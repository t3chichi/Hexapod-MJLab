"""G1 验收门：旧 LidarPDActorCritic ↔ 新 LidarPDActor5x/LidarPDCritic5x 数值等价。

流程（CPU，eval + no_grad）：

1. 同种子构造旧模型，取 ``state_dict``；
2. 经 ``ckpt_convert.convert_model_state_dict`` 迁移后 strict 载入新
   actor/critic（strict=True 本身验证键映射完整性）；
3. 喂同一随机观测批（B=8，布局 2754 = proprio 66 + proximal 256×3 +
   distal 10×64×3），逐项断言 ≤1e-6：

   - 确定性动作均值（旧 ``act_inference`` ↔ 新 ``actor(obs)``）
   - 近端特征（旧/新 ``_cached_proximal_feature``）
   - 分布均值/标准差/熵/对数概率（旧 ``action_mean`` 等 ↔ 新
     ``output_mean`` 等）
   - 采样动作（同 RNG 种子下逐位一致）
   - critic 值（旧 ``evaluate`` ↔ 新 ``critic(obs)``，经共享感知缓存）
   - 辅助高度监督损失（旧/新 ``compute_auxiliary_loss``）

运行::

    conda run --no-capture-output -n hp_mjlab python tests/rl_parity/test_parity_forward.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
from tensordict import TensorDict

TESTS_DIR = Path(__file__).resolve().parent
REPO = TESTS_DIR.parents[1]
sys.path.insert(0, str(REPO / "src"))

from hexapod_mjlab.rl.actor_v2 import LidarPDActor5x, LidarPDCritic5x  # noqa: E402
from hexapod_mjlab.rl.ckpt_convert import convert_model_state_dict  # noqa: E402
from hexapod_mjlab.rl.lidar_pd_actor_critic import LidarPDActorCritic  # noqa: E402

# ── EL4090 LiDAR-PD 布局（与 tests/pd_gru_lidar_head/symmetry_check.py 一致） ──
PROPRIO_DIM = 66
PROX_POINTS = 256
DISTAL_HISTORY = 10
DISTAL_POINTS = 64
DISTAL_TOTAL = DISTAL_HISTORY * DISTAL_POINTS  # 640
NUM_ACTIONS = 18
PROX_FEATURE = 187
DIST_FEATURE = 64
HEIGHT_DIM = 187  # 17 x 11
TOTAL_DIM = PROPRIO_DIM + PROX_POINTS * 3 + DISTAL_TOTAL * 3  # 2754
B = 8

TOL = 1e-6
SENSOR_RPY = [0.0, 0.3, -0.2]
SENSOR_POS = [0.7, 0.0, -0.05]


def build_old_model() -> LidarPDActorCritic:
    return LidarPDActorCritic(
        TOTAL_DIM,
        TOTAL_DIM,  # 旧版 critic_obs_dim 未使用（共享 GRU latent）
        NUM_ACTIONS,
        actor_hidden_dims=(1024, 512, 256, 128),
        critic_hidden_dims=(1024, 512, 256, 128),
        activation="elu",
        init_noise_std=0.8,
        noise_std_type="scalar",
        proximal_points=PROX_POINTS,
        distal_history_length=DISTAL_HISTORY,
        distal_points=DISTAL_POINTS,
        proximal_feature_dim=PROX_FEATURE,
        distal_feature_dim=DIST_FEATURE,
        proprio_obs_dim=PROPRIO_DIM,
        privileged_height_dim=HEIGHT_DIM,
        sensor_offset_rpy=SENSOR_RPY,
        sensor_offset_pos=SENSOR_POS,
        gradient_checkpointing_proximal=False,
        gradient_checkpointing_distal=True,
    )


def build_new_models(obs_td: TensorDict, obs_groups: dict):
    actor = LidarPDActor5x(
        obs_td,
        obs_groups,
        "actor",
        NUM_ACTIONS,
        hidden_dims=(1024, 512, 256, 128),
        activation="elu",
        init_noise_std=0.8,
        noise_std_type="scalar",
        proximal_points=PROX_POINTS,
        distal_history_length=DISTAL_HISTORY,
        distal_points=DISTAL_POINTS,
        proximal_feature_dim=PROX_FEATURE,
        distal_feature_dim=DIST_FEATURE,
        proprio_obs_dim=PROPRIO_DIM,
        privileged_height_dim=HEIGHT_DIM,
        sensor_offset_rpy=SENSOR_RPY,
        sensor_offset_pos=SENSOR_POS,
        gradient_checkpointing_proximal=False,
        gradient_checkpointing_distal=True,
    )
    critic = LidarPDCritic5x(
        obs_td,
        obs_groups,
        "critic",
        1,
        hidden_dims=(1024, 512, 256, 128),
        activation="elu",
        actor=actor,
    )
    return actor, critic


def main() -> int:
    out_dir = TESTS_DIR / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    failures: list[str] = []

    torch.manual_seed(1234)
    old = build_old_model().eval()
    old_sd = old.state_dict()

    torch.manual_seed(4321)  # 构造种子不同：数值等价必须来自权重迁移而非构造顺序
    obs_td = TensorDict(
        {"policy": torch.zeros(B, TOTAL_DIM), "height_grid": torch.zeros(B, HEIGHT_DIM)},
        batch_size=[B],
    )
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor, critic = build_new_models(obs_td, obs_groups)
    actor = actor.eval()
    critic = critic.eval()

    new_sds = convert_model_state_dict(old_sd)
    actor.load_state_dict(new_sds["actor_state_dict"], strict=True)
    critic.load_state_dict(new_sds["critic_state_dict"], strict=True)

    # 迁移覆盖检查：新旧参数量一致
    n_old = sum(p.numel() for p in old.parameters())
    n_new = sum(p.numel() for p in actor.parameters()) + sum(p.numel() for p in critic.parameters())
    results["param_counts"] = {"old": n_old, "new_actor_plus_critic": n_new}
    if n_old != n_new:
        failures.append(f"parameter count mismatch: old={n_old} new={n_new}")

    # 同一批随机观测（点云段钳到常用量程）
    torch.manual_seed(2024)
    obs = torch.randn(B, TOTAL_DIM)
    obs[:, PROPRIO_DIM:] = obs[:, PROPRIO_DIM:].clamp(-4.9, 4.9)
    height = torch.rand(B, HEIGHT_DIM)
    obs_td = TensorDict(
        {"policy": obs.clone(), "height_grid": height.clone()},
        batch_size=[B],
    )

    def check(name: str, a: torch.Tensor, b2: torch.Tensor, tol: float = TOL) -> float:
        a, b2 = a.detach(), b2.detach()
        if a.shape != b2.shape:
            failures.append(f"{name}: shape {tuple(a.shape)} vs {tuple(b2.shape)}")
            return float("nan")
        err = (a - b2).abs().max().item()
        results[name] = err
        if not (err <= tol) or math.isnan(err):
            failures.append(f"{name}: max abs err {err} > {tol}")
        return err

    with torch.no_grad():
        # 1) 确定性均值（旧 act_inference ↔ 新 forward(stochastic_output=False)）
        old_mean = old.act_inference(obs)
        new_mean = actor(obs_td)
        check("deterministic_action_mean", old_mean, new_mean)

        # 2) 随机路径：分布均值 / 标准差 / 熵 / log_prob / 近端特征 / latent 缓存
        torch.manual_seed(777)
        old_actions = old.act(obs)
        old_log_prob = old.get_actions_log_prob(old_actions)
        torch.manual_seed(777)  # 同 RNG → 同采样噪声
        new_actions = actor(obs_td, stochastic_output=True)
        new_log_prob = actor.get_output_log_prob(new_actions)

        check("sampled_actions_same_rng", old_actions, new_actions)
        check("distribution_mean", old.action_mean, actor.output_mean)
        check("distribution_std", old.action_std, actor.output_std)
        check("entropy", old.entropy, actor.output_entropy)
        check("actions_log_prob", old_log_prob, new_log_prob)
        check("proximal_feature_cache", old.cached_proximal_feature, actor.cached_proximal_feature)
        check(
            "actor_latent_cache",
            old._cached_actor_latent,  # noqa: SLF001
            actor._cached_actor_latent,  # noqa: SLF001
        )

        # 3) critic 值（旧 evaluate 缓存命中 ↔ 新 critic 共享感知）
        old_value = old.evaluate(obs)
        new_value = critic(obs_td)
        check("critic_value", old_value, new_value)

        # 4) aux 高度监督损失
        old_aux = old.compute_auxiliary_loss(height)
        new_aux = actor.compute_auxiliary_loss(height)
        check("aux_height_loss", old_aux.unsqueeze(0), new_aux.unsqueeze(0))

        # 5) critic 缓存未命中（换一批观测，须从输入重建而非复用缓存）
        obs2 = torch.randn(B, TOTAL_DIM)
        obs2[:, PROPRIO_DIM:] = obs2[:, PROPRIO_DIM:].clamp(-4.9, 4.9)
        obs_td2 = TensorDict({"policy": obs2.clone(), "height_grid": height.clone()}, batch_size=[B])
        old_value2 = old.evaluate(obs2)
        new_value2 = critic(obs_td2)
        check("critic_value_cache_miss", old_value2, new_value2)

    results["failures"] = failures
    results["verdict"] = "PASS" if not failures else "FAIL"
    out_path = out_dir / "g1_parity_forward.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"[G1] {'PASS' if not failures else 'FAIL'}  (details: {out_path})")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
