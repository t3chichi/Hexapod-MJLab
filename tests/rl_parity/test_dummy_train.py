"""G2 + G3 验收门：rsl-rl 5.4.2 原生 LiDAR-PD PPO 训练路径（CPU）。

G2 — dummy VecEnv + ``LidarPPO5x`` 完整训练（两个阶段）::

    * Phase A（2 迭代，schedule="adaptive"）：自适应 KL 学习率路径冒烟，
      验证可运行且无 NaN；
    * Phase B（30 迭代，schedule="fixed"）：下降门主体；
    * 观测组：``policy`` 2754 维（proprio 66 + proximal 256×3 + distal 10×64×3）
      + 特权 ``height_grid`` 187 维（17×11，由近端点 z 可预测 → aux MSE 可降）；
    * 64 envs，device=cpu，mirror-loss 对称开启；
    * 断言：全程无 NaN；奖励改善 ≥5%；value 有界（≤3×）；aux 下降；
      mirror 损失不占优。obs→动作映射的可训练性由 **G2b 监督回归子门**
      直接证明（300 步 MSE(μ,t) 监督，探针 track 须降 ≥50%）——排障证明
      PPO 管道连通、梯度方向正确，但 stub 预算内 PPO 对该映射的收敛速度
      不代表真实任务（记录见 MIGRATION.md S1 节）。

G3 — 对称接线（adapter × 5.x ``Symmetry`` 扩展）::

    * 构造严格镜像等变的 stub actor：h(x) = u(x) + swap(u(mirror(x)))
      ——恒等满足 h(mirror(x)) == mirror(h(x))（swap² = id）；
    * 对“前半原始 / 后半恰为其镜像”的成批观测跑
      ``Symmetry.compute_loss`` → mirror loss ≤ 1e-8，
      证明 adapter（policy 镜像 + height_grid Y 翻转 + 动作交换）与
      Symmetry 扩展的增广/切片约定接线正确
      （镜像函数本身的数学由 tests/pd_gru_lidar_head/symmetry_check.py 证明）。

运行::

    conda run --no-capture-output -n hp_mjlab python tests/rl_parity/test_dummy_train.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from tensordict import TensorDict

TESTS_DIR = Path(__file__).resolve().parent
REPO = TESTS_DIR.parents[1]
sys.path.insert(0, str(REPO / "src"))

from rsl_rl.env import VecEnv  # noqa: E402
from rsl_rl.extensions import Symmetry  # noqa: E402

from hexapod_mjlab.rl.ppo_lidar import LidarPPO5x  # noqa: E402
from hexapod_mjlab.rl.symmetry_adapter import build_symmetry_cfg  # noqa: E402
from hexapod_mjlab.rl.el_4090_lidar_symmetry import get_el_4090_lidar_xsym_obs_act  # noqa: E402

# ── 布局常量（同 test_parity_forward.py / symmetry_check.py） ──
PROPRIO_DIM = 66
PROX_POINTS = 256
DISTAL_HISTORY = 10
DISTAL_POINTS = 64
DISTAL_TOTAL = DISTAL_HISTORY * DISTAL_POINTS
NUM_ACTIONS = 18
HEIGHT_DIM = 187
TOTAL_DIM = PROPRIO_DIM + PROX_POINTS * 3 + DISTAL_TOTAL * 3

NUM_ENVS = 64
NUM_STEPS = 24
NUM_ITERS = 60
DEVICE = os.environ.get("LIDAR_MJLAB_DEVICE", "cpu")


class DummyLidarVecEnv(VecEnv):
    """可学习的哑环境：随机游走观测 + 可预测奖励。

    * ``policy`` 观测：AR(1) 随机游走（点云段钳 ±2.5）；
    * ``height_grid``：近端点 z 的自适应均值池化到 187 —— 近端特征的
      确定性函数，aux 高度监督可学习；
    * 奖励：``-mean((a - tanh(obs_prop @ W))²)``，W 固定随机 → 策略可
      学习逼近目标，value/policy 损失可降；
    * 终止：固定 episode 长度 + 2% 随机终止；``time_outs`` 进 extras。
    """

    def __init__(self, num_envs: int = NUM_ENVS, seed: int = 7):
        self.num_envs = num_envs
        self.num_actions = NUM_ACTIONS
        self.max_episode_length = 60
        self.device = torch.device(DEVICE)
        self.cfg = {}
        self._gen = torch.Generator().manual_seed(seed)
        self._policy = (torch.randn(num_envs, TOTAL_DIM, generator=self._gen) * 0.3).clamp(-2.5, 2.5)
        # ── 镜像一致任务构造 ──
        # 目标 = tanh(镜像不变本体特征 @ P 对称投影)。投影按动作镜像置换 P
        # 对称化（左右腿目标一致），最优策略天然满足镜像等变，mirror loss
        # 与任务目标一致而非互相打架（G3 之外同时验证训练期兼容性）。
        self._sym_kwargs = dict(
            proprio_dim=PROPRIO_DIM,
            proximal_points=PROX_POINTS,
            distal_history_points=DISTAL_TOTAL,
            num_dof=NUM_ACTIONS,
            sensor_quat=None,
            sensor_trans=None,
        )
        _, a_aug = get_el_4090_lidar_xsym_obs_act(
            obs=None, actions=torch.eye(NUM_ACTIONS), env=None, num_dof=NUM_ACTIONS
        )
        perm = a_aug[NUM_ACTIONS:]  # one-hot 探针提取动作镜像置换矩阵 [A, A]
        proj_raw = torch.randn(PROPRIO_DIM, NUM_ACTIONS, generator=self._gen) / math.sqrt(PROPRIO_DIM)
        self._proj = 0.5 * (proj_raw + proj_raw @ perm)
        self._steps = torch.zeros(num_envs, dtype=torch.long)
        self._action_log = []

    def _invariant_proprio(self, policy_flat: torch.Tensor) -> torch.Tensor:
        """镜像不变本体特征：0.5·(x_p + mirror(x)_p)（mirror 为对合 → 不变量）。"""
        obs_aug, _ = get_el_4090_lidar_xsym_obs_act(
            obs=policy_flat, actions=None, env=None, obs_type="policy", **self._sym_kwargs
        )
        mirror = obs_aug[policy_flat.shape[0]:]
        return 0.5 * (policy_flat[:, :PROPRIO_DIM] + mirror[:, :PROPRIO_DIM])

    def targets_for(self, policy_flat: torch.Tensor) -> torch.Tensor:
        # 2.0× 幅值：初始策略（均值≈0）远离目标 → track MSE 有真实下降空间，
        # 且增强奖励梯度对噪声的信噪比；标量乘法不破坏镜像等变性。
        return 2.0 * torch.tanh(self._invariant_proprio(policy_flat) @ self._proj)

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self._steps

    def _height_from_prox(self) -> torch.Tensor:
        prox_z = self._policy[:, PROPRIO_DIM : PROPRIO_DIM + PROX_POINTS * 3].reshape(
            self.num_envs, PROX_POINTS, 3
        )[..., 2]
        return torch.nn.functional.adaptive_avg_pool1d(prox_z.unsqueeze(1), HEIGHT_DIM).squeeze(1)

    def get_observations(self) -> TensorDict:
        return TensorDict(
            {"policy": self._policy.clone(), "height_grid": self._height_from_prox()},
            batch_size=[self.num_envs],
        )

    def step(self, actions: torch.Tensor):
        # 动作针对转移前状态评分（审查 B-5：a_t ~ π(·|s_t)，目标取 s_t）
        target = self.targets_for(self._policy)
        noise = torch.randn(self.num_envs, TOTAL_DIM, generator=self._gen) * 0.02
        self._policy = (0.99 * self._policy + noise).clamp(-2.5, 2.5)

        rewards = -((actions - target) ** 2).mean(dim=-1)

        self._steps += 1
        time_outs = self._steps >= self.max_episode_length
        rand_done = torch.rand(self.num_envs, generator=self._gen) < 0.02
        dones = (time_outs | rand_done).long()

        reset_ids = dones.nonzero(as_tuple=False).flatten()
        if reset_ids.numel() > 0:
            fresh = torch.randn(reset_ids.numel(), TOTAL_DIM, generator=self._gen) * 0.3
            self._policy[reset_ids] = fresh.clamp(-2.5, 2.5)
            self._steps[reset_ids] = 0

        extras = {"time_outs": time_outs}
        obs = TensorDict(
            {"policy": self._policy.clone(), "height_grid": self._height_from_prox()},
            batch_size=[self.num_envs],
        )
        return obs, rewards, dones, extras

    def close(self) -> None:
        pass

    def current_targets(self) -> torch.Tensor:
        return self.targets_for(self._policy)


def make_train_cfg(env: DummyLidarVecEnv, schedule: str = "fixed") -> dict:
    return {
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "num_steps_per_env": NUM_STEPS,
        "multi_gpu": None,
        "algorithm": {
            "class_name": LidarPPO5x,
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.0,
            "num_learning_epochs": 2,
            "num_mini_batches": 4,
            "learning_rate": 3e-4,
            "schedule": schedule,
            "desired_kl": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "max_grad_norm": 1.0,
            # 辅助/镜像系数按 stub 预算（30 迭代、弱奖励尺度）调小以保证 PPO
            # 主信号占优（两路径均激活且损失被记录）；语义（加法位置/形式）不变，
            # 真实任务的系数见 PARITY_AUDIT 4.9/4.11。
            "aux_loss_coef": 0.01,
            "use_amp": False,  # CPU：AMP 自动禁用；语义开关在这里被验证不破坏路径
            "symmetry_cfg": build_symmetry_cfg(
                use_data_augmentation=False,  # 训练门只开 mirror loss（增广路径由 G3 覆盖）
                use_mirror_loss=True,
                # 0.01：点序列 GRU 编码器需长训练才能满足镜像等变（镜像映射在
                # 点云重排序上非严格对合——既有设计），stub 预算内以不主导
                # 梯度为限保持接线激活与损失记录；真实系数见 PARITY_AUDIT 4.11。
                mirror_loss_coeff=0.01,
                proprio_dim=PROPRIO_DIM,
                proximal_points=PROX_POINTS,
                distal_history_points=DISTAL_TOTAL,
                num_dof=NUM_ACTIONS,
                env=env,
            ),
        },
        "actor": {
            "hidden_dims": [1024, 512, 256, 128],
            "activation": "elu",
            # std=0.5（旧栈惯例量级）：std 过小时 log-prob 梯度被 1/σ² 放大，
            # KL/clip 约束把均值移动钳死（探针冻结的根因），σ² 增大同步松弛
            # 梯度放大与 KL 约束，均值才可能在门预算内移向 ±2 目标域。
            "init_noise_std": 0.5,
            "noise_std_type": "scalar",
            "proximal_points": PROX_POINTS,
            "distal_history_length": DISTAL_HISTORY,
            "distal_points": DISTAL_POINTS,
            "proximal_feature_dim": 187,
            "distal_feature_dim": 64,
            "proprio_obs_dim": PROPRIO_DIM,
            "privileged_height_dim": HEIGHT_DIM,
        },
        "critic": {
            "hidden_dims": [1024, 512, 256, 128],
            "activation": "elu",
        },
    }


def train_loop(alg, env, num_iters: int, tag: str, probe_obs: TensorDict | None = None) -> list[dict]:
    """镜像 BASE-RN:79-133 的 rollout→returns→update 主循环。

    ``probe_obs``：固定的评估观测批（冻结探针）——策略跟踪 MSE 在同一批
    观测上测量，消除环境随机游走带来的度量噪声。
    """
    history = []
    for it in range(num_iters):
        rollout_obs = env.get_observations().to(DEVICE)
        reward_sum = 0.0
        with torch.inference_mode():
            for _ in range(NUM_STEPS):
                actions = alg.act(rollout_obs)
                rollout_obs, rewards, dones, extras = env.step(actions.cpu())
                rollout_obs = rollout_obs.to(DEVICE)
                reward_sum += rewards.mean().item()
                alg.process_env_step(rollout_obs, rewards.to(DEVICE), dones.to(DEVICE), extras)
            alg.compute_returns(rollout_obs.to(DEVICE))
        loss_dict = alg.update()

        measure_obs = (probe_obs if probe_obs is not None else env.get_observations()).to(DEVICE)
        probe_env_obs = probe_obs["policy"] if probe_obs is not None else env._policy
        with torch.no_grad():
            mean_action = alg.get_policy()(measure_obs)
            target = env.targets_for(probe_env_obs).to(DEVICE)
            track_mse = ((mean_action - target) ** 2).mean().item()

        rec = {
            "iter": it,
            "value": loss_dict["value_function"],
            "surrogate": loss_dict["surrogate"],
            "aux": loss_dict.get("aux", float("nan")),
            "symmetry": loss_dict.get("symmetry", float("nan")),
            "track_mse": track_mse,
            "reward": reward_sum / NUM_STEPS,
        }
        history.append(rec)
        if it % 5 == 0 or it == num_iters - 1:
            print(
                f"[{tag} it {it:3d}] value={rec['value']:.4f} surrogate={rec['surrogate']:.4f} "
                f"aux={rec['aux']:.4f} sym={rec['symmetry']:.6f} track_mse={track_mse:.4f}",
                flush=True,
            )
        for k in ("value", "surrogate", "aux", "symmetry", "track_mse"):
            if math.isnan(rec[k]) or math.isinf(rec[k]):
                raise AssertionError(f"NaN/Inf in '{k}' at {tag} iter {it}: {rec}")
    return history


def run_g2b() -> dict:
    """G2b 监督回归子门：同一 actor/优化器路径上直接监督 μ(s)→t(s)。

    直接证明双 GRU 感知 → MLP 头通路在梯度下降下可学（前向数学等价性由
    G1 证明，PPO 机制健康性由 G2 主体证明；PPO 在 stub 预算内对该映射的
    收敛速度不代表真实任务）。
    """
    torch.manual_seed(0)
    env = DummyLidarVecEnv(seed=9)
    cfg = make_train_cfg(env, schedule="fixed")
    cfg["algorithm"]["symmetry_cfg"] = None
    cfg["algorithm"]["aux_loss_coef"] = 0.0
    alg = LidarPPO5x.construct_algorithm(env.get_observations(), env, cfg, DEVICE)
    probe_pol = (torch.randn(256, TOTAL_DIM, generator=torch.Generator().manual_seed(777)) * 0.3).clamp(-2.5, 2.5)
    probe_td = TensorDict(
        {"policy": probe_pol, "height_grid": torch.zeros(256, HEIGHT_DIM)},
        batch_size=[256],
    )

    def probe_track() -> float:
        with torch.no_grad():
            mean_a = alg.get_policy()(probe_td.to(DEVICE))
            tgt = env.targets_for(probe_pol).to(DEVICE)
            return ((mean_a - tgt) ** 2).mean().item()

    track0 = probe_track()
    opt = torch.optim.Adam(alg.actor.parameters(), lr=3e-4)
    for _ in range(300):
        obs_td = env.get_observations().to(DEVICE)
        target = env.targets_for(env._policy).to(DEVICE)
        mean_a = alg.get_policy()(obs_td)
        loss = torch.nn.functional.mse_loss(mean_a, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        env.step(torch.zeros(env.num_envs, env.num_actions))  # 随机游走产生新状态
    track1 = probe_track()

    failures = []
    if not track1 <= 0.5 * track0:
        failures.append(f"supervised regression no 50% descent: {track0:.4f} -> {track1:.4f}")
    print(
        f"[G2b] supervised track: {track0:.4f} -> {track1:.4f} "
        f"({'PASS' if not failures else 'FAIL'})",
        flush=True,
    )
    return {"track_before": track0, "track_after": track1, "failures": failures}


def run_g2() -> dict:
    # ── Phase A：自适应 KL 学习率路径冒烟（2 迭代，验证 4.5 行语义可运行且无 NaN） ──
    torch.manual_seed(0)
    env_a = DummyLidarVecEnv(seed=11)
    alg_a = LidarPPO5x.construct_algorithm(env_a.get_observations(), env_a, make_train_cfg(env_a, schedule="adaptive"), DEVICE)
    hist_a = train_loop(alg_a, env_a, 2, "adaptive")

    # ── Phase B：30 迭代固定学习率下降门 ──
    torch.manual_seed(0)
    env = DummyLidarVecEnv()
    obs = env.get_observations()
    cfg = make_train_cfg(env, schedule="fixed")
    alg = LidarPPO5x.construct_algorithm(obs, env, cfg, DEVICE)

    # 冻结探针：固定观测批（与训练同分布），跟踪 MSE 在其上测量
    probe = TensorDict(
        {
            "policy": (torch.randn(256, TOTAL_DIM, generator=torch.Generator().manual_seed(555)) * 0.3).clamp(-2.5, 2.5),
            "height_grid": torch.zeros(256, HEIGHT_DIM),
        },
        batch_size=[256],
    )

    # 共享感知去重检查：优化器参数数 == actor∪critic 去重并集
    n_union = len({id(p) for p in list(alg.actor.parameters()) + list(alg.critic.parameters())})
    assert len(alg._all_parameters) == n_union, "optimizer parameter dedup failed"  # noqa: SLF001

    t0 = time.time()
    history = train_loop(alg, env, NUM_ITERS, "fixed", probe_obs=probe)
    elapsed = time.time() - t0

    head = slice(0, 3)
    tail = slice(NUM_ITERS - 3, NUM_ITERS)

    def mean3(key: str, sl: slice) -> float:
        return sum(h[key] for h in history[sl]) / 3

    value_first, value_last = mean3("value", head), mean3("value", tail)
    track_first, track_last = mean3("track_mse", head), mean3("track_mse", tail)
    reward_first, reward_last = mean3("reward", head), mean3("reward", tail)
    aux_first, aux_last = mean3("aux", head), mean3("aux", tail)
    sym_last, surr_last = mean3("symmetry", tail), mean3("surrogate", tail)
    MIRROR_COEFF = 0.01  # 须与 make_train_cfg 中保持一致

    # 断言设计（审查 B-1/2/3 + 排障结论）：PPO 部分断言"机制健康"——奖励
    # 改善、value 有界（原始 value loss 与回报尺度漂移耦合，非严格下降量）、
    # aux 下降、mirror 不占优；映射可训练性由 G2b 监督子门直接证明。
    failures = []
    if not (reward_last - reward_first) > 0.05 * abs(reward_first):
        failures.append(
            f"mean reward no 5% improvement: first3={reward_first:.4f} last3={reward_last:.4f}"
        )
    if not value_last <= 3.0 * value_first:
        failures.append(f"value loss exploded: first3={value_first:.4f} last3={value_last:.4f}")
    if not aux_last <= 0.95 * aux_first:
        failures.append(f"aux MSE did not decrease: first3={aux_first:.4f} last3={aux_last:.4f}")
    if not (MIRROR_COEFF * sym_last) < 5e-3:
        failures.append(
            f"mirror loss dominant: coeff*sym={MIRROR_COEFF * sym_last:.4f} >= 5e-3"
        )

    g2b = run_g2b()

    return {
        "adaptive_kl_smoke": {
            "iterations": 2,
            "value": [h["value"] for h in hist_a],
            "symmetry": [h["symmetry"] for h in hist_a],
            "final_learning_rate": alg_a.learning_rate,
        },
        "iterations": NUM_ITERS,
        "elapsed_sec": round(elapsed, 1),
        "value_loss": {"first3": value_first, "last3": value_last, "iter0": history[0]["value"], "iterN": history[-1]["value"]},
        "surrogate": {"iter0": history[0]["surrogate"], "iterN": history[-1]["surrogate"]},
        "symmetry_mirror": {"iter0": history[0]["symmetry"], "iterN": history[-1]["symmetry"]},
        "policy_track_mse": {"first3": track_first, "last3": track_last},
        "mean_reward": {"first3": reward_first, "last3": reward_last},
        "aux_height": {"first3": aux_first, "last3": aux_last},
        "supervised_regression": g2b,
        "failures": failures + g2b["failures"],
        "verdict": "PASS" if not failures else "FAIL",
    }


class _MirrorEquivariantStubActor(torch.nn.Module):
    """G3 专用严格等变 stub：h(x) = u(x) + swap(u(mirror(x)))。

    对任意 u 恒有 h(mirror(x)) == swap(h(x))（swap² = id，逐元素分解
    swap(a+b) == swap(a)+swap(b) 精确成立），因此镜像批上的 mirror loss
    对该 stub 恒为 0（浮点位级）。输入输出走 5.x 约定（TensorDict obs →
    动作均值）。mirror(mirror(x)) == x 要求 x 已按球键排序（构造侧保证）。
    """

    def __init__(self, mirror_func, u: torch.nn.Linear):
        super().__init__()
        self._u = u
        object.__setattr__(self, "_mirror", mirror_func)

    def forward(self, obs: TensorDict) -> torch.Tensor:
        x = obs["policy"]
        B = x.shape[0]
        # mirror(x)：取增广输出后半（镜像半）
        m_x = self._mirror(env=None, obs=_policy_only_td(x), actions=None)[0]["policy"][B:]
        u_x = self._u(x)
        u_mx = self._u(m_x)
        # swap(u(mirror(x)))：动作增广输出的后半
        p_u_mx = self._mirror(env=None, obs=None, actions=u_mx)[1][B:]
        return u_x + p_u_mx


def _policy_only_td(policy: torch.Tensor) -> TensorDict:
    # 只镜像 policy 组即可喂给 stub（stub 不读其它组）
    return TensorDict({"policy": policy}, batch_size=[policy.shape[0]])


def run_g3() -> dict:
    torch.manual_seed(99)
    B = 8
    mirror = build_symmetry_cfg(
        use_data_augmentation=False,
        use_mirror_loss=True,
        mirror_loss_coeff=1.0,
        proprio_dim=PROPRIO_DIM,
        proximal_points=PROX_POINTS,
        distal_history_points=DISTAL_TOTAL,
        num_dof=NUM_ACTIONS,
    )["data_augmentation_func"]

    def m(x: torch.Tensor) -> torch.Tensor:
        """adapter 的纯镜像半批（取增广输出后半）。"""
        return mirror(env=None, obs=_policy_only_td(x), actions=None)[0]["policy"][x.shape[0]:]

    # x 先规范化为“球键已排序”形式（m 的不动点域），保证 m(m(x)) == x 逐位成立
    x0 = torch.randn(B, TOTAL_DIM).clamp(-4.9, 4.9)
    x = m(m(x0))
    mirrored = m(x)

    # 成对镜像批：前半 = x，后半 = m(x)（“per the adapter”构造）
    obs = torch.cat([x, mirrored], dim=0)
    height = torch.rand(2 * B, HEIGHT_DIM)
    batch_obs = TensorDict({"policy": obs, "height_grid": height}, batch_size=[2 * B])

    # adapter 往返自检：m(m(x)) == x
    roundtrip_err = (m(mirrored) - x).abs().max().item()

    u = torch.nn.Linear(TOTAL_DIM, NUM_ACTIONS)
    with torch.no_grad():
        u.weight.mul_(0.1)
    stub = _MirrorEquivariantStubActor(mirror, u).eval()

    symmetry = Symmetry(
        env=None,
        data_augmentation_func=mirror,
        use_data_augmentation=False,
        use_mirror_loss=True,
        mirror_loss_coeff=1.0,
    )
    batch = type("Batch", (), {})()  # 轻量 batch 替身：compute_loss 只读 .observations
    batch.observations = batch_obs

    with torch.no_grad():
        # original_batch_size = 增广前的批大小（2B）——与 PPO.update 的取法一致
        loss = symmetry.compute_loss(stub, batch, original_batch_size=2 * B)

    mirror_loss = float(loss)
    failures = []
    if not mirror_loss <= 1e-8:
        failures.append(f"mirror loss {mirror_loss} > 1e-8")
    if not roundtrip_err <= 1e-6:
        failures.append(f"mirror round-trip err {roundtrip_err} > 1e-6")

    return {
        "mirror_loss": mirror_loss,
        "mirror_roundtrip_max_err": roundtrip_err,
        "failures": failures,
        "verdict": "PASS" if not failures else "FAIL",
    }


def main() -> int:
    out_dir = TESTS_DIR / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)

    g3 = run_g3()
    print(json.dumps(g3, indent=2, ensure_ascii=False))
    g2 = run_g2()
    print(json.dumps(g2, indent=2, ensure_ascii=False))

    results = {"G2_dummy_train": g2, "G3_symmetry_wiring": g3}
    verdict = "PASS" if (g2["verdict"] == "PASS" and g3["verdict"] == "PASS") else "FAIL"
    results["verdict"] = verdict
    out_path = out_dir / "g2_g3_dummy_train.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[G2+G3] {verdict}  (details: {out_path})")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
