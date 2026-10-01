"""旧版 LiDAR-PD checkpoint → rsl-rl 5.x 新版键名迁移。

旧版（``Lidar_legged_gym``，单策略 ActorCritic）checkpoint::

    {
      "model_state_dict": {                 # 单一 policy 的 state_dict
          "proximal_gru.weight_ih_l0", ...  # 近端 GRU
          "distal_gru.weight_ih_l0", ...    # 远端 GRU
          "height_supervisor.weight/bias",  # 高度监督头
          "actor.0.weight", ..., "actor.8.bias",        # actor MLP（nn.Sequential 索引）
          "critic.0.weight", ..., "critic.8.bias",      # critic MLP
          "std",                             # scalar 标准差参数（noise_std_type="scalar"）
          "log_std",                         # log 标准差参数（noise_std_type="log"，二选一）
      },
      "optimizer_state_dict": ...,
      "iter": int, "infos": ...,
    }

新版（rsl-rl 5.4.2，actor/critic 分离 + ``LidarPDActor5x/LidarPDCritic5x``）::

    {
      "actor_state_dict": {                 # LidarPDActor5x
          "proximal_gru.*",                 # 不变
          "distal_gru.*",                   # 不变
          "height_supervisor.*",            # 不变
          "mlp.0.weight", ..., "mlp.8.bias",# actor.* → mlp.*（层结构/索引完全相同）
          "distribution.std_param",         # std → distribution.std_param
          "distribution.log_std_param",     # log_std → distribution.log_std_param
      },
      "critic_state_dict": {                # LidarPDCritic5x（共享 GRU 只在 actor 侧）
          "mlp.0.weight", ..., "mlp.8.bias",# critic.* → mlp.*
      },
      "iter": int, "infos": ...,
    }

键映射总表（逐条）::

    旧键 (model_state_dict)            新键                                目标模型
    ─────────────────────────────────  ──────────────────────────────────  ────────────────
    proximal_gru.weight_ih_l0          proximal_gru.weight_ih_l0           actor
    proximal_gru.weight_hh_l0          proximal_gru.weight_hh_l0           actor
    proximal_gru.bias_ih_l0            proximal_gru.bias_ih_l0             actor
    proximal_gru.bias_hh_l0            proximal_gru.bias_hh_l0             actor
    distal_gru.weight_ih_l0            distal_gru.weight_ih_l0             actor
    distal_gru.weight_hh_l0            distal_gru.weight_hh_l0             actor
    distal_gru.bias_ih_l0              distal_gru.bias_ih_l0               actor
    distal_gru.bias_hh_l0              distal_gru.bias_hh_l0               actor
    height_supervisor.weight           height_supervisor.weight            actor
    height_supervisor.bias             height_supervisor.bias              actor
    actor.{i}.weight / actor.{i}.bias  mlp.{i}.weight / mlp.{i}.bias       actor
    critic.{i}.weight / critic.{i}.…   mlp.{i}.weight / mlp.{i}.bias       critic
    std                                distribution.std_param              actor
    log_std                            distribution.log_std_param          actor

显式不做迁移（有意差异，见 PARITY_AUDIT.md）：

* ``optimizer_state_dict``：旧优化器对单一 policy 参数排序，新优化器对
  去重后的 actor∪critic 参数排序，Adam 动量槽无法一一对应 → 丢弃
  （恢复训练首个迭代等效冷启动动量，策略/价值函数本身无损）。
* 观测归一化统计：旧任务 ``empirical_normalization=False``（Identity），
  新模型同样禁用归一化，无状态需迁移。
* AMP ``scaler_state_dict``：torch.cuda.amp → torch.amp 的 GradScaler
  状态字典同构，若存在则原样透传（键名不变）。
"""

from __future__ import annotations

import torch

__all__ = ["convert_checkpoint", "convert_model_state_dict", "CKPT_KEY_MAP", "RECOMMENDED_LOAD_CFG"]

# 前缀级映射（“.* 后缀原样保留”），加两条整键映射。
CKPT_KEY_MAP: dict[str, tuple[str, str]] = {
    "proximal_gru.": ("proximal_gru.", "actor"),
    "distal_gru.": ("distal_gru.", "actor"),
    "height_supervisor.": ("height_supervisor.", "actor"),
    "actor.": ("mlp.", "actor"),
    "critic.": ("mlp.", "critic"),
    "std": ("distribution.std_param", "actor"),
    "log_std": ("distribution.log_std_param", "actor"),
}

# PPO.load 的配套 load_cfg：本转换器不产出 optimizer_state_dict（见模块
# docstring），而 load 的默认 load_cfg 为 optimizer=True → 直接 KeyError
# （架构审查 MED-5）。加载转换产物时必须传入：
#     alg.load(convert_checkpoint(old), load_cfg=RECOMMENDED_LOAD_CFG, strict=True)
# 或经 runner 时传等价 load_cfg。
RECOMMENDED_LOAD_CFG: dict[str, bool] = {
    "actor": True,
    "critic": True,
    "optimizer": False,
    "iteration": False,
    "rnd": False,
}


def convert_model_state_dict(old_state_dict: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    """把旧版单一 ``model_state_dict`` 转成 ``{"actor_state_dict": …, "critic_state_dict": …}``。"""
    actor_sd: dict[str, torch.Tensor] = {}
    critic_sd: dict[str, torch.Tensor] = {}
    unmapped: list[str] = []

    for key, value in old_state_dict.items():
        if key in ("std", "log_std"):  # 整键映射优先（避免被 "actor." 等前缀误伤）
            new_key, target = CKPT_KEY_MAP[key]
            bucket = actor_sd if target == "actor" else critic_sd
            bucket[new_key] = value
            continue
        for prefix, (new_prefix, target) in CKPT_KEY_MAP.items():
            if prefix.endswith(".") and key.startswith(prefix):
                bucket = actor_sd if target == "actor" else critic_sd
                bucket[new_prefix + key[len(prefix):]] = value
                break
        else:
            unmapped.append(key)

    if unmapped:
        raise KeyError(
            f"convert_model_state_dict: keys with no migration rule: {unmapped}. "
            f"Known prefixes: {sorted(CKPT_KEY_MAP)}."
        )
    return {"actor_state_dict": actor_sd, "critic_state_dict": critic_sd}


def convert_checkpoint(old_checkpoint: dict) -> dict:
    """旧版 runner checkpoint → 新版 5.x alg.load 可用的 checkpoint。

    接受两种输入：

    * 旧 runner 保存的完整 dict（含 ``model_state_dict``）；
    * 裸的旧 policy ``state_dict``（键为 ``proximal_gru.*`` 等）。

    返回::

        {
          "actor_state_dict": …,      # LidarPDActor5x.state_dict()
          "critic_state_dict": …,     # LidarPDCritic5x.state_dict()
          "iter": …,                  # 若旧 checkpoint 带 "iter"
          "infos": …,                 # 若旧 checkpoint 带 "infos"
          "scaler_state_dict": …,     # 若旧 checkpoint 带（AMP）
        }

    注意：不迁移 ``optimizer_state_dict``（见模块 docstring 的有意差异），
    因此加载本产物必须传 ``load_cfg=RECOMMENDED_LOAD_CFG``（默认 load_cfg
    会因缺失该键而 KeyError）。
    """
    if "model_state_dict" in old_checkpoint:
        full = old_checkpoint
        old_sd = full["model_state_dict"]
    else:
        full = None
        old_sd = old_checkpoint

    new = convert_model_state_dict(old_sd)
    if full is not None:
        if "iter" in full:
            new["iter"] = full["iter"]
        if "infos" in full:
            new["infos"] = full["infos"]
        if "scaler_state_dict" in full:
            new["scaler_state_dict"] = full["scaler_state_dict"]
    return new
