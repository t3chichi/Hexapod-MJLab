"""P3 联合更新解析损失（纯 Torch）：PPO-Clip、价值裁剪、熵、包络/地面辅助、KL 与学习率调度。

仅标准库与 PyTorch；不导入 mjlab/任务模块（更新编排与单 Adam 算法见
``ppo.py``）。依据 docs/tasks/confined_locomotion/平地学习与基准测量.md 第 3 节
（「PPO 与优化器起步配方」「地面辅助与启动损失」）、docs/research/训练架构对齐.md
第 4/5 节、docs/tasks/confined_locomotion/training/首版训练实施计划.md 4.4：

* PPO 策略损失（raw log-prob）：ratio=exp(new−old)；
  L_policy = mean(max(−A·ratio, −A·clamp(ratio, 0.8, 1.2)))。A 是**完整 rollout
  一次性归一化**后的优势切片；归一化唯一入口 ``normalize_rollout_advantage``
  （N=T·B、N>1、torch.std 的 correction=1，参考实现实际语义），绝不在
  minibatch 内重新归一化。
* 价值裁剪损失（系数 1）：以 oldV 为中心 ±0.2；
  L_value = mean(max((V−R)², (oldV+clip(V−oldV, ±0.2)−R)²))。
* 熵：事件级（18 维求和）逐样本值 (K,) 的批均值进入总损失，系数 0.01 取负号：
  L_total = L_policy + 1·L_value − 0.01·H̄ + 1·L_E + 1·L_H（``total_joint_loss``）。
* 包络辅助 λ_E=1：几何有效样本上六维平方误差**求和**（不除以 6）后批平均；
  有效行**先索引后平方**，不乘 NaN×0；有效空集返回 device 零。
* 地面辅助 λ_H=1：帧内 l_s = mean_valid((H_s−H*_s)²)（有效位置先索引后平方）；
  第 j 桶 L_H,j = (4/|S|)·Σ_(s∈bucket_j) l_s，**不除以桶局部帧数**；四桶平均
  恒等于全 S 帧等权平均（独立于每帧有效位置数）；空桶/|S|=0 返回 device 零。
  桶内每来源帧的 fresh 近端前向由 ``ppo.py`` 负责，本模块只做解析聚合。
* KL（学习率调度用，精确式与已安装 rsl_rl 5.4.2 实际语义一致）：
  ``torch.distributions.kl_divergence(Normal(old_mean, old_std),
  Normal(new_mean, new_std)).sum(-1)``；没有旧版本额外 epsilon 近似。
* 学习率调度（``adjust_learning_rate``）：minibatch 均值 KL > 2·desired_kl →
  lr/1.5；0 < KL < desired_kl/2 → lr×1.5；否则保持；随后整体 clamp 到
  [1e-5, 1e-2]（desired_kl=0.01 时阈值即 0.02/0.005）。

所有函数显式校验形状/dtype/设备/有限性；exp 溢出或 NaN/Inf 一律抛
``LossContractError``——保存停止，不 clamp std、不删样本、不重试优化。
"""

from __future__ import annotations

import math

import torch

from .contracts import FLOAT_DTYPE
from .training_rollout import GROUND_NUM_BUCKETS

__all__ = [
    "ADVANTAGE_NORM_EPS",
    "DESIRED_KL_STARTUP",
    "ENTROPY_COEFF",
    "ENVELOPE_LOSS_COEFF",
    "GROUND_LOSS_COEFF",
    "GROUND_NUM_BUCKETS_CONST",
    "KL_DOWN_FACTOR",
    "KL_UP_FACTOR",
    "LR_ADJUST_FACTOR",
    "LR_MAX",
    "LR_MIN",
    "PPO_CLIP_EPSILON",
    "VALUE_CLIP_EPSILON",
    "VALUE_LOSS_COEFF",
    "LossContractError",
    "adjust_learning_rate",
    "envelope_sum_loss",
    "ground_bucket_loss",
    "ground_frame_mean_loss",
    "kl_divergence_old_new",
    "normalize_rollout_advantage",
    "ppo_policy_loss",
    "require_finite",
    "total_joint_loss",
    "value_clipped_loss",
]

# —— 已确认起步配方常数（权威：平地学习与基准测量.md「PPO 与优化器起步配方」）——
PPO_CLIP_EPSILON = 0.2        # 策略 clip（ratio 区间 [0.8, 1.2]）
VALUE_CLIP_EPSILON = 0.2      # 价值 clip（以 oldV 为中心 ±0.2）
VALUE_LOSS_COEFF = 1.0        # 价值损失系数
ENTROPY_COEFF = 0.01          # 熵系数（总损失中取负号）
ENVELOPE_LOSS_COEFF = 1.0     # λ_E
GROUND_LOSS_COEFF = 1.0       # λ_H
ADVANTAGE_NORM_EPS = 1e-8     # 完整 rollout 优势归一化分母保护
DESIRED_KL_STARTUP = 0.01     # KL 自适应目标（阈值 2×/÷2 即 0.02/0.005）
LR_ADJUST_FACTOR = 1.5        # KL 调整倍率
LR_MIN = 1e-5                 # 学习率下界（完整 clamp）
LR_MAX = 1e-2                 # 学习率上界（完整 clamp）
KL_DOWN_FACTOR = LR_ADJUST_FACTOR   # KL 过大 → lr/1.5
KL_UP_FACTOR = LR_ADJUST_FACTOR     # KL 过小 → lr×1.5
#: 与 PPO 4 minibatch 配对的地面桶数（training_rollout 已冻结为 4）。
GROUND_NUM_BUCKETS_CONST = GROUND_NUM_BUCKETS


class LossContractError(ValueError):
    """解析损失契约违规（形状/dtype/设备/非有限/std 非正/分母非法等）。"""


def require_finite(name: str, tensor: torch.Tensor) -> None:
    """有限性守卫：NaN/Inf 立即抛错（不删数据、不 clamp、不重试）。"""
    if not isinstance(tensor, torch.Tensor):
        raise LossContractError(f"{name} 必须是 torch.Tensor，实际 {type(tensor).__name__}")
    if not bool(torch.isfinite(tensor).all()):
        raise LossContractError(f"{name} 含 NaN/Inf（保存停止；不 clamp、不删样本）")


def _require_1d(name: str, tensor: torch.Tensor, *, dtype: torch.dtype, device: torch.device) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise LossContractError(f"{name} 必须是 torch.Tensor，实际 {type(tensor).__name__}")
    if tensor.dim() != 1:
        raise LossContractError(f"{name} 须为一维 (K,) 张量，实际 {tuple(tensor.shape)}")
    if tensor.dtype is not dtype:
        raise LossContractError(f"{name} dtype 应为 {dtype}，实际 {tensor.dtype}")
    if tensor.device != device:
        raise LossContractError(f"{name} 设备 {tensor.device} 与批内基准 {device} 不一致")


# ═══════════════════════════ 优势归一化（唯一入口） ═══════════════════════════


def normalize_rollout_advantage(advantage_flat: torch.Tensor) -> torch.Tensor:
    """完整 rollout 的 N=T·B 统一优势标准化（仅此一次；P2 GAE 未归一化）。

    (A − mean(A)) / (torch.std(A, correction=1) + 1e-8)——correction=1 是已安装
    参考实现的实际无偏 std；N>1 强制（N=1 时 std 无定义，显式拒绝）。
    minibatch 只能切本结果的行，不得再次归一化或覆盖原 GAE。
    """
    if not isinstance(advantage_flat, torch.Tensor) or advantage_flat.dim() != 1:
        raise LossContractError(
            f"advantage_flat 须为一维 (N,) 张量，实际 "
            f"{tuple(advantage_flat.shape) if isinstance(advantage_flat, torch.Tensor) else type(advantage_flat).__name__}"
        )
    if advantage_flat.dtype is not FLOAT_DTYPE:
        raise LossContractError(f"advantage_flat dtype 应为 {FLOAT_DTYPE}，实际 {advantage_flat.dtype}")
    n = int(advantage_flat.numel())
    if n <= 1:
        raise LossContractError(f"优势归一化要求 N>1（完整 rollout），实际 N={n}")
    require_finite("advantage_flat", advantage_flat)
    mean = advantage_flat.mean()
    std = torch.std(advantage_flat, correction=1)
    require_finite("advantage mean/std", torch.stack([mean.reshape(()), std.reshape(())]))
    normalized = (advantage_flat - mean) / (std + ADVANTAGE_NORM_EPS)
    require_finite("normalized advantage", normalized)
    return normalized


# ═══════════════════════════ PPO 策略 / 价值 / 熵 ═══════════════════════════


def ppo_policy_loss(
    new_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    normalized_advantage: torch.Tensor,
    *,
    clip_epsilon: float = PPO_CLIP_EPSILON,
) -> torch.Tensor:
    """PPO-Clip 策略损失：mean(max(−A·ratio, −A·clamp(ratio, 1−ε, 1+ε)))。

    new_log_prob 保留当前计算图（对 raw 动作、18 维求和的事件级值）；
    old_log_prob/优势为 buffer 无梯度切片。ratio=exp(new−old)，溢出立即抛错。
    """
    device = new_log_prob.device if isinstance(new_log_prob, torch.Tensor) else torch.device("cpu")
    _require_1d("new_log_prob", new_log_prob, dtype=FLOAT_DTYPE, device=device)
    _require_1d("old_log_prob", old_log_prob, dtype=FLOAT_DTYPE, device=device)
    _require_1d("normalized_advantage", normalized_advantage, dtype=FLOAT_DTYPE, device=device)
    if not (new_log_prob.shape == old_log_prob.shape == normalized_advantage.shape):
        raise LossContractError(
            f"policy loss 形状不一致: new {tuple(new_log_prob.shape)} old "
            f"{tuple(old_log_prob.shape)} adv {tuple(normalized_advantage.shape)}"
        )
    if int(new_log_prob.numel()) < 1:
        raise LossContractError("policy loss 要求非空 minibatch")
    require_finite("new_log_prob", new_log_prob)
    require_finite("old_log_prob", old_log_prob)
    require_finite("normalized_advantage", normalized_advantage)
    if not (math.isfinite(float(clip_epsilon)) and float(clip_epsilon) > 0.0):
        raise LossContractError(f"clip_epsilon 须为正有限值，实际 {clip_epsilon!r}")
    ratio = torch.exp(new_log_prob - old_log_prob)
    require_finite("policy ratio (exp 溢出)", ratio)
    clamped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    surrogate = torch.maximum(
        -normalized_advantage * ratio,
        -normalized_advantage * clamped,
    )
    loss = surrogate.mean()
    require_finite("policy loss", loss)
    return loss


def value_clipped_loss(
    new_value: torch.Tensor,
    old_value: torch.Tensor,
    returns: torch.Tensor,
    *,
    clip_epsilon: float = VALUE_CLIP_EPSILON,
) -> torch.Tensor:
    """价值裁剪损失：mean(max((V−R)², (oldV+clip(V−oldV, ±ε)−R)²))，系数 1。

    new_value (K,) 保留计算图（网络 value 头 (K,1) 的显式 squeeze 结果）；
    old_value/returns 来自 buffer（(K,) 规范化存储）。
    """
    device = new_value.device if isinstance(new_value, torch.Tensor) else torch.device("cpu")
    for name, tensor in (("new_value", new_value), ("old_value", old_value), ("returns", returns)):
        _require_1d(name, tensor, dtype=FLOAT_DTYPE, device=device)
    if not (new_value.shape == old_value.shape == returns.shape):
        raise LossContractError(
            f"value loss 形状不一致: new {tuple(new_value.shape)} old "
            f"{tuple(old_value.shape)} returns {tuple(returns.shape)}"
        )
    if int(new_value.numel()) < 1:
        raise LossContractError("value loss 要求非空 minibatch")
    for name, tensor in (("new_value", new_value), ("old_value", old_value), ("returns", returns)):
        require_finite(name, tensor)
    if not (math.isfinite(float(clip_epsilon)) and float(clip_epsilon) > 0.0):
        raise LossContractError(f"value clip_epsilon 须为正有限值，实际 {clip_epsilon!r}")
    clipped = old_value + (new_value - old_value).clamp(-clip_epsilon, clip_epsilon)
    loss = torch.maximum((new_value - returns) ** 2, (clipped - returns) ** 2).mean()
    require_finite("value loss", loss)
    return loss


def total_joint_loss(
    loss_policy: torch.Tensor,
    loss_value: torch.Tensor,
    entropy: torch.Tensor,
    loss_envelope: torch.Tensor,
    loss_ground: torch.Tensor,
    *,
    value_coeff: float = VALUE_LOSS_COEFF,
    entropy_coeff: float = ENTROPY_COEFF,
    envelope_coeff: float = ENVELOPE_LOSS_COEFF,
    ground_coeff: float = GROUND_LOSS_COEFF,
) -> torch.Tensor:
    """联合总损失：L_policy + λ_V·L_value − 0.01·H̄ + λ_E·L_E + λ_H·L_H。

    entropy 为事件级 (K,) 批值，批均值进入总损失（负号）。五个分量均为
    device 上的标量；一次 backward 的唯一对象（ppo.py 使用）。
    """
    device = loss_policy.device if isinstance(loss_policy, torch.Tensor) else torch.device("cpu")
    for name, tensor in (
        ("loss_policy", loss_policy), ("loss_value", loss_value),
        ("loss_envelope", loss_envelope), ("loss_ground", loss_ground),
    ):
        if not isinstance(tensor, torch.Tensor) or tensor.dim() != 0:
            raise LossContractError(f"{name} 须为 device 上的标量张量")
        if tensor.device != device:
            raise LossContractError(f"{name} 设备 {tensor.device} 与 {device} 不一致")
        require_finite(name, tensor)
    _require_1d("entropy", entropy, dtype=FLOAT_DTYPE, device=device)
    require_finite("entropy", entropy)
    for name, coeff in (
        ("value_coeff", value_coeff), ("entropy_coeff", entropy_coeff),
        ("envelope_coeff", envelope_coeff), ("ground_coeff", ground_coeff),
    ):
        if not (math.isfinite(float(coeff)) and float(coeff) >= 0.0):
            raise LossContractError(f"{name} 须为非负有限值，实际 {coeff!r}")
    total = (
        loss_policy
        + value_coeff * loss_value
        - entropy_coeff * entropy.mean()
        + envelope_coeff * loss_envelope
        + ground_coeff * loss_ground
    )
    require_finite("total joint loss", total)
    return total


# ═══════════════════════════ 包络辅助（六维求和尺度） ═══════════════════════════


def envelope_sum_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """L_E = mean_valid( Σ_{d=1..6} (Ê_nd − E*_nd)² )：六维求和、样本平均。

    ``prediction`` 为网络米制 Ê（未 detach，辅助路径）；``target`` 为固定标签
    (K,6)；``valid`` (K,) bool 几何有效样本。有效行**先索引后平方**（不乘
    NaN×0：无效行内容不参与任何算术）；有效空集返回 device 零标量。相同样本
    下该形式是六维取平均的 6 倍（不除以 6）。
    """
    if not isinstance(prediction, torch.Tensor) or prediction.dim() != 2 \
            or int(prediction.shape[1]) != 6:
        raise LossContractError(f"prediction 须为 (K,6) 张量，实际 {tuple(prediction.shape) if isinstance(prediction, torch.Tensor) else type(prediction).__name__}")
    device = prediction.device
    k = int(prediction.shape[0])
    if not isinstance(target, torch.Tensor) or tuple(target.shape) != (k, 6) \
            or target.dtype is not FLOAT_DTYPE or target.device != device:
        raise LossContractError(
            f"target 须为 ({k},6) {FLOAT_DTYPE} 且在 {device}，实际 "
            f"{tuple(target.shape) if isinstance(target, torch.Tensor) else type(target).__name__}"
        )
    if not isinstance(valid, torch.Tensor) or valid.dim() != 1 \
            or tuple(valid.shape) != (k,) or valid.dtype is not torch.bool or valid.device != device:
        raise LossContractError(f"valid 须为 ({k},) bool 且在 {device}")
    if k < 1:
        raise LossContractError("envelope loss 要求非空 minibatch（K>=1）")
    require_finite("envelope prediction", prediction)
    if bool(valid.any()):
        require_finite("envelope target (valid rows)", target[valid])
        # 先索引后平方：只对有效行做差与平方，无效行不进入算术。
        difference = prediction[valid] - target[valid]
        per_sample_sum = (difference * difference).sum(dim=1)
        loss = per_sample_sum.mean()
    else:
        loss = prediction.new_zeros(())
    require_finite("envelope loss", loss)
    return loss


# ═══════════════════════════ 地面辅助（两级平均 / 4 桶权重） ═══════════════════════════


def ground_frame_mean_loss(
    prediction: torch.Tensor,
    heights: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """单帧 l_s = mean_{j∈V_s}((H_sj − H*_sj)²)：帧内有效位置平均。

    有效位置**先索引后平方**（无效槽即使含 NaN 也不进入算术，不乘 NaN×0）；
    全无效帧返回 device 零（该帧本不入 S，此返回只为桶聚合的安全不变量）。
    """
    if not isinstance(prediction, torch.Tensor) or prediction.dim() != 1:
        raise LossContractError(f"帧 prediction 须为 (M,)，实际 {tuple(prediction.shape) if isinstance(prediction, torch.Tensor) else type(prediction).__name__}")
    device = prediction.device
    m = int(prediction.numel())
    if m < 1:
        raise LossContractError("地面帧须 M>=1")
    if not isinstance(heights, torch.Tensor) or tuple(heights.shape) != (m,) \
            or heights.dtype is not FLOAT_DTYPE or heights.device != device:
        raise LossContractError(f"帧 heights 须为 ({m},) {FLOAT_DTYPE} 且在 {device}")
    if not isinstance(valid, torch.Tensor) or tuple(valid.shape) != (m,) \
            or valid.dtype is not torch.bool or valid.device != device:
        raise LossContractError(f"帧 valid 须为 ({m},) bool 且在 {device}")
    require_finite("帧 prediction", prediction)
    if bool(valid.any()):
        require_finite("帧 heights (valid)", heights[valid])
        difference = prediction[valid] - heights[valid]     # 先索引
        loss = (difference * difference).mean()             # 后平方
    else:
        loss = prediction.new_zeros(())
    require_finite("ground frame loss", loss)
    return loss


def ground_bucket_loss(
    frame_predictions: torch.Tensor,
    frame_heights: torch.Tensor,
    frame_valid: torch.Tensor,
    supervision_size: int,
    *,
    num_buckets: int = GROUND_NUM_BUCKETS_CONST,
) -> torch.Tensor:
    """第 j 桶地面损失：L_H,j = (num_buckets/|S|)·Σ_(s∈bucket_j) l_s。

    **不除以桶局部帧数**：四桶平均恒等于全 S 帧等权平均（独立于每帧有效
    位置数）。空桶（K=0）或 |S|=0 返回 device 零（合法跳过，不改变 PPO/
    包络更新数）。帧内逐帧 mean_valid（``ground_frame_mean_loss``），桶内求和
    后乘统一权重 num_buckets/|S|。每帧的 fresh 近端前向由调用方完成，本函数
    只做解析聚合。
    """
    if not isinstance(frame_predictions, torch.Tensor) or frame_predictions.dim() != 2:
        raise LossContractError(
            f"frame_predictions 须为 (K,M)，实际 "
            f"{tuple(frame_predictions.shape) if isinstance(frame_predictions, torch.Tensor) else type(frame_predictions).__name__}"
        )
    device = frame_predictions.device
    k, m = int(frame_predictions.shape[0]), int(frame_predictions.shape[1])
    if not isinstance(frame_heights, torch.Tensor) or tuple(frame_heights.shape) != (k, m) \
            or frame_heights.dtype is not FLOAT_DTYPE or frame_heights.device != device:
        raise LossContractError(f"frame_heights 须为 ({k},{m}) {FLOAT_DTYPE} 且在 {device}")
    if not isinstance(frame_valid, torch.Tensor) or tuple(frame_valid.shape) != (k, m) \
            or frame_valid.dtype is not torch.bool or frame_valid.device != device:
        raise LossContractError(f"frame_valid 须为 ({k},{m}) bool 且在 {device}")
    if isinstance(supervision_size, bool) or not isinstance(supervision_size, int):
        raise LossContractError(f"supervision_size 须为 int，实际 {supervision_size!r}")
    if isinstance(num_buckets, bool) or not isinstance(num_buckets, int) or num_buckets < 1:
        raise LossContractError(f"num_buckets 须为 >=1 整数，实际 {num_buckets!r}")
    if supervision_size < 0:
        raise LossContractError(f"supervision_size |S| 须为非负，实际 {supervision_size}")
    # 完全 S 空（|S|=0）合法跳过；空桶（K=0）合法贡献零——先于一致性检查。
    if k == 0 or supervision_size == 0:
        return frame_predictions.new_zeros(())
    if supervision_size < k:
        raise LossContractError(
            f"|S|={supervision_size} 小于桶内帧数 K={k}（桶划分来自 S 的互斥子集，配置不一致）"
        )
    frame_losses = []
    for frame_index in range(k):
        frame_losses.append(
            ground_frame_mean_loss(
                frame_predictions[frame_index], frame_heights[frame_index], frame_valid[frame_index]
            )
        )
    loss = torch.stack(frame_losses).sum() * (float(num_buckets) / float(supervision_size))
    require_finite("ground bucket loss", loss)
    return loss


# ═══════════════════════════ KL 与学习率调度 ═══════════════════════════


def kl_divergence_old_new(
    old_mean: torch.Tensor,
    old_std: torch.Tensor,
    new_mean: torch.Tensor,
    new_std: torch.Tensor,
) -> torch.Tensor:
    """KL(old‖new) 事件级 (K,)：torch.distributions.kl_divergence 精确语义。

    与已安装 rsl_rl 5.4.2 实际实现一致：
    ``kl_divergence(Normal(old_mean, old_std), Normal(new_mean, new_std)).sum(-1)``
    （18 维独立对角求和；没有旧版本额外 epsilon 近似）。std 非正/非有限
    显式拒绝。本函数只做调度证据，PPO 损失不使用 KL。
    """
    for name, tensor in (("old_mean", old_mean), ("old_std", old_std),
                         ("new_mean", new_mean), ("new_std", new_std)):
        if not isinstance(tensor, torch.Tensor) or tensor.dim() != 2 \
                or int(tensor.shape[1]) != 18 or tensor.dtype is not FLOAT_DTYPE:
            raise LossContractError(
                f"{name} 须为 (K,18) {FLOAT_DTYPE}，实际 "
                f"{tuple(tensor.shape) if isinstance(tensor, torch.Tensor) else type(tensor).__name__}"
            )
    device = old_mean.device
    for name, tensor in (("old_std", old_std), ("new_mean", new_mean), ("new_std", new_std)):
        if tensor.device != device:
            raise LossContractError(f"{name} 设备 {tensor.device} 与 old_mean {device} 不一致")
    if not (old_mean.shape == old_std.shape == new_mean.shape == new_std.shape):
        raise LossContractError(
            f"KL 形状不一致: {tuple(old_mean.shape)} {tuple(old_std.shape)} "
            f"{tuple(new_mean.shape)} {tuple(new_std.shape)}"
        )
    for name, tensor in (("old_mean", old_mean), ("old_std", old_std),
                         ("new_mean", new_mean), ("new_std", new_std)):
        require_finite(name, tensor)
    for name, tensor in (("old_std", old_std), ("new_std", new_std)):
        if bool((tensor <= 0).any()):
            raise LossContractError(f"{name} 须处处为正（std=exp(log_std) 无 clamp）")
    kl = torch.distributions.kl_divergence(
        torch.distributions.Normal(old_mean, old_std),
        torch.distributions.Normal(new_mean, new_std),
    ).sum(dim=-1)
    require_finite("KL(old‖new)", kl)
    return kl


def adjust_learning_rate(
    kl_mean: float,
    learning_rate: float,
    *,
    desired_kl: float = DESIRED_KL_STARTUP,
    lr_min: float = LR_MIN,
    lr_max: float = LR_MAX,
) -> float:
    """KL 自适应学习率（纯 float，便于独立复算）：返回调整并 clamp 后的值。

    kl_mean > 2·desired_kl → lr/1.5；0 < kl_mean < desired_kl/2 → lr×1.5；
    否则保持（含 kl==0 与 [desired_kl/2, 2·desired_kl] 区间）；最终整体
    clamp 到 [lr_min, lr_max]（desired_kl=0.01 时阈值即 0.02/0.005）。
    无 early-stop；同一 Adam 的全部参数组由调用方统一套用。
    """
    for name, value in (("kl_mean", kl_mean), ("learning_rate", learning_rate),
                        ("desired_kl", desired_kl), ("lr_min", lr_min), ("lr_max", lr_max)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise LossContractError(f"{name} 须为有限数值，实际 {value!r}")
    if desired_kl <= 0.0 or lr_min <= 0.0 or lr_max <= lr_min:
        raise LossContractError(
            f"调度参数非法: desired_kl={desired_kl} lr_min={lr_min} lr_max={lr_max}"
        )
    adjusted = float(learning_rate)
    if kl_mean > 2.0 * desired_kl:
        adjusted = adjusted / KL_DOWN_FACTOR
    elif 0.0 < kl_mean < desired_kl / 2.0:
        adjusted = adjusted * KL_UP_FACTOR
    return min(max(adjusted, float(lr_min)), float(lr_max))
