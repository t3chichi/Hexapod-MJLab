"""F3 执行链：固定关节参考 + 裁剪残差目标 + 纯 PD 力矩（逐行纯计算）。

M2 起本文件为唯一正典实现（由 ``rl/joint_passage/control.py`` 兼容垫片别名），
与已验收的 joint_passage F3 执行链逐字一致：仅标准库与 PyTorch 的纯计算模块，
不导入 mjlab/任务注册器，不含 PPO/采样/标准差/reset/结束阈值/奖励权重，无全局
可变状态、无跨调用缓存。依据：docs/tasks/confined_locomotion/framework/实验记录.md、
docs/tasks/confined_locomotion/平地学习与基准测量.md 第 2 节（已确认数值）、
docs/research/训练架构对齐.md 第 1 节“执行”行。

已确认执行链（顺序固定；五个数值均为已确认的首版仿真基线，不代表实测机械
边界或硬件能力，不提供运行期覆盖参数）：

    ① clipped = clamp(raw_action, −100, +100)        # 先裁剪 raw 动作
    ② target  = q_ref + 0.25 · clipped               # 残差目标，rad
    ③ τ_raw   = 130 · (target − q) − 2 · q̇           # 纯 PD，N·m
    ④ τ       = clamp(τ_raw, −80, +80)               # 施加力矩，N·m

* q_ref 为 18 维固定名义关节参考（rad），由调用方**显式逐行**提供 (B,18)：
  本模块不读取、不缓存任何机器人构型或旧任务默认值。动作残差与 F1 的
  q−q_ref 观测共用同一参考，由调用方保证一致（平地说明第 2 节）。
* ② 之后不做任何位置裁剪：±3 rad 名义关节限位与 ±2.7 rad 软限位惩罚范围
  属于后续仿真模型限位/奖励侧（权重待定），不是实测硬限位；joint_target 可
  越出 ±3/±2.7。“±100 → ±25 rad 残差目标”只是已有方案数值的推论，不构成
  安全位置认证（docs/tasks/confined_locomotion/action_range/已有动作需求审查.md）。
* 输出语义（PDExecutionResult，行序=输入行序，逐行独立）：
      raw_action        输入动作原样（独立副本，无量纲，未做任何处理）
      clipped_action    ①（无量纲）
      joint_target      ②（rad）
      unclipped_torque  ③（N·m，未裁剪）
      applied_torque    ④（N·m，最终施加）
      torque_saturated  |③|>80 的逐元素布尔掩码（诊断记录用，非奖励项）
* 数值合法性：输入完整有限仍可能在计算中产生非有限值（如 joint_pos=−maxfloat、
  joint_vel=maxfloat 时 130·误差与 2·速度分别上溢为 ±Inf，相减得 Inf−Inf=NaN）。
  算出的关节目标、未裁剪力矩与施加力矩任一非有限即 ContractError 拒绝——
  不静默 clamp 成正常力矩、不改 PD/±100/0.25/±80、不加位置夹紧；普通力矩
  饱和（|③|>80 ⇒ ④=±80）不受影响。
* 上一步动作字段（F1 OnlineObservation.prev_action）取 raw_action 还是
  clipped_action 尚未决定（闭环接入前确认）：本模块同时返回两者，不代选。
  （M2 真实任务已确认本体 prev_action 使用裁剪后无量纲动作，由任务侧而非
  本模块决定。）
* 输入须为同设备 float32 (B,18) 且完整有限（B≥1）；输入张量不被就地修改；
  环境身份/行序由调用方维护，本函数逐行纯计算、不区分环境含义。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .contracts import FLOAT_DTYPE, JOINT_DIM, ContractError

__all__ = [
    "ACTION_CLIP",
    "PD_KP",
    "PD_KD",
    "RESIDUAL_SCALE",
    "TORQUE_CLIP",
    "PDExecutionResult",
    "compute_pd_torques",
]

# 已确认执行数值（平地学习与基准测量.md 第 2 节“动作、限位与结束”表）。
ACTION_CLIP = 100.0     # ① raw 动作对称裁剪幅值（无量纲）
RESIDUAL_SCALE = 0.25   # ② 残差比例：target = q_ref + 0.25·clipped（rad/单位动作）
PD_KP = 130.0           # ③ 比例增益 kp（N·m/rad）
PD_KD = 2.0             # ③ 微分增益 kd（N·m/(rad/s)）
TORQUE_CLIP = 80.0      # ④ 施加力矩对称裁剪幅值（N·m）


@dataclass(frozen=True)
class PDExecutionResult:
    """compute_pd_torques 的输出：执行链各阶段语义分离保存，均为 (B,18)。

    raw_action/clipped_action 无量纲；joint_target 单位 rad；unclipped_torque
    与 applied_torque 单位 N·m；torque_saturated 为 bool 掩码（|未裁剪|>80）。
    行序与输入行序一致；raw_action 为输入的独立副本（修改结果不污染调用方
    张量）；其余字段为新计算张量。上一步动作取 raw 还是 clipped 未定，调用方
    自行决定，本结果不隐含选择。
    """

    raw_action: torch.Tensor        # (B,18) float32 输入原样（独立副本）
    clipped_action: torch.Tensor    # (B,18) float32 clamp(raw, ±100)
    joint_target: torch.Tensor      # (B,18) float32 rad，q_ref + 0.25·clipped
    unclipped_torque: torch.Tensor  # (B,18) float32 N·m，130·(target−q)−2·q̇
    applied_torque: torch.Tensor    # (B,18) float32 N·m，clamp(未裁剪, ±80)
    torque_saturated: torch.Tensor  # (B,18) bool，|unclipped_torque| > 80


def _require_joint_field(name: str, value: object, ref_device: torch.device) -> torch.Tensor:
    """(B,18) float32 同设备且完整有限；拒绝而非静默修复。"""
    if not isinstance(value, torch.Tensor):
        raise ContractError(f"{name} 必须是 torch.Tensor，实际 {type(value).__name__}")
    if value.dim() != 2:
        raise ContractError(f"{name} 形状应为 (B, {JOINT_DIM})，实际 {tuple(value.shape)}")
    if int(value.shape[1]) != JOINT_DIM:
        raise ContractError(f"{name} 形状应为 (B, {JOINT_DIM})，实际 {tuple(value.shape)}")
    if value.dtype is not FLOAT_DTYPE:
        raise ContractError(f"{name} dtype 应为 {FLOAT_DTYPE}，实际 {value.dtype}")
    if value.device != ref_device:
        raise ContractError(f"{name} 设备 {value.device} 与批内基准 {ref_device} 不一致")
    if not bool(torch.isfinite(value).all()):
        raise ContractError(f"{name} 含 NaN/Inf（执行链要求完整有限输入）")
    return value


def _require_finite_computed(name: str, value: torch.Tensor) -> None:
    """有限输入下算出的中间/最终量必须仍有限，否则立即 ContractError。

    只做拒绝：不静默 clamp 成正常力矩、不改 PD/±100/0.25/±80、不加位置夹紧。
    """
    if not bool(torch.isfinite(value).all()):
        raise ContractError(
            f"{name} 含 NaN/Inf（有限输入计算出非有限值，如 Inf−Inf），拒绝而非静默裁剪"
        )


def compute_pd_torques(
    *,
    raw_action: torch.Tensor,
    q_ref: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
) -> PDExecutionResult:
    """残差动作 → 关节目标 → 纯 PD 力矩（逐行纯计算，见模块 docstring）。

    参数（均为 (B,18) float32 同设备、完整有限，B≥1）：
      raw_action  网络原始动作（无量纲；是否已在 ±100 内不设假设，先裁剪）；
      q_ref       调用方显式提供的固定名义关节参考（rad）——本函数不读取任何
                  机器人构型默认值；
      joint_pos   当前关节角 q（rad）；
      joint_vel   当前关节角速度 q̇（rad/s）。
    顺序固定为 裁剪→残差目标→PD→力矩裁剪；② 之后无位置裁剪（±3/±2.7 属后续
    模型限位/奖励，非本模块职责）。输入不被就地修改。算出的关节目标/未裁剪
    力矩/施加力矩任一非有限（有限输入下的上溢或 Inf−Inf）立即 ContractError。
    """
    if not isinstance(raw_action, torch.Tensor):
        raise ContractError(
            f"raw_action 必须是 torch.Tensor，实际 {type(raw_action).__name__}"
        )
    _require_joint_field("raw_action", raw_action, raw_action.device)
    batch = int(raw_action.shape[0])
    if batch < 1:
        raise ContractError(f"批量维度必须 >= 1，实际 {batch}")
    for name, value in (("q_ref", q_ref), ("joint_pos", joint_pos), ("joint_vel", joint_vel)):
        checked = _require_joint_field(name, value, raw_action.device)
        if int(checked.shape[0]) != batch:
            raise ContractError(
                f"{name} 批量 {int(checked.shape[0])} 与 raw_action 批量 {batch} 不一致"
            )
    clipped_action = raw_action.clamp(min=-ACTION_CLIP, max=ACTION_CLIP)
    joint_target = q_ref + RESIDUAL_SCALE * clipped_action
    _require_finite_computed("joint_target", joint_target)
    unclipped_torque = PD_KP * (joint_target - joint_pos) - PD_KD * joint_vel
    _require_finite_computed("unclipped_torque", unclipped_torque)
    applied_torque = unclipped_torque.clamp(min=-TORQUE_CLIP, max=TORQUE_CLIP)
    _require_finite_computed("applied_torque", applied_torque)
    torque_saturated = unclipped_torque.abs() > TORQUE_CLIP
    return PDExecutionResult(
        raw_action=raw_action.clone(),
        clipped_action=clipped_action,
        joint_target=joint_target,
        unclipped_torque=unclipped_torque,
        applied_torque=applied_torque,
        torque_saturated=torque_saturated,
    )
