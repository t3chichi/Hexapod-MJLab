"""F2 形态输出映射：head 原始输出 → 米制包络 Ê（首版 Softplus，r=1）。

M2 起本文件为唯一正典实现（由 ``rl/joint_passage/mapping.py`` 兼容垫片别名），
与已验收的 joint_passage F2 映射逐字一致：仅标准库与 PyTorch 的纯计算模块；
不导入 mjlab、不含 PPO/优化器/训练逻辑、不含场景真值。依据：
docs/tasks/confined_locomotion/framework/实验记录.md、
docs/research/训练架构对齐.md 第 3 节、
docs/tasks/confined_locomotion/平地学习与基准测量.md 第 2 节（确认值）。

已确认语义：
* p = p0 ⊙ softplus(raw)，r=1（F.softplus 默认 β=1），固定、不设 β/阈值参数。
* 启动正量 p0=[0.59,0.69,0.59,0.89,0.89,0.46] m 与高度参考 h_ref=0.46 m 为
  已确认启动值，作为默认值提供，可显式覆盖（切换实测基准 E0/h0 时的尺度/偏置
  处理仍待定，不在本模块内解决）。
* 对外只返回米制 Ê=[前半宽,中半宽,后半宽,前缘x,后缘x(负),Δz]，其中
  Δz=p_height−h_ref；raw 只经显式调试接口暴露，不进入在线前向返回值。
  物理映射（softplus→米制）唯一放在本形态模块内，网络其余部分不重复实现。
* 两种等价表示，mode 必须显式给出（正式表示未确认，不设默认）：
    "raw"       head 末层直接输出 raw，末层偏置应由 b=softplus⁻¹(1) 初始化；
    "residual"  raw = b + u，head 末层偏置应为 0，b 由本模块恰好加一次
                （不得在 head 偏置与本模块重复叠加）。
* b = softplus⁻¹(1) = log(e−1) ≈ 0.5413248546：softplus(b)=1 ⇒ 初始 p=p0、
  Ê=[0.59,0.69,0.59,0.89,−0.89,0] m（形态末层零权重时对任意有限输入成立）。
  可训练性来自上游 head 参数；本模块自身无可训练参数。
* p0/h_ref 注册为 buffer：进入 state_dict 可重载、随 .to() 移动设备；
  不是可训练参数，优化器不应包含它们。
* 表示模式随 checkpoint 保存（get_extra_state/set_extra_state，键 `_extra_state`）：
  跨模式或缺元数据的重载显式拒绝，不得借转换/切换表示绕过校验——同一 head
  输出在两种模式下的物理量不同（残差侧多加 b），静默混载会引入错误偏置。
* 已知数值性质（如实记录，非认证安全）：softplus 导数 sigmoid(raw) 在 raw 极负端
  衰减趋 0（docs/tasks/confined_locomotion/mapping/实验报告.md 记录过极负端恢复变慢）。
  正式可达域不在本轮：不加 EA2 硬裁剪、不换 Sigmoid、不加安全 solver。
* 数值合法性拒绝（工程数值正确性，非新增合法域/阈值）：正量 p 数值上非有限
  （有限尺度乘法上溢为 Inf、p0 buffer 被错误重载）或非正（softplus 极负端
  float32 下溢为 0、p0 重载为 0/负值）、Ê 含非有限值（非法 h_ref buffer）、
  目标高度 h_ref+Δz 的 float32 重建值非正/非有限——一律 ContractError 拒绝，
  不裁剪、不设 floor、不加 solver；正常区间输出/梯度与极负端梯度衰减性质
  不变（衰减可经 positive_quantities 合法观察）。真实运行时异常如何 reset
  不在本模块决定。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .contracts import ENVELOPE_DIM, FLOAT_DTYPE, ContractError

__all__ = [
    "ENVELOPE_MODES",
    "H_REF_STARTUP",
    "P0_STARTUP",
    "RAW_MODE",
    "RESIDUAL_MODE",
    "SOFTPLUS_INVERSE_ONE",
    "EnvelopeSoftplusMapping",
]

# 已确认启动正量与高度参考（docs/tasks/confined_locomotion/平地学习与基准测量.md 第 2 节）。
P0_STARTUP = (0.59, 0.69, 0.59, 0.89, 0.89, 0.46)
H_REF_STARTUP = 0.46

# b = softplus⁻¹(1) = log(e−1) ≈ 0.5413248546：softplus(b)=1 ⇒ 初始 p=p0。
SOFTPLUS_INVERSE_ONE = math.log(math.e - 1.0)

RAW_MODE = "raw"
RESIDUAL_MODE = "residual"
ENVELOPE_MODES = (RAW_MODE, RESIDUAL_MODE)

# checkpoint 模式元数据编码（经 get_extra_state/set_extra_state 持久化）。
_MODE_CODES = {RAW_MODE: 0, RESIDUAL_MODE: 1}
_CODE_NAMES = {0: RAW_MODE, 1: RESIDUAL_MODE}


class EnvelopeSoftplusMapping(nn.Module):
    """p = p0 ⊙ softplus(raw) 与 Ê=[fw,mw,bw,front_x,−rear,Δz] 的唯一物理映射。

    mode="raw"       head 末层直接输出 raw；末层偏置初值应为 b（见 head_terminal_bias）。
    mode="residual"  raw = b + u；head 末层偏置初值应为 0，b 由本模块恰好加一次。
    p0/h_ref 为 buffer（state_dict 可重载、随 .to() 移动）；本模块无可训练参数。
    mode 经 `_extra_state` 随 state_dict 保存并在重载时校验：跨模式或缺元数据
    的重载明确拒绝（ContractError/RuntimeError），不得借转换绕过正式表示选择。
    正量与 Ê 的数值合法性（非有限/非正/高度重建非正）在计算路径上直接拒绝
    （ContractError，不裁剪/不设 floor）：forward 与 positive_quantities 共享
    同一正量检查，buffer 被错误重载为非法值时由结果检查拒绝（不新增正式
    checkpoint 方案/原子加载）。
    """

    def __init__(self, *, mode: str, p0=P0_STARTUP, h_ref: float = H_REF_STARTUP) -> None:
        super().__init__()
        if mode not in ENVELOPE_MODES:
            raise ContractError(
                f"mode 必须为 {ENVELOPE_MODES} 之一（正式表示未确认，须显式给出），实际 {mode!r}"
            )
        p0_tensor = torch.as_tensor(p0, dtype=FLOAT_DTYPE).reshape(-1).clone()
        if int(p0_tensor.numel()) != ENVELOPE_DIM:
            raise ContractError(f"p0 应为 {ENVELOPE_DIM} 个启动正量，实际 {int(p0_tensor.numel())} 个")
        if not bool(torch.isfinite(p0_tensor).all()) or bool((p0_tensor <= 0).any()):
            raise ContractError("p0 必须全部为正的有限值（正量尺度）")
        h_ref_tensor = torch.as_tensor(h_ref, dtype=FLOAT_DTYPE).reshape(()).clone()
        if not bool(torch.isfinite(h_ref_tensor).all()):
            raise ContractError("h_ref 必须为有限值")
        self.mode = mode
        self.register_buffer("p0", p0_tensor)
        self.register_buffer("h_ref", h_ref_tensor)

    @property
    def head_terminal_bias(self) -> float:
        """形态末层偏置初值约定：raw 模式为 b=softplus⁻¹(1)，残差模式为 0。"""
        return SOFTPLUS_INVERSE_ONE if self.mode == RAW_MODE else 0.0

    def extra_repr(self) -> str:
        return f"mode={self.mode!r}"

    def get_extra_state(self) -> torch.Tensor:
        """把表示模式编码进 state_dict（键 `_extra_state`，单元素 int64）。"""
        return torch.tensor(_MODE_CODES[self.mode], dtype=torch.int64)

    def set_extra_state(self, state: object) -> None:
        """重载校验：checkpoint 模式必须与当前实例一致，错误模式明确拒绝。"""
        if not isinstance(state, torch.Tensor) or state.dtype is not torch.int64 \
                or state.numel() != 1:
            raise ContractError(f"checkpoint 模式元数据非法（应为单元素 int64）：{state!r}")
        code = int(state.reshape(-1)[0].item())
        if code not in _CODE_NAMES:
            raise ContractError(f"checkpoint 模式元数据未知编码：{code}")
        if code != _MODE_CODES[self.mode]:
            raise ContractError(
                f"envelope_mode 与 checkpoint 不一致：checkpoint 为 {_CODE_NAMES[code]!r}，"
                f"当前实例为 {self.mode!r}；跨模式重载被拒绝（不得借转换/切换表示绕过）"
            )

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs) -> None:
        # 缺模式元数据（本修正之前的 checkpoint 形式）同样拒绝：不允许静默跳过校验。
        if prefix + "_extra_state" not in state_dict:
            error_msgs.append(
                f"checkpoint 缺少 {prefix}_extra_state（envelope_mode 元数据），"
                "拒绝未经模式校验的重载"
            )
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def _checked_head_output(self, head_output: object) -> torch.Tensor:
        if not isinstance(head_output, torch.Tensor):
            raise ContractError(
                f"head 输出应为 (B, {ENVELOPE_DIM}) float32 张量，实际 {type(head_output).__name__}"
            )
        if head_output.dim() != 2 or int(head_output.shape[1]) != ENVELOPE_DIM \
                or head_output.dtype is not FLOAT_DTYPE:
            raise ContractError(
                f"head 输出应为 (B, {ENVELOPE_DIM}) float32 张量，"
                f"实际 {tuple(head_output.shape)}/{head_output.dtype}"
            )
        if not bool(torch.isfinite(head_output).all()):
            raise ContractError("head 输出含 NaN/Inf（上游接线错误或已发散，拒绝而非静默传播）")
        return head_output

    def raw_values(self, head_output: torch.Tensor) -> torch.Tensor:
        """调试接口：head 输出 → 六维 raw（残差模式加 b；raw 模式原样返回）。

        raw 不进入在线前向返回值，仅供调试/诊断，不作为训练接口。
        """
        raw = self._checked_head_output(head_output)
        if self.mode == RESIDUAL_MODE:
            return raw + SOFTPLUS_INVERSE_ONE
        return raw

    def _positive_quantities_checked(self, head_output: torch.Tensor) -> torch.Tensor:
        """正量 p = p0 ⊙ softplus(raw) 的唯一计算与校验路径（forward 与调试接口共享）。

        数值上非有限（有限尺度乘法上溢为 Inf、p0 buffer 被错误重载为 Inf/NaN）或
        非正（softplus 极负端 float32 下溢为 0、p0 重载为 0/负值）的正量一律
        ContractError 拒绝：不裁剪、不设最小尺寸 floor、不换激活/solver——这些
        退化值属于应拒绝的非法输出，不是应修复的正常输出。
        """
        raw = self.raw_values(head_output)
        positive = self.p0 * F.softplus(raw)
        if not bool(torch.isfinite(positive).all()):
            raise ContractError(
                "正量 p=p0⊙softplus(raw) 含 NaN/Inf（有限输入上溢或非法 p0 buffer），"
                "拒绝而非静默给出非有限物理量"
            )
        if not bool((positive > 0).all()):
            raise ContractError(
                "正量 p=p0⊙softplus(raw) 数值上非正（softplus 极负端下溢为 0 或非法 p0 buffer），"
                "拒绝而非裁剪/设 floor"
            )
        return positive

    def positive_quantities(self, head_output: torch.Tensor) -> torch.Tensor:
        """六个正量 p = p0 ⊙ softplus(raw)（米）；不还原后缘符号、不减 h_ref。

        与 forward 共享同一正量计算/校验路径：数值上非有限/非正的正量在此同样
        拒绝，调试接口不得绕过检查（高度重建语义仅在 forward 中检查）。
        """
        return self._positive_quantities_checked(head_output)

    def forward(self, head_output: torch.Tensor) -> torch.Tensor:
        """米制 Ê (B,6)=[前半宽,中半宽,后半宽,前缘x,后缘x(负),Δz]，单位米。

        在共享正量校验之上另查两级物理合法性（均 ContractError，不裁剪/不设
        floor）：Ê 各分量有限（非法 h_ref buffer 不得静默传播为 ±Inf/NaN）；
        目标高度 h_ref+Δz 的 float32 重建值为正且有限——p_height 远小于 h_ref
        的 ulp 时 Δz=p_height−h_ref 完全消减为 −h_ref，加回参考得 0 高度，
        违反基本几何合法性。真实运行时异常如何 reset 不在本模块决定。
        """
        positive = self._positive_quantities_checked(head_output)
        horizontal = torch.cat([positive[:, :4], -positive[:, 4:5]], dim=1)
        delta_z = positive[:, 5:6] - self.h_ref
        envelope = torch.cat([horizontal, delta_z], dim=1)
        if not bool(torch.isfinite(envelope).all()):
            raise ContractError("米制 Ê 含 NaN/Inf（如非法 h_ref buffer），拒绝而非静默传播")
        reconstructed = self.h_ref + delta_z.detach()
        if not bool(torch.isfinite(reconstructed).all()) or not bool((reconstructed > 0).all()):
            raise ContractError(
                "目标高度 h_ref+Δz 重建为非正/非有限（float32 消减退化为 0 高度），"
                "拒绝而非裁剪或设 floor"
            )
        return envelope
