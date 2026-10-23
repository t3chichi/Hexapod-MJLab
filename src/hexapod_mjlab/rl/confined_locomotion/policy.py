"""Policy adapter for ``ConfinedLocomotionNetwork``.

This module provides task policy forward, mean-action, sample/evaluation, and
parameter-deduplication APIs. It accepts an existing network and optional
explicit distribution provider; it does not create an optimizer, implement
PPO/GAE, or execute actions in an environment.

接口边界（与已确认架构一致，本模块不改任何语义）：

* ``ConfinedPolicyAdapter(nn.Module)`` 把一个**已存在**的
  ``ConfinedLocomotionNetwork`` 原样注册一次（``self.network``，单一实例，不
  重复注册、不拷贝权重）。``forward`` 与网络签名一致并精确复用其前向：返回
  ``NetworkOutput``（mean/value/envelope/ground_pred 逐字段相同）。
  63 维本体与 3 维指令的已确认倍率**只**在网络 forward 施加一次，本 adapter
  不做任何二次缩放。action/value 在网络内部消费 detach(Ê)，感知直连路径仍可
  反传；envelope 与 ground_pred 保持未 detach，供未来辅助损失路径使用。
* 没有跨控制步 hidden state（网络每次 GRU 前向 h0=0）。``reset()`` 是显式
  无操作：不清空/重建共享网络或 provider 参数，也不创建任何 GRU carry。
* 独立 mean 行为接口 ``mean_actions``：返回 raw 动作均值与 ±100 裁剪副本。
  mean-only 输出没有采样概率、entropy 或标准差，不能替代 PPO 采样数据。本
  adapter 没有任何执行环境动作的入口，不存在经 sampled 分支自动 step 环境的路径。
* 显式预留 distribution-provider 注入口：外部注入**有身份**的 nn.Module
  provider，其 ``forward(action_mean)`` 仅接收 (B,18) 动作均值，返回满足
  **event 级**契约 ``mean (B,18) / sample() (B,18) / log_prob(raw) (B,) /
  entropy() (B,)`` 的对象。没有 provider 时 ``sample`` / ``evaluate_actions`` /
  ``log_prob_actions`` 一律显式拒绝。生产文件不实现任何分布族、不给标准差或
  探索默认值；不把网络输出按名称当作完整 actor distribution。
* provider 身份必须非空且可 JSON 冻结（``allow_nan=False`` 的规范序列化），
  并在评估/采样输出中原样回传对应；log_prob 对应**未裁剪 raw** 动作，不对应
  clip 值或 PD 目标角。provider 的分布 mean 必须与网络 action_mean 一致
  （容差 1e-6，纯数值守恒；防止第二均值头/变形头接错）。所有 provider 输出
  校验 shape/dtype=float32/device/有限性，event 级要求 (B,) 一维，不静默
  猜维度、不求和补救。采样结果以 (raw, log_prob, identity) 成组返回；
  ``evaluate_actions`` 全程不做错误 detach（均值与 provider 参数保留梯度）。
* ``unique_parameters()`` 按参数 id 一次去重，涵盖网络与显式提供的 provider
  参数；普通 ``parameters()`` 亦无重复（网络只注册一份）。

本 adapter 不选择训练分布族/探索/std、不新建 optimizer。任务的显式生产
runner 使用这些策略接口并负责优化与生命周期；本模块本身仍只负责策略
前向、分布 provider 契约、采样评估与参数去重。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .contracts import FLOAT_DTYPE, JOINT_DIM, OnlineObservation
from .network import ConfinedLocomotionNetwork, NetworkOutput

__all__ = [
    "ACTION_CLIP_LIMIT",
    "ACTION_DIM",
    "ActionEvaluation",
    "ConfinedPolicyAdapter",
    "MeanActionOutput",
    "PolicyAdapterError",
    "PolicySampledAction",
]

#: 动作维数（与网络动作头一致）。
ACTION_DIM = JOINT_DIM

#: Mean-action clipping boundary (±100; the network output remains raw).
ACTION_CLIP_LIMIT = 100.0

#: provider 分布 mean 与网络 action_mean 的一致性容差（纯浮点守恒级；
#: 任何第二均值头/变形头的偏差都会远大于此）。
_PROVIDER_MEAN_ATOL = 1e-6


class PolicyAdapterError(ValueError):
    """policy adapter 契约违规（provider/身份/形状/dtype/有限性/event 级等）。"""


def _freeze_identity(identity: object) -> str:
    """把 provider 身份冻结为规范 JSON 字符串（非空、可序列化、无 NaN）。

    接受字符串或任意 JSON 可序列化结构（dict/list/数值等）；集合、张量、
    NaN/Inf 与空值一律拒绝。返回的冻结字符串即采样/评估输出回传的身份凭据。
    """
    if identity is None:
        raise PolicyAdapterError("distribution_identity 不得为 None（须显式非空身份）")
    if isinstance(identity, str) and not identity.strip():
        raise PolicyAdapterError("distribution_identity 不得为空字符串")
    try:
        frozen = json.dumps(
            identity, sort_keys=True, allow_nan=False, separators=(",", ":")
        )
    except (TypeError, ValueError) as err:
        raise PolicyAdapterError(
            f"distribution_identity 必须可 JSON 冻结（非空、无 NaN/Inf、非集合/张量），"
            f"实际 {identity!r}: {err}"
        ) from err
    if frozen in ("", "null", '""', "{}", "[]"):
        raise PolicyAdapterError(f"distribution_identity 冻结结果为空: {identity!r}")
    return frozen


def _check_action_block(
    name: str, tensor: object, *, batch: int, device: torch.device
) -> None:
    """(B,18) float32、同设备、有限——event 的动作侧形状（不做任何维度猜测）。"""
    if not isinstance(tensor, torch.Tensor):
        raise PolicyAdapterError(f"{name} 必须是 torch.Tensor，实际 {type(tensor).__name__}")
    if tensor.dtype is not FLOAT_DTYPE:
        raise PolicyAdapterError(f"{name} dtype 应为 {FLOAT_DTYPE}，实际 {tensor.dtype}")
    if tuple(tensor.shape) != (batch, ACTION_DIM):
        raise PolicyAdapterError(
            f"{name} 形状应为 ({batch}, {ACTION_DIM})，实际 {tuple(tensor.shape)}"
        )
    if tensor.device != device:
        raise PolicyAdapterError(f"{name} 设备 {tensor.device} 与动作均值 {device} 不一致")
    if not bool(torch.isfinite(tensor).all()):
        raise PolicyAdapterError(f"{name} 含 NaN/Inf")


def _check_event_scalar(
    name: str, tensor: object, *, batch: int, device: torch.device
) -> None:
    """(B,) float32、同设备、有限——event 级 log_prob/entropy 形状。

    显式要求一维逐行值：provider 返回逐维 (B,18)（未做 event 化）在此拒绝，
    本模块不静默求和或猜维度。
    """
    if not isinstance(tensor, torch.Tensor):
        raise PolicyAdapterError(f"{name} 必须是 torch.Tensor，实际 {type(tensor).__name__}")
    if tensor.dtype is not FLOAT_DTYPE:
        raise PolicyAdapterError(f"{name} dtype 应为 {FLOAT_DTYPE}，实际 {tensor.dtype}")
    if tensor.dim() != 1 or int(tensor.shape[0]) != batch:
        raise PolicyAdapterError(
            f"{name} 须为 event 级 ({batch},) 一维张量，实际 {tuple(tensor.shape)}"
            "（逐维 log_prob 未 event 化；本接口不求和补救）"
        )
    if tensor.device != device:
        raise PolicyAdapterError(f"{name} 设备 {tensor.device} 与动作均值 {device} 不一致")
    if not bool(torch.isfinite(tensor).all()):
        raise PolicyAdapterError(f"{name} 含 NaN/Inf")


@dataclass(frozen=True)
class MeanActionOutput:
    """mean 行为接口输出：raw 均值与 ±100 裁剪副本。

    这不是采样数据：没有与之对应的 old log probability、entropy 或标准差，
    不能充当 PPO 旧动作概率。裁剪副本仅供调用方按执行边界使用；本 adapter
    不执行环境动作。
    """

    action_mean: torch.Tensor     # (B,18) 网络动作均值（raw，未裁剪）
    clipped_action: torch.Tensor  # (B,18) clamp(-100,100)
    clip_limit: float = ACTION_CLIP_LIMIT


@dataclass(frozen=True)
class PolicySampledAction:
    """``sample`` 的成组输出：采样 raw 动作 + 其 log_prob + 分布身份。

    raw_action 是**未裁剪**采样动作；log_prob 严格对应该 raw 值（不对应 clip
    后的值或 PD 目标角）。调用方若将动作送入环境须自行裁剪并另行记录；本
    adapter 不提供任何环境执行路径。
    """

    raw_action: torch.Tensor        # (B,18) float32
    log_prob: torch.Tensor          # (B,) float32 event 级
    distribution_identity: Any      # 构造时身份的规范副本（JSON 反冻结）
    identity_frozen: str            # 规范冻结字符串（与评估输出同源）


@dataclass(frozen=True)
class ActionEvaluation:
    """``evaluate_actions`` 的输出（梯度保留，供未来 PPO 侧使用）。

    action_mean 为网络前向原值（未 detach）；log_prob/entropy 由 provider 在
    该图上计算，梯度同时流向网络与 provider 参数。log_prob 对应传入的未裁剪
    raw 动作。
    """

    action_mean: torch.Tensor   # (B,18)
    log_prob: torch.Tensor      # (B,)
    entropy: torch.Tensor       # (B,)
    identity_frozen: str        # provider 身份的规范冻结字符串


class ConfinedPolicyAdapter(nn.Module):
    """ConfinedLocomotionNetwork 的训练侧薄包装（单一网络实例 + 可选 provider）。

    构造参数（均无生产默认）：
      network                已存在的 ConfinedLocomotionNetwork（原样注册一次；
                             不重建、不拷贝权重、不改训练/eval 模式）；
      distribution_provider  可选的外部注入 nn.Module：``forward(action_mean)``
                             只拿 (B,18) 均值，返回满足 event 级
                             mean/sample/log_prob/entropy 契约的分布对象；
      distribution_identity  provider 的非空可 JSON 冻结身份（与 provider 必须
                             同时提供或同时缺省）。
    """

    def __init__(
        self,
        *,
        network: ConfinedLocomotionNetwork,
        distribution_provider: nn.Module | None = None,
        distribution_identity: object = None,
    ) -> None:
        super().__init__()
        if not isinstance(network, ConfinedLocomotionNetwork):
            raise PolicyAdapterError(
                "network 必须为 ConfinedLocomotionNetwork（兼容别名 "
                "JointPassageNetwork 为同一类对象），实际 "
                f"{type(network).__name__}"
            )
        if (distribution_provider is None) != (distribution_identity is None):
            raise PolicyAdapterError(
                "distribution_provider 与 distribution_identity 必须同时提供或同时"
                "缺省（无身份的 provider 与无 provider 的身份都拒绝）"
            )
        # 网络先注册（单一实例）；provider 随后作为独立子模块注册一次。
        self.network = network
        if distribution_provider is None:
            self.distribution_provider = None
            self._identity_frozen: str | None = None
        else:
            if not isinstance(distribution_provider, nn.Module):
                raise PolicyAdapterError(
                    "distribution_provider 必须是 nn.Module（外部注入的有身份"
                    f"模块），实际 {type(distribution_provider).__name__}"
                )
            self._identity_frozen = _freeze_identity(distribution_identity)
            self.distribution_provider = distribution_provider

    # ── 身份视图 ─────────────────────────────────────────────────────────────

    @property
    def has_distribution_provider(self) -> bool:
        return self.distribution_provider is not None

    @property
    def distribution_identity(self) -> Any:
        """身份的规范独立副本（JSON 反冻结；每次返回新对象，防别名修改）。"""
        if self._identity_frozen is None:
            return None
        return json.loads(self._identity_frozen)

    @property
    def identity_frozen(self) -> str | None:
        return self._identity_frozen

    def extra_repr(self) -> str:
        if self._identity_frozen is None:
            return "distribution_provider=None (mean-only; sampling/evaluate/log_prob refuse)"
        digest = hashlib.sha256(self._identity_frozen.encode("utf-8")).hexdigest()[:12]
        return f"distribution_identity_sha256_prefix={digest}"

    # ── 前向（精确复用网络；无二次缩放、无标签入参） ─────────────────────────

    def forward(
        self,
        observation: OnlineObservation,
        near_points: torch.Tensor,
        near_valid: torch.Tensor,
        far_points: torch.Tensor,
        far_valid: torch.Tensor,
    ) -> NetworkOutput:
        """与 ``ConfinedLocomotionNetwork.forward`` 完全一致的纯前向。

        输入契约、63/3 维倍率施加、detach(Ê) 边界与可选 ground_pred 全部由
        网络自身负责；本方法不添加任何变换。输出为网络的 ``NetworkOutput``
        （envelope/ground_pred 未 detach，供未来辅助路径）。
        """
        return self.network(
            observation, near_points, near_valid, far_points, far_valid
        )

    def mean_actions(
        self,
        observation: OnlineObservation,
        near_points: torch.Tensor,
        near_valid: torch.Tensor,
        far_points: torch.Tensor,
        far_valid: torch.Tensor,
    ) -> MeanActionOutput:
        """独立 mean 行为接口：raw 均值 + ±100 裁剪副本（无任何概率假值）。"""
        output = self.network(
            observation, near_points, near_valid, far_points, far_valid
        )
        return MeanActionOutput(
            action_mean=output.action_mean,
            clipped_action=output.action_mean.clamp(
                -ACTION_CLIP_LIMIT, ACTION_CLIP_LIMIT
            ),
        )

    def reset(self) -> None:
        """显式无操作：本 adapter 无跨控制步隐藏状态。

        网络 GRU 每次 forward 从零隐藏开始，不存在需要清空的 carry；本方法
        不抹除/重建共享网络或 provider 参数（权重逐位不变），仅为满足「带
        reset 口的训练侧包装」接口习惯而保留。
        """
        return None

    # ── 参数视图 ─────────────────────────────────────────────────────────────

    def unique_parameters(self) -> tuple[nn.Parameter, ...]:
        """按 id 一次去重的参数视图：网络 + 显式 provider 参数。

        网络在self上只注册一份（无重复子模块）；provider 参数经注册自然纳入
        ``self.parameters()``。供未来优化器构建使用；本模块自身不建 optimizer。
        """
        unique: dict[int, nn.Parameter] = {}
        for parameter in self.parameters():
            unique.setdefault(id(parameter), parameter)
        return tuple(unique.values())

    # ── distribution-provider 接入口（无 provider 时显式拒绝） ────────────────

    def _require_provider(self, api: str) -> nn.Module:
        provider = self.distribution_provider
        if provider is None:
            raise PolicyAdapterError(
                f"{api} 需要 distribution_provider：本 adapter 构造时未注入任何"
                "分布 provider（生产不提供默认分布族/std/探索；mean 不是采样数据）。"
                "待训练动作分布确认后由外部显式注入有身份的 provider"
            )
        return provider

    def _distribution(
        self, provider: nn.Module, action_mean: torch.Tensor
    ) -> object:
        """调用 provider 并校验其分布对象的 event 级契约与均值一致性。"""
        distribution = provider(action_mean)
        mean = getattr(distribution, "mean", None)
        batch = int(action_mean.shape[0])
        _check_action_block(
            "provider 分布 mean", mean, batch=batch, device=action_mean.device
        )
        if not bool(
            torch.allclose(mean, action_mean, rtol=0.0, atol=_PROVIDER_MEAN_ATOL)
        ):
            raise PolicyAdapterError(
                "provider 分布 mean 与网络 action_mean 不一致（容差 "
                f"{_PROVIDER_MEAN_ATOL}）：拒绝第二均值头或经变形/压算的均值接入；"
                "provider 须以传入的 (B,18) action_mean 为分布中心"
            )
        for method in ("sample", "log_prob", "entropy"):
            if not callable(getattr(distribution, method, None)):
                raise PolicyAdapterError(
                    f"provider 返回的分布对象缺少可调用的 {method}()"
                )
        return distribution

    def sample(
        self,
        observation: OnlineObservation,
        near_points: torch.Tensor,
        near_valid: torch.Tensor,
        far_points: torch.Tensor,
        far_valid: torch.Tensor,
    ) -> PolicySampledAction:
        """provider 采样（成组返回 raw/log_prob/身份；无 provider 时拒绝）。

        log_prob 严格对应返回的未裁剪 raw 采样值。采样动作**不会**被本接口
        送入任何环境（本类没有环境执行路径）；采样数据必须保留其 raw 动作、
        log_prob 和分布身份，不能替换为 mean-only 输出。
        """
        provider = self._require_provider("sample")
        output = self.network(
            observation, near_points, near_valid, far_points, far_valid
        )
        batch = int(output.action_mean.shape[0])
        distribution = self._distribution(provider, output.action_mean)
        raw_action = distribution.sample()
        _check_action_block(
            "sampled raw_action", raw_action, batch=batch, device=output.action_mean.device
        )
        log_prob = distribution.log_prob(raw_action)
        _check_event_scalar(
            "sampled log_prob", log_prob, batch=batch, device=output.action_mean.device
        )
        return PolicySampledAction(
            raw_action=raw_action,
            log_prob=log_prob,
            distribution_identity=json.loads(self._identity_frozen),
            identity_frozen=self._identity_frozen,
        )

    def evaluate_actions(
        self,
        observation: OnlineObservation,
        near_points: torch.Tensor,
        near_valid: torch.Tensor,
        far_points: torch.Tensor,
        far_valid: torch.Tensor,
        raw_actions: torch.Tensor,
    ) -> ActionEvaluation:
        """provider 评估（log_prob/entropy；梯度保留；无 provider 时拒绝）。

        raw_actions 为**未裁剪** raw 动作 (B,18)：log_prob 对应该 raw 值，不
        对应 ±100 clip 后的值，也不对应 PD 目标角。action_mean 与 provider
        输出保持在网络前向计算图上（无错误 detach），梯度可同时回传网络与
        provider 参数。
        """
        provider = self._require_provider("evaluate_actions")
        output = self.network(
            observation, near_points, near_valid, far_points, far_valid
        )
        batch = int(output.action_mean.shape[0])
        _check_action_block(
            "raw_actions", raw_actions, batch=batch, device=output.action_mean.device
        )
        distribution = self._distribution(provider, output.action_mean)
        log_prob = distribution.log_prob(raw_actions)
        entropy = distribution.entropy()
        _check_event_scalar(
            "log_prob", log_prob, batch=batch, device=output.action_mean.device
        )
        _check_event_scalar(
            "entropy", entropy, batch=batch, device=output.action_mean.device
        )
        return ActionEvaluation(
            action_mean=output.action_mean,
            log_prob=log_prob,
            entropy=entropy,
            identity_frozen=self._identity_frozen,
        )

    def log_prob_actions(
        self,
        observation: OnlineObservation,
        near_points: torch.Tensor,
        near_valid: torch.Tensor,
        far_points: torch.Tensor,
        far_valid: torch.Tensor,
        raw_actions: torch.Tensor,
    ) -> torch.Tensor:
        """仅取 log_prob 的薄入口（契约与 ``evaluate_actions`` 一致）。"""
        return self.evaluate_actions(
            observation,
            near_points,
            near_valid,
            far_points,
            far_valid,
            raw_actions,
        ).log_prob
