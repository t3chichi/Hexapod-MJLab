"""P3/P8 单 Adam 联合 PPO 与辅助更新算法（纯 Torch；无 AMP/std clamp）。

仅标准库与 PyTorch；不导入 mjlab/任务模块（真实采集薄层属 P2 任务侧，
P4 runner 注册不在本批）。依据 docs/tasks/confined_locomotion/平地学习与
基准测量.md 第 3 节、docs/research/训练架构对齐.md 第 5 节、
docs/tasks/confined_locomotion/training/首版训练实施计划.md 4.4 与 P2 验收：

* **一个**类型化 frozen ``ConfinedPPOConfig`` 绑定已确认配方：24 steps/
  5 epoch/4 minibatch、clip 0.2、γ0.99/λ0.95、value 1/entropy 0.01/
  max-grad-norm 1、单 Adam lr 1e-3、β(0.9,0.999)、ε1e-8、weight decay 0、
  desired KL 0.01、lr 范围 [1e-5,1e-2]、两辅助 λ=1。任何字段偏离已确认
  值在构造时显式拒绝——不能通过 config 隐藏改变 confirmed 配方。
* ``ConfinedPPOTrainer`` 构造传入**已有** ``ConfinedPolicyAdapter``（内含
  唯一网络与生产 ``IndependentDiagonalNormalProvider``）+ 显式 grid hash 与
  seed：``unique_parameters()`` 按 id 去重覆盖网络全部头与 log_std，每个参数
  只进**唯一一个** Adam 一次（不先建参考 actor/critic 两套再替换）。设备与
  FP32 显式校验，不偷偷 cast/move 输入、不冻结模块、不加 running norm。
  正式 update 必须 CUDA（``allow_cpu=True`` 仅供解析/CPU 检查构造，CPU 通过
  不冒充 GPU 门）。
* ``update(buffer)`` 前置校验：完整、未消费、设备/M/grid/provider 身份与
  **采集模型指纹**一致（纯 rl 本地 ``policy_state_fingerprint``，与采集层
  语义逐位一致；不 import 任务模块）——优化开始前权重与采集期一致。
  优势在完整 N=T·B 上**一次**归一化（torch.std correction=1 + 1e-8，N>1），
  minibatch 只切行、不重新归一化、不覆盖原 GAE。每 epoch 由命名派生的独立
  ``torch.Generator``（流名 ``optimization_shuffle``，与行为/命令 PRNG 分离；
  state 可存取供 P5）randperm(N)，分 4 个互斥完整 minibatch（N 不被 4 整除
  显式拒绝，不丢样本）；每 epoch 4 个地面桶独立配对（bucket_partition 按
  epoch 轮转，索引指向全部 frames()）。
* 每 minibatch：**一次**网络完整 forward（PPO/value/包络共用），加地面桶
  独立来源 fresh near_encoder→ground_head 前向（不缓存特征/预测、不跑不
  需要的 far/形态/command）；总损失 L_policy + 1·L_value − 0.01·H̄ + 1·L_E +
  1·L_H。zero_grad → **一次**总 loss backward → **一次**去重全参数联合
  norm=1 裁剪 → **一次** Adam.step（共 5×4=20 次；不分别裁 actor/critic/
  辅助、不另建辅助 optimizer）。KL 用精确 torch.distributions 语义
  （losses.kl_divergence_old_new），每 minibatch 均值调度同一 Adam 的全部
  参数组 lr（>0.02 ÷1.5、0< <0.005 ×1.5、否则保持；完整 clamp [1e-5,1e-2]），
  无 early-stop，**绝不**调用 distribution.sample 或推进行为 generator。
* 有限性检查覆盖 forward/全部损失/ratio/KL、裁剪前后梯度、step 后参数与
  Adam 状态；exp(std/logprob) 溢出或非有限立即抛错——不删数据、不 clamp
  std、不重试优化。失败 stage 明确（``PPOUpdateFault.stage``），可配
  ``on_fault(stage, error)`` 回调供 P4 保存现场；部分执行的 optimizer 状态
  不是稳定完整 checkpoint。失败算法标 faulted 禁止继续 update，**不清 buffer**。
  成功后 consume+clear（边界为空）。
* 更新记录为可 serializable metrics + 有限小诊断（epoch/minibatch 索引、
  bucket 来源 index/key、|S| 与帧有效位置数、old/new mean/std/logprob/value、
  归一化 adv/returns、各损失、KL、LR 前后、联合梯度 norm 裁剪前后、参数
  身份数量、实际 optimizer step/update 计数），反映实际执行；正常更新不
  留完整参数/梯度 20 份、不额外调用 autograd.grad 做分量探针。显式
  ``observer(stage, payload)`` 诊断钩子供验收方按阶段即时 clone 所需数组/
  梯度/state（由测试/检查使用，不属于生产行为）。
* ``state_dict``/``load_state_dict`` 只覆盖**算法局部**状态（Adam、LR、计数、
  shuffle PRNG），不声称完整环境恢复已完成（P5 职责）。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from typing import Any, Callable, ClassVar

import torch
from torch import nn

from .contracts import FLOAT_DTYPE
from .distribution import IndependentDiagonalNormalProvider
from .losses import (
    ADVANTAGE_NORM_EPS,
    DESIRED_KL_STARTUP,
    ENVELOPE_LOSS_COEFF,
    ENTROPY_COEFF,
    GROUND_LOSS_COEFF,
    LR_MAX,
    LR_MIN,
    PPO_CLIP_EPSILON,
    VALUE_LOSS_COEFF,
    adjust_learning_rate,
    envelope_sum_loss,
    ground_bucket_loss,
    kl_divergence_old_new,
    normalize_rollout_advantage,
    ppo_policy_loss,
    require_finite,
    total_joint_loss,
    value_clipped_loss,
)
from .network import ConfinedLocomotionNetwork
from .policy import ConfinedPolicyAdapter
from .training_rollout import (
    GAE_GAMMA,
    GAE_LAMBDA,
    GROUND_NUM_BUCKETS,
    NOMINAL_ROLLOUT_STEPS,
    TrainingRolloutBuffer,
)
from .symmetry import (
    MIRROR_MAPPING_SCHEMA_VERSION,
    GridPermutation,
    JointPermutation,
    MirrorMappingError,
    mirror_action_distribution,
    mirror_envelope_values,
    mirror_ground_positions,
    mirror_ground_values,
    mirror_mean_mse,
    mirror_observation,
    mirror_point_block,
)

__all__ = [
    "ADAM_BETAS",
    "ADAM_EPS",
    "ADAM_WEIGHT_DECAY",
    "INITIAL_LEARNING_RATE",
    "MAX_GRAD_NORM",
    "NUM_EPOCHS",
    "NUM_MINIBATCHES",
    "PPO_CONFIG_SCHEMA_VERSION",
    "SHUFFLE_GENERATOR_STREAM",
    "ConfinedPPOConfig",
    "ConfinedPPOTrainer",
    "MirrorTrainingConfig",
    "PPOUpdateError",
    "PPOUpdateFault",
    "derive_shuffle_generator_seed",
    "policy_state_fingerprint",
]

#: 算法身份 schema 版本（state_dict 兼容比对用；语义变化必须 bump）。
PPO_CONFIG_SCHEMA_VERSION = "confined_locomotion.ppo/v2"
SHUFFLE_SEED_SCHEMA = "confined_locomotion.ppo/v1"

#: 已确认优化轮数 / minibatch 数（5×4=20 次联合更新 / rollout）。
NUM_EPOCHS = 5
NUM_MINIBATCHES = 4

#: 单 Adam 起步配置（权威：平地学习与基准测量.md「PPO 与优化器起步配方」）。
INITIAL_LEARNING_RATE = 1e-3
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
ADAM_WEIGHT_DECAY = 0.0

#: 联合梯度范数上限（去重全参数一次裁剪）。
MAX_GRAD_NORM = 1.0

#: 优化 shuffle 随机流的显式命名（PRNG 分流：独立于行为/命令/传感流）。
SHUFFLE_GENERATOR_STREAM = "optimization_shuffle"

#: 已确认采集长度（与 buffer 名义步数一致性校验用）。
ROLLOUT_STEPS = NOMINAL_ROLLOUT_STEPS

#: 地面桶数（与 PPO minibatch 配对；training_rollout 冻结为 4）。
GROUND_BUCKETS = GROUND_NUM_BUCKETS

#: Adam 状态/诊断里需要有限性检查的键。
_ADAM_TENSOR_STATE_KEYS = ("exp_avg", "exp_avg_sq", "max_exp_avg_sq", "step")

#: observer 阶段名（payload 均为**活引用**，由观察方按需 clone）。
_STAGE_UPDATE_STARTED = "update_started"
_STAGE_MINIBATCH_FORWARD = "minibatch_forward"
_STAGE_GRAD_PRE_CLIP = "grad_pre_clip"
_STAGE_GRAD_POST_CLIP = "grad_post_clip"
_STAGE_ADAM_STEP_DONE = "adam_step_done"
_STAGE_UPDATE_SUCCEEDED = "update_succeeded"


class PPOUpdateError(ValueError):
    """PPO 更新契约违规（配置/身份/设备/形状/整除性等，未触及优化器状态）。"""


class PPOUpdateFault(RuntimeError):
    """更新执行中的故障（forward/损失/梯度/step 非有限等；现场不稳定的部分执行）。

    ``stage`` 精确到 epoch/minibatch 与步骤名；optimizer 部分执行的状态不是
    稳定完整 checkpoint。抛出前算法已标 faulted 并调用 ``on_fault``。
    """

    def __init__(self, stage: str, message: str) -> None:
        self.stage = str(stage)
        super().__init__(f"[{stage}] {message}")


def derive_shuffle_generator_seed(base_seed: int) -> int:
    """优化 shuffle generator seed 的显式命名派生（uint32 安全）。

    仅依赖 (schema, 流名, base_seed) 的稳定散列；流名 ``optimization_shuffle``
    与行为采样流 ``policy_action`` 不同，同一 base_seed 下两流 seed 必不同，
    与 num_envs/构造顺序无关。
    """
    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise PPOUpdateError(f"base_seed 必须为 int，实际 {base_seed!r}")
    material = f"{SHUFFLE_SEED_SCHEMA}|{SHUFFLE_GENERATOR_STREAM}|{int(base_seed)}".encode("utf-8")
    digest = hashlib.sha256(material).digest()
    return int.from_bytes(digest[:4], "big")


def policy_state_fingerprint(adapter: ConfinedPolicyAdapter) -> str:
    """adapter 参数的确定性指纹（纯 rl 本地实现；不 import 任务 fingerprint）。

    语义与采集层一致：按参数名排序，逐项 name + shape + dtype + 字节做
    sha256。只用于「采集期间权重不变」与身份核对，不是加密摘要，也不进入
    任何更新输入。
    """
    digest = hashlib.sha256()
    for name, parameter in sorted(adapter.named_parameters(), key=lambda item: item[0]):
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(parameter.shape)).encode("utf-8"))
        digest.update(str(parameter.dtype).encode("utf-8"))
        digest.update(parameter.detach().reshape(-1).cpu().numpy().tobytes())
    return digest.hexdigest()


def _generator_state_sha256(generator: torch.Generator) -> str:
    return hashlib.sha256(generator.get_state().cpu().numpy().tobytes()).hexdigest()


# ═══════════════════════════ 类型化 frozen 配置 ═══════════════════════════


@dataclass(frozen=True)
class MirrorTrainingConfig:
    """Typed P8 augmentation switch; the mean-consistency coefficient is fixed."""

    enabled: bool = False
    mean_loss_coeff: float = 0.6

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise PPOUpdateError(f"symmetry.enabled must be bool, got {self.enabled!r}")
        if isinstance(self.mean_loss_coeff, bool) or not isinstance(
            self.mean_loss_coeff, (int, float)
        ) or float(self.mean_loss_coeff) != 0.6:
            raise PPOUpdateError(
                "symmetry.mean_loss_coeff is fixed at 0.6 for this accepted P8 recipe"
            )

    def summary(self) -> dict:
        return {
            "schema": "confined_locomotion.symmetry_training/v1",
            "enabled": self.enabled,
            "mean_loss_coeff": 0.6,
            "mapping_schema": MIRROR_MAPPING_SCHEMA_VERSION,
        }

    @classmethod
    def from_dict(cls, value: dict) -> "MirrorTrainingConfig":
        if not isinstance(value, dict) or set(value) != {"enabled", "mean_loss_coeff"}:
            raise PPOUpdateError(
                "symmetry must contain exactly enabled and mean_loss_coeff"
            )
        return cls(**value)


@dataclass(frozen=True)
class ConfinedPPOConfig:
    """已确认 PPO/优化器配方的类型化 frozen 绑定（不可经 config 改变语义）。

    所有字段必须等于已确认起步值（平地学习与基准测量.md 第 3 节）；
    ``__post_init__`` 逐项校验，任何偏离立即抛 ``PPOUpdateError``——新增
    字段或改值须先改权威文档并 bump schema，不能借构造参数隐藏变更。
    """

    rollout_steps: int = ROLLOUT_STEPS
    num_epochs: int = NUM_EPOCHS
    num_minibatches: int = NUM_MINIBATCHES
    clip_epsilon: float = PPO_CLIP_EPSILON
    gamma: float = GAE_GAMMA
    lam: float = GAE_LAMBDA
    value_loss_coeff: float = VALUE_LOSS_COEFF
    entropy_coeff: float = ENTROPY_COEFF
    max_grad_norm: float = MAX_GRAD_NORM
    learning_rate: float = INITIAL_LEARNING_RATE
    adam_beta1: float = ADAM_BETAS[0]
    adam_beta2: float = ADAM_BETAS[1]
    adam_eps: float = ADAM_EPS
    weight_decay: float = ADAM_WEIGHT_DECAY
    desired_kl: float = DESIRED_KL_STARTUP
    lr_min: float = LR_MIN
    lr_max: float = LR_MAX
    envelope_loss_coeff: float = ENVELOPE_LOSS_COEFF
    ground_loss_coeff: float = GROUND_LOSS_COEFF

    #: (字段名, 已确认值) 全集；__post_init__ 逐项强等（非 dataclass 字段）。
    _CONFIRMED: ClassVar[tuple[tuple[str, object], ...]] = (
        ("rollout_steps", ROLLOUT_STEPS),
        ("num_epochs", NUM_EPOCHS),
        ("num_minibatches", NUM_MINIBATCHES),
        ("clip_epsilon", PPO_CLIP_EPSILON),
        ("gamma", GAE_GAMMA),
        ("lam", GAE_LAMBDA),
        ("value_loss_coeff", VALUE_LOSS_COEFF),
        ("entropy_coeff", ENTROPY_COEFF),
        ("max_grad_norm", MAX_GRAD_NORM),
        ("learning_rate", INITIAL_LEARNING_RATE),
        ("adam_beta1", ADAM_BETAS[0]),
        ("adam_beta2", ADAM_BETAS[1]),
        ("adam_eps", ADAM_EPS),
        ("weight_decay", ADAM_WEIGHT_DECAY),
        ("desired_kl", DESIRED_KL_STARTUP),
        ("lr_min", LR_MIN),
        ("lr_max", LR_MAX),
        ("envelope_loss_coeff", ENVELOPE_LOSS_COEFF),
        ("ground_loss_coeff", GROUND_LOSS_COEFF),
    )

    def __post_init__(self) -> None:
        for field, confirmed in self._CONFIRMED:
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise PPOUpdateError(f"{field} 须为数值，实际 {value!r}")
            if float(value) != float(confirmed):
                raise PPOUpdateError(
                    f"ConfinedPPOConfig.{field}={value!r} 偏离已确认配方值 {confirmed!r}"
                    "（不能通过 config 隐藏改变 confirmed 配方；变更须先改权威文档并 bump schema）"
                )
        if self.gamma != GAE_GAMMA or self.lam != GAE_LAMBDA:
            raise PPOUpdateError("gamma/lam 必须与 buffer GAE 常数一致")

    def summary(self) -> dict:
        """可 serializable 摘要（metrics/recipe 记录用）。"""
        return {
            "schema": PPO_CONFIG_SCHEMA_VERSION,
            "rollout_steps": self.rollout_steps,
            "num_epochs": self.num_epochs,
            "num_minibatches": self.num_minibatches,
            "clip_epsilon": self.clip_epsilon,
            "gamma": self.gamma,
            "lam": self.lam,
            "value_loss_coeff": self.value_loss_coeff,
            "entropy_coeff": self.entropy_coeff,
            "max_grad_norm": self.max_grad_norm,
            "learning_rate": self.learning_rate,
            "adam_betas": (self.adam_beta1, self.adam_beta2),
            "adam_eps": self.adam_eps,
            "weight_decay": self.weight_decay,
            "desired_kl": self.desired_kl,
            "lr_min": self.lr_min,
            "lr_max": self.lr_max,
            "envelope_loss_coeff": self.envelope_loss_coeff,
            "ground_loss_coeff": self.ground_loss_coeff,
            "mirror": None,
            "amp": False,
            "std_clamp": None,
        }


# ═══════════════════════════ 单 Adam 联合更新算法 ═══════════════════════════


class ConfinedPPOTrainer:
    """P3 联合更新算法：单 Adam、一次联合 backward/裁剪/step、KL 自适应 lr。

    构造参数（均显式）：
      policy_adapter     已注入生产 provider 的 ``ConfinedPolicyAdapter``
                         （网络与 log_std 均为该 adapter 唯一注册实例）；
      grid_identity_hash 显式训练网格身份（须与 buffer 一致）；
      seed               配方 base seed（shuffle 流按命名派生，独立于行为流）；
      device             显式 torch.device；正式 update 必须 CUDA
                         （``allow_cpu=True`` 仅用于解析/CPU 检查构造）；
      config             可选 ``ConfinedPPOConfig``（默认即已确认配方）；
      allow_cpu          显式 CPU 解析构造开关（默认 False；CPU 通过不冒充 GPU）；
      observer           可选诊断观察回调 ``observer(stage, payload)``，payload
                         为活引用（数组/梯度/state），观察方自行即时 clone；
                         属验收证据通道，不属于生产行为；
      on_fault           可选故障回调 ``on_fault(stage, error)``（P4 保存现场；
                         回调异常不掩盖原始故障）。
    """

    def __init__(
        self,
        *,
        policy_adapter: ConfinedPolicyAdapter,
        grid_identity_hash: str,
        seed: int,
        device: torch.device | str,
        config: ConfinedPPOConfig | None = None,
        symmetry_config: MirrorTrainingConfig | None = None,
        joint_permutation: JointPermutation | None = None,
        grid_permutation: GridPermutation | None = None,
        mount_pos_b: torch.Tensor | None = None,
        mount_quat_xyzw: torch.Tensor | None = None,
        mount_optical_witness: dict | None = None,
        allow_cpu: bool = False,
        observer: Callable[[str, dict], None] | None = None,
        on_fault: Callable[[str, BaseException], None] | None = None,
    ) -> None:
        if not isinstance(policy_adapter, ConfinedPolicyAdapter):
            raise PPOUpdateError(
                f"policy_adapter 必须为 ConfinedPolicyAdapter，实际 {type(policy_adapter).__name__}"
            )
        if not isinstance(policy_adapter.distribution_provider, IndependentDiagonalNormalProvider):
            raise PPOUpdateError(
                "policy_adapter 必须注入生产 IndependentDiagonalNormalProvider，实际 "
                f"{type(policy_adapter.distribution_provider).__name__}"
            )
        if policy_adapter.distribution_provider is None or not policy_adapter.has_distribution_provider:
            raise PPOUpdateError("policy_adapter 缺少 provider 身份（mean-only adapter 不能更新）")
        network = policy_adapter.network
        if not isinstance(network, ConfinedLocomotionNetwork):
            raise PPOUpdateError(f"adapter.network 必须为 ConfinedLocomotionNetwork，实际 {type(network).__name__}")
        if network.ground_samples is None:
            raise PPOUpdateError("网络未建地面头（ground_samples=None）：地面辅助无法联合更新")
        if not isinstance(grid_identity_hash, str) or not grid_identity_hash.strip():
            raise PPOUpdateError("grid_identity_hash 须为非空字符串")
        if isinstance(allow_cpu, bool) is False:
            raise PPOUpdateError("allow_cpu 须为 bool")
        if observer is not None and not callable(observer):
            raise PPOUpdateError("observer 须为可调用或 None")
        if on_fault is not None and not callable(on_fault):
            raise PPOUpdateError("on_fault 须为可调用或 None")
        resolved_config = config if config is not None else ConfinedPPOConfig()
        if not isinstance(resolved_config, ConfinedPPOConfig):
            raise PPOUpdateError(f"config 必须为 ConfinedPPOConfig，实际 {type(config).__name__}")
        resolved_symmetry = (
            symmetry_config if symmetry_config is not None else MirrorTrainingConfig()
        )
        if not isinstance(resolved_symmetry, MirrorTrainingConfig):
            raise PPOUpdateError(
                f"symmetry_config must be MirrorTrainingConfig, got {type(resolved_symmetry).__name__}"
            )
        if joint_permutation is not None and not isinstance(joint_permutation, JointPermutation):
            raise PPOUpdateError("joint_permutation must be a JointPermutation or None")
        if grid_permutation is not None and not isinstance(grid_permutation, GridPermutation):
            raise PPOUpdateError("grid_permutation must be a GridPermutation or None")
        if mount_optical_witness is not None and not isinstance(mount_optical_witness, dict):
            raise PPOUpdateError("mount_optical_witness must be a JSON-safe dict or None")
        if resolved_symmetry.enabled and any(value is None for value in (
            joint_permutation, grid_permutation, mount_pos_b, mount_quat_xyzw
        )):
            raise PPOUpdateError(
                "enabled symmetry requires name-derived joint/grid mappings and the actual lidar mount"
            )

        device = torch.device(device)
        if device.type != "cuda" and not allow_cpu:
            raise PPOUpdateError(
                f"正式联合更新必须在 CUDA 上执行（device={device}）；显式 CPU 解析构造"
                "须传入 allow_cpu=True，且 CPU 检查通过不冒充 GPU 门"
            )

        # 设备/FP32：全部参数（网络 + log_std）在显式设备上且 float32；不 cast/move。
        for name, parameter in policy_adapter.named_parameters():
            if parameter.device != device:
                raise PPOUpdateError(f"参数 {name} 在 {parameter.device}，要求 {device}（不静默迁移）")
            if parameter.dtype is not FLOAT_DTYPE:
                raise PPOUpdateError(f"参数 {name} dtype={parameter.dtype}，须 {FLOAT_DTYPE}（无 AMP）")

        # 去重参数视图：覆盖网络全部头 + provider.log_std，每参数恰一次。
        unique = policy_adapter.unique_parameters()
        network_unique = network.unique_parameters()
        provider = policy_adapter.distribution_provider
        expected = len(network_unique) + 1          # + log_std
        if len(unique) != expected:
            raise PPOUpdateError(
                f"去重参数数 {len(unique)} != 网络 {len(network_unique)} + log_std 1"
                "（参数覆盖不完整或重复注册）"
            )
        if id(provider.log_std) not in {id(p) for p in unique}:
            raise PPOUpdateError("provider.log_std 未入去重参数集（18 维可学习 std 必须联合更新）")
        named_ids = {id(p) for _, p in policy_adapter.named_parameters()}
        if named_ids != {id(p) for p in unique}:
            raise PPOUpdateError("去重参数集与 adapter.named_parameters 不一致（遗漏/多余参数）")

        self.config = resolved_config
        self.symmetry_config = resolved_symmetry
        self.joint_permutation = joint_permutation
        self.grid_permutation = grid_permutation
        self.mount_pos_b = self._mount_tensor(
            "mount_pos_b", mount_pos_b, (3,), device=device
        )
        self.mount_quat_xyzw = self._mount_tensor(
            "mount_quat_xyzw", mount_quat_xyzw, (4,), device=device
        )
        self.mount_optical_witness = dict(mount_optical_witness or {})
        self.symmetry_identity = self._make_symmetry_identity()
        self.policy_adapter = policy_adapter
        self.network = network
        self.provider = provider
        self.grid_identity_hash = str(grid_identity_hash)
        self.device = device
        self.allow_cpu = bool(allow_cpu)
        self.ground_samples = int(network.ground_samples)
        self._observer = observer
        self._on_fault = on_fault
        self._unique_parameters = tuple(unique)
        # 唯一一个 Adam：直接以去重参数构造（不先建参考 actor/critic 两套再替换）。
        self._adam = torch.optim.Adam(
            params=list(unique),
            lr=resolved_config.learning_rate,
            betas=(resolved_config.adam_beta1, resolved_config.adam_beta2),
            eps=resolved_config.adam_eps,
            weight_decay=resolved_config.weight_decay,
        )
        if len(self._adam.param_groups) != 1:
            raise PPOUpdateError("单 Adam 应恰有一个参数组（联合 lr 调度作用于全部组）")
        # 优化 shuffle 流：命名派生独立 generator（与行为/命令 PRNG 分离）。
        self._shuffle_generator = torch.Generator(device=device)
        self._shuffle_generator.manual_seed(derive_shuffle_generator_seed(seed))
        self._learning_rate = float(resolved_config.learning_rate)
        self._update_count = 0
        self._optimizer_step_count = 0
        self._faulted_stage: str | None = None
        self._fault_error: BaseException | None = None

    @staticmethod
    def _mount_tensor(
        name: str, value: torch.Tensor | None, shape: tuple[int, ...], *, device: torch.device
    ) -> torch.Tensor | None:
        if value is None:
            return None
        if not isinstance(value, torch.Tensor) or value.dtype is not FLOAT_DTYPE \
                or tuple(value.shape) != shape or value.device != device \
                or not bool(torch.isfinite(value).all()):
            raise PPOUpdateError(f"{name} must be finite float32 {shape} on {device}")
        return value.detach().clone()

    def _make_symmetry_identity(self) -> dict:
        def float_list(tensor):
            return None if tensor is None else [float(v) for v in tensor.detach().cpu().tolist()]

        joint = None if self.joint_permutation is None else {
            "joint_names_in_action_order": list(self.joint_permutation.joint_names),
            "indices": list(self.joint_permutation.indices),
            "same_sign_involution": True,
        }
        grid = None if self.grid_permutation is None else {
            "x_coords": list(self.grid_permutation.x_coords),
            "y_coords": list(self.grid_permutation.y_coords),
            "source_indices": list(self.grid_permutation.source_indices),
        }
        return {
            "config": self.symmetry_config.summary(),
            "mapping_schema": MIRROR_MAPPING_SCHEMA_VERSION,
            "joint_permutation": joint,
            "grid_permutation": grid,
            "mount_pos_b": float_list(self.mount_pos_b),
            "mount_quat_xyzw": float_list(self.mount_quat_xyzw),
            "mount_optical_witness": self.mount_optical_witness,
        }

    def config_identity(self) -> dict:
        """Strict serializable trainer identity, including the mirror on/off recipe."""
        return {
            "ppo": self.config.summary(),
            "symmetry": self.symmetry_config.summary(),
            "symmetry_mapping": self.symmetry_identity,
        }

    # ── 视图 ────────────────────────────────────────────────────────────────

    @property
    def optimizer(self) -> torch.optim.Adam:
        """唯一 Adam（验收检查单 optimizer/唯一参数用；不新增第二优化器）。"""
        return self._adam

    @property
    def unique_parameters(self) -> tuple[nn.Parameter, ...]:
        return self._unique_parameters

    @property
    def unique_parameter_count(self) -> int:
        return len(self._unique_parameters)

    @property
    def learning_rate(self) -> float:
        return float(self._learning_rate)

    @property
    def update_count(self) -> int:
        return self._update_count

    @property
    def optimizer_step_count(self) -> int:
        return self._optimizer_step_count

    @property
    def faulted(self) -> bool:
        return self._faulted_stage is not None

    @property
    def fault_stage(self) -> str | None:
        return self._faulted_stage

    @property
    def fault_error(self) -> BaseException | None:
        return self._fault_error

    def shuffle_generator_state(self) -> torch.Tensor:
        """优化 shuffle generator 状态独立副本（P5 保存恢复用）。"""
        return self._shuffle_generator.get_state().clone()

    def load_shuffle_generator_state(self, state: torch.Tensor) -> None:
        if not isinstance(state, torch.Tensor):
            raise PPOUpdateError(f"shuffle generator state 须为 torch.Tensor，实际 {type(state).__name__}")
        self._shuffle_generator.set_state(state.detach().cpu().clone())

    # ── 内部：故障与观察 ────────────────────────────────────────────────────

    def _notify(self, stage: str, payload: dict) -> None:
        if self._observer is None:
            return
        self._observer(stage, payload)

    def _fault(self, stage: str, error: BaseException) -> PPOUpdateFault:
        """标记 faulted（禁止继续 update、不清 buffer）并回调 on_fault。"""
        self._faulted_stage = str(stage)
        self._fault_error = error
        if self._on_fault is not None:
            try:
                self._on_fault(str(stage), error)
            except BaseException:  # 回调失败不掩盖原始故障
                pass
        return PPOUpdateFault(str(stage), f"{type(error).__name__}: {error}")

    # ── 内部：有限性守卫 ────────────────────────────────────────────────────

    def _require_finite_parameters(self, stage: str) -> None:
        for index, parameter in enumerate(self._unique_parameters):
            if not bool(torch.isfinite(parameter).all()):
                raise PPOUpdateError(f"参数槽 {index} 含 NaN/Inf（stage={stage}）")

    def _require_finite_gradients(self, stage: str) -> None:
        for parameter in self._unique_parameters:
            if parameter.grad is None:
                continue
            if not bool(torch.isfinite(parameter.grad).all()):
                raise PPOUpdateError(f"参数 {id(parameter)} 梯度含 NaN/Inf（stage={stage}）")

    def _require_finite_adam_state(self, stage: str) -> None:
        for parameter, state in self._adam.state.items():
            for key, value in state.items():
                if torch.is_tensor(value):
                    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                        raise PPOUpdateError(f"Adam state[{key}] 含 NaN/Inf（stage={stage}）")
                elif isinstance(value, float) and not math.isfinite(value):
                    raise PPOUpdateError(f"Adam state[{key}] 非有限（stage={stage}）")

    def _joint_grad_norm(self) -> float:
        """联合梯度 L2 范数（与 clip_grad_norm_ 同式：逐参数 2-范数的 2-范数）。"""
        squares = []
        for parameter in self._unique_parameters:
            if parameter.grad is None:
                continue
            squares.append(torch.sum(parameter.grad * parameter.grad).reshape(()))
        if not squares:
            return 0.0
        return float(torch.sqrt(torch.stack(squares).sum()).item())

    # ── 主入口 ──────────────────────────────────────────────────────────────

    def update(self, buffer: TrainingRolloutBuffer) -> dict:
        """对一个完整 24 步 rollout 执行 5×4=20 次单 Adam 联合更新。

        成功：consume+clear（边界为空）并返回可 serializable metrics；
        失败：抛 ``PPOUpdateFault``（stage 明确、不清 buffer、算法标 faulted）。
        前置校验失败抛 ``PPOUpdateError``（未触及优化器状态、不标 faulted）。
        """
        if self.faulted:
            raise PPOUpdateFault(
                self._faulted_stage or "unknown",
                f"算法已 faulted（{self._faulted_stage}），禁止继续 update；不清 buffer",
            )
        try:
            self._require_finite_parameters("preflight/parameters")
            self._require_finite_adam_state("preflight/optimizer")
        except Exception as err:
            raise self._fault("preflight/finiteness", err) from err
        self._validate_buffer(buffer)

        n = buffer.nominal_steps * buffer.num_envs
        behavior_rng_before = self.provider.generator_state()
        fingerprint_before = policy_state_fingerprint(self.policy_adapter)
        frames_all = buffer.ground_frames.frames()
        metrics: dict[str, Any] = {
            "schema": PPO_CONFIG_SCHEMA_VERSION,
            "device": str(self.device),
            "nominal_steps": buffer.nominal_steps,
            "num_envs": buffer.num_envs,
            "num_samples": n,
            "unique_parameter_count": self.unique_parameter_count,
            "adam_param_group_count": len(self._adam.param_groups),
            "adam_param_count": sum(len(g["params"]) for g in self._adam.param_groups),
            "update_count_before": self._update_count,
            "optimizer_step_count_before": self._optimizer_step_count,
            "learning_rate_initial": self._learning_rate,
            "model_fingerprint_before": fingerprint_before,
            "distribution_identity_frozen": buffer.distribution_identity_frozen,
            "grid_identity_hash": buffer.ground_frames.grid_identity_hash,
            "behavior_generator_state_sha256_before": hashlib.sha256(
                behavior_rng_before.cpu().numpy().tobytes()
            ).hexdigest(),
            "config": self.config_identity(),
            "mirror_enabled": self.symmetry_config.enabled,
            "symmetry": self.symmetry_identity,
            "minibatches": [],
        }
        try:
            self._current_stage = "update_started"
            self._notify(_STAGE_UPDATE_STARTED, {"buffer_summary": buffer.summary(), "trainer": self})
            minibatch_metrics = self._run_epochs(buffer, n, frames_all, metrics)
        except BaseException as err:
            raise self._fault(self._current_stage, err) from err

        # 成功边界：consume（标记已消费）→ clear（下一 rollout 复用；边界为空）。
        buffer.consume()
        buffer.clear()
        self._update_count += 1
        metrics["update_count"] = self._update_count
        metrics["optimizer_steps_this_update"] = minibatch_metrics
        metrics["optimizer_step_count"] = self._optimizer_step_count
        metrics["learning_rate_final"] = self._learning_rate
        metrics["model_fingerprint_after"] = policy_state_fingerprint(self.policy_adapter)
        metrics["weights_changed"] = metrics["model_fingerprint_after"] != fingerprint_before
        metrics["behavior_generator_state_sha256_after"] = hashlib.sha256(
            self.provider.generator_state().cpu().numpy().tobytes()
        ).hexdigest()
        metrics["behavior_generator_unchanged"] = (
            metrics["behavior_generator_state_sha256_after"]
            == metrics["behavior_generator_state_sha256_before"]
        )
        metrics["shuffle_generator_state_sha256"] = _generator_state_sha256(self._shuffle_generator)
        self._require_finite_parameters("update_succeeded")
        self._notify(_STAGE_UPDATE_SUCCEEDED, {"metrics": metrics, "trainer": self})
        return metrics

    _current_stage: str = "validate"

    # ── 校验 ────────────────────────────────────────────────────────────────

    def _validate_buffer(self, buffer: TrainingRolloutBuffer) -> None:
        if not isinstance(buffer, TrainingRolloutBuffer):
            raise PPOUpdateError(
                f"buffer 必须为 TrainingRolloutBuffer，实际 {type(buffer).__name__}"
            )
        if not buffer.is_full:
            raise PPOUpdateError(
                f"要求完整 rollout（{self.config.rollout_steps} 步），实际 {buffer.steps_filled} 步"
                "（残缺数据只作证据；空/残缺 buffer 不更新）"
            )
        if buffer.summary()["engineering_stopped_steps"]:
            raise PPOUpdateError("工程跌倒停止的rollout禁止更新，收满也不能继续")
        if buffer.consumed:
            raise PPOUpdateError("buffer 已 consume（先 clear 再开始下一 rollout）")
        if buffer.device != self.device:
            raise PPOUpdateError(f"buffer 设备 {buffer.device} 与算法 {self.device} 不一致")
        if buffer.nominal_steps != self.config.rollout_steps:
            raise PPOUpdateError(
                f"buffer.nominal_steps={buffer.nominal_steps} != 配方 {self.config.rollout_steps}"
            )
        n = buffer.nominal_steps * buffer.num_envs
        if n <= 1:
            raise PPOUpdateError(f"优势归一化要求 N=T·B>1，实际 N={n}")
        if n % self.config.num_minibatches != 0:
            raise PPOUpdateError(
                f"N={n} 不能被 {self.config.num_minibatches} minibatch 整除（显式拒绝，不丢样本）"
            )
        if str(buffer.distribution_identity_frozen or "") != str(self.policy_adapter.identity_frozen):
            raise PPOUpdateError(
                "buffer 分布身份与当前 provider 不一致（rollout 属于另一分布/模型）"
            )
        if buffer.ground_frames.grid_identity_hash != self.grid_identity_hash:
            raise PPOUpdateError(
                f"buffer grid 身份 {buffer.ground_frames.grid_identity_hash} != 算法 "
                f"{self.grid_identity_hash}（网格身份必须一致）"
            )
        if int(buffer.ground_frames.ground_samples) != self.ground_samples:
            raise PPOUpdateError(
                f"buffer ground M={buffer.ground_frames.ground_samples} != 网络 {self.ground_samples}"
            )
        fingerprint = policy_state_fingerprint(self.policy_adapter)
        if buffer.model_fingerprint != fingerprint:
            raise PPOUpdateError(
                "采集模型指纹与当前参数不一致（采集期间权重必须不动；优化开始前须一致）"
            )

    # ── 5×4 epoch/minibatch 主循环 ──────────────────────────────────────────

    def _run_epochs(
        self, buffer: TrainingRolloutBuffer, n: int, frames_all: tuple, metrics: dict
    ) -> int:
        mb_size = n // self.config.num_minibatches
        # 完整 rollout 一次性优势归一化（唯一一次；minibatch 只切行）。
        advantage_flat = buffer.advantage.reshape(-1)
        normalized = normalize_rollout_advantage(advantage_flat)
        metrics["normalized_advantage"] = normalized.detach().cpu().tolist()
        metrics["advantage_normalization"] = {
            "mean": float(advantage_flat.mean().item()),
            "std_correction_1": float(torch.std(advantage_flat, correction=1).item()),
            "eps": ADVANTAGE_NORM_EPS,
            "applied_once_over_full_rollout": True,
        }
        steps_done = 0
        for epoch in range(self.config.num_epochs):
            permutation = torch.randperm(
                n, device=self.device, generator=self._shuffle_generator
            )
            buckets = buffer.ground_frames.bucket_partition(
                num_buckets=self.config.num_minibatches, epoch=epoch
            )
            supervision_size = sum(len(bucket) for bucket in buckets)
            metrics.setdefault("ground", {
                "total_source_frames": len(frames_all),
                "per_frame_valid_positions": [f.valid_position_count for f in frames_all],
                "bucket_sizes_per_epoch": [],
            })["bucket_sizes_per_epoch"].append([len(b) for b in buckets])
            metrics["ground"]["supervision_S"] = supervision_size
            for minibatch_index in range(self.config.num_minibatches):
                stage = f"epoch{epoch}/minibatch{minibatch_index}"
                self._current_stage = stage
                entry = self._run_minibatch(
                    buffer,
                    indices=permutation[minibatch_index * mb_size:(minibatch_index + 1) * mb_size],
                    normalized_advantage=normalized,
                    frames=frames_all,
                    bucket=buckets[minibatch_index],
                    supervision_size=supervision_size,
                    epoch=epoch,
                    minibatch=minibatch_index,
                    stage=stage,
                )
                metrics["minibatches"].append(entry)
                steps_done += 1
        if steps_done != self.config.num_epochs * self.config.num_minibatches:
            raise PPOUpdateError(
                f"联合更新次数 {steps_done} != {self.config.num_epochs * self.config.num_minibatches}"
            )
        return steps_done

    @staticmethod
    def _quat_rotate_xyzw(quat: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
        q_xyz = quat[:3].view(*([1] * (vectors.dim() - 1)), 3)
        q_w = quat[3].view(*([1] * (vectors.dim() - 1)), 1)
        uv = torch.cross(q_xyz.expand_as(vectors), vectors, dim=-1)
        uuv = torch.cross(q_xyz.expand_as(vectors), uv, dim=-1)
        return vectors + 2.0 * (q_w * uv + uuv)

    def _mirror_source_directions(
        self,
        points: torch.Tensor,
        valid: torch.Tensor,
        directions: torch.Tensor | None,
        present: torch.Tensor | None,
    ) -> torch.Tensor:
        """Use retained sensor rays; recover only safe non-zero legacy endpoints."""
        if self.mount_pos_b is None or self.mount_quat_xyzw is None:
            raise PPOUpdateError("mirror mapping has no actual optical lidar mount")
        if (directions is None) != (present is None):
            raise PPOUpdateError("source directions and their presence mask must travel together")
        if directions is None:
            directions = torch.zeros_like(points)
            present = torch.zeros_like(valid)
        if tuple(directions.shape) != tuple(points.shape) or directions.dtype is not FLOAT_DTYPE \
                or directions.device != points.device:
            raise PPOUpdateError("source direction shape/device/dtype differs from prepared points")
        if tuple(present.shape) != tuple(valid.shape) or present.dtype is not torch.bool \
                or present.device != points.device:
            raise PPOUpdateError("source direction presence shape/device/dtype differs from points")
        missing = valid & ~present
        result = directions.clone()
        if bool(missing.any()):
            displacement = points - self.mount_pos_b.view(1, 1, 3)
            ranges = torch.linalg.vector_norm(displacement, dim=-1)
            if bool((missing & (ranges <= 1e-8)).any()):
                raise PPOUpdateError(
                    "mirror augmentation needs retained source rays for zero-range points"
                )
            body_dirs = torch.zeros_like(displacement)
            body_dirs[missing] = displacement[missing] / ranges[missing].unsqueeze(-1)
            q_conjugate = torch.cat((-self.mount_quat_xyzw[:3], self.mount_quat_xyzw[3:4]))
            sensor_dirs = self._quat_rotate_xyzw(q_conjugate, body_dirs)
            result[missing] = sensor_dirs[missing]
        result = torch.where(valid.unsqueeze(-1), result, torch.zeros_like(result))
        return result

    def _mirror_point_inputs(
        self,
        points: torch.Tensor,
        valid: torch.Tensor,
        *,
        dirs: torch.Tensor | None,
        dirs_present: torch.Tensor | None,
        source_indices: torch.Tensor | None,
        scan_ids: torch.Tensor | None = None,
        times: torch.Tensor | None = None,
    ):
        if self.joint_permutation is None or self.grid_permutation is None \
                or self.mount_quat_xyzw is None:
            raise PPOUpdateError("mirror augmentation mapping is incomplete")
        source_dirs = self._mirror_source_directions(points, valid, dirs, dirs_present)
        source_order = torch.arange(
            points.shape[1], dtype=torch.int64, device=points.device
        ).expand(points.shape[0], -1)
        return mirror_point_block(
            points,
            valid,
            source_dirs,
            self.mount_quat_xyzw,
            scan_ids=scan_ids,
            times=times,
            source_indices=source_indices,
            source_order=source_order,
        )

    def _mirror_prepared(self, prepared):
        if self.joint_permutation is None:
            raise PPOUpdateError("mirror augmentation has no joint permutation")
        near = self._mirror_point_inputs(
            prepared.near_points, prepared.near_valid,
            dirs=prepared.near_source_dirs_sensor,
            dirs_present=prepared.near_source_dirs_present,
            source_indices=prepared.near_source_indices,
        )
        far = self._mirror_point_inputs(
            prepared.far_points, prepared.far_valid,
            dirs=prepared.far_source_dirs_sensor,
            dirs_present=prepared.far_source_dirs_present,
            source_indices=prepared.far_source_indices,
            scan_ids=prepared.far_point_scan_ids,
            times=prepared.far_point_times,
        )
        return replace(
            prepared,
            observation=mirror_observation(prepared.observation, self.joint_permutation),
            near_points=near.points,
            near_valid=near.valid,
            near_angle_keys=near.angle_keys,
            near_source_dirs_sensor=near.source_dirs_sensor,
            near_source_dirs_present=near.valid.clone(),
            near_source_indices=near.source_indices,
            far_points=far.points,
            far_valid=far.valid,
            far_angle_keys=far.angle_keys,
            far_source_dirs_sensor=far.source_dirs_sensor,
            far_source_dirs_present=far.valid.clone(),
            far_source_indices=far.source_indices,
            far_point_scan_ids=far.scan_ids,
            far_point_times=far.times,
        )

    def _mirror_ground_frame(self, frame):
        if self.grid_permutation is None:
            raise PPOUpdateError("mirror augmentation has no grid permutation")
        near = self._mirror_point_inputs(
            frame.near_points,
            frame.near_valid,
            dirs=frame.near_source_dirs_sensor,
            dirs_present=frame.near_source_dirs_present,
            source_indices=frame.near_source_indices,
        )
        heights, height_valid = mirror_ground_values(
            frame.heights.view(1, -1), frame.valid.view(1, -1), self.grid_permutation
        )
        positions, position_valid = mirror_ground_positions(
            frame.positions.view(1, self.ground_samples, 3),
            frame.valid.view(1, -1), self.grid_permutation,
        )
        if not torch.equal(height_valid, position_valid):
            raise PPOUpdateError("mirrored ground height/position validity differs")
        return replace(
            frame,
            near_points=near.points,
            near_valid=near.valid,
            near_angle_keys=near.angle_keys,
            near_source_dirs_sensor=near.source_dirs_sensor,
            near_source_dirs_present=near.valid.clone(),
            near_source_indices=near.source_indices,
            heights=heights[0],
            positions=positions[0],
            valid=height_valid[0],
        )

    def _run_minibatch(
        self,
        buffer: TrainingRolloutBuffer,
        *,
        indices: torch.Tensor,
        normalized_advantage: torch.Tensor,
        frames: tuple,
        bucket: list[int],
        supervision_size: int,
        epoch: int,
        minibatch: int,
        stage: str,
    ) -> dict:
        self._current_stage = f"{stage}/replay"
        batch = buffer.replay(indices)
        observation = batch.prepared.observation
        mirror_enabled = self.symmetry_config.enabled
        mirrored_batch = None
        old_mirrored = None
        old_mirror_log_prob = None

        # ── 一次网络完整 forward（PPO/value/包络共用；不二次前向）。
        self._current_stage = f"{stage}/forward"
        output = self.policy_adapter(
            observation,
            batch.prepared.near_points,
            batch.prepared.near_valid,
            batch.prepared.far_points,
            batch.prepared.far_valid,
        )
        self._current_stage = f"{stage}/forward"
        for name, value in vars(output).items():
            if torch.is_tensor(value):
                require_finite(f"{stage}/forward/{name}", value)
        # provider 只构造分布与概率：绝不 sample、绝不推进行为 generator。
        distribution = self.provider(output.action_mean)
        new_log_prob = distribution.log_prob(batch.raw_action)
        new_std = distribution.std
        new_mean = output.action_mean
        entropy = distribution.entropy()
        new_value = output.value.squeeze(-1)
        if tuple(new_value.shape) != tuple(batch.old_value.shape):
            raise PPOUpdateError(
                f"new_value squeeze 后形状 {tuple(new_value.shape)} != old_value "
                f"{tuple(batch.old_value.shape)}"
            )

        mirrored_output = None
        mirrored_distribution = None
        mirrored_new_log_prob = None
        mirrored_entropy = None
        mirrored_value = None
        mirror_mean_loss = None
        old_mirror_logprob_max_abs_diff = 0.0
        if mirror_enabled:
            self._current_stage = f"{stage}/mirror_prepare"
            mirrored_prepared = self._mirror_prepared(batch.prepared)
            mirrored_batch = replace(batch, prepared=mirrored_prepared)
            old_mirrored = mirror_action_distribution(
                batch.old_mean, batch.old_std, batch.raw_action, self.joint_permutation
            )
            old_mirror_log_prob = (
                -0.5 * ((old_mirrored.raw_action - old_mirrored.mean) / old_mirrored.std).square()
                - torch.log(old_mirrored.std)
                - 0.5 * math.log(2.0 * math.pi)
            ).sum(dim=1)
            old_mirror_logprob_max_abs_diff = float(
                (old_mirror_log_prob - batch.old_log_prob).abs().max().detach().item()
            )
            if not torch.allclose(
                old_mirror_log_prob.detach(), batch.old_log_prob, atol=1e-5, rtol=1e-5
            ):
                raise PPOUpdateError(
                    "permuted old Normal log probability changed under the mirror map "
                    f"(max abs diff={old_mirror_logprob_max_abs_diff})"
                )
            self._current_stage = f"{stage}/mirror_forward"
            mirrored_output = self.policy_adapter(
                mirrored_prepared.observation,
                mirrored_prepared.near_points,
                mirrored_prepared.near_valid,
                mirrored_prepared.far_points,
                mirrored_prepared.far_valid,
            )
            for name, value in vars(mirrored_output).items():
                if torch.is_tensor(value):
                    require_finite(f"{stage}/mirror_forward/{name}", value)
            # Current provider std remains in its learned action order. Only
            # the saved OLD distribution above is A-pushed-forward.
            mirrored_distribution = self.provider(mirrored_output.action_mean)
            mirrored_new_log_prob = mirrored_distribution.log_prob(old_mirrored.raw_action)
            mirrored_entropy = mirrored_distribution.entropy()
            mirrored_value = mirrored_output.value.squeeze(-1)
            mirror_mean_loss = mirror_mean_mse(
                mirrored_output.action_mean, output.action_mean, self.joint_permutation
            )
            if tuple(mirrored_value.shape) != tuple(batch.old_value.shape):
                raise PPOUpdateError("mirrored value shape differs from the original rollout batch")

        # ── PPO / 价值 / 熵（优势是完整 rollout 归一化结果的行切片）。
        self._current_stage = f"{stage}/policy_value_losses"
        advantage_slice = normalized_advantage[indices]
        loss_policy_original = ppo_policy_loss(new_log_prob, batch.old_log_prob, advantage_slice)
        loss_value_original = value_clipped_loss(new_value, batch.old_value, batch.returns)
        if mirror_enabled:
            loss_policy_mirrored = ppo_policy_loss(
                mirrored_new_log_prob, old_mirror_log_prob, advantage_slice
            )
            loss_value_mirrored = value_clipped_loss(
                mirrored_value, batch.old_value, batch.returns
            )
            loss_policy = 0.5 * (loss_policy_original + loss_policy_mirrored)
            loss_value = 0.5 * (loss_value_original + loss_value_mirrored)
            paired_entropy = 0.5 * (entropy + mirrored_entropy)
        else:
            loss_policy_mirrored = None
            loss_value_mirrored = None
            loss_policy = loss_policy_original
            loss_value = loss_value_original
            paired_entropy = entropy

        # ── 包络辅助（Ê 未 detach：辅助路径训练形态头与共享感知）。
        self._current_stage = f"{stage}/envelope_loss"
        loss_envelope_original = envelope_sum_loss(
            output.envelope, batch.envelope_values, batch.envelope_valid
        )
        if mirror_enabled:
            loss_envelope_mirrored = envelope_sum_loss(
                mirrored_output.envelope,
                mirror_envelope_values(batch.envelope_values),
                batch.envelope_valid,
            )
            loss_envelope = 0.5 * (loss_envelope_original + loss_envelope_mirrored)
        else:
            loss_envelope_mirrored = None
            loss_envelope = loss_envelope_original

        # ── 地面辅助：本 minibatch 配对桶的独立来源 fresh 近端前向。
        #     不缓存特征/预测、不跑 far/形态/command；空桶或 |S|=0 为 device 零。
        self._current_stage = f"{stage}/ground_loss"
        ground_forward_executed = False
        ground_predictions = None
        mirrored_ground_predictions = None
        mirrored_frames = None
        if supervision_size > 0 and len(bucket) > 0:
            bucket_frames = [frames[index] for index in bucket]
            near_points = torch.cat([frame.near_points for frame in bucket_frames], dim=0)
            near_valid = torch.cat([frame.near_valid for frame in bucket_frames], dim=0)
            heights = torch.stack([frame.heights for frame in bucket_frames], dim=0)
            valid = torch.stack([frame.valid for frame in bucket_frames], dim=0)
            near_feature = self.network.near_encoder(near_points, near_valid)
            require_finite(f"{stage}/ground/near_feature", near_feature)
            ground_predictions = self.network.ground_head(near_feature)
            require_finite(f"{stage}/ground/predictions", ground_predictions)
            loss_ground_original = ground_bucket_loss(
                ground_predictions, heights, valid, supervision_size,
                num_buckets=self.config.num_minibatches,
            )
            if mirror_enabled:
                self._current_stage = f"{stage}/mirror_ground_prepare"
                mirrored_frames = [self._mirror_ground_frame(frame) for frame in bucket_frames]
                mirrored_near_points = torch.cat(
                    [frame.near_points for frame in mirrored_frames], dim=0
                )
                mirrored_near_valid = torch.cat(
                    [frame.near_valid for frame in mirrored_frames], dim=0
                )
                mirrored_heights = torch.stack([frame.heights for frame in mirrored_frames])
                mirrored_valid = torch.stack([frame.valid for frame in mirrored_frames])
                mirrored_near_feature = self.network.near_encoder(
                    mirrored_near_points, mirrored_near_valid
                )
                require_finite(f"{stage}/mirror_ground/near_feature", mirrored_near_feature)
                mirrored_ground_predictions = self.network.ground_head(mirrored_near_feature)
                require_finite(f"{stage}/mirror_ground/predictions", mirrored_ground_predictions)
                loss_ground_mirrored = ground_bucket_loss(
                    mirrored_ground_predictions, mirrored_heights, mirrored_valid,
                    supervision_size, num_buckets=self.config.num_minibatches,
                )
                loss_ground = 0.5 * (loss_ground_original + loss_ground_mirrored)
            else:
                loss_ground_mirrored = None
                loss_ground = loss_ground_original
            ground_forward_executed = True
        else:
            loss_ground_original = new_value.new_zeros(())
            loss_ground_mirrored = new_value.new_zeros(()) if mirror_enabled else None
            loss_ground = new_value.new_zeros(())

        # ── 联合总损失（一次 backward 的唯一对象）。
        self._current_stage = f"{stage}/total_loss"
        loss_total = total_joint_loss(
            loss_policy,
            loss_value,
            paired_entropy,
            loss_envelope,
            loss_ground,
            value_coeff=self.config.value_loss_coeff,
            entropy_coeff=self.config.entropy_coeff,
            envelope_coeff=self.config.envelope_loss_coeff,
            ground_coeff=self.config.ground_loss_coeff,
        )
        if mirror_enabled:
            loss_total = loss_total + self.symmetry_config.mean_loss_coeff * mirror_mean_loss
            require_finite(f"{stage}/total_loss_with_mirror", loss_total)

        # KL（调度证据；detach 保存的精确 torch.distributions 语义）。
        kl_event_original = kl_divergence_old_new(
            batch.old_mean, batch.old_std, new_mean.detach(), new_std.detach()
        )
        if mirror_enabled:
            kl_event_mirrored = kl_divergence_old_new(
                old_mirrored.mean,
                old_mirrored.std,
                mirrored_output.action_mean.detach(),
                mirrored_distribution.std.detach(),
            )
            kl_event = torch.cat((kl_event_original, kl_event_mirrored), dim=0)
        else:
            kl_event_mirrored = None
            kl_event = kl_event_original
        kl_mean = float(kl_event.detach().mean().item())

        self._notify(
            _STAGE_MINIBATCH_FORWARD,
            {
                "epoch": epoch, "minibatch": minibatch, "stage": stage,
                "batch": batch, "network_output": output, "distribution": distribution,
                "mirror_enabled": mirror_enabled,
                "original_batch": batch,
                "mirrored_batch": mirrored_batch,
                "original_output": output,
                "mirrored_output": mirrored_output,
                "mirrored_distribution": mirrored_distribution,
                "old_mirrored": old_mirrored,
                "old_mirror_log_prob": old_mirror_log_prob,
                "old_mirror_logprob_max_abs_diff": old_mirror_logprob_max_abs_diff,
                "loss_policy": loss_policy, "loss_value": loss_value,
                "loss_envelope": loss_envelope, "loss_ground": loss_ground,
                "loss_policy_original": loss_policy_original,
                "loss_policy_mirrored": loss_policy_mirrored,
                "loss_value_original": loss_value_original,
                "loss_value_mirrored": loss_value_mirrored,
                "loss_envelope_original": loss_envelope_original,
                "loss_envelope_mirrored": loss_envelope_mirrored,
                "loss_ground_original": loss_ground_original,
                "loss_ground_mirrored": loss_ground_mirrored,
                "loss_mirror_mean": mirror_mean_loss,
                "loss_total": loss_total, "kl_event": kl_event,
                "kl_event_original": kl_event_original,
                "kl_event_mirrored": kl_event_mirrored,
                "indices": indices, "normalized_advantage": advantage_slice,
                "ground_predictions": ground_predictions, "ground_bucket": list(bucket),
                "mirrored_ground_predictions": mirrored_ground_predictions,
                "mirrored_ground_frames": mirrored_frames,
                "ground_supervision_size": supervision_size,
                "symmetry_identity": self.symmetry_identity,
            },
        )

        # ── KL 学习率调度（当前step前作用于同一 Adam 的全部参数组；无 early-stop）。
        lr_before = self._learning_rate
        lr_after = adjust_learning_rate(
            kl_mean, lr_before,
            desired_kl=self.config.desired_kl,
            lr_min=self.config.lr_min, lr_max=self.config.lr_max,
        )
        for group in self._adam.param_groups:
            group["lr"] = lr_after
        self._learning_rate = lr_after

        # ── zero_grad → 一次 backward → 一次联合裁剪 → 一次 step。
        self._current_stage = f"{stage}/backward"
        self._adam.zero_grad(set_to_none=True)
        loss_total.backward()
        self._current_stage = f"{stage}/grad_pre_clip"
        self._require_finite_gradients(self._current_stage)
        grad_norm_pre = self._joint_grad_norm()
        if not math.isfinite(grad_norm_pre):
            raise PPOUpdateError("联合梯度范数非有限，禁止裁剪掩盖溢出")
        self._notify(
            _STAGE_GRAD_PRE_CLIP,
            {"epoch": epoch, "minibatch": minibatch, "stage": stage,
             "parameters": self._unique_parameters, "grad_norm_pre_clip": grad_norm_pre},
        )
        # 一次去重全参数联合 norm 裁剪（不分别裁 actor/critic/辅助）。
        torch.nn.utils.clip_grad_norm_(
            self._unique_parameters, max_norm=self.config.max_grad_norm
        )
        self._current_stage = f"{stage}/grad_post_clip"
        self._require_finite_gradients(self._current_stage)
        grad_norm_post = self._joint_grad_norm()
        if not math.isfinite(grad_norm_post):
            raise PPOUpdateError("裁剪后联合梯度范数非有限")
        self._notify(
            _STAGE_GRAD_POST_CLIP,
            {"epoch": epoch, "minibatch": minibatch, "stage": stage,
             "parameters": self._unique_parameters,
             "grad_norm_post_clip": grad_norm_post},
        )
        self._current_stage = f"{stage}/adam_step"
        self._adam.step()
        self._optimizer_step_count += 1
        self._current_stage = f"{stage}/post_step"
        self._require_finite_parameters(self._current_stage)
        self._require_finite_adam_state(self._current_stage)

        self._notify(
            _STAGE_ADAM_STEP_DONE,
            {"epoch": epoch, "minibatch": minibatch, "stage": stage,
             "optimizer": self._adam, "parameters": self._unique_parameters,
             "learning_rate": lr_after, "kl_mean": kl_mean},
        )

        return {
            "epoch": epoch,
            "minibatch": minibatch,
            "stage_last": self._current_stage,
            "indices": indices.detach().cpu().tolist(),
            "loss_policy": float(loss_policy.detach().item()),
            "loss_value": float(loss_value.detach().item()),
            "entropy_mean": float(paired_entropy.detach().mean().item()),
            "loss_envelope": float(loss_envelope.detach().item()),
            "loss_ground": float(loss_ground.detach().item()),
            "loss_total": float(loss_total.detach().item()),
            "kl_mean": kl_mean,
            "mirror_enabled": mirror_enabled,
            "loss_policy_original": float(loss_policy_original.detach().item()),
            "loss_value_original": float(loss_value_original.detach().item()),
            "loss_envelope_original": float(loss_envelope_original.detach().item()),
            "loss_ground_original": float(loss_ground_original.detach().item()),
            "loss_policy_mirrored": (
                None if loss_policy_mirrored is None
                else float(loss_policy_mirrored.detach().item())
            ),
            "loss_value_mirrored": (
                None if loss_value_mirrored is None
                else float(loss_value_mirrored.detach().item())
            ),
            "loss_envelope_mirrored": (
                None if loss_envelope_mirrored is None
                else float(loss_envelope_mirrored.detach().item())
            ),
            "loss_ground_mirrored": (
                None if loss_ground_mirrored is None
                else float(loss_ground_mirrored.detach().item())
            ),
            "loss_mirror_mean": (
                None if mirror_mean_loss is None
                else float(mirror_mean_loss.detach().item())
            ),
            "mirror_mean_loss_coeff": self.symmetry_config.mean_loss_coeff,
            "old_mirror_logprob_max_abs_diff": old_mirror_logprob_max_abs_diff,
            "kl_mean_original": float(kl_event_original.detach().mean().item()),
            "kl_mean_mirrored": (
                None if kl_event_mirrored is None
                else float(kl_event_mirrored.detach().mean().item())
            ),
            "symmetry_identity": self.symmetry_identity,
            "lr_before": lr_before,
            "lr_after": lr_after,
            "grad_norm_pre_clip": grad_norm_pre,
            "grad_norm_post_clip": grad_norm_post,
            "optimizer_step_count": self._optimizer_step_count,
            "ground": {
                "bucket_size": len(bucket),
                "bucket_frame_indices": list(map(int, bucket)),
                "bucket_keys": [
                    list(frames[index].key) for index in bucket
                ],
                "bucket_per_frame_valid_positions": [
                    frames[index].valid_position_count for index in bucket
                ],
                "supervision_S": supervision_size,
                "ground_forward_executed": ground_forward_executed,
                "mirrored_ground_forward_executed": (
                    False if not mirror_enabled else ground_forward_executed
                ),
                "mirrored_bucket_keys": (
                    None if mirrored_frames is None
                    else [list(frame.key) for frame in mirrored_frames]
                ),
            },
            "old_log_prob": batch.old_log_prob.detach().cpu().tolist(),
            "new_log_prob": new_log_prob.detach().cpu().tolist(),
            "old_value": batch.old_value.detach().cpu().tolist(),
            "new_value": new_value.detach().cpu().tolist(),
            "normalized_advantage": advantage_slice.detach().cpu().tolist(),
            "returns": batch.returns.detach().cpu().tolist(),
            "old_mean": batch.old_mean.detach().cpu().tolist(),
            "old_std": batch.old_std.detach().cpu().tolist(),
            "new_mean": new_mean.detach().cpu().tolist(),
            "new_std": new_std.detach().cpu().tolist(),
            "old_mean_mirrored": (
                None if old_mirrored is None else old_mirrored.mean.detach().cpu().tolist()
            ),
            "old_std_mirrored": (
                None if old_mirrored is None else old_mirrored.std.detach().cpu().tolist()
            ),
            "raw_action_mirrored": (
                None if old_mirrored is None else old_mirrored.raw_action.detach().cpu().tolist()
            ),
            "old_log_prob_mirrored": (
                None if old_mirror_log_prob is None
                else old_mirror_log_prob.detach().cpu().tolist()
            ),
            "new_log_prob_mirrored": (
                None if mirrored_new_log_prob is None
                else mirrored_new_log_prob.detach().cpu().tolist()
            ),
            "new_value_mirrored": (
                None if mirrored_value is None else mirrored_value.detach().cpu().tolist()
            ),
            "new_mean_mirrored": (
                None if mirrored_output is None
                else mirrored_output.action_mean.detach().cpu().tolist()
            ),
            "new_std_mirrored": (
                None if mirrored_distribution is None
                else mirrored_distribution.std.detach().cpu().tolist()
            ),
            "kl_event_original": kl_event_original.detach().cpu().tolist(),
            "kl_event_mirrored": (
                None if kl_event_mirrored is None
                else kl_event_mirrored.detach().cpu().tolist()
            ),
            "normalized_advantage_mirrored": (
                None if not mirror_enabled else advantage_slice.detach().cpu().tolist()
            ),
        }

    # ── 算法局部状态（P5 消费；不声称完整环境恢复） ─────────────────────────

    def state_dict(self) -> dict:
        """算法局部状态：Adam、LR、计数、shuffle PRNG、配置/网格身份。

        只覆盖本算法持有的状态；**不包含**环境/物理/窗口/命令/传感等任何
        运行时状态——完整 checkpoint 恢复属 P5，本 state_dict 不声称完成。
        """
        return {
            "schema": PPO_CONFIG_SCHEMA_VERSION,
            "optimizer": self._adam.state_dict(),
            "learning_rate": self._learning_rate,
            "update_count": self._update_count,
            "optimizer_step_count": self._optimizer_step_count,
            "shuffle_generator_state": self.shuffle_generator_state(),
            "config": self.config_identity(),
            "grid_identity_hash": self.grid_identity_hash,
            "shuffle_seed_derivation": {
                "stream": SHUFFLE_GENERATOR_STREAM,
                "base_seed_schema": SHUFFLE_SEED_SCHEMA,
            },
            "faulted_stage": self._faulted_stage,
        }

    def load_state_dict(self, state: dict) -> None:
        """恢复算法局部状态（要求同 schema/config/grid 的 trainer 实例）。

        optimizer 状态按参数顺序映射：加载方必须用相同结构的 adapter 构造
        本 trainer 后再加载。仍不构成完整环境恢复。
        """
        if self.faulted:
            raise PPOUpdateFault(
                self._faulted_stage or "unknown", "faulted 状态禁止 load_state_dict"
            )
        if not isinstance(state, dict) or state.get("schema") != PPO_CONFIG_SCHEMA_VERSION:
            raise PPOUpdateError(
                f"state_dict schema 不兼容：{state.get('schema') if isinstance(state, dict) else type(state).__name__}"
            )
        if state.get("grid_identity_hash") != self.grid_identity_hash:
            raise PPOUpdateError(
                f"state_dict grid 身份 {state.get('grid_identity_hash')} != {self.grid_identity_hash}"
            )
        if state.get("config") != self.config_identity():
            raise PPOUpdateError("state_dict 配置与当前 trainer 不一致（拒绝跨配方加载）")
        self._adam.load_state_dict(state["optimizer"])
        self._learning_rate = float(state["learning_rate"])
        self._update_count = int(state["update_count"])
        self._optimizer_step_count = int(state["optimizer_step_count"])
        self.load_shuffle_generator_state(state["shuffle_generator_state"])
        self._faulted_stage = state.get("faulted_stage")
