"""P2 生产动作分布：18 维独立对角 Normal 的 provider（唯一可学习 log_std）。

仅标准库与 PyTorch；不导入 mjlab/任务注册器。本模块只实现已确认的
生产动作分布（docs/tasks/confined_locomotion/平地学习与基准测量.md 第 2 节
「训练动作分布」与「策略探索起步候选」；docs/tasks/confined_locomotion/
training/首版训练实施计划.md 4.1），不实现 PPO 更新、退火或镜像：

* 18 维独立对角高斯：均值**只**来自现有网络动作头（provider 不建第二个
  均值头、不变换传入均值）；各关节独立可学习且与观测无关的 log_std(18)
  是本 provider 唯一的 ``nn.Parameter``，初始 std=0.5（log_std=ln 0.5），
  std=exp(log_std)，不 clamp、不设边界、不加状态相关标准差头。
* 事件级契约（与 ``ConfinedPolicyAdapter`` 的 provider 注入口一致）：
  ``mean``/``std``/``stddev`` (B,18)；``sample()`` (B,18)；``log_prob(raw)``
  与 ``entropy()`` 均为 18 维求和后的 (B,) 一维张量。log_prob 对应**未裁剪
  raw** 动作，不对应 clip 值或 PD 目标角。
* 采样严格使用本 provider 自有的 ``torch.Generator``（显式设备；CUDA 上为
  实际 CUDA 设备），形如 ``torch.randn(shape, device, dtype, generator=...)``；
  不使用指令/缺口/测距/本体/未来 shuffle 的任何随机流，也不额外叠加随机
  动作。seed 由命名派生入口 ``derive_action_generator_seed`` 显式给出；
  generator 状态可经 ``generator_state``/``load_generator_state`` 保存恢复
  （P5 checkpoint 消费）。
* 一切输入/输出校验 float32、同设备、形状、全部有限、std>0；违规抛
  ``DistributionContractError``——异常保存停止（由采集层冻结现场），不删除
  样本、不加 AMP、不退火、不加探索噪声、不使用截断 Normal。
* 身份 schema（``distribution_identity``）稳定描述分布族：18 维、独立对角、
  初始 std=0.5、log_std 与观测无关、采样流名称。学习后 provider 仍代表同一
  族，身份**不**随当前 std 漂移；身份不含任何服务凭据。身份用于注入
  ``ConfinedPolicyAdapter(distribution_identity=...)``（JSON 可冻结）。
* ``forward(action_mean)`` 保留计算图：log_prob/entropy 的梯度同时流向网络
  均值与 log_std（P3 联合更新用）；采集期由调用方以 ``torch.no_grad`` 包裹。
"""

from __future__ import annotations

import hashlib
import math

import torch
from torch import nn

from .contracts import FLOAT_DTYPE, JOINT_DIM

__all__ = [
    "ACTION_GENERATOR_STREAM",
    "ACTION_NORMAL_SCHEMA_VERSION",
    "DiagonalNormalDistribution",
    "DistributionContractError",
    "INITIAL_STD_STARTUP",
    "IndependentDiagonalNormalProvider",
    "LOG_STD_INIT",
    "derive_action_generator_seed",
]

#: 分布身份 schema 版本（身份冻结/比对用；语义变化必须 bump）。
ACTION_NORMAL_SCHEMA_VERSION = "confined_locomotion.action_normal/v1"

#: 已确认起步候选初始 std=0.5（未触发裁剪时目标扰动 0.125 rad；效果留 5090 验证）。
INITIAL_STD_STARTUP = 0.5

#: log_std 初始值 = ln(0.5) ≈ −0.6931471805599453。
LOG_STD_INIT = math.log(INITIAL_STD_STARTUP)

#: 策略动作采样随机流的显式命名（PRNG 分流：独立于命令/缺口/测距/本体/shuffle）。
ACTION_GENERATOR_STREAM = "policy_action"

#: log N(x|μ,σ) 与每维熵的解析常数。
_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)
_ENTROPY_CONST_PER_DIM = 0.5 + _HALF_LOG_2PI  # 0.5·(1+ln 2π)


class DistributionContractError(ValueError):
    """分布契约违规（形状/dtype/设备/有限性/std<=0/生成器设备错配等）。"""


def derive_action_generator_seed(base_seed: int) -> int:
    """策略动作采样 generator seed 的显式命名派生（uint32 安全）。

    仅依赖 (schema, 流名, base_seed) 的稳定散列：与命令流（NumPy
    SeedSequence 派生）族不同、与 num_envs/构造顺序无关，同一 base_seed
    派生结果恒定，可直接喂 ``torch.Generator.manual_seed``。
    """
    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise DistributionContractError(
            f"base_seed 必须为 int（通常是配方 seed），实际 {base_seed!r}"
        )
    material = (
        f"{ACTION_NORMAL_SCHEMA_VERSION}|{ACTION_GENERATOR_STREAM}|{int(base_seed)}"
    ).encode("utf-8")
    digest = hashlib.sha256(material).digest()
    return int.from_bytes(digest[:4], "big")


def _require_finite(name: str, tensor: torch.Tensor) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise DistributionContractError(f"{name} 含 NaN/Inf（保存停止；不删样本）")


class DiagonalNormalDistribution:
    """``provider(action_mean)`` 返回的具体分布（事件级契约对象）。

    mean 为网络前向图上的原张量（不 detach）；log_std 为 provider 参数图上的
    原张量；std = exp(log_std) 与观测无关。本对象不持有参数、不可再次采样
    之外的随机性：``sample`` 每次恰好消耗一次 provider generator。
    """

    def __init__(
        self,
        *,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        generator: torch.Generator,
    ) -> None:
        self._mean = mean
        self._log_std = log_std
        self._generator = generator

    # ── 分布状态（均保留计算图） ─────────────────────────────────────────────

    @property
    def mean(self) -> torch.Tensor:
        """(B,18) 分布中心——即网络 action_mean 原值（第二均值头拒绝）。"""
        return self._mean

    @property
    def log_std(self) -> torch.Tensor:
        """(18,) 与观测无关的 log_std（provider 参数）。"""
        return self._log_std

    @property
    def std(self) -> torch.Tensor:
        """(B,18) = exp(log_std) 广播；无 clamp。"""
        return torch.exp(self._log_std).expand_as(self._mean)

    #: 兼容别名（event 级契约只要求 mean/std 之一，本分布两者都暴露）。
    stddev = std

    # ── 采样与概率 ───────────────────────────────────────────────────────────

    def sample(self) -> torch.Tensor:
        """标准重参数化采样 raw 动作 (B,18)：mean + ε·std，ε 来自显式 generator。

        返回**未裁剪** raw 值（±100 裁剪属执行链）；本方法不读任何其它随机流，
        每次调用恰消耗一次 generator。
        """
        epsilon = torch.randn(
            self._mean.shape,
            device=self._mean.device,
            dtype=FLOAT_DTYPE,
            generator=self._generator,
        )
        raw = self._mean + epsilon * self.std
        _require_finite("sampled raw_action", raw)
        return raw

    def log_prob(self, raw: torch.Tensor) -> torch.Tensor:
        """raw 动作的对数概率，18 维求和后的 (B,) 事件级张量。

        严格对应**未裁剪 raw** 值（clip 后的值概率不同，见测试反例）。
        """
        self._require_raw("log_prob raw", raw)
        z = (raw - self._mean) / self.std
        per_dim = -0.5 * z * z - self._log_std - _HALF_LOG_2PI
        result = per_dim.sum(dim=1)
        _require_finite("raw action log_prob", result)
        return result

    def entropy(self) -> torch.Tensor:
        """分布熵，18 维求和后的 (B,) 事件级张量（与观测无关）。"""
        per_dim = self._log_std + _ENTROPY_CONST_PER_DIM   # (18,) 每维 0.5(1+ln2π)+lnσ
        result = per_dim.expand_as(self._mean).sum(dim=1)
        _require_finite("action entropy", result)
        return result

    # ── 内部校验 ─────────────────────────────────────────────────────────────

    def _require_raw(self, name: str, raw: torch.Tensor) -> None:
        if not isinstance(raw, torch.Tensor):
            raise DistributionContractError(
                f"{name} 必须是 torch.Tensor，实际 {type(raw).__name__}"
            )
        if raw.dtype is not FLOAT_DTYPE or tuple(raw.shape) != tuple(self._mean.shape):
            raise DistributionContractError(
                f"{name} 应为 {tuple(self._mean.shape)} {FLOAT_DTYPE}，"
                f"实际 {tuple(raw.shape)} {raw.dtype}"
            )
        if raw.device != self._mean.device:
            raise DistributionContractError(
                f"{name} 设备 {raw.device} 与均值 {self._mean.device} 不一致"
            )
        _require_finite(name, raw)


class IndependentDiagonalNormalProvider(nn.Module):
    """18 维独立对角 Normal 的 distribution provider（唯一参数 log_std）。

    构造（均显式，无隐藏默认）：
      device          provider 所在设备（生成器与 log_std 同设备；CUDA 须给
                      实际 CUDA 设备）。``.to()`` 不会迁移 generator——设备变更
                      须显式重建 provider（P5 保存恢复负责流迁移），forward 对
                      设备错配一律拒绝，绝不静默重播种。
      generator_seed  动作采样 generator 的显式 seed（用命名派生入口
                      ``derive_action_generator_seed(base_seed)`` 产生）。

    使用方式（P2 采集层）：构造一次注入
    ``ConfinedPolicyAdapter(network=..., distribution_provider=provider,
    distribution_identity=provider.distribution_identity())``；采集时对一次网络
    前向的 ``action_mean`` 调 ``provider(action_mean)`` 得到分布再采样/求概率，
    不得为 mean/value/辅助重复前向网络。
    """

    def __init__(self, *, device: torch.device | str | None = None, generator_seed: int) -> None:
        super().__init__()
        if isinstance(generator_seed, bool) or not isinstance(generator_seed, int):
            raise DistributionContractError(
                "generator_seed 必须显式给出为 int（命名派生入口 "
                f"derive_action_generator_seed），实际 {generator_seed!r}"
            )
        resolved = torch.device(device) if device is not None else torch.device("cpu")
        self.log_std = nn.Parameter(
            torch.full((JOINT_DIM,), LOG_STD_INIT, dtype=FLOAT_DTYPE, device=resolved)
        )
        self._generator_seed = int(generator_seed)
        # 生成器固定在构造设备上（CUDA 上为实际 CUDA 设备）；绝不使用全局
        # 指令 PRNG，也不在采样间重播种。
        self._generator = torch.Generator(device=resolved)
        self._generator.manual_seed(self._generator_seed)

    # ── provider 接入口 ──────────────────────────────────────────────────────

    def forward(self, action_mean: torch.Tensor) -> DiagonalNormalDistribution:
        """以网络 action_mean 为中心的独立对角 Normal（校验后原样引用）。"""
        if not isinstance(action_mean, torch.Tensor) or action_mean.dim() != 2 \
                or int(action_mean.shape[1]) != JOINT_DIM or int(action_mean.shape[0]) < 1:
            raise DistributionContractError(
                f"action_mean 须为 (B>=1, {JOINT_DIM}) 张量，实际 "
                f"{tuple(action_mean.shape) if isinstance(action_mean, torch.Tensor) else type(action_mean).__name__}"
            )
        if action_mean.dtype is not FLOAT_DTYPE:
            raise DistributionContractError(
                f"action_mean dtype 应为 {FLOAT_DTYPE}，实际 {action_mean.dtype}"
            )
        _require_finite("action_mean", action_mean)
        log_std = self.log_std
        if log_std.device != action_mean.device:
            raise DistributionContractError(
                f"log_std 设备 {log_std.device} 与 action_mean 设备 "
                f"{action_mean.device} 不一致（.to() 不迁移 generator；须显式重建 provider）"
            )
        self._require_generator_device(action_mean.device)
        _require_finite("log_std", log_std)
        std = torch.exp(log_std)
        if bool((std <= 0).any()) or not bool(torch.isfinite(std).all()):
            # exp 下溢为 0 属契约故障：不 clamp、不替换样本、保存停止。
            raise DistributionContractError(
                "std=exp(log_std) 须处处为正且有限（当前 log_std 使 std 下溢/溢出；"
                "不 clamp、不删样本）"
            )
        return DiagonalNormalDistribution(
            mean=action_mean, log_std=log_std, generator=self._generator
        )

    # ── 身份与随机流 ─────────────────────────────────────────────────────────

    @property
    def generator_seed(self) -> int:
        """构造时显式给出的动作采样 seed（诊断/复现记录用）。"""
        return self._generator_seed

    @property
    def generator_device(self) -> torch.device:
        return self._generator.device

    def distribution_identity(self) -> dict:
        """稳定的族级身份（JSON 可冻结；不随学习后的 std 漂移）。

        内容：schema 版本、分布族、18 维、初始 std=0.5、log_std 初始值、
        std 与观测无关、采样流名称。不含当前 std、不含设备/凭据等执行细节。
        """
        return {
            "schema": ACTION_NORMAL_SCHEMA_VERSION,
            "family": "independent_diagonal_normal",
            "action_dim": JOINT_DIM,
            "initial_std": INITIAL_STD_STARTUP,
            "log_std_init": LOG_STD_INIT,
            "std_observation_dependent": False,
            "sample_generator_stream": ACTION_GENERATOR_STREAM,
        }

    def generator_state(self) -> torch.Tensor:
        """动作采样 generator 状态的独立副本（P5 checkpoint 保存用）。"""
        return self._generator.get_state().clone()

    def load_generator_state(self, state: torch.Tensor) -> None:
        """恢复 generator 状态（同设备字节布局；跨设备迁移属 P5 显式职责）。"""
        if not isinstance(state, torch.Tensor):
            raise DistributionContractError(
                f"generator state 必须是 torch.Tensor，实际 {type(state).__name__}"
            )
        self._generator.set_state(state.detach().to(device="cpu").clone())

    # ── 内部校验 ─────────────────────────────────────────────────────────────

    def _require_generator_device(self, mean_device: torch.device) -> None:
        generator_device = self._generator.device
        if mean_device.type != generator_device.type:
            raise DistributionContractError(
                f"动作均值设备 {mean_device} 与采样 generator 设备 "
                f"{generator_device} 类型不一致（重建 provider，不静默迁移）"
            )
        if generator_device.type == "cuda":
            mean_index = mean_device.index if mean_device.index is not None \
                else torch.cuda.current_device()
            generator_index = generator_device.index if generator_device.index is not None \
                else torch.cuda.current_device()
            if int(mean_index) != int(generator_index):
                raise DistributionContractError(
                    f"动作均值在 cuda:{mean_index}，采样 generator 在 "
                    f"cuda:{generator_index}（重建 provider，不静默迁移）"
                )
