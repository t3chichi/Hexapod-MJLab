"""生产采集薄层：TrainingRuntime × ConfinedPolicyAdapter 采样到 buffer。

本文件是任务侧薄层（真实环境采集；纯 Torch 的分布/buffer/GAE 在
``hexapod_mjlab.rl.confined_locomotion.distribution`` 与
``...training_rollout``）。
采集遵循真实转移 runtime 与策略适配器的生产契约：

* ``TrainingCollector(runtime, policy_adapter, *, steps=24)`` 只核对与编排，
  **不自行创建第二网络/优化器**：runtime 必须是 ``TrainingRuntime``；
  adapter 必须是注入了生产 ``IndependentDiagonalNormalProvider`` 的
  ``ConfinedPolicyAdapter``（provider 唯一参数 log_std(18)）；网络
  ``ground_samples`` 必须与 runtime 显式训练网格 M 一致；网络/provider
  参数须在 runtime 设备上且 FP32。
* 逐步顺序（一次 online 完整前向，不为 mean/value/辅助重新前向四次）：
  ``prepare()``（同控制身份重复读同对象）→ ``labels()``（pre-action 标签
  绑定）→ ``adapter.forward(...)`` 取完整 ``NetworkOutput`` →
  ``adapter.distribution_provider(output.action_mean)`` 得分布 → 显式
  generator 采样 raw + old_log_prob/old_mean/old_std → ``runtime.step(raw)``
  → 用已有 final 快照做一次 value 前向得 V_next（bootstrap 不采样、不扫描、
  不抽噪；其 ground_pred 不计入监督集合；next_inputs 只用于下一动作）→
  组装 ``RolloutStepSample`` 入 buffer，并把 pre-action 的地面 SOURCE 帧
  交给全 rollout 去重存储。final 边界新扫描不当本步监督。
* 采集期间策略权重不变（更新留给 PPO）：collect 前后各取一次模型参数
  指纹并核对一致；记录 provider 身份（冻结 JSON）与模型指纹。仅在
  ``no_grad`` 采集期复用扫描点块的编码；rollout 仍保存原始观测/点云，PPO
  更新重新前向，缓存特征不会成为更新输入。
* 停止语义：默认 engineering 模式下自然跌倒/formal done 已由 runtime 保存现场
  并停止——collector 如实返回 incomplete 结果（``require_complete=True``
  时抛 ``TrainingCollectionIncomplete``），保留部分数据，绝不凑满 24 步、
  不偷偷 reset、不改症状阈值。显式 ``training_lifecycle_fixture`` 模式执行
  训练部分 reset/GAE 协议，身份显式记录，不伪装自然实验。
* collector 自身额外失败（分布/契约/buffer 等，且 runtime 未自行冻结时）
  调 ``runtime._save_fault(err, phase="collection")`` 保存冻结现场后原样
  raise；不输出可续跑 checkpoint。无 optimizer、无注册改动、无正式
  train/play。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
from torch import nn

from hexapod_mjlab.rl.confined_locomotion.contracts import FLOAT_DTYPE, JOINT_DIM
from hexapod_mjlab.rl.confined_locomotion.distribution import IndependentDiagonalNormalProvider
from hexapod_mjlab.rl.confined_locomotion.inference import PointCloudFeatureCache
from hexapod_mjlab.rl.confined_locomotion.network import ConfinedLocomotionNetwork
from hexapod_mjlab.rl.confined_locomotion.policy import ConfinedPolicyAdapter
from hexapod_mjlab.rl.confined_locomotion.training_rollout import (
    NOMINAL_ROLLOUT_STEPS,
    RolloutStepSample,
    TrainingRolloutBuffer,
)

from .runtime import TrainingRuntime, TrainingTransition

__all__ = [
    "CollectionResult",
    "TrainingCollectionIncomplete",
    "TrainingCollector",
    "model_state_fingerprint",
]


def model_state_fingerprint(model: nn.Module) -> str:
    """模型参数的确定性指纹（名/形状/dtype/字节；采集身份记录用）。

    只作「采集期间权重不变」与证据身份核对，不是加密摘要，也不进入
    任何更新输入。
    """
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(parameter.shape)).encode("utf-8"))
        digest.update(str(parameter.dtype).encode("utf-8"))
        digest.update(parameter.detach().reshape(-1).cpu().numpy().tobytes())
    return digest.hexdigest()


class TrainingCollectionIncomplete(RuntimeError):
    """24 步未收满（工程停止/自然跌倒后如实停止；残缺数据保留为证据）。

    绝不静默凑满名义步数、不 reset 续跑、不修改症状阈值。
    """

    def __init__(self, result: "CollectionResult") -> None:
        self.result = result
        super().__init__(
            f"collection incomplete: completed {result.completed_steps}/{result.requested_steps} "
            f"steps (reason={result.stopped_reason!r}, mode={result.mode!r}); partial data kept "
            "as evidence, never padded to the nominal length"
        )


@dataclass(frozen=True)
class CollectionResult:
    """一次 collect 的结果身份（自然/夹具模式显区分，不伪装）。"""

    buffer: TrainingRolloutBuffer
    mode: str
    requested_steps: int
    completed_steps: int
    stopped_reason: str | None
    model_fingerprint: str
    distribution_identity_frozen: str

    @property
    def complete(self) -> bool:
        return self.completed_steps == self.requested_steps and self.buffer.is_full

    def require_complete(self) -> "CollectionResult":
        if not self.complete:
            raise TrainingCollectionIncomplete(self)
        return self

    def summary(self) -> dict:
        return {
            "mode": self.mode,
            "requested_steps": self.requested_steps,
            "completed_steps": self.completed_steps,
            "complete": self.complete,
            "stopped_reason": self.stopped_reason,
            "model_fingerprint": self.model_fingerprint,
            "distribution_identity_frozen": self.distribution_identity_frozen,
            "buffer": self.buffer.summary(),
        }


class TrainingCollector:
    """生产采样 collector：把 runtime 的真实转移采样进训练 buffer。"""

    def __init__(
        self,
        runtime: TrainingRuntime,
        policy_adapter: ConfinedPolicyAdapter,
        *,
        steps: int = NOMINAL_ROLLOUT_STEPS,
        use_feature_cache: bool = True,
    ) -> None:
        if not isinstance(runtime, TrainingRuntime):
            raise TypeError(
                f"runtime 必须为 tasks.confined_locomotion.training.TrainingRuntime，"
                f"实际 {type(runtime).__name__}"
            )
        if not isinstance(policy_adapter, ConfinedPolicyAdapter):
            raise TypeError(
                "policy_adapter 必须为 rl.confined_locomotion.policy.ConfinedPolicyAdapter，"
                f"实际 {type(policy_adapter).__name__}"
            )
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError(f"steps 须为 >=1 整数（名义 24），实际 {steps!r}")
        if not isinstance(use_feature_cache, bool):
            raise TypeError(f"use_feature_cache 须为 bool，实际 {use_feature_cache!r}")
        provider = policy_adapter.distribution_provider
        if not isinstance(provider, IndependentDiagonalNormalProvider):
            raise TypeError(
                "policy_adapter 必须注入生产 IndependentDiagonalNormalProvider"
                f"（18 维对角 Normal，唯一 log_std），实际 {type(provider).__name__}"
            )
        network = policy_adapter.network
        if not isinstance(network, ConfinedLocomotionNetwork):
            raise TypeError(
                f"adapter.network 必须为 ConfinedLocomotionNetwork，实际 {type(network).__name__}"
            )
        self.runtime = runtime
        self.policy_adapter = policy_adapter
        self.provider = provider
        self.network = network
        self.steps = int(steps)
        self.use_feature_cache = use_feature_cache
        self.feature_cache = PointCloudFeatureCache(network) if use_feature_cache else None
        # collect() keeps feature rows across control steps; the public
        # collect_step() entry point owns a one-step cache lifetime.
        self._collection_cache_active = False
        # env.device 可能是字符串（P1 已修正的比较口径）：统一 torch.device。
        self.device = torch.device(runtime.device)
        self.num_envs = int(runtime.num_envs)
        self.mode = str(runtime.mode)

        # ground M：网络头维数必须与显式训练网格一致（同一配置身份驱动）。
        grid_samples = int(runtime.grid_cfg.num_points)
        if network.ground_samples is None or int(network.ground_samples) != grid_samples:
            raise ValueError(
                f"network.ground_samples ({network.ground_samples}) 必须等于显式训练网格 "
                f"M={grid_samples}（ground head/标签/存储共用网格身份）"
            )
        self.ground_samples = grid_samples
        self.grid_identity_hash = str(runtime.grid_cfg.identity_hash())

        # 设备/FP32：网络与 provider 参数全部在 runtime 设备上且 float32。
        for name, parameter in network.named_parameters():
            if parameter.device != self.device:
                raise ValueError(
                    f"network 参数 {name} 在 {parameter.device}，runtime 设备为 {self.device}"
                )
            if parameter.dtype is not FLOAT_DTYPE:
                raise ValueError(f"network 参数 {name} dtype={parameter.dtype}，须 FP32")
        log_std = provider.log_std
        if tuple(log_std.shape) != (JOINT_DIM,) or log_std.dtype is not FLOAT_DTYPE \
                or log_std.device != self.device:
            raise ValueError(
                f"provider.log_std 须为 ({JOINT_DIM},) float32 且在 {self.device}，"
                f"实际 {tuple(log_std.shape)} {log_std.dtype} {log_std.device}"
            )

    # ── 采集 ────────────────────────────────────────────────────────────────

    def make_buffer(self) -> TrainingRolloutBuffer:
        """按 runtime 身份构造训练 buffer（网格 hash/M/设备一致）。"""
        return TrainingRolloutBuffer(
            num_envs=self.num_envs,
            ground_samples=self.ground_samples,
            grid_identity_hash=self.grid_identity_hash,
            device=self.device,
            nominal_steps=self.steps,
        )

    def collect(
        self,
        *,
        buffer: TrainingRolloutBuffer | None = None,
        require_complete: bool = False,
    ) -> CollectionResult:
        """采集名义 steps 个真实控制步（torch.no_grad；权重保持不变）。

        默认 engineering 模式下自然停止如实返回 incomplete（残缺数据保留）；
        ``require_complete=True`` 时以 ``TrainingCollectionIncomplete`` 显式
        抛出。collector 自身故障先经 ``runtime._save_fault`` 冻结再 raise。
        """
        if self.feature_cache is not None:
            self.feature_cache.clear()
            self.feature_cache.reset_stats()
        self._collection_cache_active = True
        try:
            buffer = buffer if buffer is not None else self.make_buffer()
            fingerprint_before = model_state_fingerprint(self.policy_adapter)
            buffer.bind_collection_identity(
                distribution_identity_frozen=str(self.policy_adapter.identity_frozen),
                model_fingerprint=fingerprint_before,
            )
            stopped_reason: str | None = None
            completed = 0
            with torch.no_grad():
                for _ in range(self.steps):
                    transition = self.collect_step(buffer)
                    completed += 1
                    if transition.engineering_stopped:
                        stopped_reason = "engineering_stop"
                        break
            fingerprint_after = model_state_fingerprint(self.policy_adapter)
            if fingerprint_after != fingerprint_before:
                error = RuntimeError(
                    "collection-time policy weights changed (no optimizer exists in P2; "
                    "updates belong to P3)"
                )
                self._save_collector_fault(error)
                raise error
            result = CollectionResult(
                buffer=buffer,
                mode=self.mode,
                requested_steps=self.steps,
                completed_steps=completed,
                stopped_reason=stopped_reason,
                model_fingerprint=fingerprint_after,
                distribution_identity_frozen=str(self.policy_adapter.identity_frozen),
            )
            if require_complete:
                result.require_complete()
            return result
        finally:
            self._collection_cache_active = False
            if self.feature_cache is not None:
                self.feature_cache.clear()

    def collect_step(self, buffer: TrainingRolloutBuffer) -> TrainingTransition:
        """采集一个真实控制步并入 buffer（返回该步转移；no_grad 由本方法负责）。

        显式单步入口供隔离夹具在步间做**已声明**的受控操作（如终止计数
        预置）；正常训练循环用 ``collect``。工程停止照常由返回值标记。
        """
        owns_cache_lifetime = (
            self.feature_cache is not None and not self._collection_cache_active
        )
        if owns_cache_lifetime:
            self.feature_cache.clear()
        try:
            with torch.no_grad():
                try:
                    sample, transition = self._collect_one_step(buffer)
                    buffer.append(sample)
                except Exception as err:
                    self._save_collector_fault(err)
                    raise
            return transition
        finally:
            if owns_cache_lifetime:
                self.feature_cache.clear()

    # ── 单步（一次完整前向 + 采样 + 真实转移 + 自然边界价值） ────────────────

    def _collect_one_step(
        self, buffer: TrainingRolloutBuffer
    ) -> tuple[RolloutStepSample, TrainingTransition]:
        runtime = self.runtime
        adapter = self.policy_adapter
        batch = self.num_envs

        # 1) pre-action 输入与标签（同一控制身份；prepare 重复读同对象）。
        prepared = runtime.prepare()
        labels = runtime.labels()

        # 2) 一次 online 完整前向（mean/value/envelope/ground_pred 同源），
        #    再由 provider 在该 action_mean 上取分布；不二次前向网络。
        output = self._forward_prepared(prepared)
        self._check_output(output, "pre-action")
        distribution = adapter.distribution_provider(output.action_mean)
        raw_action = distribution.sample()                 # 显式 generator；未裁剪 raw
        old_log_prob = distribution.log_prob(raw_action)   # 事件级 (B,)
        old_mean = output.action_mean
        old_std = distribution.std
        if tuple(old_std.shape) != (batch, JOINT_DIM):
            raise RuntimeError(f"distribution.std 形状 {tuple(old_std.shape)} != ({batch},{JOINT_DIM})")
        old_value = output.value.squeeze(-1)               # (B,1)→(B,) 规范化
        if tuple(old_value.shape) != (batch,):
            raise RuntimeError(f"old_value squeeze 后形状 {tuple(old_value.shape)} != ({batch},)")

        # 3) 真实 4 子步转移（P1 清洁协议：实际 reward、独立 flags、自然
        #    final、终止前观测在 reset 之前封存）。
        transition = runtime.step(raw_action)

        # 4) V_next：自然边界 final 快照的一次 value 前向（不采样/不扫描/
        #    不抽噪；与采集期相同权重）。bootstrap 的 ground_pred 不进入
        #    监督集合；next_inputs 只由 runtime 用于下一动作。
        final = transition.final_inputs
        final_output = self._forward_prepared(final)
        self._check_output(final_output, "bootstrap")
        v_next = final_output.value.squeeze(-1)
        if tuple(v_next.shape) != (batch,):
            raise RuntimeError(f"v_next squeeze 后形状 {tuple(v_next.shape)} != ({batch},)")

        # 5) 组装样本（观测/点块为 runtime 已 clone 的独立张量；buffer 再验证）。
        reset_applied = torch.zeros(batch, dtype=torch.bool, device=self.device)
        if transition.reset_env_ids is not None:
            reset_applied.index_fill_(0, transition.reset_env_ids, True)
        sample = RolloutStepSample(
            control_id=int(transition.control_id) - 1,
            time_s=float(transition.time_s),
            env_ids=transition.env_ids,
            episode_ids=transition.episode_ids,
            step_ids=transition.step_ids,
            observation=transition.input.observation,
            command=transition.command,
            near_points=transition.input.near_points,
            near_valid=transition.input.near_valid,
            far_points=transition.input.far_points,
            far_valid=transition.input.far_valid,
            near_filled=transition.input.near_filled,
            near_scan_ids=transition.input.near_scan_ids,
            near_times=transition.input.near_times,
            far_point_scan_ids=transition.input.far_point_scan_ids,
            far_point_times=transition.input.far_point_times,
            near_angle_keys=transition.input.near_angle_keys,
            far_angle_keys=transition.input.far_angle_keys,
            near_source_dirs_sensor=transition.input.near_source_dirs_sensor,
            far_source_dirs_sensor=transition.input.far_source_dirs_sensor,
            near_source_dirs_present=transition.input.near_source_dirs_present,
            far_source_dirs_present=transition.input.far_source_dirs_present,
            near_source_indices=transition.input.near_source_indices,
            far_source_indices=transition.input.far_source_indices,
            raw_action=raw_action,
            old_log_prob=old_log_prob,
            old_mean=old_mean,
            old_std=old_std,
            old_value=old_value,
            reward=transition.reward,
            terminated=transition.terminated,
            truncated=transition.truncated,
            v_next=v_next,
            final_episode_ids=transition.final_inputs.observation.episode_ids,
            final_step_ids=transition.final_inputs.observation.step_ids,
            final_control_id=int(transition.control_id),
            next_episode_ids=transition.next_inputs.observation.episode_ids,
            next_step_ids=transition.next_inputs.observation.step_ids,
            reset_applied=reset_applied,
            envelope_label=labels["envelope"],
            engineering_stopped=bool(transition.engineering_stopped),
        )

        # 6) 地面 SOURCE 帧：只来自 pre-action 标签（第 2 步消费的身份）。
        self._add_ground_source_frames(buffer, transition, labels)
        return sample, transition

    def _forward_prepared(self, prepared):
        """Run the complete policy output, reusing only cached point encodings."""
        if self.feature_cache is not None:
            return self.feature_cache.forward(prepared)
        return self.policy_adapter(
            prepared.observation,
            prepared.near_points,
            prepared.near_valid,
            prepared.far_points,
            prepared.far_valid,
        )

    def _add_ground_source_frames(
        self,
        buffer: TrainingRolloutBuffer,
        transition: TrainingTransition,
        labels: dict,
    ) -> None:
        """把本步 pre-action 的地面 SOURCE 帧交给全 rollout 去重存储。

        缺失行（source_frames 为 None 或 scan<0）跳过；重复 key 由存储逐位
        校验不可变一致性（契约故障即抛）。bootstrap 前向不产生帧。
        """
        grid_hash = str(labels["ground"]["grid_identity_hash"])
        if grid_hash != buffer.ground_frames.grid_identity_hash:
            raise RuntimeError(
                f"labels 网格身份 {grid_hash} 与 buffer {buffer.ground_frames.grid_identity_hash}"
                " 不一致（同 rollout 不允许 grid 变更）"
            )
        ground_label = labels["ground"]["label"]
        metadata = labels["ground"]["metadata"]
        for row in range(self.num_envs):
            frame = labels["source_frames"][row]
            if frame is None:
                continue
            scan_id = int(frame["scan"])
            if scan_id < 0:
                continue
            pose_meta = metadata[row] if row < len(metadata) else None
            buffer.ground_frames.add_source_frame(
                env_id=int(frame["env"]),
                episode_id=int(frame["episode"]),
                scan_id=scan_id,
                near_points=transition.input.near_points[row : row + 1],
                near_valid=transition.input.near_valid[row : row + 1],
                near_angle_keys=(
                    None if transition.input.near_angle_keys is None
                    else transition.input.near_angle_keys[row : row + 1]
                ),
                near_source_dirs_sensor=(
                    None if transition.input.near_source_dirs_sensor is None
                    else transition.input.near_source_dirs_sensor[row : row + 1]
                ),
                near_source_dirs_present=(
                    None if transition.input.near_source_dirs_present is None
                    else transition.input.near_source_dirs_present[row : row + 1]
                ),
                near_source_indices=(
                    None if transition.input.near_source_indices is None
                    else transition.input.near_source_indices[row : row + 1]
                ),
                heights=ground_label.heights[row],
                positions=ground_label.positions[row],
                valid=ground_label.valid[row],
                time_s=float(frame["time_s"]),
                persistent_scan_count=int(frame["persistent_scan_count"]),
                scan_pos_w=None if pose_meta is None else pose_meta.get("base_pos_w"),
                scan_quat_wxyz=None if pose_meta is None else pose_meta.get("base_quat_wxyz"),
                control_id=int(transition.control_id) - 1,
            )

    @staticmethod
    def _check_output(output, phase):
        for name, value in vars(output).items():
            if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
                raise RuntimeError(f"collector/{phase}/{name}: network output contains NaN/Inf")

    # ── 故障冻结 ────────────────────────────────────────────────────────────

    def _save_collector_fault(self, err: BaseException) -> None:
        """collector 额外失败 → runtime 冻结现场（runtime 自身已冻结则跳过）。"""
        runtime = self.runtime
        if getattr(runtime, "_faulted", False) or getattr(runtime, "_stopped", False):
            return  # runtime 已保存该故障/停止现场，不重复覆盖
        try:
            runtime._save_fault(err, phase="collection")
        except BaseException:  # 保存失败不掩盖原始错误
            pass
