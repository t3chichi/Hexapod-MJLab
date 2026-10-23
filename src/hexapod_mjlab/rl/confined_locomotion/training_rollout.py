"""P2 训练 rollout：24 步生产采样 buffer、GAE 与地面 SOURCE 帧聚合（纯 Torch）。

仅标准库与 PyTorch；不导入 mjlab/任务模块（真实采集薄层在
``tasks/confined_locomotion/training_collection.py``）。本模块实现
docs/tasks/confined_locomotion/training/首版训练实施计划.md 4.1/4.2/4.4 中
P2 范围的训练数据 schema，**不改**旧 mean_dev rollout（``rollout.py`` 的
v1/segment schema 与守卫原样保留，两种 schema 互不通用）：

* 新训练 schema：名义 24 控制步 × B。每步冻结 pre-action 输入
  （OnlineObservation + 近 256/远 640 点与掩码 + 扫描身份）、raw_action、
  old_log_prob、old_mean/old_std、old_value（(B,1) 显式 squeeze(-1) 后以
  (B,) 规范化存储）、实际 reward、独立 terminated/truncated、自然 final
  观测身份与 V_next、reset 后 next 身份、EnvelopeLabel。pre-action 采样与
  标签绑定：envelope 标签身份必须与 pre-action 观测逐行一致
  （``bind_envelope_label``）；final 边界的新扫描不充当本步监督。完整
  rollout 保留重放所需 256/640 输入与标签；86,400 原始射线与 4 张接触表
  不进热 buffer（证据片段由检查入口另行声明留存）。
* 全部张量 detach+clone 独立、FP32；采集期由调用方 ``torch.no_grad`` 包裹
  （不用 inference_mode，保证 P3 重放可对当前模型求梯度）。提供 flatten
  (T·B) 视图并按任意 indices（可重复、可乱序）重建
  ``PreparedControlInputs`` 与 envelope/动作/优势批次：env 在 flatten 后
  重复出现是正常布局，env 身份是真实身份，绝不当作全局连续下标。
* 生命周期明确：收集 → 收满（恰 nominal 步）→ ``consume``（P3 更新完标记）
  → ``clear``（下一 rollout 复用同一 buffer 对象）；残缺批不得伪装完整
  （GAE/replay 要求恰 nominal 步）；收满后继续 append、身份回跳/重复、
  控制步不推进、时间回退、env 行序改变均显式失败。
* GAE（γ=0.99、λ=0.95，本批不归一化——全 rollout 优势标准化属 P3）：
  δ_t = r_t + γ(1−terminated_t)·V_next_t − V_t；反向递推延续掩码
  continuation = 1−(terminated OR truncated)；returns = adv + V。
  V_next 是真实自然边界价值（timeout 用终止前 final 观测；真终止 bootstrap
  为 0；普通批末用继续 episode 的 final 观测），绝不用 reset 后状态或当前
  old_value 修正 reward；original reward 原样保存（timeout reward 不改）。
* 地面 SOURCE 帧全 rollout 去重：key=(env, episode, source_scan,
  grid_identity_hash)。每 SOURCE 帧保存预动作消费时实际排序的
  near_points/near_valid、扫描标签 heights/positions/valid、来源
  time/persistent count 与扫描时位姿（诊断，不在线）。同 key 重复要求上述
  不可变输入/标签/身份逐位一致（网络预测不纳入——更新后变化合法；本体与
  command 不是 ground head 输入，不纳入 frame 一致性）。不按观测支撑裁剪
  真几何标签。先去重后取 geometry_valid.any() 的有效独立帧集合 S；全
  invalid 帧计数并排除；S 完全为空是合法跳过；near 为空但几何有效仍监督。
  不按控制步平均、不跨 rollout 永久去重（clear 后同 key 重新允许）。
  提供每帧有效位置数与 S 统计、确定性均衡 4 桶 index 划分
  （``4/|S|`` 统一权重、桶可空；loss/优化器/镜像属 P3，本模块不实现）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .contracts import (
    COMMAND_DIM,
    ENVELOPE_DIM,
    FAR_POINTS_PER_SCAN,
    FAR_WINDOW_SLOTS,
    FLOAT_DTYPE,
    INT_DTYPE,
    JOINT_DIM,
    NEAR_POINTS,
    ContractError,
    EnvelopeLabel,
    IdentityMismatchError,
    OnlineObservation,
    bind_envelope_label,
    validate_point_block,
)
from .integration import PreparedControlInputs

__all__ = [
    "GAE_GAMMA",
    "GAE_LAMBDA",
    "GROUND_NUM_BUCKETS",
    "NOMINAL_ROLLOUT_STEPS",
    "TRAINING_ROLLOUT_SCHEMA_VERSION",
    "GroundFrameStore",
    "GroundSourceFrame",
    "RolloutStepSample",
    "TrainingReplayBatch",
    "TrainingRolloutBuffer",
    "TrainingRolloutError",
    "TrainingRolloutStateError",
]

#: 新训练 schema 版本（与 mean_dev rollout/v1、rollout_segment/v1 互不通用）。
TRAINING_ROLLOUT_SCHEMA_VERSION = "confined_locomotion.training_rollout/v2"

#: 已确认采集长度：每环境 24 控制步（0.48 s），rollout 边界不触发 reset。
NOMINAL_ROLLOUT_STEPS = 24

#: 已确认 GAE 常数（权威：平地学习与基准测量.md「PPO 与优化器起步配方」）。
GAE_GAMMA = 0.99
GAE_LAMBDA = 0.95

#: 地面辅助均衡桶数（与 PPO 4 minibatch 配对；P3 消费，本模块只给划分）。
GROUND_NUM_BUCKETS = 4

_CONTROL_DT_S = 0.02
_TIME_TOL_S = 2e-6
_FAR_FLAT = FAR_WINDOW_SLOTS * FAR_POINTS_PER_SCAN  # 640


class TrainingRolloutError(ValueError):
    """训练 rollout 契约违规（形状/dtype/身份/不可变冲突/生命周期等）。"""


class TrainingRolloutStateError(TrainingRolloutError):
    """buffer 生命周期误用（残缺批读 GAE/replay、收满后 append、消费后重放等）。"""


def _require_tensor(
    name: str,
    value: object,
    *,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    device: torch.device,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TrainingRolloutError(f"{name} 必须是 torch.Tensor，实际 {type(value).__name__}")
    if value.dtype is not dtype:
        raise TrainingRolloutError(f"{name} dtype 应为 {dtype}，实际 {value.dtype}")
    if tuple(value.shape) != shape:
        raise TrainingRolloutError(f"{name} 形状应为 {shape}，实际 {tuple(value.shape)}")
    if value.device != device:
        raise TrainingRolloutError(f"{name} 设备 {value.device} 与 buffer {device} 不一致")


def _require_finite(name: str, tensor: torch.Tensor) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise TrainingRolloutError(f"{name} 含 NaN/Inf")


def _check_masked_finite_zero(values: torch.Tensor, valid: torch.Tensor, *, name: str) -> None:
    """有效处有限、无效处精确 0（与 F1 契约同规则；用于 SOURCE 帧标签）。"""
    if bool(valid.any()) and not bool(torch.isfinite(values[valid]).all()):
        raise TrainingRolloutError(f"{name}: 有效元素含 NaN/Inf")
    invalid = ~valid
    if bool(invalid.any()):
        bad = values[invalid]
        if not torch.equal(bad, torch.zeros_like(bad)):
            raise TrainingRolloutError(f"{name}: 无效元素必须精确填 0")


# ═══════════════════════════ 地面 SOURCE 帧去重 ═══════════════════════════


@dataclass(frozen=True)
class GroundSourceFrame:
    """一个去重后的地面监督 SOURCE 帧（一个 (env, episode, scan, grid) 身份）。

    near_points/near_valid 是预动作消费时的实际排序近端块（该扫描被策略消费
    的不可变输入）；heights/positions/valid 是扫描时真实几何标签；time/
    persistent count/扫描时位姿为来源诊断（不进入在线输入）。**不含**网络
    预测（ground_pred 更新后变化合法，不参与不可变身份比对），也不含本体/
    command（非 ground head 输入）。
    """

    env_id: int
    episode_id: int
    scan_id: int
    grid_identity_hash: str
    near_points: torch.Tensor                 # (1,256,3) float32 米（角键排序后）
    near_valid: torch.Tensor                  # (1,256) bool
    near_angle_keys: torch.Tensor | None       # (1,256) source optical keys, sorted
    near_source_dirs_sensor: torch.Tensor | None  # (1,256,3) source sensor rays
    near_source_dirs_present: torch.Tensor | None # (1,256) provenance mask
    near_source_indices: torch.Tensor | None   # (1,256) raw source ray ids
    heights: torch.Tensor                     # (M,) float32 米（带符号 h_B）
    positions: torch.Tensor                   # (M,3) float32 机身坐标 [x_B,y_B,h_B]
    valid: torch.Tensor                       # (M,) bool（几何有效掩码）
    time_s: float                             # 扫描来源时间（调用方时钟）
    persistent_scan_count: int                # 持续传感器扫描计数（跨 reset 不清零）
    scan_pos_w: torch.Tensor | None = None    # (3,) 扫描时根位置（诊断）
    scan_quat_wxyz: torch.Tensor | None = None  # (4,) 扫描时根姿态 wxyz（诊断）
    first_control_id: int = -1                # 首次出现的控制步（诊断）

    @property
    def key(self) -> tuple[int, int, int, str]:
        """全 rollout 去重 key：(env, episode, source_scan, grid_hash)。"""
        return (self.env_id, self.episode_id, self.scan_id, self.grid_identity_hash)

    @property
    def valid_position_count(self) -> int:
        """帧内几何有效位置数（P3 帧内均值分母）。"""
        return int(self.valid.sum().item())

    @property
    def supervises(self) -> bool:
        """是否进入有效独立帧集合 S（geometry_valid.any()；不按观测支撑裁剪）。"""
        return bool(self.valid.any().item())


class GroundFrameStore:
    """单 rollout 内的地面 SOURCE 帧存储：按 key 去重 + 逐位不可变校验。

    同一 store 绑定一个 grid 身份（hash）与 M；不同 hash/M 属于配置变更，
    在同一 rollout 内显式拒绝。跨 rollout（新 store/clear 后）同 key 重新
    允许——没有「训练过即永久跳过」。多 epoch 复用同一 store 是正常用法
    （划分按 epoch 确定性轮转）。
    """

    def __init__(self, *, ground_samples: int, grid_identity_hash: str, device: torch.device) -> None:
        if isinstance(ground_samples, bool) or not isinstance(ground_samples, int) or ground_samples < 1:
            raise TrainingRolloutError(f"ground_samples 须为 >=1 整数，实际 {ground_samples!r}")
        if not isinstance(grid_identity_hash, str) or not grid_identity_hash.strip():
            raise TrainingRolloutError("grid_identity_hash 须为非空字符串")
        if not isinstance(device, torch.device):
            raise TrainingRolloutError("device 须为 torch.device")
        self.ground_samples = int(ground_samples)
        self.grid_identity_hash = str(grid_identity_hash)
        self.device = device
        self._frames: list[GroundSourceFrame] = []
        self._index: dict[tuple[int, int, int, str], int] = {}

    # ── 添加/查询 ───────────────────────────────────────────────────────────

    def add_source_frame(
        self,
        *,
        env_id: int,
        episode_id: int,
        scan_id: int,
        near_points: torch.Tensor,
        near_valid: torch.Tensor,
        near_angle_keys: torch.Tensor | None = None,
        near_source_dirs_sensor: torch.Tensor | None = None,
        near_source_dirs_present: torch.Tensor | None = None,
        near_source_indices: torch.Tensor | None = None,
        heights: torch.Tensor,
        positions: torch.Tensor,
        valid: torch.Tensor,
        time_s: float,
        persistent_scan_count: int,
        scan_pos_w: torch.Tensor | None = None,
        scan_quat_wxyz: torch.Tensor | None = None,
        control_id: int,
    ) -> GroundSourceFrame:
        """登记一个 SOURCE 帧；同 key 重复时逐位校验不可变输入/标签/身份。

        任何不可变字段不一致抛 ``TrainingRolloutError``（契约故障，保存停止；
        不任意挑选/平均冲突记录）。near 为空（valid 全 False）但几何标签有效
        的帧合法登记并参与监督。
        """
        m = self.ground_samples
        if isinstance(scan_id, bool) or not isinstance(scan_id, int) or scan_id < 0:
            raise TrainingRolloutError(
                f"SOURCE 帧 scan_id 须为非负 int（缺失行由调用方跳过），实际 {scan_id!r}"
            )
        if not isinstance(near_points, torch.Tensor) or near_points.dim() != 3 \
                or int(near_points.shape[0]) != 1 \
                or int(near_points.shape[1]) != NEAR_POINTS or int(near_points.shape[2]) != 3:
            raise TrainingRolloutError(
                f"near_points 须为 (1,{NEAR_POINTS},3) 单行块，实际 {tuple(near_points.shape)}"
            )
        if near_points.device != self.device:
            raise TrainingRolloutError(
                f"frame near_points 设备 {near_points.device} 与 store {self.device} 不一致"
            )
        validate_point_block(near_points, near_valid, n_points=NEAR_POINTS, name="frame near_points")
        source_values = (
            near_source_dirs_sensor, near_source_dirs_present, near_source_indices,
        )
        if any(value is None for value in source_values) and any(
            value is not None for value in source_values
        ):
            raise TrainingRolloutError(
                "ground SOURCE directions, presence and raw ids must be supplied together"
            )
        if near_angle_keys is not None:
            _require_tensor("frame near_angle_keys", near_angle_keys,
                            dtype=FLOAT_DTYPE, shape=(1, NEAR_POINTS), device=self.device)
            if bool(near_valid.any()) and not bool(torch.isfinite(near_angle_keys[near_valid]).all()):
                raise TrainingRolloutError("ground SOURCE valid angle keys must be finite")
        if near_source_dirs_sensor is not None:
            _require_tensor("frame source directions", near_source_dirs_sensor,
                            dtype=FLOAT_DTYPE, shape=(1, NEAR_POINTS, 3), device=self.device)
            _require_tensor("frame source presence", near_source_dirs_present,
                            dtype=torch.bool, shape=(1, NEAR_POINTS), device=self.device)
            _require_tensor("frame source indices", near_source_indices,
                            dtype=INT_DTYPE, shape=(1, NEAR_POINTS), device=self.device)
            present = near_source_dirs_present
            if bool((present & ~near_valid).any()):
                raise TrainingRolloutError("ground SOURCE has ray provenance on invalid points")
            if not torch.equal(present, near_source_indices >= 0):
                raise TrainingRolloutError("ground SOURCE ray ids and presence mask disagree")
            if bool(present.any()):
                rays = near_source_dirs_sensor[present]
                if not bool(torch.isfinite(rays).all()) or not bool(torch.allclose(
                    torch.linalg.vector_norm(rays, dim=-1),
                    torch.ones_like(torch.linalg.vector_norm(rays, dim=-1)),
                    atol=1e-5, rtol=0.0,
                )):
                    raise TrainingRolloutError("ground SOURCE rays must be finite unit directions")
            if bool((~present).any()) and not torch.equal(
                near_source_dirs_sensor[~present],
                torch.zeros_like(near_source_dirs_sensor[~present]),
            ):
                raise TrainingRolloutError("missing ground SOURCE rays must be exact zero")
        _require_tensor("frame heights", heights, dtype=FLOAT_DTYPE, shape=(m,), device=self.device)
        _require_tensor("frame valid", valid, dtype=torch.bool, shape=(m,), device=self.device)
        _require_tensor("frame positions", positions, dtype=FLOAT_DTYPE, shape=(m, 3), device=self.device)
        _check_masked_finite_zero(heights, valid, name="frame heights")
        _check_masked_finite_zero(positions, valid[:, None].expand(m, 3), name="frame positions")
        if not math.isfinite(float(time_s)):
            raise TrainingRolloutError("frame time_s 必须有限")
        if isinstance(persistent_scan_count, bool) or not isinstance(persistent_scan_count, int) \
                or persistent_scan_count < 0:
            raise TrainingRolloutError(f"persistent_scan_count 须为非负 int，实际 {persistent_scan_count!r}")

        key = (int(env_id), int(episode_id), int(scan_id), self.grid_identity_hash)
        existing_index = self._index.get(key)
        if existing_index is not None:
            existing = self._frames[existing_index]
            differences = self._compare_frames(existing, near_points, near_valid, heights,
                                               positions, valid, near_angle_keys,
                                               near_source_dirs_sensor, near_source_dirs_present,
                                               near_source_indices, float(time_s),
                                               int(persistent_scan_count), scan_pos_w, scan_quat_wxyz)
            if differences:
                raise TrainingRolloutError(
                    f"重复 SOURCE 帧 key {key} 的不可变字段不一致: {sorted(differences)}"
                    "（契约故障：不能挑选/平均冲突记录，不比较网络预测）"
                )
            return existing

        frame = GroundSourceFrame(
            env_id=int(env_id),
            episode_id=int(episode_id),
            scan_id=int(scan_id),
            grid_identity_hash=self.grid_identity_hash,
            near_points=near_points.detach().to(device=self.device, dtype=FLOAT_DTYPE).clone(),
            near_valid=near_valid.detach().to(device=self.device).clone(),
            near_angle_keys=None if near_angle_keys is None else near_angle_keys.detach().clone(),
            near_source_dirs_sensor=(None if near_source_dirs_sensor is None
                                     else near_source_dirs_sensor.detach().clone()),
            near_source_dirs_present=(None if near_source_dirs_present is None
                                      else near_source_dirs_present.detach().clone()),
            near_source_indices=(None if near_source_indices is None
                                 else near_source_indices.detach().clone()),
            heights=heights.detach().to(device=self.device, dtype=FLOAT_DTYPE).clone(),
            positions=positions.detach().to(device=self.device, dtype=FLOAT_DTYPE).clone(),
            valid=valid.detach().to(device=self.device).clone(),
            time_s=float(time_s),
            persistent_scan_count=int(persistent_scan_count),
            scan_pos_w=None if scan_pos_w is None
            else scan_pos_w.detach().to(device=self.device, dtype=FLOAT_DTYPE).reshape(-1)[:3].clone(),
            scan_quat_wxyz=None if scan_quat_wxyz is None
            else scan_quat_wxyz.detach().to(device=self.device, dtype=FLOAT_DTYPE).reshape(-1)[:4].clone(),
            first_control_id=int(control_id),
        )
        self._index[key] = len(self._frames)
        self._frames.append(frame)
        return frame

    def _compare_frames(
        self,
        existing: GroundSourceFrame,
        near_points: torch.Tensor,
        near_valid: torch.Tensor,
        heights: torch.Tensor,
        positions: torch.Tensor,
        valid: torch.Tensor,
        near_angle_keys: torch.Tensor | None,
        near_source_dirs_sensor: torch.Tensor | None,
        near_source_dirs_present: torch.Tensor | None,
        near_source_indices: torch.Tensor | None,
        time_s: float,
        persistent_scan_count: int,
        scan_pos_w: torch.Tensor | None,
        scan_quat_wxyz: torch.Tensor | None,
    ) -> list[str]:
        """逐位比较不可变字段，返回差异字段名列表（空列表=一致）。"""
        differences: list[str] = []
        if not torch.equal(existing.near_points, near_points):
            differences.append("near_points")
        if not torch.equal(existing.near_valid, near_valid):
            differences.append("near_valid")
        for name, old, new in (
            ("near_angle_keys", existing.near_angle_keys, near_angle_keys),
            ("near_source_dirs_sensor", existing.near_source_dirs_sensor, near_source_dirs_sensor),
            ("near_source_dirs_present", existing.near_source_dirs_present, near_source_dirs_present),
            ("near_source_indices", existing.near_source_indices, near_source_indices),
        ):
            if (old is None) != (new is None):
                differences.append(name)
            elif old is not None and not torch.equal(old, new):
                differences.append(name)
        if not torch.equal(existing.heights, heights):
            differences.append("heights")
        if not torch.equal(existing.positions, positions):
            differences.append("positions")
        if not torch.equal(existing.valid, valid):
            differences.append("valid")
        if existing.time_s != time_s:
            differences.append("time_s")
        if existing.persistent_scan_count != persistent_scan_count:
            differences.append("persistent_scan_count")
        if (existing.scan_pos_w is None) != (scan_pos_w is None):
            differences.append("scan_pos_w")
        elif existing.scan_pos_w is not None and not torch.equal(existing.scan_pos_w, scan_pos_w):
            differences.append("scan_pos_w")
        if (existing.scan_quat_wxyz is None) != (scan_quat_wxyz is None):
            differences.append("scan_quat_wxyz")
        elif existing.scan_quat_wxyz is not None and not torch.equal(existing.scan_quat_wxyz, scan_quat_wxyz):
            differences.append("scan_quat_wxyz")
        return differences

    # ── 统计与桶划分 ────────────────────────────────────────────────────────

    def frames(self) -> tuple[GroundSourceFrame, ...]:
        """插入序全 SOURCE 帧（诊断/独立复算用）。"""
        return tuple(self._frames)

    def supervision_frames(self) -> tuple[GroundSourceFrame, ...]:
        """有效独立帧集合 S（geometry_valid.any()；先去重后筛选）。"""
        return tuple(frame for frame in self._frames if frame.supervises)

    def stats(self) -> dict:
        """每帧有效位置数与 S 统计（P3 两级平均与 4/|S| 权重的分母依据）。"""
        frames = self._frames
        supervision = [frame for frame in frames if frame.supervises]
        return {
            "total_source_frames": len(frames),
            "supervision_frames_S": len(supervision),
            "all_invalid_frames_excluded": len(frames) - len(supervision),
            "per_frame_valid_positions": [frame.valid_position_count for frame in frames],
            "supervision_per_frame_valid_positions": [
                frame.valid_position_count for frame in supervision
            ],
            "ground_samples": self.ground_samples,
            "grid_identity_hash": self.grid_identity_hash,
        }

    def bucket_partition(self, *, num_buckets: int = GROUND_NUM_BUCKETS, epoch: int = 0) -> tuple[list[int], ...]:
        """S 的确定性均衡划分：返回 ``num_buckets`` 个帧下标列表。

        下标指向 ``frames()``；并集恰为 S、两两不相交、每帧恰入一桶；桶大小
        差至多 1（``4/|S|`` 统一权重的均衡布局）；|S|<桶数时允许空桶。epoch
        以确定性轮转改变分配（多 epoch 复用同 S 正常），不引入随机 shuffle。
        """
        if isinstance(num_buckets, bool) or not isinstance(num_buckets, int) or num_buckets < 1:
            raise TrainingRolloutError(f"num_buckets 须为 >=1 整数，实际 {num_buckets!r}")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise TrainingRolloutError(f"epoch 须为非负 int，实际 {epoch!r}")
        order = [index for index, frame in enumerate(self._frames) if frame.supervises]
        buckets: list[list[int]] = [[] for _ in range(num_buckets)]
        if not order:
            return tuple(buckets)
        rotation = int(epoch) % len(order)
        rotated = order[rotation:] + order[:rotation]
        base, remainder = divmod(len(rotated), num_buckets)
        start = 0
        for bucket_index in range(num_buckets):
            end = start + base + (1 if bucket_index < remainder else 0)
            buckets[bucket_index] = rotated[start:end]
            start = end
        return tuple(buckets)

    def clear(self) -> None:
        """清空（跨 rollout 重新允许同 key；不永久去重）。"""
        self._frames.clear()
        self._index.clear()


# ═══════════════════════════ 单步样本与 buffer ═══════════════════════════


@dataclass(frozen=True)
class RolloutStepSample:
    """一个控制步（全部 env 行）的生产采样样本（append 时完整验证后入库）。

    observation 为 pre-action o_k（含 command=c_k）；command 为该转移实际
    计分的旧指令（必须与 observation.command 逐位一致）；old_* 为采样时刻
    分布状态（old_value 已 squeeze 为 (B,)）；v_next 为自然边界 final 观测
    的价值（非 reset 后状态）；final_*/next_* 只存身份（env 不变，存
    episode/step 与控制步）；envelope_label 与 pre-action 身份逐行绑定。
    """

    control_id: int
    time_s: float
    env_ids: torch.Tensor                    # (B,) int64 真实环境身份（行序全 rollout 固定）
    episode_ids: torch.Tensor                # (B,) int64 pre-action episode
    step_ids: torch.Tensor                   # (B,) int64 pre-action episode 内控制步
    observation: OnlineObservation           # pre-action o_k（未缩放命名字段）
    command: torch.Tensor                    # (B,3) 旧指令 c_k
    near_points: torch.Tensor                # (B,256,3)
    near_valid: torch.Tensor                 # (B,256)
    far_points: torch.Tensor                 # (B,640,3)
    far_valid: torch.Tensor                  # (B,640)
    near_filled: torch.Tensor                # (B,) bool
    near_scan_ids: torch.Tensor              # (B,) int64
    near_times: torch.Tensor                 # (B,) float32
    far_point_scan_ids: torch.Tensor         # (B,640) int64
    far_point_times: torch.Tensor            # (B,640) float32
    raw_action: torch.Tensor                 # (B,18) 未裁剪采样
    old_log_prob: torch.Tensor               # (B,) 事件级
    old_mean: torch.Tensor                   # (B,18)
    old_std: torch.Tensor                    # (B,18)
    old_value: torch.Tensor                  # (B,)（原 (B,1) 已显式 squeeze）
    reward: torch.Tensor                     # (B,) 实际 reward（原样，不改）
    terminated: torch.Tensor                 # (B,) bool 独立
    truncated: torch.Tensor                  # (B,) bool 独立
    v_next: torch.Tensor                     # (B,) 自然边界价值
    final_episode_ids: torch.Tensor          # (B,) int64 自然 final 身份
    final_step_ids: torch.Tensor             # (B,) int64
    final_control_id: int                    # 自然边界控制步（= control_id + 1）
    next_episode_ids: torch.Tensor           # (B,) int64 reset 后 next 身份
    next_step_ids: torch.Tensor              # (B,) int64
    reset_applied: torch.Tensor              # (B,) bool 该行是否发生选中 reset
    envelope_label: EnvelopeLabel            # pre-action 包络标签 E*
    engineering_stopped: bool = False        # 本步触发工程停止（残缺批标记）
    near_angle_keys: torch.Tensor | None = None
    far_angle_keys: torch.Tensor | None = None
    near_source_dirs_sensor: torch.Tensor | None = None
    far_source_dirs_sensor: torch.Tensor | None = None
    near_source_dirs_present: torch.Tensor | None = None
    far_source_dirs_present: torch.Tensor | None = None
    near_source_indices: torch.Tensor | None = None
    far_source_indices: torch.Tensor | None = None


@dataclass(frozen=True)
class TrainingReplayBatch:
    """``replay(indices)`` 的输出：按任意 indices 重建的重放批次（P3 消费）。

    prepared 为重建的 ``PreparedControlInputs``（OnlineObservation + 256/640
    点/掩码/扫描身份），可直接喂当前模型求梯度；advantage/returns 已含 GAE
    （P2 不归一化；全 rollout 标准化属 P3）。indices 可重复/乱序：env 身份
    随行保留，绝不把 env 当全局连续下标。
    """

    prepared: PreparedControlInputs
    raw_action: torch.Tensor       # (K,18)
    old_log_prob: torch.Tensor     # (K,)
    old_mean: torch.Tensor         # (K,18)
    old_std: torch.Tensor          # (K,18)
    old_value: torch.Tensor        # (K,)
    reward: torch.Tensor           # (K,)
    terminated: torch.Tensor       # (K,) bool
    truncated: torch.Tensor        # (K,) bool
    advantage: torch.Tensor        # (K,)（未归一化）
    returns: torch.Tensor          # (K,) = advantage + old_value
    v_next: torch.Tensor           # (K,) 自然边界价值（证据/独立复算用）
    envelope_values: torch.Tensor  # (K,6)
    envelope_valid: torch.Tensor   # (K,) bool
    env_ids: torch.Tensor          # (K,) int64 真实身份（可重复）
    episode_ids: torch.Tensor      # (K,) int64
    step_ids: torch.Tensor         # (K,) int64
    control_ids: torch.Tensor      # (K,) int64


class TrainingRolloutBuffer:
    """名义 24 步 × B 的生产采样 buffer（新训练 schema；非 mean_dev）。

    生命周期：``append``（恰 nominal 步收满）→ ``compute_gae``/``replay`` →
    ``consume``（P3 更新完成后标记，之后拒绝重放）→ ``clear``（清空后开始
    下一 rollout）。地面 SOURCE 帧由 ``ground_frames`` 持有并随 ``clear``
    一并清空（不跨 rollout 堆积）。收集身份（分布身份冻结串 + 模型指纹）
    由采集层 ``bind_collection_identity`` 一次绑定、之后不可变。
    """

    def __init__(
        self,
        *,
        num_envs: int,
        ground_samples: int,
        grid_identity_hash: str,
        device: torch.device,
        nominal_steps: int = NOMINAL_ROLLOUT_STEPS,
        control_dt_s: float = _CONTROL_DT_S,
    ) -> None:
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
            raise TrainingRolloutError(f"num_envs 须为 >=1 整数，实际 {num_envs!r}")
        if isinstance(nominal_steps, bool) or not isinstance(nominal_steps, int) or nominal_steps < 1:
            raise TrainingRolloutError(f"nominal_steps 须为 >=1 整数，实际 {nominal_steps!r}")
        if not math.isfinite(float(control_dt_s)) or float(control_dt_s) <= 0.0:
            raise TrainingRolloutError(f"control_dt_s 须为正有限值，实际 {control_dt_s!r}")
        if not isinstance(device, torch.device):
            raise TrainingRolloutError("device 须为 torch.device")
        self.num_envs = int(num_envs)
        self.nominal_steps = int(nominal_steps)
        self.control_dt_s = float(control_dt_s)
        self.device = device
        self.schema_version = TRAINING_ROLLOUT_SCHEMA_VERSION
        self.ground_frames = GroundFrameStore(
            ground_samples=ground_samples, grid_identity_hash=grid_identity_hash, device=device
        )

        steps, batch = self.nominal_steps, self.num_envs

        def zeros_f(shape: tuple[int, ...]) -> torch.Tensor:
            return torch.zeros((steps, batch, *shape), dtype=FLOAT_DTYPE, device=device)

        def zeros_i(shape: tuple[int, ...]) -> torch.Tensor:
            return torch.zeros((steps, batch, *shape), dtype=INT_DTYPE, device=device)

        def zeros_b(shape: tuple[int, ...]) -> torch.Tensor:
            return torch.zeros((steps, batch, *shape), dtype=torch.bool, device=device)

        # 观测（未缩放命名字段；proprio/指令原样）
        self._obs_time_s = zeros_f(())
        self._obs_lin_vel = zeros_f((3,))
        self._obs_ang_vel = zeros_f((3,))
        self._obs_gravity = zeros_f((3,))
        self._obs_joint_pos = zeros_f((JOINT_DIM,))
        self._obs_joint_vel = zeros_f((JOINT_DIM,))
        self._obs_prev_action = zeros_f((JOINT_DIM,))
        self._obs_command = zeros_f((COMMAND_DIM,))
        self._envelope_values = zeros_f((ENVELOPE_DIM,))
        self._envelope_valid = zeros_b(())
        # 点块与扫描身份
        self._near_points = zeros_f((NEAR_POINTS, 3))
        self._near_valid = zeros_b((NEAR_POINTS,))
        self._far_points = zeros_f((_FAR_FLAT, 3))
        self._far_valid = zeros_b((_FAR_FLAT,))
        self._near_filled = zeros_b(())
        self._near_scan_ids = zeros_i(())
        self._near_times = zeros_f(())
        self._far_point_scan_ids = zeros_i((_FAR_FLAT,))
        self._far_point_times = zeros_f((_FAR_FLAT,))
        # P8 source-ray provenance is rollout metadata only; it never enters
        # the network channel and remains aligned with the sorted point blocks.
        self._near_angle_keys = zeros_f((NEAR_POINTS,))
        self._far_angle_keys = zeros_f((_FAR_FLAT,))
        self._near_source_dirs_sensor = zeros_f((NEAR_POINTS, 3))
        self._far_source_dirs_sensor = zeros_f((_FAR_FLAT, 3))
        self._near_source_dirs_present = zeros_b((NEAR_POINTS,))
        self._far_source_dirs_present = zeros_b((_FAR_FLAT,))
        self._near_source_indices = torch.full(
            (steps, batch, NEAR_POINTS), -1, dtype=INT_DTYPE, device=device
        )
        self._far_source_indices = torch.full(
            (steps, batch, _FAR_FLAT), -1, dtype=INT_DTYPE, device=device
        )
        # 采样分布状态与转移量
        self._raw_action = zeros_f((JOINT_DIM,))
        self._old_log_prob = zeros_f(())
        self._old_mean = zeros_f((JOINT_DIM,))
        self._old_std = zeros_f((JOINT_DIM,))
        self._old_value = zeros_f(())
        self._reward = zeros_f(())
        self._v_next = zeros_f(())
        self._terminated = zeros_b(())
        self._truncated = zeros_b(())
        # 边界身份
        self._final_episode_ids = zeros_i(())
        self._final_step_ids = zeros_i(())
        self._next_episode_ids = zeros_i(())
        self._next_step_ids = zeros_i(())
        self._reset_applied = zeros_b(())
        # pre-action 身份存储（env 行序由首个 append 注册；之后必须逐位
        # 一致——GAE 沿固定行槽递推）
        self._pre_episode_ids = zeros_i(())
        self._pre_step_ids = zeros_i(())
        self._env_ids: torch.Tensor | None = None
        # 每 step 标量身份
        self._control_ids: list[int] = []
        self._step_times: list[float] = []
        self._engineering_stopped: list[bool] = []
        self._seen_identities: set[tuple[int, int, int]] = set()
        # 收集身份（分布/模型指纹；一次绑定）
        self._distribution_identity_frozen: str | None = None
        self._model_fingerprint: str | None = None
        # 生命周期与 GAE 结果
        self._filled = 0
        self._consumed = False
        self._advantage: torch.Tensor | None = None
        self._returns: torch.Tensor | None = None

    # ── 生命周期视图 ────────────────────────────────────────────────────────

    @property
    def steps_filled(self) -> int:
        return self._filled

    @property
    def is_full(self) -> bool:
        return self._filled == self.nominal_steps

    @property
    def consumed(self) -> bool:
        return self._consumed

    @property
    def distribution_identity_frozen(self) -> str | None:
        return self._distribution_identity_frozen

    @property
    def model_fingerprint(self) -> str | None:
        return self._model_fingerprint

    @property
    def advantage(self) -> torch.Tensor:
        """(T,B) 未归一化优势（须已 compute_gae；返回独立副本）。"""
        self._ensure_gae("advantage")
        assert self._advantage is not None
        return self._advantage.clone()

    @property
    def returns(self) -> torch.Tensor:
        """(T,B) = advantage + old_value（须已 compute_gae；独立副本）。"""
        self._ensure_gae("returns")
        assert self._returns is not None
        return self._returns.clone()

    def bind_collection_identity(
        self,
        *,
        distribution_identity_frozen: str,
        model_fingerprint: str,
    ) -> None:
        """绑定收集身份（provider 身份冻结串 + 模型参数指纹）；一次、不可变。"""
        if self._distribution_identity_frozen is not None or self._model_fingerprint is not None:
            raise TrainingRolloutStateError("收集身份已绑定，不可重复绑定")
        if not isinstance(distribution_identity_frozen, str) or not distribution_identity_frozen.strip():
            raise TrainingRolloutError("distribution_identity_frozen 须为非空字符串")
        if not isinstance(model_fingerprint, str) or not model_fingerprint.strip():
            raise TrainingRolloutError("model_fingerprint 须为非空字符串")
        self._distribution_identity_frozen = str(distribution_identity_frozen)
        self._model_fingerprint = str(model_fingerprint)

    def _require_collecting(self, api: str) -> None:
        if self._consumed:
            raise TrainingRolloutStateError(f"{api}：buffer 已 consume（先 clear 再开始下一 rollout）")
        if self._filled >= self.nominal_steps:
            raise TrainingRolloutStateError(
                f"{api}：buffer 已收满 {self.nominal_steps} 步（不伪装更多步；先 consume/clear）"
            )

    def _require_full(self, api: str) -> None:
        if self._filled != self.nominal_steps:
            raise TrainingRolloutStateError(
                f"{api} 要求恰 {self.nominal_steps} 步完整 rollout，实际 {self._filled} 步"
                "（残缺批不得伪装完整；工程停止的残缺数据只作证据留存）"
            )
        if self._consumed:
            raise TrainingRolloutStateError(f"{api}：buffer 已 consume，重放窗口关闭（先 clear）")

    # ── append（完整验证后写预分配张量） ────────────────────────────────────

    @torch.no_grad()
    def append(self, sample: RolloutStepSample) -> None:
        """验证并写入一个控制步样本（身份推进/绑定/形状/有限性全查后入库）。"""
        self._require_collecting("append")
        if not isinstance(sample, RolloutStepSample):
            raise TrainingRolloutError(
                f"sample 必须为 RolloutStepSample，实际 {type(sample).__name__}"
            )
        batch = self.num_envs
        device = self.device
        index = self._filled

        observation = sample.observation
        observation.validate()
        if int(observation.env_ids.shape[0]) != batch:
            raise TrainingRolloutError(
                f"观测批量 {int(observation.env_ids.shape[0])} != buffer 批量 {batch}"
            )
        if observation.env_ids.device != device:
            raise TrainingRolloutError(f"观测设备 {observation.env_ids.device} 与 buffer {device} 不一致")
        if not math.isfinite(float(sample.time_s)) or not bool(torch.allclose(
            observation.time_s, torch.full_like(observation.time_s, float(sample.time_s)),
            atol=_TIME_TOL_S, rtol=0.0,
        )):
            raise TrainingRolloutError("sample time_s 与动作前观测时间不一致或非有限")
        if self._env_ids is not None and not torch.equal(self._env_ids, observation.env_ids):
            raise TrainingRolloutError(
                "env 行序与首个控制步不一致（GAE 沿固定行槽递推；行序改变须另开 rollout）"
            )
        for name, value in (
            ("env_ids", sample.env_ids),
            ("episode_ids", sample.episode_ids),
            ("step_ids", sample.step_ids),
        ):
            _require_tensor(name, value, dtype=INT_DTYPE, shape=(batch,), device=device)
        if not (
            torch.equal(sample.env_ids, observation.env_ids)
            and torch.equal(sample.episode_ids, observation.episode_ids)
            and torch.equal(sample.step_ids, observation.step_ids)
        ):
            raise TrainingRolloutError("sample 身份字段与 observation 身份不一致")

        # 控制身份推进：control_id 严格 +1、时间严格递增且步长 dt。
        if self._control_ids:
            previous_control = self._control_ids[-1]
            previous_time = self._step_times[-1]
            if int(sample.control_id) != previous_control + 1:
                raise TrainingRolloutError(
                    f"control_id 未严格推进（前 {previous_control}，现 {int(sample.control_id)}；"
                    "未来/回跳/重复身份拒绝）"
                )
            if not (float(sample.time_s) > previous_time) or \
                    abs(float(sample.time_s) - (previous_time + self.control_dt_s)) > _TIME_TOL_S:
                raise TrainingRolloutError(
                    f"time_s 未按控制步长推进（前 {previous_time}，现 {float(sample.time_s)}，"
                    f"dt={self.control_dt_s}）"
                )
        if int(sample.final_control_id) != int(sample.control_id) + 1:
            raise TrainingRolloutError(
                f"final_control_id 应为 control_id+1，实际 {int(sample.final_control_id)}"
            )

        # 每行 (env, episode, step) 身份在全 rollout 内唯一（重复=数据修改/回放）。
        # 只检查不写入：所有验证通过后才提交状态（失败不留下幽灵身份）。
        env_list = observation.env_ids.detach().cpu().tolist()
        episode_list = observation.episode_ids.detach().cpu().tolist()
        step_list = observation.step_ids.detach().cpu().tolist()
        pending_identities = set()
        for row in range(batch):
            identity = (int(env_list[row]), int(episode_list[row]), int(step_list[row]))
            if identity in self._seen_identities or identity in pending_identities:
                raise TrainingRolloutError(f"重复控制身份 (env,episode,step)={identity}")
            pending_identities.add(identity)

        # 包络标签与 pre-action 观测逐行绑定（未来/旧 episode/错位拒绝）。
        try:
            bind_envelope_label(observation, sample.envelope_label)
        except (ContractError, IdentityMismatchError) as err:
            raise TrainingRolloutError(f"envelope 标签未绑定到 pre-action 身份: {err}") from err

        # 点块与扫描身份元数据。
        for name, block in (("near_points", sample.near_points), ("far_points", sample.far_points)):
            if int(block.shape[0]) != batch or block.device != device:
                raise TrainingRolloutError(f"{name} 批量/设备与 buffer 不一致")
        validate_point_block(sample.near_points, sample.near_valid, n_points=NEAR_POINTS, name="near_points")
        validate_point_block(sample.far_points, sample.far_valid, n_points=_FAR_FLAT, name="far_points")
        if (sample.near_angle_keys is None) != (sample.far_angle_keys is None):
            raise TrainingRolloutError("near/far prepared angle keys must be supplied together")
        source_metadata = (
            sample.near_source_dirs_sensor, sample.far_source_dirs_sensor,
            sample.near_source_dirs_present, sample.far_source_dirs_present,
            sample.near_source_indices, sample.far_source_indices,
        )
        if any(value is None for value in source_metadata) and any(
            value is not None for value in source_metadata
        ):
            raise TrainingRolloutError("prepared source rays, masks and raw ids must be supplied together")
        if sample.near_angle_keys is not None:
            for name, value, shape in (
                ("near_angle_keys", sample.near_angle_keys, (batch, NEAR_POINTS)),
                ("far_angle_keys", sample.far_angle_keys, (batch, _FAR_FLAT)),
            ):
                _require_tensor(name, value, dtype=FLOAT_DTYPE, shape=shape, device=device)
                _require_finite(name, value)
            for name, keys, valid in (
                ("near", sample.near_angle_keys, sample.near_valid),
                ("far", sample.far_angle_keys, sample.far_valid),
            ):
                if bool(valid.any()) and not bool(torch.isfinite(keys[valid]).all()):
                    raise TrainingRolloutError(f"{name} valid angle keys must be finite")
        if sample.near_source_dirs_sensor is not None:
            expected_metadata = (
                ("near_source_dirs_sensor", sample.near_source_dirs_sensor, FLOAT_DTYPE, (batch, NEAR_POINTS, 3)),
                ("far_source_dirs_sensor", sample.far_source_dirs_sensor, FLOAT_DTYPE, (batch, _FAR_FLAT, 3)),
                ("near_source_dirs_present", sample.near_source_dirs_present, torch.bool, (batch, NEAR_POINTS)),
                ("far_source_dirs_present", sample.far_source_dirs_present, torch.bool, (batch, _FAR_FLAT)),
                ("near_source_indices", sample.near_source_indices, INT_DTYPE, (batch, NEAR_POINTS)),
                ("far_source_indices", sample.far_source_indices, INT_DTYPE, (batch, _FAR_FLAT)),
            )
            for name, value, dtype, shape in expected_metadata:
                _require_tensor(name, value, dtype=dtype, shape=shape, device=device)
                if dtype is FLOAT_DTYPE:
                    _require_finite(name, value)
            for name, valid, present, rays, ray_ids in (
                ("near", sample.near_valid, sample.near_source_dirs_present,
                 sample.near_source_dirs_sensor, sample.near_source_indices),
                ("far", sample.far_valid, sample.far_source_dirs_present,
                 sample.far_source_dirs_sensor, sample.far_source_indices),
            ):
                if bool((present & ~valid).any()) or not torch.equal(present, ray_ids >= 0):
                    raise TrainingRolloutError(f"{name} ray metadata masks/indices disagree")
                if bool(present.any()):
                    norms = torch.linalg.vector_norm(rays[present], dim=-1)
                    if not bool(torch.allclose(norms, torch.ones_like(norms), atol=1e-5, rtol=0.0)):
                        raise TrainingRolloutError(f"{name} source directions must be unit vectors")
                if bool((~present).any()) and not torch.equal(rays[~present], torch.zeros_like(rays[~present])):
                    raise TrainingRolloutError(f"{name} missing source directions must be exact zero")
        for name, value, dtype, shape in (
            ("near_filled", sample.near_filled, torch.bool, (batch,)),
            ("near_scan_ids", sample.near_scan_ids, INT_DTYPE, (batch,)),
            ("near_times", sample.near_times, FLOAT_DTYPE, (batch,)),
            ("far_point_scan_ids", sample.far_point_scan_ids, INT_DTYPE, (batch, _FAR_FLAT)),
            ("far_point_times", sample.far_point_times, FLOAT_DTYPE, (batch, _FAR_FLAT)),
        ):
            _require_tensor(name, value, dtype=dtype, shape=shape, device=device)
        # 未来扫描拒绝（近端已填充行：扫描时间不得晚于本控制步时间）。
        filled_rows = sample.near_filled
        if bool(filled_rows.any()):
            future = filled_rows & (
                sample.near_times > observation.time_s + _TIME_TOL_S
            )
            if bool(future.any()):
                raise TrainingRolloutError(
                    f"近端扫描时间晚于控制步时间（未来扫描，行 {future.nonzero().flatten().tolist()}）"
                )

        # 采样分布状态与转移量（old_value/v_next/reward/log_prob 一律 (B,)）。
        for name, value, shape in (
            ("raw_action", sample.raw_action, (batch, JOINT_DIM)),
            ("old_mean", sample.old_mean, (batch, JOINT_DIM)),
            ("old_std", sample.old_std, (batch, JOINT_DIM)),
            ("old_log_prob", sample.old_log_prob, (batch,)),
            ("old_value", sample.old_value, (batch,)),
            ("v_next", sample.v_next, (batch,)),
            ("reward", sample.reward, (batch,)),
        ):
            _require_tensor(name, value, dtype=FLOAT_DTYPE, shape=shape, device=device)
            _require_finite(name, value)
        if bool((sample.old_std <= 0).any()):
            raise TrainingRolloutError("old_std 须处处为正（std=exp(log_std)；无 clamp）")
        for name, value in (("terminated", sample.terminated), ("truncated", sample.truncated)):
            _require_tensor(name, value, dtype=torch.bool, shape=(batch,), device=device)
        # 旧指令身份：c_k 必须就是 o_k 的指令字段。
        if not torch.equal(sample.command, observation.command):
            raise TrainingRolloutError("command 与 observation.command 不一致（c_k 身份绑定）")

        # 边界身份。
        for name, value in (
            ("final_episode_ids", sample.final_episode_ids),
            ("final_step_ids", sample.final_step_ids),
            ("next_episode_ids", sample.next_episode_ids),
            ("next_step_ids", sample.next_step_ids),
        ):
            _require_tensor(name, value, dtype=INT_DTYPE, shape=(batch,), device=device)
        _require_tensor("reset_applied", sample.reset_applied, dtype=torch.bool, shape=(batch,), device=device)
        if bool((sample.reset_applied & ~(sample.terminated | sample.truncated)).any()):
            raise TrainingRolloutError("reset_applied 行必须 terminated 或 truncated（选中 reset 协议）")
        if not torch.equal(sample.final_episode_ids, sample.episode_ids) or not torch.equal(
            sample.final_step_ids, sample.step_ids + 1
        ):
            raise TrainingRolloutError("final 身份必须是原episode的自然下一控制步")
        expected_episode = torch.where(sample.reset_applied, sample.episode_ids + 1, sample.final_episode_ids)
        expected_step = torch.where(sample.reset_applied, torch.zeros_like(sample.step_ids), sample.final_step_ids)
        if not torch.equal(sample.next_episode_ids, expected_episode) or not torch.equal(sample.next_step_ids, expected_step):
            raise TrainingRolloutError("next 身份与显式选中reset不一致")

        # 写入（全部 detach+clone 语义：预分配 copy_ 后与源无别名）。
        self._obs_time_s[index] = observation.time_s
        self._obs_lin_vel[index] = observation.lin_vel_b
        self._obs_ang_vel[index] = observation.ang_vel_b
        self._obs_gravity[index] = observation.gravity_b
        self._obs_joint_pos[index] = observation.joint_pos_rel_ref
        self._obs_joint_vel[index] = observation.joint_vel
        self._obs_prev_action[index] = observation.prev_action
        self._obs_command[index] = observation.command
        self._near_points[index] = sample.near_points
        self._near_valid[index] = sample.near_valid
        self._far_points[index] = sample.far_points
        self._far_valid[index] = sample.far_valid
        self._near_filled[index] = sample.near_filled
        self._near_scan_ids[index] = sample.near_scan_ids
        self._near_times[index] = sample.near_times
        self._far_point_scan_ids[index] = sample.far_point_scan_ids
        self._far_point_times[index] = sample.far_point_times
        if sample.near_angle_keys is not None:
            self._near_angle_keys[index] = sample.near_angle_keys
            self._far_angle_keys[index] = sample.far_angle_keys
        else:
            self._near_angle_keys[index].zero_()
            self._far_angle_keys[index].zero_()
        if sample.near_source_dirs_sensor is not None:
            self._near_source_dirs_sensor[index] = sample.near_source_dirs_sensor
            self._far_source_dirs_sensor[index] = sample.far_source_dirs_sensor
            self._near_source_dirs_present[index] = sample.near_source_dirs_present
            self._far_source_dirs_present[index] = sample.far_source_dirs_present
            self._near_source_indices[index] = sample.near_source_indices
            self._far_source_indices[index] = sample.far_source_indices
        else:
            self._near_source_dirs_sensor[index].zero_()
            self._far_source_dirs_sensor[index].zero_()
            self._near_source_dirs_present[index].zero_()
            self._far_source_dirs_present[index].zero_()
            self._near_source_indices[index].fill_(-1)
            self._far_source_indices[index].fill_(-1)
        self._raw_action[index] = sample.raw_action
        self._old_log_prob[index] = sample.old_log_prob
        self._old_mean[index] = sample.old_mean
        self._old_std[index] = sample.old_std
        self._old_value[index] = sample.old_value
        self._reward[index] = sample.reward
        self._v_next[index] = sample.v_next
        self._terminated[index] = sample.terminated
        self._truncated[index] = sample.truncated
        self._final_episode_ids[index] = sample.final_episode_ids
        self._final_step_ids[index] = sample.final_step_ids
        self._next_episode_ids[index] = sample.next_episode_ids
        self._next_step_ids[index] = sample.next_step_ids
        self._reset_applied[index] = sample.reset_applied
        self._pre_episode_ids[index] = observation.episode_ids
        self._pre_step_ids[index] = observation.step_ids
        self._envelope_values[index] = sample.envelope_label.values
        self._envelope_valid[index] = sample.envelope_label.valid
        self._control_ids.append(int(sample.control_id))
        self._step_times.append(float(sample.time_s))
        self._engineering_stopped.append(bool(sample.engineering_stopped))
        if self._env_ids is None:
            self._env_ids = observation.env_ids.detach().clone()
        self._seen_identities.update(pending_identities)
        self._filled += 1

    # ── GAE ─────────────────────────────────────────────────────────────────

    def compute_gae(self) -> tuple[torch.Tensor, torch.Tensor]:
        """按已确认公式计算 (advantage, returns)（γ=0.99、λ=0.95；不归一化）。

        δ_t = r_t + γ(1−terminated_t)·V_next_t − V_t（真终止 bootstrap 为 0；
        timeout 的 V_next 是终止前自然边界价值）；反向递推延续
        continuation = 1−(terminated OR truncated)；returns = adv + V。
        本批不做全 rollout 优势标准化（属 P3）。original reward 原样使用。
        """
        self._require_full("compute_gae")
        advantage = torch.zeros(self.nominal_steps, self.num_envs, dtype=FLOAT_DTYPE, device=self.device)
        running = torch.zeros(self.num_envs, dtype=FLOAT_DTYPE, device=self.device)
        for index in range(self.nominal_steps - 1, -1, -1):
            not_terminal = (~self._terminated[index]).to(FLOAT_DTYPE)
            delta = (
                self._reward[index]
                + GAE_GAMMA * not_terminal * self._v_next[index]
                - self._old_value[index]
            )
            continuation = (~(self._terminated[index] | self._truncated[index])).to(FLOAT_DTYPE)
            running = delta + GAE_GAMMA * GAE_LAMBDA * continuation * running
            advantage[index] = running
        returns = advantage + self._old_value
        if not bool(torch.isfinite(advantage).all()) or not bool(torch.isfinite(returns).all()):
            raise TrainingRolloutError("GAE 结果含 NaN/Inf")
        self._advantage = advantage
        self._returns = returns
        return advantage.clone(), returns.clone()

    def _ensure_gae(self, api: str) -> None:
        if self._advantage is None or self._returns is None:
            self.compute_gae()

    # ── flatten / replay ────────────────────────────────────────────────────

    def flat_identities(self) -> dict:
        """(T·B,) 展平身份视图（env 可重复=真实布局；不做全局连续下标假设）。"""
        self._require_full("flat_identities")
        assert self._env_ids is not None
        control = torch.tensor(self._control_ids, dtype=INT_DTYPE, device=self.device)
        return {
            "env_ids": self._env_ids.expand(self.nominal_steps, self.num_envs).reshape(-1).clone(),
            "episode_ids": self._pre_episode_ids.reshape(-1).clone(),
            "step_ids": self._pre_step_ids.reshape(-1).clone(),
            "control_ids": control[:, None].expand(self.nominal_steps, self.num_envs).reshape(-1).clone(),
        }

    def replay(self, indices: torch.Tensor) -> TrainingReplayBatch:
        """按任意 indices（(K,) int64，可重复/乱序，范围 [0, T·B)）重建批次。

        重建的 ``PreparedControlInputs`` 可直接喂当前模型（P3 对当前参数求
        梯度；输入本身无 grad）。GAE 惰性计算（未算则先 compute_gae）。
        """
        self._require_full("replay")
        if not isinstance(indices, torch.Tensor) or indices.dim() != 1 or indices.dtype is not INT_DTYPE:
            raise TrainingRolloutError(f"indices 须为 (K,) int64 一维张量，实际 {type(indices).__name__}")
        if indices.device != self.device:
            raise TrainingRolloutError(f"indices 设备 {indices.device} 与 buffer {self.device} 不一致")
        total = self.nominal_steps * self.num_envs
        if int(indices.numel()) > 0 and bool((indices < 0).any() | (indices >= total).any()):
            raise TrainingRolloutError(f"indices 超出 [0, {total})")
        self._ensure_gae("replay")
        assert self._advantage is not None and self._returns is not None

        def gather(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(total, *tensor.shape[2:]).index_select(0, indices)

        observation = OnlineObservation(
            env_ids=gather(self._env_ids.expand(self.nominal_steps, self.num_envs)),
            episode_ids=gather(self._pre_episode_ids),
            step_ids=gather(self._pre_step_ids),
            time_s=gather(self._obs_time_s),
            lin_vel_b=gather(self._obs_lin_vel),
            ang_vel_b=gather(self._obs_ang_vel),
            gravity_b=gather(self._obs_gravity),
            joint_pos_rel_ref=gather(self._obs_joint_pos),
            joint_vel=gather(self._obs_joint_vel),
            prev_action=gather(self._obs_prev_action),
            command=gather(self._obs_command),
        )
        prepared = PreparedControlInputs(
            observation=observation,
            near_points=gather(self._near_points),
            near_valid=gather(self._near_valid),
            far_points=gather(self._far_points),
            far_valid=gather(self._far_valid),
            near_filled=gather(self._near_filled),
            near_scan_ids=gather(self._near_scan_ids),
            near_times=gather(self._near_times),
            far_point_scan_ids=gather(self._far_point_scan_ids),
            far_point_times=gather(self._far_point_times),
            near_angle_keys=gather(self._near_angle_keys),
            far_angle_keys=gather(self._far_angle_keys),
            near_source_dirs_sensor=gather(self._near_source_dirs_sensor),
            far_source_dirs_sensor=gather(self._far_source_dirs_sensor),
            near_source_dirs_present=gather(self._near_source_dirs_present),
            far_source_dirs_present=gather(self._far_source_dirs_present),
            near_source_indices=gather(self._near_source_indices),
            far_source_indices=gather(self._far_source_indices),
        )
        control = torch.tensor(self._control_ids, dtype=INT_DTYPE, device=self.device)
        return TrainingReplayBatch(
            prepared=prepared,
            raw_action=gather(self._raw_action),
            old_log_prob=gather(self._old_log_prob),
            old_mean=gather(self._old_mean),
            old_std=gather(self._old_std),
            old_value=gather(self._old_value),
            reward=gather(self._reward),
            terminated=gather(self._terminated),
            truncated=gather(self._truncated),
            advantage=gather(self._advantage),
            returns=gather(self._returns),
            v_next=gather(self._v_next),
            envelope_values=gather(self._envelope_values),
            envelope_valid=gather(self._envelope_valid),
            env_ids=gather(self._env_ids.expand(self.nominal_steps, self.num_envs)),
            episode_ids=gather(self._pre_episode_ids),
            step_ids=gather(self._pre_step_ids),
            control_ids=gather(control[:, None].expand(self.nominal_steps, self.num_envs)),
        )

    # ── 消费/清空/摘要 ──────────────────────────────────────────────────────

    def consume(self) -> None:
        """标记本 rollout 数据已被更新消费（要求先收满；之后拒绝重放）。"""
        self._require_full("consume")
        self._consumed = True

    def clear(self) -> None:
        """清空全部步数据与 SOURCE 帧（跨 rollout 同 key 重新允许）。"""
        self._filled = 0
        self._consumed = False
        self._advantage = None
        self._returns = None
        self._env_ids = None
        self._control_ids.clear()
        self._step_times.clear()
        self._engineering_stopped.clear()
        self._seen_identities.clear()
        self._distribution_identity_frozen = None
        self._model_fingerprint = None
        self.ground_frames.clear()

    def summary(self) -> dict:
        """身份/进度/地面统计摘要（检查与独立复算用；不含大张量）。"""
        return {
            "schema_version": self.schema_version,
            "nominal_steps": self.nominal_steps,
            "steps_filled": self._filled,
            "complete": self.is_full,
            "consumed": self._consumed,
            "num_envs": self.num_envs,
            "device": str(self.device),
            "distribution_identity_frozen": self._distribution_identity_frozen,
            "model_fingerprint": self._model_fingerprint,
            "control_ids": list(self._control_ids),
            "engineering_stopped_steps": [i for i, flag in enumerate(self._engineering_stopped) if flag],
            "ground": self.ground_frames.stats(),
        }
