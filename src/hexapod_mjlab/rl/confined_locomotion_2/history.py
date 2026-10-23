"""One-current-scan window for the recurrent confined-locomotion variant.

时序语义（雷达 10Hz、控制 50Hz，比值不写死在本模块）：
* 只写入“新完成”的扫描：每环境 scan_id 与 time 在 episode 内严格递增；
  同扫描重复提交、旧扫描、异 episode、非单调时间一律显式拒绝，不静默覆盖或忽略。
* 读取纯只读：50Hz 控制步反复读取不推进/不改动 10Hz 窗口。
* 近端槽同步保留最新完成扫描；远端单槽在每次新扫描时被最新扫描替换。
* 部分复位只清理目标环境，允许不同环境异步刷新、各自独立计数。
* reset 清空该环境扫描状态并切换 episode 身份（要求新 id 与旧 id 不同，防同 id
  复用导致旧 episode 扫描混入）；reset 后每到达一个新扫描即可用（含近端），
  空窗口状态可表示，不复制首扫描补足。reset 后能否在零扫描下
  做实际控制属闭环确认事项，本模块不假设。
* 隔离：写入深拷贝进内部缓冲；读取返回全新快照，后续写入/reset 或外部修改
  不影响已返回数据，避免与 rollout 共享可变存储。
* episode 身份前提：调用方为每环境分配全局唯一的 episode id；窗口记录各环境
  历史用过的全部 id，reset 时拒绝任何历史 id 的复用/回退（不只与当前不同），
  防止跨 reset 冒充新 episode。
* 可选角键元数据（``WindowAngleKeys``）：来自原始单位射线方向的排序键可
  随写入携带（``write(near_angle_keys=..., far_angle_keys=...)``，两者同时
  提供或缺省）；``read_angle_keys`` 与 ``read()`` 使用同一远端槽旋转。旧调用
  （不携带键）行为逐字不变——``read_angle_keys`` 返回 None，clean 路径无任何
  新开销。角键是排序/身份元数据，绝不进入网络输入通道。
* 可选源射线元数据（``WindowSourceDirections``）：传感器坐标系单位方向与原始
  射线索引随相同近端槽/远端环保存；旧写入调用仍可省略，零距离缺元数据由镜像消费端拒绝。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .contracts import (
    EMPTY_SCAN_ID,
    FAR_POINTS_PER_SCAN,
    FAR_WINDOW_SLOTS,
    FLOAT_DTYPE,
    INT_DTYPE,
    NEAR_POINTS,
    POINT_CHANNELS,
    ContractError,
    ScanWindowSnapshot,
    validate_point_block,
)


class EpisodeMismatchError(ContractError):
    """写入扫描的 episode 身份与窗口当前 episode 不一致。"""


class StaleScanError(ContractError):
    """scan_id 未大于该环境最新 scan_id（重复或旧扫描）。"""


class TimeOrderError(ContractError):
    """扫描时间未晚于该环境最新时间（episode 内要求严格递增）。"""


@dataclass(frozen=True)
class WindowAngleKeys:
    """P6 可选角键元数据快照（与 ``ScanWindowSnapshot`` 平行的只读读取）。

    近端 (B,256) 键与写入时 near_points 槽序对齐；远端 (B,1,128) 键与唯一
    远端槽对齐，故 far.reshape(B,128) 与快照 far_points 的展平顺序一一对应。
    ``present`` 标记每行键是否随其最近一次扫描写入（部分 reset 只清该行；
    随后无键写入会将该行 present 置回 False，绝不残留陈旧键）。键来自原始
    单位射线方向（P6 消费副本契约），不是从存储点坐标反推；缺失行用哑零
    填充，该行点位必为无效（掩码排除），哑键不参与任何有效点排序。角键是
    排序/身份元数据，不进入网络输入通道。
    """

    near: torch.Tensor        # (B,256) float32；present=False 行为哑 0
    far: torch.Tensor         # (B,1,128) float32；present=False 行为哑 0
    present: torch.Tensor     # (B,) bool；该行当前扫描是否携带键


@dataclass(frozen=True)
class WindowSourceDirections:
    """可选逐点源射线方向与原始射线身份。"""

    near: torch.Tensor          # (B,256,3) sensor-frame unit directions; missing=0
    far: torch.Tensor           # (B,1,128,3), same chronological slots as read()
    near_present: torch.Tensor  # (B,256) bool; only true for valid sourced points
    far_present: torch.Tensor   # (B,1,128) bool
    near_indices: torch.Tensor  # (B,256) int64 raw ray id, missing/invalid=-1
    far_indices: torch.Tensor  # (B,1,128) int64 raw ray id


def _as_int_vector(values: object, *, name: str, device: torch.device) -> torch.Tensor:
    """int64 一维向量：张量须精确 int64；序列安全转换（拒绝非整数元素）。"""
    if isinstance(values, torch.Tensor):
        if values.dtype is not INT_DTYPE:
            raise ContractError(f"{name} 张量 dtype 应为 {INT_DTYPE}，实际 {values.dtype}")
        out = values.to(device)
    else:
        try:
            sequence = list(values)
            probe = torch.tensor(sequence, dtype=torch.float64)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ContractError(f"{name} 须为 int64 张量或整数序列") from exc
        if bool((probe != probe.round()).any()):
            raise ContractError(f"{name} 含非整数元素")
        out = probe.to(INT_DTYPE).to(device)
    if out.dim() != 1:
        raise ContractError(f"{name} 必须是一维")
    return out


def _as_float_vector(values: object, *, name: str, device: torch.device) -> torch.Tensor:
    """float32 一维向量：张量须精确 float32（float64 输入显式拒绝）。"""
    if isinstance(values, torch.Tensor):
        if values.dtype is not FLOAT_DTYPE:
            raise ContractError(f"{name} 张量 dtype 应为 {FLOAT_DTYPE}，实际 {values.dtype}")
        return values.to(device)
    try:
        return torch.tensor(list(values), dtype=FLOAT_DTYPE, device=device)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ContractError(f"{name} 须为 float32 张量或浮点数序列") from exc


class ScanWindow:
    """批处理扫描窗口（纯框架组件，不接触仿真/传感实现）。"""

    def __init__(self, batch_size: int, *, episode_ids: object, device: object = None) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ContractError(f"batch_size 须为正整数，实际 {batch_size!r}")
        target = torch.device(device) if device is not None else torch.device("cpu")
        episodes = _as_int_vector(episode_ids, name="episode_ids", device=target)
        if int(episodes.shape[0]) != batch_size:
            raise ContractError(
                f"episode_ids 长度 {int(episodes.shape[0])} 与 batch_size {batch_size} 不一致"
            )
        if bool((episodes < 0).any()):
            raise ContractError("episode_ids 必须非负")
        self._device = target
        self._batch = batch_size
        self._episode_ids = episodes.clone()
        self._seen_episodes = [set() for _ in range(batch_size)]
        for env in range(batch_size):
            self._seen_episodes[env].add(int(episodes[env]))
        self._near_points = torch.zeros(batch_size, NEAR_POINTS, POINT_CHANNELS, dtype=FLOAT_DTYPE, device=target)
        self._near_valid = torch.zeros(batch_size, NEAR_POINTS, dtype=torch.bool, device=target)
        self._near_filled = torch.zeros(batch_size, dtype=torch.bool, device=target)
        self._near_scan_ids = torch.full((batch_size,), EMPTY_SCAN_ID, dtype=INT_DTYPE, device=target)
        self._near_times = torch.zeros(batch_size, dtype=FLOAT_DTYPE, device=target)
        self._far_points = torch.zeros(
            batch_size, FAR_WINDOW_SLOTS, FAR_POINTS_PER_SCAN, POINT_CHANNELS, dtype=FLOAT_DTYPE, device=target
        )
        self._far_valid = torch.zeros(batch_size, FAR_WINDOW_SLOTS, FAR_POINTS_PER_SCAN, dtype=torch.bool, device=target)
        self._far_scan_ids = torch.full((batch_size, FAR_WINDOW_SLOTS), EMPTY_SCAN_ID, dtype=INT_DTYPE, device=target)
        self._far_times = torch.zeros(batch_size, FAR_WINDOW_SLOTS, dtype=FLOAT_DTYPE, device=target)
        self._far_count = torch.zeros(batch_size, dtype=INT_DTYPE, device=target)
        self._far_pos = torch.zeros(batch_size, dtype=INT_DTYPE, device=target)
        self._last_scan_id = torch.full((batch_size,), EMPTY_SCAN_ID, dtype=INT_DTYPE, device=target)
        self._last_time = torch.full((batch_size,), float("-inf"), dtype=FLOAT_DTYPE, device=target)
        # P6 可选角键元数据：未携带键的写入保持 present=False；哑 0 仅出现
        # 在 present=False 行（该行点位必为无效，由掩码排除，不参与排序）。
        self._near_angle_keys = torch.zeros(
            batch_size, NEAR_POINTS, dtype=FLOAT_DTYPE, device=target
        )
        self._far_angle_keys = torch.zeros(
            batch_size, FAR_WINDOW_SLOTS, FAR_POINTS_PER_SCAN, dtype=FLOAT_DTYPE, device=target
        )
        self._angle_keys_present = torch.zeros(batch_size, dtype=torch.bool, device=target)
        self._near_source_dirs_sensor = torch.zeros(
            batch_size, NEAR_POINTS, 3, dtype=FLOAT_DTYPE, device=target
        )
        self._far_source_dirs_sensor = torch.zeros(
            batch_size, FAR_WINDOW_SLOTS, FAR_POINTS_PER_SCAN, 3,
            dtype=FLOAT_DTYPE, device=target,
        )
        self._near_source_dirs_present = torch.zeros(
            batch_size, NEAR_POINTS, dtype=torch.bool, device=target
        )
        self._far_source_dirs_present = torch.zeros(
            batch_size, FAR_WINDOW_SLOTS, FAR_POINTS_PER_SCAN,
            dtype=torch.bool, device=target,
        )
        self._near_source_indices = torch.full(
            (batch_size, NEAR_POINTS), -1, dtype=INT_DTYPE, device=target
        )
        self._far_source_indices = torch.full(
            (batch_size, FAR_WINDOW_SLOTS, FAR_POINTS_PER_SCAN),
            -1, dtype=INT_DTYPE, device=target,
        )
        # Distinguish a source-aware scan whose selected points are all
        # invalid (all per-point presence bits are False) from legacy writes
        # that never carried ray provenance. This is metadata only.
        self._source_dirs_attached = torch.zeros(batch_size, dtype=torch.bool, device=target)

    @property
    def batch_size(self) -> int:
        return self._batch

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def episode_ids(self) -> torch.Tensor:
        """各环境当前 episode 身份（返回 clone）。"""
        return self._episode_ids.clone()

    def reset_envs(self, *, env_ids: object, new_episode_ids: object) -> None:
        """部分 reset：仅清目标环境的扫描状态并切换 episode 身份。

        其余环境不受影响；新 episode_id 不得为该环境历史上使用过的任何 id
        （含当前 id），episode 身份不得跨 reset 复用/回退；整批原子：任一行
        违规则全部不生效。
        """
        envs = _as_int_vector(env_ids, name="env_ids", device=self._device)
        fresh = _as_int_vector(new_episode_ids, name="new_episode_ids", device=self._device)
        count = int(envs.shape[0])
        if count == 0:
            raise ContractError("reset_envs 需要至少一个环境")
        if int(fresh.shape[0]) != count:
            raise ContractError("env_ids 与 new_episode_ids 长度不一致")
        if envs.numel() != torch.unique(envs).numel():
            raise ContractError("env_ids 不得重复")
        if bool((envs < 0).any()) or bool((envs >= self._batch).any()):
            raise ContractError(f"env_ids 超出 [0, {self._batch})")
        if bool((fresh < 0).any()):
            raise ContractError("new_episode_ids 必须非负")
        for i in range(count):
            env = int(envs[i])
            new_id = int(fresh[i])
            if new_id in self._seen_episodes[env]:
                raise ContractError(
                    f"env {env}: episode_id {new_id} 已在该环境使用过（当前为 "
                    f"{int(self._episode_ids[env])}）；episode 身份不得跨 reset 复用/回退"
                )
        for i in range(count):
            env = int(envs[i])
            new_id = int(fresh[i])
            self._seen_episodes[env].add(new_id)
            self._episode_ids[env] = new_id
            self._near_points[env].zero_()
            self._near_valid[env].fill_(False)
            self._near_filled[env] = False
            self._near_scan_ids[env] = EMPTY_SCAN_ID
            self._near_times[env] = 0.0
            self._far_points[env].zero_()
            self._far_valid[env].fill_(False)
            self._far_scan_ids[env].fill_(EMPTY_SCAN_ID)
            self._far_times[env].zero_()
            self._far_count[env] = 0
            self._far_pos[env] = 0
            self._last_scan_id[env] = EMPTY_SCAN_ID
            self._last_time[env] = float("-inf")
            # P6 角键元数据随该行扫描状态一并清除；其余环境不受影响。
            self._near_angle_keys[env].zero_()
            self._far_angle_keys[env].zero_()
            self._angle_keys_present[env] = False
            self._near_source_dirs_sensor[env].zero_()
            self._far_source_dirs_sensor[env].zero_()
            self._near_source_dirs_present[env].zero_()
            self._far_source_dirs_present[env].zero_()
            self._near_source_indices[env].fill_(-1)
            self._far_source_indices[env].fill_(-1)
            self._source_dirs_attached[env] = False

    def write(
        self,
        *,
        env_ids: object,
        episode_ids: object,
        scan_ids: object,
        times: object,
        near_points: torch.Tensor,
        near_valid: torch.Tensor,
        far_points: torch.Tensor,
        far_valid: torch.Tensor,
        near_angle_keys: torch.Tensor | None = None,
        far_angle_keys: torch.Tensor | None = None,
        near_source_dirs_sensor: torch.Tensor | None = None,
        far_source_dirs_sensor: torch.Tensor | None = None,
        near_source_indices: torch.Tensor | None = None,
        far_source_indices: torch.Tensor | None = None,
    ) -> None:
        """写入一批新完成的扫描（每环境至多一条；近端槽与远端环同步更新）。

        输入：near (K,256,3)/(K,256)、far (K,128,3)/(K,128)，float32/bool，
        有效点有限、无效槽精确 0（拒绝策略，见 contracts.validate_point_block）。
        可选角键：``near_angle_keys`` (K,256)/``far_angle_keys`` (K,128)
        float32，槽序与同批点块对齐，二者必须同时提供或同时缺省；提供时须
        与窗口设备一致、在有效点槽处有限（与 sort 契约一致），写入后该行
        ``present=True``；缺省写入会把该行 ``present`` 置回 False（绝不残留
        陈旧键匹配新点）。角键是排序/身份元数据，不是点云通道。
        整批原子：任一行身份/顺序/点契约违规则全部不写入。
        """
        envs = _as_int_vector(env_ids, name="env_ids", device=self._device)
        episodes = _as_int_vector(episode_ids, name="episode_ids", device=self._device)
        scans = _as_int_vector(scan_ids, name="scan_ids", device=self._device)
        stamps = _as_float_vector(times, name="times", device=self._device)
        count = int(envs.shape[0])
        if count == 0:
            raise ContractError("write 需要至少一个环境")
        for name, vec in (("episode_ids", episodes), ("scan_ids", scans), ("times", stamps)):
            if int(vec.shape[0]) != count:
                raise ContractError(f"{name} 长度与 env_ids 不一致（{int(vec.shape[0])} vs {count}）")
        if envs.numel() != torch.unique(envs).numel():
            raise ContractError("同一批 write 中 env_ids 不得重复")
        if bool((envs < 0).any()) or bool((envs >= self._batch).any()):
            raise ContractError(f"env_ids 超出 [0, {self._batch})")
        if bool((scans < 0).any()):
            raise ContractError("scan_ids 必须非负")
        if not bool(torch.isfinite(stamps).all()):
            raise ContractError("times 含 NaN/Inf")
        validate_point_block(near_points, near_valid, n_points=NEAR_POINTS, name="near_points")
        validate_point_block(far_points, far_valid, n_points=FAR_POINTS_PER_SCAN, name="far_points")
        if int(near_points.shape[0]) != count or int(far_points.shape[0]) != count:
            raise ContractError("点块批量与 env_ids 数不一致")
        if near_points.device != self._device or far_points.device != self._device:
            raise ContractError(f"写入张量必须与窗口设备一致（{self._device}）")
        if (near_angle_keys is None) != (far_angle_keys is None):
            raise ContractError(
                "near/far 角键必须同时提供或同时缺省（一次扫描同源产生）"
            )
        source_args = (
            near_source_dirs_sensor, far_source_dirs_sensor,
            near_source_indices, far_source_indices,
        )
        if any(value is None for value in source_args) and any(
            value is not None for value in source_args
        ):
            raise ContractError("近/远源方向与原始射线索引必须同时提供或同时缺省")
        if near_angle_keys is not None and far_angle_keys is not None:
            if (
                not isinstance(near_angle_keys, torch.Tensor)
                or near_angle_keys.dtype is not FLOAT_DTYPE
                or tuple(near_angle_keys.shape) != (count, NEAR_POINTS)
                or near_angle_keys.device != self._device
            ):
                raise ContractError(
                    f"near_angle_keys 须为 ({count}, {NEAR_POINTS}) float32 且与窗口同设备"
                )
            if (
                not isinstance(far_angle_keys, torch.Tensor)
                or far_angle_keys.dtype is not FLOAT_DTYPE
                or tuple(far_angle_keys.shape) != (count, FAR_POINTS_PER_SCAN)
                or far_angle_keys.device != self._device
            ):
                raise ContractError(
                    f"far_angle_keys 须为 ({count}, {FAR_POINTS_PER_SCAN}) float32 且与窗口同设备"
                )
            if bool(near_valid.any()) and not bool(
                torch.isfinite(near_angle_keys[near_valid]).all()
            ):
                raise ContractError("near_angle_keys 在有效点槽处必须有限")
            if bool(far_valid.any()) and not bool(
                torch.isfinite(far_angle_keys[far_valid]).all()
            ):
                raise ContractError("far_angle_keys 在有效点槽处必须有限")
        if near_source_dirs_sensor is not None:
            expected_sources = (
                ("near_source_dirs_sensor", near_source_dirs_sensor, (count, NEAR_POINTS, 3), FLOAT_DTYPE),
                ("far_source_dirs_sensor", far_source_dirs_sensor, (count, FAR_POINTS_PER_SCAN, 3), FLOAT_DTYPE),
                ("near_source_indices", near_source_indices, (count, NEAR_POINTS), INT_DTYPE),
                ("far_source_indices", far_source_indices, (count, FAR_POINTS_PER_SCAN), INT_DTYPE),
            )
            for name, value, shape, dtype in expected_sources:
                if not isinstance(value, torch.Tensor) or value.dtype is not dtype \
                        or tuple(value.shape) != shape or value.device != self._device:
                    raise ContractError(
                        f"{name} 须为 {shape} {dtype} 且与窗口同设备"
                    )
            for name, directions, indices, valid in (
                ("near", near_source_dirs_sensor, near_source_indices, near_valid),
                ("far", far_source_dirs_sensor, far_source_indices, far_valid),
            ):
                present = indices >= 0
                if not torch.equal(present, valid):
                    raise ContractError(f"{name} 原始射线索引必须在有效槽非负、无效槽为 -1")
                if bool(present.any()):
                    selected = directions[present]
                    if not bool(torch.isfinite(selected).all()):
                        raise ContractError(f"{name} 源方向含 NaN/Inf")
                    norms = torch.linalg.vector_norm(selected, dim=-1)
                    if not bool(torch.allclose(norms, torch.ones_like(norms), atol=1e-5, rtol=0.0)):
                        raise ContractError(f"{name} 源方向须为单位向量")
                if bool((~present).any()) and not torch.equal(
                    directions[~present], torch.zeros_like(directions[~present])
                ):
                    raise ContractError(f"{name} 无源方向槽必须精确填 0")
        for i in range(count):
            env = int(envs[i])
            if int(episodes[i]) != int(self._episode_ids[env]):
                raise EpisodeMismatchError(
                    f"env {env}: 写入 episode {int(episodes[i])} 与窗口当前 "
                    f"{int(self._episode_ids[env])} 不一致（陈旧/未声明 episode 拒绝）"
                )
            last_scan = int(self._last_scan_id[env])
            if int(scans[i]) <= last_scan:
                raise StaleScanError(
                    f"env {env}: scan_id {int(scans[i])} 未大于最新 {last_scan}（重复/旧扫描拒绝）"
                )
            last_time = float(self._last_time[env])
            if float(stamps[i]) <= last_time:
                raise TimeOrderError(
                    f"env {env}: time {float(stamps[i])} 未晚于最新 {last_time}（episode 内须严格递增）"
                )
        for i in range(count):
            env = int(envs[i])
            slot = int(self._far_pos[env])
            self._far_points[env, slot] = far_points[i]
            self._far_valid[env, slot] = far_valid[i]
            self._far_scan_ids[env, slot] = int(scans[i])
            self._far_times[env, slot] = float(stamps[i])
            self._far_count[env] = min(int(self._far_count[env]) + 1, FAR_WINDOW_SLOTS)
            self._far_pos[env] = (slot + 1) % FAR_WINDOW_SLOTS
            self._near_points[env] = near_points[i]
            self._near_valid[env] = near_valid[i]
            self._near_filled[env] = True
            self._near_scan_ids[env] = int(scans[i])
            self._near_times[env] = float(stamps[i])
            self._last_scan_id[env] = int(scans[i])
            self._last_time[env] = float(stamps[i])
            if near_angle_keys is not None and far_angle_keys is not None:
                self._near_angle_keys[env] = torch.where(near_valid[i], near_angle_keys[i], 0.0)
                self._far_angle_keys[env, slot] = torch.where(far_valid[i], far_angle_keys[i], 0.0)
                self._angle_keys_present[env] = True
            else:
                self._near_angle_keys[env].zero_()
                self._far_angle_keys[env, slot].zero_()
                self._angle_keys_present[env] = False
            if near_source_dirs_sensor is not None:
                self._near_source_dirs_sensor[env] = near_source_dirs_sensor[i]
                self._far_source_dirs_sensor[env, slot] = far_source_dirs_sensor[i]
                self._near_source_dirs_present[env] = near_valid[i]
                self._far_source_dirs_present[env, slot] = far_valid[i]
                self._near_source_indices[env] = near_source_indices[i]
                self._far_source_indices[env, slot] = far_source_indices[i]
                self._source_dirs_attached[env] = True
            else:
                self._near_source_dirs_sensor[env].zero_()
                self._far_source_dirs_sensor[env, slot].zero_()
                self._near_source_dirs_present[env].zero_()
                self._far_source_dirs_present[env, slot].zero_()
                self._near_source_indices[env].fill_(-1)
                self._far_source_indices[env, slot].fill_(-1)

    def read(self) -> ScanWindowSnapshot:
        """整批只读快照：不推进窗口；返回全新 clone，后续写入/reset 不影响已返回快照。

        far 槽按当前扫描排列；未写满时仅前 count 个槽有效，空槽全 0 且
        far_filled=False，不复制首扫描补足。
        """
        batch = self._batch
        device = self._device
        slots = torch.arange(FAR_WINDOW_SLOTS, device=device)
        sequential = slots[None, :].expand(batch, -1)
        rotated = (self._far_pos[:, None] + sequential) % FAR_WINDOW_SLOTS
        full = self._far_count >= FAR_WINDOW_SLOTS
        source = torch.where(full[:, None], rotated, sequential)
        env_index = torch.arange(batch, device=device)[:, None]
        return ScanWindowSnapshot(
            env_ids=torch.arange(batch, dtype=INT_DTYPE, device=device),
            episode_ids=self._episode_ids.clone(),
            near_points=self._near_points.clone(),
            near_valid=self._near_valid.clone(),
            near_filled=self._near_filled.clone(),
            near_scan_ids=self._near_scan_ids.clone(),
            near_times=self._near_times.clone(),
            far_points=self._far_points[env_index, source].clone(),
            far_valid=self._far_valid[env_index, source].clone(),
            far_filled=(slots[None, :] < self._far_count[:, None]).clone(),
            far_scan_ids=self._far_scan_ids[env_index, source].clone(),
            far_times=self._far_times[env_index, source].clone(),
        )

    def read_angle_keys(self) -> WindowAngleKeys | None:
        """可选角键元数据快照；从未携带键写入（clean 路径）时返回 None。

        远端键槽采用与 ``read()`` 完全相同的旋转/重排，因此
        ``far.reshape(batch, 128)`` 与 ``read().far_points.reshape(batch, 128)``
        的行/点顺序一一对应；``present=False`` 行（部分 reset 后未再扫描，
        或写入方从不携带键）要求消费者走原几何角键路径；该行可以包含
        有效旧接口点，不能直接消费哑键。全新 clone，后续写入/reset 不影响已返回数据。
        """
        if not bool(self._angle_keys_present.any()):
            return None
        batch = self._batch
        device = self._device
        slots = torch.arange(FAR_WINDOW_SLOTS, device=device)
        sequential = slots[None, :].expand(batch, -1)
        rotated = (self._far_pos[:, None] + sequential) % FAR_WINDOW_SLOTS
        full = self._far_count >= FAR_WINDOW_SLOTS
        source = torch.where(full[:, None], rotated, sequential)
        env_index = torch.arange(batch, device=device)[:, None]
        return WindowAngleKeys(
            near=self._near_angle_keys.clone(),
            far=self._far_angle_keys[env_index, source].clone(),
            present=self._angle_keys_present.clone(),
        )

    def read_source_directions(self) -> WindowSourceDirections | None:
        """Read ray provenance in the exact point/slot order of ``read()``."""
        if not bool(self._source_dirs_attached.any()):
            return None
        batch = self._batch
        device = self._device
        slots = torch.arange(FAR_WINDOW_SLOTS, device=device)
        sequential = slots[None, :].expand(batch, -1)
        rotated = (self._far_pos[:, None] + sequential) % FAR_WINDOW_SLOTS
        full = self._far_count >= FAR_WINDOW_SLOTS
        source = torch.where(full[:, None], rotated, sequential)
        env_index = torch.arange(batch, device=device)[:, None]
        return WindowSourceDirections(
            near=self._near_source_dirs_sensor.clone(),
            far=self._far_source_dirs_sensor[env_index, source].clone(),
            near_present=self._near_source_dirs_present.clone(),
            far_present=self._far_source_dirs_present[env_index, source].clone(),
            near_indices=self._near_source_indices.clone(),
            far_indices=self._far_source_indices[env_index, source].clone(),
        )

    def state_dict(self) -> dict[str, object]:
        """Return an isolated checkpoint of scan state and episode identities."""
        tensors = {
            name: value.clone()
            for name, value in vars(self).items()
            if name.startswith("_") and isinstance(value, torch.Tensor)
        }
        return {
            "batch_size": self._batch,
            "tensors": tensors,
            "seen_episodes": [sorted(ids) for ids in self._seen_episodes],
        }

    def load_state_dict(self, state: object) -> None:
        """Restore a compatible state atomically after validating all fields."""
        if not isinstance(state, dict) or state.get("batch_size") != self._batch:
            raise ContractError("ScanWindow state batch_size does not match this window")
        raw_tensors = state.get("tensors")
        raw_seen = state.get("seen_episodes")
        if not isinstance(raw_tensors, dict) or not isinstance(raw_seen, list) \
                or len(raw_seen) != self._batch:
            raise ContractError("ScanWindow state is missing tensors or per-environment episode history")
        current = {
            name: value
            for name, value in vars(self).items()
            if name.startswith("_") and isinstance(value, torch.Tensor)
        }
        if set(raw_tensors) != set(current):
            raise ContractError("ScanWindow state tensor fields do not match this implementation")
        staged: dict[str, torch.Tensor] = {}
        for name, template in current.items():
            value = raw_tensors[name]
            if not isinstance(value, torch.Tensor) or value.dtype is not template.dtype \
                    or tuple(value.shape) != tuple(template.shape):
                raise ContractError(f"ScanWindow state field {name} has incompatible shape or dtype")
            staged[name] = value.to(device=self._device).clone()

        seen: list[set[int]] = []
        for env, values in enumerate(raw_seen):
            try:
                ids = {int(value) for value in values}
            except (TypeError, ValueError, OverflowError) as exc:
                raise ContractError(f"seen_episodes[{env}] must be an integer sequence") from exc
            if not ids or min(ids) < 0 or int(staged["_episode_ids"][env]) not in ids:
                raise ContractError(f"seen_episodes[{env}] must include its nonnegative current episode")
            seen.append(ids)

        candidate = ScanWindowSnapshot(
            env_ids=torch.arange(self._batch, dtype=INT_DTYPE, device=self._device),
            episode_ids=staged["_episode_ids"],
            near_points=staged["_near_points"],
            near_valid=staged["_near_valid"],
            near_filled=staged["_near_filled"],
            near_scan_ids=staged["_near_scan_ids"],
            near_times=staged["_near_times"],
            far_points=staged["_far_points"],
            far_valid=staged["_far_valid"],
            far_filled=(staged["_far_count"][:, None] > 0),
            far_scan_ids=staged["_far_scan_ids"],
            far_times=staged["_far_times"],
        )
        candidate.validate()
        if not torch.equal(staged["_last_scan_id"], staged["_near_scan_ids"]):
            raise ContractError("ScanWindow state latest scan ids disagree with the near scan")
        if bool((staged["_far_count"] < 0).any()) or bool((staged["_far_count"] > FAR_WINDOW_SLOTS).any()):
            raise ContractError("ScanWindow state far_count is outside the one-slot window")
        expected_time = torch.where(
            staged["_near_filled"], staged["_near_times"],
            torch.full_like(staged["_last_time"], float("-inf")),
        )
        if not torch.equal(staged["_last_time"], expected_time):
            raise ContractError("ScanWindow state latest scan times disagree with the near scan")
        if bool((staged["_far_pos"] != 0).any()):
            raise ContractError("ScanWindow state far_pos must be zero for one slot")

        for name, value in staged.items():
            current[name].copy_(value)
        self._seen_episodes = seen
