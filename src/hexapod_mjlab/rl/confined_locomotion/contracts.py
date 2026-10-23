"""F1 数据契约：confined_locomotion 感知数据的命名字段、形状/单位与身份关联检查。

M2 起本文件为唯一正典实现（由 ``rl/joint_passage/contracts.py`` 兼容垫片别名），
内容与已验收的 joint_passage F1 契约逐字一致：纯计算契约，不注册任务、不导入
mjlab 或任何任务注册器，只依赖标准库与 PyTorch。上位文档：
docs/tasks/confined_locomotion/framework/实验记录.md、docs/research/训练架构对齐.md。

坐标与单位约定（本模块只定义契约，不决定物理传感）：
* 所有点为“采集时机体坐标系”的米制 xyz，恒 3 通道；有效性用独立布尔掩码表达，
  不增加第 4 个输入通道。
* 非法输入一律拒绝（拒绝策略，非本模块代为掩码）：掩码 true 的点必须有限；
  掩码 false 的点和空槽必须精确填 0。调用方负责预先掩码并填 0，本模块不做
  nan_to_num 之类静默修复，因此 NaN/Inf/缺帧/填充无法伪装成有效点。
  正常无命中的最大量程端点是普通有效有限点，无特殊处理。
* 物理量程、近/远划分角、传感器坐标外参、噪声与运动补偿由上游决定，本模块不定义、
  不施加；角度排序只接受调用方显式提供的每点角键，不隐含传感器姿态，
  也不以时间排序替代角度排序。
* 身份 (env_id, episode_id, scan_id/step_id) 随数据同行保存；标签类型与在线观测类型
  分离，标签不进入在线观测字段。控制步身份与扫描身份两套绑定分别校验，防止未来标签
  或旧 episode 数据混入。
* 环境身份约定：env_ids 为任意非负 int64 的真实环境身份，支持重排、子集与重复的
  逻辑采样行（不冻结 PPO 批布局），配对一律按 env/episode/step 或 scan 身份逐行
  执行，不做重编号；ScanWindowSnapshot 每环境至多一行（env_ids 唯一）。
  ScanWindow 内部仍按物理环境索引 0..B-1 写读，read() 返回环境顺序。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# 已确认的固定尺寸（docs/tasks/confined_locomotion/平地学习与基准测量.md 第2节）。
NEAR_POINTS = 256            # 近端当前扫描点数
FAR_POINTS_PER_SCAN = 64     # 远端每次扫描保留点数
FAR_WINDOW_SLOTS = 10        # 远端独立扫描窗口槽数
JOINT_DIM = 18               # q−q_ref / 关节速度 / 上一步动作的维数
PROPRIO_DIM = 63             # 本体向量总维数 = 3+3+3+18+18+18
COMMAND_DIM = 3              # [vx, vy, ωz]
POINT_CHANNELS = 3           # 米制 xyz；有效性不占通道
ENVELOPE_DIM = 6             # [前半宽, 中半宽, 后半宽, 前缘x, 后缘x, Δz]

# 统一 dtype：数值 float32，身份 int64，掩码 bool。
FLOAT_DTYPE = torch.float32
INT_DTYPE = torch.int64

# 空槽（无扫描）使用的 scan_id 哨兵；有效扫描 id 必须非负。
EMPTY_SCAN_ID = -1

# 本体向量拼接顺序（proprio_vector 按此顺序，不缩放、不含指令）。
PROPRIO_LAYOUT = (
    "lin_vel_b",
    "ang_vel_b",
    "gravity_b",
    "joint_pos_rel_ref",
    "joint_vel",
    "prev_action",
)


class ContractError(ValueError):
    """数据契约违规（形状/dtype/单位语义/零填充/身份顺序等）。"""


class IdentityMismatchError(ContractError):
    """绑定双方身份不一致（环境/episode/控制步/扫描，含未来或陈旧数据）。"""


def _validated_batch(env_ids: object) -> int:
    """校验 env_ids 为 (B,) 非负 int64 并返回 B。

    不要求 arange、不要求唯一：逻辑采样行可重排/可重复（同环境多控制步），
    由 episode/step 或 scan 身份区分并逐行配对；快照的每环境唯一性由各自
    validate 另行强制。
    """
    if not isinstance(env_ids, torch.Tensor) or env_ids.dim() != 1 or env_ids.dtype is not INT_DTYPE:
        raise ContractError("env_ids 必须是 (B,) int64 张量")
    batch = int(env_ids.shape[0])
    if batch < 1:
        raise ContractError("批量维度必须 >= 1")
    if bool((env_ids < 0).any()):
        raise ContractError("env_ids 必须非负（携带真实环境身份，不做重编号）")
    return batch


def _require_field(
    name: str,
    value: object,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    ref_device: torch.device,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise ContractError(f"{name} 必须是 torch.Tensor，实际 {type(value).__name__}")
    if value.dtype is not dtype:
        raise ContractError(f"{name} dtype 应为 {dtype}，实际 {value.dtype}")
    if tuple(value.shape) != shape:
        raise ContractError(f"{name} 形状应为 {shape}，实际 {tuple(value.shape)}")
    if value.device != ref_device:
        raise ContractError(f"{name} 设备 {value.device} 与批内基准 {ref_device} 不一致")


def _check_masked_finite_zero(values: torch.Tensor, valid: torch.Tensor, *, name: str) -> None:
    """values 与 valid 同形状：有效元素有限，无效元素精确 0（不得夹带异常值）。"""
    if bool(valid.any()):
        if not bool(torch.isfinite(values[valid]).all()):
            raise ContractError(f"{name}: 有效元素含 NaN/Inf")
    invalid = ~valid
    if bool(invalid.any()):
        bad = values[invalid]
        if not torch.equal(bad, torch.zeros_like(bad)):
            raise ContractError(f"{name}: 无效元素必须精确填 0")


def validate_point_block(points: object, valid: object, *, n_points: int, name: str) -> None:
    """点块契约：(K, n_points, 3) float32 + (K, n_points) bool，同批量同设备。

    掩码 true 的点必须有限（米制机体坐标，可含正常无命中的最大量程端点）；
    掩码 false 的槽必须精确填 0。
    """
    if not isinstance(points, torch.Tensor) or points.dim() != 3:
        raise ContractError(f"{name} 必须是 (K, {n_points}, 3) 张量")
    k = int(points.shape[0])
    if k < 1 or int(points.shape[1]) != n_points or int(points.shape[2]) != POINT_CHANNELS:
        raise ContractError(f"{name} 形状应为 (K, {n_points}, {POINT_CHANNELS})，实际 {tuple(points.shape)}")
    if points.dtype is not FLOAT_DTYPE:
        raise ContractError(f"{name} dtype 应为 {FLOAT_DTYPE}，实际 {points.dtype}")
    if not isinstance(valid, torch.Tensor) or valid.dtype is not torch.bool or tuple(valid.shape) != (k, n_points):
        raise ContractError(f"{name} 对应掩码应为 ({k}, {n_points}) bool 张量")
    if valid.device != points.device:
        raise ContractError(f"{name} 与其掩码设备不一致: {points.device} vs {valid.device}")
    _check_masked_finite_zero(points, valid, name=name)


@dataclass
class ScanWindowSnapshot:
    """扫描窗口只读快照（近端最新完成扫描 + 远端 10 槽独立扫描）。

    坐标：采集时机体坐标系、单位米；构造即校验（__post_init__ 调 validate）。
    * env_ids 为非负真实环境身份且每环境至多一行（唯一）；行序不要求环境顺序，
      ScanWindow.read() 返回的快照按环境顺序排列。
    * near_*：近端当前扫描。near_filled=False 表示尚无扫描（可表示空状态，不捏造数据）：
      此时 points 全 0、valid 全 False、scan_ids=EMPTY_SCAN_ID、times=0。
    * far_*：dim1 按采集先后排列（最旧→最新）；空槽 points 全 0、valid 全 False、
      scan_ids=EMPTY_SCAN_ID、times=0，绝不复制首扫描或重复点补足。
    * 已填充槽内 scan_ids/times 严格递增（写入序=采集序）。
    * 近/远关联：near_filled 与“远端是否存在已填充槽”一致；已填充行的近端槽
      scan_id/time 必须与远端最新已填充槽完全一致（两者由同一次写入同步更新）。
    """

    env_ids: torch.Tensor        # (B,) int64 非负且唯一（每环境至多一行）
    episode_ids: torch.Tensor    # (B,) int64，非负
    near_points: torch.Tensor    # (B, 256, 3) float32 米
    near_valid: torch.Tensor     # (B, 256) bool
    near_filled: torch.Tensor    # (B,) bool
    near_scan_ids: torch.Tensor  # (B,) int64
    near_times: torch.Tensor     # (B,) float32 秒（调用方时钟，episode 内递增）
    far_points: torch.Tensor     # (B, 10, 64, 3) float32 米
    far_valid: torch.Tensor      # (B, 10, 64) bool
    far_filled: torch.Tensor     # (B, 10) bool，每行前缀
    far_scan_ids: torch.Tensor   # (B, 10) int64
    far_times: torch.Tensor      # (B, 10) float32 秒

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        batch = _validated_batch(self.env_ids)
        device = self.env_ids.device
        if self.env_ids.numel() != torch.unique(self.env_ids).numel():
            raise ContractError("快照 env_ids 必须唯一（每环境至多一行，供按身份配对）")
        _require_field("episode_ids", self.episode_ids, INT_DTYPE, (batch,), device)
        _require_field("near_points", self.near_points, FLOAT_DTYPE,
                       (batch, NEAR_POINTS, POINT_CHANNELS), device)
        _require_field("near_valid", self.near_valid, torch.bool, (batch, NEAR_POINTS), device)
        _require_field("near_filled", self.near_filled, torch.bool, (batch,), device)
        _require_field("near_scan_ids", self.near_scan_ids, INT_DTYPE, (batch,), device)
        _require_field("near_times", self.near_times, FLOAT_DTYPE, (batch,), device)
        _require_field("far_points", self.far_points, FLOAT_DTYPE,
                       (batch, FAR_WINDOW_SLOTS, FAR_POINTS_PER_SCAN, POINT_CHANNELS), device)
        _require_field("far_valid", self.far_valid, torch.bool,
                       (batch, FAR_WINDOW_SLOTS, FAR_POINTS_PER_SCAN), device)
        _require_field("far_filled", self.far_filled, torch.bool, (batch, FAR_WINDOW_SLOTS), device)
        _require_field("far_scan_ids", self.far_scan_ids, INT_DTYPE, (batch, FAR_WINDOW_SLOTS), device)
        _require_field("far_times", self.far_times, FLOAT_DTYPE, (batch, FAR_WINDOW_SLOTS), device)
        if bool((self.episode_ids < 0).any()):
            raise ContractError("episode_ids 必须非负")
        if not bool(torch.isfinite(self.near_times).all()) or not bool(torch.isfinite(self.far_times).all()):
            raise ContractError("时间字段含 NaN/Inf")
        validate_point_block(self.near_points, self.near_valid, n_points=NEAR_POINTS, name="near_points")
        validate_point_block(
            self.far_points.reshape(batch * FAR_WINDOW_SLOTS, FAR_POINTS_PER_SCAN, POINT_CHANNELS),
            self.far_valid.reshape(batch * FAR_WINDOW_SLOTS, FAR_POINTS_PER_SCAN),
            n_points=FAR_POINTS_PER_SCAN,
            name="far_points",
        )
        empty_near = ~self.near_filled
        if bool(self.near_filled.any()) and bool((self.near_scan_ids[self.near_filled] < 0).any()):
            raise ContractError("已填充近端槽 scan_ids 必须非负")
        if bool(empty_near.any()):
            if bool((self.near_scan_ids[empty_near] != EMPTY_SCAN_ID).any()):
                raise ContractError("未填充近端槽 scan_ids 必须为 EMPTY_SCAN_ID")
            if bool(self.near_valid[empty_near].any()):
                raise ContractError("未填充近端槽掩码必须全 False")
            if not torch.equal(self.near_times[empty_near], torch.zeros_like(self.near_times[empty_near])):
                raise ContractError("未填充近端槽 times 必须为 0")
        slots = torch.arange(FAR_WINDOW_SLOTS, device=device)
        counts = self.far_filled.sum(dim=1, keepdim=True)
        if not torch.equal(self.far_filled, slots[None, :] < counts):
            raise ContractError("far_filled 必须是每行前缀（最旧→最新连续填充，无空洞）")
        filled, unfilled = self.far_filled, ~self.far_filled
        if bool(filled.any()) and bool((self.far_scan_ids[filled] < 0).any()):
            raise ContractError("已填充远端槽 scan_ids 必须非负")
        if bool((self.far_scan_ids[unfilled] != EMPTY_SCAN_ID).any()):
            raise ContractError("空远端槽 scan_ids 必须为 EMPTY_SCAN_ID")
        if bool(self.far_valid[unfilled].any()):
            raise ContractError("空远端槽掩码必须全 False")
        if bool(unfilled.any()) and not torch.equal(
            self.far_times[unfilled], torch.zeros_like(self.far_times[unfilled])
        ):
            raise ContractError("空远端槽 times 必须为 0")
        pair = filled[:, 1:]  # 前缀填充下，位置 j 填充 ⇒ j-1 也已填充
        if bool(pair.any()):
            if not bool((self.far_scan_ids[:, 1:][pair] > self.far_scan_ids[:, :-1][pair]).all()):
                raise ContractError("远端已填充槽 scan_ids 必须严格递增")
            if not bool((self.far_times[:, 1:][pair] > self.far_times[:, :-1][pair]).all()):
                raise ContractError("远端已填充槽 times 必须严格递增")
        # 近/远关联：近端槽与远端最新已填充槽来自同一次写入，身份与时间须完全一致，
        # 防止手工快照把远端较新扫描与近端旧扫描拼接后通过后续绑定。
        if not torch.equal(self.near_filled, self.far_filled[:, 0]):
            raise ContractError("near_filled 必须与远端是否存在已填充槽一致（近远端同步写入）")
        newest_index = (self.far_filled.sum(dim=1) - 1).clamp(min=0)
        row_index = torch.arange(batch, device=device)
        newest_ids = self.far_scan_ids[row_index, newest_index]
        newest_times = self.far_times[row_index, newest_index]
        if bool(self.near_filled.any()):
            selected = self.near_filled
            if not bool((newest_ids[selected] == self.near_scan_ids[selected]).all()):
                raise ContractError("近端槽 scan_id 必须与远端最新已填充槽一致")
            if not bool(torch.equal(newest_times[selected], self.near_times[selected])):
                raise ContractError("近端槽时间必须与远端最新已填充槽一致")


@dataclass
class OnlineObservation:
    """在线观测（actor/critic 输入；不含任何标签或场景真值字段）。

    本体字段与单位（机体坐标系）：
      lin_vel_b (B,3) m/s；ang_vel_b (B,3) rad/s；gravity_b (B,3) 重力方向单位向量（无量纲）；
      joint_pos_rel_ref (B,18) rad（q−q_ref，固定关节参考与动作执行共用）；
      joint_vel (B,18) rad/s；prev_action (B,18) 无量纲上一步动作字段——取裁剪前
      还是裁剪后的动作尚未确认，本契约不默认限定来源或数值范围（含不默认 ±100），
      闭环接入前确认。
    指令另列：command (B,3) = [vx, vy, ωz]，单位 m/s、m/s、rad/s；不进入形态头，
    也不并入本体向量。指令的机体/航向坐标系细节在闭环接入前确认。
    env_ids 为非负真实环境身份：逻辑采样行可重排、可子集、可同环境多控制步重复，
    行身份由 (env, episode, step) 区分，不做重编号。
    time_s 与扫描时间同一调用方时钟（秒），仅用于一致性检查，不定义绝对零点；
    观测倍率（×2、×0.25 等）属 F2 网络输入侧，本类不施加。
    所有数值字段必须完整有限：本体观测不定义部分有效语义。
    """

    env_ids: torch.Tensor            # (B,) int64 非负，真实环境身份（可重排/重复）
    episode_ids: torch.Tensor        # (B,) int64，非负
    step_ids: torch.Tensor           # (B,) int64，episode 内控制步计数，非负
    time_s: torch.Tensor             # (B,) float32 秒
    lin_vel_b: torch.Tensor          # (B,3) float32 m/s
    ang_vel_b: torch.Tensor          # (B,3) float32 rad/s
    gravity_b: torch.Tensor          # (B,3) float32 无量纲
    joint_pos_rel_ref: torch.Tensor  # (B,18) float32 rad
    joint_vel: torch.Tensor          # (B,18) float32 rad/s
    prev_action: torch.Tensor        # (B,18) float32 无量纲动作（来源/范围闭环前确认）
    command: torch.Tensor            # (B,3) float32 [vx, vy, ωz]

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        batch = _validated_batch(self.env_ids)
        device = self.env_ids.device
        _require_field("episode_ids", self.episode_ids, INT_DTYPE, (batch,), device)
        _require_field("step_ids", self.step_ids, INT_DTYPE, (batch,), device)
        _require_field("time_s", self.time_s, FLOAT_DTYPE, (batch,), device)
        for name in ("lin_vel_b", "ang_vel_b", "gravity_b"):
            _require_field(name, getattr(self, name), FLOAT_DTYPE, (batch, 3), device)
        for name in ("joint_pos_rel_ref", "joint_vel", "prev_action"):
            _require_field(name, getattr(self, name), FLOAT_DTYPE, (batch, JOINT_DIM), device)
        _require_field("command", self.command, FLOAT_DTYPE, (batch, COMMAND_DIM), device)
        if bool((self.episode_ids < 0).any()) or bool((self.step_ids < 0).any()):
            raise ContractError("episode_ids/step_ids 必须非负")
        for name in ("time_s", *PROPRIO_LAYOUT, "command"):
            if not bool(torch.isfinite(getattr(self, name)).all()):
                raise ContractError(f"{name} 含 NaN/Inf（在线观测必须完整有限）")

    def proprio_vector(self) -> torch.Tensor:
        """按 PROPRIO_LAYOUT 顺序拼接为 (B, 63)；不含指令、不施加观测倍率。"""
        self.validate()
        return torch.cat([getattr(self, name) for name in PROPRIO_LAYOUT], dim=1)


@dataclass
class EnvelopeLabel:
    """包络辅助标签 E*：对应当前控制步、动作发出前的状态。

    values (B,6) 米，顺序 [前半宽, 中半宽, 后半宽, 前缘x, 后缘x, Δz]；
    后缘 x 按约定存负值（更负=向后展开）、半宽为正。本契约不强行校验几何符号
    （几何合法性属 teacher/评测侧），只要求：有效行有限、无效行精确填 0 并由
    valid 排除（求解失败/超时/无参考的隔离方式；失败原因分项记录待定，不在本类内）。
    身份为控制步身份 (env_ids, episode_ids, step_ids)，经 bind_envelope_label
    与同控制步的 OnlineObservation 逐行绑定。标签不进入在线输入。
    """

    env_ids: torch.Tensor      # (B,) int64 非负，可重排/重复（与观测行序对齐）
    episode_ids: torch.Tensor  # (B,) int64
    step_ids: torch.Tensor     # (B,) int64
    values: torch.Tensor       # (B, 6) float32 米
    valid: torch.Tensor        # (B,) bool

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        batch = _validated_batch(self.env_ids)
        device = self.env_ids.device
        _require_field("episode_ids", self.episode_ids, INT_DTYPE, (batch,), device)
        _require_field("step_ids", self.step_ids, INT_DTYPE, (batch,), device)
        _require_field("values", self.values, FLOAT_DTYPE, (batch, ENVELOPE_DIM), device)
        _require_field("valid", self.valid, torch.bool, (batch,), device)
        if bool((self.episode_ids < 0).any()) or bool((self.step_ids < 0).any()):
            raise ContractError("episode_ids/step_ids 必须非负")
        _check_masked_finite_zero(
            self.values, self.valid[:, None].expand(batch, ENVELOPE_DIM), name="EnvelopeLabel.values"
        )


@dataclass
class GroundLabel:
    """地面辅助标签：与一次近端扫描配对（采样位置与根高度均取该扫描时状态）。

    2026-10-07 用户确认：heights (B,M) 米为固定机身 xy 位置沿扫描时机身 z
    轴查询地面所得的带符号相对高度 h_B；valid (B,M)。网格跟随完整机身姿态，
    根参考点为原点，positions (B,M,3) 在真实提供器中声明为采集时机身坐标
    [x_B,y_B,h_B]（世界交点另存离线元数据）。M/范围/分辨率由调用方显式配置，
    最终尺寸后置精调；本纯计算契约只校验数值/形状/掩码/身份，不自行查询几何。
    早期 WORLD 竖直差值 z_ground-z_root 的冻结实验保留原身份，不回写。
    身份为扫描身份 (env_ids, episode_ids, scan_ids)：
    * env_ids 为非负真实环境身份，逻辑行可重排/重复；
    * scan_ids = EMPTY_SCAN_ID(-1) 表示该行缺失标签（如部分 reset 后无扫描）：
      valid 必须全 False 且数值/位置零填，由掩码排除，不得伪造有效标签；
    * 含有效标签的行 scan_ids 必须非负，并经 bind_ground_label 与窗口近端槽
      当前扫描逐行匹配（同批 episode 仍严格一致，禁止未来/旧 episode）。
    同一扫描在多个控制步重复使用时的计数/权重待定，不在本类内决定。

    M2 边界：真实地面辅助网格/覆盖尚未确认，ground_samples=None 是**显式未实现
    的真实辅助**（不是删除地面目标）；本契约与绑定 API 原样保留。
    """

    env_ids: torch.Tensor      # (B,) int64 非负，可重排/重复
    episode_ids: torch.Tensor  # (B,) int64
    scan_ids: torch.Tensor     # (B,) int64，非负或 EMPTY_SCAN_ID（缺失行）
    heights: torch.Tensor      # (B, M) float32 米
    positions: torch.Tensor    # (B, M, 3) float32 调用方声明语义，本模块不解释
    valid: torch.Tensor        # (B, M) bool

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        batch = _validated_batch(self.env_ids)
        device = self.env_ids.device
        _require_field("episode_ids", self.episode_ids, INT_DTYPE, (batch,), device)
        _require_field("scan_ids", self.scan_ids, INT_DTYPE, (batch,), device)
        if not isinstance(self.heights, torch.Tensor) or self.heights.dim() != 2 \
                or int(self.heights.shape[0]) != batch or self.heights.dtype is not FLOAT_DTYPE:
            raise ContractError(f"heights 应为 ({batch}, M) float32，实际非契约形状/dtype")
        samples = int(self.heights.shape[1])
        _require_field("heights", self.heights, FLOAT_DTYPE, (batch, samples), device)
        _require_field("positions", self.positions, FLOAT_DTYPE,
                       (batch, samples, POINT_CHANNELS), device)
        _require_field("valid", self.valid, torch.bool, (batch, samples), device)
        if bool((self.episode_ids < 0).any()):
            raise ContractError("episode_ids 必须非负")
        if bool((self.scan_ids < EMPTY_SCAN_ID).any()):
            raise ContractError("scan_ids 只能为非负或 EMPTY_SCAN_ID(-1)")
        has_label = self.valid.any(dim=1)
        if bool((has_label & (self.scan_ids < 0)).any()):
            raise ContractError("含有效标签的行 scan_ids 必须非负（缺失行不得伪造有效标签）")
        _check_masked_finite_zero(self.heights, self.valid, name="GroundLabel.heights")
        _check_masked_finite_zero(
            self.positions, self.valid[:, :, None].expand(batch, samples, POINT_CHANNELS),
            name="GroundLabel.positions",
        )


def _require_same_identity(what: str, left: torch.Tensor, right: torch.Tensor) -> None:
    if left.device != right.device:
        raise IdentityMismatchError(f"{what} 身份设备不一致: {left.device} vs {right.device}")
    if not torch.equal(left, right):
        raise IdentityMismatchError(f"{what} 身份不一致: {left.tolist()} vs {right.tolist()}")


def _lookup_env_rows(source_env_ids: torch.Tensor, wanted_env_ids: torch.Tensor) -> torch.Tensor:
    """按环境身份把 wanted 每行映射到 source 行号（int64 索引）。

    source（窗口快照）每环境唯一（其 validate 保证）；wanted 可重排/重复。
    任一 wanted 身份在 source 中不存在即抛 IdentityMismatchError。
    """
    equal = wanted_env_ids[:, None] == source_env_ids[None, :]
    counts = equal.sum(dim=1)
    missing = counts != 1
    if bool(missing.any()):
        rows = missing.nonzero().flatten().tolist()
        raise IdentityMismatchError(
            f"环境身份未在窗口快照中找到（行 {rows}: {wanted_env_ids[missing].tolist()}）"
        )
    return equal.to(INT_DTYPE).argmax(dim=1)


def bind_envelope_label(observation: OnlineObservation, label: EnvelopeLabel) -> EnvelopeLabel:
    """校验包络标签与在线观测为同一逻辑采样批且逐行控制步身份一致。

    双方行序对齐：每行 (env, episode, step) 全等（支持重排/重复行，只要两批同序）；
    任何行不一致（含未来控制步标签、旧 episode 标签、行序错位）抛
    IdentityMismatchError。
    """
    observation.validate()
    label.validate()
    _require_same_identity("env", observation.env_ids, label.env_ids)
    _require_same_identity("episode", observation.episode_ids, label.episode_ids)
    _require_same_identity("step", observation.step_ids, label.step_ids)
    return label


def bind_ground_label(window: ScanWindowSnapshot, label: GroundLabel) -> GroundLabel:
    """按行把地面标签绑定到窗口近端槽当前扫描。

    标签行经环境身份查找窗口行（可重排/重复/子集，不要求补齐全部环境）；
    所有行 episode 必须与窗口一致。scan_id=EMPTY_SCAN_ID 的缺失行跳过扫描匹配
    （其 valid 已被契约强制全 False）；含有效标签的行必须与该环境近端槽
    scan_id 一致，缺失/错位/旧数据拒绝。
    """
    window.validate()
    label.validate()
    if window.env_ids.device != label.env_ids.device:
        raise ContractError("绑定双方必须位于同一设备")
    index = _lookup_env_rows(window.env_ids, label.env_ids)
    _require_same_identity("episode", window.episode_ids[index], label.episode_ids)
    has_label = label.valid.any(dim=1)
    mismatch = has_label & (
        (~window.near_filled[index]) | (window.near_scan_ids[index] != label.scan_ids)
    )
    if bool(mismatch.any()):
        rows = mismatch.nonzero().flatten().tolist()
        raise IdentityMismatchError(
            f"地面标签扫描身份与窗口近端槽不一致（行 {rows}）: "
            f"标签 scan_ids {label.scan_ids.tolist()} vs 窗口 {window.near_scan_ids.tolist()}"
        )
    return label


def bind_window_to_observation(
    window: ScanWindowSnapshot, observation: OnlineObservation
) -> ScanWindowSnapshot:
    """按行校验在线观测与窗口快照的环境/episode 身份，且无未来扫描。

    观测行经环境身份查找窗口行（可重排/重复/子集）；每行 episode 必须一致；
    该行所有有效扫描（近端与远端全部已填充槽）的采集时间不得晚于该控制步时间
    （同调用方时钟），任一违反即拒绝。快照近/远一致性由其 validate 保证，
    此处对近端与远端分别再查，不依赖单一近端时间。
    """
    window.validate()
    observation.validate()
    if window.env_ids.device != observation.env_ids.device:
        raise ContractError("绑定双方必须位于同一设备")
    index = _lookup_env_rows(window.env_ids, observation.env_ids)
    _require_same_identity("episode", window.episode_ids[index], observation.episode_ids)
    near_future = window.near_filled[index] & (window.near_times[index] > observation.time_s)
    far_future = (
        window.far_filled[index] & (window.far_times[index] > observation.time_s[:, None])
    ).any(dim=1)
    bad = near_future | far_future
    if bool(bad.any()):
        rows = bad.nonzero().flatten().tolist()
        raise IdentityMismatchError(
            f"有效扫描时间晚于控制步时间（未来扫描，行 {rows}）: "
            f"近端 {window.near_times.tolist()} / 远端最新 "
            f"{window.far_times.max(dim=1).values.tolist()} vs 控制步 {observation.time_s.tolist()}"
        )
    return window


@dataclass
class OrderedPoints:
    """排序后的点序列：无效点稳定置尾，仍由掩码排除；各字段同步置换。"""

    points: torch.Tensor                  # (B, N, 3) float32 米
    valid: torch.Tensor                   # (B, N) bool
    scan_ids: torch.Tensor | None = None  # (B, N) int64，调用方提供时同步置换
    times: torch.Tensor | None = None     # (B, N) float32 秒，调用方提供时同步置换


def sort_points_by_angle_keys(
    points: torch.Tensor,
    valid: torch.Tensor,
    angle_keys: torch.Tensor,
    *,
    scan_ids: torch.Tensor | None = None,
    times: torch.Tensor | None = None,
) -> OrderedPoints:
    """按调用方显式提供的每点角键升序稳定排序，points/valid/scan_ids/times 同步置换。

    角度键与传感器安装外参相关且尚未确认，故本函数不默认计算角度（无隐含零轴/
    量程假设），也不以时间序或写入序替代角度序；调用方可自行用任意函数预计算角键。
    契约：点块满足与 validate_point_block 相同的规则（有效点有限、无效槽精确 0）；
    角键仅在 valid=True 处要求有限，无效槽角键任意（含 NaN）。排序后无效点稳定
    置于序列尾部（保持原相对顺序），GRU 沿排序后的点序列递归、不沿扫描时间递归。
    """
    if not isinstance(points, torch.Tensor) or points.dim() != 3 or points.dtype is not FLOAT_DTYPE \
            or int(points.shape[0]) < 1 or int(points.shape[2]) != POINT_CHANNELS:
        raise ContractError(f"points 应为 (B, N, {POINT_CHANNELS}) float32 张量")
    batch, count = int(points.shape[0]), int(points.shape[1])
    if not isinstance(valid, torch.Tensor) or valid.dtype is not torch.bool \
            or tuple(valid.shape) != (batch, count) or valid.device != points.device:
        raise ContractError(f"valid 应为 ({batch}, {count}) bool 且与 points 同设备")
    if not isinstance(angle_keys, torch.Tensor) or angle_keys.dtype is not FLOAT_DTYPE \
            or tuple(angle_keys.shape) != (batch, count) or angle_keys.device != points.device:
        raise ContractError(f"angle_keys 应为 ({batch}, {count}) float32 且与 points 同设备")
    if scan_ids is not None and (not isinstance(scan_ids, torch.Tensor) or scan_ids.dtype is not INT_DTYPE
                                 or tuple(scan_ids.shape) != (batch, count) or scan_ids.device != points.device):
        raise ContractError(f"scan_ids 应为 ({batch}, {count}) int64 且与 points 同设备")
    if times is not None and (not isinstance(times, torch.Tensor) or times.dtype is not FLOAT_DTYPE
                              or tuple(times.shape) != (batch, count) or times.device != points.device):
        raise ContractError(f"times 应为 ({batch}, {count}) float32 且与 points 同设备")
    _check_masked_finite_zero(points, valid, name="points")
    if bool(valid.any()) and not bool(torch.isfinite(angle_keys[valid]).all()):
        raise ContractError("angle_keys 在有效点处必须有限（无效槽角键任意）")
    effective = torch.where(valid, angle_keys, torch.full_like(angle_keys, float("inf")))
    order = torch.argsort(effective, dim=1, stable=True)
    gathered_index = order.unsqueeze(-1).expand(batch, count, POINT_CHANNELS)
    return OrderedPoints(
        points=torch.gather(points, 1, gathered_index),
        valid=torch.gather(valid, 1, order),
        scan_ids=torch.gather(scan_ids, 1, order) if scan_ids is not None else None,
        times=torch.gather(times, 1, order) if times is not None else None,
    )


def flatten_far_window(window: ScanWindowSnapshot) -> OrderedPoints:
    """远端窗口按最旧→最新拼接为 (B, 10*64, ·) 点序列，附每点 scan_id/time。

    每点元数据按 far_valid 逐点掩码：有效点携带所属槽的 scan_id/time；无效点
    （含已填充槽内的 false 点，正常无命中的最大量程端点是 valid=True 的有效点）
    为 scan_id=EMPTY_SCAN_ID、time=0，坐标恒 0（零填充由快照契约保证）。
    不复制扫描或点补足；坐标点、远端扫描窗口内容与实际已完成扫描身份不变。
    拼接结果可直接交给 sort_points_by_angle_keys 做角度排序（角键由调用方提供）。
    """
    window.validate()
    batch = int(window.far_points.shape[0])
    per_env = FAR_WINDOW_SLOTS * FAR_POINTS_PER_SCAN
    point_valid = window.far_valid.reshape(batch, per_env)
    slot_scan_ids = (
        window.far_scan_ids[:, :, None]
        .expand(batch, FAR_WINDOW_SLOTS, FAR_POINTS_PER_SCAN)
        .reshape(batch, per_env)
    )
    slot_times = (
        window.far_times[:, :, None]
        .expand(batch, FAR_WINDOW_SLOTS, FAR_POINTS_PER_SCAN)
        .reshape(batch, per_env)
    )
    return OrderedPoints(
        points=window.far_points.reshape(batch, per_env, POINT_CHANNELS),
        valid=point_valid,
        scan_ids=torch.where(
            point_valid, slot_scan_ids, torch.full_like(slot_scan_ids, EMPTY_SCAN_ID)
        ),
        times=torch.where(point_valid, slot_times, torch.zeros_like(slot_times)),
    )
