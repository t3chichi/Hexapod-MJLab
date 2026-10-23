"""confined_locomotion 计算主包（M2 起的正典实现；原 joint_passage 位置保留兼容垫片）。

纯计算框架：不注册运行任务、不导入 mjlab 或任何任务注册器，仅依赖标准库与
PyTorch。观测/标签/快照携带任意非负 int64 真实环境身份，逻辑采样行可重排/重复，
配对按 env/episode/step 或 scan 身份逐行执行；ScanWindow 内部仍按物理环境索引
0..B-1 写读。形状/单位/坐标语义与未定项边界见
docs/tasks/confined_locomotion/framework/实验记录.md。F2 网络消费示例：

    window = ScanWindow(batch_size, episode_ids=[0] * batch_size)
    window.write(...)                                    # 仅新完成的扫描
    snapshot = window.read()                             # 50Hz 控制步只读快照
    flat = flatten_far_window(snapshot)                  # (B, 10*64, ·) 最旧→最新
    far = sort_points_by_angle_keys(flat.points, flat.valid, angle_keys,
                                    scan_ids=flat.scan_ids, times=flat.times)
    obs = OnlineObservation(...); obs.proprio_vector()   # (B, 63)，指令另列 command

网络正典类名 ``ConfinedLocomotionNetwork``（``joint_passage.network`` 的
``JointPassageNetwork`` 为同一类对象的兼容别名），按子模块路径导入：
``hexapod_mjlab.rl.confined_locomotion.network``。本包 ``__init__`` 与历史
joint_passage 包一致，只导出 F1 符号；F2/F3/F4 模块直接按子模块路径导入，
不注册任何任务 ID。真实传感/闭环（mjlab/warp 依赖）在任务包
``hexapod_mjlab.tasks.confined_locomotion``（scanner/runtime），不在本包。
"""

from .contracts import (
    COMMAND_DIM,
    EMPTY_SCAN_ID,
    ENVELOPE_DIM,
    FAR_POINTS_PER_SCAN,
    FAR_WINDOW_SLOTS,
    FLOAT_DTYPE,
    INT_DTYPE,
    JOINT_DIM,
    NEAR_POINTS,
    POINT_CHANNELS,
    PROPRIO_DIM,
    PROPRIO_LAYOUT,
    ContractError,
    EnvelopeLabel,
    GroundLabel,
    IdentityMismatchError,
    OnlineObservation,
    OrderedPoints,
    ScanWindowSnapshot,
    bind_envelope_label,
    bind_ground_label,
    bind_window_to_observation,
    flatten_far_window,
    sort_points_by_angle_keys,
    validate_point_block,
)
from .history import (
    EpisodeMismatchError,
    ScanWindow,
    StaleScanError,
    TimeOrderError,
)

__all__ = [
    "COMMAND_DIM",
    "EMPTY_SCAN_ID",
    "ENVELOPE_DIM",
    "EpisodeMismatchError",
    "FAR_POINTS_PER_SCAN",
    "FAR_WINDOW_SLOTS",
    "FLOAT_DTYPE",
    "ContractError",
    "EnvelopeLabel",
    "GroundLabel",
    "IdentityMismatchError",
    "INT_DTYPE",
    "JOINT_DIM",
    "NEAR_POINTS",
    "OnlineObservation",
    "OrderedPoints",
    "POINT_CHANNELS",
    "PROPRIO_DIM",
    "PROPRIO_LAYOUT",
    "ScanWindow",
    "ScanWindowSnapshot",
    "StaleScanError",
    "TimeOrderError",
    "bind_envelope_label",
    "bind_ground_label",
    "bind_window_to_observation",
    "flatten_far_window",
    "sort_points_by_angle_keys",
    "validate_point_block",
]
