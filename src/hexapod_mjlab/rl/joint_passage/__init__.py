"""joint_passage 兼容入口：正典实现自 M2 起位于 ``hexapod_mjlab.rl.confined_locomotion``。

本包保留历史导入路径：``__init__`` 导出与原 F1 完全相同的符号集（同一对象，
非副本），各子模块（contracts/history/mapping/control/contacts/network/
integration）为 sys.modules 别名垫片，指向正典模块——所有公有/私有导入身份、
类身份（含 JointPassageNetwork ↔ ConfinedLocomotionNetwork 同一类对象）与
历史测试行为完全不变；数学语义零改动。历史证据与冻结配方保持原路径，不迁移。

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
"""

from hexapod_mjlab.rl.confined_locomotion.contracts import (
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
from hexapod_mjlab.rl.confined_locomotion.history import (
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
