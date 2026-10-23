"""F4 集成：合成“扫描窗口—观测—网络—PD 执行”链路的准备/执行纯函数与离线标签绑定。

M2 起本文件为唯一正典实现（由 ``rl/joint_passage/integration.py`` 兼容垫片别名）；
除网络类改用正典名 ``ConfinedLocomotionNetwork``（同一类对象）外，与已验收的
joint_passage F4 集成逐字一致。仅标准库与 PyTorch 的纯计算模块（CPU 合成链路）：
不注册任务、不导入 mjlab 或任务注册器、不含真实仿真闭环、PPO/采样/logprob/
标准差、优化器、奖励或 reset/结束判定。依据：
docs/tasks/confined_locomotion/framework/实验记录.md、
docs/tasks/confined_locomotion/framework/使用说明.md。

接口分两侧（标签与在线输入严格隔离）：

* 在线侧（无标签/真值参数）：
    prepare_control_inputs(window, observation, near_angle_keys, far_angle_keys)
        对“已完成扫描窗口快照 + 当前在线观测”先做身份/时间校验（复用 F1
        bind_window_to_observation：按环境身份逐行查找窗口行、episode 逐行
        一致、近/远全部有效扫描不晚于控制步时间），再按调用方**显式提供**的
        近端 256 / 远端 640 角键准备行对齐、掩码保留的排序输入，并记录每行
        最新近端扫描身份。点张量与身份元数据为全新张量、与窗口快照不共享
        存储（窗口侧无 live 别名）；observation 字段借用传入对象而非克隆
        （调用方不得在 prepare 之后修改该对象，rollout 需另存独立观测快照）。
    execute_control_step(model, prepared, q_ref, joint_pos, joint_vel)
        F2 网络在线前向（不接受任何标签）→ action_mean → F3 纯 PD 分级结果。
        q_ref/q/q̇ 由调用方显式逐行提供：本模块不从观测推导、不读取任何机器
        人构型默认；action_mean 直接作为 raw_action 执行仅为**确定性合成检查**，
        不定义正式动作分布或采样策略。（M2 真实闭环由任务侧 runtime 直接把
        action_mean 交 env.step，由环境按 200Hz 重算 PD；本函数仍为纯计算
        检查入口。）
* 离线侧（训练标签绑定，与在线接口分离）：
    bind_offline_labels(observation, *, envelope_label, ground_label, window)
        轻量复用 F1 bind_envelope_label / bind_ground_label：包络标签 E* 对应
        当前控制步、动作发出前状态（控制步身份、行序对齐）；地面标签对应最新
        近端扫描时刻（扫描身份，值 z_ground−z_root 与有效掩码）。ground 标签的
        逻辑行须与观测**严格同 env/episode/批量/行序**（双方同序的重排/重复行可），
        再按扫描身份对窗口近端槽绑定；标签行配到观测之外的环境、缺行/多行/错行/
        设备不等一律拒绝；缺失行（scan_id=-1、全 False 掩码）合法。
        未来步/错 episode/错扫描/行序错位一律 IdentityMismatchError；
        无 loss/权重——属后续确认项。

边界（未确认项一律显式传参，不设默认）：
* 角度规则尚未在真实资产上复核：近/远角键必须由调用方显式供给；本模块与合成
  夹具均不冻结真实雷达方位角、不定义零轴/外参、不处理无命中端点；调用方若要
  保留近端原生扫描顺序，可显式传恒等键（0..N-1）。远端点保持“采集时机体坐标
  系”原米制数值：无运动补偿、无点缩放、无第 4 通道。
* 环境身份：观测逻辑行可重排/重复/子集，一律按 env 身份逐行查找窗口行配对，
  绝不把 env_id 当连续数组下标；缺失环境/错 episode/未来扫描由 F1 契约拒绝。
* 零扫描状态可准备（全 False 掩码编码，GRU 空行返回零特征，不捏造数据、不复制
  扫描补槽）；零扫描下是否允许实际控制仍待闭环确认，本模块不设 fallback、
  不做许可判断。
* 执行一致性：q_ref 与观测 joint_pos_rel_ref 共用同一参考、joint_pos/joint_vel
  与观测字段一致由调用方保证（F1/F3 契约语义），本模块不另设容差检查；本模块
  只核对执行输入与观测同批/同设备/同维。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .contracts import (
    INT_DTYPE,
    JOINT_DIM,
    ContractError,
    EnvelopeLabel,
    GroundLabel,
    IdentityMismatchError,
    OnlineObservation,
    ScanWindowSnapshot,
    bind_envelope_label,
    bind_ground_label,
    bind_window_to_observation,
    flatten_far_window,
    sort_points_by_angle_keys,
)
from .control import PDExecutionResult, compute_pd_torques
from .network import ConfinedLocomotionNetwork, NetworkOutput

__all__ = [
    "BoundTrainingLabels",
    "IntegratedControlResult",
    "PreparedControlInputs",
    "bind_offline_labels",
    "execute_control_step",
    "prepare_control_inputs",
]


@dataclass(frozen=True)
class PreparedControlInputs:
    """prepare_control_inputs 的输出：一次控制步的在线输入与身份记录。

    行序与传入观测的逻辑批行序一致（支持重排/重复/子集的环境行）；不含任何
    标签/真值字段。点张量与身份元数据为全新存储（行对齐与排序均产生新张量），
    不与窗口快照共享内存；近/远点保持原米制机体坐标（无缩放、无补偿），排序
    仅按调用方显式角键置换。零扫描行：near_filled=False、near_scan_ids=
    EMPTY_SCAN_ID、近/远掩码全 False（由掩码编码，不捏造数据）。
    observation 字段**借用**传入的 OnlineObservation 对象（非克隆）：调用方
    不得在 prepare 之后修改该对象；rollout 需要留存历史观测时须自行另存独立
    快照，本类不提供观测克隆、也不声称全部字段无别名（点云与扫描身份元数据
    的窗口存储确实隔离）。
    """

    observation: OnlineObservation       # 已通过身份/时间绑定的当前控制步观测（pre-action 状态）
    near_points: torch.Tensor            # (B,256,3) 米，近端角键升序、无效点稳定置尾
    near_valid: torch.Tensor             # (B,256) bool
    far_points: torch.Tensor             # (B,640,3) 米，最旧→最新拼接后按远端角键升序
    far_valid: torch.Tensor              # (B,640) bool
    near_filled: torch.Tensor            # (B,) bool，该行是否存在已完成扫描
    near_scan_ids: torch.Tensor          # (B,) int64，最新近端扫描身份（零扫描为 EMPTY_SCAN_ID）
    near_times: torch.Tensor             # (B,) float32 秒，与扫描同一调用方时钟
    far_point_scan_ids: torch.Tensor     # (B,640) int64，排序后每点所属扫描（空槽/空点 -1）
    far_point_times: torch.Tensor        # (B,640) float32 秒
    near_angle_keys: torch.Tensor | None = None  # (B,256) 已排序的 optical angle keys
    far_angle_keys: torch.Tensor | None = None   # (B,640) 已排序的 optical angle keys
    near_source_dirs_sensor: torch.Tensor | None = None  # (B,256,3) sensor-frame rays
    far_source_dirs_sensor: torch.Tensor | None = None   # (B,640,3) sensor-frame rays
    near_source_dirs_present: torch.Tensor | None = None # (B,256) source direction identity
    far_source_dirs_present: torch.Tensor | None = None  # (B,640) source direction identity
    near_source_indices: torch.Tensor | None = None      # (B,256) original ray ids
    far_source_indices: torch.Tensor | None = None       # (B,640) original ray ids


def _lookup_window_rows(
    window: ScanWindowSnapshot, observation: OnlineObservation
) -> torch.Tensor:
    """按环境身份把观测每行映射到窗口行号（int64 索引，供 index_select 使用）。

    仅在 bind_window_to_observation 通过后调用：快照 env_ids 唯一且观测 env
    全部存在于快照已由绑定保证；env_id 本身绝不作为数组下标使用。
    """
    equal = observation.env_ids[:, None] == window.env_ids[None, :]
    return equal.to(INT_DTYPE).argmax(dim=1)


def prepare_control_inputs(
    *,
    window: ScanWindowSnapshot,
    observation: OnlineObservation,
    near_angle_keys: torch.Tensor,
    far_angle_keys: torch.Tensor,
    near_source_dirs_sensor: torch.Tensor | None = None,
    far_source_dirs_sensor: torch.Tensor | None = None,
    near_source_dirs_present: torch.Tensor | None = None,
    far_source_dirs_present: torch.Tensor | None = None,
    near_source_indices: torch.Tensor | None = None,
    far_source_indices: torch.Tensor | None = None,
) -> PreparedControlInputs:
    """身份/时间校验 + 行对齐 + 显式角键排序，产出网络在线输入（纯函数）。

    * 先复用 F1 bind_window_to_observation 做逐行身份校验（环境存在、episode
      一致、近/远全部有效扫描不晚于控制步时间），缺失/错位/未来扫描一律拒绝。
    * 角键为调用方显式提供的每点 float32 键（近 (B,256)/远 (B,640)，行序对齐
      观测逻辑批）：角度规则/雷达外参未确认，本模块不默认计算角度、不冻结
      真实方位角；键契约（有效点处有限等）由 F1 sort_points_by_angle_keys
      强制。
    * 远端先经 F1 flatten_far_window（10 槽最旧→最新拼接为 640 点，附每点
      scan_id/time），再按远端角键排序；近端按近端角键排序；两路均掩码保留、
      无效点稳定置尾。GRU 沿角序递归，不以时间序或写入序替代。
    * 零扫描行以全 False 掩码通过（是否允许零扫描实际控制待闭环确认，本模块
      不设 fallback）。
    """
    bind_window_to_observation(window, observation)
    rows = _lookup_window_rows(window, observation)
    def stable_order(valid: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        effective = torch.where(valid, keys, torch.full_like(keys, float("inf")))
        return torch.argsort(effective, dim=1, stable=True)

    def gather(order: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        if values.dim() == 3:
            return torch.gather(values, 1, order.unsqueeze(-1).expand_as(values))
        return torch.gather(values, 1, order)

    near_raw_points = window.near_points.index_select(0, rows)
    near_raw_valid = window.near_valid.index_select(0, rows)
    near_order = stable_order(near_raw_valid, near_angle_keys)
    near_sorted = sort_points_by_angle_keys(near_raw_points, near_raw_valid, near_angle_keys)
    flat = flatten_far_window(window)
    far_raw_points = flat.points.index_select(0, rows)
    far_raw_valid = flat.valid.index_select(0, rows)
    far_order = stable_order(far_raw_valid, far_angle_keys)
    far_sorted = sort_points_by_angle_keys(
        far_raw_points,
        far_raw_valid,
        far_angle_keys,
        scan_ids=flat.scan_ids.index_select(0, rows),
        times=flat.times.index_select(0, rows),
    )
    source_args = (
        near_source_dirs_sensor, far_source_dirs_sensor,
        near_source_dirs_present, far_source_dirs_present,
        near_source_indices, far_source_indices,
    )
    if any(value is None for value in source_args) and any(
        value is not None for value in source_args
    ):
        raise ContractError("near/far source directions, presence masks and ray ids must be supplied together")
    near_dirs_sorted = far_dirs_sorted = None
    near_dirs_present_sorted = far_dirs_present_sorted = None
    near_source_indices_sorted = far_source_indices_sorted = None
    if near_source_dirs_sensor is not None:
        window_batch = int(window.env_ids.shape[0])
        expected = (
            ("near_source_dirs_sensor", near_source_dirs_sensor, (window_batch, near_raw_points.shape[1], 3), torch.float32),
            ("far_source_dirs_sensor", far_source_dirs_sensor, (window_batch, far_raw_points.shape[1], 3), torch.float32),
            ("near_source_dirs_present", near_source_dirs_present, (window_batch, near_raw_valid.shape[1]), torch.bool),
            ("far_source_dirs_present", far_source_dirs_present, (window_batch, far_raw_valid.shape[1]), torch.bool),
            ("near_source_indices", near_source_indices, (window_batch, near_raw_valid.shape[1]), INT_DTYPE),
            ("far_source_indices", far_source_indices, (window_batch, far_raw_valid.shape[1]), INT_DTYPE),
        )
        for name, value, shape, dtype in expected:
            if not isinstance(value, torch.Tensor) or value.dtype is not dtype \
                    or tuple(value.shape) != shape or value.device != near_raw_points.device:
                raise ContractError(f"{name} must be {shape}/{dtype} on {near_raw_points.device}")
        near_present = near_source_dirs_present.index_select(0, rows)
        far_present = far_source_dirs_present.index_select(0, rows)
        near_ids = near_source_indices.index_select(0, rows)
        far_ids = far_source_indices.index_select(0, rows)
        near_dirs = near_source_dirs_sensor.index_select(0, rows)
        far_dirs = far_source_dirs_sensor.index_select(0, rows)
        for name, directions, present, valid, indices in (
            ("near", near_dirs, near_present, near_raw_valid, near_ids),
            ("far", far_dirs, far_present, far_raw_valid, far_ids),
        ):
            if bool((present & ~valid).any()):
                raise ContractError(f"{name} source direction presence is true for an invalid point")
            if not torch.equal(present, indices >= 0):
                raise ContractError(f"{name} source direction presence and raw ray ids disagree")
            if bool(present.any()):
                selected = directions[present]
                if not bool(torch.isfinite(selected).all()):
                    raise ContractError(f"{name} source directions contain NaN/Inf")
                norms = torch.linalg.vector_norm(selected, dim=-1)
                if not bool(torch.allclose(norms, torch.ones_like(norms), atol=1e-5, rtol=0.0)):
                    raise ContractError(f"{name} source directions must be unit vectors")
            if bool((~present).any()) and not torch.equal(
                directions[~present], torch.zeros_like(directions[~present])
            ):
                raise ContractError(f"{name} missing source directions must be exactly zero")
        near_dirs_sorted = gather(near_order, near_dirs)
        far_dirs_sorted = gather(far_order, far_dirs)
        near_dirs_present_sorted = gather(near_order, near_present)
        far_dirs_present_sorted = gather(far_order, far_present)
        near_source_indices_sorted = gather(near_order, near_ids)
        far_source_indices_sorted = gather(far_order, far_ids)

    return PreparedControlInputs(
        observation=observation,
        near_points=near_sorted.points,
        near_valid=near_sorted.valid,
        far_points=far_sorted.points,
        far_valid=far_sorted.valid,
        near_filled=window.near_filled.index_select(0, rows),
        near_scan_ids=window.near_scan_ids.index_select(0, rows),
        near_times=window.near_times.index_select(0, rows),
        far_point_scan_ids=far_sorted.scan_ids,
        far_point_times=far_sorted.times,
        near_angle_keys=gather(near_order, near_angle_keys),
        far_angle_keys=gather(far_order, far_angle_keys),
        near_source_dirs_sensor=near_dirs_sorted,
        far_source_dirs_sensor=far_dirs_sorted,
        near_source_dirs_present=near_dirs_present_sorted,
        far_source_dirs_present=far_dirs_present_sorted,
        near_source_indices=near_source_indices_sorted,
        far_source_indices=far_source_indices_sorted,
    )


@dataclass(frozen=True)
class IntegratedControlResult:
    """execute_control_step 的输出：网络命名输出 + F3 PD 分级结果（无标签字段）。

    network.envelope 为米制 Ê（未 detach，辅助路径用；构造期初始值=启动基准）；
    pd 由 network.action_mean 作为 raw_action 直接进入 F3 执行链（裁剪→残差
    目标→纯 PD→力矩裁剪，分级语义分离保存）。mean 直接执行仅为确定性合成
    检查的记录，不定义正式动作分布或采样策略；无奖励/reset 输出。
    """

    network: NetworkOutput
    pd: PDExecutionResult


def execute_control_step(
    *,
    model: ConfinedLocomotionNetwork,
    prepared: PreparedControlInputs,
    q_ref: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
) -> IntegratedControlResult:
    """F2 网络在线前向（无标签入参）→ action_mean → F3 纯 PD（纯函数）。

    * model 须为正典 ConfinedLocomotionNetwork（兼容别名 JointPassageNetwork
      为同一类对象）：其 forward 签名不含任何标签/真值参数，标签在接口上
      无法进入在线路径。
    * q_ref/joint_pos/joint_vel (B,18) 由调用方显式逐行提供，须与观测逻辑批
      同批、同设备（dtype/有限性由 F3 compute_pd_torques 强制）；本模块不从
      观测的 joint_pos_rel_ref/joint_vel 推导执行状态——与观测字段的一致性
      由调用方按 F1/F3 契约保证，不在此另设容差检查。
    * 返回的 PD 结果是把 action_mean 直接当 raw_action 执行的确定性合成检查：
      不含采样、logprob、标准差，也不构成对正式动作分布的选择。
    """
    if not isinstance(model, ConfinedLocomotionNetwork):
        raise ContractError("model 必须为 ConfinedLocomotionNetwork（在线前向不接受标签）")
    if not isinstance(prepared, PreparedControlInputs):
        raise ContractError("prepared 必须为 prepare_control_inputs 的输出")
    observation = prepared.observation
    batch = int(observation.env_ids.shape[0])
    device = observation.env_ids.device
    for name, value in (("q_ref", q_ref), ("joint_pos", joint_pos), ("joint_vel", joint_vel)):
        if not isinstance(value, torch.Tensor) or value.dim() != 2 \
                or int(value.shape[0]) != batch or int(value.shape[1]) != JOINT_DIM:
            raise ContractError(
                f"{name} 应为 ({batch}, {JOINT_DIM}) 张量且与观测逻辑批同批，"
                f"实际 {tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__}"
            )
        if value.device != device:
            raise ContractError(f"{name} 设备 {value.device} 与观测 {device} 不一致")
    network_output = model(
        observation,
        prepared.near_points,
        prepared.near_valid,
        prepared.far_points,
        prepared.far_valid,
    )
    pd_result = compute_pd_torques(
        raw_action=network_output.action_mean,
        q_ref=q_ref,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
    )
    return IntegratedControlResult(network=network_output, pd=pd_result)


@dataclass(frozen=True)
class BoundTrainingLabels:
    """bind_offline_labels 的输出：已通过身份绑定的离线标签（原对象透传）。

    envelope 对应当前控制步动作发出前状态（控制步身份）；ground 对应最新近端
    扫描时刻（扫描身份，值 z_ground−z_root 与有效掩码）。两者时刻不同、各自
    独立绑定，不强行对齐；本结构只服务于训练侧，不得传入在线前向。
    """

    envelope: EnvelopeLabel | None = None
    ground: GroundLabel | None = None


def _require_label_row_identity(
    label_name: str,
    observation: OnlineObservation,
    env_ids: torch.Tensor,
    episode_ids: torch.Tensor,
) -> None:
    """离线标签的逻辑行与观测严格同 env/episode/批量/行序。

    支持双方同序的重排与重复行（同环境多控制步）；缺行/多行（批量不等）与
    设备不等抛 ContractError，行内容错位抛 IdentityMismatchError。本检查只
    约束“标签行属于本次观测的逻辑批”，不替代 F1 的扫描身份绑定——标签行
    配到观测之外的环境（即使该环境在窗口中合法存在）必须在此拒绝。
    """
    reference = observation.env_ids
    for name, value in (("env_ids", env_ids), ("episode_ids", episode_ids)):
        if not isinstance(value, torch.Tensor) or value.dim() != 1 or value.dtype is not INT_DTYPE:
            raise ContractError(f"{label_name}.{name} 必须是 (B,) int64 张量")
        if value.device != reference.device:
            raise ContractError(
                f"{label_name}.{name} 设备 {value.device} 与观测 {reference.device} 不一致"
            )
        if int(value.shape[0]) != int(reference.shape[0]):
            raise ContractError(
                f"{label_name} 行数 {int(value.shape[0])} 与观测逻辑批 "
                f"{int(reference.shape[0])} 不一致（缺行/多行拒绝）"
            )
    if not torch.equal(env_ids, reference):
        raise IdentityMismatchError(
            f"{label_name}.env_ids 与观测逻辑批行序不一致: "
            f"{env_ids.tolist()} vs {reference.tolist()}"
        )
    if not torch.equal(episode_ids, observation.episode_ids):
        raise IdentityMismatchError(
            f"{label_name}.episode_ids 与观测逻辑批行序不一致: "
            f"{episode_ids.tolist()} vs {observation.episode_ids.tolist()}"
        )


def bind_offline_labels(
    observation: OnlineObservation,
    *,
    envelope_label: EnvelopeLabel | None = None,
    ground_label: GroundLabel | None = None,
    window: ScanWindowSnapshot | None = None,
) -> BoundTrainingLabels:
    """训练侧标签绑定：轻量复用 F1 bind_envelope_label / bind_ground_label。

    * envelope_label：与观测同一逻辑采样批且行序对齐（逐行 (env, episode,
      step) 全等）；未来控制步标签、旧 episode、行序错位一律拒绝。
    * ground_label：先要求其逻辑行与观测**严格同 env/episode/批量/行序**
      （_require_label_row_identity：支持双方同序的重排/重复行；缺行/多行/
      设备不等抛 ContractError、行内容错位抛 IdentityMismatchError），再按
      扫描身份绑定到窗口近端槽当前扫描——不得以控制步身份替代扫描身份。
      须同时传入与 prepare_control_inputs **相同的窗口快照**（本函数会先
      校验该快照与观测的身份/时间一致性）；scan_id=-1 缺失行（如部分 reset
      后无扫描）合法：valid 全 False、零填、由掩码排除，且该行的 env/episode
      仍须与观测对应；错扫描/错 episode 拒绝。
    * 提供窗口时（即使只绑包络标签）同样执行窗口—观测绑定校验，防止用另一
      个窗口的标签冒充本控制步数据。无 loss/权重/teacher 失败分项——这些
      属后续确认项，不在本接口内决定。
    """
    if envelope_label is None and ground_label is None:
        raise ContractError("至少提供一类标签（envelope_label/ground_label），空绑定无意义")
    if ground_label is not None and window is None:
        raise ContractError("ground 标签按扫描身份绑定，须传入与 prepare_control_inputs 相同的窗口快照")
    if window is not None:
        bind_window_to_observation(window, observation)
    bound_envelope = (
        None if envelope_label is None else bind_envelope_label(observation, envelope_label)
    )
    bound_ground = None
    if ground_label is not None:
        _require_label_row_identity(
            "ground_label", observation, ground_label.env_ids, ground_label.episode_ids
        )
        bound_ground = bind_ground_label(window, ground_label)
    return BoundTrainingLabels(envelope=bound_envelope, ground=bound_ground)
