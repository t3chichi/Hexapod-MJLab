"""F3 接触：逐接触判定、非足环境/自碰撞去重计数与旧足端绊碰（纯计算）。

M2 起本文件为唯一正典实现（由 ``rl/joint_passage/contacts.py`` 兼容垫片别名），
与已验收的 joint_passage F3 接触计算逐字一致：仅标准库与 PyTorch 的纯计算模块，
不导入 mjlab/任务注册器，不含 PPO/奖励权重/reset/结束判定，无全局可变状态。
依据：docs/tasks/confined_locomotion/framework/实验记录.md、
docs/tasks/confined_locomotion/平地学习与基准测量.md 第 2 节“接触惩罚”表、
docs/research/训练架构对齐.md 第 1 节“接触与结束”行。

已确认规则（本模块只计数与诊断，不含任何权重/启用安排）：
* 逐接触合格判据：distance ≤ 0 且 ‖force‖ > 0.1 N（force 为该接触自身的
  力向量范数）。逐条独立判断——多接触点不得先求和再过滤：同一 body 上
  方向相反的两条合格接触既不互相抵消，也不会因去重而重复计数。
* 环境碰撞计数：合格接触中“恰一侧为机器人非足 body、另一侧为外部 body”
  者，按机器人侧 body 去重（每 body 每物理子步计 1）。hip、雷达安装件等
  一切调用方列入机器人集合的非足 body 均在其中——本模块不持有、不默认
  任何 body 清单或 footID。普通足-环境接触（含正常地面支撑）不进入该计数。
* 自碰撞计数：合格接触中“两侧均为机器人 body”者，按无序 body 对去重，
  独立于环境计数；含足在内的自碰撞对不因正常地面支撑豁免。
* 每控制步恰 4 个物理子步：各子步分别去重计数后取平均（可为分数）；
  持续接触每个控制步正常计数，不叠加持续时间倍数（本模块为纯函数，
  无跨调用状态）。
* 原始几何接触独立保留在诊断输出（qualified 掩码与分类码）中，不因力
  阈值被删除；不把几何分类/接触法向加入任何奖励（场景分类与法向仅诊断，
  首版无墙/顶棚分类奖惩）。
* 足端绊碰独立判定（feet_stumble_flag）：复用旧任务力比判据，输入为控制
  时刻世界坐标每足**净力**（不得用逐接触力替代），见该函数 docstring。

body 归属语义：调用方显式提供机器人 body 集合与 6 个足 body 集合，二者
决定一切分类；不在机器人集合中的 body 一律为外部 body（未知 body 按实际
membership 判定为外部，不得把全部接触默认算作机器人）。body id 为任意
非负 int64，仅作身份比较（isin/unique），绝不当作数组连续下标使用。
环境索引同理：env_ids 为任意非负、非连续可的真实身份，接触记录以批次行号
关联，返回批次顺序与输入 env_ids 顺序一致。

实际传感器覆盖未验证：GPU 上 mjlab 是否报告全部环境接触与自碰撞、力向量
的符号/参考系约定、距离字段语义，由调用方在闭环接入前核对；本模块只做
上述纯计算，不验证覆盖、不写 reset。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .contracts import FLOAT_DTYPE, INT_DTYPE, ContractError

__all__ = [
    "CONTACT_CATEGORY_EXTERNAL_ONLY",
    "CONTACT_CATEGORY_FOOT_ENV",
    "CONTACT_CATEGORY_NAMES",
    "CONTACT_CATEGORY_NONFOOT_ENV",
    "CONTACT_CATEGORY_SELF",
    "CONTACT_DISTANCE_MAX_M",
    "CONTACT_FORCE_MIN_N",
    "FOOT_COUNT",
    "PHYSICS_SUBSTEPS_PER_CONTROL_STEP",
    "STUMBLE_FORCE_RATIO",
    "ContactControlStepResult",
    "ControlStepContacts",
    "count_control_step_contacts",
    "feet_stumble_flag",
]

# 已确认常量（平地学习与基准测量.md 第 2 节）。
FOOT_COUNT = 6                              # 六足结构（足 body 集合恰 6 个）
PHYSICS_SUBSTEPS_PER_CONTROL_STEP = 4       # 每控制步物理子步数（50Hz/200Hz）
CONTACT_FORCE_MIN_N = 0.1                   # 逐接触力向量范数阈值（N，严格大于）
CONTACT_DISTANCE_MAX_M = 0.0                # 逐接触距离上限（m，<= 为穿透/触及）
STUMBLE_FORCE_RATIO = 5.0                   # 旧足端绊碰力比：‖F_xy‖ > 5·|F_z|

# 逐接触 body 归属分类码（诊断用；与合格判据相互独立）。
CONTACT_CATEGORY_NONFOOT_ENV = 0       # 恰一侧机器人非足 body × 外部 body → 环境计数
CONTACT_CATEGORY_FOOT_ENV = 1          # 恰一侧机器人足 body × 外部 body → 不入环境计数
CONTACT_CATEGORY_SELF = 2              # 机器人 body × 机器人 body → 自碰撞无序对
CONTACT_CATEGORY_EXTERNAL_ONLY = 3     # 外部 × 外部 → 与机器人无关，不计数
CONTACT_CATEGORY_NAMES = (
    "nonfoot_robot_env",
    "foot_robot_env",
    "robot_self",
    "external_only",
)


def _require_tensor(
    name: str,
    value: object,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    ref_device: torch.device,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ContractError(f"{name} 必须是 torch.Tensor，实际 {type(value).__name__}")
    if value.dtype is not dtype:
        raise ContractError(f"{name} dtype 应为 {dtype}，实际 {value.dtype}")
    if tuple(value.shape) != shape:
        raise ContractError(f"{name} 形状应为 {shape}，实际 {tuple(value.shape)}")
    if value.device != ref_device:
        raise ContractError(f"{name} 设备 {value.device} 与批内基准 {ref_device} 不一致")
    return value


@dataclass
class ControlStepContacts:
    """一个控制步、B 个环境的原始逐接触记录（调用方从仿真读取后显式提供）。

    * env_ids (B,) int64：非负真实环境身份且唯一（单个控制步每环境恰一行，
      允许非连续/任意顺序）；返回结果的批次顺序与此一致。
    * 每条接触共 C 行（行序保留，与返回的 qualified/categories 对齐）：
        contact_env_rows (C,) int64 ∈ [0,B)：接触所属**批次行号**（行下标，
          不是环境 id——环境身份只写在 env_ids）；
        substeps (C,) int64 ∈ [0,4)：所属物理子步（每控制步恰 4 个）；
        body_a/body_b (C,) int64：两侧 body id（非负、可非连续、仅作身份，
          不作数组下标；同一行 body_a ≠ body_b）；
        distance (C,) float32 米：调用方语义（mjlab 惯例 ≤0 为穿透/触及，
          >0 为碰撞 margin 内的近距候选）；
        force (C,3) float32 牛顿：该接触自身的力向量（范数用于合格判据；
          符号/参考系约定由调用方与传感器侧核对）。
    * C=0（该批无接触）合法。所有张量同设备同批一致、完整有限；构造即校验。
    """

    env_ids: torch.Tensor           # (B,) int64 非负唯一
    contact_env_rows: torch.Tensor  # (C,) int64 ∈ [0,B)
    substeps: torch.Tensor          # (C,) int64 ∈ [0,4)
    body_a: torch.Tensor            # (C,) int64 非负
    body_b: torch.Tensor            # (C,) int64 非负，且逐行 body_a ≠ body_b
    distance: torch.Tensor          # (C,) float32 米
    force: torch.Tensor             # (C,3) float32 N

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.env_ids, torch.Tensor) or self.env_ids.dim() != 1 \
                or self.env_ids.dtype is not INT_DTYPE:
            raise ContractError("env_ids 必须是 (B,) int64 张量")
        batch = int(self.env_ids.shape[0])
        if batch < 1:
            raise ContractError("批量维度必须 >= 1")
        if bool((self.env_ids < 0).any()):
            raise ContractError("env_ids 必须非负（真实环境身份，不做重编号）")
        if self.env_ids.numel() != torch.unique(self.env_ids).numel():
            raise ContractError("env_ids 必须唯一（单个控制步每环境恰一行）")
        device = self.env_ids.device
        count: int | None = None
        for name in ("contact_env_rows", "substeps", "body_a", "body_b"):
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor) or value.dim() != 1 \
                    or value.dtype is not INT_DTYPE:
                raise ContractError(f"{name} 必须是 (C,) int64 张量")
            if count is None:
                count = int(value.shape[0])
            elif int(value.shape[0]) != count:
                raise ContractError(
                    f"逐接触字段接触数 C 不一致：{name} 为 {int(value.shape[0])}，应为 {count}"
                )
            if value.device != device:
                raise ContractError(f"{name} 设备 {value.device} 与 env_ids 设备 {device} 不一致")
        assert count is not None
        _require_tensor("distance", self.distance, FLOAT_DTYPE, (count,), device)
        _require_tensor("force", self.force, FLOAT_DTYPE, (count, 3), device)
        if not bool(torch.isfinite(self.distance).all()):
            raise ContractError("distance 含 NaN/Inf")
        if not bool(torch.isfinite(self.force).all()):
            raise ContractError("force 含 NaN/Inf")
        if count > 0:
            if bool((self.contact_env_rows < 0).any()) or bool((self.contact_env_rows >= batch).any()):
                raise ContractError(
                    f"contact_env_rows 必须在 [0,{batch})（批次行号，非环境 id）"
                )
            if bool((self.substeps < 0).any()) \
                    or bool((self.substeps >= PHYSICS_SUBSTEPS_PER_CONTROL_STEP).any()):
                raise ContractError(
                    f"substeps 必须在 [0,{PHYSICS_SUBSTEPS_PER_CONTROL_STEP})（每控制步恰 4 个物理子步）"
                )
            if bool((self.body_a < 0).any()) or bool((self.body_b < 0).any()):
                raise ContractError("body_a/body_b 必须非负")
            if bool((self.body_a == self.body_b).any()):
                raise ContractError("同一接触的 body_a 与 body_b 不得为同一 body")


@dataclass(frozen=True)
class ContactControlStepResult:
    """count_control_step_contacts 的输出：批次顺序与输入 env_ids 一致。

    env_collision_counts/self_collision_counts (B,) float32：4 个物理子步各自
    去重计数后的平均（可为分数，如仅 1 个子步合格为 0.25）；两项相互独立。
    *_per_substep (B,4) int64：第 k 个物理子步去重后的 body 数 / 无序 body
    对数（去重中间结果，供诊断）。qualified (C,) bool 与 categories (C,)
    int64 为逐接触诊断：前者是合格判据结果（distance≤0 且 ‖force‖>0.1），
    后者是 body 归属分类（CONTACT_CATEGORY_*，与合格与否无关）；两者均按
    输入接触行序对齐，原始几何接触不因阈值被删除。
    """

    env_ids: torch.Tensor                       # (B,) int64，与输入批次顺序一致
    env_collision_counts: torch.Tensor          # (B,) float32，非足环境接触平均计数
    self_collision_counts: torch.Tensor         # (B,) float32，自碰撞平均计数
    env_collision_counts_per_substep: torch.Tensor   # (B,4) int64
    self_collision_counts_per_substep: torch.Tensor  # (B,4) int64
    qualified: torch.Tensor                     # (C,) bool
    categories: torch.Tensor                    # (C,) int64


def _validated_body_ids(
    name: str,
    value: object,
    ref_device: torch.device,
    *,
    exact_count: int | None = None,
) -> torch.Tensor:
    """body 身份集合：(K,) int64 非负唯一同设备；K 有明确要求时精确校验。"""
    if not isinstance(value, torch.Tensor) or value.dim() != 1 or value.dtype is not INT_DTYPE:
        raise ContractError(f"{name} 必须是 (K,) int64 张量（真实 body 身份集合）")
    if value.device != ref_device:
        raise ContractError(f"{name} 设备 {value.device} 与批内基准 {ref_device} 不一致")
    if int(value.shape[0]) < 1:
        raise ContractError(f"{name} 不可为空（归属由实际集合判定，不接受隐式默认）")
    if exact_count is not None and int(value.shape[0]) != exact_count:
        raise ContractError(f"{name} 必须恰有 {exact_count} 个 body，实际 {int(value.shape[0])}")
    if bool((value < 0).any()):
        raise ContractError(f"{name} 必须非负")
    if value.numel() != torch.unique(value).numel():
        raise ContractError(f"{name} 必须唯一（body 身份不得重复）")
    return value


def count_control_step_contacts(
    contacts: ControlStepContacts,
    *,
    robot_body_ids: torch.Tensor,
    foot_body_ids: torch.Tensor,
) -> ContactControlStepResult:
    """逐接触判定 + 非足环境/自碰撞去重计数 + 4 子步平均（纯计算，见模块 docstring）。

    robot_body_ids (R,) 与 foot_body_ids (恰 FOOT_COUNT=6) 为调用方显式提供的
    body 身份集合（int64 非负唯一同设备；足集须为机器人集合的子集）。归属
    完全由这两个集合决定：不在机器人集合中的 body 一律按外部处理；不默认
    遗漏 hip/雷达等任何 body，也不把外部接触算作机器人。权重与奖励不在此处。
    """
    contacts.validate()
    device = contacts.env_ids.device
    _validated_body_ids("robot_body_ids", robot_body_ids, device)
    _validated_body_ids("foot_body_ids", foot_body_ids, device, exact_count=FOOT_COUNT)
    if not bool(torch.isin(foot_body_ids, robot_body_ids).all()):
        raise ContractError("foot_body_ids 必须是 robot_body_ids 的子集（足属机器人 body）")

    body_a, body_b = contacts.body_a, contacts.body_b
    a_robot = torch.isin(body_a, robot_body_ids)
    b_robot = torch.isin(body_b, robot_body_ids)
    one_robot = a_robot ^ b_robot
    both_robot = a_robot & b_robot
    # 单侧机器人为真时，机器人侧是否为足（both_robot 行由 SELF 覆盖、无机器人行无关）。
    robot_side_is_foot = torch.where(
        a_robot, torch.isin(body_a, foot_body_ids), torch.isin(body_b, foot_body_ids)
    )
    categories = torch.full_like(body_a, CONTACT_CATEGORY_EXTERNAL_ONLY)
    categories[one_robot & ~robot_side_is_foot] = CONTACT_CATEGORY_NONFOOT_ENV
    categories[one_robot & robot_side_is_foot] = CONTACT_CATEGORY_FOOT_ENV
    categories[both_robot] = CONTACT_CATEGORY_SELF

    force_norm = torch.linalg.vector_norm(contacts.force, dim=-1)
    qualified = (contacts.distance <= CONTACT_DISTANCE_MAX_M) & (force_norm > CONTACT_FORCE_MIN_N)

    batch = int(contacts.env_ids.shape[0])
    group = contacts.contact_env_rows * PHYSICS_SUBSTEPS_PER_CONTROL_STEP + contacts.substeps
    total_groups = batch * PHYSICS_SUBSTEPS_PER_CONTROL_STEP

    env_selected = qualified & (categories == CONTACT_CATEGORY_NONFOOT_ENV)
    robot_side = torch.where(a_robot, body_a, body_b)
    env_keys = torch.stack((group[env_selected], robot_side[env_selected]), dim=1)
    env_unique = torch.unique(env_keys, dim=0)
    env_per_substep = torch.bincount(
        env_unique[:, 0].contiguous(), minlength=total_groups
    ).reshape(batch, PHYSICS_SUBSTEPS_PER_CONTROL_STEP)

    self_selected = qualified & (categories == CONTACT_CATEGORY_SELF)
    pair_lo = torch.minimum(body_a, body_b)
    pair_hi = torch.maximum(body_a, body_b)
    self_keys = torch.stack((group[self_selected], pair_lo[self_selected], pair_hi[self_selected]), dim=1)
    self_unique = torch.unique(self_keys, dim=0)
    self_per_substep = torch.bincount(
        self_unique[:, 0].contiguous(), minlength=total_groups
    ).reshape(batch, PHYSICS_SUBSTEPS_PER_CONTROL_STEP)

    return ContactControlStepResult(
        env_ids=contacts.env_ids,
        env_collision_counts=(
            env_per_substep.sum(dim=1).to(FLOAT_DTYPE) / PHYSICS_SUBSTEPS_PER_CONTROL_STEP
        ),
        self_collision_counts=(
            self_per_substep.sum(dim=1).to(FLOAT_DTYPE) / PHYSICS_SUBSTEPS_PER_CONTROL_STEP
        ),
        env_collision_counts_per_substep=env_per_substep,
        self_collision_counts_per_substep=self_per_substep,
        qualified=qualified,
        categories=categories,
    )


def feet_stumble_flag(foot_forces: torch.Tensor) -> torch.Tensor:
    """旧足端绊碰 0/1 判定（每环境一个控制步，与接触计数相互独立）。

    * foot_forces (B, FOOT_COUNT, 3) float32 牛顿、**世界坐标系**、控制时刻每足
      净力（该足全部接触的合力，由调用方/传感器给出——不得以单条接触力替代）。
    * 判据复用旧任务（el_4090_lidar feet_stumble）：任一足 ‖F_xy‖ > 5·|F_z|
      即该环境记 1，否则 0；多足触发不累计；严格大于（恰等于比值不触发）；
      无 0.1 N 门限（小于 0.1 N 的水平净力仍按比值判定；零力不触发：
      0 > 5·0 为假）。
    * 单控制步单次判定：不做 4 子步平均、不跨控制步累计、无全局状态；
      行序=输入行序。返回 (B,) float32，取值恰为 0.0/1.0。
    """
    if not isinstance(foot_forces, torch.Tensor) or foot_forces.dim() != 3 \
            or int(foot_forces.shape[1]) != FOOT_COUNT or int(foot_forces.shape[2]) != 3:
        raise ContractError(
            f"foot_forces 应为 (B, {FOOT_COUNT}, 3) float32 张量（世界坐标净足力）"
        )
    if foot_forces.dtype is not FLOAT_DTYPE:
        raise ContractError(f"foot_forces dtype 应为 {FLOAT_DTYPE}，实际 {foot_forces.dtype}")
    if not bool(torch.isfinite(foot_forces).all()):
        raise ContractError("foot_forces 含 NaN/Inf")
    horizontal = torch.linalg.vector_norm(foot_forces[..., :2], dim=-1)
    vertical = foot_forces[..., 2].abs()
    trip = (horizontal > STUMBLE_FORCE_RATIO * vertical).any(dim=1)
    return trip.to(FLOAT_DTYPE)
