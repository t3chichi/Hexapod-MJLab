#!/usr/bin/env python
"""F3 自检：执行链与接触计数（合成输入，CPU，unittest）。

覆盖真实风险反例：动作裁剪顺序（先 ±100 裁剪再 0.25 残差，无 ±3/±2.7 位置
裁剪——目标可越出名义限位）、PD 增益与 ±80 N·m 力矩上限（恰 80 不算饱和、
80.5 饱和）、输入不被就地修改、raw/clipped 语义分离；逐接触 distance=0/
正距与 force 范数 0.1/刚大于 0.1 边界、同 body 多接触点与方向相反接触力
（不先求和再过滤、不重复计数）、自碰撞无序 body 对（(a,b)/(b,a) 同对）、
含足自碰撞不豁免、普通足-环境接触不入非足环境计数、hip/雷达由调用方集合
决定（移出机器人集合即变外部）、4 子步先去重计数再平均（分数值）、持续接触
跨控制步不叠乘、非连续环境身份按批次行号关联、C=0 空接触；足端绊碰旧力比
判据（任一足 0/1、恰等于比值不触发、零力不触发、<0.1 N 水平净力无门限）；
输入契约边界（形状/dtype/NaN/范围/设备）；模块只依赖标准库与 torch、无奖励
权重/reset。

合成夹具声明：期望值全部由本文件用纯 Python 算术独立解析计算（不调用被测
模块内部路径）；执行链与 PD 边界夹具取二进制精确值（k/8、k/16 型 dyadic
数），使 float32 全链精确、可用 torch.equal 严格比较；力阈值边界 (0.1,0,0)
在 float32/f64 两种累加路径下范数都精确落在 float32(0.1) 上（严格 > 排除），
0.1000001 与 0.07 级别的值则与边界保持多个 ulp 以上距离。机器人 body 集合
（躯干 0、足 1-6、hip 7、腿 8、雷达 13、地面 100、墙 101）为合成身份，
不代表任务默认配置或真实 footID。本检查由 GLM 编写，本机 headless 无审批
客户端未运行；由 Codex 在交付后执行，失败以非零退出传播。不含真实仿真、
GPU 接触覆盖或训练结论。

修正轮（2026-10-05）：Codex 真实首跑 23 项中 22 项通过、1 项 ERROR——
test_pd_and_torque_clip_boundaries 的 scenarios 五元组 (err, qdot, 期望未裁剪,
期望施加, 期望饱和) 被误解包为四个名字，解包阶段即 ValueError（测试编码
错误，非生产缺陷；独立随机接触集合解析、4 子步均值、PD 裁剪顺序与小水平
净足力绊碰检查均通过）。修正为忽略前两个输入字段、取后三个期望值；全部
scenarios、断言与 23 项检查保留，未删测试、未放宽边界。本轮修正为静态
修改，本机未运行，由 Codex 复跑。
"""

import ast
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import torch

from hexapod_mjlab.rl.joint_passage.contacts import (
    CONTACT_CATEGORY_EXTERNAL_ONLY,
    CONTACT_CATEGORY_FOOT_ENV,
    CONTACT_CATEGORY_NONFOOT_ENV,
    CONTACT_CATEGORY_SELF,
    FOOT_COUNT,
    PHYSICS_SUBSTEPS_PER_CONTROL_STEP,
    ContactControlStepResult,
    ControlStepContacts,
    count_control_step_contacts,
    feet_stumble_flag,
)
from hexapod_mjlab.rl.joint_passage.control import (
    ACTION_CLIP,
    PD_KP,
    PD_KD,
    RESIDUAL_SCALE,
    TORQUE_CLIP,
    PDExecutionResult,
    compute_pd_torques,
)
from hexapod_mjlab.rl.joint_passage.contracts import JOINT_DIM, ContractError

JOINTS = JOINT_DIM  # 18

# 合成 body 身份（夹具，非任务默认）：躯干 0、足 1-6、非足机器人 body
# 7(hip)/8(腿)/13(雷达安装件)，外部 body 100(地面)/101(墙)。
FEET = [1, 2, 3, 4, 5, 6]
GROUND, WALL = 100, 101
ROBOT_BODIES = [0] + FEET + [7, 8, 13]


# ---------------------------------------------------------------------------
# 独立解析参考（纯 Python，不调用被测模块）
# ---------------------------------------------------------------------------

def reference_chain(raw, q_ref, q, qdot):
    """执行链规则的手工解析：clip±100 → q_ref+0.25·clip → 130·Δq−2·q̇ → clip±80。"""
    clipped = max(-ACTION_CLIP, min(ACTION_CLIP, float(raw)))
    target = float(q_ref) + RESIDUAL_SCALE * clipped
    unclipped = PD_KP * (target - float(q)) - PD_KD * float(qdot)
    applied = max(-TORQUE_CLIP, min(TORQUE_CLIP, unclipped))
    return clipped, target, unclipped, applied, abs(unclipped) > TORQUE_CLIP


def chain_tensors(raw_rows, q_ref_rows, q_rows, qdot_rows):
    """按 reference_chain 逐元素解析期望张量（float64 精确计算后转 float32）。"""
    batch = len(raw_rows)
    fields = [[] for _ in range(5)]
    for r in range(batch):
        for j in range(JOINTS):
            values = reference_chain(raw_rows[r][j], q_ref_rows[r][j], q_rows[r][j], qdot_rows[r][j])
            for k, value in enumerate(values):
                fields[k].append(value)
    return [
        torch.tensor(fields[0], dtype=torch.float32).reshape(batch, JOINTS),   # clipped
        torch.tensor(fields[1], dtype=torch.float32).reshape(batch, JOINTS),   # target
        torch.tensor(fields[2], dtype=torch.float32).reshape(batch, JOINTS),   # unclipped
        torch.tensor(fields[3], dtype=torch.float32).reshape(batch, JOINTS),   # applied
        torch.tensor(fields[4], dtype=torch.bool).reshape(batch, JOINTS),      # saturated
    ]


def make_contacts(env_ids, rows, substeps, pairs, distances=None, forces=None):
    """构造一个控制步的合成接触批；默认全部合格（distance<0、力 3N）。"""
    count = len(rows)
    if distances is None:
        distances = [-0.01] * count
    if forces is None:
        forces = [[3.0, 0.0, 0.0]] * count
    return ControlStepContacts(
        env_ids=torch.tensor(env_ids, dtype=torch.int64),
        contact_env_rows=torch.tensor(rows, dtype=torch.int64),
        substeps=torch.tensor(substeps, dtype=torch.int64),
        body_a=torch.tensor([pair[0] for pair in pairs], dtype=torch.int64),
        body_b=torch.tensor([pair[1] for pair in pairs], dtype=torch.int64),
        distance=torch.tensor(distances, dtype=torch.float32),
        force=torch.tensor(forces, dtype=torch.float32).reshape(count, 3),
    )


def run_count(contacts, robot=None, feet=None):
    return count_control_step_contacts(
        contacts,
        robot_body_ids=torch.tensor(ROBOT_BODIES if robot is None else robot, dtype=torch.int64),
        foot_body_ids=torch.tensor(FEET if feet is None else feet, dtype=torch.int64),
    )


# ---------------------------------------------------------------------------
# 执行链（control.py）
# ---------------------------------------------------------------------------

class ComputePDTorquesTest(unittest.TestCase):
    """残差目标与 PD 力矩：数值/顺序/边界用独立解析值严格比较。"""

    def test_execution_chain_matches_independent_values(self):
        """裁剪→残差→PD→力矩裁剪全链与纯 Python 解析值逐元素相等。

        夹具含 raw=±1e6/±100（含边界值本身）/±99.9375/0/37.25 等混合值；
        所有量取 dyadic 精确值，float32 全链无舍入，torch.equal 严格成立。
        """
        raw_cycle = [-1000000.0, -100.0, -99.9375, -0.5, 0.0, 37.25, 99.9375, 100.0, 1000000.0]
        raw_rows, q_ref_rows, q_rows, qdot_rows = [], [], [], []
        err_cycles = [[0.5, -0.25, 0.0], [2.0, 0.0, -1.0]]
        for r in range(2):
            raw_rows.append([raw_cycle[(j + 3 * r) % 9] for j in range(JOINTS)])
            q_ref_rows.append([(j * 0.125 if r == 0 else 1.0 - 0.125 * j) for j in range(JOINTS)])
            err = err_cycles[r]
            q_rows.append([q_ref_rows[r][j] - err[j % 3] for j in range(JOINTS)])
            qdot_rows.append([(0.25 * (j % 4) - 0.375 if r == 0 else -0.5 * (j % 3)) for j in range(JOINTS)])
        raw = torch.tensor(raw_rows, dtype=torch.float32)
        q_ref = torch.tensor(q_ref_rows, dtype=torch.float32)
        q = torch.tensor(q_rows, dtype=torch.float32)
        qdot = torch.tensor(qdot_rows, dtype=torch.float32)
        result = compute_pd_torques(raw_action=raw, q_ref=q_ref, joint_pos=q, joint_vel=qdot)
        self.assertIsInstance(result, PDExecutionResult)
        exp_clipped, exp_target, exp_unclipped, exp_applied, exp_sat = chain_tensors(
            raw_rows, q_ref_rows, q_rows, qdot_rows
        )
        self.assertTrue(torch.equal(result.clipped_action, exp_clipped))
        self.assertTrue(torch.equal(result.joint_target, exp_target))
        self.assertTrue(torch.equal(result.unclipped_torque, exp_unclipped))
        self.assertTrue(torch.equal(result.applied_torque, exp_applied))
        self.assertTrue(torch.equal(result.torque_saturated, exp_sat))
        self.assertTrue(torch.equal(result.raw_action, raw))
        for name in ("raw_action", "clipped_action", "joint_target", "unclipped_torque", "applied_torque"):
            self.assertEqual(tuple(getattr(result, name).shape), (2, JOINTS), name)
            self.assertEqual(getattr(result, name).dtype, torch.float32, name)
        self.assertEqual(result.torque_saturated.dtype, torch.bool)

    def test_residual_target_reaches_25rad_without_position_clip(self):
        """±100 裁剪 → 残差目标恰 ±25 rad（已有方案数值，非安全位置认证）。

        q_ref=1.0、raw=±1e6 → target 精确 26.0/−24.0，远超 ±3/±2.7：若误加
        位置裁剪，此处会得到 ±3/±2.7 而严格相等失败。
        """
        raw = torch.zeros(1, JOINTS, dtype=torch.float32)
        raw[0, 0] = 1000000.0
        raw[0, 1] = -1000000.0
        raw[0, 2] = 100.0        # 边界值本身：clip 含端点
        raw[0, 3] = -100.0
        raw[0, 4] = 99.9375      # 界内原样通过
        q_ref = torch.ones(1, JOINTS, dtype=torch.float32)
        result = compute_pd_torques(
            raw_action=raw, q_ref=q_ref,
            joint_pos=torch.zeros(1, JOINTS, dtype=torch.float32),
            joint_vel=torch.zeros(1, JOINTS, dtype=torch.float32),
        )
        self.assertEqual(float(result.clipped_action[0, 0]), 100.0)
        self.assertEqual(float(result.clipped_action[0, 1]), -100.0)
        self.assertEqual(float(result.clipped_action[0, 2]), 100.0)
        self.assertEqual(float(result.clipped_action[0, 3]), -100.0)
        self.assertEqual(float(result.clipped_action[0, 4]), 99.9375)
        self.assertEqual(float(result.joint_target[0, 0]), 26.0)     # 1 + 25
        self.assertEqual(float(result.joint_target[0, 1]), -24.0)    # 1 − 25
        self.assertEqual(float(result.joint_target[0, 2]), 26.0)
        self.assertEqual(float(result.joint_target[0, 3]), -24.0)
        self.assertEqual(float(result.joint_target[0, 4]), 1.0 + 0.25 * 99.9375)
        # 残差幅值上限 = RESIDUAL_SCALE·ACTION_CLIP = 25（不是 ±3/±2.7）。
        self.assertEqual(RESIDUAL_SCALE * ACTION_CLIP, 25.0)
        self.assertGreater(float((result.joint_target - q_ref).abs().max()), 3.0)

    def test_pd_and_torque_clip_boundaries(self):
        """130·Δq−2·q̇ 与 ±80 上限的解析边界：恰 ±80 不饱和、±80.5 饱和。"""
        # (误差, 关节速度, 期望未裁剪, 期望施加, 期望饱和)：全部 dyadic 精确。
        scenarios = [
            (0.5, -7.5, 80.0, 80.0, False),     # 65+15=80 恰在上限 → 不算饱和
            (0.5, -7.75, 80.5, 80.0, True),
            (-0.5, 7.5, -80.0, -80.0, False),
            (-0.5, 7.75, -80.5, -80.0, True),
            (2.0, 0.0, 260.0, 80.0, True),
            (-2.0, 0.0, -260.0, -80.0, True),
            (0.125, 0.5, 15.25, 15.25, False),  # 界内原样通过：16.25−1.0
            (0.0, 0.0, 0.0, 0.0, False),
        ]
        err = [scenarios[j % 8][0] for j in range(JOINTS)]
        qdot = [scenarios[j % 8][1] for j in range(JOINTS)]
        q_ref = torch.ones(1, JOINTS, dtype=torch.float32)
        raw = torch.zeros(1, JOINTS, dtype=torch.float32)   # target = q_ref
        q = torch.tensor([[1.0 - err[j] for j in range(JOINTS)]], dtype=torch.float32)
        qdot_t = torch.tensor([qdot], dtype=torch.float32)
        result = compute_pd_torques(raw_action=raw, q_ref=q_ref, joint_pos=q, joint_vel=qdot_t)
        for j in range(JOINTS):
            # 每条 scenario 为 (err, qdot, 期望未裁剪, 期望施加, 期望饱和)：
            # 忽略前两个输入字段，取后三个期望值。
            _, _, exp_unclipped, exp_applied, exp_sat = scenarios[j % 8]
            self.assertEqual(float(result.unclipped_torque[0, j]), exp_unclipped, f"joint {j}")
            self.assertEqual(float(result.applied_torque[0, j]), exp_applied, f"joint {j}")
            self.assertEqual(bool(result.torque_saturated[0, j]), exp_sat, f"joint {j}")

    def test_inputs_not_modified_and_raw_action_is_independent_copy(self):
        """输入不被就地修改；raw_action 字段为独立副本，改结果不污染调用方。"""
        torch.manual_seed(11)
        raw = torch.randn(3, JOINTS) * 60
        q_ref = torch.randn(3, JOINTS) * 0.5
        q = torch.randn(3, JOINTS) * 0.5
        qdot = torch.randn(3, JOINTS) * 2.0
        snapshots = [t.clone() for t in (raw, q_ref, q, qdot)]
        result = compute_pd_torques(raw_action=raw, q_ref=q_ref, joint_pos=q, joint_vel=qdot)
        for original, snapshot in zip((raw, q_ref, q, qdot), snapshots):
            self.assertTrue(torch.equal(original, snapshot))
        self.assertNotEqual(result.raw_action.data_ptr(), raw.data_ptr())
        result.raw_action.zero_()
        self.assertTrue(torch.equal(raw, snapshots[0]))     # 调用方张量不受影响

    def test_rows_are_independent_and_order_preserved(self):
        """逐行独立：改一行输入只影响该行输出；行序与输入一致。"""
        base = torch.zeros(3, JOINTS, dtype=torch.float32)
        q_ref = torch.zeros(3, JOINTS, dtype=torch.float32)
        first = compute_pd_torques(raw_action=base.clone(), q_ref=q_ref.clone(),
                                   joint_pos=base.clone(), joint_vel=base.clone())
        changed = base.clone()
        changed[1, 0] = 50.0
        second = compute_pd_torques(raw_action=changed, q_ref=q_ref, joint_pos=base, joint_vel=base)
        self.assertTrue(torch.equal(second.joint_target[0], first.joint_target[0]))
        self.assertTrue(torch.equal(second.joint_target[2], first.joint_target[2]))
        self.assertFalse(torch.equal(second.joint_target[1], first.joint_target[1]))
        self.assertEqual(float(second.joint_target[1, 0]), 12.5)   # 0 + 0.25·50

    def test_control_contract_rejections(self):
        """形状/dtype/有限性/批量/设备契约违约一律 ContractError。"""
        raw = torch.zeros(2, JOINTS, dtype=torch.float32)
        q_ref = torch.zeros(2, JOINTS, dtype=torch.float32)
        joint_pos = torch.zeros(2, JOINTS, dtype=torch.float32)
        joint_vel = torch.zeros(2, JOINTS, dtype=torch.float32)
        compute_pd_torques(raw_action=raw, q_ref=q_ref, joint_pos=joint_pos, joint_vel=joint_vel)

        def bad(**overrides):
            kwargs = dict(raw_action=raw, q_ref=q_ref, joint_pos=joint_pos, joint_vel=joint_vel)
            kwargs.update(overrides)
            compute_pd_torques(**kwargs)

        nan_raw = raw.clone()
        nan_raw[1, 4] = float("nan")
        cases = [
            dict(raw_action=raw[:, :17]),                     # 关节维 17
            dict(raw_action=raw[0]),                          # 缺批量维
            dict(raw_action=raw.unsqueeze(-1)),               # 多余维
            dict(raw_action=raw.to(torch.float64)),
            dict(raw_action=[[0.0] * JOINTS] * 2),            # 非 tensor
            dict(q_ref=torch.zeros(3, JOINTS)),               # 批量不一致
            dict(joint_pos=torch.full((2, JOINTS), float("nan"))),
            dict(joint_vel=torch.full((2, JOINTS), float("inf"))),
            dict(raw_action=nan_raw),
            dict(raw_action=torch.zeros(0, JOINTS)),          # 空批
            dict(joint_vel=torch.zeros(2, JOINTS, dtype=torch.float32, device="meta")),
        ]
        for i, case in enumerate(cases):
            with self.subTest(case=i):
                self.assertRaises(ContractError, bad, **case)


# ---------------------------------------------------------------------------
# 接触计数（contacts.py）
# ---------------------------------------------------------------------------

class PerContactQualificationTest(unittest.TestCase):
    """逐接触 distance≤0 且 ‖force‖>0.1 的判定与边界。"""

    def test_qualification_boundaries(self):
        """distance 0/负距/正距 × force 范数恰 0.1/刚大于 0.1/零力/小力。"""
        cases = [
            # (distance, force, 期望合格)
            (0.0, [0.1, 0.0, 0.0], False),          # 范数恰为 float32(0.1)：严格 > 排除
            (0.0, [0.1000001, 0.0, 0.0], True),    # 刚大于 0.1
            (0.0, [-0.1000001, 0.0, 0.0], True),   # 负方向同样按范数判定
            (-0.000001, [0.07, 0.07, 0.0], False),  # 距离合格但范数≈0.099<0.1
            (-0.000001, [0.2, 0.0, 0.0], True),
            (0.000001, [30.0, 0.0, 0.0], False),   # 正距：力再大也不合格
            (0.0, [0.0, 0.0, 0.0], False),         # 零力不合格
            (0.0, [1.0, 1.0, 1.0], True),          # 范数 √3≈1.73
        ]
        contacts = make_contacts(
            env_ids=[0],
            rows=[0] * len(cases),
            substeps=[0] * len(cases),
            pairs=[(7, GROUND)] * len(cases),
            distances=[c[0] for c in cases],
            forces=[c[1] for c in cases],
        )
        result = run_count(contacts)
        expected = torch.tensor([c[2] for c in cases], dtype=torch.bool)
        self.assertTrue(torch.equal(result.qualified, expected))
        # 全部为非足机器人 body × 外部：分类一致；合格者同 body 同子步 → 计 1。
        self.assertTrue(torch.equal(
            result.categories,
            torch.full((len(cases),), CONTACT_CATEGORY_NONFOOT_ENV, dtype=torch.int64),
        ))
        self.assertTrue(torch.equal(result.env_collision_counts_per_substep,
                                    torch.tensor([[1, 0, 0, 0]], dtype=torch.int64)))
        self.assertTrue(torch.equal(result.env_collision_counts,
                                    torch.tensor([0.25], dtype=torch.float32)))
        self.assertTrue(torch.equal(result.self_collision_counts,
                                    torch.tensor([0.0], dtype=torch.float32)))

    def test_env_dedup_by_body_opposing_forces_and_duplicate_points(self):
        """同 body 多接触点/方向相反力：逐条判定、按 body 去重，不先求和。"""
        contacts = make_contacts(
            env_ids=[0, 1],
            rows=[0, 0, 0, 0],
            substeps=[0, 0, 0, 0],
            pairs=[(7, GROUND), (7, GROUND), (8, GROUND), (8, GROUND)],
            forces=[[3.0, 0.0, 0.0], [-3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 2.0, 0.0]],
        )
        result = run_count(contacts)
        # 先求和再过滤会漏掉 body 7（+3x 与 −3x 相消）；不去重会得到 4。
        self.assertTrue(torch.equal(result.env_collision_counts_per_substep,
                                    torch.tensor([[2, 0, 0, 0], [0, 0, 0, 0]], dtype=torch.int64)))
        self.assertTrue(torch.equal(result.env_collision_counts,
                                    torch.tensor([0.5, 0.0], dtype=torch.float32)))

    def test_self_collision_unordered_pairs_including_feet(self):
        """自碰撞按无序 body 对去重；(a,b)/(b,a) 同对；含足自碰撞不豁免。"""
        contacts = make_contacts(
            env_ids=[0],
            rows=[0] * 4,
            substeps=[0, 0, 0, 0],
            pairs=[(0, 7), (7, 0), (1, 0), (1, GROUND)],   # 足 1 × 躯干 0 是自碰撞
            forces=[[1.0, 0.0, 0.0], [0.0, -5.0, 0.0], [2.0, 0.0, 0.0], [0.5, 0.5, 30.0]],
        )
        result = run_count(contacts)
        # 子步 0 的自碰撞无序对：{0,7}（两条方向相反记录同对）与 {0,1} → 2。
        self.assertTrue(torch.equal(result.self_collision_counts_per_substep,
                                    torch.tensor([[2, 0, 0, 0]], dtype=torch.int64)))
        self.assertTrue(torch.equal(result.self_collision_counts,
                                    torch.tensor([0.5], dtype=torch.float32)))
        # 足-地面支撑（含大竖直力）不进入非足环境计数，也不影响自碰撞。
        self.assertTrue(torch.equal(result.env_collision_counts_per_substep,
                                    torch.tensor([[0, 0, 0, 0]], dtype=torch.int64)))
        expected_categories = torch.tensor(
            [CONTACT_CATEGORY_SELF, CONTACT_CATEGORY_SELF, CONTACT_CATEGORY_SELF,
             CONTACT_CATEGORY_FOOT_ENV],
            dtype=torch.int64,
        )
        self.assertTrue(torch.equal(result.categories, expected_categories))


class MembershipTest(unittest.TestCase):
    """机器人/外部归属完全由调用方集合决定。"""

    def test_hip_and_radar_follow_caller_set_and_unknown_bodies_are_external(self):
        """hip/雷达入集合则计数；移出即变外部；外部×外部与机器人无关。"""
        contacts = make_contacts(
            env_ids=[0],
            rows=[0] * 4,
            substeps=[0, 0, 1, 2],
            pairs=[(7, GROUND), (13, GROUND), (1, 13), (GROUND, WALL)],
            forces=[[3.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.2, 0.1, 5.0], [9.0, 0.0, 0.0]],
        )
        # ① 雷达 13 在机器人集合：sub0 {7,13} → 2；(1,13) 为自碰撞对 {1,13}。
        with_radar = run_count(contacts)
        self.assertTrue(torch.equal(with_radar.env_collision_counts_per_substep,
                                    torch.tensor([[2, 0, 0, 0]], dtype=torch.int64)))
        self.assertTrue(torch.equal(with_radar.env_collision_counts,
                                    torch.tensor([0.5], dtype=torch.float32)))
        self.assertTrue(torch.equal(with_radar.self_collision_counts_per_substep,
                                    torch.tensor([[0, 1, 0, 0]], dtype=torch.int64)))
        self.assertEqual(int(with_radar.categories[3]), CONTACT_CATEGORY_EXTERNAL_ONLY)
        self.assertEqual(int(with_radar.categories[2]), CONTACT_CATEGORY_SELF)
        # ② 雷达 13 移出机器人集合：13 变外部——(13,GROUND) 为外部×外部不计，
        #    (1,13) 变足×外部（FOOT_ENV，不入非足环境计数），不把全部接触算机器人。
        without_radar = run_count(contacts, robot=[0] + FEET + [7, 8])
        self.assertTrue(torch.equal(without_radar.env_collision_counts_per_substep,
                                    torch.tensor([[1, 0, 0, 0]], dtype=torch.int64)))
        self.assertTrue(torch.equal(without_radar.env_collision_counts,
                                    torch.tensor([0.25], dtype=torch.float32)))
        self.assertTrue(torch.equal(without_radar.self_collision_counts,
                                    torch.tensor([0.0], dtype=torch.float32)))
        self.assertEqual(int(without_radar.categories[1]), CONTACT_CATEGORY_EXTERNAL_ONLY)
        self.assertEqual(int(without_radar.categories[2]), CONTACT_CATEGORY_FOOT_ENV)

    def test_foot_membership_decides_foot_env_category(self):
        """足身份由 foot 集合决定：同一 body 移出足集即按非足环境计数。"""
        body = 3
        contacts = make_contacts(
            env_ids=[0], rows=[0], substeps=[0], pairs=[(body, GROUND)],
            forces=[[2.0, 0.0, 0.0]],
        )
        as_foot = run_count(contacts)
        self.assertEqual(int(as_foot.categories[0]), CONTACT_CATEGORY_FOOT_ENV)
        self.assertTrue(torch.equal(as_foot.env_collision_counts,
                                    torch.tensor([0.0], dtype=torch.float32)))
        # 合成夹具：把 body 3 移出足集、body 7(hip) 顶替为“足”——足身份只由
        # 集合判定，body 3 随即按非足环境接触计数。
        as_nonfoot = run_count(contacts, feet=[1, 2, 4, 5, 6, 7])
        self.assertEqual(int(as_nonfoot.categories[0]), CONTACT_CATEGORY_NONFOOT_ENV)
        self.assertTrue(torch.equal(as_nonfoot.env_collision_counts,
                                    torch.tensor([0.25], dtype=torch.float32)))


class SubstepAveragingTest(unittest.TestCase):
    """每控制步恰 4 个物理子步：先各子步去重计数再平均；持续接触不叠乘。"""

    def test_counts_average_over_exactly_four_substeps(self):
        """仅 1/4 子步合格 → 0.25；不同子步不同 body → (1,2,0,0)→0.75。"""
        contacts = make_contacts(
            env_ids=[0, 1],
            rows=[0, 1, 1, 1, 1],
            substeps=[2, 0, 1, 1, 2],
            pairs=[(7, GROUND), (7, GROUND), (7, GROUND), (8, GROUND), (7, GROUND)],
            distances=[-0.01, -0.01, -0.01, -0.01, 0.001],   # 行 4 正距：sub2 不合格
        )
        result = run_count(contacts)
        self.assertTrue(torch.equal(
            result.env_collision_counts_per_substep,
            torch.tensor([[0, 0, 1, 0], [1, 2, 0, 0]], dtype=torch.int64),
        ))
        self.assertTrue(torch.equal(result.env_collision_counts,
                                    torch.tensor([0.25, 0.75], dtype=torch.float32)))
        # 平均值恰为各子步去重计数之和 / 4（持续整个控制步的接触计 1.0 而非 4.0）。
        self.assertTrue(torch.equal(
            result.env_collision_counts,
            result.env_collision_counts_per_substep.sum(dim=1).to(torch.float32) / 4,
        ))
        self.assertEqual(result.env_collision_counts.dtype, torch.float32)
        self.assertEqual(result.env_collision_counts_per_substep.dtype, torch.int64)

    def test_sustained_contact_counts_each_control_step_without_accumulation(self):
        """同一接触内容重复控制步：每步计数相同，无跨调用累计，输入不变。"""
        contacts = make_contacts(
            env_ids=[4],
            rows=[0, 0],
            substeps=[0, 0],
            pairs=[(0, 7), (13, GROUND)],
        )
        snapshot = {name: getattr(contacts, name).clone()
                    for name in ("env_ids", "contact_env_rows", "substeps", "body_a", "body_b",
                                 "distance", "force")}
        first = run_count(contacts)
        second = run_count(contacts)   # 下一控制步、同样的持续接触
        self.assertIsInstance(first, ContactControlStepResult)
        for name in ("env_collision_counts", "self_collision_counts",
                     "env_collision_counts_per_substep", "self_collision_counts_per_substep",
                     "qualified", "categories"):
            self.assertTrue(torch.equal(getattr(first, name), getattr(second, name)), name)
        self.assertTrue(torch.equal(first.env_collision_counts,
                                    torch.tensor([0.25], dtype=torch.float32)))
        self.assertTrue(torch.equal(first.self_collision_counts,
                                    torch.tensor([0.25], dtype=torch.float32)))
        for name, tensor in snapshot.items():
            self.assertTrue(torch.equal(getattr(contacts, name), tensor), name)

    def test_self_pair_sustained_across_all_substeps_counts_once(self):
        """全 4 子步持续的自碰撞对：平均恰 1.0（不乘子步数）。"""
        contacts = make_contacts(
            env_ids=[0],
            rows=[0] * 4,
            substeps=[0, 1, 2, 3],
            pairs=[(2, 8)] * 4,   # 足 2 × 腿 8：含足自碰撞对，不豁免
        )
        result = run_count(contacts)
        self.assertTrue(torch.equal(result.self_collision_counts_per_substep,
                                    torch.tensor([[1, 1, 1, 1]], dtype=torch.int64)))
        self.assertTrue(torch.equal(result.self_collision_counts,
                                    torch.tensor([1.0], dtype=torch.float32)))
        self.assertTrue(torch.equal(result.env_collision_counts,
                                    torch.tensor([0.0], dtype=torch.float32)))


class BatchOrderTest(unittest.TestCase):
    """非连续环境身份与批次顺序。"""

    def test_noncontiguous_env_ids_use_batch_rows_not_env_ids(self):
        """接触按批次行号关联（非环境 id 数值）；返回顺序与输入 env_ids 一致。"""
        contacts = make_contacts(
            env_ids=[100, 7],
            rows=[1, 1, 0],
            substeps=[0, 1, 3],
            pairs=[(7, GROUND), (7, GROUND), (8, GROUND)],
        )
        result = run_count(contacts)
        self.assertTrue(torch.equal(result.env_ids, torch.tensor([100, 7], dtype=torch.int64)))
        self.assertTrue(torch.equal(
            result.env_collision_counts_per_substep,
            torch.tensor([[0, 0, 0, 1], [1, 1, 0, 0]], dtype=torch.int64),
        ))
        self.assertTrue(torch.equal(result.env_collision_counts,
                                    torch.tensor([0.25, 0.5], dtype=torch.float32)))

    def test_zero_contacts_batch_is_representable(self):
        """C=0：全部计数为零，诊断字段为合法空张量。"""
        contacts = make_contacts(env_ids=[3, 9], rows=[], substeps=[], pairs=[])
        result = run_count(contacts)
        self.assertTrue(torch.equal(result.env_collision_counts, torch.zeros(2, dtype=torch.float32)))
        self.assertTrue(torch.equal(result.self_collision_counts, torch.zeros(2, dtype=torch.float32)))
        self.assertTrue(torch.equal(result.env_collision_counts_per_substep,
                                    torch.zeros(2, PHYSICS_SUBSTEPS_PER_CONTROL_STEP, dtype=torch.int64)))
        self.assertTrue(torch.equal(result.self_collision_counts_per_substep,
                                    torch.zeros(2, PHYSICS_SUBSTEPS_PER_CONTROL_STEP, dtype=torch.int64)))
        self.assertEqual(tuple(result.qualified.shape), (0,))
        self.assertEqual(result.qualified.dtype, torch.bool)
        self.assertEqual(tuple(result.categories.shape), (0,))
        self.assertEqual(result.categories.dtype, torch.int64)


class ContactContractTest(unittest.TestCase):
    """ControlStepContacts 与成员集合的契约边界。"""

    def test_contact_record_rejections(self):
        """逐接触记录的形状/dtype/范围/有限性/设备违约一律 ContractError。"""

        def build(**overrides):
            kwargs = dict(
                env_ids=torch.tensor([0, 1], dtype=torch.int64),
                contact_env_rows=torch.tensor([0, 0], dtype=torch.int64),
                substeps=torch.tensor([0, 1], dtype=torch.int64),
                body_a=torch.tensor([7, 8], dtype=torch.int64),
                body_b=torch.tensor([GROUND, GROUND], dtype=torch.int64),
                distance=torch.tensor([-0.01, -0.01], dtype=torch.float32),
                force=torch.tensor([[3.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float32),
            )
            kwargs.update(overrides)
            ControlStepContacts(**kwargs)

        cases = [
            dict(substeps=torch.tensor([0, PHYSICS_SUBSTEPS_PER_CONTROL_STEP], dtype=torch.int64)),
            dict(substeps=torch.tensor([-1, 1], dtype=torch.int64)),
            dict(body_a=torch.tensor([7, 8], dtype=torch.int64),
                 body_b=torch.tensor([7, GROUND], dtype=torch.int64)),          # 同 body 记录
            dict(contact_env_rows=torch.tensor([0, 2], dtype=torch.int64)),     # 行号越界（=B）
            dict(contact_env_rows=torch.tensor([0, -1], dtype=torch.int64)),
            dict(env_ids=torch.tensor([5, 5], dtype=torch.int64)),              # 环境重复
            dict(env_ids=torch.tensor([-1, 1], dtype=torch.int64)),
            dict(body_a=torch.tensor([-7, 8], dtype=torch.int64)),              # 负 body id
            dict(contact_env_rows=torch.tensor([0], dtype=torch.int64)),        # C 不一致
            dict(substeps=torch.tensor([0, 1], dtype=torch.float32)),           # dtype
            dict(distance=torch.tensor([-0.01, 0.01], dtype=torch.float64)),
            dict(distance=torch.tensor([float("nan"), -0.01], dtype=torch.float32)),
            dict(force=torch.tensor([[float("nan"), 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32)),
            dict(force=torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)),   # (C,2)
            dict(force=torch.zeros(2, 3, dtype=torch.float32, device="meta")),  # 设备不一致
        ]
        for i, case in enumerate(cases):
            with self.subTest(case=i):
                self.assertRaises(ContractError, build, **case)

    def test_membership_rejections(self):
        """机器人/足集合的形状/dtype/唯一性/子集/设备违约一律 ContractError。"""
        contacts = make_contacts(env_ids=[0], rows=[0], substeps=[0], pairs=[(7, GROUND)])
        robot = torch.tensor(ROBOT_BODIES, dtype=torch.int64)
        cases = [
            # 足数正确但含非机器人 body：子集违约。
            dict(robot_body_ids=torch.tensor([0] + FEET + [7, 8], dtype=torch.int64),
                 foot_body_ids=torch.tensor([1, 2, 3, 4, 5, 13], dtype=torch.int64)),
            dict(robot_body_ids=robot, foot_body_ids=torch.tensor(FEET[:5], dtype=torch.int64)),
            dict(robot_body_ids=robot,
                 foot_body_ids=torch.tensor([1, 1, 2, 3, 4, 5], dtype=torch.int64)),   # 足重复
            dict(robot_body_ids=torch.tensor([0, 0] + FEET + [7, 8, 13], dtype=torch.int64),  # 机器人重复
                 foot_body_ids=torch.tensor(FEET, dtype=torch.int64)),
            dict(robot_body_ids=torch.zeros(0, dtype=torch.int64),                    # 机器人集合为空
                 foot_body_ids=torch.tensor(FEET, dtype=torch.int64)),
            dict(robot_body_ids=robot, foot_body_ids=torch.tensor(FEET, dtype=torch.float32)),
            dict(robot_body_ids=robot,
                 foot_body_ids=torch.tensor(FEET, dtype=torch.int64, device="meta")),  # 设备
        ]
        for i, case in enumerate(cases):
            with self.subTest(case=i):
                self.assertRaises(ContractError, count_control_step_contacts,
                                  contacts, **case)


class FeetStumbleTest(unittest.TestCase):
    """旧足端绊碰：世界坐标净足力的 0/1 力比判定。"""

    def test_stumble_ratio_rules(self):
        forces = torch.zeros(5, FOOT_COUNT, 3, dtype=torch.float32)
        # env0：两足触发 → 仍记 1（不按足数累计）。
        forces[0, 0] = torch.tensor([6.0, 0.0, 1.0])     # 6 > 5×1
        forces[0, 3] = torch.tensor([12.0, 0.0, 2.0])    # 12 > 5×2
        # env1：所有足恰等于比值（5 = 5×1）→ 严格 > 不触发。
        forces[1, :] = torch.tensor([5.0, 0.0, 1.0])
        # env2：水平净力 0.05 < 0.1 N、Fz=0 → 无 0.1 门限，仍按比值触发。
        forces[2, 2] = torch.tensor([0.05, 0.0, 0.0])
        # env3：0.05 水平 + 0.001 竖直 → 0.05 > 0.005 触发。
        forces[3, 5] = torch.tensor([0.05, 0.0, 0.001])
        # env4：竖直支撑为主（0.1 > 50 为假）+ 其余零力 → 0。
        forces[4, 0] = torch.tensor([0.1, 0.0, 10.0])
        flag = feet_stumble_flag(forces)
        self.assertEqual(flag.dtype, torch.float32)
        self.assertTrue(torch.equal(flag, torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0], dtype=torch.float32)))
        self.assertTrue(bool(((flag == 0.0) | (flag == 1.0)).all()))

    def test_stumble_is_not_four_substep_averaged_nor_accumulating(self):
        """单控制步单次判定：同一输入重复调用结果不变（无子步平均、无状态）。"""
        forces = torch.zeros(1, FOOT_COUNT, 3, dtype=torch.float32)
        forces[0, 1] = torch.tensor([0.0, 7.5, 1.0])     # 7.5 > 5
        first = feet_stumble_flag(forces)
        second = feet_stumble_flag(forces)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(first, torch.tensor([1.0], dtype=torch.float32)))

    def test_stumble_input_rejections(self):
        good = torch.zeros(2, FOOT_COUNT, 3, dtype=torch.float32)
        feet_stumble_flag(good)
        cases = [
            torch.zeros(2, 5, 3, dtype=torch.float32),                    # 足数 5
            torch.zeros(2, FOOT_COUNT, 2, dtype=torch.float32),
            torch.zeros(FOOT_COUNT, 3, dtype=torch.float32),              # 缺批量维
            torch.zeros(2, FOOT_COUNT, 3, dtype=torch.float64),
            torch.full((2, FOOT_COUNT, 3), float("nan")),
            [[0.0] * 3] * FOOT_COUNT,                                     # 非 tensor
        ]
        for i, case in enumerate(cases):
            with self.subTest(case=i):
                self.assertRaises(ContractError, feet_stumble_flag, case)


class ModulePurityTest(unittest.TestCase):
    """模块边界：仅标准库 + torch + 本包 contracts；不导出奖励/reset 逻辑。"""

    def test_modules_depend_only_on_stdlib_and_torch(self):
        import hexapod_mjlab.rl.joint_passage.contacts as contacts_module
        import hexapod_mjlab.rl.joint_passage.control as control_module

        allowed_top = {"torch", "dataclasses", "__future__", "math"}
        for module in (control_module, contacts_module):
            tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertIn(alias.name.split(".")[0], allowed_top,
                                      f"{module.__name__} 导入 {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    if node.level:   # 相对导入仅允许 .contracts（F1 契约）
                        self.assertEqual(node.module, "contracts", module.__name__)
                    else:
                        self.assertIn((node.module or "").split(".")[0], allowed_top,
                                      module.__name__)

    def test_no_reward_reset_or_weight_symbols_exported(self):
        import hexapod_mjlab.rl.joint_passage.contacts as contacts_module
        import hexapod_mjlab.rl.joint_passage.control as control_module

        for module in (control_module, contacts_module):
            for name in vars(module):
                lowered = name.lower()
                for keyword in ("reward", "weight", "reset", "terminat", "curriculum", "done"):
                    self.assertNotIn(keyword, lowered,
                                     f"{module.__name__}.{name} 不应出现在执行/接触纯计算模块")


if __name__ == "__main__":
    unittest.main()
