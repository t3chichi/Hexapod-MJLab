#!/usr/bin/env python
"""语义修正轮自检：三类实现缺陷修复的工程数值正确性（合成输入，CPU，unittest）。

对应 Codex 实际反例（logs/joint_passage/framework/20261005_semantic_audit/
counterexamples_initial.json 与 before/ 原版本，均不改动）：
① EnvelopeSoftplusMapping：极负 raw 下 softplus float32 下溢使半宽退化为 0、
   float32 高度消减（Δz=p_height−h_ref）加回参考重建为 0 高度、有限尺度乘法
   上溢为 Inf、p0/h_ref buffer 被错误重载——修复后一律 ContractError 拒绝
   （不裁剪/不设 floor/不加 EA2 范围、最小尺寸或 solver）；正常区间输出/梯度、
   raw/residual 等价与高度还原、极负端梯度衰减（经 positive_quantities 合法
   观察）保持；
② compute_pd_torques：有限输入（joint_pos=−maxfloat、joint_vel=maxfloat）产生
   Inf−Inf=NaN 的力矩——修复后立即 ContractError；普通 ±80 饱和与
   ±100→±25 残差目标不变；
③ flatten_far_window：已填充槽内 false 点的元数据按 far_valid 逐点掩码为
   scan_id=−1、time=0，有效点保留所属扫描身份；maxrange 无命中端点
   （valid=True）保留扫描身份；坐标点与远端扫描窗口内容不变。

期望值全部由本文件用纯 Python 算术独立解析（math/struct 的 float64 计算 +
struct 的 float32 舍入模拟、sorted 稳定排序参考），不调用被测模块内部路径
生成期望。这些是工程数值正确性检查：只验证"拒绝非有限/非正的退化数值、
保持正常数值与已确认语义"，不选择 loss/预算，不新增合法值域、阈值、EA2
范围、clamp、最小尺寸、solver、奖励/reset/动作分布，也不决定真实运行时
异常如何 reset。本检查由 GLM 编写，本机 headless 无审批客户端未运行；
由 Codex 在交付后执行，失败以非零退出传播。合成夹具不代表任务默认配置，
不含真实训练结论。
"""

import math
import struct
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import torch

from hexapod_mjlab.rl.joint_passage.contracts import (
    EMPTY_SCAN_ID,
    ENVELOPE_DIM,
    FAR_POINTS_PER_SCAN,
    FAR_WINDOW_SLOTS,
    JOINT_DIM,
    NEAR_POINTS,
    ContractError,
    ScanWindowSnapshot,
    flatten_far_window,
    sort_points_by_angle_keys,
)
from hexapod_mjlab.rl.joint_passage.control import compute_pd_torques
from hexapod_mjlab.rl.joint_passage.mapping import EnvelopeSoftplusMapping

# 已确认启动值（docs/tasks/confined_locomotion/平地学习与基准测量.md 第 2 节；
# 测试内独立硬编码，不引自被测模块常量）。
P0_DOC = (0.59, 0.69, 0.59, 0.89, 0.89, 0.46)
H_REF_DOC = 0.46
# IEEE float32 最大有限值（2^128−2^103，独立写出）。
F32_MAX = 3.4028234663852886e+38


def f32(value):
    """IEEE float32 舍入的独立模拟（struct 实现，不经 torch/被测模块）。"""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def reference_pd(raw, q_ref, q, qdot):
    """执行链规则的纯 Python 解析：clip±100 → q_ref+0.25·clip → 130·Δq−2·q̇ → clip±80。"""
    clipped = max(-100.0, min(100.0, float(raw)))
    target = float(q_ref) + 0.25 * clipped
    unclipped = 130.0 * (target - float(q)) - 2.0 * float(qdot)
    applied = max(-80.0, min(80.0, unclipped))
    return target, unclipped, applied, abs(unclipped) > 80.0


def snapshot_with_scans(scans):
    """B=1 合法快照：scans=[(scan_id, time_s, [有效点坐标列表])] 前缀填充远端槽，
    近端槽与最新已填充扫描同步（取其前两个坐标为近端有效点）。"""
    far_points = torch.zeros(1, FAR_WINDOW_SLOTS, FAR_POINTS_PER_SCAN, 3)
    far_valid = torch.zeros(1, FAR_WINDOW_SLOTS, FAR_POINTS_PER_SCAN, dtype=torch.bool)
    far_scan_ids = torch.full((1, FAR_WINDOW_SLOTS), EMPTY_SCAN_ID, dtype=torch.int64)
    far_times = torch.zeros(1, FAR_WINDOW_SLOTS, dtype=torch.float32)
    for slot, (scan_id, time_s, coords) in enumerate(scans):
        far_scan_ids[0, slot] = scan_id
        far_times[0, slot] = time_s
        for k, coord in enumerate(coords):
            far_points[0, slot, k] = torch.tensor(coord)
            far_valid[0, slot, k] = True
    newest_id, newest_time, newest_coords = scans[-1]
    near_points = torch.zeros(1, NEAR_POINTS, 3)
    near_valid = torch.zeros(1, NEAR_POINTS, dtype=torch.bool)
    for k, coord in enumerate(newest_coords[:2]):
        near_points[0, k] = torch.tensor(coord)
        near_valid[0, k] = True
    return ScanWindowSnapshot(
        env_ids=torch.tensor([0], dtype=torch.int64),
        episode_ids=torch.tensor([11], dtype=torch.int64),
        near_points=near_points,
        near_valid=near_valid,
        near_filled=torch.tensor([True]),
        near_scan_ids=torch.tensor([newest_id], dtype=torch.int64),
        near_times=torch.tensor([newest_time], dtype=torch.float32),
        far_points=far_points,
        far_valid=far_valid,
        far_filled=torch.tensor(
            [[True] * len(scans) + [False] * (FAR_WINDOW_SLOTS - len(scans))]
        ),
        far_scan_ids=far_scan_ids,
        far_times=far_times,
    )


class MappingRejectionSemanticsTest(unittest.TestCase):
    """Softplus 映射的退化数值拒绝（反例驱动；只拒绝，不裁剪/不设 floor）。"""

    def test_extreme_negative_raw_width_degeneration_rejected(self):
        """极负 raw 宽度退化拒绝：softplus(-110) 在 float32 下溢为 0 ⇒ 半宽=0。"""
        mapping = EnvelopeSoftplusMapping(mode="raw")
        head = torch.full((1, ENVELOPE_DIM), -110.0)
        # 独立解析：exp(-110)≈1.7e-48 低于 float32 最小次正规数（≈1.4e-45），
        # float32 下舍入为 0 ⇒ softplus=log1p(0)=0 ⇒ p=p0·0=0（宽度退化）。
        self.assertEqual(f32(math.exp(-110.0)), 0.0)
        with self.assertRaisesRegex(ContractError, "正量"):
            mapping(head)                      # forward 拒绝
        with self.assertRaisesRegex(ContractError, "正量"):
            mapping.positive_quantities(head)  # 调试接口共享同一路径，不得绕过

    def test_zero_reconstructed_height_rejected(self):
        """减/加高度参考导致零重建高度拒绝：raw=-30 时 p_height 微正但完全消减。"""
        mapping = EnvelopeSoftplusMapping(mode="raw")
        # 独立解析（float64 计算 + float32 舍入模拟）：
        p_height = f32(H_REF_DOC * math.log1p(math.exp(-30.0)))    # ≈4.3e-14，微正
        delta_z = f32(p_height - f32(H_REF_DOC))                   # 消减为 −0.46f
        reconstructed = f32(f32(H_REF_DOC) + delta_z)              # 加回参考得 0
        self.assertGreater(p_height, 0.0)      # 正量本身合法（微正且有限）
        self.assertEqual(reconstructed, 0.0)   # 重建高度为 0：几何非法
        with self.assertRaisesRegex(ContractError, "重建"):
            mapping(torch.full((1, ENVELOPE_DIM), -30.0))
        # 正量微正且有限：调试接口按共享正量检查放行（高度语义仅在 forward 检查）
        positives = mapping.positive_quantities(torch.full((1, ENVELOPE_DIM), -30.0))
        self.assertTrue(bool((positives > 0).all()))
        self.assertTrue(bool(torch.isfinite(positives).all()))

    def test_finite_scale_multiplication_overflow_rejected(self):
        """有限尺度乘法溢出拒绝：raw=maxfloat、p0=2 ⇒ 2·max 超出 float32 → Inf。"""
        # 独立解析：2·f32_max 已超出 float32 表示范围，乘积必然上溢为 Inf。
        self.assertGreater(2.0 * F32_MAX, F32_MAX)
        overflow = EnvelopeSoftplusMapping(
            mode="raw", p0=(2.0, 2.0, 2.0, 2.0, 2.0, 2.0), h_ref=H_REF_DOC
        )
        head = torch.full((1, ENVELOPE_DIM), torch.finfo(torch.float32).max)
        with self.assertRaisesRegex(ContractError, "正量"):
            overflow(head)
        with self.assertRaisesRegex(ContractError, "正量"):
            overflow.positive_quantities(head)
        # 对照：默认 p0<1 时 maxfloat 经 softplus 阈值分支原样通过，乘积仍有限
        # （只拒绝非有限/非正，不新增任何值域范围/上限）。
        default = EnvelopeSoftplusMapping(mode="raw")
        envelope = default(head)
        self.assertTrue(bool(torch.isfinite(envelope).all()))
        self.assertTrue(bool((default.positive_quantities(head) > 0).all()))

    def test_illegal_buffer_reload_rejected_by_result_checks(self):
        """buffer 被错误 checkpoint 重载为非法值：由 forward 结果检查拒绝。"""
        head = torch.zeros(1, ENVELOPE_DIM)
        cases = {
            "p0_negative": {"p0": -torch.ones(ENVELOPE_DIM)},
            "p0_zero": {"p0": torch.zeros(ENVELOPE_DIM)},
            "p0_inf": {"p0": torch.full((ENVELOPE_DIM,), float("inf"))},
            "h_ref_inf": {"h_ref": torch.tensor(float("inf"))},
        }
        for name, override in cases.items():
            with self.subTest(case=name):
                mapping = EnvelopeSoftplusMapping(mode="raw")
                state = mapping.state_dict()
                state.update(override)
                mapping.load_state_dict(state)
                with self.assertRaises(ContractError):
                    mapping(head)


class MappingNormalRangeSemanticsTest(unittest.TestCase):
    """正常区间不变：初始化基准、上端微扰梯度不饱和、raw/residual 等价与高度还原。"""

    def test_startup_baseline_and_upper_perturbation_gradient(self):
        """正常初始化（两模式 Ê=p0 基准）与上端微扰梯度不饱和（=p0·sigmoid）。"""
        b = math.log(math.e - 1.0)   # softplus⁻¹(1)，独立由 math 计算
        self.assertAlmostEqual(b, 0.5413248546, places=9)
        for mode, head in (
            ("raw", torch.full((1, ENVELOPE_DIM), b)),
            ("residual", torch.zeros(1, ENVELOPE_DIM)),
        ):
            with self.subTest(mode=mode):
                envelope = EnvelopeSoftplusMapping(mode=mode)(head)
                for value, expected in zip(
                    envelope[0].tolist(), (0.59, 0.69, 0.59, 0.89, -0.89, 0.0)
                ):
                    self.assertAlmostEqual(value, expected, delta=1e-6)
        # 梯度独立解析：dÊ_j/draw_j = ±p0_j·sigmoid(raw_j)（后缘维反向）。
        mapping = EnvelopeSoftplusMapping(mode="raw")
        signs = (1.0, 1.0, 1.0, 1.0, -1.0, 1.0)
        for offset in (0.0, 2.0):   # 初始化点 b 与正常上端 b+2
            with self.subTest(offset=offset):
                head = torch.full((1, ENVELOPE_DIM), b + offset, requires_grad=True)
                mapping(head).sum().backward()
                for j, p0_j in enumerate(P0_DOC):
                    expected = signs[j] * p0_j / (1.0 + math.exp(-(b + offset)))
                    self.assertAlmostEqual(float(head.grad[0, j]), expected, delta=1e-6)
                    # 不饱和：初始点与上端梯度均显著非零（无死区、无截断）
                    self.assertGreater(abs(float(head.grad[0, j])), 0.2)

    def test_raw_residual_equivalence_and_height_restoration(self):
        """正常 raw/residual 等价（b 恰加一次）与 h_ref+Δz 高度还原。"""
        b = math.log(math.e - 1.0)
        raw_mapping = EnvelopeSoftplusMapping(mode="raw")
        residual_mapping = EnvelopeSoftplusMapping(mode="residual")
        grid = [-1.0, -0.25, 0.0, 0.75, 1.25]
        head = torch.tensor([[value] * ENVELOPE_DIM for value in grid], dtype=torch.float32)
        # 残差侧恰加一次 b：residual(u) ≡ raw(u+b)（同算术路径，逐位一致）
        self.assertTrue(torch.equal(residual_mapping(head), raw_mapping(head + b)))
        envelope = raw_mapping(head)
        expected_rows = []
        for value in grid:
            sp = math.log1p(math.exp(value))
            p = [f32(scale * sp) for scale in P0_DOC]
            expected_rows.append(
                [p[0], p[1], p[2], p[3], -p[4], f32(p[5] - f32(H_REF_DOC))]
            )
        self.assertTrue(
            torch.allclose(envelope, torch.tensor(expected_rows), rtol=0.0, atol=1e-6)
        )
        # 高度还原：h_ref+Δz 在正常区间还原独立解析的 p_height（消减误差≪量级）
        for value, row in zip(grid, envelope):
            p_height = f32(P0_DOC[5] * math.log1p(math.exp(value)))
            restored = f32(f32(H_REF_DOC) + float(row[5]))
            self.assertAlmostEqual(restored, p_height, delta=1e-6)
            self.assertGreater(restored, 0.0)

    def test_negative_end_gradient_decay_survives_via_debug_path(self):
        """极负端梯度衰减仍存在：经 positive_quantities 合法观察（无 clamp/floor 痕迹）。"""
        b = math.log(math.e - 1.0)
        mapping = EnvelopeSoftplusMapping(mode="raw")
        low = torch.full((1, ENVELOPE_DIM), b - 40.0, requires_grad=True)
        mapping.positive_quantities(low).sum().backward()
        mid = torch.full((1, ENVELOPE_DIM), b, requires_grad=True)
        mapping.positive_quantities(mid).sum().backward()
        self.assertLess(float(low.grad.abs().max()), 1e-10)
        self.assertLess(float(low.grad.abs().max()), float(mid.grad.abs().max()))
        # 同一输入的前向因高度重建为精确 0 被拒绝（与 raw=−30 反例同一算术形态：
        # b−40≈−39.46 更负，p_height≈3.4e-18 同样远小于 ulp(0.46)，完全消减）。
        p_height = f32(P0_DOC[5] * math.log1p(math.exp(b - 40.0)))
        self.assertEqual(f32(f32(H_REF_DOC) + f32(p_height - f32(H_REF_DOC))), 0.0)
        with self.assertRaisesRegex(ContractError, "重建"):
            mapping(low.detach())


class PDTorqueSemanticsTest(unittest.TestCase):
    """PD 执行链：有限输入的非有限计算拒绝；普通饱和/残差目标不变。"""

    def test_finite_inputs_producing_nonfinite_computation_rejected(self):
        """joint_pos=−max、joint_vel=max：130·误差与 2·速度分别上溢 → Inf−Inf=NaN。"""
        # 独立解析：130·max 与 2·max 均超出 float32 表示范围（各自上溢为 Inf），
        # 有限减法 Inf−Inf=NaN；力矩 clamp 不修复 NaN。
        self.assertGreater(130.0 * F32_MAX, F32_MAX)
        self.assertGreater(2.0 * F32_MAX, F32_MAX)
        zeros = torch.zeros(1, JOINT_DIM)
        with self.assertRaisesRegex(ContractError, "unclipped_torque"):
            compute_pd_torques(
                raw_action=zeros, q_ref=zeros,
                joint_pos=torch.full((1, JOINT_DIM), -F32_MAX),
                joint_vel=torch.full((1, JOINT_DIM), F32_MAX),
            )
        # 单项上溢形态：误差 0、速度 −max → 0−2·(−max) 上溢为 +Inf，同样拒绝。
        with self.assertRaisesRegex(ContractError, "unclipped_torque"):
            compute_pd_torques(
                raw_action=zeros, q_ref=zeros, joint_pos=zeros,
                joint_vel=torch.full((1, JOINT_DIM), -F32_MAX),
            )
        # 批内单行单关节的非有限计算也拒绝（不因其余行正常而放过）。
        joint_pos = torch.zeros(2, JOINT_DIM)
        joint_vel = torch.zeros(2, JOINT_DIM)
        joint_pos[1, 3] = -F32_MAX
        joint_vel[1, 3] = F32_MAX
        with self.assertRaisesRegex(ContractError, "unclipped_torque"):
            compute_pd_torques(
                raw_action=torch.zeros(2, JOINT_DIM), q_ref=torch.zeros(2, JOINT_DIM),
                joint_pos=joint_pos, joint_vel=joint_vel,
            )

    def test_normal_saturation_and_residual_targets_unchanged(self):
        """普通 ±80 饱和与 ±100→±25 残差目标不因有限性检查改变（dyadic 精确）。"""
        # (raw, q_ref, q, qdot)：期望由 reference_pd 独立解析，全部 dyadic 精确。
        scenarios = [
            (1000000.0, 1.0, 1.0, 0.0),    # clip 100 → target=26 → 3250 饱和
            (-1000000.0, 1.0, 1.0, 0.0),   # target=−24 → −3250 饱和
            (100.0, 1.0, 1.0, 0.0),        # 裁剪边界值本身
            (-100.0, 0.0, 0.0, 0.0),       # q_ref=0 → target=−25（恰 −25 rad）
            (0.0, 1.0, 0.5, -7.5),         # 65+15=80 恰在上限 → 不算饱和
            (0.0, 1.0, 1.5, 7.5),          # −80 恰在下限 → 不算饱和
            (0.0, 1.0, 0.875, 0.5),        # 16.25−1=15.25 界内原样通过
            (99.9375, 1.0, 1.0, 0.0),      # 界内 raw 原样：target=25.984375
        ]
        expected = [reference_pd(*scenario) for scenario in scenarios]
        raw = torch.tensor(
            [[scenarios[j % len(scenarios)][0] for j in range(JOINT_DIM)]],
            dtype=torch.float32,
        )
        q_ref = torch.tensor(
            [[scenarios[j % len(scenarios)][1] for j in range(JOINT_DIM)]],
            dtype=torch.float32,
        )
        q = torch.tensor(
            [[scenarios[j % len(scenarios)][2] for j in range(JOINT_DIM)]],
            dtype=torch.float32,
        )
        qdot = torch.tensor(
            [[scenarios[j % len(scenarios)][3] for j in range(JOINT_DIM)]],
            dtype=torch.float32,
        )
        result = compute_pd_torques(raw_action=raw, q_ref=q_ref, joint_pos=q, joint_vel=qdot)
        for j in range(JOINT_DIM):
            target, unclipped, applied, saturated = expected[j % len(scenarios)]
            self.assertEqual(float(result.joint_target[0, j]), target, f"joint {j}")
            self.assertEqual(float(result.unclipped_torque[0, j]), unclipped, f"joint {j}")
            self.assertEqual(float(result.applied_torque[0, j]), applied, f"joint {j}")
            self.assertEqual(bool(result.torque_saturated[0, j]), saturated, f"joint {j}")
        # ±100 裁剪 → 残差目标恰 ±25 rad（无位置夹紧）；普通饱和仍报 ±80。
        self.assertEqual(float(result.joint_target[0, 0]), 26.0)
        self.assertEqual(float(result.joint_target[0, 3]), -25.0)
        self.assertGreater(float(result.unclipped_torque[0, 0]), 80.0)
        self.assertEqual(float(result.applied_torque[0, 0]), 80.0)
        self.assertTrue(bool(result.torque_saturated[0, 0]))


class FlattenMetadataSemanticsTest(unittest.TestCase):
    """flatten_far_window 每点元数据掩码与 maxrange 无命中端点的扫描身份。"""

    def test_partial_false_points_masked_and_sort_metadata_consistent(self):
        """已填充槽内部分 false 点：元数据 -1/0；排序后元数据与独立排序参考一致。"""
        scans = (
            (5, 0.1, [[1.00, 0.5, 0.0], [1.00, 0.5, 1.0], [1.00, 0.5, 2.0]]),
            (7, 0.2, [[1.01, 0.5, 100.0], [1.01, 0.5, 101.0]]),
        )
        snapshot = snapshot_with_scans(scans)
        flat = flatten_far_window(snapshot)
        # 独立解析期望：flat 索引 = slot·64+k；有效点保留槽 scan/time，
        # false 点（含已填充槽内）为 -1/0；z 通道编码点身份。
        expected_ids, expected_times, expected_z = [], [], []
        for scan_id, time_s, coords in scans:
            for k in range(FAR_POINTS_PER_SCAN):
                valid = k < len(coords)
                expected_ids.append(scan_id if valid else -1)
                expected_times.append(time_s if valid else 0.0)
                expected_z.append(coords[k][2] if valid else 0.0)
        for _ in range((FAR_WINDOW_SLOTS - len(scans)) * FAR_POINTS_PER_SCAN):
            expected_ids.append(-1)
            expected_times.append(0.0)
            expected_z.append(0.0)
        self.assertTrue(
            torch.equal(flat.scan_ids[0], torch.tensor(expected_ids, dtype=torch.int64))
        )
        self.assertTrue(
            torch.equal(flat.times[0], torch.tensor(expected_times, dtype=torch.float32))
        )
        self.assertTrue(
            torch.equal(flat.valid[0], torch.tensor([i != -1 for i in expected_ids]))
        )
        self.assertTrue(
            torch.equal(flat.points[0, :, 2], torch.tensor(expected_z, dtype=torch.float32))
        )
        # 坐标点与远端扫描窗口内容不变：拼接坐标与快照逐位一致，无效点精确 0。
        self.assertTrue(
            torch.equal(
                flat.points[0],
                snapshot.far_points[0].reshape(FAR_WINDOW_SLOTS * FAR_POINTS_PER_SCAN, 3),
            )
        )
        self.assertEqual(float(flat.points[0][~flat.valid[0]].abs().sum()), 0.0)
        # 排序元数据一致性：显式互异角键排序后，元数据/坐标随独立稳定排序参考置换。
        count = FAR_WINDOW_SLOTS * FAR_POINTS_PER_SCAN
        keys = torch.tensor(
            [[((i * 11) % count) + 0.25 for i in range(count)]], dtype=torch.float32
        )
        ordered = sort_points_by_angle_keys(
            flat.points, flat.valid, keys, scan_ids=flat.scan_ids, times=flat.times
        )
        valid_idx = [i for i in range(count) if expected_ids[i] != -1]
        invalid_idx = [i for i in range(count) if expected_ids[i] == -1]
        order = sorted(valid_idx, key=lambda i: float(keys[0, i])) + invalid_idx
        self.assertTrue(
            torch.equal(
                ordered.scan_ids[0],
                torch.tensor([expected_ids[i] for i in order], dtype=torch.int64),
            )
        )
        self.assertTrue(
            torch.equal(
                ordered.times[0],
                torch.tensor([expected_times[i] for i in order], dtype=torch.float32),
            )
        )
        self.assertTrue(
            torch.equal(
                ordered.points[0, :, 2],
                torch.tensor([expected_z[i] for i in order], dtype=torch.float32),
            )
        )
        self.assertTrue(
            torch.equal(
                ordered.valid[0],
                torch.tensor([expected_ids[i] != -1 for i in order]),
            )
        )

    def test_maxrange_no_hit_keeps_valid_and_scan_identity(self):
        """正常 maxrange 无命中端点（valid=True）保留所属扫描身份。"""
        snapshot = snapshot_with_scans([(3, 0.1, [[8.0, 0.0, 0.0]])])
        flat = flatten_far_window(snapshot)
        self.assertTrue(bool(flat.valid[0, 0]))                    # 无命中是有效点
        self.assertEqual(flat.scan_ids[0, 0].item(), 3)            # 保留扫描身份
        self.assertEqual(float(flat.times[0, 0]), f32(0.1))
        # 同槽其余 false 点与全部空槽：-1/0、掩码 False（槽位仍存在）。
        self.assertTrue(bool((flat.scan_ids[0, 1:] == EMPTY_SCAN_ID).all()))
        self.assertTrue(bool((flat.times[0, 1:] == 0).all()))
        self.assertFalse(bool(flat.valid[0, 1:].any()))
        self.assertEqual(tuple(flat.points.shape),
                         (1, FAR_WINDOW_SLOTS * FAR_POINTS_PER_SCAN, 3))


if __name__ == "__main__":
    unittest.main(verbosity=2)
