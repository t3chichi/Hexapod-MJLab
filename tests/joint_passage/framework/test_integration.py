#!/usr/bin/env python
"""F4 自检：合成扫描—观测—网络—PD 集成链路与统一检查入口（合成输入，CPU，unittest）。

覆盖真实跨模块风险：接口组合成立（身份校验→行对齐→角键排序→网络前向→PD 分级，
期望值由本文件用纯 Python 稳定排序/F3 规则独立解析）；50Hz 控制读取不推进 10Hz
窗口（每 5 次读取窗不增长，仅新扫描写入，同扫描重复提交拒绝）；10 个独立扫描填满
远端环、第 11 个淘汰最旧且各槽内容互不相同（不复制同扫描补槽）；记录 pre-action
控制步与最新扫描身份、近/远原米制输入（与窗口数值逐位一致）及 Ê/action_mean/
PD 分级结果；缺失环境/错 episode/未来扫描拒绝；重排/重复/非连续 env 身份按身份
逐行配对（env_id 不作数组下标）且与单行执行数值一致；零扫描/未满窗口以掩码编码
（GRU 空行零特征、不捏造数据）；部分 reset 只清目标环境、旧 episode 不泄漏；
角键契约（有效点处 NaN/形状/dtype 拒绝）；准备结果不与快照共享存储（live 别名）；
执行使用显式 q_ref/q/q̇（不从观测推导）并拒绝错批/错设备；离线标签绑定辅助接口
（包络=控制步身份、地面=扫描身份、缺失行合法、未来/错位/错扫描拒绝）；在线通路
签名与结果字段无任何标签/teacher/真值（inspect/数据类结构断言）；state_dict 重载
输出一致（重新构造显式保持夹具配置，raw/residual 两模式）；必要网络配置缺失/
非法拒绝；接触约束仍调用 F3 纯计数与绊碰（合成接触期望解析已知，无 reward/reset）。

合成夹具声明：网络配置固定 mlp_activation=torch.tanh、init_fn=None、
envelope_mode="raw"（重载用例另测 residual）、ground_samples=3、固定 torch 种子；
q_ref 取显式合成值；近/远角键用 atan2(y,x) 仅作**测试排序键**（验证排序通路与
元数据置换），不代表真实雷达方位角/外参/零轴，也不冻结无命中处理；body 集合为
合成身份（躯干 0、足 1-6、hip 7、腿 8、雷达 13、地面 100、墙 101）。以上均非
任务默认配置。本检查不读真实资产、不启用物理/优化器/训练；由 GLM 编写，本机
headless 无审批客户端**未运行**，由 Codex 在交付后执行，失败以非零退出传播。

修正轮（2026-10-05）：Codex 真实首跑 16 项中 **14 通过、1 FAIL、1 ERROR**——
①test_angle_key_contract_enforced 的远端键长度反例误切第一维（B=1 未缩短），
改为截断第二维（点数维）；②test_full_chain_records_identity_inputs_and_outputs
的 far 期望元数据误把 (B,10) 槽级 scan_id 直接 reshape (B,640) 触发 ERROR，
改为逐槽扩展 64 点、无效点按掩码置 EMPTY/0、按独立 orders 置换，并补
far_point_times 元数据检查；runner 实际返回 1、报告预存在返回 2 均与设计一致。
独立反例（logs/joint_passage/framework/20261005_build/F4_independent_review_
initial.json，不覆盖）另确认两处真实缺陷并由本轮修正：③bind_offline_labels
未保证 ground 标签行与观测同 env/episode/批量/行序（标签行可配到观测之外的
环境），新增行身份校验与 GroundRowIdentityTest；④run_checks 报告写入存在
TOCTOU（预检查后 write_text 覆盖并发证据），改排他创建 open(..., "x") 并新增
RunChecksTest（仅 mock 子进程返回对象，不替代真实四文件 CPU 执行，判定只看
实际退出码）。本轮为静态修改、未运行；保留全部原断言与合成框架边界，不删
检查、不放宽规则；由 Codex 复跑。
"""

import ast
import dataclasses
import importlib.util
import inspect
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import torch

from hexapod_mjlab.rl.joint_passage import (
    EMPTY_SCAN_ID,
    FAR_POINTS_PER_SCAN,
    FAR_WINDOW_SLOTS,
    NEAR_POINTS,
    ContractError,
    EnvelopeLabel,
    GroundLabel,
    IdentityMismatchError,
    OnlineObservation,
    ScanWindow,
    ScanWindowSnapshot,
    StaleScanError,
)
from hexapod_mjlab.rl.joint_passage.contacts import (
    ControlStepContacts,
    count_control_step_contacts,
    feet_stumble_flag,
)
from hexapod_mjlab.rl.joint_passage.control import compute_pd_torques
from hexapod_mjlab.rl.joint_passage.integration import (
    IntegratedControlResult,
    PreparedControlInputs,
    bind_offline_labels,
    execute_control_step,
    prepare_control_inputs,
)
from hexapod_mjlab.rl.joint_passage.network import JointPassageNetwork, NetworkOutput

NEAR = NEAR_POINTS                     # 256
FAR_FLAT = FAR_WINDOW_SLOTS * FAR_POINTS_PER_SCAN   # 640
SNAPSHOT_FIELDS = (
    "env_ids", "episode_ids", "near_points", "near_valid", "near_filled",
    "near_scan_ids", "near_times", "far_points", "far_valid", "far_filled",
    "far_scan_ids", "far_times",
)
# 启动基准物理输出（F2 构造期强制：任意有限输入的初始 Ê）。
BASELINE = torch.tensor([0.59, 0.69, 0.59, 0.89, -0.89, 0.0], dtype=torch.float32)
# 合成 body 身份（夹具，非任务默认）：躯干 0、足 1-6、非足机器人 7(hip)/8(腿)/
# 13(雷达安装件)，外部 100(地面)/101(墙)。
FEET = [1, 2, 3, 4, 5, 6]
ROBOT_BODIES = [0] + FEET + [7, 8, 13]


def load_run_checks():
    """按文件路径加载 run_checks.py（tests 目录非包，不依赖执行方式/启动目录）。"""
    path = Path(__file__).resolve().parent / "run_checks.py"
    spec = importlib.util.spec_from_file_location("joint_passage_framework_run_checks", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# 合成夹具（确定性；种子/配置均为测试选择，不是任务默认）
# ---------------------------------------------------------------------------

def synth_block(n_points, valid_count, *, base, seed):
    """确定性合成点块：有效点 = base·x̂ + 0.05·randn（种子唯一），无效槽精确 0。"""
    generator = torch.Generator().manual_seed(seed)
    points = torch.randn(n_points, 3, generator=generator).mul(0.05)
    points[:, 0] += base
    valid = torch.zeros(n_points, dtype=torch.bool)
    valid[:valid_count] = True
    points = torch.where(valid[:, None], points, torch.zeros(()))
    return points[None].contiguous(), valid[None].contiguous()


def near_block(scan_id, valid_count=NEAR):
    return synth_block(NEAR, valid_count, base=0.5 + 0.05 * scan_id, seed=1000 + 10 * scan_id)


def far_block(scan_id, valid_count=FAR_POINTS_PER_SCAN):
    return synth_block(FAR_POINTS_PER_SCAN, valid_count, base=2.0 + 0.05 * scan_id, seed=2000 + 10 * scan_id)


def write_scan(window, *, env, scan_id, time_s,
               near_valid=NEAR, far_valid=FAR_POINTS_PER_SCAN):
    """向单个环境写入一次新完成扫描（episode 取窗口当前值）。"""
    episode = int(window.episode_ids[env])
    near_points, near_valid_t = near_block(scan_id, near_valid)
    far_points, far_valid_t = far_block(scan_id, far_valid)
    window.write(
        env_ids=[env],
        episode_ids=[episode],
        scan_ids=[scan_id],
        times=[time_s],
        near_points=near_points,
        near_valid=near_valid_t,
        far_points=far_points,
        far_valid=far_valid_t,
    )


def make_chain_window():
    """B=3 窗口夹具：env0 三次扫描、env1 一次、env2 两次（异步、未满环）。"""
    window = ScanWindow(3, episode_ids=[11, 11, 11])
    write_scan(window, env=0, scan_id=0, time_s=0.10)
    write_scan(window, env=2, scan_id=0, time_s=0.10)
    write_scan(window, env=0, scan_id=1, time_s=0.20)
    write_scan(window, env=1, scan_id=0, time_s=0.15)
    write_scan(window, env=2, scan_id=1, time_s=0.20)
    write_scan(window, env=0, scan_id=2, time_s=0.30)
    return window


def make_observation(env_ids, *, episodes, step_id, time_s, command=(0.5, -0.25, 0.8)):
    """确定性在线观测（pre-action 状态；字段取互不相同的常数便于接线检查）。"""
    batch = len(env_ids)

    def block(value, dim):
        if isinstance(value, (tuple, list)):
            tensor = torch.tensor(value, dtype=torch.float32)
            assert tensor.numel() == dim
        else:
            tensor = torch.full((dim,), float(value), dtype=torch.float32)
        return tensor.unsqueeze(0).repeat(batch, 1)

    return OnlineObservation(
        env_ids=torch.tensor(env_ids, dtype=torch.int64),
        episode_ids=torch.tensor(episodes, dtype=torch.int64),
        step_ids=torch.full((batch,), step_id, dtype=torch.int64),
        time_s=torch.full((batch,), float(time_s), dtype=torch.float32),
        lin_vel_b=block((0.5, 0.25, -1.0), 3),
        ang_vel_b=block((2.0, -4.0, 0.5), 3),
        gravity_b=block((0.0, 0.0, -1.0), 3),
        joint_pos_rel_ref=block(0.3, 18),
        joint_vel=block(4.0, 18),
        prev_action=block(-7.0, 18),
        command=block(command, 3),
    )


def make_model(*, seed=5, envelope_mode="raw", ground_samples=3, init_fn=None,
               mlp_activation=torch.tanh):
    """构造网络：合成夹具配置全部显式（被测构造函数本身无默认值）。"""
    torch.manual_seed(seed)
    return JointPassageNetwork(
        mlp_activation=mlp_activation,
        init_fn=init_fn,
        envelope_mode=envelope_mode,
        ground_samples=ground_samples,
    )


def test_angle_keys(points):
    """合成测试排序键 atan2(y,x)：仅验证排序通路，不代表真实雷达方位角/外参。"""
    return torch.atan2(points[..., 1], points[..., 0])


def window_row_index(snapshot, env_ids):
    """夹具侧按身份查找窗口行（与被测 prepare 的查找语义相同，仅测试用）。"""
    wanted = torch.tensor(env_ids, dtype=torch.int64)
    equal = wanted[:, None] == snapshot.env_ids[None, :]
    assert bool((equal.sum(dim=1) == 1).all()), "夹具环境身份须存在于快照"
    return equal.to(torch.int64).argmax(dim=1)


def caller_keys(snapshot, env_ids):
    """模拟“调用方持有各环境传感数据”计算行对齐的近/远角键（夹具）。"""
    rows = window_row_index(snapshot, env_ids)
    near_keys = test_angle_keys(snapshot.near_points[rows])
    far_keys = test_angle_keys(snapshot.far_points[rows].reshape(len(env_ids), FAR_FLAT, 3))
    return near_keys, far_keys


def reference_order(keys_row, valid_row):
    """独立稳定排序参考（纯 Python sorted，稳定性与 torch stable argsort 一致）：
    有效点按键升序，无效点保持原相对顺序置尾。"""
    count = int(keys_row.shape[0])
    valid_idx = [i for i in range(count) if bool(valid_row[i])]
    invalid_idx = [i for i in range(count) if not bool(valid_row[i])]
    return sorted(valid_idx, key=lambda i: float(keys_row[i])) + invalid_idx


def gather_points(points, orders):
    order = torch.tensor(orders, dtype=torch.int64)
    return torch.gather(points, 1, order.unsqueeze(-1).expand(-1, -1, points.shape[-1]))


def gather_flat(flat_tensor, orders):
    order = torch.tensor(orders, dtype=torch.int64)
    return torch.gather(flat_tensor, 1, order)


def chain_q_fixtures(batch):
    """显式 q_ref/q/q̇ 合成值（与观测字段一致性由调用方保证，这里保持一致）。"""
    q_ref = torch.linspace(-0.1, 0.1, 18, dtype=torch.float32).unsqueeze(0).repeat(batch, 1)
    joint_pos = q_ref + 0.3                      # = q_ref + joint_pos_rel_ref
    joint_vel = torch.linspace(-2.0, 2.0, 18, dtype=torch.float32).unsqueeze(0).repeat(batch, 1)
    return q_ref, joint_pos, joint_vel


class FullChainTest(unittest.TestCase):
    """接口组合：准备→前向→PD 全链与独立解析参考一致。"""

    def test_full_chain_records_identity_inputs_and_outputs(self):
        window = make_chain_window()
        snapshot = window.read()
        obs = make_observation([0, 1, 2], episodes=[11, 11, 11], step_id=20, time_s=0.40)
        near_keys, far_keys = caller_keys(snapshot, [0, 1, 2])
        prepared = prepare_control_inputs(
            window=snapshot, observation=obs,
            near_angle_keys=near_keys, far_angle_keys=far_keys,
        )
        # 记录 pre-action 控制步身份与最新近端扫描身份/时间。
        self.assertTrue(torch.equal(prepared.observation.env_ids, obs.env_ids))
        self.assertTrue(torch.equal(prepared.observation.step_ids, obs.step_ids))
        self.assertTrue(torch.equal(prepared.observation.time_s, obs.time_s))
        self.assertTrue(torch.equal(
            prepared.near_scan_ids, torch.tensor([2, 0, 1], dtype=torch.int64)))
        self.assertTrue(torch.equal(
            prepared.near_times, torch.tensor([0.30, 0.15, 0.20], dtype=torch.float32)))
        self.assertTrue(bool(prepared.near_filled.all()))
        # 近端原米制输入 = 窗口近端行的独立参考置换（无缩放/无补偿）。
        rows = window_row_index(snapshot, [0, 1, 2])
        near_orders = [reference_order(near_keys[b], snapshot.near_valid[rows][b]) for b in range(3)]
        self.assertTrue(torch.equal(
            prepared.near_points, gather_points(snapshot.near_points[rows], near_orders)))
        self.assertTrue(torch.equal(
            prepared.near_valid, gather_flat(snapshot.near_valid[rows], near_orders)))
        # 远端原米制输入 = 10×64 拼接（最旧→最新）后的独立参考置换，元数据同步。
        far_points_flat = snapshot.far_points[rows].reshape(3, FAR_FLAT, 3)
        far_valid_flat = snapshot.far_valid[rows].reshape(3, FAR_FLAT)
        # 期望元数据独立构造（不取被测 prepare 输出当期望）：槽级 scan_id/time
        # 逐槽扩展 64 点再 flatten，无效点按 far_valid 置 EMPTY_SCAN_ID/0 后按
        # 独立 orders 置换。本夹具已填充槽内 64 点全有效，故该替换只作用于
        # 未填充槽（窗口本就 EMPTY/0 填充），参考与被测输出逐位一致可判。
        far_scan_flat = (
            snapshot.far_scan_ids[rows][:, :, None]
            .expand(3, FAR_WINDOW_SLOTS, FAR_POINTS_PER_SCAN)
            .reshape(3, FAR_FLAT)
        )
        far_times_flat = (
            snapshot.far_times[rows][:, :, None]
            .expand(3, FAR_WINDOW_SLOTS, FAR_POINTS_PER_SCAN)
            .reshape(3, FAR_FLAT)
        )
        expected_far_scan = torch.where(
            far_valid_flat, far_scan_flat, torch.full_like(far_scan_flat, EMPTY_SCAN_ID))
        expected_far_times = torch.where(
            far_valid_flat, far_times_flat, torch.zeros_like(far_times_flat))
        far_orders = [reference_order(far_keys[b], far_valid_flat[b]) for b in range(3)]
        self.assertTrue(torch.equal(
            prepared.far_points, gather_points(far_points_flat, far_orders)))
        self.assertTrue(torch.equal(
            prepared.far_valid, gather_flat(far_valid_flat, far_orders)))
        self.assertTrue(torch.equal(
            prepared.far_point_scan_ids, gather_flat(expected_far_scan, far_orders)))
        self.assertTrue(torch.equal(
            prepared.far_point_times, gather_flat(expected_far_times, far_orders)))
        # 掩码保留：每行有效远点数 = 已填充槽数 × 64（env0=3、env1=1、env2=2）。
        self.assertTrue(torch.equal(
            prepared.far_valid.sum(dim=1), torch.tensor([192, 64, 128], dtype=torch.int64)))
        for row, expected in ((0, {0, 1, 2}), (1, {0}), (2, {0, 1})):
            present = set(prepared.far_point_scan_ids[row][prepared.far_valid[row]].tolist())
            self.assertEqual(present, expected)

        # 执行：网络前向（初始 Ê=启动基准，数值锚点）→ action_mean → PD 分级。
        model = make_model(seed=5)
        q_ref, joint_pos, joint_vel = chain_q_fixtures(3)
        result = execute_control_step(
            model=model, prepared=prepared,
            q_ref=q_ref, joint_pos=joint_pos, joint_vel=joint_vel,
        )
        self.assertIsInstance(result, IntegratedControlResult)
        self.assertTrue(torch.allclose(result.network.envelope, BASELINE.unsqueeze(0).repeat(3, 1),
                                       rtol=0.0, atol=1e-6))
        action = result.network.action_mean
        self.assertEqual(tuple(action.shape), (3, 18))
        self.assertEqual(action.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(action).all()))
        self.assertEqual(tuple(result.network.value.shape), (3, 1))
        self.assertIsNotNone(result.network.ground_pred)
        self.assertEqual(tuple(result.network.ground_pred.shape), (3, 3))
        # PD 分级 = F3 规则的内联重算与直接调用逐位一致（同输入同代码路径）。
        clipped = action.clamp(min=-100.0, max=100.0)
        self.assertTrue(torch.equal(result.pd.raw_action, action))
        self.assertTrue(torch.equal(result.pd.clipped_action, clipped))
        self.assertTrue(torch.equal(result.pd.joint_target, q_ref + 0.25 * clipped))
        expected_unclipped = 130.0 * (q_ref + 0.25 * clipped - joint_pos) - 2.0 * joint_vel
        self.assertTrue(torch.equal(result.pd.unclipped_torque, expected_unclipped))
        self.assertTrue(torch.equal(
            result.pd.applied_torque, expected_unclipped.clamp(min=-80.0, max=80.0)))
        self.assertTrue(torch.equal(
            result.pd.torque_saturated, expected_unclipped.abs() > 80.0))
        direct = compute_pd_torques(
            raw_action=action, q_ref=q_ref, joint_pos=joint_pos, joint_vel=joint_vel)
        for field in ("raw_action", "clipped_action", "joint_target",
                      "unclipped_torque", "applied_torque", "torque_saturated"):
            self.assertTrue(torch.equal(getattr(result.pd, field), getattr(direct, field)))

    def test_control_reads_do_not_advance_window(self):
        """50Hz 控制读取/10Hz 完成扫描：两次写入间 5 次读取窗不增长、内容不变。"""
        window = ScanWindow(2, episode_ids=[11, 11])
        model = make_model(seed=6)
        previous_cycle_near = None
        for cycle in range(2):
            scan_id, scan_time = cycle, 0.1 * (cycle + 1)
            near_points, near_valid = near_block(scan_id)
            far_points, far_valid = far_block(scan_id)
            window.write(
                env_ids=[0, 1], episode_ids=[11, 11],
                scan_ids=[scan_id, scan_id], times=[scan_time, scan_time],
                near_points=torch.cat([near_points] * 2), near_valid=torch.cat([near_valid] * 2),
                far_points=torch.cat([far_points] * 2), far_valid=torch.cat([far_valid] * 2),
            )
            previous_snapshot = None
            for j in range(5):
                snapshot = window.read()
                if previous_snapshot is not None:
                    for name in SNAPSHOT_FIELDS:
                        self.assertTrue(
                            torch.equal(getattr(snapshot, name), getattr(previous_snapshot, name)),
                            f"{name} 在无写入的连续控制读取间发生变化",
                        )
                previous_snapshot = snapshot
                obs = make_observation([0, 1], episodes=[11, 11],
                                       step_id=10 * cycle + j, time_s=scan_time + 0.02 * j)
                near_keys, far_keys = caller_keys(snapshot, [0, 1])
                prepared = prepare_control_inputs(
                    window=snapshot, observation=obs,
                    near_angle_keys=near_keys, far_angle_keys=far_keys)
                q_ref, joint_pos, joint_vel = chain_q_fixtures(2)
                result = execute_control_step(
                    model=model, prepared=prepared,
                    q_ref=q_ref, joint_pos=joint_pos, joint_vel=joint_vel)
                self.assertTrue(bool(torch.isfinite(result.pd.applied_torque).all()))
            # 新写入后窗口确实推进（上面“不变”不是空洞成立）。
            if previous_cycle_near is not None:
                self.assertFalse(torch.equal(previous_snapshot.near_scan_ids, previous_cycle_near))
            previous_cycle_near = previous_snapshot.near_scan_ids.clone()
            # 同扫描重复提交拒绝且内容不变。
            before = window.read()
            with self.assertRaises(StaleScanError):
                write_scan(window, env=0, scan_id=scan_id, time_s=scan_time + 0.5)
            after = window.read()
            for name in SNAPSHOT_FIELDS:
                self.assertTrue(torch.equal(getattr(before, name), getattr(after, name)))

    def test_far_window_fills_then_evicts_oldest_without_copying(self):
        """10 个独立扫描填满远端环；第 11 个淘汰最旧，各槽内容互不相同。"""
        window = ScanWindow(1, episode_ids=[11])
        for scan_id in range(10):
            write_scan(window, env=0, scan_id=scan_id, time_s=0.1 * (scan_id + 1))
        full = window.read()
        self.assertTrue(bool(full.far_filled.all()))
        write_scan(window, env=0, scan_id=10, time_s=1.1)
        evicted = window.read()
        self.assertTrue(torch.equal(
            evicted.far_scan_ids[0], torch.arange(1, 11, dtype=torch.int64)))
        self.assertEqual(int(evicted.near_scan_ids[0]), 10)
        # 每槽内容 = 对应 scan 的确定性合成块（夹具种子重生成，逐位一致），
        # 不复制同扫描补槽：slot k 属于 scan k+1。
        for slot in range(10):
            points, _ = far_block(slot + 1)
            self.assertTrue(torch.equal(evicted.far_points[0, slot], points[0]),
                            f"slot {slot} 内容与其 scan 不符")
        self.assertFalse(torch.equal(evicted.far_points[0, 0], evicted.far_points[0, 1]))
        obs = make_observation([0], episodes=[11], step_id=60, time_s=1.2)
        near_keys, far_keys = caller_keys(evicted, [0])
        prepared = prepare_control_inputs(
            window=evicted, observation=obs,
            near_angle_keys=near_keys, far_angle_keys=far_keys)
        present = set(prepared.far_point_scan_ids[0][prepared.far_valid[0]].tolist())
        self.assertEqual(present, set(range(1, 11)))
        self.assertNotIn(0, present)
        self.assertTrue(torch.equal(
            prepared.near_points, gather_points(
                evicted.near_points, [reference_order(near_keys[0], evicted.near_valid[0])])))


class IdentityMismatchTest(unittest.TestCase):
    """身份/时间错位拒绝与任意环境身份行配对。"""

    def test_missing_env_wrong_episode_and_future_scan_rejected(self):
        window = make_chain_window()
        snapshot = window.read()
        model = make_model(seed=7)
        good = make_observation([0], episodes=[11], step_id=20, time_s=0.40)
        near_keys, far_keys = caller_keys(snapshot, [0])
        prepared = prepare_control_inputs(
            window=snapshot, observation=good,
            near_angle_keys=near_keys, far_angle_keys=far_keys)
        q_ref, joint_pos, joint_vel = chain_q_fixtures(1)
        result = execute_control_step(
            model=model, prepared=prepared,
            q_ref=q_ref, joint_pos=joint_pos, joint_vel=joint_vel)
        self.assertTrue(bool(torch.isfinite(result.pd.applied_torque).all()))
        # 环境缺失：观测 env=7 不在快照（env_ids=0..2）。
        missing = make_observation([7], episodes=[11], step_id=20, time_s=0.40)
        with self.assertRaises(IdentityMismatchError):
            prepare_control_inputs(
                window=snapshot, observation=missing,
                near_angle_keys=test_angle_keys(snapshot.near_points[[0]]),
                far_angle_keys=test_angle_keys(snapshot.far_points[[0]].reshape(1, FAR_FLAT, 3)))
        # 错 episode：env0 当前 episode=11。
        wrong_episode = make_observation([0], episodes=[12], step_id=20, time_s=0.40)
        with self.assertRaises(IdentityMismatchError):
            prepare_control_inputs(
                window=snapshot, observation=wrong_episode,
                near_angle_keys=near_keys, far_angle_keys=far_keys)
        # 未来扫描：env0 最新扫描 time=0.30 晚于控制步 0.25。
        past = make_observation([0], episodes=[11], step_id=10, time_s=0.25)
        with self.assertRaises(IdentityMismatchError):
            prepare_control_inputs(
                window=snapshot, observation=past,
                near_angle_keys=near_keys, far_angle_keys=far_keys)

    def test_reordered_duplicated_and_noncontiguous_env_rows(self):
        """重排/重复逻辑行与非连续快照 env_ids 按身份配对；单行与批行数值一致。"""
        window = ScanWindow(2, episode_ids=[11, 11])
        write_scan(window, env=0, scan_id=3, time_s=0.10)
        write_scan(window, env=1, scan_id=4, time_s=0.12)
        base = window.read()
        # 手工重映射快照 env_ids 为非连续身份 [7, 3]（快照契约允许：唯一非负），
        # 验证 prepare 用身份匹配而非把 env_id 当连续下标。
        fields = {name: getattr(base, name).clone() for name in SNAPSHOT_FIELDS}
        fields["env_ids"] = torch.tensor([7, 3], dtype=torch.int64)
        snapshot = ScanWindowSnapshot(**fields)
        obs = make_observation([3, 7, 7], episodes=[11, 11, 11], step_id=9, time_s=0.20)
        near_keys, far_keys = caller_keys(snapshot, [3, 7, 7])
        prepared = prepare_control_inputs(
            window=snapshot, observation=obs,
            near_angle_keys=near_keys, far_angle_keys=far_keys)
        self.assertTrue(torch.equal(
            prepared.near_scan_ids, torch.tensor([4, 3, 3], dtype=torch.int64)))
        self.assertTrue(torch.equal(
            prepared.near_times, torch.tensor([0.12, 0.10, 0.10], dtype=torch.float32)))
        # 重复行（同环境同控制步）输入完全一致。
        self.assertTrue(torch.equal(prepared.near_points[1], prepared.near_points[2]))
        self.assertTrue(torch.equal(prepared.far_points[1], prepared.far_points[2]))
        model = make_model(seed=8)
        q_ref, joint_pos, joint_vel = chain_q_fixtures(3)
        batched = execute_control_step(
            model=model, prepared=prepared,
            q_ref=q_ref, joint_pos=joint_pos, joint_vel=joint_vel)
        # 单行执行与批行数值一致（allclose：批内矩阵乘末位差异，F2 先例）。
        for env_id, batch_row in ((7, 1), (3, 0)):
            single_obs = make_observation([env_id], episodes=[11], step_id=9, time_s=0.20)
            single_near_keys, single_far_keys = caller_keys(snapshot, [env_id])
            single = execute_control_step(
                model=model,
                prepared=prepare_control_inputs(
                    window=snapshot, observation=single_obs,
                    near_angle_keys=single_near_keys, far_angle_keys=single_far_keys),
                q_ref=q_ref[:1], joint_pos=joint_pos[:1], joint_vel=joint_vel[:1])
            self.assertTrue(torch.allclose(single.network.envelope,
                                           batched.network.envelope[batch_row:batch_row + 1],
                                           rtol=0.0, atol=1e-6))
            self.assertTrue(torch.allclose(single.network.action_mean,
                                           batched.network.action_mean[batch_row:batch_row + 1],
                                           rtol=0.0, atol=1e-6))
            # 力矩容差较宽：批内/单行矩阵乘末位差被 PD 增益（130·0.25）放大约
            # 30 倍，atol=1e-4 相对 ±80 N·m 量级仍约为 1e-6 相对差（数值一致性
            # 检查，非新设正式验收门，F2 单行/批行先例同此说明）。
            self.assertTrue(torch.allclose(single.pd.applied_torque,
                                           batched.pd.applied_torque[batch_row:batch_row + 1],
                                           rtol=0.0, atol=1e-4))


class SensorMaskTest(unittest.TestCase):
    """零扫描/未满窗口的掩码编码与部分 reset 隔离。"""

    def test_zero_and_partial_scan_rows_are_mask_encoded(self):
        window = ScanWindow(2, episode_ids=[11, 11])
        write_scan(window, env=0, scan_id=0, time_s=0.10, far_valid=30)  # env1 无扫描
        snapshot = window.read()
        obs = make_observation([0, 1], episodes=[11, 11], step_id=5, time_s=0.20)
        near_keys, far_keys = caller_keys(snapshot, [0, 1])
        prepared = prepare_control_inputs(
            window=snapshot, observation=obs,
            near_angle_keys=near_keys, far_angle_keys=far_keys)
        self.assertTrue(torch.equal(
            prepared.near_filled, torch.tensor([True, False])))
        self.assertTrue(torch.equal(
            prepared.near_scan_ids, torch.tensor([0, EMPTY_SCAN_ID], dtype=torch.int64)))
        # 零扫描行：掩码全 False、数值精确 0，不捏造数据、不复制补足。
        self.assertFalse(bool(prepared.near_valid[1].any()))
        self.assertTrue(torch.equal(prepared.near_points[1], torch.zeros_like(prepared.near_points[1])))
        self.assertFalse(bool(prepared.far_valid[1].any()))
        self.assertTrue((prepared.far_point_scan_ids[1] == EMPTY_SCAN_ID).all())
        # 未满窗口的掩码保留：env0 远端 1 槽 × 30 有效点。
        self.assertTrue(torch.equal(
            prepared.far_valid.sum(dim=1), torch.tensor([30, 0], dtype=torch.int64)))
        model = make_model(seed=9)
        q_ref, joint_pos, joint_vel = chain_q_fixtures(2)
        result = execute_control_step(
            model=model, prepared=prepared,
            q_ref=q_ref, joint_pos=joint_pos, joint_vel=joint_vel)
        # GRU 空行返回精确零特征（F2 保证），形态输出仍为启动基准（数值锚点）。
        self.assertTrue(torch.equal(
            result.network.near_feature[1], torch.zeros_like(result.network.near_feature[1])))
        self.assertTrue(torch.equal(
            result.network.far_feature[1], torch.zeros_like(result.network.far_feature[1])))
        self.assertTrue(torch.allclose(result.network.envelope[1], BASELINE, rtol=0.0, atol=1e-6))
        self.assertTrue(bool(torch.isfinite(result.pd.applied_torque).all()))

    def test_partial_reset_clears_only_target_envs(self):
        window = make_chain_window()
        before = window.read()
        window.reset_envs(env_ids=[1], new_episode_ids=[12])
        after = window.read()
        self.assertFalse(bool(after.near_filled[1]))
        self.assertTrue((after.far_scan_ids[1] == EMPTY_SCAN_ID).all())
        for name in ("near_points", "near_valid", "near_scan_ids", "near_times",
                     "far_points", "far_valid", "far_scan_ids", "far_times"):
            self.assertTrue(torch.equal(getattr(after, name)[[0, 2]], getattr(before, name)[[0, 2]]),
                            f"reset 影响了未目标环境: {name}")
        # 新 episode 下可正常准备，零扫描掩码编码、不泄漏旧 episode 扫描。
        obs = make_observation([0, 1, 2], episodes=[11, 12, 11], step_id=30, time_s=0.40)
        near_keys, far_keys = caller_keys(after, [0, 1, 2])
        prepared = prepare_control_inputs(
            window=after, observation=obs,
            near_angle_keys=near_keys, far_angle_keys=far_keys)
        self.assertFalse(bool(prepared.near_valid[1].any()))
        self.assertTrue((prepared.far_point_scan_ids[1] == EMPTY_SCAN_ID).all())
        # 旧 episode 观测（env1 episode=11）拒绝。
        stale = make_observation([1], episodes=[11], step_id=30, time_s=0.40)
        with self.assertRaises(IdentityMismatchError):
            prepare_control_inputs(
                window=after, observation=stale,
                near_angle_keys=test_angle_keys(after.near_points[[1]]),
                far_angle_keys=test_angle_keys(after.far_points[[1]].reshape(1, FAR_FLAT, 3)))


class PrepareContractTest(unittest.TestCase):
    """角键契约与准备结果无 live 别名。"""

    def test_angle_key_contract_enforced(self):
        window = make_chain_window()
        snapshot = window.read()
        obs = make_observation([0], episodes=[11], step_id=20, time_s=0.40)
        near_keys, far_keys = caller_keys(snapshot, [0])
        bad = near_keys.clone()
        bad[0, 0] = float("nan")     # 有效点处 NaN（该行近端全有效）
        with self.assertRaises(ContractError):
            prepare_control_inputs(
                window=snapshot, observation=obs,
                near_angle_keys=bad, far_angle_keys=far_keys)
        with self.assertRaises(ContractError):
            prepare_control_inputs(
                window=snapshot, observation=obs,
                near_angle_keys=near_keys[:, :NEAR - 1].contiguous(), far_angle_keys=far_keys)
        with self.assertRaises(ContractError):
            prepare_control_inputs(
                window=snapshot, observation=obs,
                near_angle_keys=near_keys.double(), far_angle_keys=far_keys)
        with self.assertRaises(ContractError):
            prepare_control_inputs(
                window=snapshot, observation=obs,
                near_angle_keys=near_keys,
                far_angle_keys=far_keys[:, :FAR_FLAT - 2].contiguous())   # 截断第二维（点数）

    def test_prepared_inputs_do_not_alias_snapshot(self):
        window = make_chain_window()
        snapshot = window.read()
        obs = make_observation([0, 1, 2], episodes=[11, 11, 11], step_id=20, time_s=0.40)
        near_keys, far_keys = caller_keys(snapshot, [0, 1, 2])
        prepared = prepare_control_inputs(
            window=snapshot, observation=obs,
            near_angle_keys=near_keys, far_angle_keys=far_keys)
        saved = {
            "near_points": prepared.near_points.clone(),
            "near_valid": prepared.near_valid.clone(),
            "far_points": prepared.far_points.clone(),
            "far_valid": prepared.far_valid.clone(),
            "near_scan_ids": prepared.near_scan_ids.clone(),
            "far_point_scan_ids": prepared.far_point_scan_ids.clone(),
        }
        # 就地破坏快照（快照是可变 dataclass）不影响已准备输入。
        snapshot.near_points.mul_(0.0)
        snapshot.far_points.mul_(0.0)
        snapshot.near_valid.fill_(False)
        snapshot.near_scan_ids.fill_(99)
        snapshot.far_scan_ids.fill_(99)
        for name, value in saved.items():
            self.assertTrue(torch.equal(getattr(prepared, name), value),
                            f"prepared.{name} 与快照共享存储")
        # 后续窗口写入/重读同样不影响。
        write_scan(window, env=0, scan_id=99, time_s=9.9)
        for name, value in saved.items():
            self.assertTrue(torch.equal(getattr(prepared, name), value))


class ExecutionTest(unittest.TestCase):
    """执行链使用显式状态并拒绝错位输入。"""

    def test_execute_uses_explicit_state_and_rejects_misaligned_inputs(self):
        window = make_chain_window()
        snapshot = window.read()
        obs = make_observation([0, 1, 2], episodes=[11, 11, 11], step_id=20, time_s=0.40)
        near_keys, far_keys = caller_keys(snapshot, [0, 1, 2])
        prepared = prepare_control_inputs(
            window=snapshot, observation=obs,
            near_angle_keys=near_keys, far_angle_keys=far_keys)
        model = make_model(seed=10)
        q_ref, _, joint_vel = chain_q_fixtures(3)
        # 故意给出与观测字段不一致的显式 q：执行必须按显式值而非观测推导。
        joint_pos = q_ref + obs.joint_pos_rel_ref + 0.07
        result = execute_control_step(
            model=model, prepared=prepared,
            q_ref=q_ref, joint_pos=joint_pos, joint_vel=joint_vel)
        target = q_ref + 0.25 * result.network.action_mean.clamp(min=-100.0, max=100.0)
        expected = 130.0 * (target - joint_pos) - 2.0 * joint_vel
        self.assertTrue(torch.equal(result.pd.unclipped_torque, expected))
        # 错批/错设备/非张量拒绝（设备检查用 meta 张量，无需真实第二设备）。
        with self.assertRaises(ContractError):
            execute_control_step(model=model, prepared=prepared,
                                 q_ref=q_ref[:2], joint_pos=joint_pos, joint_vel=joint_vel)
        with self.assertRaises(ContractError):
            execute_control_step(
                model=model, prepared=prepared,
                q_ref=torch.empty(3, 18, dtype=torch.float32, device="meta"),
                joint_pos=joint_pos, joint_vel=joint_vel)
        with self.assertRaises(ContractError):
            execute_control_step(model=model, prepared=prepared,
                                 q_ref=q_ref.tolist(), joint_pos=joint_pos, joint_vel=joint_vel)
        with self.assertRaises(ContractError):
            execute_control_step(model=model, prepared=None,
                                 q_ref=q_ref, joint_pos=joint_pos, joint_vel=joint_vel)
        with self.assertRaises(ContractError):
            execute_control_step(model=object(), prepared=prepared,
                                 q_ref=q_ref, joint_pos=joint_pos, joint_vel=joint_vel)


class LabelIsolationTest(unittest.TestCase):
    """标签离线绑定辅助接口与在线通路隔离。"""

    def test_offline_label_binding_helper(self):
        window = make_chain_window()
        snapshot = window.read()
        obs = make_observation([0, 1], episodes=[11, 11], step_id=20, time_s=0.40)
        envelope = EnvelopeLabel(
            env_ids=torch.tensor([0, 1], dtype=torch.int64),
            episode_ids=torch.tensor([11, 11], dtype=torch.int64),
            step_ids=torch.tensor([20, 20], dtype=torch.int64),
            values=torch.tensor([[0.60, 0.70, 0.60, 0.90, -0.90, 0.05],
                                 [0.65, 0.72, 0.61, 0.95, -0.85, 0.00]], dtype=torch.float32),
            valid=torch.tensor([True, True]),
        )
        ground = GroundLabel(
            env_ids=torch.tensor([0, 1], dtype=torch.int64),
            episode_ids=torch.tensor([11, 11], dtype=torch.int64),
            scan_ids=torch.tensor([2, 0], dtype=torch.int64),   # 近端槽：env0=2、env1=0
            heights=torch.tensor([[0.1, -0.2, 0.0, 0.0], [0.3, 0.1, -0.1, 0.0]], dtype=torch.float32),
            positions=torch.zeros(2, 4, 3, dtype=torch.float32),
            valid=torch.tensor([[True, True, False, False], [True, True, True, False]]),
        )
        bound = bind_offline_labels(obs, envelope_label=envelope,
                                    ground_label=ground, window=snapshot)
        self.assertIs(bound.envelope, envelope)
        self.assertIs(bound.ground, ground)
        # 未来控制步包络标签拒绝。
        future = EnvelopeLabel(
            env_ids=envelope.env_ids, episode_ids=envelope.episode_ids,
            step_ids=torch.tensor([21, 21], dtype=torch.int64),
            values=envelope.values, valid=envelope.valid)
        with self.assertRaises(IdentityMismatchError):
            bind_offline_labels(obs, envelope_label=future)
        # 包络标签行序错位（env 重排）拒绝。
        swapped = EnvelopeLabel(
            env_ids=torch.tensor([1, 0], dtype=torch.int64),
            episode_ids=torch.tensor([11, 11], dtype=torch.int64),
            step_ids=torch.tensor([20, 20], dtype=torch.int64),
            values=envelope.values, valid=envelope.valid)
        with self.assertRaises(IdentityMismatchError):
            bind_offline_labels(obs, envelope_label=swapped)
        # 错扫描身份地面标签拒绝。
        wrong_scan = GroundLabel(
            env_ids=ground.env_ids, episode_ids=ground.episode_ids,
            scan_ids=torch.tensor([1, 0], dtype=torch.int64),
            heights=ground.heights, positions=ground.positions, valid=ground.valid)
        with self.assertRaises(IdentityMismatchError):
            bind_offline_labels(obs, ground_label=wrong_scan, window=snapshot)
        # 缺失行（scan_id=-1、全 False 掩码、零填）合法，不阻碍有效行。
        missing = GroundLabel(
            env_ids=ground.env_ids, episode_ids=ground.episode_ids,
            scan_ids=torch.tensor([2, EMPTY_SCAN_ID], dtype=torch.int64),
            heights=torch.tensor([[0.1, -0.2, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
            positions=torch.zeros(2, 4, 3, dtype=torch.float32),
            valid=torch.tensor([[True, True, False, False], [False, False, False, False]]))
        bound_missing = bind_offline_labels(obs, ground_label=missing, window=snapshot)
        self.assertIs(bound_missing.ground, missing)
        # 结构性要求：ground 必须伴随窗口；空绑定拒绝。
        with self.assertRaises(ContractError):
            bind_offline_labels(obs, ground_label=ground)
        with self.assertRaises(ContractError):
            bind_offline_labels(obs)

    def test_online_path_has_no_label_arguments_or_fields(self):
        expected = {
            "prepare_control_inputs": {"window", "observation", "near_angle_keys", "far_angle_keys"},
            "execute_control_step": {"model", "prepared", "q_ref", "joint_pos", "joint_vel"},
        }
        for name, params in expected.items():
            function = {"prepare_control_inputs": prepare_control_inputs,
                        "execute_control_step": execute_control_step}[name]
            self.assertEqual(set(inspect.signature(function).parameters), params, name)
        forward_params = set(inspect.signature(JointPassageNetwork.forward).parameters)
        self.assertEqual(
            forward_params, {"self", "observation", "near_points", "near_valid",
                             "far_points", "far_valid"})
        self.assertEqual(
            {f.name for f in dataclasses.fields(PreparedControlInputs)},
            {"observation", "near_points", "near_valid", "far_points", "far_valid",
             "near_filled", "near_scan_ids", "near_times",
             "far_point_scan_ids", "far_point_times"})
        self.assertEqual({f.name for f in dataclasses.fields(IntegratedControlResult)},
                         {"network", "pd"})
        for holder in (PreparedControlInputs, IntegratedControlResult, NetworkOutput):
            for field in dataclasses.fields(holder):
                low = field.name.lower()
                self.assertFalse(
                    any(token in low for token in ("label", "teacher", "truth")),
                    f"{holder.__name__}.{field.name} 疑似标签字段泄漏到在线通路")


class GroundRowIdentityTest(unittest.TestCase):
    """bind_offline_labels 的 ground 标签行身份（独立反例修正：标签行不得配到
    观测逻辑批之外的环境）。"""

    @staticmethod
    def _ground(env_ids, episodes, scan_ids, *, valid_rows=None, samples=4):
        """夹具地面标签：valid_rows 为 True 的行第 0 个采样有效（值非零），
        其余槽精确 0 且掩码 False；缺失/无效行全零。"""
        batch = len(env_ids)
        heights = torch.zeros(batch, samples, dtype=torch.float32)
        valid = torch.zeros(batch, samples, dtype=torch.bool)
        for row in range(batch):
            if valid_rows is None or valid_rows[row]:
                valid[row, 0] = True
                heights[row, 0] = 0.1 * (row + 1)
        return GroundLabel(
            env_ids=torch.tensor(env_ids, dtype=torch.int64),
            episode_ids=torch.tensor(episodes, dtype=torch.int64),
            scan_ids=torch.tensor(scan_ids, dtype=torch.int64),
            heights=heights,
            positions=torch.zeros(batch, samples, 3, dtype=torch.float32),
            valid=valid,
        )

    def test_ground_rows_must_match_observation_identity(self):
        window = make_chain_window()
        snapshot = window.read()
        obs1 = make_observation([0], episodes=[11], step_id=20, time_s=0.40)
        # 独立反例：观测只有 env0，标签只有 env1 行——env1 在窗口中合法存在、
        # episode/scan 均正确，但不在本次观测逻辑批内，必须拒绝。
        env1_only = self._ground([1], [11], [0], valid_rows=[True])
        with self.assertRaises(IdentityMismatchError):
            bind_offline_labels(obs1, ground_label=env1_only, window=snapshot)
        # episode 不一致（env 相同）。
        wrong_episode = self._ground([0], [12], [2])
        with self.assertRaises(IdentityMismatchError):
            bind_offline_labels(obs1, ground_label=wrong_episode, window=snapshot)
        # 行重排：观测 [0,1]、标签 [1,0]（双方不同序的重排拒绝）。
        obs2 = make_observation([0, 1], episodes=[11, 11], step_id=20, time_s=0.40)
        reordered = self._ground([1, 0], [11, 11], [0, 2])
        with self.assertRaises(IdentityMismatchError):
            bind_offline_labels(obs2, ground_label=reordered, window=snapshot)
        # 缺行：观测 2 行、标签 1 行。
        missing_rows = self._ground([0], [11], [2])
        with self.assertRaises(ContractError):
            bind_offline_labels(obs2, ground_label=missing_rows, window=snapshot)
        # 多行：观测 1 行、标签 2 行。
        extra_rows = self._ground([0, 1], [11, 11], [2, 0])
        with self.assertRaises(ContractError):
            bind_offline_labels(obs1, ground_label=extra_rows, window=snapshot)
        # 设备不等（借 meta 张量触发设备分支，不做数据访问）。
        device_case = self._ground([0], [11], [2])
        device_case.env_ids = device_case.env_ids.to("meta")
        with self.assertRaises(ContractError):
            bind_offline_labels(obs1, ground_label=device_case, window=snapshot)

    def test_ground_rows_accept_aligned_reorder_duplicates_and_missing_frames(self):
        snapshot = make_chain_window().read()
        # 双方同序的重排行（env [2,0]）可接受：每行 env/episode 与观测一致，
        # 扫描身份仍按 F1 对窗口近端槽逐行绑定。
        obs_re = make_observation([2, 0], episodes=[11, 11], step_id=20, time_s=0.40)
        reordered = self._ground([2, 0], [11, 11], [1, 2])
        bound = bind_offline_labels(obs_re, ground_label=reordered, window=snapshot)
        self.assertIs(bound.ground, reordered)
        # 双方同序的重复行（同 env 多控制步）可接受。
        obs_dup = make_observation([0, 0], episodes=[11, 11], step_id=20, time_s=0.40)
        duplicate = self._ground([0, 0], [11, 11], [2, 2])
        self.assertIs(
            bind_offline_labels(obs_dup, ground_label=duplicate, window=snapshot).ground,
            duplicate)
        # 部分 reset 缺帧：env1 reset→episode12 无扫描；标签缺失行（scan=-1、
        # 全 False 掩码）合法，且该行 env/episode 仍与观测逐行对应。
        reset_window = make_chain_window()
        reset_window.reset_envs(env_ids=[1], new_episode_ids=[12])
        reset_snapshot = reset_window.read()
        obs_reset = make_observation([0, 1], episodes=[11, 12], step_id=30, time_s=0.40)
        mixed = self._ground([0, 1], [11, 12], [2, EMPTY_SCAN_ID],
                             valid_rows=[True, False])
        self.assertIs(
            bind_offline_labels(obs_reset, ground_label=mixed, window=reset_snapshot).ground,
            mixed)


class ConsistencyTest(unittest.TestCase):
    """checkpoint 重载一致与必要配置缺失拒绝。"""

    def test_state_dict_roundtrip_preserves_outputs(self):
        window = make_chain_window()
        snapshot = window.read()
        obs = make_observation([0, 1, 2], episodes=[11, 11, 11], step_id=20, time_s=0.40)
        near_keys, far_keys = caller_keys(snapshot, [0, 1, 2])
        prepared = prepare_control_inputs(
            window=snapshot, observation=obs,
            near_angle_keys=near_keys, far_angle_keys=far_keys)
        q_ref, joint_pos, joint_vel = chain_q_fixtures(3)
        for mode in ("raw", "residual"):
            with self.subTest(mode=mode):
                torch.manual_seed(5)
                original = JointPassageNetwork(
                    mlp_activation=torch.tanh, init_fn=None,
                    envelope_mode=mode, ground_samples=3)
                first = execute_control_step(
                    model=original, prepared=prepared,
                    q_ref=q_ref, joint_pos=joint_pos, joint_vel=joint_vel)
                state = original.state_dict()
                # 重新构造显式保持夹具配置（不同随机初始化），再加载 checkpoint。
                torch.manual_seed(900)
                reloaded = JointPassageNetwork(
                    mlp_activation=torch.tanh, init_fn=None,
                    envelope_mode=mode, ground_samples=3)
                reloaded.load_state_dict(state)
                second = execute_control_step(
                    model=reloaded, prepared=prepared,
                    q_ref=q_ref, joint_pos=joint_pos, joint_vel=joint_vel)
                for field in ("envelope", "action_mean", "value", "ground_pred"):
                    self.assertTrue(torch.equal(getattr(first.network, field),
                                                getattr(second.network, field)),
                                    f"{mode}/{field} 重载输出不一致")
                self.assertTrue(torch.equal(first.pd.applied_torque, second.pd.applied_torque))
                self.assertTrue(torch.allclose(first.network.envelope,
                                               BASELINE.unsqueeze(0).repeat(3, 1),
                                               rtol=0.0, atol=1e-6))

    def test_required_network_config_missing_or_invalid_rejected(self):
        with self.assertRaises(TypeError):
            JointPassageNetwork(mlp_activation=torch.tanh)   # 缺 init_fn/envelope_mode/ground_samples
        with self.assertRaises(ContractError):
            make_model(envelope_mode="bogus")               # 非法表示模式


class ContactsChainTest(unittest.TestCase):
    """接触约束串联：仍调用 F3 纯计数与绊碰（合成期望解析已知，无 reward/reset）。"""

    def test_contact_counting_and_stumble_in_control_step_batch(self):
        env_ids = torch.tensor([2, 0, 1], dtype=torch.int64)   # 重排真实环境身份
        contacts = ControlStepContacts(
            env_ids=env_ids,
            contact_env_rows=torch.tensor([0, 0, 0, 0, 0, 0, 1, 2], dtype=torch.int64),
            substeps=torch.tensor([0, 0, 0, 1, 2, 3, 1, 2], dtype=torch.int64),
            body_a=torch.tensor([7, 7, 7, 7, 7, 7, 1, 100], dtype=torch.int64),
            body_b=torch.tensor([100, 100, 8, 8, 8, 8, 100, 101], dtype=torch.int64),
            distance=torch.zeros(8, dtype=torch.float32),
            force=torch.tensor([
                [2.0, 0.0, 0.0], [-2.0, 0.0, 0.0],          # 同 body 反方向不抵消，去重仍计 1
                [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],           # (7,8) 4 子步全有 → 平均 1.0
                [0.0, 0.0, 30.0],                           # 足-环境：不入非足计数
                [5.0, 0.0, 0.0],                            # 外部×外部：不计数
            ], dtype=torch.float32),
        )
        result = count_control_step_contacts(
            contacts,
            robot_body_ids=torch.tensor(ROBOT_BODIES, dtype=torch.int64),
            foot_body_ids=torch.tensor(FEET, dtype=torch.int64))
        self.assertTrue(torch.equal(result.env_ids, env_ids))
        # 解析已知：行0 非足环境（hip×地面，仅子步0，去重）=0.25；
        # 行0 自碰撞 (7,8) 全部 4 子步 =1.0；行1 足-环境、行2 外部×外部均 0。
        self.assertTrue(torch.equal(
            result.env_collision_counts, torch.tensor([0.25, 0.0, 0.0], dtype=torch.float32)))
        self.assertTrue(torch.equal(
            result.self_collision_counts, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)))
        # 绊碰诊断（世界坐标净足力，夹具）：行0 足0 (5.1,0,1) 触发，其余零力。
        foot_forces = torch.zeros(3, 6, 3, dtype=torch.float32)
        foot_forces[0, 0] = torch.tensor([5.1, 0.0, 1.0])
        self.assertTrue(torch.equal(
            feet_stumble_flag(foot_forces), torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)))


class RunChecksTest(unittest.TestCase):
    """统一 runner（独立反例修正 TOCTOU 后的锁定检查）。

    仅 mock 子进程返回对象（SimpleNamespace），不替代真实四文件 CPU 执行、
    也不依赖任何 PASS 文本字段；runner 的文件存在性预检查仍作用于本目录
    真实存在的四个 test 文件。
    """

    def test_failure_gate_uses_exit_code_not_pass_text(self):
        checks = load_run_checks()
        lying = SimpleNamespace(returncode=1, stdout="PASS\nRan 4 tests in 0.01s\n", stderr="")
        with mock.patch.object(checks.subprocess, "run", return_value=lying):
            self.assertEqual(checks.main([]), 1)   # 子进程写 PASS 也按退出码 1 失败
        quiet = SimpleNamespace(returncode=0, stdout="Ran 4 tests in 0.01s\n", stderr="")
        with mock.patch.object(checks.subprocess, "run", return_value=quiet):
            self.assertEqual(checks.main([]), 0)   # 无 PASS 字样也按退出码 0 通过

    def test_preexisting_report_refused_without_overwrite(self):
        checks = load_run_checks()
        passing = SimpleNamespace(returncode=0, stdout="Ran 4 tests in 0.01s\n", stderr="")
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.md"
            report.write_text("旧证据", encoding="utf-8")
            with mock.patch.object(checks.subprocess, "run", return_value=passing):
                self.assertEqual(checks.main(["--report", str(report)]), 2)
            self.assertEqual(report.read_text(encoding="utf-8"), "旧证据")

    def test_report_created_during_run_is_not_overwritten(self):
        checks = load_run_checks()
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.md"
            concurrent = "并发进程写入的证据"

            def fake_run(*_args, **_kwargs):
                # 模拟检查运行期间另一进程创建同路径报告（TOCTOU 场景）。
                report.write_text(concurrent, encoding="utf-8")
                return SimpleNamespace(returncode=0, stdout="Ran 4 tests\n", stderr="")

            with mock.patch.object(checks.subprocess, "run", side_effect=fake_run):
                self.assertEqual(checks.main(["--report", str(report)]), 2)
            self.assertEqual(report.read_text(encoding="utf-8"), concurrent)


class ModulePurityTest(unittest.TestCase):
    """integration 模块只依赖标准库+torch+本包 F1-F3 模块。"""

    def test_integration_module_imports_only_framework_modules(self):
        import hexapod_mjlab.rl.joint_passage.integration as integration_module
        source = Path(integration_module.__file__).read_text(encoding="utf-8")
        allowed_absolute = {"torch", "dataclasses", "__future__"}
        allowed_relative = {"contracts", "control", "network", "mapping"}
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertIn(alias.name.split(".")[0], allowed_absolute,
                                  f"非法绝对导入: {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.level == 1:
                    self.assertIn(node.module, allowed_relative,
                                  f"非法相对导入: {node.module}")
                else:
                    self.assertIn((node.module or "").split(".")[0], allowed_absolute,
                                  f"非法 from 导入: {node.module}")
        for forbidden in ("mujoco", "torch.optim", "subprocess", "import mjlab"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
