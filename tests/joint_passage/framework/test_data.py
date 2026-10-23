#!/usr/bin/env python
"""F1 自检：数据契约与批处理扫描窗口（合成输入，CPU，unittest）。

覆盖真实风险反例：重复控制读取/重复扫描不增窗、>10 次刷新淘汰最旧、部分 reset
与陈旧 episode 拒绝（含跨 reset id 复用/回退）、不同环境异步刷新、空槽/部分有效点
零填充、无命中端点有效、NaN/Inf 拒绝（含掩码 false 位置）、角度排序保持元数据且
不冒充时间排序、外部输入与快照互不污染、控制步/扫描两套标签身份关联及未来数据
拒绝、整批写入原子性、快照近/远关联一致、任意环境身份的重排/重复/子集逻辑采样行
配对、地面标签缺失行不阻碍有效行。

修正轮（2026-10-05）：首跑 18/19 通过，test_flatten_far_window_then_angle_sort
因 B=2 夹具误断言 (1,640,3) 失败，已修正夹具与断言一致且不弱化角度/metadata 检查；
并按 Codex 独立反例（logs/joint_passage/framework/20261005_build/
F1_independent_review.json）修正环境身份契约、快照近/远关联与地面标签缺失行。
本检查由 GLM 编写，本机 headless 无审批客户端未运行；由 Codex 在交付后执行，
失败以非零退出传播。合成夹具不代表任务默认配置。
"""

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import torch

from hexapod_mjlab.rl.joint_passage import (
    COMMAND_DIM,
    EMPTY_SCAN_ID,
    FAR_POINTS_PER_SCAN,
    FAR_WINDOW_SLOTS,
    NEAR_POINTS,
    PROPRIO_DIM,
    PROPRIO_LAYOUT,
    ContractError,
    EnvelopeLabel,
    EpisodeMismatchError,
    GroundLabel,
    IdentityMismatchError,
    OnlineObservation,
    ScanWindow,
    ScanWindowSnapshot,
    StaleScanError,
    TimeOrderError,
    bind_envelope_label,
    bind_ground_label,
    bind_window_to_observation,
    flatten_far_window,
    sort_points_by_angle_keys,
)

SNAPSHOT_FIELDS = (
    "env_ids",
    "episode_ids",
    "near_points",
    "near_valid",
    "near_filled",
    "near_scan_ids",
    "near_times",
    "far_points",
    "far_valid",
    "far_filled",
    "far_scan_ids",
    "far_times",
)


def point_block(n_points, valid_count, *, radius=1.0):
    """确定性合成点块：前 valid_count 个为不同有效点，其余精确 0 且掩码 False。"""
    idx = torch.arange(n_points, dtype=torch.float32)
    block = torch.stack([radius + 0.001 * idx, 0.01 * idx, -0.05 + 0.002 * idx], dim=-1)
    valid = idx < float(valid_count)
    points = torch.where(valid[:, None], block, torch.zeros(1))
    return points[None].contiguous(), valid[None].contiguous()


def make_window(batch=2, episode=11):
    return ScanWindow(batch, episode_ids=[episode] * batch)


def write_scan(window, *, env, episode, scan_id, time_s,
               near_valid_count=NEAR_POINTS, far_valid_count=FAR_POINTS_PER_SCAN):
    near_points, near_valid = point_block(NEAR_POINTS, near_valid_count, radius=0.5 + 0.01 * scan_id)
    far_points, far_valid = point_block(FAR_POINTS_PER_SCAN, far_valid_count, radius=2.0 + 0.01 * scan_id)
    window.write(
        env_ids=[env],
        episode_ids=[episode],
        scan_ids=[scan_id],
        times=[time_s],
        near_points=near_points,
        near_valid=near_valid,
        far_points=far_points,
        far_valid=far_valid,
    )


def clone_snapshot(snapshot):
    return ScanWindowSnapshot(**{name: getattr(snapshot, name).clone() for name in SNAPSHOT_FIELDS})


def assert_snapshots_equal(test, left, right):
    for name in SNAPSHOT_FIELDS:
        test.assertTrue(
            torch.equal(getattr(left, name), getattr(right, name)),
            f"{name} 在两次读取间发生变化",
        )


def make_obs_rows(*, env_ids, episode_ids, step_ids, time_s):
    """任意非负环境身份的在线观测（支持重排/重复逻辑采样行）。"""
    batch = len(env_ids)

    def zeros(*shape):
        return torch.zeros(batch, *shape, dtype=torch.float32)

    return OnlineObservation(
        env_ids=torch.tensor(env_ids, dtype=torch.int64),
        episode_ids=torch.tensor(episode_ids, dtype=torch.int64),
        step_ids=torch.tensor(step_ids, dtype=torch.int64),
        time_s=torch.tensor(time_s, dtype=torch.float32),
        lin_vel_b=zeros(3),
        ang_vel_b=zeros(3),
        gravity_b=torch.tensor([[0.0, 0.0, 1.0]]).repeat(batch, 1),
        joint_pos_rel_ref=zeros(18),
        joint_vel=zeros(18),
        prev_action=zeros(18),
        command=zeros(3),
    )


def make_obs(*, episode=11, step=5, time_s=0.30, batch=2):
    return make_obs_rows(
        env_ids=list(range(batch)),
        episode_ids=[episode] * batch,
        step_ids=[step] * batch,
        time_s=[time_s] * batch,
    )


def make_envelope_label_rows(*, env_ids, episode_ids, step_ids):
    batch = len(env_ids)
    return EnvelopeLabel(
        env_ids=torch.tensor(env_ids, dtype=torch.int64),
        episode_ids=torch.tensor(episode_ids, dtype=torch.int64),
        step_ids=torch.tensor(step_ids, dtype=torch.int64),
        values=torch.tensor([[0.59, 0.69, 0.59, 0.89, -0.89, 0.0]]).repeat(batch, 1),
        valid=torch.ones(batch, dtype=torch.bool),
    )


def make_envelope_label(*, episode=11, step=5, batch=2):
    return make_envelope_label_rows(
        env_ids=list(range(batch)),
        episode_ids=[episode] * batch,
        step_ids=[step] * batch,
    )


def make_ground_label_rows(*, env_ids, episode_ids, scan_ids, missing=None, samples=4):
    """missing[i]=True 的行构造为缺失标签行（scan_id=-1、valid 全 False、零填）。"""
    batch = len(env_ids)
    valid = torch.ones(batch, samples, dtype=torch.bool)
    heights = torch.tensor([[0.01, -0.02, 0.0, 0.03]]).repeat(batch, 1)
    if missing is not None:
        for row, is_missing in enumerate(missing):
            if is_missing:
                valid[row] = False
                heights[row] = 0.0
    return GroundLabel(
        env_ids=torch.tensor(env_ids, dtype=torch.int64),
        episode_ids=torch.tensor(episode_ids, dtype=torch.int64),
        scan_ids=torch.tensor(scan_ids, dtype=torch.int64),
        heights=heights,
        positions=torch.zeros(batch, samples, 3),
        valid=valid,
    )


def make_ground_label(*, episode=11, scan_id=3, batch=2):
    return make_ground_label_rows(
        env_ids=list(range(batch)),
        episode_ids=[episode] * batch,
        scan_ids=[scan_id] * batch,
    )


class WindowTests(unittest.TestCase):
    def test_repeated_control_reads_do_not_advance_window(self):
        window = make_window()
        for scan_id in (1, 2, 3):
            write_scan(window, env=0, episode=11, scan_id=scan_id, time_s=0.1 * scan_id)
        first = window.read()
        for _ in range(5):
            assert_snapshots_equal(self, first, window.read())
        self.assertEqual(int(first.far_filled[0].sum()), 3)
        self.assertEqual(first.near_scan_ids[0].item(), 3)

    def test_more_than_ten_refreshes_evict_oldest(self):
        window = make_window()
        for scan_id in range(1, 13):  # 12 次刷新，穿插控制读取
            write_scan(window, env=0, episode=11, scan_id=scan_id, time_s=0.1 * scan_id)
            window.read()
        snapshot = window.read()
        self.assertTrue(bool(snapshot.far_filled[0].all()))
        self.assertEqual(snapshot.far_scan_ids[0].tolist(), list(range(3, 13)))
        self.assertEqual(snapshot.near_scan_ids[0].item(), 12)
        self.assertTrue(bool((snapshot.far_times[0][1:] > snapshot.far_times[0][:-1]).all()))
        # 位置内容与身份一致：最旧槽为 scan 3，最新槽与近端同为 scan 12
        self.assertAlmostEqual(float(snapshot.far_points[0, 0, 0, 0]), 2.03, places=6)
        self.assertAlmostEqual(float(snapshot.far_points[0, 9, 0, 0]), 2.12, places=6)
        self.assertAlmostEqual(float(snapshot.near_points[0, 0, 0]), 0.62, places=6)

    def test_duplicate_and_stale_scans_rejected_without_mutation(self):
        window = make_window()
        write_scan(window, env=0, episode=11, scan_id=1, time_s=0.1)
        write_scan(window, env=0, episode=11, scan_id=2, time_s=0.2)
        before = window.read()
        with self.assertRaises(StaleScanError):
            write_scan(window, env=0, episode=11, scan_id=2, time_s=0.3)  # 同扫描重复提交
        with self.assertRaises(StaleScanError):
            write_scan(window, env=0, episode=11, scan_id=1, time_s=0.3)  # 旧扫描
        with self.assertRaises(TimeOrderError):
            write_scan(window, env=0, episode=11, scan_id=3, time_s=0.2)  # 非单调时间
        assert_snapshots_equal(self, before, window.read())

    def test_partial_reset_isolates_envs_and_rejects_stale_episode(self):
        window = make_window()
        write_scan(window, env=0, episode=11, scan_id=1, time_s=0.1)
        write_scan(window, env=1, episode=11, scan_id=7, time_s=0.1)
        with self.assertRaises(ContractError):
            window.reset_envs(env_ids=[1], new_episode_ids=[11])  # 同 id 复用拒绝
        window.reset_envs(env_ids=[0], new_episode_ids=[12])
        snapshot = window.read()
        self.assertFalse(bool(snapshot.near_filled[0]))              # 仅 env0 被清空
        self.assertEqual(int(snapshot.episode_ids[0]), 12)
        self.assertTrue(bool(snapshot.near_filled[1]))               # env1 不受影响
        self.assertEqual(snapshot.near_scan_ids[1].item(), 7)
        with self.assertRaises(EpisodeMismatchError):
            write_scan(window, env=0, episode=11, scan_id=2, time_s=0.2)  # 陈旧 episode
        with self.assertRaises(EpisodeMismatchError):
            write_scan(window, env=0, episode=13, scan_id=2, time_s=0.2)  # 未声明的新 episode
        # 新 episode 内从零计数即可用，无需等满窗口
        write_scan(window, env=0, episode=12, scan_id=1, time_s=0.0)
        after = window.read()
        self.assertTrue(bool(after.near_filled[0]))
        self.assertEqual(after.far_scan_ids[0, 0].item(), 1)
        self.assertTrue(bool((after.far_filled[0, 1:] == 0).all()))
        self.assertEqual(after.near_scan_ids[1].item(), 7)

    def test_reset_rejects_episode_id_reuse_across_resets(self):
        window = make_window()  # 初始 episode 11
        window.reset_envs(env_ids=[0], new_episode_ids=[12])
        with self.assertRaises(ContractError):  # 回退到历史 id（非当前）同样拒绝
            window.reset_envs(env_ids=[0], new_episode_ids=[11])
        window.reset_envs(env_ids=[0], new_episode_ids=[13])  # 全新 id 正常
        self.assertEqual(window.episode_ids.tolist(), [13, 11])
        snapshot = window.read()
        self.assertEqual(int(snapshot.episode_ids[0]), 13)

    def test_envs_refresh_independently(self):
        window = make_window()
        write_scan(window, env=0, episode=11, scan_id=1, time_s=0.1)
        write_scan(window, env=0, episode=11, scan_id=2, time_s=0.2)
        snapshot = window.read()
        self.assertFalse(bool(snapshot.near_filled[1]))
        self.assertFalse(bool(snapshot.far_filled[1].any()))
        write_scan(window, env=1, episode=11, scan_id=5, time_s=0.3)  # 与 env0 不同步的扫描号
        snapshot = window.read()
        self.assertEqual(snapshot.near_scan_ids.tolist(), [2, 5])
        expected = torch.zeros(2, FAR_WINDOW_SLOTS, dtype=torch.bool)
        expected[0, :2] = True
        expected[1, 0] = True
        self.assertTrue(torch.equal(snapshot.far_filled, expected))

    def test_empty_window_and_partial_points_are_representable(self):
        window = make_window()
        snapshot = window.read()  # 空状态可表示，不捏造数据
        snapshot.validate()
        self.assertFalse(bool(snapshot.near_filled.any()))
        self.assertTrue(bool((snapshot.near_scan_ids == EMPTY_SCAN_ID).all()))
        self.assertFalse(bool(snapshot.near_valid.any()))
        self.assertEqual(float(snapshot.near_points.abs().sum()), 0.0)
        self.assertFalse(bool(snapshot.far_filled.any()))
        # 部分有效点：无效点精确 0 并被掩码排除；空槽不复制首扫描或点补足
        write_scan(window, env=0, episode=11, scan_id=1, time_s=0.1,
                   near_valid_count=10, far_valid_count=0)
        snapshot = window.read()
        self.assertEqual(int(snapshot.near_valid[0].sum()), 10)
        self.assertEqual(float(snapshot.near_points[0][~snapshot.near_valid[0]].abs().sum()), 0.0)
        self.assertTrue(bool(snapshot.far_filled[0, 0]))            # 全无效点仍是占用槽位的真实扫描
        self.assertFalse(bool(snapshot.far_valid[0, 0].any()))
        self.assertFalse(bool(snapshot.far_filled[0, 1:].any()))
        self.assertTrue(bool((snapshot.far_points[0, 1:] == 0).all()))
        self.assertTrue(bool((snapshot.far_scan_ids[0, 1:] == EMPTY_SCAN_ID).all()))

    def test_no_hit_max_range_endpoint_is_valid(self):
        window = make_window()
        near_points, near_valid = point_block(NEAR_POINTS, 0)
        far_points, far_valid = point_block(FAR_POINTS_PER_SCAN, 0)
        near_points[0, 0] = torch.tensor([8.0, 0.0, 0.0])  # 正常无命中的最大量程端点
        near_valid[0, 0] = True
        window.write(
            env_ids=[0], episode_ids=[11], scan_ids=[1], times=[0.1],
            near_points=near_points, near_valid=near_valid,
            far_points=far_points, far_valid=far_valid,
        )
        snapshot = window.read()
        self.assertTrue(bool(snapshot.near_valid[0, 0]))
        self.assertEqual(float(snapshot.near_points[0, 0, 0]), 8.0)

    def test_nan_inf_and_masked_nan_inputs_rejected(self):
        window = make_window()
        near_points, near_valid = point_block(NEAR_POINTS, NEAR_POINTS)
        far_points, far_valid = point_block(FAR_POINTS_PER_SCAN, FAR_POINTS_PER_SCAN)
        kwargs = dict(env_ids=[0], episode_ids=[11], scan_ids=[1], times=[0.1],
                      far_points=far_points, far_valid=far_valid)
        for bad_value in (float("nan"), float("inf")):  # 有效位置 NaN/Inf 拒绝
            near_points[0, 3, 1] = bad_value
            with self.assertRaises(ContractError):
                window.write(near_points=near_points.clone(), near_valid=near_valid, **kwargs)
        near_points[0, 3, 1] = 0.0
        masked_points = near_points.clone()  # 掩码 false 位置 NaN 同样拒绝，不得夹带
        masked_valid = near_valid.clone()
        masked_valid[0, 5] = False
        masked_points[0, 5] = float("nan")
        with self.assertRaises(ContractError):
            window.write(near_points=masked_points, near_valid=masked_valid, **kwargs)
        with self.assertRaises(ContractError):  # float64 输入显式拒绝
            window.write(near_points=near_points.double(), near_valid=near_valid, **kwargs)
        with self.assertRaises(ContractError):  # 点数不符拒绝
            window.write(near_points=near_points[:, :200, :], near_valid=near_valid[:, :200], **kwargs)
        snapshot = window.read()  # 全部失败写入后窗口保持空
        self.assertFalse(bool(snapshot.near_filled.any()))

    def test_batched_write_is_atomic(self):
        window = make_window()
        near_points, near_valid = point_block(NEAR_POINTS, NEAR_POINTS)
        far_points, far_valid = point_block(FAR_POINTS_PER_SCAN, FAR_POINTS_PER_SCAN)
        with self.assertRaises(EpisodeMismatchError):
            window.write(
                env_ids=[0, 1], episode_ids=[11, 99], scan_ids=[1, 1], times=[0.1, 0.1],
                near_points=torch.cat([near_points, near_points]),
                near_valid=torch.cat([near_valid, near_valid]),
                far_points=torch.cat([far_points, far_points]),
                far_valid=torch.cat([far_valid, far_valid]),
            )
        snapshot = window.read()
        self.assertFalse(bool(snapshot.near_filled.any()))  # 合法行 0 也未被写入

    def test_external_mutation_and_snapshot_stability(self):
        window = make_window()
        near_points, near_valid = point_block(NEAR_POINTS, NEAR_POINTS)
        far_points, far_valid = point_block(FAR_POINTS_PER_SCAN, FAR_POINTS_PER_SCAN)
        window.write(
            env_ids=[0], episode_ids=[11], scan_ids=[1], times=[0.1],
            near_points=near_points, near_valid=near_valid,
            far_points=far_points, far_valid=far_valid,
        )
        near_points[0, 0, 0] = 99.0  # 外部修改输入不污染缓冲
        far_valid[0, 0] = False
        first = window.read()
        self.assertAlmostEqual(float(first.near_points[0, 0, 0]), 1.0, places=6)
        self.assertTrue(bool(first.far_valid[0, 0, 0]))
        write_scan(window, env=0, episode=11, scan_id=2, time_s=0.2)
        window.reset_envs(env_ids=[1], new_episode_ids=[12])
        second = window.read()  # 旧快照不被后续写入/reset 改变
        self.assertEqual(first.near_scan_ids[0].item(), 1)
        self.assertEqual(second.near_scan_ids[0].item(), 2)
        self.assertAlmostEqual(float(first.near_points[0, 0, 0]), 1.0, places=6)
        self.assertEqual(int(first.episode_ids[1]), 11)
        self.assertEqual(int(second.episode_ids[1]), 12)


class SortingTests(unittest.TestCase):
    def test_angle_sort_keeps_metadata_and_is_not_time_order(self):
        points = torch.zeros(1, 5, 3)
        keys = torch.tensor([[0.9, 0.1, float("nan"), 0.3, 0.7]])
        valid = torch.tensor([[True, True, False, True, True]])
        scan_ids = torch.tensor([[11, 11, 11, 12, 12]], dtype=torch.int64)
        times = torch.tensor([[0.1, 0.1, 0.1, 0.2, 0.2]])
        for i in range(5):
            if valid[0, i]:
                points[0, i] = torch.tensor([1.0, float(keys[0, i]), 10.0 + i])
        ordered = sort_points_by_angle_keys(points, valid, keys, scan_ids=scan_ids, times=times)
        # 有效点按角键升序：0.1(i=1) 0.3(i=3) 0.7(i=4) 0.9(i=0)；无效点 i=2 稳定置尾
        self.assertEqual([round(v, 6) for v in ordered.points[0, :, 2].tolist()],
                         [11.0, 13.0, 14.0, 10.0, 0.0])
        self.assertEqual(ordered.scan_ids[0].tolist(), [11, 12, 12, 11, 11])
        self.assertEqual([round(v, 6) for v in ordered.times[0].tolist()],
                         [0.1, 0.2, 0.2, 0.1, 0.1])
        self.assertTrue(bool(ordered.valid[0, :4].all()))
        self.assertFalse(bool(ordered.valid[0, 4]))
        self.assertNotEqual(ordered.points[0, :, 2].tolist(),
                            [10.0, 11.0, 0.0, 13.0, 14.0])  # 非时间/写入序
        with self.assertRaises(ContractError):  # 有效点角键 NaN 拒绝
            bad_keys = keys.clone()
            bad_keys[0, 0] = float("nan")
            sort_points_by_angle_keys(points, valid, bad_keys)
        with self.assertRaises(ContractError):  # 无效点非零拒绝
            bad_points = points.clone()
            bad_points[0, 2, 0] = 5.0
            sort_points_by_angle_keys(bad_points, valid, keys)

    def test_flatten_far_window_then_angle_sort(self):
        def far_scan(scan_id, angles):
            points = torch.zeros(1, FAR_POINTS_PER_SCAN, 3)
            valid = torch.zeros(1, FAR_POINTS_PER_SCAN, dtype=torch.bool)
            for j, angle in enumerate(angles):
                points[0, j] = torch.tensor(
                    [math.cos(angle), math.sin(angle), scan_id + 0.1 * j]
                )
                valid[0, j] = True
            return points, valid

        window = make_window()  # B=2：仅 env0 有扫描，env1 为空（夹具与断言一致）
        near_points, near_valid = point_block(NEAR_POINTS, NEAR_POINTS)
        for scan_id, (angles, time_s) in {
            1: ([0.2, 0.6], 0.1),
            2: ([0.1, 0.9], 0.2),
        }.items():
            far_points, far_valid = far_scan(scan_id, angles)
            window.write(
                env_ids=[0], episode_ids=[11], scan_ids=[scan_id], times=[time_s],
                near_points=near_points, near_valid=near_valid,
                far_points=far_points, far_valid=far_valid,
            )
        flat = flatten_far_window(window.read())
        self.assertEqual(tuple(flat.points.shape),
                         (2, FAR_WINDOW_SLOTS * FAR_POINTS_PER_SCAN, 3))
        keys = torch.atan2(flat.points[:, :, 1], flat.points[:, :, 0])
        ordered = sort_points_by_angle_keys(flat.points, flat.valid, keys,
                                            scan_ids=flat.scan_ids, times=flat.times)
        # env0 行：拼接后按角度全局排序 0.1(s2) 0.2(s1) 0.6(s1) 0.9(s2)，元数据随点同步
        self.assertEqual(int(ordered.valid[0].sum()), 4)
        self.assertEqual(ordered.scan_ids[0, :4].tolist(), [2, 1, 1, 2])
        self.assertEqual([round(v, 6) for v in ordered.times[0, :4].tolist()], [0.2, 0.1, 0.1, 0.2])
        self.assertEqual([round(v, 6) for v in ordered.points[0, :4, 2].tolist()],
                         [2.0, 1.0, 1.1, 2.1])
        self.assertFalse(bool(ordered.valid[0, 4:].any()))  # 其余为空槽/无效点
        # env1 行：无扫描，全无效点排序后仍全无效
        self.assertFalse(bool(ordered.valid[1].any()))
        self.assertTrue(bool((ordered.points[1] == 0).all()))


class BindingTests(unittest.TestCase):
    def test_envelope_label_binds_control_identity_only(self):
        observation = make_obs(episode=11, step=5)
        matched = make_envelope_label(episode=11, step=5)
        self.assertIs(bind_envelope_label(observation, matched), matched)
        with self.assertRaises(IdentityMismatchError):  # 未来控制步标签
            bind_envelope_label(observation, make_envelope_label(episode=11, step=6))
        with self.assertRaises(IdentityMismatchError):  # 旧 episode 标签
            bind_envelope_label(observation, make_envelope_label(episode=10, step=5))

    def test_ground_label_binds_scan_identity_only(self):
        window = make_window()
        write_scan(window, env=0, episode=11, scan_id=3, time_s=0.3)
        write_scan(window, env=1, episode=11, scan_id=3, time_s=0.3)
        snapshot = window.read()
        matched = make_ground_label(episode=11, scan_id=3)
        self.assertIs(bind_ground_label(snapshot, matched), matched)
        with self.assertRaises(IdentityMismatchError):  # 扫描身份不符
            bind_ground_label(snapshot, make_ground_label(episode=11, scan_id=2))
        with self.assertRaises(IdentityMismatchError):  # 环境身份不在窗口（错位/子集越界）
            bind_ground_label(snapshot, make_ground_label_rows(
                env_ids=[5], episode_ids=[11], scan_ids=[3]))
        with self.assertRaises(IdentityMismatchError):  # 窗口无扫描时不可绑定有效标签
            bind_ground_label(make_window().read(), make_ground_label(episode=11, scan_id=3))

    def test_ground_label_missing_rows_do_not_block_valid_rows(self):
        window = make_window()
        write_scan(window, env=1, episode=11, scan_id=4, time_s=0.4)
        window.reset_envs(env_ids=[0], new_episode_ids=[12])  # 部分 reset：env0 无扫描
        snapshot = window.read()
        self.assertFalse(bool(snapshot.near_filled[0]))
        mixed = make_ground_label_rows(
            env_ids=[0, 1], episode_ids=[12, 11],
            scan_ids=[EMPTY_SCAN_ID, 4], missing=[True, False],
        )
        self.assertIs(bind_ground_label(snapshot, mixed), mixed)  # 缺失行不阻碍有效行
        with self.assertRaises(IdentityMismatchError):  # 有效标签行扫描身份不符
            bind_ground_label(snapshot, make_ground_label_rows(
                env_ids=[0, 1], episode_ids=[12, 11],
                scan_ids=[EMPTY_SCAN_ID, 3], missing=[True, False]))
        with self.assertRaises(IdentityMismatchError):  # 对无扫描环境伪造有效标签
            bind_ground_label(snapshot, make_ground_label_rows(
                env_ids=[0, 1], episode_ids=[12, 11],
                scan_ids=[4, 4], missing=[False, False]))
        with self.assertRaises(IdentityMismatchError):  # 缺失行的 episode 也须同批一致
            bind_ground_label(snapshot, make_ground_label_rows(
                env_ids=[0, 1], episode_ids=[13, 11],
                scan_ids=[EMPTY_SCAN_ID, 4], missing=[True, False]))
        with self.assertRaises(ContractError):  # scan_id=-1 却声明有效标签：契约拒绝
            make_ground_label_rows(
                env_ids=[0, 1], episode_ids=[12, 11],
                scan_ids=[EMPTY_SCAN_ID, 4], missing=[False, False])

    def test_window_observation_binding_rejects_future_scan(self):
        window = make_window()
        write_scan(window, env=0, episode=11, scan_id=1, time_s=0.30)
        write_scan(window, env=1, episode=11, scan_id=1, time_s=0.30)
        snapshot = window.read()
        current = make_obs(episode=11, step=15, time_s=0.32)
        self.assertIs(bind_window_to_observation(snapshot, current), snapshot)
        with self.assertRaises(IdentityMismatchError):  # 扫描晚于控制步（未来扫描）
            bind_window_to_observation(snapshot, make_obs(episode=11, step=14, time_s=0.28))
        with self.assertRaises(IdentityMismatchError):  # episode 不一致
            bind_window_to_observation(snapshot, make_obs(episode=12, step=15, time_s=0.32))

    def test_env_identity_reordering_and_pairing(self):
        window = make_window()
        write_scan(window, env=0, episode=11, scan_id=1, time_s=0.30)
        write_scan(window, env=1, episode=11, scan_id=1, time_s=0.30)
        snapshot = window.read()
        reordered = make_obs_rows(  # 重排逻辑采样行，按身份逐行配对
            env_ids=[1, 0], episode_ids=[11, 11], step_ids=[9, 5], time_s=[0.32, 0.32])
        self.assertIs(bind_window_to_observation(snapshot, reordered), snapshot)
        duplicated = make_obs_rows(  # 同环境多控制步组成一批
            env_ids=[0, 0], episode_ids=[11, 11], step_ids=[5, 6], time_s=[0.32, 0.32])
        self.assertIs(bind_window_to_observation(snapshot, duplicated), snapshot)
        with self.assertRaises(IdentityMismatchError):  # 环境身份不在窗口
            bind_window_to_observation(snapshot, make_obs_rows(
                env_ids=[7], episode_ids=[11], step_ids=[5], time_s=[0.32]))
        with self.assertRaises(IdentityMismatchError):  # 一行 episode 不符
            bind_window_to_observation(snapshot, make_obs_rows(
                env_ids=[1, 0], episode_ids=[11, 12], step_ids=[9, 5], time_s=[0.32, 0.32]))

    def test_envelope_label_supports_reordered_logical_batches(self):
        observation = make_obs_rows(
            env_ids=[7, 3], episode_ids=[11, 11], step_ids=[5, 9], time_s=[0.3, 0.3])
        observation.validate()  # 任意非负环境身份（非 arange）
        aligned = make_envelope_label_rows(
            env_ids=[7, 3], episode_ids=[11, 11], step_ids=[5, 9])
        self.assertIs(bind_envelope_label(observation, aligned), aligned)
        with self.assertRaises(IdentityMismatchError):  # 行序错位：按行配对而非集合匹配
            bind_envelope_label(observation, make_envelope_label_rows(
                env_ids=[3, 7], episode_ids=[11, 11], step_ids=[5, 9]))
        with self.assertRaises(IdentityMismatchError):  # 未来控制步标签
            bind_envelope_label(observation, make_envelope_label_rows(
                env_ids=[7, 3], episode_ids=[11, 11], step_ids=[5, 10]))


class ContractTests(unittest.TestCase):
    def test_online_observation_layout_and_validation(self):
        observation = make_obs()
        vector = observation.proprio_vector()
        self.assertEqual(tuple(vector.shape), (2, PROPRIO_DIM))
        self.assertTrue(torch.equal(vector[:, 6:9], observation.gravity_b))
        self.assertTrue(torch.equal(vector[:, 45:63], observation.prev_action))
        self.assertEqual(tuple(observation.command.shape), (2, COMMAND_DIM))
        observation.command = observation.command + 1.0  # 指令不进入本体向量
        self.assertTrue(torch.equal(observation.proprio_vector(), vector))
        expected_fields = {"env_ids", "episode_ids", "step_ids", "time_s", *PROPRIO_LAYOUT, "command"}
        self.assertEqual(set(OnlineObservation.__dataclass_fields__), expected_fields)  # 无标签字段
        bad = make_obs()
        bad.gravity_b[0, 2] = float("nan")
        with self.assertRaises(ContractError):
            bad.validate()
        wrong_shape = make_obs()
        wrong_shape.joint_vel = torch.zeros(2, 17)
        with self.assertRaises(ContractError):
            wrong_shape.validate()
        wrong_dtype = make_obs()
        wrong_dtype.command = wrong_dtype.command.double()
        with self.assertRaises(ContractError):
            wrong_dtype.validate()

    def test_label_invalid_rows_are_zero_filled_and_isolated(self):
        label = make_envelope_label()
        label.valid = torch.tensor([True, False])
        label.values = label.values.clone()
        label.values[1] = 0.0
        label.validate()
        label.values[1, 0] = 0.5  # 无效行非零拒绝
        with self.assertRaises(ContractError):
            label.validate()
        label.values[1, 0] = 0.0
        label.values[0, 0] = float("nan")  # 有效行 NaN 拒绝
        with self.assertRaises(ContractError):
            label.validate()
        ground = make_ground_label()
        ground.valid[:, 2] = False
        ground.heights = ground.heights.clone()
        ground.heights[:, 2] = 0.0
        ground.validate()
        ground.heights[0, 2] = float("inf")  # 无效格 Inf 拒绝
        with self.assertRaises(ContractError):
            ground.validate()

    def test_snapshot_structural_invariants(self):
        window = make_window()
        write_scan(window, env=0, episode=11, scan_id=1, time_s=0.1)
        write_scan(window, env=0, episode=11, scan_id=2, time_s=0.2)
        snapshot = window.read()
        holed = clone_snapshot(snapshot)
        holed.far_filled[0, 0] = False  # 前缀空洞拒绝
        with self.assertRaises(ContractError):
            holed.validate()
        unordered = clone_snapshot(snapshot)
        unordered.far_scan_ids[0, 1] = 1  # 递增破坏拒绝
        with self.assertRaises(ContractError):
            unordered.validate()
        stale_near = clone_snapshot(snapshot)
        stale_near.near_filled[0] = False  # 未填充近端槽残留有效数据拒绝
        with self.assertRaises(ContractError):
            stale_near.validate()

    def test_snapshot_near_far_association_enforced(self):
        window = make_window()
        write_scan(window, env=0, episode=11, scan_id=1, time_s=0.1)
        write_scan(window, env=0, episode=11, scan_id=2, time_s=0.2)
        snapshot = window.read()  # near 与远端最新槽同步为 scan 2 @ 0.2
        time_drift = clone_snapshot(snapshot)
        time_drift.far_times[0, 1] = 0.5  # 远端最新时间与近端脱节（近端 0.2）
        with self.assertRaises(ContractError):
            time_drift.validate()
        id_drift = clone_snapshot(snapshot)
        id_drift.far_scan_ids[0, 1] = 99  # 远端最新 scan_id 与近端脱节
        with self.assertRaises(ContractError):
            id_drift.validate()
        orphan_near = clone_snapshot(snapshot)  # 近端有扫描但远端全空
        orphan_near.far_points[0].zero_()
        orphan_near.far_valid[0].fill_(False)
        orphan_near.far_scan_ids[0].fill_(EMPTY_SCAN_ID)
        orphan_near.far_times[0].zero_()
        orphan_near.far_filled[0].fill_(False)
        with self.assertRaises(ContractError):
            orphan_near.validate()
        duplicate_env = clone_snapshot(snapshot)
        duplicate_env.env_ids[1] = 0  # 快照每环境至多一行
        with self.assertRaises(ContractError):
            duplicate_env.validate()

    def test_package_is_pure_framework(self):
        import sys

        loaded = set(sys.modules)
        self.assertFalse(any(name == "mjlab" or name.startswith("mjlab.") for name in loaded))
        self.assertFalse(any(name.startswith("hexapod_mjlab.tasks") for name in loaded))


if __name__ == "__main__":
    unittest.main(verbosity=2)
