"""CPU contracts for lidar configuration and pure scan selection math."""
from __future__ import annotations

import math
import traceback

import torch

from hexapod_mjlab.tasks.confined_locomotion.scanner import (
    ConfinedLidarCfg,
    ScanCoverageError,
    assert_mesh_covers_sensor,
    direction_angle_keys,
    farthest_point_sampling_indices,
    partition_cone_forward_x,
    sensor_frame_angle_keys,
    uniform_angular_sample_indices,
)


def test_lidar_configuration_is_explicit_and_noise_free():
    cfg = ConfinedLidarCfg()
    assert (cfg.airy_n_azimuth, cfg.airy_n_elevation) == (900, 96)
    assert cfg.airy_horizontal_resolution_deg == 0.4
    assert cfg.pool_azimuth_stride == 1
    assert (cfg.airy_n_azimuth // cfg.pool_azimuth_stride) * cfg.airy_n_elevation == 86400
    assert (cfg.near_points, cfg.far_points_per_scan, cfg.far_window_slots) == (256, 64, 10)
    assert cfg.max_range_m == 15.0 and cfg.scan_frequency_hz == 10.0
    assert (cfg.split_theta_deg, cfg.split_axis_tilt_deg) == (28.0, 56.0)
    assert cfg.split_max_forward_x_m == 3.5
    legacy = ConfinedLidarCfg(
        split_theta_deg=32.0, split_axis_tilt_deg=56.0,
        split_max_forward_x_m=None,
    )
    assert legacy.split_max_forward_x_m is None
    assert cfg.enable_sensor_noise is False
    assert cfg.pixel_dropout_prob == 0.0 and cfg.pixel_std_dev_multiplier == 0.0
    assert cfg.expected_plane_friction == (1.0, 0.005, 0.0001)
    for stride in (0, -1, True, 1.5, 7):
        try:
            ConfinedLidarCfg(pool_azimuth_stride=stride)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid pool stride accepted: {stride!r}")
    for bad in (-1.0, 0, float("nan"), float("inf"), True, "3.5"):
        try:
            ConfinedLidarCfg(split_max_forward_x_m=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid forward cutoff accepted: {bad!r}")
    for bad in (0, 90, -1, float("nan"), float("inf"), True):
        try:
            ConfinedLidarCfg(split_theta_deg=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid cone theta accepted: {bad!r}")
    for kwargs in ({"enable_sensor_noise": True}, {"pixel_dropout_prob": 0.1},
                   {"pixel_std_dev_multiplier": 1.0}):
        try:
            ConfinedLidarCfg(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"noise setting accepted: {kwargs!r}")


def test_B8_near_far_partition_respects_body_x_not_range_and_p6_gap():
    # All beams point INTO the 62-degree half-angle cone, but only consumed
    # BASE-x positions at or before the 3.5m plane belong to near.
    angle = torch.tensor([1., 1., 1., 1., 0., 1.], dtype=torch.float32)
    pts = torch.tensor([
        [3.49, 2.0, -1.0], [3.50, 0., 0.],
        [3.51, 0., 0.], [0., 9., -1.],
        [1., 0., 0.], [15., 0., 0.],
    ])
    near, far = partition_cone_forward_x(
        angle, pts, theta_deg=28., forward_x_max_m=3.5
    )
    assert near.tolist() == [True, True, False, True, False, False]
    assert torch.equal(near | far, torch.ones(6, dtype=torch.bool))
    assert not bool((near & far).any())

    valid = torch.tensor([True, False, True, True, True, True])
    n, f = partition_cone_forward_x(
        angle, pts, theta_deg=28., forward_x_max_m=3.5, valid=valid
    )
    assert n.tolist() == [True, False, False, True, False, False]
    assert torch.equal(n | f, valid)
    assert not bool((n & f).any())
    # Endpoint distance d_obs=0 has a meaningless xyz direction; the cone
    # alignment is ORIGINAL ray direction, so a zero-distance hit at mount x
    # remains near, and a far-ray at the same endpoint remains far.
    near0, far0 = partition_cone_forward_x(
        torch.tensor([1., 0.], dtype=torch.float32),
        torch.tensor([[0.645, 0., 0.], [0.645, 0., 0.]]),
        theta_deg=28., forward_x_max_m=3.5
    )
    assert near0.tolist() == [True, False] and far0.tolist() == [False, True]
    # No cutoff reproduces the angular-only group exactly.
    legacy_near, legacy_far = partition_cone_forward_x(
        angle, pts, theta_deg=32., forward_x_max_m=None
    )
    assert legacy_near.tolist() == [True, True, True, True, False, True]
    assert torch.equal(legacy_far, ~legacy_near)


def test_scan_selection_never_repeats_and_handles_short_or_empty_groups():
    generator = torch.Generator().manual_seed(17)
    points = torch.randn(300, 3, generator=generator) * 3.0
    group = torch.zeros(300, dtype=torch.bool)
    group[:100] = True

    indices, count = farthest_point_sampling_indices(points, group, 256)
    assert count == 100 and indices.unique().numel() == 100
    assert bool(group[indices].all())

    indices, count = farthest_point_sampling_indices(points, group, 5)
    assert count == 5 and indices.unique().numel() == 5

    tied = torch.cat([torch.zeros(50, 3), torch.randn(50, 3, generator=generator)])
    indices, count = farthest_point_sampling_indices(tied, torch.ones(100, dtype=torch.bool), 10)
    assert count == 10 and indices.unique().numel() == 10

    indices, count = farthest_point_sampling_indices(points, torch.zeros(300, dtype=torch.bool), 10)
    assert count == 0 and indices.numel() == 0

    keys = torch.rand(2880, generator=generator)
    group = torch.zeros(2880, dtype=torch.bool)
    group[:1000] = True
    indices, count = uniform_angular_sample_indices(keys, group, 64)
    assert count == 64 and indices.unique().numel() == 64
    assert bool((keys[indices[1:]] >= keys[indices[:-1]]).all())

    short = torch.zeros_like(group)
    short[:10] = True
    indices, count = uniform_angular_sample_indices(keys, short, 64)
    assert count == 10 and indices.unique().numel() == 10
    assert set(indices.tolist()) == set(range(10))

    empty, count = uniform_angular_sample_indices(keys, torch.zeros_like(group), 8)
    assert count == 0 and empty.numel() == 0


def test_sensor_frame_angle_keys_are_finite_at_poles_and_follow_mount():
    mount_pos = torch.tensor([0.644974, 0.0, -0.164512])
    mount_q = torch.tensor([0.0, 0.0, 0.0, 1.0])
    points = torch.tensor([[1.0, 2.0, -3.0], [0.0, 0.5, 0.5]])
    keys = sensor_frame_angle_keys(points, mount_pos, mount_q)
    rel = points - mount_pos
    expected = (
        torch.atan2(rel[:, 2], torch.hypot(rel[:, 0], rel[:, 1])) * (2.0 * math.pi)
        + torch.atan2(rel[:, 1], rel[:, 0])
    )
    assert torch.allclose(keys, expected, atol=1e-6)

    poles = mount_pos + torch.tensor([[0.0, 0.0, 5.0], [0.0, 0.0, -5.0]])
    pole_keys = sensor_frame_angle_keys(poles, mount_pos, mount_q)
    assert bool(torch.isfinite(pole_keys).all())
    assert abs(float(pole_keys[0]) - math.pi * math.pi) < 1e-6
    assert abs(float(pole_keys[1]) + math.pi * math.pi) < 1e-6

    # A zero-distance consumed endpoint has no direction of its own. The
    # source ray's direction key remains the independent angular identity.
    directions = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    direction_keys = direction_angle_keys(directions)
    independent_keys = (
        torch.atan2(directions[:, 2], torch.hypot(directions[:, 0], directions[:, 1]))
        * (2.0 * math.pi)
        + torch.atan2(directions[:, 1], directions[:, 0])
    )
    assert torch.allclose(direction_keys, independent_keys, atol=1e-7)
    zero_endpoint = directions[1] * 0.0
    assert torch.equal(zero_endpoint, torch.zeros(3))
    endpoint_key = sensor_frame_angle_keys(zero_endpoint.reshape(1, 3), torch.zeros(3), mount_q)
    assert torch.allclose(direction_keys[1], torch.tensor(math.pi / 2), atol=1e-7)
    assert torch.allclose(endpoint_key[0], torch.tensor(0.0), atol=1e-7)
    assert direction_keys[1] != endpoint_key[0]


def test_sensor_mesh_coverage_rejects_reach_outside_compiled_bounds():
    try:
        assert_mesh_covers_sensor((-10.0, 10.0, -10.0, 10.0), (0.0, 0.0), 15.0)
    except ScanCoverageError:
        pass
    else:
        raise AssertionError("a mesh smaller than the 15 m ray reach must be rejected")

    assert_mesh_covers_sensor((-100.0, 100.0, -100.0, 100.0), (0.0, 0.0), 15.0)
    try:
        assert_mesh_covers_sensor((-100.0, 100.0, -100.0, 100.0), (90.0, 0.0), 15.0)
    except ScanCoverageError:
        pass
    else:
        raise AssertionError("sensor origin outside mesh must be rejected")


def run_cpu_checks() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} CPU checks passed")
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(run_cpu_checks())

