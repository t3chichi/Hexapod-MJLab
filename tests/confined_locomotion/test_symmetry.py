"""CPU checks for pure mirror mappings and their independent references."""

from __future__ import annotations

import math
import traceback
import unittest

import torch

from hexapod_mjlab.rl.confined_locomotion.contracts import OnlineObservation
from hexapod_mjlab.rl.confined_locomotion.symmetry import (
    GridPermutation,
    JointPermutation,
    MirrorMappingError,
    mirror_action_distribution,
    mirror_axial_vector,
    mirror_command,
    mirror_envelope_values,
    mirror_ground_positions,
    mirror_ground_values,
    mirror_mean_mse,
    mirror_observation,
    mirror_point_block,
    mirror_polar_vector,
    mirror_proprio63,
)


_JOINT_NAMES = (
    "RF_HAA", "RF_HFE", "RF_KFE",
    "RM_HAA", "RM_HFE", "RM_KFE",
    "RB_HAA", "RB_HFE", "RB_KFE",
    "LF_HAA", "LF_HFE", "LF_KFE",
    "LM_HAA", "LM_HFE", "LM_KFE",
    "LB_HAA", "LB_HFE", "LB_KFE",
)


def _observation() -> OnlineObservation:
    batch = 2
    values = torch.arange(batch * 18, dtype=torch.float32).reshape(batch, 18)
    return OnlineObservation(
        env_ids=torch.tensor([4, 1], dtype=torch.int64),
        episode_ids=torch.tensor([3, 8], dtype=torch.int64),
        step_ids=torch.tensor([12, 91], dtype=torch.int64),
        time_s=torch.tensor([0.24, 1.82], dtype=torch.float32),
        lin_vel_b=torch.tensor([[1.0, 2.0, 3.0], [-4.0, 5.0, 6.0]]),
        ang_vel_b=torch.tensor([[7.0, 8.0, 9.0], [-1.0, 2.0, -3.0]]),
        gravity_b=torch.tensor([[0.1, -0.2, -0.9], [0.3, 0.4, -0.8]]),
        joint_pos_rel_ref=values.clone(),
        joint_vel=values + 40.0,
        prev_action=values + 80.0,
        command=torch.tensor([[0.5, -0.3, 0.7], [-0.2, 0.4, -0.6]]),
    )


def _log_prob(raw: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (-0.5 * ((raw - mean) / std).square() - std.log() - 0.5 * math.log(2 * math.pi)).sum(-1)


def test_joint_permutation_is_name_based_same_sign_and_reversible_for_shuffled_names():
    shuffled = ("LM_KFE", "RB_HFE", "RF_KFE", "LB_HAA", "LF_HFE", "RM_KFE",
               "RB_KFE", "LF_HAA", "RM_HAA", "RF_HFE", "LM_HAA", "RB_HAA",
               "LB_HFE", "RM_HFE", "RF_HAA", "LB_KFE", "LF_KFE", "LM_HFE")
    permutation = JointPermutation.from_joint_names(shuffled)
    action = torch.arange(36, dtype=torch.float32).reshape(2, 18)
    mirrored = permutation.apply(action)

    assert torch.equal(permutation.apply(mirrored), action)
    assert torch.equal(mirrored[0], action[0, torch.tensor(permutation.indices)])
    assert all(shuffled[permutation.indices[i]].split("_")[1] == shuffled[i].split("_")[1]
               for i in range(18))
    assert all(shuffled[permutation.indices[i]].split("_")[0] != shuffled[i].split("_")[0]
               for i in range(18))
    with unittest.TestCase().assertRaisesRegex(MirrorMappingError, "missing mirror joint"):
        JointPermutation.from_joint_names(_JOINT_NAMES[:-1] + ("LB_ANKLE",))


def test_observation_maps_polar_axial_joint_and_command_fields_without_aliasing():
    permutation = JointPermutation.from_joint_names(_JOINT_NAMES)
    observation = _observation()
    proprio = observation.proprio_vector()
    mirrored = mirror_observation(observation, permutation)
    mapped_proprio = mirror_proprio63(proprio, permutation)

    assert torch.equal(mapped_proprio, mirrored.proprio_vector())
    assert torch.equal(mirrored.lin_vel_b, mirror_polar_vector(observation.lin_vel_b))
    assert torch.equal(mirrored.gravity_b, mirror_polar_vector(observation.gravity_b))
    assert torch.equal(mirrored.ang_vel_b, mirror_axial_vector(observation.ang_vel_b))
    assert torch.equal(mirrored.command, mirror_command(observation.command))
    assert torch.equal(mirrored.env_ids, observation.env_ids)
    assert torch.equal(mirrored.episode_ids, observation.episode_ids)
    assert torch.equal(mirrored.step_ids, observation.step_ids)
    assert torch.equal(mirrored.time_s, observation.time_s)
    assert torch.equal(mirror_observation(mirrored, permutation).proprio_vector(), proprio)
    assert torch.equal(mirror_observation(mirrored, permutation).command, observation.command)
    assert mirrored.joint_pos_rel_ref.data_ptr() != observation.joint_pos_rel_ref.data_ptr()


def test_asymmetric_old_distribution_pushforward_keeps_probability_invariant():
    permutation = JointPermutation.from_joint_names(_JOINT_NAMES)
    mean = torch.arange(36, dtype=torch.float32).reshape(2, 18) / 13.0
    std = torch.linspace(0.2, 2.4, 36, dtype=torch.float32).reshape(2, 18)
    raw = torch.flip(mean, dims=(1,)) + torch.linspace(-0.9, 0.7, 18)
    mapped = mirror_action_distribution(mean, std, raw, permutation)

    assert torch.equal(mapped.mean, permutation.apply(mean))
    assert torch.equal(mapped.std, permutation.apply(std))
    assert torch.equal(mapped.raw_action, permutation.apply(raw))
    assert torch.allclose(_log_prob(raw, mean, std), _log_prob(mapped.raw_action, mapped.mean, mapped.std))
    assert torch.equal(mirror_action_distribution(
        mapped.mean, mapped.std, mapped.raw_action, permutation
    ).raw_action, raw)


def test_point_cloud_reflects_optical_rays_and_preserves_mask_history_and_double_map():
    # A 90-degree sensor tilt makes -old-angle an incorrect shortcut: +sensor-x
    # becomes body +y, then body reflection maps it to sensor -x.
    half = math.sqrt(0.5)
    mount_quat = torch.tensor([0.0, 0.0, half, half], dtype=torch.float32)
    raw_points = torch.tensor([[[0.2, 0.4, 0.1], [0.0, 0.0, 0.0],
                                [0.3, 0.2, 0.8], [-0.1, 0.5, -0.7],
                                [0.0, 0.0, 0.0], [0.4, -0.2, 0.0],
                                [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    raw_valid = torch.tensor([[True, False, True, True, False, True, True, False]])
    raw_dirs = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                              [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
                              [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
                              [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    raw_scan_ids = torch.tensor([[12, -1, 12, 12, -1, 13, 13, -1]], dtype=torch.int64)
    raw_times = torch.tensor([[0.2, 0.0, 0.2, 0.2, 0.0, 0.3, 0.3, 0.0]])
    raw_source_ids = torch.tensor([[10, -1, 20, 21, -1, 30, 31, -1]], dtype=torch.int64)

    # Prepare the original optical order exactly as the F1 stable key sort does.
    phi = torch.atan2(raw_dirs[..., 2], torch.hypot(raw_dirs[..., 0], raw_dirs[..., 1]))
    azimuth = torch.atan2(raw_dirs[..., 1], raw_dirs[..., 0])
    raw_keys = phi * (2.0 * math.pi) + azimuth
    raw_keys = torch.where(raw_valid, raw_keys, torch.full_like(raw_keys, float("inf")))
    base_order = torch.argsort(raw_keys, dim=1, stable=True)
    points = torch.gather(raw_points, 1, base_order.unsqueeze(-1).expand_as(raw_points))
    valid = torch.gather(raw_valid, 1, base_order)
    dirs = torch.gather(raw_dirs, 1, base_order.unsqueeze(-1).expand_as(raw_dirs))
    scan_ids = torch.gather(raw_scan_ids, 1, base_order)
    times = torch.gather(raw_times, 1, base_order)
    source_ids = torch.gather(raw_source_ids, 1, base_order)
    order_rank = torch.arange(points.shape[1], dtype=torch.int64).view(1, -1)

    mirrored = mirror_point_block(
        points, valid, dirs, mount_quat,
        scan_ids=scan_ids, times=times, source_indices=source_ids,
        source_order=order_rank,
    )
    assert torch.equal(mirrored.valid, torch.tensor([[True, True, True, True, True, False, False, False]]))
    assert torch.equal(mirrored.points[~mirrored.valid], torch.zeros_like(mirrored.points[~mirrored.valid]))
    assert bool(torch.isfinite(mirrored.angle_keys).all())
    valid_count = int(mirrored.valid.sum())
    assert bool((mirrored.angle_keys[:, :valid_count - 1]
                 <= mirrored.angle_keys[:, 1:valid_count]).all())
    assert torch.equal(mirrored.scan_ids, torch.gather(scan_ids, 1, mirrored.permutation))
    assert torch.equal(mirrored.times, torch.gather(times, 1, mirrored.permutation))
    assert torch.equal(mirrored.source_indices, torch.gather(source_ids, 1, mirrored.permutation))

    # The zero-coordinate valid point still gets the reflected source-ray key.
    zero_source_column = int(torch.nonzero(source_ids[0] == 31, as_tuple=False)[0, 0])
    zero_output_column = int(torch.nonzero(mirrored.source_indices[0] == 31, as_tuple=False)[0, 0])
    assert torch.equal(points[0, zero_source_column], torch.zeros(3))
    zero_key = mirrored.angle_keys[0, zero_output_column].item()
    assert math.isclose(abs(zero_key), math.pi, abs_tol=2e-6)

    twice = mirror_point_block(
        mirrored.points, mirrored.valid, mirrored.source_dirs_sensor, mount_quat,
        scan_ids=mirrored.scan_ids, times=mirrored.times,
        source_indices=mirrored.source_indices, source_order=mirrored.source_order,
    )
    assert torch.equal(twice.points, points)
    assert torch.equal(twice.valid, valid)
    assert torch.allclose(twice.source_dirs_sensor, dirs, atol=2e-6, rtol=0)
    assert torch.equal(twice.scan_ids, scan_ids)
    assert torch.equal(twice.times, times)
    assert torch.equal(twice.source_indices, source_ids)
    assert torch.equal(twice.source_order, order_rank)


def test_point_cloud_rejects_missing_direction_contracts_and_bad_mount():
    p = torch.zeros((1, 2, 3), dtype=torch.float32)
    valid = torch.tensor([[True, False]])
    dirs = torch.tensor([[[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    q = torch.tensor([0.0, 0.0, 0.0, 1.0])
    with unittest.TestCase().assertRaisesRegex(MirrorMappingError, "unit sensor-frame"):
        mirror_point_block(p, valid, dirs, q)
    dirs[0, 0] = torch.tensor([1.0, 0.0, 0.0])
    with unittest.TestCase().assertRaisesRegex(MirrorMappingError, "unit length"):
        mirror_point_block(p, valid, dirs, torch.tensor([0.0, 0.0, 0.0, 2.0]))
    bad_points = p.clone()
    bad_points[0, 1, 0] = 1.0
    with unittest.TestCase().assertRaisesRegex(MirrorMappingError, "exactly zero"):
        mirror_point_block(bad_points, valid, dirs, q)


def test_near_pole_optical_key_uses_stable_householder_for_pitch_and_actual_mount():
    def rotate_xyzw(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        xyz, w = quat[:3], quat[3]
        uv = torch.cross(xyz, vector, dim=-1)
        return vector + 2.0 * (w * uv + torch.cross(xyz, uv, dim=-1))

    def expected_mirror_direction(direction: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
        # Independent FP64 analytic R^-1 Q R expectation. Normalize only the
        # numerical quaternion/direction representation, not any ray component.
        q = quat.to(torch.float64)
        q = q / torch.linalg.vector_norm(q)
        conjugate = torch.cat((-q[:3], q[3:4]))
        normal = rotate_xyzw(conjugate, torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64))
        normal = normal / torch.linalg.vector_norm(normal)
        reflection = torch.eye(3, dtype=torch.float64) - 2.0 * torch.outer(normal, normal)
        return reflection @ direction.to(torch.float64)

    def key(direction: torch.Tensor) -> float:
        x, y, z = (float(value) for value in direction)
        return math.atan2(z, math.hypot(x, y)) * 2.0 * math.pi + math.atan2(y, x)

    direction = torch.tensor([1e-8, 2e-8, 1.0], dtype=torch.float64)
    direction = direction / torch.linalg.vector_norm(direction)
    pitch_quat = torch.tensor(
        [0.0, math.sin(0.05), 0.0, math.cos(0.05)], dtype=torch.float32
    )
    points = torch.zeros((1, 1, 3), dtype=torch.float32)
    valid = torch.ones((1, 1), dtype=torch.bool)
    dirs = direction.to(torch.float32).reshape(1, 1, 3)

    pitch_mapped = mirror_point_block(points, valid, dirs, pitch_quat)
    pitch_expected = expected_mirror_direction(direction, pitch_quat)
    assert math.isclose(float(pitch_mapped.source_dirs_sensor[0, 0, 0]), 1e-8,
                        rel_tol=0, abs_tol=1e-12)
    assert abs(float(pitch_mapped.angle_keys[0, 0]) - key(pitch_expected)) <= 1e-5

    # Use the compiled lidar_sensor site and its runtime FP32 quaternion, as the
    # scanner does. The actual optical mount has tiny quaternion component
    # asymmetries that FP32 collapses; this near-pole witness exercises that
    # real pipeline precision without running a simulator or changing scans.
    from hexapod_mjlab.tasks.el_4090_lidar.mdp.lidar import resolve_lidar_mount

    resolved = resolve_lidar_mount()
    assert resolved is not None
    actual_quat = torch.tensor(resolved[1], dtype=torch.float32)
    actual_mapped = mirror_point_block(points, valid, dirs, actual_quat)
    actual_expected = expected_mirror_direction(direction, actual_quat)
    assert abs(float(actual_mapped.angle_keys[0, 0]) - key(actual_expected)) <= 1e-5


def test_grid_mapping_uses_metadata_pairs_values_masks_and_positions():
    grid = GridPermutation.from_axes(x_coords=(-0.4, 0.0, 0.6), y_coords=(-1.0, 0.0, 1.0))
    assert grid.source_indices == (6, 7, 8, 3, 4, 5, 0, 1, 2)
    valid = torch.tensor([[True, False, True, True, True, False, True, True, True],
                          [False, True, True, True, False, True, True, True, True]])
    heights = torch.arange(18, dtype=torch.float32).reshape(2, 9)
    heights = torch.where(valid, heights, torch.zeros_like(heights))
    mapped_heights, mapped_valid = mirror_ground_values(heights, valid, grid)
    twice_heights, twice_valid = mirror_ground_values(mapped_heights, mapped_valid, grid)
    assert torch.equal(twice_heights, heights)
    assert torch.equal(twice_valid, valid)

    coords = []
    for y in (-1.0, 0.0, 1.0):
        for x in (-0.4, 0.0, 0.6):
            coords.append((x, y, -0.3 + 0.1 * x + 0.02 * y))
    positions = torch.tensor(coords, dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1)
    positions = torch.where(valid.unsqueeze(-1), positions, torch.zeros_like(positions))
    mapped_positions, mapped_position_valid = mirror_ground_positions(positions, valid, grid)
    twice_positions, twice_position_valid = mirror_ground_positions(
        mapped_positions, mapped_position_valid, grid
    )
    assert torch.allclose(twice_positions, positions, atol=1e-7, rtol=0)
    assert torch.equal(twice_position_valid, valid)
    assert torch.allclose(mapped_positions[0, 0], torch.tensor([-0.4, -1.0, -0.32]), atol=1e-7, rtol=0)
    assert torch.equal(mirror_envelope_values(torch.ones((2, 6))), torch.ones((2, 6)))

    mismatched = positions.clone()
    mismatched[0, 0, 0] += 0.01
    with unittest.TestCase().assertRaisesRegex(MirrorMappingError, "grid coordinate metadata"):
        mirror_ground_positions(mismatched, valid, grid)


def test_grid_rejects_nonpaired_or_float32_collapsed_metadata():
    with unittest.TestCase().assertRaisesRegex(MirrorMappingError, "not paired"):
        GridPermutation.from_axes((0.0, 1.0), (-1.0, 0.0, 2.0))
    with unittest.TestCase().assertRaisesRegex(MirrorMappingError, "strictly increasing"):
        GridPermutation.from_axes((0.0, 1e-50), (0.0,))

    with unittest.TestCase().assertRaisesRegex(MirrorMappingError, "match reflected coordinate"):
        GridPermutation((-1.0, 1.0), (-1.0, 1.0), (0, 3, 2, 1))


def test_mirror_mean_mse_uses_detached_target_and_batch_times_18_mean():
    permutation = JointPermutation.from_joint_names(_JOINT_NAMES)
    original = torch.arange(36, dtype=torch.float32).reshape(2, 18).requires_grad_()
    mirrored = torch.zeros((2, 18), dtype=torch.float32, requires_grad=True)
    expected_target = permutation.apply(original.detach())
    expected = expected_target.square().mean()

    loss = mirror_mean_mse(mirrored, original, permutation)
    assert torch.equal(loss, expected)
    loss.backward()
    assert original.grad is None
    assert torch.allclose(mirrored.grad, -2.0 * expected_target / (2 * 18), atol=2e-7, rtol=0)

    wrong_indices = tuple(range(18))
    with unittest.TestCase().assertRaisesRegex(MirrorMappingError, "named left/right"):
        JointPermutation(_JOINT_NAMES, wrong_indices)


def main() -> int:
    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print("PASS", name, flush=True)
        except BaseException as err:  # noqa: BLE001 — report all independent checks
            failures.append(name)
            print("FAIL", name, repr(err), flush=True)
            traceback.print_exc()
    print(f"{len(tests) - len(failures)}/{len(tests)} passed", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
