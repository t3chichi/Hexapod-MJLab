#!/usr/bin/env python
"""Real-grid optical left/right mapping and persistent window rotation checks."""
from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import torch

from _common import add_common_arguments, close_runtime, execute_gpu_check


MODES = ("mapping", "window_rotation")
WINDOW_ROTATION_NEW_SCANS = 31
WINDOW_ROTATION_MAX_CONTROLS = 180
RAW_RAYS_PER_SCAN = 86_400


def _args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--mode", required=True, choices=MODES)
  add_common_arguments(parser)
  return parser.parse_args(argv)


def _build_runtime(out: Path, *, args, grid):
  from confined_locomotion.support.fixtures import build_runtime

  return build_runtime(
    out, mode="engineering", device=f"cuda:{args.gpu}", seed=args.seed,
    num_envs=args.num_envs, ground_grid=grid,
  )


def _rotation_xyzw(quaternion: torch.Tensor) -> torch.Tensor:
  """Independent float64 quaternion rotation matrix, xyzw input order."""
  x, y, z, w = [float(item) for item in quaternion.detach().cpu().to(torch.float64)]
  return torch.tensor([
    [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
    [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
    [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
  ], dtype=torch.float64)


def _independent_mirror_points(points, valid, directions, mount_quat):
  """Recompute Q reflection, optical directions, angular key and stable sort."""
  q = mount_quat.detach().cpu()
  rotation = _rotation_xyzw(q)
  reflection = torch.diag(torch.tensor([1.0, -1.0, 1.0], dtype=torch.float64))
  xform = rotation.T @ reflection @ rotation
  pts64 = points.detach().cpu().to(torch.float64)
  dirs64 = directions.detach().cpu().to(torch.float64)
  reflected_points = pts64.clone()
  reflected_points[..., 1] *= -1.0
  reflected_dirs = torch.einsum("ij,bnj->bni", xform, dirs64)
  xyz = reflected_dirs
  azimuth = torch.atan2(xyz[..., 1], xyz[..., 0])
  elevation = torch.atan2(xyz[..., 2], torch.hypot(xyz[..., 0], xyz[..., 1]))
  key = elevation * (2.0 * math.pi) + azimuth
  effective = torch.where(valid.detach().cpu(), key, torch.full_like(key, float("inf")))
  order = torch.argsort(effective, dim=-1, stable=True)
  gather_points = order[..., None].expand(*order.shape, 3)
  return {
    "points": torch.gather(reflected_points, 1, gather_points).to(points.dtype),
    "valid": torch.gather(valid.detach().cpu(), 1, order),
    "directions": torch.gather(reflected_dirs, 1, gather_points).to(directions.dtype),
    "indices": order,
    "keys": torch.gather(key, 1, order).to(points.dtype),
  }


def _mapping_checks(out: Path, device: str, recipe: dict, *, args, grid) -> dict:
  from confined_locomotion.support.snapshots import cpu_tree
  from hexapod_mjlab.rl.confined_locomotion.symmetry import (
    GridPermutation, JointPermutation, mirror_action_distribution,
    mirror_axial_vector, mirror_command, mirror_ground_positions,
    mirror_ground_values, mirror_observation, mirror_point_block,
    mirror_polar_vector,
  )
  from hexapod_mjlab.tasks.el_4090_lidar.mdp.lidar import resolve_lidar_mount

  env, wrapper, runtime = _build_runtime(out, args=args, grid=grid)
  checks = {}
  try:
    prepared = runtime.prepare()
    labels = runtime.labels()
    action = env.action_manager.get_term("residual_pd")
    joint = JointPermutation.from_joint_names(tuple(action._target_names))
    from hexapod_mjlab.tasks.el_4090_lidar.t0.witness import frozen_witness

    t1 = frozen_witness()
    t1_targets = torch.tensor(t1.joint_targets, dtype=torch.float32, device=device)
    actual_q_ref = action.q_ref.detach().clone()
    q_ref_error = float((actual_q_ref - t1_targets.expand_as(actual_q_ref)).abs().max())
    left_right_q_ref_delta = actual_q_ref[0] - joint.apply(actual_q_ref[0])
    checks["compiled_pd_reference_matches_frozen_t1_targets"] = (
      tuple(action._target_names) == tuple(t1.joint_names)
      and actual_q_ref.shape == (args.num_envs, 18)
      and q_ref_error <= 1e-7
    )
    checks["actual_t1_left_right_pd_asymmetry_is_preserved"] = (
      float(left_right_q_ref_delta.abs().max()) > 1e-4
    )
    index_by_name = {name: i for i, name in enumerate(action._target_names)}
    expected_indices = []
    for name in action._target_names:
      leg, joint_kind = name.split("_", 1)
      partner = {"RF": "LF", "LF": "RF", "RM": "LM", "LM": "RM", "RB": "LB", "LB": "RB"}[leg]
      expected_indices.append(index_by_name[f"{partner}_{joint_kind}"])
    checks["compiled_joint_names_form_same_sign_involution"] = (
      tuple(joint.indices) == tuple(expected_indices)
      and all(joint.indices[joint.indices[i]] == i for i in range(18))
    )
    mapping = GridPermutation.from_axes(grid.x_coords, grid.y_coords)
    expected_grid_indices = []
    # GridPermutation's contract is y-outer/x-inner, matching the runner's
    # registered flat-grid serialization order.
    coords = [(x, y) for y in grid.y_coords for x in grid.x_coords]
    for x, y in coords:
      expected_grid_indices.append(coords.index((x, -y)))
    checks["full_grid_positions_pair_at_same_x_and_negated_y"] = (
      tuple(mapping.source_indices) == tuple(expected_grid_indices)
      and len(set(mapping.source_indices)) == grid.num_points
    )

    mount = resolve_lidar_mount()
    if mount is None:
      raise AssertionError("actual compiled optical lidar site was not resolved")
    q = torch.tensor(mount[1], dtype=torch.float32, device=device)
    mount_pos = torch.tensor(mount[0], dtype=torch.float32, device=device)
    checks["mirror_mount_matches_actual_runtime_scanner"] = (
      torch.allclose(runtime.pipeline._mount_t[0], mount_pos, atol=1e-7, rtol=0.0)
      and torch.allclose(runtime.pipeline._mount_q[0], q, atol=1e-7, rtol=0.0)
    )

    observation = prepared.observation
    mirrored_observation = mirror_observation(observation, joint)
    expected_lin = torch.stack((observation.lin_vel_b[:, 0], -observation.lin_vel_b[:, 1], observation.lin_vel_b[:, 2]), dim=-1)
    expected_ang = torch.stack((-observation.ang_vel_b[:, 0], observation.ang_vel_b[:, 1], -observation.ang_vel_b[:, 2]), dim=-1)
    expected_command = torch.stack((observation.command[:, 0], -observation.command[:, 1], -observation.command[:, 2]), dim=-1)
    checks["polar_axial_joint_and_command_fields_map_independently"] = (
      torch.equal(mirrored_observation.lin_vel_b, expected_lin)
      and torch.equal(mirrored_observation.gravity_b, mirror_polar_vector(observation.gravity_b))
      and torch.equal(mirrored_observation.ang_vel_b, expected_ang)
      and torch.equal(mirrored_observation.command, expected_command)
      and torch.equal(mirrored_observation.joint_pos_rel_ref, joint.apply(observation.joint_pos_rel_ref))
      and torch.equal(mirrored_observation.env_ids, observation.env_ids)
    )
    checks["observation_map_is_involutive"] = (
      torch.equal(joint.apply(mirrored_observation.joint_pos_rel_ref), observation.joint_pos_rel_ref)
      and torch.equal(mirror_command(mirrored_observation.command), observation.command)
      and torch.equal(mirror_axial_vector(mirrored_observation.ang_vel_b), observation.ang_vel_b)
    )

    near_scan_ids = torch.where(
      prepared.near_valid,
      prepared.near_scan_ids[:, None].expand_as(prepared.near_valid),
      torch.full_like(prepared.near_source_indices, -1),
    )
    near_times = torch.where(
      prepared.near_valid,
      prepared.near_times[:, None].expand_as(prepared.near_valid),
      torch.zeros_like(prepared.near_times[:, None].expand_as(prepared.near_valid)),
    )
    near_input_order = torch.arange(256, device=device).expand(args.num_envs, -1)
    far_input_order = torch.arange(640, device=device).expand(args.num_envs, -1)
    near = mirror_point_block(
      prepared.near_points, prepared.near_valid,
      prepared.near_source_dirs_sensor, q,
      scan_ids=near_scan_ids, times=near_times,
      source_indices=prepared.near_source_indices,
      source_order=near_input_order,
    )
    near_expected = _independent_mirror_points(
      prepared.near_points, prepared.near_valid, prepared.near_source_dirs_sensor, q
    )
    far = mirror_point_block(
      prepared.far_points, prepared.far_valid,
      prepared.far_source_dirs_sensor, q,
      scan_ids=prepared.far_point_scan_ids,
      times=prepared.far_point_times,
      source_indices=prepared.far_source_indices,
      source_order=far_input_order,
    )
    far_expected = _independent_mirror_points(
      prepared.far_points, prepared.far_valid, prepared.far_source_dirs_sensor, q
    )
    checks["near_optical_reflection_and_stable_order_match_independent_q_reflection"] = (
      torch.allclose(near.points.cpu(), near_expected["points"], atol=2e-6, rtol=0.0)
      and torch.equal(near.valid.cpu(), near_expected["valid"])
      and torch.allclose(near.source_dirs_sensor.cpu(), near_expected["directions"], atol=2e-6, rtol=0.0)
      and torch.equal(near.source_indices.cpu(), torch.gather(prepared.near_source_indices.cpu(), 1, near_expected["indices"]))
      and torch.allclose(near.angle_keys.cpu(), near_expected["keys"], atol=1e-5, rtol=0.0)
    )
    checks["far_optical_reflection_preserves_source_history"] = (
      torch.allclose(far.points.cpu(), far_expected["points"], atol=2e-6, rtol=0.0)
      and torch.equal(far.valid.cpu(), far_expected["valid"])
      and torch.allclose(far.source_dirs_sensor.cpu(), far_expected["directions"], atol=2e-6, rtol=0.0)
      and torch.equal(far.source_indices.cpu(), torch.gather(prepared.far_source_indices.cpu(), 1, far_expected["indices"]))
      and torch.equal(far.scan_ids.cpu(), torch.gather(prepared.far_point_scan_ids.cpu(), 1, far_expected["indices"]))
      and torch.equal(far.times.cpu(), torch.gather(prepared.far_point_times.cpu(), 1, far_expected["indices"]))
    )
    twice = mirror_point_block(
      near.points, near.valid, near.source_dirs_sensor, q,
      scan_ids=near.scan_ids, times=near.times,
      source_indices=near.source_indices,
      source_order=near.source_order,
    )
    # Clean F1 sorts translated point geometry; mirror sorts source-ray angles.
    # Compare each original source after undoing the returned permutation.
    # The independent single-map check above separately enforces optical order.
    restore_order = torch.argsort(twice.source_order, dim=1, stable=True)
    def restore_source(value):
      index = restore_order[..., None].expand_as(value) if value.ndim == 3 else restore_order
      return torch.gather(value, 1, index)
    checks["double_mapping_preserves_each_original_source_point_mask_and_metadata"] = (
      torch.allclose(restore_source(twice.points), prepared.near_points, atol=2e-6, rtol=0.0)
      and torch.equal(restore_source(twice.valid), prepared.near_valid)
      and torch.equal(restore_source(twice.source_indices), prepared.near_source_indices)
      and torch.allclose(restore_source(twice.source_dirs_sensor), prepared.near_source_dirs_sensor, atol=2e-6, rtol=0.0)
      and torch.equal(restore_source(twice.scan_ids), near_scan_ids)
      and torch.equal(restore_source(twice.times), near_times)
      and torch.equal(restore_source(twice.source_order), near_input_order)
    )
    far_twice = mirror_point_block(
      far.points, far.valid, far.source_dirs_sensor, q,
      scan_ids=far.scan_ids, times=far.times,
      source_indices=far.source_indices, source_order=far.source_order,
    )
    checks["far_source_order_and_metadata_survive_double_mapping"] = (
      torch.allclose(far_twice.points, prepared.far_points, atol=2e-6, rtol=0.0)
      and torch.equal(far_twice.valid, prepared.far_valid)
      and torch.allclose(far_twice.source_dirs_sensor, prepared.far_source_dirs_sensor, atol=2e-6, rtol=0.0)
      and torch.equal(far_twice.source_indices, prepared.far_source_indices)
      and torch.equal(far_twice.scan_ids, prepared.far_point_scan_ids)
      and torch.equal(far_twice.times, prepared.far_point_times)
      and torch.equal(far_twice.source_order, far_input_order)
    )

    ground = labels["ground"]["label"]
    height, height_valid = mirror_ground_values(ground.heights, ground.valid, mapping)
    positions, positions_valid = mirror_ground_positions(ground.positions, ground.valid, mapping)
    expected_positions = ground.positions.index_select(
      1, torch.tensor(expected_grid_indices, dtype=torch.long, device=device)
    ).clone()
    expected_positions[..., 1] *= -1.0
    checks["ground_height_mask_and_xyz_map_share_full_grid_permutation"] = (
      torch.equal(height, ground.heights.index_select(1, torch.tensor(expected_grid_indices, device=device)))
      and torch.equal(height_valid, ground.valid.index_select(1, torch.tensor(expected_grid_indices, device=device)))
      and torch.equal(positions_valid, height_valid)
      and torch.allclose(positions, expected_positions, atol=2e-6, rtol=0.0)
    )

    # Asymmetric old per-joint distribution witness: the full raw old Normal
    # (mean, std, action) is permuted together and independently rescored.
    mean = torch.linspace(-1.2, 1.4, 18, device=device).view(1, 18).expand(args.num_envs, 18)
    std = torch.linspace(0.2, 2.2, 18, device=device).view(1, 18).expand(args.num_envs, 18)
    raw = mean + torch.linspace(-0.9, 0.7, 18, device=device).view(1, 18)
    mapped = mirror_action_distribution(mean, std, raw, joint)
    def log_prob(a, m, s):
      return (-0.5 * ((a - m) / s).square() - torch.log(s) - 0.5 * math.log(2.0 * math.pi)).sum(dim=-1)
    checks["asymmetric_old_raw_normal_probability_is_invariant"] = torch.allclose(
      log_prob(raw, mean, std), log_prob(mapped.raw_action, mapped.mean, mapped.std), atol=1e-5, rtol=1e-5
    )
    torch.save(cpu_tree({
      "joint_names": action._target_names,
      "joint_permutation": joint.indices,
      "grid_identity": grid.identity(),
      "grid_permutation": mapping.source_indices,
      "mount_pos": mount[0], "mount_quat_xyzw": mount[1],
      "prepared": prepared,
      "near_mirrored": near,
      "near_double_mirrored": twice,
      "near_double_restore_order": restore_order,
      "near_independent": near_expected,
      "far_mirrored": far,
      "far_independent": far_expected,
      "ground_label": ground,
      "actual_t1_q_ref": actual_q_ref,
      "t1_expected_targets": t1_targets,
      "left_right_q_ref_delta": left_right_q_ref_delta,
      "checks": checks,
    }), out / "symmetry_mapping_evidence.pt")
  finally:
    close_runtime(env, runtime)
  checks["passed"] = all(checks.values())
  from confined_locomotion.support.evidence import write_json

  write_json(out / "symmetry_mapping_checks.json", checks)
  return checks


def _tensor_sha256(value) -> str:
  tensor = value.detach().cpu().contiguous()
  digest = hashlib.sha256()
  digest.update(str(tensor.dtype).encode())
  digest.update(repr(tuple(tensor.shape)).encode())
  digest.update(tensor.numpy().tobytes())
  return digest.hexdigest()


def _selected_source_record(frame, *, scan_id: int, episode_id: int,
                            persistent_count: int, time_s: float) -> dict:
  """Retain selected ray IDs/directions and full-frame identity digests."""
  from confined_locomotion.support.snapshots import cpu_tree

  if tuple(frame.raw_ray_dirs.shape) != (RAW_RAYS_PER_SCAN, 3):
    raise AssertionError(
      f"actual source frame ray shape {tuple(frame.raw_ray_dirs.shape)} "
      f"!= ({RAW_RAYS_PER_SCAN}, 3)"
    )
  result = {
    "env_row": int(frame.env_row),
    "episode_id": int(episode_id),
    "scan_id": int(scan_id),
    "persistent_scan_count": int(persistent_count),
    "time_s": float(time_s),
    "raw_ray_count": int(frame.raw_ray_dirs.shape[0]),
    "raw_dirs_sha256": _tensor_sha256(frame.raw_ray_dirs),
    "raw_dists_sha256": _tensor_sha256(frame.raw_dists),
    "gap_fired": bool(frame.gap_fired),
  }
  for name in ("near", "far"):
    indices = getattr(frame, f"{name}_indices").detach().to(dtype=torch.long)
    valid = getattr(frame, f"{name}_valid").detach().to(dtype=torch.bool)
    if not torch.equal(indices >= 0, valid):
      raise AssertionError(f"{name} raw source IDs differ from selected valid mask")
    directions = torch.zeros(
      (*indices.shape, 3), dtype=frame.raw_ray_dirs.dtype,
      device=frame.raw_ray_dirs.device,
    )
    if bool(valid.any()):
      selected = indices[valid]
      if bool(((selected < 0) | (selected >= RAW_RAYS_PER_SCAN)).any()):
        raise AssertionError(f"{name} selected source ray is outside the full frame")
      directions[valid] = frame.raw_ray_dirs.index_select(0, selected)
    result[name] = {
      "indices": cpu_tree(indices),
      "valid": cpu_tree(valid),
      "source_dirs_sensor": cpu_tree(directions),
      "valid_count": int(valid.sum().item()),
    }
  return result


def _full_edge_frame(frame) -> dict:
  """Keep only first/last raw frames; all other scans use compact provenance."""
  from confined_locomotion.support.snapshots import cpu_tree

  fields = (
    "raw_ray_dirs", "raw_dists", "raw_points_sensor", "raw_points_base",
    "no_hit_mask", "near_indices", "near_valid", "near_points_base",
    "near_angle_keys", "far_indices", "far_valid", "far_points_base",
    "far_angle_keys", "sensor_pos_w", "sensor_quat_xyzw", "base_pos_w",
    "base_quat_wxyz",
  )
  return {
    "scan_id": int(frame.scan_id),
    "episode_id": int(frame.episode_id),
    "time_s": float(frame.time_s),
    "env_row": int(frame.env_row),
    "raw_ray_count": int(frame.raw_ray_dirs.shape[0]),
    **{name: cpu_tree(getattr(frame, name)) for name in fields},
  }


def _window_read_row_differences(before, after, *, row: int, num_envs: int) -> list[str]:
  """Compare one unselected row in every live and read-only window field."""
  from dataclasses import fields as dataclass_fields, is_dataclass
  from confined_locomotion.support.snapshots import tree_equal

  failures = []
  if set(before["ring"]["fields"]) != set(after["ring"]["fields"]):
    failures.append("ring.field_names")
  else:
    for name in sorted(before["ring"]["fields"]):
      left = before["ring"]["fields"][name]
      right = after["ring"]["fields"][name]
      if not (torch.is_tensor(left) and torch.is_tensor(right)
              and left.ndim > 0 and right.ndim > 0
              and left.shape[0] == num_envs and right.shape[0] == num_envs):
        failures.append(f"ring.{name}.shape")
      elif not torch.equal(left[row], right[row]):
        failures.append(f"ring.{name}.row{row}")
  if before["ring"]["seen_episodes"][row] != after["ring"]["seen_episodes"][row]:
    failures.append(f"ring.seen_episodes.row{row}")
  for section in ("chronological", "angle_keys", "source_directions"):
    left, right = before[section], after[section]
    if left is None or right is None:
      if left is not right:
        failures.append(f"{section}.presence")
      continue
    if is_dataclass(left) and is_dataclass(right):
      values = ((field.name, getattr(left, field.name), getattr(right, field.name))
                for field in dataclass_fields(left))
    elif isinstance(left, dict) and isinstance(right, dict) and set(left) == set(right):
      values = ((name, left[name], right[name]) for name in left)
    else:
      failures.append(f"{section}.structure")
      continue
    for name, a, b in values:
      if torch.is_tensor(a) and torch.is_tensor(b) and a.ndim > 0 and a.shape[0] == num_envs:
        differences = tree_equal(a[row], b[row], f"{section}.{name}.row{row}")
      else:
        differences = tree_equal(a, b, f"{section}.{name}")
      failures.extend(differences)
  return failures


def _window_rotation_checks(runtime, *, grid, num_envs: int,
                            initial_cursor, initial_counts,
                            source_records_by_key: dict) -> tuple[dict, dict]:
  """Check the real ring, source-ray identities and cursor-rotated reads."""
  from confined_locomotion.support.snapshots import (
    cpu_tree, live_window_snapshot,
  )

  checks = {}
  details = {}
  ring = live_window_snapshot(runtime.window)
  fields = ring["fields"]
  expected_shapes = {
    "_near_valid": (num_envs, 256),
    "_near_scan_ids": (num_envs,),
    "_near_source_dirs_sensor": (num_envs, 256, 3),
    "_near_source_dirs_present": (num_envs, 256),
    "_near_source_indices": (num_envs, 256),
    "_far_valid": (num_envs, 10, 64),
    "_far_scan_ids": (num_envs, 10),
    "_far_times": (num_envs, 10),
    "_far_source_dirs_sensor": (num_envs, 10, 64, 3),
    "_far_source_dirs_present": (num_envs, 10, 64),
    "_far_source_indices": (num_envs, 10, 64),
    "_far_count": (num_envs,),
    "_far_pos": (num_envs,),
    "_near_filled": (num_envs,),
    "_source_dirs_attached": (num_envs,),
  }
  checks["complete_persistent_window_fields_and_shapes"] = all(
    name in fields and torch.is_tensor(fields[name])
    and tuple(fields[name].shape) == shape and fields[name].numel() > 0
    for name, shape in expected_shapes.items()
  )
  if not checks["complete_persistent_window_fields_and_shapes"]:
    details["missing_or_mismatched_fields"] = [
      name for name, shape in expected_shapes.items()
      if name not in fields or not torch.is_tensor(fields[name])
      or tuple(fields[name].shape) != shape or fields[name].numel() == 0
    ]
    return checks, details

  counts = fields["_far_count"].to(torch.int64)
  cursors = fields["_far_pos"].to(torch.int64)
  raw_far_ids = fields["_far_scan_ids"].to(torch.int64)
  read = cpu_tree(runtime.window.read())
  angles = cpu_tree(runtime.window.read_angle_keys())
  source_directions = cpu_tree(runtime.window.read_source_directions())
  checks["all_rows_have_full_ten_scan_ring"] = bool((counts == 10).all())
  checks["all_rows_retain_ten_distinct_source_scan_ids"] = all(
    torch.unique(raw_far_ids[row]).numel() == 10
    and bool((raw_far_ids[row] >= 0).all())
    for row in range(num_envs)
  )
  checks["31_new_scans_rotate_each_cursor_by_one_slot"] = all(
    int((cursors[row] - int(initial_cursor[row])) % 10)
    == WINDOW_ROTATION_NEW_SCANS % 10
    for row in range(num_envs)
  )
  checks["raw_ring_masks_and_source_presence_match"] = (
    torch.equal(fields["_near_valid"], fields["_near_source_dirs_present"])
    and torch.equal(fields["_far_valid"], fields["_far_source_dirs_present"])
    and torch.equal(fields["_near_valid"], fields["_near_source_indices"] >= 0)
    and torch.equal(fields["_far_valid"], fields["_far_source_indices"] >= 0)
    and bool(fields["_source_dirs_attached"].all())
    and bool(fields["_near_filled"].all())
  )
  # Clean frames intentionally omit cached optical keys; runtime computes keys.
  # When supplied, keys must use the same cursor as every other read view.
  angle_presence_ok = (angles is None and not bool(fields["_angle_keys_present"].any())) or (
    angles is not None and torch.equal(angles.present, fields["_angle_keys_present"]))
  checks["read_views_follow_live_ring_cursor"] = angle_presence_ok and source_directions is not None
  chronological_scan_ids = []
  source_records = {}
  for row in range(num_envs):
    cursor = int(cursors[row])
    raw_order = torch.cat((torch.arange(cursor, 10), torch.arange(0, cursor)))
    expected_ids = raw_far_ids[row].index_select(0, raw_order)
    expected_times = fields["_far_times"][row].index_select(0, raw_order)
    expected_angle = fields["_far_angle_keys"][row].index_select(0, raw_order)
    expected_dirs = fields["_far_source_dirs_sensor"][row].index_select(0, raw_order)
    expected_present = fields["_far_source_dirs_present"][row].index_select(0, raw_order)
    expected_indices = fields["_far_source_indices"][row].index_select(0, raw_order)
    chronological_scan_ids.append(expected_ids.detach().cpu().tolist())
    views_ok = (
      torch.equal(read.far_scan_ids[row], expected_ids)
      and torch.equal(read.far_times[row], expected_times)
      and angle_presence_ok
      and (angles is None or torch.equal(angles.far[row], expected_angle))
      and source_directions is not None
      and torch.equal(source_directions.far[row], expected_dirs)
      and torch.equal(source_directions.far_present[row], expected_present)
      and torch.equal(source_directions.far_indices[row], expected_indices)
      and torch.equal(source_directions.near_present[row], read.near_valid[row])
      and torch.equal(source_directions.near_indices[row] >= 0, source_directions.near_present[row])
    )
    if not views_ok:
      checks["read_views_follow_live_ring_cursor"] = False

    episode_id = int(runtime.episode_ids[row].item())
    ordered_keys = sorted(
      (key for key in source_records_by_key
       if key[0] == row and key[1] == episode_id),
      key=lambda key: key[2],
    )
    recent_keys = ordered_keys[-10:]
    expected_recent_ids = [key[2] for key in recent_keys]
    if expected_recent_ids != expected_ids.detach().cpu().tolist():
      checks[f"retained_scans_match_recent_actual_sources_env{row}"] = False
    else:
      checks[f"retained_scans_match_recent_actual_sources_env{row}"] = True
    for key in recent_keys:
      source_records[(row, key[2])] = source_records_by_key[key]

    near_scan = int(read.near_scan_ids[row].item())
    near_record = source_records_by_key.get((row, episode_id, near_scan))
    near_ok = near_record is not None
    if near_record is not None:
      near_valid = read.near_valid[row].detach().cpu()
      actual_indices = source_directions.near_indices[row][near_valid]
      expected_indices_near = near_record["near"]["indices"][near_record["near"]["valid"]]
      near_ok = torch.equal(
        torch.sort(actual_indices).values,
        torch.sort(expected_indices_near).values,
      )
      expected_by_id = {
        int(index): direction
        for index, direction, valid in zip(
          near_record["near"]["indices"].tolist(),
          near_record["near"]["source_dirs_sensor"],
          near_record["near"]["valid"].tolist(),
        ) if valid
      }
      actual_dirs = source_directions.near[row][near_valid]
      actual_ids_list = actual_indices.tolist()
      near_ok = near_ok and all(
        torch.equal(direction, expected_by_id[source_id])
        for source_id, direction in zip(actual_ids_list, actual_dirs)
      )
      near_ok = near_ok and abs(
        float(read.near_times[row].item()) - float(near_record["time_s"])
      ) <= 1e-7
    checks[f"near_directions_and_ids_match_actual_raw_scan_env{row}"] = near_ok

    far_ok = True
    far_valid = read.far_valid[row].reshape(-1)
    far_scan_ids = read.far_scan_ids[row, :, None].expand_as(read.far_valid[row]).reshape(-1)
    far_indices = source_directions.far_indices[row].reshape(-1)
    far_dirs = source_directions.far[row].reshape(-1, 3)
    far_times = read.far_times[row, :, None].expand_as(read.far_valid[row]).reshape(-1)
    for scan_id in expected_recent_ids:
      source = source_records_by_key.get((row, episode_id, scan_id))
      selected_mask = far_valid & (far_scan_ids == scan_id)
      actual_indices = far_indices[selected_mask].detach().cpu()
      actual_dirs = far_dirs[selected_mask].detach().cpu()
      actual_times = far_times[selected_mask].detach().cpu()
      if source is None:
        far_ok = False
        continue
      source_valid = source["far"]["valid"]
      expected_indices_far = source["far"]["indices"][source_valid]
      expected_dirs_far = source["far"]["source_dirs_sensor"][source_valid]
      far_ok = far_ok and torch.equal(
        torch.sort(actual_indices).values, torch.sort(expected_indices_far).values
      )
      expected_by_id = {
        int(index): direction
        for index, direction, valid in zip(
          source["far"]["indices"].tolist(),
          source["far"]["source_dirs_sensor"],
          source_valid.tolist(),
        ) if valid
      }
      far_ok = far_ok and all(
        torch.equal(direction, expected_by_id[source_id])
        for source_id, direction in zip(actual_indices.tolist(), actual_dirs)
      )
      far_ok = far_ok and bool(torch.allclose(
        actual_times,
        torch.full_like(actual_times, float(source["time_s"])),
        atol=1e-7, rtol=0.0,
      ))
    checks[f"far_directions_ids_and_times_match_all_ten_raw_scans_env{row}"] = far_ok
  details.update({
    "far_count": counts.detach().cpu().tolist(),
    "far_pos": cursors.detach().cpu().tolist(),
    "initial_far_pos": list(initial_cursor),
    "initial_sensor_scan_counts": list(initial_counts),
    "raw_far_scan_ids_by_env": raw_far_ids.detach().cpu().tolist(),
    "chronological_far_scan_ids_by_env": chronological_scan_ids,
    "source_records_for_retained_scans": source_records,
  })

  prepared = runtime.prepare()
  provenance_checks = []
  for row in range(num_envs):
    episode = int(prepared.observation.episode_ids[row].item())
    near_scan = int(prepared.near_scan_ids[row].item())
    source = source_records_by_key.get((row, episode, near_scan))
    if source is None:
      provenance_checks.append({"env": row, "near": False, "far": False})
      continue
    prepared_near_valid = prepared.near_valid[row].detach().cpu()
    prepared_near_ids = prepared.near_source_indices[row].detach().cpu()[prepared_near_valid]
    source_near_ids = source["near"]["indices"][source["near"]["valid"]]
    near_ok = torch.equal(torch.sort(prepared_near_ids).values,
                          torch.sort(source_near_ids).values)
    expected_near_dir_by_id = {
      int(index): direction
      for index, direction, valid in zip(
        source["near"]["indices"].tolist(),
        source["near"]["source_dirs_sensor"],
        source["near"]["valid"].tolist(),
      ) if valid
    }
    prepared_near_dirs = prepared.near_source_dirs_sensor[row].detach().cpu()[prepared_near_valid]
    near_ok = near_ok and all(
      torch.equal(direction, expected_near_dir_by_id[source_id])
      for source_id, direction in zip(prepared_near_ids.tolist(), prepared_near_dirs)
    )
    near_ok = near_ok and abs(
      float(prepared.near_times[row].item()) - float(source["time_s"])
    ) <= 1e-7

    far_ok = True
    flat_valid = prepared.far_valid[row].reshape(-1)
    flat_scan = prepared.far_point_scan_ids[row].reshape(-1)
    flat_indices = prepared.far_source_indices[row].reshape(-1)
    flat_dirs = prepared.far_source_dirs_sensor[row].reshape(-1, 3)
    flat_times = prepared.far_point_times[row].reshape(-1)
    for scan_id in details["chronological_far_scan_ids_by_env"][row]:
      source = source_records_by_key.get((row, episode, int(scan_id)))
      selected = flat_valid & (flat_scan == int(scan_id))
      actual_ids = flat_indices[selected].detach().cpu()
      actual_dirs = flat_dirs[selected].detach().cpu()
      actual_times = flat_times[selected].detach().cpu()
      if source is None:
        far_ok = False
        continue
      source_valid = source["far"]["valid"]
      expected_ids = source["far"]["indices"][source_valid]
      expected_dirs = source["far"]["source_dirs_sensor"][source_valid]
      far_ok = far_ok and torch.equal(torch.sort(actual_ids).values,
                                      torch.sort(expected_ids).values)
      expected_by_id = {
        int(index): direction
        for index, direction, valid in zip(
          source["far"]["indices"].tolist(),
          source["far"]["source_dirs_sensor"],
          source_valid.tolist(),
        ) if valid
      }
      far_ok = far_ok and all(
        torch.equal(direction, expected_by_id[source_id])
        for source_id, direction in zip(actual_ids.tolist(), actual_dirs)
      )
      far_ok = far_ok and bool(torch.allclose(
        actual_times,
        torch.full_like(actual_times, float(source["time_s"])),
        atol=1e-7, rtol=0.0,
      ))
    provenance_checks.append({"env": row, "near": near_ok, "far": far_ok})
  checks["prepared_f1_sources_match_actual_scan_ray_ids_directions_and_times"] = all(
    item["near"] and item["far"] for item in provenance_checks
  )
  details["prepared_source_checks"] = provenance_checks
  details["prepared_identity"] = cpu_tree(prepared)
  return checks, details


def _window_rotation(out: Path, device: str, recipe: dict, *, args, grid) -> dict:
  from confined_locomotion.support.snapshots import cpu_tree, live_window_snapshot
  from hexapod_mjlab.tasks.el_4090_lidar.t0.witness import frozen_witness

  if args.num_envs != 2:
    raise ValueError("window_rotation requires exactly two rows for unselected-row isolation")
  env, wrapper, runtime = _build_runtime(out, args=args, grid=grid)
  checks = {}
  details = {}
  source_records = []
  source_records_by_key = {}
  edge_frames = {"first": {}, "last": {}}
  latest_frames = {}
  holder_steps = []
  initial_window = None
  reset_record = None
  reset_snapshots = None
  try:
    action = env.action_manager.get_term("residual_pd")
    witness = frozen_witness()
    t1_targets = torch.tensor(witness.joint_targets, dtype=torch.float32, device=device)
    actual_q_ref = action.q_ref.detach().clone()
    checks["actual_pd_reference_matches_frozen_t1_targets"] = (
      tuple(action._target_names) == tuple(witness.joint_names)
      and tuple(actual_q_ref.shape) == (2, 18)
      and float((actual_q_ref - t1_targets.expand_as(actual_q_ref)).abs().max()) <= 1e-7
    )
    initial_counts = runtime.sensor_scan_counts.detach().cpu().tolist()
    if initial_counts != [1, 1]:
      raise AssertionError(f"runtime must begin with one real takeover scan per row; got {initial_counts}")
    initial_cursor = runtime.window._far_pos.detach().cpu().to(torch.int64).tolist()
    initial_window = live_window_snapshot(runtime.window)
    initial_frames = {
      int(record["frame"].env_row): record["frame"]
      for record in runtime.pipeline.scan_records
      if isinstance(record, dict) and "frame" in record
    }
    if set(initial_frames) != {0, 1}:
      raise AssertionError("runtime did not retain both actual takeover source frames")

    def retain_frame(frame, *, scan_id, episode_id, persistent, time_s):
      row = int(frame.env_row)
      record = _selected_source_record(
        frame, scan_id=scan_id, episode_id=episode_id,
        persistent_count=persistent, time_s=time_s,
      )
      key = (row, int(episode_id), int(scan_id))
      if key in source_records_by_key:
        raise AssertionError(f"duplicate actual source scan identity {key}")
      source_records_by_key[key] = record
      source_records.append(record)
      if row not in edge_frames["first"]:
        edge_frames["first"][row] = _full_edge_frame(frame)
      latest_frames[row] = frame

    for row, frame in initial_frames.items():
      retain_frame(
        frame, scan_id=int(frame.scan_id), episode_id=int(frame.episode_id),
        persistent=int(runtime.sensor_scan_counts[row].item()), time_s=float(frame.time_s),
      )

    original_scan_rows = runtime.pipeline.scan_rows

    def record_new_scans(rows, **kwargs):
      frames = original_scan_rows(rows, **kwargs)
      for frame in frames:
        row = int(frame.env_row)
        retain_frame(
          frame,
          scan_id=int(runtime.scan_counters[row].item()) + 1,
          episode_id=int(runtime.episode_ids[row].item()),
          persistent=int(runtime.sensor_scan_counts[row].item()) + 1,
          time_s=float(runtime.control_time_s()),
        )
      return frames

    runtime.pipeline.scan_rows = record_new_scans
    # Keep the real scanner's diagnostic raw-frame list bounded to one current
    # pair. Complete selected-ray provenance is independently retained above.
    runtime.scan_audit_bound = args.num_envs
    start_control = int(runtime.control_steps_taken)
    start_counts = list(initial_counts)
    zero_action = torch.zeros((2, 18), dtype=torch.float32, device=device)
    natural_stop = None
    for _ in range(WINDOW_ROTATION_MAX_CONTROLS):
      transition = runtime.step(zero_action)
      context = transition.context_record
      pd_records = context.get("pd", ())
      pd_diffs = [
        float((item["target"] - action.q_ref).abs().max())
        for item in pd_records
      ]
      raw_ok = torch.equal(context["raw_action"], zero_action)
      clipped_ok = torch.equal(context["clipped_action"], zero_action)
      target_ok = len(pd_diffs) == 4 and all(diff == 0.0 for diff in pd_diffs)
      contact_count = context.get("c_nonfoot_env")
      holder_steps.append({
        "control_id": int(transition.control_id),
        "raw_zero": bool(raw_ok and torch.isfinite(context["raw_action"]).all()),
        "clipped_zero": bool(clipped_ok and torch.isfinite(context["clipped_action"]).all()),
        "four_substep_pd_targets_exact_t1": bool(target_ok),
        "max_target_abs_diff_rad": max(pd_diffs, default=None),
        "nonfoot_contact_count": (
          int(contact_count.sum().item()) if torch.is_tensor(contact_count) else None
        ),
      })
      if not (raw_ok and clipped_ok and target_ok):
        break
      if transition.engineering_stopped:
        natural_stop = {
          "control_id": int(transition.control_id),
          "terminated": cpu_tree(transition.terminated),
          "truncated": cpu_tree(transition.truncated),
          "stop_record": cpu_tree(runtime.stop_record),
        }
        break
      new_counts = [
        int(runtime.sensor_scan_counts[row].item()) - start_counts[row]
        for row in range(2)
      ]
      if all(value >= WINDOW_ROTATION_NEW_SCANS for value in new_counts):
        break

    new_counts = [
      int(runtime.sensor_scan_counts[row].item()) - start_counts[row]
      for row in range(2)
    ]
    checks["every_real_holder_step_used_zero_action_and_four_exact_t1_substeps"] = (
      bool(holder_steps)
      and all(item["raw_zero"] and item["clipped_zero"]
              and item["four_substep_pd_targets_exact_t1"] for item in holder_steps)
    )
    checks["31_actual_new_86400_ray_scans_per_row_within_180_controls"] = (
      all(value >= WINDOW_ROTATION_NEW_SCANS for value in new_counts)
      and len(holder_steps) <= WINDOW_ROTATION_MAX_CONTROLS
      and natural_stop is None
      and not runtime._stopped
      and all(
        record["raw_ray_count"] == RAW_RAYS_PER_SCAN
        for record in source_records
      )
    )
    checks["no_reset_was_used_to_pad_the_long_clock_gate"] = (
      [record["episode_id"] for record in source_records]
      == [0] * len(source_records)
      and len(source_records) == 2 + sum(new_counts)
    )
    details.update({
      "start_control_step": start_control,
      "end_control_step": int(runtime.control_steps_taken),
      "controls_executed": len(holder_steps),
      "initial_sensor_scan_counts": start_counts,
      "new_sensor_scans_by_row": new_counts,
      "total_source_frames_captured": len(source_records),
      "natural_stop": natural_stop,
      "holder_steps": holder_steps,
      "scan_index": [
        {key: value for key, value in record.items()
         if key not in ("near", "far")}
        for record in source_records
      ],
    })

    if checks["31_actual_new_86400_ray_scans_per_row_within_180_controls"]:
      edge_frames["last"] = {
        row: _full_edge_frame(frame) for row, frame in latest_frames.items()
      }
      pre_reset_window = live_window_snapshot(runtime.window)
      pre_reset_read = cpu_tree(runtime.window.read())
      pre_reset_angles = cpu_tree(runtime.window.read_angle_keys())
      pre_reset_sources = cpu_tree(runtime.window.read_source_directions())
      ring_checks, ring_details = _window_rotation_checks(
        runtime, grid=grid, num_envs=args.num_envs,
        initial_cursor=initial_cursor, initial_counts=initial_counts,
        source_records_by_key=source_records_by_key,
      )
      checks.update(ring_checks)
      details.update(ring_details)
      details["initial_far_cursor"] = initial_cursor
      details["pre_reset_window"] = pre_reset_window
      details["pre_reset_read"] = pre_reset_read
      details["pre_reset_angle_keys"] = pre_reset_angles
      details["pre_reset_source_directions"] = pre_reset_sources
      details["first_and_last_full_frames"] = edge_frames

      if all(checks.values()):
        before_reset = {
          "ring": live_window_snapshot(runtime.window),
          "chronological": cpu_tree(runtime.window.read()),
          "angle_keys": cpu_tree(runtime.window.read_angle_keys()),
          "source_directions": cpu_tree(runtime.window.read_source_directions()),
        }
        episodes_before = runtime.episode_ids.detach().cpu().clone()
        scan_counts_before = runtime.sensor_scan_counts.detach().cpu().clone()
        reset_record = runtime.reset(torch.tensor([0], dtype=torch.long, device=device))
        after_reset = {
          "ring": live_window_snapshot(runtime.window),
          "chronological": cpu_tree(runtime.window.read()),
          "angle_keys": cpu_tree(runtime.window.read_angle_keys()),
          "source_directions": cpu_tree(runtime.window.read_source_directions()),
        }
        reset_snapshots = {"before": before_reset, "after": after_reset}
        unselected_diffs = _window_read_row_differences(
          before_reset, after_reset, row=1, num_envs=args.num_envs,
        )
        episodes_after = runtime.episode_ids.detach().cpu()
        scan_counts_after = runtime.sensor_scan_counts.detach().cpu()
        checks["selected_reset_preserves_unselected_full_window_and_read_views"] = not unselected_diffs
        checks["selected_reset_clears_only_selected_ring_and_advances_its_first_scan"] = (
          int(episodes_after[0]) == int(episodes_before[0]) + 1
          and int(scan_counts_after[0]) == int(scan_counts_before[0]) + 1
          and int(scan_counts_after[1]) == int(scan_counts_before[1])
          and int(runtime.window._far_count[0].item()) == 1
          and int(runtime.window._far_pos[0].item()) == 1
          and bool(runtime.window._near_filled[0].item())
          and bool(runtime.window._source_dirs_attached[0].item())
          and int(runtime.window._far_count[1].item()) == 10
        )
        details["partial_reset"] = {
          "selected_rows": reset_record.get("reset_rows"),
          "new_episodes": reset_record.get("new_episodes"),
          "episodes_before": episodes_before.tolist(),
          "episodes_after": episodes_after.tolist(),
          "persistent_scans_before": scan_counts_before.tolist(),
          "persistent_scans_after": scan_counts_after.tolist(),
          "unselected_row_differences": unselected_diffs,
          "fixture_is_explicit_and_after_long_scan_gate": True,
        }
      else:
        checks["selected_reset_preserves_unselected_full_window_and_read_views"] = False
        checks["selected_reset_clears_only_selected_ring_and_advances_its_first_scan"] = False
        details["partial_reset"] = {"skipped": "pre-reset long-window evidence failed"}
    else:
      for name in (
        "complete_persistent_window_fields_and_shapes",
        "all_rows_have_full_ten_scan_ring",
        "all_rows_retain_ten_distinct_source_scan_ids",
        "31_new_scans_rotate_each_cursor_by_one_slot",
        "raw_ring_masks_and_source_presence_match",
        "read_views_follow_live_ring_cursor",
        "prepared_f1_sources_match_actual_scan_ray_ids_directions_and_times",
        "selected_reset_preserves_unselected_full_window_and_read_views",
        "selected_reset_clears_only_selected_ring_and_advances_its_first_scan",
      ):
        checks[name] = False
      details["long_window_checks_skipped"] = "31-scan real-backend gate did not pass"

    checks["passed"] = bool(checks) and all(checks.values())
    torch.save(cpu_tree({
      "mode": "window_rotation",
      "checks": checks,
      "details": details,
      "compact_actual_scan_sources": source_records,
      "first_and_last_full_source_frames": edge_frames,
      "partial_reset_record": reset_record,
      "partial_reset_window_before_after": reset_snapshots,
    }), out / "symmetry_window_rotation_evidence.pt")
    from confined_locomotion.support.evidence import write_json

    write_json(out / "symmetry_window_rotation_checks.json", {
      "checks": checks,
      "scan_summary": {
        key: value for key, value in details.items()
        if key not in ("pre_reset_window", "pre_reset_read", "pre_reset_angle_keys",
                       "pre_reset_source_directions", "prepared_identity",
                       "source_records_for_retained_scans", "partial_reset",
                       "first_and_last_full_frames", "holder_steps")
      },
      "partial_reset": details.get("partial_reset"),
      "source_index_count": len(source_records),
      "claim": "31 real new source scans per row rotate the persistent ten-scan window; no PPO or optimizer claim",
    })
    return checks
  finally:
    close_runtime(env, runtime)


def main(argv=None) -> int:
  args = _args(argv)
  from _common import configure_imports

  configure_imports()
  from confined_locomotion.support.fixtures import m12_training_grid

  grid = m12_training_grid()
  fn = _mapping_checks if args.mode == "mapping" else _window_rotation
  return execute_gpu_check(
    args,
    feature="symmetry",
    mode=args.mode,
    runner=lambda out, device, recipe: fn(out, device, recipe, args=args, grid=grid),
    optimizer=False,
    grid=grid,
    extra_recipe={
      "mapping_mode_uses_actual_compiled_optical_mount": True,
      "window_rotation_new_real_scans_per_row": (
        WINDOW_ROTATION_NEW_SCANS if args.mode == "window_rotation" else 0
      ),
      "window_rotation_max_real_controls": (
        WINDOW_ROTATION_MAX_CONTROLS if args.mode == "window_rotation" else 0
      ),
    },
  )


if __name__ == "__main__":
  raise SystemExit(main())
