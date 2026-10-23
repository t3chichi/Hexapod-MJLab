#!/usr/bin/env python
"""Real-backend evidence for cached observation perturbations and scan schedules."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from _common import add_common_arguments, close_runtime, execute_gpu_check


MODES = ("observation_reuse", "schedule")
SCHEDULE_SCAN_TARGET = 40
MAX_SCHEDULE_CONTROLS = 250


def _args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--mode", required=True, choices=MODES)
  add_common_arguments(parser)
  return parser.parse_args(argv)


def _build_runtime(out: Path, *, args, grid, preset):
  from confined_locomotion.support.fixtures import build_runtime

  env, _wrapper, runtime = build_runtime(
    out,
    mode="training",
    device=f"cuda:{args.gpu}",
    seed=args.seed,
    num_envs=args.num_envs,
    ground_grid=grid,
    perturbation_cfg=preset,
  )
  return env, runtime


def _range_consumption_audit(frame, pipeline) -> dict:
  raw = frame.raw_dists
  no_hit = frame.no_hit_mask
  valid = frame.consumed_valid
  epsilon = frame.range_epsilon
  consumed = frame.consumed_dists
  if valid is None or epsilon is None or consumed is None or frame.range_applied_mask is None:
    return {"passed": False, "reason": "range audit fields absent from actual scan frame"}
  expected_applied = (~no_hit) & valid
  expected = raw.clone()
  expected[expected_applied] = (raw[expected_applied] + epsilon[expected_applied]).clamp(0.0, 15.0)
  consumed_points = consumed[:, None] * frame.raw_ray_dirs
  from hexapod_mjlab.tasks.confined_locomotion.scanner import _quat_apply_xyzw
  import math

  pool_idx = frame.pool_raw_indices
  dirs_base = _quat_apply_xyzw(
    pipeline.mount_q.expand(int(frame.raw_ray_dirs.shape[0]), 4),
    frame.raw_ray_dirs,
  )
  # Independently recompute the expected B8 partition, not via the scanner
  # helper. The P6 cone uses RAW direction while x uses CONSUMED endpoint.
  cosang = (dirs_base @ pipeline.cone_axis_body).index_select(0, pool_idx)
  points = frame.pool_points_base
  expected_near = (
    frame.consumed_valid.index_select(0, pool_idx)
    & (cosang >= math.sin(math.radians(28.0)))
    & (points[:, 0] <= 3.5)
  )
  expected_far = frame.consumed_valid.index_select(0, pool_idx) & ~expected_near
  selected_near = frame.near_indices[frame.near_valid]
  selected_far = frame.far_indices[frame.far_valid]
  checks = {
    "B8_cfg_bound_to_real_pipeline": (
      pipeline.cfg.split_theta_deg == 28.0
      and pipeline.cfg.split_axis_tilt_deg == 56.0
      and pipeline.cfg.split_max_forward_x_m == 3.5
    ),
    "B8_near_matches_independent_raw_direction_and_consumed_x": torch.equal(
      frame.near_mask, expected_near
    ),
    "B8_far_is_exact_valid_complement": torch.equal(frame.far_mask, expected_far),
    "B8_selected_near_is_unique_and_inside_partition": (
      int(torch.unique(selected_near).numel()) == int(selected_near.numel())
      and bool(frame.near_mask[selected_near].all())
    ),
    "B8_selected_far_is_unique_and_inside_partition": (
      int(torch.unique(selected_far).numel()) == int(selected_far.numel())
      and bool(frame.far_mask[selected_far].all())
    ),
    "raw_frame_86400": int(raw.numel()) == 86_400,
    "raw_distances_retained": frame.raw_dists.data_ptr() != frame.consumed_dists.data_ptr(),
    "applied_mask_is_hit_and_consumed_valid": torch.equal(frame.range_applied_mask, expected_applied),
    "consumed_copy_matches_independent_clipped_epsilon": torch.equal(consumed, expected),
    "no_hit_stays_exact_15m": torch.equal(consumed[no_hit], raw[no_hit]),
    "gap_excluded_hit_stays_raw": torch.equal(consumed[(~valid) & (~no_hit)], raw[(~valid) & (~no_hit)]),
    "all_points_follow_consumed_range_and_original_direction": torch.equal(frame.points_sensor, consumed_points),
    "raw_hit_no_hit_identity_unchanged": torch.equal(no_hit, raw == 15.0),
  }
  return {"passed": all(checks.values()), "checks": checks,
          "ray_counts": {"raw": int(raw.numel()), "hits": int((~no_hit).sum()),
                         "valid": int(valid.sum()), "applied": int(expected_applied.sum())}}


def _observation_reuse(out: Path, device: str, recipe: dict, *, args, grid) -> dict:
  from confined_locomotion.support.snapshots import cpu_tree, runtime_snapshot, tree_equal
  from hexapod_mjlab.tasks.confined_locomotion.perturbations import ALL_THREE_PRESET

  env, runtime = _build_runtime(out, args=args, grid=grid, preset=ALL_THREE_PRESET)
  checks = {}
  try:
    before_physics = runtime_snapshot(runtime)["physics"]
    first = runtime.prepare()
    first_obs = cpu_tree(first.observation)
    first_noise = cpu_tree(runtime._perturbations.proprio_records())
    first_counts = runtime._perturbations.draw_counts()
    scan_records = [item for item in runtime.pipeline.scan_records if isinstance(item, dict) and "frame" in item]
    if len(scan_records) != args.num_envs:
      raise AssertionError(f"expected one actual takeover scan per env, got {len(scan_records)}")
    range_audits = {
      str(item["env_row"]): _range_consumption_audit(item["frame"], runtime.pipeline)
      for item in scan_records
    }
    second = runtime.prepare()
    second_obs = cpu_tree(second.observation)
    repeated_obs = second_obs
    second_noise = cpu_tree(runtime._perturbations.proprio_records())
    checks["prepare_returns_identity_cached_inputs"] = second is first
    checks["repeated_prepare_does_not_draw_streams"] = runtime._perturbations.draw_counts() == first_counts
    checks["cached_proprio_values_and_noise_are_bit_exact"] = (
      torch.equal(first_obs.lin_vel_b, second_obs.lin_vel_b)
      and torch.equal(first_obs.joint_pos_rel_ref, second_obs.joint_pos_rel_ref)
      and torch.equal(first_obs.command, second_obs.command)
      and all(torch.equal(a["noise"], b["noise"])
              for a, b in zip(first_noise, second_noise))
    )
    checks["noise_does_not_mutate_true_physics"] = not bool(tree_equal(
      runtime_snapshot(runtime)["physics"], before_physics
    ))
    checks["actual_range_noise_consumer_contract"] = all(item["passed"] for item in range_audits.values())
    active_counts = runtime._perturbations.draw_counts()
    checks["active_streams_are_per_environment_and_drawn"] = (
      active_counts["gap_jitter"] is not None
      and active_counts["range"] is not None
      and active_counts["proprio"] is not None
      and len(active_counts["range"]) == args.num_envs
      and len(active_counts["proprio"]) == args.num_envs
      and all(value > 0 for value in active_counts["range"])
      and all(value > 0 for value in active_counts["proprio"])
    )
    # The formal training scheduler must reject a manual command. Exercise the
    # rejected API without changing its command/cache/physics state.
    training_before_refusal = runtime_snapshot(runtime)
    training_command_refused = False
    try:
      runtime.set_manual_command(
        torch.arange(args.num_envs, device=device, dtype=torch.long),
        torch.tensor([[0.25, -0.15, 0.2]], device=device).expand(args.num_envs, 3),
      )
    except RuntimeError as error:
      training_command_refused = "play mode" in str(error)
    checks["training_mode_refuses_manual_command_without_mutation"] = (
      training_command_refused
      and not bool(tree_equal(training_before_refusal, runtime_snapshot(runtime)))
    )
  finally:
    close_runtime(env, runtime)
  checks["passed"] = all(checks.values())
  torch.save(cpu_tree({
    "first_observation": first_obs,
    "first_proprio_records": first_noise,
    "repeated_observation": repeated_obs,
    "draw_counts": active_counts,
    "range_audits": range_audits,
    "scan_records": [
      {key: value for key, value in item.items() if key != "frame"}
      for item in scan_records
    ],
    "checks": checks,
  }), out / "perturbation_reuse_evidence.pt")
  from confined_locomotion.support.evidence import write_json

  write_json(out / "perturbation_reuse_checks.json", checks)
  return checks


def _schedule(out: Path, device: str, recipe: dict, *, args, grid) -> dict:
  from confined_locomotion.support.snapshots import cpu_tree
  from hexapod_mjlab.tasks.confined_locomotion.perturbations import GAP_ONLY_PRESET

  env, runtime = _build_runtime(out, args=args, grid=grid, preset=GAP_ONLY_PRESET)
  checks = {}
  controls = 0
  captured_gap_frames = []
  original_scan_rows = runtime.pipeline.scan_rows

  def scan_and_capture_gaps(*scan_args, **scan_kwargs):
    frames = original_scan_rows(*scan_args, **scan_kwargs)
    captured_gap_frames.extend(frame for frame in frames if frame.gap_fired)
    return frames

  runtime.pipeline.scan_rows = scan_and_capture_gaps
  try:
    # A zero-residual T1 holder is the declared short scheduling witness. It
    # still uses the actual scanner and solver; a natural stop ends the run.
    while controls < MAX_SCHEDULE_CONTROLS and bool(
      (runtime.sensor_scan_counts < SCHEDULE_SCAN_TARGET).any()
    ) and not runtime._stopped and not runtime._faulted:
      runtime.step(torch.zeros((args.num_envs, 18), device=device))
      controls += 1
    rows = runtime._perturbations.gap_schedule_record()
    actual_scan_log = list(runtime.scan_log)
    expected_count = [int(value) for value in runtime.sensor_scan_counts.detach().cpu().tolist()]
    per_row = []
    for row in range(args.num_envs):
      row_scans = [item for item in actual_scan_log if int(item["env"]) == row]
      row_events = rows[row]["events"] if rows[row] is not None else []
      fired = [item for item in row_events if not item["skipped"]]
      per_row.append({
        "env": row,
        "actual_persistent_scan_count": expected_count[row],
        "observed_scan_log_count": len(row_scans),
        "fired_events": fired,
        "actual_gap_fired_scan_ids": [int(item["persistent_scan_count"]) for item in row_scans if item["gap_fired"]],
      })
    checks["actual_sensor_scans_reached_schedule_horizon"] = (
      all(value >= SCHEDULE_SCAN_TARGET for value in expected_count)
      and controls <= MAX_SCHEDULE_CONTROLS
    )
    checks["three_or_more_periodic_events_per_environment"] = all(
      len(item["fired_events"]) >= 3 for item in per_row
    )
    checks["scanner_gap_flags_match_real_persistent_scan_ids"] = all(
      item["actual_gap_fired_scan_ids"] == [int(event["fired_scan"]) for event in item["fired_events"]]
      for item in per_row
    )
    checks["runner_stopped_only_on_real_natural_stop_or_fault"] = (
      not runtime._faulted and (not runtime._stopped or runtime.stop_record is not None)
    )
    checks["gap_schedule_phase_remained_persistent"] = all(
      row is not None and 1 <= int(row["phase"]) <= int(GAP_ONLY_PRESET.gap.period_scans)
      for row in rows
    )
    # B8 near/far must remain a strict partition of the actually consumed
    # pool even on triggered real scans (the gap excludes 80x96 source rays).
    # The runtime intentionally retains only a bounded tail of full frames;
    # observe each real gap at acquisition rather than inspecting that tail.
    fired_frames = captured_gap_frames
    b8_checks = []
    import math
    from hexapod_mjlab.tasks.confined_locomotion.scanner import _quat_apply_xyzw
    for frame in fired_frames:
      p = runtime.pipeline
      mask = frame.consumed_valid.index_select(0, frame.pool_raw_indices)
      rotated = _quat_apply_xyzw(
        p.mount_q.expand(frame.raw_ray_dirs.shape[0], 4), frame.raw_ray_dirs,
      )
      angle = (rotated @ p.cone_axis_body).index_select(0, frame.pool_raw_indices)
      n = mask & (angle >= math.sin(math.radians(28))) & (
        frame.pool_points_base[:, 0] <= 3.5
      )
      f = mask & ~n
      selected_n = frame.near_indices[frame.near_valid]
      selected_f = frame.far_indices[frame.far_valid]
      b8_checks.append({
        "env": int(frame.env_row),
        "scan": int(frame.scan_id),
        "gap_rays": int((~frame.consumed_valid).sum().item()),
        "gap_count_7680": int((~frame.consumed_valid).sum().item()) == 7680,
        "near_equal": bool(torch.equal(frame.near_mask, n)),
        "far_equal": bool(torch.equal(frame.far_mask, f)),
        "nonempty": bool(mask.any()),
        "unique_near": int(torch.unique(selected_n).numel()) == int(selected_n.numel()),
        "unique_far": int(torch.unique(selected_f).numel()) == int(selected_f.numel()),
        "near_gap_excluded": not bool((~frame.consumed_valid)[selected_n].any()),
        "far_gap_excluded": not bool((~frame.consumed_valid)[selected_f].any()),
      })
    checks["B8_gap_fired_real_frames_partition_and_source_ids"] = (
      len(fired_frames) >= 3 * args.num_envs
      and all(all(value for key, value in item.items() if key not in ("env", "scan", "gap_rays"))
              for item in b8_checks)
    )
    checks["passed"] = all(checks.values())
    torch.save(cpu_tree({
      "controls_executed": controls,
      "sensor_scan_counts": runtime.sensor_scan_counts,
      "scan_counters": runtime.scan_counters,
      "gap_schedule": rows,
      "scan_log": actual_scan_log,
      "stopped": runtime._stopped,
      "stop_record": runtime.stop_record,
      "faulted": runtime._faulted,
      "last_fault": runtime.last_fault,
      "checks": checks,
    }), out / "gap_schedule_evidence.pt")
    from confined_locomotion.support.evidence import write_json

    write_json(out / "gap_schedule_checks.json", {
      "checks": checks,
      "controls_executed": controls,
      "per_environment": per_row,
      "gap_phases_and_events": rows,
      "B8_per_fired_frame": b8_checks,
      "natural_stop": runtime.stop_record,
      "fault": runtime.last_fault,
    })
  finally:
    runtime.pipeline.scan_rows = original_scan_rows
    close_runtime(env, runtime)
  return checks


def main(argv=None) -> int:
  args = _args(argv)
  from _common import configure_imports

  configure_imports()
  from confined_locomotion.support.fixtures import m12_training_grid

  grid = m12_training_grid()
  fn = _observation_reuse if args.mode == "observation_reuse" else _schedule
  return execute_gpu_check(
    args,
    feature="perturbations",
    mode=args.mode,
    runner=lambda out, device, recipe: fn(out, device, recipe, args=args, grid=grid),
    optimizer=False,
    grid=grid,
    extra_recipe={
      "perturbation_preset": "ALL_THREE" if args.mode == "observation_reuse" else "GAP_ONLY",
      "schedule_horizon_real_scans": SCHEDULE_SCAN_TARGET if args.mode == "schedule" else 1,
      "natural_stop_ends_run": True,
    },
  )


if __name__ == "__main__":
  raise SystemExit(main())
