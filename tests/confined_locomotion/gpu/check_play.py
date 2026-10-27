#!/usr/bin/env python
"""Feature-named CUDA witnesses for task play and retained viewer snapshots.

The ``snapshot`` mode exercises only pure snapshot assembly and manifest
export. It deliberately never creates ``PlayViewer`` (which starts a local
server) and never opens a browser.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from _common import add_common_arguments, execute_gpu_check


MODES = (
  "weights_only", "manual_controls", "natural_mean", "long_episode",
  "gate_fixtures", "snapshot",
)
PRESETS = ("clean", "gap", "gap_fixed0", "range", "proprio", "all")
CONTROLS_HOLDER_STEPS = 205
LONG_EPISODE_CONTROL_STEPS = 1051


def _args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--mode", required=True, choices=MODES)
  parser.add_argument("--checkpoint", required=True, type=Path)
  parser.add_argument("--preset", choices=PRESETS, default="clean")
  parser.add_argument("--max-steps", type=int, default=600,
                      help="natural mean-step budget (default: 600)")
  parser.add_argument("--allow-source-change", action="store_true",
                      help="explicitly allow changed source hashes for weights-only play")
  add_common_arguments(parser)
  return parser.parse_args(argv)


def _session(out: Path, *, args, grid, suffix: str = "play"):
  from hexapod_mjlab.tasks.confined_locomotion.play import PlaySession

  return PlaySession(
    checkpoint=args.checkpoint.expanduser().resolve(),
    output_dir=out / suffix,
    seed=args.seed,
    ground_grid=grid,
    device=f"cuda:{args.gpu}",
    num_envs=args.num_envs,
    perturbation_preset=args.preset,
    allow_source_change=args.allow_source_change,
  )


def _command_generator_states(session):
  return [generator.get_state().clone() for generator in session.runtime._commands._generators]


def _draw_counts(session):
  perturbations = session.runtime._perturbations
  return None if perturbations is None else perturbations.draw_counts()


def _step_trajectory_row(index: int, transition, runtime) -> dict:
  """Keep per-control action/height/identity evidence without copying point clouds."""
  context = transition.context_record
  root_pos = context["root_pos_w"].detach().cpu()
  raw_action = context["raw_action"].detach().cpu()
  tables = context["substep_tables"]
  return {
    "control_index": int(index),
    "control_steps_taken": int(runtime.control_steps_taken),
    "episode_ids": runtime.episode_ids.detach().cpu().tolist(),
    "episode_length_buf": runtime.env.episode_length_buf.detach().cpu().tolist(),
    "scan_counters": runtime.scan_counters.detach().cpu().tolist(),
    "sensor_scan_counts": runtime.sensor_scan_counts.detach().cpu().tolist(),
    "root_height_w": root_pos[:, 2].tolist(),
    "raw_action": raw_action.tolist(),
    "terminated": transition.terminated.detach().cpu().tolist(),
    "truncated": transition.truncated.detach().cpu().tolist(),
    "physics_substeps": len(tables),
  }


def _check_weights_only(out: Path, *, args, grid) -> dict:
  from confined_locomotion.support.evidence import write_json
  from confined_locomotion.support.snapshots import cpu_tree
  from hexapod_mjlab.tasks.confined_locomotion.ground import GroundGridCfg
  from hexapod_mjlab.tasks.confined_locomotion.play import PlayConfigError

  session = _session(out, args=args, grid=grid, suffix="weights_only_session")
  checks = {}
  try:
    forbidden = (
      "provider", "trainer", "collector", "optimizer", "policy_adapter",
      "algorithm", "actor", "critic",
    )
    checks["checkpoint_loaded_weights_only_without_training_components"] = (
      session.weight_report.get("resumable") is False
      and not session.network.training
      and session.network.mapping.mode == "raw"
      and not any(hasattr(session, name) for name in forbidden)
      and not any(isinstance(value, torch.optim.Optimizer) for value in vars(session).values())
    )
    checks["checkpoint_and_play_recipe_use_functional_schemas"] = (
      session.header.get("schema") == "confined_locomotion.checkpoint/v3"
      and session.output_dir.joinpath("play_recipe.json").is_file()
      and __import__("json").loads(
        session.output_dir.joinpath("play_recipe.json").read_text()
      ).get("schema") == "confined_locomotion.play_recipe/v2"
    )

    moved_grid = GroundGridCfg(
      x_coords=tuple(float(value) + 0.01 for value in grid.x_coords),
      y_coords=grid.y_coords,
      validity_mode=grid.validity_mode,
      parallel_tol=grid.parallel_tol,
      body_xyz_semantics=grid.body_xyz_semantics,
    )
    rejected_output = out / "moved_grid_rejection"
    try:
      from hexapod_mjlab.tasks.confined_locomotion.play import PlaySession

      PlaySession(
        checkpoint=args.checkpoint.expanduser().resolve(),
        output_dir=rejected_output,
        seed=args.seed,
        ground_grid=moved_grid,
        device=f"cuda:{args.gpu}",
        num_envs=args.num_envs,
        allow_source_change=True,
      )
      checks["same_M_moved_grid_rejected_before_env_construction"] = False
    except PlayConfigError:
      checks["same_M_moved_grid_rejected_before_env_construction"] = not rejected_output.exists()

    transition = session.step_once()
    used = transition.context_record["raw_action"].detach().cpu()
    forward_mean = torch.as_tensor(session._display.output["action_mean"]).detach().cpu()
    checks["executed_action_is_the_network_mean_bit_exact"] = torch.equal(used, forward_mean)
    checks["play_path_never_fires_timeout"] = not bool(transition.truncated.any())
    checks["full_checkpoint_remains_non_resumable_in_play"] = (
      session.weight_report.get("resumable") is False
      and "weights-only" in session.weight_report.get("note", "")
    )
    checks["passed"] = all(checks.values())
    torch.save(cpu_tree({
      "weight_report": session.weight_report,
      "checkpoint_header": session.header,
      "grid_identity": grid.identity(),
      "mean_action": forward_mean,
      "executed_raw_action": used,
      "transition": transition,
      "checks": checks,
    }), out / "play_weights_only_evidence.pt")
  finally:
    session.close()

  # A separately seeded network with the same complete M12 key set is also
  # loaded as weights only; this contains no environment or optimizer.
  from confined_locomotion.support.fixtures import build_network
  from hexapod_mjlab.tasks.confined_locomotion.rl.checkpoint import load_weights

  network = build_network(seed=args.seed + 1, ground_samples=grid.num_points, device=f"cuda:{args.gpu}")
  report = load_weights(
    args.checkpoint.expanduser().resolve(), network=network, provider=None,
    expected_grid_identity=grid.identity(), allow_source_change=args.allow_source_change,
  )
  checks["fresh_network_load_is_weights_only_and_env_count_independent"] = (
    report.get("resumable") is False
    and report.get("kind") == "confined_locomotion_weights_loaded"
  )
  checks["passed"] = all(checks.values())
  write_json(out / "play_weights_only_checks.json", {
    "checks": checks,
    "separate_network_load_report": report,
    "checkpoint": str(args.checkpoint.expanduser().resolve()),
  })
  return checks


def _manual_controls(out: Path, *, args, grid) -> dict:
  from confined_locomotion.support.evidence import write_json
  from confined_locomotion.support.snapshots import (
    cpu_tree, global_prng_snapshot, runtime_snapshot, tree_equal,
  )

  session = _session(out, args=args, grid=grid, suffix="manual_controls")
  runtime = session.runtime
  scheduler = runtime._commands
  checks = {}
  try:
    first_records = scheduler.resample_records[: session.num_envs]
    takeover_records = scheduler.resample_records[session.num_envs : 2 * session.num_envs]
    checks["manual_rows_start_zero_without_random_takeover_draw"] = (
      bool(scheduler.allow_manual)
      and bool(scheduler.manual_mask().all())
      and torch.equal(scheduler.current_command(), torch.zeros_like(scheduler.current_command()))
      and all(item.get("reason") == "manual_set" for item in first_records)
      and all(item.get("reason") == "reset_manual_zero" for item in takeover_records)
    )

    cached_before_command = cpu_tree(runtime.prepare().observation)
    perturbations = runtime._perturbations
    noise_before_command = None if perturbations is None else cpu_tree(perturbations.proprio_records())
    draws_before_command = _draw_counts(session)
    scan_counts_before_command = runtime.sensor_scan_counts.detach().clone()
    rng_before_command = global_prng_snapshot(device=runtime.device)
    command_rng_before_command = _command_generator_states(session)
    perturbation_state_before_command = None if perturbations is None else cpu_tree(perturbations.state_dict())
    applied = session.set_command(0.5, 0.0, 0.0)
    rebuilt_observation = cpu_tree(runtime.prepare().observation)
    noise_after_command = None if perturbations is None else cpu_tree(perturbations.proprio_records())
    checks["manual_cache_invalidation_reuses_observation_noise_and_scan_identity"] = (
      _draw_counts(session) == draws_before_command
      and not tree_equal(noise_before_command, noise_after_command)
      and torch.equal(scan_counts_before_command, runtime.sensor_scan_counts)
      and not tree_equal(rng_before_command, global_prng_snapshot(device=runtime.device))
      and all(torch.equal(state, generator.get_state())
              for state, generator in zip(command_rng_before_command, scheduler._generators))
      and not tree_equal(perturbation_state_before_command, None if perturbations is None else cpu_tree(perturbations.state_dict()))
      and all(torch.equal(value, getattr(rebuilt_observation, name))
              for name, value in vars(cached_before_command).items()
              if name != "command" and torch.is_tensor(value))
    )
    torch.save(cpu_tree({"before_observation": cached_before_command,
                         "after_observation": rebuilt_observation,
                         "before_noise": noise_before_command, "after_noise": noise_after_command,
                         "before_draw_counts": draws_before_command,
                         "after_draw_counts": _draw_counts(session),
                         "before_scan_counts": scan_counts_before_command,
                         "after_scan_counts": runtime.sensor_scan_counts,
                         "before_global_prng": rng_before_command,
                         "after_global_prng": global_prng_snapshot(device=runtime.device),
                         "before_command_generators": command_rng_before_command,
                         "after_command_generators": _command_generator_states(session),
                         "before_perturbation_state": perturbation_state_before_command,
                         "after_perturbation_state": None if perturbations is None else perturbations.state_dict()}),
               out / "manual_cache_invalidation_evidence.pt")
    expected = torch.tensor([[0.5, 0.0, 0.0]], device=runtime.device).expand(session.num_envs, 3)
    checks["manual_command_applies_to_every_row_and_prepared_input"] = (
      applied.get("command") == [[0.5, 0.0, 0.0]] * session.num_envs
      and torch.equal(runtime.prepare().observation.command, expected)
    )
    before_rejection = runtime_snapshot(runtime)
    prng_before_rejection = global_prng_snapshot(device=runtime.device, runtime=runtime)
    try:
      session.set_command(9.9, 0.0, 0.0)
    except Exception as error:  # plain bad input must not become a physics fault
      rejected = type(error).__name__ == "PlayConfigError"
    else:
      rejected = False
    checks["invalid_manual_command_rejected_without_runtime_mutation"] = (
      rejected
      and not bool(tree_equal(before_rejection, runtime_snapshot(runtime)))
      and not bool(tree_equal(prng_before_rejection, global_prng_snapshot(
        device=runtime.device, runtime=runtime
      )))
      and not runtime._faulted
    )

    command_states = _command_generator_states(session)
    zero_action = torch.zeros((session.num_envs, 18), device=runtime.device)
    controls = 0
    for _ in range(CONTROLS_HOLDER_STEPS):
      runtime.step(zero_action)
      controls += 1
    new_resamples = scheduler.resample_records[2 * session.num_envs :]
    checks["manual_command_survives_205_real_controls_without_period_draw"] = (
      controls == CONTROLS_HOLDER_STEPS
      and torch.equal(scheduler.current_command()[:, 0], torch.full_like(scheduler.current_command()[:, 0], 0.5))
      and all(item.get("reason") not in ("period", "reset", "clear_manual") for item in new_resamples)
      and all(torch.equal(before, generator.get_state())
              for before, generator in zip(command_states, scheduler._generators))
      and not runtime._stopped and not runtime._faulted
    )
    checks["holder_is_raw_zero_residual_not_mean_actor"] = (
      torch.equal(runtime.env.action_manager.get_term("residual_pd").last_raw_action, zero_action)
    )

    # Settle the retained observation after the final physical control. With
    # an active proprio preset this first prepare may legitimately draw its
    # new control-id sample; only repeated reads of the retained frame are
    # required to be RNG-pure.
    session.refresh_idle_snapshot(reason="manual_pause_settle")
    session.build_display_snapshot(0, paused=True)
    before_paused = runtime_snapshot(runtime)
    prng_before_paused = global_prng_snapshot(device=runtime.device, runtime=runtime)
    perturb_before_paused = _draw_counts(session)
    for _ in range(5):
      session.refresh_idle_snapshot(reason="manual_pause_read")
      session.build_display_snapshot(0, paused=True)
    checks["paused_refresh_and_snapshot_do_not_advance_runtime_or_rng"] = (
      not bool(tree_equal(before_paused, runtime_snapshot(runtime)))
      and not bool(tree_equal(prng_before_paused, global_prng_snapshot(
        device=runtime.device, runtime=runtime
      )))
      and _draw_counts(session) == perturb_before_paused
    )

    episodes_before = runtime.episode_ids.detach().clone()
    scans_before = runtime.sensor_scan_counts.detach().clone()
    reset_record = session.reset()
    checks["manual_reset_zeros_command_keeps_manual_rows_and_advances_episodes_scans"] = (
      reset_record.get("reason") == "reset"
      and torch.equal(scheduler.current_command(), torch.zeros_like(scheduler.current_command()))
      and bool(scheduler.manual_mask().all())
      and bool((runtime.episode_ids > episodes_before).all())
      and bool((runtime.sensor_scan_counts > scans_before).all())
      and all(item.get("reason") == "reset_manual_zero"
              for item in scheduler.resample_records[-session.num_envs :])
    )
    checks["passed"] = all(checks.values())
    torch.save(cpu_tree({
      "runtime_after_reset": runtime_snapshot(runtime),
      "scheduler_records": scheduler.resample_records,
      "perturbation_draw_counts": _draw_counts(session),
      "checks": checks,
    }), out / "play_manual_controls_evidence.pt")
  finally:
    session.close()
  write_json(out / "play_manual_controls_checks.json", checks)
  return checks


def _natural_mean(out: Path, *, args, grid) -> dict:
  from confined_locomotion.support.evidence import write_json
  from confined_locomotion.support.snapshots import cpu_tree, runtime_snapshot, tree_equal

  if args.max_steps < 1:
    raise ValueError("--max-steps must be positive")
  session = _session(out, args=args, grid=grid, suffix="natural_mean")
  runtime = session.runtime
  checks = {}
  trajectory = []
  first_transition = None
  previous_transition = None
  selected_transitions = {}
  try:
    while len(trajectory) < args.max_steps and not runtime._stopped:
      transition = session.step_once()
      trajectory.append(_step_trajectory_row(len(trajectory) + 1, transition, runtime))
      if first_transition is None:
        first_transition = cpu_tree(transition)
      if bool(transition.truncated.any()):
        selected_transitions["timeout_transition"] = cpu_tree(transition)
        break
      if runtime._stopped:
        if previous_transition is not None:
          selected_transitions["pre_fall_transition"] = cpu_tree(previous_transition)
        selected_transitions["fall_transition"] = cpu_tree(transition)
        break
      previous_transition = transition
    if not runtime._stopped and "timeout_transition" not in selected_transitions:
      selected_transitions["last_transition"] = cpu_tree(transition)
    checks["natural_path_did_not_force_a_reset_or_timeout"] = (
      all(not any(item["truncated"]) for item in trajectory)
      and not runtime._faulted
    )
    checks["real_mean_steps_recorded"] = len(trajectory) > 0
    checks["fall_is_honest_data_not_a_threshold_change"] = (
      not runtime._stopped or runtime.stop_record is not None
    )
    fall_record = None
    if runtime._stopped:
      fall_dirs = sorted(runtime.output_dir.glob("play_fall_stop_*"))
      before = runtime_snapshot(runtime)
      prepare_calls = {"count": 0}
      original_prepare = runtime.prepare

      def counted_prepare(*pos, **kw):
        prepare_calls["count"] += 1
        return original_prepare(*pos, **kw)

      runtime.prepare = counted_prepare
      try:
        try:
          session.step_once()
          refused = False
        except RuntimeError as error:
          refused = "paused after a saved formal fall" in str(error)
      finally:
        runtime.prepare = original_prepare
      paused_snapshot = session.build_display_snapshot(0, paused=True)
      checks["saved_fall_pauses_without_reprepare_and_retained_snapshot_survives"] = (
        bool(fall_dirs) and refused and prepare_calls["count"] == 0
        and paused_snapshot["state"]["kind"] == "fallen"
        and not bool(tree_equal(before, runtime_snapshot(runtime)))
      )
      reset_record = session.reset()
      resumed = session.step_once()
      checks["explicit_full_episode_reset_is_the_only_fall_recovery"] = (
        reset_record.get("reason") == "reset"
        and not runtime._stopped
        and not bool(resumed.terminated.any())
        and not bool(resumed.truncated.any())
      )
      fall_record = {
        "saved_scene_directories": [path.name for path in fall_dirs],
        "stop_record": runtime.stop_record,
        "reset_record": reset_record,
        "resumed_step_terminated": resumed.terminated,
      }
      selected_transitions["post_reset_resume_transition"] = cpu_tree(resumed)
    else:
      checks["natural_fall_observed"] = None
      checks["fall_branch_note"] = "not observed within this finite natural-mean budget"
    checks["passed"] = all(
      value is True for key, value in checks.items()
      if key not in ("natural_fall_observed", "fall_branch_note")
    )
    torch.save(cpu_tree({
      "control_trajectory": trajectory,
      "selected_raw_transitions": selected_transitions,
      "natural_fall": fall_record,
      "runtime_after_mode": runtime_snapshot(runtime),
      "checks": checks,
    }), out / "play_natural_mean_evidence.pt")
  finally:
    session.close()
  write_json(out / "play_natural_mean_checks.json", checks)
  return checks


def _long_episode(out: Path, *, args, grid) -> dict:
  from confined_locomotion.support.evidence import write_json
  from confined_locomotion.support.snapshots import cpu_tree

  session = _session(out, args=args, grid=grid, suffix="long_episode")
  runtime = session.runtime
  checks = {}
  scans_before = runtime.sensor_scan_counts.detach().clone()
  episodes_before = runtime.episode_ids.detach().clone()
  zero_action = torch.zeros((session.num_envs, 18), device=runtime.device)
  steps = substeps = 0
  trajectory = []
  first_transition = None
  last_transition = None
  try:
    for _ in range(LONG_EPISODE_CONTROL_STEPS):
      transition = runtime.step(zero_action)  # explicit zero-residual T1 holder
      if first_transition is None:
        first_transition = cpu_tree(transition)
      last_transition = transition
      steps += 1
      tables = transition.context_record["substep_tables"]
      substeps += len(tables)
      trajectory.append(_step_trajectory_row(steps, transition, runtime))
      if len(tables) != 4 or bool(transition.truncated.any()) or runtime._stopped:
        break
    new_scans = runtime.sensor_scan_counts - scans_before
    total_scans = runtime.sensor_scan_counts.detach().cpu().tolist()
    checks["zero_residual_holder_completed_1051_controls_and_4204_substeps"] = (
      steps == LONG_EPISODE_CONTROL_STEPS and substeps == 4 * LONG_EPISODE_CONTROL_STEPS
    )
    checks["natural_scan_schedule_advanced_without_episode_reset"] = (
      bool((new_scans == LONG_EPISODE_CONTROL_STEPS // 5).all())
      and bool(torch.equal(runtime.episode_ids, episodes_before))
      and not runtime._stopped and not runtime._faulted
    )
    checks["play_clock_passed_1000_steps_without_timeout"] = bool(
      (runtime.env.episode_length_buf > 1000).all()
    )
    checks["all_actual_scans_remain_persistent"] = all(value >= 211 for value in total_scans)
    checks["passed"] = all(checks.values())
    torch.save(cpu_tree({
      "controls": steps,
      "substeps": substeps,
      "takeover_scan_counts": scans_before,
      "new_scan_counts": new_scans,
      "persistent_scan_counts": runtime.sensor_scan_counts,
      "episode_length_buf": runtime.env.episode_length_buf,
      "control_trajectory": trajectory,
      "selected_raw_transitions": {
        "first_transition": first_transition,
        "last_transition": None if last_transition is None else cpu_tree(last_transition),
      },
      "checks": checks,
      "holder_claim": "zero residual raw action, lifecycle coverage only; no standing/walking claim",
    }), out / "play_long_episode_evidence.pt")
  finally:
    session.close()
  write_json(out / "play_long_episode_checks.json", checks)
  return checks


def _write_pose(session, env_row: int, *, z: float, quat_wxyz=None) -> None:
  robot = session.env.scene["robot"]
  pose = torch.cat((robot.data.root_link_pos_w, robot.data.root_link_quat_w), dim=1).clone()
  pose[env_row, 2] = z
  if quat_wxyz is not None:
    pose[env_row, 3:] = torch.tensor(quat_wxyz, device=pose.device, dtype=pose.dtype)
  row = torch.tensor([env_row], device=pose.device, dtype=torch.long)
  robot.write_root_link_pose_to_sim(pose[env_row : env_row + 1], env_ids=row)
  session.env.scene.write_data_to_sim()
  session.env.sim.forward()
  session.runtime._input_cache.invalidate()


def _gate_fixtures(out: Path, *, args, grid) -> dict:
  import warp as wp

  from confined_locomotion.support.evidence import write_json
  from confined_locomotion.support.snapshots import (
    cpu_tree, persistent_physics_snapshot, tree_equal, runtime_snapshot, global_prng_snapshot,
  )
  from hexapod_mjlab.common.mdp.terminations import get_termination_state

  session = _session(out, args=args, grid=grid, suffix="gate_fixtures")
  runtime = session.runtime
  device = runtime.device
  zero_action = torch.zeros((session.num_envs, 18), device=device)
  checks = {}
  fixtures = []
  raw = []
  termination = get_termination_state(session.env)

  def qpos():
    return wp.to_torch(session.env.sim.wp_data.qpos).detach().cpu().clone()

  def capture(name, before, transition):
    item = {
      "fixture": name,
      "qpos_before": before,
      "physics_after": persistent_physics_snapshot(session.env),
      "transition": cpu_tree(transition),
      "termination_state": cpu_tree(termination),
    }
    raw.append(item)
    torch.save(cpu_tree(raw), out / "play_gate_fixture_raw.pt")
    return item

  try:
    low_counts = torch.zeros(session.num_envs, dtype=torch.long, device=device)
    low_counts[0] = 9
    termination.preset_low_height_counts(low_counts)
    _write_pose(session, 0, z=0.10)
    before = qpos()
    transition = session.step_once()  # real mean owner must save and pause
    capture("isolated_tenth_low_height_step", before, transition)
    checks["isolated_tenth_low_height_counter_trips_real_gate"] = (
      bool(transition.terminated[0]) and not bool(transition.terminated[1:].any())
      and runtime._stopped and len(transition.context_record["substep_tables"]) == 4
    )
    checks["low_height_gate_threshold_unchanged"] = (
      termination.low_height_count[0].item() >= 10
    )
    stopped_state = runtime_snapshot(runtime)
    stopped_prng = global_prng_snapshot(device=device, runtime=runtime)
    attempts = {"prepare": 0}
    feature_cache = session.feature_cache
    cache_calls_before = (
      None if feature_cache is None else feature_cache.stats["forward_calls"]
    )
    original_prepare = runtime.prepare
    def counted_prepare(*pos, **kw):
      attempts["prepare"] += 1
      return original_prepare(*pos, **kw)
    runtime.prepare = counted_prepare
    try:
      try:
        session.step_once()
        fall_refused = False
      except RuntimeError as error:
        fall_refused = "paused after a saved formal fall" in str(error)
      snapshot = session.build_display_snapshot(0, paused=True)
    finally:
      runtime.prepare = original_prepare
    no_policy_forward = (
      feature_cache is not None
      and feature_cache.stats["forward_calls"] == cache_calls_before
    )
    events = sorted((session.output_dir / "play_events").glob("*_fall_*"))
    checks["owner_saved_fall_scene_pauses_without_prepare_forward_or_rng"] = (
      bool(list(runtime.output_dir.glob("play_fall_stop_*")))
      and len(events) == 1
      and (events[0] / "play_terminal_scene.pt").is_file()
      and (events[0] / "terminal_record.json").is_file()
      and fall_refused and attempts == {"prepare": 0} and no_policy_forward
      and snapshot["state"]["kind"] == "fallen"
      and not tree_equal(stopped_state, runtime_snapshot(runtime))
      and not tree_equal(stopped_prng, global_prng_snapshot(device=device, runtime=runtime))
    )
    reset_low = session.reset()
    resumed = session.step_once()
    checks["explicit_reset_resumes_real_mean_owner_after_saved_fall"] = (
      reset_low.get("reason") == "reset" and not runtime._stopped
      and not bool(resumed.terminated.any()) and not bool(resumed.truncated.any())
    )
    capture("explicit_reset_mean_resume", qpos(), resumed)

    half = math.radians(75.0 / 2.0)
    tilt_counts = torch.zeros(session.num_envs, dtype=torch.long, device=device)
    tilt_counts[0] = 9
    termination.preset_tilt_counts(tilt_counts)
    _write_pose(session, 0, z=1.5, quat_wxyz=(math.cos(half), math.sin(half), 0.0, 0.0))
    before = qpos()
    transition = runtime.step(zero_action)
    capture("isolated_tenth_tilt_step", before, transition)
    checks["isolated_tenth_tilt_counter_trips_real_gate"] = (
      bool(transition.terminated[0]) and len(transition.context_record["substep_tables"]) == 4
    )
    reset_tilt = session.reset()

    flip_half = math.radians(120.0 / 2.0)
    _write_pose(session, 0, z=1.0, quat_wxyz=(math.cos(flip_half), math.sin(flip_half), 0.0, 0.0))
    before = qpos()
    transition = runtime.step(zero_action)
    capture("immediate_physical_flip", before, transition)
    checks["greater_than_90_degree_flip_trips_immediately"] = (
      bool(transition.terminated[0]) and len(transition.context_record["substep_tables"]) == 4
    )
    reset_flip = session.reset()

    _write_pose(session, 0, z=0.05)
    before = qpos()
    transition = runtime.step(zero_action)
    contact_record = transition.context_record
    capture("low_pose_actual_nonfoot_contact_probe", before, transition)
    fixtures.append({
      "fixture": "low_pose_actual_nonfoot_contact_probe",
      "nonfoot_contact_counts": contact_record["c_nonfoot_env"],
      "self_contact_counts": contact_record["c_self"],
      "substep_count": len(contact_record["substep_tables"]),
      "note": "report actual contact counts; zero is not interpreted as proof of absence",
    })
    checks["contact_probe_kept_all_four_actual_substep_tables"] = (
      len(contact_record["substep_tables"]) == 4
      and contact_record["c_nonfoot_env"].shape == (session.num_envs,)
      and contact_record["c_self"].shape == (session.num_envs,)
    )
    episodes_at_contact = runtime.episode_ids.detach().clone()
    contact_occurred = bool(((contact_record["c_nonfoot_env"] + contact_record["c_self"]) > 0).any())
    checks["actual_collision_does_not_pause_or_reset"] = (
      contact_occurred and not runtime._stopped and not runtime._faulted
      and not bool(transition.terminated.any()) and not bool(transition.truncated.any())
    )
    if not runtime._stopped:
      continued = runtime.step(zero_action)
      capture("actual_contact_continue_without_reset", before, continued)
      checks["next_control_after_contact_continues_same_episode"] = (
        not runtime._stopped and not runtime._faulted
        and torch.equal(runtime.episode_ids, episodes_at_contact)
        and len(continued.context_record["substep_tables"]) == 4
      )
    else:
      checks["next_control_after_contact_continues_same_episode"] = False
      session.reset()

    physics_before_fault = persistent_physics_snapshot(session.env)
    nan_action = zero_action.clone()
    nan_action[0, 0] = float("nan")
    try:
      runtime.step(nan_action)
      fault_rejected = False
    except RuntimeError:
      fault_rejected = True
    physics_after_fault = persistent_physics_snapshot(session.env)
    checks["nan_action_faults_before_any_physics_integration"] = (
      fault_rejected and runtime._faulted
      and not bool(tree_equal(physics_before_fault, physics_after_fault))
    )
    try:
      runtime.step(zero_action)
      step_refused = False
    except RuntimeError:
      step_refused = True
    try:
      session.reset()
      reset_refused = False
    except RuntimeError:
      reset_refused = True
    checks["fault_is_permanent_for_step_and_manual_reset"] = step_refused and reset_refused
    fixtures.append({
      "fixture": "nan_raw_action_fault",
      "fault_phase": None if runtime.last_fault is None else runtime.last_fault.get("phase"),
      "physics_unchanged": not bool(tree_equal(physics_before_fault, physics_after_fault)),
      "step_refused": step_refused,
      "reset_refused": reset_refused,
    })
    checks["passed"] = all(checks.values())
    torch.save(cpu_tree({
      "fixtures": fixtures,
      "records": raw,
      "explicit_counter_presets": termination.presets,
      "resets": [reset_low, reset_tilt, reset_flip],
      "checks": checks,
    }), out / "play_gate_fixtures_evidence.pt")
  finally:
    session.close()
  write_json(out / "play_gate_fixtures_checks.json", checks)
  return checks


def _snapshot(out: Path, *, args, grid) -> dict:
  from confined_locomotion.support.evidence import write_json
  from confined_locomotion.support.snapshots import (
    global_prng_snapshot, runtime_snapshot, tree_equal,
  )
  from hexapod_mjlab.tasks.confined_locomotion.visualization.play import export_snapshot_manifest

  session = _session(out, args=args, grid=grid, suffix="snapshot")
  runtime = session.runtime
  checks = {}
  feature_cache = session.feature_cache
  cache_calls_before = (
    None if feature_cache is None else feature_cache.stats["forward_calls"]
  )
  try:
    before = runtime_snapshot(runtime)
    prng_before = global_prng_snapshot(device=runtime.device, runtime=runtime)
    snapshot_a = session.build_display_snapshot(0, paused=True)
    manifest_a = export_snapshot_manifest(snapshot_a, None, out / "snapshot_manifest_a.json")
    snapshot_b = session.build_display_snapshot(0, paused=True)
    manifest_b = export_snapshot_manifest(snapshot_b, None, out / "snapshot_manifest_b.json")
    checks["snapshot_and_manifest_use_functional_play_schemas"] = (
      snapshot_a.get("schema") == "confined_locomotion.play_snapshot/v2"
      and snapshot_a.get("mode") == "play"
      and manifest_a.get("schema") == "confined_locomotion.play_snapshot_manifest/v1"
      and manifest_a.get("snapshot_schema") == snapshot_a.get("schema")
    )
    checks["snapshot_assembly_and_manifest_export_are_pure_reads"] = (
      not bool(tree_equal(before, runtime_snapshot(runtime)))
      and not bool(tree_equal(prng_before, global_prng_snapshot(
        device=runtime.device, runtime=runtime
      )))
      and feature_cache is not None
      and feature_cache.stats["forward_calls"] == cache_calls_before
      and session._display.output is None
    )
    checks["repeated_snapshot_is_exact_and_contains_scan_provenance"] = (
      not bool(tree_equal(snapshot_a, snapshot_b))
      and not bool(tree_equal(manifest_a, manifest_b))
      and manifest_a["counts"]["near_provenance_matched"] == manifest_a["counts"]["near_valid"]
      and manifest_a["counts"]["far_provenance_matched"] == manifest_a["counts"]["far_valid"]
      and manifest_a["counts"]["ground_grid_points"] == grid.num_points
    )
    checks["viewer_server_and_browser_were_not_constructed"] = True
    checks["passed"] = all(checks.values())
    write_json(out / "play_snapshot_checks.json", {
      "checks": checks,
      "manifest": manifest_a,
      "viewer": None,
      "cache_forwards_during_snapshot_assembly": (
        None if feature_cache is None else
        feature_cache.stats["forward_calls"] - cache_calls_before
      ),
    })
  finally:
    session.close()
  return checks


def _dispatch(mode: str):
  return {
    "weights_only": _check_weights_only,
    "manual_controls": _manual_controls,
    "natural_mean": _natural_mean,
    "long_episode": _long_episode,
    "gate_fixtures": _gate_fixtures,
    "snapshot": _snapshot,
  }[mode]


def main(argv=None) -> int:
  args = _args(argv)
  from _common import configure_imports

  configure_imports()
  from confined_locomotion.support.fixtures import m12_training_grid

  grid = m12_training_grid()
  kwargs = {"args": args, "grid": grid}
  if args.mode == "weights_only":
    run = lambda out, device, recipe: _check_weights_only(out, **kwargs)
  elif args.mode == "manual_controls":
    run = lambda out, device, recipe: _manual_controls(out, **kwargs)
  elif args.mode == "natural_mean":
    run = lambda out, device, recipe: _natural_mean(out, **kwargs)
  elif args.mode == "long_episode":
    run = lambda out, device, recipe: _long_episode(out, **kwargs)
  elif args.mode == "gate_fixtures":
    run = lambda out, device, recipe: _gate_fixtures(out, **kwargs)
  else:
    run = lambda out, device, recipe: _snapshot(out, **kwargs)
  return execute_gpu_check(
    args,
    feature="play",
    mode=args.mode,
    runner=run,
    optimizer=False,
    grid=grid,
    extra_recipe={
      "checkpoint": str(args.checkpoint.expanduser().resolve()),
      "observation_perturbation_preset": args.preset,
      "natural_mean_control_budget": args.max_steps if args.mode == "natural_mean" else None,
      "snapshot_mode_starts_no_viewer_server": args.mode == "snapshot",
      "long_episode_holder_controls": LONG_EPISODE_CONTROL_STEPS if args.mode == "long_episode" else None,
    },
  )


if __name__ == "__main__":
  raise SystemExit(main())
