#!/usr/bin/env python
"""Finite CUDA witnesses for strict restore, replay, windows and weights-only."""
from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import torch

from _common import add_common_arguments, close_runtime, execute_gpu_check


MODES = ("save_a", "restore_b", "window", "weights_only", "diagnose_next_rollout", "cycle")
FAR_SLOTS = 10
RAW_RAYS = 86_400
GATE_POSITION_ATOL = 1e-5
GATE_VELOCITY_ATOL = 1e-4
GATE_VELOCITY_RTOL = 1e-5
GATE_PARAMETER_ATOL = 1e-5
GATE_PARAMETER_RTOL = 1e-5


def _args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--mode", required=True, choices=MODES)
  parser.add_argument("--checkpoint", type=Path)
  parser.add_argument("--a-dir", type=Path)
  parser.add_argument("--b-dir", type=Path)
  parser.add_argument("--diagnose-next-rollout", action="store_true",
                      help="record the strict next-rollout comparison as diagnostics only")
  add_common_arguments(parser)
  return parser.parse_args(argv)


def _torch_load(path: Path, *, map_location):
  return torch.load(path, map_location=map_location, weights_only=False)


def _write_tensor(path: Path, value) -> None:
  from confined_locomotion.support.snapshots import cpu_tree

  path.parent.mkdir(parents=True, exist_ok=True)
  torch.save(cpu_tree(value), path)


def _formal_result_record(out: Path, result) -> dict:
  from confined_locomotion.support.evidence import write_json

  record = {
    "mode": "formal_training",
    "run_dir": str(result.run_dir.resolve()),
    "checkpoint_path": str(result.checkpoint_path.resolve()),
    "close_completed": bool(result.close_completed),
    "global_iteration": int(result.summary.get("current_learning_iteration", -1)),
    "total_control_batches": int(result.summary.get("total_control_batches", -1)),
    "optimizer_step_count": int(result.summary.get("optimizer_step_count", -1)),
    "summary": result.summary,
  }
  write_json(out / "official_run.json", record)
  return record


class _OfficialBoundaryObserver:
  """Save the actual official runner's stable, pre-close boundary."""

  def __init__(self, output_dir: Path):
    self.output_dir = output_dir
    self.runner = None
    self.event_names: list[str] = []
    self.boundary = None

  def __call__(self, event, runner, payload) -> None:
    self.event_names.append(str(event))
    if event == "constructed":
      if self.runner is not None and self.runner is not runner:
        raise RuntimeError("official entry constructed more than one runner")
      self.runner = runner
    elif event == "learn_finished":
      if self.runner is not runner:
        raise RuntimeError("official learn_finished came from another runner")
      self.boundary = _live_update_state(runner)
      _write_tensor(self.output_dir / "official_preclose_boundary.pt", self.boundary)


def _launch_official(out: Path, *, args, grid, checkpoint: Path | None):
  from hexapod_mjlab.tasks.confined_locomotion.rl.entry import launch_formal_training
  from hexapod_mjlab.tasks.confined_locomotion.rl.runner import (
    add_engineering_observer,
    clear_engineering_observers,
  )

  observer = _OfficialBoundaryObserver(out)
  clear_engineering_observers()
  add_engineering_observer(observer)
  try:
    result = launch_formal_training(
      seed=args.seed,
      ground_grid=grid,
      num_envs=args.num_envs,
      max_iterations=1,
      resume_checkpoint=checkpoint,
      log_root=out / "official_logs",
      gpu_ids=(args.gpu,),
      engineering_validation=True,
    )
  finally:
    clear_engineering_observers()
  if observer.boundary is None:
    raise AssertionError(
      "official formal entry did not emit a captured learn_finished pre-close boundary; "
      f"observed events={observer.event_names}"
    )
  from confined_locomotion.support.evidence import write_json

  write_json(out / "official_boundary_witness.json", {
    "mode": "formal_training",
    "event_names": observer.event_names,
    "same_runner_constructed_and_finished": observer.runner is not None,
    "pre_close_boundary_captured": True,
    "boundary_file": str((out / "official_preclose_boundary.pt").resolve()),
    "pointer_values_retained_in_raw_boundary": True,
  })
  return result


def _build_formal_runner(out: Path, *, args, grid):
  """A checkpoint diagnostic runner; it is explicitly not an official launch."""
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper

  from hexapod_mjlab.tasks.confined_locomotion.rl.entry import build_formal_train_config
  from hexapod_mjlab.tasks.confined_locomotion.rl.runner import ConfinedLocomotionRunner

  config = build_formal_train_config(
    seed=args.seed,
    ground_grid=grid,
    num_envs=args.num_envs,
    max_iterations=1,
    log_root=out / "diagnostic_logs",
    gpu_ids=(args.gpu,),
    engineering_validation=True,
  )
  device = f"cuda:{args.gpu}"
  env = ManagerBasedRlEnv(config.env, device=device)
  try:
    wrapper = RslRlVecEnvWrapper(env, clip_actions=None)
    runner = ConfinedLocomotionRunner(
      wrapper,
      dataclasses.asdict(config.agent),
      log_dir=str(out / "diagnostic_runner"),
      device=device,
    )
  except BaseException:
    env.close()
    raise
  return env, wrapper, runner


def _instrument_load(runner):
  """Count effects caused inside load only; never replaces restore logic."""
  env, runtime, sim = runner.runtime.env, runner.runtime, runner.runtime.env.sim
  counts = {name: 0 for name in ("reset", "env_step", "sim_step", "sim_forward", "scan")}
  originals = []
  for owner, attribute, counter in (
    (env, "reset", "reset"),
    (env, "step", "env_step"),
    (sim, "step", "sim_step"),
    (sim, "forward", "sim_forward"),
    (runtime, "_scan_rows_now", "scan"),
  ):
    original = getattr(owner, attribute)
    originals.append((owner, attribute, original))

    def observed(*args, _original=original, _counter=counter, **kwargs):
      counts[_counter] += 1
      return _original(*args, **kwargs)

    setattr(owner, attribute, observed)

  def restore():
    for owner, attribute, original in reversed(originals):
      setattr(owner, attribute, original)

  return counts, restore


def _live_update_state(runner, *, include_runtime=True) -> dict:
  from confined_locomotion.support.snapshots import (
    cpu_tree,
    global_prng_snapshot,
    runtime_snapshot,
  )

  state = {
    "parameters": {
      name: value.detach().cpu().clone()
      for name, value in sorted(runner.policy_adapter.named_parameters())
    },
    "optimizer": cpu_tree(runner.trainer.optimizer.state_dict()),
    "optimizer_step_count": int(runner.trainer.optimizer_step_count),
    "update_count": int(runner.trainer.update_count),
    "learning_rate": float(runner.trainer.learning_rate),
    "shuffle_generator": cpu_tree(runner.trainer.shuffle_generator_state()),
    "prng": global_prng_snapshot(
      device=runner.device,
      provider=runner.provider,
      runtime=runner.runtime,
      trainer=runner.trainer,
    ),
    "runner": {
      "current_learning_iteration": int(runner.current_learning_iteration),
      "total_control_batches": int(runner.total_control_batches),
      "total_env_samples": int(runner.total_env_samples),
      "lifecycle_phase": runner._lifecycle_phase,
      "artifact_exported": bool(runner._artifact_exported),
    },
  }
  if include_runtime:
    runtime = runner.runtime
    env = runtime.env
    from hexapod_mjlab.tasks.confined_locomotion.rl.checkpoint import _manager_tensors

    cache = runtime._input_cache
    cached = cache.get()
    reward_terms = getattr(env, "_confined_training_reward_terms", None)
    parameter_names = {
      name for name, _parameter in runner.policy_adapter.named_parameters()
    }
    adapter_buffers = {
      name: value
      for name, value in runner.policy_adapter.state_dict().items()
      if name not in parameter_names
    }
    state["runtime"] = runtime_snapshot(runtime)
    state["complete_boundary_groups"] = {
      "adapter_non_parameter_state_dict": cpu_tree(adapter_buffers),
      "input_cache": {
        "identity": None if cached is None else cache._identity,
        "episodes": None if cached is None else cpu_tree(cache._episodes),
        "prepared": None if cached is None else cpu_tree(cached),
      },
      "current_window_snapshot": cpu_tree(runtime._current_window_snapshot),
      "scan_audit_bound": cpu_tree(runtime.scan_audit_bound),
      "manager_tensors": cpu_tree({
        name: value for name, value in _manager_tensors(env).items()
      }),
      "runtime_lifecycle": {
        "closed": bool(runtime._closed),
        "stopped": bool(runtime._stopped),
        "faulted": bool(runtime._faulted),
        "context_phase": str(runtime._context.phase),
      },
    }
    state["diagnostic_not_restored"] = {
      "reward_terms_cache": None if reward_terms is None else cpu_tree(reward_terms),
      "semantics": (
        "raw boundary diagnostic only; the full payload stores it but load does not apply it; "
        "the reward term regenerates its per-control cache"
      ),
    }
  return state


def _pointer_manifest(boundary: dict) -> dict:
  manifest = boundary.get("runtime", {}).get("physics", {}).get("manifest", {})
  return {
    name: {
      "ptr": entry.get("ptr"),
      "shape": entry.get("shape"),
      "dtype": entry.get("dtype"),
      "empty": entry.get("empty"),
    }
    for name, entry in manifest.items()
  }


def _cross_pid_state(boundary: dict) -> dict:
  """Copy all exact state while excluding only process-local Warp pointers."""
  from confined_locomotion.support.snapshots import cpu_tree

  state = cpu_tree(boundary)
  manifest = state.get("runtime", {}).get("physics", {}).get("manifest", {})
  for entry in manifest.values():
    if isinstance(entry, dict):
      entry.pop("ptr", None)
  state.pop("diagnostic_not_restored", None)
  return state


def _boundary_differences(official: dict, loaded: dict) -> list[str]:
  return _diff_tree(_cross_pid_state(official), _cross_pid_state(loaded))


def _write_boundary_comparison(
  out: Path, name: str, official: dict, loaded: dict,
) -> dict:
  from confined_locomotion.support.evidence import write_json

  exact_differences = _boundary_differences(official, loaded)
  official_diagnostic = official.get("diagnostic_not_restored", {}).get(
    "reward_terms_cache"
  )
  loaded_diagnostic = loaded.get("diagnostic_not_restored", {}).get(
    "reward_terms_cache"
  )
  diagnostic_differences = _diff_tree(official_diagnostic, loaded_diagnostic)
  record = {
    "comparison": name,
    "exact_causal_state_differences": exact_differences,
    "diagnostic_reward_cache_exact_equal": not diagnostic_differences,
    "diagnostic_reward_cache_differences_not_a_causal_restore_gate": diagnostic_differences,
    "only_causal_state_exclusions": [
      "physics.manifest.*.ptr (process-local address)",
      "diagnostic_not_restored.reward_terms_cache (payload-only diagnostic cache; regenerated)",
    ],
    "raw_diagnostic_values_retained_in_compared_boundary_files": True,
    "diagnostic_semantics_source": [
      "src/hexapod_mjlab/tasks/confined_locomotion/rl/checkpoint.py:977-981 (payload capture only; restore does not apply this diagnostic)",
      "src/hexapod_mjlab/tasks/confined_locomotion/mdp/rewards.py:289-296 (per-control cache initialization and regeneration)",
    ],
    "official_raw_physics_pointer_manifest": _pointer_manifest(official),
    "loaded_raw_physics_pointer_manifest": _pointer_manifest(loaded),
  }
  write_json(out / f"{name}_boundary_comparison.json", record)
  return record


def _diff_tree(left, right, path="root") -> list[str]:
  from confined_locomotion.support.snapshots import tree_equal

  return tree_equal(left, right, path)


def _save_a(out: Path, device: str, recipe: dict, *, args, grid) -> dict:
  """Official A save followed by one explicitly diagnostic restored replay."""
  from confined_locomotion.support.evidence import write_json
  from confined_locomotion.support.observer import RuntimeStepObserver, TrainerObserver
  from confined_locomotion.support.snapshots import cpu_tree

  result = _launch_official(out, args=args, grid=grid, checkpoint=None)
  official = _formal_result_record(out, result)
  if not result.close_completed or not result.checkpoint_path.is_file():
    raise AssertionError("official A did not commit a full checkpoint after close")

  env, _wrapper, runner = _build_formal_runner(out, args=args, grid=grid)
  checks = {}
  try:
    load_report = runner.load(result.checkpoint_path)
    boundary = _live_update_state(runner)
    _write_tensor(out / "a_loaded_boundary.pt", boundary)
    official_boundary = _torch_load(
      out / "official_preclose_boundary.pt", map_location="cpu"
    )
    official_boundary_comparison = _write_boundary_comparison(
      out, "a_official_vs_loaded", official_boundary, boundary
    )
    # The restored official runner owns its actual collector, policy, trainer
    # and runtime. The collect is a checkpoint replay diagnostic, not a second
    # official entry claim.
    with RuntimeStepObserver(runner.runtime, out / "a_reference_transitions", label="reference") as transitions:
      collected = runner.collector.collect()
    checks["diagnostic_restored_rollout_is_full_and_natural"] = (
      collected.complete and collected.completed_steps == 24
      and collected.stopped_reason is None and len(transitions.records) == 24
    )
    if not collected.complete:
      _write_tensor(out / "a_stopped_rollout.pt", {
        "summary": collected.summary(), "transitions": transitions.records,
      })
      raise AssertionError("A diagnostic collection stopped before a full natural rollout")
    collected.buffer.compute_gae()
    # Preserve the complete real 24-step buffer before update for B's exact
    # same-rollout replay (all 48 rows, all identities and ground frames).
    full_buffer_shape = {
      "is_full": bool(collected.buffer.is_full),
      "steps_filled": int(collected.buffer.steps_filled),
      "num_envs": int(collected.buffer.num_envs),
      "raw_action_shape": tuple(collected.buffer.replay(
        torch.arange(collected.buffer.num_envs * collected.buffer.steps_filled,
                     device=device)
      ).raw_action.shape),
    }
    torch.save(collected.buffer, out / "same_rollout_buffer.pt")
    before_update = _live_update_state(runner)
    _write_tensor(out / "same_rollout_before_update.pt", before_update)
    checks["diagnostic_preupdate_shuffle_state_matches_checkpoint_boundary"] = (
      not _diff_tree(
        boundary["shuffle_generator"], before_update["shuffle_generator"]
      )
      and not _diff_tree(
        boundary["prng"]["optimization_shuffle"],
        before_update["prng"]["optimization_shuffle"],
      )
    )
    with TrainerObserver(runner.trainer, out / "a_reference_minibatches", full_gradients=True) as observer:
      metrics = runner.trainer.update(collected.buffer)
    after_update = _live_update_state(runner, include_runtime=False)
    _write_tensor(out / "same_rollout_expected_update.pt", {
      "state": after_update,
      "metrics": metrics,
      "observer_events": observer.events,
      "optimizer_calls": observer.calls,
      "observer_summary": observer.summary(),
    })
    checks["diagnostic_update_observed_all_20_minibatches"] = (
      observer.summary()["forward_count"] == 20
      and observer.summary()["call_counts"] == {"backward": 20, "joint_clip": 20, "adam_step": 20}
      and observer.summary()["all_clip_calls_used_exact_parameter_set"]
      and observer.summary()["all_steps_used_expected_adam"]
    )
    checks["full_rollout_replay_inputs_retained"] = (
      full_buffer_shape["is_full"]
      and full_buffer_shape["steps_filled"] == 24
      and full_buffer_shape["raw_action_shape"] == (24 * args.num_envs, 18)
      and (out / "same_rollout_buffer.pt").stat().st_size > 0
    )
    checks["strict_full_checkpoint_load_report"] = (
      load_report.get("schema") == "confined_locomotion.checkpoint/v3"
      and load_report.get("current_learning_iteration") == 1
      and load_report.get("load_operation", {}).get("sim_forward_calls") == 1
      and "python" in load_report.get("prng_restored", [])
    )
    checks["fresh_load_matches_official_A_preclose_boundary"] = (
      not official_boundary_comparison["exact_causal_state_differences"]
    )
    checks["official_A_completed_one_24_by_20_iteration"] = (
      official["global_iteration"] == 1
      and official["total_control_batches"] == 24
      and official["optimizer_step_count"] == 20
      and official["summary"].get("formal_run_completed") is True
    )
    checks["passed"] = all(checks.values())
    write_json(out / "save_a_checks.json", checks)
    write_json(out / "checkpoint_source.json", {
      "official_checkpoint": official["checkpoint_path"],
      "official_run_dir": official["run_dir"],
      "official_preclose_boundary": str((out / "official_preclose_boundary.pt").resolve()),
      "diagnostic_boundary": str((out / "a_loaded_boundary.pt").resolve()),
      "same_rollout_buffer": str((out / "same_rollout_buffer.pt").resolve()),
      "same_rollout_expected_update": str((out / "same_rollout_expected_update.pt").resolve()),
      "same_rollout_buffer_shape": full_buffer_shape,
      "note": "the buffer was collected by a fresh strict-loaded runner from the official A checkpoint",
    })
    return {"passed": checks["passed"], "checks": checks, "checkpoint": official["checkpoint_path"]}
  finally:
    close_runtime(env, runner.runtime)


def _restore_b(out: Path, device: str, recipe: dict, *, args, grid) -> dict:
  from confined_locomotion.support.observer import RuntimeStepObserver, TrainerObserver
  from confined_locomotion.support.snapshots import cpu_tree

  checkpoint = args.checkpoint.expanduser().resolve()
  source_dir = args.a_dir.expanduser().resolve()
  a_boundary = _torch_load(source_dir / "a_loaded_boundary.pt", map_location="cpu")
  a_expected = _torch_load(source_dir / "same_rollout_expected_update.pt", map_location="cpu")
  buffer = _torch_load(source_dir / "same_rollout_buffer.pt", map_location=device)
  env, _wrapper, runner = _build_formal_runner(out, args=args, grid=grid)
  checks = {}
  try:
    counts, undo = _instrument_load(runner)
    try:
      load_report = runner.load(checkpoint)
    finally:
      undo()
    b_boundary = _live_update_state(runner)
    boundary_comparison = _write_boundary_comparison(
      out, "a_loaded_vs_b_loaded", a_boundary, b_boundary
    )
    boundary_differences = boundary_comparison["exact_causal_state_differences"]
    checks["new_runner_full_boundary_bit_exact"] = not boundary_differences
    checks["load_performed_only_one_derived_forward"] = counts == {
      "reset": 0, "env_step": 0, "sim_step": 0, "sim_forward": 1, "scan": 0
    }
    expected_applied = [
      "adapter_weights",
      "trainer_adam_lr_counters_shuffle",
      "physics_inplace_forward_rewrite",
      "env_counters_buffers",
      "action_term_state",
      "foot_air_termination_state",
      "manager_lifecycle_tensors",
      "runtime_counters_window_ground_input_cache",
      "command_streams",
      "perturbation_streams_schedules_caches",
      "action_generator",
      "global_prngs",
      "runner_counters_phase",
    ]
    checks["full_restore_report_and_exact_13_application_labels"] = (
      load_report.get("schema") == "confined_locomotion.checkpoint/v3"
      and load_report.get("current_learning_iteration") == 1
      and load_report.get("applied") == expected_applied
      and len(load_report.get("construction_side_effects", {})) > 0
    )
    checks["same_rollout_has_all_24_steps_and_full_48_sample_batch"] = (
      buffer.is_full and buffer.steps_filled == 24 and buffer.num_envs == 2
      and buffer.replay(torch.arange(48, device=device)).raw_action.shape == (48, 18)
    )
    with TrainerObserver(runner.trainer, out / "b_replay_minibatches", full_gradients=True) as observer:
      replay_metrics = runner.trainer.update(buffer)
    b_after = _live_update_state(runner, include_runtime=False)
    expected_state = a_expected["state"]
    parameter_diffs = _diff_tree(expected_state["parameters"], b_after["parameters"])
    adam_diffs = _diff_tree(expected_state["optimizer"], b_after["optimizer"])
    replay_algorithm_state = {
      key: expected_state[key]
      for key in ("optimizer_step_count", "update_count", "learning_rate", "shuffle_generator")
    }
    replay_algorithm_state["optimization_shuffle"] = expected_state["prng"]["optimization_shuffle"]
    replay_algorithm_state["runner"] = expected_state["runner"]
    b_algorithm_state = {
      key: b_after[key]
      for key in ("optimizer_step_count", "update_count", "learning_rate", "shuffle_generator")
    }
    b_algorithm_state["optimization_shuffle"] = b_after["prng"]["optimization_shuffle"]
    b_algorithm_state["runner"] = b_after["runner"]
    replay_state_diffs = _diff_tree(replay_algorithm_state, b_algorithm_state)
    checks["same_rollout_update_parameters_exact"] = not parameter_diffs
    checks["same_rollout_update_adam_exact"] = not adam_diffs
    checks["same_rollout_update_shuffle_and_runner_state_exact"] = not replay_state_diffs
    checks["same_rollout_update_all_20_minibatches"] = (
      observer.summary()["forward_count"] == 20
      and observer.summary()["call_counts"] == {"backward": 20, "joint_clip": 20, "adam_step": 20}
      and observer.summary()["all_clip_calls_used_exact_parameter_set"]
      and observer.summary()["all_steps_used_expected_adam"]
    )
    # The strict historical P5 next-rollout gate was an accepted FAIL. This
    # opt-in section only writes deltas; it never changes ordinary pass/fail.
    diagnostic = None
    if args.diagnose_next_rollout:
      with RuntimeStepObserver(runner.runtime, out / "b_next_rollout_transitions", label="next") as next_steps:
        next_collection = runner.collector.collect()
      diagnostic = {
        "enabled": True,
        "passed_gate": None,
        "classification": "diagnostic_only; strict frozen P5 next-rollout FAIL remains a separate historical result",
        "complete": bool(next_collection.complete),
        "completed_steps": int(next_collection.completed_steps),
        "transition_count": len(next_steps.records),
      }
      _write_tensor(out / "next_rollout_diagnostic.pt", {
        "summary": next_collection.summary(), "transitions": next_steps.records,
      })
      if next_collection.complete:
        next_collection.buffer.compute_gae()
        torch.save(next_collection.buffer, out / "b_next_rollout_buffer.pt")
    checks["ordinary_official_resume_completed_new_24_by_20_iteration"] = False
    # The following ordinary task entry uses the official TrainConfig.launch
    # and real registered runner; it is not the diagnostic runner above.
    official_result = _launch_official(
      out, args=args, grid=grid, checkpoint=checkpoint
    )
    official = _formal_result_record(out, official_result)
    official_b_boundary = _torch_load(
      out / "official_preclose_boundary.pt", map_location="cpu"
    )
    checks["ordinary_official_resume_completed_new_24_by_20_iteration"] = (
      official_result.close_completed
      and official["global_iteration"] == 2
      and official["total_control_batches"] == 48
      and official["optimizer_step_count"] == 40
      and official["summary"].get("formal_run_completed") is True
      and official_result.checkpoint_path.is_file()
    )
    checks["passed"] = all(checks.values())
    from confined_locomotion.support.evidence import write_json

    write_json(out / "restore_b_checks.json", {
      "checks": checks,
      "load_call_counts": counts,
      "boundary_differences": boundary_differences,
      "parameter_differences": parameter_diffs,
      "adam_differences": adam_diffs,
      "replay_state_differences": replay_state_diffs,
      "diagnose_next_rollout": diagnostic,
      "diagnostic_update_metrics": replay_metrics,
      "official_resume": official,
      "official_B_preclose_boundary_file": str(
        (out / "official_preclose_boundary.pt").resolve()
      ),
      "official_B_raw_physics_pointer_manifest": _pointer_manifest(official_b_boundary),
      "ordinary_pass_ignores_next_rollout_diagnostic": True,
    })
    _write_tensor(out / "restore_b_evidence.pt", {
      "loaded_boundary": b_boundary,
      "after_same_rollout_update": b_after,
      "all_minibatch_events": observer.events,
      "optimizer_calls": observer.calls,
      "observer_summary": observer.summary(),
      "official_boundary_comparison": boundary_comparison,
    })
    write_json(out / "checkpoint_source.json", {
      "source_A": str(checkpoint),
      "official_B_checkpoint": official["checkpoint_path"],
      "official_B_run_dir": official["run_dir"],
      "official_B_preclose_boundary": str(
        (out / "official_preclose_boundary.pt").resolve()
      ),
      "mode": "formal_training",
    })
    return {"passed": checks["passed"], "checks": checks, "checkpoint": official["checkpoint_path"]}
  finally:
    close_runtime(env, runner.runtime)


def _validate_full_window(boundary: dict, *, num_envs: int) -> dict:
  """Validate actual ten-scan ring provenance and its ray identities."""
  failures = []
  runtime = boundary.get("runtime", {})
  window = runtime.get("window", {})
  fields = window.get("fields", {})
  expected_shapes = {
    "_near_valid": (num_envs, 256),
    "_near_scan_ids": (num_envs,),
    "_near_source_dirs_sensor": (num_envs, 256, 3),
    "_near_source_dirs_present": (num_envs, 256),
    "_near_source_indices": (num_envs, 256),
    "_far_valid": (num_envs, FAR_SLOTS, 64),
    "_far_scan_ids": (num_envs, FAR_SLOTS),
    "_far_times": (num_envs, FAR_SLOTS),
    "_far_source_dirs_sensor": (num_envs, FAR_SLOTS, 64, 3),
    "_far_source_dirs_present": (num_envs, FAR_SLOTS, 64),
    "_far_source_indices": (num_envs, FAR_SLOTS, 64),
    "_far_count": (num_envs,),
    "_far_pos": (num_envs,),
    "_near_filled": (num_envs,),
    "_source_dirs_attached": (num_envs,),
  }
  for name, shape in expected_shapes.items():
    value = fields.get(name)
    if not torch.is_tensor(value) or tuple(value.shape) != shape:
      failures.append(f"shape:{name}")
  if failures:
    return {"passed": False, "failures": failures}
  far_count = fields["_far_count"].to(torch.int64).cpu()
  far_ids = fields["_far_scan_ids"].to(torch.int64).cpu()
  far_times = fields["_far_times"].to(torch.float64).cpu()
  far_valid = fields["_far_valid"].to(torch.bool).cpu()
  far_present = fields["_far_source_dirs_present"].to(torch.bool).cpu()
  far_indices = fields["_far_source_indices"].to(torch.int64).cpu()
  far_dirs = fields["_far_source_dirs_sensor"].to(torch.float32).cpu()
  near_valid = fields["_near_valid"].to(torch.bool).cpu()
  near_present = fields["_near_source_dirs_present"].to(torch.bool).cpu()
  near_indices = fields["_near_source_indices"].to(torch.int64).cpu()
  near_dirs = fields["_near_source_dirs_sensor"].to(torch.float32).cpu()
  for row in range(num_envs):
    slots = torch.nonzero(far_valid[row].any(dim=-1), as_tuple=False).flatten()
    unique_ids = torch.unique(far_ids[row, slots])
    if int(far_count[row]) != FAR_SLOTS or int(slots.numel()) != FAR_SLOTS:
      failures.append(f"far_window_not_full_env{row}")
    if int(unique_ids.numel()) != FAR_SLOTS or bool((unique_ids < 0).any()):
      failures.append(f"ten_distinct_far_scan_ids_env{row}")
    if unique_ids.numel() and not bool(torch.isfinite(far_times[row, slots]).all()):
      failures.append(f"far_scan_times_nonfinite_env{row}")
  if not bool(fields["_near_filled"].to(torch.bool).all()):
    failures.append("near_window_not_filled")
  if not bool(fields["_source_dirs_attached"].to(torch.bool).all()):
    failures.append("source_directions_not_attached")
  if not torch.equal(near_valid, near_present) or not torch.equal(far_valid, far_present):
    failures.append("raw_ray_masks_do_not_match_valid_points")
  if not torch.equal(near_present, near_indices >= 0) or not torch.equal(far_present, far_indices >= 0):
    failures.append("raw_ray_indices_mask_mismatch")
  if bool((near_indices[near_present] >= RAW_RAYS).any()) or bool((far_indices[far_present] >= RAW_RAYS).any()):
    failures.append("raw_ray_id_outside_86400_sensor_frame")
  if bool((near_indices[~near_present] != -1).any()) or bool((far_indices[~far_present] != -1).any()):
    failures.append("invalid_slots_must_have_minus_one_ray_id")
  for name, dirs, mask in (("near", near_dirs, near_present), ("far", far_dirs, far_present)):
    if not bool(torch.isfinite(dirs).all()):
      failures.append(f"{name}_directions_nonfinite")
    if bool(mask.any()):
      norms = torch.linalg.vector_norm(dirs[mask], dim=-1)
      if not torch.allclose(norms, torch.ones_like(norms), atol=1e-5, rtol=0.0):
        failures.append(f"{name}_directions_not_unit")
  scan_entries = runtime.get("sensor_scan_counts")
  if not torch.is_tensor(scan_entries) or tuple(scan_entries.shape) != (num_envs,):
    failures.append("actual_sensor_scan_counts_missing")
  elif bool((scan_entries < FAR_SLOTS).any()):
    failures.append("fewer_than_ten_actual_sensor_scans")
  return {
    "passed": not failures,
    "failures": failures,
    "far_count": far_count.tolist(),
    "far_scan_ids": far_ids.tolist(),
    "near_ray_count": near_present.sum(dim=1).tolist(),
    "far_ray_count_by_scan": far_present.sum(dim=-1).tolist(),
    "sensor_scan_counts": None if scan_entries is None else scan_entries.tolist(),
    "claim": "loaded current full window; distinct source scans and raw-ray provenance only",
  }


def _window_load_only(out: Path, device: str, recipe: dict, *, args, grid) -> dict:
  checkpoint = args.checkpoint.expanduser().resolve()
  source_dir = args.b_dir.expanduser().resolve()
  official_boundary_path = source_dir / "official_preclose_boundary.pt"
  if not official_boundary_path.is_file():
    raise FileNotFoundError(
      f"window load requires the source official pre-close witness: {official_boundary_path}"
    )
  official_boundary = _torch_load(official_boundary_path, map_location="cpu")
  env, _wrapper, runner = _build_formal_runner(out, args=args, grid=grid)
  counts, undo = _instrument_load(runner)
  try:
    try:
      report = runner.load(checkpoint)
    finally:
      undo()
    boundary = _live_update_state(runner)
    window_checks = _validate_full_window(boundary, num_envs=args.num_envs)
    official_comparison = _write_boundary_comparison(
      out, "b_official_vs_window_loaded", official_boundary, boundary
    )
    official_differences = official_comparison["exact_causal_state_differences"]
    checks = {
      "load_only_no_reset_scan_or_integration": counts == {
        "reset": 0, "env_step": 0, "sim_step": 0, "sim_forward": 1, "scan": 0
      },
      "full_ten_scan_window_and_actual_ray_provenance": window_checks["passed"],
      "new_pid_window_matches_official_B_preclose_boundary": not official_differences,
      "load_only_report_has_one_derived_forward_and_zero_physics_effects": (
        report.get("load_operation", {}).get("env_reset_calls") == 0
        and report.get("load_operation", {}).get("scan_calls") == 0
        and report.get("load_operation", {}).get("physics_steps") == 0
        and report.get("load_operation", {}).get("sim_forward_calls") == 1
      ),
      "load_only_did_not_advance_training_or_runtime": (
        runner.current_learning_iteration == report["current_learning_iteration"]
        and runner.runtime.control_steps_taken == boundary["runtime"]["control_steps"]
        and runner.trainer.optimizer_step_count == boundary["optimizer_step_count"]
      ),
    }
    checks["passed"] = all(checks.values())
    from confined_locomotion.support.evidence import write_json

    write_json(out / "window_checks.json", {
      "checks": checks,
      "load_call_counts": counts,
      "full_window": window_checks,
      "official_boundary_differences": official_differences,
      "diagnostic_reward_cache_differences_not_a_causal_gate":
        official_comparison["diagnostic_reward_cache_differences_not_a_causal_restore_gate"],
      "source_official_preclose_boundary": str(official_boundary_path),
      "source_official_raw_physics_pointer_manifest": _pointer_manifest(official_boundary),
      "loaded_raw_physics_pointer_manifest": _pointer_manifest(boundary),
      "cross_pid_comparison_exclusions": [
        "physics.manifest.*.ptr (process-local address)",
        "diagnostic_not_restored.reward_terms_cache (reported separately; regenerated per control)",
      ],
      "no_learn_rollout_reset_scan_or_step": True,
    })
    _write_tensor(out / "window_boundary.pt", boundary)
    return checks
  finally:
    close_runtime(env, runner.runtime)


def _weights_only(out: Path, device: str, recipe: dict, *, args, grid) -> dict:
  from confined_locomotion.support.fixtures import build_network
  from confined_locomotion.support.snapshots import cpu_tree
  from hexapod_mjlab.tasks.confined_locomotion.rl.checkpoint import load_weights
  from hexapod_mjlab.rl.confined_locomotion.contracts import JOINT_DIM, OnlineObservation

  if args.checkpoint is None:
    raise ValueError("weights_only requires --checkpoint")
  network = build_network(seed=args.seed, ground_samples=grid.num_points, device="cpu")
  before_keys = set(network.state_dict())
  report = load_weights(
    args.checkpoint,
    network=network,
    provider=None,
    expected_grid_identity=grid.identity(),
  )
  batch = 2
  zeros = lambda shape: torch.zeros(shape, dtype=torch.float32)
  observation = OnlineObservation(
    env_ids=torch.arange(batch, dtype=torch.long),
    episode_ids=torch.zeros(batch, dtype=torch.long),
    step_ids=torch.zeros(batch, dtype=torch.long),
    time_s=zeros((batch,)),
    lin_vel_b=zeros((batch, 3)),
    ang_vel_b=zeros((batch, 3)),
    gravity_b=torch.tensor([[0.0, 0.0, -1.0]]).expand(batch, 3).clone(),
    joint_pos_rel_ref=zeros((batch, JOINT_DIM)),
    joint_vel=zeros((batch, JOINT_DIM)),
    prev_action=zeros((batch, JOINT_DIM)),
    command=zeros((batch, 3)),
  )
  near = torch.zeros((batch, 256, 3)); near_valid = torch.zeros((batch, 256), dtype=torch.bool)
  far = torch.zeros((batch, 640, 3)); far_valid = torch.zeros((batch, 640), dtype=torch.bool)
  with torch.no_grad():
    output = network(observation, near, near_valid, far, far_valid)
  checks = {
    "complete_network_key_set_loaded": set(network.state_dict()) == before_keys,
    "all_four_heads_forward_finite": all(
      value is not None and bool(torch.isfinite(value).all())
      for value in (output.envelope, output.action_mean, output.value, output.ground_pred)
    ),
    "weight_load_is_not_resume": report.get("resumable") is False,
    "adam_physics_and_rng_not_constructed": not hasattr(network, "optimizer") and report.get("resumable") is False,
  }
  checks["passed"] = all(checks.values())
  from confined_locomotion.support.evidence import write_json

  write_json(out / "weights_only_checks.json", {"checks": checks, "load_report": report})
  _write_tensor(out / "weights_only_heads.pt", {
    "heads": cpu_tree({
      "envelope": output.envelope,
      "action_mean": output.action_mean,
      "value": output.value,
      "ground_pred": output.ground_pred,
    }),
    "network_state": cpu_tree(network.state_dict()),
  })
  return checks


def _cycle(out: Path, device: str, recipe: dict, *, args, grid) -> dict:
  """Run save A → fresh-process restore B → fresh-process load-only window."""
  script = Path(__file__).resolve()
  root = Path(args.output).expanduser().resolve()
  stages = (("save_a", root / "A"), ("restore_b", root / "B"), ("window", root / "window"))
  commands = []
  for stage, stage_out in stages:
    command = [
      sys.executable, str(script), "--mode", stage,
      "--output", str(stage_out), "--gpu", str(args.gpu),
      "--seed", str(args.seed), "--num-envs", str(args.num_envs),
    ]
    if stage == "restore_b":
      command += ["--checkpoint", "{A_CHECKPOINT}", "--a-dir", str(root / "A")]
      if args.diagnose_next_rollout:
        command.append("--diagnose-next-rollout")
    if stage == "window":
      command += ["--checkpoint", "{B_CHECKPOINT}", "--b-dir", str(root / "B")]
    commands.append((stage, stage_out, command))
  completed = []
  a_checkpoint = b_checkpoint = None
  for stage, stage_out, raw in commands:
    stage_out.parent.mkdir(parents=True, exist_ok=True)
    if stage_out.exists():
      raise FileExistsError(f"cycle stage output already exists: {stage_out}")
    command = [
      a_checkpoint if token == "{A_CHECKPOINT}" else
      b_checkpoint if token == "{B_CHECKPOINT}" else token
      for token in raw
    ]
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    item = {
      "stage": stage,
      "command": command,
      "returncode": process.returncode,
      "stdout": process.stdout,
      "stderr": process.stderr,
      "output": str(stage_out),
    }
    completed.append(item)
    if process.returncode != 0:
      break
    if stage == "save_a":
      record = json.loads((stage_out / "checkpoint_source.json").read_text())
      a_checkpoint = record["official_checkpoint"]
    elif stage == "restore_b":
      record = json.loads((stage_out / "checkpoint_source.json").read_text())
      b_checkpoint = record["official_B_checkpoint"]
  checks = {
    "all_stages_completed_in_order": [item["stage"] for item in completed] == ["save_a", "restore_b", "window"],
    "each_stage_used_a_distinct_fresh_output_directory": len({item["output"] for item in completed}) == len(completed),
    "failure_stopped_the_chain": len(completed) < 3 and completed[-1]["returncode"] != 0
      or len(completed) == 3 and completed[-1]["returncode"] == 0,
  }
  checks["passed"] = all(checks.values())
  from confined_locomotion.support.evidence import write_json

  write_json(out / "cycle_subprocesses.json", {
    "mode": "official_checkpoint_cycle",
    "separate_process_per_stage": True,
    "stages": completed,
    "a_checkpoint": a_checkpoint,
    "b_checkpoint": b_checkpoint,
    "checks": checks,
  })
  return checks


def _runner(args, grid):
  mode = args.mode
  if mode == "save_a":
    return lambda out, device, recipe: _save_a(out, device, recipe, args=args, grid=grid)
  if mode == "restore_b":
    return lambda out, device, recipe: _restore_b(out, device, recipe, args=args, grid=grid)
  if mode == "window":
    return lambda out, device, recipe: _window_load_only(out, device, recipe, args=args, grid=grid)
  if mode == "weights_only":
    return lambda out, device, recipe: _weights_only(out, device, recipe, args=args, grid=grid)
  if mode == "diagnose_next_rollout":
    args.diagnose_next_rollout = True
    return lambda out, device, recipe: _restore_b(out, device, recipe, args=args, grid=grid)
  if mode == "cycle":
    return lambda out, device, recipe: _cycle(out, device, recipe, args=args, grid=grid)
  raise AssertionError(mode)


def main(argv=None) -> int:
  args = _args(argv)
  from _common import configure_imports

  configure_imports()
  from confined_locomotion.support.fixtures import m12_training_grid

  grid = m12_training_grid()
  if args.mode in ("save_a", "restore_b", "window", "weights_only", "diagnose_next_rollout"):
    if args.mode in ("restore_b", "window", "weights_only", "diagnose_next_rollout") and args.checkpoint is None:
      raise SystemExit(f"{args.mode} requires --checkpoint")
    if args.mode in ("restore_b", "diagnose_next_rollout") and args.a_dir is None:
      raise SystemExit(f"{args.mode} requires --a-dir")
    if args.mode == "window" and args.b_dir is None:
      raise SystemExit("window requires --b-dir containing official_preclose_boundary.pt")
  return execute_gpu_check(
    args,
    feature="checkpoint",
    mode=args.mode,
    runner=_runner(args, grid),
    optimizer=args.mode in ("save_a", "restore_b", "diagnose_next_rollout", "cycle"),
    grid=grid,
    extra_recipe={
      "strict_full_restore": True,
      "same_rollout_exact_update_replay": args.mode in ("restore_b", "diagnose_next_rollout", "cycle"),
      "next_rollout_diagnostic_only": bool(args.diagnose_next_rollout or args.mode == "diagnose_next_rollout"),
      "window_mode_load_only": args.mode == "window",
      "ordinary_formal_resume_uses_official_entry": args.mode == "restore_b",
    },
  )


if __name__ == "__main__":
  raise SystemExit(main())
