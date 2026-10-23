#!/usr/bin/env python
"""Verify the real official TrainConfig.launch formal-training path on CUDA."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import add_common_arguments, execute_gpu_check


MODES = ("official_formal",)


class _OfficialTrainerEvidence:
  """Attach raw minibatch capture only to the runner created by official launch."""

  def __init__(self, output_dir: Path):
    self.output_dir = output_dir
    self.observer = None
    self.runner = None

  def __call__(self, event, runner, payload):
    from confined_locomotion.support.observer import TrainerObserver

    if event == "constructed":
      if self.runner is not None:
        raise RuntimeError("official launch constructed more than one task runner")
      self.runner = runner
      self.observer = TrainerObserver(
        runner.trainer, self.output_dir / "official_minibatches", full_gradients=True
      )
      self.observer.__enter__()
    elif event in ("learn_finished", "learn_failed", "learn_not_passed"):
      if self.observer is not None:
        self.observer.__exit__(None, None, None)

  def close(self):
    if self.observer is not None and self.observer._active:
      self.observer.__exit__(None, None, None)


def _run_official_formal(out: Path, device: str, recipe: dict, *, args, grid) -> dict:
  from confined_locomotion.support.entry_witness import (
    WrapperInitialResetWitness,
    clear_official_entry_observers,
    install_official_entry_observer,
    write_witness,
  )
  from confined_locomotion.support.snapshots import cpu_tree
  from hexapod_mjlab.tasks.confined_locomotion.rl.entry import launch_formal_training
  from hexapod_mjlab.tasks.confined_locomotion.rl.runner import FORMAL_RUNNER_IDENTITY

  del device
  checks = {}
  result = None
  witness = install_official_entry_observer(out / "official_events")
  trainer_evidence = _OfficialTrainerEvidence(out)
  from hexapod_mjlab.tasks.confined_locomotion.rl.runner import add_engineering_observer

  add_engineering_observer(trainer_evidence)
  wrapper_reset = WrapperInitialResetWitness()
  try:
    with wrapper_reset:
      result = launch_formal_training(
        seed=args.seed,
        ground_grid=grid,
        num_envs=args.num_envs,
        max_iterations=1,
        log_root=out / "official_logs",
        gpu_ids=(args.gpu,),
        engineering_validation=True,
      )
  finally:
    trainer_evidence.close()
    clear_official_entry_observers()

  if result is None:
    raise RuntimeError("official launch returned without a FormalLaunchResult")
  runner = witness.runner
  run_dir = result.run_dir
  summary = result.summary
  events = witness.summary()
  trainer = trainer_evidence.observer
  minibatches = [] if trainer is None else [
    item for item in trainer.events if item["stage"] == "minibatch_forward"
  ]
  checks["official_formal_runner_identity_and_mode"] = (
    runner is not None
    and runner.runner_identity == FORMAL_RUNNER_IDENTITY
    and runner.formal_training is True
    and runner.runtime.mode == "training"
    and runner.recipe.get("mode") == "formal_training"
  )
  checks["official_wrapper_constructor_reset_exactly_once"] = (
    wrapper_reset.summary()["exactly_one_wrapper_reset_each"]
    and wrapper_reset.summary()["initial_reset_counts"] == [1]
  )
  checks["official_initial_takeover_counters_were_fresh"] = (
    events["initial_counters"] is not None
    and events["initial_counters"]["runtime_control_steps"] == 0
    and events["initial_counters"]["env_common_step_counter"] == 0
    and events["initial_counters"]["simulation_step_counter"] == 0
    and events["initial_counters"]["episode_length_buf"].tolist() == [0, 0]
  )
  checks["official_learn_and_pre_close_boundary_witnessed"] = (
    events["event_names"] == ["constructed", "iteration_completed", "learn_finished"]
    and events["constructed_runner_observed"]
    and events["official_wrapper_observed"]
    and events["pre_close_boundary_observed"]
  )
  boundary = witness.live_boundary
  checks["official_24_control_20_adam_iteration_completed"] = (
    boundary is not None
    and boundary["runner"]["current_learning_iteration"] == 1
    and boundary["runner"]["total_control_batches"] == 24
    and boundary["runner"]["total_env_samples"] == 24 * args.num_envs
    and boundary["runner"]["optimizer_step_count"] == 20
    and boundary["runner"]["update_count"] == 1
    and boundary["runner"]["adam_param_groups"] == 1
  )
  checks["official_single_adam_full_minibatch_evidence"] = (
    trainer is not None
    and len(minibatches) == 20
    and trainer.summary()["call_counts"] == {
      "backward": 20, "joint_clip": 20, "adam_step": 20
    }
    and trainer.summary()["actual_call_order"] == [
      "backward", "joint_clip", "adam_step"
    ] * 20
    and trainer.summary()["all_clip_calls_used_exact_parameter_set"]
    and trainer.summary()["all_steps_used_expected_adam"]
    and all("batch" in item and "network_output" in item for item in minibatches)
  )
  checks["official_formal_commit_and_close_proof"] = (
    result.close_completed is True
    and summary.get("formal_entry_verification", {}).get("official_launch_returned") is True
    and summary.get("formal_entry_verification", {}).get("official_close_completed") is True
    and summary.get("formal_run_completed") is True
    and summary.get("environment_close", {}).get("completed") is True
    and summary.get("final_checkpoint_commit", {}).get("state") == "committed"
    and result.checkpoint_path.is_file()
  )
  checks["functional_logs_and_explicit_grid_identity"] = (
    summary.get("mode") == "formal_training"
    and summary.get("grid_identity_hash") == grid.identity_hash()
    and runner.recipe.get("grid", {}).get("identity") == grid.identity()
    and (run_dir / "run_recipe.json").is_file()
    and (run_dir / "iteration_records.jsonl").is_file()
    and (run_dir / "training_summary.json").is_file()
    and result.checkpoint_path.name == "final_checkpoint_full_000001.pt"
  )
  checks["no_learning_effect_claim"] = (
    summary.get("learning_effect_ready") is False
    and summary.get("capabilities", {}).get("learning_effect_ready") is False
  )
  checks["passed"] = all(checks.values())
  trainer_evidence.close()
  write_witness(out, witness, checks, wrapper_reset=wrapper_reset)
  from confined_locomotion.support.evidence import write_json

  write_json(out / "official_result.json", {
    "mode": "formal_training",
    "run_dir": str(run_dir),
    "checkpoint_path": str(result.checkpoint_path),
    "close_completed": result.close_completed,
    "summary": summary,
  })
  import torch

  torch.save(cpu_tree({
    "initial_counters": events["initial_counters"],
    "pre_close_boundary": boundary,
    "trainer_events_all_minibatches": [] if trainer is None else trainer.events,
    "optimizer_calls": [] if trainer is None else trainer.calls,
    "checks": checks,
  }), out / "official_entry_evidence.pt")
  return checks


def _parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--mode", required=True, choices=MODES)
  add_common_arguments(parser)
  return parser.parse_args(argv)


def main(argv=None) -> int:
  args = _parse_args(argv)
  from _common import configure_imports

  configure_imports()
  from confined_locomotion.support.fixtures import m12_training_grid

  grid = m12_training_grid()
  return execute_gpu_check(
    args,
    feature="entry",
    mode=args.mode,
    runner=lambda out, device, recipe: _run_official_formal(
      out, device, recipe, args=args, grid=grid
    ),
    optimizer=True,
    grid=grid,
    extra_recipe={
      "official_entry": "mjlab.scripts.train.TrainConfig.launch_training",
      "formal_mode": True,
      "max_iterations": 1,
      "wrapper_initial_reset_witness": True,
    },
  )


if __name__ == "__main__":
  raise SystemExit(main())
