#!/usr/bin/env python
"""Real-runner lifecycle gates for post-commit logging and close failures."""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from _common import add_common_arguments, close_runtime, execute_gpu_check


PRECOMMIT_MODES = ("runtime_failure", "ppo_failure", "final_save_failure", "record_log_failure")
MODES = ("postcommit", "close_failure", *PRECOMMIT_MODES)


def _args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--mode", required=True, choices=MODES)
  add_common_arguments(parser)
  return parser.parse_args(argv)


def _build_runner(out: Path, *, args, grid):
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
  env = ManagerBasedRlEnv(config.env, device=f"cuda:{args.gpu}")
  try:
    wrapper = RslRlVecEnvWrapper(env, clip_actions=None)
    runner = ConfinedLocomotionRunner(
      wrapper, dataclasses.asdict(config.agent),
      log_dir=str(out / "diagnostic_runner"), device=f"cuda:{args.gpu}"
    )
  except BaseException:
    env.close()
    raise
  return env, runner


def _precommit_failure(out: Path, *, args, grid) -> dict:
  """Inject each frozen pre-commit failure at its real owning boundary."""
  from confined_locomotion.support.evidence import write_json

  env, runner = _build_runner(out, args=args, grid=grid)
  runtime = runner.runtime
  original_learn_close = runner._close_after_learn_error
  originals = {}
  close_events = []
  injected_error = None

  def record_close():
    try:
      original_learn_close()
    finally:
      close_events.append({
        "runtime_closed": bool(runtime._closed),
        "official_environment_closed": bool(runner._official_environment_closed),
        "runtime_faulted": bool(runtime._faulted),
        "runner_phase": runner._lifecycle_phase,
      })

  runner._close_after_learn_error = record_close
  try:
    if args.mode == "runtime_failure":
      originals["scan"] = runtime._scan_rows_now

      def injected_scan(_rows):
        raise RuntimeError("injected functional runtime scan fault")

      runtime._scan_rows_now = injected_scan
    elif args.mode == "ppo_failure":
      originals["adam_step"] = runner.trainer.optimizer.step

      def injected_step(*_args, **_kwargs):
        raise RuntimeError("injected functional PPO Adam-step fault")

      runner.trainer.optimizer.step = injected_step
    elif args.mode == "final_save_failure":
      originals["runner_save"] = runner.save

      def injected_save(_path, *_args, **_kwargs):
        from hexapod_mjlab.tasks.confined_locomotion.rl.checkpoint import CheckpointFault

        raise CheckpointFault("injected functional final full-checkpoint save fault")

      runner.save = injected_save
    elif args.mode == "record_log_failure":
      originals["append_record"] = runner._append_jsonl_record

      def injected_append(record):
        if isinstance(record, dict) and record.get("kind") == "formal_final_full_checkpoint":
          raise OSError("injected functional final checkpoint record log fault")
        return originals["append_record"](record)

      runner._append_jsonl_record = injected_append
    else:
      raise AssertionError(args.mode)

    try:
      runner.learn(1)
    except BaseException as error:
      injected_error = error
    checks = {}
    checks["selected_failure_was_reached_and_closed_by_runner_error_path"] = (
      injected_error is not None
      and len(close_events) == 1
      and close_events[0]["runtime_closed"]
      and close_events[0]["official_environment_closed"]
    )
    completed = [
      row for row in runner._records
      if isinstance(row, dict) and row.get("kind") == "iteration"
      and row.get("status") == "completed"
    ]
    full_checkpoints = list(Path(runner.log_dir).rglob("final_checkpoint_full_*.pt"))
    fault_files = list(Path(runner.log_dir).rglob("fault.json"))
    nonresumable = list(Path(runner.log_dir).rglob("NOT_A_RESUMABLE_CHECKPOINT.txt"))
    summary_path = Path(runner.log_dir) / "training_summary.json"
    summary = None
    if summary_path.is_file():
      import json

      summary = json.loads(summary_path.read_text())
    checks["failure_did_not_commit_a_successful_full_checkpoint"] = (
      not full_checkpoints and runner._final_checkpoint_path is None
    )
    checks["failure_kept_nonresumable_diagnostics"] = bool(fault_files and nonresumable)
    checks["closed_summary_does_not_claim_completion"] = (
      summary is not None
      and summary.get("formal_run_completed") is False
      and summary.get("final_checkpoint_commit", {}).get("state") != "committed"
      and summary.get("environment_close", {}).get("completed") is True
    )
    checks["failure_error_is_the_injected_boundary"] = (
      injected_error is not None
      and {
        "runtime_failure": "functional runtime scan fault",
        "ppo_failure": "functional PPO Adam-step fault",
        "final_save_failure": "functional final full-checkpoint save fault",
        "record_log_failure": "functional final checkpoint record log fault",
      }[args.mode] in str(injected_error)
    )
    if args.mode == "runtime_failure":
      checks["runtime_fault_persisted_and_froze_before_close"] = (
        runtime._faulted and bool(fault_files)
        and close_events[0]["runtime_faulted"]
        and runner.trainer.optimizer_step_count == 0
      )
    elif args.mode == "ppo_failure":
      checks["optimizer_failure_faulted_trainer_and_runtime"] = (
        runner.trainer.faulted and runtime._faulted
        and runner.trainer.fault_stage is not None
        and bool(fault_files)
      )
    else:
      checks["finalization_failure_followed_real_24_control_20_adam_iteration"] = (
        len(completed) == 1
        and completed[0].get("collect", {}).get("completed_steps") == 24
        and runner.total_control_batches == 24
        and runner.trainer.optimizer_step_count == 20
      )
      checks["pending_checkpoint_remains_uncommitted"] = (
        runner._pending_final_checkpoint_path is not None
        and runner._final_checkpoint_path is None
      )
    checks["passed"] = all(checks.values())
    write_json(out / "precommit_failure_checks.json", {
      "mode": args.mode,
      "checks": checks,
      "exception": None if injected_error is None else {
        "type": type(injected_error).__name__, "message": str(injected_error),
      },
      "completed_iterations": completed,
      "close_path": close_events,
      "summary_path": str(summary_path),
      "fault_files": [str(path) for path in fault_files],
      "nonresumable_markers": [str(path) for path in nonresumable],
      "full_checkpoints": [str(path) for path in full_checkpoints],
      "optimizer_step_count": runner.trainer.optimizer_step_count,
      "runtime_fault": runtime.last_fault,
      "trainer_fault_stage": runner.trainer.fault_stage,
    })
    return checks
  finally:
    runner._close_after_learn_error = original_learn_close
    if "scan" in originals:
      runtime._scan_rows_now = originals["scan"]
    if "adam_step" in originals:
      runner.trainer.optimizer.step = originals["adam_step"]
    if "runner_save" in originals:
      runner.save = originals["runner_save"]
    if "append_record" in originals:
      runner._append_jsonl_record = originals["append_record"]
    if not runtime._closed:
      try:
        runner.close()
      except BaseException:
        pass
    if not runner._official_environment_closed:
      try:
        env.close()
      except BaseException:
        pass


def _run(out: Path, device: str, recipe: dict, *, args, grid) -> dict:
  from confined_locomotion.support.evidence import write_json

  if args.mode in PRECOMMIT_MODES:
    return _precommit_failure(out, args=args, grid=grid)

  env, runner = _build_runner(out, args=args, grid=grid)
  checks = {}
  try:
    # This is an actual guarded formal runner/update with the accepted finite
    # 24/20 budget. It is a lifecycle diagnostic, not an official launch.
    runner.learn(1)
    checkpoint = Path(runner._final_checkpoint_path)
    checks["real_full_iteration_committed_before_close_probe"] = (
      runner.current_learning_iteration == 1
      and runner.trainer.optimizer_step_count == 20
      and checkpoint.is_file()
    )
    original_close = runner._official_env_close
    if args.mode == "postcommit":
      def summary_io_failure():
        raise OSError("injected post-commit summary refresh failure")

      runner._write_summary = summary_io_failure
      runner._close_official_environment()
      checks["official_environment_close_still_completed"] = runner._official_environment_closed
      checks["post_commit_summary_failure_is_best_effort"] = any(
        "injected post-commit summary refresh failure" in message
        for message in runner._observer_errors
      )
      checks["committed_full_checkpoint_is_preserved"] = checkpoint.is_file()
      checks["summary_fault_did_not_reclassify_committed_run"] = (
        runner._final_checkpoint_path == checkpoint
        and runner._lifecycle_phase == "ready"
        and runner.runtime._closed
        and not runner.runtime._faulted
      )
      close_record = {
        "mode": "formal_training",
        "runner_phase": runner._lifecycle_phase,
        "runtime_closed": runner.runtime._closed,
        "runtime_faulted": runner.runtime._faulted,
        "checkpoint": str(checkpoint),
        "summary_error_recorded_in_memory": runner._observer_errors,
        "close_completed": runner._official_environment_closed,
      }
    else:
      calls = {"official_env_close": 0}

      def counted_original_close():
        calls["official_env_close"] += 1
        return original_close()

      runner._official_env_close = counted_original_close

      def runtime_close_failure():
        raise RuntimeError("injected runtime close failure")

      original_runner_close = runner.close
      runner.close = runtime_close_failure
      try:
        try:
          runner._close_official_environment()
        except RuntimeError as error:
          checks["close_error_preserved_after_native_close_attempt"] = (
            "injected runtime close failure" in str(error)
          )
        else:
          checks["close_error_preserved_after_native_close_attempt"] = False
      finally:
        runner.close = original_runner_close
      checks["native_environment_close_was_still_attempted"] = calls["official_env_close"] == 1
      checks["close_failure_freezes_runner_and_runtime"] = (
        runner._lifecycle_phase == "faulted"
        and runner.runtime._faulted
        and "injected runtime close failure" in str(runner._environment_close_error)
      )
      try:
        runner.learn(1)
      except RuntimeError:
        checks["learn_refused_after_close_failure"] = True
      else:
        checks["learn_refused_after_close_failure"] = False
      try:
        runner.save(out / "refused_after_close_failure.pt")
      except RuntimeError:
        checks["save_refused_after_close_failure"] = True
      else:
        checks["save_refused_after_close_failure"] = False
      try:
        runner.load(out / "missing_checkpoint.pt")
      except RuntimeError:
        checks["load_refused_before_checkpoint_read_after_close_failure"] = True
      else:
        checks["load_refused_before_checkpoint_read_after_close_failure"] = False
      try:
        runner.runtime.step(__import__("torch").zeros((args.num_envs, 18), device=device))
      except RuntimeError:
        checks["physics_step_refused_after_close_failure"] = True
      else:
        checks["physics_step_refused_after_close_failure"] = False
      checks["no_resume_artifact_created_after_close_failure"] = not (
        out / "refused_after_close_failure.pt"
      ).exists()
      close_record = {
        "mode": "formal_training",
        "runner_phase": runner._lifecycle_phase,
        "runtime_faulted": runner.runtime._faulted,
        "official_env_close_calls": calls["official_env_close"],
        "checkpoint_pre_close_failure": str(checkpoint),
        "environment_close_error": runner._environment_close_error,
      }
    checks["passed"] = all(checks.values())
    write_json(out / "lifecycle_checks.json", {"checks": checks, "evidence": close_record})
    return checks
  finally:
    try:
      runner.close = runner.__class__.close.__get__(runner, runner.__class__)
      if not runner.runtime._closed:
        runner.close()
    except BaseException:
      pass
    if not runner._official_environment_closed:
      try:
        env.close()
      except BaseException:
        pass


def main(argv=None) -> int:
  args = _args(argv)
  from _common import configure_imports

  configure_imports()
  from confined_locomotion.support.fixtures import m12_training_grid

  grid = m12_training_grid()
  return execute_gpu_check(
    args,
    feature="lifecycle",
    mode=args.mode,
    runner=lambda out, device, recipe: _run(out, device, recipe, args=args, grid=grid),
    optimizer=True,
    grid=grid,
    extra_recipe={"real_formal_mode": True, "iteration_budget": 1, "fault_injection": args.mode},
  )


if __name__ == "__main__":
  raise SystemExit(main())
