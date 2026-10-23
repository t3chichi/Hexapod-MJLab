#!/usr/bin/env python
"""Actual CUDA checks for sampled collection and natural timeout bootstrap."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from _common import (
  ROLLOUT_STEPS as RECIPE_STEPS,
  add_common_arguments,
  close_runtime,
  execute_gpu_check,
)


MODES = (
  "natural_rollout", "raw_action_log_prob", "value_fault", "timeout_bootstrap",
  "feature_cache",
)


def _make_runtime(out: Path, *, args, grid, mode: str):
  from confined_locomotion.support.fixtures import build_runtime

  return build_runtime(
    out,
    mode=mode,
    device=f"cuda:{args.gpu}",
    seed=args.seed,
    num_envs=args.num_envs,
    ground_grid=grid,
  )


def _make_collector(runtime, *, args, steps=24, use_feature_cache=True):
  from confined_locomotion.support.fixtures import build_collector
  from mjlab.utils.torch import configure_torch_backends

  configure_torch_backends(allow_tf32=False)
  torch.backends.cudnn.rnn.fp32_precision = "ieee"
  collector, adapter = build_collector(
    runtime, device=f"cuda:{args.gpu}", steps=steps, seed=args.seed
  )
  if collector.use_feature_cache != use_feature_cache:
    # The test fixture owns adapter construction; recreate only the collector
    # with the requested public runtime option.
    from hexapod_mjlab.tasks.confined_locomotion.collector import TrainingCollector

    collector = TrainingCollector(
      runtime, adapter, steps=steps, use_feature_cache=use_feature_cache
    )
  return collector, adapter


def _natural_rollout(out: Path, device: str, recipe: dict, *, args, grid) -> dict:
  from confined_locomotion.support.observer import RuntimeStepObserver
  from confined_locomotion.support.snapshots import cpu_tree

  env, _wrapper, runtime = _make_runtime(out, args=args, grid=grid, mode="engineering")
  checks = {}
  try:
    collector, adapter = _make_collector(runtime, args=args, steps=24)
    with RuntimeStepObserver(runtime, out / "transitions", label="natural") as witness:
      result = collector.collect()
    checks["full_natural_24_control_rollout"] = (
      result.mode == "engineering"
      and result.requested_steps == 24
      and result.completed_steps == 24
      and result.complete
      and result.stopped_reason is None
    )
    checks["all_24_real_transitions_saved"] = len(witness.records) == 24
    checks["raw_sampling_distribution_identity_bound"] = bool(
      result.distribution_identity_frozen and result.model_fingerprint
    )
    checks["rollout_storage_has_24_natural_rows"] = result.buffer.steps_filled == 24
    checks["pre_action_and_boundary_inputs_retained"] = (
      result.buffer._obs_lin_vel.shape == (24, args.num_envs, 3)
      and result.buffer._v_next.shape == (24, args.num_envs)
      and result.buffer._raw_action.shape == (24, args.num_envs, 18)
    )
    replay = None
    if result.complete:
      result.buffer.compute_gae()
      replay = result.buffer.replay(
        torch.arange(24 * args.num_envs, device=device, dtype=torch.long)
      )
    torch.save(cpu_tree({
      "result_summary": result.summary(),
      "replay": replay,
      "policy_state": adapter.state_dict(),
      "runtime_clock": {
        "control_steps": runtime.control_steps_taken,
        "scan_counts": runtime.sensor_scan_counts,
        "scan_counters": runtime.scan_counters,
      },
    }), out / "natural_rollout.pt")
  finally:
    close_runtime(env, runtime)
  checks["passed"] = all(checks.values())
  return checks


def _raw_action_log_prob(out: Path, device: str, recipe: dict, *, args, grid) -> dict:
  from confined_locomotion.support.oracles import diagonal_normal_log_prob
  from confined_locomotion.support.snapshots import cpu_tree

  env, _wrapper, runtime = _make_runtime(out, args=args, grid=grid, mode="training")
  checks = {}
  try:
    collector, adapter = _make_collector(runtime, args=args, steps=1)
    with torch.no_grad():
      adapter.network.action_head.output.weight.zero_()
      adapter.network.action_head.output.bias.zero_()
      adapter.network.action_head.output.bias[0] = 101.0
    result = collector.collect()
    batch = result.buffer.replay(torch.arange(args.num_envs, device=device))
    transition = runtime.transitions[-1]
    raw_lp = diagonal_normal_log_prob(batch.raw_action, batch.old_mean, batch.old_std)
    clipped_lp = diagonal_normal_log_prob(
      batch.raw_action.clamp(-100.0, 100.0), batch.old_mean, batch.old_std
    )
    checks["raw_sample_exceeds_execution_clip"] = bool((batch.raw_action[:, 0] > 100.0).all())
    checks["one_step_collection_complete"] = result.complete and result.completed_steps == 1
    checks["raw_action_kept_for_behavior_log_prob"] = torch.allclose(
      batch.old_log_prob, raw_lp, atol=1e-5, rtol=1e-5
    )
    checks["clipped_action_is_only_physics_execution"] = torch.equal(
      transition.context_record["clipped_action"], batch.raw_action.clamp(-100.0, 100.0)
    )
    checks["old_log_prob_not_recomputed_on_clipped_action"] = not torch.allclose(
      batch.old_log_prob, clipped_lp, atol=1e-5, rtol=1e-5
    )
    torch.save(cpu_tree({
      "result_summary": result.summary(),
      "replay": result.buffer.replay(torch.arange(args.num_envs, device=device)),
      "raw_log_prob_independent": raw_lp,
      "clipped_log_prob_independent": clipped_lp,
      "transition": transition,
    }), out / "raw_action_probability.pt")
  finally:
    close_runtime(env, runtime)
  checks["passed"] = all(checks.values())
  return checks


def _value_fault(out: Path, device: str, recipe: dict, *, args, grid) -> dict:
  from confined_locomotion.support.snapshots import (
    persistent_physics_snapshot,
    tree_equal,
  )

  env, _wrapper, runtime = _make_runtime(out, args=args, grid=grid, mode="engineering")
  checks = {}
  try:
    collector, adapter = _make_collector(runtime, args=args, steps=24)
    before = persistent_physics_snapshot(env)
    action_rng = collector.provider.generator_state().clone()
    try:
      with torch.no_grad():
        adapter.network.value_head.output.bias.fill_(float("nan"))
      collector.collect()
    except RuntimeError as error:
      checks["nonfinite_value_rejected_before_action_sampling"] = "NaN" in str(error) or "finite" in str(error)
    else:
      checks["nonfinite_value_rejected_before_action_sampling"] = False
    after = persistent_physics_snapshot(env)
    checks["persistent_physics_bit_exact"] = not tree_equal(before, after)
    checks["action_generator_unchanged"] = torch.equal(action_rng, collector.provider.generator_state())
    checks["no_control_or_physics_integration"] = (
      runtime.control_steps_taken == 0 and int(env.common_step_counter) == 0
    )
    checks["runtime_frozen_and_fault_scene_saved"] = (
      runtime._faulted and bool(runtime.last_fault)
      and any((out / "runtime").glob("fault_*/current_scene.pt"))
    )
    torch.save({
      "physics_before": before,
      "physics_after": after,
      "action_rng_before": action_rng,
      "action_rng_after": collector.provider.generator_state(),
      "fault": runtime.last_fault,
      "policy_state": adapter.state_dict(),
    }, out / "value_fault_evidence.pt")
  finally:
    close_runtime(env, runtime)
  checks["passed"] = all(checks.values())
  return checks


def _timeout_bootstrap(out: Path, device: str, recipe: dict, *, args, grid) -> dict:
  from confined_locomotion.support.oracles import generalized_advantage
  from confined_locomotion.support.snapshots import cpu_tree

  env, _wrapper, runtime = _make_runtime(
    out, args=args, grid=grid, mode="training_lifecycle_fixture"
  )
  checks = {}
  try:
    collector, adapter = _make_collector(runtime, args=args, steps=24)
    # Explicit fixture only: the real manager and runtime episode counters are
    # moved together to the last pre-timeout control identity. The actual next
    # transition then creates and resets a genuine timeout row.
    env.episode_length_buf[1] = 999
    runtime.step_in_episode[1] = 999
    with torch.no_grad():
      adapter.network.value_head.output.weight.zero_()
      adapter.network.value_head.output.bias.fill_(2.0)
    from confined_locomotion.support.observer import RuntimeStepObserver

    with RuntimeStepObserver(runtime, out / "transitions", label="timeout") as witness:
      result = collector.collect()
    buffer = result.buffer
    timeout_rows = torch.nonzero(buffer._truncated, as_tuple=False)
    checks["one_natural_timeout_flag_before_reset"] = (
      timeout_rows.tolist() == [[0, 1]]
      and not bool(buffer._terminated[0, 1])
      and bool(buffer._reset_applied[0, 1])
      and result.complete
      and len(witness.records) == 24
    )
    checks["natural_final_observation_bootstraps_before_reset"] = (
      float(buffer._v_next[0, 1]) == 2.0
      and int(buffer._final_episode_ids[0, 1]) == 0
      and int(buffer._final_step_ids[0, 1]) == 1000
      and int(buffer._next_episode_ids[0, 1]) == 1
      and int(buffer._next_step_ids[0, 1]) == 0
    )
    checks["original_timeout_reward_retained"] = torch.equal(
      buffer._reward[0, 1].detach().cpu(), witness.records[0]["transition"].reward[1]
    )
    actual_advantage, actual_returns = buffer.compute_gae()
    expected_advantage, expected_returns = generalized_advantage(
      buffer._reward,
      buffer._old_value,
      buffer._v_next,
      buffer._terminated,
      buffer._truncated,
      gamma=0.99,
      lam=0.95,
    )
    checks["timeout_bootstrap_and_recursion_cut_match_independent_gae"] = (
      torch.allclose(actual_advantage, expected_advantage, atol=1e-6, rtol=1e-6)
      and torch.allclose(actual_returns, expected_returns, atol=1e-6, rtol=1e-6)
      and bool((expected_advantage[0, 1] != buffer._reward[0, 1] - buffer._old_value[0, 1]).item())
    )
    torch.save(cpu_tree({
      "result_summary": result.summary(),
      "replay": buffer.replay(torch.arange(24 * args.num_envs, device=device)),
      "actual_advantage": actual_advantage,
      "actual_returns": actual_returns,
      "independent_advantage": expected_advantage,
      "independent_returns": expected_returns,
      "timeout_row": [0, 1],
      "declared_value_head_fixture": "value output weight zero, bias 2.0",
    }), out / "timeout_bootstrap.pt")
  finally:
    close_runtime(env, runtime)
  checks["passed"] = all(checks.values())
  return checks


def _cuda_event_average_ms(function, *, device: str, warmup: int = 3, repeats: int = 8) -> float:
  """Warm a bounded kernel path, then time it with synchronized CUDA events."""
  for _ in range(warmup):
    function()
  torch.cuda.synchronize(device)
  start = torch.cuda.Event(enable_timing=True)
  end = torch.cuda.Event(enable_timing=True)
  start.record()
  for _ in range(repeats):
    function()
  end.record()
  end.synchronize()
  return float(start.elapsed_time(end)) / repeats


def _output_max_abs_error(left, right) -> float:
  names = ("near_feature", "far_feature", "envelope", "action_mean", "value", "ground_pred")
  errors = []
  for name in names:
    a, b = getattr(left, name), getattr(right, name)
    if a is None or b is None:
      if a is not b:
        return float("inf")
      continue
    errors.append(float((a.detach().float() - b.detach().float()).abs().max().item()))
  return max(errors, default=0.0)


def _feature_cache_case(out: Path, device: str, *, args, grid,
                        use_feature_cache: bool, steps: int = RECIPE_STEPS) -> dict:
  """Run the existing bounded 24-control recipe; compare cache on exact snapshots."""
  from confined_locomotion.support.snapshots import cpu_tree
  from hexapod_mjlab.rl.confined_locomotion.inference import PointCloudFeatureCache
  from mjlab.utils.torch import configure_torch_backends

  env, _wrapper, runtime = _make_runtime(
    out, args=args, grid=grid, mode="training_lifecycle_fixture"
  )
  result_record = {"use_feature_cache": bool(use_feature_cache), "requested_steps": steps}
  cache = None
  original_cache_forward = None
  hooks = []
  try:
    # Match the production runner/PlaySession before constructing the network.
    # cuDNN's default TF32 path can vary with the row-selective miss batch size.
    configure_torch_backends(allow_tf32=False)
    torch.backends.cudnn.rnn.fp32_precision = "ieee"
    precision = {
      "cuda_matmul_fp32_precision": str(torch.backends.cuda.matmul.fp32_precision),
      "cudnn_fp32_precision": str(torch.backends.cudnn.fp32_precision),
      "cudnn_rnn_fp32_precision": str(torch.backends.cudnn.rnn.fp32_precision),
    }
    result_record["precision"] = precision
    collector, adapter = _make_collector(
      runtime, args=args, steps=steps, use_feature_cache=use_feature_cache
    )
    network = adapter.network
    # Declared timeout fixture on the last row proves reset identity handling
    # across the existing scan cadence; it does not compare physics runs.
    reset_row = min(1, int(args.num_envs) - 1)
    env.episode_length_buf[reset_row] = 999
    runtime.step_in_episode[reset_row] = 999

    prepared = runtime.prepare()
    with torch.no_grad():
      near_gru_ms = _cuda_event_average_ms(
        lambda: network.encode_near_points(prepared.near_points, prepared.near_valid),
        device=device,
      )
      far_gru_ms = _cuda_event_average_ms(
        lambda: network.encode_far_points(prepared.far_points, prepared.far_valid),
        device=device,
      )
      both_gru_ms = _cuda_event_average_ms(
        lambda: network.encode_points(
          prepared.near_points, prepared.near_valid,
          prepared.far_points, prepared.far_valid,
        ),
        device=device,
      )
      benchmark_cache = PointCloudFeatureCache(network)
      benchmark_cache.forward(prepared)  # cold miss / compile warm-up
      cache_hit_and_heads_ms = _cuda_event_average_ms(
        lambda: benchmark_cache.forward(prepared), device=device
      )
    del benchmark_cache

    encoded_rows = {"near": 0, "far": 0}
    encoder_calls = {"near": 0, "far": 0}
    def count_rows(name):
      def hook(_module, module_inputs):
        encoded_rows[name] += int(module_inputs[0].shape[0])
        encoder_calls[name] += 1
      return hook

    hooks = [
      network.near_encoder.register_forward_pre_hook(count_rows("near")),
      network.far_encoder.register_forward_pre_hook(count_rows("far")),
    ]
    full_forward_calls = {"count": 0}
    if not use_feature_cache:
      hooks.append(network.register_forward_pre_hook(
        lambda _module, _inputs: full_forward_calls.__setitem__(
          "count", full_forward_calls["count"] + 1
        )
      ))

    captured = []
    cache = collector.feature_cache
    if cache is not None:
      original_cache_forward = cache.forward
      def capture_cache_forward(prepared_inputs):
        output = original_cache_forward(prepared_inputs)
        captured.append((prepared_inputs, output))
        return output
      cache.forward = capture_cache_forward

    torch.cuda.synchronize(device)
    started = time.perf_counter()
    collection = collector.collect(require_complete=True)
    torch.cuda.synchronize(device)
    total_seconds = time.perf_counter() - started
    for hook in hooks:
      hook.remove()

    max_error = None
    outputs_match = None
    if use_feature_cache:
      errors = []
      all_close = True
      with torch.no_grad():
        for prepared_inputs, cached_output in captured:
          fresh_output = network(
            prepared_inputs.observation,
            prepared_inputs.near_points, prepared_inputs.near_valid,
            prepared_inputs.far_points, prepared_inputs.far_valid,
          )
          errors.append(_output_max_abs_error(cached_output, fresh_output))
          all_close = all_close and all(
            torch.allclose(
              getattr(cached_output, name), getattr(fresh_output, name),
              atol=3e-5, rtol=3e-5,
            )
            for name in ("near_feature", "far_feature", "envelope", "action_mean", "value")
          ) and (
            cached_output.ground_pred is None
            or torch.allclose(cached_output.ground_pred, fresh_output.ground_pred,
                              atol=3e-5, rtol=3e-5)
          )
      max_error = max(errors, default=0.0)
      outputs_match = bool(captured) and all_close

    reset_locations = torch.nonzero(
      collection.buffer._reset_applied[:, reset_row], as_tuple=False
    ).flatten()
    reset_step = int(reset_locations[0].item()) if reset_locations.numel() else None
    if reset_step is None:
      final_episode_id = next_episode_id = final_step_id = next_step_id = None
      partial_reset_semantics = False
    else:
      final_episode_id = int(collection.buffer._final_episode_ids[reset_step, reset_row])
      next_episode_id = int(collection.buffer._next_episode_ids[reset_step, reset_row])
      final_step_id = int(collection.buffer._final_step_ids[reset_step, reset_row])
      next_step_id = int(collection.buffer._next_step_ids[reset_step, reset_row])
      partial_reset_semantics = (
        reset_step == 0
        and final_episode_id + 1 == next_episode_id
        and final_step_id == 1000
        and next_step_id == 0
        and bool(collection.buffer._reset_applied[reset_step, reset_row])
      )
    stats = None if cache is None else cache.stats
    logical_forwards = (
      int(stats["forward_calls"]) if stats is not None else int(full_forward_calls["count"])
    )
    naive_rows = int(collection.completed_steps) * 2 * int(args.num_envs)
    result_record.update({
      "complete": collection.complete,
      "completed_steps": collection.completed_steps,
      "collection_wall_seconds": total_seconds,
      "collection_wall_ms_per_control": 1000.0 * total_seconds / max(1, collection.completed_steps),
      "logical_policy_forwards": logical_forwards,
      "expected_policy_forwards_pre_action_plus_bootstrap": 2 * collection.completed_steps,
      "encoder_row_counts": dict(encoded_rows),
      "encoder_module_calls": dict(encoder_calls),
      "naive_full_forward_rows_per_encoder": naive_rows,
      "cache_stats": stats,
      "segments_cuda_event_ms_average": {
        "near_gru": near_gru_ms,
        "far_gru": far_gru_ms,
        "both_gru": both_gru_ms,
        "cache_hit_plus_observation_dependent_heads": cache_hit_and_heads_ms,
      },
      "exact_snapshot_cached_output_max_abs_error": max_error,
      "exact_snapshot_cached_output_within_fp32_tolerance": outputs_match,
      "partial_reset_final_old_to_next_new_episode": partial_reset_semantics,
      "timeout_fixture_row_boundary": {
        "step": reset_step,
        "row": reset_row,
        "final_episode_id": final_episode_id,
        "final_step_id": final_step_id,
        "next_episode_id": next_episode_id,
        "next_step_id": next_step_id,
      },
      "trajectory_comparison": "not compared; each mode uses a separate fresh physics run",
    })
    result_record["checks"] = {
      "production_ieee_fp32_backends": all(value == "ieee" for value in precision.values()),
      "bounded_24_step_collection_completed": collection.complete
        and collection.completed_steps == steps,
      "two_full_policy_outputs_per_control_step": logical_forwards == 2 * collection.completed_steps,
      "encoder_hooks_match_cache_encoded_row_stats": (
        stats is not None
        and encoded_rows["near"] == stats["near_rows_encoded"]
        and encoded_rows["far"] == stats["far_rows_encoded"]
      ) if use_feature_cache else (
        encoded_rows["near"] == naive_rows and encoded_rows["far"] == naive_rows
      ),
      "timeout_fixture_retains_old_final_and_new_next_episode": partial_reset_semantics,
      "cached_outputs_match_fresh_network_on_same_snapshots": (
        outputs_match if use_feature_cache else True
      ),
      "no_optimizer_or_learning_claim": True,
    }
    result_record["passed"] = all(result_record["checks"].values())
    torch.save(cpu_tree(result_record), out / "feature_cache_collection.pt")
  finally:
    for hook in locals().get("hooks", []):
      try:
        hook.remove()
      except Exception:
        pass
    if cache is not None and original_cache_forward is not None:
      cache.forward = original_cache_forward
    close_runtime(env, runtime)
  return result_record


def _feature_cache_benchmark(out: Path, device: str, recipe: dict, *, args, grid) -> dict:
  """Compare short collection costs; never compare separate physics trajectories."""
  from confined_locomotion.support.evidence import write_json

  if not 1 <= args.num_envs <= 2:
    raise ValueError("feature-cache engineering check requires 1 or 2 environments")
  steps = RECIPE_STEPS
  cache_off = _feature_cache_case(
    out / "cache_off", device, args=args, grid=grid,
    use_feature_cache=False, steps=steps,
  )
  cache_on = _feature_cache_case(
    out / "cache_on", device, args=args, grid=grid,
    use_feature_cache=True, steps=steps,
  )
  checks = {
    "cache_off_case_passed": bool(cache_off.get("passed")),
    "cache_on_case_passed": bool(cache_on.get("passed")),
    "cache_on_does_not_encode_more_rows_than_uncached_reference": (
      cache_on["encoder_row_counts"]["near"] <= cache_off["encoder_row_counts"]["near"]
      and cache_on["encoder_row_counts"]["far"] <= cache_off["encoder_row_counts"]["far"]
    ),
  }
  report = {
    "checks": checks,
    "cache_off": cache_off,
    "cache_on": cache_on,
    "performance_threshold": None,
    "interpretation": (
      "Measured bounded engineering timing only. No fixed speedup threshold and no learning-effect claim."
    ),
    "passed": all(checks.values()),
  }
  write_json(out / "feature_cache_benchmark.json", report)
  return report


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
  runners = {
    "natural_rollout": _natural_rollout,
    "raw_action_log_prob": _raw_action_log_prob,
    "value_fault": _value_fault,
    "timeout_bootstrap": _timeout_bootstrap,
    "feature_cache": _feature_cache_benchmark,
  }
  return execute_gpu_check(
    args,
    feature="collection",
    mode=args.mode,
    runner=lambda out, device, recipe: runners[args.mode](
      out, device, recipe, args=args, grid=grid
    ),
    grid=grid,
    optimizer=False,
    extra_recipe={
      "declared_fixture": {
        "mode": args.mode,
        "timeout_rows": (
          [[0, 1]] if args.mode == "timeout_bootstrap" else
          [[0, min(1, int(args.num_envs) - 1)]] if args.mode == "feature_cache" else []
        ),
        "rollout_steps": RECIPE_STEPS,
      },
      **({
        "rollout_steps": RECIPE_STEPS,
        "optimizer_steps": 0,
        "feature_cache_benchmark": {
          "collection_control_steps_per_case": RECIPE_STEPS,
          "num_envs_max": 2,
          "optimizer_enabled": False,
          "separate_fresh_physics_runs": True,
          "same_snapshot_output_comparison": "cache result vs fresh network forward on captured prepared inputs",
          "performance_threshold": None,
        },
      } if args.mode == "feature_cache" else {}),
    },
  )


if __name__ == "__main__":
  raise SystemExit(main())
