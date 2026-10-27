#!/usr/bin/env python
"""Actual CUDA P3 update checks with full 20-minibatch raw evidence."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from _common import add_common_arguments, close_runtime, execute_gpu_check


MODES = ("joint_update", "mirror_update")
FORMULA_ATOL = 1e-5
FORMULA_RTOL = 1e-5


def _get(value, name):
  return value[name] if isinstance(value, dict) else getattr(value, name)


def _float(value) -> float:
  return float(value.detach().cpu().item()) if torch.is_tensor(value) else float(value)


def _compare_scalar(expected, actual, name: str, report: list[dict]) -> bool:
  expected_value, actual_value = _float(expected), _float(actual)
  difference = abs(expected_value - actual_value)
  allowed = FORMULA_ATOL + FORMULA_RTOL * abs(expected_value)
  report.append({
    "name": name,
    "independent": expected_value,
    "observed": actual_value,
    "absolute_difference": difference,
    "atol": FORMULA_ATOL,
    "rtol": FORMULA_RTOL,
    "within_gate": difference <= allowed,
  })
  return difference <= allowed


def _ground_loss(prediction, selected_frames, supervision_size, ground_bucket_loss) -> torch.Tensor:
  if prediction is None or not selected_frames or supervision_size == 0:
    if prediction is not None:
      return prediction.new_zeros(())
    return torch.zeros((), dtype=torch.float32)
  target = torch.stack([frame.heights for frame in selected_frames]).to(prediction.device)
  valid = torch.stack([frame.valid for frame in selected_frames]).to(prediction.device)
  return ground_bucket_loss(prediction, target, valid, supervision_size, bucket_count=4)


def _recompute_minibatch(record: dict, frames: tuple, *, mirror_indices=None) -> dict:
  from confined_locomotion.support.oracles import (
    clipped_value_loss,
    diagonal_normal_entropy,
    diagonal_normal_kl,
    diagonal_normal_log_prob,
    envelope_sum_loss,
    ground_bucket_loss,
    joint_loss,
    mirror_mean_mse,
    ppo_policy_loss,
  )

  batch = record["batch"]
  output = record["network_output"]
  distribution = record["distribution_values"]
  mean, std = distribution["mean"], distribution["std"]
  raw_action = distribution["raw_action"]
  normalized_advantage = record["normalized_advantage"]
  new_log_prob = diagonal_normal_log_prob(raw_action, mean, std)
  old_log_prob = _get(batch, "old_log_prob")
  policy_original = ppo_policy_loss(new_log_prob, old_log_prob, normalized_advantage)
  value_original = clipped_value_loss(
    _get(output, "value").squeeze(-1),
    _get(batch, "old_value"),
    _get(batch, "returns"),
  )
  envelope_target = _get(batch, "envelope_values")
  envelope_valid = _get(batch, "envelope_valid")
  envelope_original = envelope_sum_loss(
    _get(output, "envelope"), envelope_target, envelope_valid
  )
  entropy_original = diagonal_normal_entropy(std)
  bucket = list(record.get("ground_bucket") or [])
  supervision_size = int(record.get("ground_supervision_size") or 0)
  ground_original = _ground_loss(
    record.get("ground_predictions"),
    tuple(frames[index] for index in bucket),
    supervision_size,
    ground_bucket_loss,
  )
  kl_original = diagonal_normal_kl(
    _get(batch, "old_mean"), _get(batch, "old_std"), mean, std
  )
  result = {
    "new_log_prob_original": new_log_prob,
    "policy_original": policy_original,
    "value_original": value_original,
    "envelope_original": envelope_original,
    "ground_original": ground_original,
    "entropy_original": entropy_original,
    "kl_original": kl_original,
  }

  mirrored_distribution = record.get("mirrored_distribution_values")
  if mirrored_distribution is not None:
    mirrored_output = record["mirrored_output"]
    old_mirrored = record["old_mirrored"]
    mirrored_mean = mirrored_distribution["mean"]
    mirrored_std = mirrored_distribution["std"]
    mirrored_raw = mirrored_distribution["raw_action"]
    old_mirror_mean = _get(old_mirrored, "mean")
    old_mirror_std = _get(old_mirrored, "std")
    old_mirror_raw = _get(old_mirrored, "raw_action")
    old_mirror_log_prob = diagonal_normal_log_prob(
      old_mirror_raw, old_mirror_mean, old_mirror_std
    )
    new_mirror_log_prob = diagonal_normal_log_prob(
      mirrored_raw, mirrored_mean, mirrored_std
    )
    policy_mirrored = ppo_policy_loss(
      new_mirror_log_prob, old_mirror_log_prob, normalized_advantage
    )
    value_mirrored = clipped_value_loss(
      _get(mirrored_output, "value").squeeze(-1),
      _get(batch, "old_value"),
      _get(batch, "returns"),
    )
    # The confirmed six envelope labels are invariant under left/right y reflection.
    envelope_mirrored = envelope_sum_loss(
      _get(mirrored_output, "envelope"), envelope_target, envelope_valid
    )
    mirrored_frames = tuple(record.get("mirrored_ground_frames") or ())
    ground_mirrored = _ground_loss(
      record.get("mirrored_ground_predictions"),
      mirrored_frames,
      supervision_size,
      ground_bucket_loss,
    )
    entropy_mirrored = diagonal_normal_entropy(mirrored_std)
    kl_mirrored = diagonal_normal_kl(
      old_mirror_mean, old_mirror_std, mirrored_mean, mirrored_std
    )
    mirror_mean = mirror_mean_mse(
      _get(mirrored_output, "action_mean"),
      _get(output, "action_mean"),
      mirror_indices,
    )
    result.update({
      "old_mirror_log_prob_independent": old_mirror_log_prob,
      "policy_mirrored": policy_mirrored,
      "value_mirrored": value_mirrored,
      "envelope_mirrored": envelope_mirrored,
      "ground_mirrored": ground_mirrored,
      "entropy_mirrored": entropy_mirrored,
      "kl_mirrored": kl_mirrored,
      "mirror_mean": mirror_mean,
      "policy": 0.5 * (policy_original + policy_mirrored),
      "value": 0.5 * (value_original + value_mirrored),
      "envelope": 0.5 * (envelope_original + envelope_mirrored),
      "ground": 0.5 * (ground_original + ground_mirrored),
      "entropy": 0.5 * (entropy_original + entropy_mirrored),
      "kl": torch.cat((kl_original, kl_mirrored)),
      "total": joint_loss(
        0.5 * (policy_original + policy_mirrored),
        0.5 * (value_original + value_mirrored),
        0.5 * (entropy_original + entropy_mirrored),
        0.5 * (envelope_original + envelope_mirrored),
        0.5 * (ground_original + ground_mirrored),
      ) + 0.6 * mirror_mean,
    })
  else:
    result.update({
      "policy": policy_original,
      "value": value_original,
      "envelope": envelope_original,
      "ground": ground_original,
      "entropy": entropy_original,
      "kl": kl_original,
      "total": joint_loss(
        policy_original, value_original, entropy_original,
        envelope_original, ground_original,
      ),
    })
  return result


def _trainer_for(runtime, adapter, *, grid, device: str, seed: int, mirror: bool):
  from hexapod_mjlab.rl.confined_locomotion.ppo import (
    ConfinedPPOTrainer,
    MirrorTrainingConfig,
  )

  kwargs = {}
  symmetry = MirrorTrainingConfig(enabled=mirror, mean_loss_coeff=0.6)
  if mirror:
    from hexapod_mjlab.rl.confined_locomotion.symmetry import GridPermutation, JointPermutation
    from hexapod_mjlab.common.lidar import resolve_lidar_mount

    action = runtime.env.action_manager.get_term("residual_pd")
    joint = JointPermutation.from_joint_names(tuple(action._target_names))
    grid_permutation = GridPermutation.from_axes(grid.x_coords, grid.y_coords)
    mount = resolve_lidar_mount()
    if mount is None:
      raise RuntimeError("actual compiled optical lidar mount was not resolved")
    mount_pos = torch.tensor(mount[0], dtype=torch.float32, device=device)
    mount_quat = torch.tensor(mount[1], dtype=torch.float32, device=device)
    pipeline_pos = runtime.pipeline._mount_t[0]
    pipeline_quat = runtime.pipeline._mount_q[0]
    mount_matches = (
      torch.allclose(pipeline_pos, mount_pos, atol=1e-7, rtol=0.0)
      and torch.allclose(pipeline_quat, mount_quat, atol=1e-7, rtol=0.0)
    )
    if not mount_matches:
      raise AssertionError("resolved optical mount differs from real runtime scan mount")
    kwargs = {
      "joint_permutation": joint,
      "grid_permutation": grid_permutation,
      "mount_pos_b": pipeline_pos.detach().clone(),
      "mount_quat_xyzw": pipeline_quat.detach().clone(),
      "mount_optical_witness": {
        "source": "compiled lidar_sensor site via resolve_lidar_mount",
        "resolved_matches_runtime": True,
        "absolute_tolerance": 1e-7,
      },
    }
  trainer = ConfinedPPOTrainer(
    policy_adapter=adapter,
    grid_identity_hash=grid.identity_hash(),
    seed=seed,
    device=device,
    symmetry_config=symmetry,
    allow_cpu=False,
    **kwargs,
  )
  return trainer


def _run_update(out: Path, device: str, recipe: dict, *, args, grid, mirror: bool) -> dict:
  from confined_locomotion.support.observer import TrainerObserver
  from confined_locomotion.support.snapshots import cpu_tree
  from confined_locomotion.support.fixtures import build_collector, build_runtime
  from hexapod_mjlab.rl.confined_locomotion.symmetry import JointPermutation
  from mjlab.utils.torch import configure_torch_backends

  env, _wrapper, runtime = build_runtime(
    out,
    mode="engineering",
    device=device,
    seed=args.seed,
    num_envs=args.num_envs,
    ground_grid=grid,
  )
  checks = {}
  try:
    configure_torch_backends(allow_tf32=False)
    torch.backends.cudnn.rnn.fp32_precision = "ieee"
    checks["production_ieee_fp32_backends"] = (
      torch.backends.cuda.matmul.fp32_precision == "ieee"
      and torch.backends.cudnn.fp32_precision == "ieee"
      and torch.backends.cudnn.rnn.fp32_precision == "ieee"
    )
    collector, adapter = build_collector(
      runtime, device=device, steps=24, seed=args.seed
    )
    from confined_locomotion.support.observer import RuntimeStepObserver

    with RuntimeStepObserver(runtime, out / "collection_steps", label="rollout") as collect_observer:
      result = collector.collect()
    checks["natural_rollout_complete_without_stop"] = (
      result.complete and result.completed_steps == 24
      and result.mode == "engineering" and result.stopped_reason is None
    )
    checks["all_24_pre_action_and_transition_inputs_saved"] = len(collect_observer.records) == 24
    if not result.complete:
      torch.save(cpu_tree({"result_summary": result.summary()}), out / "not_passed_rollout.pt")
      checks["passed"] = False
      return checks

    buffer = result.buffer
    buffer.compute_gae()
    full_indices = torch.arange(24 * args.num_envs, device=device, dtype=torch.long)
    full_replay = buffer.replay(full_indices)
    all_ground_frames = buffer.ground_frames.frames()
    ground_frames = tuple(frame for frame in all_ground_frames if bool(frame.valid.any()))
    trainer = _trainer_for(
      runtime, adapter, grid=grid, device=device, seed=args.seed, mirror=mirror
    )
    expected_ids = {id(parameter) for _name, parameter in adapter.named_parameters()}
    unique_ids = {id(parameter) for parameter in trainer.unique_parameters}
    before_params = {name: parameter.detach().cpu().clone() for name, parameter in adapter.named_parameters()}
    behavior_rng_before = collector.provider.generator_state().clone()
    torch.save(cpu_tree({
      "result_summary": result.summary(),
      "full_replay": full_replay,
      "ground_source_frames": ground_frames,
      "policy_state_before_update": before_params,
      "behavior_rng_before": behavior_rng_before,
      "symmetry_identity": trainer.symmetry_identity,
    }), out / "update_full_replay.pt")
    observer = TrainerObserver(trainer, out / "minibatches", full_gradients=True)
    sensor_counts_before_update = runtime.sensor_scan_counts.detach().clone()
    with observer:
      metrics = trainer.update(buffer)
    observer_summary = observer.summary()
    checks["actual_20_minibatch_updates"] = (
      len(metrics["minibatches"]) == 20
      and trainer.optimizer_step_count == 20
      and trainer.update_count == 1
      and observer_summary["forward_count"] == 20
      and observer_summary["grad_pre_count"] == 20
      and observer_summary["grad_post_count"] == 20
      and observer_summary["adam_step_done_count"] == 20
    )
    checks["one_backward_joint_clip_adam_sequence_per_minibatch"] = (
      observer_summary["call_counts"] == {"backward": 20, "joint_clip": 20, "adam_step": 20}
      and observer_summary["actual_call_order"] == ["backward", "joint_clip", "adam_step"] * 20
    )
    checks["one_joint_adam_over_exact_unique_named_parameter_set"] = (
      len(trainer.optimizer.param_groups) == 1
      and trainer.optimizer.param_groups[0]["params"]
      and {id(parameter) for parameter in trainer.optimizer.param_groups[0]["params"]} == expected_ids == unique_ids
      and len(unique_ids) == len(expected_ids)
      and observer_summary["all_clip_calls_used_exact_parameter_set"]
      and observer_summary["all_steps_used_expected_adam"]
    )
    checks["five_epochs_four_complete_minibatches_each"] = (
      {(int(item["epoch"]), int(item["minibatch"])) for item in metrics["minibatches"]}
      == {(epoch, minibatch) for epoch in range(5) for minibatch in range(4)}
      and all(len([entry for entry in metrics["minibatches"] if entry["epoch"] == epoch]) == 4
              for epoch in range(5))
    )
    checks["all_48_samples_seen_once_per_epoch"] = all(
      sorted(index for item in metrics["minibatches"] if item["epoch"] == epoch
             for index in item["indices"]) == list(range(24 * args.num_envs))
      for epoch in range(5)
    )
    checks["actual_gradients_finite_and_clipped"] = (
      all(math.isfinite(float(item["grad_norm_pre_clip"])) for item in observer.events
          if item["stage"] == "grad_pre_clip")
      and all(math.isfinite(float(item["grad_norm_post_clip"]))
              and float(item["grad_norm_post_clip"]) <= 1.0 + 1e-4
              for item in observer.events if item["stage"] == "grad_post_clip")
      and len([item for item in observer.events if item["stage"] == "grad_pre_clip"]) == 20
      and len([item for item in observer.events if item["stage"] == "grad_post_clip"]) == 20
    )
    checks["weight_update_and_behavior_prng_separation"] = (
      metrics["weights_changed"] is True
      and metrics["behavior_generator_unchanged"] is True
      and torch.equal(behavior_rng_before, collector.provider.generator_state())
    )
    checks["successful_update_consumed_and_cleared_buffer"] = (
      buffer.steps_filled == 0 and not buffer.is_full
    )
    checks["update_reuses_scans_without_new_sensor_work"] = torch.equal(
      sensor_counts_before_update, runtime.sensor_scan_counts
    )

    formula_comparisons = []
    forward_records = [item for item in observer.events if item["stage"] == "minibatch_forward"]
    recalc_ok = len(forward_records) == 20
    for item in forward_records:
      calculated = _recompute_minibatch(
        item,
        tuple(ground_frames),
        mirror_indices=(trainer.joint_permutation.indices if mirror else None),
      )
      expected_fields = {
        "loss_policy": calculated["policy"],
        "loss_value": calculated["value"],
        "loss_envelope": calculated["envelope"],
        "loss_ground": calculated["ground"],
        "loss_total": calculated["total"],
      }
      if mirror:
        expected_fields.update({
          "loss_mirror_mean": calculated["mirror_mean"],
          "loss_policy_original": calculated["policy_original"],
          "loss_policy_mirrored": calculated["policy_mirrored"],
          "loss_value_original": calculated["value_original"],
          "loss_value_mirrored": calculated["value_mirrored"],
          "loss_envelope_original": calculated["envelope_original"],
          "loss_envelope_mirrored": calculated["envelope_mirrored"],
          "loss_ground_original": calculated["ground_original"],
          "loss_ground_mirrored": calculated["ground_mirrored"],
        })
        old_lp = _get(item, "old_mirror_log_prob")
        independently_pushed_lp = calculated["old_mirror_log_prob_independent"]
        recalc_ok &= _compare_scalar(
          (old_lp - independently_pushed_lp).abs().max(), 0.0,
          f"epoch{item['epoch']}_mb{item['minibatch']}_old_mirror_log_prob", formula_comparisons,
        )
      for metric_name, expected in expected_fields.items():
        observed = item.get(metric_name)
        recalc_ok &= observed is not None and _compare_scalar(
          expected, observed,
          f"epoch{item['epoch']}_mb{item['minibatch']}_{metric_name}",
          formula_comparisons,
        )
      if "kl" in calculated:
        observed_kl = item.get("kl_event")
        recalc_ok &= observed_kl is not None and torch.allclose(
          calculated["kl"], observed_kl, atol=FORMULA_ATOL, rtol=FORMULA_RTOL
        )
        if observed_kl is not None:
          formula_comparisons.append({
            "name": f"epoch{item['epoch']}_mb{item['minibatch']}_kl_event",
            "max_abs_difference": float((calculated["kl"] - observed_kl).abs().max()),
            "atol": FORMULA_ATOL,
            "rtol": FORMULA_RTOL,
            "within_gate": bool(torch.allclose(
              calculated["kl"], observed_kl, atol=FORMULA_ATOL, rtol=FORMULA_RTOL
            )),
          })
    checks["all_20_minibatches_independently_recomputed_from_raw_tensors"] = recalc_ok
    source_count = len(ground_frames)
    checks["independent_source_frames_counted_once_in_four_buckets_each_epoch"] = (
      len({tuple(frame.key) for frame in ground_frames}) == source_count
      and all(int(item["ground_supervision_size"]) == source_count for item in forward_records)
      and all(
        sorted(index for item in forward_records if item["epoch"] == epoch
               for index in item["ground_bucket"]) == list(range(source_count))
        for epoch in range(5)
      )
    )
    checks["reported_update_contract_and_learning_rate"] = (
      metrics["optimizer_steps_this_update"] == 20
      and metrics["learning_rate_final"] >= 1e-5
      and metrics["learning_rate_final"] <= 1e-2
    )
    if mirror:
      permutation = torch.tensor(trainer.joint_permutation.indices, dtype=torch.long)
      checks["current_mirrored_std_stays_native_and_old_distribution_is_pushed_together"] = all(
        torch.equal(item["mirrored_distribution_values"]["std"], item["distribution_values"]["std"])
        and all(torch.equal(
          _get(item["old_mirrored"], field),
          _get(item["batch"], old_field).index_select(-1, permutation),
        ) for field, old_field in (("mean", "old_mean"), ("std", "old_std"), ("raw_action", "raw_action")))
        and len(item["mirrored_ground_frames"]) == len(item["ground_bucket"])
        for item in forward_records
      )
      checks["mirror_coefficient_and_old_probability_invariance"] = (
        all(item.get("mirror_enabled") is True for item in metrics["minibatches"])
        and all(item.get("mirror_mean_loss_coeff") == 0.6 for item in metrics["minibatches"])
        and all(item.get("old_mirror_logprob_max_abs_diff", float("inf")) <= 1e-5
                for item in metrics["minibatches"])
      )
    torch.save(cpu_tree({
      "update_metrics": metrics,
      "observer_summary": observer_summary,
      "observer_events_all_20": observer.events,
      "optimizer_calls": observer.calls,
      "independent_formula_comparisons": formula_comparisons,
      "policy_state_after_update": {
        name: parameter.detach().cpu().clone()
        for name, parameter in adapter.named_parameters()
      },
    }), out / "update_evidence.pt")
  finally:
    close_runtime(env, runtime)
  checks["passed"] = all(checks.values())
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
  mirror = args.mode == "mirror_update"
  return execute_gpu_check(
    args,
    feature="update",
    mode=args.mode,
    runner=lambda out, device, recipe: _run_update(
      out, device, recipe, args=args, grid=grid, mirror=mirror
    ),
    optimizer=True,
    grid=grid,
    extra_recipe={
      "optimizer": "one real Adam; exactly 5 epochs x 4 minibatches",
      "mirror_enabled": mirror,
      "formula_comparison_gate": {"atol": FORMULA_ATOL, "rtol": FORMULA_RTOL},
    },
  )


if __name__ == "__main__":
  raise SystemExit(main())
