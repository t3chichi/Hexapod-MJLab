"""Finite real-CUDA integration gate; never a formal learning-effect test."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
import traceback

import torch
import numpy as np


def differences(a, b, prefix=""):
    if torch.is_tensor(a):
        return [] if torch.is_tensor(b) and a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.cpu(), b.cpu()) else [prefix]
    if isinstance(a, np.ndarray):
        return [] if isinstance(b, np.ndarray) and np.array_equal(a, b) else [prefix]
    if hasattr(a, "__dataclass_fields__"):
        return differences(vars(a), vars(b), prefix)
    if isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            return [prefix + ".keys"]
        return [p for key in a for p in differences(a[key], b[key], prefix + "." + str(key))]
    if isinstance(a, (tuple, list)):
        if not isinstance(b, type(a)) or len(a) != len(b):
            return [prefix]
        return [p for i,(x,y) in enumerate(zip(a,b)) for p in differences(x,y,prefix+f"[{i}]")]
    return [] if a == b else [prefix]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--num-envs", type=int, default=2, choices=(1, 2))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--raycast-mode", choices=("full", "sparse"), default="full")
    parser.add_argument("--sparse-selection")
    parser.add_argument("--far-sampling", choices=("all_endpoints", "hit_only_10m"), default="all_endpoints")
    parser.add_argument("--save-interval", type=int, default=50)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    checks = {}
    started = time.monotonic()
    runner = None
    env = None
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("real CUDA required; CPU cannot pass this gate")
        device = f"cuda:{args.gpu}"
        torch.cuda.set_device(args.gpu)
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
        from hexapod_mjlab.tasks.confined_locomotion_2 import make_training_env_cfg
        from hexapod_mjlab.tasks.confined_locomotion_2.rl_cfg import confined_locomotion_formal_rl_cfg
        from hexapod_mjlab.tasks.confined_locomotion_2.rl.runner import ConfinedLocomotionRunner

        cfg = confined_locomotion_formal_rl_cfg(
            seed=args.seed, num_envs=args.num_envs, max_iterations=1)
        cfg.engineering_enabled = True
        cfg.save_interval = args.save_interval
        cfg.lidar_config = {"raycast_mode": args.raycast_mode, "sparse_selection_path": args.sparse_selection,
                            "far_sampling": args.far_sampling}
        (out / "recipe.json").write_text(json.dumps(asdict(cfg), default=str, indent=2))
        env = ManagerBasedRlEnv(make_training_env_cfg(num_envs=args.num_envs, seed=args.seed), device=device)
        wrapper = RslRlVecEnvWrapper(env, clip_actions=None)
        runner = ConfinedLocomotionRunner(wrapper, asdict(cfg), out / "training", device)
        prepared = runner.runtime.prepare()
        checks["scan_shapes"] = (prepared.near_points.shape[1:] == (256, 3)
                                 and prepared.far_points.shape[1:] == (128, 3))
        checks["no_ground_dependencies"] = not hasattr(runner.runtime, "_ground_provider") and not hasattr(runner.network, "ground_head")
        checks["mirror_enabled"] = runner.symmetry_config.enabled
        runner.learn(1)
        checks["one_24_step_20_optimizer_iteration"] = (
            runner.total_control_batches == 24 and runner.trainer.optimizer_step_count == 20
            and runner.current_learning_iteration == 1)
        checks["checkpoint_written"] = Path(runner.final_checkpoint).is_file()
        saved_hidden = runner.collector.hidden.clone()
        saved_mirror = runner.collector.mirror_hidden.clone()
        # Continue two natural rollouts from exactly the same saved boundary.
        reference = runner.collector.collect()
        replay_buffer = copy.deepcopy(reference)
        reference_hidden = runner.collector.hidden.clone()
        reference_mirror = runner.collector.mirror_hidden.clone()
        runner.load(runner.final_checkpoint)
        checks["both_histories_restored"] = torch.equal(saved_hidden, runner.collector.hidden) and torch.equal(saved_mirror, runner.collector.mirror_hidden)
        boundary_copy = runner.save(out / "restored_boundary.pt")
        initial_payload = torch.load(runner.final_checkpoint, map_location="cpu", weights_only=False)
        restored_payload = torch.load(boundary_copy, map_location="cpu", weights_only=False)
        boundary_diff = differences(initial_payload, restored_payload)
        checks["restore_boundary_bit_exact"] = not boundary_diff
        checks["restore_boundary_differences"] = boundary_diff
        checks["episode_metrics_restored"] = not differences(
            initial_payload["learning_metrics"], runner.learning_metrics.state_dict())
        restored = runner.collector.collect()
        checks["resume_actions_bit_exact"] = all(torch.equal(a.action, b.action) for a, b in zip(reference.steps, restored.steps))
        checks["resume_rewards_bit_exact"] = all(torch.equal(a.reward, b.reward) for a, b in zip(reference.steps, restored.steps))
        checks["resume_observations_bit_exact"] = all(
            torch.equal(a.observation.proprio_vector(), b.observation.proprio_vector())
            and torch.equal(a.near_points, b.near_points) and torch.equal(a.far_points, b.far_points)
            for a, b in zip(reference.steps, restored.steps))
        checks["resume_histories_bit_exact"] = torch.equal(reference_hidden, runner.collector.hidden) and torch.equal(reference_mirror, runner.collector.mirror_hidden)
        checks["resume_differences"] = [{
            "step": t,
            "action": float((a.action-b.action).abs().max()),
            "mean": float((a.old_mean-b.old_mean).abs().max()),
            "std": float((a.old_std-b.old_std).abs().max()),
            "proprio": float((a.observation.proprio_vector()-b.observation.proprio_vector()).abs().max()),
            "points": float((a.near_points-b.near_points).abs().max()),
            "hidden_before": float((a.hidden_before-b.hidden_before).abs().max()),
        } for t,(a,b) in enumerate(zip(reference.steps,restored.steps)) if not torch.equal(a.action,b.action) or not torch.equal(a.observation.proprio_vector(),b.observation.proprio_vector())]
        # Future simulation is diagnostic; exact acceptance is boundary state
        # and a repeated optimizer update on IDENTICAL retained raw inputs.
        runner.load(runner.final_checkpoint)
        runner.trainer.update(reference)
        expected = copy.deepcopy({"network": runner.network.state_dict(), "provider": runner.provider.state_dict(), "trainer": runner.trainer.state_dict()})
        runner.load(runner.final_checkpoint)
        runner.trainer.update(replay_buffer)
        replay_diff = differences(expected, {"network": runner.network.state_dict(), "provider": runner.provider.state_dict(), "trainer": runner.trainer.state_dict()})
        checks["same_rollout_optimizer_replay_bit_exact"] = not replay_diff
        checks["optimizer_replay_differences"] = replay_diff
        runner.load(runner.final_checkpoint)
        # Explicit timeout fixture: validates natural final bootstrap versus
        # post-reset observation, not a claim of natural episode completion.
        env.episode_length_buf[0] = env.max_episode_length - 1
        runner.runtime.step_in_episode[0] = env.max_episode_length - 1
        runner.runtime._input_cache.invalidate()
        timeout = runner.collector.collect()
        checks["timeout_fixture_reset"] = bool(timeout.steps[0].truncated[0]) and int(timeout.steps[1].observation.episode_ids[0]) == 1
        checks["timeout_hidden_zero"] = bool((timeout.steps[1].hidden_before[0] == 0).all()) and bool((timeout.steps[1].mirror_hidden_before[0] == 0).all())
        checks["timeout_bootstrap_finite"] = bool(torch.isfinite(timeout.steps[0].next_value).all())
        if args.save_interval == 1:
            # A deliberately shortened interval tests real serialization and
            # cumulative resume scheduling; CPU checks cover the default 50.
            first_checkpoint = runner.final_checkpoint
            first_periodic = out / "training/checkpoint_full_000001.pt"
            checks["periodic_first_boundary"] = first_periodic.is_file()
            checks["final_reuses_periodic_storage"] = (
                Path(first_checkpoint).stat().st_ino == first_periodic.stat().st_ino)
            runner.load(first_checkpoint)
            runner.learn(1)
            checks["periodic_resume_cumulative_boundary"] = (
                runner.current_learning_iteration == 2
                and (out / "training/checkpoint_full_000002.pt").is_file())
        records = [json.loads(line) for line in
                   (out / "training/learning_metrics.jsonl").read_text().splitlines()]
        checks["learning_metrics_finite"] = all(
            math.isfinite(value) for record in records for value in record.values())
        checks["learning_metrics_reward_reconstruction"] = all(
            abs(row["Reward/actual"] - row["Reward/reconstructed"]) < 1e-6
            for row in records)
        checks["no_invented_completed_episode_mean"] = all(
            row["Episode/completed_count"] > 0 or "Episode/mean_return" not in row
            for row in records)
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        events = EventAccumulator(str(out / "training/tensorboard")).Reload()
        required_tags = {"Loss/total", "Proprio/vx_rmse_mps", "Reward/actual",
                         "Scan/far_valid_count", "Optimizer/preclip_grad_norm_mean"}
        checks["tensorboard_learning_tags"] = required_tags.issubset(events.Tags()["scalars"])
        diagnostics = {"resume_actions_bit_exact", "resume_rewards_bit_exact", "resume_observations_bit_exact", "resume_histories_bit_exact", "resume_differences", "restore_boundary_differences", "optimizer_replay_differences"}
        checks["passed"] = all(value for name,value in checks.items() if name not in diagnostics)
    except BaseException as error:
        checks["passed"] = False
        checks["error"] = repr(error)
        checks["traceback"] = traceback.format_exc()
    finally:
        if runner is not None:
            runner.close()
        elif env is not None:
            env.close()
        checks["elapsed_s"] = time.monotonic() - started
        if torch.cuda.is_available():
            checks["gpu"] = torch.cuda.get_device_name(args.gpu)
            checks["peak_allocated_MiB"] = torch.cuda.max_memory_allocated(args.gpu) / 2**20
        (out / "checks.json").write_text(json.dumps(checks, indent=2, ensure_ascii=False))
        print(json.dumps(checks, indent=2, ensure_ascii=False))
    if not checks["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
