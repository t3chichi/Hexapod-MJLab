"""Finite full-vs-sparse training-stage CUDA benchmark from saved boundaries.

For each mode, repeatedly restore its own resumable checkpoint, time one real
24-control-step collector rollout and one real 20-optimizer-step PPO update,
then restore again before the next sample. This is not a learning run and does
not write a checkpoint.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import sys
import time
import traceback

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def _validate_checkpoint(path: Path, *, mode: str, seed: int, num_envs: int) -> dict:
    from hexapod_mjlab.tasks.confined_locomotion_2.rl.checkpoint import read_checkpoint

    if not path.is_file():
        raise FileNotFoundError(path)
    payload = read_checkpoint(path)
    identity = payload["identity"]
    if identity.get("seed") != seed:
        raise ValueError(
            f"{mode} checkpoint seed {identity.get('seed')} != requested seed {seed}"
        )
    if identity.get("num_envs") != num_envs:
        raise ValueError(
            f"{mode} checkpoint num_envs {identity.get('num_envs')} != requested {num_envs}"
        )
    scan_identity = identity.get("scan_identity", {})
    if scan_identity.get("mode") != mode:
        raise ValueError(
            f"{mode} checkpoint has scan mode {scan_identity.get('mode')!r}"
        )
    summary = {
        "path": str(path),
        "seed": identity["seed"],
        "num_envs": identity["num_envs"],
        "scan_mode": scan_identity.get("mode"),
        "counters": payload.get("counters"),
        "optimizer_step_count": payload.get("trainer", {}).get("optimizer_step_count"),
    }
    del payload
    return summary


def _measure_mode(*, mode: str, checkpoint: Path, selection: Path,
                  seed: int, num_envs: int, gpu: int, warmup: int,
                  samples: int, log_dir: Path) -> dict:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
    from hexapod_mjlab.tasks.confined_locomotion_2 import make_training_env_cfg
    from hexapod_mjlab.tasks.confined_locomotion_2.config.flat_baseline import (
        confined_locomotion_formal_rl_cfg,
    )
    from hexapod_mjlab.tasks.confined_locomotion_2.rl.runner import (
        ConfinedLocomotionRunner,
    )

    device = torch.device(f"cuda:{gpu}")
    cfg = confined_locomotion_formal_rl_cfg(
        seed=seed, num_envs=num_envs, max_iterations=1
    )
    cfg.engineering_enabled = True
    cfg.lidar_config = {
        "raycast_mode": mode,
        "sparse_selection_path": str(selection) if mode == "sparse" else None,
    }

    env = None
    runner = None
    try:
        env = ManagerBasedRlEnv(
            make_training_env_cfg(num_envs=num_envs, seed=seed),
            device=str(device),
        )
        wrapper = RslRlVecEnvWrapper(env, clip_actions=None)
        runner = ConfinedLocomotionRunner(
            wrapper, asdict(cfg), log_dir, str(device),
        )
        if not runner.symmetry_config.enabled or runner.symmetry_config.mean_loss_coeff != 0.6:
            raise RuntimeError("formal benchmark recipe must use mirror augmentation at 0.6")
        if runner.trainer.config.epochs * runner.trainer.config.minibatches != 20:
            raise RuntimeError("formal benchmark recipe must perform exactly 20 PPO updates")

        measured = []
        warmup_results = []
        stable_boundary = None
        total_runs = warmup + samples
        for run_index in range(total_runs):
            # Every run starts at the checkpoint's same stable physics, runtime,
            # recurrent, network, distribution and optimizer boundary.
            runner.load(str(checkpoint))
            torch.cuda.synchronize(device)
            boundary = {
                "control_steps": runner.runtime.control_steps_taken,
                "env_common_step_counter": int(runner.runtime.env.common_step_counter),
                "episode_ids": runner.runtime.episode_ids.detach().cpu().tolist(),
                "step_in_episode": runner.runtime.step_in_episode.detach().cpu().tolist(),
                "scan_counters": runner.runtime.scan_counters.detach().cpu().tolist(),
            }
            if stable_boundary is None:
                stable_boundary = boundary
            elif boundary != stable_boundary:
                raise RuntimeError("checkpoint restore did not return to the same stable boundary")
            baseline_allocated = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)

            control_before = runner.runtime.control_steps_taken
            optimizer_before = runner.trainer.optimizer_step_count
            update_count_before = runner.trainer.update_count
            collect_start = time.perf_counter()
            buffer = runner.collector.collect()
            torch.cuda.synchronize(device)
            collect_s = time.perf_counter() - collect_start
            if len(buffer.steps) != 24:
                raise RuntimeError(f"collector returned {len(buffer.steps)} steps, expected 24")
            if runner.runtime.control_steps_taken - control_before != 24:
                raise RuntimeError("runtime control clock did not advance by 24 steps")

            update_start = time.perf_counter()
            metrics = runner.trainer.update(buffer)
            torch.cuda.synchronize(device)
            update_s = time.perf_counter() - update_start
            if runner.trainer.optimizer_step_count - optimizer_before != 20:
                raise RuntimeError("trainer did not perform exactly 20 optimizer steps")
            if runner.trainer.update_count - update_count_before != 1:
                raise RuntimeError("trainer did not complete exactly one update")

            peak_allocated = torch.cuda.max_memory_allocated(device)
            elapsed = collect_s + update_s
            result = {
                "collect_s": collect_s,
                "update_s": update_s,
                "collect_plus_update_s": elapsed,
                "train_env_steps_per_s": 24 * num_envs / elapsed,
                "optimizer_steps_this_run": runner.trainer.optimizer_step_count - optimizer_before,
                "optimizer_steps_total_after": int(metrics["optimizer_steps"]),
                "buffer_steps": len(buffer.steps),
                "restored_boundary": boundary,
                "baseline_allocated_MiB": baseline_allocated / 2**20,
                "peak_allocated_total_MiB": peak_allocated / 2**20,
                "peak_allocated_incremental_MiB": max(
                    0, peak_allocated - baseline_allocated
                ) / 2**20,
            }
            del buffer, metrics
            if run_index < warmup:
                warmup_results.append(result)
            else:
                measured.append(result)

        return {
            "mode": mode,
            "checkpoint": str(checkpoint),
            "stable_boundary": stable_boundary,
            "warmup_runs": warmup_results,
            "measured_runs": measured,
            "median": {
                key: statistics.median(run[key] for run in measured)
                for key in (
                    "collect_s", "update_s", "collect_plus_update_s",
                    "train_env_steps_per_s", "peak_allocated_total_MiB",
                    "peak_allocated_incremental_MiB",
                )
            },
        }
    finally:
        if runner is not None:
            runner.close()
        elif env is not None:
            env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-checkpoint", required=True, type=Path)
    parser.add_argument("--sparse-checkpoint", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-envs", type=int, choices=(1, 2), default=2)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    full_checkpoint = args.full_checkpoint.expanduser().resolve()
    sparse_checkpoint = args.sparse_checkpoint.expanduser().resolve()
    selection = args.selection.expanduser().resolve()
    if args.samples < 1 or args.warmup < 0:
        raise ValueError("samples must be >=1 and warmup must be >=0")
    if not selection.is_file():
        raise FileNotFoundError(selection)
    if output.exists():
        raise FileExistsError(f"benchmark output already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)

    report: dict = {
        "schema": "confined_locomotion_2.fixed_training_gpu_benchmark/v1",
        "passed": False,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "gpu_index": args.gpu,
        "samples": args.samples,
        "warmup": args.warmup,
        "selection": str(selection),
        "started_unix_s": time.time(),
    }
    started = time.monotonic()
    try:
        if not torch.cuda.is_available() or not 0 <= args.gpu < torch.cuda.device_count():
            raise RuntimeError("a valid real CUDA device is required")
        torch.cuda.set_device(args.gpu)
        report["gpu"] = torch.cuda.get_device_name(args.gpu)
        report["checkpoints"] = {
            "full": _validate_checkpoint(
                full_checkpoint, mode="full", seed=args.seed, num_envs=args.num_envs
            ),
            "sparse": _validate_checkpoint(
                sparse_checkpoint, mode="sparse", seed=args.seed, num_envs=args.num_envs
            ),
        }

        # Environments and runners are strictly sequential so the two modes
        # never overlap on the GPU. Each is closed before the next is built.
        report["modes"] = {}
        for mode, checkpoint in (
            ("full", full_checkpoint), ("sparse", sparse_checkpoint),
        ):
            report["modes"][mode] = _measure_mode(
                mode=mode,
                checkpoint=checkpoint,
                selection=selection,
                seed=args.seed,
                num_envs=args.num_envs,
                gpu=args.gpu,
                warmup=args.warmup,
                samples=args.samples,
                log_dir=output / mode / "runner",
            )
        report["passed"] = True
    except BaseException as error:
        report["error"] = repr(error)
        report["traceback"] = traceback.format_exc()
    finally:
        report["elapsed_s"] = time.monotonic() - started
        (output / "benchmark.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
