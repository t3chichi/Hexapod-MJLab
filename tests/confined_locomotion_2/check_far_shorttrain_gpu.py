"""Finite same-seed GPU comparison of the two far-point sampling policies.

Run this script once per ``--far-sampling`` value in a fresh process. Each run
uses the formal task recipe and saves the official full checkpoint at the
stable boundary after the requested number of complete iterations. Ten
iterations (480 environment transitions with two envs) are only a finite
stability and initial-motion probe; this script makes no learning claim.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
SELECTION = (
    ROOT / "src/hexapod_mjlab/tasks/confined_locomotion_2/assets/fixed_channels"
    / "v1_hit10/near_x6_far_x10.npz"
)
SOURCE_FILES = (
    "tests/confined_locomotion_2/check_far_shorttrain_gpu.py",
    "src/hexapod_mjlab/tasks/confined_locomotion_2/rl/runner.py",
    "src/hexapod_mjlab/tasks/confined_locomotion_2/config/flat_baseline.py",
    "src/hexapod_mjlab/tasks/confined_locomotion_2/scanner.py",
    "src/hexapod_mjlab/tasks/confined_locomotion_2/runtime.py",
    "src/hexapod_mjlab/tasks/confined_locomotion_2/collector.py",
    "src/hexapod_mjlab/tasks/confined_locomotion_2/flat_env_cfg.py",
    "src/hexapod_mjlab/tasks/confined_locomotion_2/mdp/rewards.py",
    "src/hexapod_mjlab/tasks/confined_locomotion_2/mdp/terminations.py",
    "src/hexapod_mjlab/assets/robots/el_4090/mjcf/el_4090_lidar.xml",
    "src/hexapod_mjlab/rl/confined_locomotion_2/network.py",
    "src/hexapod_mjlab/rl/confined_locomotion_2/ppo.py",
    "src/hexapod_mjlab/rl/confined_locomotion_2/training_rollout.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_stats(value) -> dict:
    array = torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float64)
    flat = array.reshape(-1)
    finite = torch.isfinite(flat)
    finite_values = flat[finite]
    return {
        "count": int(flat.numel()),
        "finite_count": int(finite.sum()),
        "finite": bool(finite.all()),
        "mean": float(finite_values.mean()) if finite_values.numel() else None,
        "min": float(finite_values.min()) if finite_values.numel() else None,
        "max": float(finite_values.max()) if finite_values.numel() else None,
        "rms": float(finite_values.square().mean().sqrt()) if finite_values.numel() else None,
    }


def _nvml_process_memory(gpu_index: int) -> dict:
    """Return NVML's per-process memory usage, separately from torch reserved."""
    try:
        import pynvml
    except (ImportError, OSError) as error:
        return {"available": False, "reason": f"pynvml unavailable: {error!r}"}
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        pid = os.getpid()
        processes = []
        for getter_name in (
            "nvmlDeviceGetComputeRunningProcesses",
            "nvmlDeviceGetGraphicsRunningProcesses",
        ):
            getter = getattr(pynvml, getter_name, None)
            if getter is None:
                continue
            try:
                processes.extend(getter(handle))
            except Exception:
                pass
        matches = [process for process in processes if int(process.pid) == pid]
        # A process can appear in both lists; report its largest observed entry.
        memory_bytes = max(
            (int(process.usedGpuMemory) for process in matches
             if process.usedGpuMemory is not None and int(process.usedGpuMemory) >= 0),
            default=None,
        )
        return {
            "available": True,
            "pid": pid,
            "used_gpu_memory_bytes": memory_bytes,
            "used_gpu_memory_mib": None if memory_bytes is None else memory_bytes / 2**20,
            "matched_process_entries": len(matches),
            "source": "NVML per-process usedGpuMemory; distinct from torch reserved memory",
        }
    except Exception as error:
        return {"available": False, "reason": f"NVML query failed: {error!r}"}


def _root_position(env) -> torch.Tensor | None:
    try:
        return env.scene["robot"].data.root_link_pos_w.detach().clone()
    except (AttributeError, KeyError, TypeError):
        return None


def _step_diagnostic(runtime, action, transition) -> dict:
    action_stats = _tensor_stats(action)
    context = transition.context_record
    fall = context.get("engineering_fall") if isinstance(context, dict) else None
    if fall is None:
        fall = transition.terminated
    position = context.get("root_pos_w") if isinstance(context, dict) else None
    height_stats = None
    displacement = None
    if position is not None:
        root = torch.as_tensor(position).detach().cpu()
        height_stats = _tensor_stats(root[:, 2])
    return {
        "control_id": int(transition.control_id),
        "action": action_stats,
        "reward": _tensor_stats(transition.reward),
        "terminated_count": int(transition.terminated.sum()),
        "truncated_count": int(transition.truncated.sum()),
        "fall_count": int(torch.as_tensor(fall).sum()),
        "engineering_stopped": bool(transition.engineering_stopped),
        "root_height": height_stats,
        "root_xy_displacement_m": displacement,
    }


def _buffer_diagnostics(buffer) -> dict:
    actions = torch.stack([step.action for step in buffer.steps])
    rewards = torch.stack([step.reward for step in buffer.steps])
    terminated = torch.stack([step.terminated for step in buffer.steps])
    truncated = torch.stack([step.truncated for step in buffer.steps])
    far_counts = torch.stack([step.far_valid.sum(dim=1) for step in buffer.steps])
    far_pct = far_counts.to(torch.float32) / max(1, buffer.steps[0].far_valid.shape[1]) * 100.0
    # Prepared inputs retain physical root state only indirectly. Velocities and
    # valid-10m coverage are useful initial-motion diagnostics and remain causal.
    base_speed = torch.stack([
        step.observation.lin_vel_b[:, :2].norm(dim=1) for step in buffer.steps
    ])
    env_action = _tensor_stats(actions)
    env_reward = _tensor_stats(rewards)
    env_speed = _tensor_stats(base_speed)
    far_values = far_pct.detach().cpu()
    return {
        "control_steps": len(buffer.steps),
        "environment_transitions": len(buffer.steps) * int(buffer.num_envs),
        "action": env_action,
        "reward": env_reward,
        "done_count": int((terminated | truncated).sum()),
        "terminated_count": int(terminated.sum()),
        "truncated_count": int(truncated.sum()),
        "fall_count": int(terminated.sum()),
        "base_body_horizontal_speed_m_s": env_speed,
        "far_valid_count_min": int(far_counts.min()),
        "far_valid_count_max": int(far_counts.max()),
        "far_valid_pct_mean": float(far_values.mean()),
        "far_valid_pct_min": float(far_values.min()),
        "far_valid_pct_max": float(far_values.max()),
        "far_valid_pct_first_control": far_values[0].tolist(),
        "far_valid_pct_last_control": far_values[-1].tolist(),
        "far_valid_pct_change_first_to_last": (far_values[-1] - far_values[0]).tolist(),
    }


def _finite_tree(value) -> bool:
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    if torch.is_tensor(value) and value.is_floating_point():
        return bool(torch.isfinite(value).all())
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--far-sampling", required=True,
                        choices=("all_endpoints", "hit_only_10m"))
    parser.add_argument("--gpu0", action="store_true",
                        help="use CUDA device 0 (the default and only supported device)")
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--num-envs", type=int, default=2, choices=(2,))
    parser.add_argument("--iterations", type=int, default=10, choices=(10, 20))
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    report: dict = {
        "schema": "confined_locomotion_2.far_shorttrain_probe/v1",
        "passed": False,
        "interpretation": (
            "finite short-run stability and initial-motion evidence only; "
            "not a learning-effect or policy-quality result"
        ),
        "far_sampling": args.far_sampling,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "iterations_requested": args.iterations,
        "gpu_index": 0,
        "selection": str(SELECTION),
        "started_unix_s": time.time(),
        "source_hashes": {},
        "iterations": [],
    }
    runner = env = None
    clip_grad_original = torch.nn.utils.clip_grad_norm_
    grad_norms: list[dict] = []
    started = time.monotonic()
    try:
        for relative in SOURCE_FILES:
            source_path = ROOT / relative
            report["source_hashes"][relative] = _sha256(source_path)
        if not SELECTION.is_file():
            raise FileNotFoundError(f"fixed candidate selection is missing: {SELECTION}")
        report["source_hashes"][str(SELECTION.relative_to(ROOT))] = _sha256(SELECTION)
        report["selection_sha256"] = report["source_hashes"][str(SELECTION.relative_to(ROOT))]
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise RuntimeError("a real CUDA device 0 is required")
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        report["gpu"] = torch.cuda.get_device_name(0)
        report["torch_cuda"] = torch.version.cuda

        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
        from hexapod_mjlab.tasks.confined_locomotion_2 import make_training_env_cfg
        from hexapod_mjlab.tasks.confined_locomotion_2.config.flat_baseline import (
            confined_locomotion_formal_rl_cfg,
        )
        from hexapod_mjlab.tasks.confined_locomotion_2.rl.runner import (
            ConfinedLocomotionRunner,
        )

        cfg = confined_locomotion_formal_rl_cfg(
            seed=args.seed, num_envs=args.num_envs, max_iterations=args.iterations,
        )
        cfg.lidar_config = {
            "raycast_mode": "sparse",
            "sparse_selection_path": str(SELECTION),
            "far_sampling": args.far_sampling,
        }
        env = ManagerBasedRlEnv(
            make_training_env_cfg(num_envs=args.num_envs, seed=args.seed),
            device=str(device),
        )
        wrapper = RslRlVecEnvWrapper(env, clip_actions=None)
        runner = ConfinedLocomotionRunner(
            wrapper, asdict(cfg), output / "training", str(device),
        )
        if (not runner.symmetry_config.enabled
                or runner.symmetry_config.mean_loss_coeff != 0.6
                or runner.trainer.config.epochs * runner.trainer.config.minibatches != 20
                or runner.runtime.lidar_cfg.far_sampling != args.far_sampling):
            raise RuntimeError("formal recipe, mirror settings, or far sampling did not resolve as requested")
        parameter_count = sum(parameter.numel() for parameter in runner.network.parameters())
        from hexapod_mjlab.tasks.confined_locomotion_2.collector import model_state_fingerprint
        initial_fingerprint = model_state_fingerprint(runner.network)
        report["resolved_recipe"] = {
            "formal": True,
            "observation_perturbations": runner.recipe_identity["perturbations"],
            "symmetry": asdict(runner.symmetry_config),
            "ppo": asdict(runner.trainer.config),
            "scan_identity": runner.runtime.scan_identity,
            "architecture": runner.recipe_identity["architecture"],
            "parameter_count": int(parameter_count),
            "initial_network_sha256": initial_fingerprint,
            "selection_sha256": report["selection_sha256"],
        }
        (output / "recipe.json").write_text(
            json.dumps({"runner_cfg": asdict(cfg), **report["resolved_recipe"]},
                       indent=2, ensure_ascii=False, default=str, allow_nan=False)
        )

        # Preserve the formal collect/update path; these wrappers observe the
        # real rollout and optimizer calls without replaying them.
        original_collect = runner.collector.collect
        original_update = runner.trainer.update
        transition_diagnostics: list[dict] = []
        original_step = runner.runtime.step

        def timed_collect():
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            t0 = time.perf_counter()
            report["_iteration_step_start"] = len(transition_diagnostics)
            buffer = original_collect()
            torch.cuda.synchronize(device)
            report["_collect_s"] = time.perf_counter() - t0
            report["_buffer"] = _buffer_diagnostics(buffer)
            report["_iteration_memory_start_allocated_bytes"] = int(torch.cuda.memory_allocated(device))
            return buffer

        def record_step(action):
            before = _root_position(runner.runtime.env)
            transition = original_step(action)
            after = _root_position(runner.runtime.env)
            diagnostic = _step_diagnostic(runner.runtime, action, transition)
            if before is not None and after is not None:
                displacement = (after[:, :2] - before[:, :2]).norm(dim=1)
                diagnostic["root_xy_displacement_m"] = _tensor_stats(displacement)
                if diagnostic["root_height"] is None:
                    diagnostic["root_height"] = _tensor_stats(after[:, 2])
            transition_diagnostics.append(diagnostic)
            return transition

        def timed_update(buffer):
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            metrics = original_update(buffer)
            torch.cuda.synchronize(device)
            update_s = time.perf_counter() - t0
            index = len(report["iterations"]) + 1
            peak_allocated = int(torch.cuda.max_memory_allocated(device))
            peak_reserved = int(torch.cuda.max_memory_reserved(device))
            allocated = int(torch.cuda.memory_allocated(device))
            reserved = int(torch.cuda.memory_reserved(device))
            start_allocated = int(report.pop("_iteration_memory_start_allocated_bytes"))
            collect_s = float(report.pop("_collect_s"))
            buffer_metrics = report.pop("_buffer")
            step_start = int(report.pop("_iteration_step_start"))
            iteration_steps = transition_diagnostics[step_start:]
            if len(iteration_steps) != 24:
                raise RuntimeError(
                    f"iteration {index} recorded {len(iteration_steps)} runtime steps; expected 24"
                )
            norms = grad_norms[-20:]
            entry = {
                "iteration": index,
                "collect_s": collect_s,
                "update_s": update_s,
                "collect_and_update_s": collect_s + update_s,
                "training_metrics": metrics,
                "training_metrics_finite": _finite_tree(metrics),
                "gradient_norms": norms,
                "gradient_norms_finite": all(item["finite"] for item in norms),
                "optimizer_steps_this_iteration": len(norms),
                "optimizer_steps_total": int(runner.trainer.optimizer_step_count),
                "buffer": buffer_metrics,
                "steps": iteration_steps,
                "runtime_step_diagnostic_count": len(iteration_steps),
                "torch_cuda_memory": {
                    "allocated_bytes_after_update": allocated,
                    "reserved_bytes_after_update": reserved,
                    "peak_allocated_bytes_iteration": peak_allocated,
                    "peak_reserved_bytes_iteration": peak_reserved,
                    "peak_allocated_increment_bytes": max(0, peak_allocated - start_allocated),
                },
                "nvml_process_memory_after_update": _nvml_process_memory(0),
            }
            report["iterations"].append(entry)
            return metrics

        def measured_clip(parameters, max_norm, norm_type=2.0, error_if_nonfinite=False, foreach=None):
            parameter_list = list(parameters)
            finite_before = all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                for parameter in parameter_list
            )
            norm = clip_grad_original(
                parameter_list, max_norm, norm_type=norm_type,
                error_if_nonfinite=error_if_nonfinite, foreach=foreach,
            )
            scalar = float(norm.detach().item()) if torch.is_tensor(norm) else float(norm)
            grad_norms.append({"norm_before_clip": scalar, "finite": bool(finite_before and np.isfinite(scalar))})
            return norm

        runner.collector.collect = timed_collect
        runner.runtime.step = record_step
        runner.trainer.update = timed_update
        torch.nn.utils.clip_grad_norm_ = measured_clip
        runner.learn(args.iterations)
        torch.nn.utils.clip_grad_norm_ = clip_grad_original
        expected_control_steps = 24 * args.iterations
        report["checkpoint"] = runner.final_checkpoint
        report["checkpoint_exists"] = bool(runner.final_checkpoint and Path(runner.final_checkpoint).is_file())
        report["counters"] = {
            "current_learning_iteration": int(runner.current_learning_iteration),
            "total_control_batches": int(runner.total_control_batches),
            "total_env_samples": int(runner.total_env_samples),
            "runtime_control_steps": int(runner.runtime.control_steps_taken),
            "optimizer_steps": int(runner.trainer.optimizer_step_count),
            "expected_control_steps": expected_control_steps,
        }
        report["fall_evidence"] = {
            "terminated_count": sum(item["buffer"]["terminated_count"] for item in report["iterations"]),
            "truncated_count": sum(item["buffer"]["truncated_count"] for item in report["iterations"]),
            "engineering_stopped_steps": sum(
                step["engineering_stopped"] for iteration in report["iterations"] for step in iteration["steps"]
            ),
        }
        report["passed"] = bool(
            report["checkpoint_exists"]
            and runner.current_learning_iteration == args.iterations
            and runner.total_control_batches == expected_control_steps
            and runner.runtime.control_steps_taken == expected_control_steps
            and runner.total_env_samples == expected_control_steps * args.num_envs
            and runner.trainer.optimizer_step_count == args.iterations * 20
            and all(item["training_metrics_finite"] and item["gradient_norms_finite"]
                    and item["optimizer_steps_this_iteration"] == 20
                    and item["runtime_step_diagnostic_count"] == 24
                    for item in report["iterations"])
        )
    except BaseException as error:
        torch.nn.utils.clip_grad_norm_ = clip_grad_original
        report["error"] = repr(error)
        report["traceback"] = traceback.format_exc()
        report["failure_files"] = [
            str(path.relative_to(output)) for path in output.rglob("failure.json")
        ]
        if runner is not None:
            report["runtime_faulted"] = bool(getattr(runner.runtime, "_faulted", False))
            report["runner_phase"] = getattr(runner, "phase", None)
    finally:
        if runner is not None:
            try:
                torch.cuda.synchronize(0)
            except Exception:
                pass
            try:
                runner.close()
            except Exception as close_error:
                report["close_error"] = repr(close_error)
        elif env is not None:
            try:
                env.close()
            except Exception as close_error:
                report["close_error"] = repr(close_error)
        report["elapsed_s"] = time.monotonic() - started
        if torch.cuda.is_available():
            try:
                report["torch_cuda_memory_final"] = {
                    "allocated_bytes": int(torch.cuda.memory_allocated(0)),
                    "reserved_bytes": int(torch.cuda.memory_reserved(0)),
                }
            except Exception as memory_error:
                report["torch_cuda_memory_error"] = repr(memory_error)
        iteration_memory = [item.get("torch_cuda_memory", {}) for item in report["iterations"]]
        report["torch_cuda_memory_max_across_recorded_iterations"] = {
            "peak_allocated_bytes": max(
                (int(item["peak_allocated_bytes_iteration"]) for item in iteration_memory
                 if "peak_allocated_bytes_iteration" in item),
                default=None,
            ),
            "peak_reserved_bytes": max(
                (int(item["peak_reserved_bytes_iteration"]) for item in iteration_memory
                 if "peak_reserved_bytes_iteration" in item),
                default=None,
            ),
            "scope": "maximum of per-iteration peaks; peak counters reset before each rollout",
        }
        nvml_samples = [
            item.get("nvml_process_memory_after_update", {}) for item in report["iterations"]
        ]
        nvml_values = [
            int(item["used_gpu_memory_bytes"]) for item in nvml_samples
            if item.get("available") and item.get("used_gpu_memory_bytes") is not None
        ]
        report["nvml_process_memory_max_observed_after_update"] = {
            "used_gpu_memory_bytes": max(nvml_values, default=None),
            "used_gpu_memory_mib": None if not nvml_values else max(nvml_values) / 2**20,
            "samples_with_process_memory": len(nvml_values),
            "scope": "maximum of point samples after PPO updates; not a continuous peak",
        }
        (output / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str, allow_nan=False)
        )
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str, allow_nan=False))
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
