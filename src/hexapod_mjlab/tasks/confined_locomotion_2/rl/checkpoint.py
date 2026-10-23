"""Independent full-state checkpoints, including both recurrent histories."""

from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass
import hashlib
import os
from pathlib import Path
import random
import uuid

import numpy as np
import torch

from hexapod_mjlab.tasks.confined_locomotion_2.sampling import sampling_identity
from hexapod_mjlab.tasks.confined_locomotion.rl.checkpoint import (
    _compiled_model_identity, _manager_tensors, _physics_arrays,
    _write_persistent_fields,
)

SCHEMA = "confined_locomotion_2.full/v1"
TASK_ID = "El-4090-Confined-Locomotion-Flat_2"


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if is_dataclass(value) and not isinstance(value, type):
        return type(value)(**{field.name: cpu_tree(getattr(value, field.name)) for field in fields(value)})
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def require_finite(value):
    if torch.is_tensor(value):
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError("checkpoint contains non-finite tensors")
    elif isinstance(value, dict):
        for item in value.values():
            require_finite(item)
    elif is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            require_finite(getattr(value, field.name))
    elif isinstance(value, (list, tuple)):
        for item in value:
            require_finite(item)


def architecture_identity(far_sampling="all_endpoints"):
    return {"task": TASK_ID, "near": 256, "far": 128, "window": 1,
            "grouping": "fixed_t1_channels/v1",
            "frame_gru": [3, 187], "temporal_gru": [196, 187],
            "envelope_heads": [6], "ground_head": False,
            "scan_age": False, "control_hz": 50, "scan_hz": 10,
            "sampling": sampling_identity(far_sampling)}


def identity(runner):
    source = {}
    root = Path(__file__).resolve().parents[2]
    for directory in (root / "confined_locomotion_2", root.parent / "rl" / "confined_locomotion_2",
                      root / "confined_locomotion", root.parent / "rl" / "confined_locomotion",
                      root.parent / "utils" / "lidar", root / "el_4090_lidar"):
        for path in sorted(directory.rglob("*.py")):
            source[str(path.relative_to(root.parent))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"architecture": architecture_identity(runner.runtime.lidar_cfg.far_sampling), "seed": runner.seed,
            "num_envs": runner.num_envs, "device": str(runner.device),
            "cuda_name": torch.cuda.get_device_name(runner.device) if runner.device.type == "cuda" else None,
            "torch": torch.__version__, "recipe": runner.recipe_identity,
            "scan_identity": runner.runtime.scan_identity,
            "compiled_model": _compiled_model_identity(runner.runtime.env),
            "source": source}


def require_stable(runner):
    runtime = runner.runtime
    if runner.phase != "ready" or runner._active_buffer is not None:
        raise RuntimeError("full save requires a completed collect/update boundary")
    if runtime._faulted or runtime._stopped or runtime._closed or runtime._context.phase != "idle" or runner.trainer.faulted:
        raise RuntimeError("faulted/stopped/closed or active runtime is not resumable")
    if runner.total_control_batches != 24 * runner.current_learning_iteration:
        raise RuntimeError("incomplete rollout accounting")
    if runner.trainer.update_count != runner.current_learning_iteration or runner.trainer.optimizer_step_count != 20 * runner.current_learning_iteration:
        raise RuntimeError("incomplete optimizer update accounting")
    if runtime.control_steps_taken != runner.total_control_batches:
        raise RuntimeError("unupdated environment transitions exist")
    if bool(runtime.env._manual_reset_pending.any()):
        raise RuntimeError("pending reset is not a stable save boundary")


def save_full_checkpoint(runner, path, *, infos=None):
    require_stable(runner)
    target = Path(path)
    if target.exists():
        raise FileExistsError(target)
    env = runner.runtime.env
    physics, manifest = _physics_arrays(env)
    action = env.action_manager.get_term("residual_pd")
    action_fields = ("prev_action", "last_raw_action", "last_clipped_action",
                     "last_target", "_processed_actions", "_raw_actions")
    env_fields = ("episode_length_buf", "_manual_reset_pending", "reset_terminated",
                  "reset_time_outs", "reset_buf", "reward_buf")
    payload = {
        "schema": SCHEMA, "task": TASK_ID, "kind": "full", "resumable": True,
        "identity": identity(runner), "infos": infos,
        "network": runner.network.state_dict(), "provider": runner.provider.state_dict(),
        "trainer": runner.trainer.state_dict(), "runtime": runner.runtime.state_dict(),
        "collector": runner.collector.state_dict(),
        "learning_metrics": runner.learning_metrics.state_dict(),
        "physics": physics, "physics_manifest": manifest,
        "env": {name: getattr(env, name, None) for name in env_fields},
        "env_counters": {"common_step_counter": int(env.common_step_counter),
                         "_sim_step_counter": int(env._sim_step_counter)},
        "action": {name: getattr(action, name) for name in action_fields},
        "action_control_steps": int(action.control_steps_seen),
        "managers": _manager_tensors(env),
        "counters": {"iteration": runner.current_learning_iteration,
                     "control_batches": runner.total_control_batches,
                     "env_samples": runner.total_env_samples},
        "random": {"python": random.getstate(), "numpy": np.random.get_state(),
                   "torch": torch.get_rng_state(),
                   "cuda": torch.cuda.get_rng_state(runner.device) if runner.device.type == "cuda" else None,
                   "action": runner.provider.generator_state()},
    }
    payload = cpu_tree(payload)
    require_finite(payload)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp-" + uuid.uuid4().hex)
    try:
        torch.save(payload, temporary)
        # A same-directory hard link commits atomically without overwriting.
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def read_checkpoint(path, *, expected_far_sampling=None):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA or payload.get("task") != TASK_ID:
        raise ValueError("checkpoint is not a confined_locomotion_2 full checkpoint")
    if payload.get("kind") != "full" or payload.get("resumable") is not True:
        raise ValueError("checkpoint is not resumable")
    saved_architecture = payload.get("identity", {}).get("architecture")
    supported_sampling = ("all_endpoints", "hit_only_10m")
    if expected_far_sampling is not None:
        if expected_far_sampling not in supported_sampling:
            raise ValueError("unsupported expected far sampling mode")
        expected_architecture = architecture_identity(expected_far_sampling)
        if saved_architecture != expected_architecture:
            raise ValueError("checkpoint architecture does not match this task/far sampling mode")
    elif not any(saved_architecture == architecture_identity(mode) for mode in supported_sampling):
        raise ValueError("checkpoint architecture does not match a supported task/far sampling mode")
    require_finite(payload)
    return payload


def _validate_tensors(saved, live, label):
    if saved.keys() != live.keys():
        raise ValueError(f"{label} keys differ")
    for name, value in live.items():
        other = saved[name]
        if not torch.is_tensor(other) or other.shape != value.shape or other.dtype != value.dtype:
            raise ValueError(f"{label}.{name} shape/dtype differs")


def load_full_checkpoint(runner, path):
    if runner.phase not in ("fresh", "ready") or runner._active_buffer is not None:
        raise RuntimeError("load requires a fresh or stable runner")
    payload = read_checkpoint(
        path, expected_far_sampling=runner.runtime.lidar_cfg.far_sampling
    )
    if payload["identity"] != identity(runner):
        raise ValueError("full resume identity differs (task/source/recipe/model/device)")
    env = runner.runtime.env
    physics, _ = _physics_arrays(env)
    _validate_tensors(payload["physics"], physics, "physics")
    _validate_tensors(payload["network"], runner.network.state_dict(), "network")
    _validate_tensors(payload["provider"], runner.provider.state_dict(), "provider")
    _validate_tensors(payload["managers"], _manager_tensors(env), "managers")
    try:
        runner.network.load_state_dict(payload["network"], strict=True)
        runner.provider.load_state_dict(payload["provider"], strict=True)
        runner.trainer.load_state_dict(payload["trainer"])
        _write_persistent_fields(env, payload["physics"], "v2_restore")
        env.sim.forward()
        _write_persistent_fields(env, payload["physics"], "v2_post_forward_rewrite")
        runner.runtime._context.sensor._invalidate_cache()
        for name, value in payload["env_counters"].items():
            setattr(env, name, value)
        with torch.no_grad():
            for name, saved in payload["env"].items():
                if saved is None:
                    continue
                live = getattr(env, name, None)
                if live is None:
                    setattr(env, name, saved.to(runner.device).clone())
                else:
                    live.copy_(saved.to(live.device))
            action = env.action_manager.get_term("residual_pd")
            for name, saved in payload["action"].items():
                live = getattr(action, name)
                live.copy_(saved.to(live.device))
            action.control_steps_seen = payload["action_control_steps"]
            for name, live in _manager_tensors(env).items():
                live.copy_(payload["managers"][name].to(live.device))
        runner.runtime.load_state_dict(payload["runtime"])
        runner.collector.load_state_dict(payload["collector"])
        runner.learning_metrics.load_state_dict(payload["learning_metrics"])
        counters = payload["counters"]
        runner.current_learning_iteration = counters["iteration"]
        runner.total_control_batches = counters["control_batches"]
        runner.total_env_samples = counters["env_samples"]
        runner.provider.load_generator_state(payload["random"]["action"])
        random.setstate(payload["random"]["python"])
        np.random.set_state(payload["random"]["numpy"])
        torch.set_rng_state(payload["random"]["torch"])
        if payload["random"]["cuda"] is not None:
            torch.cuda.set_rng_state(payload["random"]["cuda"], runner.device)
        runner.phase = "ready"
    except BaseException as error:
        runner.phase = "faulted"
        runner.runtime._save_fault(error, phase="v2_checkpoint_restore")
        raise
    return payload.get("infos")


def load_weights(network, path, *, expected_far_sampling=None):
    """Play starts a new episode, with zero history; it is not full resume."""
    payload = read_checkpoint(path, expected_far_sampling=expected_far_sampling)
    network.load_state_dict(payload["network"], strict=True)
    return payload
