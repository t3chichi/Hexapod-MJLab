"""Independent full-state checkpoints, including both recurrent histories."""

from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass
import os
from pathlib import Path
import random
import uuid

import numpy as np
import torch

from hexapod_mjlab.tasks.confined_locomotion_2.sampling import sampling_identity
from hexapod_mjlab.common.checkpoint import (
    compiled_model_identity as _compiled_model_identity,
    manager_tensors as _manager_tensors,
    physics_arrays as _physics_arrays,
    write_persistent_fields as _write_persistent_fields,
)

SCHEMA = "confined_locomotion_2.full/v1"
TASK_ID = "El-4090-Confined-Locomotion-Flat_2"
TERRAIN_TASK_ID = "El-4090-Confined-Locomotion-Terrain-Memory_2"


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


def architecture_identity(far_sampling="all_endpoints", *, terrain_memory=False):
    result = {"task": TASK_ID, "near": 256, "far": 128, "window": 1,
            "grouping": "fixed_t1_channels/v1",
            "frame_gru": [3, 187], "temporal_gru": [196, 187],
            "envelope_heads": [6], "ground_head": False,
            "scan_age": False, "control_hz": 50, "scan_hz": 10,
            "sampling": sampling_identity(far_sampling)}
    if terrain_memory:
        result.update(task=TERRAIN_TASK_ID, ground_head=[187, 187],
                      height_grid_shape=[11, 17], height_prediction_to_actor=False)
    return result


def identity(runner):
    return {"architecture": runner.recipe_identity["architecture"], "seed": runner.seed,
            "num_envs": runner.num_envs, "device": str(runner.device),
            "cuda_name": torch.cuda.get_device_name(runner.device) if runner.device.type == "cuda" else None,
            "torch": torch.__version__, "recipe": runner.recipe_identity,
            "scan_identity": runner.runtime.scan_identity,
            "compiled_model": _compiled_model_identity(runner.runtime.env)}


#: Fields that determine whether the saved training state is compatible.
#: Device/Torch metadata and source hashes are recorded but do not gate resume.
_IDENTITY_COMPARE_KEYS = ("architecture", "seed", "num_envs", "scan_identity")
_MODEL_COMPARE_KEYS = ("dims", "joint_names", "actuator_names", "actuator_gear", "q_ref", "options")

#: Required for every v2/tripod checkpoint. Optional semantic fields are
#: compared only when the saved file carries them, so legacy files remain
#: readable while new files gain stricter action/reward/termination checks.
_REQUIRED_RECIPE_KEYS = ("ppo", "symmetry")
_OPTIONAL_RECIPE_KEYS = (
    "perturbations",
    "height_grid",
    "height_loss_weight",
    "actuator_sha256",
    "lidar_config",
    "distribution",
    "proprio_noise_level",
    "running_normalization",
    "lidar_perturbations",
    "env_semantics",
    "legacy_reward_scales",
)


def _model_structure(value):
    if not isinstance(value, dict):
        return None
    return {key: value.get(key) for key in _MODEL_COMPARE_KEYS}


def _compare_recipe(current, saved):
    if not isinstance(current, dict) or not isinstance(saved, dict):
        raise ValueError("checkpoint identity is missing the training recipe")
    for key in _REQUIRED_RECIPE_KEYS:
        if key not in saved:
            raise ValueError(f"checkpoint identity is missing training recipe field {key!r}")
        if current.get(key) != saved[key]:
            raise ValueError(f"full resume identity differs on the training recipe.{key}")
    for key in _OPTIONAL_RECIPE_KEYS:
        if key not in saved:
            continue  # legacy files did not record this semantic field
        if key not in current:
            raise ValueError(f"checkpoint identity is missing training recipe field {key!r}")
        if current[key] != saved[key]:
            raise ValueError(f"full resume identity differs on the training recipe.{key}")


def compare_identity(current, saved, *, extra_keys=()):
    """Compare only state/semantic identity fields; ignore metadata extras."""
    if not isinstance(saved, dict):
        raise ValueError("checkpoint identity is not an object")
    for key in (*_IDENTITY_COMPARE_KEYS, *extra_keys):
        if key not in saved:
            raise ValueError(f"checkpoint identity is missing {key!r}")
        if current.get(key) != saved[key]:
            raise ValueError(f"full resume identity differs on {key!r}")
    current_model = _model_structure(current.get("compiled_model"))
    saved_model = _model_structure(saved.get("compiled_model"))
    if current_model is None or saved_model is None:
        raise ValueError("checkpoint identity is missing the compiled-model structure")
    if current_model != saved_model:
        raise ValueError("full resume identity differs on the compiled-model structure")
    _compare_recipe(current.get("recipe"), saved.get("recipe"))


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
        "schema": SCHEMA, "task": runner.task_id, "kind": "full", "resumable": True,
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


def read_checkpoint(path, *, expected_far_sampling=None, expected_task_id=TASK_ID):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if expected_task_id not in (TASK_ID, TERRAIN_TASK_ID):
        raise ValueError("unsupported checkpoint task")
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA or payload.get("task") != expected_task_id:
        raise ValueError("checkpoint is not a confined_locomotion_2 full checkpoint")
    if payload.get("kind") != "full" or payload.get("resumable") is not True:
        raise ValueError("checkpoint is not resumable")
    saved_architecture = payload.get("identity", {}).get("architecture")
    supported_sampling = ("all_endpoints", "hit_only_10m")
    if expected_far_sampling is not None:
        if expected_far_sampling not in supported_sampling:
            raise ValueError("unsupported expected far sampling mode")
        expected_architecture = architecture_identity(expected_far_sampling, terrain_memory=expected_task_id == TERRAIN_TASK_ID)
        if saved_architecture != expected_architecture:
            raise ValueError("checkpoint architecture does not match this task/far sampling mode")
    elif not any(saved_architecture == architecture_identity(mode, terrain_memory=expected_task_id == TERRAIN_TASK_ID) for mode in supported_sampling):
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
        path, expected_far_sampling=runner.runtime.lidar_cfg.far_sampling, expected_task_id=runner.task_id
    )
    compare_identity(identity(runner), payload["identity"])
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


def load_weights(network, path, *, expected_far_sampling=None, expected_task_id=TASK_ID):
    """Play starts a new episode, with zero history; it is not full resume."""
    payload = read_checkpoint(path, expected_far_sampling=expected_far_sampling, expected_task_id=expected_task_id)
    network.load_state_dict(payload["network"], strict=True)
    return payload
