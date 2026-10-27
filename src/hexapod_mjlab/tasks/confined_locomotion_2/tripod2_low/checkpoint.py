"""Independent strict full-state checkpoints for the migrated tripod recipe."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
import os
from pathlib import Path
import random
import uuid

import numpy as np
import torch

from ..rl import checkpoint as base

SCHEMA = "confined_locomotion_flat_tripod2_low.full/v1"
TASK_ID = "El-4090-Confined-Locomotion-Flat-Tripod2-Low_2"
WARMUP_CONTROL_STEPS = 1
ACTION_FIELDS = ("prev_action", "last_raw_action", "last_clipped_action",
                 "last_target", "_processed_actions", "_raw_actions")
ENV_FIELDS = ("episode_length_buf", "_manual_reset_pending", "reset_terminated",
              "reset_time_outs", "reset_buf", "reward_buf")
MODEL_FIELDS = ("body_mass", "body_inertia", "geom_friction", "pair_friction")
REQUIRED = frozenset(("schema", "task", "kind", "resumable", "identity", "network",
                     "provider", "trainer", "runtime", "collector", "learning_metrics",
                     "physics", "physics_manifest", "env", "env_counters", "action",
                     "action_control_steps", "model", "managers", "counters", "random"))


def architecture_identity(far_sampling="all_endpoints"):
    return {**base.architecture_identity(far_sampling), "task": TASK_ID}


def identity(runner):
    result = base.identity(runner)
    result["warmup_control_steps"] = WARMUP_CONTROL_STEPS
    return result


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
    if runtime.control_steps_taken != runner.total_control_batches + WARMUP_CONTROL_STEPS:
        raise RuntimeError("environment transitions differ from rollouts plus startup warmup")
    if bool(runtime.env._manual_reset_pending.any()):
        raise RuntimeError("pending reset is not a stable save boundary")


def _model_tensors(env):
    if tuple(env._tripod_model_fields) != MODEL_FIELDS:
        raise ValueError("tripod randomized model fields differ")
    return {name: getattr(env.sim.model, name) for name in MODEL_FIELDS}


def _action_tensors(env):
    action = env.action_manager.get_term("residual_pd")
    extra = tuple(action._tripod_checkpoint_fields)
    if extra != ("sea_hidden_state", "sea_cell_state", "last_effort"):
        raise ValueError("tripod actuator must declare its hidden/cell checkpoint fields")
    return action, {name: getattr(action, name) for name in (*ACTION_FIELDS, *extra)}


def save_full_checkpoint(runner, path, *, infos=None):
    require_stable(runner)
    if runner.task_id != TASK_ID:
        raise ValueError("unsupported checkpoint task")
    target = Path(path)
    if target.exists():
        raise FileExistsError(target)
    env = runner.runtime.env
    physics, manifest = base._physics_arrays(env)
    action, action_state = _action_tensors(env)
    payload = {
        "schema": SCHEMA, "task": TASK_ID, "kind": "full", "resumable": True,
        "identity": identity(runner), "infos": infos,
        **{name: getattr(runner, name).state_dict() for name in
           ("network", "provider", "trainer", "runtime", "collector", "learning_metrics")},
        "physics": physics, "physics_manifest": manifest,
        "env": {name: getattr(env, name, None) for name in ENV_FIELDS},
        "env_counters": {name: int(getattr(env, name)) for name in
                         ("common_step_counter", "_sim_step_counter")},
        "action": action_state, "action_control_steps": int(action.control_steps_seen),
        "model": _model_tensors(env), "managers": base._manager_tensors(env),
        "counters": {"iteration": runner.current_learning_iteration,
                     "control_batches": runner.total_control_batches,
                     "env_samples": runner.total_env_samples},
        "random": {"python": random.getstate(), "numpy": np.random.get_state(),
                   "torch": torch.get_rng_state(), "action": runner.provider.generator_state(),
                   "cuda": torch.cuda.get_rng_state(runner.device) if runner.device.type == "cuda" else None},
    }
    payload = base.cpu_tree(payload)
    base.require_finite(payload)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp-" + uuid.uuid4().hex)
    try:
        torch.save(payload, temporary)
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def read_checkpoint(path, *, expected_far_sampling=None, expected_task_id=TASK_ID):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if expected_task_id != TASK_ID or not isinstance(payload, dict) or not REQUIRED <= payload.keys():
        raise ValueError("incomplete or unsupported tripod checkpoint")
    if payload["schema"] != SCHEMA or payload["task"] != TASK_ID:
        raise ValueError("checkpoint is not a tripod full checkpoint")
    if payload["kind"] != "full" or payload["resumable"] is not True:
        raise ValueError("checkpoint is not resumable")
    modes = (expected_far_sampling,) if expected_far_sampling is not None else ("all_endpoints", "hit_only_10m")
    if any(mode not in ("all_endpoints", "hit_only_10m") for mode in modes) or not any(
            payload["identity"].get("architecture") == architecture_identity(mode) for mode in modes):
        raise ValueError("checkpoint architecture does not match tripod/far sampling mode")
    base.require_finite(payload)
    return payload


def _validate_tree(saved, live, label):
    """Reject missing nested state and incompatible tensors before applying it."""
    if isinstance(live, dict):
        if not isinstance(saved, dict) or saved.keys() != live.keys():
            raise ValueError(f"{label} keys differ")
        for name, item in live.items():
            _validate_tree(saved[name], item, f"{label}.{name}")
    elif torch.is_tensor(live):
        if not torch.is_tensor(saved) or saved.shape != live.shape or saved.dtype != live.dtype:
            raise ValueError(f"{label} shape/dtype differs")
    elif is_dataclass(live):
        if type(saved) is not type(live):
            raise ValueError(f"{label} dataclass type differs")
        for field in fields(live):
            _validate_tree(getattr(saved, field.name), getattr(live, field.name), f"{label}.{field.name}")
    elif isinstance(live, (list, tuple)) and not isinstance(saved, type(live)):
        raise ValueError(f"{label} sequence type differs")


def _validate_runtime_cache(state):
    """Validate optional scan caches even when the fresh runtime has no cache."""
    if state.get("schema") != "tripod2_low.runtime/v1":
        return
    if state.get("warmup_control_steps") != WARMUP_CONTROL_STEPS:
        raise ValueError("runtime startup warmup differs")
    from hexapod_mjlab.rl.confined_locomotion_2.contracts import ScanWindowSnapshot, OnlineObservation
    from hexapod_mjlab.rl.confined_locomotion_2.integration import PreparedControlInputs
    cache = state["cache"]
    prepared, snapshot = cache["_prepared"], state["window_snapshot"]
    batch = len(state["tensors"]["episode_ids"])
    proprio = state["proprio_cached"]
    if proprio is not None:
        _validate_tree(proprio, {name: torch.empty(batch, 3) for name in
                                ("lin_vel_b", "ang_vel_b", "gravity_b")}, "runtime proprio_cached")
    if snapshot is not None:
        if type(snapshot) is not ScanWindowSnapshot:
            raise ValueError("runtime window_snapshot type differs")
        snapshot.validate()
        if snapshot.env_ids.shape != (batch,):
            raise ValueError("runtime window_snapshot batch differs")
    if prepared is None:
        if cache["_identity"] is not None or cache["_episodes"] is not None:
            raise ValueError("runtime empty cache identity differs")
        return
    if type(prepared) is not PreparedControlInputs or type(prepared.observation) is not OnlineObservation:
        raise ValueError("runtime cache prepared type differs")
    prepared.observation.validate()
    if type(cache["_identity"]) is not int or cache["_identity"] != state["control_steps"]:
        raise ValueError("runtime cache control identity differs")
    episodes = cache["_episodes"]
    _validate_tree(episodes, state["tensors"]["episode_ids"], "runtime cache episodes")
    if not torch.equal(episodes, prepared.observation.episode_ids) or episodes.shape != (batch,):
        raise ValueError("runtime cache episode identity differs")
    for side, count in (("near", 256), ("far", 128)):
        layout = {"points": ((batch, count, 3), torch.float32),
                  "valid": ((batch, count), torch.bool),
                  "angle_keys": ((batch, count), torch.float32),
                  "source_dirs_sensor": ((batch, count, 3), torch.float32),
                  "source_dirs_present": ((batch, count), torch.bool),
                  "source_indices": ((batch, count), torch.int64)}
        for name, (shape, dtype) in layout.items():
            value = getattr(prepared, side + "_" + name)
            if value is None and name not in ("points", "valid"):
                continue
            _validate_tree(value, torch.empty(shape, dtype=dtype), f"runtime cache {side}_{name}")
    for name, shape, dtype in (("near_filled", (batch,), torch.bool),
                               ("near_scan_ids", (batch,), torch.int64),
                               ("near_times", (batch,), torch.float32),
                               ("far_point_scan_ids", (batch, 128), torch.int64),
                               ("far_point_times", (batch, 128), torch.float32)):
        _validate_tree(getattr(prepared, name), torch.empty(shape, dtype=dtype), "runtime cache " + name)


def load_full_checkpoint(runner, path):
    if runner.phase not in ("fresh", "ready") or runner._active_buffer is not None:
        raise RuntimeError("load requires a fresh or stable runner")
    payload = read_checkpoint(path, expected_far_sampling=runner.runtime.lidar_cfg.far_sampling,
                              expected_task_id=runner.task_id)
    base.compare_identity(identity(runner), payload["identity"],
                          extra_keys=("warmup_control_steps",))
    env = runner.runtime.env
    action, action_state = _action_tensors(env)
    physics, _ = base._physics_arrays(env)
    for name, live in (("physics", physics), ("network", runner.network.state_dict()),
                       ("provider", runner.provider.state_dict()), ("model", _model_tensors(env)),
                       ("managers", base._manager_tensors(env)), ("action", action_state),
                       ("runtime", runner.runtime.state_dict())):
        _validate_tree(payload[name], live, name)
    for section, keys in (("env", ENV_FIELDS),
                          ("env_counters", ("common_step_counter", "_sim_step_counter")),
                          ("counters", ("iteration", "control_batches", "env_samples")),
                          ("random", ("python", "numpy", "torch", "action", "cuda"))):
        if not isinstance(payload[section], dict) or set(payload[section]) != set(keys):
            raise ValueError(f"{section} keys differ")
    _validate_runtime_cache(payload["runtime"])
    try:
        runner.network.load_state_dict(payload["network"], strict=True)
        runner.provider.load_state_dict(payload["provider"], strict=True)
        runner.trainer.load_state_dict(payload["trainer"])
        with torch.no_grad():
            for name, live in _model_tensors(env).items():
                live.copy_(payload["model"][name].to(live.device))
        from mjlab.managers.event_manager import RecomputeLevel
        env.sim.recompute_constants(RecomputeLevel.set_const)
        base._write_persistent_fields(env, payload["physics"], "tripod_restore")
        env.sim.forward()
        base._write_persistent_fields(env, payload["physics"], "tripod_post_forward")
        runner.runtime._context.sensor._invalidate_cache()
        for name, value in payload["env_counters"].items():
            setattr(env, name, value)
        with torch.no_grad():
            for name, saved in payload["env"].items():
                if saved is not None:
                    live = getattr(env, name, None)
                    if live is None:
                        setattr(env, name, saved.to(runner.device).clone())
                    else:
                        live.copy_(saved.to(live.device))
            for name, live in action_state.items():
                live.copy_(payload["action"][name].to(live.device))
            action.control_steps_seen = payload["action_control_steps"]
            for name, live in base._manager_tensors(env).items():
                live.copy_(payload["managers"][name].to(live.device))
        for name in ("runtime", "collector", "learning_metrics"):
            getattr(runner, name).load_state_dict(payload[name])
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
        require_stable(runner)
    except BaseException as error:
        runner.phase = "faulted"
        runner.runtime._save_fault(error, phase="tripod_checkpoint_restore")
        raise
    return payload.get("infos")


def load_weights(network, path, *, expected_far_sampling=None, expected_task_id=TASK_ID):
    payload = read_checkpoint(path, expected_far_sampling=expected_far_sampling, expected_task_id=expected_task_id)
    network.load_state_dict(payload["network"], strict=True)
    return payload
