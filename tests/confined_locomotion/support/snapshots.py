"""CPU snapshots of live training objects used by independent GPU witnesses."""
from __future__ import annotations

import dataclasses
import hashlib
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def cpu_tree(value: Any, *, omit: frozenset[str] = frozenset()) -> Any:
  """Detach and clone tensors recursively without discarding raw values."""
  import torch
  import numpy as np

  if torch.is_tensor(value):
    return value.detach().cpu().clone()
  if dataclasses.is_dataclass(value) and not isinstance(value, type):
    return type(value)(**{
      field.name: cpu_tree(getattr(value, field.name), omit=omit)
      for field in dataclasses.fields(value)
    })
  if isinstance(value, Mapping):
    return {
      key: cpu_tree(item, omit=omit)
      for key, item in value.items()
      if str(key) not in omit
    }
  if isinstance(value, tuple):
    return tuple(cpu_tree(item, omit=omit) for item in value)
  if isinstance(value, list):
    return [cpu_tree(item, omit=omit) for item in value]
  if isinstance(value, set):
    return sorted((cpu_tree(item, omit=omit) for item in value), key=repr)
  if isinstance(value, np.ndarray):
    return value.copy()
  if isinstance(value, np.generic):
    return value.item()
  if isinstance(value, Path):
    return str(value)
  if value is None or isinstance(value, (str, int, float, bool, bytes)):
    return value
  return repr(value)


def tensor_sha256(value) -> str:
  """Hash a tensor's dtype, shape and contiguous CPU bytes."""
  import torch

  if not torch.is_tensor(value):
    raise TypeError(f"expected torch.Tensor, got {type(value).__name__}")
  tensor = value.detach().cpu().contiguous()
  digest = hashlib.sha256()
  digest.update(str(tensor.dtype).encode())
  digest.update(repr(tuple(tensor.shape)).encode())
  digest.update(tensor.numpy().tobytes())
  return digest.hexdigest()


def tree_equal(left: Any, right: Any, path: str = "root") -> list[str]:
  """Return exact structural/tensor differences; no tolerances are applied."""
  import torch
  import numpy as np

  if torch.is_tensor(left) or torch.is_tensor(right):
    if not (torch.is_tensor(left) and torch.is_tensor(right)):
      return [path]
    if left.dtype != right.dtype or tuple(left.shape) != tuple(right.shape):
      return [path]
    return [] if torch.equal(left.detach().cpu(), right.detach().cpu()) else [path]
  if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
    if not (isinstance(left, np.ndarray) and isinstance(right, np.ndarray)):
      return [path]
    if left.dtype != right.dtype or left.shape != right.shape:
      return [path]
    return [] if np.array_equal(left, right, equal_nan=True) else [path]
  if dataclasses.is_dataclass(left) or dataclasses.is_dataclass(right):
    if not (
      dataclasses.is_dataclass(left)
      and dataclasses.is_dataclass(right)
      and not isinstance(left, type)
      and not isinstance(right, type)
      and type(left) is type(right)
    ):
      return [path]
    return [
      diff
      for field in dataclasses.fields(left)
      for diff in tree_equal(
        getattr(left, field.name), getattr(right, field.name),
        f"{path}.{field.name}",
      )
    ]
  if type(left) is not type(right):
    return [path]
  if isinstance(left, Mapping):
    if set(left) != set(right):
      return [f"{path}.keys"]
    return [diff for key in sorted(left, key=str)
            for diff in tree_equal(left[key], right[key], f"{path}.{key}")]
  if isinstance(left, (tuple, list)):
    if len(left) != len(right):
      return [f"{path}.length"]
    return [diff for index, (a, b) in enumerate(zip(left, right))
            for diff in tree_equal(a, b, f"{path}[{index}]")]
  return [] if left == right else [path]


def live_window_snapshot(window) -> dict:
  """Copy all persistent scan-ring arrays, including source provenance."""
  fields = (
    "_episode_ids", "_near_points", "_near_valid", "_near_filled",
    "_near_scan_ids", "_near_times", "_far_points", "_far_valid",
    "_far_scan_ids", "_far_times", "_far_count", "_far_pos",
    "_last_scan_id", "_last_time", "_near_angle_keys", "_far_angle_keys",
    "_angle_keys_present", "_near_source_dirs_sensor", "_far_source_dirs_sensor",
    "_near_source_dirs_present", "_far_source_dirs_present", "_near_source_indices",
    "_far_source_indices", "_source_dirs_attached",
  )
  arrays = {}
  for name in fields:
    if not hasattr(window, name):
      raise AttributeError(f"live scan window is missing required field {name}")
    value = getattr(window, name)
    if value is None or (hasattr(value, "numel") and value.numel() == 0):
      raise ValueError(f"live scan window field {name} is empty")
    arrays[name] = cpu_tree(value)
  if not hasattr(window, "_seen_episodes"):
    raise AttributeError("live scan window is missing _seen_episodes")
  seen = window._seen_episodes
  if not seen or any(not episode_set for episode_set in seen):
    raise ValueError("live scan window has no retained episode identities")
  return {"fields": arrays, "seen_episodes": cpu_tree(seen)}


def persistent_physics_snapshot(env) -> dict:
  """Use the checkpoint's live manifest to capture every nonempty physics field."""
  from hexapod_mjlab.tasks.confined_locomotion.rl.checkpoint import _physics_arrays

  arrays, manifest = _physics_arrays(env)
  return {"arrays": arrays, "manifest": cpu_tree(manifest)}


def runtime_snapshot(runtime) -> dict:
  """Capture physical and row-local state relevant to selected-reset isolation."""
  env = runtime.env
  action = env.action_manager.get_term("residual_pd")
  foot_air = getattr(env, "_confined_foot_air_state", None)
  termination = getattr(env, "_confined_termination_state", None)
  robot = env.scene["robot"]
  derived_names = (
    "root_link_pos_w", "root_link_quat_w", "root_link_lin_vel_w",
    "root_link_ang_vel_w", "root_link_lin_vel_b", "root_link_ang_vel_b",
    "projected_gravity_b", "joint_pos", "joint_vel", "body_link_pos_w",
    "body_link_lin_vel_w",
  )
  result = {
    "physics": persistent_physics_snapshot(env),
    "episode_ids": cpu_tree(runtime.episode_ids),
    "step_in_episode": cpu_tree(runtime.step_in_episode),
    "scan_counters": cpu_tree(runtime.scan_counters),
    "sensor_scan_counts": cpu_tree(runtime.sensor_scan_counts),
    "next_due_scan_step": cpu_tree(runtime.next_due_scan_step),
    "control_steps": int(runtime.control_steps_taken),
    "common_step_counter": int(env.common_step_counter),
    "sim_step_counter": int(env._sim_step_counter),
    "episode_length_buf": cpu_tree(env.episode_length_buf),
    "manual_reset_pending": cpu_tree(env._manual_reset_pending),
    "commands": cpu_tree(runtime._commands.state_dict()),
    "perturbations": (
      None if runtime._perturbations is None
      else cpu_tree(runtime._perturbations.state_dict())
    ),
    "action_state": cpu_tree({
      "prev_action": action.prev_action,
      "last_raw_action": action.last_raw_action,
      "last_clipped_action": action.last_clipped_action,
      "last_target": action.last_target,
      "processed_actions": action._processed_actions,
      "raw_actions": action._raw_actions,
      "control_steps_seen": action.control_steps_seen,
    }),
    "foot_air_state": None if foot_air is None else cpu_tree(vars(foot_air)),
    "termination_state": None if termination is None else cpu_tree({
      name: getattr(termination, name)
      for name in (
        "low_height_count", "tilt_count", "low_height_done", "tilt_done",
        "flip_done", "any_fall", "control_id",
      )
      if hasattr(termination, name)
    }),
    "derived_robot_state": cpu_tree({
      name: getattr(robot.data, name)
      for name in derived_names
      if hasattr(robot.data, name)
    }),
    "window": live_window_snapshot(runtime.window),
    "window_read": cpu_tree(runtime.window.read()),
    "window_source_directions": cpu_tree(runtime.window.read_source_directions()),
    "ground_cache": cpu_tree(runtime._ground_cache),
    "input_cache_identity": getattr(runtime._input_cache, "_identity", None),
  }
  for name in ("reset_terminated", "reset_time_outs", "reset_buf", "reward_buf"):
    value = getattr(env, name, None)
    if value is not None:
      result[name] = cpu_tree(value)
  return result


def global_prng_snapshot(*, device=None, provider=None, runtime=None, trainer=None) -> dict:
  """Capture every live generator relevant to collection/update continuation."""
  import numpy as np
  import torch

  state = {
    "python": random.getstate(),
    "numpy": np.random.get_state(),
    "torch_cpu": torch.get_rng_state().clone(),
    "torch_cuda": None,
  }
  if device is not None and torch.device(device).type == "cuda":
    state["torch_cuda"] = torch.cuda.get_rng_state(device).clone()
  if provider is not None:
    state["action_generator"] = cpu_tree(provider.generator_state())
  if runtime is not None:
    state["command_generators"] = cpu_tree(runtime._commands.state_dict())
    perturb = runtime._perturbations
    state["perturbation_generators"] = None if perturb is None else cpu_tree(perturb.state_dict())
  if trainer is not None:
    state["optimization_shuffle"] = cpu_tree(trainer.shuffle_generator_state())
  return state

