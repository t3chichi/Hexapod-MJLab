"""Shared checkpoint state helpers for the formal locomotion tasks."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import torch
import warp as wp

from .transition import physics_field_manifest, require_finite


class CheckpointFault(RuntimeError):
  """Save/apply failure after validation (the runner is frozen, not usable)."""


#: Persistent Data fields checked BEYOND the frozen P0 tuple that can affect
#: the next integration in this mujoco_warp version (actuator activation
#: history / plugin state). Recorded in the manifest either way; saved and
#: restored in place whenever actually non-empty.
EXTRA_PERSISTENT_PHYSICS_FIELDS = ("history", "plugin_state")


def compiled_model_identity(env) -> dict:
  """Structure digest of the ACTUAL compiled model (order/gear/options)."""
  import mujoco

  mm = env.sim.mj_model
  joint_names = [str(mm.joint(i).name) for i in range(mm.njnt)]
  actuator_names = [
    mujoco.mj_id2name(mm, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"actuator{i}"
    for i in range(mm.nu)
  ]
  action = env.action_manager.get_term("residual_pd")
  payload = {
    "dims": {
      "nq": int(mm.nq),
      "nv": int(mm.nv),
      "nu": int(mm.nu),
      "na": int(mm.na),
      "nbody": int(mm.nbody),
      "ngeom": int(mm.ngeom),
      "njnt": int(mm.njnt),
      "nsensordata": int(mm.nsensordata),
      "nmocap": int(mm.nmocap),
      "neq": int(mm.neq),
      "nuserdata": int(mm.nuserdata),
    },
    "joint_names": joint_names,
    "actuator_names": actuator_names,
    "actuator_gear": np.asarray(mm.actuator_gear)[:, 0].tolist(),
    "q_ref": action.q_ref.detach().cpu().tolist(),
    "options": {
      "timestep": float(mm.opt.timestep),
      "integrator": int(mm.opt.integrator),
      "solver": int(mm.opt.solver),
      "iterations": int(mm.opt.iterations),
      "ls_iterations": int(mm.opt.ls_iterations),
      "impratio": float(mm.opt.impratio),
      "tolerance": float(mm.opt.tolerance),
      "jacobian": int(mm.opt.jacobian),
      "disableflags": int(mm.opt.disableflags),
      "enableflags": int(mm.opt.enableflags),
    },
  }
  binary = np.empty(mujoco.mj_sizeModel(mm), dtype=np.uint8)
  mujoco.mj_saveModel(mm, None, binary)
  payload["compiled_binary_sha256"] = hashlib.sha256(binary.tobytes()).hexdigest()
  blob = json.dumps(payload, sort_keys=True, allow_nan=False).encode("utf-8")
  return {**payload, "digest": hashlib.sha256(blob).hexdigest()}


def manager_tensors(env) -> dict:
  result = {}
  for name in ("_action", "_prev_action", "_prev_prev_action"):
    result["action_manager." + name] = getattr(env.action_manager, name)
  for name in ("_reward_buf", "_step_reward"):
    result["reward_manager." + name] = getattr(env.reward_manager, name)
  for name, tensor in env.reward_manager._episode_sums.items():
    result["reward_sum." + name] = tensor
  for name in ("_truncated_buf", "_terminated_buf"):
    result["termination_manager." + name] = getattr(env.termination_manager, name)
  for name, tensor in env.termination_manager._term_dones.items():
    result["termination_done." + name] = tensor
  for field in ("_reset_term_last_triggered_step_id", "_reset_term_last_triggered_once", "_interval_term_time_left"):
    for index, tensor in enumerate(getattr(env.event_manager, field, [])):
      result[f"event_manager.{field}.{index}"] = tensor
  robot = env.scene["robot"]
  result["entity.joint_effort_target"] = robot.data.joint_effort_target
  sensor = env.scene["feet_any_contact"]
  if sensor._history_state is not None:
    for name, tensor in sensor._history_state.items():
      result["sensor_history." + name] = tensor
  if sensor._air_time_state is not None:
    for name in ("current_air_time", "last_air_time", "current_contact_time", "last_contact_time"):
      result["sensor_air." + name] = getattr(sensor._air_time_state, name)
  return result


def physics_arrays(env) -> tuple[dict, dict]:
  """(non-empty CPU arrays, manifest incl. the extra checked fields)."""
  arrays: dict[str, torch.Tensor] = {}
  manifest: dict[str, dict] = dict(physics_field_manifest(env))
  for name in EXTRA_PERSISTENT_PHYSICS_FIELDS:
    array = getattr(env.sim.wp_data, name, None)
    if array is None:
      manifest[name] = {"shape": [0], "dtype": "absent", "ptr": 0, "empty": True}
      continue
    manifest[name] = {
      "shape": list(array.shape),
      "dtype": str(array.dtype),
      "ptr": int(array.ptr or 0),
      "empty": not array.size,
    }
    if array.size:
      arrays[name] = wp.to_torch(array).detach().cpu().clone()
  for name, entry in manifest.items():
    if not entry["empty"]:
      tensor = wp.to_torch(getattr(env.sim.wp_data, name))
      arrays[name] = tensor.detach().cpu().clone()
      entry["tensor_shape"] = list(tensor.shape)
      entry["tensor_dtype"] = str(tensor.dtype)
    if entry["empty"] and name in arrays:
      raise CheckpointFault(f"physics field {name} manifest/size mismatch")
  return arrays, manifest


def write_persistent_fields(env, arrays: dict, stage: str) -> None:
  for name, tensor in arrays.items():
    target = wp.to_torch(getattr(env.sim.wp_data, name))
    candidate = tensor.detach().to(device=target.device, dtype=target.dtype)
    require_finite(f"physics/{stage}/{name}", candidate)
    with torch.no_grad():
      target.copy_(candidate)
