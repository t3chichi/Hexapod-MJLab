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


MODEL_PHYSICS_SCHEMA = "hexapod_mjlab.model_physics/v1"

# Compiled, nominal physical inputs, not compiler caches, file paths or visual
# materials. Keep this list explicit: adding/changing its meaning requires a
# schema change. Live per-env domain randomization belongs to checkpoint state,
# not this signature of the host MjModel used to construct the simulation.
_MODEL_PHYSICS_FIELDS = tuple("""
  qpos0 qpos_spring
  body_parentid body_mocapid body_pos body_quat body_ipos body_iquat
  body_mass body_inertia body_gravcomp
  jnt_type jnt_bodyid jnt_qposadr jnt_dofadr jnt_pos jnt_axis jnt_limited
  jnt_range jnt_margin jnt_stiffness jnt_stiffnesspoly jnt_solref jnt_solimp
  jnt_actfrclimited jnt_actfrcrange jnt_actgravcomp
  dof_bodyid dof_jntid dof_parentid dof_armature dof_damping dof_dampingpoly
  dof_frictionloss dof_solref dof_solimp
  geom_type geom_bodyid geom_dataid geom_pos geom_quat geom_size
  geom_contype geom_conaffinity geom_condim geom_priority geom_friction
  geom_solmix geom_solref geom_solimp geom_margin geom_gap geom_fluid
  geom_adhesion geom_surfacevel
  site_bodyid site_pos site_quat
  pair_geom1 pair_geom2 pair_dim pair_friction pair_solref pair_solreffriction
  pair_solimp pair_margin pair_gap pair_adhesion exclude_signature
  eq_type eq_obj1id eq_obj2id eq_objtype eq_active0 eq_solref eq_solimp eq_data
  tendon_adr tendon_num tendon_limited tendon_range tendon_margin
  tendon_stiffness tendon_stiffnesspoly tendon_damping tendon_dampingpoly
  tendon_armature tendon_frictionloss tendon_lengthspring
  tendon_solref_lim tendon_solimp_lim tendon_solref_fri tendon_solimp_fri
  tendon_actfrclimited tendon_actfrcrange wrap_type wrap_objid wrap_prm
  actuator_trntype actuator_trnid actuator_dyntype actuator_dynprm
  actuator_gaintype actuator_gainprm actuator_biastype actuator_biasprm
  actuator_actearly actuator_actlimited actuator_actrange actuator_actadr
  actuator_actnum actuator_ctrllimited actuator_ctrlrange actuator_ctrladr
  actuator_ctrlnum actuator_ctrlspec actuator_forcelimited actuator_forcerange
  actuator_gear actuator_cranklength actuator_lengthrange actuator_group
  actuator_armature actuator_damping actuator_dampingpoly actuator_delay
  actuator_history actuator_historyadr
  mesh_vertadr mesh_vertnum mesh_faceadr mesh_facenum mesh_vert mesh_face
  mesh_pos mesh_quat mesh_scale
  hfield_adr hfield_nrow hfield_ncol hfield_size hfield_data
""".split())

_MODEL_OPTION_FIELDS = tuple("""
  timestep gravity wind magnetic density viscosity integrator solver cone
  jacobian iterations tolerance ls_iterations ls_tolerance noslip_iterations
  noslip_tolerance ccd_iterations ccd_tolerance impratio disableflags
  enableflags disableactuator o_margin o_solref o_solimp o_friction
  sdf_initpoints sdf_iterations sleep_tolerance
""".split())


def _numeric_signature(value) -> dict:
  """Canonical numeric content, independent of native dtype and byte order."""
  array = np.asarray(value)
  if array.dtype.kind in "biu":
    kind, canonical = "integer", array.astype("<i8", copy=True)
  elif array.dtype.kind == "f":
    kind, canonical = "float", array.astype("<f8", copy=True)
    if np.isnan(canonical).any():
      raise ValueError("physical model contains NaN")
    canonical[canonical == 0] = 0  # +0 and -0 have the same physical meaning.
  else:
    raise ValueError(f"unsupported physical model dtype: {array.dtype}")
  return {"shape": list(array.shape), "kind": kind,
          "sha256": hashlib.sha256(canonical.tobytes(order="C")).hexdigest()}


def model_physics_identity(model) -> dict:
  """Fingerprint nominal rigid-body dynamics, including collision geometry.

  Unsupported physical features fail closed instead of receiving a misleading
  compatibility guarantee. EL4090 tasks use neither flexes nor plugins.
  """
  if model.nflex or model.nplugin:
    raise ValueError("physical identity does not support flex/plugin models")
  fields = {name: _numeric_signature(getattr(model, name))
            for name in _MODEL_PHYSICS_FIELDS}
  fields.update({"opt." + name: _numeric_signature(getattr(model.opt, name))
                 for name in _MODEL_OPTION_FIELDS})
  return {"schema": MODEL_PHYSICS_SCHEMA, "fields": fields}


def compare_model_physics(current, saved) -> None:
  """Validate physical compatibility before any live training state is applied.

  Legacy files without a physical signature remain weights-only inputs. A
  source or binary digest cannot prove their physical semantics portably.
  """
  for label, value in (("current", current), ("checkpoint", saved)):
    if (not isinstance(value, dict) or value.get("schema") != MODEL_PHYSICS_SCHEMA
        or not isinstance(value.get("fields"), dict)):
      raise ValueError(
        f"{label} lacks a supported physical model identity; full resume cannot "
        "verify dynamics. Use weights-only Play; it is not full resume."
      )
  if current["fields"].keys() != saved["fields"].keys():
    raise ValueError("physical model identity field set differs")
  if current.get("restored_fields", []) != saved.get("restored_fields", []):
    raise ValueError("physical model restored-field contract differs")
  changed = [name for name in current["fields"]
             if current["fields"][name] != saved["fields"][name]]
  if changed:
    raise ValueError(f"physical model differs on {', '.join(changed)}")


def compiled_model_identity(env, *, restored_model_fields=()) -> dict:
  """Structure digest of the ACTUAL compiled model (order/gear/options)."""
  import mujoco

  mm = env.sim.mj_model
  physics = model_physics_identity(mm)
  # Also cover actual Warp model overrides. Only explicitly checkpointed model
  # state (Tripod DR) may differ live; its nominal host values remain enforced.
  physics["restored_fields"] = sorted(restored_model_fields)
  for name in _MODEL_PHYSICS_FIELDS:
    array = getattr(env.sim.wp_model, name, None)
    if isinstance(array, wp.array) and name not in restored_model_fields:
      physics["fields"]["live." + name] = _numeric_signature(
        wp.to_torch(array).detach().cpu().numpy()
      )
  for name in _MODEL_OPTION_FIELDS:
    value = getattr(env.sim.wp_model.opt, name, None)
    if value is not None:
      if isinstance(value, wp.array):
        value = wp.to_torch(value).detach().cpu().numpy()
      physics["fields"]["live.opt." + name] = _numeric_signature(value)
  joint_names = [str(mm.joint(i).name) for i in range(mm.njnt)]
  actuator_names = [
    mujoco.mj_id2name(mm, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"actuator{i}"
    for i in range(mm.nu)
  ]
  action = env.action_manager.get_term("residual_pd")
  payload = {
    "physics": physics,
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
