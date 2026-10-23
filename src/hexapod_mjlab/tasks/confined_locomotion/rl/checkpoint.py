"""Strict full checkpoint save/resume for confined-locomotion runs.

Three STRICTLY separated modes (authority: 首版训练实施计划.md §4.6):

1. **Full resume** (``save_full_checkpoint``/``load_full_checkpoint``): the
   complete continuation state of one stable control boundary — network
   heads/buffers + 18 log_std, the single Adam (step/exp_avg/exp_avg_sq),
   LR/counters, the optimization-shuffle PRNG, ALL global PRNG streams
   (Python/NumPy/Torch CPU/CUDA) plus the named task streams (action
   generator, per-env command generators), the REAL Warp persistent physics
   fields (P0 ``PERSISTENT_PHYSICS_FIELDS`` + the additionally checked
   ``history``/``plugin_state``), the scan window ring (v2: including the
   P6 per-row angle-key metadata), ground cache, the
   ControlInputCache/``_current_window_snapshot`` first-action inputs,
   env/action-manager/foot-air/termination lifecycle state, the runner
   counters and — v2 (P6) — the observation-perturbation continuation state
   (per-env named gap/range/proprio generators, persistent gap schedule
   phase/next-event/scheduled-ε, per-row proprio noise cache, draw counters)
   whenever the run is perturbed; a clean run saves an explicit None and
   never resumes into a perturbed runtime (the noise recipe is part of the
   identity). Same CUDA device/model/config/env-count/grid/source/perturbation-
   recipe only; a source, dependency, grid (same M with moved positions
   included), model, device, env-count or noise-recipe mismatch is REJECTED —
   never silently relaxed into a "cross-device equivalent" resume. v1 (P5)
   files are rejected by the schema outright.
2. **Weights only** (``save_weights_only``/``load_weights``): the shared
   network + four heads + all buffers and the OPTIONAL 18-dim log_std. No
   Adam/physics/RNG/lifecycle is touched and NOTHING about continuation is
   claimed (``resumable=False``). The env count may differ; the grid
   positions/sizes and the full key set are still strictly matched (not just
   the actor head). External source changes are only accepted behind the
   explicit independent rule ``allow_source_change=True`` — never for full
   resume.
3. Fault/stop scenes and engineering artifacts stay NON-RESUMABLE
   diagnostics; containing an Adam state never makes them a full resume.

Stable save boundary: a COMPLETED 24-step collect + 20-update iteration has
finished (counters verified), the rollout buffer is consumed/empty
(``_active_buffer is None``), the transition context is idle, no terminal row
is pending and the next action's inputs were already produced by the last
transition's final cache (``prepare`` re-reads are identity-keyed, so saving
performs NO reset, NO rescan and NO random draws). Save is atomic (same-dir
tmp + ``os.replace``) and REFUSES to overwrite an existing file.

Load validates EVERYTHING first (schema/kind, full identity, payload key
completeness, per-tensor shape/dtype/finiteness, PRNG structure) — a failed
validation mutates NOTHING. Only then is the state applied in a fixed order:
weights → Adam → physics in-place (``wp.to_torch`` alias ``copy_`` into the
original Warp arrays, preserving CUDA-graph storage; one ``sim.forward`` to
rebuild derived state/caches followed by a re-write of the persistent fields
that forward clobbers, e.g. the solver warmstart; contact-sensor cache
invalidation; overflow check) → env/action/foot/termination state → runtime
counters/window/ground/input cache → commands → action generator → ALL
remaining PRNGs LAST (nothing between the PRNG restore and the first action
consumes randomness). An apply-time failure freezes the runner
(``_lifecycle_phase='faulted'``) after saving feasible diagnostics; it never
continues half-applied. Construction-time side effects of a fresh official
wrapper (initial reset, ONE takeover first scan, reset-time command sampling)
are RECORDED at load entry and then fully overwritten by the saved state —
load itself performs 0 resets, 0 scans and 0 physics steps (one forward is
the documented derived-state rebuild).

Bounded diagnostics (``runtime.transitions``/``scan_log``,
``pipeline.scan_records``) affect no future behavior and are deliberately NOT
serialized; their presence in the payload is a recorded count only. The
runner's natural-fall/fault save+stop rules and the one-shot P4 artifact
export are unchanged; no periodic auto-save is introduced here.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import warp as wp

from hexapod_mjlab.rl.confined_locomotion.network import ConfinedLocomotionNetwork
from hexapod_mjlab.rl.confined_locomotion.ppo import (
  NUM_EPOCHS,
  NUM_MINIBATCHES,
)
from hexapod_mjlab.tasks.confined_locomotion.scanner import ConfinedLidarCfg

from ..flat_env_cfg import TASK_ID, DECIMATION, SIM_TIMESTEP_S, _ROBOT_XML
from ..mdp.rewards import CONTROL_DT_S, TRAINING_REWARD_WEIGHTS
from ..mdp.terminations import (
  FLIP_THRESHOLD_COS,
  LOW_HEIGHT_PERSISTENCE_STEPS,
  LOW_HEIGHT_THRESHOLD_M,
  TILT_THRESHOLD_COS,
  TRAINING_TERMINATION_TIME_OUT_FLAGS,
)
from ..transition import physics_field_manifest, require_finite

__all__ = [
  "CHECKPOINT_KIND_FULL",
  "CHECKPOINT_KIND_WEIGHTS",
  "CHECKPOINT_SCHEMA",
  "CheckpointError",
  "CheckpointFault",
  "EXTRA_PERSISTENT_PHYSICS_FIELDS",
  "load_full_checkpoint",
  "load_weights",
  "save_full_checkpoint",
  "save_weights_only",
  "validate_payload_envelope",
]

#: Schema stamp; v3 (P8) binds typed symmetry on/off + named joint/grid/mount
#: identity and source-ray provenance in window/prepared snapshots. v2 (P6)
#: and earlier files are rejected rather than partially resumed.
CHECKPOINT_SCHEMA = "confined_locomotion.checkpoint/v3"

#: The two payload kinds; fault/stop scenes and engineering artifacts remain
#: external diagnostics and are NEVER checkpoints.
CHECKPOINT_KIND_FULL = "confined_locomotion_full_resume"
CHECKPOINT_KIND_WEIGHTS = "confined_locomotion_weights_only"

#: Persistent Data fields checked BEYOND the frozen P0 tuple that can affect
#: the next integration in this mujoco_warp version (actuator activation
#: history / plugin state). Recorded in the manifest either way; saved and
#: restored in place whenever actually non-empty.
EXTRA_PERSISTENT_PHYSICS_FIELDS = ("history", "plugin_state")

#: Production source files hashed into the checkpoint identity: the runtime,
#: MDP, scanning, geometry, policy, PPO, and official-entry dependency closure
#: plus the compiled robot MJCF; never logs, tests, or docs.
_SOURCE_MODULE_NAMES = (
  "hexapod_mjlab.tasks.confined_locomotion",
  "hexapod_mjlab.tasks.confined_locomotion.rl",
  "hexapod_mjlab.tasks.confined_locomotion.mdp",
  "hexapod_mjlab.rl.confined_locomotion",
  "hexapod_mjlab.rl.confined_locomotion.inference",
  "hexapod_mjlab.tasks.confined_locomotion.rl.entry",
  "hexapod_mjlab.tasks.confined_locomotion.perturbations",
  "hexapod_mjlab.tasks.confined_locomotion.runtime",
  "hexapod_mjlab.tasks.confined_locomotion.collector",
  "hexapod_mjlab.tasks.confined_locomotion.transition",
  "hexapod_mjlab.tasks.confined_locomotion.diagnostics",
  "hexapod_mjlab.tasks.confined_locomotion.commands",
  "hexapod_mjlab.tasks.confined_locomotion.observation",
  "hexapod_mjlab.tasks.confined_locomotion.scanner",
  "hexapod_mjlab.tasks.confined_locomotion.ground",
  "hexapod_mjlab.tasks.confined_locomotion.rl.checkpoint",
  "hexapod_mjlab.tasks.confined_locomotion.rl.runner",
  "hexapod_mjlab.tasks.confined_locomotion.rl_cfg",
  "hexapod_mjlab.tasks.confined_locomotion.flat_env_cfg",
  "hexapod_mjlab.tasks.confined_locomotion.mdp.events",
  "hexapod_mjlab.tasks.confined_locomotion.mdp.observations",
  "hexapod_mjlab.tasks.confined_locomotion.mdp.rewards",
  "hexapod_mjlab.tasks.confined_locomotion.mdp.terminations",
  "hexapod_mjlab.tasks.confined_locomotion.mdp.actions",
  "hexapod_mjlab.tasks.el_4090_lidar",
  "hexapod_mjlab.tasks.el_4090_lidar.t0",
  "hexapod_mjlab.tasks.el_4090_lidar.t0.witness",
  "hexapod_mjlab.tasks.el_4090_lidar.mdp",
  "hexapod_mjlab.tasks.el_4090_lidar.mdp.lidar",
  "hexapod_mjlab.utils.lidar",
  "hexapod_mjlab.utils.lidar.lidar_sensor",
  "hexapod_mjlab.utils.lidar.base_sensor",
  "hexapod_mjlab.utils.lidar.sensor_config",
  "hexapod_mjlab.utils.lidar.sensor_config.lidar_sensor_config",
  "hexapod_mjlab.utils.lidar.sensor_config.base_sensor_config",
  "hexapod_mjlab.utils.lidar.sensor_pattern",
  "hexapod_mjlab.utils.lidar.sensor_pattern.sensor_lidar",
  "hexapod_mjlab.utils.lidar.sensor_pattern.sensor_lidar.genera_lidar_scan_pattern",
  "hexapod_mjlab.utils.lidar.sensor_kernels",
  "hexapod_mjlab.utils.lidar.sensor_kernels.lidar_kernels_warp",
  "hexapod_mjlab.rl.confined_locomotion.network",
  "hexapod_mjlab.rl.confined_locomotion.policy",
  "hexapod_mjlab.rl.confined_locomotion.distribution",
  "hexapod_mjlab.rl.confined_locomotion.ppo",
  "hexapod_mjlab.rl.confined_locomotion.losses",
  "hexapod_mjlab.rl.confined_locomotion.training_rollout",
  "hexapod_mjlab.rl.confined_locomotion.contacts",
  "hexapod_mjlab.rl.confined_locomotion.contracts",
  "hexapod_mjlab.rl.confined_locomotion.integration",
  "hexapod_mjlab.rl.confined_locomotion.control",
  "hexapod_mjlab.rl.confined_locomotion.mapping",
  "hexapod_mjlab.rl.confined_locomotion.history",
  "hexapod_mjlab.rl.confined_locomotion.symmetry",
)

_JOINT_UPDATE_STEPS_PER_ROLLOUT = NUM_EPOCHS * NUM_MINIBATCHES  # 5×4 = 20

_PREPARED_OBSERVATION_FIELDS = (
  "env_ids",
  "episode_ids",
  "step_ids",
  "time_s",
  "lin_vel_b",
  "ang_vel_b",
  "gravity_b",
  "joint_pos_rel_ref",
  "joint_vel",
  "prev_action",
  "command",
)

_WINDOW_TENSOR_FIELDS = (
  "episode_ids",
  "near_points",
  "near_valid",
  "near_filled",
  "near_scan_ids",
  "near_times",
  "far_points",
  "far_valid",
  "far_scan_ids",
  "far_times",
  "far_count",
  "far_pos",
  "last_scan_id",
  "last_time",
  "source_dirs_attached",
  "near_source_dirs_sensor",
  "far_source_dirs_sensor",
  "near_source_dirs_present",
  "far_source_dirs_present",
  "near_source_indices",
  "far_source_indices",
)

_SNAPSHOT_TENSOR_FIELDS = (
  "env_ids",
  "episode_ids",
  "near_points",
  "near_valid",
  "near_filled",
  "near_scan_ids",
  "near_times",
  "far_points",
  "far_valid",
  "far_filled",
  "far_scan_ids",
  "far_times",
)

_GROUND_CACHE_TENSOR_FIELDS = (
  "heights",
  "positions",
  "valid",
  "world_points",
  "base_pos_w",
  "base_quat_wxyz",
)


class CheckpointError(RuntimeError):
  """Checkpoint contract violation / incompatibility (nothing was mutated)."""


class CheckpointFault(RuntimeError):
  """Save/apply failure after validation (the runner is frozen, not usable)."""


# ═══════════════════════════ identity builders ═══════════════════════════


def _sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1 << 20), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _source_identity() -> dict:
  import importlib

  files: dict[str, str] = {}
  for module_name in _SOURCE_MODULE_NAMES:
    module = importlib.import_module(module_name)
    source = Path(getattr(module, "__file__", ""))
    if not source.is_file():
      raise CheckpointFault(f"source file missing for {module_name}: {source}")
    files[module_name] = _sha256_file(source)
  if not _ROBOT_XML.is_file():
    raise CheckpointFault(f"robot MJCF asset missing: {_ROBOT_XML}")
  files[_ROBOT_XML.name] = _sha256_file(_ROBOT_XML)
  return files


def _dependency_versions() -> dict:
  import mjlab
  import mujoco

  from importlib.metadata import version
  return {
    "rsl_rl": version("rsl-rl-lib"),
    "mujoco_warp": version("mujoco-warp"),
    "torch": str(torch.__version__),
    "mjlab": version("mjlab"),
    "mujoco": str(mujoco.__version__),
    "warp": str(wp.config.version),
    "numpy": str(np.__version__),
    "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
  }


def _compiled_model_identity(env) -> dict:
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


def _lidar_identity(lidar_cfg: ConfinedLidarCfg) -> dict:
  import dataclasses

  return {
    field.name: (
      list(field_value) if isinstance(field_value := getattr(lidar_cfg, field.name), tuple)
      else field_value
    )
    for field in dataclasses.fields(ConfinedLidarCfg)
  }


def _torch_precision_record() -> dict:
  matmul = getattr(torch.backends.cuda.matmul, "fp32_precision", None)
  cudnn = getattr(torch.backends.cudnn, "fp32_precision", None)
  rnn = getattr(getattr(torch.backends.cudnn, "rnn", None), "fp32_precision", None)
  if matmul is None:
    matmul = "allow_tf32=True" if torch.backends.cuda.matmul.allow_tf32 else "allow_tf32=False"
  if cudnn is None:
    cudnn = "allow_tf32=True" if torch.backends.cudnn.allow_tf32 else "allow_tf32=False"
  return {
    "cuda_matmul_fp32_precision": str(matmul),
    "cudnn_fp32_precision": str(cudnn),
    "cudnn_rnn_fp32_precision": str(cudnn if rnn is None else rnn),
    "tf32_disabled": all(
      str(value) in {"ieee", "allow_tf32=False"}
      for value in (matmul, cudnn, cudnn if rnn is None else rnn)
    ),
    "default_dtype": str(torch.get_default_dtype()),
  }


def _comparable_identity(runner) -> dict:
  runtime = runner.runtime
  env = runtime.env
  device = torch.device(runner.device)
  if device.type != "cuda" or torch.cuda.device_count() == 0:
    raise CheckpointError(
      f"full checkpoint identity requires the actual CUDA device, got {device}"
    )
  return {
    "schema": CHECKPOINT_SCHEMA,
    "kind": CHECKPOINT_KIND_FULL,
    "resumable": True,
    "task": TASK_ID,
    "runner_identity": getattr(
      runner,
      "runner_identity",
      "El-4090-Confined-Locomotion-Flat/ConfinedLocomotionRunner/mode=engineering",
    ),
    "source_sha256": _source_identity(),
    "dependencies": _dependency_versions(),
    "device": {
      "type": "cuda",
      "index": int(device.index if device.index is not None else torch.cuda.current_device()),
      "name": torch.cuda.get_device_name(device),
    },
    "torch_precision": _torch_precision_record(),
    "num_envs": int(runtime.num_envs),
    "env_seed": int(env.cfg.seed) if env.cfg.seed is not None else None,
    "runtime": {"mode": str(runtime.mode), "seed": int(runtime.seed)},
    "grid": {
      "identity": runtime.grid_cfg.identity(),
      "identity_hash": runtime.grid_cfg.identity_hash(),
    },
    "model": _compiled_model_identity(env),
    "sim": {
      "timestep": float(env.cfg.sim.mujoco.timestep),
      "decimation": int(env.cfg.decimation),
      "step_dt": float(env.step_dt),
      "control_dt": float(CONTROL_DT_S),
      "sim_timestep_constant": float(SIM_TIMESTEP_S),
      "decimation_constant": int(DECIMATION),
      "episode_length_s": float(env.cfg.episode_length_s),
      "max_episode_length": int(env.max_episode_length),
      "auto_reset": bool(env.cfg.auto_reset),
    },
    "rewards": {"weights": {k: float(v) for k, v in TRAINING_REWARD_WEIGHTS.items()}},
    "terminations": {
      "time_out_flags": dict(TRAINING_TERMINATION_TIME_OUT_FLAGS),
      "low_height_threshold_m": float(LOW_HEIGHT_THRESHOLD_M),
      "low_height_persistence_steps": int(LOW_HEIGHT_PERSISTENCE_STEPS),
      "tilt_threshold_cos": float(TILT_THRESHOLD_COS),
      "flip_threshold_cos": float(FLIP_THRESHOLD_COS),
    },
    "lidar": _lidar_identity(runtime.lidar_cfg),
    "wrapper": {"clip_actions": None},
    "ppo_config": runner.trainer.config.summary(),
    "trainer_config": runner.trainer.config_identity(),
    "symmetry": runner.trainer.symmetry_identity,
    "distribution_identity_frozen": str(runner.policy_adapter.identity_frozen),
    "observation_perturbations": (
        {"mode": "clean"}
        if getattr(runtime, "perturbation_cfg", None) is None
        else runtime.perturbation_cfg.identity()
    ),
  }


# ═══════════════════════════ serialization helpers ═════════════════════════


def _to_cpu(value):
  if torch.is_tensor(value):
    return value.detach().cpu().clone()
  if isinstance(value, dict):
    return {key: _to_cpu(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    converted = [_to_cpu(item) for item in value]
    return type(value)(converted) if isinstance(value, tuple) else converted
  return value


def _on_device(value, device: torch.device):
  if torch.is_tensor(value):
    return value.detach().to(device=device)
  if isinstance(value, dict):
    return {key: _on_device(item, device) for key, item in value.items()}
  if isinstance(value, list):
    return [_on_device(item, device) for item in value]
  return value


def _serialize_prepared(prepared) -> dict:
  observation = prepared.observation
  payload = {
    "observation": {
      name: getattr(observation, name).detach().cpu().clone()
      for name in _PREPARED_OBSERVATION_FIELDS
    },
    **{
      name: getattr(prepared, name).detach().cpu().clone()
      for name in (
        "near_points",
        "near_valid",
        "far_points",
        "far_valid",
        "near_filled",
        "near_scan_ids",
        "near_times",
        "far_point_scan_ids",
        "far_point_times",
      )
    },
  }
  for name in (
    "near_angle_keys", "far_angle_keys",
    "near_source_dirs_sensor", "far_source_dirs_sensor",
    "near_source_dirs_present", "far_source_dirs_present",
    "near_source_indices", "far_source_indices",
  ):
    value = getattr(prepared, name)
    payload[name] = None if value is None else value.detach().cpu().clone()
  return payload


def _deserialize_prepared(state: dict, device: torch.device):
  from hexapod_mjlab.rl.confined_locomotion.contracts import (
    OnlineObservation,
  )
  from hexapod_mjlab.rl.confined_locomotion.integration import (
    PreparedControlInputs,
  )

  tensors = _on_device(state, device)
  observation = OnlineObservation(**tensors["observation"])
  return PreparedControlInputs(
    observation=observation,
    near_points=tensors["near_points"],
    near_valid=tensors["near_valid"],
    far_points=tensors["far_points"],
    far_valid=tensors["far_valid"],
    near_filled=tensors["near_filled"],
    near_scan_ids=tensors["near_scan_ids"],
    near_times=tensors["near_times"],
    far_point_scan_ids=tensors["far_point_scan_ids"],
    far_point_times=tensors["far_point_times"],
    near_angle_keys=tensors.get("near_angle_keys"),
    far_angle_keys=tensors.get("far_angle_keys"),
    near_source_dirs_sensor=tensors.get("near_source_dirs_sensor"),
    far_source_dirs_sensor=tensors.get("far_source_dirs_sensor"),
    near_source_dirs_present=tensors.get("near_source_dirs_present"),
    far_source_dirs_present=tensors.get("far_source_dirs_present"),
    near_source_indices=tensors.get("near_source_indices"),
    far_source_indices=tensors.get("far_source_indices"),
  )


def _serialize_snapshot(snapshot) -> dict:
  return {
    name: getattr(snapshot, name).detach().cpu().clone()
    for name in _SNAPSHOT_TENSOR_FIELDS
  }


def _deserialize_snapshot(state: dict, device: torch.device):
  from hexapod_mjlab.rl.confined_locomotion.contracts import (
    ScanWindowSnapshot,
  )

  return ScanWindowSnapshot(**_on_device(state, device))


def _serialize_window(window) -> dict:
  keys = window.read_angle_keys()
  return {
    **{
      name: getattr(window, f"_{name}").detach().cpu().clone()
      for name in _WINDOW_TENSOR_FIELDS
    },
    "seen_episodes": [sorted(int(v) for v in seen) for seen in window._seen_episodes],
    "angle_keys": (
        None
        if keys is None
        else {
            # Other window fields are raw ring storage; keys must use the
            # same storage order, not read_angle_keys()' chronological view.
            "near": window._near_angle_keys.detach().cpu().clone(),
            "far": window._far_angle_keys.detach().cpu().clone(),
            "present": window._angle_keys_present.detach().cpu().clone(),
        }
    ),
  }


def _serialize_ground_cache(runtime) -> list[dict | None]:
  entries: list[dict | None] = []
  for entry in runtime._ground_cache:
    if entry is None:
      entries.append(None)
      continue
    entries.append(
      {
        "env_row": int(entry["env_row"]),
        "scan_id": int(entry["scan_id"]),
        "episode_id": int(entry["episode_id"]),
        "persistent_scan_count": int(entry["persistent_scan_count"]),
        "time_s": float(entry["time_s"]),
        **{
          name: entry[name].detach().cpu().clone()
          for name in _GROUND_CACHE_TENSOR_FIELDS
        },
      }
    )
  return entries


def _validate_float_finiteness(tree, prefix: str = "") -> None:
  if torch.is_tensor(tree):
    if tree.is_floating_point() and tree.numel() and not bool(torch.isfinite(tree).all()):
      raise CheckpointError(f"payload tensor {prefix or '<root>'} contains NaN/Inf")
  elif isinstance(tree, dict):
    for key, value in tree.items():
      _validate_float_finiteness(value, f"{prefix}.{key}" if prefix else str(key))
  elif isinstance(tree, (list, tuple)):
    for index, value in enumerate(tree):
      _validate_float_finiteness(value, f"{prefix}[{index}]")
  elif isinstance(tree, float) and not math.isfinite(tree):
    raise CheckpointError(f"payload float {prefix or '<root>'} is not finite")


def _validate_payload_finiteness(payload: dict) -> None:
  """Blanket float finiteness with ONE documented exception.

  ``runtime.window.last_time`` legitimately holds ``-inf`` for rows with no
  completed scan yet (the F1 empty-window marker, mirrored by
  ``last_scan_id == EMPTY_SCAN_ID``); everything else must be finite.
  """
  from hexapod_mjlab.rl.confined_locomotion.contracts import EMPTY_SCAN_ID

  window = payload.get("runtime", {}).get("window", {})
  last_time = window.get("last_time")
  last_scan_id = window.get("last_scan_id")
  if torch.is_tensor(last_time):
    saved_last_time = last_time.clone()
    saved_last_scan = last_scan_id.clone() if torch.is_tensor(last_scan_id) else None
    window["last_time"] = last_time.clone().where(
      torch.isfinite(last_time), torch.zeros_like(last_time)
    )
    try:
      _validate_float_finiteness(payload)
    finally:
      window["last_time"] = saved_last_time
    inf_rows = (~torch.isfinite(saved_last_time)).nonzero(as_tuple=False).flatten()
    for row in inf_rows.tolist():
      if not bool(torch.isneginf(saved_last_time[row])) or saved_last_scan is None or int(saved_last_scan[row]) != EMPTY_SCAN_ID:
        raise CheckpointError(
          f"runtime.window.last_time[{row}] is -inf but last_scan_id is not "
          "EMPTY_SCAN_ID (inconsistent empty-window state)"
        )
  else:
    _validate_float_finiteness(payload)


def _require_same_tensors(name: str, saved, live: dict) -> None:
  if not isinstance(saved, dict) or set(saved) != set(live):
    raise CheckpointError(
      f"{name}: saved tensor key set != live "
      f"(missing {sorted(set(live) - set(saved))}, extra {sorted(set(saved) - set(live))})"
    )
  for key, live_tensor in live.items():
    candidate = saved[key]
    if not torch.is_tensor(live_tensor):
      if candidate != live_tensor:
        raise CheckpointError(f"{name}.{key}: non-tensor model state mismatch")
      continue
    if not torch.is_tensor(candidate):
      raise CheckpointError(f"{name}.{key}: expected a tensor, got {type(candidate).__name__}")
    if tuple(candidate.shape) != tuple(live_tensor.shape):
      raise CheckpointError(
        f"{name}.{key}: shape {tuple(candidate.shape)} != live {tuple(live_tensor.shape)}"
      )
    if key.endswith("_extra_state") and not torch.equal(candidate.cpu(), live_tensor.cpu()):
      raise CheckpointError(f"{name}.{key}: model representation mismatch")
    if str(candidate.dtype) != str(live_tensor.dtype):
      raise CheckpointError(f"{name}.{key}: dtype {candidate.dtype} != live {live_tensor.dtype}")


# ═══════════════════════════ full save ═══════════════════════════


def _parameter_names(runner):
  names = {id(p): name for name,p in runner.policy_adapter.named_parameters()}
  return [names[id(p)] for p in runner.trainer.unique_parameters]


def _manager_tensors(env) -> dict:
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


def _physics_arrays(env) -> tuple[dict, dict]:
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


def _require_stable_save_boundary(runner) -> None:
  runtime = runner.runtime
  env = runtime.env
  reasons: list[str] = []
  if getattr(runner, "_lifecycle_phase", None) != "ready":
    reasons.append(
      f"runner lifecycle phase is {getattr(runner, '_lifecycle_phase', None)!r}, "
      "not 'ready' (full save needs a completed collect+update boundary)"
    )
  if getattr(runtime, "_closed", False):
    reasons.append("training runtime closed")
  if getattr(runtime, "_faulted", False):
    reasons.append("training runtime faulted (fault scenes are NON-RESUMABLE)")
  if getattr(runtime, "_stopped", False):
    reasons.append(
      "training runtime stopped after an engineering fall/done (the stop scene "
      "is NOT a resumable boundary)"
    )
  if runner.trainer.faulted:
    reasons.append(
      f"trainer faulted (stage={runner.trainer.fault_stage!r}); a partially "
      "executed optimizer state is never a stable checkpoint"
    )
  if getattr(runner, "_log_failed", False):
    reasons.append("local log IO failed earlier; the run is stopped")
  if runner._active_buffer is not None:
    reasons.append("an active rollout buffer exists (mid-update boundary)")
  if runtime._context.phase != "idle":
    reasons.append(
      f"transition context phase is {runtime._context.phase!r}, not 'idle'"
    )
  if int(runtime._control_steps) != int(runner.total_control_batches):
    reasons.append("unupdated collection exists beyond the last complete iteration")
  if int(runner.trainer.update_count) != int(runner.current_learning_iteration):
    reasons.append("trainer and runner update counters differ")
  if int(runner.total_env_samples) != int(runner.total_control_batches) * int(runtime.num_envs):
    reasons.append("environment sample counter mismatch")
  if int(runtime._control_steps) != int(env.common_step_counter):
    reasons.append("runtime/env control-step counters desynced")
  if bool(torch.any(env._manual_reset_pending)):
    reasons.append("terminal rows are pending an explicit selected reset")
  if int(runner.current_learning_iteration) < 1:
    reasons.append(
      "no completed 24-step collect + 20-update iteration yet (a fresh "
      "construction or mid-iteration boundary is not a save point)"
    )
  if int(runner.trainer.update_count) < 1:
    reasons.append("trainer update_count is 0")
  from ..rl_cfg import ROLLOUT_CONTROL_STEPS

  if int(runner.total_control_batches) != ROLLOUT_CONTROL_STEPS * int(
    runner.current_learning_iteration
  ):
    reasons.append(
      f"total_control_batches {runner.total_control_batches} != "
      f"{ROLLOUT_CONTROL_STEPS}×{runner.current_learning_iteration}"
    )
  if int(runner.trainer.optimizer_step_count) != _JOINT_UPDATE_STEPS_PER_ROLLOUT * int(
    runner.trainer.update_count
  ):
    reasons.append(
      f"optimizer_step_count {runner.trainer.optimizer_step_count} != "
      f"{_JOINT_UPDATE_STEPS_PER_ROLLOUT}×{runner.trainer.update_count} "
      "(partially executed update)"
    )
  if reasons:
    raise CheckpointError(
      "full checkpoint save refused — the boundary is not the stable "
      "post-update/pre-action point: " + "; ".join(reasons)
    )


def save_full_checkpoint(runner, path, *, infos: dict | None = None) -> Path:
  """Refusals are harmless; an actual save failure freezes and retains evidence."""
  _require_stable_save_boundary(runner)
  cache = runner.runtime._input_cache
  if (cache.get() is None or cache._identity != runner.runtime._control_steps
      or not torch.equal(cache._episodes, runner.runtime.episode_ids)):
    raise CheckpointError("full save requires the prepared first-action cache; prepare after selected reset first")
  target = Path(path)
  if target.exists():
    raise CheckpointError(f"refusing to overwrite an existing file: {target}")
  try:
    return _save_full_checkpoint(runner, target, infos=infos)
  except CheckpointFault:
    raise
  except BaseException as error:
    runner._lifecycle_phase = "faulted"
    runner.runtime._save_fault(error, phase="checkpoint/save_prepare")
    raise CheckpointFault(f"checkpoint save failed: {error!r}") from error


def _save_full_checkpoint(runner, path, *, infos: dict | None = None) -> Path:
  """Save the complete same-GPU resume state atomically to a NEW file."""
  _require_stable_save_boundary(runner)
  target = Path(path)
  if target.exists():
    raise CheckpointError(
      f"refusing to overwrite an existing file: {target} (history is never "
      "rewritten; write a NEW checkpoint file instead)"
    )
  target.parent.mkdir(parents=True, exist_ok=True)

  runtime = runner.runtime
  env = runtime.env
  physics, manifest = _physics_arrays(env)
  cache = runtime._input_cache
  cached = cache.get()
  payload = {
    "schema": CHECKPOINT_SCHEMA,
    "kind": CHECKPOINT_KIND_FULL,
    "resumable": True,
    "task": TASK_ID,
    "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "identity": _comparable_identity(runner),
    "infos": dict(infos) if infos else None,
    "physics": {
      "manifest": manifest,
      "arrays": physics,
      "restore_protocol": (
        "in-place copy_ into the original wp arrays (CUDA-graph storage "
        "preserved) -> one sim.forward() rebuilds derived state/caches -> "
        "the persistent fields are written AGAIN so solver-clobbered values "
        "like qacc_warmstart match the saved boundary bit-exactly"
      ),
    },
    "algorithm": {
      "adapter_state_dict": _to_cpu(runner.policy_adapter.state_dict()),
      "trainer_state": _to_cpu(runner.trainer.state_dict()),
      "model_fingerprint": None,  # filled below (post-cpu, live fingerprint)
      "parameter_names": _parameter_names(runner),
      "optimizer_param_count": sum(
        len(group["params"]) for group in runner.trainer.optimizer.param_groups
      ),
    },
    "runtime": {
      "control_steps": int(runtime._control_steps),
      "episode_ids": runtime.episode_ids.detach().cpu().clone(),
      "step_in_episode": runtime.step_in_episode.detach().cpu().clone(),
      "scan_counters": runtime.scan_counters.detach().cpu().clone(),
      "sensor_scan_counts": runtime.sensor_scan_counts.detach().cpu().clone(),
      "next_due_scan_step": runtime.next_due_scan_step.detach().cpu().clone(),
      "window": _serialize_window(runtime.window),
      "ground_cache": _serialize_ground_cache(runtime),
      "input_cache": {
        "identity": None if cached is None else int(cache._identity),
        "episodes": None if cached is None else cache._episodes.detach().cpu().clone(),
        "prepared": None if cached is None else _serialize_prepared(cached),
      },
      "current_window_snapshot": (
        None if runtime._current_window_snapshot is None
        else _serialize_snapshot(runtime.window.read())
      ),
      "commands": _to_cpu(runtime._commands.state_dict()),
      "perturbations": (
        None
        if getattr(runtime, "_perturbations", None) is None
        else _to_cpu(runtime._perturbations.state_dict())
      ),
      "scan_audit_bound": runtime.scan_audit_bound,
    },
    "env_state": {
      "common_step_counter": int(env.common_step_counter),
      "sim_step_counter": int(env._sim_step_counter),
      "episode_length_buf": env.episode_length_buf.detach().cpu().clone(),
      "manual_reset_pending": env._manual_reset_pending.detach().cpu().clone(),
      "reset_terminated": (
        None if not hasattr(env, "reset_terminated")
        else env.reset_terminated.detach().cpu().clone()
      ),
      "reset_time_outs": (
        None if not hasattr(env, "reset_time_outs")
        else env.reset_time_outs.detach().cpu().clone()
      ),
      "reset_buf": (
        None if not hasattr(env, "reset_buf")
        else env.reset_buf.detach().cpu().clone()
      ),
      "reward_buf": (
        None if not hasattr(env, "reward_buf")
        else env.reward_buf.detach().cpu().clone()
      ),
    },
    "action_state": {
      "prev_action": env.action_manager.get_term("residual_pd").prev_action.detach().cpu().clone(),
      "last_raw_action": (
        env.action_manager.get_term("residual_pd").last_raw_action.detach().cpu().clone()
      ),
      "last_clipped_action": (
        env.action_manager.get_term("residual_pd").last_clipped_action.detach().cpu().clone()
      ),
      "last_target": env.action_manager.get_term("residual_pd").last_target.detach().cpu().clone(),
      "processed_actions": (
        env.action_manager.get_term("residual_pd")._processed_actions.detach().cpu().clone()
      ),
      "raw_actions": (
        env.action_manager.get_term("residual_pd")._raw_actions.detach().cpu().clone()
      ),
      "control_steps_seen": int(env.action_manager.get_term("residual_pd").control_steps_seen),
    },
    "foot_air_state": None,  # filled below
    "termination_state": None,  # filled below
    "manager_state": _to_cpu(_manager_tensors(env)),
    "reward_terms_cache": None,  # diagnostic; filled below
    "prng": None,  # filled below
    "runner_state": {
      "current_learning_iteration": int(runner.current_learning_iteration),
      "total_control_batches": int(runner.total_control_batches),
      "total_env_samples": int(runner.total_env_samples),
      "artifact_exported": bool(runner._artifact_exported),
      "learn_calls_recorded": len(runner._learn_calls),
    },
    "diagnostics_not_saved": {
      "runtime_transitions_count": len(runtime.transitions),
      "runtime_scan_log_count": len(runtime.scan_log),
      "pipeline_scan_records_count": len(runtime.pipeline.scan_records),
      "note": (
        "bounded diagnostic histories affect no future behavior and are "
        "deliberately not serialized; counts recorded for audit only"
      ),
    },
  }
  from ..collector import model_state_fingerprint

  payload["algorithm"]["model_fingerprint"] = model_state_fingerprint(runner.policy_adapter)

  foot = getattr(env, "_confined_foot_air_state", None)
  payload["foot_air_state"] = None if foot is None else {
    "air_time_s": foot.air_time_s.detach().cpu().clone(),
    "last_support": foot.last_support.detach().cpu().clone(),
    "control_id": foot.control_id,
    "landing": _to_cpu(foot.landing),
    "consumed_air_s": _to_cpu(foot.consumed_air_s),
  }
  termination = getattr(env, "_confined_termination_state", None)
  payload["termination_state"] = None if termination is None else {
    "low_height_count": termination.low_height_count.detach().cpu().clone(),
    "tilt_count": termination.tilt_count.detach().cpu().clone(),
    "low_height_done": termination.low_height_done.detach().cpu().clone(),
    "tilt_done": termination.tilt_done.detach().cpu().clone(),
    "flip_done": termination.flip_done.detach().cpu().clone(),
    "any_fall": termination.any_fall.detach().cpu().clone(),
    "control_id": termination.control_id,
    "presets_count": len(termination.presets),
  }
  terms = getattr(env, "_confined_training_reward_terms", None)
  payload["reward_terms_cache"] = None if terms is None else {
    "control_id": terms.get("control_id"),
    "raws": _to_cpu(terms.get("raws", {})),
    "note": "per-control-id reward cache of the last completed step (diagnostic)",
  }
  payload["prng"] = {
    "python": random.getstate(),
    "numpy": np.random.get_state(),
    "torch_cpu": torch.get_rng_state().clone(),
    "torch_cuda": torch.cuda.get_rng_state(runner.device).clone(),
    "action_generator": runner.provider.generator_state().clone(),
    "command_generators_inside": "runtime.commands (state saved in runtime.commands)",
    "optimization_shuffle_inside": "trainer_state (state saved in algorithm.trainer_state)",
  }

  _validate_payload_finiteness(payload)

  tmp = target.with_name(f"{target.name}.tmp-{os.getpid()}")
  try:
    torch.save(payload, tmp)
    os.replace(tmp, target)
  except BaseException as error:
    try:
      tmp.unlink(missing_ok=True)
    except OSError:
      pass
    runner._lifecycle_phase = "faulted"
    runtime._save_fault(error, phase="checkpoint/save_io")
    raise CheckpointFault(f"atomic save to {target} failed: {error!r}") from error
  return target


# ═══════════════════════════ full load ═══════════════════════════


def _compare_identity(current: dict, saved: dict) -> None:
  for key in sorted(set(current) | set(saved)):
    if key not in current:
      raise CheckpointError(f"saved identity has unknown extra key {key!r}")
    if key not in saved:
      raise CheckpointError(f"saved identity is missing key {key!r}")
    if current[key] != saved[key]:
      raise CheckpointError(
        f"identity mismatch on {key!r}: checkpoint "
        f"{json.dumps(saved[key], default=str, sort_keys=True)[:400]} != current "
        f"{json.dumps(current[key], default=str, sort_keys=True)[:400]} — full "
        "resume is same-device/model/config/env-count/grid/source only"
      )


def validate_payload_envelope(payload: dict) -> None:
  """Pure schema/kind/section validation of a full-resume payload (no runner).

  Used by the full loader AND by CPU contract tests: a weights-only file, a
  NON-RESUMABLE artifact, another schema or a missing section is rejected
  before anything live is touched.
  """
  required_sections = (
    "schema", "kind", "resumable", "task", "created", "identity", "physics",
    "algorithm", "runtime", "env_state", "action_state", "foot_air_state",
    "termination_state", "prng", "runner_state", "manager_state",
  )
  missing = [key for key in required_sections if key not in payload]
  if missing:
    raise CheckpointError(f"payload missing required sections: {missing}")
  if payload["schema"] != CHECKPOINT_SCHEMA:
    raise CheckpointError(
      f"schema {payload['schema']!r} != {CHECKPOINT_SCHEMA!r} (old/other-schema "
      "checkpoints are rejected, not loosely migrated)"
    )
  if payload["kind"] != CHECKPOINT_KIND_FULL:
    raise CheckpointError(
      f"kind {payload['kind']!r} is not {CHECKPOINT_KIND_FULL!r} (weights-only "
      "files and NON_RESUMABLE diagnostics are not full-resume payloads; use "
      "load_weights for the weights mode)"
    )
  if payload["resumable"] is not True:
    raise CheckpointError("payload is marked non-resumable")
  if payload["task"] != TASK_ID:
    raise CheckpointError("checkpoint task mismatch")


def _validate_full_payload(runner, payload: dict) -> None:
  runtime = runner.runtime
  env = runtime.env
  validate_payload_envelope(payload)
  _require_same_tensors("manager_state", payload["manager_state"], _manager_tensors(env))
  _compare_identity(_comparable_identity(runner), payload["identity"])
  _validate_payload_finiteness(payload)

  # ── physics ──────────────────────────────────────────────────────────────
  physics = payload["physics"]
  if not isinstance(physics, dict) or "manifest" not in physics or "arrays" not in physics:
    raise CheckpointError("physics section malformed (manifest/arrays missing)")
  current_manifest = _physics_arrays(env)[1]
  saved_manifest = physics["manifest"]
  if set(saved_manifest) != set(current_manifest):
    raise CheckpointError(
      f"physics manifest field set mismatch: {sorted(set(saved_manifest))} vs "
      f"{sorted(set(current_manifest))}"
    )
  for name, entry in current_manifest.items():
    saved_entry = saved_manifest[name]
    if any(saved_entry.get(k) != entry.get(k) for k in ("shape", "dtype", "empty", "tensor_shape", "tensor_dtype")):
      raise CheckpointError(
        f"physics field {name}: saved manifest {saved_entry} incompatible with "
        f"current {entry} (model/structure changed)"
      )
    if not entry["empty"]:
      tensor = physics["arrays"].get(name)
      if not torch.is_tensor(tensor) or tuple(tensor.shape) != tuple(entry["tensor_shape"]) or str(tensor.dtype) != entry["tensor_dtype"]:
        raise CheckpointError(f"physics array {name} missing or wrong shape")
    elif name in physics["arrays"]:
      raise CheckpointError(f"physics array {name} present although the field is empty")

  expected_arrays = {k for k,v in current_manifest.items() if not v["empty"]}
  if set(physics["arrays"]) != expected_arrays:
    raise CheckpointError("physics non-empty array key set mismatch")

  # ── algorithm ────────────────────────────────────────────────────────────
  algorithm = payload["algorithm"]
  for key in ("adapter_state_dict", "trainer_state", "optimizer_param_count"):
    if key not in algorithm:
      raise CheckpointError(f"algorithm section missing {key!r}")
  _require_same_tensors(
    "algorithm.adapter_state_dict",
    algorithm["adapter_state_dict"],
    runner.policy_adapter.state_dict(),
  )
  log_std = algorithm["adapter_state_dict"]["distribution_provider.log_std"]
  if not bool(torch.isfinite(log_std.exp()).all()) or not bool((log_std.exp() > 0).all()):
    raise CheckpointError("saved log_std does not define a finite positive Normal std")
  trainer_state = algorithm["trainer_state"]
  for key in ("schema", "optimizer", "learning_rate", "update_count",
              "optimizer_step_count", "shuffle_generator_state"):
    if key not in trainer_state:
      raise CheckpointError(f"trainer_state missing {key!r}")
  if trainer_state["schema"] != runner.trainer.state_dict()["schema"]:
    raise CheckpointError("trainer_state schema mismatch")
  optimizer_state = trainer_state["optimizer"]
  groups = optimizer_state.get("param_groups")
  if not isinstance(groups, list) or len(groups) != 1:
    raise CheckpointError("trainer_state optimizer must have exactly one param group")
  unique = list(runner.trainer.unique_parameters)
  if algorithm.get("parameter_names") != _parameter_names(runner):
    raise CheckpointError("optimizer parameter name/order mismatch")
  if groups[0].get("params") != list(range(len(unique))):
    raise CheckpointError(
      "optimizer param indices do not match the unique-parameter order "
      "(structure changed since the checkpoint)"
    )
  if int(algorithm["optimizer_param_count"]) != len(unique):
    raise CheckpointError("optimizer_param_count mismatch")
  state_map = optimizer_state.get("state")
  if not isinstance(state_map, dict) or set(map(int, state_map)) != set(range(len(unique))):
    raise CheckpointError(
      "optimizer state must cover exactly the unique parameters "
      f"({len(unique)}); got {sorted(map(int, state_map)) if isinstance(state_map, dict) else type(state_map).__name__}"
    )
  for index, entry in state_map.items():
    parameter = unique[int(index)]
    for moment in ("exp_avg", "exp_avg_sq"):
      tensor = entry.get(moment)
      if not torch.is_tensor(tensor) or tuple(tensor.shape) != tuple(parameter.shape) or tensor.dtype != parameter.dtype:
        raise CheckpointError(
          f"optimizer state[{index}][{moment}] shape mismatch vs parameter"
        )
    step = entry.get("step")
    if not torch.is_tensor(step) or step.numel() != 1:
      raise CheckpointError(f"optimizer state[{index}]['step'] must be a scalar tensor")
  if trainer_state.get("config") != runner.trainer.config_identity() or trainer_state.get("grid_identity_hash") != runner.trainer.grid_identity_hash or trainer_state.get("faulted_stage") is not None:
    raise CheckpointError("trainer config/grid/faulted state mismatch")
  if not runner.trainer.config.lr_min <= float(trainer_state["learning_rate"]) <= runner.trainer.config.lr_max or groups[0].get("lr") != trainer_state["learning_rate"]:
    raise CheckpointError("optimizer group LR/range mismatch")
  live_group = runner.trainer.optimizer.state_dict()["param_groups"][0]
  if {k:v for k,v in groups[0].items() if k != "lr"} != {k:v for k,v in live_group.items() if k != "lr"}:
    raise CheckpointError("optimizer group configuration mismatch")
  shuffle = trainer_state["shuffle_generator_state"]
  if not torch.is_tensor(shuffle) or shuffle.dtype is not torch.uint8 or shuffle.dim() != 1:
    raise CheckpointError("shuffle_generator_state malformed (1-D uint8 tensor required)")
  if not (0.0 < float(trainer_state["learning_rate"])):
    raise CheckpointError("trainer_state learning_rate must be positive")

  # ── runtime tensors ──────────────────────────────────────────────────────
  runtime_state = payload["runtime"]
  for key in (
    "control_steps", "episode_ids", "step_in_episode", "scan_counters",
    "sensor_scan_counts", "next_due_scan_step", "window", "ground_cache",
    "input_cache", "current_window_snapshot", "commands", "perturbations",
  ):
    if key not in runtime_state:
      raise CheckpointError(f"runtime section missing {key!r}")
  for key, live in (
    ("episode_ids", runtime.episode_ids),
    ("step_in_episode", runtime.step_in_episode),
    ("scan_counters", runtime.scan_counters),
    ("sensor_scan_counts", runtime.sensor_scan_counts),
    ("next_due_scan_step", runtime.next_due_scan_step),
  ):
    _require_same_tensors(f"runtime.{key}", {key: runtime_state[key]}, {key: live})
  window = runtime_state["window"]
  live_window = {name: getattr(runtime.window, f"_{name}") for name in _WINDOW_TENSOR_FIELDS}
  _require_same_tensors(
    "runtime.window",
    {name: window[name] for name in _WINDOW_TENSOR_FIELDS},
    live_window,
  )
  source_attached = window["source_dirs_attached"]
  if source_attached.dtype is not torch.bool or tuple(source_attached.shape) != (runtime.num_envs,):
    raise CheckpointError("runtime.window source_dirs_attached must be (num_envs,) bool")
  for prefix in ("near", "far"):
    valid = window[f"{prefix}_valid"]
    present = window[f"{prefix}_source_dirs_present"]
    indices = window[f"{prefix}_source_indices"]
    directions = window[f"{prefix}_source_dirs_sensor"]
    if not torch.equal(present, valid) or not torch.equal(present, indices >= 0):
      raise CheckpointError(f"runtime.window {prefix} source direction masks/raw ids mismatch")
    attached_mask = source_attached.reshape(
      (runtime.num_envs,) + (1,) * (valid.dim() - 1)
    )
    if bool((valid & ~attached_mask).any()):
      raise CheckpointError(
        f"runtime.window {prefix} has valid points without attached source-ray metadata"
      )
    if bool(present.any()):
      norms = torch.linalg.vector_norm(directions[present], dim=-1)
      if not bool(torch.allclose(norms, torch.ones_like(norms), atol=1e-5, rtol=0.0)):
        raise CheckpointError(f"runtime.window {prefix} source directions are not unit vectors")
    if bool((~present).any()) and not torch.equal(
      directions[~present], torch.zeros_like(directions[~present])
    ):
      raise CheckpointError(f"runtime.window {prefix} absent source directions are nonzero")
  seen = window.get("seen_episodes")
  if not isinstance(seen, list) or len(seen) != runtime.num_envs:
    raise CheckpointError("runtime.window.seen_episodes malformed")
  # P6 window angle-key metadata: presence must match the live window (a
  # perturbed-lidar window carries keys; a clean window never does).
  saved_window_keys = window.get("angle_keys")
  live_window_keys = runtime.window.read_angle_keys()
  if (saved_window_keys is None) != (live_window_keys is None):
    raise CheckpointError(
      "runtime.window.angle_keys presence mismatch (the checkpoint and the "
      "live window must both be key-carrying or both clean)"
    )
  if saved_window_keys is not None:
    from hexapod_mjlab.rl.confined_locomotion.contracts import (
      FAR_WINDOW_SLOTS as _FAR_SLOTS,
    )
    from hexapod_mjlab.rl.confined_locomotion.contracts import (
      FAR_POINTS_PER_SCAN as _FAR_PER_SCAN,
    )
    from hexapod_mjlab.rl.confined_locomotion.contracts import (
      NEAR_POINTS as _NEAR,
    )
    batch = int(runtime.num_envs)
    expected_key_shapes = {
      "near": ((batch, _NEAR), torch.float32),
      "far": ((batch, _FAR_SLOTS, _FAR_PER_SCAN), torch.float32),
      "present": ((batch,), torch.bool),
    }
    if not isinstance(saved_window_keys, dict) or set(saved_window_keys) != set(
      expected_key_shapes
    ):
      raise CheckpointError("runtime.window.angle_keys malformed (field set)")
    for name, (shape, dtype) in expected_key_shapes.items():
      tensor = saved_window_keys[name]
      if not torch.is_tensor(tensor) or tuple(tensor.shape) != shape or tensor.dtype is not dtype:
        raise CheckpointError(
          f"runtime.window.angle_keys.{name}: expected {shape}/{dtype}, got "
          f"{tuple(tensor.shape) if torch.is_tensor(tensor) else type(tensor).__name__}"
        )
      if dtype is torch.float32 and not bool(torch.isfinite(tensor).all()):
        raise CheckpointError(f"runtime.window.angle_keys.{name}: non-finite keys")
    if not torch.equal(saved_window_keys["present"].cpu(), live_window_keys.present.cpu()):
      raise CheckpointError(
        "runtime.window.angle_keys.present differs from the live window "
        "(per-row key presence is part of the restored state)"
      )
  cache_state = runtime_state["input_cache"]
  if not isinstance(cache_state, dict):
    raise CheckpointError("runtime.input_cache malformed")
  if cache_state["prepared"] is not None:
    prepared_state = cache_state["prepared"]
    if set(prepared_state) != {
      "observation", "near_points", "near_valid", "far_points", "far_valid",
      "near_filled", "near_scan_ids", "near_times", "far_point_scan_ids",
      "far_point_times", "near_angle_keys", "far_angle_keys",
      "near_source_dirs_sensor", "far_source_dirs_sensor",
      "near_source_dirs_present", "far_source_dirs_present",
      "near_source_indices", "far_source_indices",
    }:
      raise CheckpointError("input-cache prepared field set mismatch")
    if set(prepared_state["observation"]) != set(_PREPARED_OBSERVATION_FIELDS):
      raise CheckpointError("input-cache observation field set mismatch")
    from hexapod_mjlab.rl.confined_locomotion.contracts import (
      FAR_POINTS_PER_SCAN,
      FAR_WINDOW_SLOTS,
      JOINT_DIM,
      NEAR_POINTS,
    )

    batch = int(runtime.num_envs)
    far_sequence = FAR_WINDOW_SLOTS * FAR_POINTS_PER_SCAN  # 640
    expected_prepared_shapes = {
      "env_ids": (batch,), "episode_ids": (batch,), "step_ids": (batch,),
      "time_s": (batch,),
      "lin_vel_b": (batch, 3), "ang_vel_b": (batch, 3),
      "gravity_b": (batch, 3), "command": (batch, 3),
      "joint_pos_rel_ref": (batch, JOINT_DIM), "joint_vel": (batch, JOINT_DIM),
      "prev_action": (batch, JOINT_DIM),
      "near_points": (batch, NEAR_POINTS, 3), "near_valid": (batch, NEAR_POINTS),
      "far_points": (batch, far_sequence, 3), "far_valid": (batch, far_sequence),
      "near_angle_keys": (batch, NEAR_POINTS),
      "far_angle_keys": (batch, far_sequence),
      "near_source_dirs_sensor": (batch, NEAR_POINTS, 3),
      "far_source_dirs_sensor": (batch, far_sequence, 3),
      "near_source_dirs_present": (batch, NEAR_POINTS),
      "far_source_dirs_present": (batch, far_sequence),
      "near_source_indices": (batch, NEAR_POINTS),
      "far_source_indices": (batch, far_sequence),
      "near_filled": (batch,), "near_scan_ids": (batch,), "near_times": (batch,),
      "far_point_scan_ids": (batch, far_sequence),
      "far_point_times": (batch, far_sequence),
    }
    flat_prepared = {**prepared_state["observation"], **prepared_state}
    flat_prepared.pop("observation")
    for key, shape in expected_prepared_shapes.items():
      tensor = flat_prepared.get(key)
      if not torch.is_tensor(tensor) or tuple(tensor.shape) != shape:
        raise CheckpointError(
          f"input-cache prepared.{key}: expected shape {shape}, got "
          f"{tuple(tensor.shape) if torch.is_tensor(tensor) else type(tensor).__name__}"
        )
  if cache_state.get("prepared") is None or cache_state.get("identity") != runtime_state["control_steps"] or not torch.equal(cache_state["episodes"], runtime_state["episode_ids"]):
    raise CheckpointError("stable first-action cache missing or identity mismatch")
  integer_names = {
    "env_ids", "episode_ids", "step_ids", "near_scan_ids",
    "far_point_scan_ids", "near_source_indices", "far_source_indices",
  }
  bool_names = {
    "near_valid", "far_valid", "near_filled",
    "near_source_dirs_present", "far_source_dirs_present",
  }
  for name,tensor in flat_prepared.items():
    if tensor is None:
      raise CheckpointError(f"prepared {name} missing in v3 source-ray state")
    dtype = torch.int64 if name in integer_names else torch.bool if name in bool_names else torch.float32
    if tensor.dtype != dtype:
      raise CheckpointError(f"prepared {name} dtype mismatch")
  for name in (
    "near_angle_keys", "far_angle_keys", "near_source_dirs_sensor",
    "far_source_dirs_sensor",
  ):
    if not bool(torch.isfinite(flat_prepared[name]).all()):
      raise CheckpointError(f"prepared {name} contains NaN/Inf")
  for prefix in ("near", "far"):
    valid = flat_prepared[f"{prefix}_valid"]
    present = flat_prepared[f"{prefix}_source_dirs_present"]
    indices = flat_prepared[f"{prefix}_source_indices"]
    directions = flat_prepared[f"{prefix}_source_dirs_sensor"]
    if not torch.equal(present, valid) or not torch.equal(present, indices >= 0):
      raise CheckpointError(f"prepared {prefix} source direction masks/raw ids mismatch")
    if bool(present.any()):
      norms = torch.linalg.vector_norm(directions[present], dim=-1)
      if not bool(torch.allclose(norms, torch.ones_like(norms), atol=1e-5, rtol=0.0)):
        raise CheckpointError(f"prepared {prefix} source directions are not unit vectors")
    if bool((~present).any()) and not torch.equal(
      directions[~present], torch.zeros_like(directions[~present])
    ):
      raise CheckpointError(f"prepared {prefix} missing source direction slots are nonzero")
  _deserialize_prepared(cache_state["prepared"], torch.device("cpu")).observation.validate()
  obs = cache_state["prepared"]["observation"]
  if not torch.equal(obs["episode_ids"], runtime_state["episode_ids"]) or not torch.equal(obs["step_ids"], runtime_state["step_in_episode"]):
    raise CheckpointError("prepared observation lifecycle identity mismatch")
  snapshot_state = runtime_state["current_window_snapshot"]
  if snapshot_state is not None and set(snapshot_state) != set(_SNAPSHOT_TENSOR_FIELDS):
    raise CheckpointError("current_window_snapshot field set mismatch")
  if snapshot_state is None:
    raise CheckpointError("stable window snapshot missing")
  _require_same_tensors("window_snapshot", snapshot_state, _serialize_snapshot(runtime.window.read()))
  ground_cache = runtime_state["ground_cache"]
  if not isinstance(ground_cache, list) or len(ground_cache) != runtime.num_envs:
    raise CheckpointError("runtime.ground_cache malformed")
  for row, entry in enumerate(ground_cache):
    if entry is None:
      continue
    if set(entry) != {
      "env_row", "scan_id", "episode_id", "persistent_scan_count", "time_s",
      *_GROUND_CACHE_TENSOR_FIELDS,
    }:
      raise CheckpointError(f"ground_cache[{row}] field set mismatch")
    if int(entry["env_row"]) != row:
      raise CheckpointError(f"ground_cache[{row}] env_row mismatch")
    grid_points = int(runtime.grid_cfg.num_points)
    if tuple(entry["heights"].shape) != (grid_points,):
      raise CheckpointError(f"ground_cache[{row}].heights shape mismatch (grid M changed?)")
    shapes = {"heights": (grid_points,), "positions": (grid_points,3), "valid": (grid_points,), "world_points": (grid_points,3), "base_pos_w": (3,), "base_quat_wxyz": (4,)}
    for name, shape in shapes.items():
      tensor = entry[name]
      dtype = torch.bool if name == "valid" else torch.float32
      if not torch.is_tensor(tensor) or tuple(tensor.shape) != shape or tensor.dtype != dtype:
        raise CheckpointError(f"ground cache {row}/{name} malformed")
    if tuple(entry["positions"].shape) != (grid_points, 3):
      raise CheckpointError(f"ground_cache[{row}].positions shape mismatch")
  commands = runtime_state["commands"]
  scheduler = runtime._commands
  for key in ("num_envs", "seed", "period", "allow_manual"):
    if commands.get(key) != getattr(scheduler, key):
      raise CheckpointError(
        f"command scheduler {key} mismatch: {commands.get(key)!r} != "
        f"{getattr(scheduler, key)!r}"
      )
  if not isinstance(commands.get("generators"), list) or len(commands["generators"]) != runtime.num_envs:
    raise CheckpointError("command generator state list malformed")

  # ── P6 perturbation continuation state ───────────────────────────────────
  # Presence must match the live runtime (clean ↔ clean, perturbed ↔
  # perturbed); the structure is validated by restoring it into a THROWAWAY
  # owner built from the live cfg/seed — nothing live is touched and the
  # live owner's own load runs later only after everything validated.
  saved_perturbations = runtime_state.get("perturbations")
  live_perturbations = getattr(runtime, "_perturbations", None)
  if (saved_perturbations is None) != (live_perturbations is None):
    raise CheckpointError(
      "observation-perturbation presence mismatch: a clean checkpoint cannot "
      "resume into a perturbed runtime and vice versa (the noise recipe is "
      "part of the identity)"
    )
  if live_perturbations is not None:
    if not isinstance(saved_perturbations, dict):
      raise CheckpointError("runtime.perturbations malformed")
    from ..perturbations import ObservationPerturbations, PerturbationContractError

    try:
      throwaway = ObservationPerturbations(
        runtime.num_envs,
        cfg=runtime.perturbation_cfg,
        seed=runtime.seed,
        device="cpu",
      )
      throwaway.load_state_dict(saved_perturbations)
      if throwaway.range_enabled and not torch.equal(
          saved_perturbations["range_draw_counts"], runtime_state["sensor_scan_counts"].cpu()):
        raise PerturbationContractError("range draw count differs from persistent scan count")
      if throwaway.proprio_enabled:
        halves = torch.tensor([.1] * 3 + [.2] * 3 + [.05] * 3 + [.05] * 18 + [1.5] * 18)
        obs_fields = ("lin_vel_b", "ang_vel_b", "gravity_b", "joint_pos_rel_ref", "joint_vel")
        for row, entry in enumerate(saved_perturbations["proprio_cache"]):
          if entry is None or (entry["episode_id"], entry["step_id"], entry["control_step"]) != (
              int(runtime_state["episode_ids"][row]), int(runtime_state["step_in_episode"][row]), runtime_state["control_steps"]):
            raise PerturbationContractError(f"proprio cache {row} observation identity differs")
          if bool((entry["noise"].abs().cpu() > halves).any()):
            raise PerturbationContractError(f"proprio cache {row} noise outside confirmed bounds")
          expected_noisy = entry["clean"].cpu() + entry["noise"].cpu()
          expected_noisy[6:9] /= torch.linalg.vector_norm(expected_noisy[6:9])
          observed = torch.cat([obs[name][row].cpu() for name in obs_fields])
          if not torch.equal(expected_noisy, entry["noisy"].cpu()) or not torch.equal(observed, entry["noisy"].cpu()):
            raise PerturbationContractError(f"proprio cache {row} noise/observation mismatch")
      if throwaway.gap_enabled:
        for row, gap in enumerate(saved_perturbations["gap_state"]):
          count = int(runtime_state["sensor_scan_counts"][row])
          nominal = gap["phase"] + 10 * gap["next_event_index"]
          epsilon = gap["scheduled_epsilon"]
          if ((epsilon is None and count >= nominal - 1)
              or (epsilon is not None and not nominal - 1 <= count < nominal + epsilon)):
            raise PerturbationContractError(f"gap schedule {row} persistent clock mismatch")
    except PerturbationContractError as error:
      raise CheckpointError(
        f"runtime.perturbations does not validate as the confirmed "
        f"continuation state: {error}"
      ) from error

  # ── env / action / lifecycle state ───────────────────────────────────────
  env_state = payload["env_state"]
  _require_same_tensors(
    "env_state.episode_length_buf",
    {"episode_length_buf": env_state["episode_length_buf"]},
    {"episode_length_buf": env.episode_length_buf},
  )
  _require_same_tensors(
    "env_state.manual_reset_pending",
    {"manual_reset_pending": env_state["manual_reset_pending"]},
    {"manual_reset_pending": env._manual_reset_pending},
  )
  for key, attr, expected_dtype in (
    ("reset_terminated", "reset_terminated", torch.bool),
    ("reset_time_outs", "reset_time_outs", torch.bool),
    ("reset_buf", "reset_buf", torch.bool),
    ("reward_buf", "reward_buf", torch.float32),
  ):
    saved = env_state[key]
    if saved is not None:
      # (num_envs,) last-step outputs; a FRESH env has not stepped yet, so the
      # live attribute may be absent — validate the saved tensor directly.
      if not torch.is_tensor(saved) or tuple(saved.shape) != (int(runtime.num_envs),):
        raise CheckpointError(f"env_state.{key} malformed shape")
      if saved.dtype is not expected_dtype:
        raise CheckpointError(f"env_state.{key} dtype {saved.dtype} != {expected_dtype}")
      continue
    if getattr(env, attr, None) is not None:
      raise CheckpointError(f"env_state.{key} absent although the live env has one")
  action = env.action_manager.get_term("residual_pd")
  for key, live in (
    ("prev_action", action.prev_action),
    ("last_raw_action", action.last_raw_action),
    ("last_clipped_action", action.last_clipped_action),
    ("last_target", action.last_target),
    ("processed_actions", action._processed_actions),
    ("raw_actions", action._raw_actions),
  ):
    _require_same_tensors(f"action_state.{key}", {key: payload["action_state"][key]}, {key: live})
  foot = getattr(env, "_confined_foot_air_state", None)
  saved_foot = payload["foot_air_state"]
  if (saved_foot is None) != (foot is None):
    raise CheckpointError("foot air state presence mismatch")
  if foot is not None:
    for key, live in (
      ("air_time_s", foot.air_time_s),
      ("last_support", foot.last_support),
    ):
      _require_same_tensors(f"foot_air_state.{key}", {key: saved_foot[key]}, {key: live})
  if saved_foot is not None:
    for name in ("landing", "consumed_air_s"):
      tensor = saved_foot[name]
      if tensor is not None and (not torch.is_tensor(tensor) or tuple(tensor.shape) != (runtime.num_envs,6) or tensor.dtype != (torch.bool if name == "landing" else torch.float32)):
        raise CheckpointError(f"foot cache {name} malformed")
  termination = getattr(env, "_confined_termination_state", None)
  saved_termination = payload["termination_state"]
  if (saved_termination is None) != (termination is None):
    raise CheckpointError("termination state presence mismatch")
  if termination is not None:
    for key, live in (
      ("low_height_count", termination.low_height_count),
      ("tilt_count", termination.tilt_count),
      ("low_height_done", termination.low_height_done),
      ("tilt_done", termination.tilt_done),
      ("flip_done", termination.flip_done),
      ("any_fall", termination.any_fall),
    ):
      _require_same_tensors(
        f"termination_state.{key}", {key: saved_termination[key]}, {key: live}
      )

  # ── PRNG structure ───────────────────────────────────────────────────────
  prng = payload["prng"]
  for key in ("python", "numpy", "torch_cpu", "torch_cuda", "action_generator"):
    if key not in prng:
      raise CheckpointError(f"prng section missing {key!r}")
  python_state = prng["python"]
  if not (isinstance(python_state, tuple) and len(python_state) == 3):
    raise CheckpointError("prng.python state malformed")
  numpy_state = prng["numpy"]
  if not (isinstance(numpy_state, tuple) and len(numpy_state) == 5):
    raise CheckpointError("prng.numpy state malformed")
  for key in ("torch_cpu", "torch_cuda", "action_generator"):
    tensor = prng[key]
    if not torch.is_tensor(tensor) or tensor.dtype is not torch.uint8 or tensor.dim() != 1:
      raise CheckpointError(f"prng.{key} malformed (1-D uint8 tensor required)")

  # Validate random states on isolated generators before mutating any live state.
  try:
    random.Random().setstate(prng["python"])
    np.random.RandomState().set_state(prng["numpy"])
    torch.Generator(device="cpu").set_state(prng["torch_cpu"].cpu())
    for key in ("torch_cuda", "action_generator"):
      torch.Generator(device=runner.device).set_state(prng[key].cpu())
    torch.Generator(device=runner.device).set_state(shuffle.cpu())
    for state in commands["generators"]:
      torch.Generator(device="cpu").set_state(state.cpu())
  except Exception as error:
    raise CheckpointError(f"invalid PRNG state: {error}") from error
  _require_same_tensors("commands", {k:commands[k] for k in ("command", "steps_since_resample", "manual", "sampled_once")}, {k:scheduler.state_dict()[k] for k in ("command", "steps_since_resample", "manual", "sampled_once")})

  # ── runner counters ──────────────────────────────────────────────────────
  runner_state = payload["runner_state"]
  for key in ("current_learning_iteration", "total_control_batches", "total_env_samples"):
    if not isinstance(runner_state.get(key), int):
      raise CheckpointError(f"runner_state.{key} malformed")

  iteration = runner_state["current_learning_iteration"]
  controls = runner_state["total_control_batches"]
  if (iteration < 1 or controls != 24 * iteration
      or runner_state["total_env_samples"] != controls * runtime.num_envs
      or trainer_state["update_count"] != iteration
      or trainer_state["optimizer_step_count"] != _JOINT_UPDATE_STEPS_PER_ROLLOUT * iteration
      or runtime_state["control_steps"] != controls
      or env_state["common_step_counter"] != controls
      or env_state["sim_step_counter"] != DECIMATION * controls
      or payload["action_state"]["control_steps_seen"] != controls):
    raise CheckpointError("saved stable-boundary counters are inconsistent")
  if bool(env_state["manual_reset_pending"].any()):
    raise CheckpointError("saved boundary has pending terminal/reset rows")
  if not torch.equal(env_state["episode_length_buf"], runtime_state["step_in_episode"]):
    raise CheckpointError("saved episode-length counters differ")
  if not torch.equal(obs["command"].cpu(), commands["command"].cpu()):
    raise CheckpointError("first-action command cache differs from scheduler")
  if not torch.equal(obs["env_ids"], torch.arange(runtime.num_envs, dtype=torch.int64)):
    raise CheckpointError("first-action env ordering differs")
  if not torch.equal(window["episode_ids"], runtime_state["episode_ids"]):
    raise CheckpointError("scan window episode identity differs")
  for row, entry in enumerate(ground_cache):
    if entry is None or (entry["episode_id"] != int(runtime_state["episode_ids"][row])
                        or entry["scan_id"] != int(window["last_scan_id"][row])
                        or entry["persistent_scan_count"] != int(runtime_state["sensor_scan_counts"][row])):
      raise CheckpointError("ground cache source identity differs")


def _entry_accounting(runner) -> dict:
  runtime = runner.runtime
  env = runtime.env
  return {
    "common_step_counter": int(env.common_step_counter),
    "runtime_control_steps": int(runtime._control_steps),
    "episode_ids": runtime.episode_ids.detach().cpu().tolist(),
    "scan_counters": runtime.scan_counters.detach().cpu().tolist(),
    "sensor_scan_counts": runtime.sensor_scan_counts.detach().cpu().tolist(),
    "pipeline_scan_records": len(runtime.pipeline.scan_records),
    "prng_digests": {
      "python": hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
      "numpy": hashlib.sha256(
        np.asarray(np.random.get_state()[1]).tobytes()
      ).hexdigest(),
      "torch_cpu": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
      "torch_cuda": (
        hashlib.sha256(torch.cuda.get_rng_state(runner.device).numpy().tobytes()).hexdigest()
      ),
      "action_generator": hashlib.sha256(
        runner.provider.generator_state().numpy().tobytes()
      ).hexdigest(),
    },
    "note": (
      "state AFTER the fresh official construction (wrapper initial reset + "
      "the ONE takeover first scan + reset-time command sampling): these "
      "construction side effects are recorded and then fully overwritten by "
      "the loaded checkpoint state"
    ),
  }


def _write_persistent_fields(env, arrays: dict, stage: str) -> None:
  for name, tensor in arrays.items():
    target = wp.to_torch(getattr(env.sim.wp_data, name))
    candidate = tensor.detach().to(device=target.device, dtype=target.dtype)
    require_finite(f"physics/{stage}/{name}", candidate)
    with torch.no_grad():
      target.copy_(candidate)


def _clear_collector_feature_cache(collector) -> None:
  """Drop inference rows before checkpoint weights can become stale."""
  feature_cache = getattr(collector, "feature_cache", None)
  if feature_cache is not None:
    feature_cache.clear()


def _apply_full_checkpoint(runner, payload: dict, report: dict) -> None:
  # This is also the safeguard for callers that invoke the checkpoint loader
  # directly instead of going through ConfinedLocomotionRunner.load().
  _clear_collector_feature_cache(getattr(runner, "collector", None))
  runtime = runner.runtime
  env = runtime.env
  device = runtime.device
  applied: list[str] = []

  # 1) network heads/buffers + 18 log_std (strict full key match).
  runner.policy_adapter.load_state_dict(
    payload["algorithm"]["adapter_state_dict"], strict=True
  )
  applied.append("adapter_weights")

  # 2) single Adam (step/exp_avg/exp_avg_sq), LR, counters, shuffle PRNG.
  runner.trainer.load_state_dict(payload["algorithm"]["trainer_state"])
  applied.append("trainer_adam_lr_counters_shuffle")

  # 3) physics: in-place writes -> one forward -> rewrite clobbered persistent
  #    fields -> contact-cache invalidation -> overflow check.
  arrays = payload["physics"]["arrays"]
  _write_persistent_fields(env, arrays, "restore")
  env.sim.forward()
  report["load_operation"]["sim_forward_calls"] += 1
  _write_persistent_fields(env, arrays, "post_forward_rewrite")
  overflow = wp.to_torch(env.sim.wp_data.overflow)
  if bool((overflow != 0).any()):
    raise CheckpointFault(f"physics overflow bits set after restore: {overflow.tolist()}")
  context = getattr(env, "_confined_transition_context", None)
  if context is None:
    raise CheckpointFault("transition context missing on the runtime env")
  context.sensor._invalidate_cache()
  applied.append("physics_inplace_forward_rewrite")

  # 4) env counters and last-step output buffers.
  env_state = payload["env_state"]
  env.common_step_counter = int(env_state["common_step_counter"])
  env._sim_step_counter = int(env_state["sim_step_counter"])
  with torch.no_grad():
    env.episode_length_buf.copy_(env_state["episode_length_buf"].to(device=device))
    env._manual_reset_pending.copy_(
      env_state["manual_reset_pending"].to(device=device)
    )
    for key, attr in (
      ("reset_terminated", "reset_terminated"),
      ("reset_time_outs", "reset_time_outs"),
      ("reset_buf", "reset_buf"),
      ("reward_buf", "reward_buf"),
    ):
      saved = env_state[key]
      if saved is None:
        continue
      live = getattr(env, attr, None)
      restored = saved.to(device=device)
      if live is None:
        setattr(env, attr, restored.clone())  # fresh env never stepped yet
      else:
        live.copy_(restored)
  applied.append("env_counters_buffers")

  # 5) action manager state (prev/raw/clipped/target/processed).
  action = env.action_manager.get_term("residual_pd")
  action_state = payload["action_state"]
  with torch.no_grad():
    action.prev_action.copy_(action_state["prev_action"].to(device=device))
    action.last_raw_action.copy_(action_state["last_raw_action"].to(device=device))
    action.last_clipped_action.copy_(
      action_state["last_clipped_action"].to(device=device)
    )
    action.last_target.copy_(action_state["last_target"].to(device=device))
    action._processed_actions.copy_(
      action_state["processed_actions"].to(device=device)
    )
    action._raw_actions.copy_(action_state["raw_actions"].to(device=device))
  action.control_steps_seen = int(action_state["control_steps_seen"])
  applied.append("action_term_state")

  # 6) foot-air / termination lifecycle caches.
  saved_foot = payload["foot_air_state"]
  if saved_foot is not None:
    foot = getattr(env, "_confined_foot_air_state")
    with torch.no_grad():
      foot.air_time_s.copy_(saved_foot["air_time_s"].to(device=device))
      foot.last_support.copy_(saved_foot["last_support"].to(device=device))
    foot.control_id = saved_foot["control_id"]
    foot.landing = _on_device(saved_foot["landing"], torch.device(device))
    foot.consumed_air_s = _on_device(saved_foot["consumed_air_s"], torch.device(device))
  saved_termination = payload["termination_state"]
  if saved_termination is not None:
    termination = getattr(env, "_confined_termination_state")
    with torch.no_grad():
      termination.low_height_count.copy_(
        saved_termination["low_height_count"].to(device=device)
      )
      termination.tilt_count.copy_(saved_termination["tilt_count"].to(device=device))
      termination.low_height_done.copy_(
        saved_termination["low_height_done"].to(device=device)
      )
      termination.tilt_done.copy_(saved_termination["tilt_done"].to(device=device))
      termination.flip_done.copy_(saved_termination["flip_done"].to(device=device))
      termination.any_fall.copy_(saved_termination["any_fall"].to(device=device))
    termination.control_id = saved_termination["control_id"]
  applied.append("foot_air_termination_state")

  with torch.no_grad():
    for key, live in _manager_tensors(env).items():
      live.copy_(payload["manager_state"][key].to(device=live.device))
  applied.append("manager_lifecycle_tensors")

  # 7) runtime counters, window ring, ground cache, first-action input cache.
  runtime_state = payload["runtime"]
  runtime._control_steps = int(runtime_state["control_steps"])
  with torch.no_grad():
    runtime.episode_ids.copy_(runtime_state["episode_ids"].to(device=device))
    runtime.step_in_episode.copy_(runtime_state["step_in_episode"].to(device=device))
    runtime.scan_counters.copy_(runtime_state["scan_counters"].to(device=device))
    runtime.sensor_scan_counts.copy_(
      runtime_state["sensor_scan_counts"].to(device=device)
    )
    runtime.next_due_scan_step.copy_(
      runtime_state["next_due_scan_step"].to(device=device)
    )
    window_state = runtime_state["window"]
    for name in _WINDOW_TENSOR_FIELDS:
      getattr(runtime.window, f"_{name}").copy_(
        window_state[name].to(device=device)
      )
  runtime.window._seen_episodes = [
    {int(value) for value in seen} for seen in window_state["seen_episodes"]
  ]
  window_keys_state = window_state.get("angle_keys")
  if window_keys_state is not None:
    with torch.no_grad():
      runtime.window._near_angle_keys.copy_(
        window_keys_state["near"].to(device=device)
      )
      runtime.window._far_angle_keys.copy_(
        window_keys_state["far"].to(device=device)
      )
      runtime.window._angle_keys_present.copy_(
        window_keys_state["present"].to(device=device)
      )
  runtime._ground_cache = []
  for entry in runtime_state["ground_cache"]:
    if entry is None:
      runtime._ground_cache.append(None)
      continue
    runtime._ground_cache.append(
      {
        "env_row": int(entry["env_row"]),
        "scan_id": int(entry["scan_id"]),
        "episode_id": int(entry["episode_id"]),
        "persistent_scan_count": int(entry["persistent_scan_count"]),
        "time_s": float(entry["time_s"]),
        **{
          name: entry[name].detach().to(device=device).clone()
          for name in _GROUND_CACHE_TENSOR_FIELDS
        },
      }
    )
  cache_state = runtime_state["input_cache"]
  runtime._input_cache.invalidate()
  if cache_state["prepared"] is not None:
    runtime._input_cache.store(
      int(cache_state["identity"]),
      cache_state["episodes"].to(device=device),
      _deserialize_prepared(cache_state["prepared"], torch.device(device)),
    )
  runtime._current_window_snapshot = (
    None if runtime_state["current_window_snapshot"] is None
    else _deserialize_snapshot(
      runtime_state["current_window_snapshot"], torch.device(device)
    )
  )
  applied.append("runtime_counters_window_ground_input_cache")

  # 8) per-env command streams (values, counters, generator states) and the
  #    P6 perturbation continuation state (named streams + persistent gap
  #    schedule + proprio per-row noise cache; restore never redraws).
  runtime._commands.load_state_dict(runtime_state["commands"])
  applied.append("command_streams")
  perturbations_state = runtime_state.get("perturbations")
  live_perturbations = getattr(runtime, "_perturbations", None)
  if perturbations_state is not None and live_perturbations is not None:
    live_perturbations.load_state_dict(perturbations_state)
    applied.append("perturbation_streams_schedules_caches")

  # 9) action-sampling generator of the provider.
  runner.provider.load_generator_state(payload["prng"]["action_generator"])
  applied.append("action_generator")

  # 10) GLOBAL PRNGs LAST: nothing after this consumes randomness before the
  #     first action (sim.forward and all copies above are deterministic).
  random.setstate(payload["prng"]["python"])
  np.random.set_state(payload["prng"]["numpy"])
  torch.set_rng_state(payload["prng"]["torch_cpu"])
  torch.cuda.set_rng_state(payload["prng"]["torch_cuda"], device=runner.device)
  applied.append("global_prngs")

  # 11) runner counters + one-shot artifact bookkeeping + phase.
  runner_state = payload["runner_state"]
  runner.current_learning_iteration = int(runner_state["current_learning_iteration"])
  runner.total_control_batches = int(runner_state["total_control_batches"])
  runner.total_env_samples = int(runner_state["total_env_samples"])
  runner._artifact_exported = bool(runner_state["artifact_exported"])
  runner._lifecycle_phase = "ready"
  applied.append("runner_counters_phase")

  report["applied"] = applied


def load_full_checkpoint(
  runner, path, *, load_cfg=None, strict=True, map_location=None
) -> dict:
  """Full same-GPU resume of a stable-boundary checkpoint onto ``runner``."""
  if load_cfg not in (None, {}):
    raise CheckpointError(
      f"load(load_cfg={load_cfg!r}) refused: this runner has no actor-only or "
      "partial-load mode; the ONLY non-full mode is the explicit "
      "load_weights() weights path (which never claims resume)"
    )
  if strict is not True:
    raise CheckpointError(
      "load(strict=False) refused: full resume is strict by construction "
      "(weights-only relaxation belongs to load_weights)"
    )
  device = torch.device(runner.device)
  if map_location is not None:
    requested = torch.device(map_location)
    if requested.type != "cuda" or (
      requested.index is not None
      and device.index is not None
      and requested.index != device.index
    ):
      raise CheckpointError(
        f"map_location={map_location!r} refused: full resume is same-CUDA-device "
        "only (cross-device resume is not supported and never claimed; the "
        "weights-only mode has its own explicit rule)"
      )
  phase = getattr(runner, "_lifecycle_phase", None)
  if phase not in ("fresh", "ready"):
    raise CheckpointError(
      f"load refused: runner lifecycle phase is {phase!r} (load accepts a "
      "freshly constructed runner or a completed ready boundary — never a "
      "collect/update/faulted/stopped state)"
    )
  runtime = runner.runtime
  if phase == "ready":
    _require_stable_save_boundary(runner)
  if getattr(runtime, "_faulted", False) or getattr(runtime, "_stopped", False) or getattr(runtime, "_closed", False):
    raise CheckpointError("load refused: the training runtime is faulted/stopped/closed")
  if runner.trainer.faulted:
    raise CheckpointError("load refused: the trainer is faulted")
  if runtime._context.phase != "idle":
    raise CheckpointError(
      f"load refused: transition context phase is {runtime._context.phase!r}"
    )

  target = Path(path)
  if not target.is_file():
    raise CheckpointError(f"checkpoint file not found: {target}")
  entry = _entry_accounting(runner)
  try:
    payload = torch.load(target, map_location=None, weights_only=False)
  except BaseException as error:
    raise CheckpointError(f"checkpoint unreadable: {target} ({error!r})") from error

  _validate_full_payload(runner, payload)

  report = {
    "kind": "confined_locomotion_full_resume_loaded",
    "path": str(target),
    "schema": payload["schema"],
    "created": payload.get("created"),
    "construction_side_effects": entry,
    "load_operation": {
      "env_reset_calls": 0,
      "scan_calls": 0,
      "physics_steps": 0,
      "sim_forward_calls": 0,
      "note": (
        "load itself performs no reset/scan/physics-step; the single "
        "sim.forward is the documented derived-state rebuild (persistent "
        "fields are rewritten afterwards)"
      ),
    },
    "prng_restored": [
      "python", "numpy", "torch_cpu", "torch_cuda",
      "action_generator", "command_generators(per-env)", "optimization_shuffle",
      "perturbation_streams(gap/range/proprio, per-env, when active)",
    ],
    "derived_state_note": (
      "env.obs_buf and the manager observation group are recomputed at the "
      "next step (the training runtime builds its own named observations "
      "from the restored live state); sensordata was rebuilt by the load's "
      "forward from the restored qpos/qvel; the entity-held effort targets "
      "are overwritten by the first substep's PD recompute and are never "
      "read before that"
    ),
    "applied": [],
  }
  try:
    _apply_full_checkpoint(runner, payload, report)
    # Post-apply readback assurance: the restored persistent state matches
    # the saved arrays bit-exactly (the post-forward rewrite held).
    for name, tensor in payload["physics"]["arrays"].items():
      current = wp.to_torch(getattr(runtime.env.sim.wp_data, name))
      restored = tensor.to(device=current.device, dtype=current.dtype)
      if not torch.equal(current, restored):
        raise CheckpointFault(
          f"post-apply readback mismatch on physics.{name}"
        )
  except BaseException as error:
    runner._lifecycle_phase = "faulted"
    diagnostics = {
      "kind": "load_fault",
      "mode": runner.recipe["mode"],
      "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
      "path": str(target),
      "error": repr(error),
      "applied_before_fault": list(report["applied"]),
      "note": (
        "validation passed but the apply stage failed; the runner is frozen "
        "(NOT usable, NOT a resumable state) — feasible in-memory evidence "
        "kept, nothing further was mutated by the reporting itself"
      ),
    }
    try:
      (Path(runner.log_dir) / "load_fault.json").write_text(
        json.dumps(diagnostics, indent=2, default=str)
      )
    except OSError:
      pass
    runner._records.append(diagnostics)
    runtime._save_fault(error, phase="checkpoint/load_apply")
    raise CheckpointFault(
      f"checkpoint apply failed after {report['applied']}: {error!r} "
      "(runner frozen; see load_fault.json)"
    ) from error

  load_record = {
    "kind": "checkpoint_loaded",
    "mode": runner.recipe["mode"],
    "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "path": str(target),
    "current_learning_iteration": runner.current_learning_iteration,
    "total_control_batches": runner.total_control_batches,
    "optimizer_step_count": runner.trainer.optimizer_step_count,
    "model_fingerprint": payload["algorithm"]["model_fingerprint"],
  }
  runner._records.append(load_record)
  try:
    runner._append_jsonl("checkpoint_events.jsonl", load_record)
  except OSError as error:
    runner._lifecycle_phase = "faulted"
    runner._handle_log_io_failure(error, stage="checkpoint_event")
    raise CheckpointFault("checkpoint loaded but event log failed; run frozen") from error
  report["model_fingerprint"] = payload["algorithm"]["model_fingerprint"]
  report["current_learning_iteration"] = runner.current_learning_iteration
  return report


# ═══════════════════════════ weights-only mode ═══════════════════════════


def save_weights_only(path, policy_adapter, *, grid_identity: dict | None = None, reason: str) -> Path:
  """Save ONLY the shared network + buffers + log_std (never resumable)."""
  target = Path(path)
  if target.exists():
    raise CheckpointError(f"refusing to overwrite an existing file: {target}")
  target.parent.mkdir(parents=True, exist_ok=True)
  state = _to_cpu(policy_adapter.state_dict())
  _validate_float_finiteness(state)
  payload = {
    "schema": CHECKPOINT_SCHEMA,
    "kind": CHECKPOINT_KIND_WEIGHTS,
    "resumable": False,
    "task": TASK_ID,
    "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "reason": str(reason),
    "identity": {
      "source_sha256": _source_identity(),
      "dependencies": _dependency_versions(),
      "distribution_identity_frozen": str(policy_adapter.identity_frozen),
      "network_state_keys": sorted(state),
      "ground_samples": int(policy_adapter.network.ground_samples or 0),
      "grid_identity": grid_identity,
    },
    "state": state,
  }
  tmp = target.with_name(f"{target.name}.tmp-{os.getpid()}")
  try:
    torch.save(payload, tmp)
    os.replace(tmp, target)
  except BaseException as error:
    try:
      tmp.unlink(missing_ok=True)
    except OSError:
      pass
    raise CheckpointFault(f"atomic weights save to {target} failed: {error!r}") from error
  return target


def load_weights(
  path,
  *,
  network: ConfinedLocomotionNetwork | None = None,
  provider=None,
  expected_grid_identity: dict | None = None,
  allow_source_change: bool = False,
) -> dict:
  """Load ALL shared network keys (+ optional 18-dim log_std) — weights ONLY.

  No Adam, physics, PRNG or lifecycle state is touched and continuation is
  never claimed (``resumable`` stays False in the report). The env count may
  differ from the checkpoint's. Grid positions/sizes are strictly checked
  whenever ``expected_grid_identity`` is supplied; the full model key set
  must match exactly (not just the actor head). A source change is accepted
  only under the explicit independent rule ``allow_source_change=True``
  (recorded; never available to full resume).
  """
  if not isinstance(network, ConfinedLocomotionNetwork):
    raise CheckpointError(
      "load_weights requires an explicitly constructed ConfinedLocomotionNetwork "
      "(the caller owns its configuration/device; nothing is created silently)"
    )
  target = Path(path)
  if not target.is_file():
    raise CheckpointError(f"weights file not found: {target}")
  try:
    payload = torch.load(target, map_location="cpu", weights_only=False)
  except BaseException as error:
    raise CheckpointError(f"weights file unreadable: {target} ({error!r})") from error
  if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA:
    raise CheckpointError(
      f"weights file schema mismatch: {payload.get('schema') if isinstance(payload, dict) else type(payload).__name__}"
    )
  kind = payload.get("kind")
  if kind not in (CHECKPOINT_KIND_WEIGHTS, CHECKPOINT_KIND_FULL):
    raise CheckpointError(
      f"kind {kind!r} carries no loadable weights (fault/stop scenes and "
      "NON_RESUMABLE artifacts are never weight sources)"
    )
  state = (
    payload["state"] if kind == CHECKPOINT_KIND_WEIGHTS
    else payload["algorithm"]["adapter_state_dict"]
  )
  identity = payload.get("identity") or {}
  if kind == CHECKPOINT_KIND_WEIGHTS:
    saved_m = identity.get("ground_samples")
  else:
    saved_m = (
      ((payload.get("identity") or {}).get("grid") or {})
      .get("identity", {}).get("num_points")
    )
  if saved_m is None or int(saved_m) != int(network.ground_samples or -1):
    raise CheckpointError(
      f"ground head size mismatch: file M={saved_m} vs network "
      f"M={network.ground_samples} (grid M must match strictly)"
    )
  if expected_grid_identity is not None:
    saved_grid = (
      identity.get("grid_identity")
      if kind == CHECKPOINT_KIND_WEIGHTS
      else ((payload.get("identity") or {}).get("grid") or {}).get("identity")
    )
    if saved_grid is None or saved_grid != expected_grid_identity:
      raise CheckpointError(
        "grid identity mismatch (positions/sizes/resolution must match "
        f"strictly): file {saved_grid} != expected {expected_grid_identity}"
      )
  current_source = _source_identity()
  if identity.get("source_sha256") != current_source and not allow_source_change:
    raise CheckpointError(
      "source files changed since the checkpoint (weights-only loading under "
      "external code changes requires the explicit allow_source_change=True "
      "rule; full resume NEVER accepts source changes)"
    )

  network_keys = {
    key.removeprefix("network."): value for key, value in state.items()
    if key.startswith("network.")
  }
  live_keys = set(network.state_dict())
  if set(network_keys) != live_keys:
    raise CheckpointError(
      f"network key set mismatch (strict full match, not actor-only): "
      f"missing {sorted(live_keys - set(network_keys))}, extra "
      f"{sorted(set(network_keys) - live_keys)}"
    )
  _validate_float_finiteness(network_keys)
  _require_same_tensors("network", network_keys, network.state_dict())

  log_std_key = "distribution_provider.log_std"
  log_std_report = "absent"
  if set(state) != {"network." + key for key in live_keys} | {log_std_key}:
    raise CheckpointError("adapter state keys include unexpected or missing fields")
  if provider is not None:
    if log_std_key not in state:
      raise CheckpointError(
        "provider given but the file carries no distribution_provider.log_std"
      )
    log_std = state[log_std_key]
    if tuple(log_std.shape) != tuple(provider.log_std.shape):
      raise CheckpointError("log_std shape mismatch")
    _validate_float_finiteness({"log_std": log_std})
    if log_std.dtype != provider.log_std.dtype:
      raise CheckpointError("log_std dtype mismatch")
    if not bool(torch.isfinite(log_std.exp()).all()) or not bool((log_std.exp() > 0).all()):
      raise CheckpointError("saved log_std does not define a finite positive Normal std")
  # A caller that owns a collector clears it before entering this loader.
  # Direct network-only callers have no cache owner here; cached rows are
  # guarded by the network parameter-version signature on their next use.
  network.load_state_dict(network_keys, strict=True)
  if provider is not None:
    with torch.no_grad():
      provider.log_std.data.copy_(log_std.to(device=provider.log_std.device))
    log_std_report = "loaded"
  elif log_std_key in state:
    log_std_report = "present_not_loaded(no provider given)"

  return {
    "kind": "confined_locomotion_weights_loaded",
    "resumable": False,
    "source_kind": kind,
    "path": str(target),
    "network_keys": sorted(network_keys),
    "log_std": log_std_report,
    "source_change_allowed": bool(allow_source_change),
    "source_matches_current": identity.get("source_sha256") == current_source,
    "note": (
      "weights-only load: no Adam/physics/PRNG/lifecycle restored; this is "
      "NOT a resume and never claims one"
    ),
  }
