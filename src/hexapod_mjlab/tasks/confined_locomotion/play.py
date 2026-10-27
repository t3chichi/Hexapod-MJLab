"""Task-local checkpoint play: real mean closed loop with manual control.

This is the CONFIRMED play entry of ``El-4090-Confined-Locomotion-Flat``
(平地学习与基准测量.md §2 "Play运行规则"): it loads an EXPLICIT checkpoint
into a REAL simulation and drives it with the network's ACTION MEAN — no
optimizer, no PPO, no Normal provider, no action sampling, no parameter
updates, ever.

Construction chain (each component created exactly once, nothing replaced):

    make_play_env_cfg(num_envs, seed)          # NEW independent cfg object:
        the training factory's output with the ONE episode_timeout term
        REMOVED (all 17 rewards, the three formal fall terms, auto_reset=
        False, Null commands, nan_policy='error' stay; episode_length_s
        stays 20 s / max_episode_length 1000 as RECORDED metadata only)
    -> ManagerBasedRlEnv(cfg, device)          # official env
    -> RslRlVecEnvWrapper(env, clip_actions=None)   # ONE initial reset
    -> TrainingRuntime(..., mode="play")       # manual commands
                                                # zeroed BEFORE the takeover
                                                # sample_reset; formal-fall
                                                # save+pause; explicit manual
                                                # reset; 20 s truncation
                                                # structurally absent
    -> ConfinedLocomotionNetwork(ELU, init_fn=None, envelope_mode="raw",
                                 ground_samples=ground_grid.num_points)
    -> checkpoint.load_weights(network=..., provider=None,
                               expected_grid_identity=...)  # overwritten by
        the checkpoint's COMPLETE four-head + buffer key set (never a
        silent random initialization; log_std stays unloaded — no provider
        exists here). network.eval() + forward under torch.no_grad().

Checkpoint compatibility is checked strictly BEFORE anything runs: the
current checkpoint schema, kind = weights-only or FULL resume (a full current
training checkpoint may
serve as a WEIGHTS-ONLY source — fresh T1 physics, a new play seed and the
manual command lifecycle; this is NOT a training continuation and never
restores Adam/physics/RNG/lifecycle), task id, ground-head M, grid identity
(same-M-moved-positions included), compiled-model structure, sim, lidar,
rewards and termination semantics. Source hashes are not a compatibility
condition. mean_dev artifacts, fault/stop scenes and v1 files are rejected
outright.

Ownership/threading: ONE owner loop advances everything (``step_once``:
prepare at the pre-action identity → single no-grad forward →
``runtime.step(action_mean)``); the Viser viewer's callbacks only submit
intents. Pausing, layer toggles, env switching and picking re-render the
RETAINED snapshot — they never step physics, launch scans, redraw noise or
re-run the network. Manual command changes are validated FIRST (an illegal
command is a plain rejected input, never a physics fault) and then applied
through ``TrainingRuntime.set_manual_command`` (input-cache invalidation at
the same identity; per-row noise stays cached).

Lifecycle: falls decided ONLY by the confirmed formal gates save the scene
and PAUSE (``step`` refuses; the explicit full-episode manual reset is the
only continuation — real T1 reset, cleared history/prev-action/fall
counters, unique first scan, zeroed manual command); faults save and FREEZE
(step AND reset refused forever); collisions never pause anything; the 20 s
training truncation never fires in play.

Evidence: ``play_recipe.json`` records the checkpoint file hash, the saved
identity, the weight-load report, the actual model/source/dependency/
device/precision, env/grid/lidar/perturbation/seed/command configuration;
``write_terminal_record`` leaves the final state (counters, window, sensing,
command, the last input/output/action/labels and the reason) with an
explicit NON_RESUMABLE statement. Nothing here is a checkpoint and nothing
claims a training continuation.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from hexapod_mjlab.rl.confined_locomotion.integration import (
  PreparedControlInputs,
)
from hexapod_mjlab.rl.confined_locomotion.inference import PointCloudFeatureCache
from hexapod_mjlab.rl.confined_locomotion.network import ConfinedLocomotionNetwork

from .rl.checkpoint import (
  CHECKPOINT_KIND_FULL,
  CHECKPOINT_KIND_WEIGHTS,
  CHECKPOINT_SCHEMA,
  _compiled_model_identity,
  _lidar_identity,
  load_weights,
)
from .commands import TrainingCommandScheduler
from hexapod_mjlab.common.mdp.terminations import get_termination_state
from .observation import clone_prepared
from hexapod_mjlab.common.perturbations import ObservationPerturbationCfg, preset_by_name
from hexapod_mjlab.common.transition import require_finite
from .flat_env_cfg import make_training_env_cfg
from .ground import GroundGridCfg, VALIDITY_GEOMETRY_ONLY_FLAT_TRAINING
from .runtime import TrainingRuntime, TrainingRuntimeFrozenError
from .visualization.play import PLAY_SNAPSHOT_SCHEMA, resolve_point_sources

__all__ = [
  "COMMAND_RANGES",
  "PLAY_TASK_NOTE",
  "PlayConfigError",
  "PlaySession",
  "make_play_env_cfg",
  "peek_checkpoint_header",
  "resolve_play_perturbation_cfg",
]

#: Human-readable identity of this entry (recorded in every artifact).
PLAY_TASK_NOTE = "El-4090-Confined-Locomotion-Flat/PlaySession"

#: Legal manual command ranges (the confirmed scheduler constants).
COMMAND_RANGES = (
  TrainingCommandScheduler.RANGE_VX,
  TrainingCommandScheduler.RANGE_VY,
  TrainingCommandScheduler.RANGE_OMEGA,
)


class PlayConfigError(RuntimeError):
  """Invalid play configuration/checkpoint/request (never a physics fault)."""


# ═════════════════════════ play env cfg ════════════════════════════════════


def make_play_env_cfg(*, num_envs: int = 1, seed: int | None = None):
  """Independent PLAY env config (registered/training defaults untouched).

  Builds a NEW object via the training factory (formal rewards /
  terminations / Null commands / auto_reset=False / nan_policy='error') and
  removes the ONE ``episode_timeout`` truncation term from it — the three
  formal fall terms and every reward/MDP term stay. ``episode_length_s``
  stays 20 s (``max_episode_length`` 1000) as RECORDED metadata: the play
  runtime refuses any truncation flag, so the timeout is structurally
  inert rather than "hacked" with a giant episode length.
  """
  cfg = make_training_env_cfg(
    num_envs=int(num_envs), episode_length_s=20.0, seed=seed
  )
  removed = cfg.terminations.pop("episode_timeout", None)
  if removed is None or not bool(getattr(removed, "time_out", False)):
    raise PlayConfigError(
      "the training env cfg did not carry the episode_timeout time_out term; "
      "the play factory refuses to proceed on an unexpected config shape"
    )
  remaining_time_out = [
    name for name, term in cfg.terminations.items()
    if bool(getattr(term, "time_out", False))
  ]
  if remaining_time_out:
    raise PlayConfigError(
      f"unexpected time_out terms remain after removal: {remaining_time_out}"
    )
  return cfg


# ═════════════════════════ checkpoint header peek ══════════════════════════


def _file_sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1 << 20), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _cpu_tree(value):
  """Copy a bounded runtime/display tree to CPU without numeric rounding."""
  from dataclasses import fields, is_dataclass

  if torch.is_tensor(value):
    return value.detach().cpu().clone()
  if isinstance(value, np.ndarray):
    return value.copy()
  if is_dataclass(value):
    return {field.name: _cpu_tree(getattr(value, field.name)) for field in fields(value)}
  if isinstance(value, dict):
    return {key: _cpu_tree(item) for key, item in value.items()}
  if isinstance(value, tuple):
    return tuple(_cpu_tree(item) for item in value)
  if isinstance(value, list):
    return [_cpu_tree(item) for item in value]
  return value


def _slice_batch_payload(payload: dict, row: int, batch_size: int) -> dict:
  """Select one env's row-shaped labels while preserving shared metadata."""
  return {
    key: (
      value[row].copy()
      if isinstance(value, np.ndarray) and value.ndim > 0
      and value.shape[0] == batch_size
      else value[row]
      if isinstance(value, list) and len(value) == batch_size
      else value
    )
    for key, value in payload.items()
  }


def _compare_compiled_model_identity(saved: dict, current: dict) -> tuple[list[str], bool]:
  """Compare a FULL identity across env counts, normalizing repeated q_ref.

  ``q_ref`` is stored as B×18 and the overall digest includes that batch. A
  weights-only play may use another B, so it compares one identical T1 row
  after verifying every row in both identities repeats that same reference;
  all compiled structure/binary fields still compare exactly.
  """
  fields = (set(saved) | set(current)) - {"q_ref", "digest"}
  mismatches = [
    key for key in sorted(fields) if saved.get(key) != current.get(key)
  ]
  try:
    saved_q_ref = np.asarray(saved["q_ref"], dtype=np.float64)
    current_q_ref = np.asarray(current["q_ref"], dtype=np.float64)
  except (KeyError, TypeError, ValueError):
    return mismatches, False
  q_ref_valid = (
    saved_q_ref.ndim == 2 and current_q_ref.ndim == 2
    and saved_q_ref.shape[1:] == (18,) and current_q_ref.shape[1:] == (18,)
    and saved_q_ref.shape[0] > 0 and current_q_ref.shape[0] > 0
    and np.array_equal(saved_q_ref, np.broadcast_to(saved_q_ref[0], saved_q_ref.shape))
    and np.array_equal(current_q_ref, np.broadcast_to(current_q_ref[0], current_q_ref.shape))
    and np.array_equal(saved_q_ref[0], current_q_ref[0])
  )
  return mismatches, bool(q_ref_valid)


def peek_checkpoint_header(path) -> dict:
  """Read a checkpoint's IDENTITY header (schema/kind/task/M/grid).

  Loads the file once on CPU (the same payload ``load_weights`` consumes
  later) and returns the identity fields the play construction needs plus
  the checkpoint FILE sha256 for the recipe. mean_dev artifacts, fault/stop
  scenes and any non-current-schema payload are rejected here — they are never
  checkpoint weight sources.
  """
  target = Path(path)
  if not target.is_file():
    raise PlayConfigError(f"checkpoint file not found: {target}")
  try:
    payload = torch.load(target, map_location="cpu", weights_only=False)
  except BaseException as error:  # unreadable file is a config error here
    raise PlayConfigError(f"checkpoint unreadable: {target} ({error!r})") from error
  if not isinstance(payload, dict):
    raise PlayConfigError(
      f"checkpoint payload is not a dict: {type(payload).__name__}"
    )
  if payload.get("schema") != CHECKPOINT_SCHEMA:
    raise PlayConfigError(
      f"checkpoint schema {payload.get('schema')!r} != {CHECKPOINT_SCHEMA!r}; "
      "older-schema files and non-checkpoint artifacts (mean_dev/fault scenes) are "
      "rejected outright — no migration, no silent fallback"
    )
  kind = payload.get("kind")
  if kind not in (CHECKPOINT_KIND_WEIGHTS, CHECKPOINT_KIND_FULL):
    raise PlayConfigError(
      f"checkpoint kind {kind!r} carries no loadable weights (fault/stop "
      "scenes and NON_RESUMABLE artifacts are never weight sources)"
    )
  identity = payload.get("identity") or {}
  if kind == CHECKPOINT_KIND_WEIGHTS:
    ground_samples = identity.get("ground_samples")
    grid_identity = identity.get("grid_identity")
    grid_identity_hash = None
  else:
    grid_block = identity.get("grid") or {}
    grid_identity = grid_block.get("identity")
    grid_identity_hash = grid_block.get("identity_hash")
    ground_samples = (grid_identity or {}).get("num_points")
  if ground_samples is None or not isinstance(ground_samples, int):
    raise PlayConfigError(
      f"checkpoint carries no integer ground-head size M: {ground_samples!r}"
    )
  return {
    "path": str(target),
    "file_sha256": _file_sha256(target),
    "schema": payload.get("schema"),
    "kind": kind,
    "task": payload.get("task"),
    "created": payload.get("created"),
    "ground_samples": int(ground_samples),
    "grid_identity": grid_identity,
    "grid_identity_hash": grid_identity_hash,
    "dependencies": identity.get("dependencies"),
    "saved_identity": identity if kind == CHECKPOINT_KIND_FULL else None,
    "source_kind_note": (
      "weights-only file" if kind == CHECKPOINT_KIND_WEIGHTS
      else "FULL current-schema training checkpoint used WEIGHTS-ONLY (fresh T1 physics, "
      "new play seed, manual command lifecycle; no Adam/physics/RNG/"
      "lifecycle restored — NOT a training continuation)"
    ),
  }


def ground_grid_from_checkpoint_header(header: dict) -> GroundGridCfg:
  """Recover the ORIGINAL grid exactly for checkpoint-based Play.

  No coordinate regeneration or default-preset lookup: even same-M shifted
  grids change what a ground-head output means. The returned identity must
  equal the saved identity byte-for-value, including tolerance and semantics.
  PlaySession still independently checks the header against its supplied grid.
  """
  if not isinstance(header, dict):
    raise PlayConfigError("checkpoint header must be a dictionary")
  identity = header.get("grid_identity")
  if not isinstance(identity, dict):
    raise PlayConfigError("checkpoint is missing its complete ground-grid identity")
  try:
    grid = GroundGridCfg(
      x_coords=tuple(identity["x_coords"]),
      y_coords=tuple(identity["y_coords"]),
      validity_mode=identity["validity_mode"],
      parallel_tol=identity["parallel_tol"],
      body_xyz_semantics=identity["body_xyz_semantics"],
    )
  except (KeyError, TypeError, ValueError) as error:
    raise PlayConfigError(f"checkpoint ground-grid cannot be reconstructed: {error}") from error
  if grid.identity() != identity:
    raise PlayConfigError("checkpoint ground-grid identity has unknown or changed fields")
  if header.get("ground_samples") != grid.num_points:
    raise PlayConfigError("checkpoint ground-head size differs from its saved grid")
  saved_hash = header.get("grid_identity_hash")
  if saved_hash is not None and saved_hash != grid.identity_hash():
    raise PlayConfigError("checkpoint ground-grid identity hash mismatch")
  return grid


def resolve_play_perturbation_cfg(
  preset_name: str,
) -> ObservationPerturbationCfg | None:
  """Named observation preset → typed cfg (None/'clean' = the clean path)."""
  if preset_name in ("", "clean", "None", "none"):
    return None
  return preset_by_name(preset_name)


def _dependency_record() -> dict:
  import importlib.metadata as importlib_metadata

  import mjlab
  import mujoco
  import warp as wp

  return {
    "rsl_rl": importlib_metadata.version("rsl-rl-lib"),
    "mujoco_warp": importlib_metadata.version("mujoco-warp"),
    "torch": str(torch.__version__),
    "mjlab": getattr(mjlab, "__version__", None) or importlib_metadata.version("mjlab"),
    "mujoco": str(mujoco.__version__),
    "warp": str(wp.config.version),
    "numpy": str(np.__version__),
  }


def _torch_precision_record() -> dict:
  matmul = getattr(torch.backends.cuda.matmul, "fp32_precision", None)
  cudnn = getattr(torch.backends.cudnn, "fp32_precision", None)
  rnn = getattr(getattr(torch.backends.cudnn, "rnn", None), "fp32_precision", None)
  if matmul is None:
    matmul = (
      "allow_tf32=True" if torch.backends.cuda.matmul.allow_tf32
      else "allow_tf32=False"
    )
  if cudnn is None:
    cudnn = (
      "allow_tf32=True" if torch.backends.cudnn.allow_tf32
      else "allow_tf32=False"
    )
  return {
    "cuda_matmul_fp32_precision": str(matmul),
    "cudnn_fp32_precision": str(cudnn),
    "cudnn_rnn_fp32_precision": str(cudnn if rnn is None else rnn),
    "tf32_disabled": all(
      str(value) in {"ieee", "allow_tf32=False"}
      for value in (matmul, cudnn, cudnn if rnn is None else rnn)
    ),
    "default_dtype": str(torch.get_default_dtype()),
    "amp": "none (no autocast/GradScaler anywhere in play)",
  }


@dataclass
class _DisplayRetention:
  """The retained o_k display frame (rebuilt once per control step).

  Always FULL-BATCH numpy copies: ``pose`` holds (B,3)/(B,4)/(B,nq) arrays,
  ``output`` holds (B,·) arrays; per-row slicing happens only when a
  snapshot is assembled for a selected env row.
  """

  input_clone: PreparedControlInputs
  window: object
  angle_keys: object
  labels_cpu: dict
  output: dict | None
  pose: dict
  contacts: dict | None
  input_control_step: int
  input_episode_ids: np.ndarray
  input_step_in_episode: np.ndarray
  input_persistent_scan_counts: np.ndarray
  input_capture_entries: dict[int, list]
  pose_control_step: int
  pose_episode_ids: np.ndarray
  pose_step_in_episode: np.ndarray
  pose_persistent_scan_counts: np.ndarray


# ═════════════════════════ the play session ════════════════════════════════


class PlaySession:
  """Owns the real play lifecycle of one checkpoint in one env batch.

  The SINGLE owner of stepping: ``step_once`` performs prepare → one
  no-grad mean forward → ``TrainingRuntime.step``. Viewers/CLI threads only
  submit intents; every public mutator here validates plain input errors
  BEFORE touching the runtime (illegal values are rejected, never turned
  into physics faults).
  """

  def __init__(
    self,
    *,
    checkpoint,
    output_dir,
    seed: int,
    ground_grid: GroundGridCfg,
    device: str = "cuda:0",
    num_envs: int = 1,
    perturbation_preset: str = "clean",
    initial_command: tuple[float, float, float] = (0.0, 0.0, 0.0),
    env_seed: int | None = None,
    use_feature_cache: bool = True,
  ) -> None:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
    from mjlab.utils.torch import configure_torch_backends

    self.checkpoint_path = Path(checkpoint)
    if not isinstance(ground_grid, GroundGridCfg):
      raise PlayConfigError("ground_grid must be an explicitly supplied GroundGridCfg")
    if ground_grid.validity_mode != VALIDITY_GEOMETRY_ONLY_FLAT_TRAINING:
      raise PlayConfigError(
        "task play requires ground_grid validity_mode="
        f"{VALIDITY_GEOMETRY_ONLY_FLAT_TRAINING!r}, got {ground_grid.validity_mode!r}"
      )
    if isinstance(seed, bool) or not isinstance(seed, int):
      raise PlayConfigError(f"play seed must be an explicit int, got {seed!r}")
    self.seed = int(seed)
    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
      raise PlayConfigError(f"num_envs must be a positive int, got {num_envs!r}")
    self.num_envs = int(num_envs)
    self.perturbation_preset = str(perturbation_preset)
    if not isinstance(use_feature_cache, bool):
      raise PlayConfigError(
        f"use_feature_cache must be an explicit bool, got {use_feature_cache!r}"
      )
    self.use_feature_cache = use_feature_cache
    self._validated_initial_command = self._validate_command(initial_command)

    device_obj = torch.device(device)

    # ── checkpoint identity FIRST (strict, before anything is constructed) ──
    self.header = peek_checkpoint_header(self.checkpoint_path)
    from .flat_env_cfg import TASK_ID

    if self.header["task"] != TASK_ID:
      raise PlayConfigError(
        f"checkpoint task {self.header['task']!r} != {TASK_ID!r}"
      )
    self.grid_cfg = ground_grid
    if int(self.header["ground_samples"]) != int(self.grid_cfg.num_points):
      raise PlayConfigError(
        f"checkpoint ground-head M={self.header['ground_samples']} != the "
        f"explicit grid M={self.grid_cfg.num_points}"
      )
    if self.header["grid_identity"] != self.grid_cfg.identity():
      raise PlayConfigError(
        "checkpoint grid identity differs from the explicit ground_grid "
        "(including same-M moved positions)"
      )
    expected_grid_hash = self.grid_cfg.identity_hash()
    if (
      self.header["kind"] == CHECKPOINT_KIND_FULL
      and self.header["grid_identity_hash"] != expected_grid_hash
    ):
      raise PlayConfigError(
        "checkpoint ground_grid identity hash differs from the explicit grid"
      )
    self.perturbation_cfg = resolve_play_perturbation_cfg(self.perturbation_preset)

    if device_obj.type != "cuda" or not torch.cuda.is_available():
      raise PlayConfigError(
        f"task play requires actual CUDA, got device={device!r} "
        "(CPU construction claims nothing about the accepted GPU path)"
      )
    if device_obj.index is None:
      device_obj = torch.device(f"cuda:{torch.cuda.current_device()}")
    self.device = device_obj
    self.started_at = time.monotonic()
    self.created_at = time.strftime("%Y-%m-%dT%H:%M:%S")

    # TF32 OFF before the network exists (same rule as the accepted runner).
    configure_torch_backends(allow_tf32=False)
    # Recent torch backends expose an independent RNN precision override.
    rnn_backend = getattr(torch.backends.cudnn, "rnn", None)
    if rnn_backend is not None and hasattr(rnn_backend, "fp32_precision"):
      rnn_backend.fp32_precision = "ieee"
    self.precision_record = _torch_precision_record()

    self.output_dir = Path(output_dir)
    self.output_dir.mkdir(parents=True, exist_ok=False)
    self._terminal_written = False
    self._play_event_counter = 0
    self._persisted_fault_event = False
    self._attempted_policy = None
    self._closed_env = False

    # ── the real chain: env → wrapper (ONE reset) → play runtime ────────────
    env_cfg = make_play_env_cfg(num_envs=self.num_envs, seed=env_seed)
    self.env_cfg = env_cfg
    self.env = ManagerBasedRlEnv(cfg=env_cfg, device=str(self.device))
    try:
      self.wrapper = RslRlVecEnvWrapper(self.env, clip_actions=None)
      self.runtime = TrainingRuntime(
        self.wrapper,
        grid_cfg=self.grid_cfg,
        seed=self.seed,
        output_dir=self.output_dir / "runtime",
        mode="play",
        perturbation_cfg=self.perturbation_cfg,
        scan_audit_bound_per_env=1,
      )
    except BaseException:
      self.env.close()
      self._closed_env = True
      raise

    try:
      self.full_identity_report = self._validate_full_checkpoint_identity()
    except BaseException:
      self.close()
      raise

    # ── the network: created once, then FULLY overwritten by the checkpoint ─
    try:
      torch.manual_seed(self.seed)  # the one explicit transient initialization
      self.network = ConfinedLocomotionNetwork(
        mlp_activation=torch.nn.ELU(),
        init_fn=None,
        envelope_mode="raw",
        ground_samples=int(self.grid_cfg.num_points),
      ).to(self.device)
      self.weight_report = load_weights(
        self.checkpoint_path,
        network=self.network,
        provider=None,  # mean-only play: no Normal provider is ever created
        expected_grid_identity=self.grid_cfg.identity(),
      )
    except BaseException:
      self.close()
      raise
    self.network.eval()
    self.feature_cache = (
      PointCloudFeatureCache(self.network) if self.use_feature_cache else None
    )
    if self.network.mapping.mode != "raw":
      self.close()
      raise PlayConfigError("the checkpoint network must use raw envelope mapping")

    # ── optional explicit initial command (validated FIRST) ─────────────────
    self.initial_command_record = None
    try:
      if self._validated_initial_command != (0.0, 0.0, 0.0):
        self.initial_command_record = self.set_command(
          *self._validated_initial_command
        )
    except BaseException:
      self.close()
      raise

    self._display: _DisplayRetention | None = None
    try:
      self.refresh_idle_snapshot(reason="construction")
      self._write_recipe()
    except BaseException:
      self.close()
      raise

  # ── configuration / identity ──────────────────────────────────────────────

  @property
  def play_source_scans(self) -> dict[int, list]:
    """The runtime's bounded per-env provenance rings (read-only access)."""
    if self.runtime._play_source_scans is None:
      raise RuntimeError("play source scans exist only in play mode")
    return self.runtime._play_source_scans

  def _write_recipe(self) -> None:
    import dataclasses

    recipe = {
      "schema": "confined_locomotion.play_recipe/v2",
      "mode": "play",
      "task_note": PLAY_TASK_NOTE,
      "created": self.created_at,
      "checkpoint": self.header,
      "checkpoint_compatibility": self.full_identity_report,
      "weight_load_report": self.weight_report,
      "network": {
        "class": type(self.network).__name__,
        "mlp_activation": "ELU",
        "init_fn": None,
        "envelope_mode": "raw",
        "ground_samples": int(self.grid_cfg.num_points),
        "mode": "eval (mean-only; no Normal provider, no sampling)",
      },
      "construction_chain": (
        "make_play_env_cfg (episode_timeout removed on a NEW object) → "
        "ManagerBasedRlEnv → RslRlVecEnvWrapper(clip_actions=None) [one "
        "initial reset] → TrainingRuntime(mode='play') → "
        "ConfinedLocomotionNetwork → checkpoint.load_weights(network, "
        "provider=None, expected_grid_identity)"
      ),
      "env": {
        "num_envs": self.num_envs,
        "seed": self.env_cfg.seed,
        "auto_reset": False,
        "episode_length_s_recorded": float(self.env_cfg.episode_length_s),
        "max_episode_length_recorded": int(self.env.max_episode_length),
        "timeout_note": (
          "episode_timeout term REMOVED on this independent play object; the "
          "20 s/1000-step values are recorded metadata and the runtime "
          "refuses any truncation flag (play never resets on time)"
        ),
        "terminations": sorted(self.env_cfg.terminations),
        "commands": "Null manager; manual scheduler rows (allow_manual=True)",
      },
      "grid": {
        "declaration": dataclasses.asdict(self.grid_cfg),
        "identity": self.grid_cfg.identity(),
        "identity_hash": self.grid_cfg.identity_hash(),
        "note": "explicit complete GroundGridCfg used by the checkpoint",
      },
      "lidar": {
        "airy_rays": int(self.runtime.lidar_cfg.airy_n_azimuth)
        * int(self.runtime.lidar_cfg.airy_n_elevation),
        "near_points": int(self.runtime.lidar_cfg.near_points),
        "far_points_per_scan": int(self.runtime.lidar_cfg.far_points_per_scan),
        "far_window_slots": int(self.runtime.lidar_cfg.far_window_slots),
      },
      "display_retention": {
        "full_scan_frames_per_env": 1,
        "sampled_source_metadata_scans_per_env": 10,
        "note": (
          "one current raw frame supports the optional pool layer; older "
          "provenance keeps sampled raw ids/masks/keys/no-hit only"
        ),
      },
      "observation_perturbations": (
        None if self.perturbation_cfg is None
        else self.perturbation_cfg.identity()
      ),
      "seed": self.seed,
      "device": {
        "resolved": str(self.device),
        "cuda_name": torch.cuda.get_device_name(self.device),
      },
      "torch_precision": self.precision_record,
      "dependencies": _dependency_record(),
      "initial_command": self.initial_command_record,
      "optimizer": False,
      "sampling": False,
      "resumable": False,
      "t1_pd_note": (
        "T1 fixed witness reset + q_ref PD (kp=130, kd=2, ±80 N·m, 0.25 rad "
        "residual, ±100 clip) — unchanged accepted execution seam"
      ),
      "weights_only_compatibility_assumption": (
        "weights-only files do not carry saved compiled-model/PD/sim/lidar "
        "identity fields; compatibility is limited to exact task, M/grid, "
        "source rule (unless explicitly relaxed), full network key/shape/dtype/"
        "finiteness checks, and the saved raw mapping metadata. Current live "
        "T1/PD/sim/lidar identity is recorded separately. No absent checkpoint "
        "metadata is inferred or fabricated."
        if self.header["kind"] == CHECKPOINT_KIND_WEIGHTS
        else "FULL model/PD/T1/sim/lidar/grid identity was compared strictly; "
        "the file is still loaded as weights-only and no training state is restored."
      ),
      "current_runtime_identity": self._current_runtime_identity(),
      "network_mapping_identity": {
        "mode": self.network.mapping.mode,
        "p0": self.network.mapping.p0.detach().cpu().tolist(),
        "h_ref": float(self.network.mapping.h_ref.detach().cpu()),
      },
    }
    (self.output_dir / "play_recipe.json").write_text(
      json.dumps(recipe, indent=2, default=str)
    )

  # ── manual command (validated FIRST; plain errors never fault physics) ────

  def set_command(self, vx: float, vy: float, omega: float) -> dict:
    """Set the manual body command on ALL rows (legal ranges enforced).

    Out-of-range/non-finite values are rejected here WITHOUT touching the
    runtime — a harmless illegal instruction must not fabricate a fault.
    """
    values = self._validate_command((vx, vy, omega))
    rows = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
    triplet = torch.tensor([values], dtype=torch.float32, device=self.device)
    return self.runtime.set_manual_command(
      rows, triplet.expand(self.num_envs, 3).contiguous()
    )

  @staticmethod
  def _validate_command(command) -> tuple[float, float, float]:
    try:
      raw = tuple(command)
    except TypeError as error:
      raise PlayConfigError("manual command must contain exactly three values") from error
    if len(raw) != 3:
      raise PlayConfigError("manual command must contain exactly three values")
    try:
      values = tuple(float(value) for value in raw)
    except (TypeError, ValueError, OverflowError) as error:
      raise PlayConfigError("manual command values must be numeric") from error
    for name, value, bounds in zip(("vx", "vy", "omega"), values, COMMAND_RANGES):
      if not np.isfinite(value) or not (bounds[0] <= value <= bounds[1]):
        raise PlayConfigError(
          f"manual {name}={value!r} outside the legal range {bounds} "
          "(rejected before touching the runtime)"
        )
    return values

  def _current_runtime_identity(self) -> dict:
    from .flat_env_cfg import DECIMATION, SIM_TIMESTEP_S
    from hexapod_mjlab.common.mdp.rewards import CONTROL_DT_S, TRAINING_REWARD_WEIGHTS
    from hexapod_mjlab.common.mdp.terminations import (
      FLIP_THRESHOLD_COS,
      LOW_HEIGHT_PERSISTENCE_STEPS,
      LOW_HEIGHT_THRESHOLD_M,
      TILT_THRESHOLD_COS,
      TRAINING_TERMINATION_TIME_OUT_FLAGS,
    )

    action = self.env.action_manager.get_term("residual_pd")
    sim = {
      "timestep": float(self.env.cfg.sim.mujoco.timestep),
      "decimation": int(self.env.cfg.decimation),
      "step_dt": float(self.env.step_dt),
      "control_dt": float(CONTROL_DT_S),
      "sim_timestep_constant": float(SIM_TIMESTEP_S),
      "decimation_constant": int(DECIMATION),
      "episode_length_s": float(self.env.cfg.episode_length_s),
      "max_episode_length": int(self.env.max_episode_length),
      "auto_reset": bool(self.env.cfg.auto_reset),
      "play_timeout_term_present": "episode_timeout" in self.env.cfg.terminations,
    }
    return {
      "task": self.header["task"],
      "num_envs": self.num_envs,
      "grid": self.grid_cfg.identity(),
      "model": _compiled_model_identity(self.env),
      "sim": sim,
      "lidar": _lidar_identity(self.runtime.lidar_cfg),
      "pd": {
        "raw_clip": float(action.cfg.raw_clip),
        "residual_scale": float(action.cfg.residual_scale),
        "stiffness": float(action.cfg.stiffness),
        "damping": float(action.cfg.damping),
        "effort_limit": float(action.cfg.effort_limit),
        "q_ref": action.q_ref.detach().cpu().tolist(),
      },
      "formal_fall_gates": {
        "low_height_threshold_m": float(LOW_HEIGHT_THRESHOLD_M),
        "low_height_persistence_steps": int(LOW_HEIGHT_PERSISTENCE_STEPS),
        "tilt_threshold_cos": float(TILT_THRESHOLD_COS),
        "flip_threshold_cos": float(FLIP_THRESHOLD_COS),
      },
      "training_time_out_flags": dict(TRAINING_TERMINATION_TIME_OUT_FLAGS),
      "rewards": {key: float(value) for key, value in TRAINING_REWARD_WEIGHTS.items()},
    }

  def _validate_full_checkpoint_identity(self) -> dict:
    if self.header["kind"] != CHECKPOINT_KIND_FULL:
      return {
        "kind": self.header["kind"],
        "saved_model_identity_available": False,
        "verified": False,
        "assumption": "weights-only checkpoint has no saved model/PD/sim/lidar fields",
      }
    saved = self.header.get("saved_identity")
    if not isinstance(saved, dict):
      raise PlayConfigError("FULL checkpoint has no identity object")
    current = self._current_runtime_identity()
    saved_model = saved.get("model")
    if not isinstance(saved_model, dict):
      raise PlayConfigError("FULL checkpoint has no compiled-model identity")
    model_mismatches, q_ref_valid = _compare_compiled_model_identity(
      saved_model, current["model"]
    )
    checks = {
      "task": (self.header["task"], saved.get("task")),
      "grid": (current["grid"], (saved.get("grid") or {}).get("identity")),
      "sim": (
        {key: value for key, value in current["sim"].items()
         if key != "play_timeout_term_present"},
        saved.get("sim"),
      ),
      "lidar": (current["lidar"], saved.get("lidar")),
      "formal_fall_gates": (
        current["formal_fall_gates"],
        {
          key: (saved.get("terminations") or {}).get(key)
          for key in current["formal_fall_gates"]
        },
      ),
      "training_time_out_flags": (
        current["training_time_out_flags"],
        (saved.get("terminations") or {}).get("time_out_flags"),
      ),
      "rewards": (current["rewards"], (saved.get("rewards") or {}).get("weights")),
    }
    mismatches = [name for name, (actual, expected) in checks.items() if actual != expected]
    if model_mismatches:
      mismatches.append(f"model_fields:{model_mismatches}")
    if not q_ref_valid:
      mismatches.append("model.q_ref_not_same_T1_reference_across_env_counts")
    if mismatches:
      raise PlayConfigError(
        "FULL weights-only play identity mismatch: "
        f"fields={mismatches}"
      )
    return {
      "kind": CHECKPOINT_KIND_FULL,
      "saved_model_identity_available": True,
      "verified": True,
      "matched_fields": sorted(checks),
      "model_fields_matched_ignoring_repeated_q_ref_batch_and_batch_digest": True,
      "pd_configuration_basis": (
        "the FULL identity has no separate PD-gain fields; the live raw_clip/"
        "scale/stiffness/damping/effort_limit are recorded, and saved T1 q_ref "
        "was compared across env counts"
      ),
      "full_resume": False,
      "note": "validated FULL identity, then loaded network weights only",
    }

  def zero_command(self) -> dict:
    return self.set_command(0.0, 0.0, 0.0)

  def current_command(self) -> torch.Tensor:
    return self.runtime.current_command()

  # ── the single-owner control step ─────────────────────────────────────────

  def step_once(self):
    """One real control step: prepare(o_k) → one no-grad mean forward → step.

    Returns the ``TrainingTransition``. Non-finite network outputs are
    caught BEFORE any physics runs: the scene is saved and the runtime
    frozen (a forward/save fault stops the run honestly). After a saved
    fall the call refuses UP FRONT — no re-prepare, no forward — the paused
    display keeps reading the retained last snapshot only.
    """
    runtime = self.runtime
    if runtime._closed:
      raise TrainingRuntimeFrozenError("play runtime closed")
    if runtime._faulted:
      raise TrainingRuntimeFrozenError(
        "play runtime frozen after a fault; step refuses permanently"
      )
    if runtime._stopped:
      raise TrainingRuntimeFrozenError(
        "play runtime paused after a saved formal fall; step refuses until "
        "the explicit full-episode manual reset (no forward is run while "
        "paused-after-fall; the display reads the retained snapshot only)"
      )
    self._attempted_policy = None
    try:
      prepared = runtime.prepare()
      input_clone = clone_prepared(prepared)
      window = runtime.window.read()
      angle_keys = runtime.window.read_angle_keys()
      labels = runtime.labels()
      input_control_step = int(runtime.control_steps_taken)
      capture_entries = self._snapshot_capture_entries()
      input_persistent_scan_counts = (
        runtime.sensor_scan_counts.detach().cpu().numpy().copy()
      )
    except BaseException as error:
      if not runtime._faulted:
        runtime._save_fault(error, phase="play/prepare")
      self._persist_terminal_event("fault", f"play/prepare: {error!r}")
      raise
    self._attempted_policy = {"input": input_clone, "labels": labels, "output": None,
                              "input_control_step": input_control_step}
    try:
      with torch.no_grad():
        if self.feature_cache is not None:
          output = self.feature_cache.forward(prepared)
        else:
          output = self.network(
            prepared.observation,
            prepared.near_points,
            prepared.near_valid,
            prepared.far_points,
            prepared.far_valid,
          )
      self._attempted_policy["output"] = output
      require_finite("play_forward/action_mean", output.action_mean)
      require_finite("play_forward/envelope", output.envelope)
      require_finite("play_forward/value", output.value)
      if output.ground_pred is not None:
        require_finite("play_forward/ground_pred", output.ground_pred)
    except BaseException as error:
      runtime._save_fault(error, phase="play/forward")
      self._persist_terminal_event("fault", f"play/forward: {error!r}")
      raise

    # Retain the input and mean before entering physics so a failure inside
    # runtime.step still leaves the exact action that was attempted visible
    # in the read-only fault scene. On success this provisional pose is
    # replaced with the sealed t_(k+1) pose below.
    robot = self.env.scene["robot"]
    try:
      self._retain_display(
        input_clone=input_clone,
        window=window,
        angle_keys=angle_keys,
        labels=labels,
        input_control_step=input_control_step,
        input_persistent_scan_counts=input_persistent_scan_counts,
        input_capture_entries=capture_entries,
        output=output,
        pose={
          "root_pos_w": robot.data.root_link_pos_w.detach().cpu().numpy().copy(),
          "root_quat_wxyz": robot.data.root_link_quat_w.detach().cpu().numpy().copy(),
          "qpos_all": self._qpos_batch_from_sim(),
          "source": (
            "pre-action boundary t_k of the action just computed; if the step "
            "faults, this retained pose is paired with its consumed o_k"
          ),
        },
        contacts=None,
      )
    except BaseException as error:
      runtime._save_fault(error, phase="play/pre-step-display-retention")
      self._persist_terminal_event("fault", f"play/pre-step-display-retention: {error!r}")
      raise

    try:
      transition = runtime.step(output.action_mean)
    except BaseException as error:
      self._persist_terminal_event("fault", f"play/physics: {error!r}")
      raise
    record = transition.context_record
    try:
      self._retain_display(
        input_clone=input_clone,
        window=window,
        angle_keys=angle_keys,
        labels=labels,
        input_control_step=input_control_step,
        input_persistent_scan_counts=input_persistent_scan_counts,
        input_capture_entries=capture_entries,
        output=output,
        pose={
          "root_pos_w": record["root_pos_w"].detach().cpu().numpy().copy(),
          "root_quat_wxyz": record["root_quat_w"].detach().cpu().numpy().copy(),
          "qpos_all": self._qpos_batch(record["qpos"]),
          "source": (
            "post-step boundary t_(k+1) of the last completed control step "
            "(sealed transition record)"
          ),
        },
        contacts={
          "c_nonfoot_env": record["c_nonfoot_env"].detach().cpu().numpy().copy(),
          "c_self": record["c_self"].detach().cpu().numpy().copy(),
        },
      )
    except BaseException as error:
      runtime._save_fault(error, phase="play/post-step-display-retention")
      self._persist_terminal_event("fault", f"play/post-step-display-retention: {error!r}")
      raise
    if runtime._stopped:
      self._persist_terminal_event("fall", "formal fall saved and paused")
    return transition

  def reset(self) -> dict:
    """Explicit full-episode manual reset (the only path past a saved fall)."""
    if self.runtime._closed or self.runtime._faulted:
      raise TrainingRuntimeFrozenError("play reset refused: runtime closed/faulted")
    if self.feature_cache is not None:
      self.feature_cache.clear()
    try:
      record = self.runtime.reset(
        torch.arange(self.num_envs, dtype=torch.long, device=self.device)
      )
      self._attempted_policy = None
      self.refresh_idle_snapshot(reason="manual_reset")
      return record
    except BaseException as error:
      if not self.runtime._faulted:
        self.runtime._save_fault(error, phase="play/reset")
      self._persist_terminal_event("fault", f"play/reset: {error!r}")
      raise

  # ── display retention / snapshot building ─────────────────────────────────

  def refresh_idle_snapshot(self, *, reason: str) -> None:
    """Rebuild the retained display frame from the CURRENT state (no step).

    Used after construction and after a manual reset: the reset already
    refreshed derived quantities, so the reads are valid. No network
    forward is performed — the viewer shows 'no control step yet'.
    """
    runtime = self.runtime
    prepared = runtime.prepare()
    robot = self.env.scene["robot"]
    self._retain_display(
      input_clone=clone_prepared(prepared),
      window=runtime.window.read(),
      angle_keys=runtime.window.read_angle_keys(),
      labels=runtime.labels(),
      input_control_step=int(runtime.control_steps_taken),
      input_persistent_scan_counts=(
        runtime.sensor_scan_counts.detach().cpu().numpy().copy()
      ),
      input_capture_entries=self._snapshot_capture_entries(),
      output=None,
      pose={
        "root_pos_w": robot.data.root_link_pos_w.detach().cpu().numpy().copy(),
        "root_quat_wxyz": robot.data.root_link_quat_w.detach().cpu().numpy().copy(),
        "qpos_all": self._qpos_batch_from_sim(),
        "source": (
          f"current state after {reason} (no control step executed since; the "
          "retained inputs below are the NEXT action's o_k)"
        ),
      },
      contacts=None,
    )

  def _qpos_batch(self, qpos_tensor) -> np.ndarray:
    nq = int(self.env.sim.mj_model.nq)
    qpos = qpos_tensor.detach()
    if qpos.dim() == 1:
      expected = self.num_envs * nq
      if int(qpos.numel()) != expected:
        raise RuntimeError(
          f"flat qpos size {int(qpos.numel())} != num_envs*nq ({expected})"
        )
      qpos = qpos.view(self.num_envs, nq)
    if int(qpos.shape[1]) != nq:
      raise RuntimeError(f"qpos row width {int(qpos.shape[1])} != nq {nq}")
    return qpos.cpu().numpy().copy()

  def _qpos_batch_from_sim(self) -> np.ndarray:
    import warp as wp

    return self._qpos_batch(wp.to_torch(self.env.sim.wp_data.qpos))

  def _retain_display(
    self,
    *,
    input_clone,
    window,
    angle_keys,
    labels,
    input_control_step,
    input_persistent_scan_counts,
    input_capture_entries,
    output,
    pose,
    contacts,
  ) -> None:
    ground = labels["ground"]["label"]
    self._display = _DisplayRetention(
      input_clone=input_clone,
      window=window,
      angle_keys=angle_keys,
      labels_cpu={
        "ground_env_ids": ground.env_ids.detach().cpu().numpy().copy(),
        "ground_episode_ids": ground.episode_ids.detach().cpu().numpy().copy(),
        "ground_scan_ids": ground.scan_ids.detach().cpu().numpy().copy(),
        "ground_heights": ground.heights.detach().cpu().numpy().copy(),
        "ground_valid": ground.valid.detach().cpu().numpy().copy(),
        "grid_xy_base": ground.positions.detach().cpu().numpy().copy()[:, :, :2],
        "envelope_env_ids": labels["envelope"].env_ids.detach().cpu().numpy().copy(),
        "envelope_episode_ids": labels["envelope"].episode_ids.detach().cpu().numpy().copy(),
        "envelope_step_ids": labels["envelope"].step_ids.detach().cpu().numpy().copy(),
        "envelope_label": labels["envelope"].values.detach().cpu().numpy().copy(),
        "grid_identity_hash": labels["ground"]["grid_identity_hash"],
        "ground_metadata": _cpu_tree(labels["ground"]["metadata"]),
      },
      output=(
        None if output is None
        else {
          "envelope": output.envelope.detach().cpu().numpy().copy(),
          "ground_pred": (
            None if output.ground_pred is None
            else output.ground_pred.detach().cpu().numpy().copy()
          ),
          "action_mean": output.action_mean.detach().cpu().numpy().copy(),
          "value": output.value.detach().cpu().numpy().copy(),
        }
      ),
      pose=pose,
      contacts=contacts,
      input_control_step=int(input_control_step),
      input_episode_ids=input_clone.observation.episode_ids.detach().cpu().numpy().copy(),
      input_step_in_episode=input_clone.observation.step_ids.detach().cpu().numpy().copy(),
      input_persistent_scan_counts=np.asarray(
        input_persistent_scan_counts, dtype=np.int64
      ).copy(),
      input_capture_entries=input_capture_entries,
      pose_control_step=int(self.runtime.control_steps_taken),
      pose_episode_ids=self.runtime.episode_ids.detach().cpu().numpy().copy(),
      pose_step_in_episode=self.runtime.step_in_episode.detach().cpu().numpy().copy(),
      pose_persistent_scan_counts=self.runtime.sensor_scan_counts.detach()
      .cpu().numpy().copy(),
    )

  def _snapshot_capture_entries(self) -> dict[int, list]:
    return {
      int(row): [
        {
          key: (value.detach().cpu().clone() if torch.is_tensor(value) else value)
          for key, value in entry.items()
        }
        for entry in entries
      ]
      for row, entries in self.play_source_scans.items()
    }

  def build_display_snapshot(self, env_row: int, *, paused: bool = False) -> dict:
    """Assemble the viewer snapshot for one env row (pure assembly).

    ``paused`` is the DRIVER's pause state; the fall/fault state comes from
    the runtime. The pose is the post-step boundary t_(k+1) (or the current
    state before any step); the inputs/predictions are the o_k the LAST
    action consumed — both labelled explicitly so a pre-action prediction
    is never presented as a future post-state.
    """
    if not 0 <= int(env_row) < self.num_envs:
      raise PlayConfigError(f"env row out of range: {env_row!r}")
    row = int(env_row)
    disp = self._display
    if disp is None:
      raise RuntimeError("no display frame retained")
    prepared = disp.input_clone
    observation = prepared.observation
    window = disp.window
    if not bool(window.near_filled[row]):
      raise RuntimeError("retained window has no near scan for this row")
    prov = resolve_point_sources(
      snapshot=window,
      angle_keys=disp.angle_keys,
      geometric_keys_fn=self.runtime.pipeline.angle_keys,
      prepared=prepared,
      capture_entries=disp.input_capture_entries[row],
      env_row=row,
      episode_id=int(disp.input_episode_ids[row]),
    )
    prov_numpy = {
      key: (value.detach().cpu().numpy() if torch.is_tensor(value) else value)
      for key, value in prov.items()
    }
    near_time = float(prepared.near_times[row])
    input_time = float(observation.time_s[row])
    far_times = prepared.far_point_times[row].detach().cpu().numpy()
    far_valid = prepared.far_valid[row].detach().cpu().numpy()
    far_ages = np.full((int(far_valid.shape[0]),), np.inf)
    if bool(far_valid.any()):
      far_ages[far_valid] = np.maximum(input_time - far_times[far_valid], 0.0)
    fall_state = get_termination_state(self.env)
    if self.runtime._faulted:
      state_kind = "faulted"
      state_note = (
        "fault scene saved; step AND reset are refused permanently "
        "(a fault is never recoverable)"
      )
    elif self.runtime._stopped:
      state_kind = "fallen"
      state_note = (
        "formal-fall scene saved; stepping refuses until the explicit "
        "manual reset (continue cannot bypass a fall)"
      )
    elif paused:
      state_kind = "paused"
      state_note = "clock stopped by the operator; no physics/scan/noise advance"
    else:
      state_kind = "running"
      state_note = ""
    ground_z = float(
      np.asarray(self.runtime.pipeline.mesh_record["z"], dtype=np.float64)
      .reshape(-1)[0]
    )
    network_out = None if disp.output is None else {
      "envelope": disp.output["envelope"][row],
      "ground_pred": (
        None if disp.output["ground_pred"] is None
        else disp.output["ground_pred"][row]
      ),
      "action_mean": disp.output["action_mean"][row],
      "value": disp.output["value"][row],
    }
    return {
      "schema": PLAY_SNAPSHOT_SCHEMA,
      "mode": "play",
      "env": row,
      "identity": {
        "episode_id": int(disp.input_episode_ids[row]),
        "control_step": int(disp.pose_control_step),
        "control_time_s": float(disp.pose_control_step) * 0.02,
        "step_in_episode": int(disp.input_step_in_episode[row]),
        "persistent_scan_count": int(disp.input_persistent_scan_counts[row]),
        "input_episode_id": int(disp.input_episode_ids[row]),
        "input_control_step": int(disp.input_control_step),
        "input_step_in_episode": int(disp.input_step_in_episode[row]),
        "input_persistent_scan_count": int(disp.input_persistent_scan_counts[row]),
        "input_near_scan_id": int(prepared.near_scan_ids[row]),
        "pose_episode_id": int(disp.pose_episode_ids[row]),
        "pose_control_step": int(disp.pose_control_step),
        "pose_step_in_episode": int(disp.pose_step_in_episode[row]),
        "pose_persistent_scan_count": int(disp.pose_persistent_scan_counts[row]),
      },
      "state": {"kind": state_kind, "note": state_note},
      "pose": {
        "root_pos_w": disp.pose["root_pos_w"][row],
        "root_quat_wxyz": disp.pose["root_quat_wxyz"][row],
        "qpos": disp.pose["qpos_all"][row],
        "source": disp.pose["source"],
      },
      "command": observation.command[row].detach().cpu().numpy().tolist(),
      "current_command": self.current_command().detach().cpu().numpy().tolist(),
      "input": {
        "source_label": (
          "pre-action o_k (the input of the last executed action; "
          "predictions paired with it, NOT a future post-state)"
        ),
        "near": {
          "points_base": prepared.near_points[row].detach().cpu().numpy(),
          "valid": prepared.near_valid[row].detach().cpu().numpy(),
          "scan_id": int(prepared.near_scan_ids[row]),
          "time_s": near_time,
        },
        "far": {
          "points_base": prepared.far_points[row].detach().cpu().numpy(),
          "valid": far_valid,
          "scan_ids": prepared.far_point_scan_ids[row].detach().cpu().numpy(),
          "times": far_times,
          "age_s": far_ages,
        },
        "window_far_fill": int(window.far_filled[row].sum()),
        "input_time_s": input_time,
        "prev_action": observation.prev_action[row].detach().cpu().numpy(),
        "lin_vel_b": observation.lin_vel_b[row].detach().cpu().numpy(),
        "ang_vel_b": observation.ang_vel_b[row].detach().cpu().numpy(),
        "gravity_b": observation.gravity_b[row].detach().cpu().numpy(),
      },
      "network": network_out,
      "labels": _slice_batch_payload(disp.labels_cpu, row, self.num_envs),
      "provenance": prov_numpy,
      "pool": self._pool_layer(row, int(window.near_scan_ids[row])),
      "metrics": {
        "root_height_m": float(disp.pose["root_pos_w"][row][2]) - ground_z,
        "scan_age_s": max(input_time - near_time, 0.0),
        "far_window_fill": int(window.far_filled[row].sum()),
        "c_nonfoot_env": (
          0.0 if disp.contacts is None else float(disp.contacts["c_nonfoot_env"][row])
        ),
        "c_self": (
          0.0 if disp.contacts is None else float(disp.contacts["c_self"][row])
        ),
        "fall_low_height_count": int(fall_state.low_height_count[row]),
        "fall_tilt_count": int(fall_state.tilt_count[row]),
        "ground_z": ground_z,
      },
      "checkpoint": {
        "path": self.header["path"],
        "kind": self.header["kind"],
        "ground_samples": self.header["ground_samples"],
        "grid_identity_hash": (
          self.header["grid_identity_hash"] or self.grid_cfg.identity_hash()
        ),
        "file_sha256": self.header["file_sha256"],
      },
    }

  def _pool_layer(self, row: int, near_scan_id: int) -> dict | None:
    """The retained CURRENT raw frame's candidate groups (display-only)."""
    for record in reversed(self.runtime.pipeline.scan_records):
      frame = record["frame"]
      if int(record["env_row"]) == row and int(frame.scan_id) == near_scan_id:
        return {
          "points_base": frame.pool_points_base.detach().cpu().numpy(),
          "near_mask": frame.near_mask.detach().cpu().numpy(),
          "far_mask": frame.far_mask.detach().cpu().numpy(),
          "raw_indices": frame.pool_raw_indices.detach().cpu().numpy(),
          "scan_id": int(frame.scan_id),
          "count": int(frame.pool_raw_indices.numel()),
        }
    return None

  # ── terminal evidence ─────────────────────────────────────────────────────

  def _persist_terminal_event(self, kind: str, reason: str) -> None:
    """Save the actual fall/fault scene immediately, before UI/reset/exit.

    Each event gets a fresh directory. These diagnostics never latch the
    independent final exit record or become training checkpoints.
    """
    if kind == "fault" and self._persisted_fault_event:
      return
    self._play_event_counter += 1
    event_dir = self.output_dir / "play_events" / (
      f"{self._play_event_counter:04d}_{kind}_c{self.runtime.control_steps_taken:08d}"
    )
    try:
      event_dir.mkdir(parents=True, exist_ok=False)
      self.write_terminal_record(reason, _event_output_dir=event_dir)
      if kind == "fault":
        self._persisted_fault_event = True
    except BaseException as error:
      if not self.runtime._faulted:
        self.runtime._save_fault(error, phase="play/event_scene")
      if self.runtime.last_fault is not None:
        self.runtime.last_fault["play_event_save_error"] = repr(error)
      raise

  def write_terminal_record(self, reason: str, *, extra: dict | None = None,
                            _event_output_dir: Path | None = None) -> None:
    """Write a concise summary and an exact CPU terminal scene.

    The ``.pt`` companion keeps the actual terminal arrays and runtime
    caches without optimizer/RNG restoration semantics. It is deliberately
    marked NON_RESUMABLE and is never accepted by ``load_weights``.
    """
    if _event_output_dir is None and self._terminal_written:
      return
    destination = self.output_dir if _event_output_dir is None else _event_output_dir
    runtime = self.runtime
    disp = self._display
    try:
      action = self.env.action_manager.get_term("residual_pd")
      weight_state = _cpu_tree(self.network.state_dict())
      model_digest = hashlib.sha256()
      for name in sorted(weight_state):
        tensor = weight_state[name]
        model_digest.update(name.encode("utf-8"))
        if torch.is_tensor(tensor):
          model_digest.update(str(tensor.dtype).encode("ascii"))
          model_digest.update(str(tuple(tensor.shape)).encode("ascii"))
          model_digest.update(tensor.contiguous().numpy().tobytes())
        else:
          model_digest.update(repr(tensor).encode("utf-8"))
      display_state = None if disp is None else {
        "input": _cpu_tree(disp.input_clone),
        "window": _cpu_tree(disp.window),
        "angle_keys": _cpu_tree(disp.angle_keys),
        "source_scan_metadata": _cpu_tree(disp.input_capture_entries),
        "input_identity": {
          "control_step": disp.input_control_step,
          "episode_ids": disp.input_episode_ids.copy(),
          "step_in_episode": disp.input_step_in_episode.copy(),
          "persistent_scan_counts": disp.input_persistent_scan_counts.copy(),
        },
        "pose_identity": {
          "control_step": disp.pose_control_step,
          "episode_ids": disp.pose_episode_ids.copy(),
          "step_in_episode": disp.pose_step_in_episode.copy(),
          "persistent_scan_counts": disp.pose_persistent_scan_counts.copy(),
        },
        "pose": _cpu_tree(disp.pose),
        "contacts": _cpu_tree(disp.contacts),
        "network_output": _cpu_tree(disp.output),
        "action_mean": (
          None if disp.output is None
          else np.asarray(disp.output["action_mean"]).copy()
        ),
        "labels": _cpu_tree(disp.labels_cpu),
      }
      scene = {
        "schema": "confined_locomotion.play_terminal_scene/v2",
        "mode": "play",
        "resumable": False,
        "kind": "play_terminal_scene_non_resumable",
        "reason": str(reason),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "checkpoint": _cpu_tree(self.header),
        "checkpoint_compatibility": _cpu_tree(self.full_identity_report),
        "live_runtime_identity": _cpu_tree(self._current_runtime_identity()),
        "weight_load_report": _cpu_tree(self.weight_report),
        "weights": {
          "state_dict": weight_state,
          "state_dict_sha256": model_digest.hexdigest(),
          "checkpoint_file_sha256": self.header["file_sha256"],
        },
        "physical": runtime._physics_snapshot(),
        "sensing": {
          "lidar_config": _cpu_tree(runtime.lidar_cfg),
          "mount_pos": np.asarray(runtime.pipeline.mount_pos).copy(),
          "mount_quat_xyzw": np.asarray(runtime.pipeline.mount_quat_xyzw).copy(),
          "plane_mesh": _cpu_tree(runtime.pipeline.mesh_record),
          "sampled_source_scan_metadata": _cpu_tree(runtime._play_source_scans),
          "window_is_saved_in_runtime_section": True,
          "full_raw_ray_frames": "not duplicated; bounded sampled-ID/no-hit metadata and exact network windows are saved",
        },
        "managers": {
          "residual_pd": {
            key: _cpu_tree(getattr(action, key))
            for key in (
              "q_ref", "prev_action", "last_raw_action",
              "last_clipped_action", "last_target", "_processed_actions",
              "_raw_actions",
            )
            if hasattr(action, key)
          },
          "residual_pd_config": {
            "raw_clip": float(action.cfg.raw_clip),
            "residual_scale": float(action.cfg.residual_scale),
            "stiffness": float(action.cfg.stiffness),
            "damping": float(action.cfg.damping),
            "effort_limit": float(action.cfg.effort_limit),
          },
          "termination": _cpu_tree(get_termination_state(self.env)),
          "reward_terms": _cpu_tree(
            getattr(self.env, "_confined_training_reward_terms", None)
          ),
        },
        "runtime": {
          "mode": runtime.mode,
          "control_steps_taken": runtime.control_steps_taken,
          "control_time_s": runtime.control_time_s(),
          "episode_ids": _cpu_tree(runtime.episode_ids),
          "step_in_episode": _cpu_tree(runtime.step_in_episode),
          "scan_counters": _cpu_tree(runtime.scan_counters),
          "sensor_scan_counts": _cpu_tree(runtime.sensor_scan_counts),
          "next_due_scan_step": _cpu_tree(runtime.next_due_scan_step),
          "window": _cpu_tree(runtime.window.read()),
          "window_angle_keys": _cpu_tree(runtime.window.read_angle_keys()),
          "ground_cache": _cpu_tree(runtime._ground_cache),
          "input_cache": _cpu_tree(runtime._input_cache.get()),
          "current_window_snapshot": _cpu_tree(runtime._current_window_snapshot),
          "commands": _cpu_tree(runtime._commands.state_dict()),
          "perturbations": (
            None if runtime._perturbations is None
            else _cpu_tree(runtime._perturbations.state_dict())
          ),
          "play_source_scans": _cpu_tree(runtime._play_source_scans),
          "transition_context": {
            "phase": runtime._context.phase,
            "record": _cpu_tree(runtime._context.record),
            "pd": _cpu_tree(runtime._context.pd),
            "tables": _cpu_tree(runtime._context.tables),
          },
          "last_completed_transition": (
            None if not runtime.transitions
            else _cpu_tree(runtime.transitions[-1])
          ),
          "stopped": bool(runtime._stopped),
          "faulted": bool(runtime._faulted),
          "stop_record": _cpu_tree(runtime.stop_record),
          "last_fault": _cpu_tree(runtime.last_fault),
          "closed": bool(runtime._closed),
        },
        "command": _cpu_tree(self.current_command()),
        "last_display_frame": display_state,
        "attempted_policy": _cpu_tree(self._attempted_policy),
        "optimizer": None,
        "rng_continuation": {
          "global_rng_state_saved": False,
          "task_stream_states_included_as_runtime_diagnostics": True,
          "restorable": False,
        },
        "resume_note": (
          "NON_RESUMABLE diagnostic scene only; physics/managers/runtime are "
          "stored for inspection; named task-stream states are diagnostic, "
          "there is no load/apply path, optimizer, global RNG continuation "
          "or training-resume claim"
        ),
      }
      scene_path = destination / "play_terminal_scene.pt"
      torch.save(scene, scene_path)

      last_output = (
        None if disp is None or disp.output is None
        else {
          name: np.asarray(value).tolist()
          for name, value in disp.output.items()
          if value is not None
        }
      )
      record = {
        "schema": "confined_locomotion.play_terminal/v2",
        "mode": "play",
        "resumable": False,
        "terminal_scene": scene_path.name,
        "reason": str(reason),
        "task_note": PLAY_TASK_NOTE,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_s": time.monotonic() - self.started_at,
        "checkpoint": self.header,
        "weight_load_report": self.weight_report,
        "runtime": {
          "mode": runtime.mode,
          "control_steps_taken": runtime.control_steps_taken,
          "control_time_s": runtime.control_time_s(),
          "stopped": bool(runtime._stopped),
          "faulted": bool(runtime._faulted),
          "stop_record": runtime.stop_record,
          "last_fault": _cpu_tree(runtime.last_fault),
          "episode_ids": runtime.episode_ids.detach().cpu().tolist(),
          "scan_counters": runtime.scan_counters.detach().cpu().tolist(),
          "sensor_scan_counts": runtime.sensor_scan_counts.detach().cpu().tolist(),
          "closed": bool(runtime._closed),
        },
        "command": self.current_command().detach().cpu().tolist(),
        "sensing": {
          "lidar_rays": int(runtime.lidar_cfg.airy_n_azimuth)
          * int(runtime.lidar_cfg.airy_n_elevation),
          "window_far_fill": (
            None if disp is None
            else disp.window.far_filled.sum(dim=1).detach().cpu().tolist()
          ),
          "near_scan_ids": (
            None if disp is None
            else disp.window.near_scan_ids.detach().cpu().tolist()
          ),
        },
        "last_display_frame": {
          "input_control_step": None if disp is None else disp.input_control_step,
          "pose_control_step": None if disp is None else disp.pose_control_step,
          "network_output": last_output,
        },
        "torch_memory": {
          "max_allocated_bytes": int(torch.cuda.max_memory_allocated(self.device)),
          "max_reserved_bytes": int(torch.cuda.max_memory_reserved(self.device)),
          "note": "torch allocator peaks only; NOT the whole Warp/device memory",
        },
        "note": (
          "NON_RESUMABLE terminal scene; exact CPU arrays are in the companion "
          "PT file, and no training continuation is claimed"
        ),
      }
      if extra:
        record.update(extra)
      record_path = destination / "terminal_record.json"
      record_path.write_text(json.dumps(record, indent=2, default=str))
      if _event_output_dir is None:
        self._terminal_written = True
    except BaseException as error:
      if not runtime._faulted:
        runtime._save_fault(error, phase="play/terminal_record")
      elif runtime.last_fault is not None:
        runtime.last_fault["terminal_record_error"] = repr(error)
      raise

  # ── lifecycle ─────────────────────────────────────────────────────────────

  def close(self) -> None:
    """Restore the instance hooks and close the env (idempotent)."""
    feature_cache = getattr(self, "feature_cache", None)
    if feature_cache is not None:
      feature_cache.clear()
    runtime = getattr(self, "runtime", None)
    if runtime is not None:
      runtime.close()
    env = getattr(self, "env", None)
    if env is not None and not getattr(self, "_closed_env", False):
      self._closed_env = True
      env.close()
