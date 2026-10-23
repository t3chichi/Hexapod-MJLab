"""Preflighted official entry for explicit formal training.

Known recipe and resume-file errors are rejected before official environment
construction. The official mjlab chain constructs the environment before it
constructs this task runner and does not wrap runner construction in a
``finally`` close; this entry does not patch that generic chain. Once
the task runner owns the environment, its learn-failure path attempts close
without masking the original error. A successful ``launch_training`` return
proves that the official ``env.close()`` call returned; it does not prove
cleanup after an earlier environment/runner construction exception.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Sequence

from mjlab.scripts.train import TrainConfig, launch_training
from mjlab.utils.os import get_checkpoint_path

from ..flat_env_cfg import TASK_ID
from ..ground import GroundGridCfg, VALIDITY_GEOMETRY_ONLY_FLAT_TRAINING
from ..rl_cfg import ConfinedLocomotionRlCfg, confined_locomotion_formal_rl_cfg
from .runner import FORMAL_RUNNER_IDENTITY, _resolve_formal_cfg

__all__ = [
  "FormalTrainingConfigError",
  "FormalLaunchResult",
  "build_formal_train_config",
  "preflight_formal_train_config",
  "launch_formal_training",
]


class FormalTrainingConfigError(ValueError):
  """An explicit formal recipe or its pre-env launch config is invalid."""


@dataclass(frozen=True)
class FormalLaunchResult:
  """Result returned only after the official train call and close return.

  ``close_completed`` comes from the successful official call returning after
  ``env.close``. ``summary`` is the runner's persisted best-effort report; its
  ``formal_entry_verification`` distinguishes a persisted close record from
  the official return proof if the post-commit summary refresh could not be
  written.
  """

  config: TrainConfig
  run_dir: Path
  checkpoint_path: Path
  summary: dict
  close_completed: bool


def _require_seed(seed: object) -> int:
  if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
    raise FormalTrainingConfigError(
      f"seed must be an explicit integer in [0, 2**32), got {seed!r}"
    )
  return int(seed)


def _resolve_resume_checkpoint(
  cfg: TrainConfig, explicit_path: str | Path | None
) -> tuple[TrainConfig, Path | None]:
  if explicit_path is None:
    if cfg.agent.resume:
      raise FormalTrainingConfigError(
        "resume=True requires an explicit local full-checkpoint path"
      )
    if cfg.wandb_run_path is not None or cfg.wandb_checkpoint_name is not None:
      raise FormalTrainingConfigError("formal resume does not use W&B checkpoints")
    return cfg, None

  target = Path(explicit_path).expanduser().resolve()
  if not target.is_file():
    raise FormalTrainingConfigError(f"resume checkpoint file not found: {target}")
  if target.parent.parent == target.parent:
    raise FormalTrainingConfigError(f"invalid checkpoint run path: {target}")
  experiment_dir = target.parent.parent
  cfg.agent.experiment_name = experiment_dir.name
  cfg.agent.load_run = f"^{re.escape(target.parent.name)}$"
  cfg.agent.load_checkpoint = f"^{re.escape(target.name)}$"
  cfg.agent.resume = True
  cfg = replace(cfg, log_root=str(experiment_dir.parent))
  return cfg, target


def build_formal_train_config(
  *,
  seed: int,
  ground_grid: GroundGridCfg,
  num_envs: int,
  max_iterations: int,
  resume_checkpoint: str | Path | None = None,
  log_root: str | Path = "logs/confined_locomotion",
  gpu_ids: Sequence[int] = (0,),
  engineering_validation: bool = False,
) -> TrainConfig:
  """Build an explicit task recipe on top of the official task TrainConfig.

  Seed, grid, environment count, and requested update budget are mandatory
  positional-by-keyword inputs. The validation switch only authorizes the
  already-bounded 1-update / <=2-env check; it does not choose those values.
  """
  seed = _require_seed(seed)
  if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
    raise FormalTrainingConfigError(f"num_envs must be a positive explicit int, got {num_envs!r}")
  if (
    isinstance(max_iterations, bool)
    or not isinstance(max_iterations, int)
    or max_iterations < 1
  ):
    raise FormalTrainingConfigError(
      f"max_iterations must be a positive explicit int, got {max_iterations!r}"
    )
  if not isinstance(ground_grid, GroundGridCfg):
    raise FormalTrainingConfigError("ground_grid must be an explicitly built GroundGridCfg")
  if ground_grid.validity_mode != VALIDITY_GEOMETRY_ONLY_FLAT_TRAINING:
    raise FormalTrainingConfigError(
      "formal training requires GroundGridCfg validity_mode="
      f"{VALIDITY_GEOMETRY_ONLY_FLAT_TRAINING!r}"
    )
  if not isinstance(engineering_validation, bool):
    raise FormalTrainingConfigError("engineering_validation must be bool")
  if engineering_validation and (num_envs > 2 or max_iterations != 1):
    raise FormalTrainingConfigError(
      "engineering validation is limited to at most 2 envs and exactly 1 update"
    )
  if isinstance(gpu_ids, (str, bytes)) or len(gpu_ids) != 1:
    raise FormalTrainingConfigError("formal training requires exactly one explicit GPU id")
  selected_gpu = gpu_ids[0]
  if isinstance(selected_gpu, bool) or not isinstance(selected_gpu, int) or selected_gpu < 0:
    raise FormalTrainingConfigError(f"gpu id must be a non-negative int, got {selected_gpu!r}")

  cfg = TrainConfig.from_task(TASK_ID)
  agent_cfg = confined_locomotion_formal_rl_cfg(
    seed=seed, ground_grid=ground_grid, num_envs=num_envs,
    max_iterations=max_iterations,
  )
  if engineering_validation:
    agent_cfg.engineering_enabled = True
  agent_cfg.run_name = f"formal_{uuid.uuid4().hex}"
  cfg = replace(cfg, agent=agent_cfg)
  cfg.env.seed = seed
  cfg.env.scene.num_envs = num_envs
  cfg = replace(
    cfg,
    log_root=str(Path(log_root).expanduser()),
    gpu_ids=[selected_gpu],
  )
  cfg, _ = _resolve_resume_checkpoint(cfg, resume_checkpoint)
  return cfg


def _static_resume_identity(cfg: TrainConfig, path: Path) -> dict:
  from .checkpoint import (
    CHECKPOINT_KIND_FULL,
    CHECKPOINT_SCHEMA,
    _source_identity,
    validate_payload_envelope,
  )

  try:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    validate_payload_envelope(payload)
  except Exception as err:
    raise FormalTrainingConfigError(f"resume checkpoint is not a valid full payload: {err}") from err
  if payload.get("schema") != CHECKPOINT_SCHEMA or payload.get("kind") != CHECKPOINT_KIND_FULL:
    raise FormalTrainingConfigError(
      "formal resume accepts only the current full-resume checkpoint kind/schema"
    )
  identity = payload.get("identity")
  if not isinstance(identity, dict):
    raise FormalTrainingConfigError("formal resume checkpoint identity must be an object")
  resolved = _resolve_formal_cfg(asdict(cfg.agent))
  expected_grid = resolved["grid"]
  expected = {
    "task": TASK_ID,
    "runner_identity": FORMAL_RUNNER_IDENTITY,
    "source_sha256": _source_identity(),
    "num_envs": int(cfg.agent.formal_num_envs),
    "env_seed": int(cfg.env.seed),
    "runtime": {"mode": "training", "seed": int(cfg.agent.seed)},
    "grid": {
      "identity": expected_grid.identity(),
      "identity_hash": expected_grid.identity_hash(),
    },
  }
  for name, value in expected.items():
    if identity.get(name) != value:
      raise FormalTrainingConfigError(
        f"resume checkpoint {name} identity differs from the explicit formal recipe"
      )
  return {
    "schema": payload["schema"],
    "kind": payload["kind"],
    "task": identity.get("task"),
    "runner_identity": identity.get("runner_identity"),
    "source_sha256": identity.get("source_sha256"),
    "num_envs": identity.get("num_envs"),
    "env_seed": identity.get("env_seed"),
    "runtime": identity.get("runtime"),
    "grid": identity.get("grid"),
  }


def preflight_formal_train_config(cfg: TrainConfig) -> dict:
  """Reject malformed formal/resume configs before official env construction."""
  if not isinstance(cfg, TrainConfig):
    raise TypeError(f"cfg must be the official TrainConfig, got {type(cfg).__name__}")
  if cfg.agent.formal_training_enabled is not True:
    raise FormalTrainingConfigError("formal_training_enabled must be explicitly True")
  resolved = _resolve_formal_cfg(asdict(cfg.agent))
  env_seed = cfg.env.seed
  if (
    isinstance(env_seed, bool)
    or not isinstance(env_seed, int)
    or not 0 <= env_seed < 2**32
  ):
    raise FormalTrainingConfigError(
      f"env.seed must be an explicit int in [0, 2**32), got {env_seed!r}"
    )
  if resolved["seed"] != env_seed:
    raise FormalTrainingConfigError("agent and environment seed must match exactly")
  env_num_envs = cfg.env.scene.num_envs
  if (
    isinstance(env_num_envs, bool)
    or not isinstance(env_num_envs, int)
    or env_num_envs < 1
  ):
    raise FormalTrainingConfigError(
      f"env.scene.num_envs must be a positive int, got {env_num_envs!r}"
    )
  if resolved["num_envs"] != env_num_envs:
    raise FormalTrainingConfigError("formal_num_envs must match env.scene.num_envs")
  if cfg.registry_name is not None:
    raise FormalTrainingConfigError("formal entry does not use registry motion downloads")
  if cfg.video or cfg.wandb_run_path is not None or cfg.wandb_checkpoint_name is not None:
    raise FormalTrainingConfigError("formal entry disallows video and W&B launch modes")
  if cfg.enable_nan_guard:
    raise FormalTrainingConfigError("formal entry keeps the accepted env NaN guard configuration")
  if cfg.torchrunx_log_dir is not None:
    raise FormalTrainingConfigError("formal entry is single process; torchrunx is not supported")
  if not isinstance(cfg.gpu_ids, list) or len(cfg.gpu_ids) != 1:
    raise FormalTrainingConfigError("formal entry requires exactly one selected GPU")
  gpu_id = cfg.gpu_ids[0]
  if isinstance(gpu_id, bool) or not isinstance(gpu_id, int) or gpu_id < 0:
    raise FormalTrainingConfigError(f"invalid GPU id {gpu_id!r}")

  resume_record = None
  if cfg.agent.resume:
    log_path = Path(cfg.log_root).expanduser() / cfg.agent.experiment_name
    try:
      target = get_checkpoint_path(
        log_path, cfg.agent.load_run, cfg.agent.load_checkpoint
      ).resolve()
    except Exception as err:
      raise FormalTrainingConfigError(f"cannot resolve explicit resume checkpoint: {err}") from err
    if not target.is_file():
      raise FormalTrainingConfigError(f"resume checkpoint file not found: {target}")
    resume_record = _static_resume_identity(cfg, target)
  elif cfg.wandb_run_path is not None or cfg.wandb_checkpoint_name is not None:
    raise FormalTrainingConfigError("resume is local-only; W&B checkpoint fields must be unset")

  return {
    "task_id": TASK_ID,
    "formal_training_enabled": True,
    "implemented_classvar": ConfinedLocomotionRlCfg.IMPLEMENTED,
    "mode": "formal_training",
    "seed": resolved["seed"],
    "num_envs": resolved["num_envs"],
    "requested_new_iterations": resolved["max_iterations"],
    "engineering_validation": resolved["engineering_validation"],
    "ordinary_formal_launch_allowed": bool(ConfinedLocomotionRlCfg.IMPLEMENTED),
    "learning_effect_ready": False,
    "learning_effect_acceptance": "separate evidence-based gate deferred",
    "grid_identity": resolved["grid"].identity(),
    "grid_identity_hash": resolved["grid"].identity_hash(),
    "resume": resume_record,
    "pre_env_rejection": True,
  }


def _find_run_dir(cfg: TrainConfig) -> Path:
  experiment_dir = Path(cfg.log_root).expanduser().resolve() / cfg.agent.experiment_name
  run_name = cfg.agent.run_name
  matches = sorted(
    path for path in experiment_dir.iterdir()
    if path.is_dir() and path.name.endswith("_" + run_name)
  ) if experiment_dir.is_dir() else []
  if len(matches) != 1:
    raise RuntimeError(
      f"expected exactly one official run directory for {run_name!r}, got {matches}"
    )
  return matches[0]


def launch_formal_training(
  *,
  seed: int,
  ground_grid: GroundGridCfg,
  num_envs: int,
  max_iterations: int,
  resume_checkpoint: str | Path | None = None,
  log_root: str | Path = "logs/confined_locomotion",
  gpu_ids: Sequence[int] = (0,),
  engineering_validation: bool = False,
) -> FormalLaunchResult:
  """Preflight then call the unmodified official ``launch_training`` path."""
  cfg = build_formal_train_config(
    seed=seed,
    ground_grid=ground_grid,
    num_envs=num_envs,
    max_iterations=max_iterations,
    resume_checkpoint=resume_checkpoint,
    log_root=log_root,
    gpu_ids=gpu_ids,
    engineering_validation=engineering_validation,
  )
  preflight_formal_train_config(cfg)
  try:
    launch_training(TASK_ID, args=cfg)
  except BaseException as error:
    try:
      run_dir = _find_run_dir(cfg)
      note = f"formal_training run directory: {run_dir}"
      if hasattr(error, "add_note"):
        error.add_note(note)
      setattr(error, "formal_run_dir", run_dir)
    except Exception:
      pass
    raise

  run_dir = _find_run_dir(cfg)
  candidates = sorted(run_dir.glob("final_checkpoint_full_*.pt"))
  if len(candidates) != 1 or not candidates[0].is_file():
    raise RuntimeError(
      f"official training returned without exactly one full final checkpoint: {candidates}"
    )
  # The full target is the commit proof.  ``launch_training`` returns only
  # after the official ``env.close()`` returned, so its successful return is
  # the close proof.  The runner refreshes its summary best-effort during that
  # close; do not turn a post-commit summary-write failure into a failed launch
  # or remove the valid full checkpoint to disguise it.
  summary_path = run_dir / "training_summary.json"
  summary_read_error = None
  try:
    summary = json.loads(summary_path.read_text())
    if not isinstance(summary, dict):
      summary = {}
      summary_read_error = "summary payload is not an object"
  except Exception as err:
    summary = {}
    summary_read_error = repr(err)
  environment_close = summary.get("environment_close")
  checkpoint_commit = summary.get("final_checkpoint_commit")
  summary_checkpoint = summary.get("final_checkpoint_path")
  persisted_close_record = (
    isinstance(environment_close, dict)
    and environment_close.get("completed") is True
    and isinstance(checkpoint_commit, dict)
    and checkpoint_commit.get("state") == "committed"
    and isinstance(summary_checkpoint, str)
    and Path(summary_checkpoint).expanduser().resolve() == candidates[0].resolve()
  )
  summary["formal_entry_verification"] = {
    "official_launch_returned": True,
    "full_checkpoint_exists": True,
    "official_close_completed": True,
    "close_record_persisted": persisted_close_record,
    "summary_path": str(summary_path),
    "summary_read_error": summary_read_error,
  }
  return FormalLaunchResult(
    config=cfg,
    run_dir=run_dir,
    checkpoint_path=candidates[0],
    summary=summary,
    close_completed=True,
  )
