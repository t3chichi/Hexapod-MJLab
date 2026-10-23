"""Guarded official-chain runner for the confined flat-ground task.

The registered defaults stay intentionally incomplete and are refused before
network or optimizer construction. Explicit engineering recipes are bounded
to one iteration and two environments. Explicit formal recipes require seed,
complete ground-grid identity, environment count, and update budget; learning
effect acceptance is reported separately.

The official mjlab runner chain remains in place. This runner constructs one
network, one distribution provider, one optimizer, one runtime, and one
collector. It uses local JSON/JSONL records, performs updates only after a
complete rollout, and saves full checkpoints only at a stable iteration
boundary. Full resume validates the complete saved state before applying it.
Checkpoint play uses the separate task PlaySession because the generic play
loop does not implement this task's scanner and command lifecycle.

Engineering observers expose live objects to checks without replacing any
official component; observer errors are recorded and do not break a run.
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import torch

from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.utils.torch import configure_torch_backends

from hexapod_mjlab.rl.confined_locomotion.distribution import (
  IndependentDiagonalNormalProvider,
  derive_action_generator_seed,
)
from hexapod_mjlab.rl.confined_locomotion.network import ConfinedLocomotionNetwork
from hexapod_mjlab.rl.confined_locomotion.policy import ConfinedPolicyAdapter
from hexapod_mjlab.rl.confined_locomotion.ppo import (
  ConfinedPPOConfig,
  MirrorTrainingConfig,
)
from hexapod_mjlab.rl.confined_locomotion.symmetry import (
  GridPermutation,
  JointPermutation,
)

from .checkpoint import (
  load_full_checkpoint,
  load_weights as load_weights_only,
  save_full_checkpoint,
)
from ..flat_env_cfg import TASK_ID
from ..rl_cfg import (
  ENGINEERING_MAX_ITERATIONS,
  ENGINEERING_MAX_NUM_ENVS,
  ROLLOUT_CONTROL_STEPS,
  ConfinedLocomotionRlCfg,
)
from ..runtime import TrainingRuntime, TrainingRuntimeFrozenError
from ..collector import TrainingCollector, model_state_fingerprint
from ..ground import GroundGridCfg, VALIDITY_GEOMETRY_ONLY_FLAT_TRAINING
from ..perturbations import ALL_THREE_PRESET, ObservationPerturbationCfg

_GROUND_GRID_FIELDS = {
  "x_coords", "y_coords", "validity_mode", "parallel_tol", "body_xyz_semantics"
}


def _resolve_ground_grid(value, *, context: str) -> GroundGridCfg:
  if not isinstance(value, dict) or set(value) != _GROUND_GRID_FIELDS:
    raise RuntimeError(
      f"{context} ground_grid must be the complete GroundGridCfg with fields "
      f"{sorted(_GROUND_GRID_FIELDS)}"
    )
  try:
    grid = GroundGridCfg(
      x_coords=tuple(value["x_coords"]),
      y_coords=tuple(value["y_coords"]),
      validity_mode=value["validity_mode"],
      parallel_tol=value["parallel_tol"],
      body_xyz_semantics=value["body_xyz_semantics"],
    )
    if grid.validity_mode != VALIDITY_GEOMETRY_ONLY_FLAT_TRAINING:
      raise ValueError(
        f"validity_mode must be {VALIDITY_GEOMETRY_ONLY_FLAT_TRAINING!r}"
      )
  except (TypeError, ValueError) as err:
    raise RuntimeError(f"{context} ground_grid is invalid: {err}") from err
  return grid

__all__ = [
  "ConfinedLocomotionRunner",
  "PENDING_CAPABILITY_MSG",
  "RUNNER_GUARDED_MSG",
  "RUNNER_IDENTITY",
  "FORMAL_RUNNER_IDENTITY",
  "TASK_PLAY_ENTRY_MSG",
  "TrainingEntryNotPassed",
  "add_engineering_observer",
  "clear_engineering_observers",
  "validate_engineering_learn_arguments",
]

#: Implementation identity recorded in every local artifact of this runner.
RUNNER_IDENTITY = "El-4090-Confined-Locomotion-Flat/ConfinedLocomotionRunner/mode=engineering"
FORMAL_RUNNER_IDENTITY = "El-4090-Confined-Locomotion-Flat/ConfinedLocomotionRunner/mode=formal_training"

#: Why the default (non-engineering) construction is refused.
RUNNER_GUARDED_MSG = (
  "El-4090-Confined-Locomotion-Flat registered defaults are intentionally "
  "incomplete and guarded: "
  "max_iterations=0, engineering_enabled=False, "
  "formal_training_enabled=False, and no approved seed/env/budget. The "
  "registered S1-187 grid does not authorize training. This runner "
  "requires either the bounded engineering factory or the separate "
  "explicit formal training recipe; the registry default never runs. "
  "IMPLEMENTED is a class-level implementation marker and cannot be overridden by config."
)

#: Shared refusal text for APIs that this training runner does not implement.
PENDING_CAPABILITY_MSG = (
  "generic inference, policy export, and eval are not exposed by this task "
  "training runner; task play uses the separate PlaySession. The inherited "
  "reference implementation is never silently used."
)

#: Why ``play()`` is refused on the TRAINING runner and where the real task
#: play entry lives (the generic mjlab play script does not implement this
#: task's sensor/command lifecycle and must not be claimed to work).
TASK_PLAY_ENTRY_MSG = (
  "play() is refused on the TRAINING runner: task play is the "
  "explicit mean-only entry "
  "hexapod_mjlab.tasks.confined_locomotion.play.PlaySession (CLI: "
  "python scripts/experiments/confined_locomotion/play.py --checkpoint "
  "<file> --output <new-dir>). It loads the full four-head network "
  "weights-only, runs the manual command lifecycle with the real viewer; "
  "the inherited generic play implementation is never silently used."
)

#: Schema stamp of local runner records. Version 2 stores complete grid identity.
RUNNER_RECORD_SCHEMA = "confined_locomotion.runner/v2"


class TrainingEntryNotPassed(RuntimeError):
  """The bounded engineering learn did not pass (no update was executed).

  Raised for an incomplete 24-step rollout or ANY engineering stop (natural
    independent fall / formal done, including one landing exactly on the last
    step). The runtime has already saved the stop evidence; this exception
  only reports the honest not-passed outcome.
  """

  def __init__(self, record: dict) -> None:
    self.record = record
    collect = record.get("collect") or {}
    super().__init__(
      f"engineering learn not passed: completed "
      f"{collect.get('completed_steps')}/{collect.get('requested_steps')} steps "
      f"(stopped_reason={collect.get('stopped_reason')!r}); no update was "
      "executed and the run stops here — the terminal evidence was saved by "
      "the training runtime"
    )


# ── engineering observers (acceptance-evidence channel; explicit only) ─────

_ENGINEERING_OBSERVERS: list = []


def add_engineering_observer(observer) -> None:
  """Register ``observer(event, runner, payload)`` for the engineering path.

  Gives independent checks live access to the REAL objects (runner →
  runtime/env/wrapper/trainer) without replacing any official component.
  Observer exceptions are recorded locally and never break the run.
  """
  if not callable(observer):
    raise TypeError("engineering observer must be callable(event, runner, payload)")
  _ENGINEERING_OBSERVERS.append(observer)


def clear_engineering_observers() -> None:
  """Remove all engineering observers (test isolation)."""
  _ENGINEERING_OBSERVERS.clear()


def validate_engineering_learn_arguments(
  num_learning_iterations, init_at_random_ep_len
) -> None:
  """Validate official ``learn`` kwargs against the bounded engineering budget.

  ``num_learning_iterations`` must be exactly 1 (this batch's hard budget):
  0 is an explicit no-op that must not be reported as a pass, and anything
  above the budget is rejected — the engineering entry never auto-runs long.
  """
  if isinstance(num_learning_iterations, bool) or not isinstance(num_learning_iterations, int):
    raise TypeError(
      f"num_learning_iterations must be int, got {num_learning_iterations!r}"
    )
  if num_learning_iterations < 0:
    raise ValueError(
      f"num_learning_iterations must be non-negative, got {num_learning_iterations!r}"
    )
  if num_learning_iterations == 0:
    raise RuntimeError(
      "engineering budget 0: a zero-iteration learn executes nothing and is "
      "explicitly rejected (not a pass); nothing was run"
    )
  if num_learning_iterations > ENGINEERING_MAX_ITERATIONS:
    raise RuntimeError(
      f"engineering budget exceeded: at most {ENGINEERING_MAX_ITERATIONS} "
      f"iteration(s) per learn call on this batch, got {num_learning_iterations} "
      "(no unbounded auto-running; longer budgets need an explicit new recipe)"
    )
  if not isinstance(init_at_random_ep_len, bool):
    raise TypeError(f"init_at_random_ep_len must be bool, got {init_at_random_ep_len!r}")


def validate_formal_learn_arguments(
  num_learning_iterations, init_at_random_ep_len, *, expected_iterations: int
) -> None:
  """Require the official learn call to match this run's explicit budget."""
  if isinstance(num_learning_iterations, bool) or not isinstance(num_learning_iterations, int):
    raise TypeError(
      f"num_learning_iterations must be int, got {num_learning_iterations!r}"
    )
  if num_learning_iterations <= 0:
    raise ValueError("formal num_learning_iterations must be positive")
  if num_learning_iterations != expected_iterations:
    raise RuntimeError(
      "official learn budget must equal the explicit formal max_iterations "
      f"({expected_iterations}), got {num_learning_iterations}"
    )
  if not isinstance(init_at_random_ep_len, bool):
    raise TypeError(f"init_at_random_ep_len must be bool, got {init_at_random_ep_len!r}")


def _now_iso() -> str:
  return time.strftime("%Y-%m-%dT%H:%M:%S")


def _torch_precision_record() -> dict:
  """Actual torch matmul/cudnn FP32-precision state (raw values, no fiction)."""
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


def _resolve_engineering_cfg(train_cfg: dict) -> dict:
  """Validate the asdict-ed engineering agent cfg; resolve typed bindings.

  Every deviation is rejected explicitly — the runner never silently fixes a
  value (a wrong config stops the run, it does not become a new recipe).
  """
  if not isinstance(train_cfg, dict):
    raise TypeError(
      f"train_cfg must be the asdict() dict of ConfinedLocomotionRlCfg, "
      f"got {type(train_cfg).__name__}"
    )
  if train_cfg.get("class_name") != "ConfinedLocomotionRunner":
    raise RuntimeError(
      f"class_name must be 'ConfinedLocomotionRunner', got "
      f"{train_cfg.get('class_name')!r}"
    )
  seed = train_cfg.get("seed")
  if isinstance(seed, bool) or not isinstance(seed, int):
    raise RuntimeError(f"seed must be an explicit int, got {seed!r}")

  max_iterations = train_cfg.get("max_iterations")
  if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
    raise RuntimeError(f"max_iterations must be int, got {max_iterations!r}")
  if max_iterations <= 0:
    raise RuntimeError(
      "engineering cfg max_iterations must be positive: a non-positive budget executes nothing and "
      "is explicitly rejected (not a pass)"
    )
  if max_iterations > ENGINEERING_MAX_ITERATIONS:
    raise RuntimeError(
      f"engineering cfg max_iterations={max_iterations} exceeds this batch's "
      f"budget {ENGINEERING_MAX_ITERATIONS} (explicitly rejected; no "
      "auto-running)"
    )

  steps = train_cfg.get("num_steps_per_env")
  if steps != ROLLOUT_CONTROL_STEPS:
    raise RuntimeError(
      f"num_steps_per_env must be the confirmed {ROLLOUT_CONTROL_STEPS} "
      f"control steps, got {steps!r}"
    )
  if train_cfg.get("clip_actions") is not None:
    raise RuntimeError(
      "engineering cfg must keep clip_actions=None (raw actions are clipped "
      f"by the action term itself), got {train_cfg.get('clip_actions')!r}"
    )
  if not isinstance(train_cfg.get("resume"), bool):
    raise RuntimeError("resume must be an explicit bool (strict full load; default remains False)")
  if train_cfg.get("save_interval") != 0:
    raise RuntimeError(
      f"save_interval must be 0 (no periodic checkpoints), got "
      f"{train_cfg.get('save_interval')!r}"
    )
  if train_cfg.get("upload_model") is not False:
    raise RuntimeError("upload_model must be False (no remote upload)")
  if tuple(train_cfg.get("wandb_tags") or ()) != ():
    raise RuntimeError("wandb_tags must be empty (no remote logging)")
  if train_cfg.get("logger") != "tensorboard":
    raise RuntimeError(
      "logger must be the declared 'tensorboard' value (this runner writes "
      f"local JSON logs only and initializes no remote logger), got "
      f"{train_cfg.get('logger')!r}"
    )

  grid = _resolve_ground_grid(train_cfg.get("ground_grid"), context="engineering")
  try:
    GridPermutation.from_axes(grid.x_coords, grid.y_coords)
  except (TypeError, ValueError) as err:
    raise RuntimeError(f"engineering ground_grid has no unique mirror pairing: {err}") from err

  ppo_dict = train_cfg.get("ppo_config")
  if not isinstance(ppo_dict, dict):
    raise RuntimeError("ppo_config missing from the agent cfg dict")
  try:
    ppo_config = ConfinedPPOConfig(**ppo_dict)
  except (TypeError, ValueError) as err:
    raise RuntimeError(
      f"ppo_config does not reconstruct to the confirmed ConfinedPPOConfig "
      f"recipe (any deviation is rejected): {err}"
    ) from err

  # Observation-perturbation recipe: absent (or None) = the CLEAN path;
  # an explicit dict must reconstruct EXACTLY to a typed confirmed recipe —
  # unknown keys/numbers are rejected, never silently defaulted. The recipe
  # enters the runtime (streams/schedules) and the checkpoint identity; it
  # never opens any formal-training gate by itself.
  perturbation_dict = train_cfg.get("observation_perturbations")
  if perturbation_dict is None:
    perturbation_cfg = None
  elif isinstance(perturbation_dict, dict):
    from ..perturbations import ObservationPerturbationCfg, PerturbationContractError

    try:
      perturbation_cfg = ObservationPerturbationCfg.from_dict(perturbation_dict)
    except PerturbationContractError as err:
      raise RuntimeError(
        f"observation_perturbations does not reconstruct to the confirmed "
        f"typed recipe (any deviation is rejected): {err}"
      ) from err
  else:
    raise RuntimeError(
      "observation_perturbations must be absent/None (clean) or the asdict "
      f"dict of an ObservationPerturbationCfg, got "
      f"{type(perturbation_dict).__name__}"
    )

  symmetry_dict = train_cfg.get("symmetry")
  if symmetry_dict is None:
    symmetry_cfg = MirrorTrainingConfig()
  elif isinstance(symmetry_dict, dict):
    try:
      symmetry_cfg = MirrorTrainingConfig.from_dict(symmetry_dict)
    except (TypeError, ValueError) as err:
      raise RuntimeError(
        f"symmetry does not reconstruct to the typed mirror recipe: {err}"
      ) from err
  else:
    raise RuntimeError("symmetry must be an asdict MirrorTrainingConfig or None")

  return {
    "seed": int(seed),
    "max_iterations": int(max_iterations),
    "ppo_config": ppo_config,
    "grid": grid,
    "observation_perturbations": perturbation_cfg,
    "symmetry": symmetry_cfg,
  }


def _resolve_formal_cfg(train_cfg: dict) -> dict:
  """Strictly reconstruct the explicit formal recipe from the official asdict cfg."""
  if not isinstance(train_cfg, dict):
    raise TypeError("formal train_cfg must be the official asdict() agent dictionary")
  if train_cfg.get("formal_training_enabled") is not True:
    raise RuntimeError("formal_training_enabled must be explicitly True")
  if not isinstance(train_cfg.get("engineering_enabled"), bool):
    raise RuntimeError("engineering_enabled must be an explicit bool")
  if train_cfg.get("class_name") != "ConfinedLocomotionRunner":
    raise RuntimeError("class_name must be 'ConfinedLocomotionRunner'")

  seed = train_cfg.get("seed")
  if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
    raise RuntimeError(f"formal seed must be an explicit int in [0, 2**32), got {seed!r}")
  budget = train_cfg.get("max_iterations")
  if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
    raise RuntimeError(f"formal max_iterations must be a positive explicit int, got {budget!r}")
  num_envs = train_cfg.get("formal_num_envs")
  if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
    raise RuntimeError(f"formal_num_envs must be a positive explicit int, got {num_envs!r}")
  if train_cfg["engineering_enabled"] is True and (
    budget != ENGINEERING_MAX_ITERATIONS or num_envs > ENGINEERING_MAX_NUM_ENVS
  ):
    raise RuntimeError(
      "formal engineering validation is bounded to exactly one iteration and "
      f"at most {ENGINEERING_MAX_NUM_ENVS} environments"
    )

  if train_cfg.get("num_steps_per_env") != ROLLOUT_CONTROL_STEPS:
    raise RuntimeError(
      f"num_steps_per_env must be {ROLLOUT_CONTROL_STEPS}, got "
      f"{train_cfg.get('num_steps_per_env')!r}"
    )
  for name, expected in (
    ("clip_actions", None), ("save_interval", 0), ("upload_model", False),
    ("wandb_tags", ()), ("logger", "tensorboard"),
  ):
    actual = train_cfg.get(name)
    if name == "wandb_tags":
      actual = tuple(actual or ())
    if actual != expected:
      raise RuntimeError(f"formal cfg {name} must be {expected!r}, got {actual!r}")
  if not isinstance(train_cfg.get("resume"), bool):
    raise RuntimeError("resume must be an explicit bool")

  grid = _resolve_ground_grid(train_cfg.get("ground_grid"), context="formal")
  try:
    GridPermutation.from_axes(grid.x_coords, grid.y_coords)
  except (TypeError, ValueError) as err:
    raise RuntimeError(f"formal ground_grid lacks unique mirror pairing: {err}") from err

  ppo_dict = train_cfg.get("ppo_config")
  if not isinstance(ppo_dict, dict):
    raise RuntimeError("formal ppo_config must be the complete typed recipe")
  try:
    ppo_config = ConfinedPPOConfig(**ppo_dict)
  except (TypeError, ValueError) as err:
    raise RuntimeError(f"formal ppo_config differs from the accepted recipe: {err}") from err

  perturbation_dict = train_cfg.get("observation_perturbations")
  if not isinstance(perturbation_dict, dict):
    raise RuntimeError("formal training requires explicit ALL_THREE perturbations")
  try:
    perturbation_cfg = ObservationPerturbationCfg.from_dict(perturbation_dict)
  except (TypeError, ValueError) as err:
    raise RuntimeError(f"formal perturbations are invalid: {err}") from err
  if perturbation_cfg != ALL_THREE_PRESET:
    raise RuntimeError("formal training requires the accepted ALL_THREE perturbation preset")

  symmetry_dict = train_cfg.get("symmetry")
  if not isinstance(symmetry_dict, dict):
    raise RuntimeError("formal training requires an explicit symmetry recipe")
  try:
    symmetry_cfg = MirrorTrainingConfig.from_dict(symmetry_dict)
  except (TypeError, ValueError) as err:
    raise RuntimeError(f"formal symmetry recipe is invalid: {err}") from err
  if symmetry_cfg != MirrorTrainingConfig(enabled=True, mean_loss_coeff=0.6):
    raise RuntimeError("formal training requires ALL_THREE mirror augmentation at coeff 0.6")

  return {
    "seed": int(seed),
    "max_iterations": int(budget),
    "num_envs": int(num_envs),
    "ppo_config": ppo_config,
    "grid": grid,
    "observation_perturbations": perturbation_cfg,
    "symmetry": symmetry_cfg,
    "engineering_validation": train_cfg["engineering_enabled"] is True,
  }


class ConfinedLocomotionRunner(MjlabOnPolicyRunner):
  """Guarded official-chain runner for explicit engineering or formal recipes."""

  def __init__(self, env, train_cfg, log_dir=None, device="cuda:0", **kwargs) -> None:
    if kwargs:
      raise RuntimeError(
        f"unsupported runner entry options {sorted(kwargs)}: this runner has "
        "no registry/video/DDP options; nothing silently falls back to the "
        "reference defaults"
      )
    # ── FIRST GUARD: refuse the registered default before anything else ──
    # (before reading/resetting the env and before any network/Adam exists;
    # IMPLEMENTED is a ClassVar on the cfg class and is never read from the
    # dict, so no configuration string can re-enable it)
    if not isinstance(train_cfg, dict):
      raise RuntimeError(RUNNER_GUARDED_MSG)
    formal = train_cfg.get("formal_training_enabled") is True
    engineering = train_cfg.get("engineering_enabled") is True
    if formal:
      resolved = _resolve_formal_cfg(train_cfg)
      runtime_mode = "training"
      runner_identity = FORMAL_RUNNER_IDENTITY
    elif engineering:
      resolved = _resolve_engineering_cfg(train_cfg)
      runtime_mode = "engineering"
      runner_identity = RUNNER_IDENTITY
    else:
      raise RuntimeError(RUNNER_GUARDED_MSG)

    # ── device: actual CUDA required (launch_training must have selected a ─
    # GPU; run_train without CUDA_VISIBLE_DEVICES selects CPU and is NOT the
    # GPU gate) — validated BEFORE any local directory is created
    device_obj = torch.device(device)
    if device_obj.type != "cuda" or not torch.cuda.is_available():
      raise RuntimeError(
        f"the explicit {runtime_mode} path requires actual CUDA, got device={device!r} "
        "(CPU construction/config checks never claim the GPU gate; call the "
        "official launch_training with explicit gpu_ids)"
      )
    if device_obj.index is None:  # normalize "cuda" to the concrete index
      device_obj = torch.device(f"cuda:{torch.cuda.current_device()}")
    self.device = device_obj
    cuda_name = torch.cuda.get_device_name(device_obj)

    if log_dir is None:
      raise RuntimeError(
        "the runner requires the official log_dir (the official "
        "chain always passes it); no implicit cwd logging"
      )
    self.log_dir = Path(log_dir)
    self.log_dir.mkdir(parents=True, exist_ok=True)
    self.cfg = dict(train_cfg)
    self.seed = resolved["seed"]
    self.max_iterations = resolved["max_iterations"]
    self.runner_identity = runner_identity
    self.formal_training = formal
    self.engineering_validation = bool(formal and resolved.get("engineering_validation"))

    # ── TF32 OFF before any network/Adam exists (the official chain's own ──
    # configure_torch_backends() defaults to allow_tf32=True; re-disable it
    # here and RECORD the actual backend state — no dependency source is
    # modified, no AMP anywhere)
    configure_torch_backends(allow_tf32=False)
    # Recent torch backends expose an independent RNN precision override.
    rnn_backend = getattr(torch.backends.cudnn, "rnn", None)
    if rnn_backend is not None and hasattr(rnn_backend, "fp32_precision"):
      rnn_backend.fp32_precision = "ieee"
    precision_record = _torch_precision_record()

    # ── env/wrapper structural checks (fresh, officially-reset state) ──────
    from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper

    if not isinstance(env, RslRlVecEnvWrapper):
      raise TypeError(
        "env must be the official RslRlVecEnvWrapper (its construction "
        f"already performed the one initial reset), got {type(env).__name__}"
      )
    if env.clip_actions is not None:
      raise RuntimeError(
        f"wrapper.clip_actions must be None (raw-action seam), got "
        f"{env.clip_actions!r}"
      )
    self.env = env
    self.num_envs = int(env.num_envs)
    if formal and self.num_envs != resolved["num_envs"]:
      raise RuntimeError(
        f"constructed env count {self.num_envs} != explicit formal_num_envs "
        f"{resolved['num_envs']}"
      )
    if engineering and not formal and not 1 <= self.num_envs <= ENGINEERING_MAX_NUM_ENVS:
      raise RuntimeError(
        f"the engineering path allows at most {ENGINEERING_MAX_NUM_ENVS} envs "
        f"on this machine, got {self.num_envs} (override env.scene.num_envs "
        "BEFORE environment construction; the registered 64-env default is "
        "never constructed by checks)"
      )
    if self.engineering_validation and not 1 <= self.num_envs <= ENGINEERING_MAX_NUM_ENVS:
      raise RuntimeError(
        f"formal engineering validation allows at most {ENGINEERING_MAX_NUM_ENVS} "
        f"envs, got {self.num_envs}"
      )
    env_device = torch.device(env.unwrapped.device)
    if env_device.type != "cuda" or (
      device_obj.index is not None
      and env_device.index is not None
      and env_device.index != device_obj.index
    ):
      raise RuntimeError(
        f"env device {env_device} differs from the runner device {device_obj}"
      )

    grid = resolved["grid"]
    self.grid_cfg = grid
    self.ground_samples = int(grid.num_points)

    # ── each production component is created EXACTLY ONCE ─────────────────
    torch.manual_seed(self.seed)  # the ONE explicit initial network (no re-init)
    self.network = ConfinedLocomotionNetwork(
      mlp_activation=torch.nn.ELU(),
      init_fn=None,
      envelope_mode="raw",
      ground_samples=self.ground_samples,
    ).to(self.device)
    self.provider = IndependentDiagonalNormalProvider(
      device=self.device, generator_seed=derive_action_generator_seed(self.seed)
    )
    self.policy_adapter = ConfinedPolicyAdapter(
      network=self.network,
      distribution_provider=self.provider,
      distribution_identity=self.provider.distribution_identity(),
    )
    self.symmetry_config = resolved["symmetry"]
    self.runtime = TrainingRuntime(
      env,
      grid_cfg=grid,
      seed=self.seed,
      output_dir=self.log_dir / "confined_runtime",
      mode=runtime_mode,
      perturbation_cfg=resolved["observation_perturbations"],
    )
    self._build_mirror_mapping()
    self.trainer = self._build_trainer(resolved["ppo_config"])
    self.collector = TrainingCollector(self.runtime, self.policy_adapter, steps=ROLLOUT_CONTROL_STEPS)

    # ── counters / local records ───────────────────────────────────────────
    self.current_learning_iteration = 0
    self.total_control_batches = 0
    self.total_env_samples = 0
    self._learn_calls: list[dict] = []
    self._records: list[dict] = []
    self._observer_errors: list[str] = []
    self._source_records: list[dict] = []
    self._active_buffer = None
    self._log_failed: bool = False
    self._artifact_exported: bool = False
    self._final_checkpoint_path: Path | None = None
    self._pending_final_checkpoint_path: Path | None = None
    self._official_environment_closed = False
    self._environment_close_error: str | None = None
    #: Explicit loop lifecycle, maintained from the REAL method entries/exits
    #: (fresh → collecting → updating → ready per completed iteration; any
    #: terminal frozen state — fault or engineering stop — lands on
    #: 'faulted'). Save requires 'ready' + the full stable-boundary guards in
    #: checkpoint._require_stable_save_boundary; load accepts fresh/ready.
    self._lifecycle_phase: str = "fresh"

    self.recipe = {
      "schema": RUNNER_RECORD_SCHEMA,
      "mode": "formal_training" if formal else "engineering",
      "task": TASK_ID,
      "runner": self.runner_identity,
      "created": _now_iso(),
      "implemented_classvar": ConfinedLocomotionRlCfg.IMPLEMENTED,
      "capabilities": {
        "engineering_learn": (
          "bounded one-iteration validation" if not formal else
          "bounded formal validation" if self.engineering_validation else "not selected"
        ),
        "formal_training": (
          "bounded validation in real training mode; no "
          "learning-effect claim"
          if formal and self.engineering_validation
          else "explicit formal recipe; learning-effect acceptance deferred"
          if formal else "not selected"
        ),
        "learning_effect_ready": False,
        "learning_effect_acceptance": "separate evidence-based gate deferred",
        "checkpoint_resume": (
          "strict full same-GPU resume and explicit weights-only loading"
        ),
        "task_play": "mean-only checkpoint play via PlaySession",
      },
      "seed": self.seed,
      "max_iterations": self.max_iterations,
      "num_envs": self.num_envs,
      "device": {
        "requested": str(device),
        "resolved": str(self.device),
        "cuda_name": cuda_name,
        "torch_version": torch.__version__,
      },
      "torch_precision": precision_record,
      "wrapper": {
        "type": type(env).__name__,
        "clip_actions": env.clip_actions,
        "initial_reset": "performed once by the wrapper constructor; the "
        "runtime takes over the current T1 state without resetting",
      },
      "runtime_takeover": {
        "mode": runtime_mode,
        "fresh_counters_required": True,
        "verified_by": "TrainingRuntime construction (common_step_counter==0, "
        "episode_length_buf all zero, auto_reset=False, decimation=4, "
        "control dt=0.02 s)",
        "standing_walking_required": False,
      },
      "grid": {
        "declaration": grid.identity(),
        "identity": grid.identity(),
        "identity_hash": grid.identity_hash(),
        "note": "explicit GroundGridCfg; all coordinates and metadata are part of checkpoint identity",
      },
      "model": {
        "network_class": type(self.network).__name__,
        "mlp_activation": "ELU",
        "init_fn": None,
        "envelope_mode": "raw",
        "ground_samples": self.ground_samples,
        "unique_parameter_count": self.trainer.unique_parameter_count,
        "adam_param_group_count": len(self.trainer.optimizer.param_groups),
        "model_fingerprint_initial": model_state_fingerprint(self.policy_adapter),
        "distribution_identity_frozen": self.policy_adapter.identity_frozen,
        "initializations": 1,
      },
      "ppo_config": resolved["ppo_config"].summary(),
      "symmetry": self.trainer.config_identity()["symmetry"],
      "symmetry_mapping": self.trainer.symmetry_identity,
      "observation_perturbations": (
        None if resolved["observation_perturbations"] is None
        else resolved["observation_perturbations"].identity()
      ),
      "local_logging": "local JSON/JSONL only; no WandB/TensorBoard init",
    }
    try:
      self._write_json(self.log_dir / "run_recipe.json", self.recipe)
    except OSError as error:
      self._handle_log_io_failure(error, stage="construction/recipe")
      raise
    # Official train closes the wrapper/environment, not the runner.
    # Close the runtime so the transition boundary restores the env methods.
    self._official_env_close = self.env.unwrapped.close
    self.env.unwrapped.close = self._close_official_environment
    self._notify_observers("constructed", self.recipe)

  # ── construction helpers ─────────────────────────────────────────────────

  def _build_mirror_mapping(self) -> None:
    """Bind named joints/grid and prove the symmetry mount matches the optical site."""
    action = self.env.unwrapped.action_manager.get_term("residual_pd")
    self.joint_permutation = JointPermutation.from_joint_names(tuple(action._target_names))
    self.grid_permutation = GridPermutation.from_axes(
      self.grid_cfg.x_coords, self.grid_cfg.y_coords
    )
    from hexapod_mjlab.tasks.el_4090_lidar.mdp.lidar import resolve_lidar_mount

    resolved_mount = resolve_lidar_mount()
    if resolved_mount is None:
      raise RuntimeError("actual compiled lidar_sensor optical mount could not be resolved")
    resolved_pos, resolved_quat = resolved_mount
    pipeline_pos = self.runtime.pipeline._mount_t[0]
    pipeline_quat = self.runtime.pipeline._mount_q[0]
    resolver_pos = torch.tensor(resolved_pos, dtype=torch.float32, device=self.device)
    resolver_quat = torch.tensor(resolved_quat, dtype=torch.float32, device=self.device)
    if not torch.allclose(pipeline_pos, resolver_pos, atol=1e-7, rtol=0.0) or not torch.allclose(
      pipeline_quat, resolver_quat, atol=1e-7, rtol=0.0
    ):
      raise RuntimeError(
        "runtime scanner mount differs from resolve_lidar_mount optical site witness"
      )
    self.mount_optical_witness = {
      "source": "compiled lidar_sensor site via resolve_lidar_mount",
      "pipeline_source": "TrainingRuntime ConfinedScanPipeline actual mount",
      "position_matches": True,
      "quaternion_xyzw_matches": True,
      "absolute_tolerance": 1e-7,
      "resolved_pos_b": [float(value) for value in resolved_pos],
      "resolved_quat_xyzw": [float(value) for value in resolved_quat],
    }
    self.mount_pos_b = pipeline_pos.detach().clone()
    self.mount_quat_xyzw = pipeline_quat.detach().clone()

  def _build_trainer(self, ppo_config: ConfinedPPOConfig):
    from hexapod_mjlab.rl.confined_locomotion.ppo import ConfinedPPOTrainer

    return ConfinedPPOTrainer(
      policy_adapter=self.policy_adapter,
      grid_identity_hash=self.grid_cfg.identity_hash(),
      seed=self.seed,
      device=self.device,
      config=ppo_config,
      symmetry_config=self.symmetry_config,
      joint_permutation=self.joint_permutation,
      grid_permutation=self.grid_permutation,
      mount_pos_b=self.mount_pos_b,
      mount_quat_xyzw=self.mount_quat_xyzw,
      mount_optical_witness=self.mount_optical_witness,
      on_fault=self._on_ppo_fault,
    )

  # ── official-chain API ───────────────────────────────────────────────────

  def add_git_repo_to_log(self, repo_file_path: str) -> None:
    """Official-chain compatibility: LOCAL record only.

    The base implementation appends to the W&B-backed logger; this runner has
    no logger and performs no Git/W&B write and no remote connection — it
    records the source path and the implementation identity locally.
    """
    entry = {
      "repo_file_path": str(repo_file_path),
      "recorded_at": _now_iso(),
      "implementation": self.runner_identity,
      "mode": self.recipe["mode"],
      "note": "local record only; no git subprocess, no wandb, no remote",
    }
    self._source_records.append(entry)
    try:
      self._append_jsonl("source_repositories.jsonl", entry)
    except OSError as err:
      self._handle_log_io_failure(err, stage="source_repo_record")
      raise

  def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
    """Run the explicit engineering bound or the exact formal learn budget."""
    try:
      if self.formal_training:
        validate_formal_learn_arguments(
          num_learning_iterations,
          init_at_random_ep_len,
          expected_iterations=self.max_iterations,
        )
      else:
        validate_engineering_learn_arguments(
          num_learning_iterations, init_at_random_ep_len
        )
      self._guard_learn_state()
      self._learn_calls.append(
        {
          "call": len(self._learn_calls),
          "at": _now_iso(),
          "num_learning_iterations": num_learning_iterations,
          "init_at_random_ep_len": {
            "received": init_at_random_ep_len,
            "action": "ignored/disabled" if init_at_random_ep_len else "not requested",
            "episode_length_buf_randomized": False,
          },
        }
      )
      for _ in range(num_learning_iterations):
        self._run_one_iteration(init_at_random_ep_len)
      if self.formal_training:
        final_path = self.log_dir / (
          f"final_checkpoint_full_{self.current_learning_iteration:06d}.pt"
        )
        self._pending_final_checkpoint_path = final_path
        final_record = {
          "kind": "formal_final_full_checkpoint",
          "status": "pending",
          "checkpoint_committed": False,
          "path": str(final_path),
          "requested_new_iterations": int(num_learning_iterations),
          "global_learning_iteration": int(self.current_learning_iteration),
          "at": _now_iso(),
        }
        self._remember_record(final_record)
        # Required evidence is flushed before the atomic full-checkpoint
        # commit. If either log write fails, no new full file is created.
        self._append_jsonl_record(final_record)
        self._write_summary()
        self._final_checkpoint_path = self.save(
          final_path,
          infos={
            "purpose": "formal_training_run_final",
            "requested_new_iterations": int(num_learning_iterations),
            "updates_executed_before_save": int(num_learning_iterations),
            "global_learning_iteration": int(self.current_learning_iteration),
          },
        )
        self._pending_final_checkpoint_path = None
        final_record["status"] = "committed"
        final_record["checkpoint_committed"] = True
        final_record["committed_at"] = _now_iso()
      else:
        # The NON-RESUMABLE engineering artifact remains one-shot evidence; a resumed
        # engineering runner skips a duplicate export.
        if self._artifact_exported:
          skip = {
            "kind": "artifact_export_skipped",
            "at": _now_iso(),
            "reason": "already exported once (bounded evidence; no per-iteration copies)",
          }
          self._remember_record(skip)
          self._append_jsonl_record(skip)
        else:
          self.export_engineering_artifact(reason="learn_completed")
        self._write_summary()
    except TrainingEntryNotPassed:
      self._lifecycle_phase = "faulted"  # terminal: engineering stop froze the run
      self._write_summary_best_effort()
      self._notify_observers(
        "learn_not_passed", self._records[-1] if self._records else None
      )
      self._close_after_learn_error()
      raise
    except BaseException as error:
      self._lifecycle_phase = "faulted"
      if isinstance(error, OSError) and not self._log_failed:
        self._handle_log_io_failure(error, stage="learn/record_or_artifact")
      elif not self.runtime._stopped and not self.runtime._faulted:
        self.runtime._save_fault(error, phase="runner/learn")
      record = {
        "kind": "learn_error",
        "at": _now_iso(),
        "error": repr(error),
        "traceback": traceback.format_exc(),
        "current_learning_iteration": self.current_learning_iteration,
        "total_control_batches": self.total_control_batches,
        "optimizer_step_count": self.trainer.optimizer_step_count,
      }
      self._remember_record(record)
      self._append_jsonl_best_effort(record)
      self._write_summary_best_effort()
      self._notify_observers("learn_failed", record)
      self._close_after_learn_error()
      raise
    self._notify_observers("learn_finished", self._summary_snapshot())

  def _close_after_learn_error(self) -> None:
    """Close the runner-owned env on learn failure without masking that error."""
    try:
      self._close_official_environment()
    except BaseException as close_error:
      self._observer_errors.append(f"environment close after learn error: {close_error!r}")

  # ── full-checkpoint surface ──────────────────────────────────────────────

  def save(self, path, infos=None):
    """FULL same-GPU resume checkpoint at the stable boundary (atomic, new file).

    Delegates to :func:`.checkpoint.save_full_checkpoint`, which enforces the
    completed-24/20-iteration stable boundary (phase 'ready', idle context,
    consumed/empty buffer, no pending terminal rows, consistent counters)
    and refuses mid-collect/mid-update/faulted/stopped/closed states. The
    official ``infos`` are recorded inside the payload.
    """
    return save_full_checkpoint(self, path, infos=infos)

  def load(self, path, load_cfg=None, strict=True, map_location=None) -> dict:
    """Complete same-device restore of a full checkpoint (official signature).

    ``load_cfg`` must be empty (actor-only configs are refused — the
    weights-only mode is the explicit :meth:`load_weights`), ``strict`` must
    be True and ``map_location`` must stay on the same CUDA device. The
    loader validates identity/payload completely BEFORE mutating anything,
    then applies weights → Adam → physics (in-place, forward, rewrite) →
    lifecycle caches → PRNGs last; a fresh official construction's reset/
    first-scan side effects are recorded and fully overwritten.
    """
    feature_cache = getattr(
      getattr(self, "collector", None), "feature_cache", None
    )
    if feature_cache is not None:
      feature_cache.clear()
    return load_full_checkpoint(
      self, path, load_cfg=load_cfg, strict=strict, map_location=map_location
    )

  def load_weights(
    self,
    path,
    *,
    network=None,
    provider=None,
    expected_grid_identity=None,
    allow_source_change: bool = False,
  ) -> dict:
    """Weights-only load for task play: full key set + optional 18-dim log_std.

    Defaults to THIS runner's live network/provider/grid identity when the
    caller does not supply its own freshly constructed objects (play builds a
    network without Adam or behavior sampling). No Adam/physics/RNG/
    lifecycle is touched; the returned report never claims resume.
    """
    feature_cache = getattr(
      getattr(self, "collector", None), "feature_cache", None
    )
    if feature_cache is not None:
      feature_cache.clear()
    return load_weights_only(
      path,
      network=self.network if network is None else network,
      provider=self.provider if provider is None else provider,
      expected_grid_identity=(
        self.grid_cfg.identity() if expected_grid_identity is None
        else expected_grid_identity
      ),
      allow_source_change=allow_source_change,
    )

  # ── explicitly refused pending capabilities ──────────────────────────────

  def get_inference_policy(self, device=None):
    raise RuntimeError(f"get_inference_policy() refused: {PENDING_CAPABILITY_MSG}")

  def export_policy_to_onnx(self, path, filename="policy.onnx", verbose=False) -> None:
    raise RuntimeError(f"export_policy_to_onnx() refused: {PENDING_CAPABILITY_MSG}")

  def eval(self, *args, **kwargs):
    raise RuntimeError(f"eval() refused: {PENDING_CAPABILITY_MSG}")

  def play(self, *args, **kwargs):
    raise RuntimeError(f"play() refused: {TASK_PLAY_ENTRY_MSG}")

  # ── engineering artifact (review only; NOT a model checkpoint) ───────────

  def export_engineering_artifact(self, *, reason: str) -> Path:
    """Export a clearly NON-RESUMABLE engineering artifact once per runner.

    Contains the network/Adam/metrics for review only; the naming must never
    be mistaken for a model checkpoint (full checkpoints have a separate path). Not
    automatically copied per iteration — explicit checks export it.
    """
    if self._artifact_exported:
      raise RuntimeError(
        "engineering artifact already exported once (bounded evidence; no "
        "per-iteration copies)"
      )
    directory = self.log_dir / "engineering_artifact_NON_RESUMABLE"
    directory.mkdir(parents=False, exist_ok=False)
    payload = {
      "kind": "engineering_artifact",
      "mode": self.recipe["mode"],
      "resumable": False,
      "reason": str(reason),
      "schema": RUNNER_RECORD_SCHEMA,
      "task": TASK_ID,
      "runner": RUNNER_IDENTITY,
      "created": _now_iso(),
      "seed": self.seed,
      "device": self.recipe["device"],
      "torch_precision": self.recipe["torch_precision"],
      "grid_identity_hash": self.grid_cfg.identity_hash(),
      "current_learning_iteration": self.current_learning_iteration,
      "total_control_batches": self.total_control_batches,
      "total_env_samples": self.total_env_samples,
      "optimizer_step_count": self.trainer.optimizer_step_count,
      "learning_rate": self.trainer.learning_rate,
      "model_fingerprint": model_state_fingerprint(self.policy_adapter),
      "distribution_identity_frozen": self.policy_adapter.identity_frozen,
      "adapter_state_dict": self.policy_adapter.state_dict(),
      "algorithm_state": self.trainer.state_dict(),
      "behavior_generator_state": self.provider.generator_state(),
      "summary": self._summary_snapshot(),
    }
    torch.save(payload, directory / "engineering_artifact_NON_RESUMABLE.pt")
    (directory / "NOT_A_MODEL_CHECKPOINT.txt").write_text(
      "Engineering artifact for REVIEW ONLY.\n\n"
      "This is NOT a model checkpoint and NOT resumable: complete "
      "checkpoint/resume uses the separate strict full-checkpoint surface. It contains the "
      "engineering network/Adam/metrics of one bounded run and must never be "
      "loaded to continue training.\n"
    )
    self._artifact_exported = True
    return directory

  def _close_official_environment(self) -> None:
    if self._official_environment_closed:
      return
    self._official_environment_closed = True
    original = self._official_env_close
    close_error = None
    try:
      self.close()
    except BaseException as err:
      close_error = err
    finally:
      # An exception while restoring the runtime must never skip the actual
      # official env close. Learn failure callers swallow only this close
      # error so the original training exception remains primary.
      try:
        original()
      except BaseException as err:
        if close_error is None:
          close_error = err
    self._environment_close_error = None if close_error is None else repr(close_error)
    if close_error is not None:
      # The official env.close was attempted, but the runtime/context restore
      # or the env close itself failed.  Freeze every continuation surface
      # without pretending runtime.close completed or snapshotting a closed
      # simulator as a new physical boundary.
      self._lifecycle_phase = "faulted"
      self.runtime._faulted = True
    self._write_summary_best_effort()
    if close_error is not None:
      raise close_error

  def close(self) -> None:
    """Restore the runtime and environment instance hooks; no reset."""
    if hasattr(self, "_official_env_close"):
      self.env.unwrapped.close = self._official_env_close
    self.runtime.close()

  # ── one bounded iteration ────────────────────────────────────────────────

  def _run_one_iteration(self, init_at_random_ep_len: bool) -> None:
    iteration_index = self.current_learning_iteration
    self._lifecycle_phase = "collecting"
    kwargs_record = {
      "init_at_random_ep_len": {
        "received": init_at_random_ep_len,
        "action": "ignored/disabled" if init_at_random_ep_len else "not requested",
        "episode_length_buf_randomized": False,
      }
    }
    fingerprint_before = model_state_fingerprint(self.policy_adapter)
    result = self.collector.collect()
    collect_summary = result.summary()
    runtime_stopped = bool(getattr(self.runtime, "_stopped", False))
    stopped_reason = collect_summary.get("stopped_reason")
    if (not result.complete) or stopped_reason is not None or runtime_stopped:
      record = {
        "kind": "iteration",
        "iteration": iteration_index,
        "status": "not_passed",
        "at": _now_iso(),
        "learn_kwargs": kwargs_record,
        "collect": collect_summary,
        "runtime_stop_record": self.runtime.stop_record,
        "model_fingerprint": collect_summary.get("model_fingerprint"),
        "note": (
          "engineering stop (natural independent fall or formal done, "
          "including one landing exactly on the last step): the runtime "
          "already saved the terminal evidence; no update is executed and "
          "the run stops honestly"
        ),
      }
      self._remember_record(record)
      self._append_jsonl_record(record)  # an IO failure here must also stop the run
      self._lifecycle_phase = "faulted"  # terminal: engineering stop froze the run
      raise TrainingEntryNotPassed(record)

    # Complete and NOT stopped: persist the collection boundary BEFORE the
    # update so a local log-IO failure stops the run before any Adam step.
    pre_record = {
      "kind": "iteration_collect_boundary",
      "iteration": iteration_index,
      "status": "collect_completed",
      "at": _now_iso(),
      "learn_kwargs": kwargs_record,
      "collect": collect_summary,
      "model_fingerprint": collect_summary.get("model_fingerprint"),
    }
    try:
      self._append_jsonl_record(pre_record)
    except OSError as err:
      self._handle_log_io_failure(err, stage=f"iteration{iteration_index}/pre_update_record")
      raise

    self._active_buffer = result.buffer
    self._lifecycle_phase = "updating"
    try:
      metrics = self.trainer.update(result.buffer)
    finally:
      self._active_buffer = None
      feature_cache = getattr(
        getattr(self, "collector", None), "feature_cache", None
      )
      if feature_cache is not None:
        feature_cache.clear()
    post_summary = result.buffer.summary()
    # consume()+clear() leave an EMPTY boundary: zero steps filled, nothing
    # consumable (clear resets the consumed flag along with the data).
    if post_summary.get("steps_filled") != 0 or post_summary.get("complete"):
      raise RuntimeError(
        f"post-update buffer boundary is not empty: {post_summary}"
      )

    self.current_learning_iteration += 1
    self.total_control_batches += int(result.completed_steps)
    self.total_env_samples += int(result.completed_steps) * self.num_envs
    # Completed 24-collect/20-update boundary: this is the (only) save point.
    self._lifecycle_phase = "ready"
    minibatches = metrics.get("minibatches") or []
    kls = [entry.get("kl_mean") for entry in minibatches if entry.get("kl_mean") is not None]
    lrs = [entry.get("lr_after") for entry in minibatches if entry.get("lr_after") is not None]
    record = {
      "kind": "iteration",
      "iteration": iteration_index,
      "status": "completed",
      "at": _now_iso(),
      "learn_kwargs": kwargs_record,
      "collect": collect_summary,
      "update": metrics,
      "post_update_buffer": post_summary,
      "model_fingerprint_before": fingerprint_before,
      "counters": {
        "current_learning_iteration": self.current_learning_iteration,
        "total_control_batches": self.total_control_batches,
        "total_env_samples": self.total_env_samples,
        "optimizer_step_count": self.trainer.optimizer_step_count,
        "learning_rate_final": self.trainer.learning_rate,
        "kl_mean_over_minibatches": (
          {"min": min(kls), "mean": sum(kls) / len(kls), "max": max(kls)}
          if kls
          else None
        ),
        "lr_range_over_minibatches": (
          {"min": min(lrs), "max": max(lrs)} if lrs else None
        ),
      },
    }
    self._remember_record(record)
    self._append_jsonl_record(record)
    self._notify_observers("iteration_completed", record)

  # ── PPO fault callback (update faults only) ─────────────────────────────

  def _on_ppo_fault(self, stage: str, error: BaseException) -> None:
    """PPO ``on_fault`` callback: save the scene + diagnostics; never mask.

    The runtime fault save keeps the physics/window/commands scene; the
    algorithm diagnostics and the current (un-consumed) rollout are saved as
    PURE DIAGNOSTICS — never a resumable continuation. Collection faults and
    engineering falls are NOT routed here (runtime/collector own them).
    """
    phase = f"runner/update/{stage}"
    try:
      self.runtime._save_fault(error, phase=phase)
    except BaseException:  # saving must never mask the original fault
      pass
    try:
      index = len(self._records)
      directory = self.log_dir / f"update_fault_{index:02d}_NON_RESUMABLE"
      directory.mkdir(parents=False, exist_ok=False)
      payload = {
        "kind": "update_fault_diagnostics",
        "mode": self.recipe["mode"],
        "resumable": False,
        "stage": str(stage),
        "phase": phase,
        "error": repr(error),
        "created": _now_iso(),
        "learning_rate": self.trainer.learning_rate,
        "update_count": self.trainer.update_count,
        "optimizer_step_count": self.trainer.optimizer_step_count,
        "model_fingerprint": model_state_fingerprint(self.policy_adapter),
        "adapter_state_dict": self.policy_adapter.state_dict(),
        "algorithm_state": self.trainer.state_dict(),
      }
      torch.save(payload, directory / "algorithm_diagnostics_NON_RESUMABLE.pt")
      (directory / "fault.json").write_text(
        json.dumps({key: value for key, value in payload.items()
                    if key not in ("adapter_state_dict", "algorithm_state")},
                   indent=2, default=str)
      )
      buffer = self._active_buffer
      if buffer is not None:
        try:
          torch.save(
            {"kind": "partial_rollout_diagnostics", "resumable": False,
             "buffer": buffer},
            directory / "partial_rollout_NON_RESUMABLE.pt",
          )
        except BaseException:
          (directory / "partial_rollout_summary_fallback.json").write_text(
            json.dumps(buffer.summary(), indent=2, default=str)
          )
    except BaseException:
      pass

  # ── local logging ────────────────────────────────────────────────────────

  @staticmethod
  def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str))

  def _append_jsonl(self, name: str, record: dict) -> None:
    with (self.log_dir / name).open("a") as stream:
      stream.write(json.dumps(record, default=str) + "\n")

  def _append_jsonl_record(self, record: dict) -> None:
    record.setdefault("mode", self.recipe["mode"])
    self._append_jsonl("iteration_records.jsonl", record)

  def _remember_record(self, record: dict) -> None:
    """Keep formal-run memory bounded; the JSONL file remains full history."""
    self._records.append(record)
    if not self.formal_training:
      return
    if record.get("kind") == "iteration" and record.get("status") == "completed":
      latest_fault = [
        old for old in self._records
        if old.get("kind") in ("learn_error", "log_io_failure")
      ][-1:]
      self._records = latest_fault + [record]
      return
    last_complete = next(
      (
        old for old in reversed(self._records[:-1])
        if old.get("kind") == "iteration" and old.get("status") == "completed"
      ),
      None,
    )
    self._records = ([] if last_complete is None else [last_complete]) + [record]

  def _append_jsonl_best_effort(self, record: dict) -> None:
    try:
      self._append_jsonl_record(record)
    except OSError as err:
      self._observer_errors.append(f"iteration log IO failure: {err!r}")

  def _handle_log_io_failure(self, error: OSError, *, stage: str) -> None:
    """A local log IO failure: save what is feasible, then stop the run."""
    self._log_failed = True
    try:
      self.runtime._save_fault(error, phase=f"runner/log_io/{stage}")
    except BaseException:
      pass
    emergency = {
      "kind": "log_io_failure",
      "mode": self.recipe["mode"],
      "stage": stage,
      "error": repr(error),
      "at": _now_iso(),
      "records_in_memory": len(self._records),
      "note": "run stopped before the next update; in-memory records kept "
              "as the feasible evidence",
    }
    self._remember_record(emergency)
    try:
      self._write_json(self.log_dir / "log_io_failure.json", emergency)
    except OSError:
      pass

  def _summary_snapshot(self) -> dict:
    minibatch_entries = []
    for record in self._records:
      if record.get("kind") == "iteration" and record.get("status") == "completed":
        for entry in (record.get("update") or {}).get("minibatches") or []:
          minibatch_entries.append(entry)
    kls = [e.get("kl_mean") for e in minibatch_entries if e.get("kl_mean") is not None]
    return {
      "schema": RUNNER_RECORD_SCHEMA,
      "mode": self.recipe["mode"],
      "task": TASK_ID,
      "runner": self.runner_identity,
      "implemented_classvar": ConfinedLocomotionRlCfg.IMPLEMENTED,
      "capabilities": self.recipe["capabilities"],
      "seed": self.seed,
      "num_envs": self.num_envs,
      "device": self.recipe["device"],
      "torch_precision": self.recipe["torch_precision"],
      "grid_identity_hash": self.grid_cfg.identity_hash(),
      "model_fingerprint_final": model_state_fingerprint(self.policy_adapter),
      "distribution_identity_frozen": self.policy_adapter.identity_frozen,
      "unique_parameter_count": self.trainer.unique_parameter_count,
      "adam_param_group_count": len(self.trainer.optimizer.param_groups),
      "learn_calls": list(self._learn_calls),
      "current_learning_iteration": self.current_learning_iteration,
      "requested_learning_iterations": self.max_iterations,
      "formal_training": self.formal_training,
      "engineering_validation": self.engineering_validation,
      "learning_effect_ready": False,
      "learning_effect_acceptance": "separate evidence-based gate deferred",
      "final_checkpoint_path": (
        None if self._final_checkpoint_path is None else str(self._final_checkpoint_path)
      ),
      "final_checkpoint_commit": {
        "state": (
          "pending_after_previous_commit"
          if self._final_checkpoint_path is not None and self._pending_final_checkpoint_path is not None
          else "pending" if self._pending_final_checkpoint_path is not None
          else "committed" if self._final_checkpoint_path is not None
          else "none"
        ),
        "committed_path": (
          None if self._final_checkpoint_path is None else str(self._final_checkpoint_path)
        ),
        "pending_path": (
          None if self._pending_final_checkpoint_path is None
          else str(self._pending_final_checkpoint_path)
        ),
      },
      "formal_run_completed": bool(
        self.formal_training
        and self._final_checkpoint_path is not None
        and self._official_environment_closed
        and self._environment_close_error is None
        and self._lifecycle_phase == "ready"
      ),
      "environment_close": {
        "completed": bool(self._official_environment_closed and self._environment_close_error is None),
        "error": self._environment_close_error,
      },
      "total_control_batches": self.total_control_batches,
      "total_env_samples": self.total_env_samples,
      "lifecycle_phase": getattr(self, "_lifecycle_phase", None),
      "checkpoint_events_log": "checkpoint_events.jsonl (local JSONL only)",
      "optimizer_step_count": self.trainer.optimizer_step_count,
      "learning_rate_final": self.trainer.learning_rate,
      "kl_stats_over_all_minibatches": (
        {"count": len(kls), "min": min(kls), "mean": sum(kls) / len(kls), "max": max(kls)}
        if kls
        else None
      ),
      "runtime_state": {
        "mode": self.runtime.mode,
        "control_steps_taken": self.runtime.control_steps_taken,
        "stopped": bool(getattr(self.runtime, "_stopped", False)),
        "faulted": bool(getattr(self.runtime, "_faulted", False)),
      },
      "trainer_faulted": self.trainer.faulted,
      "observer_errors": list(self._observer_errors),
      "source_repos": list(self._source_records),
      "records": list(self._records),
      "record_retention": {
        "in_memory": "latest completed iteration plus latest final/fault record"
        if self.formal_training else "all records for bounded engineering runs",
        "complete_iteration_history": "iteration_records.jsonl",
        "kl_stats_scope": "retained records" if self.formal_training else "all records",
      },
    }

  def _write_summary(self) -> None:
    self._write_json(self.log_dir / "training_summary.json", self._summary_snapshot())

  def _write_summary_best_effort(self) -> None:
    try:
      self._write_summary()
    except OSError as err:
      self._observer_errors.append(f"summary log IO failure: {err!r}")

  # ── guards ───────────────────────────────────────────────────────────────

  def _guard_learn_state(self) -> None:
    if self._log_failed:
      raise RuntimeError(
        "local log IO failed earlier: the run is stopped (no further updates)"
      )
    phase = getattr(self, "_lifecycle_phase", None)
    if phase not in ("fresh", "ready"):
      raise RuntimeError(
        f"learn refused: runner lifecycle phase is {phase!r} (iterations only "
        "start from a fresh construction or a completed ready boundary)"
      )
    if getattr(self.runtime, "_closed", False):
      raise TrainingRuntimeFrozenError("training runtime closed")
    if getattr(self.runtime, "_faulted", False):
      raise TrainingRuntimeFrozenError(
        "training runtime frozen after a fault; learn cannot recover it"
      )
    if getattr(self.runtime, "_stopped", False):
      raise TrainingRuntimeFrozenError(
        "training runtime stopped after an engineering fall/done; the whole "
        "run ends there (no auto-reset continuation)"
      )
    if self.trainer.faulted:
      raise RuntimeError(
        f"the PPO trainer is faulted (stage={self.trainer.fault_stage!r}); "
        "learn refuses to continue"
      )

  # ── observers ────────────────────────────────────────────────────────────

  def _notify_observers(self, event: str, payload) -> None:
    for observer in tuple(_ENGINEERING_OBSERVERS):
      try:
        observer(event, self, payload)
      except BaseException as err:  # evidence channel must not break the run
        self._observer_errors.append(f"observer {event}: {err!r}")
