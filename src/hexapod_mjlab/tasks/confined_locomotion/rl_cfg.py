"""Typed runner recipes for the registered confined flat-ground task.

The registry holds the confirmed current S1-187 grid but stays guarded:
no seed/budget/entry authorization is inferred from this grid. Engineering
and formal factories require an explicit full GroundGridCfg; formal training
also requires explicit seed, environment count, and iteration budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from mjlab.rl import RslRlBaseRunnerCfg

from hexapod_mjlab.rl.confined_locomotion.ppo import (
  ROLLOUT_STEPS,
  ConfinedPPOConfig,
  MirrorTrainingConfig,
)
from .ground import VALIDITY_GEOMETRY_ONLY_FLAT_TRAINING, GroundGridCfg
from .ground_presets import s1_187_ground_grid
from .perturbations import ALL_THREE_PRESET, ObservationPerturbationCfg

__all__ = [
  "ConfinedLocomotionRlCfg",
  "confined_locomotion_formal_rl_cfg",
  "confined_locomotion_engineering_rl_cfg",
  "confined_locomotion_rl_cfg",
]

#: Confirmed rollout length (24 control steps per update).
ROLLOUT_CONTROL_STEPS = ROLLOUT_STEPS

#: Engineering iteration budget of this batch — exactly ONE learn iteration
#: per explicit engineering run (4050-bounded; larger budgets are rejected).
ENGINEERING_MAX_ITERATIONS = 1

#: Hard env-count bound of the engineering path on this machine (4050).
ENGINEERING_MAX_NUM_ENVS = 2

@dataclass
class ConfinedLocomotionRlCfg(RslRlBaseRunnerCfg):
  """Guarded typed recipe for the registered task.

  ``IMPLEMENTED`` is a ``ClassVar`` (never a field, never in ``asdict``):
  it cannot be changed through any configuration value. The implementation
  supports explicit engineering and formal recipes; learning-effect
  acceptance remains a separate evidence-based decision. The registered
  default is guarded (missing seed/iteration authorization) and rejected
  before network/Adam construction even though the current grid is known.
  """

  IMPLEMENTED: ClassVar[bool] = True

  class_name: str = "ConfinedLocomotionRunner"
  max_iterations: int = 0
  num_steps_per_env: int = ROLLOUT_CONTROL_STEPS
  clip_actions: float | None = None
  logger: str = "tensorboard"
  upload_model: bool = False
  wandb_tags: tuple[str, ...] = ()
  resume: bool = False
  save_interval: int = 0  # stable-boundary saves only; no periodic autosave
  experiment_name: str = "confined_locomotion_flat"

  #: Explicit bounded engineering gate (never set by the registry).
  engineering_enabled: bool = False
  #: Complete serialized grid identity, required by runnable recipes.
  #: The registered recipe selects S1-187; standalone manually built
  #: recipes may still supply None and are rejected by the runner.
  ground_grid: GroundGridCfg | None = None
  #: Separate explicit production-runtime gate; the registered default stays off.
  formal_training_enabled: bool = False
  #: Caller-specified environment count for the formal recipe.
  formal_num_envs: int | None = None
  #: Typed observation-perturbation recipe (serializable, asdict-safe).
  #: None = CLEAN everywhere: the registered default and the engineering
  #: default stay clean; the field can EXPRESS the three confirmed
  #: perturbations (gap / range / proprio, single or combined) for explicit
  #: recipes; selecting one does not establish learning-effect acceptance.
  observation_perturbations: ObservationPerturbationCfg | None = None
  #: Frozen typed numeric recipe (construction-rejecting; asdict-safe).
  ppo_config: ConfinedPPOConfig = field(default_factory=ConfinedPPOConfig)
  #: Typed same-sign mirror augmentation; OFF in the registered default.
  symmetry: MirrorTrainingConfig = field(default_factory=MirrorTrainingConfig)


def confined_locomotion_rl_cfg() -> ConfinedLocomotionRlCfg:
  """Registered default factory: guarded full recipe, learning refused.

  ``max_iterations=0`` and both training authorization flags OFF cause
  the runner to refuse before network/Adam construction. The default S1-187
  grid merely describes current task geometry, not an approved training
  budget/seed. Explicit training entry still preflights the entire recipe.
  """
  return ConfinedLocomotionRlCfg(ground_grid=s1_187_ground_grid())


def confined_locomotion_engineering_rl_cfg(
  *,
  seed: int,
  ground_grid: GroundGridCfg,
  perturbations: ObservationPerturbationCfg | None = None,
  symmetry: MirrorTrainingConfig | None = None,
) -> ConfinedLocomotionRlCfg:
  """Build an explicit engineering recipe (bounded to one iteration).

  Sets ``engineering_enabled=True`` and ``max_iterations=1`` with explicit
  seed and complete grid identity. ``num_envs`` is NOT set here — the
  engineering entry overrides ``env.scene.num_envs`` (≤ 2) BEFORE
  environment construction; the runner independently rejects more than 2.

  ``perturbations`` selects the observation-perturbation recipe of the
  check — default None (clean) or an explicit ``ObservationPerturbationCfg``
  built from the confirmed presets (``perturbations.GAP_ONLY_PRESET`` /
  ``GAP_FIXED_JITTER_OFF_PRESET`` / ``RANGE_ONLY_PRESET`` /
  ``PROPRIO_ONLY_PRESET`` / ``ALL_THREE_PRESET`` / ``preset_by_name``); the
  recipe is recorded in every artifact and enters the checkpoint identity.
  """
  if isinstance(seed, bool) or not isinstance(seed, int):
    raise ValueError(f"engineering seed must be an explicit int, got {seed!r}")
  if not isinstance(ground_grid, GroundGridCfg):
    raise TypeError("engineering ground_grid must be an explicitly built GroundGridCfg")
  if ground_grid.validity_mode != VALIDITY_GEOMETRY_ONLY_FLAT_TRAINING:
    raise ValueError(
      "engineering requires "
      f"{VALIDITY_GEOMETRY_ONLY_FLAT_TRAINING!r}, got {ground_grid.validity_mode!r}"
    )
  if perturbations is not None and not isinstance(
    perturbations, ObservationPerturbationCfg
  ):
    raise ValueError(
      "perturbations must be an ObservationPerturbationCfg or None "
      f"(clean default), got {perturbations!r}"
    )
  if symmetry is not None and not isinstance(symmetry, MirrorTrainingConfig):
    raise ValueError(
      "symmetry must be an explicit MirrorTrainingConfig or None "
      "(mirror-off default)"
    )
  return ConfinedLocomotionRlCfg(
    seed=seed,
    max_iterations=ENGINEERING_MAX_ITERATIONS,
    engineering_enabled=True,
    ground_grid=ground_grid,
    observation_perturbations=perturbations,
    symmetry=symmetry if symmetry is not None else MirrorTrainingConfig(),
    experiment_name="confined_locomotion_flat_engineering",
    run_name="engineering",
  )


def confined_locomotion_formal_rl_cfg(
  *,
  seed: int,
  ground_grid: GroundGridCfg,
  num_envs: int,
  max_iterations: int,
) -> ConfinedLocomotionRlCfg:
  """Build a formal training recipe from caller-supplied essentials.

  There are intentionally no defaults for seed, grid, environment count, or
  update budget. The bounded validation variant may be produced explicitly
  with ``dataclasses.replace(cfg, engineering_enabled=True, formal_num_envs=2, max_iterations=1)``;
  the runner then enforces at most two environments and exactly one update.
  Learning-effect acceptance is reported separately from this runnable
  implementation recipe.
  """
  if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
    raise ValueError(f"formal seed must be an explicit int in [0, 2**32), got {seed!r}")
  for name, value, minimum in (
    ("num_envs", num_envs, 1),
    ("max_iterations", max_iterations, 1),
  ):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
      raise ValueError(f"formal {name} must be an int >= {minimum}, got {value!r}")
  if not isinstance(ground_grid, GroundGridCfg):
    raise TypeError(
      "formal ground_grid must be an explicitly built GroundGridCfg, got "
      f"{type(ground_grid).__name__}"
    )
  if ground_grid.validity_mode != VALIDITY_GEOMETRY_ONLY_FLAT_TRAINING:
    raise ValueError(
      "formal training requires "
      f"{VALIDITY_GEOMETRY_ONLY_FLAT_TRAINING!r}, got {ground_grid.validity_mode!r}"
    )
  try:
    from hexapod_mjlab.rl.confined_locomotion.symmetry import GridPermutation

    GridPermutation.from_axes(ground_grid.x_coords, ground_grid.y_coords)
  except (TypeError, ValueError) as err:
    raise ValueError(f"formal ground grid has no unique mirror pairing: {err}") from err
  return ConfinedLocomotionRlCfg(
    seed=seed,
    max_iterations=max_iterations,
    engineering_enabled=False,
    formal_training_enabled=True,
    formal_num_envs=num_envs,
    ground_grid=ground_grid,
    observation_perturbations=ALL_THREE_PRESET,
    symmetry=MirrorTrainingConfig(enabled=True, mean_loss_coeff=0.6),
    experiment_name="confined_locomotion_flat",
    run_name="formal_training",
  )
