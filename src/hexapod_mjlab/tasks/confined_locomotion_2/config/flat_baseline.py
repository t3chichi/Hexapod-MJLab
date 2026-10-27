"""Flat Baseline environment factories and explicit RL recipes.

Environment construction and execution stay in the existing task modules;
this module is the task-specific configuration entry point.
"""

from dataclasses import dataclass, field
from typing import ClassVar

from mjlab.rl import RslRlBaseRunnerCfg

from hexapod_mjlab.rl.confined_locomotion.ppo import MirrorTrainingConfig
from hexapod_mjlab.rl.confined_locomotion_2.ppo import TemporalPPOConfig
from hexapod_mjlab.common.perturbations import (
    ALL_THREE_PRESET, ObservationPerturbationCfg,
)

from ..flat_env_cfg import TASK_ID, make_flat_env_cfg, make_training_env_cfg
from ..play import make_play_env_cfg


@dataclass
class ConfinedLocomotionRlCfg(RslRlBaseRunnerCfg):
    IMPLEMENTED: ClassVar[bool] = True
    class_name: str = "ConfinedLocomotionRunner"
    experiment_name: str = "confined_locomotion_flat_2"
    max_iterations: int = 0
    num_steps_per_env: int = 24
    clip_actions: float | None = None
    save_interval: int = 50
    execution_mode: str = "low_memory"
    logger: str = "tensorboard"
    upload_model: bool = False
    engineering_enabled: bool = False
    formal_training_enabled: bool = False
    formal_num_envs: int | None = None
    # Kept as a plain mapping so the official mjlab config serializer can
    # pass it through to the guarded runner without knowing task-local types.
    lidar_config: dict = field(default_factory=lambda: {
        "raycast_mode": "full", "sparse_selection_path": None,
    })
    observation_perturbations: ObservationPerturbationCfg | None = None
    ppo_config: TemporalPPOConfig = field(default_factory=TemporalPPOConfig)
    symmetry: MirrorTrainingConfig = field(default_factory=MirrorTrainingConfig)


def _seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an explicit uint32 integer")


def confined_locomotion_rl_cfg():
    """Registration describes the task; it does not authorize a learning run."""
    return ConfinedLocomotionRlCfg()


def confined_locomotion_engineering_rl_cfg(*, seed, perturbations=None, symmetry=None):
    _seed(seed)
    return ConfinedLocomotionRlCfg(
        seed=seed, max_iterations=1, engineering_enabled=True,
        observation_perturbations=perturbations,
        symmetry=symmetry if symmetry is not None else MirrorTrainingConfig(),
        run_name="engineering", experiment_name="confined_locomotion_flat_2_engineering",
    )


def confined_locomotion_formal_rl_cfg(*, seed, num_envs, max_iterations):
    _seed(seed)
    for name, value in (("num_envs", num_envs), ("max_iterations", max_iterations)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive explicit integer")
    return ConfinedLocomotionRlCfg(
        seed=seed, max_iterations=max_iterations, formal_num_envs=num_envs,
        formal_training_enabled=True, observation_perturbations=ALL_THREE_PRESET,
        symmetry=MirrorTrainingConfig(enabled=True, mean_loss_coeff=0.6),
        run_name="formal_training",
    )
