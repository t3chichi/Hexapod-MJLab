"""Registered confined flat-ground locomotion task and current user API."""

from mjlab.tasks.registry import register_mjlab_task

from .flat_env_cfg import (
  TASK_ID,
  make_flat_env_cfg,
  make_training_env_cfg,
)
from .ground import GroundGridCfg
from .play import (
  PlayConfigError,
  PlaySession,
  make_play_env_cfg,
  peek_checkpoint_header,
  resolve_play_perturbation_cfg,
)
from .rl_cfg import (
  ConfinedLocomotionRlCfg,
  confined_locomotion_engineering_rl_cfg,
  confined_locomotion_formal_rl_cfg,
  confined_locomotion_rl_cfg,
)
from .rl.runner import ConfinedLocomotionRunner
from .scanner import ConfinedLidarCfg, ConfinedScanPipeline

REGISTERED_TRAINING_NUM_ENVS = 64

register_mjlab_task(
  task_id=TASK_ID,
  env_cfg=make_training_env_cfg(num_envs=REGISTERED_TRAINING_NUM_ENVS),
  play_env_cfg=make_flat_env_cfg(play=True),
  rl_cfg=confined_locomotion_rl_cfg(),
  runner_cls=ConfinedLocomotionRunner,
)

__all__ = [
  "TASK_ID",
  "REGISTERED_TRAINING_NUM_ENVS",
  "GroundGridCfg",
  "make_flat_env_cfg",
  "make_training_env_cfg",
  "make_play_env_cfg",
  "PlayConfigError",
  "PlaySession",
  "peek_checkpoint_header",
  "resolve_play_perturbation_cfg",
  "ConfinedLocomotionRunner",
  "ConfinedLocomotionRlCfg",
  "confined_locomotion_rl_cfg",
  "confined_locomotion_engineering_rl_cfg",
  "confined_locomotion_formal_rl_cfg",
  "ConfinedLidarCfg",
  "ConfinedScanPipeline",
]
