"""Registered Flat_2 task identity and its task-level environment interface.

The single flat environment implementation lives in
``hexapod_mjlab.common.flat_env``. This module keeps the task ID and the
environment names already used by task, checkpoint, play, test and experiment
code; it does not build a second environment.
"""

from hexapod_mjlab.common.flat_env import (
  FLAT_GROUND_FRICTION,
  GENERIC_EPISODE_LENGTH_S,
  GENERIC_NUM_ENVS,
  make_flat_env_cfg,
  make_training_env_cfg,
)
from hexapod_mjlab.common.robot import (
  DECIMATION,
  FOOT_BODIES,
  JOINT_NAMES,
  SIM_TIMESTEP_S,
)

TASK_ID = "El-4090-Confined-Locomotion-Flat_2"

__all__ = [
  "TASK_ID",
  "FLAT_GROUND_FRICTION",
  "GENERIC_EPISODE_LENGTH_S",
  "GENERIC_NUM_ENVS",
  "make_flat_env_cfg",
  "make_training_env_cfg",
  "DECIMATION",
  "SIM_TIMESTEP_S",
  "FOOT_BODIES",
  "JOINT_NAMES",
]
