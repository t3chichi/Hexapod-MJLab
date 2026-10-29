"""Shared confined-locomotion package with one v2-family registration entry.

Flat Baseline, Terrain Memory and Tripod2 Low have independent real
configurations; all three are registered here and use the shared temporal PPO
implementation while retaining task-owned environment and runtime semantics.
"""

from mjlab.tasks.registry import register_mjlab_task

from .config import terrain_memory_config as terrain_memory
from .config.flat_baseline_config import (
    ConfinedLocomotionRlCfg, confined_locomotion_engineering_rl_cfg,
    confined_locomotion_formal_rl_cfg, confined_locomotion_rl_cfg,
)
from .flat_env_cfg import TASK_ID, make_flat_env_cfg, make_training_env_cfg
from .play import PlaySession, make_play_env_cfg
from .rl.runner import ConfinedLocomotionRunner
from .tripod_env_cfg import TASK_ID as TRIPOD_TASK_ID
from .tripod_env_cfg import make_env_cfg as make_tripod_env_cfg
from .config.tripod2_low_config import TripodRlCfg
from .rl.tripod_runner import TripodRunner

register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=make_training_env_cfg(num_envs=64),
    play_env_cfg=make_play_env_cfg(),
    rl_cfg=confined_locomotion_rl_cfg(),
    runner_cls=ConfinedLocomotionRunner,
)

register_mjlab_task(
    task_id=terrain_memory.TASK_ID,
    env_cfg=terrain_memory.make_training_env_cfg(num_envs=64),
    play_env_cfg=terrain_memory.make_play_env_cfg(),
    rl_cfg=terrain_memory.terrain_memory_rl_cfg(),
    runner_cls=ConfinedLocomotionRunner,
)

register_mjlab_task(
    task_id=TRIPOD_TASK_ID,
    env_cfg=make_tripod_env_cfg(),
    play_env_cfg=make_tripod_env_cfg(num_envs=1, play=True),
    rl_cfg=TripodRlCfg(),
    runner_cls=TripodRunner,
)

__all__ = ["TASK_ID", "ConfinedLocomotionRlCfg", "ConfinedLocomotionRunner", "PlaySession",
           "make_flat_env_cfg", "make_training_env_cfg", "make_play_env_cfg",
           "confined_locomotion_rl_cfg", "confined_locomotion_engineering_rl_cfg",
           "confined_locomotion_formal_rl_cfg"]
