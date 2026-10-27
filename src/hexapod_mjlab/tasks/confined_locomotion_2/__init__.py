"""Shared confined-locomotion package with one v2-family registration entry.

Flat Baseline, Terrain Memory and Tripod2 Low have independent real
configurations; all three are registered here and share the recurrent/legacy
implementations where their recipes explicitly say so.
"""

from mjlab.tasks.registry import register_mjlab_task

from .config import terrain_memory
from .config.flat_baseline import (
    ConfinedLocomotionRlCfg, confined_locomotion_engineering_rl_cfg,
    confined_locomotion_formal_rl_cfg, confined_locomotion_rl_cfg,
)
from .flat_env_cfg import TASK_ID, make_flat_env_cfg, make_training_env_cfg
from .play import PlaySession, make_play_env_cfg
from .rl.runner import ConfinedLocomotionRunner
from .tripod2_low.env import TASK_ID as TRIPOD_TASK_ID
from .tripod2_low.env import make_env_cfg as make_tripod_env_cfg
from .tripod2_low.runner import TripodRlCfg, TripodRunner

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
