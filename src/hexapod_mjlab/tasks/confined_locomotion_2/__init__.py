"""Registered independent hierarchical recurrent flat-ground task."""

from mjlab.tasks.registry import register_mjlab_task

from .flat_env_cfg import TASK_ID, make_flat_env_cfg, make_training_env_cfg
from .rl_cfg import (
    ConfinedLocomotionRlCfg, confined_locomotion_engineering_rl_cfg,
    confined_locomotion_formal_rl_cfg, confined_locomotion_rl_cfg,
)
from .rl.runner import ConfinedLocomotionRunner
from .play import PlaySession, make_play_env_cfg


register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=make_training_env_cfg(num_envs=64),
    play_env_cfg=make_play_env_cfg(),
    rl_cfg=confined_locomotion_rl_cfg(),
    runner_cls=ConfinedLocomotionRunner,
)

__all__ = ["TASK_ID", "ConfinedLocomotionRlCfg", "ConfinedLocomotionRunner", "PlaySession",
           "make_flat_env_cfg", "make_training_env_cfg", "make_play_env_cfg",
           "confined_locomotion_rl_cfg", "confined_locomotion_engineering_rl_cfg",
           "confined_locomotion_formal_rl_cfg"]
