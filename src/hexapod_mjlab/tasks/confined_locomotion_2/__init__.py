"""Shared confined-locomotion package with explicit per-task registration.

Flat Baseline and Terrain Memory have independent real configurations; both
reuse the recurrent implementation and guarded runner.
"""

from mjlab.tasks.registry import register_mjlab_task

from .config.flat_baseline import (
    TASK_ID, make_flat_env_cfg, make_training_env_cfg, make_play_env_cfg,
    ConfinedLocomotionRlCfg, confined_locomotion_engineering_rl_cfg,
    confined_locomotion_formal_rl_cfg, confined_locomotion_rl_cfg,
)
from .rl.runner import ConfinedLocomotionRunner
from .play import PlaySession


register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=make_training_env_cfg(num_envs=64),
    play_env_cfg=make_play_env_cfg(),
    rl_cfg=confined_locomotion_rl_cfg(),
    runner_cls=ConfinedLocomotionRunner,
)

from .config import terrain_memory

register_mjlab_task(
    task_id=terrain_memory.TASK_ID,
    env_cfg=terrain_memory.make_training_env_cfg(num_envs=64),
    play_env_cfg=terrain_memory.make_play_env_cfg(),
    rl_cfg=terrain_memory.terrain_memory_rl_cfg(),
    runner_cls=ConfinedLocomotionRunner,
)

__all__ = ["TASK_ID", "ConfinedLocomotionRlCfg", "ConfinedLocomotionRunner", "PlaySession",
           "make_flat_env_cfg", "make_training_env_cfg", "make_play_env_cfg",
           "confined_locomotion_rl_cfg", "confined_locomotion_engineering_rl_cfg",
           "confined_locomotion_formal_rl_cfg"]
