"""Independent el4090_tripod2_low recipe using the frozen Flat Baseline policy."""
from mjlab.tasks.registry import register_mjlab_task

from .env import TASK_ID, make_env_cfg
from .runner import TripodRlCfg, TripodRunner

register_mjlab_task(task_id=TASK_ID, env_cfg=make_env_cfg(),
                   play_env_cfg=make_env_cfg(num_envs=1, play=True),
                   rl_cfg=TripodRlCfg(), runner_cls=TripodRunner)
