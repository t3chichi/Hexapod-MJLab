"""Robot-specific task configurations + registration for the EL4090 LiDAR task."""

import warnings

from mjlab.tasks.registry import register_mjlab_task

from hexapod_mjlab.tasks.el_4090_lidar.rl import El4090LidarRunner

from .env_cfgs import el_4090_lidar_head_env_cfg
from .rl_cfg import el_4090_lidar_ppo_runner_cfg

EL4090_TASK_ID = "El-4090-Lidar"

# dr.body_mass warns on every call (advisory: use pseudo_inertia for uniform
# density changes). The old stack's added_base_mass IS a point-mass-at-COM
# model, which is exactly what dr.body_mass implements, so the warning is
# noise here (it fires once at startup; the filter is belt-and-braces).
warnings.filterwarnings(
  "ignore",
  message="dr.body_mass only randomizes mass",
  category=UserWarning,
)

# Module-level registration (mirrors the official mjlab task packages):
# importing this package registers the task in the mjlab registry; the
# entry-point in pyproject.toml makes `import mjlab` import this package.
register_mjlab_task(
  task_id=EL4090_TASK_ID,
  env_cfg=el_4090_lidar_head_env_cfg(),
  play_env_cfg=el_4090_lidar_head_env_cfg(play=True),
  rl_cfg=el_4090_lidar_ppo_runner_cfg(),
  runner_cls=El4090LidarRunner,
)


def register_el_4090_lidar_head() -> None:
  """Re-register the task (only valid after a registry clear; re-registering
  with the same id raises in ``register_mjlab_task`` — kept for symmetry with
  the official packages' export surface)."""
  register_mjlab_task(
    task_id=EL4090_TASK_ID,
    env_cfg=el_4090_lidar_head_env_cfg(),
    play_env_cfg=el_4090_lidar_head_env_cfg(play=True),
    rl_cfg=el_4090_lidar_ppo_runner_cfg(),
    runner_cls=El4090LidarRunner,
  )
