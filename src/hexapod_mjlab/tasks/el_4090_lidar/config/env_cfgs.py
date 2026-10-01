"""EL4090 head-LiDAR pillar-field environment configuration."""

from mjlab.envs import ManagerBasedRlEnvCfg

from hexapod_mjlab.tasks.el_4090_lidar.el_4090_lidar_env_cfg import (
  make_el_4090_lidar_env_cfg,
)


def el_4090_lidar_head_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the EL4090 LiDAR head pillar-field task configuration."""
  return make_el_4090_lidar_env_cfg(play=play)
