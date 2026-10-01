"""Observation terms for the EL4090 LiDAR task.

Proprio group (66-dim, exact port of the old ``compute_observations`` layout,
order preserved)::

    base_lin_vel * 2.0 | base_ang_vel * 0.25 | projected_gravity |
    commands[:3] * [2.0, 2.0, 0.25] | (dof_pos - default) * 1.0 |
    dof_vel * 0.05 | actions                              (3+3+3+3+18+18+18)

Noise: mjlab applies noise BEFORE ``scale``; the old stack applied the
(obs-scale-multiplied) noise width AFTER scaling. To reproduce the old
effective widths, the raw noise magnitudes here are the old widths divided
by the term scale (e.g. lin_vel: old +/-0.3 effective -> raw +/-0.15 with
scale 2.0).

Privileged group ``height_grid`` (187-dim): old ``aux_obs_buf`` semantics
``clip(root_z - 0.5 - measured_heights, -1, 1) * 5.0`` with the 17x11
forward grid (x in [1.2, 3.8], y in [-0.9, 0.9]) yaw-rotated about the base
and sampled with the old min-of-3-neighbors kernel.

LiDAR terms: proximal 256x3 + distal 640x3 blocks, appended after proprio in
the ``policy`` group (total 2754). Produced by the real raycast pipeline in
:mod:`hexapod_mjlab.tasks.el_4090_lidar.mdp.lidar` (warp LidarSensor vs the M2
terrain mesh, 10 Hz tick, pool sampling, old LidarWrapper semantics).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .lidar import lidar_distal_obs, lidar_prox_obs
from .terrain import get_el_4090_terrain_view, sample_terrain_heights_min3

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

# Old obs scales (El4090SpiderCfg.normalization.obs_scales).
OBS_SCALE_LIN_VEL = 2.0
OBS_SCALE_ANG_VEL = 0.25
OBS_SCALE_DOF_POS = 1.0
OBS_SCALE_DOF_VEL = 0.05
OBS_SCALE_HEIGHT = 5.0
COMMANDS_SCALE = (OBS_SCALE_LIN_VEL, OBS_SCALE_LIN_VEL, OBS_SCALE_ANG_VEL)

# Old noise widths (noise_scales * noise_level 1.5), split into raw magnitude
# (pre-scale) for mjlab's noise->scale ordering.
NOISE_LIN_VEL = 0.1 * 1.5  # effective ±0.3 after ×2.0
NOISE_ANG_VEL = 0.2 * 1.5  # effective ±0.075 after ×0.25
NOISE_GRAVITY = 0.05 * 1.5  # effective ±0.075 (scale 1.0)
NOISE_DOF_POS = 0.05 * 1.5  # effective ±0.075 (scale 1.0)
NOISE_DOF_VEL = 1.5 * 1.5  # effective ±0.1125 after ×0.05

# Measured-height grid (MEASURED_GRID_* in the old config).
MEASURED_GRID_X_RANGE = (1.2, 3.8)
MEASURED_GRID_Y_RANGE = (-0.9, 0.9)
MEASURED_GRID_X_COUNT = 17
MEASURED_GRID_Y_COUNT = 11
HEIGHT_GRID_DIM = MEASURED_GRID_X_COUNT * MEASURED_GRID_Y_COUNT  # 187

# LiDAR pool layout (pd network constants).
PROXIMAL_POINTS = 256
DISTAL_POINTS = 64
DISTAL_HISTORY = 10
PROPRIO_DIM = 66
POLICY_OBS_DIM = (
  PROPRIO_DIM + PROXIMAL_POINTS * 3 + DISTAL_HISTORY * DISTAL_POINTS * 3
)  # 2754


def _get_heights_grid(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Cached 17x11 grid points in the base (yaw) frame, flattened x-major."""
  grid = env.__dict__.get("_el_4090_height_grid")
  if grid is None:
    xs = torch.linspace(
      MEASURED_GRID_X_RANGE[0],
      MEASURED_GRID_X_RANGE[1],
      MEASURED_GRID_X_COUNT,
      device=env.device,
    )
    ys = torch.linspace(
      MEASURED_GRID_Y_RANGE[0],
      MEASURED_GRID_Y_RANGE[1],
      MEASURED_GRID_Y_COUNT,
      device=env.device,
    )
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)  # (187, 2)
    env._el_4090_height_grid = grid
  return grid


def measured_heights(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terrain heights under the yaw-rotated 17x11 grid (old ``_get_heights``).

  Returns shape ``(B, 187)`` in meters (already multiplied by the vertical
  scale, like the old ``measured_heights``).
  """
  asset: Entity = env.scene[asset_cfg.name]
  view = get_el_4090_terrain_view(env)
  grid = _get_heights_grid(env)  # (N, 2)

  # quat_apply_yaw: rotate by base yaw only.
  yaw = asset.data.heading_w  # (B,)
  cos, sin = torch.cos(yaw), torch.sin(yaw)
  x = grid[None, :, 0]
  y = grid[None, :, 1]
  rot_x = cos[:, None] * x - sin[:, None] * y
  rot_y = sin[:, None] * x + cos[:, None] * y
  base_xy = asset.data.root_link_pos_w[:, None, :2]
  points_w = torch.stack([rot_x + base_xy[..., 0], rot_y + base_xy[..., 1]], dim=-1)

  return sample_terrain_heights_min3(view, points_w)


def height_grid_obs(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Privileged height-grid observation (old ``aux_obs_buf``).

  ``clip(root_z - 0.5 - measured_heights, -1, 1) * 5.0``, shape ``(B, 187)``.
  """
  asset: Entity = env.scene[asset_cfg.name]
  root_z = asset.data.root_link_pos_w[:, 2:3]
  heights = measured_heights(env, asset_cfg)
  return (
    torch.clip(root_z - 0.5 - heights, -1.0, 1.0) * OBS_SCALE_HEIGHT
  )


def base_lin_vel(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_lin_vel_b


def base_ang_vel(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_ang_vel_b


__all__ = [
  "base_lin_vel",
  "base_ang_vel",
  "measured_heights",
  "height_grid_obs",
  "lidar_prox_obs",
  "lidar_distal_obs",
  "PROPRIO_DIM",
  "POLICY_OBS_DIM",
  "HEIGHT_GRID_DIM",
  "PROXIMAL_POINTS",
  "DISTAL_POINTS",
  "DISTAL_HISTORY",
  "OBS_SCALE_LIN_VEL",
  "OBS_SCALE_ANG_VEL",
  "OBS_SCALE_DOF_POS",
  "OBS_SCALE_DOF_VEL",
  "OBS_SCALE_HEIGHT",
  "COMMANDS_SCALE",
  "NOISE_LIN_VEL",
  "NOISE_ANG_VEL",
  "NOISE_GRAVITY",
  "NOISE_DOF_POS",
  "NOISE_DOF_VEL",
]
