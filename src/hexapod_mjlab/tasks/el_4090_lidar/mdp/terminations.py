"""Termination terms — port of the old ``check_termination`` extensions.

Old stack: base terminations are the episode time limit plus (enabled in the
head config, ``enable_fall_termination``)::

    flipped   = projected_gravity[:, 2] > -0.1
    low_base  = base_pos[:, 2] < 0.12
    reset    |= flipped | low_base
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

FALL_PROJECTED_GRAVITY_Z_THRESHOLD = -0.1
FALL_BASE_HEIGHT_THRESHOLD = 0.12


def fell_over(
  env: ManagerBasedRlEnv,
  gravity_z_threshold: float = FALL_PROJECTED_GRAVITY_Z_THRESHOLD,
  base_height_threshold: float = FALL_BASE_HEIGHT_THRESHOLD,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Old fall termination: flipped (gravity z above threshold) or low base."""
  asset: Entity = env.scene[asset_cfg.name]
  flipped = asset.data.projected_gravity_b[:, 2] > gravity_z_threshold
  low_base = asset.data.root_link_pos_w[:, 2] < base_height_threshold
  return flipped | low_base
