"""Event terms — exact ports of old-stack randomization events."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

from mjlab.envs.mdp.events import resolve_env_ids  # noqa: F401

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def push_by_overwriting_velocity(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  velocity_range: dict[str, tuple[float, float]],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Old ``_push_robots``: OVERWRITE root linear velocity x/y with a uniform
  sample (the official ``push_by_setting_velocity`` *adds* a delta instead —
  not semantics-equivalent, hence this port)."""
  env_ids = resolve_env_ids(env, env_ids)
  asset: Entity = env.scene[asset_cfg.name]
  vel_w = asset.data.root_link_vel_w[env_ids].clone()
  lo_x, hi_x = velocity_range.get("x", (0.0, 0.0))
  lo_y, hi_y = velocity_range.get("y", (0.0, 0.0))
  n = len(env_ids)
  vel_w[:, 0] = torch.empty(n, device=env.device).uniform_(lo_x, hi_x)
  vel_w[:, 1] = torch.empty(n, device=env.device).uniform_(lo_y, hi_y)
  asset.write_root_link_velocity_to_sim(vel_w, env_ids=env_ids)
