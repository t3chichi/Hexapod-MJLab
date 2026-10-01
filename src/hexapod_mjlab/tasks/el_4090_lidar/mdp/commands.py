"""Velocity command with the old task's deadzone semantics.

Builds on the official ``UniformVelocityCommand`` (mjlab tasks/velocity/mdp/
velocity_command.py), configured to match the old ``El4090LidarHeadCfg.commands``:

* heading_command=True with all envs heading-controlled (rel_heading_envs=1.0)
  and heading_control_stiffness=0.5 == the old ``clip(0.5 * wrap_to_pi(err))``
  conversion (the official implementation clips to the ang_vel_z range
  (-1.2, 1.2), the old clipped to (-1, 1); see MIGRATION.md deviation note);
* resampling every 20 s (resampling_time_range=(20, 20));
* ranges lin_vel_x/y (-1, 1), ang_vel_z (-1.2, 1.2), heading (-3.14, 3.14).

Added here: the old deterministic command deadzone
(``cmd_deadzone = 0.2``: linear commands with norm <= 0.2 are zeroed at
resample — old ``_resample_commands`` tail).

The old commands curriculum (expanding lin_vel_x by ±0.5 up to ±2.5 when
tracking reward exceeds 80% of scale) lives in
:mod:`hexapod_mjlab.tasks.el_4090_lidar.mdp.curriculums`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)


@dataclass(kw_only=True)
class El4090VelocityCommandCfg(UniformVelocityCommandCfg):
  cmd_deadzone: float = 0.2
  """Zero linear commands whose norm is below this at resample (old default)."""

  def build(self, env):
    return El4090VelocityCommand(self, env)


class El4090VelocityCommand(UniformVelocityCommand):
  cfg: El4090VelocityCommandCfg

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    # Old _resample_commands tail: small-command deadzone.
    norm = torch.norm(self.vel_command_b[env_ids, :2], dim=1)
    self.vel_command_b[env_ids, :2] *= (
      norm > self.cfg.cmd_deadzone
    ).unsqueeze(1)
