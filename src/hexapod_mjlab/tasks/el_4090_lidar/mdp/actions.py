"""Joint PD-torque action term (port of the old ``_compute_torques`` "P" mode).

Old semantics (el_4090.py / legged_gym)::

    torques = p_gains * (action * action_scale + default_dof_pos - dof_pos)
              - d_gains * dof_vel
    torques = clip(torques, -torque_limits, +torque_limits)   # ±80 Nm
    ctrl = torques  (direct torque control through <motor> actuators)

The policy outputs 18 joint-position targets. mjlab's action manager calls
``apply_action()`` before EVERY physics substep
(``ManagerBasedRlEnv.step`` decimation loop), so the PD gains are re-applied
at the 200 Hz physics rate against instantaneous q/qd with the target held
for the 4 substeps — identical to the old loop which recomputed torques each
``dof_pos`` refresh inside its decimation loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.envs.mdp.actions.actions import BaseAction, BaseActionCfg


@dataclass(kw_only=True)
class JointPDTorqueActionCfg(BaseActionCfg):
  """Configuration for the PD-torque joint action (motors/effort targets)."""

  stiffness: float = 130.0
  """Proportional gain kp [N*m/rad] (old control.stiffness; uniform 130)."""
  damping: float = 2.0
  """Derivative gain kd [N*m*s/rad] (old control.damping; uniform 2)."""
  effort_limit: float = 80.0
  """Torque clip (old torque_limits = ±80 Nm, also the motor ctrlrange)."""
  use_default_offset: bool = True

  def build(self, env):
    return JointPDTorqueAction(self, env)


class JointPDTorqueAction(BaseAction):
  """Position targets -> torch PD -> clipped torques -> effort targets."""

  cfg: JointPDTorqueActionCfg

  def __init__(self, cfg: JointPDTorqueActionCfg, env):
    super().__init__(cfg=cfg, env=env)
    if cfg.use_default_offset:
      self._offset = self._entity.data.default_joint_pos[:, self._target_ids].clone()

  def apply_actions(self) -> None:
    target = self._processed_actions  # a * scale + default_dof_pos
    joint_pos = self._entity.data.joint_pos[:, self._target_ids]
    joint_vel = self._entity.data.joint_vel[:, self._target_ids]
    torques = (
      self.cfg.stiffness * (target - joint_pos) - self.cfg.damping * joint_vel
    )
    torques = torch.clamp(torques, -self.cfg.effort_limit, self.cfg.effort_limit)
    self._entity.set_joint_effort_target(torques, joint_ids=self._target_ids)
