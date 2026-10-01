"""Reward terms — exact port of the old ``el4090_lidar_head`` reward set.

Sources (old repo, resolved MRO): ``legged_robot_rew_mixin.py`` for the base
penalties, ``el_4090/spider_nomal/el_4090.py`` for ``feet_air_time``,
``shank_perp2ground``, the sync/async gait helpers and the per-step foot
contact timers, ``elspider_air/elspider.py::ElSpider._reward_gait_2_step``
for the hexapod 2-step gait (the active override in the old MRO), and
``el_4090_lidar_head.py``/configs for weights.

A small per-policy-step state (``_El4090StepState``) is cached on the env and
refreshed once per step on first access (guarded by ``common_step_counter``):
the old env updated foot contact timers and stored ``last_dof_vel`` inside
``post_physics_step`` before reward computation; replicating that here keeps
identical semantics under mjlab's manager architecture.

Foot ordering: the old stack sorted foot bodies alphabetically
(LB, LF, LM, RB, RF, RM); the gait groups are defined on that order:
group 1 = (LB, LF, RM) = indices (0, 1, 5), group 2 = (LM, RB, RF) = (2, 3, 4).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

from .observations import measured_heights

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

#: Old cfg rewards constants.
TRACKING_SIGMA = 0.25
BASE_HEIGHT_TARGET = 0.25  # head config overrides the spider default 0.5
MAX_CONTACT_FORCE = 500.0
FEET_AIR_TIME_TARGET = 0.25  # spider cfg rewards.feet_air_time_target
SPEED_MIN = 0.2  # LeggedRobotRewMixin speed_min

#: Feet in the old alphabetical order.
FEET_NAMES = ("LB_FOOT", "LF_FOOT", "LM_FOOT", "RB_FOOT", "RF_FOOT", "RM_FOOT")
HFE_NAMES = ("RF_HFE", "RM_HFE", "RB_HFE", "LF_HFE", "LM_HFE", "LB_HFE")
KFE_NAMES = ("RF_KFE", "RM_KFE", "RB_KFE", "LF_KFE", "LM_KFE", "LB_KFE")

#: Old gait groups (alphabetical foot indices).
GAIT_GROUP_1 = (0, 1, 5)  # LB, LF, RM
GAIT_GROUP_2 = (2, 3, 4)  # LM, RB, RF


class _El4090StepState:
  """Per-policy-step state ported from the old ``post_physics_step``."""

  def __init__(self, env: ManagerBasedRlEnv, feet_sensor_name: str):
    self.env = env
    self.feet_sensor_name = feet_sensor_name
    self.step_marker = -1
    n = env.num_envs
    d = env.device
    self.last_dof_vel = torch.zeros(n, 18, device=d)
    self._prev_dof_vel = torch.zeros(n, 18, device=d)
    self.feet_air_time = torch.zeros(n, 6, device=d)
    self.feet_contact_time = torch.zeros(n, 6, device=d)
    self.last_contacts = torch.zeros(n, 6, dtype=torch.bool, device=d)
    self.feet_air_time_on_contact = torch.zeros(n, 6, device=d)
    self.feet_first_contact = torch.zeros(n, 6, dtype=torch.bool, device=d)
    # Alphabetical reorder permutation for the contact sensor primaries.
    sensor: ContactSensor = env.scene[feet_sensor_name]
    primary_names = list(sensor.primary_names)
    self._perm = torch.tensor(
      [primary_names.index(name) for name in FEET_NAMES], device=d
    )

  def feet_force(self) -> torch.Tensor:
    """(B, 6, 3) net contact force per foot, alphabetical order.

    The z component is NEGATED to match the old IsaacGym convention
    (contact force on the foot points up, +z; the MuJoCo netforce
    sensordata carries the opposite sign). Norm-based terms are unaffected;
    the ``z > 1 N`` contact threshold and the stumble comparison use |z| or
    the corrected sign.
    """
    sensor: ContactSensor = self.env.scene[self.feet_sensor_name]
    force = sensor.data.force
    assert force is not None
    n_prim = len(sensor.primary_names)
    force = force.view(self.env.num_envs, n_prim, 3).clone()
    force = force[:, self._perm]
    force[..., 2] = -force[..., 2]
    return force

  def update(self) -> None:
    """Advance foot timers + last_dof_vel (old ordering: before rewards)."""
    env = self.env
    if self.step_marker == env.common_step_counter:
      return
    self.step_marker = env.common_step_counter

    robot: Entity = env.scene["robot"]
    # dof_acc bookkeeping: expose previous step's dof_vel, store current.
    self._prev_dof_vel = self.last_dof_vel
    self.last_dof_vel = robot.data.joint_vel.clone()

    # _update_feet_contact_timers (el_4090.py), dt = policy step dt.
    dt = env.step_dt
    contact = self.feet_force()[..., 2] > 1.0
    contact_filt = torch.logical_or(contact, self.last_contacts)

    air_time_next = self.feet_air_time + dt
    contact_time_next = self.feet_contact_time + dt
    self.feet_first_contact = (self.feet_air_time > 0.0) & contact_filt
    self.feet_air_time_on_contact = air_time_next
    self.feet_air_time = air_time_next * ~contact_filt
    self.feet_contact_time = contact_time_next * contact_filt
    self.last_contacts = contact


def _get_state(env: ManagerBasedRlEnv, feet_sensor_name: str) -> _El4090StepState:
  state = env.__dict__.get("_el_4090_step_state")
  if state is None:
    state = _El4090StepState(env, feet_sensor_name)
    env._el_4090_step_state = state
  state.update()
  return state


def reset_gait_state(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  feet_sensor_name: str = "feet_ground_contact",
) -> None:
  """Port of the old ``reset_idx`` timer zeroing (event term, mode="reset").

  Old stack zeroed ``feet_air_time``/``feet_contact_time``/``last_dof_vel``
  (and the implicit ``feet_air_time_on_contact`` snapshot) for done envs.
  """
  state = env.__dict__.get("_el_4090_step_state")
  if state is None:
    return  # no reward computed yet: nothing to clear
  from mjlab.envs.mdp.events import resolve_env_ids

  ids = resolve_env_ids(env, env_ids)
  state.last_dof_vel[ids] = 0.0
  state._prev_dof_vel[ids] = 0.0
  state.feet_air_time[ids] = 0.0
  state.feet_contact_time[ids] = 0.0
  state.feet_air_time_on_contact[ids] = 0.0
  state.last_contacts[ids] = False
  state.feet_first_contact[ids] = False


##
# Tracking.
##


def tracking_lin_vel(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  lin_vel_error = torch.sum(
    torch.square(command[:, :2] - asset.data.root_link_lin_vel_b[:, :2]), dim=1
  )
  return torch.exp(-lin_vel_error / TRACKING_SIGMA)


def tracking_ang_vel(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  ang_vel_error = torch.square(
    command[:, 2] - asset.data.root_link_ang_vel_b[:, 2]
  )
  return torch.exp(-ang_vel_error / TRACKING_SIGMA)


##
# Base penalties (legged_robot_rew_mixin).
##


def lin_vel_z(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.square(asset.data.root_link_lin_vel_b[:, 2])


def ang_vel_xy(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)


def orientation(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def torques(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.actuator_force), dim=1)


def dof_vel(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.joint_vel), dim=1)


def dof_acc(
  env: ManagerBasedRlEnv,
  feet_sensor_name: str = "feet_ground_contact",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  state = _get_state(env, feet_sensor_name)
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(
    torch.square((state._prev_dof_vel - asset.data.joint_vel) / env.step_dt),
    dim=1,
  )


def base_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  base_height_val = torch.mean(
    asset.data.root_link_pos_w[:, 2].unsqueeze(1) - measured_heights(env),
    dim=1,
  )
  return torch.square(base_height_val - BASE_HEIGHT_TARGET)


def action_rate_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  return torch.sum(
    torch.square(env.action_manager.action - env.action_manager.prev_action),
    dim=1,
  )


def dof_pos_limits(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  limits = asset.data.soft_joint_pos_limits
  assert limits is not None
  joint_pos = asset.data.joint_pos
  out_of_limits = -(joint_pos - limits[..., 0]).clip(max=0.0)
  out_of_limits += (joint_pos - limits[..., 1]).clip(min=0.0)
  return torch.sum(out_of_limits, dim=1)


def shank_perp2ground(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  names = list(asset.joint_names)
  hfe = torch.tensor([names.index(n) for n in HFE_NAMES], device=env.device)
  kfe = torch.tensor([names.index(n) for n in KFE_NAMES], device=env.device)
  return torch.square(
    asset.data.joint_pos[:, hfe] - asset.data.joint_pos[:, kfe]
  ).sum(dim=1)


##
# Contact-based terms.
##


def _foot_velocities(env: ManagerBasedRlEnv) -> torch.Tensor:
  """(B, 6, 3) world-frame linear velocity of foot bodies, alphabetical."""
  robot: Entity = env.scene["robot"]
  names = list(robot.body_names)
  ids = torch.tensor([names.index(n) for n in FEET_NAMES], device=env.device)
  return robot.data.body_link_lin_vel_w[:, ids]


def feet_slip(
  env: ManagerBasedRlEnv, feet_sensor_name: str = "feet_ground_contact"
) -> torch.Tensor:
  state = _get_state(env, feet_sensor_name)
  contact = state.feet_force()[..., 2] > 1.0
  contact_filt = torch.logical_or(contact, state.last_contacts)
  foot_velocities = torch.square(
    torch.norm(_foot_velocities(env)[:, :, 0:2], dim=2)
  )
  return torch.sum(contact_filt * foot_velocities, dim=1)


def feet_air_time(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  feet_sensor_name: str = "feet_ground_contact",
) -> torch.Tensor:
  """EL4090 override: (air_on_contact - target) * first_contact, cmd-gated."""
  state = _get_state(env, feet_sensor_name)
  command = env.command_manager.get_command(command_name)
  assert command is not None
  rew = torch.sum(
    (state.feet_air_time_on_contact - FEET_AIR_TIME_TARGET)
    * state.feet_first_contact,
    dim=1,
  )
  rew *= torch.norm(command[:, :2], dim=1) > 0.1  # no reward for zero command
  return rew


def feet_stumble(
  env: ManagerBasedRlEnv, feet_sensor_name: str = "feet_ground_contact"
) -> torch.Tensor:
  state = _get_state(env, feet_sensor_name)
  force = state.feet_force()
  return torch.any(
    torch.norm(force[..., :2], dim=2) > 5 * torch.abs(force[..., 2]), dim=1
  ).float()


def feet_contact_forces(
  env: ManagerBasedRlEnv, feet_sensor_name: str = "feet_ground_contact"
) -> torch.Tensor:
  state = _get_state(env, feet_sensor_name)
  return torch.sum(
    (torch.norm(state.feet_force(), dim=-1) - MAX_CONTACT_FORCE).clip(min=0.0),
    dim=1,
  )


def collision(
  env: ManagerBasedRlEnv, penalised_sensor_name: str = "penalised_contact"
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[penalised_sensor_name]
  force = sensor.data.force
  assert force is not None
  n_prim = len(sensor.primary_names)
  force = force.view(env.num_envs, n_prim, 3)
  return torch.sum(
    (torch.norm(force, dim=-1) > 0.1).float(), dim=1
  )


def stand_still2(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  feet_sensor_name: str = "feet_ground_contact",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Old stand_still2 with the old active weights (only the joint-position
  penalty, weight 2.0; contact-count/stability/time weights are 0 upstream)."""
  state = _get_state(env, feet_sensor_name)
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  small_command_mask = (torch.norm(command[:, :2], dim=1) < SPEED_MIN) & (
    torch.abs(command[:, 2]) < SPEED_MIN
  )
  joint_position_penalty = torch.sum(
    torch.abs(
      asset.data.joint_pos - asset.data.default_joint_pos
    ),
    dim=1,
  )
  return joint_position_penalty * 2.0 * small_command_mask.float()


##
# Gait (ElSpider._reward_gait_2_step + its sync/async helpers).
##


def _sync_reward_func(
  air_time: torch.Tensor, contact_time: torch.Tensor, foot_0: int, foot_1: int,
  max_err: float = 2.0,
) -> torch.Tensor:
  se_air = torch.clip(
    torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=max_err**2
  )
  se_contact = torch.clip(
    torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]),
    max=max_err**2,
  )
  return se_air + se_contact


def _async_reward_func(
  air_time: torch.Tensor, contact_time: torch.Tensor, foot_0: int, foot_1: int,
  max_err: float = 2.0,
) -> torch.Tensor:
  se_act_0 = torch.clip(
    torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=max_err**2
  )
  se_act_1 = torch.clip(
    torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=max_err**2
  )
  return se_act_0 + se_act_1


def gait_2_step(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  feet_sensor_name: str = "feet_ground_contact",
) -> torch.Tensor:
  state = _get_state(env, feet_sensor_name)
  command = env.command_manager.get_command(command_name)
  assert command is not None
  air_time = state.feet_air_time
  contact_time = state.feet_contact_time

  # Group 1 (LB, LF, RM) internal sync.
  sync_lb_lf = _sync_reward_func(air_time, contact_time, 0, 1)
  sync_lb_rm = _sync_reward_func(air_time, contact_time, 0, 5)
  sync_lf_rm = _sync_reward_func(air_time, contact_time, 1, 5)
  sync_group1 = (sync_lb_lf + sync_lb_rm + sync_lf_rm) / 3

  # Group 2 (LM, RB, RF) internal sync.
  sync_lm_rb = _sync_reward_func(air_time, contact_time, 2, 3)
  sync_lm_rf = _sync_reward_func(air_time, contact_time, 2, 4)
  sync_rb_rf = _sync_reward_func(air_time, contact_time, 3, 4)
  sync_group2 = (sync_lm_rb + sync_lm_rf + sync_rb_rf) / 3

  # Cross-group async (all 9 pairs).
  async_pairs = (
    (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (5, 2), (5, 3), (5, 4)
  )
  async_reward = sum(
    _async_reward_func(air_time, contact_time, a, b) for a, b in async_pairs
  ) / 9

  sync_reward = (sync_group1 + sync_group2) / 2

  # Sync-all (standing) reward across all 15 pairs.
  sync_all_pairs = [
    sync_lb_lf, sync_lb_rm, sync_lf_rm, sync_lm_rb, sync_lm_rf, sync_rb_rf,
    _sync_reward_func(air_time, contact_time, 0, 2),
    _sync_reward_func(air_time, contact_time, 0, 3),
    _sync_reward_func(air_time, contact_time, 0, 4),
    _sync_reward_func(air_time, contact_time, 1, 2),
    _sync_reward_func(air_time, contact_time, 1, 3),
    _sync_reward_func(air_time, contact_time, 1, 4),
    _sync_reward_func(air_time, contact_time, 5, 2),
    _sync_reward_func(air_time, contact_time, 5, 3),
    _sync_reward_func(air_time, contact_time, 5, 4),
  ]
  sync_all_reward = sum(sync_all_pairs) / len(sync_all_pairs)

  command_magnitude = torch.logical_or(
    torch.norm(command[:, :2], dim=1) > SPEED_MIN,
    torch.abs(command[:, 2]) > SPEED_MIN,
  )
  return torch.where(
    command_magnitude,
    sync_reward + async_reward,
    sync_all_reward,
  )
