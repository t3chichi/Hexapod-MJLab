"""EL4090 LiDAR task configuration (factory, mirroring velocity_env_cfg.py).

Robot-specific values come from the M1 artifacts (mjcf + tests/
m1_joint_table.json, sourced from the old ``el_4090_spider_config.py``); MDP
semantics from the old ``pd_gru_lidar_head`` env/config (see MIGRATION.md M3).
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco

from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr

from . import mdp
from .mdp.commands import El4090VelocityCommandCfg
from .mdp.terrain import El4090PillarFieldCfg, make_el_4090_terrain_config

# ── Robot (M1 + radar variant) ────────────────────────────────────────────
# The radar variant MJCF (strict superset of the base: fixed radar links +
# lidar_sensor site). The base el_4090.xml remains the M1 acceptance target
# in tests/robot/.
_EL4090_XML = (
  Path(__file__).parents[2] / "assets" / "robots" / "el_4090" / "mjcf" / "el_4090_lidar.xml"
)

JOINT_NAMES = (
  "RF_HAA", "RF_HFE", "RF_KFE",
  "RM_HAA", "RM_HFE", "RM_KFE",
  "RB_HAA", "RB_HFE", "RB_KFE",
  "LF_HAA", "LF_HFE", "LF_KFE",
  "LM_HAA", "LM_HFE", "LM_KFE",
  "LB_HAA", "LB_HFE", "LB_KFE",
)

# Old control constants (m1_joint_table.json old_config).
ACTION_SCALE = 0.25
STIFFNESS = 130.0
DAMPING = 2.0
TORQUE_LIMIT = 80.0
DECIMATION = 4
SIM_TIMESTEP = 0.005
INIT_POS = (0.0, 0.0, 0.5)  # old init_state.pos (drops 1.4 cm to stand)
SPAWN_OFFSET_RANGE = 0.2
ROT_RANDOMIZATION_RANGE = (-3.14, 3.14)
INIT_VEL_RANGE = 0.5
EPISODE_LENGTH_S = 20.0

# Old domain_rand ranges.（旧摩擦范围 (0.3,1.25) 已由地形摩擦校准
# U[0.65,1.125] 替代——见 events.terrain_friction 注释）
ADDED_MASS_RANGE = (-10.0, 10.0)
PUSH_INTERVAL_S = 3
MAX_PUSH_VEL_XY = 1.0


def _get_el_4090_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(_EL4090_XML))


def get_el_4090_robot_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=_get_el_4090_spec,
    articulation=EntityArticulationInfoCfg(
      actuators=(XmlActuatorCfg(target_names_expr=JOINT_NAMES),),
    ),
    init_state=EntityCfg.InitialStateCfg(
      pos=INIT_POS,
      joint_pos={name: 0.0 for name in JOINT_NAMES},
    ),
  )


def make_el_4090_lidar_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the EL4090 head-LiDAR pillar-field task configuration."""

  ##
  # Sensors (contact; the LiDAR is env-side raycasting vs the terrain warp
  # mesh, see mdp/lidar.py — not a MuJoCo scene sensor).
  ##
  feet_ground_contact = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="body", pattern=mdp.FEET_NAMES, entity="robot"
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )
  penalised_contact = ContactSensorCfg(
    name="penalised_contact",
    # Old penalize_contacts_on=["BASE","SHANK","THIGH"] (substring match).
    primary=ContactMatch(
      mode="body",
      pattern=("BASE", ".*_SHANK", ".*_THIGH"),
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )

  ##
  # Observations: policy = proprio 66 + prox 768 + distal 1920 (= 2754);
  # privileged height_grid = 187.
  ##
  policy_terms = {
    "base_lin_vel": ObservationTermCfg(
      func=mdp.base_lin_vel,
      scale=mdp.OBS_SCALE_LIN_VEL,
      noise=Unoise(n_min=-mdp.NOISE_LIN_VEL, n_max=mdp.NOISE_LIN_VEL),
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.base_ang_vel,
      scale=mdp.OBS_SCALE_ANG_VEL,
      noise=Unoise(n_min=-mdp.NOISE_ANG_VEL, n_max=mdp.NOISE_ANG_VEL),
    ),
    "projected_gravity": ObservationTermCfg(
      func=envs_mdp.projected_gravity,
      noise=Unoise(n_min=-mdp.NOISE_GRAVITY, n_max=mdp.NOISE_GRAVITY),
    ),
    "command": ObservationTermCfg(
      func=envs_mdp.generated_commands,
      params={"command_name": "twist"},
      scale=mdp.COMMANDS_SCALE,
    ),
    "joint_pos": ObservationTermCfg(
      func=envs_mdp.joint_pos_rel,
      params={"asset_cfg": mdp.SceneEntityCfg("robot", joint_names=(".*",))},
      scale=mdp.OBS_SCALE_DOF_POS,
      noise=Unoise(n_min=-mdp.NOISE_DOF_POS, n_max=mdp.NOISE_DOF_POS),
    ),
    "joint_vel": ObservationTermCfg(
      func=envs_mdp.joint_vel_rel,
      params={"asset_cfg": mdp.SceneEntityCfg("robot", joint_names=(".*",))},
      scale=mdp.OBS_SCALE_DOF_VEL,
      noise=Unoise(n_min=-mdp.NOISE_DOF_VEL, n_max=mdp.NOISE_DOF_VEL),
    ),
    "actions": ObservationTermCfg(func=envs_mdp.last_action),
    "lidar_prox": ObservationTermCfg(func=mdp.lidar_prox_obs),
    "lidar_distal": ObservationTermCfg(func=mdp.lidar_distal_obs),
  }
  observations = {
    "policy": ObservationGroupCfg(
      terms=policy_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "height_grid": ObservationGroupCfg(
      terms={"height_grid": ObservationTermCfg(func=mdp.height_grid_obs)},
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  ##
  # Actions: 18 position targets -> PD torques (kp 130 / kd 2), clip ±80 Nm.
  ##
  actions = {
    "joint_pd_torque": mdp.JointPDTorqueActionCfg(
      entity_name="robot",
      actuator_names=JOINT_NAMES,
      scale=ACTION_SCALE,
      stiffness=STIFFNESS,
      damping=DAMPING,
      effort_limit=TORQUE_LIMIT,
      use_default_offset=True,
    )
  }

  ##
  # Commands (old El4090LidarHeadCfg.commands).
  ##
  commands = {
    "twist": El4090VelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(20.0, 20.0),
      heading_command=True,
      heading_control_stiffness=0.5,
      rel_heading_envs=1.0,
      rel_standing_envs=0.0,
      rel_world_envs=0.0,
      rel_forward_envs=0.0,
      ranges=El4090VelocityCommandCfg.Ranges(
        lin_vel_x=(-1.0, 1.0),
        lin_vel_y=(-1.0, 1.0),
        ang_vel_z=(-1.2, 1.2),
        heading=(-3.14, 3.14),
      ),
      cmd_deadzone=0.2,
    )
  }

  ##
  # Events: old reset randomization + friction/mass/push DR.
  ##
  events = {
    "reset_base": EventTermCfg(
      func=envs_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        # Old _reset_root_states: yaw ±3.14, xy offset ±0.2, z none.
        "pose_range": {
          "x": (-SPAWN_OFFSET_RANGE, SPAWN_OFFSET_RANGE),
          "y": (-SPAWN_OFFSET_RANGE, SPAWN_OFFSET_RANGE),
          "yaw": ROT_RANDOMIZATION_RANGE,
        },
        # Old root velocity init ±0.5 on all 6 DOF.
        "velocity_range": {
          "x": (-INIT_VEL_RANGE, INIT_VEL_RANGE),
          "y": (-INIT_VEL_RANGE, INIT_VEL_RANGE),
          "z": (-INIT_VEL_RANGE, INIT_VEL_RANGE),
          "roll": (-INIT_VEL_RANGE, INIT_VEL_RANGE),
          "pitch": (-INIT_VEL_RANGE, INIT_VEL_RANGE),
          "yaw": (-INIT_VEL_RANGE, INIT_VEL_RANGE),
        },
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=envs_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": mdp.SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    # Old push: overwrite root lin vel x/y with U(-1, 1) every 3 s
    # (official push_by_setting_velocity adds a delta — old semantics set).
    "push_robot": EventTermCfg(
      func=mdp.push_by_overwriting_velocity,
      mode="interval",
      interval_range_s=(PUSH_INTERVAL_S, PUSH_INTERVAL_S),
      params={
        "velocity_range": {
          "x": (-MAX_PUSH_VEL_XY, MAX_PUSH_VEL_XY),
          "y": (-MAX_PUSH_VEL_XY, MAX_PUSH_VEL_XY),
        }
      },
    ),
    # 摩擦语义校准（M5a 探针定案，tests/data/m5/friction_semantics.md）：
    # 旧栈 PhysX 对 (robot μ~U[0.3,1.25], terrain 1.0) 取平均 → 有效摩擦
    # U[0.65, 1.125]；MuJoCo 取 max-mix。地形 geom 已设 priority=1（见
    # mdp/terrain.py），其摩擦整胜出，故随机化地形滑动摩擦 U[0.65, 1.125]
    # 即复现旧栈有效分布；每 env 一采样、跨地形 geom 共享（≙ 旧栈一次采样
    # 应用于全部几何体的语义）。startup 模式 = 旧栈 _create_envs 时机。
    "terrain_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": mdp.SceneEntityCfg("terrain", geom_names=(".*",)),
        "operation": "abs",
        "ranges": (0.65, 1.125),
        "axes": [0],
        "shared_random": True,
      },
    ),
    "added_mass": EventTermCfg(
      mode="startup",
      func=dr.body_mass,
      params={
        # Old _process_rigid_body_props: props[0].mass += U(-10, 10)
        # (base body only).
        "asset_cfg": mdp.SceneEntityCfg("robot", body_names=("BASE",)),
        "operation": "add",
        "ranges": ADDED_MASS_RANGE,
      },
    ),
    # Old reset_idx tail: zero gait timers + last_dof_vel for done envs.
    "reset_gait_state": EventTermCfg(
      mode="reset",
      func=mdp.reset_gait_state,
    ),
  }

  ##
  # Rewards: stage-0 weights (multi-stage switch in curriculum below).
  ##
  rewards = {
    "tracking_lin_vel": RewardTermCfg(
      func=mdp.tracking_lin_vel, weight=1.0, params={"command_name": "twist"}
    ),
    "tracking_ang_vel": RewardTermCfg(
      func=mdp.tracking_ang_vel, weight=0.5, params={"command_name": "twist"}
    ),
    "lin_vel_z": RewardTermCfg(func=mdp.lin_vel_z, weight=-8.0),
    "ang_vel_xy": RewardTermCfg(func=mdp.ang_vel_xy, weight=-0.5),
    "orientation": RewardTermCfg(func=mdp.orientation, weight=-8.0),
    "torques": RewardTermCfg(func=mdp.torques, weight=-0.0001),
    "dof_vel": RewardTermCfg(func=mdp.dof_vel, weight=-0.0001),
    "dof_acc": RewardTermCfg(func=mdp.dof_acc, weight=-5e-8),
    "base_height": RewardTermCfg(
      func=mdp.base_height, weight=mdp.STAGE_WEIGHTS["base_height"][0]
    ),
    "feet_slip": RewardTermCfg(
      func=mdp.feet_slip, weight=mdp.STAGE_WEIGHTS["feet_slip"][0]
    ),
    "feet_air_time": RewardTermCfg(
      func=mdp.feet_air_time,
      weight=mdp.STAGE_WEIGHTS["feet_air_time"][0],
      params={"command_name": "twist"},
    ),
    "collision": RewardTermCfg(func=mdp.collision, weight=-1.0),
    "feet_stumble": RewardTermCfg(
      func=mdp.feet_stumble, weight=mdp.STAGE_WEIGHTS["feet_stumble"][0]
    ),
    "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.005),
    "stand_still2": RewardTermCfg(
      func=mdp.stand_still2,
      weight=-0.6,
      params={"command_name": "twist"},
    ),
    "dof_pos_limits": RewardTermCfg(func=mdp.dof_pos_limits, weight=-1.0),
    "feet_contact_forces": RewardTermCfg(
      func=mdp.feet_contact_forces,
      weight=mdp.STAGE_WEIGHTS["feet_contact_forces"][0],
    ),
    "shank_perp2ground": RewardTermCfg(func=mdp.shank_perp2ground, weight=-0.05),
    "gait_2_step": RewardTermCfg(
      func=mdp.gait_2_step,
      weight=mdp.STAGE_WEIGHTS["gait_2_step"][0],
      params={"command_name": "twist"},
    ),
  }

  ##
  # Terminations.
  ##
  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(func=mdp.fell_over),
  }

  ##
  # Curriculum.
  ##
  curriculum = {
    "reward_stage": CurriculumTermCfg(func=mdp.reward_stage),
    "command_vel": CurriculumTermCfg(
      func=mdp.commands_vel_el_4090, params={"command_name": "twist"}
    ),
  }

  ##
  # Assemble.
  ##
  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
          size=(8.0, 8.0),
          num_rows=2,
          num_cols=2,
          border_width=0.0,  # M2 border lives in the heightfield/LiDAR mesh
          curriculum=False,  # old terrain curriculum = False
          sub_terrains={
            "pillar_field": El4090PillarFieldCfg(
              proportion=1.0,
              size=(8.0, 8.0),
              builder_cfg=make_el_4090_terrain_config(),
            )
          },
        ),
        max_init_terrain_level=None,
      ),
      entities={"robot": get_el_4090_robot_cfg()},
      sensors=(feet_ground_contact, penalised_contact),
      num_envs=4096,  # target hardware RTX 5090; override via CLI on 6 GB dev
      env_spacing=8.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    sim=SimulationCfg(
      mujoco=MujocoCfg(timestep=SIM_TIMESTEP, gravity=(0.0, 0.0, -9.81)),
    ),
    decimation=DECIMATION,
    episode_length_s=EPISODE_LENGTH_S,
  )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}

  return cfg
