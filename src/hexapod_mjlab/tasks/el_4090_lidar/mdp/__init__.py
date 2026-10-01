"""MDP terms for the EL4090 LiDAR task (mirrors mjlab tasks/velocity/mdp)."""

from mjlab.envs.mdp import *  # noqa: F401, F403
from mjlab.envs.mdp import dr as dr  # noqa: F401
from mjlab.managers.scene_entity_config import SceneEntityCfg  # noqa: F401

from . import rewards as rewards  # noqa: F401
from .actions import JointPDTorqueAction, JointPDTorqueActionCfg  # noqa: F401
from .commands import El4090VelocityCommand, El4090VelocityCommandCfg  # noqa: F401
from .curriculums import (  # noqa: F401
  STAGE_WEIGHTS,
  commands_vel_el_4090,
  reward_stage,
)
from .events import push_by_overwriting_velocity  # noqa: F401
from .lidar import lidar_distal_obs, lidar_prox_obs  # noqa: F401
from .observations import (  # noqa: F401
  COMMANDS_SCALE,
  NOISE_ANG_VEL,
  NOISE_DOF_POS,
  NOISE_DOF_VEL,
  NOISE_GRAVITY,
  NOISE_LIN_VEL,
  OBS_SCALE_ANG_VEL,
  OBS_SCALE_DOF_POS,
  OBS_SCALE_DOF_VEL,
  OBS_SCALE_LIN_VEL,
  base_ang_vel,
  base_lin_vel,
  height_grid_obs,
  measured_heights,
)
from .rewards import (  # noqa: F401
  FEET_NAMES,
  action_rate_l2,
  ang_vel_xy,
  base_height,
  collision,
  dof_acc,
  dof_pos_limits,
  dof_vel,
  feet_air_time,
  feet_contact_forces,
  feet_slip,
  feet_stumble,
  gait_2_step,
  lin_vel_z,
  orientation,
  reset_gait_state,
  shank_perp2ground,
  stand_still2,
  torques,
  tracking_ang_vel,
  tracking_lin_vel,
)
from .terminations import fell_over  # noqa: F401
from .terrain import (  # noqa: F401
  EL4090_TERRAIN_SEED,
  El4090PillarFieldCfg,
  El4090TerrainView,
  get_el_4090_terrain_view,
)

