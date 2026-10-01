"""EL4090 head-mounted LiDAR locomotion task on mjlab.

Port of ``Lidar_legged_gym/legged_gym/legged_gym/envs/el_4090/pd_gru_lidar_head``
(``el4090_lidar_head``): pillar-field terrain, 66-dim proprioception, 18-DoF
PD-torque actions, Airy LiDAR perception feeding the LidarPD actor/critic
(obs groups ``policy`` 2754-dim + privileged ``height_grid`` 187-dim).
"""

from .config import *  # noqa: F401,F403  (registers the task)
from .config import register_el_4090_lidar_head  # noqa: F401
