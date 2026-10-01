"""RL runner for the EL4090 LiDAR task (counterpart of velocity/rl/runner.py).

Extends :class:`mjlab.rl.runner.MjlabOnPolicyRunner` only to expose the LiDAR
mount transform on the wrapped env so that
``LidarPPO5x.construct_algorithm`` finds the symmetry sensor binding tensors
(``_sensor_offset_quat`` / ``_sensor_translation``; see the warning note in
``hexapod_mjlab/rl/ppo_lidar.py``). Without this, the mirror function would
silently assume an identity mount.
"""

from __future__ import annotations

import math

import torch

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner

from hexapod_mjlab.tasks.el_4090_lidar.mdp.lidar import (
  LIDAR_MAX_RANGE,
  PROXIMAL_POINTS,
)

# LiDAR mount (old El4090LidarHeadCfg.lidar): head-front mount with pitch tilt.
SENSOR_OFFSET_POS = (0.7, 0.0, -0.05)
PITCH_TILT_DOWN_RAD = 0.1
SENSOR_OFFSET_RPY = (0.0, math.pi / 2.0 + PITCH_TILT_DOWN_RAD, 0.0)

NUM_DOF = 18
PROPRIO_DIM = 66
DISTAL_POINTS = 64
DISTAL_HISTORY = 10


def _euler_xyz_to_quat_xyzw(roll: float, pitch: float, yaw: float) -> torch.Tensor:
  """q = qz(yaw) * qy(pitch) * qx(roll), isaacgym [x, y, z, w] convention
  (same formula as the S2 spike / old env ``quat_from_euler_xyz``)."""
  cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
  cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
  cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
  qx = torch.tensor([sr, 0.0, 0.0, cr])
  qy = torch.tensor([0.0, sp, 0.0, cp])
  qz = torch.tensor([0.0, 0.0, sy, cy])

  def mul(a, b):
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack([
      aw * bx + ax * bw + ay * bz - az * by,
      aw * by - ax * bz + ay * bw + az * bx,
      aw * bz + ax * by - ay * bx + az * bw,
      aw * bw - ax * bx - ay * by - az * bz,
    ], dim=-1)

  return mul(qz, mul(qy, qx)).view(1, 4)


class El4090LidarRunner(MjlabOnPolicyRunner):
  """MjlabOnPolicyRunner + LiDAR mount exposure for LidarPPO5x symmetry."""

  env: RslRlVecEnvWrapper

  def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
    # Expose the mount transform on the VecEnv wrapper before the rsl_rl
    # OnPolicyRunner constructor builds the algorithm (construct_algorithm
    # looks for these attributes on the env it receives).
    if not hasattr(env, "_sensor_offset_quat"):
      device_str = str(device)
      env._sensor_offset_quat = _euler_xyz_to_quat_xyzw(*SENSOR_OFFSET_RPY).to(
        device_str
      )
      env._sensor_translation = torch.tensor(
        SENSOR_OFFSET_POS, dtype=torch.float32, device=device_str
      ).view(1, 3)
    super().__init__(env, train_cfg, log_dir, device)
