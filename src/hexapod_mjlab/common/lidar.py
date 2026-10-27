"""Model-derived EL4090 LiDAR mount, independent of any task package."""

from __future__ import annotations

import functools
from pathlib import Path

EL4090_LIDAR_MJCF = (
  Path(__file__).resolve().parents[1]
  / "assets"
  / "robots"
  / "el_4090"
  / "mjcf"
  / "el_4090_lidar.xml"
)


@functools.lru_cache(maxsize=1)
def resolve_lidar_mount(
  mjcf_path: Path = EL4090_LIDAR_MJCF,
  site_name: str = "lidar_sensor",
  base_body: str = "BASE",
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]] | None:
  """Constant BASE→optical transform read from the model's ``lidar_sensor`` site.

  The optical +z scan pole is aligned to housing +y by the empty URDF/MJCF
  sensor frame. Do not add another 90-degree correction here. Optical +x
  stays housing +x; this azimuth origin is not a hardware calibration.

  The site is rigidly fixed to BASE, so the transform is one constant per
  model: evaluated once at the ``stand`` keyframe (identity base quat, so
  base frame == world frame up to translation). Returns ((x, y, z),
  (x, y, z, w) quat) in the BASE body frame, or None when the MJCF or the
  site does not exist (base variant → callers fall back to hand-written
  cfg values).
  """
  if not mjcf_path.exists():
    return None
  import mujoco
  import numpy as np

  model = mujoco.MjModel.from_xml_path(str(mjcf_path))
  sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
  bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body)
  if sid < 0 or bid < 0:
    return None
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  r_base_t = np.asarray(data.xmat[bid]).reshape(3, 3).T
  p_rel = r_base_t @ (np.asarray(data.site_xpos[sid]) - np.asarray(data.xpos[bid]))
  r_rel = r_base_t @ np.asarray(data.site_xmat[sid]).reshape(3, 3)
  q_wxyz = np.zeros(4)
  mujoco.mju_mat2Quat(q_wxyz, r_rel.flatten())
  if q_wxyz[0] < 0:
    q_wxyz = -q_wxyz
  pos = (float(p_rel[0]), float(p_rel[1]), float(p_rel[2]))
  quat_xyzw = (float(q_wxyz[1]), float(q_wxyz[2]), float(q_wxyz[3]), float(q_wxyz[0]))
  return pos, quat_xyzw
