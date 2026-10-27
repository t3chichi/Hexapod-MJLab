"""Lifted T1-witness reset state shared by the formal locomotion tasks."""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np

from .robot import FOOT_BODIES, FOOT_CLEARANCE_M, ROBOT_XML
from .witness import SETTLED_QPOS, frozen_witness


def lowest_collision_point_z(model, data) -> float:
  """Lowest world-z over all collision geoms of the given bodies' subtree.

  Uses compiled mesh vertices (the real collision representation), not foot
  origins. ``model``/``data`` are classic mujoco objects with the witness
  state set.
  """
  foot_body_ids = []
  for name in FOOT_BODIES:
    bid = _body_id(model, name)
    if bid >= 0:
      foot_body_ids.append(bid)
  if not foot_body_ids:
    raise RuntimeError("no foot bodies found in model")
  z_min = np.inf
  for gid in range(model.ngeom):
    if model.geom_bodyid[gid] not in foot_body_ids:
      continue
    if model.geom_contype[gid] == 0 and model.geom_conaffinity[gid] == 0:
      continue  # visual-only
    gtype = model.geom_type[gid]
    xpos = data.geom_xpos[gid]
    xmat = data.geom_xmat[gid].reshape(3, 3)
    if gtype == 7:  # mjGEOM_MESH
      m = model.geom_dataid[gid]
      adr, num = model.mesh_vertadr[m], model.mesh_vertnum[m]
      # mesh vertices live in the geom frame (geom_xpos/geom_xmat already
      # include the mesh placement offset geom_pos)
      local = np.asarray(model.mesh_vert[adr : adr + num], dtype=np.float64)
      world = local @ xmat.T + xpos
      z_min = min(z_min, float(world[:, 2].min()))
    else:
      # spheres/capsules/etc: conservative - use geom rbound
      z_min = min(z_min, float(xpos[2]) - float(model.geom_rbound[gid]))
  return z_min


def _body_id(model, name: str) -> int:
  import mujoco

  return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))


def compute_lifted_root(
  xml_path: Path,
  qpos: tuple | list,
  clearance_m: float = FOOT_CLEARANCE_M,
):
  """Compute the lifted root state for the drop-in reset.

  Returns ``(lifted_qpos, min_foot_z_before, min_foot_z_after)``. The root is
  raised so the lowest foot collision point sits exactly ``clearance_m``
  above the world z=0 plane of the flat terrain.
  """
  import mujoco

  model = mujoco.MjModel.from_xml_path(str(xml_path))
  data = mujoco.MjData(model)
  data.qpos[:] = qpos
  mujoco.mj_forward(model, data)
  z_min = lowest_collision_point_z(model, data)
  lift = clearance_m - z_min
  lifted = list(qpos)
  lifted[2] = qpos[2] + lift
  data.qpos[:] = lifted
  mujoco.mj_forward(model, data)
  z_after = lowest_collision_point_z(model, data)
  return lifted, float(z_min), float(z_after)


@functools.lru_cache(maxsize=1)
def lifted_witness_state() -> tuple[tuple, tuple, float, float]:
  """Lifted reset state from the frozen T1 witness, computed once.

  Returns ``(root_pos_xyz, root_quat_wxyz, min_foot_z_before, min_foot_z_after)``.
  The lift is computed from compiled collision meshes via the read-only T1
  witness geometry utility (never from foot-origin kinematics), raising if
  the achieved clearance is not 2 mm within 1e-6 m.
  """
  lifted, z_before, z_after = compute_lifted_root(
    ROBOT_XML, SETTLED_QPOS, clearance_m=FOOT_CLEARANCE_M
  )
  if abs(z_after - FOOT_CLEARANCE_M) > 1e-6:
    raise RuntimeError(
      f"witness lift failed: min foot z after lift = {z_after} "
      f"(expected {FOOT_CLEARANCE_M})"
    )
  return tuple(lifted[0:3]), tuple(lifted[3:7]), float(z_before), float(z_after)


def settled_reset_joint_pos() -> tuple:
  """Reset joint angles = the settled T1 witness qpos (NOT q_ref)."""
  return tuple(frozen_witness().joint_pos)


def reset_env_preserving_unselected_physics(env, env_ids):
  """Run the real env reset while preserving other rows' persistent physics.

  mjlab's selected reset still calls a batch-wide ``sim.forward()`` and
  ``sim.sense()``. MuJoCo may rewrite persistent solver state such as
  ``qacc_warmstart`` during that forward, so preserve the unselected rows of
  every non-empty checkpoint physics field around the ordinary reset. The
  selected rows still receive the real reset, forward, sensing and manager
  updates. We deliberately do not forward again after restoring warmstart:
  that would overwrite the value we are restoring.
  """
  import torch

  from .checkpoint import EXTRA_PERSISTENT_PHYSICS_FIELDS
  from .transition import PERSISTENT_PHYSICS_FIELDS

  num_envs = int(env.num_envs)
  selected_rows = env_ids.detach().to(dtype=torch.long)
  selected_mask = torch.ones(
    num_envs, dtype=torch.bool, device=selected_rows.device
  )
  selected_mask[selected_rows] = False
  unselected_rows = torch.nonzero(selected_mask, as_tuple=False).flatten()
  if unselected_rows.numel() == 0:
    # Whole-batch resets need no snapshots or post-reset copies.
    return env.reset(env_ids=env_ids)

  import warp as wp

  data = env.sim.wp_data
  required_fields = set(PERSISTENT_PHYSICS_FIELDS)
  field_names = (*PERSISTENT_PHYSICS_FIELDS, *EXTRA_PERSISTENT_PHYSICS_FIELDS)
  snapshots = []
  for name in field_names:
    array = getattr(data, name, None)
    if array is None:
      if name in required_fields:
        raise RuntimeError(
          f"backend persistent physics field absent during partial reset: {name}"
        )
      continue
    if not array.size:
      continue
    tensor = wp.to_torch(array)
    if tensor.ndim == 0 or tensor.shape[0] != num_envs:
      raise RuntimeError(
        f"persistent physics field {name} has no env-row axis: "
        f"shape={tuple(tensor.shape)}, num_envs={num_envs}"
      )
    field_rows = unselected_rows.to(device=tensor.device)
    snapshots.append(
      (tensor, field_rows, tensor.index_select(0, field_rows).clone())
    )

  try:
    return env.reset(env_ids=env_ids)
  finally:
    # Copy into the existing Warp-backed tensors to retain their storage
    # identity (including any captured CUDA-graph references).
    with torch.no_grad():
      for tensor, rows, saved in snapshots:
        tensor.index_copy_(0, rows, saved)
