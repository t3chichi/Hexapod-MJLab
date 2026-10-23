"""Production contact-table reader used by the transition boundary."""

from __future__ import annotations

import mujoco_warp
import numpy as np
import torch
import warp as wp

from hexapod_mjlab.tasks.el_4090_lidar.t0 import witness as t1_witness

CONTACT_DIST_MAX_M = 0.0
CONTACT_FORCE_MIN_N = 0.1
EXPECTED_SUBSTEPS_PER_CONTROL_STEP = 4
FORCE_API = "mujoco_warp.contact_force(world_frame=True) -> [Fx,Fy,Fz,Tx,Ty,Tz]"


class ContactTableReader:
  """Read raw contacts and derive per-substep contact counts for one env.

  This reader has no hooks, control-step bracketing, or retained history. The
  caller chooses when to read the actual backend state and how to aggregate
  the returned tables.
  """

  def __init__(self, env) -> None:
    self.env = env
    self.device = env.device
    import mujoco

    mm = env.sim.mj_model
    self.body_names = [
      mujoco.mj_id2name(mm, mujoco.mjtObj.mjOBJ_BODY, int(b)) or ""
      for b in range(mm.nbody)
    ]
    geom_body = torch.tensor(mm.geom_bodyid, dtype=torch.long)
    geom_contype = torch.tensor(mm.geom_contype, dtype=torch.long)
    geom_conaffinity = torch.tensor(mm.geom_conaffinity, dtype=torch.long)
    collidable = (geom_contype != 0) | (geom_conaffinity != 0)

    is_robot_body = torch.tensor(
      [name.startswith("robot/") for name in self.body_names], dtype=torch.bool
    )
    foot_body_names = set(t1_witness.FOOT_BODIES)
    robot_foot_shortnames = {
      name.split("/")[-1] for name in self.body_names if name.startswith("robot/")
    }
    missing = foot_body_names - robot_foot_shortnames
    if missing:
      raise RuntimeError(
        f"foot body names not found in the actual asset: {sorted(missing)}"
      )
    is_foot_body = torch.tensor(
      [name.split("/")[-1] in foot_body_names for name in self.body_names],
      dtype=torch.bool,
    )
    robot_collidable_geoms = collidable & is_robot_body[geom_body]
    owns_collidable = torch.zeros(int(mm.nbody), dtype=torch.long)
    owns_collidable.index_add_(0, geom_body, robot_collidable_geoms.long())
    self.owns_collidable_geom = owns_collidable.bool()
    self.robot_body_ids = torch.nonzero(is_robot_body).flatten()
    self.foot_body_ids = torch.nonzero(is_foot_body).flatten()
    self.nonfoot_robot_body_ids = torch.nonzero(
      is_robot_body & self.owns_collidable_geom & ~is_foot_body
    ).flatten()

    self._geom_body = geom_body.to(env.device)
    self._is_robot_body = is_robot_body.to(env.device)
    self._is_foot_body = is_foot_body.to(env.device)
    self._wp_model = env.sim.wp_model
    self._wp_data = env.sim.wp_data
    self._wp_device = self._wp_data.contact.geom.device

  def body_report(self) -> dict:
    """Actual-asset body sets used by the contact classifier."""
    return {
      "robot_bodies": [self.body_names[int(i)] for i in self.robot_body_ids],
      "foot_bodies": [self.body_names[int(i)] for i in self.foot_body_ids],
      "nonfoot_collidable_robot_bodies": [
        self.body_names[int(i)] for i in self.nonfoot_robot_body_ids
      ],
      "force_api": FORCE_API,
    }

  @staticmethod
  def _empty_table() -> dict:
    return {
      "n_contacts": 0,
      "geom": np.zeros((0, 2), dtype=np.int64),
      "dist": np.zeros((0,), dtype=np.float32),
      "pos": np.zeros((0, 3), dtype=np.float32),
      "frame": np.zeros((0, 3, 3), dtype=np.float32),
      "dim": np.zeros((0,), dtype=np.int64),
      "force6_world": np.zeros((0, 6), dtype=np.float32),
      "worldid": np.zeros((0,), dtype=np.int64),
      "body1": np.zeros((0,), dtype=np.int64),
      "body2": np.zeros((0,), dtype=np.int64),
      "body1_name": [],
      "body2_name": [],
    }

  def read_substep_table(self) -> dict:
    """Snapshot every raw contact at the current solver substep."""
    data = self._wp_data
    wp.synchronize_device(self._wp_device)
    n = int(wp.to_torch(data.nacon).item())
    if n == 0:
      return self._empty_table()

    geom_all = wp.to_torch(data.contact.geom).reshape(-1, 2).long()
    dist_all = wp.to_torch(data.contact.dist)
    pos_all = wp.to_torch(data.contact.pos).reshape(-1, 3)
    frame_all = wp.to_torch(data.contact.frame).reshape(-1, 9)
    dim_all = wp.to_torch(data.contact.dim).long()
    worldid_all = wp.to_torch(data.contact.worldid).long()
    short = [
      name
      for name, array in (
        ("geom", geom_all),
        ("dist", dist_all),
        ("pos", pos_all),
        ("frame", frame_all),
        ("dim", dim_all),
        ("worldid", worldid_all),
      )
      if int(array.shape[0]) < n
    ]
    if short:
      raise RuntimeError(
        f"contact backing arrays shorter than nacon={n}: {short} "
        f"(worldid capacity={int(worldid_all.shape[0])})"
      )
    geom = geom_all[:n].detach().clone()
    dist = dist_all[:n].detach().clone()
    pos = pos_all[:n].detach().clone()
    frame = frame_all[:n].detach().clone()
    dim = dim_all[:n].detach().clone()
    worldid = worldid_all[:n].detach().clone()
    if not bool(((worldid >= 0) & (worldid < self.env.num_envs)).all()):
      raise RuntimeError(
        f"contact worldid out of range [0, {self.env.num_envs}): "
        f"{worldid.tolist()}"
      )

    contact_ids = wp.array(
      np.arange(n, dtype=np.int32), dtype=wp.int32, device=self._wp_device
    )
    force_wp = wp.zeros(n, dtype=wp.spatial_vector, device=self._wp_device)
    mujoco_warp.contact_force(
      self._wp_model, data, contact_ids, True, force_wp
    )
    wp.synchronize_device(self._wp_device)
    force6 = wp.to_torch(force_wp).reshape(n, 6).detach().cpu().numpy().copy()

    geom1 = geom[:, 0].to(self.device)
    geom2 = geom[:, 1].to(self.device)
    body1 = torch.where(
      geom1 >= 0,
      self._geom_body[torch.clamp(geom1, min=0)],
      -1,
    )
    body2 = torch.where(
      geom2 >= 0,
      self._geom_body[torch.clamp(geom2, min=0)],
      -1,
    )
    return {
      "n_contacts": n,
      "geom": geom.detach().cpu().numpy().copy(),
      "dist": dist.detach().cpu().numpy().copy(),
      "pos": pos.detach().cpu().numpy().copy(),
      "frame": frame.reshape(n, 3, 3).detach().cpu().numpy().copy(),
      "dim": dim.detach().cpu().numpy().copy(),
      "force6_world": force6,
      "worldid": worldid.detach().cpu().numpy().copy(),
      "body1": body1.detach().cpu().numpy().copy(),
      "body2": body2.detach().cpu().numpy().copy(),
      "body1_name": [
        self.body_names[int(i)] if i >= 0 else "" for i in body1.tolist()
      ],
      "body2_name": [
        self.body_names[int(i)] if i >= 0 else "" for i in body2.tolist()
      ],
    }

  def substep_counts(self, table: dict) -> dict:
    """Count distinct qualifying nonfoot/environment contacts and self pairs."""
    n_envs = self.env.num_envs
    zeros = torch.zeros(n_envs, device=self.device)
    result = {"nonfoot_env": zeros.clone(), "self_pairs": zeros.clone()}
    if table["n_contacts"] == 0:
      return result
    dist = torch.as_tensor(table["dist"], device=self.device)
    worldid = torch.as_tensor(table["worldid"], device=self.device)
    body1 = torch.as_tensor(table["body1"], device=self.device)
    body2 = torch.as_tensor(table["body2"], device=self.device)
    force6 = torch.as_tensor(table["force6_world"], dtype=torch.float32)
    magnitude = force6[:, :3].norm(dim=1).to(self.device)
    qualifying = (dist <= CONTACT_DIST_MAX_M) & (magnitude > CONTACT_FORCE_MIN_N)
    robot1 = self._is_robot_body[body1]
    robot2 = self._is_robot_body[body2]
    foot1 = self._is_foot_body[body1]
    foot2 = self._is_foot_body[body2]
    nonfoot_env = qualifying & (
      (robot1 & ~foot1 & ~robot2) | (robot2 & ~foot2 & ~robot1)
    )
    self_contact = qualifying & robot1 & robot2 & (body1 != body2)

    if bool(nonfoot_env.any()):
      body = torch.where(robot1 & ~foot1, body1, body2)[nonfoot_env]
      env_ids = worldid[nonfoot_env]
      bodies: list[set] = [set() for _ in range(n_envs)]
      for env_id, body_id in zip(env_ids.tolist(), body.tolist()):
        bodies[env_id].add(body_id)
      result["nonfoot_env"] = torch.tensor(
        [len(items) for items in bodies], dtype=torch.float32, device=self.device
      )
    if bool(self_contact.any()):
      lo = torch.minimum(body1, body2)[self_contact]
      hi = torch.maximum(body1, body2)[self_contact]
      env_ids = worldid[self_contact]
      pairs: list[set] = [set() for _ in range(n_envs)]
      for env_id, low, high in zip(env_ids.tolist(), lo.tolist(), hi.tolist()):
        pairs[env_id].add((low, high))
      result["self_pairs"] = torch.tensor(
        [len(items) for items in pairs], dtype=torch.float32, device=self.device
      )
    return result
