"""Training-only height labels and clean LiDAR queries on compiled Heightfield.

No observation, network, loss or physics lifecycle is owned here. Callers supply
the current pre-action identity and the rows due for a physical LiDAR scan.
The first-stage implementation supports exactly one static, horizontal-axis
Heightfield. It uses the actual compiled model and selected live geometry poses.
"""

from copy import copy
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch
import warp as wp

from hexapod_mjlab.rl.confined_locomotion_2.contracts import HeightLabels

from .channels import build_channel_catalog
from .config.terrain_geometry import TERRAIN_GROUP
from .sampling import far_candidate_mask, farthest_point_sampling_masked
from .scanner import _quat_apply_xyzw, _quat_mul_xyzw
from hexapod_mjlab.common.lidar import resolve_lidar_mount


@dataclass(frozen=True)
class HeightGridCfg:
    """Explicit engineering candidate, not a final training grid extent."""

    x_range_m: tuple[float, float]
    y_range_m: tuple[float, float]

    def __post_init__(self):
        for bounds in (self.x_range_m, self.y_range_m):
            if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
                raise ValueError("grid bounds must be finite, strictly increasing pairs")
        if self.x_range_m[0] != -self.x_range_m[1] or self.y_range_m[0] != -self.y_range_m[1]:
            raise ValueError("the root-centred 11x17 grid must have symmetric bounds")

    def xy(self, device="cpu"):
        y, x = torch.meshgrid(
            torch.linspace(*self.y_range_m, 11, device=device),
            torch.linspace(*self.x_range_m, 17, device=device), indexing="ij")
        return torch.stack((x, y), dim=-1).reshape(187, 2)

    def identity(self):
        return {"nx": 17, "ny": 11, "x_range_m": self.x_range_m,
                "y_range_m": self.y_range_m, "order": "iy*17+ix",
                "frame": "current_root_xy_yaw_only", "height": "world_surface_z-root_z_m"}


def grid_world_xy(root_pos, root_quat_wxyz, grid: HeightGridCfg):
    """Yaw-only grid; changing roll/pitch at fixed yaw leaves XY unchanged."""
    if root_pos.ndim != 2 or root_pos.shape[1] != 3 or root_quat_wxyz.shape != (len(root_pos), 4):
        raise ValueError("root poses must be [B,3] and [B,4]")
    if not bool(torch.isfinite(root_pos).all()) or not bool(torch.isfinite(root_quat_wxyz).all()):
        raise ValueError("nonfinite root pose")
    if not bool(torch.allclose(root_quat_wxyz.norm(dim=-1), torch.ones_like(root_pos[:, 0]), atol=1e-6, rtol=0)):
        raise ValueError("root quaternion must be normalized wxyz")
    w, x, y, z = root_quat_wxyz.unbind(-1)
    yaw = torch.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    xy = grid.xy(root_pos.device)
    cx, sy = yaw.cos()[:, None], yaw.sin()[:, None]
    return root_pos[:, None, :2] + torch.stack(
        (cx*xy[:, 0] - sy*xy[:, 1], sy*xy[:, 0] + cx*xy[:, 1]), dim=-1)


@dataclass(frozen=True)
class TerrainScan:
    env_ids: torch.Tensor
    source_ids: torch.Tensor
    distances: torch.Tensor           # [B,R], miss uses the physical max range
    hit: torch.Tensor
    points_world: torch.Tensor        # miss endpoints included for raw audit
    points_base: torch.Tensor         # acquisition-time full BODY coordinates
    near_points: torch.Tensor
    near_valid: torch.Tensor
    far_points: torch.Tensor
    far_valid: torch.Tensor
    near_source_ids: torch.Tensor
    far_source_ids: torch.Tensor


class CompiledHeightfield:
    """CPU reference derived from actual compiled samples and geom transform."""

    def __init__(self, model):
        ids = np.flatnonzero(np.asarray(model.geom_group) == TERRAIN_GROUP)
        if len(ids) != 1 or model.geom_type[ids[0]] != mujoco.mjtGeom.mjGEOM_HFIELD:
            raise ValueError("terrain group must contain exactly one Heightfield")
        self.geom_id = int(ids[0])
        body = int(model.geom_bodyid[self.geom_id])
        while body:
            if model.body_jntnum[body] or model.body_mocapid[body] >= 0:
                raise ValueError("Heightfield and its ancestors must be static")
            body = int(model.body_parentid[body])
        if model.geom_contype[self.geom_id] == 0 and model.geom_conaffinity[self.geom_id] == 0:
            raise ValueError("Heightfield must be physically collidable")
        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)
        self.position = data.geom_xpos[self.geom_id].copy()
        self.rotation = data.geom_xmat[self.geom_id].reshape(3, 3).copy()
        if not np.allclose(self.rotation[:, 2], [0, 0, 1], atol=1e-8, rtol=0):
            raise ValueError("first-stage Heightfield axes must be horizontal")
        hid = int(model.geom_dataid[self.geom_id])
        self.size = np.asarray(model.hfield_size[hid]).copy()
        nr, nc = int(model.hfield_nrow[hid]), int(model.hfield_ncol[hid])
        adr = int(model.hfield_adr[hid])
        self.samples = np.asarray(model.hfield_data[adr:adr+nr*nc], dtype=np.float64).reshape(nr, nc).copy()
        self.top_z = float(self.position[2] + self.samples.max() * self.size[2])
        self.bottom_z = float(self.position[2] - self.size[3])
        self.identity = {"geom_id": self.geom_id, "geom_name": model.geom(self.geom_id).name,
                         "hfield_name": model.hfield(hid).name,
                         "position": self.position.tolist(), "rotation": self.rotation.tolist(),
                         "size": self.size.tolist(), "shape_yx": [nr, nc],
                         "samples_sha256": hashlib.sha256(self.samples.tobytes()).hexdigest()}
        self.identity["sha256"] = hashlib.sha256(json.dumps(self.identity, sort_keys=True).encode()).hexdigest()

    def reference_heights(self, xy_w):
        """Exact triangle interpolation; not bilinear and not a label fallback."""
        xy = np.asarray(xy_w, dtype=np.float64)
        local = (xy - self.position[:2]) @ self.rotation[:2, :2]
        nr, nc = self.samples.shape
        u = (local[..., 0] + self.size[0]) * (nc-1) / (2*self.size[0])
        v = (local[..., 1] + self.size[1]) * (nr-1) / (2*self.size[1])
        valid = np.isfinite(xy).all(axis=-1) & (u >= 0) & (u <= nc-1) & (v >= 0) & (v <= nr-1)
        # Index clamping only protects invalid reference queries; invalid values
        # remain masked and never become edge-height labels.
        u = np.clip(np.nan_to_num(u), 0, nc-1)
        v = np.clip(np.nan_to_num(v), 0, nr-1)
        c = np.minimum(np.floor(u).astype(int), nc-2)
        r = np.minimum(np.floor(v).astype(int), nr-2)
        a, b = u-c, v-r
        h00, h10 = self.samples[r, c], self.samples[r, c+1]
        h01, h11 = self.samples[r+1, c], self.samples[r+1, c+1]
        h = np.where(a >= b, (1-a)*h00 + (a-b)*h10 + b*h11,
                     (1-b)*h00 + a*h11 + (b-a)*h01)
        return np.where(valid, self.position[2] + h*self.size[2], 0), valid


class TerrainGeometryQuery:
    """Selected-world native raycasts; no mutation of simulation Data or clocks."""

    def __init__(self, env):
        self.env, self.device = env, torch.device(env.device)
        if env.sim.mj_model.nflex:
            raise ValueError("first-stage terrain queries do not support flex geometry")
        self.surface = CompiledHeightfield(env.sim.mj_model)
        # First stage explicitly excludes world-varying geometry/model fields.
        for name in ("geom_dataid", "geom_size", "geom_matid", "geom_rgba", "mat_rgba"):
            if getattr(env.sim.wp_model, name).shape[0] != 1:
                raise ValueError(f"world-varying ray model field {name} is unsupported")
        self.catalog = build_channel_catalog()
        self._mount = resolve_lidar_mount()
        if self._mount is None:
            raise ValueError("actual lidar mount is missing")
        self._ray_contexts = {}

    def _synchronize_physics(self):
        if self.device.type == "cuda":
            wp.synchronize_device(str(self.device))

    def _rows(self, env_ids):
        rows = torch.as_tensor(env_ids, device=self.device)
        if rows.dtype != torch.int64 or rows.ndim != 1 or not len(rows):
            raise ValueError("env_ids must be a nonempty int64 vector")
        if bool(((rows < 0) | (rows >= self.env.num_envs)).any()) or len(torch.unique(rows)) != len(rows):
            raise ValueError("env_ids must be unique valid world indices")
        return rows

    @torch.no_grad()
    def raycast(self, env_ids, origins_w, directions_w):
        """Return native distances/geomid; misses are -1, no max-range conversion."""
        rows = self._rows(env_ids)
        if origins_w.ndim != 3 or origins_w.shape != directions_w.shape or origins_w.shape[0] != len(rows) or origins_w.shape[2] != 3:
            raise ValueError("ray arrays must have matching [selected_worlds,rays,3] shapes")
        for value in (origins_w, directions_w):
            if value.device != self.device or value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
                raise ValueError("ray arrays must be finite float32 on the environment device")
        if not bool(torch.allclose(directions_w.norm(dim=-1), torch.ones_like(directions_w[..., 0]), atol=1e-5, rtol=0)):
            raise ValueError("ray directions must be unit length")
        self._synchronize_physics()
        data = copy(self.env.sim.wp_data)
        data.nworld = len(rows)
        positions = wp.to_torch(self.env.sim.wp_data.geom_xpos).index_select(0, rows).contiguous()
        matrices = wp.to_torch(self.env.sim.wp_data.geom_xmat).index_select(0, rows).contiguous()
        data.geom_xpos = wp.from_torch(positions, dtype=wp.vec3)
        data.geom_xmat = wp.from_torch(matrices, dtype=wp.mat33)
        shape = origins_w.shape[:2]
        distance = wp.empty(shape, dtype=float, device=str(self.device))
        geoms = wp.empty(shape, dtype=int, device=str(self.device))
        normals = wp.empty(shape, dtype=wp.vec3, device=str(self.device))
        exclude = wp.full(shape[1], -1, dtype=int, device=str(self.device))
        groups = wp.types.vector(length=6, dtype=float)(0, 0, 0, -1, 0, 0)
        origins, directions = origins_w.contiguous(), directions_w.contiguous()
        stream = (wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(self.device)))
                  if self.device.type == "cuda" else nullcontext())
        with stream, wp.ScopedDevice(str(self.device)):
            # Use native compiled-HField BVH, not an independently generated
            # surface. BVH plus explicit stream ordering is the verified path;
            # early direct-path errors were not uniquely isolated from stream
            # ordering and are retained in the first-stage probe evidence.
            context = self._ray_contexts.pop(len(rows), None)
            if context is None:
                # Partial resets vary the selected-world count. Retain only the
                # two most recent sizes instead of accumulating render contexts.
                if len(self._ray_contexts) >= 2:
                    self._ray_contexts.pop(next(iter(self._ray_contexts)))
                context = mjwarp.create_render_context(
                    self.env.sim.mj_model, nworld=len(rows),
                    enabled_geom_groups=[TERRAIN_GROUP],
                    cam_active=[False]*self.env.sim.mj_model.ncam,
                    use_textures=False, use_shadows=False,
                )
            self._ray_contexts[len(rows)] = context
            mjwarp.refit_bvh(self.env.sim.wp_model, data, context)
            mjwarp.rays(
                self.env.sim.wp_model, data,
                wp.from_torch(origins, dtype=wp.vec3),
                wp.from_torch(directions, dtype=wp.vec3),
                groups, True, exclude, distance, geoms, normals, rc=context,
            )
        dist, ids = wp.to_torch(distance).clone(), wp.to_torch(geoms).clone()
        if not bool(torch.isfinite(dist).all()):
            raise RuntimeError("native terrain raycast produced nonfinite distances")
        if bool(((ids >= 0) & (ids != self.surface.geom_id)).any()):
            raise RuntimeError("native raycast returned a non-terrain geom")
        return dist, ids

    @torch.no_grad()
    def world_heights(self, env_ids, xy):
        """Read absolute surface Z at [B,P,2] world XY, without root reference."""
        rows = self._rows(env_ids)
        if xy.ndim != 3 or xy.shape[0] != len(rows) or xy.shape[2] != 2:
            raise ValueError("world XY must have shape [selected_worlds,points,2]")
        origins = torch.cat((xy, torch.full_like(xy[..., :1], self.surface.top_z + 1.0)), dim=-1)
        directions = torch.zeros_like(origins)
        directions[..., 2] = -1
        dist, ids = self.raycast(rows, origins, directions)
        # Exclude the finite Heightfield skirt: labels describe its top surface.
        local = (xy - torch.as_tensor(self.surface.position[:2], device=self.device, dtype=torch.float32)) @ torch.as_tensor(
            self.surface.rotation[:2, :2], device=self.device, dtype=torch.float32)
        inside = (local.abs() <= torch.as_tensor(self.surface.size[:2], device=self.device, dtype=torch.float32)).all(dim=-1)
        valid = (ids == self.surface.geom_id) & (dist >= 0) & inside
        height = torch.where(valid, origins[..., 2] - dist, torch.zeros_like(dist))
        return height, valid

    @torch.no_grad()
    def height_labels(self, env_ids, episode_ids, step_ids, grid: HeightGridCfg):
        self._synchronize_physics()
        rows = self._rows(env_ids)
        for identity in (episode_ids, step_ids):
            if identity.shape != rows.shape or identity.dtype != torch.int64 or identity.device != self.device or bool((identity < 0).any()):
                raise ValueError("episode/step identity must be selected-row nonnegative int64 vectors")
        if not torch.equal(step_ids, self.env.episode_length_buf.index_select(0, rows)):
            raise ValueError("labels must name the current pre-action control step")
        robot = self.env.scene["robot"]
        position = robot.data.root_link_pos_w.index_select(0, rows).detach().clone()
        quat = robot.data.root_link_quat_w.index_select(0, rows).detach().clone()
        xy = grid_world_xy(position, quat, grid)
        world_z, valid = self.world_heights(rows, xy)
        height = torch.where(valid, world_z - position[:, None, 2], torch.zeros_like(world_z))
        return HeightLabels(rows.clone(), episode_ids.clone(), step_ids.clone(),
                            height.reshape(-1, 11, 17), valid.reshape(-1, 11, 17),
                            xy.reshape(-1, 11, 17, 2).clone(), position, grid.identity(),
                            self.surface.identity["sha256"])

    @torch.no_grad()
    def scan(self, env_ids, *, source_ids=None, far_sampling="hit_only_10m"):
        self._synchronize_physics()
        rows = self._rows(env_ids)
        source = (torch.arange(86400, device=self.device) if source_ids is None
                  else torch.as_tensor(source_ids, device=self.device))
        if source.dtype != torch.int64 or source.ndim != 1 or not len(source) or bool(((source < 0) | (source >= 86400)).any()) or bool((source[1:] <= source[:-1]).any()):
            raise ValueError("source_ids must be sorted unique int64 Airy channel IDs")
        dirs = torch.tensor(self.catalog.directions_sensor, device=self.device, dtype=torch.float32).index_select(0, source)
        mount_p = torch.tensor(self._mount[0], device=self.device, dtype=torch.float32)
        mount_q = torch.tensor(self._mount[1], device=self.device, dtype=torch.float32)
        robot = self.env.scene["robot"]
        base_p = robot.data.root_link_pos_w.index_select(0, rows).clone()
        base_q = robot.data.root_link_quat_w.index_select(0, rows)[:, [1, 2, 3, 0]].clone()
        sensor_p = base_p + _quat_apply_xyzw(base_q, mount_p.expand(len(rows), 3))
        sensor_q = _quat_mul_xyzw(base_q, mount_q.expand(len(rows), 4))
        directions = _quat_apply_xyzw(sensor_q[:, None, :].expand(-1, len(source), -1), dirs[None].expand(len(rows), -1, -1))
        origins = sensor_p[:, None].expand_as(directions).contiguous()
        raw, ids = self.raycast(rows, origins, directions)
        hit = (ids >= 0) & (raw >= 0) & (raw < 15.0)
        distances = torch.where(hit, raw, torch.full_like(raw, 15.0))
        world = origins + distances[..., None] * directions
        sensor_points = distances[..., None] * dirs[None]
        base = _quat_apply_xyzw(mount_q[None, None].expand(len(rows), len(source), -1), sensor_points) + mount_p
        near_pool = torch.tensor(self.catalog.near_mask, device=self.device).index_select(0, source)[None].expand(len(rows), -1)
        far_pool = far_candidate_mask(~near_pool, distances, ~hit, far_sampling)
        ni = farthest_point_sampling_masked(base, near_pool, 256)
        fi = farthest_point_sampling_masked(base, far_pool, 128)
        nv, fv = ni >= 0, fi >= 0
        def gather(index, valid):
            safe_index = index.clamp_min(0)
            points = base.gather(1, safe_index[..., None].expand(-1, -1, 3))
            points = torch.where(valid[..., None], points, torch.zeros_like(points))
            sources = torch.where(valid, source[safe_index], torch.full_like(index, -1))
            return points, sources
        near, ns = gather(ni, nv)
        far, fs = gather(fi, fv)
        return TerrainScan(rows.clone(), source.clone(), distances, hit, world, base,
                           near, nv, far, fv, ns, fs)
