"""LiDAR observation terms + perception pipeline (Phase B: real raycasting).

Pipeline port (old ``EL4090_LidarHead`` + runner ``LidarWrapper``, exact
semantics; see ``Lidar_legged_gym/.../pd_gru_lidar_head/el_4090_lidar_head.py``
and ``rsl_rl/utils/lidar_wrapper.py``)::

    10 Hz tick (every 5th 50 Hz policy step, forced on any env reset):
        sensor pose <- base pose (wxyz->xyzw) x mount — read from the model's
                       lidar_sensor optical site (radar variant; scan +z
                       faces forward, slightly down), falling back to the old hand-written
                       (0.7, 0, -0.05; rpy (0, pi/2 + 0.1, 0)) on the base
                       variant
        LidarSensor.update()          warp raycast, Airy 900 az x 96 el,
                                      vs the M2 terrain warp mesh (26x26 m
                                      field incl. border, world frame)
        sensor-frame domain rand      2% point mask (fake dist U(0, 0.3)),
                                      2% distance noise, clamp r <= 15 m
        sensor frame -> body frame    + proximal/distal pool gather
                                      (angle_stride 30 -> 2,880 points)

    every policy step (at obs compute):
        LidarWrapper.wrap_obs         proximal FPS 256 + angular-key sort,
                                      distal 64-point uniform downsample ->
                                      10-frame ring buffer -> sort; done-env
                                      windows zeroed AFTER obs assembly.

Observation layout (policy group, mirrors the old runner's wrapped obs)::

    [proprio 66 | proximal 256*3 | distal 10*64*3]   (2754 total)

Cadence driver: mjlab computes observations exactly once per policy step
(``ManagerBasedRlEnv.step`` -> ``observation_manager.compute`` after the
auto-reset and ``sim.forward()``), which is where the old stack's
``post_physics_step`` tick + runner ``wrap_obs`` sat.  The tick is additionally
forced when any env reset this step (old ``_reset_lidar_buffers`` forced a
full-frame tick on every reset batch), refreshing the freshly reset envs'
pool from their post-reset pose.  Both the tick and the wrap are guarded by
``common_step_counter`` so repeated obs computes within one step are no-ops.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import warp as wp

from hexapod_mjlab.utils.lidar import LidarConfig, LidarSensor, LidarType
from hexapod_mjlab.utils.lidar.pool_samplers import build_pool_sampler
from hexapod_mjlab.rl.lidar_wrapper import LidarWrapper

from .terrain import El4090TerrainView, get_el_4090_terrain_view

# ── Pool / network layout (pd network constants; keep in sync with
#    observations.py and the old El4090LidarHeadCfg pd_risknet block). ──────
PROXIMAL_POINTS = 256
DISTAL_POINTS = 64
DISTAL_HISTORY = 10
POOL_NUM_POINTS = 2880  # 900 azimuth / stride 30 x 96 elevation
LIDAR_MAX_RANGE = 15.0

PROXIMAL_DIM = PROXIMAL_POINTS * 3  # 768
DISTAL_DIM = DISTAL_HISTORY * DISTAL_POINTS * 3  # 1920

# ── Proprio split point inside the policy group (after the 66-dim proprio). ─
PROPRIO_DIM = 66


# ── Model-derived mount (single source of truth). ───────────────────────────
EL4090_LIDAR_MJCF = (
  Path(__file__).parents[3]
  / "assets" / "robots" / "el_4090" / "mjcf" / "el_4090_lidar.xml"
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


@dataclass(frozen=True)
class El4090LidarCfg:
  """Old ``cfg.lidar`` + ``cfg.pool`` + ``cfg.pd_risknet`` + LiDAR DR fields."""

  # Sensor (Airy full-frame).
  airy_n_elevation: int = 96
  airy_n_azimuth: int = 900  # hardcoded upstream (int(360/0.4) FP edge)
  airy_horizontal_resolution_deg: float = 0.4
  update_frequency_hz: float = 10.0  # env-side decimation from 50 Hz control
  max_range: float = LIDAR_MAX_RANGE
  min_range: float = 0.2
  enable_sensor_noise: bool = True
  pixel_std_dev_multiplier: float = 0.02
  pixel_dropout_prob: float = 0.02

  # Mount. Single source of truth is the lidar_sensor site in the radar
  # variant MJCF (mount_from_model=True; optical +z points forward/down at the
  # head; optical-frame alignment corrected 2026-10-05). The values below are the
  # fallback for the base variant (no site) — old cfg.lidar offset_pos /
  # sensor_offset_rpy, isaacgym [x,y,z,w].
  offset_pos: tuple[float, float, float] = (0.7, 0.0, -0.05)
  pitch_tilt_down_rad: float = 0.1
  sensor_offset_rpy: tuple[float, float, float] = (0.0, math.pi / 2 + 0.1, 0.0)
  mount_from_model: bool = True

  # Pool (old cfg.pool: angle_stride).
  pool_strategy: str = "angle_stride"
  pool_azimuth_stride: int = 30
  pool_seed: int = 42

  # Proximal/distal split cone (old cfg.pd_risknet).
  split_theta_deg: float = 32.0
  split_axis_tilt_deg: float = 56.0

  # Env-side point-cloud domain rand (old cfg.domain_rand).
  lidar_point_mask_ratio: float = 0.02
  lidar_point_mask_value_range: tuple[float, float] = (0.0, 0.3)
  lidar_distance_noise_ratio: float = 0.02


# ── xyzw quaternion helpers (isaacgym convention; same formulas as the S2
#    spike and the old env's isaacgym.torch_utils usage). ────────────────────


def _quat_mul_xyzw(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  ax, ay, az, aw = a.unbind(-1)
  bx, by, bz, bw = b.unbind(-1)
  return torch.stack(
    [
      aw * bx + ax * bw + ay * bz - az * by,
      aw * by - ax * bz + ay * bw + az * bx,
      aw * bz + ax * by - ay * bx + az * bw,
      aw * bw - ax * bx - ay * by - az * bz,
    ],
    dim=-1,
  )


def _quat_apply_xyzw(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  qv, qw = q[..., :3], q[..., 3:4]
  t = 2.0 * torch.linalg.cross(qv, v, dim=-1)
  return v + qw * t + torch.linalg.cross(qv, t, dim=-1)


def _quat_from_euler_xyz(roll: float, pitch: float, yaw: float) -> torch.Tensor:
  """q = qz(yaw) * qy(pitch) * qx(roll), xyzw, shape (1, 4)."""
  cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
  cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
  cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
  qx = torch.tensor([sr, 0.0, 0.0, cr], dtype=torch.float32)
  qy = torch.tensor([0.0, sp, 0.0, cp], dtype=torch.float32)
  qz = torch.tensor([0.0, 0.0, sy, cy], dtype=torch.float32)
  return _quat_mul_xyzw(qz, _quat_mul_xyzw(qy, qx)).view(1, 4)


class El4090LidarPipeline:
  """Env-cached LiDAR state machine (old env sensor stack + runner wrapper).

  Owns the warp raycast sensor against the M2 terrain mesh, the 10 Hz tick,
  the pool sampler and the proximal/distal production (via the ported
  :class:`hexapod_mjlab.rl.lidar_wrapper.LidarWrapper`).
  """

  def __init__(self, env, cfg: El4090LidarCfg | None = None):
    import numpy as np

    self.env = env
    self.cfg = cfg or El4090LidarCfg()
    self.num_envs = env.num_envs
    self.device = env.device
    device_str = str(env.device)
    c = self.cfg

    assert c.airy_n_azimuth * c.airy_horizontal_resolution_deg == 360.0, (
      "airy azimuth count inconsistent with horizontal resolution"
    )
    raw_num_rays = c.airy_n_azimuth * c.airy_n_elevation
    assert (c.airy_n_azimuth // c.pool_azimuth_stride) * c.airy_n_elevation == (
      POOL_NUM_POINTS
    ), "pool config does not yield the 2,880-point pool"

    # ── Terrain warp mesh: full M2 field (with border) in the mjlab world
    #    frame (warp-OWNED copies; wp.from_torch aliases die with the torch
    #    tensor — S2 finding). ─────────────────────────────────────────────
    wp.init()
    self.terrain_view: El4090TerrainView = get_el_4090_terrain_view(env)
    td = self.terrain_view.data
    off = self.terrain_view.world_offset_xy
    verts = np.asarray(td.vertices, dtype=np.float32).copy()
    verts[:, 0] -= off[0]
    verts[:, 1] -= off[1]
    tris = np.ascontiguousarray(td.triangles_flat, dtype=np.int32)
    with wp.ScopedDevice(device_str):
      self._wp_mesh = wp.Mesh(
        points=wp.array(np.ascontiguousarray(verts), dtype=wp.vec3),
        indices=wp.array(tris),
      )
      self.mesh_ids = wp.array([self._wp_mesh.id], dtype=wp.uint64)

    # ── Sensor pose tensors (kept alive: the sensor's warp arrays zero-copy
    #    alias them; every tick writes the fresh pose into them). ──────────
    self.sensor_pos_tensor = torch.zeros(
      self.num_envs, 3, device=self.device, dtype=torch.float32
    )
    self.sensor_quat_tensor = torch.zeros(
      self.num_envs, 4, device=self.device, dtype=torch.float32
    )  # xyzw (wp.quat_rotate convention, verified vs warp 1.17)

    lidar_sensor_cfg = LidarConfig(
      sensor_type=LidarType.AIRY,
      dt=float(env.step_dt),
      update_frequency=float(c.update_frequency_hz),
      max_range=float(c.max_range),
      min_range=float(c.min_range),
      num_sensors=1,
      # pattern lidar overrides these two internally (kept for print parity)
      horizontal_line_num=int(c.airy_n_azimuth),
      vertical_line_num=int(c.airy_n_elevation),
      airy_horizontal_resolution_deg=float(c.airy_horizontal_resolution_deg),
      return_pointcloud=True,
      pointcloud_in_world_frame=False,  # sensor frame (pixel = dist * dir)
      randomize_placement=False,  # must be explicit False on the warp path
      enable_sensor_noise=bool(c.enable_sensor_noise),
      pixel_std_dev_multiplier=float(c.pixel_std_dev_multiplier),
      pixel_dropout_prob=float(c.pixel_dropout_prob),
    )
    lidar_env = {
      "device": device_str,
      "num_envs": self.num_envs,
      "num_sensors": 1,
      "sensor_pos_tensor": self.sensor_pos_tensor,
      "sensor_quat_tensor": self.sensor_quat_tensor,
      "mesh_ids": self.mesh_ids,
    }
    self.sensor = LidarSensor(
      lidar_env, None, lidar_sensor_cfg, num_sensors=1, device=device_str
    )
    self.num_rays = raw_num_rays

    # ── Mount transform (xyzw). Model-derived when the lidar_sensor site
    #    exists (radar variant); falls back to the hand-written cfg values
    #    for the base variant. ──────────────────────────────────────────────
    mount = resolve_lidar_mount() if c.mount_from_model else None
    if mount is not None:
      mount_pos, mount_quat_xyzw = mount
      self._mount_t = torch.tensor(
        mount_pos, dtype=torch.float32, device=self.device
      ).view(1, 3)
      self._mount_q = torch.tensor(
        mount_quat_xyzw, dtype=torch.float32, device=self.device
      ).view(1, 4)
    else:
      self._mount_t = torch.tensor(
        c.offset_pos, dtype=torch.float32, device=self.device
      ).view(1, 3)
      self._mount_q = _quat_from_euler_xyz(*c.sensor_offset_rpy).to(self.device)
    self._sensor_translation = self._mount_t.expand(self.num_envs, 3)
    self._sensor_offset_quat = self._mount_q.expand(self.num_envs, 4)

    # ── Proximal/distal split cone (old _init_cone_axis, scheme A). ───────
    delta = math.radians(c.split_axis_tilt_deg)
    theta = math.radians(c.split_theta_deg)
    dome_axis = _quat_apply_xyzw(
      self._mount_q.to(self.device),
      torch.tensor([[0.0, 0.0, 1.0]], device=self.device),
    )[0]
    cd, sd = math.cos(delta), math.sin(delta)
    axis = torch.stack(
      [cd * dome_axis[0] + sd * dome_axis[2],
       dome_axis[1],
       -sd * dome_axis[0] + cd * dome_axis[2]]
    )
    cone_axis_body = axis / torch.linalg.norm(axis)
    self.cone_axis_body = cone_axis_body
    self.cone_sin_thr = math.sin(theta)

    # ── Pool sampler (angle_stride default). ──────────────────────────────
    self.pool_cfg = _PoolCfgShim(
      strategy=c.pool_strategy, seed=c.pool_seed,
      azimuth_stride=c.pool_azimuth_stride,
    )
    self.pool_sampler = build_pool_sampler(
      self.pool_cfg,
      n_azimuth=c.airy_n_azimuth,
      n_elevation=c.airy_n_elevation,
      raw_num_rays=raw_num_rays,
      device=self.device,
    )
    if self.pool_sampler.fixed_output_size != POOL_NUM_POINTS:
      raise ValueError(
        f"pool sampler output {self.pool_sampler.fixed_output_size} != "
        f"POOL_NUM_POINTS {POOL_NUM_POINTS}"
      )

    # ── Buffers (old env: lidar_points_base / raycast_distances). ─────────
    self.lidar_points_base = torch.zeros(
      self.num_envs, POOL_NUM_POINTS, 3, device=self.device, dtype=torch.float32
    )
    self.raycast_distances = torch.full(
      (self.num_envs, POOL_NUM_POINTS), LIDAR_MAX_RANGE,
      device=self.device, dtype=torch.float32,
    )
    self._lidar_frame_count = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.long
    )
    self.total_ticks = 0
    #: common_step_counter of every tick (cadence audit: spacing 5, or a
    #: run of 1s when env resets force ticks).
    self.tick_log: list[int] = []

    # ── Prox/distal production (old runner LidarWrapper; proprio pass-through
    #    is empty — the obs manager concatenates the proprio terms itself). ─
    self.wrapper = LidarWrapper(
      num_envs=self.num_envs,
      num_lidar_points=POOL_NUM_POINTS,
      distal_history_length=DISTAL_HISTORY,
      proximal_points=PROXIMAL_POINTS,
      distal_points=DISTAL_POINTS,
      phi_threshold_deg=c.split_theta_deg,
      proprio_dim=0,
      device=self.device,
      sensor_offset_quat=self._mount_q,
      sensor_translation=self._mount_t,
      cone_axis_body=cone_axis_body,
    )
    self._empty_proprio = torch.zeros(
      self.num_envs, 0, device=self.device, dtype=torch.float32
    )

    # Decimation from the 50 Hz control step to the LiDAR rate (old formula).
    self.pool_decimation = max(
      1, int(round(1.0 / (env.step_dt * c.update_frequency_hz)))
    )
    self._decimation_counter = 0
    self._seen_step = -1
    self._produced_step = -1
    self._prox_block = torch.zeros(
      self.num_envs, PROXIMAL_DIM, device=self.device, dtype=torch.float32
    )
    self._distal_block = torch.zeros(
      self.num_envs, DISTAL_DIM, device=self.device, dtype=torch.float32
    )

    # Old _init_buffers ran one tick at construction so the first obs is valid
    # (and set the decimation phase to 0: the first counter tick lands on the
    # FIRST policy step, so ticks land on steps 5, 10, ... like the old loop).
    self._seen_step = int(env.common_step_counter)
    self._tick()
    self._produce(reset_buf=None)

  # ── 10 Hz tick (old _update_lidar_history) ────────────────────────────────

  def _tick(self) -> bool:
    """One LiDAR tick: raycast -> DR -> body frame -> pool."""
    env = self.env
    robot = env.scene["robot"]

    # SYNC 1: pending physics kernels -> pose reads (S2 recipe).
    if str(self.device).startswith("cuda"):
      wp.synchronize_device()

    base_pos = robot.data.root_link_pos_w  # (B, 3)
    base_quat_wxyz = robot.data.root_link_quat_w  # (B, 4) wxyz (mjlab)
    base_quat = base_quat_wxyz[:, [1, 2, 3, 0]]  # -> xyzw (wp convention)

    # Sensor world pose (old formula, xyzw).
    sensor_q = _quat_mul_xyzw(base_quat, self._mount_q.expand(self.num_envs, 4))
    sensor_p = base_pos + _quat_apply_xyzw(
      base_quat, self._mount_t.expand(self.num_envs, 3)
    )
    self.sensor_quat_tensor.copy_(sensor_q)
    self.sensor_pos_tensor.copy_(sensor_p)
    if str(self.device).startswith("cuda"):
      torch.cuda.synchronize()  # SYNC 2: pose writes -> raycast kernel

    points_sensor, dist = self.sensor.update()

    if str(self.device).startswith("cuda"):
      wp.synchronize_device()  # SYNC 3: raycast outputs -> torch

    points_sensor = points_sensor.view(self.num_envs, -1, 3).clone()
    dist = dist.view(self.num_envs, -1).clone()

    # Env-side domain rand in the SENSOR frame (old _apply_lidar_domain_rand).
    points_sensor, dist = self._apply_domain_rand(points_sensor, dist)

    # Sensor frame -> body frame (old full-frame transform).
    n_total = int(points_sensor.numel() // 3)
    points_base = _quat_apply_xyzw(
      self._mount_q.expand(n_total, 4), points_sensor.reshape(-1, 3)
    ).reshape(self.num_envs, points_sensor.shape[1], 3) + self._sensor_translation.unsqueeze(1)

    # Pool layer (old gather into the preallocated buffers).
    idx = self.pool_sampler.sample(points_base, dist, self._lidar_frame_count)
    self.lidar_points_base.copy_(
      torch.gather(points_base, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
    )
    self.raycast_distances.copy_(torch.gather(dist, 1, idx))

    self._lidar_frame_count += 1
    self.total_ticks += 1
    self.tick_log.append(int(env.common_step_counter))
    return True

  def _apply_domain_rand(
    self, points_sensor: torch.Tensor, dist: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    c = self.cfg
    if c.lidar_point_mask_ratio > 0.0:
      rand_mask = torch.rand_like(dist) < c.lidar_point_mask_ratio
      lo, hi = c.lidar_point_mask_value_range
      fake_dist = torch.rand_like(dist) * (hi - lo) + lo
      dir_norm = torch.linalg.norm(points_sensor, dim=-1, keepdim=True).clamp(min=1e-6)
      dir_unit = points_sensor / dir_norm
      points_sensor = torch.where(
        rand_mask.unsqueeze(-1), dir_unit * fake_dist.unsqueeze(-1), points_sensor
      )
      dist = torch.where(rand_mask, fake_dist, dist)

    if c.lidar_distance_noise_ratio > 0.0:
      scale = 1.0 + (2.0 * torch.rand_like(dist) - 1.0) * c.lidar_distance_noise_ratio
      points_sensor = points_sensor * scale.unsqueeze(-1)
      dist = dist * scale

    # Deployment contract: no point beyond max_range (miss sentinel is 15 m).
    d_max = float(c.max_range)
    dist = dist.clamp(min=0.0, max=d_max)
    norm = torch.linalg.norm(points_sensor, dim=-1, keepdim=True)
    over = (norm > d_max).expand_as(points_sensor)
    if bool(over.any()):
      points_sensor = torch.where(
        over, points_sensor / norm.clamp(min=1e-6) * d_max, points_sensor
      )
    return points_sensor, dist

  # ── Cadence + obs production ──────────────────────────────────────────────

  def _maybe_tick(self) -> None:
    """Old cadence: tick every ``pool_decimation``-th policy step, forced when
    any env reset this step (old ``_reset_lidar_buffers``)."""
    env = self.env
    step = int(env.common_step_counter)
    if step == self._seen_step:
      return  # already advanced this policy step
    self._seen_step = step
    self._decimation_counter += 1
    reset_buf = getattr(env, "reset_buf", None)
    forced = reset_buf is not None and bool(reset_buf.any())
    if forced or self._decimation_counter >= self.pool_decimation:
      self._decimation_counter = 0
      if forced and reset_buf is not None:
        done_ids = reset_buf.nonzero(as_tuple=False).squeeze(-1)
        self._lidar_frame_count[done_ids] = 0
      self._tick()

  def _produce(self, reset_buf: torch.Tensor | None) -> None:
    """Prox/distal blocks from the current pool (old wrap_obs per step)."""
    self._maybe_tick()
    if reset_buf is None:
      reset_buf = torch.zeros(
        self.num_envs, dtype=torch.bool, device=self.device
      )
    wrapped = self.wrapper.wrap_obs(
      self._empty_proprio, self.lidar_points_base, reset_buf
    )
    self._prox_block = wrapped[:, :PROXIMAL_DIM].clone()
    self._distal_block = wrapped[:, PROXIMAL_DIM:].clone()
    self._produced_step = int(self.env.common_step_counter)

  def ensure_current(self) -> None:
    """Idempotent per policy step: tick if due, then rebuild the blocks."""
    if self._produced_step == int(self.env.common_step_counter):
      return
    self._produce(getattr(self.env, "reset_buf", None))

  # Diagnostics (used by the gate scripts).
  @property
  def distal_window(self) -> torch.Tensor:
    return self.wrapper._distal_window

  @property
  def distal_frame_count(self) -> torch.Tensor:
    return self.wrapper._distal_frame_count

  @property
  def hit_mask(self) -> torch.Tensor:
    """Pool rays that hit something (dist below the miss sentinel)."""
    return self.raycast_distances < (LIDAR_MAX_RANGE - 1e-4)


@dataclass
class _PoolCfgShim:
  """Attribute shim: exposes the pool fields the sampler factory reads."""

  strategy: str = "angle_stride"
  seed: int = 42
  azimuth_stride: int = 30


def get_lidar_pipeline(env, cfg: El4090LidarCfg | None = None) -> El4090LidarPipeline:
  """Fetch (and cache on the env) the shared LiDAR pipeline.

  ``cfg`` is only consumed on first creation; the default keeps the legacy
  randomized sensor configuration (old task behavior unchanged). The T0 flat
  variant pre-creates the pipeline with a noise-free ``El4090LidarCfg``
  through the observation-term ``params`` before any obs is computed.
  """
  pipeline = env.__dict__.get("_el_4090_lidar_pipeline")
  if pipeline is None:
    pipeline = El4090LidarPipeline(env, cfg)
    env._el_4090_lidar_pipeline = pipeline
  return pipeline


def lidar_prox_obs(env, lidar_cfg: El4090LidarCfg | None = None) -> torch.Tensor:
  """Proximal point block, ``(B, 768)`` (old: FPS 256 + angular-key sort)."""
  get_lidar_pipeline(env, lidar_cfg).ensure_current()
  return get_lidar_pipeline(env)._prox_block


def lidar_distal_obs(env, lidar_cfg: El4090LidarCfg | None = None) -> torch.Tensor:
  """Distal history block, ``(B, 1920)`` (old: ring buffer + sort).

  The done-env zeroing happens INSIDE the production step (old runner called
  wrap_obs with this step's dones and zeroed the windows after assembly), so
  this term just returns the freshly produced block.
  """
  get_lidar_pipeline(env, lidar_cfg).ensure_current()
  return get_lidar_pipeline(env)._distal_block
