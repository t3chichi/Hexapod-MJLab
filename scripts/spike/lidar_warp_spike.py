#!/usr/bin/env python
"""S2 spike: ported LidarSensor on a mujoco-warp batched scene.

Integration proof + cost measurement for the Lidar_mjlab migration (M0 -> M3).

What this script does
---------------------
1. Builds one MJCF scene: ground plane + 5 static box obstacles + a small
   freejoint "robot proxy" box that starts 0.5 m above the ground near the
   origin (with a small yaw/pitch tilt so the quat pose flow is nontrivial).
2. Runs it under mujoco-warp batched physics (``mjwarp.put_model`` /
   ``put_data(nworld=N)`` / ``mjwarp.step``), dt=0.005 s, 50 Hz control
   cadence (4 physics steps per control step, no actuators -> ctrl is a
   structural no-op).
3. Mirrors the SAME geometry (plane + boxes, robot excluded) as a trimesh in
   numpy -> ``wp.Mesh`` (vertices float32 (V,3); triangles int32 FLATTENED
   1-D — warp 1.17 requirement) and instantiates
   ``hexapod_mjlab.utils.lidar.LidarSensor`` with the REAL task config from the old
   repo (el_4090 pd_gru_lidar_head: Airy 900 az x 96 el = 86,400 rays,
   mount translation [0.7, 0, -0.05] with rpy [0, pi/2+0.1, 0], 10 Hz,
   max_range 15 m, multiplicative-Gaussian noise 0.02 + dropout 0.02).
4. Every 5th control step (10 Hz LiDAR tick): reads robot-proxy xpos/xquat
   for ALL worlds from mujoco-warp Data -> torch (float32, cuda) -> writes
   the sensor pose tensors (zero-copy aliased by the sensor's warp arrays),
   then calls ``sensor.update()`` (warp raycast kernel + torch noise).
   LiDAR update runs OUTSIDE any graph-captured region, between
   ``mjwarp.step`` batches. Explicit synchronization is done at every
   warp->torch and torch->warp hand-off (see SYNC points below).
5. 50 LiDAR ticks (+3 warmup excluded from stats). Asserts: outputs finite,
   pixel norm == dist (sensor-frame contract pixel = dist * unit_dir),
   hit rate strictly in (0, 1), obstacles provably produce hits (mesh-swap
   differential), robot pose actually moves. Measures per-tick wall time
   (raycast+sync), physics batch time, torch peak allocation and
   nvidia-smi process VRAM.
6. Attempts nworld=4096 in an isolated subprocess (OOM caught cleanly);
   on failure walks a fallback ladder (2048, 1024) to record the largest
   nworld that fits, plus the 64 -> 512 memory scaling slope.
7. Probes CUDA graph capture behaviour under driver 535 (CUDA 12.2) in an
   isolated subprocess: ``wp.is_conditional_graph_supported()``, capture of
   ``mjwarp.step``, and capture of the LidarSensor kernel. A crash in the
   probe cannot take down the measurement run.

LidarSensor call pattern (the M3 integration recipe), mirroring the old env
``el_4090_lidar_head.py::_init_lidar_sensor`` / ``_update_lidar_history``:

    wp.init()                                            # FIRST warp op, always
    verts  = <np.float32 (V,3)>
    tris   = <np.int32 (T,3)>
    # NOTE: mesh arrays must be warp-OWNED copies (wp.from_numpy + device=);
    # wp.from_torch aliases torch storage without keeping it alive -> mesh
    # points can be recycled under you -> illegal memory access in raycasts.
    mesh   = wp.Mesh(points=wp.from_numpy(verts, dtype=wp.vec3, device="cuda:0"),
                     indices=wp.from_numpy(tris.flatten(), dtype=wp.int32,
                                           device="cuda:0"))
    mesh_ids = wp.array([mesh.id], dtype=wp.uint64)
    pos_t  = torch.zeros(num_envs, 3,  device="cuda:0")   # aliased by sensor
    quat_t = torch.zeros(num_envs, 4,  device="cuda:0")   # (xyzw, isaac style)
    cfg = LidarConfig(sensor_type=LidarType.AIRY, dt=0.02, update_frequency=10.0,
                      max_range=15.0, min_range=0.2, num_sensors=1,
                      horizontal_line_num=900, vertical_line_num=96,
                      airy_horizontal_resolution_deg=0.4,
                      return_pointcloud=True, pointcloud_in_world_frame=False,
                      randomize_placement=False,          # must be explicit!
                      enable_sensor_noise=True, pixel_std_dev_multiplier=0.02,
                      pixel_dropout_prob=0.02)
    env = {"device": "cuda:0", "num_envs": num_envs, "num_sensors": 1,
           "sensor_pos_tensor": pos_t, "sensor_quat_tensor": quat_t,
           "mesh_ids": mesh_ids}
    sensor = LidarSensor(env, None, cfg, num_sensors=1, device="cuda:0")
    # per tick:
    #   SYNC  wp.synchronize_device()          # mujoco-warp -> torch poses
    #   pos_t.copy_(base_pos + quat_apply(base_quat_xyzw, t_mount))
    #   quat_t.copy_(quat_mul(base_quat_xyzw, q_mount))          # xyzw!
    #   SYNC  torch.cuda.synchronize()         # torch writes -> warp kernel
    #   pixels, dists = sensor.update()        # (N,1,R,1,3)/(N,1,R,1) torch
    #   SYNC  wp.synchronize_device()          # raycast outputs -> torch

Usage (conda env mjlab):
    python scripts/spike/lidar_warp_spike.py                 # gates: 64, 512 + 4096 ladder + graph probe
    python scripts/spike/lidar_warp_spike.py --nworlds 64,512 --ladder 4096,2048,1024
    python scripts/spike/lidar_warp_spike.py --run-single 64     # one worker (JSON result line)
    python scripts/spike/lidar_warp_spike.py --graph-probe 64    # graph capture probe worker
    python scripts/spike/lidar_warp_spike.py --ticks 10 --nworlds 64 --inprocess   # quick debug
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time

# ── CRITICAL SAFETY ────────────────────────────────────────────────────────
# wp.init() must be the FIRST warp operation in every entry point. Launching
# mesh_query_ray kernels without it segfaults with no traceback (warp 1.17).
import warp as wp

wp.init()

import numpy as np  # noqa: E402
import torch  # noqa: E402

import mujoco  # noqa: E402
import mujoco_warp as mjwarp  # noqa: E402

# hexapod_mjlab importable even when the package is not pip-installed
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if os.path.isdir(_SRC):
    sys.path.insert(0, _SRC)

from hexapod_mjlab.utils.lidar import LidarConfig, LidarSensor, LidarType  # noqa: E402

DEVICE = "cuda:0"

# ── Task config constants (READ from old repo el_4090_lidar_head_config.py) ─
AIRY_N_AZIMUTH = 900            # hardcoded upstream; int(360/0.4) is on a FP edge
AIRY_N_ELEVATION = 96           # -> 86,400 rays per env per tick
AIRY_H_RES_DEG = 0.4
LIDAR_UPDATE_HZ = 10.0
LIDAR_MAX_RANGE = 15.0          # PD_LIDAR_MAX_RANGE (far_plane AND sentinel)
LIDAR_MIN_RANGE = 0.2
SENSOR_OFFSET_POS = [0.7, 0.0, -0.05]
PITCH_TILT_DOWN_RAD = 0.1
SENSOR_OFFSET_RPY = [0.0, math.pi / 2.0 + PITCH_TILT_DOWN_RAD, 0.0]
SENSOR_NOISE = True
PIXEL_STD_DEV_MULTIPLIER = 0.02
PIXEL_DROPOUT_PROB = 0.02
CONTROL_DT = 0.02               # 50 Hz control cadence
PHYS_DT = 0.005                 # mujoco timestep
STEPS_PER_CONTROL = int(round(CONTROL_DT / PHYS_DT))              # 4
# decimation from control rate to LiDAR rate (old env _pool_decimation
# formula: max(1, round(1/(dt*hz)))  — NOT dt*hz, which truncates to 0!)
CONTROL_STEPS_PER_TICK = max(1, int(round(1.0 / (CONTROL_DT * LIDAR_UPDATE_HZ))))  # 5 -> 10 Hz
assert STEPS_PER_CONTROL == 4 and CONTROL_STEPS_PER_TICK == 5

# ── Scene geometry (single source of truth for MJCF *and* the LiDAR trimesh) ─
# (name, cx, cy, cz, hx, hy, hz): center + half sizes; heights 0.3..1.0 m,
# all within 6 m of the origin.
OBSTACLES = [
    ("obstacle_0", 2.5, 1.5, 0.45, 0.30, 0.30, 0.45),  # h=0.9
    ("obstacle_1", -2.0, 2.5, 0.50, 0.50, 0.50, 0.50),  # h=1.0
    ("obstacle_2", 1.0, -3.0, 0.15, 0.40, 0.40, 0.15),  # h=0.3
    ("obstacle_3", 4.0, -1.0, 0.30, 0.50, 0.50, 0.30),  # h=0.6
    ("obstacle_4", -3.5, -3.5, 0.35, 0.35, 0.35, 0.35),  # h=0.7
]
PLANE_HALF = 20.0               # >= max_range=15 m
ROBOT_HALF = 0.10
ROBOT_START_Z = ROBOT_HALF + 0.50   # lowest face 0.5 m above ground
ROBOT_TILT_YAW = 0.50           # nontrivial initial orientation
ROBOT_TILT_PITCH = 0.12

WARMUP_TICKS = 3                # excluded from steady-state timing stats


# ═══════════════════════════════════════════════════════════════════════════
# Small torch quat helpers — isaacgym [x, y, z, w] convention, exactly the
# semantics the old env used via isaacgym.torch_utils (no isaacgym here).
# ═══════════════════════════════════════════════════════════════════════════
def quat_from_euler_xyz(roll: float, pitch: float, yaw: float) -> torch.Tensor:
    """q = qz(yaw) * qy(pitch) * qx(roll), xyzw, shape (1, 4)."""
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    # generic composition q = qz(yaw) * qy(pitch) * qx(roll) via quat_mul
    qx = torch.tensor([sr, 0.0, 0.0, cr], dtype=torch.float32, device=DEVICE)
    qy = torch.tensor([0.0, sp, 0.0, cp], dtype=torch.float32, device=DEVICE)
    qz = torch.tensor([0.0, 0.0, sy, cy], dtype=torch.float32, device=DEVICE)
    return quat_mul(qz, quat_mul(qy, qx)).view(1, 4)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Elementwise Hamilton product of xyzw quats; (..., 4)."""
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


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate v (..., 3) by unit quat q (..., 4) xyzw."""
    qv = q[..., :3]
    qw = q[..., 3:4]
    t = 2.0 * torch.linalg.cross(qv, v, dim=-1)
    return v + qw * t + torch.linalg.cross(qv, t, dim=-1)


def _self_test_quat_math() -> None:
    """Cheap analytic checks so a quat-convention bug cannot pass silently."""
    qy90 = quat_from_euler_xyz(0.0, math.pi / 2, 0.0)
    x = torch.tensor([[1.0, 0.0, 0.0]], device=DEVICE)
    got = quat_apply(qy90, x).squeeze(0).cpu().numpy()
    assert np.allclose(got, [0.0, 0.0, -1.0], atol=1e-5), f"R_y(90)x != -z: {got}"
    q_off = quat_from_euler_xyz(*SENSOR_OFFSET_RPY)
    dome = quat_apply(q_off, torch.tensor([[0.0, 0.0, 1.0]], device=DEVICE))
    want = [math.cos(PITCH_TILT_DOWN_RAD), 0.0, -math.sin(PITCH_TILT_DOWN_RAD)]
    assert np.allclose(dome.squeeze(0).cpu().numpy(), want, atol=1e-5), \
        f"dome axis mismatch: {dome}"
    q_i = quat_mul(qy90, quat_from_euler_xyz(0.0, -math.pi / 2, 0.0))
    assert np.allclose(q_i.squeeze(0).cpu().numpy(), [0, 0, 0, 1], atol=1e-5)
    print("[spike] quat math self-test passed (xyzw, isaacgym semantics)")


# ═══════════════════════════════════════════════════════════════════════════
# Scene construction
# ═══════════════════════════════════════════════════════════════════════════
def _wxyz_quat_str(yaw: float, pitch: float) -> str:
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    # q = qz(yaw) * qy(pitch), wxyz
    w, x = cy * cp, sy * sp
    y, z = cy * sp, sy * cp
    return f"{w:.6f} {x:.6f} {y:.6f} {z:.6f}"


def build_mjcf() -> str:
    obs_xml = "\n".join(
        f'    <body name="{n}" pos="{cx:.3f} {cy:.3f} {cz:.3f}">'
        f'<geom name="{n}_geom" type="box" size="{hx:.3f} {hy:.3f} {hz:.3f}"/>'
        f"</body>"
        for n, cx, cy, cz, hx, hy, hz in OBSTACLES
    )
    return f"""
<mujoco model="lidar_mjwarp_spike">
  <option timestep="{PHYS_DT}" integrator="Euler" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="{PLANE_HALF} {PLANE_HALF} 0.1"
          friction="1.0 0.005 0.0001"/>
{obs_xml}
    <body name="robot_proxy" pos="0 0 {ROBOT_START_Z:.3f}"
          quat="{_wxyz_quat_str(ROBOT_TILT_YAW, ROBOT_TILT_PITCH)}">
      <freejoint name="robot_joint"/>
      <geom name="robot_geom" type="box" size="{ROBOT_HALF} {ROBOT_HALF} {ROBOT_HALF}"
            mass="2.0" friction="1.0 0.005 0.0001"/>
    </body>
  </worldbody>
</mujoco>
"""


def build_trimesh(include_obstacles: bool = True):
    """(vertices (V,3) float32 numpy, triangles (T,3) int32 numpy).

    Identical geometry to the MJCF: z=0 plane + obstacle boxes. The robot
    proxy is NOT part of the LiDAR mesh (same contract as the old env, whose
    mesh is terrain-only).
    """
    verts = [
        (-PLANE_HALF, -PLANE_HALF, 0.0), (PLANE_HALF, -PLANE_HALF, 0.0),
        (PLANE_HALF, PLANE_HALF, 0.0), (-PLANE_HALF, PLANE_HALF, 0.0),
    ]
    tris = [(0, 1, 2), (0, 2, 3)]
    if include_obstacles:
        for _, cx, cy, cz, hx, hy, hz in OBSTACLES:
            b = len(verts)
            # 8 corners (bit i: +x, +y, +z)
            for k in range(8):
                verts.append((
                    cx + (hx if k & 1 else -hx),
                    cy + (hy if k & 2 else -hy),
                    cz + (hz if k & 4 else -hz),
                ))
            tris += [
                (b + 0, b + 3, b + 2), (b + 0, b + 2, b + 1),  # bottom
                (b + 4, b + 5, b + 6), (b + 4, b + 6, b + 7),  # top
                (b + 0, b + 1, b + 5), (b + 0, b + 5, b + 4),  # y-
                (b + 2, b + 3, b + 7), (b + 2, b + 7, b + 6),  # y+
                (b + 3, b + 0, b + 4), (b + 3, b + 4, b + 7),  # x-
                (b + 1, b + 2, b + 6), (b + 1, b + 6, b + 5),  # x+
            ]
    vertices = np.asarray(verts, dtype=np.float32)
    triangles = np.asarray(tris, dtype=np.int32)
    return vertices, triangles


def make_lidar_mesh(include_obstacles: bool = True):
    """wp.Mesh + mesh_ids exactly like the old env (1-D flattened indices).

    MEMORY OWNERSHIP (spike finding): ``wp.from_torch`` zero-copy aliases the
    torch tensor and does NOT keep it alive — if the tensor is garbage
    collected, torch's allocator recycles the block and the mesh points
    become garbage (raycast then illegal-memory-accesses nondeterministically
    once the memory is reused). The old env kept the same latent hazard. Use
    warp-owned copies (wp.from_numpy of a host array with device=) so the
    mesh owns its memory outright.
    """
    vertices, triangles = build_trimesh(include_obstacles)
    mesh = wp.Mesh(
        points=wp.from_numpy(vertices, dtype=wp.vec3, device=DEVICE),
        indices=wp.from_numpy(triangles.flatten(), dtype=wp.int32,
                              device=DEVICE),
    )
    mesh_ids = wp.array([mesh.id], dtype=wp.uint64)
    return mesh, mesh_ids, len(vertices), len(triangles)


def make_sensor(nworld: int, mesh_ids, enable_noise: bool = SENSOR_NOISE):
    """LidarSensor with the REAL task config (see header for provenance).

    Returns (sensor, sensor_pos_tensor, sensor_quat_tensor): keep the tensor
    handles alive — the sensor's warp arrays zero-copy alias them, and every
    tick we write the fresh pose into these very tensors (LidarSensor holds
    them as env['sensor_pos_tensor']/env['sensor_quat_tensor'], exposed on
    the instance as lidar_positions_tensor / lidar_quat_tensor).
    """
    sensor_pos_tensor = torch.zeros(nworld, 3, dtype=torch.float32, device=DEVICE)
    sensor_quat_tensor = torch.zeros(nworld, 4, dtype=torch.float32, device=DEVICE)
    lidar_cfg = LidarConfig(
        sensor_type=LidarType.AIRY,
        dt=float(CONTROL_DT),               # old env passes its control dt
        update_frequency=LIDAR_UPDATE_HZ,
        max_range=LIDAR_MAX_RANGE,
        min_range=LIDAR_MIN_RANGE,
        num_sensors=1,
        # pattern lidar overrides these two internally (kept for print parity)
        horizontal_line_num=AIRY_N_AZIMUTH,
        vertical_line_num=AIRY_N_ELEVATION,
        airy_horizontal_resolution_deg=AIRY_H_RES_DEG,
        return_pointcloud=True,
        pointcloud_in_world_frame=False,    # sensor frame (pixel = dist*dir)
        randomize_placement=False,          # MUST be explicit False (warp path)
        enable_sensor_noise=enable_noise,
        pixel_std_dev_multiplier=PIXEL_STD_DEV_MULTIPLIER,
        pixel_dropout_prob=PIXEL_DROPOUT_PROB,
    )
    lidar_env = {
        "device": DEVICE,
        "num_envs": nworld,
        "num_sensors": 1,
        "sensor_pos_tensor": sensor_pos_tensor,
        "sensor_quat_tensor": sensor_quat_tensor,
        "mesh_ids": mesh_ids,
    }
    sensor = LidarSensor(lidar_env, None, lidar_cfg, num_sensors=1,
                         device=DEVICE)
    assert sensor.num_rays == AIRY_N_AZIMUTH * AIRY_N_ELEVATION, \
        f"Airy ray count {sensor.num_rays} != 86400"
    return sensor, sensor_pos_tensor, sensor_quat_tensor


# ═══════════════════════════════════════════════════════════════════════════
# VRAM helpers
# ═══════════════════════════════════════════════════════════════════════════
def smi_process_mb() -> float | None:
    """This process's VRAM usage in MB via nvidia-smi (None if not listed)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        ).stdout
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2 and int(parts[0]) == os.getpid():
                return float(parts[1])
    except Exception:
        pass
    return None


def pct(xs, q: float) -> float:
    s = sorted(xs)
    if not s:
        return float("nan")
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


# ═══════════════════════════════════════════════════════════════════════════
# Single-nworld run (worker, also used --inprocess)
# ═══════════════════════════════════════════════════════════════════════════
def run_single(nworld: int, ticks: int, enable_noise: bool = SENSOR_NOISE) -> dict:
    res = {"nworld": nworld, "ticks": ticks, "status": "ok", "error": None,
           "noise": enable_noise}
    torch.zeros(1, device=DEVICE)   # force CUDA context creation
    vram_baseline = smi_process_mb()
    torch.cuda.reset_peak_memory_stats()

    # ---- physics side ----------------------------------------------------
    model = mujoco.MjModel.from_xml_string(build_mjcf())
    mjd = mujoco.MjData(model)
    mujoco.mj_forward(model, mjd)   # seeds ncon/nefc sizing heuristics
    mjw_model = mjwarp.put_model(model)
    mjw_data = mjwarp.put_data(model, mjd, nworld=nworld)
    robot_body = model.body("robot_proxy").id
    print(f"[spike] mujoco-warp up: nworld={nworld} nbody={model.nbody} "
          f"robot_body_id={robot_body}")

    # ---- LiDAR side ------------------------------------------------------
    mesh, mesh_ids, n_verts, n_tris = make_lidar_mesh(include_obstacles=True)
    plane_mesh, plane_mesh_ids, _, _ = make_lidar_mesh(include_obstacles=False)
    print(f"[spike] lidar mesh: {n_verts} verts / {n_tris} tris "
          f"(+plane-only diag mesh)")
    sensor, sensor_pos_t, sensor_quat_t = make_sensor(
        nworld, mesh_ids, enable_noise=enable_noise)

    mount_t = torch.tensor(SENSOR_OFFSET_POS, dtype=torch.float32,
                           device=DEVICE).view(1, 3)
    mount_q = quat_from_euler_xyz(*SENSOR_OFFSET_RPY)   # (1, 4) xyzw

    # ---- loop -------------------------------------------------------------
    hit_thr = LIDAR_MAX_RANGE - 6.0 * PIXEL_STD_DEV_MULTIPLIER * LIDAR_MAX_RANGE
    first_tick_ms = None
    lidar_ms, physics_ms, hit_rates, hit_rates_thr = [], [], [], []
    base_pos_first = base_pos_last = None
    n_rays = sensor.num_rays

    for tick in range(WARMUP_TICKS + ticks):
        # -- physics batch: 5 control steps x 4 mjwarp.step (50 Hz cadence);
        #    no actuators in the scene, control boundaries are structural only
        t_phys = time.perf_counter()
        for _ in range(CONTROL_STEPS_PER_TICK):
            for _ in range(STEPS_PER_CONTROL):
                mjwarp.step(mjw_model, mjw_data)
        wp.synchronize_device()          # SYNC 1: mujoco-warp -> host/torch
        physics_ms.append((time.perf_counter() - t_phys) * 1e3)

        # -- pose flow: mujoco-warp (wxyz) -> torch xyzw -> sensor tensors
        base_pos = wp.to_torch(mjw_data.xpos)[:, robot_body, :]        # (W,3)
        base_quat_wxyz = wp.to_torch(mjw_data.xquat)[:, robot_body, :]  # (W,4)
        base_quat = base_quat_wxyz[:, [1, 2, 3, 0]]                    # -> xyzw
        if tick == 0:
            base_pos_first = base_pos.clone()
        base_pos_last = base_pos
        # identical formula to old env _update_lidar_history
        sensor_quat_t.copy_(quat_mul(base_quat, mount_q.expand(nworld, 4)))
        sensor_pos_t.copy_(base_pos + quat_apply(
            base_quat, mount_t.expand(nworld, 3)))
        torch.cuda.synchronize()         # SYNC 2: torch writes -> warp kernel

        # -- LiDAR tick (OUTSIDE any graph-captured region) -----------------
        t0 = time.perf_counter()
        pixels, dists = sensor.update()
        wp.synchronize_device()          # SYNC 3: raycast outputs -> torch
        dt_ms = (time.perf_counter() - t0) * 1e3
        if tick == 0:
            first_tick_ms = dt_ms
        if tick >= WARMUP_TICKS:
            lidar_ms.append(dt_ms)

        if os.environ.get("SPIKE_DEBUG"):
            print(f"[dbg] tick={tick} z0={base_pos[0, 2].item():.4f} "
                  f"qz={base_quat[0, :].cpu().numpy()} "
                  f"spos_z={sensor_pos_t[0, 2].item():.4f}")

        # -- checks (outside the timing window) -----------------------------
        p = pixels.view(nworld, -1, 3)
        d = dists.view(nworld, -1)
        assert torch.isfinite(p).all().item(), "non-finite points"
        assert torch.isfinite(d).all().item(), "non-finite dists"
        assert (d >= 0.0).all().item(), "negative dist"
        # sensor-side noise bound: miss rays carry Gaussian(15, 0.02*15) noise
        # (sensor does NOT clamp; the env-side layer does r <- min(r, 15)).
        # Over 10^7-10^8 samples the max reaches ~+5.5 sigma -> bound 8 sigma.
        assert (d <= LIDAR_MAX_RANGE * (1.0 + 8.0 * PIXEL_STD_DEV_MULTIPLIER)
                ).all().item(), "dist beyond 8-sigma noise bound of far plane"
        # sensor-frame contract: pixel == dist * unit_dir  =>  |p| == d
        err = (torch.linalg.norm(p, dim=-1) - d).abs().max().item()
        assert err < 2e-3, f"|p|-d contract violated: {err}"
        rate = (d < LIDAR_MAX_RANGE - 1e-4).float().mean().item()
        rate_thr = (d < hit_thr).float().mean().item()
        hit_rates.append(rate)
        hit_rates_thr.append(rate_thr)
        assert 0.0 < rate < 1.0, f"raw hit rate not in (0,1): {rate}"
        assert 0.02 < rate_thr < 0.98, \
            f"conservative hit rate insane: {rate_thr}"
        torch.cuda.synchronize()         # raises if any async CUDA error

    # -- robot actually moved (pose flow is live, not frozen) ---------------
    move = (base_pos_last - base_pos_first).norm(dim=-1)
    robot_moved = bool((move > 0.05).float().mean().item() > 0.5)
    assert robot_moved, "robot proxy never moved — pose flow suspect"

    # -- obstacle visibility proof: same rays, plane-only mesh --------------
    wp.synchronize_device()
    d_boxes = dists.view(nworld, -1).clone()
    sensor.mesh_ids = plane_mesh_ids     # direct launch path -> swap is seen
    _, d_plane = sensor.update()
    wp.synchronize_device()
    sensor.mesh_ids = mesh_ids           # restore
    d_plane = d_plane.view(nworld, -1)
    obstacle_blocked = int((d_boxes < d_plane - 0.35).sum().item())
    assert obstacle_blocked > 100, \
        f"obstacles produced only {obstacle_blocked} blocked rays"
    torch.cuda.synchronize()             # final stream-health check

    vram_end = smi_process_mb()
    res.update({
        "status": "ok",
        "n_rays": n_rays,
        "first_tick_ms": round(first_tick_ms, 3),
        "lidar_ms_mean": round(sum(lidar_ms) / len(lidar_ms), 3),
        "lidar_ms_p50": round(pct(lidar_ms, 0.50), 3),
        "lidar_ms_p95": round(pct(lidar_ms, 0.95), 3),
        "physics_ms_mean": round(sum(physics_ms[WARMUP_TICKS:])
                                 / (len(physics_ms) - WARMUP_TICKS), 3),
        "hit_rate_raw_mean": round(sum(hit_rates) / len(hit_rates), 5),
        "hit_rate_raw_min": round(min(hit_rates), 5),
        "hit_rate_raw_max": round(max(hit_rates), 5),
        "hit_rate_thr_mean": round(sum(hit_rates_thr) / len(hit_rates_thr), 5),
        "hit_thr_m": round(hit_thr, 3),
        "torch_peak_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1),
        "smi_vram_baseline_mb": vram_baseline,
        "smi_vram_end_mb": vram_end,
        "robot_moved": robot_moved,
        "obstacle_blocked_rays": obstacle_blocked,
        "sim_time_s": round((WARMUP_TICKS + ticks) * CONTROL_STEPS_PER_TICK
                            * STEPS_PER_CONTROL * PHYS_DT, 2),
        "cuda_errors": "none (all torch.cuda.synchronize() clean)",
    })
    print(f"[spike] nworld={nworld} done: lidar "
          f"{res['lidar_ms_mean']:.1f} ms/tick (p95 {res['lidar_ms_p95']:.1f}), "
          f"hit_rate~{res['hit_rate_thr_mean']:.3f}, "
          f"torch_peak {res['torch_peak_mb']} MB, smi {vram_end} MB")
    return res


def run_single_guarded(nworld: int, ticks: int,
                       enable_noise: bool = SENSOR_NOISE) -> dict:
    try:
        return run_single(nworld, ticks, enable_noise=enable_noise)
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        return {"nworld": nworld, "status": "oom",
                "error": f"torch OOM: {str(e)[:300]}"}
    except Exception as e:  # noqa: BLE001 — worker must report, not crash
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return {"nworld": nworld, "status": "error",
                "error": f"{type(e).__name__}: {str(e)[:300]}"}


# ═══════════════════════════════════════════════════════════════════════════
# CUDA graph capture probe (driver 535 / CUDA 12.2) — isolated subprocess
# ═══════════════════════════════════════════════════════════════════════════
def graph_probe(nworld: int) -> dict:
    out = {
        "probe": "cuda_graph_capture",
        "driver_note": "driver 535 = CUDA 12.2 (< 12.4)",
        "is_conditional_graph_supported": bool(wp.is_conditional_graph_supported()),
    }
    model = mujoco.MjModel.from_xml_string(build_mjcf())
    mjd = mujoco.MjData(model)
    mujoco.mj_forward(model, mjd)
    m = mjwarp.put_model(model)
    d = mjwarp.put_data(model, mjd, nworld=nworld)
    robot_body = model.body("robot_proxy").id

    # (a) capture mujoco-warp step. Warm up first: mid-capture JIT module
    # loads are rejected by the driver (observed: "operation not permitted
    # when stream is capturing" on wp_cuda_load_module), so every kernel the
    # step pipeline needs must be loaded BEFORE capture_begin.
    try:
        for _ in range(2):
            mjwarp.step(m, d)
        wp.synchronize_device()
        with wp.ScopedCapture() as cap:
            for _ in range(4):
                mjwarp.step(m, d)
        q0 = wp.to_torch(d.qpos).clone()
        wp.synchronize_device()
        wp.capture_launch(cap.graph)
        wp.capture_launch(cap.graph)
        wp.synchronize_device()
        q1 = wp.to_torch(d.qpos)
        changed = bool((q1 - q0).abs().max().item() > 1e-9)
        out["mjwarp_step_capture"] = f"ok (2 replays, state advanced: {changed})"
    except Exception as e:  # noqa: BLE001
        out["mjwarp_step_capture"] = f"failed: {type(e).__name__}: {str(e)[:200]}"

    # (a2) retry with graph_conditional=False (mujoco-warp's documented
    # fallback for CUDA < 12.4: replaces conditional nodes with an
    # unconditional launch that self-gates on the device side)
    try:
        m.opt.graph_conditional = False
        with wp.ScopedCapture() as cap2:
            mjwarp.step(m, d)
            mjwarp.step(m, d)
        q0 = wp.to_torch(d.qpos).clone()
        wp.synchronize_device()
        wp.capture_launch(cap2.graph)
        wp.capture_launch(cap2.graph)
        wp.synchronize_device()
        q1 = wp.to_torch(d.qpos)
        changed = bool((q1 - q0).abs().max().item() > 1e-9)
        out["mjwarp_step_capture_nocond"] = \
            f"ok (2 replays, state advanced: {changed})"
    except Exception as e:  # noqa: BLE001
        out["mjwarp_step_capture_nocond"] = \
            f"failed: {type(e).__name__}: {str(e)[:200]}"

    # (b) capture the LidarSensor kernel itself (NOT used in the main loop —
    #     the spike deliberately runs LiDAR updates outside any graph)
    try:
        mesh_ids = make_lidar_mesh(True)[1]
        sensor, pos_t, quat_t = make_sensor(nworld, mesh_ids)
        n = nworld
        pos_t.copy_(torch.tensor(
            [[0.7, 0.0, 0.45]] * n, dtype=torch.float32, device=DEVICE))
        quat_t.copy_(quat_from_euler_xyz(*SENSOR_OFFSET_RPY).expand(n, 4))
        torch.cuda.synchronize()
        sensor.capture()                     # wp.capture_begin/end path
        wp.synchronize_device()
        _, dists = sensor.update()           # wp.capture_launch path
        wp.synchronize_device()
        hit = (dists.view(n, -1) < LIDAR_MAX_RANGE - 1e-4).float().mean().item()
        out["lidar_kernel_capture"] = f"ok (hit rate under graph: {hit:.3f})"
    except Exception as e:  # noqa: BLE001
        out["lidar_kernel_capture"] = f"failed: {type(e).__name__}: {str(e)[:200]}"

    print("SPIKE_GRAPH_PROBE " + json.dumps(out))
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════════════════════
def run_worker(args) -> int:
    if args.graph_probe:
        graph_probe(args.graph_probe)
        return 0
    if args.run_single:
        res = run_single_guarded(args.run_single, args.ticks,
                                 enable_noise=not args.noise_off)
        print("SPIKE_SINGLE_RESULT " + json.dumps(res))
        return 0 if res["status"] == "ok" else 1
    raise SystemExit("worker mode needs --run-single or --graph-probe")


def spawn_worker(flag: str, value: int, ticks: int, timeout: int = 900):
    """Run a worker in a subprocess so OOM/crash is contained cleanly."""
    cmd = [sys.executable, os.path.abspath(__file__), flag, str(value),
           "--ticks", str(ticks)]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        out, err, rc = p.stdout, p.stderr, p.returncode
    except subprocess.TimeoutExpired:
        return None, {"status": "timeout"}, f"timeout after {timeout}s"
    took = time.time() - t0
    result = None
    line = None
    for ln in out.splitlines():
        if ln.startswith("SPIKE_SINGLE_RESULT "):
            line = ln
    if line:
        try:
            result = json.loads(line[len("SPIKE_SINGLE_RESULT "):])
        except json.JSONDecodeError:
            result = None
    if result is None:
        tail = (out + "\n" + err).strip().splitlines()[-8:]
        result = {"status": f"crashed (rc={rc})", "tail": tail}
    return result, None, took


def fmt_mb(x) -> str:
    return "?" if x is None else f"{x:.0f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nworlds", default="64,512",
                    help="comma-separated nworld list for the gated runs")
    ap.add_argument("--ladder", default="4096,2048,1024",
                    help="descending nworld ladder tried after the gated runs")
    ap.add_argument("--ticks", type=int, default=50, help="timed LiDAR ticks")
    ap.add_argument("--run-single", type=int, default=None, metavar="N",
                    help="worker: single nworld run, prints JSON result")
    ap.add_argument("--graph-probe", type=int, default=None, metavar="N",
                    help="worker: CUDA graph capture probe")
    ap.add_argument("--inprocess", action="store_true",
                    help="run the gated nworlds in-process (debug; VRAM "
                         "numbers then include leftovers of prior runs)")
    ap.add_argument("--no-ladder", action="store_true")
    ap.add_argument("--noise-off", action="store_true",
                    help="worker: disable sensor noise (attribution runs; "
                         "the gated runs always use the real task config)")
    ap.add_argument("--no-graph-probe", action="store_true")
    args = ap.parse_args()
    if args.run_single is not None or args.graph_probe is not None:
        return run_worker(args)

    _self_test_quat_math()
    nworlds = [int(x) for x in args.nworlds.split(",") if x.strip()]
    print(f"[spike] GPU: {torch.cuda.get_device_name(0)}, "
          f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB, "
          f"torch {torch.__version__}, warp {wp.__version__}, "
          f"mujoco {mujoco.__version__}")

    results = {}
    if args.inprocess:
        for n in nworlds:
            results[n] = run_single_guarded(n, args.ticks)
    else:
        for n in nworlds:
            print(f"\n[spike] === nworld={n} (subprocess) ===")
            res, _, took = spawn_worker("--run-single", n, args.ticks)
            res["wall_s"] = round(took, 1) if isinstance(took, float) else None
            results[n] = res
            print(f"[spike] nworld={n}: {res['status']} ({took:.0f} s wall)")

    # ---- ladder: 4096 if it fits, else largest that fits ------------------
    ladder_report = {"attempts": [], "max_feasible": None}
    if not args.no_ladder:
        for n in [int(x) for x in args.ladder.split(",") if x.strip()]:
            print(f"\n[spike] === ladder attempt nworld={n} ===")
            res, _, took = spawn_worker("--run-single", n, args.ticks)
            status = res.get("status")
            ladder_report["attempts"].append({
                "nworld": n, "status": status,
                "error": res.get("error"),
                "torch_peak_mb": res.get("torch_peak_mb"),
            })
            print(f"[spike] ladder nworld={n}: {status} "
                  f"({res.get('error') or ''})")
            if status == "ok":
                res["wall_s"] = round(took, 1)
                results[n] = res
                ladder_report["max_feasible"] = n
                break

    # ---- graph capture probe ---------------------------------------------
    probe = None
    if not args.no_graph_probe:
        print("\n[spike] === CUDA graph capture probe (subprocess) ===")
        cmd = [sys.executable, os.path.abspath(__file__), "--graph-probe",
               str(nworlds[0])]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            for ln in p.stdout.splitlines():
                if ln.startswith("SPIKE_GRAPH_PROBE "):
                    probe = json.loads(ln[len("SPIKE_GRAPH_PROBE "):])
            if probe is None:
                tail = (p.stdout + p.stderr).strip().splitlines()[-8:]
                probe = {"status": f"crashed (rc={p.returncode})", "tail": tail}
        except subprocess.TimeoutExpired:
            probe = {"status": "timeout"}
        print(f"[spike] graph probe: {json.dumps(probe)}")

    # ---- report & gates ---------------------------------------------------
    print("\n" + "=" * 78)
    print("S2 SPIKE REPORT — LidarSensor on mujoco-warp")
    print("=" * 78)
    hdr = (f"{'nworld':>7} {'status':>6} {'first_ms':>9} {'ms/tick':>8} "
           f"{'p95_ms':>7} {'phys_ms':>8} {'hit_raw':>8} {'hit_thr':>8} "
           f"{'torchMB':>8} {'smiMB':>7}")
    print(hdr)
    for n in sorted(results):
        r = results[n]
        print(f"{n:>7} {r['status']:>6} "
              f"{r.get('first_tick_ms', -1):>9.1f} "
              f"{r.get('lidar_ms_mean', float('nan')):>8.2f} "
              f"{r.get('lidar_ms_p95', float('nan')):>7.2f} "
              f"{r.get('physics_ms_mean', float('nan')):>8.2f} "
              f"{r.get('hit_rate_raw_mean', float('nan')):>8.4f} "
              f"{r.get('hit_rate_thr_mean', float('nan')):>8.4f} "
              f"{r.get('torch_peak_mb', -1):>8.0f} "
              f"{fmt_mb(r.get('smi_vram_end_mb')):>7}")
    for n in sorted(results):
        r = results[n]
        print(f"  detail nworld={n}: {json.dumps(r)}")

    # memory slope 64 -> 512
    slope = None
    if 64 in results and 512 in results and all(
            results[n].get("status") == "ok" for n in (64, 512)):
        d_t = results[512]["torch_peak_mb"] - results[64]["torch_peak_mb"]
        slope = {"torch_mb_per_world": round(d_t / (512 - 64), 3)}
        a, b = results[64].get("smi_vram_end_mb"), results[512].get("smi_vram_end_mb")
        if a is not None and b is not None:
            slope["smi_mb_per_world"] = round((b - a) / (512 - 64), 3)
        print(f"[spike] memory slope 64->512: {json.dumps(slope)}")
    print(f"[spike] ladder: {json.dumps(ladder_report)}")

    # ---- gates -------------------------------------------------------------
    gates = {}

    def st(n):
        return results.get(n, {}).get("status")

    gates["G1_no_crash_or_sync_error_64_512"] = st(64) == "ok" and st(512) == "ok"
    ok64, ok512 = results.get(64, {}), results.get(512, {})
    gates["G2_hits_and_finite_64_512"] = all(
        0.0 < r.get("hit_rate_raw_mean", 0) < 1.0
        and 0.02 < r.get("hit_rate_thr_mean", 0) < 0.98
        and r.get("obstacle_blocked_rays", 0) > 100
        for r in (ok64, ok512)
    ) if gates["G1_no_crash_or_sync_error_64_512"] else False
    gates["G3_timing_table_present"] = st(64) == "ok" and st(512) == "ok" \
        and "lidar_ms_mean" in ok512
    g4 = None
    if st(512) == "ok":
        smi = ok512.get("smi_vram_end_mb")
        g4 = ok512["torch_peak_mb"] < 2500 and (smi is None or smi < 2500)
    gates["G4_vram_512_below_2p5GB"] = g4
    g5 = "attempted" if ladder_report["attempts"] else "skipped"
    gates["G5_4096_attempt"] = {
        "result": g5,
        "max_feasible": ladder_report["max_feasible"],
        "attempts": ladder_report["attempts"],
    }
    print("\n---- GATES ----")
    all_ok = True
    for k, v in gates.items():
        ok = bool(v) if not isinstance(v, dict) else True
        all_ok &= ok
        print(f"  {k}: {json.dumps(v) if isinstance(v, dict) else v}")
    if probe:
        print(f"\nCUDA-graph behaviour (driver 535): {json.dumps(probe, indent=2)}")
    print(f"\n[spike] ALL GATES {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
