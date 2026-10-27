"""Shared display-only geometry and viewer primitives for formal tasks."""

from __future__ import annotations

import math
import threading

import numpy as np
import torch

H_REF_M = 0.46

HEX_VERTEX_ORDER = ("B", "D", "F", "E", "C", "A")

CONE_TILT_DEG = 56.0


def hex_vertices(envelope) -> dict[str, tuple[float, float]]:
    """Predicted horizontal hexagon, EA2 order B,D,F,E,C,A.

    ``envelope`` = Ê = [fw, mw, bw, front_x, rear_x(negative), Δz]; vertices
    B=(front,fw), D=(0,mw), F=(back,bw), E=(back,-bw), C=(0,-mw),
    A=(front,-fw) — the HORIZONTAL footprint prediction only (the hexagon is
    closed by returning the ordered labels; Δz is NOT consumed here).
    """
    fw, mw, bw, front, rear, _dz = (float(v) for v in envelope)
    recipe = ((front, fw), (0.0, mw), (rear, bw), (rear, -bw), (0.0, -mw),
              (front, -fw))
    return dict(zip(HEX_VERTEX_ORDER, recipe))


def quat_apply_xyzw(q, v) -> torch.Tensor:
    """Apply an xyzw quaternion to vectors (same cross form as the scanner's
    BASE→sensor mount math; the mount quaternion from the corrected site is
    unit).

    dtype-uniform by construction: both operands are promoted to float64
    BEFORE the cross products — the CPU attempt-1 failure was exactly a
    float64 quaternion crossed with a float32 vector. Pure geometry helper;
    never touches network inputs."""
    q64 = torch.as_tensor(q, dtype=torch.float64).reshape(-1, 4)
    v64 = torch.as_tensor(v, dtype=torch.float64).reshape(-1, 3)
    qv, qw = q64[..., :3], q64[..., 3:4]
    t = 2.0 * torch.linalg.cross(qv, v64, dim=-1)
    return v64 + qw * t + torch.linalg.cross(qv, t, dim=-1)


def cone_axis_base(mount_quat_xyzw, tilt_deg: float = CONE_TILT_DEG) -> np.ndarray:
    """Grouping-cone axis in BASE coordinates — the EXACT scanner predicate:
    the corrected mount's z axis tilted by δ about y, then normalized.

    Accepts any array-like (nested (1,4) archive, flat list, tensor); the
    quaternion is normalized to unit first (same predicate, dtype-safe)."""
    quat = np.asarray(mount_quat_xyzw, dtype=np.float64).reshape(-1)
    if quat.size != 4:
        raise ValueError(
            f"mount quaternion must have 4 components, got shape "
            f"{np.shape(mount_quat_xyzw)}")
    qnorm = float(np.linalg.norm(quat))
    if not math.isfinite(qnorm) or qnorm <= 0.0:
        raise ValueError(
            f"mount quaternion is degenerate: {quat.tolist()}")
    quat = quat / qnorm
    dome = quat_apply_xyzw(
        torch.from_numpy(quat).view(1, 4), torch.tensor([[0.0, 0.0, 1.0]])
    )[0].numpy()
    delta = math.radians(tilt_deg)
    cd, sd = math.cos(delta), math.sin(delta)
    ax, ay, az = (float(v) for v in dome)
    axis = np.array([cd * ax + sd * az, ay, -sd * ax + cd * az])
    return axis / np.linalg.norm(axis)


def cone_wireframe(origin, axis, half_angle_deg: float, length: float,
                   n_rim: int = 24) -> np.ndarray:
    """Cone wireframe as line segments (N, 2, 3): generators + rim circle.

    ``length`` is the DISPLAY LENGTH ALONG THE AXIS: every rim point is
    ``origin + R @ [L·tanθ·cosφ, L·tanθ·sinφ, L]`` with R the rotation that
    maps +z ONTO the axis — so on ANY axis (arbitrary tilt, reversed) each
    rim point's projection on the axis is exactly L, its radial radius is
    L·tanθ, and every generator direction d satisfies dot(d, axis)/||d|| =
    cos(half_angle) (the actual scanner uses cos(angle)>=sin(θ), so
    half_angle=90°−θ; real scan inputs are
    untouched by this display geometry; the apex sits at ``origin``)."""
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    axis_norm = float(np.linalg.norm(axis))
    if not math.isfinite(axis_norm) or axis_norm <= 0.0:
        raise ValueError(f"cone axis is degenerate: {axis.tolist()}")
    axis = axis / axis_norm
    if not (math.isfinite(float(length)) and float(length) > 0.0):
        raise ValueError(f"length must be finite and positive, got {length!r}")
    a = np.array([0.0, 0.0, 1.0])
    c = float(np.dot(a, axis))
    v = np.cross(a, axis)
    if float(np.linalg.norm(v)) < 1e-12:
        # axis parallel (+z → identity) or ANTI-parallel (π about x) to +z.
        rmat = np.eye(3) if c > 0.0 else np.diag([1.0, -1.0, -1.0])
    else:
        # Rodrigues mapping +z ONTO the axis. The cross vector v stays
        # UNNORMALIZED: ‖v‖ = sinα, so I + [v]× + [v]×²/(1+c) IS the full
        # rotation — normalizing v first drops the sinα factor of the
        # generator term and produces a wrong (tilted, non-orthogonal) R.
        skew = np.array(
            [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
        rmat = np.eye(3) + skew + skew @ skew * (1.0 / (1.0 + c))
    tan = math.tan(math.radians(half_angle_deg))
    rim = []
    for i in range(n_rim):
        phi = 2.0 * math.pi * i / n_rim
        # d is deliberately NOT normalized: its z component is exactly 1, so
        # (R @ d)·L lands at axial distance L with radial radius L·tanθ.
        d = np.array([math.cos(phi) * tan, math.sin(phi) * tan, 1.0])
        rim.append(origin + (rmat @ d) * float(length))
    segs = []
    for i in range(n_rim):
        segs.append((origin, rim[i]))           # generator (apex → rim)
        segs.append((rim[i], rim[(i + 1) % n_rim]))  # rim edge
    return np.asarray(segs, dtype=np.float64)


def quat_wxyz_to_matrix(q) -> np.ndarray:
    """wxyz quaternion → 3×3 rotation matrix (pure)."""
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def robot_meshes_at_qpos(mj_model, qpos, *, mj_data=None) -> list[dict]:
    """World vertices/faces of every MESH geom at the given qpos.

    Host ``mj_forward`` on the live compiled template model — kinematics for
    display ONLY (the same approach as the accepted M3 replay viewer's
    ``robot_meshes``); the sim's own state is never touched. ``mj_data`` may
    be a reused MjData (viewer cache); a fresh one is created when omitted.
    """
    import mujoco

    q = np.asarray(qpos, dtype=np.float64).reshape(-1)
    if int(q.size) != int(mj_model.nq):
        raise ValueError(
            f"display qpos size {int(q.size)} != compiled model nq "
            f"{int(mj_model.nq)}"
        )
    data = mj_data if mj_data is not None else mujoco.MjData(mj_model)
    data.qpos[:] = q
    mujoco.mj_forward(mj_model, data)
    out = []
    for gid in range(mj_model.ngeom):
        if int(mj_model.geom_type[gid]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        mid = int(mj_model.geom_dataid[gid])
        if mid < 0:
            continue
        vs = int(mj_model.mesh_vertadr[mid])
        vc = int(mj_model.mesh_vertnum[mid])
        fs = int(mj_model.mesh_faceadr[mid])
        fc = int(mj_model.mesh_facenum[mid])
        verts = np.asarray(mj_model.mesh_vert[vs : vs + vc], dtype=np.float64)
        faces = np.asarray(mj_model.mesh_face[fs : fs + fc], dtype=np.int64)
        r = np.asarray(data.geom_xmat[gid], dtype=np.float64).reshape(3, 3)
        t = np.asarray(data.geom_xpos[gid], dtype=np.float64)
        name = (
            mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom{gid}"
        )
        body = int(mj_model.geom_bodyid[gid])
        body_name = (
            mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
        )
        out.append(
            {
                "geom": name,
                "body": body_name,
                "vertices": verts @ r.T + t,
                "faces": faces,
            }
        )
    return out


class PendingControls:
    """Thread-safe intent mailbox: GUI threads only SUBMIT, the single owner
    loop TAKES. No runtime/env/network/scanner call ever happens from a GUI
    callback — the owner validates every intent before acting on it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._command = None          # (vx, vy, wz) raw slider values
        self._zero_command = False
        self._pause_toggle = False    # owner toggles its current state
        self._step = False
        self._reset = False
        self._exit = False
        self._rerender = False
        self._pick = None             # ("near"|"far", index)
        self._env_row = None

    def submit_command(self, vx: float, vy: float, omega: float) -> None:
        with self._lock:
            self._command = (float(vx), float(vy), float(omega))

    def submit_zero_command(self) -> None:
        with self._lock:
            self._zero_command = True

    def submit_pause_toggle(self) -> None:
        with self._lock:
            self._pause_toggle = not self._pause_toggle

    def submit_step(self) -> None:
        with self._lock:
            self._step = True

    def submit_reset(self) -> None:
        with self._lock:
            self._reset = True

    def submit_exit(self) -> None:
        with self._lock:
            self._exit = True

    def submit_rerender(self) -> None:
        with self._lock:
            self._rerender = True

    def submit_pick(self, kind: str, index: int) -> None:
        with self._lock:
            self._pick = (str(kind), int(index))

    def submit_env_row(self, row: int) -> None:
        with self._lock:
            self._env_row = int(row)

    def take(self) -> dict:
        with self._lock:
            pending = {
                "command": self._command,
                "zero_command": self._zero_command,
                "pause_toggle": self._pause_toggle,
                "step": self._step,
                "reset": self._reset,
                "exit": self._exit,
                "rerender": self._rerender,
                "pick": self._pick,
                "env_row": self._env_row,
            }
            self._command = None
            self._zero_command = False
            self._pause_toggle = False
            self._step = False
            self._reset = False
            self._rerender = False
            self._pick = None
            self._env_row = None
            # Exit is sticky until this owner-loop read, then consumed.
            self._exit = False
        return pending

    def exit_requested(self) -> bool:
        with self._lock:
            return bool(self._exit)
