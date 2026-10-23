"""Pure geometry shared by task visualization and play display."""

from __future__ import annotations

import math

import numpy as np
import torch

STARTUP_ENVELOPE = (0.59, 0.69, 0.59, 0.89, -0.89, 0.0)

H_REF_M = 0.46

HEX_VERTEX_ORDER = ("B", "D", "F", "E", "C", "A")

CONE_THETA_DEG = 28.0

CONE_TILT_DEG = 56.0

DEFAULT_MAX_RANGE_M = 15.0

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

def root_height_target_m(envelope, h_ref: float = H_REF_M) -> float:
    """Root-height TARGET h_init = h_ref + Δz (world vertical, ground ref).

    Δz is the ROOT HEIGHT target offset — NOT a ceiling, NOT a min/max
    collision height, and no prism/clearance certification is implied.
    """
    return float(h_ref) + float(envelope[5])

def root_height_marker_world(root_pos_w, ground_z_m: float, envelope,
                             h_ref: float = H_REF_M) -> np.ndarray:
    """World-frame center of the root-height target marker.

    Constructed IN WORLD COORDINATES first — (current root x, current root y,
    ground_z + h_ref + Δz) — so the height is referenced to the GROUND plane
    (the recorded plane z), never to the current root height. Δz is the root
    height target offset, NOT a ceiling."""
    p = np.asarray(root_pos_w, dtype=np.float64).reshape(-1)
    if p.size != 3:
        raise ValueError(f"root_pos_w must be 3 components, got {p.shape}")
    return np.array([float(p[0]), float(p[1]),
                     float(ground_z_m) + float(h_ref) + float(envelope[5])])

def world_to_base(p_w, root_pos_w, root_quat_wxyz) -> np.ndarray:
    """World → BASE (root-link acquisition frame) via the INVERSE current
    pose: Rᵀ(p − t), tilt INCLUDED — the exact inverse of the viewer's
    BASE→WORLD view transform; never a z-only subtraction."""
    r = quat_wxyz_to_matrix(root_quat_wxyz)
    t = np.asarray(root_pos_w, dtype=np.float64).reshape(3)
    return (np.asarray(p_w, dtype=np.float64).reshape(3) - t) @ r

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

def normalize_mount(pos_like, quat_like) -> tuple[np.ndarray, np.ndarray]:
    """Archived mount → plain (3,)/(4,) float64 vectors, quaternion unit.

    The driver's scans.pt archives the pipeline's (1,3)/(1,4) mount tensors
    as nested lists; flat lists/tuples/tensors are accepted equally. Shape or
    a non-finite/degenerate quaternion is a hard error (never a silent asset
    fallback)."""
    pos = np.asarray(pos_like, dtype=np.float64).reshape(-1)
    quat = np.asarray(quat_like, dtype=np.float64).reshape(-1)
    if pos.size != 3:
        raise ValueError(
            f"mount position must have 3 components, got shape "
            f"{np.shape(pos_like)}")
    if quat.size != 4:
        raise ValueError(
            f"mount quaternion must have 4 components, got shape "
            f"{np.shape(quat_like)}")
    n = float(np.linalg.norm(quat))
    if not math.isfinite(n) or n <= 0.0:
        raise ValueError(f"mount quaternion is degenerate: {quat.tolist()}")
    return pos, quat / n

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

def clip_segments_base_x(segments, *, x_max_m: float | None) -> np.ndarray:
    """Clip BASE-frame line segments against x <= x_max for B8 Play geometry.

    The cone wireframe is cosmetic and may be shorter than actual 15m range.
    This operation clips EACH generator/rim segment before transforming to
    the chosen BASE/WORLD display frame. It does not create a false cone rim
    at x_max and does not change the production point-cloud source.
    """
    lines = np.asarray(segments, dtype=np.float64).reshape(-1, 2, 3)
    if x_max_m is None:
        return lines.copy()
    xmax = float(x_max_m)
    if not math.isfinite(xmax):
        raise ValueError("x_max_m must be finite or None")
    clipped = []
    for p0, p1 in lines:
        a, b = p0.copy(), p1.copy()
        if a[0] > xmax and b[0] > xmax:
            continue
        if a[0] > xmax or b[0] > xmax:
            t = (xmax - a[0]) / (b[0] - a[0])
            boundary = a + t * (b - a)
            boundary[0] = xmax
            if a[0] > xmax:
                a = boundary
            else:
                b = boundary
        clipped.append((a, b))
    if not clipped:
        return np.empty((0, 2, 3), dtype=np.float64)
    return np.asarray(clipped, dtype=np.float64)


def rotation_matrix_to_wxyz(m: np.ndarray) -> tuple[float, float, float, float]:
    """Row-major 3×3 rotation → viser wxyz quaternion (Shepperd's method)."""
    m = np.asarray(m, dtype=np.float64).reshape(3, 3)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    n = math.sqrt(w * w + x * x + y * y + z * z)
    return w / n, x / n, y / n, z / n

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

