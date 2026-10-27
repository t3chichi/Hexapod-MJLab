"""Pure geometry shared by task visualization and play display."""

from __future__ import annotations

import math

import numpy as np

from hexapod_mjlab.common.visualization import (
    CONE_TILT_DEG,
    H_REF_M,
    HEX_VERTEX_ORDER,
    cone_axis_base,
    cone_wireframe,
    hex_vertices,
    quat_apply_xyzw,
    quat_wxyz_to_matrix,
)

STARTUP_ENVELOPE = (0.59, 0.69, 0.59, 0.89, -0.89, 0.0)

CONE_THETA_DEG = 28.0

DEFAULT_MAX_RANGE_M = 15.0


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
