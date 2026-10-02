"""Hexagon envelope geometry helpers for ``envelope_adaptive_2``.

This module implements the geometric core described in README sections
2.2.3 and 2.8:

* six-vertex body-frame hexagon construction (legacy-compatible order),
* exact half-plane margin offset (not the legacy radial approximation),
* convex point-in-hex tests,
* grid collision ratio against a 2D occupancy map,
* 5-parameter -> 8-dimensional condition conversion reusing the shared
  ``apply_env_morphology_priors`` implementation.

The legacy rule-based implementation in ``envelope_adaptive/envelope_computer.py``
is intentionally not modified; this module only mirrors its vertex order and
uses the shared spider_envelop condition spec for prior derivation.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

from hexapod_mjlab.tasks.envelope_adaptive_2 import contracts as ea2c
from hexapod_mjlab.tasks.envelope_adaptive_2.config.condition import (
    EnvelopeConditionSpec,
    apply_env_morphology_priors,
)

# Contract vertex order: B, D, F, E, C, A.
_HEX_VERTEX_NAMES: Tuple[str, ...] = (
    "B",
    "D",
    "F",
    "E",
    "C",
    "A",
)


def compute_hex_vertices(
    front_width: Tensor,
    middle_width: Tensor,
    back_width: Tensor,
    forward_limit: Tensor,
    backward_limit: Tensor,
) -> Tensor:
    """Build the 6-vertex body-frame hexagon in legacy-compatible order.

    Vertex order is ``B, D, F, E, C, A`` (counter-clockwise for valid
    envelope parameters):

    .. code-block:: text

        B=( forward_limit,  front_width)
        D=(0,              middle_width)
        F=( backward_limit, back_width)
        E=( backward_limit, -back_width)
        C=(0,              -middle_width)
        A=( forward_limit,  -front_width)

    Args:
        front_width: Half-width at the forward edge, shape ``(...,)``.
        middle_width: Half-width at the middle edge, shape ``(...,)``.
        back_width: Half-width at the backward edge, shape ``(...,)``.
        forward_limit: Forward x limit, shape ``(...,)``.
        backward_limit: Backward x limit (negative), shape ``(...,)``.

    Returns:
        Vertices with shape ``(..., 6, 2)``. The last dimension is ``(x, y)``.
    """
    z = torch.zeros_like(forward_limit)
    B = torch.stack([forward_limit, front_width], dim=-1)
    D = torch.stack([z, middle_width], dim=-1)
    F = torch.stack([backward_limit, back_width], dim=-1)
    E_v = torch.stack([backward_limit, -back_width], dim=-1)
    C = torch.stack([z, -middle_width], dim=-1)
    A = torch.stack([forward_limit, -front_width], dim=-1)
    return torch.stack([B, D, F, E_v, C, A], dim=-2)


def offset_hexagon(vertices: Tensor, margin: float) -> Tensor:
    """Return the exact half-plane offset of a convex hexagon.

    Each edge is translated outward by ``margin`` along its unit outward
    normal; the offset vertices are the intersections of adjacent translated
    edge lines.  This is the precise offset used by the M1 collision
    geometry (README 2.8), unlike the legacy radial vertex scaling.
    Vertices with collinear adjacent edges (valid degenerate convex forms
    such as the rectangle-like minimum envelope) are supported: their offset
    positions are the original vertex translated by ``margin`` along the
    shared edge normal, which lies on the exact offset boundary.

    Args:
        vertices: Convex CCW polygon vertices, shape ``(..., V, 2)``.
        margin: Non-negative offset distance in the same units as vertices.

    Returns:
        Offset polygon vertices with the same shape as ``vertices``.
    """
    if margin < 0.0:
        raise ValueError("margin must be non-negative")
    if vertices.shape[-1] != 2:
        raise ValueError("vertices must have last dimension 2 (x, y)")

    v = vertices
    v_next = torch.roll(v, shifts=-1, dims=-2)  # next vertex for each edge
    edges = v_next - v  # (..., V, 2)
    lengths = torch.norm(edges, dim=-1, keepdim=True).clamp_min(1e-12)
    # Outward normal for CCW polygon: rotate edge clockwise by 90 degrees.
    normals = torch.stack([edges[..., 1], -edges[..., 0]], dim=-1) / lengths
    constants = (normals * v).sum(dim=-1) + margin  # (..., V)

    # Offset vertex ``i`` is the intersection of offset edge ``i-1`` and
    # offset edge ``i``.  Rolling the previous line into position i keeps
    # the returned vertex order identical to the input order.
    prev_normals = torch.roll(normals, shifts=1, dims=-2)
    prev_constants = torch.roll(constants, shifts=1, dims=-1)

    a1 = prev_normals[..., 0]
    b1 = prev_normals[..., 1]
    c1 = prev_constants
    a2 = normals[..., 0]
    b2 = normals[..., 1]
    c2 = constants

    det = a1 * b2 - a2 * b1  # (..., V)
    parallel = det.abs() < 1e-12

    # A vertex whose two adjacent edges are collinear (e.g. a rectangle
    # represented in the six-vertex B,D,F,E,C,A order) has no unique line
    # intersection.  Its offset position is the original vertex translated by
    # ``margin`` along the shared outward edge normal, which lies exactly on
    # the offset boundary.
    safe_det = torch.where(parallel, torch.ones_like(det), det)
    x = (c1 * b2 - c2 * b1) / safe_det
    y = (a1 * c2 - a2 * c1) / safe_det
    line_intersections = torch.stack([x, y], dim=-1)
    collinear_offset = v + margin * normals
    return torch.where(parallel.unsqueeze(-1), collinear_offset, line_intersections)


def point_in_hex(pts_xy: Tensor, vertices: Tensor) -> Tensor:
    """Test whether points lie inside (or on the boundary of) a convex hexagon.

    The hexagon is assumed to be convex and counter-clockwise.  Boundary
    points are considered inside (``>= 0`` on all edge cross products).

    Args:
        pts_xy: Points with shape ``(..., N, 2)``.
        vertices: Hexagon vertices with shape ``(..., 6, 2)`` or ``(6, 2)``.
            Leading batch dimensions broadcast with ``pts_xy``.

    Returns:
        Boolean mask with shape ``batch_shape + (N,)``, where ``batch_shape``
        is the broadcast of the leading batch dimensions of ``pts_xy`` and
        ``vertices``.
    """
    if pts_xy.shape[-1] < 2:
        raise ValueError("pts_xy must have at least 2 columns (x, y)")
    if vertices.shape[-1] != 2:
        raise ValueError("vertices must have last dimension 2 (x, y)")

    pts = pts_xy[..., :2]
    verts = vertices[..., :2]

    batch_shape = torch.broadcast_shapes(pts.shape[:-2], verts.shape[:-2])
    pts_b = pts.expand(batch_shape + pts.shape[-2:])
    verts_b = verts.expand(batch_shape + verts.shape[-2:])

    num_vertices = verts_b.shape[-2]
    next_idx = (torch.arange(num_vertices, device=verts_b.device) + 1) % num_vertices
    edges = verts_b[..., next_idx, :] - verts_b  # (batch, V, 2)

    rel = pts_b.unsqueeze(-2) - verts_b.unsqueeze(-3)  # (batch, N, V, 2)
    # cross2d(edge, rel) = edge.x * rel.y - edge.y * rel.x
    cross = (
        edges.unsqueeze(-3)[..., 0] * rel[..., 1]
        - edges.unsqueeze(-3)[..., 1] * rel[..., 0]
    )
    return (cross >= 0).all(dim=-1)


def hex_body_sample_points(params5: Tensor) -> Tensor:
    """Return 34 body-frame sample points for one or more hexagons.

    The sample set is:
      * 6 vertices (B, D, F, E, C, A),
      * 18 edge interpolation points (t = 0.25, 0.5, 0.75 on each edge),
      * 1 centre,
      * 9 interior points from a coarse 3x3 grid.

    Args:
        params5: Envelope parameters ``(..., 5)`` in the canonical order
            ``[front_width, middle_width, back_width, forward_limit,
            backward_limit]``.

    Returns:
        Tensor of shape ``(..., 34, 2)`` in the body frame (+x forward,
        +y left).
    """
    if params5.shape[-1] != 5:
        raise ValueError(f"Expected params5 width 5, got {params5.shape[-1]}")

    verts = compute_hex_vertices(
        params5[..., 0],
        params5[..., 1],
        params5[..., 2],
        params5[..., 3],
        params5[..., 4],
    )  # (..., 6, 2)
    next_v = torch.roll(verts, shifts=-1, dims=-2)

    edge_points = []
    for t in (0.25, 0.5, 0.75):
        edge_points.append(verts * (1.0 - t) + next_v * t)

    center = verts.mean(dim=-2, keepdim=True)  # (..., 1, 2)

    # Coarse 3x3 interior grid in the body frame.  The hexagon is not a
    # rectangle, so for each longitudinal position the lateral half-width is
    # linearly interpolated between back/middle/front widths.
    front_width = params5[..., 0]
    middle_width = params5[..., 1]
    back_width = params5[..., 2]
    forward_limit = params5[..., 3]
    backward_limit = params5[..., 4]

    sx = torch.tensor(
        [0.25, 0.5, 0.75], dtype=params5.dtype, device=params5.device
    )
    sy = torch.tensor(
        [-0.5, 0.0, 0.5], dtype=params5.dtype, device=params5.device
    )

    x = backward_limit.unsqueeze(-1) + sx * (
        forward_limit - backward_limit
    ).unsqueeze(-1)  # (..., 3)

    # half-width at each x (upper boundary of the convex hexagon)
    x_pos = x >= 0.0
    w_pos = middle_width.unsqueeze(-1) + (
        front_width - middle_width
    ).unsqueeze(-1) * (
        x / forward_limit.unsqueeze(-1).clamp_min(1e-6)
    )
    w_neg = back_width.unsqueeze(-1) + (
        middle_width - back_width
    ).unsqueeze(-1) * (
        (x - backward_limit.unsqueeze(-1))
        / (0.0 - backward_limit.unsqueeze(-1)).clamp_min(1e-6)
    )
    half_w = torch.where(x_pos, w_pos, w_neg).clamp_min(0.0)  # (..., 3)

    y = half_w.unsqueeze(-1) * sy  # (..., 3, 3)
    x_grid = x.unsqueeze(-1).expand(*x.shape, 3)  # (..., 3, 3)
    interior = torch.stack([x_grid, y], dim=-1).reshape(
        *params5.shape[:-1], 9, 2
    )

    points = torch.cat([verts] + edge_points + [center, interior], dim=-2)
    # Nudge boundary samples a tiny amount toward the centroid so floating-point
    # round-off cannot classify them as strictly outside the convex hexagon.
    return center + (points - center) * (1.0 - 1e-4)


def _bilinear_field_sample(dist: Tensor, wx: Tensor, wy: Tensor) -> Tensor:
    """Bilinearly interpolate the distance field at world positions.

    Cell ``k`` covers ``[world_min + k*res, world_min + (k+1)*res)`` and its
    value is attributed to the cell centre; the interpolation therefore uses
    the four surrounding cell centres.  Out-of-bounds positions read the
    clamped edge value.  Used by the continuous collision variant: unlike the
    nearest-cell read it is continuous across cell boundaries, so penalties
    scale with sub-cell penetration depth instead of jumping a full
    soft-margin step when a sample crosses a raster line.
    """
    res = ea2c.EA2_RESOLUTION_M
    wmin = ea2c.EA2_WORLD_MIN_XY
    height, width = dist.shape
    u = (wx - wmin) / res - 0.5
    v = (wy - wmin) / res - 0.5
    ix0 = torch.floor(u).long()
    iy0 = torch.floor(v).long()
    fx = (u - ix0.to(u.dtype)).clamp(0.0, 1.0)
    fy = (v - iy0.to(v.dtype)).clamp(0.0, 1.0)
    ix1 = (ix0 + 1).clamp(0, width - 1)
    iy1 = (iy0 + 1).clamp(0, height - 1)
    cix0 = ix0.clamp(0, width - 1)
    ciy0 = iy0.clamp(0, height - 1)
    d00 = dist[ciy0, cix0]
    d10 = dist[ciy0, ix1]
    d01 = dist[iy1, cix0]
    d11 = dist[iy1, ix1]
    top = d00 + (d10 - d00) * fx
    bot = d01 + (d11 - d01) * fx
    return top + (bot - top) * fy


def _hex_sample_violations(
    params5: Tensor,
    heading: Tensor,
    base_pos_xy: Tensor,
    distance_field,
    margin: float,
    soft_margin: float,
    sampling: str = "nearest",
) -> Tensor:
    """Per-sample smooth bounded collision violations for all hex samples.

    Args:
        params5: Envelope parameters, shape ``(..., 5)``.
        heading: Body yaw, shape ``(...,)``.
        base_pos_xy: World ``(x, y)`` position, shape ``(..., 2)``.
        distance_field: 2D unsigned distance-to-obstacle field, shape
            ``(H, W)`` in metres.  Accepts ``numpy.ndarray`` or ``torch.Tensor``.
        margin: Safe distance threshold (m).
        soft_margin: Width over which the violation ramps from 0 to 1 (m).
        sampling: ``"nearest"`` (default) reads the containing cell's value
            (piecewise-constant in sample position, the historical semantics);
            ``"bilinear"`` interpolates the four surrounding cell centres
            (continuous in sample position, so penalties scale with sub-cell
            penetration depth instead of jumping a full ramp step at raster
            lines).

    Returns:
        Violation tensor shape ``(..., 34)``, each value in ``[0, 1]``.
    """
    if soft_margin <= 0.0:
        raise ValueError("soft_margin must be positive")
    if sampling not in ("nearest", "bilinear"):
        raise ValueError(f"unknown sampling mode: {sampling!r}")

    body_pts = hex_body_sample_points(params5)  # (..., 34, 2)

    cos_h = torch.cos(heading).unsqueeze(-1)
    sin_h = torch.sin(heading).unsqueeze(-1)
    px = body_pts[..., 0]
    py = body_pts[..., 1]
    wx = base_pos_xy[..., 0:1] + cos_h * px - sin_h * py
    wy = base_pos_xy[..., 1:2] + sin_h * px + cos_h * py

    if isinstance(distance_field, torch.Tensor):
        dist = distance_field
    else:
        dist = torch.as_tensor(distance_field, dtype=torch.float32)

    if sampling == "bilinear":
        clearance = _bilinear_field_sample(dist, wx, wy)
    else:
        ix = torch.floor(
            (wx - ea2c.EA2_WORLD_MIN_XY) / ea2c.EA2_RESOLUTION_M
        ).long()
        iy = torch.floor(
            (wy - ea2c.EA2_WORLD_MIN_XY) / ea2c.EA2_RESOLUTION_M
        ).long()

        height, width = dist.shape
        ix = ix.clamp(0, width - 1)
        iy = iy.clamp(0, height - 1)

        dist_flat = dist.reshape(-1)
        clearance = dist_flat[iy * width + ix]

    return ((margin - clearance) / soft_margin).clamp(0.0, 1.0)


def hex_collision_violation(
    params5: Tensor,
    heading: Tensor,
    base_pos_xy: Tensor,
    distance_field,
    margin: float,
    soft_margin: float,
    sampling: str = "nearest",
) -> Tensor:
    """Worst-point smooth bounded collision violation from a distance field.

    Args:
        params5: Envelope parameters, shape ``(..., 5)``.
        heading: Body yaw, shape ``(...,)``.
        base_pos_xy: World ``(x, y)`` position, shape ``(..., 2)``.
        distance_field: 2D unsigned distance-to-obstacle field, shape
            ``(H, W)`` in metres.  Accepts ``numpy.ndarray`` or ``torch.Tensor``.
        margin: Safe distance threshold (m).
        soft_margin: Width over which the violation ramps from 0 to 1 (m).

    Returns:
        Violation tensor shape ``(...,)`` in ``[0, 1]``.  It is the maximum
        per-sample ``clamp((margin - clearance) / soft_margin, 0, 1)``.
    """
    _, hard = hex_collision_terms(
        params5,
        heading,
        base_pos_xy,
        distance_field,
        margin,
        soft_margin,
        sampling=sampling,
    )
    return hard


def hex_collision_edge_sum(
    params5: Tensor,
    heading: Tensor,
    base_pos_xy: Tensor,
    distance_field,
    margin: float,
    soft_margin: float,
    sampling: str = "nearest",
) -> Tensor:
    """Sum of smooth collision violations over the 24 hexagon boundary samples.

    Only the hexagon outline is included (6 vertices + 18 edge interpolation
    points).  Interior samples are intentionally excluded from the dense
    reward; they remain available through :func:`hex_collision_violation` for
    hard-collision monitoring.

    Args:
        params5: Envelope parameters, shape ``(..., 5)``.
        heading: Body yaw, shape ``(...,)``.
        base_pos_xy: World ``(x, y)`` position, shape ``(..., 2)``.
        distance_field: 2D unsigned distance-to-obstacle field, shape
            ``(H, W)`` in metres.  Accepts ``numpy.ndarray`` or ``torch.Tensor``.
        margin: Safe distance threshold (m).
        soft_margin: Width over which the violation ramps from 0 to 1 (m).

    Returns:
        Sum tensor shape ``(...,)`` in ``[0, 24]``.
    """
    dense, _ = hex_collision_terms(
        params5,
        heading,
        base_pos_xy,
        distance_field,
        margin,
        soft_margin,
        sampling=sampling,
    )
    return dense


def hex_collision_terms(
    params5: Tensor,
    heading: Tensor,
    base_pos_xy: Tensor,
    distance_field,
    margin: float,
    soft_margin: float,
    sampling: str = "nearest",
) -> Tuple[Tensor, Tensor]:
    """Return ``(dense_edge_sum, hard_max)`` from a single sampling pass.

    ``dense_edge_sum`` matches :func:`hex_collision_edge_sum` and
    ``hard_max`` matches :func:`hex_collision_violation`.
    """
    violation = _hex_sample_violations(
        params5,
        heading,
        base_pos_xy,
        distance_field,
        margin,
        soft_margin,
        sampling=sampling,
    )
    return violation[..., :24].sum(dim=-1), violation.max(dim=-1).values


def envelope_params_to_condition(
    params5: Tensor,
    spec: EnvelopeConditionSpec,
) -> Tensor:
    """Convert 5 symmetric envelope parameters to the 8-dim condition tensor.

    The conversion follows README 2.6: append three zero placeholder prior
    values and let the shared ``apply_env_morphology_priors`` derive the real
    directional-ratio priors.  This guarantees M1 uses exactly the same
    downstream prior logic as the locomotion environment.

    Args:
        params5: Envelope parameters with shape ``(..., 5)`` in order
            ``[front_width, middle_width, back_width, forward_limit,
            backward_limit]``.
        spec: ``EnvelopeConditionSpec`` loaded from the spider_envelop config
            (see ``load_envelope_condition_spec``).

    Returns:
        Condition tensor with shape ``(..., 8)``.  The first five columns are
        unchanged and the last three columns are the derived morphology
        priors.
    """
    if params5.shape[-1] != 5:
        raise ValueError(f"Expected params5 width 5, got {params5.shape[-1]}")
    placeholders = torch.zeros_like(params5[..., :3])
    condition8 = torch.cat([params5, placeholders], dim=-1)
    return apply_env_morphology_priors(condition8, spec)
