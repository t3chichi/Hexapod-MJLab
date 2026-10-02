"""Grid-based per-parameter envelope oracle for ``envelope_adaptive_2``.

This module computes a "theoretical target envelope" directly from the 2D
distance field, without using the LiDAR point cloud.  The envelope family is
parameterised with five independent scale factors, each derived from a group
of hexagon edge directions inspired by the legacy fan-sector grouping in
``envs/el_4090/envelope_adaptive``.

Entry points are provided:

* :func:`compute_oracle_params` returns the raw uncorrected oracle;
* :func:`compute_direct_oracle_params_with_stats` returns the Active-set
  direct per-parameter oracle used by training (with raw hard violation for
  monitoring).
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from hexapod_mjlab.tasks.envelope_adaptive_2.geometry import (
    _hex_sample_violations,
    hex_body_sample_points,
    hex_collision_terms,
)

# The first 24 of the 34 hex samples are boundary samples (6 vertices + 18
# edge interpolation points).  Grouping is based on the legacy fan sectors.
GROUP_INDICES = {
    "front_width": [0, 1, 4, 5, 6, 10, 12, 16, 18, 22],
    "middle_width": [1, 4, 7, 10, 13, 15, 18, 21],
    "back_width": [2, 3, 7, 9, 13, 15, 19, 21],
    "forward_limit": [0, 5, 11, 17, 23],
    "backward_limit": [8, 14, 20],
}
PARAM_NAMES: Sequence[str] = (
    "front_width",
    "middle_width",
    "back_width",
    "forward_limit",
    "backward_limit",
)

# Number of march steps evaluated per batched sampler call in
# _axis_march_crossing_batched.  Chunking bounds the intermediate tensor
# (t_chunk, E, D) without changing any arithmetic -- the result is bitwise
# identical for every chunk size.  8 measured best at N=1024 and stays
# efficient at N=4096, where one shot (t_chunk >= T) degrades badly.
_AXIS_MARCH_T_CHUNK = 8

# Boundary-only dependency groups for the direct oracle.  Unlike the legacy
# GROUP_INDICES, every boundary sample is assigned to *all* envelope
# parameters that influence its coordinates.  This still uses only the 24
# boundary samples; interior points are intentionally excluded.
#
# Hex vertex order: B(0)=fwd_width+fwd_limit, D(1)=middle_width,
# F(2)=back_width+back_limit, E(3)=back_width+back_limit,
# C(4)=middle_width, A(5)=fwd_width+fwd_limit.
# Edge interpolation indices:
#   B-D: 6,12,18; D-F: 7,13,19; F-E: 8,14,20;
#   E-C: 9,15,21; C-A: 10,16,22; A-B: 11,17,23.
_DIRECT_BOUNDARY_GROUPS = {
    "front_width": [0, 5, 6, 10, 11, 12, 16, 17, 18, 22, 23],
    "middle_width": [1, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22],
    "back_width": [2, 3, 7, 8, 9, 13, 14, 15, 19, 20, 21],
    "forward_limit": [0, 5, 6, 10, 11, 12, 16, 17, 18, 22, 23],
    "backward_limit": [2, 3, 7, 8, 9, 13, 14, 15, 19, 20, 21],
}

# Inverse mapping used by the active-set shrink: for every boundary sample
# index (0..23), which of the five envelope parameters influence it.
_SAMPLE_TO_PARAM_MASK = torch.zeros(24, len(PARAM_NAMES), dtype=torch.bool)
for _j, _idxs in enumerate(_DIRECT_BOUNDARY_GROUPS.values()):
    for _s in _idxs:
        _SAMPLE_TO_PARAM_MASK[_s, _j] = True



# The direct oracle intentionally uses only the first 24 hexagon boundary
# samples (vertices + edge interpolations).  Interior sampling is not used
# for the oracle target; it is kept for hard-collision monitoring.


_EDGE_COUNT = 24


def _physical_min_max(
    low: Tensor,
    high: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return the physical minimum/maximum envelope parameter vectors.

    The backward_limit bound is reversed: a more negative value is a larger
    physical rear extent, so ``max_v`` uses ``low[4]`` and ``min_v`` uses
    ``high[4]``.
    """
    min_v = torch.stack(
        [low[0], low[1], low[2], low[3], high[4]]
    )
    max_v = torch.stack(
        [high[0], high[1], high[2], high[3], low[4]]
    )
    return min_v, max_v


def _full_edge_directions(
    low: Tensor,
    high: Tensor,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Return ``(edge_xy, radii)`` for the full-size hex boundary samples."""
    min_v, max_v = _physical_min_max(low, high)
    full = hex_body_sample_points(max_v.unsqueeze(0))  # (1, 34, 2)
    edge = full[0, :_EDGE_COUNT].to(device)  # (24, 2)
    radii = torch.norm(edge, dim=-1)
    return edge, radii


def _make_field_sampler(dist: Tensor, interp: bool, max_dist: float):
    """Return a vectorised clearance sampler over the distance field.

    Nearest-cell lookup when interp is false; bilinear in cell centres
    otherwise (see :func: for why the interpolated
    crossing needs a continuous field).  Out-of-bounds positions read a
    finite clear sentinel so interpolation denominators stay finite.
    """
    height, width = dist.shape
    world_min = -37.0
    res = 0.1
    clear_sentinel = max_dist + 1.0
    dist_flat = dist.reshape(-1)

    def sample(px: Tensor, py: Tensor) -> Tensor:
        if not interp:
            ix = torch.floor((px - world_min) / res).long()
            iy = torch.floor((py - world_min) / res).long()
            in_bounds = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
            safe_ix = ix.clamp(0, width - 1)
            safe_iy = iy.clamp(0, height - 1)
            return torch.where(
                in_bounds,
                dist_flat[safe_iy * width + safe_ix],
                torch.full_like(ix, clear_sentinel, dtype=torch.float32),
            )
        u = (px - world_min) / res - 0.5
        v = (py - world_min) / res - 0.5
        ix0 = torch.floor(u).long()
        iy0 = torch.floor(v).long()
        fx = (u - ix0.float()).clamp(0.0, 1.0)
        fy = (v - iy0.float()).clamp(0.0, 1.0)
        ix1 = ix0 + 1
        iy1 = iy0 + 1
        in_bounds = (ix0 >= 0) & (iy0 >= 0) & (ix1 < width) & (iy1 < height)
        cix0 = ix0.clamp(0, width - 1)
        cix1 = ix1.clamp(0, width - 1)
        ciy0 = iy0.clamp(0, height - 1)
        ciy1 = iy1.clamp(0, height - 1)
        d00 = dist_flat[ciy0 * width + cix0]
        d10 = dist_flat[ciy0 * width + cix1]
        d01 = dist_flat[ciy1 * width + cix0]
        d11 = dist_flat[ciy1 * width + cix1]
        top = d00 + (d10 - d00) * fx
        bot = d01 + (d11 - d01) * fx
        out = top + (bot - top) * fy
        return torch.where(
            in_bounds, out, torch.full_like(out, clear_sentinel)
        )

    return sample


def _axis_march_crossing_batched(sample, start_x: Tensor, start_y: Tensor,
                                 dir_x: Tensor, dir_y: Tensor,
                                 margin: float, step: float, max_dist: float,
                                 interp: bool, t_chunk: int = 8) -> Tensor:
    """Batched axis-aligned ray march: first crossing distance per direction.

    Returns the distance at which the clearance first drops below ``margin``
    along each direction, interpolated when ``interp`` is set, and ``inf`` when
    the ray never crosses.

    The original implementation advanced one ``step`` at a time in a Python
    loop, issuing ~21 rounds of tiny kernels per call; with four calls per
    control step and 1024 envs that dominated the oracle cost.  This version
    evaluates the same march positions in batched sampler calls and reduces the
    "first crossing" out of them.  The superseded loop version is preserved in
    ``tests/ea2/_oracle_reference.py`` as the bitwise reference.

    The result is *bitwise* identical to the loop: every op is elementwise
    and re-associated only across the independent ``t`` axis, and the
    cross-iteration state (``prev_clear``, carried as the previous row's
    clearance) is reproduced exactly.  Chunking over ``t`` bounds the
    intermediate memory without changing any arithmetic.

    Two details that are easy to get wrong and are covered by tests in
    ``tests/ea2/test_oracle_march_equivalence.py``:

    * the explicit ``t = 0`` sample is taken BEFORE the loop and overrides any
      later crossing; it must be kept, and kept first;
    * ``prev_clear`` is not one value per chunk -- inside a chunk row ``i``
      uses row ``i-1``'s clearance, and only row 0 of the first chunk uses the
      ``t = 0`` sample.
    """
    device = start_x.device
    # `start_*` may be (E, 1) broadcast against dir_* of (E, D), so the
    # direction count must come from dir_*, not start_*.
    E, D = dir_x.shape[0], dir_x.shape[1]
    if start_x.shape[1] == 1:
        start_x = start_x.expand(E, D)
    if start_y.shape[1] == 1:
        start_y = start_y.expand(E, D)

    # t = 0 sample, taken before any marching (see _axis_march_crossing).
    prev_clear = sample(start_x, start_y)
    bad0 = prev_clear < margin

    ts: list[float] = []
    t = step
    while t <= max_dist + 1e-9:
        ts.append(t)
        t += step
    T = len(ts)

    # No marching step fits: the result is decided by the t=0 sample alone.
    if T == 0:
        return torch.where(
            bad0,
            torch.zeros_like(prev_clear),
            torch.full_like(prev_clear, float("inf")),
        )

    first_bad = torch.where(
        bad0, torch.zeros_like(prev_clear), torch.full_like(prev_clear, float("inf"))
    )
    prev_t_all = torch.tensor([0.0] + ts[:-1], dtype=torch.float32, device=device)
    t_all = torch.tensor(ts, dtype=torch.float32, device=device)
    rows_all = torch.arange(T, device=device, dtype=torch.long)
    inf_row = T + 1

    for s in range(0, T, t_chunk):
        e = min(s + t_chunk, T)
        c = e - s
        tv = t_all[s:e].view(c, 1, 1)
        ptv = prev_t_all[s:e].view(c, 1, 1)

        clearance = sample(start_x.view(1, E, D) + tv * dir_x.view(1, E, D),
                          start_y.view(1, E, D) + tv * dir_y.view(1, E, D))
        bad = clearance < margin

        if interp:
            # row i uses row i-1's clearance; row 0 of this chunk continues
            # from `prev_clear` (the t=0 sample for the first chunk).
            prev = torch.cat([prev_clear.unsqueeze(0), clearance[:-1]], dim=0)
            frac = (prev - margin) / (prev - clearance).clamp_min(1e-6)
            frac = frac.clamp(0.0, 1.0)
            t_cross = ptv + frac * (tv - ptv)
        else:
            t_cross = tv.expand(c, E, D).contiguous()

        rows = rows_all[s:e].view(c, 1, 1).expand(c, E, D)
        cand = torch.where(bad, rows, torch.full_like(rows, inf_row))
        best = cand.amin(dim=0)                    # first bad row in this chunk
        accept = (best < inf_row) & torch.isinf(first_bad)

        idx = (best - s).clamp(0, c - 1)
        val = t_cross.permute(1, 2, 0).reshape(E * D, c).gather(
            1, idx.reshape(E * D, 1)).reshape(E, D)
        first_bad = torch.where(accept, val, first_bad)

        prev_clear = clearance[-1]                # carry into the next chunk

    return first_bad


def _axis_decoupled_scales(
    heading: Tensor,
    base_pos_xy: Tensor,
    low: Tensor,
    high: Tensor,
    margin: float,
    step: float,
    sample,
    max_dist: float,
    interp_crossing: bool,
    t_chunk: int = _AXIS_MARCH_T_CHUNK,
) -> Tensor:
    """Axis-decoupled scales: four axis-aligned marches, no main march.

    Extracted from the ``"axis"`` branch of :func:`_compute_raw_scales` so that
    the branch can run *before* the 24-direction march and thus skip that march
    entirely (it was dead code in this mode).

    The marches use :func:`_axis_march_crossing_batched`, which collapses the
    per-step Python loop into one batched sampler call per chunk --
    bitwise-identical, but ~7x faster in total because the loop is host
    dispatch bound.

    The march range is capped at ``fwd_min + fwd_span + margin + step``: past
    that distance the scale clamps to fully open, so marching further only
    burns iterations.
    """
    fwd_min = float(low[3])
    fwd_span = float(high[3] - low[3])
    lat_min = float(low[0])
    lat_span = float(high[0] - low[0])
    bx = base_pos_xy[:, 0:1]
    by = base_pos_xy[:, 1:2]
    cos_h = torch.cos(heading).unsqueeze(-1)
    sin_h = torch.sin(heading).unsqueeze(-1)
    # longitudinal: both body +-x marches batched in one loop.  The axis
    # march only needs to cover the largest extent plus margin -- beyond
    # that the scale clamps to fully open -- so its range is capped.
    axis_max_dist = min(max_dist, fwd_min + fwd_span + margin + step)
    d_long_x = torch.cat([cos_h, -cos_h], dim=1)   # (E, 2)
    d_long_y = torch.cat([sin_h, -sin_h], dim=1)
    x_cross = _axis_march_crossing_batched(
        sample, bx, by, d_long_x, d_long_y, margin, step, axis_max_dist,
        interp_crossing, t_chunk,
    )  # (E, 2): [front, rear]
    s_fwd = ((x_cross[:, 0:1] - fwd_min) / fwd_span).clamp(0.0, 1.0)
    s_bwd = ((x_cross[:, 1:2] - fwd_min) / fwd_span).clamp(0.0, 1.0)
    # lateral: vertical marches at the already-shrunk apex positions,
    # batched (front apex up/down + rear apex up/down) in one loop
    fx_local = fwd_min + s_fwd * fwd_span
    bwd_local = fwd_min + s_bwd * fwd_span
    sx_f = bx + fx_local * cos_h
    sy_f = by + fx_local * sin_h
    sx_b = bx - bwd_local * cos_h
    sy_b = by - bwd_local * sin_h
    start_x = torch.cat([sx_f, sx_f, sx_b, sx_b], dim=1)
    start_y = torch.cat([sy_f, sy_f, sy_b, sy_b], dim=1)
    d_lat_x = torch.cat([-sin_h, sin_h, -sin_h, sin_h], dim=1)
    d_lat_y = torch.cat([cos_h, -cos_h, cos_h, -cos_h], dim=1)
    y_cross = _axis_march_crossing_batched(
        sample, start_x, start_y, d_lat_x, d_lat_y, margin, step,
        axis_max_dist, interp_crossing, t_chunk,
    )  # (E, 4): [fwd up, fwd dn, bwd up, bwd dn]
    y_front = torch.minimum(y_cross[:, 0:1], y_cross[:, 1:2])
    s_fw = ((y_front - lat_min) / lat_span).clamp(0.0, 1.0)
    y_back = torch.minimum(y_cross[:, 2:3], y_cross[:, 3:4])
    s_bw = ((y_back - lat_min) / lat_span).clamp(0.0, 1.0)
    # middle_width uses the same vertical-march semantics at x = 0 (the
    # D/C vertices are the only middle samples with x == 0), sharing the
    # exact affine mapping of the other widths instead of the legacy
    # radial ray-min, which over-allowed by the min_v offset.
    y_up_m = _axis_march_crossing_batched(
        sample, bx, by, -sin_h, cos_h, margin, step, axis_max_dist,
        interp_crossing, t_chunk,
    )
    y_dn_m = _axis_march_crossing_batched(
        sample, bx, by, sin_h, -cos_h, margin, step, axis_max_dist,
        interp_crossing, t_chunk,
    )
    y_mid = torch.minimum(y_up_m, y_dn_m)
    lat_min_mw = float(low[1])  # middle_width's own floor (== low[0] in the
    mw_span = float(high[1] - low[1])  # current contract, but not by necessity)
    s_mw = ((y_mid - lat_min_mw) / mw_span).clamp(0.0, 1.0)
    return torch.cat(
        [
            s_fw.clamp(0.0, 1.0),
            s_mw.clamp(0.0, 1.0),
            s_bw.clamp(0.0, 1.0),
            s_fwd,
            s_bwd,
        ],
        dim=1,
    )


def _coupled_scales(
    heading: Tensor,
    base_pos_xy: Tensor,
    low: Tensor,
    high: Tensor,
    margin: float,
    step: float,
    sample,
    max_dist: float,
    interp_crossing: bool,
    device: torch.device,
) -> Tensor:
    """Historical 'coupled' scales from the 24-direction boundary march.

    Kept as-is: production runs ``"axis"``, but several tests and validation
    scripts still exercise ``"coupled"``, and one test compares the two modes.
    A frozen copy for equivalence checking lives in
    ``tests/ea2/_oracle_reference.py``.
    """
    edge, radii = _full_edge_directions(low, high, device)

    # Rotate body-frame edge directions into world frame.
    cos_h = torch.cos(heading).unsqueeze(-1)
    sin_h = torch.sin(heading).unsqueeze(-1)
    wx = cos_h * edge[:, 0] - sin_h * edge[:, 1]
    wy = sin_h * edge[:, 0] + cos_h * edge[:, 1]
    norm = torch.hypot(wx, wy).clamp_min(1e-6)
    dir_x = wx / norm
    dir_y = wy / norm

    first_bad = torch.full(
        (base_pos_xy.shape[0], _EDGE_COUNT),
        float("inf"),
        dtype=torch.float32,
        device=device,
    )
    prev_t = torch.zeros_like(first_bad)
    prev_clear = torch.full_like(first_bad, max_dist + 1.0)

    t = step
    while t <= max_dist + 1e-9:
        px = base_pos_xy[:, 0:1] + t * dir_x
        py = base_pos_xy[:, 1:2] + t * dir_y
        clearance = sample(px, py)
        bad = clearance < margin
        if interp_crossing:
            frac = (prev_clear - margin) / (prev_clear - clearance).clamp_min(1e-6)
            frac = frac.clamp(0.0, 1.0)
            t_cross = prev_t + frac * (t - prev_t)
            first_bad = torch.where(
                bad & torch.isinf(first_bad), t_cross, first_bad
            )
        else:
            first_bad = torch.where(bad & torch.isinf(first_bad), t, first_bad)
        prev_t = torch.full_like(prev_t, t)
        prev_clear = clearance
        t += step

    if interp_crossing:
        allowed = torch.where(
            torch.isfinite(first_bad),
            first_bad / radii.unsqueeze(0),
            torch.ones_like(first_bad),
        )
    else:
        allowed = torch.where(
            torch.isfinite(first_bad),
            ((first_bad - step) / radii.unsqueeze(0)).clamp(0.0, 1.0),
            torch.ones_like(first_bad),
        )

    scales = []
    for name in PARAM_NAMES:
        idx = GROUP_INDICES[name]
        group_allowed = allowed[:, idx]
        s = group_allowed.min(dim=-1).values
        scales.append(s.clamp(0.0, 1.0))

    return torch.stack(scales, dim=-1)  # (E, 5)


def _compute_raw_scales(
    heading: Tensor,
    base_pos_xy: Tensor,
    distance_field,
    low: Tensor,
    high: Tensor,
    margin: float,
    step: float,
    max_dist: float,
    interp_crossing: bool = False,
    group_mode: str = "coupled",
) -> Tensor:
    """Return raw per-parameter scales, shape ``(E, 5)``.

    With ``interp_crossing=False`` (default) the first obstructed sample is
    quantised to the ray-march grid and backed off one full step, exactly as
    before.  With ``interp_crossing=True`` the ``clearance == margin``
    crossing is linearly interpolated between the last clear and the first
    obstructed sample, removing the staircase artefact (and its up to one
    step of extra conservatism) at the source.

    ``group_mode``:

    * ``"coupled"`` (default) -- historical semantics: the shared sample
      groups of (front_width, forward_limit) and (back_width,
      backward_limit) take a common ray-scale minimum, which couples the
      pairs (a dead-ahead obstacle also shrinks front_width).
    * ``"axis"`` -- the degenerate pairs decouple along their natural
      coordinate axes (the A-B edge is the line x = forward_limit with
      y = +-front_width; F-E likewise): forward_limit/backward_limit come
      from body +-x axis marches, front_width/back_width from vertical
      marches at the already-shrunk apex position.  The composed hexagon is
      still verified by the active-set stage, which backstops the
      diagonal-visibility loss of axis marches.

    In ``"axis"`` mode the 24-direction main march below is dead code (its
    ``allowed`` result is never read), so the function branches to
    :func:`_axis_decoupled_scales` before performing it.
    """
    if distance_field is None:
        raise ValueError("distance_field must be provided")
    if low.shape[-1] != 5 or high.shape[-1] != 5:
        raise ValueError("low/high must have last dimension 5")
    if step <= 0.0:
        raise ValueError("step must be positive")
    if max_dist <= 0.0:
        raise ValueError("max_dist must be positive")

    device = base_pos_xy.device
    if isinstance(distance_field, torch.Tensor):
        dist = distance_field
    else:
        dist = torch.as_tensor(distance_field, dtype=torch.float32, device=device)
    if dist.device != device:
        dist = dist.to(device)

    # The four axis marches below return a complete result on their own: the
    # 24-direction main march is *never* consumed in this mode (it computes
    # `allowed`, which the axis branch never reads).  Branching early avoids
    # ~576 discarded ray samples per env per control step.
    #
    # This is a pure dead-code elimination -- `sample` is the only piece the
    # axis branch needs from the shared setup, and it is built here as before.
    # Verified bitwise-identical against the frozen reference in
    # tests/ea2/_oracle_reference.py.
    sample = _make_field_sampler(dist, interp_crossing, max_dist)
    if group_mode == "axis":
        return _axis_decoupled_scales(
            heading, base_pos_xy, low, high, margin, step, sample,
            max_dist, interp_crossing)
    if group_mode != "coupled":
        raise ValueError(
            f"unknown group_mode: {group_mode!r} (expected 'coupled' or 'axis')")

    return _coupled_scales(
        heading, base_pos_xy, low, high, margin, step, sample,
        max_dist, interp_crossing, device)


def _params_from_scales(
    scale_tensor: Tensor,
    low: Tensor,
    high: Tensor,
) -> Tensor:
    """Convert ``(E, 5)`` scales to envelope parameters."""
    min_v, max_v = _physical_min_max(low, high)
    return min_v.unsqueeze(0) + scale_tensor * (max_v - min_v).unsqueeze(0)



def _active_set_shrink_params(
    params: Tensor,
    low: Tensor,
    high: Tensor,
    heading: Tensor,
    base_pos_xy: Tensor,
    distance_field,
    margin: float,
    soft_margin: float,
    threshold: float = 0.05,
    num_iter: int = 6,
    max_iters: int = 2,
) -> Tensor:
    """Shrink only the parameters involved in current collisions (active set).

    Instead of the old per-parameter coordinate descent, this finds each
    violating boundary sample, collects *all* envelope parameters that
    influence it, and jointly binary-searches a shrink multiplier for exactly
    that active subset.  Inactive parameters are left untouched, so this is
    not a global safety multiplier.  Several outer iterations handle newly
    exposed violations caused by shape coupling.
    """
    min_v, max_v = _physical_min_max(low, high)
    current = params.clone()
    sample_mask = _SAMPLE_TO_PARAM_MASK.to(
        device=current.device, dtype=torch.bool
    )

    denom = max_v - min_v

    for _ in range(max(1, max_iters)):
        violation = _hex_sample_violations(
            current,
            heading,
            base_pos_xy,
            distance_field,
            margin,
            soft_margin,
        )[..., :24]  # (E, 24)
        bad = violation > threshold  # (E, 24)
        if not bool(bad.any().item()):
            break

        # Parameters that influence at least one currently violating sample.
        active = (bad.unsqueeze(-1) & sample_mask.unsqueeze(0)).any(dim=1)

        scales = ((current - min_v) / denom).clamp(0.0, 1.0)

        # Per-environment binary search on a common shrink multiplier applied
        # only to active parameters.
        lo = torch.zeros(
            current.shape[0], dtype=current.dtype, device=current.device
        )
        hi = torch.ones_like(lo)

        for _ in range(max(1, num_iter)):
            mid = 0.5 * (lo + hi)
            factor = torch.where(
                active,
                mid.unsqueeze(-1),
                torch.ones_like(mid).unsqueeze(-1),
            )
            cand_scales = (scales * factor).clamp(0.0, 1.0)
            cand = min_v + cand_scales * denom

            cand_viol = _hex_sample_violations(
                cand,
                heading,
                base_pos_xy,
                distance_field,
                margin,
                soft_margin,
            )[..., :24]
            safe = (cand_viol <= threshold).all(dim=-1)
            lo = torch.where(safe, mid, lo)
            hi = torch.where(safe, hi, mid)

        factor = torch.where(
            active,
            lo.unsqueeze(-1),
            torch.ones_like(lo).unsqueeze(-1),
        )
        current = min_v + (scales * factor).clamp(0.0, 1.0) * denom

    return current


def compute_oracle_params(
    heading: Tensor,
    base_pos_xy: Tensor,
    distance_field,
    low: Tensor,
    high: Tensor,
    margin: float = 0.10,
    step: float = 0.05,
    max_dist: float = 5.0,
    interp_crossing: bool = False,
    group_mode: str = "coupled",
) -> Tensor:
    """Compute raw per-parameter theoretical envelope params from the grid map.

    This is the uncorrected oracle.  Use
    :func:`compute_direct_oracle_params_with_stats` when the training reward
    should target the Active-set direct envelope.

    Args:
        heading: Body yaw, shape ``(E,)``.
        base_pos_xy: World ``(x, y)`` position, shape ``(E, 2)``.
        distance_field: 2D unsigned distance-to-obstacle field, shape
            ``(H, W)`` in metres.  Accepts ``numpy.ndarray`` or ``torch.Tensor``.
        low: Five envelope lower bounds, shape ``(5,)``.
        high: Five envelope upper bounds, shape ``(5,)``.
        margin: Safety clearance threshold used during ray marching.
        step: Ray-march step size in metres.
        max_dist: Max ray-march distance in metres.
        interp_crossing: Linearly interpolate the ``clearance == margin``
            crossing (see :func:`_compute_raw_scales`).

    Returns:
        Oracle envelope parameters, shape ``(E, 5)``.
    """
    scales = _compute_raw_scales(
        heading,
        base_pos_xy,
        distance_field,
        low,
        high,
        margin,
        step,
        max_dist,
        interp_crossing,
        group_mode,
    )
    return _params_from_scales(scales, low, high)


def compute_direct_oracle_params_with_stats(
    heading: Tensor,
    base_pos_xy: Tensor,
    distance_field,
    low: Tensor,
    high: Tensor,
    margin: float = 0.10,
    step: float = 0.05,
    max_dist: float = 5.0,
    safety_threshold: float = 0.05,
    num_iter: int = 6,
    max_iters: int = 2,
    collision_margin: float = 0.10,
    collision_soft_margin: float = 0.10,
    soft_dof_pos_limit: float = 0.9,
    interp_crossing: bool = False,
    group_mode: str = "coupled",
) -> tuple[Tensor, Tensor]:
    """Direct per-parameter oracle using active-set joint shrink.

    The raw per-parameter oracle is still computed by ray-marching.  Instead
    of a global multiplier, only the parameters involved in current boundary
    violations are jointly shrunk (active set).  The returned tuple is
    ``(params, raw_hard)``.

    Ordering contract: the expansion-side soft caps are applied BEFORE the
    active-set shrink, so the shrink stage is the last to touch the params
    and its per-sample safety verification holds for the returned values.
    (A post-shrink clamp would move samples after verification and can push
    a corner back into an obstacle cell.)

    No global safety verification/fallback is performed here; use the returned
    params only after checking the monitored ``oracle_unsafe_*`` ratios.
    """
    scales = _compute_raw_scales(
        heading,
        base_pos_xy,
        distance_field,
        low,
        high,
        margin,
        step,
        max_dist,
        interp_crossing,
        group_mode,
    )
    raw_params = _params_from_scales(scales, low, high)

    # Align the oracle target with the policy's soft action range on the
    # *expansion* side only.  The soft bounds exist to keep actions away from
    # the hard limits; their per-parameter floors (e.g. back_width >= 0.315)
    # must NOT act as a minimum-size constraint -- when the active-set shrink
    # has to go below the soft floor to clear an obstacle, safety wins.
    # For backward_limit the expansion side is soft_low (more negative =
    # larger rear extent), so only that side is softened.
    soft_margin = (1.0 - soft_dof_pos_limit) * (high - low) / 2.0
    soft_high = high - soft_margin
    soft_low_bwd = (low + soft_margin)[4]
    clamped = [
        torch.clamp(raw_params[..., j], min=low[j].item(), max=soft_high[j].item())
        for j in range(4)
    ]
    clamped.append(
        torch.clamp(raw_params[..., 4], min=soft_low_bwd.item(), max=high[4].item())
    )
    raw_params = torch.stack(clamped, dim=-1)

    _, raw_hard = hex_collision_terms(
        raw_params,
        heading,
        base_pos_xy,
        distance_field,
        margin=collision_margin,
        soft_margin=collision_soft_margin,
    )

    direct_params = _active_set_shrink_params(
        raw_params,
        low,
        high,
        heading,
        base_pos_xy,
        distance_field,
        margin=collision_margin,
        soft_margin=collision_soft_margin,
        threshold=safety_threshold,
        num_iter=num_iter,
        max_iters=max_iters,
    )

    return direct_params, raw_hard
