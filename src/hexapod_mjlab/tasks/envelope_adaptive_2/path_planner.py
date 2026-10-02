"""A* path planning, LOS simplification, resampling, noise and heading helpers.

This module implements README v2 sections 2.2.4-2.2.6 for the
``envelope_adaptive_2`` M1 task.  It depends only on numpy and the frozen
``_contracts`` dataclasses.
"""

from __future__ import annotations

import heapq
import math
from typing import List, Sequence, Tuple

import numpy as np

from .contracts import EA2_GRID_SHAPE, EA2_RESOLUTION_M, EA2_WORLD_MIN_XY, PathCfg, PathData

# 8-connectivity in grid coordinates (dy, dx), matching (iy, ix).
_NEIGHBORS: Tuple[Tuple[int, int], ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

# Number of independent noise draws attempted before a path is rejected.
_MAX_NOISE_ATTEMPTS = 3


def wrap_to_pi(angle: float) -> float:
    """Wrap an angle to the interval [-pi, pi).

    Works element-wise for numpy arrays as well as Python scalars.
    """
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def _world_to_grid(x: float, y: float) -> Tuple[int, int]:
    """Convert world coordinates to grid indices ``(iy, ix)``.

    Follows the README formula: ``ix = floor((x + 37.0) / 0.1)``.
    """
    ix = int(math.floor((x - EA2_WORLD_MIN_XY) / EA2_RESOLUTION_M))
    iy = int(math.floor((y - EA2_WORLD_MIN_XY) / EA2_RESOLUTION_M))
    return iy, ix


def _grid_to_world(ix: int, iy: int) -> Tuple[float, float]:
    """Convert grid indices to world coordinates of the cell center."""
    x = EA2_WORLD_MIN_XY + (ix + 0.5) * EA2_RESOLUTION_M
    y = EA2_WORLD_MIN_XY + (iy + 0.5) * EA2_RESOLUTION_M
    return x, y


def _in_bounds(ix: int, iy: int, shape: Tuple[int, int]) -> bool:
    """Return whether grid indices lie inside the map."""
    rows, cols = shape
    return 0 <= ix < cols and 0 <= iy < rows


def _cell_free(inflated: np.ndarray, ix: int, iy: int) -> bool:
    """Return whether the grid cell is inside the map and not blocked."""
    if not _in_bounds(ix, iy, inflated.shape):
        return False
    return int(inflated[iy, ix]) == 0


def _point_free(xy: Sequence[float], inflated: np.ndarray) -> bool:
    """Return whether a world-space point lies in a free inflated cell."""
    iy, ix = _world_to_grid(float(xy[0]), float(xy[1]))
    return _cell_free(inflated, ix, iy)


def _min_obstacle_distance_world(occupancy: np.ndarray, xy: Sequence[float]) -> float:
    """Return the Euclidean distance from a world point to the nearest raw obstacle.

    Distances are measured to occupied cell centers.  An empty occupancy grid
    yields ``inf``.
    """
    cells = np.argwhere(occupancy > 0)
    if cells.size == 0:
        return float("inf")
    xs = EA2_WORLD_MIN_XY + (cells[:, 1].astype(np.float64) + 0.5) * EA2_RESOLUTION_M
    ys = EA2_WORLD_MIN_XY + (cells[:, 0].astype(np.float64) + 0.5) * EA2_RESOLUTION_M
    return float(np.min(np.hypot(xs - float(xy[0]), ys - float(xy[1]))))


def _segment_clear(p0: Sequence[float], p1: Sequence[float], inflated: np.ndarray) -> bool:
    """Return whether the straight segment between two world points avoids blocked cells.

    A conservative grid traversal (Amanatides-Woo DDA) enumerates every cell
    crossed by the segment.  Endpoints outside the map are treated as blocked.
    The traversal stops at the cell containing ``p1`` and never steps into a
    cell whose boundary crossing lies beyond the segment endpoint.
    """
    x0, y0 = (float(p0[0]) - EA2_WORLD_MIN_XY) / EA2_RESOLUTION_M, (
        float(p0[1]) - EA2_WORLD_MIN_XY
    ) / EA2_RESOLUTION_M
    x1, y1 = (float(p1[0]) - EA2_WORLD_MIN_XY) / EA2_RESOLUTION_M, (
        float(p1[1]) - EA2_WORLD_MIN_XY
    ) / EA2_RESOLUTION_M

    ix = int(math.floor(x0))
    iy = int(math.floor(y0))
    end_ix = int(math.floor(x1))
    end_iy = int(math.floor(y1))
    if not _cell_free(inflated, ix, iy):
        return False

    dx = x1 - x0
    dy = y1 - y0
    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        return True

    step_x = 0 if abs(dx) <= 1e-12 else (1 if dx > 0 else -1)
    step_y = 0 if abs(dy) <= 1e-12 else (1 if dy > 0 else -1)
    t_delta_x = abs(1.0 / dx) if abs(dx) > 1e-12 else math.inf
    t_delta_y = abs(1.0 / dy) if abs(dy) > 1e-12 else math.inf

    if step_x > 0:
        t_max_x = (ix + 1.0 - x0) * t_delta_x
    elif step_x < 0:
        t_max_x = (x0 - ix) * t_delta_x
    else:
        t_max_x = math.inf
    if step_y > 0:
        t_max_y = (iy + 1.0 - y0) * t_delta_y
    elif step_y < 0:
        t_max_y = (y0 - iy) * t_delta_y
    else:
        t_max_y = math.inf

    _EPS = 1e-12
    while (ix, iy) != (end_ix, end_iy):
        if t_max_x + _EPS < t_max_y:
            next_t = t_max_x
            if next_t > 1.0 + 1e-12:
                break
            ix += step_x
            t_max_x += t_delta_x
        elif t_max_y + _EPS < t_max_x:
            next_t = t_max_y
            if next_t > 1.0 + 1e-12:
                break
            iy += step_y
            t_max_y += t_delta_y
        else:
            next_t = t_max_x
            if next_t > 1.0 + 1e-12:
                break
            # Exact (or numerically near-exact) corner crossing: step both axes
            # to avoid an optimistic diagonal skip and to avoid flagging a
            # cell that the segment only touches at its corner.
            ix += step_x
            iy += step_y
            t_max_x += t_delta_x
            t_max_y += t_delta_y
        if not _cell_free(inflated, ix, iy):
            return False
    return True


def _path_clear(points: np.ndarray, inflated: np.ndarray) -> bool:
    """Return whether every point is free and every segment avoids blocked cells.

    Args:
        points: ``(P, 2)`` world-coordinate polyline.
        inflated: Inflated occupancy grid; 1 = blocked.

    Returns:
        True when all points are free in the inflated grid and all consecutive
        segments are clear.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        return False
    if pts.shape[0] == 0:
        return True
    for p in pts:
        if not _point_free(p, inflated):
            return False
    for p0, p1 in zip(pts[:-1], pts[1:]):
        if not _segment_clear(p0, p1, inflated):
            return False
    return True


def _astar(
    inflated: np.ndarray,
    start_xy: Sequence[float],
    goal_xy: Sequence[float],
) -> List[Tuple[float, float]]:
    """Run 8-neighbour A* on the inflated grid.

    Cost and heuristic are Euclidean.  The returned polyline starts/ends at the
    exact requested world coordinates and uses cell centers for intermediate
    waypoints.
    """
    start_iy, start_ix = _world_to_grid(float(start_xy[0]), float(start_xy[1]))
    goal_iy, goal_ix = _world_to_grid(float(goal_xy[0]), float(goal_xy[1]))
    if not _cell_free(inflated, start_ix, start_iy):
        raise ValueError(f"start {tuple(start_xy)} is not free in inflated grid")
    if not _cell_free(inflated, goal_ix, goal_iy):
        raise ValueError(f"goal {tuple(goal_xy)} is not free in inflated grid")

    start = (start_iy, start_ix)
    goal = (goal_iy, goal_ix)
    if start == goal:
        return [(float(start_xy[0]), float(start_xy[1])), (float(goal_xy[0]), float(goal_xy[1]))]

    def _heuristic(cell: Tuple[int, int]) -> float:
        return math.hypot(cell[0] - goal[0], cell[1] - goal[1])

    open_heap: List[Tuple[float, int, Tuple[int, int]]] = []
    counter = 0
    heapq.heappush(open_heap, (_heuristic(start), counter, start))
    g_score = {start: 0.0}
    came_from = {start: None}
    closed: set = set()

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            break
        closed.add(current)
        for dy, dx in _NEIGHBORS:
            nxt = (current[0] + dy, current[1] + dx)
            if not _cell_free(inflated, nxt[1], nxt[0]):
                continue
            tentative = g_score[current] + math.hypot(dx, dy)
            if tentative < g_score.get(nxt, math.inf):
                g_score[nxt] = tentative
                came_from[nxt] = current
                counter += 1
                heapq.heappush(open_heap, (tentative + _heuristic(nxt), counter, nxt))

    if goal not in g_score:
        raise ValueError("A* found no path from start to goal on inflated grid")

    cells: List[Tuple[int, int]] = []
    cell = goal
    while cell is not None:
        cells.append(cell)
        cell = came_from[cell]
    cells.reverse()

    points: List[Tuple[float, float]] = []
    for idx, (iy, ix) in enumerate(cells):
        if idx == 0:
            points.append((float(start_xy[0]), float(start_xy[1])))
        elif idx == len(cells) - 1:
            points.append((float(goal_xy[0]), float(goal_xy[1])))
        else:
            points.append(_grid_to_world(ix, iy))
    return points


def _simplify_path(points: Sequence[Sequence[float]], inflated: np.ndarray) -> List[Tuple[float, float]]:
    """Greedy line-of-sight simplification of an A* polyline.

    A waypoint is kept only when the straight segment from the last kept
    waypoint to the candidate would cross an inflated occupied cell.
    """
    if len(points) <= 2:
        return [(float(p[0]), float(p[1])) for p in points]

    simplified: List[Tuple[float, float]] = [(float(points[0][0]), float(points[0][1]))]
    for i in range(1, len(points)):
        candidate = (float(points[i][0]), float(points[i][1]))
        if not _segment_clear(simplified[-1], candidate, inflated):
            simplified.append((float(points[i - 1][0]), float(points[i - 1][1])))
    simplified.append((float(points[-1][0]), float(points[-1][1])))

    # Remove duplicates caused by the greedy rule.
    deduped: List[Tuple[float, float]] = []
    for p in simplified:
        if not deduped or math.hypot(p[0] - deduped[-1][0], p[1] - deduped[-1][1]) > 1e-9:
            deduped.append(p)
    return deduped


def _cumulative_arc_lengths(points: np.ndarray) -> np.ndarray:
    """Return cumulative arc length of a polyline, starting at 0.

    Args:
        points: ``(P, 2)`` polyline in world coordinates.

    Returns:
        ``(P,)`` cumulative Euclidean arc length; ``arc[0]`` is 0.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    seg_lens = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(seg_lens))).astype(np.float64)


def _resample_polyline(
    points: Sequence[Sequence[float]], step: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample a polyline at fixed arc-length intervals.

    Returns ``(points, arc)`` where ``points`` has shape ``(P, 2)`` and ``arc``
    is the cumulative arc length starting at 0.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 2:
        return pts.copy(), np.zeros((pts.shape[0],), dtype=np.float64)

    seg_lens = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate(([0.0], np.cumsum(seg_lens)))
    total = float(cum[-1])

    if total < 1e-9:
        return pts.copy(), np.zeros((pts.shape[0],), dtype=np.float64)

    # Include the original polyline vertices in the resampled output.  Without
    # them, two resampled points on either side of a very short segment can be
    # connected by a chord that cuts across an obstacle even though the
    # simplified polyline itself is clear.
    targets = np.unique(np.concatenate((np.arange(0.0, total, step), cum)))

    out = np.empty((len(targets), 2), dtype=np.float64)
    for k, s in enumerate(targets):
        idx = int(np.searchsorted(cum, s, side="right") - 1)
        idx = min(max(idx, 0), pts.shape[0] - 2)
        seg_len = float(cum[idx + 1] - cum[idx])
        if seg_len <= 1e-12:
            out[k] = pts[idx]
        else:
            t = float((s - cum[idx]) / seg_len)
            out[k] = pts[idx] * (1.0 - t) + pts[idx + 1] * t

    return out, targets.astype(np.float64)


def _tangent_directions(points: np.ndarray) -> np.ndarray:
    """Return unit tangent vectors for each point of a polyline."""
    n = points.shape[0]
    if n == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if n == 1:
        return np.array([[1.0, 0.0]], dtype=np.float64)

    tangents = np.empty_like(points)
    for i in range(n):
        if i == 0:
            d = points[1] - points[0]
        elif i == n - 1:
            d = points[-1] - points[-2]
        else:
            d = points[i + 1] - points[i - 1]
        norm = float(np.hypot(d[0], d[1]))
        if norm > 1e-12:
            tangents[i] = d / norm
        else:
            tangents[i] = np.array([1.0, 0.0], dtype=np.float64)
    return tangents


def _tangent_yaws(points: np.ndarray) -> np.ndarray:
    """Compute tangent yaw at each path point from local direction."""
    tangents = _tangent_directions(points)
    return np.arctan2(tangents[:, 1], tangents[:, 0])


def _low_pass_noise_offsets(
    n: int,
    amp: float,
    rng: np.random.Generator,
    fc_hz: float,
    ds: float,
) -> np.ndarray:
    """Generate a bounded, spatially low-pass lateral noise sequence.

    The sequence is a first-order low-pass filtered ``U(-1,1)`` stream, then
    clipped to [-1, 1] and scaled by ``amp``.  ``ds`` is the nominal arc step.
    The cutoff frequency is ``fc_hz`` as configured; no additional moving
    average is applied here (re-smoothing, if needed, is handled by the caller
    after an ``R_min`` rejection).
    """
    if n <= 0:
        return np.zeros((0,), dtype=np.float64)
    alpha = 1.0 - math.exp(-2.0 * math.pi * fc_hz * max(ds, 1e-6))
    u = np.empty(n, dtype=np.float64)
    state = 0.0
    for i in range(n):
        raw = rng.uniform(-1.0, 1.0)
        state += alpha * (raw - state)
        u[i] = float(np.clip(state, -1.0, 1.0))
    # Taper the first/last 20% of the sequence to zero so the noisy path
    # departs from and returns to the original reference smoothly; otherwise a
    # large offset at the endpoints dominates the finite-difference tangent.
    n_taper = max(2, int(n * 0.2))
    taper = np.ones(n, dtype=np.float64)
    ramp = np.sin(np.linspace(0.0, np.pi / 2.0, n_taper))
    taper[:n_taper] = ramp
    taper[-n_taper:] = ramp[::-1]
    return amp * u * taper


def _smooth_offsets_once(offsets: np.ndarray) -> np.ndarray:
    """Apply one 3-tap moving-average pass to a noise offset sequence.

    This is used only as a README-allowed re-smoothing step after an
    ``R_min`` rejection, never inside :func:`_low_pass_noise_offsets`.
    """
    n = offsets.shape[0]
    if n < 3:
        return offsets.copy()
    padded = np.concatenate(([offsets[0]], offsets, [offsets[-1]]))
    return 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]


def _acceptable_noisy_point(
    cand: np.ndarray,
    prev: np.ndarray,
    index: int,
    points: np.ndarray,
    inflated: np.ndarray,
    cfg: PathCfg,
) -> bool:
    """Return whether a candidate noisy point is safe at ``index``.

    The candidate must be free in the inflated grid, the segment from the
    previously accepted point must be clear, and (for the last interior point)
    the segment to the fixed path endpoint must also be clear and locally
    smooth enough to satisfy the final-segment ``R_min`` estimate.
    """
    if not _point_free(cand, inflated):
        return False
    if not _segment_clear(prev, cand, inflated):
        return False
    n = points.shape[0]
    if index == n - 2:
        end = points[-1]
        if not _segment_clear(cand, end, inflated):
            return False
        ds_after = float(np.linalg.norm(end - cand))
        if ds_after <= 1e-12:
            return True
        before = math.atan2(float(cand[1] - prev[1]), float(cand[0] - prev[0]))
        after = math.atan2(float(end[1] - cand[1]), float(end[0] - cand[0]))
        max_dyaw = ds_after / max(cfg.min_turn_radius, 1e-6)
        if abs(wrap_to_pi(after - before)) > max_dyaw + 1e-9:
            return False
    return True


def _restored_prefix_is_clear(
    noisy: np.ndarray,
    points: np.ndarray,
    first_restored: int,
    index: int,
    inflated: np.ndarray,
) -> bool:
    """Return whether restoring ``points[first_restored:index+1]`` is safe.

    The check validates every restored point is free and every segment from
    ``noisy[first_restored - 1]`` through ``points[index]`` is clear.  For the
    last interior point it also validates the segment to the fixed endpoint.

    Args:
        noisy: Current working polyline (indices before ``first_restored`` are
            the already-accepted prefix).
        points: Original resampled reference polyline.
        first_restored: First index restored to the original path.
        index: Current index being placed (inclusive end of restored suffix).
        inflated: Inflated occupancy grid.

    Returns:
        True when the restored suffix is clear from the previous accepted point
        through ``index`` (and to the endpoint when ``index`` is the last
        interior point).
    """
    if first_restored < 1 or index >= points.shape[0] - 1:
        return False
    for k in range(first_restored, index + 1):
        if not _point_free(points[k], inflated):
            return False
        if not _segment_clear(noisy[k - 1], points[k], inflated):
            return False
    if index == points.shape[0] - 2:
        if not _segment_clear(points[index], noisy[-1], inflated):
            return False
    return True


def _apply_path_noise(
    points: np.ndarray,
    inflated: np.ndarray,
    cfg: PathCfg,
    rng: np.random.Generator,
    smoothing_passes: int = 0,
) -> np.ndarray:
    """Apply bounded lateral noise to every path point with rejection sampling.

    Each proposed point must be free in the inflated grid and every segment
    from the previous accepted point must stay clear.  The segment from the
    last noisy interior point to the fixed endpoint is validated as well.
    Invalid proposals fall back to the original point after
    ``cfg.noise_retries`` attempts.  If the original point is not reachable
    from the current prefix, a suffix of earlier noisy points is restored to
    the original polyline; every restored segment is re-validated before being
    accepted, so the returned polyline is always clear.

    Args:
        points: ``(P, 2)`` resampled reference path.
        inflated: 0.35 m inflated occupancy grid.
        cfg: Path configuration.
        rng: Seeded NumPy random generator.
        smoothing_passes: Number of extra moving-average passes applied to the
            noise offsets (used as an ``R_min`` re-smoothing retry).
    """
    if points.shape[0] < 2:
        return points.copy()

    amp = rng.uniform(cfg.noise_amp_range[0], cfg.noise_amp_range[1])
    offsets = _low_pass_noise_offsets(
        points.shape[0], amp, rng, cfg.noise_fc_hz, cfg.resample_dist
    )
    for _ in range(max(0, smoothing_passes)):
        offsets = _smooth_offsets_once(offsets)

    tangents = _tangent_directions(points)
    normals = np.stack((-tangents[:, 1], tangents[:, 0]), axis=1)

    noisy = points.copy()
    prev = points[0].copy()
    for i in range(1, points.shape[0] - 1):
        accepted = False
        for attempt in range(cfg.noise_retries):
            # The first try uses the spatially low-passed offset; later tries
            # redraw a fresh bounded lateral offset for this point.
            off = offsets[i] if attempt == 0 else rng.uniform(-amp, amp)
            cand = points[i] + normals[i] * off
            if not _acceptable_noisy_point(cand, prev, i, points, inflated, cfg):
                continue
            noisy[i] = cand
            prev = noisy[i].copy()
            accepted = True
            break

        if not accepted:
            # Try the original point with the current accepted prefix.
            orig_i = points[i].copy()
            if _acceptable_noisy_point(orig_i, prev, i, points, inflated, cfg):
                noisy[i] = orig_i
                prev = orig_i.copy()
                accepted = True
            else:
                # Restore a suffix of earlier noisy points to the original
                # polyline and re-validate every affected segment.  Trying
                # shorter suffixes first keeps as much accepted noise as
                # possible while guaranteeing a clear prefix.
                restored = False
                for j in range(i, 0, -1):
                    if _restored_prefix_is_clear(noisy, points, j, i, inflated):
                        noisy[j : i + 1] = points[j : i + 1]
                        prev = noisy[i].copy()
                        restored = True
                        break
                if not restored:
                    # Last resort: restore the entire prefix up to i.  The
                    # original resampled A* polyline is clear, so this can only
                    # fail on a corrupted map.
                    noisy[1 : i + 1] = points[1 : i + 1]
                    if not _restored_prefix_is_clear(noisy, points, 1, i, inflated):
                        raise ValueError(
                            "path noise fallback cannot restore a clear polyline"
                        )
                    prev = noisy[i].copy()
    return noisy


def _deduplicate_close_points(
    points: np.ndarray,
    min_spacing: float,
    inflated: np.ndarray,
) -> np.ndarray:
    """Drop near-duplicate points without ever producing a blocked chord.

    Noise fallback can leave nearly coincident points; they add zero-length
    segments and artificial curvature spikes.  A naive distance-only removal is
    unsafe: dropping a point that is close to the last kept point can create a
    longer chord between the last kept point and the next point, and that chord
    may cut through an inflated occupied cell even though the original short
    segments were clear.

    This version only skips a point when the replacement chord
    ``last_kept -> next_point`` is itself clear, so the returned polyline is
    guaranteed to remain clear.
    """
    if points.shape[0] < 2:
        return points

    keep = [0]
    i = 1
    n = points.shape[0]
    while i < n:
        last_kept = points[keep[-1]]
        # Try to drop a near-duplicate only if the skip is geometrically safe.
        if (
            i + 1 < n
            and float(np.linalg.norm(points[i] - last_kept)) < min_spacing
            and _segment_clear(last_kept, points[i + 1], inflated)
        ):
            i += 1
            continue
        keep.append(i)
        i += 1
    return points[np.asarray(keep, dtype=int)]


def _compute_segment_dirs(points: np.ndarray) -> np.ndarray:
    """Return the direction angle of each polyline segment (radians)."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 2:
        return np.zeros((0,), dtype=np.float64)
    d = np.diff(pts, axis=0)
    return np.arctan2(d[:, 1], d[:, 0])


def _detect_corners(
    points: np.ndarray,
    arc: np.ndarray,
    angle_threshold: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect interior corners of a polyline.

    A corner is a vertex whose incoming and outgoing segment directions differ
    by more than ``angle_threshold`` radians.  The returned arrays contain the
    arc length of the corner vertex and the direction of the outgoing segment,
    which is the heading the robot must align to before continuing.
    """
    pts = np.asarray(points, dtype=np.float64)
    arc_arr = np.asarray(arc, dtype=np.float64)
    seg_dirs = _compute_segment_dirs(pts)
    if seg_dirs.shape[0] < 2:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    corner_arcs = []
    corner_targets = []
    for i in range(1, seg_dirs.shape[0]):
        turn = wrap_to_pi(seg_dirs[i] - seg_dirs[i - 1])
        if abs(float(turn)) > angle_threshold:
            corner_arcs.append(float(arc_arr[i]))
            corner_targets.append(float(seg_dirs[i]))
    return (
        np.asarray(corner_arcs, dtype=np.float64),
        np.asarray(corner_targets, dtype=np.float64),
    )


def _smooth_yaws_to_min_radius(
    points: np.ndarray, yaws: np.ndarray, cfg: PathCfg, max_iterations: int = 500
) -> np.ndarray:
    """Smooth tangent yaws until the discrete curvature bound is satisfied.

    A* on an 8-neighbour grid produces right-angle polyline corners whose
    exact tangent curvature is infinite.  M1 is a kinematic surrogate: the
    position stays on the collision-checked reference polyline, while the
    heading/tangent reference is a smoothed version of the finite-difference
    yaw.  Endpoint yaws are pinned to their segment directions; interior yaws
    are smoothed with a 1:2:1 circular mean.  The heading controller still
    clips ``omega`` to ``cfg.omega_max``.
    """
    y = np.asarray(yaws, dtype=np.float64).copy()
    ds = np.linalg.norm(np.diff(points, axis=0), axis=1)
    max_curvature = 1.0 / max(float(cfg.min_turn_radius), 1e-6)

    def _violates(cur_yaws: np.ndarray) -> bool:
        dy = np.abs(wrap_to_pi(np.diff(cur_yaws)))
        for k in range(len(ds)):
            if ds[k] > 1e-9 and dy[k] / ds[k] > max_curvature + 1e-9:
                return True
        return False

    if y.shape[0] == 1:
        return y
    # Pin endpoints to their segment directions: path-noise jitter otherwise
    # leaves large finite-difference yaws at the first/last point that no
    # interior-only smoothing can remove.
    y[0] = np.arctan2(points[1, 1] - points[0, 1], points[1, 0] - points[0, 0])
    y[-1] = np.arctan2(
        points[-1, 1] - points[-2, 1], points[-1, 0] - points[-2, 0]
    )
    if y.shape[0] == 2:
        return y

    for _ in range(max_iterations):
        if not _violates(y):
            return y
        z = np.exp(1j * y)
        z_new = z.copy()
        z_new[1:-1] = 0.25 * z[:-2] + 0.5 * z[1:-1] + 0.25 * z[2:]
        z_new[0] = z[0]
        z_new[-1] = z[-1]
        y = np.angle(z_new)
    return y


def _build_path_data(
    points: np.ndarray,
    cfg: PathCfg,
    inflated: np.ndarray,
    rng: np.random.Generator,
) -> PathData:
    """Resample, noise, smooth tangent yaws and package a :class:`PathData`.

    The geometry (points) is validated against the inflated grid; the tangent
    yaws are the finite-difference tangents smoothed until the discrete
    ``R_min`` curvature bound holds.  Grid right-angle corners are therefore
    tolerated in the reference polyline instead of rejecting every tile-map
    path.
    """
    # A* output may contain many collinear grid waypoints; simplify first.
    simple = _simplify_path(points, inflated)
    resampled, _ = _resample_polyline(simple, cfg.resample_dist)

    if len(resampled) < 2:
        raise ValueError("path contains fewer than two points after resampling")

    smoothing_passes = (0, 1, 2, 4, 8, 16)
    for passes in smoothing_passes:
        for _ in range(_MAX_NOISE_ATTEMPTS):
            noisy = _apply_path_noise(
                resampled, inflated, cfg, rng, smoothing_passes=passes
            )
            min_spacing = max(1e-3, float(cfg.resample_dist) * 0.1)

            # Safety net: if noise ever returned a non-clear polyline, fall back
            # to the clear resampled reference before attempting cleanup.
            if not _path_clear(noisy, inflated):
                noisy = resampled

            # Clearness-aware dedup: by construction it cannot introduce a
            # blocked chord.  Keep a copy for the final invariant below.
            raw_noisy = noisy
            noisy = _deduplicate_close_points(noisy, min_spacing, inflated)
            if noisy.shape[0] < 2:
                continue

            # Final invariant: never return a blocked path.  If cleanup ever
            # produces one, fall back to the pre-dedup clear polyline.
            if not _path_clear(noisy, inflated):
                noisy = raw_noisy
            if not _path_clear(noisy, inflated):
                noisy = resampled
            if not _path_clear(noisy, inflated):
                raise ValueError(
                    "path cleanup could not produce a clear polyline"
                )

            yaws = _smooth_yaws_to_min_radius(
                noisy, _tangent_yaws(noisy), cfg
            )
            noisy_arc = _cumulative_arc_lengths(noisy)
            segment_dirs = _compute_segment_dirs(noisy)
            corner_arcs, corner_targets = _detect_corners(
                noisy, noisy_arc, angle_threshold=0.05
            )
            return PathData(
                points=noisy,
                yaws=yaws,
                arc=noisy_arc,
                segment_dirs=segment_dirs,
                corner_arcs=corner_arcs,
                corner_targets=corner_targets,
            )

    raise ValueError(
        "path noise could not produce a feasible realization; reject and "
        "resample a new path"
    )


def plan_path(
    occupancy: np.ndarray,
    inflated: np.ndarray,
    start_xy: Sequence[float],
    goal_xy: Sequence[float],
    cfg: PathCfg,
    rng: np.random.Generator,
) -> PathData:
    """Plan one feasible reference path for one env episode.

    Pipeline (README 2.2.4-2.2.6):
    A* on the inflated grid -> line-of-sight simplification -> fixed-distance
    resampling -> bounded lateral noise with rejection sampling -> tangent-yaw
    smoothing to the configured ``min_turn_radius`` bound.

    Args:
        occupancy: Raw occupancy grid ``(740, 740)`` uint8 (kept for API
            compatibility; A* safety uses ``inflated``).
        inflated: 0.35 m inflated grid; 1 = blocked.
        start_xy: World-space start ``(x, y)``, must be free in ``inflated``.
        goal_xy: World-space goal ``(x, y)``, must be free in ``inflated``.
        cfg: Frozen path configuration.
        rng: Seeded NumPy random generator for reproducible noise.

    Returns:
        :class:`PathData` with world points ``(P, 2)``, tangent yaws and
        cumulative arc lengths.

    Raises:
        ValueError: If start/goal are blocked, A* has no solution, the path is
            shorter than ``cfg.min_path_len``, or path-noise rejection sampling
            cannot produce a clear polyline.
    """
    if occupancy is None:
        raise ValueError("occupancy grid must be provided")
    if not isinstance(inflated, np.ndarray) or inflated.shape != EA2_GRID_SHAPE:
        raise ValueError(f"inflated grid must have shape {EA2_GRID_SHAPE}")

    # A* itself validates start/goal free cells and connectivity.
    raw_points = _astar(inflated, start_xy, goal_xy)
    if len(raw_points) < 2:
        raise ValueError("A* produced fewer than two points")

    # Endpoint sampling contract: goals must keep clearance from raw obstacles.
    goal_clearance = _min_obstacle_distance_world(occupancy, goal_xy)
    if goal_clearance + 1e-9 < cfg.goal_min_obstacle_dist:
        raise ValueError(
            f"goal {tuple(goal_xy)} is only {goal_clearance:.3f} m from nearest "
            f"obstacle (requires >= {cfg.goal_min_obstacle_dist} m)"
        )

    # Sanity-check the raw A* polyline length before spending time on noise.
    dists = np.linalg.norm(np.diff(np.asarray(raw_points, dtype=np.float64), axis=0), axis=1)
    total_len = float(np.sum(dists))
    if total_len < cfg.min_path_len - 1e-9:
        raise ValueError(
            f"path length {total_len:.3f} m is shorter than cfg.min_path_len "
            f"({cfg.min_path_len} m)"
        )

    data = _build_path_data(raw_points, cfg, inflated, rng)

    # The acceptance criterion is the returned noisy path, not the raw A*
    # polyline: LOS simplification and lateral noise can shorten it.
    if data.arc[-1] < cfg.min_path_len - 1e-9:
        raise ValueError(
            f"returned path length {data.arc[-1]:.3f} m is shorter than "
            f"cfg.min_path_len ({cfg.min_path_len} m)"
        )

    return data




__all__ = [
    "PathData",
    "PathCfg",
    "plan_path",
    "wrap_to_pi",
]
