"""Deterministic seeded pillar-field map generator for envelope_adaptive_2.

This module ports ``pd_gru_lidar``'s ``pillar_field_terrain``
(``legged_gym/legged_gym/utils/terrain.py``) into the EA2 warp-mesh pipeline:

* one global fixed map split into ``n_tiles x n_tiles`` tiles of
  ``tile_size_m`` with ``border_size_m`` outer border (pd_gru default:
  4 x 4 tiles of 16m x 16m, 5m border -> 74m x 74m);
* every tile independently runs the pd_gru random-cuboid generator
  (long/short side split, ring placement, AABB + min-separation rejection);
* all cuboids are axis-aligned; there are no boundary walls;
* occupancy is the authoritative geometry and the warp mesh is built from the
  same box primitives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from .contracts import (
    EA2_TILE_PILLAR,
    MapData,
    MapGenCfg,
    PillarFieldCfg,
    RectPrimitive,
)

_GROUND_THICKNESS_M = 0.02
_NEAR_OBSTACLE_SAMPLE_CAP = 2000


@dataclass(frozen=True)
class _Mesh:
    vertices: np.ndarray
    triangles: np.ndarray


def _world_to_grid(x: float, y: float, cfg: MapGenCfg) -> Tuple[int, int]:
    half = cfg.size_m / 2.0
    ix = int(np.floor((x + half) / cfg.resolution_m))
    iy = int(np.floor((y + half) / cfg.resolution_m))
    return ix, iy


# ---------------------------------------------------------------------------
# pd_gru pillar_field port
# ---------------------------------------------------------------------------


def _generate_pillar_field_tile(
    rng: np.random.Generator,
    cfg: MapGenCfg,
    pillar_cfg: PillarFieldCfg,
    tile_center: Tuple[float, float],
    difficulty: float,
) -> List[RectPrimitive]:
    """Generate one pillar-field tile with pd_gru semantics (continuous)."""
    count_min = int(pillar_cfg.count_min)
    count_max = int(pillar_cfg.count_max)
    count = int(count_min + difficulty * (count_max - count_min))
    if count <= 0:
        return []

    clear_radius = float(pillar_cfg.center_clear_radius)
    spawn_radius = float(pillar_cfg.spawn_radius)
    margin = float(pillar_cfg.min_separation) / 2.0
    half_tile = cfg.tile_size_m / 2.0

    # Pre-generate all sizes with pd_gru's long/short side split.
    sizes = []
    for _ in range(count):
        split = float(rng.uniform(0.2, 0.8))
        x_range = float(pillar_cfg.size_x_max - pillar_cfg.size_x_min)
        y_range = float(pillar_cfg.size_y_max - pillar_cfg.size_y_min)
        if rng.random() > 0.5:
            long_min = float(pillar_cfg.size_x_min) + split * x_range
            sx = long_min + float(rng.uniform(0.0, 1.0)) * (
                float(pillar_cfg.size_x_max) - long_min
            )
            short_max = float(pillar_cfg.size_y_min) + split * y_range
            sy = float(pillar_cfg.size_y_min) + float(rng.uniform(0.0, 1.0)) * (
                short_max - float(pillar_cfg.size_y_min)
            )
        else:
            long_min = float(pillar_cfg.size_y_min) + split * y_range
            sy = long_min + float(rng.uniform(0.0, 1.0)) * (
                float(pillar_cfg.size_y_max) - long_min
            )
            short_max = float(pillar_cfg.size_x_min) + split * x_range
            sx = float(pillar_cfg.size_x_min) + float(rng.uniform(0.0, 1.0)) * (
                short_max - float(pillar_cfg.size_x_min)
            )
        sizes.append((sx, sy))

    placed: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    max_attempts_per = max(1, count * 100)
    for sx, sy in sizes:
        if sx >= cfg.tile_size_m or sy >= cfg.tile_size_m:
            continue
        hx = sx / 2.0
        hy = sy / 2.0
        for _ in range(max_attempts_per):
            r = float(rng.uniform(clear_radius, spawn_radius))
            theta = float(rng.uniform(0.0, 2.0 * math.pi))
            cx = tile_center[0] + r * math.cos(theta)
            cy = tile_center[1] + r * math.sin(theta)

            # pd_gru keeps the whole cuboid inside its own tile.
            if (
                cx - hx < tile_center[0] - half_tile
                or cx + hx > tile_center[0] + half_tile
                or cy - hy < tile_center[1] - half_tile
                or cy + hy > tile_center[1] + half_tile
            ):
                continue
            # Center clearance (pd_gru: keep a clear circle around tile center).
            if math.hypot(cx - tile_center[0], cy - tile_center[1]) < clear_radius:
                continue

            # AABB + margin overlap check against already placed cuboids.
            overlaps = False
            for (px, py), (psx, psy) in placed:
                if (
                    abs(cx - px) < (hx + psx / 2.0 + margin)
                    and abs(cy - py) < (hy + psy / 2.0 + margin)
                ):
                    overlaps = True
                    break
            if overlaps:
                continue

            placed.append(((cx, cy), (sx, sy)))
            break

    rects: List[RectPrimitive] = []
    for (cx, cy), (sx, sy) in placed:
        h_target = float(
            rng.uniform(float(pillar_cfg.height_min), float(pillar_cfg.height_max))
        )
        if bool(pillar_cfg.allow_height_variation):
            height = float(rng.uniform(0.6 * h_target, h_target))
        else:
            height = h_target
        rects.append(
            RectPrimitive(
                center=(cx, cy),
                size=(sx, sy),
                yaw=0.0,
                height=height,
            )
        )
    return rects


def _tile_center(cfg: MapGenCfg, row: int, col: int) -> Tuple[float, float]:
    start = -cfg.size_m / 2.0 + cfg.border_size_m
    x = start + (float(col) + 0.5) * cfg.tile_size_m
    y = start + (float(row) + 0.5) * cfg.tile_size_m
    return x, y


def _generate_primitive_set(
    cfg: MapGenCfg,
    pillar_cfg: PillarFieldCfg,
    rng: np.random.Generator,
) -> Tuple[List[RectPrimitive], np.ndarray, dict]:
    tile_types = np.full(
        (cfg.n_tiles, cfg.n_tiles), EA2_TILE_PILLAR, dtype=np.uint8
    )
    rects: List[RectPrimitive] = []
    placed_counts = []
    for row in range(cfg.n_tiles):
        for col in range(cfg.n_tiles):
            # pd_gru randomized_terrain difficulty selection.
            difficulty = float(rng.choice([0.5, 0.75, 0.9]))
            center = _tile_center(cfg, row, col)
            tile_rects = _generate_pillar_field_tile(
                rng, cfg, pillar_cfg, center, difficulty
            )
            placed_counts.append(len(tile_rects))
            rects.extend(tile_rects)

    counts = {
        "n_tiles": float(cfg.n_tiles * cfg.n_tiles),
        "n_rects": float(len(rects)),
        "tiles_with_pillars": float(sum(1 for n in placed_counts if n > 0)),
        "min_rects_per_tile": float(min(placed_counts)),
        "max_rects_per_tile": float(max(placed_counts)),
    }
    return rects, tile_types, counts


# ---------------------------------------------------------------------------
# Rasterization / inflation
# ---------------------------------------------------------------------------


def _rasterize_rect(rect: RectPrimitive, cfg: MapGenCfg) -> np.ndarray:
    world_min = -cfg.size_m / 2.0
    res = cfg.resolution_m
    c = math.cos(rect.yaw)
    s = math.sin(rect.yaw)
    half_x = rect.size[0] / 2.0
    half_y = rect.size[1] / 2.0
    extent_x = abs(c) * half_x + abs(s) * half_y
    extent_y = abs(s) * half_x + abs(c) * half_y

    ix_min = max(0, int(np.floor((rect.center[0] - extent_x - world_min) / res)))
    ix_max = min(
        cfg.grid_shape[1] - 1,
        int(np.floor((rect.center[0] + extent_x - world_min) / res)),
    )
    iy_min = max(0, int(np.floor((rect.center[1] - extent_y - world_min) / res)))
    iy_max = min(
        cfg.grid_shape[0] - 1,
        int(np.floor((rect.center[1] + extent_y - world_min) / res)),
    )
    if ix_min > ix_max or iy_min > iy_max:
        return np.zeros(cfg.grid_shape, dtype=np.uint8)

    xs = world_min + (np.arange(ix_min, ix_max + 1) + 0.5) * res
    ys = world_min + (np.arange(iy_min, iy_max + 1) + 0.5) * res
    gx, gy = np.meshgrid(xs, ys)
    dx = gx - rect.center[0]
    dy = gy - rect.center[1]
    local_x = c * dx + s * dy
    local_y = -s * dx + c * dy
    mask = (np.abs(local_x) <= half_x) & (np.abs(local_y) <= half_y)

    out = np.zeros(cfg.grid_shape, dtype=np.uint8)
    out[iy_min : iy_max + 1, ix_min : ix_max + 1] = mask.astype(np.uint8)
    return out


def _rasterize_occupancy(
    rects: Sequence[RectPrimitive], cfg: MapGenCfg
) -> np.ndarray:
    occupancy = np.zeros(cfg.grid_shape, dtype=np.uint8)
    for rect in rects:
        occupancy |= _rasterize_rect(rect, cfg)
    return occupancy


def _inflate_occupancy(occupancy: np.ndarray, cfg: MapGenCfg) -> np.ndarray:
    from scipy.ndimage import binary_dilation

    if cfg.inflation_cells <= 0:
        inflated = occupancy.copy()
    else:
        radius = int(cfg.inflation_cells)
        structure = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
        inflated = binary_dilation(
            occupancy.astype(bool), structure=structure, border_value=1
        ).astype(np.uint8)
    if cfg.boundary_occupied:
        inflated[0, :] = 1
        inflated[-1, :] = 1
        inflated[:, 0] = 1
        inflated[:, -1] = 1
    return inflated


# ---------------------------------------------------------------------------
# Mesh builders
# ---------------------------------------------------------------------------


def _build_box(
    center: Tuple[float, float, float],
    size: Tuple[float, float],
    yaw: float,
    height: float,
    z_min: float = 0.0,
) -> _Mesh:
    sx, sy = size
    local = np.array(
        [
            [-sx / 2.0, -sy / 2.0, 0.0],
            [sx / 2.0, -sy / 2.0, 0.0],
            [-sx / 2.0, sy / 2.0, 0.0],
            [sx / 2.0, sy / 2.0, 0.0],
            [-sx / 2.0, -sy / 2.0, height],
            [sx / 2.0, -sy / 2.0, height],
            [-sx / 2.0, sy / 2.0, height],
            [sx / 2.0, sy / 2.0, height],
        ],
        dtype=np.float32,
    )
    c = math.cos(yaw)
    s = math.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    xy = local[:, :2] @ rot.T
    vertices = np.empty_like(local)
    vertices[:, :2] = xy + np.asarray(center[:2], dtype=np.float32)
    vertices[:, 2] = local[:, 2] + center[2] + z_min

    faces = [
        (0, 2, 3, 1),
        (4, 5, 7, 6),
        (0, 1, 5, 4),
        (2, 6, 7, 3),
        (0, 4, 6, 2),
        (1, 3, 7, 5),
    ]
    triangles: List[Tuple[int, int, int]] = []
    for a, b, c_i, d in faces:
        triangles.append((a, b, c_i))
        triangles.append((a, c_i, d))
    return _Mesh(vertices=vertices, triangles=np.asarray(triangles, dtype=np.int32))


def _build_ground_mesh(cfg: MapGenCfg) -> _Mesh:
    extent = cfg.size_m / 2.0 + cfg.ground_margin_m
    return _build_box(
        center=(0.0, 0.0, 0.0),
        size=(2.0 * extent, 2.0 * extent),
        yaw=0.0,
        height=_GROUND_THICKNESS_M,
        z_min=-_GROUND_THICKNESS_M,
    )


def _build_mesh(
    rects: Sequence[RectPrimitive], cfg: MapGenCfg
) -> Tuple[np.ndarray, np.ndarray]:
    meshes: List[_Mesh] = [_build_ground_mesh(cfg)]
    for rect in rects:
        meshes.append(
            _build_box(
                center=(rect.center[0], rect.center[1], 0.0),
                size=rect.size,
                yaw=rect.yaw,
                height=rect.height,
                z_min=0.0,
            )
        )

    vertex_list: List[np.ndarray] = []
    triangle_list: List[np.ndarray] = []
    offset = 0
    for mesh in meshes:
        vertex_list.append(mesh.vertices)
        triangle_list.append(mesh.triangles + offset)
        offset += mesh.vertices.shape[0]
    vertices = np.concatenate(vertex_list, axis=0).astype(np.float32)
    triangles = np.concatenate(triangle_list, axis=0).astype(np.int32)
    return vertices, triangles


# ---------------------------------------------------------------------------
# Acceptance
# ---------------------------------------------------------------------------


def _border_mask(shape: Tuple[int, ...]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _estimate_near_obstacle_ratio(
    occupancy: np.ndarray,
    inflated: np.ndarray,
    cfg: MapGenCfg,
    rng: np.random.Generator,
) -> float:
    from scipy.ndimage import distance_transform_edt

    free_mask = occupancy == 0
    dist_to_obstacle = distance_transform_edt(
        free_mask, sampling=(cfg.resolution_m, cfg.resolution_m)
    )
    safe = (inflated == 0) & free_mask
    safe_idx = np.argwhere(safe)
    if safe_idx.shape[0] == 0:
        return 0.0
    n_sample = min(_NEAR_OBSTACLE_SAMPLE_CAP, safe_idx.shape[0])
    chosen = rng.choice(safe_idx.shape[0], size=n_sample, replace=False)
    sample_dist = dist_to_obstacle[safe_idx[chosen, 0], safe_idx[chosen, 1]]
    lo, hi = cfg.near_obstacle_range
    return float(np.mean((sample_dist >= lo) & (sample_dist <= hi)))


def _build_acceptance(
    occupancy: np.ndarray,
    inflated: np.ndarray,
    counts: dict,
    tile_types: np.ndarray,
    rects: Sequence[RectPrimitive],
    cfg: MapGenCfg,
    rng: np.random.Generator,
) -> dict:
    border = _border_mask(occupancy.shape)
    occupancy_border_free = float(not bool(np.any(occupancy[border] == 1)))
    planning_border_blocked = float(
        bool(np.all(inflated[border] == 1)) if cfg.boundary_occupied else 1.0
    )
    all_axis_aligned = float(
        all(abs(rect.yaw % (math.pi / 2.0)) < 1e-6 for rect in rects)
    )
    near_obstacle_ratio = _estimate_near_obstacle_ratio(
        occupancy, inflated, cfg, rng
    )

    from scipy.ndimage import label as _label
    from scipy.ndimage import sum as _nd_sum

    free = inflated == 0
    labels, n_components = _label(free)
    if n_components == 0:
        largest_free_component_ratio = 0.0
    else:
        sizes = _nd_sum(free, labels, index=range(1, n_components + 1))
        largest_free_component_ratio = float(sizes.max() / max(1, int(free.sum())))

    acceptance: dict = {
        "occupancy_border_free": occupancy_border_free,
        "planning_border_blocked": planning_border_blocked,
        "physical_boundary_walls": 0.0,
        "all_rects_axis_aligned": all_axis_aligned,
        "largest_free_component_ratio": largest_free_component_ratio,
        "n_tiles": float(counts["n_tiles"]),
        "tile_types": float(tile_types.astype(np.float32).mean()),
        "n_rects": float(counts["n_rects"]),
        "tiles_with_pillars": float(counts["tiles_with_pillars"]),
        "min_rects_per_tile": float(counts["min_rects_per_tile"]),
        "max_rects_per_tile": float(counts["max_rects_per_tile"]),
        "near_obstacle_ratio_estimate": near_obstacle_ratio,
        "near_obstacle_ratio": near_obstacle_ratio,
        "path_near_obstacle_ratio": near_obstacle_ratio,
    }
    return acceptance


def _static_acceptance(
    occupancy: np.ndarray, acceptance: dict, cfg: MapGenCfg
) -> bool:
    if acceptance["occupancy_border_free"] != 1.0:
        return False
    if acceptance["planning_border_blocked"] != 1.0:
        return False
    if acceptance["all_rects_axis_aligned"] != 1.0:
        return False
    if acceptance["largest_free_component_ratio"] < cfg.min_free_component_ratio:
        return False
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_map(
    cfg: MapGenCfg, pillar_cfg: PillarFieldCfg, seed: int
) -> MapData:
    """Generate a deterministic pillar-field map from ``cfg`` and ``seed``."""
    rng = np.random.default_rng(seed)

    for _ in range(max(1, cfg.max_gen_attempts)):
        rects, tile_types, counts = _generate_primitive_set(
            cfg, pillar_cfg, rng
        )
        occupancy = _rasterize_occupancy(rects, cfg)
        inflated = _inflate_occupancy(occupancy, cfg)
        acceptance = _build_acceptance(
            occupancy, inflated, counts, tile_types, rects, cfg, rng
        )
        if _static_acceptance(occupancy, acceptance, cfg):
            break
    else:
        raise RuntimeError(
            "map_generator: failed to generate a statically acceptable map "
            f"after {cfg.max_gen_attempts} attempts"
        )

    vertices, triangles = _build_mesh(rects, cfg)

    from scipy.ndimage import distance_transform_edt

    distance_field = distance_transform_edt(
        occupancy == 0,
        sampling=(cfg.resolution_m, cfg.resolution_m),
    ).astype(np.float32)

    return MapData(
        occupancy=occupancy,
        inflated=inflated,
        vertices=vertices,
        triangles=triangles,
        distance_field=distance_field,
        tile_types=tile_types,
        rects=tuple(rects),
        pillars=(),
        acceptance=acceptance,
    )
