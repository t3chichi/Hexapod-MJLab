"""Versioned task ground-grid recipes and the confirmed current baseline.

S1-187 is the guarded registered-task grid and the task-local formal/Play
CLI default. Explicitly requested alternative coordinates remain supported;
the engineering/formal Python factory contracts still require a complete
GroundGridCfg object. Historic M12 engineering fixtures stay unchanged.
Changing coordinates changes checkpoint identity even when M stays at 187.

S1_187 predicts SIGNED scan-time BODY-z height; it never queries map-world
absolute height. The stable rectangle is based on nine static flat-ground
B8 pose raycasts; it is not guaranteed visible in extreme pitches or
occluded/rough terrain.
"""
from __future__ import annotations

from .ground import (
    DEFAULT_PARALLEL_TOL,
    GroundGridCfg,
    VALIDITY_GEOMETRY_ONLY_FLAT_TRAINING,
)

S1_187_PRESET_NAME = "s1_187"
GROUND_GRID_PRESET_NAMES = (S1_187_PRESET_NAME,)

# ── Human-editable rectangle (scan-time BASE coordinates, metres) ──────────
# Four bounds are all that is needed to edit the rectangular supervision
# region. Keep 17 columns x 11 rows (M=187); the full ordered coordinates are
# calculated automatically below. Current accepted baseline: DO NOT adjust
# these values just to visualize a different rectangle with an old checkpoint.
S1_187_X_RANGE_M = (0.8, 3.2)  # forward: nearest, farthest (accepted S1 baseline)
S1_187_Y_RANGE_M = (-0.5, 0.5)  # lateral: right, left; mirror symmetry REQUIRED
S1_187_NX = 17                 # number of x columns
S1_187_NY = 11                 # number of y rows
GRID_COORD_DECIMALS = 6        # preserve the original checkpoint identity


def ground_grid_from_rectangle(
    *,
    x_range_m: tuple[float, float],
    y_range_m: tuple[float, float],
    nx: int = S1_187_NX,
    ny: int = S1_187_NY,
    parallel_tol: float = DEFAULT_PARALLEL_TOL,
) -> GroundGridCfg:
    """Expand four BASE-xy bounds into an ordered rectangular grid.

    Uses the already-validated GroundGridCfg.from_ranges linspace and then
    rounds each coordinate to 6 decimal places. The rounding is essential:
    otherwise binary float tails change the stored grid identity, even for
    geometrically identical coordinates. The final GroundGridCfg validates
    that the rounded coordinates stay strictly increasing.

    The current symmetric six-legged policy/mirror loss requires y pairs
    about 0. A non-symmetric lateral rectangle is rejected here, rather
    than allowing the runner to fail later while building a permutation.
    """
    raw = GroundGridCfg.from_ranges(
        x_range_m, y_range_m, nx, ny,
        validity_mode=VALIDITY_GEOMETRY_ONLY_FLAT_TRAINING,
        parallel_tol=parallel_tol,
    )
    x_coords = tuple(round(v, GRID_COORD_DECIMALS) for v in raw.x_coords)
    y_coords = tuple(round(v, GRID_COORD_DECIMALS) for v in raw.y_coords)
    if any(y_coords[i] != -y_coords[-1 - i] for i in range(len(y_coords))):
        raise ValueError(
            "ground supervision y range must be symmetric around BASE y=0 "
            "for the current left/right mirror training (y_min = -y_max)"
        )
    return GroundGridCfg(
        x_coords=x_coords,
        y_coords=y_coords,
        validity_mode=VALIDITY_GEOMETRY_ONLY_FLAT_TRAINING,
        parallel_tol=parallel_tol,
    )


def s1_187_ground_grid(*, parallel_tol: float = DEFAULT_PARALLEL_TOL) -> GroundGridCfg:
    """Return the current 17x11 baseline by expanding four rectangle bounds."""
    return ground_grid_from_rectangle(
        x_range_m=S1_187_X_RANGE_M,
        y_range_m=S1_187_Y_RANGE_M,
        nx=S1_187_NX,
        ny=S1_187_NY,
        parallel_tol=parallel_tol,
    )


# Backward-compatible constants for visualization, analysis and tests;
# these are GENERATED from the exact same four bounds as the production grid.
S1_187_X_COORDS = s1_187_ground_grid().x_coords
S1_187_Y_COORDS = s1_187_ground_grid().y_coords


def resolve_explicit_ground_grid(
    *, grid_preset: str | None,
    grid_x: list[float] | tuple[float, ...] | None,
    grid_y: list[float] | tuple[float, ...] | None,
    parallel_tol: float = DEFAULT_PARALLEL_TOL,
) -> GroundGridCfg:
    """Resolve a CLI-supplied grid without changing the explicit contract.

    Either choose --grid-preset s1_187, or provide BOTH --grid-x and --grid-y.
    Mixing the two routes, or silently using a default, is forbidden.
    """
    if grid_preset is not None:
        if grid_x is not None or grid_y is not None:
            raise ValueError("grid preset must not be combined with --grid-x/--grid-y")
        if grid_preset != S1_187_PRESET_NAME:
            raise ValueError(f"unknown grid preset {grid_preset!r}")
        return s1_187_ground_grid(parallel_tol=parallel_tol)
    if grid_x is None or grid_y is None:
        raise ValueError("explicit --grid-x AND --grid-y or --grid-preset s1_187 required")
    return GroundGridCfg(
        x_coords=tuple(grid_x),
        y_coords=tuple(grid_y),
        validity_mode=VALIDITY_GEOMETRY_ONLY_FLAT_TRAINING,
        parallel_tol=parallel_tol,
    )


def resolve_task_ground_grid(
    *, grid_preset: str | None,
    grid_x: list[float] | tuple[float, ...] | None,
    grid_y: list[float] | tuple[float, ...] | None,
    parallel_tol: float = DEFAULT_PARALLEL_TOL,
) -> GroundGridCfg:
    """Resolve the current formal/Play CLI grid; explicit overrides are strict.

    No selector => S1-187 current task baseline (NOT automatic authorization
    to train). A custom grid requires BOTH ordered axes and its own checkpoint
    identity. A named preset cannot be mixed with custom axes.
    """
    if grid_preset is None and grid_x is None and grid_y is None:
        return s1_187_ground_grid(parallel_tol=parallel_tol)
    return resolve_explicit_ground_grid(
        grid_preset=grid_preset,
        grid_x=grid_x,
        grid_y=grid_y,
        parallel_tol=parallel_tol,
    )
