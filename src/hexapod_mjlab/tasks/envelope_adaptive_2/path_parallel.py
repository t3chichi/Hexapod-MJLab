"""Fork-based parallel path planning workers for ``el4090_ea2``.

The worker processes are created with the ``fork`` context after Isaac Gym /
CUDA has already been initialized in the parent.  They only run CPU A* path
planning and never touch Isaac/CUDA, so they can safely share the parent's
already-loaded ``path_planner`` module.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .contracts import PathCfg
from .path_planner import plan_path

# Worker-global state, set once per worker by ``init_worker``.
_OCC: Optional[np.ndarray] = None
_INFL: Optional[np.ndarray] = None
_CFG: Optional[PathCfg] = None


def init_worker(occupied: np.ndarray, inflated: np.ndarray, cfg: PathCfg) -> None:
    """Store the fixed map and path config in each worker process."""
    global _OCC, _INFL, _CFG
    _OCC = occupied
    _INFL = inflated
    _CFG = cfg


def plan_path_task(
    seed: int,
    start_xy: Tuple[float, float],
    goal_xy: Tuple[float, float],
):
    """Plan one path in a worker process.

    The worker must have been initialized first via :func:`init_worker`.
    Returns the ``PathData`` produced by ``path_planner.plan_path``.
    """
    if _OCC is None or _INFL is None or _CFG is None:
        raise RuntimeError("path_parallel worker was not initialized")
    rng = np.random.default_rng(int(seed))
    return plan_path(_OCC, _INFL, start_xy, goal_xy, _CFG, rng)
