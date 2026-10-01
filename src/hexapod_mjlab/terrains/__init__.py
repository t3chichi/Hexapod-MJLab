"""Numpy-only terrain system for hexapod_mjlab (M2).

Vendored isaacgym terrain generators + the ported ``TerrainBuilder`` with
mujoco hfield / warp mesh outputs.  See ``builder.py`` and
``_isaacgym_vendored.py`` docstrings for provenance.
"""

from hexapod_mjlab.terrains.builder import (
    TerrainBuilder,
    TerrainConfig,
    TerrainData,
    make_warp_mesh,
)

__all__ = [
    "TerrainBuilder",
    "TerrainConfig",
    "TerrainData",
    "make_warp_mesh",
]
