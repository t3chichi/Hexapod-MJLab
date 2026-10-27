"""Unregistered, geometry-only Heightfield environment for Terrain Memory.

This is a finite engineering environment, not a Terrain Memory training recipe.
Robot, PD and proprioception construction reuse the existing implementation.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.terrain_generator import (
    SubTerrainCfg, TerrainGeneratorCfg, TerrainGeometry, TerrainOutput,
)

from hexapod_mjlab.common.flat_env import FLAT_GROUND_FRICTION, make_flat_env_cfg

TERRAIN_GROUP = 3
SURFACE_NAME = "terrain_memory_surface"


@dataclass(kw_only=True)
class HeightfieldTerrainCfg(SubTerrainCfg):
    """One static surface; elevations are metres, ordered [y row, x column].

    ``spawn_origin`` is a caller-validated surface point relative to the field
    centre. The existing witness reset is translated to this point. Its actual
    collision clearance must be checked; this does not solve arbitrary spawns.
    """

    elevations_m: tuple[tuple[float, ...], ...]
    spawn_origin: tuple[float, float, float]

    def __post_init__(self):
        heights = np.asarray(self.elevations_m, dtype=np.float64)
        if heights.ndim != 2 or min(heights.shape) < 2 or not np.isfinite(heights).all():
            raise ValueError("elevations_m must be a finite >=2x2 [y,x] table")
        if len(self.size) != 2 or not np.isfinite(self.size).all() or min(self.size) <= 0:
            raise ValueError("heightfield size must contain two positive finite lengths")
        if len(self.spawn_origin) != 3 or not np.isfinite(self.spawn_origin).all():
            raise ValueError("spawn_origin must be a finite surface point")
        if abs(self.spawn_origin[0]) >= self.size[0] / 2 or abs(self.spawn_origin[1]) >= self.size[1] / 2:
            raise ValueError("spawn_origin must be inside the heightfield")

    def function(self, difficulty, spec, rng):
        del difficulty, rng
        self.__post_init__()
        heights = np.asarray(self.elevations_m, dtype=np.float64)
        floor = float(heights.min())
        span = max(float(heights.max()) - floor, 1e-3)
        field = spec.add_hfield(
            name="terrain_memory_hfield",
            size=[self.size[0] / 2, self.size[1] / 2, span, 0.1],
            nrow=heights.shape[0], ncol=heights.shape[1],
            userdata=((heights - floor) / span).ravel().tolist(),
        )
        geom = spec.body("terrain").add_geom(
            name=SURFACE_NAME, type=mujoco.mjtGeom.mjGEOM_HFIELD,
            hfieldname=field.name,
            pos=[self.size[0] / 2, self.size[1] / 2, floor],
            group=TERRAIN_GROUP, priority=1, friction=list(FLAT_GROUND_FRICTION),
        )
        origin = np.asarray(self.spawn_origin) + [self.size[0] / 2, self.size[1] / 2, 0]
        return TerrainOutput(origin=origin, geometries=[TerrainGeometry(geom=geom, hfield=field)])


def make_geometry_env_cfg(*, terrain: HeightfieldTerrainCfg, num_envs: int, seed: int):
    """Build real non-flat geometry, without rewards, learner or task registration."""
    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
        raise ValueError("num_envs must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be a uint32 integer")
    cfg = make_flat_env_cfg(num_envs=num_envs)
    cfg.seed, cfg.auto_reset = seed, False
    # mjlab refreshes the compiled initial pose before the first reset. Keep
    # that bootstrap pose above this field; the explicit reset still writes
    # the actual witness pose relative to the caller-validated spawn platform.
    cfg.scene.entities["robot"].init_state.pos = (
        terrain.spawn_origin[0], terrain.spawn_origin[1],
        float(np.max(terrain.elevations_m)) + 0.8,
    )
    cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=seed, size=terrain.size, num_rows=1, num_cols=1,
            border_width=0.0, curriculum=False, color_scheme="none",
            sub_terrains={"terrain_memory_geometry": terrain},
        ),
        max_init_terrain_level=None,
    )
    return cfg
