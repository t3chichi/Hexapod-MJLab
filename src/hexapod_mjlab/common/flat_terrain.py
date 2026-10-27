"""Single infinite-plane terrain config for the shared flat environment.

Kept separate from ``common.flat_env`` so that importing the flat environment
module does not trigger mjlab's external task entry-point discovery while the
module itself is still initializing.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import (
  SubTerrainCfg,
  TerrainGeometry,
  TerrainOutput,
)

from hexapod_mjlab.common.flat_env import FLAT_GROUND_FRICTION


@dataclass(kw_only=True)
class SinglePlaneSubTerrainCfg(SubTerrainCfg):
  """One ``mjGEOM_PLANE`` geom: a single horizontal INFINITE ground.

  MuJoCo plane collision is infinite in extent; the half-extent only sizes
  the rendered patch. No heightfield, no mesh, no old finite terrain field:
  the only environment geometry is this plane with the confirmed friction
  (1.0, 0.005, 0.0001). priority=1 makes the plane's friction win the
  max-mix combine against robot geoms (same mechanism as the old flat pin).
  """

  friction: tuple[float, float, float] = FLAT_GROUND_FRICTION
  plane_half_extent: float = 100.0
  _built: bool = False

  def function(self, difficulty, spec, rng):
    if self._built:
      raise RuntimeError(
        "SinglePlaneSubTerrainCfg.function called twice; the flat config "
        "must contain exactly one ground plane"
      )
    self._built = True
    geo = spec.body("terrain").add_geom(
      name="confined_flat_ground",
      type=mujoco.mjtGeom.mjGEOM_PLANE,
      size=[self.plane_half_extent, self.plane_half_extent, 0.1],
      priority=1,
      friction=list(self.friction),
    )
    return TerrainOutput(
      origin=np.array([self.size[0] / 2.0, self.size[1] / 2.0, 0.0]),
      geometries=[TerrainGeometry(geom=geo)],
      flat_patches=None,
    )
