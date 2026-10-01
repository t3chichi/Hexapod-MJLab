"""Custom terrain sub-terrain: M2 pillar-field heightfield in the official
mjlab terrain generator.

Official mechanism (``docs/source/terrain.rst``): the scene terrain is a
``TerrainEntityCfg(terrain_type="generator")`` whose ``TerrainGeneratorCfg``
assembles a grid of ``SubTerrainCfg`` patches; each sub-terrain's
``function(difficulty, spec, rng)`` writes geometry into the shared
``MjSpec``. The official ``Hf*`` sub-terrains (mjlab/terrains/
heightfield_terrains.py) show the heightfield pattern: min-max normalized
elevation -> ``spec.add_hfield`` -> one ``mjGEOM_HFIELD`` geom.

This module feeds the M2 pillar-field heightfield
(``hexapod_mjlab.terrains.builder.TerrainBuilder``, parity-tested against the old
``legged_gym.utils.terrain.Terrain``) into that mechanism:

* The full M2 field (2x2 sub-terrains of 8x8 m + 5 m flat border, i.e.
  26x26 m) is built ONCE with the verbatim M2 RNG stream (golden parity per
  seed), then served patch-by-patch: generator call #k maps to M2 patch
  (row=k // num_cols, col=k % num_cols) because the official generator's
  random mode iterates patches row-major
  (``TerrainGenerator._generate_random_terrains``).
* Each patch (the borderless 80x80 inner slice) becomes one MuJoCo hfield
  geom placed by the official generator. Elevation data is TRANSPOSED
  before flattening: MuJoCo hfield data rows run along local y while the M2
  heightfield axis 0 runs along x (verified empirically in M2 with mujoco
  3.11; see ``TerrainData.mujoco_hfield_data``).
* The full field (with border) is exposed for the LiDAR warp mesh and for
  the old ``_get_heights`` semantics (min-of-3-neighbors sampling) via
  :func:`get_el_4090_terrain_view`.

World-frame convention: the official generator centers the patch grid at the
world origin (patch (row, col) corner at ``(-rows*size/2 + row*size, ...)``),
so M2 field coordinates (which include the border) map to world coordinates
as ``world = field_xy - offset_xy`` with
``offset_xy = border_size + grid_extent / 2`` (13 m for the default config).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import mujoco
import numpy as np
import torch

from hexapod_mjlab.terrains.builder import TerrainBuilder, TerrainConfig, TerrainData
from mjlab.terrains.terrain_generator import (
  SubTerrainCfg,
  TerrainGeometry,
  TerrainOutput,
)

#: Terrain seed (M2 golden terrain is seed 42).
EL4090_TERRAIN_SEED = 42


def make_el_4090_terrain_config() -> TerrainConfig:
  """M2 TerrainConfig with El4090LidarHeadCfg.terrain semantics.

  Defaults of :class:`TerrainConfig` already equal the old resolved config
  (100% pillar_field, 2x2 sub-terrains of 8x8 m, 5 m border, hs 0.1 / vs
  0.005, pillar params 12 count / 0.5-4.0 m footprint / 1.0-2.0 m height /
  2.2 m separation / 3.0 m center clear / 7.5 m spawn radius).
  """
  return TerrainConfig()


@dataclass(kw_only=True)
class El4090PillarFieldCfg(SubTerrainCfg):
  """Pillar-field sub-terrain backed by the M2 TerrainBuilder heightfield.

  The whole M2 field is generated deterministically on the first
  ``function()`` call (per config instance) and sliced per patch afterwards;
  ``difficulty``/``rng`` are unused because the M2 build already draws
  per-patch difficulties from its verbatim RNG stream (matching the old
  ``Terrain.randomized_terrain`` exactly).
  """

  terrain_seed: int = EL4090_TERRAIN_SEED
  builder_cfg: TerrainConfig = field(default_factory=make_el_4090_terrain_config)

  # Runtime stash (plain attributes, not constructor fields).
  _terrain_data: TerrainData | None = None
  _patch_count: int = 0
  _border_px: int = 0
  _patch_rows_px: int = 0
  _patch_cols_px: int = 0
  _world_offset_xy: tuple[float, float] = (0.0, 0.0)

  def _ensure_built(self) -> None:
    if self._terrain_data is not None:
      return
    td: TerrainData = TerrainBuilder(self.builder_cfg).build(self.terrain_seed)
    self._terrain_data = td
    self._border_px = int(self.builder_cfg.border_size / td.horizontal_scale)
    self._patch_rows_px = int(self.builder_cfg.terrain_length / td.horizontal_scale)
    self._patch_cols_px = int(self.builder_cfg.terrain_width / td.horizontal_scale)
    # field xy -> world xy: subtract border + half grid extent (the official
    # generator centers the *patch grid* at the world origin; the M2 field's
    # patch (0,0) starts at border_size in field coordinates).
    self._world_offset_xy = (
      float(
        self.builder_cfg.border_size
        + self._patch_rows_px * td.horizontal_scale * td.num_sub_rows / 2.0
      ),
      float(
        self.builder_cfg.border_size
        + self._patch_cols_px * td.horizontal_scale * td.num_sub_cols / 2.0
      ),
    )

  @property
  def terrain_data(self) -> TerrainData:
    self._ensure_built()
    assert self._terrain_data is not None
    return self._terrain_data

  @property
  def world_offset_xy(self) -> tuple[float, float]:
    self._ensure_built()
    return self._world_offset_xy

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    del difficulty, rng  # M2 build already consumed the old RNG stream.
    self._ensure_built()
    td = self.terrain_data

    # Official random-mode generator call order is row-major over
    # (num_rows, num_cols); mirror it into the M2 patch grid.
    i, j = divmod(self._patch_count, td.num_sub_cols)
    self._patch_count += 1
    if i >= td.num_sub_rows:
      raise RuntimeError(
        f"El4090PillarFieldCfg got more generator calls ({self._patch_count}) "
        f"than M2 patches ({td.num_sub_rows}x{td.num_sub_cols}); "
        "check TerrainGeneratorCfg.num_rows/num_cols."
      )

    r0 = self._border_px + i * self._patch_rows_px
    r1 = r0 + self._patch_rows_px
    c0 = self._border_px + j * self._patch_cols_px
    c1 = c0 + self._patch_cols_px
    patch_raw = td.heightfield_raw[r0:r1, c0:c1]

    elevation_min = int(patch_raw.min())
    elevation_max = int(patch_raw.max())
    elevation_range = (
      elevation_max - elevation_min if elevation_max != elevation_min else 1
    )
    vertical_scale = td.vertical_scale
    max_physical_height = elevation_range * vertical_scale
    base_thickness = max(max_physical_height, 1e-3)

    normalized = ((patch_raw - elevation_min) / elevation_range).astype(np.float32)
    # MuJoCo hfield rows run along local y (M2 axis 1) -> transpose payload.
    data = np.ascontiguousarray(normalized.T, dtype=np.float32).reshape(-1)

    unique_id = uuid.uuid4().hex
    hfield = spec.add_hfield(
      name=f"hfield_pillar_{unique_id}",
      size=[
        self.size[0] / 2,
        self.size[1] / 2,
        max_physical_height,
        base_thickness,
      ],
      nrow=int(normalized.shape[1]),  # rows along y == M2 cols
      ncol=int(normalized.shape[0]),  # cols along x == M2 rows
      userdata=data.tolist(),
    )

    body = spec.body("terrain")
    hfield_z_offset = elevation_min * vertical_scale
    hfield_geom = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_HFIELD,
      hfieldname=hfield.name,
      pos=[self.size[0] / 2, self.size[1] / 2, hfield_z_offset],
      # 摩擦语义校准（M5a）：priority=1 使地形摩擦整胜出（MuJoCo max-mix 下
      # 高优先级 geom 的摩擦参数完全生效），配合 startup 事件随机化地形滑动
      # 摩擦 U[0.65,1.125] 复现旧栈 PhysX 平均的有效摩擦分布。
      priority=1,
    )

    # Spawn origin: patch center; ground height from the M2 env_origins
    # (old add_terrain_to_map: max over the center 2x2 m platform; the
    # pillar field keeps the center clear so this is the flat ground height).
    origin = np.array([
      self.size[0] / 2,
      self.size[1] / 2,
      float(td.env_origins[i, j, 2]),
    ])

    return TerrainOutput(
      origin=origin,
      geometries=[TerrainGeometry(geom=hfield_geom, hfield=hfield)],
      flat_patches=None,
    )


class El4090TerrainView:
  """View of the built M2 terrain in the mjlab world frame.

  Exposes the full heightfield (with border) in meters, the trimesh arrays,
  and the field->world offset; consumed by the LiDAR warp mesh and the
  height sampling helpers.
  """

  def __init__(self, cfg: El4090PillarFieldCfg):
    cfg._ensure_built()
    self.cfg = cfg
    self.data: TerrainData = cfg.terrain_data
    self.world_offset_xy: tuple[float, float] = cfg.world_offset_xy


def get_el_4090_terrain_view(env) -> El4090TerrainView:
  """Fetch (and cache on the env) the El4090 terrain view from the scene."""
  view = env.__dict__.get("_el_4090_terrain_view")
  if view is not None:
    return view
  terrain = env.scene.terrain
  if terrain is None:
    raise RuntimeError("El4090 task requires the pillar-field generator terrain.")
  gen_cfg = terrain.cfg.terrain_generator
  sub = gen_cfg.sub_terrains.get("pillar_field") if gen_cfg is not None else None
  if not isinstance(sub, El4090PillarFieldCfg):
    raise RuntimeError(
      "Scene terrain generator has no 'pillar_field' El4090PillarFieldCfg."
    )
  view = El4090TerrainView(sub)
  env._el_4090_terrain_view = view
  return view


def sample_terrain_heights_min3(
  view: El4090TerrainView, points_w: torch.Tensor
) -> torch.Tensor:
  """Port of the old ``_get_heights`` sampling (min of 3 nearest samples).

  Old semantics (legged_robot.py::_get_heights): convert world xy to
  heightfield indices by flooring (no bilinear), then take
  ``min(h[px,py], h[px+1,py], h[px,py+1])`` (conservative lower bound),
  clipped to the field bounds.

  Args:
    view: terrain view.
    points_w: world-frame sample positions, shape ``(B, N, 2 or 3)``.

  Returns:
    Terrain heights in meters, shape ``(B, N)``.
  """
  td = view.data
  hs = td.horizontal_scale
  field = torch.as_tensor(td.heightfield, device=points_w.device, dtype=torch.float32)
  offset = torch.tensor(
    view.world_offset_xy, device=points_w.device, dtype=torch.float32
  )
  idx = ((points_w[..., :2] + offset) / hs).long()
  px = idx[..., 0].clamp(0, field.shape[0] - 2)
  py = idx[..., 1].clamp(0, field.shape[1] - 2)
  h1 = field[px, py]
  h2 = field[px + 1, py]
  h3 = field[px, py + 1]
  return torch.minimum(torch.minimum(h1, h2), h3)
