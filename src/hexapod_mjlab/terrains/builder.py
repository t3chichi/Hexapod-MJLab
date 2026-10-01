# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

"""Numpy-only terrain builder (M2 migration: Lidar_legged_gym -> hexapod_mjlab).

Port of ``legged_gym/legged_gym/utils/terrain.py::Terrain`` (source: dirty
working tree of commit 888e493, see MIGRATION.md).  The old class depends on
``isaacgym.terrain_utils``; here that dependency is replaced by the vendored
subset in :mod:`hexapod_mjlab.terrains._isaacgym_vendored`.  No isaacgym / rsl /
legged_gym imports remain -- everything is numpy + scipy(interpolation only,
inside the vendored module).

Configuration source of truth: the ``terrain`` section of
``legged_gym.envs.el_4090.pd_gru_lidar_head.El4090LidarHeadCfg`` (plus parent
values it inherits), resolved as:

===========================  =============================
mesh_type                    'trimesh'
curriculum / selected        False / False  -> randomized_terrain
terrain_length x width       8 m x 8 m, num_rows=num_cols=2 (2x2 subterrains)
border_size                  5 m
horizontal/vertical scale    0.1 m / 0.005 m
slope_treshold               0.75
terrain_proportions          [0]*7 + [1.0] + [0]  (100% pillar_field, idx 7)
difficulty_scale             1.0
pillar_count_min/max         12 / 12
pillar_size_x_min/max        0.5 / 4.0 m
pillar_size_y_min/max        0.5 / 4.0 m
pillar_height_min/max        1.00 / 2.00 m
pillar_min_separation        2.2 m
pillar_center_clear_radius   3.0 m
pillar_spawn_radius          7.5 m
pillar_allow_height_variation True
===========================  =============================

:class:`TerrainConfig` defaults reproduce this table exactly, so
``TerrainBuilder(TerrainConfig()).build(seed)`` is deterministic per seed and
bit-identical to the old pipeline for the same seed (golden parity test:
``tests/terrain/test_terrain_parity.py``).

Port-time deviations from the old ``Terrain`` (judgement calls, none affect
the heightfield for the parity config):

1. ``build(seed)`` seeds the global ``np.random`` (old class relied on the
   caller's ``set_seed``); all RNG *call ordering* is kept verbatim so the
   generated stream matches the old code bit-for-bit.
2. ``mesh_type`` 'none'/'plane' raises ``ValueError`` (old class silently
   returned an unusable object); the builder must always produce a heightfield.
3. The config object is never mutated (old class wrote
   ``cfg.num_sub_terrains`` back into the config).
4. ``selected_terrain`` looks generators up in a registry instead of
   ``eval``-ing a name, and fixes the latent attribute bugs of the original
   (``self.vertical_scale`` / ``length=self.width_per_env_pixels`` ->
   ``cfg.vertical_scale`` / ``length_per_env_pixels``).  Dead code of the old
   module (``curved_corridor_terrain``, ``trapezoid_corridor_terrain``,
   ``_draw_circle`` -- not referenced by ``make_terrain``) was not ported.
5. Extra outputs beyond the old class: float32 heightfield in meters, mujoco
   ``<hfield>`` fields (normalized data + size tuple) and a flattened 1-D
   int32 triangle index copy (``warp.Mesh`` requires 1-D indices).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from . import _isaacgym_vendored as terrain_utils


@dataclass
class TerrainConfig:
    """Resolved terrain configuration (defaults == El4090LidarHeadCfg.terrain).

    Any duck-typed object exposing the same attributes is accepted by
    :class:`TerrainBuilder`; pillar_* and channel_* fields are read with
    ``getattr`` defaults exactly like the old code.
    """

    mesh_type: str = "trimesh"
    horizontal_scale: float = 0.1  # [m]
    vertical_scale: float = 0.005  # [m]
    border_size: float = 5.0  # [m]
    curriculum: bool = False
    selected: bool = False
    terrain_length: float = 8.0
    terrain_width: float = 8.0
    num_rows: int = 2
    num_cols: int = 2
    # [smooth slope, rough slope, stairs up, stairs down, discrete obstacles,
    #  stepping stones, gap, pillar_field, sin_curve_channel]
    terrain_proportions: list = field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    )
    difficulty_scale: float = 1.0
    terrain_kwargs: Any = None  # Dict of arguments for selected terrain
    # trimesh only:
    slope_treshold: float = 0.75  # slopes above this become vertical surfaces
    # stepping stones (from LeggedRobotCfg.terrain):
    stepping_stones_size: float = 0.5
    stepping_stones_distance: float = 1.0
    stepping_stones_max_height: float = 0.2
    stepping_stones_platform_size: float = 3.0
    # pillar field (getattr defaults of the old pillar_field_terrain kept):
    pillar_count_min: int = 12
    pillar_count_max: int = 12
    pillar_size_x_min: float = 0.5
    pillar_size_x_max: float = 4.0
    pillar_size_y_min: float = 0.5
    pillar_size_y_max: float = 4.0
    pillar_height_min: float = 1.00
    pillar_height_max: float = 2.00
    pillar_min_separation: float = 2.2
    pillar_center_clear_radius: float = 3.0
    pillar_spawn_radius: float = 7.5
    pillar_allow_height_variation: bool = True


@dataclass
class TerrainData:
    """Output of :meth:`TerrainBuilder.build`.

    Array conventions (identical to the old code):
      * heightfield axis 0 == world x (row), axis 1 == world y (col);
        pixel (i, j) center at world ``(i * x_resolution, j * y_resolution)``.
      * ``vertices[:, 0] == x``, ``vertices[:, 1] == y`` (z up).
    """

    heightfield_raw: np.ndarray  # int16 (nrow, ncol), raw units (vs multiples)
    heightfield: np.ndarray  # float32 (nrow, ncol), meters (raw * vertical_scale)
    vertices: np.ndarray  # float32 (N, 3), meters
    triangles: np.ndarray  # int32 (M, 3)
    triangles_flat: np.ndarray  # int32 (3*M,) flattened copy for wp.Mesh
    nrow: int  # heightfield rows (x direction)
    ncol: int  # heightfield cols (y direction)
    x_resolution: float  # meters per row
    y_resolution: float  # meters per col
    size_x: float  # total x extent [m] == nrow * x_resolution
    size_y: float  # total y extent [m] == ncol * y_resolution
    num_sub_rows: int
    num_sub_cols: int
    env_origins: np.ndarray  # (num_sub_rows, num_sub_cols, 3), meters
    horizontal_scale: float
    vertical_scale: float
    # mujoco <hfield> mapping (verified empirically against mujoco 3.11):
    #   * the loader MIN-MAX normalizes elevation values itself -- our data is
    #     already min-max normalized, so values pass through unchanged;
    #   * world height of a value s is ``pos_z + s * elevation_z``; the geom
    #     origin therefore sits at the field's minimum world height;
    #   * ``base_z`` is only the depth of the solid skirt below the origin
    #     (must be strictly positive, does not affect the surface height);
    #   * the FIRST data index (rows) runs along MuJoCo y, the SECOND (cols)
    #     along x -- the TRANSPOSE of our axis-0=x convention. Rays
    #     (mj_ray/mj_multiRay) and collision both agree on this layout
    #     (verified: mujoco surface at world (x, y) == numpy bilinear of
    #     heightfield at [x/hs, y/hs] only after transposing the payload).
    hfield_base_z: float  # == min world height [m]
    hfield_elevation_z: float  # == max - min world height [m]

    # MJCF hfield `size` requires ALL four parameters strictly positive, so a
    # fixed positive skirt depth is used (it does not affect the surface).
    _MJCF_BASE_Z = 1.0

    def mujoco_hfield_dims(self) -> tuple:
        """(nrow, ncol) attribute values for the MJCF ``<hfield>`` element:
        MuJoCo rows run along y, cols along x (transposed w.r.t. us)."""
        return (self.ncol, self.nrow)

    def mujoco_hfield_data(self) -> np.ndarray:
        """float32 1-D array of normalized elevations in [0, 1], laid out for
        MuJoCo: row-major with the row index along y (heightfield TRANSPOSED,
        C-order flattened)."""
        data = (self.heightfield - self.hfield_base_z) / self.hfield_elevation_z
        return np.ascontiguousarray(data.T, dtype=np.float32).reshape(-1)

    def mujoco_hfield_file_bytes(self) -> bytes:
        """Payload for a MuJoCo ``<hfield file="...bin"/>`` asset.

        MuJoCo's custom binary hfield format: two little-endian int32 header
        values (nrow, ncol -- along y, x respectively) followed by nrow*ncol
        float32 elevations in the layout of :meth:`mujoco_hfield_data`.
        """
        nrow_mj, ncol_mj = self.mujoco_hfield_dims()
        header = np.array([nrow_mj, ncol_mj], dtype=np.int32).tobytes()
        return header + self.mujoco_hfield_data().tobytes()

    def mujoco_hfield_size(self) -> tuple:
        """MJCF ``<hfield size="rx ry ez bz"/>`` (all entries positive)."""
        return (
            self.size_x / 2.0,
            self.size_y / 2.0,
            self.hfield_elevation_z,
            self._MJCF_BASE_Z,
        )

    def mujoco_hfield_placement(self) -> tuple:
        """World-frame ``pos`` for the hfield geom: local x, y in [-r, r]
        map to world [0, size_x] x [0, size_y] and z = hfield_base_z (the
        field's minimum world height), so that ``world z = pos_z + d * ez``
        reconstructs the true height for any (even negative) z_min."""
        return (self.size_x / 2.0, self.size_y / 2.0, self.hfield_base_z)


class TerrainBuilder:
    """Numpy-only port of the old ``legged_gym.utils.terrain.Terrain``."""

    def __init__(self, cfg: Any = None) -> None:
        self.cfg = cfg if cfg is not None else TerrainConfig()
        self.type = self.cfg.mesh_type
        if self.type in ("none", "plane"):
            raise ValueError(
                f"mesh_type={self.type!r} produces no heightfield; the "
                "hexapod_mjlab terrain builder requires 'trimesh'/'heightfield'"
            )
        self.env_length = self.cfg.terrain_length
        self.env_width = self.cfg.terrain_width

        # Normalize terrain_proportions from config to ensure they sum to 1
        raw = np.array(self.cfg.terrain_proportions, dtype=float)
        s = raw.sum()
        if s == 0:
            # avoid division by zero: fallback to uniform distribution
            normalized = np.ones_like(raw) / len(raw)
        else:
            normalized = raw / s
        # store raw and normalized versions for debugging/inspection
        self.raw_proportions = raw.tolist()
        self.normalized_proportions = normalized.tolist()
        # cumulative probabilities used to select terrain types
        self.proportions = list(np.cumsum(normalized))

        self.num_sub_terrains = self.cfg.num_rows * self.cfg.num_cols
        self.env_origins = np.zeros((self.cfg.num_rows, self.cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / self.cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / self.cfg.horizontal_scale)

        self.border = int(self.cfg.border_size / self.cfg.horizontal_scale)
        self.tot_cols = int(self.cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(self.cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

    # ------------------------------------------------------------------ #
    # public API                                                         #
    # ------------------------------------------------------------------ #
    def build(self, seed: int) -> TerrainData:
        """Generate the terrain deterministically for ``seed``.

        Seeds the *global* numpy RNG (same stream the old pipeline consumed),
        so callers must not rely on ambient np.random state across builds.
        """
        np.random.seed(seed)
        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        if self.cfg.curriculum:
            if hasattr(self.cfg, "difficulty_scale"):
                self.curiculum(self.cfg.difficulty_scale)
            else:
                self.curiculum()
        elif self.cfg.selected:
            self.selected_terrain()
        else:
            if hasattr(self.cfg, "difficulty_scale"):
                self.randomized_terrain(self.cfg.difficulty_scale)
            else:
                self.randomized_terrain()

        self.heightsamples = self.height_field_raw

        vertices = triangles = None
        if self.type == "trimesh":
            vertices, triangles = terrain_utils.convert_heightfield_to_trimesh(
                self.height_field_raw,
                self.cfg.horizontal_scale,
                self.cfg.vertical_scale,
                self.cfg.slope_treshold,
            )
            vertices = np.asarray(vertices, dtype=np.float32)
            triangles = np.asarray(triangles, dtype=np.int32).reshape(-1, 3)

        heightfield_m = (
            self.height_field_raw.astype(np.float32) * np.float32(self.cfg.vertical_scale)
        )
        z_min = float(heightfield_m.min())
        z_max = float(heightfield_m.max())
        # guard against perfectly flat fields (would divide by zero)
        elevation = (z_max - z_min) if (z_max - z_min) > 0.0 else 1.0

        return TerrainData(
            heightfield_raw=self.height_field_raw,
            heightfield=heightfield_m,
            vertices=vertices,
            triangles=triangles,
            triangles_flat=np.ascontiguousarray(
                triangles, dtype=np.int32
            ).reshape(-1),
            nrow=self.tot_rows,
            ncol=self.tot_cols,
            x_resolution=self.cfg.horizontal_scale,
            y_resolution=self.cfg.horizontal_scale,
            size_x=self.tot_rows * self.cfg.horizontal_scale,
            size_y=self.tot_cols * self.cfg.horizontal_scale,
            num_sub_rows=self.cfg.num_rows,
            num_sub_cols=self.cfg.num_cols,
            env_origins=self.env_origins,
            horizontal_scale=self.cfg.horizontal_scale,
            vertical_scale=self.cfg.vertical_scale,
            hfield_base_z=z_min,
            hfield_elevation_z=elevation,
        )

    # ------------------------------------------------------------------ #
    # terrain generation modes (RNG call order kept verbatim from the    #
    # old Terrain class -- parity-critical, do not reorder draws)        #
    # ------------------------------------------------------------------ #
    def randomized_terrain(self, difficulty_scale=1.0):
        for k in range(self.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9]) * difficulty_scale
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)

    def curiculum(self, difficulty_scale=1.0):
        # (sic) method name kept from the old class for diff-ability
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows * difficulty_scale
                choice = j / self.cfg.num_cols + 0.001

                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        kwargs_all = dict(self.cfg.terrain_kwargs)
        terrain_type = kwargs_all.pop("type")
        kwargs = dict(kwargs_all.get("terrain_kwargs", {}))
        for k in range(self.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain(
                "terrain",
                width=self.width_per_env_pixels,
                length=self.length_per_env_pixels,
                vertical_scale=self.cfg.vertical_scale,
                horizontal_scale=self.cfg.horizontal_scale,
            )

            generator = _TERRAIN_GENERATORS[terrain_type]
            generator(terrain, **kwargs)
            self.add_terrain_to_map(terrain, i, j)

    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(
            "terrain",
            width=self.width_per_env_pixels,
            length=self.width_per_env_pixels,
            vertical_scale=self.cfg.vertical_scale,
            horizontal_scale=self.cfg.horizontal_scale,
        )
        slope = difficulty * 0.4
        step_height = 0.05 + 0.18 * difficulty
        discrete_obstacles_height = 0.02 + difficulty * 0.2

        if hasattr(self.cfg, "stepping_stones_size"):
            stepping_stones_size = self.cfg.stepping_stones_size + difficulty * 0.5
            stepping_stones_max_height = self.cfg.stepping_stones_max_height + difficulty * 0.3
            stepping_stones_platform_size = self.cfg.stepping_stones_platform_size
            stone_distance = self.cfg.stepping_stones_distance + difficulty * 0.5
        else:
            # Configs whose terrain class shadows LeggedRobotCfg.terrain lack these
            # fields; fall back to the fork's legacy difficulty-driven parametrization.
            stepping_stones_size = 1.5 * (1.05 - difficulty)
            stepping_stones_max_height = 0.
            stepping_stones_platform_size = 4.
            stone_distance = 0.05 if difficulty == 0 else 0.1

        gap_size = 1. * difficulty
        pit_depth = 1. * difficulty
        if choice < self.proportions[0]:
            if choice < self.proportions[0] / 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        elif choice < self.proportions[1]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.03, max_height=0.03, step=0.005, downsampled_scale=0.2)
        elif choice < self.proportions[3]:
            if choice < self.proportions[2]:
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
        elif choice < self.proportions[4]:
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height,
                                                     rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
        elif choice < self.proportions[5]:
            terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size,
                                                  stone_distance=stone_distance, max_height=stepping_stones_max_height, platform_size=stepping_stones_platform_size)
        elif choice < self.proportions[6]:
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.)
        elif choice < self.proportions[7]:
            pillar_field_terrain(terrain, difficulty, self.cfg)
        elif choice < self.proportions[8]:
            sin_curve_channel_terrain(terrain, difficulty, self.cfg)
        else:
            pit_terrain(terrain, depth=pit_depth, platform_size=4.)

        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]


# ---------------------------------------------------------------------- #
# fork-specific terrain generators, ported verbatim from                 #
# legged_gym/utils/terrain.py (numpy-only in the source already)         #
# ---------------------------------------------------------------------- #

def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size

    terrain.height_field_raw[center_x-x2: center_x + x2, center_y-y2: center_y + y2] = -1000
    terrain.height_field_raw[center_x-x1: center_x + x1, center_y-y1: center_y + y1] = 0


def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth


def sin_curve_channel_terrain(terrain, difficulty, cfg):
    """Generate a sine-wave channel with walls on both sides.

    Channel runs along terrain width (world x-axis).  The centre varies
    along terrain length (world y-axis):

        centre = L/2 + amplitude * sin(2*pi*periods * world_pos / W_world)

    Parameters (from cfg):
        channel_width:   passable channel width (m).  Default 3.0.
        wall_height:     wall height (m).              Default 2.0.
        curve_amplitude: sine amplitude (m).           Default 1.0.
        curve_periods:   number of full sine periods.  Default 1.5.
    """
    channel_width = float(getattr(cfg, "channel_width", 3.0))
    wall_height = float(getattr(cfg, "wall_height", 2.0))
    curve_amplitude = float(getattr(cfg, "curve_amplitude", 1.0))
    curve_periods = float(getattr(cfg, "curve_periods", 1.5))

    hs = terrain.horizontal_scale
    vs = terrain.vertical_scale
    L = terrain.length
    W = terrain.width
    L_world = L * hs
    W_world = W * hs

    wall_px = int(wall_height / vs)
    half_px = int(channel_width / 2.0 / hs)

    for py in range(W):
        world_pos = py * hs
        centre = L_world / 2.0 + curve_amplitude * np.sin(
            2.0 * np.pi * curve_periods * world_pos / W_world)
        centre_px = int(centre / hs)
        lo = min(L, max(0, centre_px - half_px))
        hi = max(0, min(L, centre_px + half_px))
        terrain.height_field_raw[:lo, py] = wall_px
        terrain.height_field_raw[hi:, py] = wall_px


def pillar_field_terrain(terrain, difficulty, cfg):
    """
    生成随机分布的四棱柱障碍物地形（矩形截面）。

    预生成全部尺寸后逐个放置：每个障碍物在环带内随机采样位置，
    通过占用网格做 O(1) 碰撞检测（AABB + margin），兼顾效率和正确性。
    """
    # 数量范围（随难度插值）
    count_min = getattr(cfg, "pillar_count_min", 0)
    count_max = getattr(cfg, "pillar_count_max", 25)
    # 矩形边长范围（米）
    size_x_min = getattr(cfg, "pillar_size_x_min", 0.15)
    size_x_max = getattr(cfg, "pillar_size_x_max", 0.30)
    size_y_min = getattr(cfg, "pillar_size_y_min", 0.15)
    size_y_max = getattr(cfg, "pillar_size_y_max", 0.30)
    # 高度范围（米）
    height_min = getattr(cfg, "pillar_height_min", 0.20)
    height_max = getattr(cfg, "pillar_height_max", 0.50)
    # 间距与放置
    min_separation = getattr(cfg, "pillar_min_separation", 1.2)
    center_clear_radius = getattr(cfg, "pillar_center_clear_radius", 1.2)
    spawn_radius = getattr(cfg, "pillar_spawn_radius", 4.0)
    allow_height_variation = getattr(cfg, "pillar_allow_height_variation", True)

    count = int(count_min + difficulty * (count_max - count_min))

    min_sep_px = int(min_separation / terrain.horizontal_scale)
    clear_radius_px = int(center_clear_radius / terrain.horizontal_scale)
    spawn_radius_px = int(spawn_radius / terrain.horizontal_scale)
    margin = min_sep_px // 2

    L = terrain.length
    W = terrain.width
    center_y = L // 2
    center_x = W // 2

    # ---- 预生成全部尺寸（长/短边 split + 随机方向） ----
    half_x_px = []
    half_y_px = []
    occ_hx_px = []
    occ_hy_px = []
    height_px = []

    for _ in range(count):
        split = np.random.uniform(0.2, 0.8)
        x_range = size_x_max - size_x_min
        y_range = size_y_max - size_y_min

        if np.random.rand() > 0.5:
            long_min = size_x_min + split * x_range
            sx = long_min + np.random.uniform(0.0, 1.0) * (size_x_max - long_min)
            short_max = size_y_min + split * y_range
            sy = size_y_min + np.random.uniform(0.0, 1.0) * (short_max - size_y_min)
        else:
            long_min = size_y_min + split * y_range
            sy = long_min + np.random.uniform(0.0, 1.0) * (size_y_max - long_min)
            short_max = size_x_min + split * x_range
            sx = size_x_min + np.random.uniform(0.0, 1.0) * (short_max - size_x_min)

        hx = int(sx / terrain.horizontal_scale) // 2
        hy = int(sy / terrain.horizontal_scale) // 2
        h_val = height_min + np.random.uniform(0.0, 1.0) * (height_max - height_min)

        half_x_px.append(hx)
        half_y_px.append(hy)
        occ_hx_px.append(hx + margin)
        occ_hy_px.append(hy + margin)
        height_px.append(int(h_val / terrain.vertical_scale))

    # ---- 占用网格 ----
    occupied = np.zeros((L, W), dtype=bool)

    max_attempts_per = count * 100
    positions = []

    for i in range(count):
        hx = half_x_px[i]
        hy = half_y_px[i]
        occ_hx = occ_hx_px[i]
        occ_hy = occ_hy_px[i]

        # 障碍物太大跳过
        if hy * 2 >= L or hx * 2 >= W:
            continue

        for _ in range(max_attempts_per):
            r = np.random.uniform(clear_radius_px, spawn_radius_px)
            theta = np.random.uniform(0, 2 * np.pi)
            cx = int(center_x + r * np.cos(theta))
            cy = int(center_y + r * np.sin(theta))

            # 边界检查（仅本体）
            if cx - hx < 0 or cx + hx >= W or cy - hy < 0 or cy + hy >= L:
                continue

            # 中心点距离检查
            if np.hypot(cx - center_x, cy - center_y) < clear_radius_px:
                continue

            # O(1) 碰撞检测：占用网格切片
            oy1 = cy - occ_hy
            oy2 = cy + occ_hy
            ox1 = cx - occ_hx
            ox2 = cx + occ_hx
            if oy1 < 0 or oy2 > L or ox1 < 0 or ox2 > W:
                continue
            if occupied[oy1:oy2, ox1:ox2].any():
                continue

            # 放置成功
            occupied[oy1:oy2, ox1:ox2] = True
            positions.append((cx, cy, height_px[i], hx, hy))
            break

    # ---- 绘制 ----
    for cx, cy, h_px, hx, hy in positions:
        if allow_height_variation:
            h = np.random.randint(int(h_px * 0.6), h_px + 1)
        else:
            h = h_px

        x1 = max(0, cx - hx)
        x2 = min(W, cx + hx)
        y1 = max(0, cy - hy)
        y2 = min(L, cy + hy)
        terrain.height_field_raw[x1:x2, y1:y2] = h

    return terrain


# registry used by selected_terrain (replaces the old eval() of the type name)
_TERRAIN_GENERATORS = {
    "pyramid_sloped_terrain": terrain_utils.pyramid_sloped_terrain,
    "random_uniform_terrain": terrain_utils.random_uniform_terrain,
    "pyramid_stairs_terrain": terrain_utils.pyramid_stairs_terrain,
    "discrete_obstacles_terrain": terrain_utils.discrete_obstacles_terrain,
    "stepping_stones_terrain": terrain_utils.stepping_stones_terrain,
    "gap_terrain": gap_terrain,
    "pillar_field_terrain": pillar_field_terrain,
    "sin_curve_channel_terrain": sin_curve_channel_terrain,
    "pit_terrain": pit_terrain,
}


# ---------------------------------------------------------------------- #
# warp helper                                                             #
# ---------------------------------------------------------------------- #

def make_warp_mesh(terrain_data: TerrainData, device: Optional[str] = None):
    """Build a ``warp.Mesh`` from :class:`TerrainData` for ray casting.

    warp 1.17 requires points as a contiguous ``wp.vec3`` array (float32) and
    triangle indices as a *flattened 1-D* int32 array (2-D (M, 3) indices are
    rejected: "Mesh indices should be a flattened 1d array of indices") --
    hence ``TerrainData.triangles_flat``.

    SAFETY: ``wp.init()`` is called first.  warp 1.17 segfaults if
    mesh/ray kernels run before the runtime is initialized (verified 5/5);
    ``wp.init()`` is idempotent.
    """
    import warp as wp  # lazy: the builder core stays numpy-only

    wp.init()

    points = np.ascontiguousarray(terrain_data.vertices, dtype=np.float32)
    indices = np.ascontiguousarray(terrain_data.triangles_flat, dtype=np.int32)

    if device is None:
        return wp.Mesh(
            points=wp.array(points, dtype=wp.vec3),
            indices=wp.array(indices),
        )
    with wp.ScopedDevice(device):
        return wp.Mesh(
            points=wp.array(points, dtype=wp.vec3),
            indices=wp.array(indices),
        )
