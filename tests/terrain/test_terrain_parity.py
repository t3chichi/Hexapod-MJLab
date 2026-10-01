#!/usr/bin/env python
"""G2 -- golden parity test: TerrainBuilder vs the old legged_gym Terrain.

Runs in the ``hp_mjlab`` env (no isaacgym).  The golden reference
(tests/data/golden_terrain/heightfield_seed42.npy) was produced by
scripts/golden_terrain_gen.py in the ``li_leggym`` env from the OLD
``legged_gym.utils.terrain.Terrain`` with El4090LidarHeadCfg.terrain and
``np.random.seed(42)``; see the JSON next to it for the resolved config and
git provenance.

Parity expectation: EXACT match (allclose atol=1e-9 rtol=0, plus straight
int16 equality).  NumPy's legacy RandomState stream is frozen (NEP 19), and
the parity-relevant terrain type (100% pillar_field at proportions index 7)
never calls random_uniform_terrain, so the interp2d -> RectBivariateSpline
substitution in the vendored module cannot perturb this field.  If this test
ever reports a nonzero max diff, first suspect a real divergence in RNG call
order, not float noise; a tolerance bump would only be justified (<= 1e-6)
for terrain mixes that go through random_uniform_terrain.

Run:

    conda run --no-capture-output -n hp_mjlab python tests/terrain/test_terrain_parity.py
"""

import json
import sys
from pathlib import Path

import numpy as np

from hexapod_mjlab.terrains import TerrainBuilder, TerrainConfig

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "golden_terrain"
SEED = 42
# task-mandated exact tolerance; see module docstring for when a <=1e-6
# tolerance would be justifiable (never for the pillar_field mix)
ATOL, RTOL = 1e-9, 0.0


def check(cond, msg):
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)
    print(f"ok: {msg}")


def main():
    golden = np.load(DATA_DIR / f"heightfield_seed{SEED}.npy")
    meta = json.loads((DATA_DIR / f"golden_config_seed{SEED}.json").read_text())

    # 1. the builder config defaults must equal the resolved old config
    cfg = TerrainConfig()
    for key, old_val in meta["terrain"].items():
        new_val = getattr(cfg, key)
        if isinstance(old_val, list):
            check(new_val == old_val, f"cfg.{key} == {old_val}")
        else:
            check(new_val == old_val, f"cfg.{key} == {old_val!r} (new: {new_val!r})")

    # 2. exact heightfield parity
    data = TerrainBuilder(cfg).build(seed=SEED)
    new_hf = data.heightfield
    check(golden.shape == new_hf.shape, f"shape {new_hf.shape} == golden {golden.shape}")

    # The meters-valued field is stored float32; compare against the golden
    # int16 field scaled in float32 so atol=1e-9 tests generation equality,
    # not float64-vs-float32 product rounding.
    golden_f32 = golden.astype(np.float32) * np.float32(cfg.vertical_scale)
    max_diff = float(np.max(np.abs(new_hf - golden_f32)))
    # transparency: diff vs a float64 reference product (pure float32 storage
    # rounding of k*0.005, NOT a generation divergence)
    max_diff_f64ref = float(
        np.max(np.abs(new_hf.astype(np.float64) - golden * cfg.vertical_scale))
    )
    exact_int = bool(np.array_equal(data.heightfield_raw, golden))
    check(
        np.allclose(new_hf, golden_f32, atol=ATOL, rtol=RTOL),
        f"heightfield allclose(float32) atol={ATOL} rtol={RTOL} (max_diff={max_diff:.3e} m)",
    )
    check(exact_int, "raw int16 fields bit-identical (np.array_equal)")
    print(f"info: max diff vs float64 reference product = {max_diff_f64ref:.3e} m "
          "(float32 storage rounding of k*vertical_scale; int16 equality above "
          "proves the generated field is exact)")

    # 3. structural invariants
    check(data.heightfield.dtype == np.float32, "heightfield dtype float32")
    check(data.heightfield_raw.dtype == np.int16, "heightfield_raw dtype int16")
    check(data.vertices.dtype == np.float32 and data.vertices.shape == (data.nrow * data.ncol, 3),
          f"vertices {data.vertices.shape} float32")
    check(data.triangles.dtype == np.int32
          and data.triangles.shape == (2 * (data.nrow - 1) * (data.ncol - 1), 3),
          f"triangles {data.triangles.shape} int32")
    check(data.triangles_flat.dtype == np.int32 and data.triangles_flat.ndim == 1
          and data.triangles_flat.size == 3 * data.triangles.shape[0]
          and np.array_equal(data.triangles_flat, data.triangles.reshape(-1)),
          "triangles_flat: int32 1-D flattened copy of triangles")
    check(abs(float(data.vertices[:, 2].max()) - float(golden.max()) * cfg.vertical_scale) <= 1e-6,
          f"vertex z max {data.vertices[:, 2].max():.6f} == golden max * vertical_scale "
          f"(float32 storage, tol 1e-6)")
    check((data.nrow, data.ncol) == golden.shape, "nrow/ncol match golden shape")

    # 4. mujoco hfield fields
    hf_data = data.mujoco_hfield_data()
    check(hf_data.dtype == np.float32 and hf_data.shape == (data.nrow * data.ncol,),
          f"mujoco hfield data {hf_data.shape} float32")
    check(float(hf_data.min()) >= 0.0 and float(hf_data.max()) <= 1.0,
          f"hfield data in [0,1] ({hf_data.min():.6f}..{hf_data.max():.6f})")
    rx, ry, ez, bz = data.mujoco_hfield_size()
    px, py, pz = data.mujoco_hfield_placement()
    check(rx > 0 and ry > 0 and ez > 0 and bz > 0,
          f"hfield size=({rx}, {ry}, {ez:.6f}, {bz}) all positive (MJCF requirement)")
    check(abs(rx - data.size_x / 2) < 1e-12 and abs(ry - data.size_y / 2) < 1e-12
          and abs(px - data.size_x / 2) < 1e-12 and abs(py - data.size_y / 2) < 1e-12,
          "hfield x/y placement maps local [-r, r] to world [0, size]")
    check(abs(pz + ez * float(hf_data.max()) - data.heightfield.max()) < 1e-6,
          f"hfield reconstruction pos_z + d*ez == world height at d=1 "
          f"({pz + ez * float(hf_data.max()):.6f} == {data.heightfield.max():.6f})")
    check(data.x_resolution == cfg.horizontal_scale and data.y_resolution == cfg.horizontal_scale,
          "hfield x/y resolution == horizontal_scale")
    check(abs(data.size_x - data.nrow * cfg.horizontal_scale) < 1e-12
          and abs(data.size_y - data.ncol * cfg.horizontal_scale) < 1e-12,
          f"total size {data.size_x} x {data.size_y} m")

    # 5. determinism
    again = TerrainBuilder(TerrainConfig()).build(seed=SEED)
    other = TerrainBuilder(TerrainConfig()).build(seed=SEED + 1)
    check(np.array_equal(data.heightfield_raw, again.heightfield_raw),
          "build(seed=42) reproducible")
    check(not np.array_equal(data.heightfield_raw, other.heightfield_raw),
          "build(seed=43) differs from seed=42")

    # 6. env_origins parity vs values recorded at golden-gen time
    golden_origins = np.array(
        [[[4.0, 4.0, 0.0], [4.0, 12.0, 0.0]], [[12.0, 4.0, 0.0], [12.0, 12.0, 0.0]]]
    )
    check(data.env_origins.shape == (2, 2, 3)
          and np.allclose(data.env_origins, golden_origins, atol=1e-12),
          f"env_origins == old Terrain values\n{data.env_origins}")

    print("\nG2 PASS: TerrainBuilder(seed=42) is bit-identical to the old "
          f"Terrain pipeline (max_diff={max_diff:.3e} m, int16 exact={exact_int})")


if __name__ == "__main__":
    main()
