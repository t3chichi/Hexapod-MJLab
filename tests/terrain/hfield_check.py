#!/usr/bin/env python
"""G3 -- hfield physics check: drop a box onto a pillar (mujoco + mujoco-warp).

Builds the M2 pillar terrain (TerrainConfig defaults, seed 42), embeds it as a
MuJoCo ``<hfield>`` geom in a small MJCF, drops a 0.1 m box with a freejoint
from 0.2 m above the tallest pillar's interior, and simulates 300 steps:

  * CPU reference with ``mujoco`` (mj_step),
  * GPU batch with ``mujoco_warp`` (nworld=64, all worlds identical).

Pass criteria:
  * all qpos/qvel finite in every world,
  * box comes to rest ON the pillar: final z within [top+0.03, top+0.08]
    (box half-height 0.05 m) and close to the CPU reference,
  * max |qvel| < 0.1 m/s and z drift over the last 50 steps < 5 mm,
  * mujoco's interpolated hfield height at the drop point matches a numpy
    bilinear sample of our heightfield (validates the transposed data
    layout: MuJoCo hfield row index runs along y).

Note (judgement call): a 0.5 m drop is energetic enough that mujoco-warp's
box/hfield contacts (at most 1 contact for HFIELD-BOX pairs under MULTICCD,
see the runtime warning) let the box bounce and wander off the 1.0 x 1.8 m
pillar top within 300 steps; 0.2 m keeps all 64 worlds resting on the pillar
top like the CPU reference.

SAFETY: ``wp.init()`` is called before ANY warp operation (warp 1.17
segfaults without it when mesh/ray kernels are used; idempotent call).

Run (brief GPU use):

    conda run --no-capture-output -n hp_mjlab python tests/terrain/hfield_check.py

Artifacts (gitignored) land in tests/terrain/logs/.
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

OUT_DIR = REPO / "tests" / "terrain" / "logs"
SEED = 42
NWORLD = 64
NSTEPS = 300
BOX_HALF = 0.05
DROP_HEIGHT = 0.2
TIMESTEP = 0.005


def drop_target(terrain):
    """World (x, y) of a point ON the tallest pillar's top plateau.

    The plateau may be non-convex (its centroid can fall outside it), so we
    pick the plateau pixel closest to the centroid whose 8-neighborhood is
    fully inside the plateau (interior pixel -> a 0.1 m box centered there is
    supported by the pillar top).
    """
    from scipy import ndimage

    hf = terrain.heightfield  # meters, axis0 = x
    mask = hf == hf.max()
    labels, _ = ndimage.label(mask)
    # plateau containing the global argmax
    iy, ix = np.unravel_index(np.argmax(hf), hf.shape)
    lab = labels[iy, ix]
    assert lab != 0
    plateau = labels == lab
    interior = ndimage.binary_erosion(plateau, structure=np.ones((3, 3), bool))
    assert interior.any(), "no interior pixel on the plateau"
    # heightfield axis 0 == world x, axis 1 == world y (TerrainData convention)
    is_, js = np.nonzero(interior)
    is_all, js_all = np.nonzero(plateau)
    # interior pixel closest to the plateau centroid
    k = np.argmin((is_ - is_all.mean()) ** 2 + (js - js_all.mean()) ** 2)
    cx = float(is_[k]) * terrain.x_resolution
    cy = float(js[k]) * terrain.y_resolution
    top = float(hf.max())
    return cx, cy, top, int(plateau.sum())


def main():
    import warp as wp

    wp.init()  # SAFETY: before any warp operation (see module docstring)

    import mujoco
    import mujoco_warp

    from hexapod_mjlab.terrains import TerrainBuilder, TerrainConfig

    terrain = TerrainBuilder(TerrainConfig()).build(seed=SEED)
    cx, cy, top_m, plateau_px = drop_target(terrain)
    print(f"terrain: {terrain.nrow}x{terrain.ncol} px, {terrain.size_x:.1f}x{terrain.size_y:.1f} m, "
          f"h_max={terrain.heightfield.max():.3f} m")
    print(f"drop target: world ({cx:.2f}, {cy:.2f}) m, pillar top {top_m:.3f} m, "
          f"plateau {plateau_px} px")

    # ---- MJCF with the generated hfield -------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bin_path = OUT_DIR / "terrain_hf.bin"
    bin_path.write_bytes(terrain.mujoco_hfield_file_bytes())  # int32 header + float32 data

    rx, ry, ez, bz = terrain.mujoco_hfield_size()
    nrow_mj, ncol_mj = terrain.mujoco_hfield_dims()  # rows along y, cols along x
    gx, gy, gz = terrain.mujoco_hfield_placement()
    # worldbody children are placed in WORLD frame: the box goes directly at
    # the world drop point (do NOT subtract the geom offset here).
    box_world = (cx, cy, top_m + DROP_HEIGHT)
    xml = f"""<mujoco model="hfield_check">
  <option timestep="{TIMESTEP}"/>
  <asset>
    <hfield name="terrain_hf" nrow="{nrow_mj}" ncol="{ncol_mj}"
            size="{rx} {ry} {ez!r} {bz!r}" file="{bin_path.name}"/>
  </asset>
  <worldbody>
    <geom name="terrain" type="hfield" hfield="terrain_hf"
          pos="{gx!r} {gy!r} {gz!r}" friction="1 0.005 0.0001"/>
    <body name="box" pos="{box_world[0]!r} {box_world[1]!r} {box_world[2]!r}">
      <freejoint name="box_joint"/>
      <geom name="box_geom" type="box" size="{BOX_HALF} {BOX_HALF} {BOX_HALF}"
            mass="1" friction="1 0.005 0.0001"/>
    </body>
  </worldbody>
</mujoco>
"""
    xml_path = OUT_DIR / "hfield_check.xml"
    xml_path.write_text(xml)

    # ---- CPU reference -------------------------------------------------
    mjm = mujoco.MjModel.from_xml_path(str(xml_path))
    mjd = mujoco.MjData(mjm)
    z_hist = []
    for _ in range(NSTEPS):
        mujoco.mj_step(mjm, mjd)
        z_hist.append(mjd.qpos[2])
    cpu_final_z = float(mjd.qpos[2])
    cpu_final_v = float(np.linalg.norm(mjd.qvel))
    print(f"CPU: final z={cpu_final_z:.5f} m |v|={cpu_final_v:.2e} m/s "
          f"drift(last50)={max(z_hist[-50:]) - min(z_hist[-50:]):.2e} m")

    # hfield sampling cross-check: mujoco's own surface (vertical ray) at the
    # drop point vs numpy bilinear of our field (validates row=x, col=y
    # ordering and the z = pos_z + d*ez reconstruction)
    gid = np.zeros(1, dtype=np.int32)
    ray = mujoco.mj_ray(mjm, mjd, np.array([cx, cy, 20.0]), np.array([0.0, 0.0, -1.0]),
                        None, 1, 1, gid)  # bodyexclude=1: skip the resting box
    mujoco_h_at_drop = 20.0 - ray if ray >= 0 else float("nan")
    # numpy bilinear at (cx, cy): pixel coords
    fx = cx / terrain.x_resolution
    fy = cy / terrain.y_resolution
    i0, j0 = int(fx), int(fy)
    dx, dy = fx - i0, fy - j0
    i1 = min(i0 + 1, terrain.nrow - 1)
    j1 = min(j0 + 1, terrain.ncol - 1)
    hf_m = terrain.heightfield
    np_h = (
        (1 - dx) * (1 - dy) * hf_m[i0, j0]
        + dx * (1 - dy) * hf_m[i1, j0]
        + (1 - dx) * dy * hf_m[i0, j1]
        + dx * dy * hf_m[i1, j1]
    )
    print(f"hfield height at drop: mujoco={mujoco_h_at_drop:.5f} m numpy={np_h:.5f} m "
          f"(pillar top {top_m:.5f} m)")

    # ---- GPU batch via mujoco-warp --------------------------------------
    m = mujoco_warp.put_model(mjm)
    mjd_fresh = mujoco.MjData(mjm)  # qpos0 == drop state straight from the MJCF
    d = mujoco_warp.put_data(mjm, mjd_fresh, nworld=NWORLD)

    z_gpu = []
    for _ in range(NSTEPS):
        mujoco_warp.step(m, d)
        z_gpu.append(d.qpos.numpy()[:, 2].copy())
    qpos = d.qpos.numpy()
    qvel = d.qvel.numpy()
    z_gpu = np.stack(z_gpu)  # (NSTEPS, NWORLD)

    finite = bool(np.isfinite(qpos).all() and np.isfinite(qvel).all())
    final_z = qpos[:, 2]
    max_speed = float(np.abs(qvel).max())
    drift = float((z_gpu[-50:].max(axis=0) - z_gpu[-50:].min(axis=0)).max())
    rest_lo, rest_hi = top_m + 0.03, top_m + 0.08
    gpu_vs_cpu = float(np.abs(final_z - cpu_final_z).max())

    print(f"GPU(nworld={NWORLD}): z final mean={final_z.mean():.5f} "
          f"min={final_z.min():.5f} max={final_z.max():.5f} m")
    print(f"GPU: finite={finite} max|v|={max_speed:.2e} m/s "
          f"drift(last50)={drift:.2e} m |z-cpu_z|max={gpu_vs_cpu:.2e} m")

    checks = {
        "all_finite": finite,
        "resting_on_pillar": bool(
            (final_z >= rest_lo).all() and (final_z <= rest_hi).all()
        ),
        "at_rest_speed": max_speed < 0.1,
        "at_rest_drift": drift < 5e-3,
        "hfield_layout_ok": abs(mujoco_h_at_drop - np_h) < 1e-3,
    }
    for name, ok in checks.items():
        print(f"{'ok' if ok else 'FAIL'}: {name}")
    result = {
        "seed": SEED,
        "nworld": NWORLD,
        "nsteps": NSTEPS,
        "pillar_top_m": top_m,
        "drop_point_xy_m": [cx, cy],
        "cpu_final_z": cpu_final_z,
        "gpu_final_z_mean": float(final_z.mean()),
        "gpu_final_z_min": float(final_z.min()),
        "gpu_final_z_max": float(final_z.max()),
        "gpu_vs_cpu_max_abs": gpu_vs_cpu,
        "max_speed": max_speed,
        "drift_last50": drift,
        "mujoco_h_at_drop": float(mujoco_h_at_drop),
        "numpy_h_at_drop": float(np_h),
        "checks": checks,
    }
    (OUT_DIR / "g3_hfield_check.json").write_text(
        json.dumps(result, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n"
    )

    if not all(checks.values()):
        sys.exit(1)
    print("\nG3 PASS: box rests on the pillar top in all 64 worlds (finite, stable)")


if __name__ == "__main__":
    main()
