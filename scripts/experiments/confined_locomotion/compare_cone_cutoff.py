#!/usr/bin/env python
"""Stage A / B: compare original near cone with widened-cone + BASE-x cutoff.

One REAL initial CUDA lidar scan and the production geometry-only ground grid.
Six candidate partitions applied offline to that SAME source scan. Uses the
production metric FPS and uniform angular sampler for the candidate groups.
No changes to production grouping, controller, learning, or task reward.

Provisional one-pose/clean/flat-scene diagnostic only; not a final training
setting or a sim-to-real certification. Writes a reproducible report and two
measured-data BEV figures. Requires a fresh --output directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

CANDIDATES = (
    ("A0", 32.0, 56.0, None),
    ("B1", 32.0, 56.0, 1.6),
    ("B2", 30.0, 56.0, 1.6),
    ("B3", 30.0, 56.0, 1.8),
    ("B4", 28.0, 56.0, 1.8),
    ("B5", 28.0, 56.0, 2.0),
)
GRID_X = (-0.8, -0.4, 0.0, 0.4, 0.8, 1.2, 1.6, 2.0)
GRID_Y = (-0.8, -0.4, 0.0, 0.4, 0.8)
THRESHOLDS = (0.1, 0.2, 0.3)


def args_parser(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True, help="new, non-existing evidence directory")
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--seed", type=int, default=20261014)
    return p.parse_args(argv)


def cpu(tensor):
    return tensor.detach().cpu().numpy().copy()


def evaluate(frame, pipeline, grid_xyz, grid_valid):
    import numpy as np
    import torch
    from scipy.spatial import cKDTree
    from hexapod_mjlab.tasks.confined_locomotion.scanner import (
        farthest_point_sampling_indices,
        uniform_angular_sample_indices,
    )

    pool_xyz = frame.pool_points_base
    pool_raw = frame.pool_raw_indices
    pool_miss = frame.no_hit_mask.index_select(0, pool_raw)
    keys = frame.pool_angle_keys
    sensor_xyz = pipeline.mount_t[0]
    r = pool_xyz - sensor_xyz
    norms = torch.linalg.vector_norm(r, dim=-1)
    axis = pipeline.cone_axis_body
    alignment = (r @ axis) / (norms + 1e-9)
    assert bool(torch.isfinite(alignment).all())

    grid_queries = grid_xyz[grid_valid]
    records, figure_payload = [], {}
    for name, theta, tilt, xmax in CANDIDATES:
        if tilt != pipeline.cfg.split_axis_tilt_deg:
            raise AssertionError("this first round fixes the exact production axis")
        cone = alignment >= float(np.sin(np.deg2rad(theta)))
        near = cone if xmax is None else cone & (pool_xyz[:, 0] <= xmax)
        far = ~near
        if name == "A0":
            assert torch.equal(near, frame.near_mask), "A0 must be exactly the production clean grouping"
            assert torch.equal(far, frame.far_mask)
        if bool((near & far).any()) or not bool((near | far).all()):
            raise AssertionError(f"near/far partition not exhaustive in {name}")

        near_idx, near_k = farthest_point_sampling_indices(pool_xyz, near, 256)
        far_idx, far_k = uniform_angular_sample_indices(keys, far, 64)
        assert near_k == 256 and far_k == 64
        sel_near_raw = pool_raw[near_idx]
        sel_far_raw = pool_raw[far_idx]
        assert len(torch.unique(sel_near_raw)) == near_k
        assert len(torch.unique(sel_far_raw)) == far_k
        if name == "A0":
            assert torch.equal(sel_near_raw, frame.near_indices), "A0 near FPS differs from actual output"
            assert torch.equal(sel_far_raw, frame.far_indices), "A0 far sampling differs from actual output"
        hit_near_pool = near & ~pool_miss
        hit_far_pool = far & ~pool_miss
        hit_near_sample = ~pool_miss[near_idx]
        hit_far_sample = ~pool_miss[far_idx]
        pool_hits = cpu(pool_xyz[hit_near_pool])
        sample_hits = cpu(pool_xyz[near_idx[hit_near_sample]])
        far_sample_hits = cpu(pool_xyz[far_idx[hit_far_sample]])

        def nn_dist(samples):
            if len(samples) == 0:
                return None, None
            d = cKDTree(samples).query(grid_queries)[0]
            return d, {
                "p50_m": round(float(np.percentile(d, 50)), 5),
                "p95_m": round(float(np.percentile(d, 95)), 5),
                "max_m": round(float(d.max()), 5),
                "count_by_threshold": {
                    f"{threshold:.2f}m": int((d <= threshold).sum())
                    for threshold in THRESHOLDS
                },
            }

        pool_dist, pool_support = nn_dist(pool_hits)
        fps_dist, fps_support = nn_dist(sample_hits)
        plot_grid_dists = np.full(len(grid_xyz), np.nan, dtype=float)
        if pool_dist is not None:
            plot_grid_dists[grid_valid] = pool_dist
        bins = []
        # Width of the observed real-hit footprint (not analytical cone surface).
        for xcenter in (0.8, 1.2, 1.6, 2.0):
            sub = pool_hits[np.abs(pool_hits[:, 0] - xcenter) <= 0.1]
            bins.append({
                "x_center_m": xcenter, "x_tolerance_m": 0.1,
                "hits": len(sub),
                "y_p05_m": float(np.percentile(sub[:, 1], 5)) if len(sub) else None,
                "y_p95_m": float(np.percentile(sub[:, 1], 95)) if len(sub) else None,
            })

        record = {
            "id": name,
            "theta_deg": theta, "tilt_deg": tilt,
            "half_angle_deg": 90 - theta,
            "x_max_body_m": xmax,
            "near_total_candidates": int(near.sum().item()),
            "near_real_hit_candidates": int(hit_near_pool.sum().item()),
            "near_nohit_candidates": int((near & pool_miss).sum().item()),
            "far_total_candidates": int(far.sum().item()),
            "far_real_hit_candidates": int(hit_far_pool.sum().item()),
            "near_fps_points": near_k,
            "near_fps_real_hits": int(hit_near_sample.sum().item()),
            "far_angular_points": far_k,
            "far_angular_real_hits": int(hit_far_sample.sum().item()),
            "far_angular_real_hits_x_gt_2m": int((far_sample_hits[:, 0] > 2.0).sum()),
            "near_hit_x_max_m": float(pool_hits[:, 0].max()) if len(pool_hits) else None,
            "grid": {
                "geometry_valid": int(grid_valid.sum()),
                "all": int(len(grid_xyz)),
                "hit_pool_nn": pool_support,
                "fps_nn": fps_support,
            },
            "observed_lateral_hit_quantiles": bins,
        }
        records.append(record)
        figure_payload[name] = {
            "near_hit_pool": pool_hits,
            "near_hit_fps": sample_hits,
            "far_hit_selected": far_sample_hits,
            "grid_nn": plot_grid_dists,
            "x_max": xmax,
            "theta": theta,
        }
        print(
            f"{name} theta={theta:.0f} xmax={xmax} near_hits={record['near_real_hit_candidates']}"
            f" grid@0.2m(pool)={pool_support['count_by_threshold']['0.20m'] if pool_support else '-'}"
            f" fps={fps_support['count_by_threshold']['0.20m'] if fps_support else '-'}"
            f" far_hits_64={record['far_angular_real_hits']}",
            flush=True,
        )
    return records, figure_payload


def plot_bev(outdir, records, data, grid_xyz, sensor_xyz):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    # This is the frozen flat-task startup REFERENCE ENVELOPE, not a measured
    # robot mesh silhouette. Use the documented [-0.89,+0.89] x and 0.69 y
    # extents as a conservative, clearly labelled robot-scale reference.
    envelope = np.array([
        [-0.89, -0.59], [0.0, -0.69], [0.89, -0.59],
        [0.89, 0.59], [0.0, 0.69], [-0.89, 0.59],
    ], dtype=float)

    def decorate(ax, *, show_key=False):
        ax.add_patch(Polygon(envelope, closed=True, facecolor="none",
                             edgecolor="#232629", linewidth=1.4, ls="--", zorder=5))
        ax.scatter([0], [0], marker="+", color="black", s=65, linewidths=1.7, zorder=7)
        ax.scatter([sensor_xyz[0]], [sensor_xyz[1]], marker="o", color="#dc791c",
                   s=48, linewidths=1, edgecolors="white", zorder=8)
        ax.annotate("", xy=(0.5, 0), xytext=(0.06, 0),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.8))
        ax.set_xlim(-1.15, 2.25)
        ax.set_ylim(-1.25, 1.25)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.16, linewidth=0.6)
        ax.set_xlabel("BASE x [m]")
        if show_key:
            ax.set_ylabel("BASE y [m]")

    fig, axs = plt.subplots(2, 3, figsize=(17, 9.5), layout="constrained")
    for ix, (rec, ax) in enumerate(zip(records, axs.flat)):
        d = data[rec["id"]]
        p = d["near_hit_pool"]
        # Plot deterministic thinning ONLY FOR DISPLAY; all metric counts
        # are computed on the entire real hit pool.
        stride = max(1, len(p) // 5500)
        visible = p[::stride]
        ax.scatter(visible[:, 0], visible[:, 1], s=1.25, color="#3079ac",
                   alpha=0.22, rasterized=True, zorder=2)
        grid_valid = np.isfinite(d["grid_nn"])
        cols = np.where(d["grid_nn"] <= 0.2, "#2c9d58",
                        np.where(d["grid_nn"] <= 0.3, "#e6a12b", "#c73f42"))
        ax.scatter(grid_xyz[grid_valid, 0], grid_xyz[grid_valid, 1], c=cols[grid_valid],
                   s=44, marker="x", linewidths=1.6, zorder=6)
        ax.axvline(float(d["x_max"]), color="#ac5b0b", linestyle="--", lw=1.7,
                   zorder=4) if d["x_max"] is not None else None
        decorate(ax, show_key=(ix % 3 == 0))
        ax.set_title(
            f"{rec['id']}: theta={rec['theta_deg']:.0f}°, "
            + ("no cutoff" if rec["x_max_body_m"] is None else f"x_max={rec['x_max_body_m']:.1f} m")
            + f"\nNear real hits {rec['near_real_hit_candidates']:,}"
            + f" | grid@0.2 m {rec['grid']['hit_pool_nn']['count_by_threshold']['0.20m']}/40",
            fontsize=11.4, weight="bold",
        )
    fig.suptitle(
        "REAL scan: near hit pools and grid support | one clean T1 reset\n"
        "Blue = near real hit endpoints (display downsampled); "
        "grid: green <=0.2 m, amber <=0.3 m, red >0.3 m\n"
        "Dashed hexagon = startup reference envelope (NOT robot mesh); "
        "orange dot = lidar mount; vertical dash = x cutoff",
        fontsize=13,
    )
    fig.savefig(outdir / "near_hit_grid_6panels.png", dpi=155)
    plt.close(fig)

    fig, axs = plt.subplots(2, 3, figsize=(17, 9.5), layout="constrained")
    for ix, (rec, ax) in enumerate(zip(records, axs.flat)):
        d = data[rec["id"]]
        nearp, farp = d["near_hit_fps"], d["far_hit_selected"]
        if len(nearp):
            ax.scatter(nearp[:, 0], nearp[:, 1], s=13, c="#2763bc", alpha=0.75,
                       edgecolors="none", zorder=3)
        if len(farp):
            ax.scatter(farp[:, 0], farp[:, 1], s=16, c="#ac4fa4", alpha=0.85,
                       marker="^", zorder=4)
        ax.scatter(grid_xyz[:, 0], grid_xyz[:, 1], s=38,
                   c="#d14935", marker="+", linewidths=1.2, zorder=5)
        if d["x_max"] is not None:
            ax.axvline(d["x_max"], color="#ac5b0b", linestyle="--", lw=1.7)
        decorate(ax, show_key=(ix % 3 == 0))
        ax.set_title(
            f"{rec['id']}: FPS near-hit {rec['near_fps_real_hits']}/256"
            f" | far real-hit {rec['far_angular_real_hits']}/64"
            f" | grid@0.2 m FPS {rec['grid']['fps_nn']['count_by_threshold']['0.20m']}/40",
            fontsize=11.0, weight="bold",
        )
    fig.suptitle(
        "REAL scan: production metric FPS256 and angular far64 applied to each candidate\n"
        "Blue = near FPS true hits, purple triangles = far64 true hits, red + = ground grid\n"
        "Near/far miss endpoints excluded from plotting but counted in report",
        fontsize=13,
    )
    fig.savefig(outdir / "fps_far_6panels.png", dpi=155)
    plt.close(fig)


def main(argv=None):
    import numpy as np
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
    from hexapod_mjlab.rl.confined_locomotion.symmetry import GridPermutation
    from hexapod_mjlab.tasks.confined_locomotion.flat_env_cfg import make_training_env_cfg
    from hexapod_mjlab.tasks.confined_locomotion.ground import (
        GroundGridCfg, VALIDITY_GEOMETRY_ONLY_FLAT_TRAINING
    )
    from hexapod_mjlab.tasks.confined_locomotion.runtime import TrainingRuntime
    from hexapod_mjlab.tasks.confined_locomotion.scanner import ConfinedLidarCfg

    opts = args_parser(argv)
    grid = GroundGridCfg(x_coords=GRID_X, y_coords=GRID_Y,
                         validity_mode=VALIDITY_GEOMETRY_ONLY_FLAT_TRAINING)
    GridPermutation.from_axes(grid.x_coords, grid.y_coords)
    if opts.gpu_id < 0 or not torch.cuda.is_available() or opts.gpu_id >= torch.cuda.device_count():
        raise RuntimeError("one actual selected CUDA device required")
    out = opts.output.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    paths = [
        Path(__file__),
        ROOT / "src/hexapod_mjlab/tasks/confined_locomotion/scanner.py",
        ROOT / "src/hexapod_mjlab/tasks/confined_locomotion/ground.py",
        ROOT / "src/hexapod_mjlab/tasks/confined_locomotion/runtime.py",
        ROOT / "src/hexapod_mjlab/assets/robots/el_4090/mjcf/el_4090_lidar.xml",
    ]
    report = {
        "schema": "confined_locomotion.cone_cutoff_compare/v1",
        "claim": "one reset-time production CUDA scan, same raw rays, clean flat geometry; offline repartition; no model/training changes",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_id": opts.gpu_id, "seed": opts.seed,
        "candidate_grid": grid.identity(), "candidate_grid_hash": grid.identity_hash(),
        "candidate_partitions": [dict(id=n, theta_deg=t, tilt_deg=d, x_max_body_m=x) for n,t,d,x in CANDIDATES],
        "rule": "near = cosine_cone_of_original_BASE_points AND (x_BASE <= x_max if given); far = complement of near",
        "near_criterion": "cos(axis, sensor->endpoint) >= sin(theta); actual half angle = 90-theta",
        "limitation": "one nominal T1 pose, ideal flat plane only; robot mesh not occluding lidar; x cut uses observed endpoints, not ray angle; no extra physics steps",
        "sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "state": "started",
    }
    (out / "recipe.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    env = runtime = None
    try:
        torch.cuda.set_device(opts.gpu_id)
        env = ManagerBasedRlEnv(make_training_env_cfg(num_envs=1, seed=opts.seed),
                                device=f"cuda:{opts.gpu_id}")
        wrapper = RslRlVecEnvWrapper(env, clip_actions=None)
        runtime = TrainingRuntime(
            wrapper, grid_cfg=grid, seed=opts.seed,
            output_dir=out / "runtime", mode="engineering",
            lidar_cfg=ConfinedLidarCfg(), perturbation_cfg=None,
        )
        labels = runtime.labels()
        frame_items = [v["frame"] for v in runtime.pipeline.scan_records if "frame" in v]
        if len(frame_items) != 1:
            raise AssertionError(f"one first raw scan required, got {len(frame_items)}")
        frame = frame_items[0]
        if (frame.episode_id, frame.scan_id) != (0, 1):
            raise AssertionError("unexpected initial scan identity")
        ground_label = labels["ground"]["label"]
        grid_xyz = cpu(ground_label.positions[0])
        grid_valid = cpu(ground_label.valid[0]).astype(bool)
        scan_id = dict(episode=frame.episode_id, scan=frame.scan_id, time_s=frame.time_s,
                       control_steps=runtime.control_steps_taken)
        records, plot_data = evaluate(frame, runtime.pipeline, grid_xyz, grid_valid)
        plot_bev(out, records, plot_data, grid_xyz, runtime.pipeline.mount_pos)
        csv_fields = (
            "id", "theta_deg", "x_max_body_m", "near_total_candidates",
            "near_real_hit_candidates", "near_fps_real_hits", "far_total_candidates",
            "far_angular_real_hits", "grid_pool_0.2m", "grid_fps_0.2m",
            "grid_pool_0.3m", "grid_fps_0.3m",
        )
        with (out / "metrics.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=csv_fields)
            writer.writeheader()
            for r in records:
                writer.writerow({
                    **{k:r[k] for k in csv_fields if k in r},
                    "grid_pool_0.2m": r["grid"]["hit_pool_nn"]["count_by_threshold"]["0.20m"],
                    "grid_fps_0.2m": r["grid"]["fps_nn"]["count_by_threshold"]["0.20m"],
                    "grid_pool_0.3m": r["grid"]["hit_pool_nn"]["count_by_threshold"]["0.30m"],
                    "grid_fps_0.3m": r["grid"]["fps_nn"]["count_by_threshold"]["0.30m"],
                })
        report.update(state="completed", measurements=records,
                      raw_ray_count=len(frame.no_hit_mask),
                      raw_hit_count=int((~frame.no_hit_mask).sum().item()),
                      scan_identity=scan_id, lidar_mount_xyz=runtime.pipeline.mount_pos.tolist())
    except BaseException as exc:
        report.update(state="failed", error=repr(exc))
        raise
    finally:
        try:
            if runtime is not None:
                runtime.close()
        finally:
            if env is not None:
                env.close()
            (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
            print(f"compare_cone_cutoff: {report['state']}: {out / 'report.json'}",flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
