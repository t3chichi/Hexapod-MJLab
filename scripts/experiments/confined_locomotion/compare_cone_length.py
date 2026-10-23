#!/usr/bin/env python
"""Stage-A B-scheme LENGTH sweep on ONE real initial CUDA scan.

Constrain near = original cone (theta=28, tilt=56) AND observed BASE-x <= xmax.
A0 keeps the unmodified original theta=32/tilt=56 cone for reference.
All cases use exactly the same measured 86,400 source rays, plus actual
production FPS256 / angular far64 routines; no model, optimizer, or training.

The existing ground grid is a temporary REQUIRED RUNTIME FIXTURE only. Neither
its shape nor its grid-support metric enters the comparison or recommendation.
This one-pose flat-ground clean scan is diagnostic, NOT an adoption gate.

Usage:
  python scripts/experiments/confined_locomotion/compare_cone_length.py \\
    --output logs/verification/stageA_B_length_v1 --gpu-id 0
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

CANDIDATES = (("A0", 32.0, None), ("B5", 28.0, 2.0),
              ("B6", 28.0, 2.5), ("B7", 28.0, 3.0),
              ("B8", 28.0, 3.5), ("B9", 28.0, 4.0),
              ("B10", 28.0, 5.0), ("B11", 28.0, 6.0))
X_SLICE_CENTERS = (0.8, 1.2, 1.6, 2.0, 3.0, 4.0, 5.0, 6.0)
X_BINS = (0., 1., 2., 3., 4., 5., 6., 7., 15.)
CELL_M = 0.25
REGION_XY = (0.5, 6.5, -3.0, 3.0)


def _args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True, help="fresh, absent evidence directory")
    p.add_argument("--seed", type=int, default=20261014)
    p.add_argument("--gpu-id", type=int, default=0)
    return p.parse_args(argv)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy().copy()


def count_x_bins(points, *, breaks=X_BINS):
    import numpy as np
    x = points[:, 0]
    return {f"{breaks[k]:g}..{breaks[k+1]:g}": int(((x >= breaks[k]) & (x < breaks[k + 1])).sum())
            for k in range(len(breaks) - 1)}


def occupied_cells(points):
    """Fixed 0.25m XY cells in the same 0.5..6.5 x, ±3m y analysis window."""
    import numpy as np
    xmin, xmax, ymin, ymax = REGION_XY
    if not len(points):
        return set()
    p = points[(points[:, 0] >= xmin) & (points[:, 0] < xmax)
               & (points[:, 1] >= ymin) & (points[:, 1] < ymax)]
    i = np.floor((p[:, 0] - xmin) / CELL_M).astype(np.int64)
    j = np.floor((p[:, 1] - ymin) / CELL_M).astype(np.int64)
    return set(zip(i.tolist(), j.tolist()))


def measure(frame, pipeline):
    import numpy as np
    import torch
    from hexapod_mjlab.tasks.confined_locomotion.scanner import (
        farthest_point_sampling_indices, uniform_angular_sample_indices,
    )

    xyz = frame.pool_points_base
    raw = frame.pool_raw_indices
    miss = frame.no_hit_mask.index_select(0, raw)
    keys = frame.pool_angle_keys
    sep = xyz - pipeline.mount_t[0]
    cos = (sep @ pipeline.cone_axis_body) / (torch.linalg.vector_norm(sep, dim=-1) + 1e-9)
    assert bool(torch.isfinite(cos).all())
    if len(raw) != 86400:
        raise AssertionError("full raw pool expected; no pre-thinning")
    raw_hit_count = int((~miss).sum().item())

    results, data = [], {}
    for name, theta, xmax in CANDIDATES:
        mask_cone = cos >= float(np.sin(np.deg2rad(theta)))
        near = mask_cone if xmax is None else mask_cone & (xyz[:, 0] <= xmax)
        far = ~near
        if bool((near & far).any()) or not bool((near | far).all()):
            raise AssertionError(f"{name} not a disjoint full partition")
        idx_n, n = farthest_point_sampling_indices(xyz, near, 256)
        idx_f, f = uniform_angular_sample_indices(keys, far, 64)
        if (n, f) != (256, 64):
            raise AssertionError(f"{name} insufficient candidates for 256/64")
        near_hit_mask = near & ~miss
        far_hit_mask = far & ~miss
        near_all = to_numpy(xyz[near_hit_mask])
        near_selected = to_numpy(xyz[idx_n[~miss[idx_n]]])
        far_selected = to_numpy(xyz[idx_f[~miss[idx_f]]])
        near_misses = to_numpy(xyz[idx_n[miss[idx_n]]])
        if name == "A0":
            assert torch.equal(near, frame.near_mask), "A0 does not match production near"
            assert torch.equal(raw[idx_n], frame.near_indices), "A0 FPS mismatch"
            assert torch.equal(raw[idx_f], frame.far_indices), "A0 far64 mismatch"
        all_cells = occupied_cells(near_all)
        fps_cells = occupied_cells(near_selected)
        if not fps_cells <= all_cells:
            raise AssertionError(f"{name} FPS selected point outside real pool")
        slices = []
        for x in X_SLICE_CENTERS:
            hits = near_all[np.abs(near_all[:, 0] - x) <= 0.1]
            selected = near_selected[np.abs(near_selected[:, 0] - x) <= 0.1]
            slices.append({
                "x_center_m": x,
                "x_window_halfwidth_m": 0.1,
                "all_hit_count": int(len(hits)),
                "fps_hit_count": int(len(selected)),
                "all_y_p05_m": float(np.percentile(hits[:, 1], 5)) if len(hits) else None,
                "all_y_p95_m": float(np.percentile(hits[:, 1], 95)) if len(hits) else None,
                "fps_y_p05_m": float(np.percentile(selected[:, 1], 5)) if len(selected) >= 5 else None,
                "fps_y_p95_m": float(np.percentile(selected[:, 1], 95)) if len(selected) >= 5 else None,
            })
        record = {
            "id": name, "theta_deg": theta, "tilt_deg": 56.0,
            "actual_cone_half_angle_deg": 90.0 - theta,
            "x_max_body_m": xmax,
            "near_candidates": int(near.sum().item()),
            "near_candidate_real_hits": int(near_hit_mask.sum().item()),
            "near_candidate_misses": int((near & miss).sum().item()),
            "near_fps_valid": n,
            "near_fps_real_hits": int((~miss[idx_n]).sum().item()),
            "near_fps_misses": int(miss[idx_n].sum().item()),
            "near_real_hit_min_x": float(near_all[:, 0].min()),
            "near_real_hit_max_x": float(near_all[:, 0].max()),
            "near_fps_real_hit_max_x": float(near_selected[:, 0].max()) if len(near_selected) else None,
            "far_candidates": int(far.sum().item()),
            "far_candidate_real_hits": int(far_hit_mask.sum().item()),
            "far_candidate_misses": int((far & miss).sum().item()),
            "far_angular_valid": f,
            "far_angular_real_hits": int((~miss[idx_f]).sum().item()),
            "far_angular_misses": int(miss[idx_f].sum().item()),
            "far_selected_real_hit_x_ge_2m": int((far_selected[:, 0] >= 2).sum()),
            "far_selected_real_hit_x_ge_4m": int((far_selected[:, 0] >= 4).sum()),
            "far_selected_real_hit_x_ge_6m": int((far_selected[:, 0] >= 6).sum()),
            "real_hits_in_x_intervals": count_x_bins(near_all),
            "fps_hits_in_x_intervals": count_x_bins(near_selected),
            "analysis_xy_region_m": dict(x_min=REGION_XY[0], x_max=REGION_XY[1],
                                          y_min=REGION_XY[2], y_max=REGION_XY[3]),
            "cell_size_m": CELL_M,
            "near_candidate_hit_cells": len(all_cells),
            "near_fps_hit_cells": len(fps_cells),
            "near_fps_occupied_cell_retention": len(fps_cells) / len(all_cells) if all_cells else None,
            "lateral_slices": slices,
        }
        if name == "B5" and (
            record["near_candidate_real_hits"] != 31031
            or record["near_fps_real_hits"] != 256
        ):
            raise AssertionError("B5 previous same-seed clean baseline no longer matches")
        results.append(record)
        data[name] = {
            "near_pool_hits": near_all, "near_fps_hits": near_selected,
            "near_fps_misses": near_misses, "far_selected_hits": far_selected,
        }
        print(f"{name} xmax={xmax} pool_hit={record['near_candidate_real_hits']}"
              f" fps_hit={record['near_fps_real_hits']} fps_cell={len(fps_cells)}/{len(all_cells)}"
              f" far64_hit={record['far_angular_real_hits']}"
              f" far_ge4m={record['far_selected_real_hit_x_ge_4m']}", flush=True)
    return results, data, raw_hit_count


def plot_results(out, rows, data, mount):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    # Horizontal reference shape, NOT the real robot's projected mesh.
    ref = np.array([
        [-0.89, -0.59], [0, -0.69], [0.89, -0.59],
        [0.89, 0.59], [0, 0.69], [-0.89, 0.59],
    ], dtype=float)

    def setup(ax, row):
        ax.add_patch(Polygon(ref, closed=True, fill=False, ls="--", lw=1.2,
                             edgecolor="#242424", zorder=5))
        ax.scatter([mount[0]], [mount[1]], color="#ed790a", edgecolors="white", s=43,zorder=6)
        ax.scatter([0], [0], marker="+", color="#222222", s=48, zorder=6)
        ax.annotate("", (0.5, 0), (0.05, 0),
                    arrowprops=dict(arrowstyle="->", lw=1.3, color="#222222"))
        if row["x_max_body_m"] is not None:
            ax.axvline(row["x_max_body_m"], color="#af511b", lw=1.4,
                       linestyle="--", zorder=5)
        ax.set(xlim=(-1, 6.7), ylim=(-3, 3))
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.14, linewidth=0.6)
        ax.set_xlabel("BASE x [m]")
        ax.set_ylabel("BASE y [m]")

    for mode in ("pool", "fps"):
        fig, axs = plt.subplots(2, 4, figsize=(18.8, 10.8), layout="constrained")
        for row,ax in zip(rows, axs.flat):
            d = data[row["id"]]
            if mode == "pool":
                pts = d["near_pool_hits"]
                pts = pts[::max(1,len(pts)//6500)]  # display-only deterministic thinning
                ax.scatter(pts[:,0],pts[:,1],s=1.25,c="#2479b6",alpha=0.28,rasterized=True)
                line2 = f"real near hits={row['near_candidate_real_hits']:,}"
            else:
                pts = d["near_fps_hits"]
                ax.scatter(pts[:,0],pts[:,1],s=9,c="#1867c1",alpha=0.8)
                if len(d["near_fps_misses"]):
                    p=d["near_fps_misses"]
                    ax.scatter(p[:,0],p[:,1],s=23,c="#eaac3a",marker="^")
                far=d["far_selected_hits"]
                ax.scatter(far[:,0],far[:,1],s=16,c="#934cac",alpha=0.9,marker="^")
                line2 = f"FPS hit={row['near_fps_real_hits']}/256  | far hit={row['far_angular_real_hits']}/64"
            setup(ax,row)
            label = "A0 original" if row["x_max_body_m"] is None else f"x_max={row['x_max_body_m']:.1f} m"
            ax.set_title(f"{row['id']}  | {label}\n{line2}",fontsize=10.7,weight="bold")
        title = (
            "Stage A: ACTUAL initial-scan near real-hit pool | theta=28°, tilt=56° for B candidates\n"
            "Same actual 86,400 rays for all panels; blue = measured floor hit endpoints. "
            "No provisional ground grid used for selection.\n"
            "A0 theta=32° no cutoff; dashed hexagon=nominal reference envelope, NOT robot mesh."
        ) if mode=="pool" else (
            "Stage A: ACTUAL initial-scan FPS256 / angular far64 | theta=28°, tilt=56° for B candidates\n"
            "Blue dots=near FPS real hits; orange triangles=near miss endpoints; purple triangles=far64 real hits.\n"
            "Dashed hexagon=nominal reference envelope, NOT robot mesh; far miss endpoints omitted from plot."
        )
        fig.suptitle(title,fontsize=12)
        fig.savefig(out / f"length_sweep_{mode}_8panels.png",dpi=160)
        plt.close(fig)


def main(argv=None):
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
    from hexapod_mjlab.tasks.confined_locomotion.flat_env_cfg import make_training_env_cfg
    from hexapod_mjlab.tasks.confined_locomotion.ground import (
        GroundGridCfg, VALIDITY_GEOMETRY_ONLY_FLAT_TRAINING,
    )
    from hexapod_mjlab.tasks.confined_locomotion.runtime import TrainingRuntime
    from hexapod_mjlab.tasks.confined_locomotion.scanner import ConfinedLidarCfg

    args = _args(argv)
    if args.gpu_id < 0 or not torch.cuda.is_available() or args.gpu_id >= torch.cuda.device_count():
        raise RuntimeError("explicit real CUDA device required")
    out = args.output.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    # Only to honor TrainingRuntime's explicit ground-grid contract; the
    # candidate grid will be redesigned AFTER a near spatial range is chosen.
    grid = GroundGridCfg(
        x_coords=(-0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6, 2.0),
        y_coords=(-0.8, -0.4, 0, 0.4, 0.8),
        validity_mode=VALIDITY_GEOMETRY_ONLY_FLAT_TRAINING,
    )
    paths = [
        Path(__file__),
        ROOT/"src/hexapod_mjlab/tasks/confined_locomotion/scanner.py",
        ROOT/"src/hexapod_mjlab/tasks/confined_locomotion/runtime.py",
        ROOT/"src/hexapod_mjlab/assets/robots/el_4090/mjcf/el_4090_lidar.xml",
    ]
    report = {
        "schema": "confined_locomotion.stageA_B_length_sweep/v1",
        "claim": "eight offline near/far partitions of the SAME real clean CUDA T1 initial scan; no train or source sensor changes",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed":args.seed,"device":f"cuda:{args.gpu_id}",
        "candidates":[{"id":n,"theta_deg":t,"tilt_deg":56.0,"x_max":x} for n,t,x in CANDIDATES],
        "grouping_rule": "near = cos(sensor-to-BASE-point, cone_axis) >= sin(theta) AND BASE_x <= x_max if present; far=valid complement",
        "ground_grid_status": "TEMPORARY runtime fixture only; never used in metrics, comparison, ranking, or chart.",
        "plane_scene": "flat only; robot-body occlusion absent; clean 15m instantaneous scan.",
        "interpretation_limits": "Single fixed restart pose, no obstacles/walls, clean scan only. Longer near region may dilute FPS256 even if candidate coverage rises. No learned performance or real-lidar safety conclusion.",
        "sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "state":"started",
    }
    (out/"recipe.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    env=runtime=None
    try:
        torch.cuda.set_device(args.gpu_id)
        env=ManagerBasedRlEnv(make_training_env_cfg(num_envs=1,seed=args.seed),
                              device=f"cuda:{args.gpu_id}")
        wrapper=RslRlVecEnvWrapper(env,clip_actions=None)
        runtime=TrainingRuntime(wrapper,grid_cfg=grid,seed=args.seed,
                                output_dir=out/"runtime",mode="engineering",
                                lidar_cfg=ConfinedLidarCfg(),perturbation_cfg=None)
        scans=[item["frame"] for item in runtime.pipeline.scan_records if "frame" in item]
        assert len(scans)==1 and scans[0].episode_id==0 and scans[0].scan_id==1
        rows,data,num_hit=measure(scans[0],runtime.pipeline)
        plot_results(out,rows,data,runtime.pipeline.mount_pos)
        csv_keys=("id","theta_deg","x_max_body_m","near_candidates","near_candidate_real_hits",
                  "near_candidate_misses","near_fps_real_hits","near_fps_misses",
                  "near_candidate_hit_cells","near_fps_hit_cells",
                  "near_fps_occupied_cell_retention","far_candidates",
                  "far_angular_real_hits","far_selected_real_hit_x_ge_4m",
                  "near_real_hit_max_x")
        with (out/"summary.csv").open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=csv_keys);w.writeheader()
            for row in rows:w.writerow({k:row[k] for k in csv_keys})
        report.update(state="completed",scan_identity=dict(
            episode_id=scans[0].episode_id,scan_id=scans[0].scan_id,
            time_s=scans[0].time_s,control_steps=runtime.control_steps_taken),
            raw_rays=86400,raw_real_hits=num_hit,
            mount_base_xyz=runtime.pipeline.mount_pos.tolist(), measurements=rows)
    except BaseException as error:
        report.update(state="failed",error=repr(error))
        raise
    finally:
        try:
            if runtime is not None:runtime.close()
        finally:
            if env is not None:env.close()
            (out/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
            print(f"[length_sweep] {report['state']} : {out/'report.json'}",flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
