#!/usr/bin/env python
"""Checkpoint-free stage-A scan and ground-grid coverage diagnostic.

Creates one REAL confined flat environment and its task TrainingRuntime, reads
the reset-time scan and geometry-only ground labels, and writes a BEV figure
and reproducible numeric evidence. No Actor/Critic, optimizer, PPO update,
physical control step, or automatic adoption of grid/cone parameters.

Use on the actual CUDA device with an EXPLICIT candidate grid. The output
directory must not exist. A single T1 reset is a diagnostic case, not evidence
of dynamic gait coverage or final parameter acceptance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True, help="new non-existing evidence directory")
    p.add_argument("--grid-x", type=float, nargs="+", required=True, help="ordered candidate body-X coordinates (m)")
    p.add_argument("--grid-y", type=float, nargs="+", required=True, help="ordered mirror-paired body-Y coordinates (m)")
    p.add_argument("--seed", type=int, default=20261014)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--theta", type=float, default=32.0, help="candidate near cone split theta (deg)")
    p.add_argument("--tilt", type=float, default=56.0, help="candidate cone-axis tilt delta (deg)")
    return p.parse_args(argv)


def _cpu(tensor):
    return tensor.detach().cpu().numpy().copy()


def coverage_metrics(frame, ground_label, *, thresholds=(0.10, 0.20, 0.30)):
    """Pure numerical metrics from production provenance, not a reconstructed FPS.

    All spatial distances are computed in the same acquisition-time BASE XYZ.
    Coverage is proximity evidence, not a proof of observable ground surfaces.
    Geometry-valid ground labels remain supervised even when unsupported.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    raw_miss = _cpu(frame.no_hit_mask).astype(bool)
    near_mask = _cpu(frame.near_mask).astype(bool)
    far_mask = _cpu(frame.far_mask).astype(bool)
    pool_raw_ids = _cpu(frame.pool_raw_indices).astype(np.int64)
    pool_pts = _cpu(frame.pool_points_base)
    selected_raw = _cpu(frame.near_indices).astype(np.int64)
    selected_valid = _cpu(frame.near_valid).astype(bool)
    far_raw = _cpu(frame.far_indices).astype(np.int64)
    far_valid = _cpu(frame.far_valid).astype(bool)
    ground_xyz = _cpu(ground_label.positions[0])
    ground_valid = _cpu(ground_label.valid[0]).astype(bool)

    assert len(raw_miss) == 86400, "the diagnostic requires the full Airy scan"
    assert len(np.unique(selected_raw[selected_valid])) == int(selected_valid.sum())
    assert len(np.unique(far_raw[far_valid])) == int(far_valid.sum())
    assert np.all(selected_raw[~selected_valid] == -1)
    assert np.all(far_raw[~far_valid] == -1)
    assert not np.any(near_mask & far_mask)
    assert np.array_equal(near_mask | far_mask, np.ones_like(near_mask))
    assert np.all(np.isin(selected_raw[selected_valid], pool_raw_ids[near_mask]))
    assert np.all(np.isin(far_raw[far_valid], pool_raw_ids[far_mask]))

    near_hit_pool = near_mask & ~raw_miss[pool_raw_ids]
    near_hit_sample = selected_valid & ~raw_miss[np.maximum(selected_raw, 0)]
    far_hit_sample = far_valid & ~raw_miss[np.maximum(far_raw, 0)]
    query = ground_xyz[ground_valid]
    pool_ground = pool_pts[near_hit_pool]
    sampled_ground = _cpu(frame.near_points_base)[near_hit_sample]

    def distances(support):
        if not len(query) or not len(support):
            return None
        d = cKDTree(support).query(query, k=1)[0]
        return {
            "nearest_3d_m": {
                "p50": float(np.percentile(d, 50)),
                "p95": float(np.percentile(d, 95)),
                "max": float(d.max()),
            },
            "grid_points_within_distance": {
                f"{threshold:.2f}m": {
                    "count": int((d <= threshold).sum()),
                    "fraction_of_valid_grid": float(np.mean(d <= threshold)),
                }
                for threshold in thresholds
            },
        }

    details = {
        "raw_rays": len(raw_miss),
        "raw_hits": int((~raw_miss).sum()),
        "raw_misses": int(raw_miss.sum()),
        "pool_total": int(len(pool_raw_ids)),
        "near_candidates": int(near_mask.sum()),
        "far_candidates": int(far_mask.sum()),
        "near_candidates_hit": int(near_hit_pool.sum()),
        "near_valid_selected": int(selected_valid.sum()),
        "near_selected_hit": int(near_hit_sample.sum()),
        "near_selected_miss": int(selected_valid.sum() - near_hit_sample.sum()),
        "far_valid_selected": int(far_valid.sum()),
        "far_selected_hit": int(far_hit_sample.sum()),
        "far_selected_miss": int(far_valid.sum() - far_hit_sample.sum()),
        "ground_grid_total": int(ground_valid.size),
        "ground_grid_geometry_valid": int(ground_valid.sum()),
        "ground_grid_geometry_valid_fraction": float(ground_valid.mean()),
        "near_hit_pool_to_ground_grid": distances(pool_ground),
        "near_hit_FPS_to_ground_grid": distances(sampled_ground),
        "distance_definition": "acquisition-time BASE frame 3D Euclidean nearest-neighbor; genuine ground hits only",
        "limitation": (
            "A small nearest-neighbor distance indicates geometric proximity, not "
            "ray visibility, terrain semantic correctness, sufficient learned information, "
            "or dynamic gait coverage. A miss endpoint at 15m never counts as a hit."
        ),
    }
    plot = {
        "near_pool_hits": pool_ground,
        "near_FPS_hits": sampled_ground,
        "near_FPS_misses": _cpu(frame.near_points_base)[selected_valid & ~near_hit_sample],
        "far_samples_hits": _cpu(frame.far_points_base)[far_hit_sample],
        "grid_points": ground_xyz,
        "grid_valid": ground_valid,
    }
    return details, plot


def save_figure(output, plot, metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8), layout="constrained")
    for key, label, size, alpha, c in (
        ("near_pool_hits", "Near full-pool real hits", 1.3, 0.20, "#7b8b92"),
        ("near_FPS_hits", "Near FPS256 real hits", 18, 0.95, "#1877c9"),
        ("near_FPS_misses", "Near FPS256 15m miss endpoints", 12, 0.45, "#e29d39"),
        ("far_samples_hits", "Far selected real hits", 12, 0.60, "#9b59b6"),
    ):
        xyz = plot[key]
        if len(xyz):
            ax.scatter(xyz[:, 0], xyz[:, 1], s=size, alpha=alpha, c=c, label=label, rasterized=True)
    grid = plot["grid_points"]
    valid = plot["grid_valid"]
    ax.scatter(grid[valid, 0], grid[valid, 1], marker="x", c="#c6342e",
               s=50, linewidths=1.25, label="Geometry-valid supervision grid")
    if (~valid).any():
        ax.scatter(grid[~valid, 0], grid[~valid, 1], marker="x", c="black",
                   s=40, label="Geometry-invalid grid")
    ax.scatter([0], [0], marker="+", s=140, c="black", label="BASE origin")
    ax.set(xlabel="BASE x (m)", ylabel="BASE y (m)",
           title="Confined flat / real production scan / T1 reset / clean\n"
                 f"near FPS hit={metrics['near_selected_hit']}/256, "
                 f"geometry-valid grid={metrics['ground_grid_geometry_valid']}/{metrics['ground_grid_total']}")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlim(-2, 5)
    ax.set_ylim(-4, 4)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(output / "coverage_bev.png", dpi=160)
    plt.close(fig)


def main(argv=None):
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
    from hexapod_mjlab.rl.confined_locomotion.symmetry import GridPermutation
    from hexapod_mjlab.tasks.confined_locomotion.flat_env_cfg import make_training_env_cfg
    from hexapod_mjlab.tasks.confined_locomotion.ground import GroundGridCfg, VALIDITY_GEOMETRY_ONLY_FLAT_TRAINING
    from hexapod_mjlab.tasks.confined_locomotion.runtime import TrainingRuntime
    from hexapod_mjlab.tasks.confined_locomotion.scanner import ConfinedLidarCfg
    args = parse_args(argv)
    grid = GroundGridCfg(
        x_coords=tuple(args.grid_x), y_coords=tuple(args.grid_y),
        validity_mode=VALIDITY_GEOMETRY_ONLY_FLAT_TRAINING,
    )
    GridPermutation.from_axes(grid.x_coords, grid.y_coords)
    lidar_cfg = ConfinedLidarCfg(split_theta_deg=args.theta, split_axis_tilt_deg=args.tilt)
    if not (0 < args.theta < 90 and -180 < args.tilt < 180):
        raise ValueError("candidate cone angles outside diagnostic range")
    if args.gpu_id < 0 or not torch.cuda.is_available() or args.gpu_id >= torch.cuda.device_count():
        raise RuntimeError("a real selected CUDA device is required")
    out = args.output.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    paths = [
        Path(__file__),
        ROOT / "src/hexapod_mjlab/tasks/confined_locomotion/scanner.py",
        ROOT / "src/hexapod_mjlab/tasks/confined_locomotion/ground.py",
        ROOT / "src/hexapod_mjlab/tasks/confined_locomotion/runtime.py",
        ROOT / "src/hexapod_mjlab/assets/robots/el_4090/mjcf/el_4090_lidar.xml",
    ]
    evidence = {
        "schema": "confined_locomotion.preflight_scan/v1",
        "claim": "one real CUDA reset-time production scan; no PPO or physics stepping; candidate ground-grid fixture only",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed, "device": f"cuda:{args.gpu_id}",
        "grid": grid.identity(), "grid_hash": grid.identity_hash(),
        "near_partition": {
            "theta_deg": args.theta,
            "tilt_deg": args.tilt,
            "base_x_max_m": lidar_cfg.split_max_forward_x_m,
            "rule": "cone AND BASE endpoint x<=base_x_max_m; far=complement",
        },
        "clean_observation": True,
        "sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "state": "started",
    }
    (out / "recipe.json").write_text(json.dumps(evidence, indent=2))
    env = runtime = None
    try:
        torch.cuda.set_device(args.gpu_id)
        env = ManagerBasedRlEnv(make_training_env_cfg(num_envs=1, seed=args.seed), device=f"cuda:{args.gpu_id}")
        wrapper = RslRlVecEnvWrapper(env, clip_actions=None)
        runtime = TrainingRuntime(
            wrapper, grid_cfg=grid, seed=args.seed, output_dir=out / "runtime",
            mode="engineering", lidar_cfg=lidar_cfg, perturbation_cfg=None,
        )
        prepared = runtime.prepare()
        labels = runtime.labels()
        scans = [item["frame"] for item in runtime.pipeline.scan_records if "frame" in item]
        if len(scans) != 1:
            raise RuntimeError(f"expected exactly one initial physical scan, got {len(scans)}")
        frame = scans[0]
        if frame.scan_id != 1 or frame.episode_id != 0:
            raise RuntimeError("initial scan identity is not episode 0, scan 1")
        metrics, plot = coverage_metrics(frame, labels["ground"]["label"])
        evidence["state"] = "measured"
        evidence["measurements"] = metrics
        evidence["scan_identity"] = {
            "scan_id": frame.scan_id, "episode_id": frame.episode_id, "time_s": frame.time_s,
            "num_control_steps": runtime.control_steps_taken,
        }
        evidence["plane"] = runtime.pipeline.plane_record
        evidence["lidar_mount_base_xyz"] = runtime.pipeline.mount_pos.tolist()
        evidence["ground_identity"] = labels["ground"]["plane"]
        evidence["input_identity"] = {
            "step_ids": _cpu(prepared.observation.step_ids).tolist(),
            "episode_ids": _cpu(prepared.observation.episode_ids).tolist(),
        }
        save_figure(out, plot, metrics)
        evidence["state"] = "completed"
    except BaseException as err:
        evidence["state"] = "failed"
        evidence["error"] = repr(err)
        raise
    finally:
        try:
            if runtime is not None:
                runtime.close()
        finally:
            if env is not None:
                env.close()
            (out / "report.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2, default=str))
            print(f"[preflight_scan] {evidence['state']}: {out / 'report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
