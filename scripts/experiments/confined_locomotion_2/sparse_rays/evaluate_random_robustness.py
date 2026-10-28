#!/usr/bin/env python3
"""Seeded robustness probe for the frozen Airy channel recipes in a manifest.

READ-ONLY experiment, no production/FPS/training changes. Test three baselines:
  A0 fixed max-norm FPS seed, clean rays (once per scene);
  B  random FPS start, CLEAN rays, repeated per-trial;
  C  random FPS start + physically aligned range noise sigma=0.015m,
     applied once per ray before FPS. Misses stay exactly 15m; fixed
     catalog near/far ownership never changes.
Both B and C use the same independent random start quantiles per trial.
C draws an identical full-frame ray-ID-indexed noise vector across all budgets.
This is an INDEPENDENT reproduction of the current simulation's 15mm
observation-level range perturbation; no production perturbation is called.

Geometry correctness is scored on ORIGINAL clean ray hit coordinates, NOT on
the noisy versions (which could spuriously inflate apparent support).
Reports include object hit retention, geometry support, sample-ID variability,
and paired trials; offline random-start sampler is not the production FPS.

Use a new output directory; do not overwrite earlier evidence.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from select_channels import (
    FULL_RAYS, MAX_RANGE_M, NEAR_OUT, FAR_OUT, THETA_DEG,
    NEAR_MAX_X_M, airy_directions, nominal_mount, sha256,
)
from evaluate_moving_footprint import cone_axis_body, read_frozen_design, resolve_recipe_names
from evaluate_obstacles import SCENES, scan_obstacles, surface_probes
from compare_sampling import fps_ids
from hexapod_mjlab.tasks.confined_locomotion_2.components.lidar.channels import build_channel_catalog


# Main multiplier selection uses traversable/task-relevant geometry only.
# Do NOT use 3/5 cm thin posts as selection criteria: even the full 384-point
# FPS retains only ~2 points there; that primarily tests FPS micro-feature
# preservation rather than the sparse ray budget. Older boundary diagnostics
# remain immutable in their historical evidence directory.
EVAL_NAMES = (
    "step_10cm", "step_25cm", "step_10cm_pitch_down_5",
    "step_25cm_pitch_up_5", "slim_curb_10cm", "front_wall_60cm",
)


def stress_scenes():
    chosen = [s for s in SCENES if s["name"] in EVAL_NAMES]
    # Keep dimensions substantial, vary only the geometric placement.
    for dx in (-0.25, +0.25):
        chosen.append({
            "name": f"step_25cm_shift_x_{dx:+.2f}",
            "pitch": 0.0, "roll": 0.0,
            "boxes": (("step", (1.3 + dx, -0.9, 0.0),
                       (2.8 + dx, 0.9, 0.25)),),
        })
    # Side-wall faces representative of 1.6 m confined corridors; separate
    # object labels let the benchmark report each wall independently.
    chosen.extend([
        {
            "name": "corridor_left_wall_roll_5",
            "pitch": 0.0, "roll": 5.0,
            "boxes": (("left_wall", (-1.5, 0.80, 0.0),
                       (5.0, 0.83, 2.0)),),
        },
        {
            "name": "corridor_right_wall_roll_5",
            "pitch": 0.0, "roll": 5.0,
            "boxes": (("right_wall", (-1.5, -0.83, 0.0),
                       (5.0, -0.80, 2.0)),),
        },
    ])
    if len(chosen) != len(EVAL_NAMES) + 4:
        raise RuntimeError("evaluation scene list changed")
    return chosen


def random_start_fps(points: np.ndarray, ids: np.ndarray, k: int,
                     uniform_start: float) -> np.ndarray:
    """Same max-min Euclidean FPS rule, replacing ONLY its first seed.

    A seed is selected via a quantile of the compact sorted valid pool.
    Thus the same trial's random quantile is paired across budgets.
    Distances use float32, first-index ties and unique source IDs.
    """
    ids = np.asarray(ids, dtype=np.int64)
    if not len(ids):
        return ids
    if not 0 <= uniform_start < 1:
        raise ValueError("start must lie in [0,1)")
    ids = np.sort(ids)
    if len(ids) <= k:
        return ids.copy()
    pool = np.asarray(points[ids], dtype=np.float32)
    cur = min(int(uniform_start * len(ids)), len(ids) - 1)
    out = np.empty(k, dtype=np.int64)
    nearest = np.full(len(ids), np.inf, dtype=np.float32)
    used = np.zeros(len(ids), dtype=bool)
    for i in range(k):
        out[i] = ids[cur]
        used[cur] = True
        delta = pool - pool[cur]
        d2 = np.einsum("ij,ij->i", delta, delta, optimize=False)
        np.minimum(nearest, d2, out=nearest)
        nearest[used] = -1
        if i + 1 < k:
            cur = int(np.argmax(nearest))
    return out


def noisy_geometry(scan: dict, source_dirs: np.ndarray,
                   mount: dict, sigma_m: float,
                   rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Perturb true hit distances once per ray; preserve real no-hit endpoint.

    Fixed near/far channel ownership is invariant to measurement noise. Noise
    changes points consumed by FPS, never the catalog's source-ID assignment.
    """
    if sigma_m <= 0:
        raise ValueError("sigma must be positive")
    body_dirs = (mount["mount_rot_base"] @ source_dirs.astype(np.float64).T).T
    origin = mount["mount_pos_base"]
    raw_dist = scan["distances_m"]
    miss = scan["hit_label"] == -1
    epsilon = rng.normal(0.0, sigma_m, size=FULL_RAYS)
    noisy_dist = np.where(miss, MAX_RANGE_M,
                          np.clip(raw_dist + epsilon, 0.0, MAX_RANGE_M))
    pts = origin + noisy_dist[:, None] * body_dirs
    return pts, build_channel_catalog().near_mask


def select_both(xyz: np.ndarray, near: np.ndarray,
                source_ids: np.ndarray, *,
                start_near: float | None, start_far: float | None) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(source_ids, dtype=np.int64)
    near_ids = ids[near[ids]]
    far_ids = ids[~near[ids]]
    if start_near is None:
        return fps_ids(xyz, near_ids, NEAR_OUT), fps_ids(xyz, far_ids, FAR_OUT)
    return (random_start_fps(xyz, near_ids, NEAR_OUT, start_near),
            random_start_fps(xyz, far_ids, FAR_OUT, start_far))


def measurement(scan: dict, picked: tuple[np.ndarray, np.ndarray],
                probe: np.ndarray, label: int) -> dict:
    """Use clean true-surface returns of selected source IDs as the oracle.

    This makes B/C measures comparable on the same ground truth geometry,
    even when noisy coordinates influence FPS decisions.
    """
    near, far = picked
    keep = np.concatenate((near, far))
    sel = keep[scan["hit_label"][keep] == label]
    if len(sel) == 0:
        return {"hits": 0, "near_hits": 0, "far_hits": 0,
                "support20": 0.0, "support10": 0.0}
    dist = cKDTree(scan["points"][sel]).query(probe, k=1, workers=1)[0]
    return {
        "hits": int(len(sel)),
        "near_hits": int(np.sum(scan["hit_label"][near] == label)),
        "far_hits": int(np.sum(scan["hit_label"][far] == label)),
        "support20": float(np.mean(dist <= 0.20)),
        "support10": float(np.mean(dist <= 0.10)),
    }


def wilson_upper_zero_failures(n: int, z: float = 1.96) -> float:
    if n <= 0:
        raise ValueError("need at least one trial")
    return z * z / (n + z * z)


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("no measurement")
    hits = np.array([r["hits"] for r in rows], dtype=np.int64)
    cover = np.array([r["support20"] for r in rows], dtype=np.float64)
    support10 = np.array([r["support10"] for r in rows], dtype=np.float64)
    near_count = np.array([r["near_candidates"] for r in rows])
    far_count = np.array([r["far_candidates"] for r in rows])
    return {
        "trials": int(len(rows)),
        "hits_mean": float(hits.mean()),
        "hits_p05": float(np.percentile(hits, 5)),
        "hits_min": int(hits.min()),
        "hits_max": int(hits.max()),
        "zero_hit_count": int(np.sum(hits == 0)),
        "zero_hit_fraction": float(np.mean(hits == 0)),
        "zero_failure_wilson95_upper_when_no_misses": (
            wilson_upper_zero_failures(len(rows))
            if not np.any(hits == 0) else None
        ),
        "support20_mean": float(cover.mean()),
        "support20_p05": float(np.percentile(cover, 5)),
        "support20_min": float(cover.min()),
        "support10_mean": float(support10.mean()),
        "near_candidate_count_min": int(near_count.min()),
        "near_candidate_count_max": int(near_count.max()),
        "far_candidate_count_min": int(far_count.min()),
        "far_candidate_count_max": int(far_count.max()),
    }


def describe_sequences(sequences: list[tuple[np.ndarray, np.ndarray]],
                       true_points: np.ndarray | None = None,
                       true_hit_mask: np.ndarray | None = None) -> dict:
    if len(sequences) <= 1:
        return {"unique_near_source_ids_across_trials": None,
                "consecutive_selected_id_jaccard_mean": None}
    near_sets = [set(p[0].tolist()) for p in sequences]
    total = set().union(*near_sets)
    jacc = []
    spatial_p95 = []
    if (true_points is None) != (true_hit_mask is None):
        raise ValueError("true points and hit mask must be specified together")
    for (a, b), (seq_a, seq_b) in zip(
        zip(near_sets[:-1], near_sets[1:]),
        zip(sequences[:-1], sequences[1:]),
    ):
        jacc.append(len(a & b) / len(a | b))
        if true_points is not None:
            # Compare the FINAL selected TRUE surface geometry from each
            # independently sampled observation. Do not count far-endpoint
            # no-hits as actual geometric surface, or noisy xyz as truth.
            ia = seq_a[0][true_hit_mask[seq_a[0]]]
            ib = seq_b[0][true_hit_mask[seq_b[0]]]
            if len(ia) and len(ib):
                xa, xb = true_points[ia], true_points[ib]
                d_ab = cKDTree(xb).query(xa, k=1, workers=1)[0]
                d_ba = cKDTree(xa).query(xb, k=1, workers=1)[0]
                spatial_p95.append(float(np.percentile(
                    np.concatenate((d_ab, d_ba)), 95
                )))
    return {
        "unique_near_source_ids_across_trials": len(total),
        "consecutive_selected_id_jaccard_mean": float(np.mean(jacc)),
        "consecutive_selected_id_jaccard_p05": float(np.percentile(jacc, 5)),
        "consecutive_true_near_surface_symmetric_nn_p95_m_mean": (
            float(np.mean(spatial_p95)) if spatial_p95 else None
        ),
        "consecutive_true_near_surface_symmetric_nn_p95_m_p95": (
            float(np.percentile(spatial_p95, 95)) if spatial_p95 else None
        ),
        "spatial_pair_count": len(spatial_p95),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="must NOT exist")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=20261023)
    parser.add_argument("--sigma-m", type=float, default=0.015)
    parser.add_argument("--recipes", nargs="+", help="optional manifest recipe subset; default: all")
    args = parser.parse_args(argv)
    if args.trials < 2 or args.base_seed < 0 or not np.isfinite(args.sigma_m) or args.sigma_m <= 0:
        raise ValueError("need >=2 trials, nonnegative seed and positive finite sigma")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"evidence already exists: {output}")
    design = args.design.expanduser().resolve()
    source_dirs = airy_directions()
    mount = nominal_mount()
    manifest, recipes = read_frozen_design(design, source_dirs)
    wanted = resolve_recipe_names(recipes, args.recipes)

    result = {}
    for scene_i, case in enumerate(stress_scenes()):
        scan = scan_obstacles(source_dirs, mount, case)
        if len(case["boxes"]) != 1:
            # Multi-object scenes need separate per-object counters; here
            # deliberately select single-object cases for clarity.
            continue
        object_label = 1
        oracle = surface_probes(scan["points"][scan["hit_label"] == object_label])
        if not len(oracle):
            raise RuntimeError(f"unobservable scenario: {case['name']}")
        baseline_full = select_both(
            scan["points"], scan["near"],
            np.arange(FULL_RAYS, dtype=np.int64),
            start_near=None, start_far=None,
        )
        full_metrics = measurement(scan, baseline_full, oracle, object_label)
        modes = ("random_start_clean", "random_start_range_noise")
        buffers = {
            key: {"series": {mode: [] for mode in modes},
                  "sequences": {mode: [] for mode in modes}}
            for key in wanted
        }
        fixed_metrics = {}
        for key in wanted:
            fixed = select_both(scan["points"], scan["near"], recipes[key],
                                start_near=None, start_far=None)
            fixed_metrics[key] = measurement(scan, fixed, oracle, object_label)
        # The full-frame noisy observation is drawn ONCE per trial and
        # reused by all budgets; fixed channel ownership is unchanged.
        for trial in range(args.trials):
            rng_start = np.random.default_rng(np.random.SeedSequence(
                [args.base_seed, scene_i, trial, 1]))
            u_near, u_far = rng_start.random(2)
            rng_noise = np.random.default_rng(np.random.SeedSequence(
                [args.base_seed, scene_i, trial, 2]))
            noisy_xyz, noisy_near = noisy_geometry(
                scan, source_dirs, mount, args.sigma_m, rng_noise
            )
            for mode, xyz, near in (
                ("random_start_clean", scan["points"], scan["near"]),
                ("random_start_range_noise", noisy_xyz, noisy_near),
            ):
                for key in wanted:
                    ids = recipes[key]
                    pool_near = ids[near[ids]]
                    picked = select_both(
                        xyz, near, ids,
                        start_near=float(u_near), start_far=float(u_far)
                    )
                    meas = measurement(scan, picked, oracle, object_label)
                    meas["near_candidates"] = int(len(pool_near))
                    meas["far_candidates"] = int(len(ids) - len(pool_near))
                    buffers[key]["series"][mode].append(meas)
                    buffers[key]["sequences"][mode].append(picked)
        recipe_runs = {
            key: {
                "static_raycasts": int(len(recipes[key])),
                "a0_fixed_clean": fixed_metrics[key],
                "modes": {
                    mode: {
                        **aggregate(buffers[key]["series"][mode]),
                        **describe_sequences(
                            buffers[key]["sequences"][mode],
                            true_points=scan["points"],
                            true_hit_mask=(scan["hit_label"] != -1),
                        ),
                    } for mode in modes
                },
            }
            for key in wanted
        }
        result[case["name"]] = {
            "geometry": {"pitch_deg": case["pitch"], "roll_deg": case["roll"],
                         "objects": case["boxes"]},
            "full_scan_fixed_clean": full_metrics,
            "full_raw_hit_rays": int(np.sum(scan["hit_label"] == 1)),
            "full_raw_surface_probes": int(len(oracle)),
            "multipliers": recipe_runs,
        }
        print(f"PASS {case['name']}: {len(oracle)} independent surface probes", flush=True)

    output.mkdir(parents=True, exist_ok=False)
    payload = {
        "schema": "confined_locomotion_2.sparse_rays.random_robustness/v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "design_manifest_sha256": sha256(design / "manifest.json"),
        "evaluator_sha256": sha256(Path(__file__)),
        "trials_per_mode": args.trials,
        "seed": args.base_seed,
        "measurement": {
            "same_ray_xyz_truth_from_clean_analytic_geometry": True,
            "spatial_diversity": "between consecutive random-start FPS outputs; symmetric 3D NN P95 over real near-surface points only, excluding 15m no-hits",
            "noise_m": args.sigma_m,
            "full_frame_normal_draw_per_new_trial_and_scene": True,
            "no_hit_exact_max_range_m": MAX_RANGE_M,
            "reapply_B8_endpoint_x_after_noise": False,
            "static_catalog_ownership": True,
            "fixed_channel_ownership_survives_noise": True,
            "random_start": "uniform on sorted CURRENT valid pool (paired quantiles)",
            "actual_output": "near256 + far128, with no replacement, separate FPS",
            "range_noise": "applied before FPS; never changes fixed channel ownership",
            "FPS_shape": "originally fixed farthest-norm start; only offline experiment replaces seed",
            "no_dataset_train_or_sim2real_success_claim": True,
            "zero_failure_wilson95_upper": "approx formula z^2/(n+z^2), only if 0 observed misses",
            "source_order_identity": "original Airy az*96+el ray IDs",
        },
        "scenes": result,
    }
    (output / "report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = []
    for name, scene in result.items():
        for key, data in scene["multipliers"].items():
            row = {"case": name, "scheme": key, "fixed_hits": data["a0_fixed_clean"]["hits"]}
            for mode in data["modes"]:
                agg = data["modes"][mode]
                row[mode + "_zero"] = agg["zero_hit_count"]
                row[mode + "_support20"] = round(agg["support20_mean"], 3)
                row[mode + "_jaccard"] = round(agg["consecutive_selected_id_jaccard_mean"], 3)
            summary.append(row)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    print(json.dumps({"output": str(output),
                      "scenes": len(result), "schemes": len(wanted),
                      "trials_per_mode": args.trials},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
