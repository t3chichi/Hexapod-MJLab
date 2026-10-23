#!/usr/bin/env python3
"""Compare fixed sparse physical Airy IDs with full-ray XYZ FPS analytically.

Reads frozen experiment .npz/manifest files; does NOT modify training or FPS.
All scenes are analytic *diagnostics*, NOT Warp raycasts / real terrain.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from select_channels import (
    FULL_RAYS, MAX_RANGE_M, NEAR_OUT, FAR_OUT, airy_directions,
    nominal_mount, sha256, nominal_scan, spatial_order, THETA_DEG, TILT_DEG, NEAR_MAX_X_M,
)
from hexapod_mjlab.tasks.confined_locomotion_2.channels import build_channel_catalog


# With BASE +x forward and +z up, scipy +Y pitch tips the nose DOWN.
# Historical stage1 reports mislabeled +5deg as "pitch_up"; retain those raw
# reports, but new runs use physically correct case names.
CASES = (
    ("flat", None, 0.0, 0.0),
    ("corridor_1p6", 1.6, 0.0, 0.0),
    ("corridor_2p0", 2.0, 0.0, 0.0),
    ("flat_pitch_up_5", None, -5.0, 0.0),
    ("flat_pitch_down_5", None, 5.0, 0.0),
    ("flat_pitch_down_10", None, 10.0, 0.0),
    ("corridor_1p6_pitch_up_5", 1.6, -5.0, 0.0),
    ("corridor_1p6_pitch_down_5", 1.6, 5.0, 0.0),
    ("corridor_2p0_roll_left_5", 2.0, 0.0, 5.0),
)


def intersect_scene(sensor_dirs: np.ndarray, mount: dict, *, width: float | None,
                    pitch_deg: float, roll_deg: float,
                    height_delta_m: float = 0.0) -> dict:
    """Sensor-first hit with infinite z=0 floor and optional finite vertical walls.

    World = BASE nominal axes at world x/y zero and world z=root stand height.
    Mount stays rigidly attached to BASE under roll/pitch perturbations.
    Sign: positive pitch is NOSE-DOWN, negative pitch is NOSE-UP for +x forward.
    """
    rot_delta = Rotation.from_euler("xyz", [roll_deg, pitch_deg, 0], degrees=True).as_matrix()
    rot_body_world = mount["base_rot_world"] @ rot_delta
    body_dirs = (mount["mount_rot_base"] @ sensor_dirs.astype(np.float64).T).T
    world_dirs = (rot_body_world @ body_dirs.T).T
    origin_body = mount["mount_pos_base"]
    root_world = mount["base_pos_world"] + np.array([0.0, 0.0, height_delta_m])
    origin_world = root_world + rot_body_world @ origin_body
    dz = world_dirs[:, 2]
    floor_t = np.full(FULL_RAYS, np.inf)
    np.divide(-origin_world[2], dz, out=floor_t, where=np.abs(dz) > 1e-12)
    floor_t[(floor_t <= 0) | (floor_t >= MAX_RANGE_M)] = np.inf
    hit_type = np.zeros(FULL_RAYS, dtype=np.int8)  # 0 miss, 1 floor, 2 wall
    best = floor_t.copy()
    hit_type[np.isfinite(floor_t)] = 1

    if width is not None:
        for y in (-width / 2.0, width / 2.0):
            dy = world_dirs[:, 1]
            t = np.full(FULL_RAYS, np.inf)
            np.divide(y - origin_world[1], dy, out=t, where=np.abs(dy) > 1e-12)
            x = origin_world[0] + t * world_dirs[:, 0]
            z = origin_world[2] + t * world_dirs[:, 2]
            ok = (t > 0) & (t < MAX_RANGE_M) & (x >= -2.0) & (x <= 6.0) & (z >= 0) & (z <= 4.0)
            nearer = ok & (t < best)
            best[nearer] = t[nearer]
            hit_type[nearer] = 2

    dist = np.where(np.isfinite(best), best, MAX_RANGE_M)
    points_base = origin_body + dist[:, None] * body_dirs
    delta = np.deg2rad(TILT_DEG)
    dome = mount["mount_rot_base"] @ np.array([0.0, 0.0, 1.0])
    axis = np.array([np.cos(delta) * dome[0] + np.sin(delta) * dome[2], dome[1],
                     -np.sin(delta) * dome[0] + np.cos(delta) * dome[2]])
    axis /= np.linalg.norm(axis)
    in_cone = (body_dirs @ axis) >= np.sin(np.deg2rad(THETA_DEG))
    # Channel ownership is frozen from the catalog's T1 nominal-plane B8 mask.
    # Keep per-scene recomputation only as an explicit diagnostic field.
    dynamic_near = in_cone & (points_base[:, 0] <= NEAR_MAX_X_M)
    fixed_near = build_channel_catalog().near_mask if _FIXED_NEAR is None else _FIXED_NEAR
    return dict(points=points_base, hit_type=hit_type, near=fixed_near.copy(),
                far=~fixed_near, dynamic_near=dynamic_near)


_FIXED_NEAR = None


def fps_ids(points: np.ndarray, ids: np.ndarray, count: int) -> np.ndarray:
    """CPU numerical reference with _2 XYZ FPS seed/metric/source-order semantics."""
    ids = np.sort(np.asarray(ids, dtype=np.int64))
    if len(ids) <= count:
        return ids.copy()
    p = np.asarray(points[ids], dtype=np.float32)
    seed = int(np.argmax(np.linalg.norm(p, axis=1)))
    selected = np.empty(count, dtype=np.int64)
    used = np.zeros(len(ids), dtype=bool)
    mind = np.full(len(ids), np.inf, dtype=np.float32)
    current = seed
    for i in range(count):
        selected[i] = ids[current]
        used[current] = True
        d = np.sum((p - p[current]) ** 2, axis=1)
        np.minimum(mind, d, out=mind)
        mind[used] = -1
        if i + 1 < count:
            current = int(np.argmax(mind))
    return selected


def coverage(reference: np.ndarray, chosen: np.ndarray) -> dict:
    """3D 0.2m voxelized reference hits, de-biased from raw angular density."""
    if len(reference) == 0:
        return {"status": "no_real_hit_probes"}
    vox = np.floor(reference / 0.2).astype(np.int64)
    _, unique_idx = np.unique(vox, axis=0, return_index=True)
    probes = reference[np.sort(unique_idx)]
    if len(chosen) == 0:
        return {"status": "no_selected_hit", "probes": len(probes)}
    nn = cKDTree(chosen).query(probes, k=1, workers=1)[0]
    return {
        "status": "ok",
        "probes": int(len(probes)),
        "mean_nn_m": float(nn.mean()),
        "p95_nn_m": float(np.percentile(nn, 95)),
        "max_nn_m": float(nn.max()),
        "within_0p20_fraction": float((nn <= 0.20).mean()),
        "within_0p25_fraction": float((nn <= 0.25).mean()),
    }


def evaluate(s: dict, ids: np.ndarray, full_near: np.ndarray, full_far: np.ndarray) -> dict:
    near_ids = ids[s["near"][ids]]
    far_ids = ids[s["far"][ids]]
    nsel = fps_ids(s["points"], near_ids, NEAR_OUT)
    fsel = fps_ids(s["points"], far_ids, FAR_OUT)
    near_hit = (s["hit_type"] != 0) & s["near"]
    far_hit = (s["hit_type"] != 0) & s["far"]
    near_ref = s["points"][near_hit]
    far_ref = s["points"][far_hit]
    return {
        "candidate_near": int(len(near_ids)), "candidate_far": int(len(far_ids)),
        "selected_near": int(len(nsel)), "selected_far": int(len(fsel)),
        "near_real_hit_selected": int(np.sum(s["hit_type"][nsel] != 0)),
        "far_real_hit_selected": int(np.sum(s["hit_type"][fsel] != 0)),
        "near_hit_coverage": coverage(near_ref, s["points"][nsel[s["hit_type"][nsel] != 0]]),
        "far_hit_coverage": coverage(far_ref, s["points"][fsel[s["hit_type"][fsel] != 0]]),
        "full_near_selected_overlap_fraction": float(
            len(np.intersect1d(nsel, full_near)) / max(1, len(full_near))),
        "full_far_selected_overlap_fraction": float(
            len(np.intersect1d(fsel, full_far)) / max(1, len(full_far))),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--design", type=Path, required=True, help="existing select_channels output")
    ap.add_argument("--output", type=Path, required=True, help="new comparison directory")
    args = ap.parse_args(argv)
    design = args.design.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}")
    global _FIXED_NEAR
    catalog = build_channel_catalog()
    _FIXED_NEAR = catalog.near_mask
    manifest_path = design / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != catalog.identity["schema"]:
        raise ValueError("unsupported fixed-channel manifest schema")
    if manifest.get("catalog_identity") != catalog.identity:
        raise ValueError("design catalog identity differs from current T1/AIRY baseline")

    mount = nominal_mount()
    rays = airy_directions()
    recipes = {}
    for key, entry in manifest["files"].items():
        file = design / entry["file"]
        if sha256(file) != entry["npz_sha256"]:
            raise ValueError(f"tampered/mismatched recipe: {key}")
        with np.load(file, allow_pickle=False) as pack:
            ids = pack["source_ray_ids"].astype(np.int64)
            d = pack["source_ray_directions_sensor"]
        if (len(ids) != len(np.unique(ids)) or
            np.any((ids < 0) | (ids >= FULL_RAYS)) or
            not np.allclose(d, rays[ids], rtol=0, atol=2e-7)):
            raise ValueError(f"invalid source ray ID/direction mapping: {key}")
        recipes[key] = ids

    # Fair same-budget angular-domain competitor. It is also composed only of
    # original Airy IDs, and follows the identical nominal B8 source pools.
    nominal = nominal_scan(rays, mount)
    near_pool = np.flatnonzero(catalog.near_mask)
    far_pool = np.flatnonzero(~catalog.near_mask)
    max_near = max(v["selection"]["near_count"] for v in manifest["files"].values())
    max_far = max(v["selection"]["far_count"] for v in manifest["files"].values())
    near_dir_order = spatial_order(nominal["directions_base"], near_pool, max_near)
    far_dir_order = spatial_order(nominal["directions_base"], far_pool, max_far)
    angular_recipes = {
        label: np.sort(np.concatenate((near_dir_order[:entry["selection"]["near_count"]],
                                        far_dir_order[:entry["selection"]["far_count"]])))
        for label, entry in manifest["files"].items()
    }
    runs = {}
    all_ids = np.arange(FULL_RAYS, dtype=np.int64)
    for name, width, pitch, roll in CASES:
        scan = intersect_scene(rays, mount, width=width, pitch_deg=pitch, roll_deg=roll)
        full_near = fps_ids(scan["points"], all_ids[scan["near"]], NEAR_OUT)
        full_far = fps_ids(scan["points"], all_ids[scan["far"]], FAR_OUT)
        trial = {}
        angular_trial = {}
        for name_id, ids in recipes.items():
            trial[name_id] = evaluate(scan, ids, full_near, full_far)
            angular_trial[name_id] = evaluate(scan, angular_recipes[name_id],
                                              full_near, full_far)
        runs[name] = {
            "geometry": {"width_m": width, "pitch_deg": pitch, "roll_deg": roll},
            "full": evaluate(scan, all_ids, full_near, full_far),
            "sparse": trial,
            "uniform_direction_baseline": angular_trial,
        }

    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "confined_locomotion_2.sparse_rays.analytic_compare/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "design_manifest_sha256": sha256(manifest_path),
        "assumptions": [
            "Independent analytic horizontal floor and optional vertical corridor walls",
            "Rigid stand-pose BASE with optional pitch/roll, fixed source rays; +pitch means nose-down",
            "Near/far ownership is fixed from catalog T1 nominal-plane B8 mask for every scene",
            "Per-scene dynamic B8 endpoint classification is exposed as a separate diagnostic mask and is not consumed by default metrics",
            "CPU XYZ FPS reference; no claim of GPU bitwise sameness",
            "FPS metric at 3D 0.2m full-hit probes; no learning/terrain observability claim",
            "No CUDA Warp benchmark or runtime integration",
            "Angular comparison is a fixed unit-ray-direction k-centre design, NOT uniform azimuth/elevation stride",
        ],
        "cases": runs,
        "comparison_decision": "UNDETERMINED: inspect wall/pose loss and validate real GPU before selecting multiplier",
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    concise = {}
    for scene, value in runs.items():
        concise[scene] = {
            k: {"near": v["near_hit_coverage"].get("within_0p20_fraction"),
                "near_points": v["selected_near"], "far_points": v["selected_far"]}
            for k, v in value["sparse"].items()
        }
    print(json.dumps({"output": str(output), "nominal_coverage": concise},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
