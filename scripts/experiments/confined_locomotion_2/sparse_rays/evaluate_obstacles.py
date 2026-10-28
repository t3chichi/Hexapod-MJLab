#!/usr/bin/env python3
"""Static REAL-channel obstacle observability comparison for confined_locomotion_2.

Independent analytic scenes: flat ground + finite axis-aligned boxes. Each box
is a real geometric obstacle (raised step, thin post, wall or overhead beam).
All 86,400 physical Airy rays are intersected with exactly the same surfaces
as the frozen sparse source-ID subsets, then the catalog's fixed T1 near/far
ownership and deterministic near256/far128 XYZ FPS are applied.

Obstacle visibility is diagnosed at THREE stages: raycast, B8 assignment, and
FPS. Equal object-labeled surface probes from full physical-ray hits provide
an independent comparison. These are NOT moving-ground-grid metrics and not
a network or sim2real success claim.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from select_channels import (
    FULL_RAYS, MAX_RANGE_M, NEAR_OUT, FAR_OUT, NEAR_MAX_X_M,
    airy_directions, nominal_mount, sha256,
)
from evaluate_moving_footprint import cone_axis_body, read_frozen_design, resolve_recipe_names
from compare_sampling import fps_ids
from hexapod_mjlab.tasks.confined_locomotion_2.components.lidar.channels import build_channel_catalog


# World-aligned opaque solid AABBs. Full-mesh Warp validation is independent.
# Each object has its own label and is separately scored; all poses are rigid
# BASE pitch/roll with robot optical site attached (not deforming the world).
SCENES = (
    {"name": "step_10cm", "pitch": 0, "roll": 0, "boxes": (
        ("step", (1.3, -0.9, 0.0), (2.8, 0.9, 0.10)),)},
    {"name": "step_25cm", "pitch": 0, "roll": 0, "boxes": (
        ("step", (1.3, -0.9, 0.0), (2.8, 0.9, 0.25)),)},
    {"name": "step_10cm_pitch_down_5", "pitch": 5, "roll": 0, "boxes": (
        ("step", (1.3, -0.9, 0.0), (2.8, 0.9, 0.10)),)},
    {"name": "step_25cm_pitch_up_5", "pitch": -5, "roll": 0, "boxes": (
        ("step", (1.3, -0.9, 0.0), (2.8, 0.9, 0.25)),)},
    {"name": "slim_curb_10cm", "pitch": 0, "roll": 0, "boxes": (
        ("slim_curb", (2.3, -0.60, 0.0), (2.38, 0.60, 0.10)),)},
    {"name": "post_8cm_centre", "pitch": 0, "roll": 0, "boxes": (
        ("post", (1.76, -0.04, 0.0), (1.84, 0.04, 0.60)),)},
    {"name": "post_8cm_offset", "pitch": 0, "roll": 0, "boxes": (
        ("post", (2.46, 0.40, 0.0), (2.54, 0.48, 0.60)),)},
    {"name": "post_15cm_far", "pitch": 0, "roll": 0, "boxes": (
        ("post", (3.05, -0.35, 0.0), (3.20, -0.20, 0.70)),)},
    {"name": "post_3cm_centre", "pitch": 0, "roll": 0, "boxes": (
        ("thin_post", (2.40, 0.0, 0.0), (2.43, 0.03, 0.45)),)},
    {"name": "post_3cm_offset", "pitch": 0, "roll": 0, "boxes": (
        ("thin_post", (2.40, 0.40, 0.0), (2.43, 0.43, 0.45)),)},
    {"name": "post_5cm_far", "pitch": 0, "roll": 0, "boxes": (
        ("thin_post", (3.20, 0.27, 0.0), (3.25, 0.32, 0.45)),)},
    {"name": "front_wall_60cm", "pitch": 0, "roll": 0, "boxes": (
        ("front_wall", (2.20, -0.9, 0), (2.24, 0.9, 0.60)),)},
    {"name": "corridor_1p6", "pitch": 0, "roll": 0, "boxes": (
        ("left_wall", (-1.5, 0.80, 0), (5.0, 0.83, 2.0)),
        ("right_wall", (-1.5, -0.83, 0), (5.0, -0.80, 2.0)))},
    {"name": "corridor_1p6_roll_left_5", "pitch": 0, "roll": 5, "boxes": (
        ("left_wall", (-1.5, 0.80, 0), (5.0, 0.83, 2.0)),
        ("right_wall", (-1.5, -0.83, 0), (5.0, -0.80, 2.0)))},
    {"name": "corridor_2p0_pitch_down_5", "pitch": 5, "roll": 0, "boxes": (
        ("left_wall", (-1.5, 1.0, 0), (5.0, 1.03, 2.0)),
        ("right_wall", (-1.5, -1.03, 0), (5.0, -1.0, 2.0)))},
    {"name": "overhead_beam", "pitch": 0, "roll": 0, "boxes": (
        ("beam", (1.70, -0.7, 0.65), (1.95, 0.7, 0.90)),)},
    {"name": "slim_curb_and_post", "pitch": 0, "roll": 0, "boxes": (
        ("slim_curb", (2.30, -0.65, 0), (2.38, 0.65, 0.10)),
        ("post", (1.76, 0.33, 0), (1.84, 0.41, 0.60)))},
)


def ray_aabb_entry(origin: np.ndarray, dirs: np.ndarray,
                   lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """First forward ray/AABB intersection; slab geometry incl. axial parallels.

    Returns inf if no entry within max range; origin-inside rays use exit t.
    Floating point comparisons are strict on range and surface boundaries.
    """
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    dirs = np.asarray(dirs, dtype=np.float64)
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    if np.any(hi <= lo):
        raise ValueError("AABB bounds require positive thickness in every dimension")
    parallel = np.abs(dirs) < 1e-12
    outside = parallel & ((origin < lo) | (origin > hi))[None, :]
    # Avoid warnings/NaNs from division by zero: a parallel axis applies
    # (-inf, +inf) interval when the origin is within the slab.
    inv = np.zeros_like(dirs)
    np.divide(1.0, dirs, out=inv, where=~parallel)
    t0 = (lo - origin) * inv
    t1 = (hi - origin) * inv
    enter = np.where(parallel, -np.inf, np.minimum(t0, t1))
    leave = np.where(parallel, np.inf, np.maximum(t0, t1))
    near = np.max(enter, axis=1)
    far = np.min(leave, axis=1)
    t = np.where(near > 1e-8, near, far)
    valid = (~outside.any(axis=1)) & (far >= near) & (far > 1e-8) & (t > 1e-8) & (t < MAX_RANGE_M)
    return np.where(valid, t, np.inf)


def scan_obstacles(directions_sensor: np.ndarray, mount: dict, scenario: dict) -> dict:
    """Independent real-ray/full-coordinate first-hit geometry for static boxes."""
    pitch = scenario["pitch"]
    roll = scenario["roll"]
    r_delta = Rotation.from_euler("xyz", [roll, pitch, 0], degrees=True).as_matrix()
    r_wb = mount["base_rot_world"] @ r_delta
    direction_body = (mount["mount_rot_base"] @ directions_sensor.astype(np.float64).T).T
    direction_world = (r_wb @ direction_body.T).T
    origin_body = mount["mount_pos_base"]
    root_world = mount["base_pos_world"] + np.array([0.0, 0.0, 0.0])
    origin_world = root_world + r_wb @ origin_body
    dz = direction_world[:, 2]
    floor_t = np.full(len(dz), np.inf)
    np.divide(-origin_world[2], dz, out=floor_t, where=np.abs(dz) > 1e-12)
    floor_t[(floor_t <= 1e-8) | (floor_t >= MAX_RANGE_M)] = np.inf

    distance = floor_t.copy()
    hit_label = np.where(np.isfinite(floor_t), 0, -1).astype(np.int16)
    for index, (name, lo, hi) in enumerate(scenario["boxes"], start=1):
        t = ray_aabb_entry(origin_world, direction_world, np.array(lo), np.array(hi))
        nearer = t < distance
        distance[nearer] = t[nearer]
        hit_label[nearer] = index

    dist = np.where(np.isfinite(distance), distance, MAX_RANGE_M)
    points_base = origin_body[None, :] + dist[:, None] * direction_body
    cone = (direction_body @ cone_axis_body(mount)) >= np.sin(np.deg2rad(28))
    dynamic_near = cone & (points_base[:, 0] <= NEAR_MAX_X_M)
    near = build_channel_catalog().near_mask
    return {
        "points": points_base, "distances_m": dist, "hit_label": hit_label,
        "near": near.copy(), "far": ~near, "dynamic_near": dynamic_near,
        "sensor_pos_world": origin_world, "directions_world": direction_world,
    }


def surface_probes(points: np.ndarray, voxel_m: float = 0.10) -> np.ndarray:
    """One full-ray obstacle return per hit voxel, not angular-weighted."""
    if not len(points):
        return np.empty((0, 3), dtype=np.float64)
    key = np.floor(points / voxel_m).astype(np.int64)
    _, where = np.unique(key, axis=0, return_index=True)
    return points[np.sort(where)]


def surface_support(reference: np.ndarray, points: np.ndarray) -> dict:
    """Same-object only; never let another object or the floor mask misses."""
    p = surface_probes(reference)
    if not len(p):
        return {"status": "not_visible_in_full_raw", "probes": 0}
    if not len(points):
        return {
            "status": "missed", "probes": len(p),
            "fraction_within_0p10m": 0.0, "fraction_within_0p20m": 0.0,
            "p95_nn_m": None,
        }
    d = cKDTree(points).query(p, k=1, workers=1)[0]
    return {
        "status": "observed", "probes": int(len(p)),
        "fraction_within_0p10m": float(np.mean(d <= 0.10)),
        "fraction_within_0p20m": float(np.mean(d <= 0.20)),
        "p95_nn_m": float(np.percentile(d, 95)),
        "max_nn_m": float(np.max(d)),
    }


def evaluate_scenario(s: dict, source_ids: np.ndarray, *, full_reference: dict) -> dict:
    ids = np.sort(np.asarray(source_ids, dtype=np.int64))
    points = s["points"]
    mask_near = s["near"][ids]
    candidate_near = ids[mask_near]
    candidate_far = ids[~mask_near]
    select_near = fps_ids(points, candidate_near, NEAR_OUT)
    select_far = fps_ids(points, candidate_far, FAR_OUT)
    selected = np.concatenate((select_near, select_far))
    near_labels = s["hit_label"][select_near]
    far_labels = s["hit_label"][select_far]
    out = {
        "source_rays": int(len(ids)),
        "fixed_near_candidate_rays": int(len(candidate_near)),
        "fixed_far_candidate_rays": int(len(candidate_far)),
        "selected_near": int(len(select_near)),
        "selected_far": int(len(select_far)),
        "objects": {},
    }
    for i, obj in enumerate(full_reference["objects"], start=1):
        ref_points = full_reference["points"][full_reference["hit_label"] == i]
        raw_obj = ids[s["hit_label"][ids] == i]
        near_raw = candidate_near[s["hit_label"][candidate_near] == i]
        far_raw = candidate_far[s["hit_label"][candidate_far] == i]
        near_fps = select_near[near_labels == i]
        far_fps = select_far[far_labels == i]
        fps_obj = selected[s["hit_label"][selected] == i]
        out["objects"][obj["label"]] = {
            "full_raw_visible_rays": obj["full_raw_visible_rays"],
            "full_raw_near": obj["full_raw_near"],
            "full_raw_far": obj["full_raw_far"],
            "raw_visible": len(raw_obj),
            "near_raw_visible": len(near_raw),
            "far_raw_visible": len(far_raw),
            "near_fps_visible": len(near_fps),
            "far_fps_visible": len(far_fps),
            "combined_fps_visible": len(fps_obj),
            "raw_surface_coverage": surface_support(ref_points, points[raw_obj]),
            "fps_surface_coverage": surface_support(ref_points, points[fps_obj]),
            "bounds": obj["bounds"],
        }
    return out


def make_full_reference(s: dict, scenario: dict) -> dict:
    ids = np.arange(len(s["hit_label"]), dtype=np.int64)
    near_ids = ids[s["near"]]
    far_ids = ids[s["far"]]
    picked_near = fps_ids(s["points"], near_ids, NEAR_OUT)
    picked_far = fps_ids(s["points"], far_ids, FAR_OUT)
    labels = s["hit_label"]
    objects = []
    for i, (name, lo, hi) in enumerate(scenario["boxes"], start=1):
        objects.append({
            "label": name, "bounds": {"min": lo, "max": hi},
            "full_raw_visible_rays": int(np.sum(labels == i)),
            "full_raw_near": int(np.sum(labels[near_ids] == i)),
            "full_raw_far": int(np.sum(labels[far_ids] == i)),
            "full_fps_near": int(np.sum(labels[picked_near] == i)),
            "full_fps_far": int(np.sum(labels[picked_far] == i)),
        })
    return {"objects": objects, "hit_label": labels, "points": s["points"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--design", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True, help="new, absent evidence directory")
    ap.add_argument("--recipes", nargs="+", help="optional manifest recipe subset; default: all")
    args = ap.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"evidence directory exists: {output}")
    rays = airy_directions()
    mount = nominal_mount()
    frozen, choices = read_frozen_design(args.design.expanduser().resolve(), rays)
    recipe_names = resolve_recipe_names(choices, args.recipes)
    choices = {name: choices[name] for name in recipe_names}
    all_ids = np.arange(FULL_RAYS, dtype=np.int64)
    cases = {}
    summary = {
        "observable_object_cases": 0,
        "per_recipe": {key: {"raw_missed": [], "fps_missed": []} for key in recipe_names},
    }
    # Retain historical summary keys when their exact recipes are selected.
    for key, raw_alias, fps_alias in (
        ("near_x4_far_x4", "raw_missed_4x", "fps_missed_4x"),
        ("near_x5_far_x4", None, "fps_missed_5x"),
    ):
        if key in choices:
            if raw_alias:
                summary[raw_alias] = []
            summary[fps_alias] = []
    for case in SCENES:
        scan = scan_obstacles(rays, mount, case)
        full = make_full_reference(scan, case)
        full_eval = evaluate_scenario(scan, all_ids, full_reference=full)
        sparse = {
            key: evaluate_scenario(scan, ids, full_reference=full)
            for key, ids in choices.items()
        }
        for obj in full["objects"]:
            if not obj["full_raw_visible_rays"]:
                continue
            summary["observable_object_cases"] += 1
            name = obj["label"]
            label = case["name"] + "/" + name
            for key in recipe_names:
                result = sparse[key]["objects"][name]
                if not result["raw_visible"]:
                    summary["per_recipe"][key]["raw_missed"].append(label)
                if not result["combined_fps_visible"]:
                    summary["per_recipe"][key]["fps_missed"].append(label)
            if "near_x4_far_x4" in choices:
                k4 = sparse["near_x4_far_x4"]["objects"][name]
                if not k4["raw_visible"]:
                    summary["raw_missed_4x"].append(label)
                if not k4["combined_fps_visible"]:
                    summary["fps_missed_4x"].append(label)
            if "near_x5_far_x4" in choices:
                k5 = sparse["near_x5_far_x4"]["objects"][name]
                if not k5["combined_fps_visible"]:
                    summary["fps_missed_5x"].append(label)
        cases[case["name"]] = {
            "pose": {"pitch_deg": case["pitch"], "roll_deg": case["roll"]},
            "scene_objects": full["objects"],
            "full": full_eval,
            "sparse": sparse,
        }
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "schema": "confined_locomotion_2.sparse_rays.obstacles/v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "design_manifest_sha256": sha256(args.design / "manifest.json"),
        "evaluator_sha256": sha256(Path(__file__)),
        "scenes": cases,
        "summary": summary,
        "scope": [
            "Only static analytic floor + axis-aligned solid boxes; no real Warp validation in this report",
            "All rays from real frozen Airy IDs; same geometry and same source labels for full and sparse",
            "No robot self-occlusion, rolling LiDAR timing, range noise, dropouts or physical motion",
            "Hit count and same-object 3D surface support from full raw, fixed-pool raw and FPS256/128 separately",
            "Large object/vertical wall may be outside the nominal moving-flat-ground target",
            "Full raw visibility is not equivalent to full *network-input* visibility",
            "No learned policy or real obstacle traversal claim; no multiplier chosen here",
        ],
    }
    (output / "report.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    brief = {
        name: {
            obj["label"]: {
                "full_raw": obj["full_raw_visible_rays"],
                "full_fps": obj["full_fps_near"] + obj["full_fps_far"],
                "per_recipe": {
                    key: {
                        "raw_visible": sparse[key]["objects"][obj["label"]]["raw_visible"],
                        "combined_fps_visible": sparse[key]["objects"][obj["label"]]["combined_fps_visible"],
                    } for key in recipe_names
                },
            } for obj in case["scene_objects"]
        }
        for name, case in cases.items()
        for sparse in [case["sparse"]]
    }
    for scene in brief.values():
        for obj in scene.values():
            for key, raw_alias, fps_alias in (
                ("near_x4_far_x4", "4x_raw", "4x_fps"),
                ("near_x5_far_x4", "5x_raw", "5x_fps"),
            ):
                if key in obj["per_recipe"]:
                    obj[raw_alias] = obj["per_recipe"][key]["raw_visible"]
                    obj[fps_alias] = obj["per_recipe"][key]["combined_fps_visible"]
    print(json.dumps({"output": str(output), "summary": summary, "objects": brief},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
