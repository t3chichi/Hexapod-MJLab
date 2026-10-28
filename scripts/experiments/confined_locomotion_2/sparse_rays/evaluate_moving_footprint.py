#!/usr/bin/env python3
"""Evaluate static physical-ray choices against a *pose-following* ground footprint.

The reference is an independent, deterministic planar XY grid in the NOMINAL
robot BASE frame, filtered by the *continuous* nominal angular cone. Each grid
point defines a virtual probe DIRECTION from the actual optical mount.
These diagnostic directions are NEVER passed to the sensor; they only define
an independent evaluation field of view.

At every test pose, rotate the probe directions and mount with BASE and
intersect the real horizontal world floor. The fixed angular field's ground
footprint shifts naturally under pitch/roll. Do not hold
the world-ground XY probes static and do not define probes using selected IDs.

This is a flat-plane geometry test, not obstacle observability, learning or
real-world scan-skew validation. The frozen physical source IDs and existing
256/128 XYZ FPS are NOT modified.
"""
from __future__ import annotations

import argparse
import hashlib
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
    FULL_RAYS, MAX_RANGE_M, NEAR_OUT, NEAR_MAX_X_M,
    THETA_DEG, TILT_DEG, MJCF, airy_directions, nominal_mount, sha256,
)
from compare_sampling import intersect_scene, fps_ids
from hexapod_mjlab.tasks.confined_locomotion_2.components.lidar.channels import build_channel_catalog, load_sparse_selection


# Uniform *grid* in nominal BASE horizontal world-ground coordinates.
# Unlike a full B8 footprint bound, this explicitly evaluated region is finite.
# All grid cells have equal nominal area. This grid is NOT the S1 ground head.
GRID = {"x_start_m": 0.75, "x_stop_m": 3.45, "x_count": 28,
        "y_start_m": -1.9, "y_stop_m": 1.9, "y_count": 39}
CASES = (
    ("flat", 0.0, 0.0, 0.0),
    ("pitch_down_2", 2.0, 0.0, 0.0),
    ("pitch_down_5", 5.0, 0.0, 0.0),
    ("pitch_down_10", 10.0, 0.0, 0.0),
    ("pitch_up_5", -5.0, 0.0, 0.0),
    ("roll_left_5", 0.0, 5.0, 0.0),
    ("roll_right_5", 0.0, -5.0, 0.0),
    ("body_up_8cm", 0.0, 0.0, 0.08),
    ("body_down_8cm", 0.0, 0.0, -0.08),
)


def cone_axis_body(mount: dict) -> np.ndarray:
    dome = mount["mount_rot_base"] @ np.array([0.0, 0.0, 1.0])
    delta = np.deg2rad(TILT_DEG)
    axis = np.array([
        np.cos(delta) * dome[0] + np.sin(delta) * dome[2],
        dome[1],
        -np.sin(delta) * dome[0] + np.cos(delta) * dome[2],
    ])
    return axis / np.linalg.norm(axis)


def make_independent_template(mount: dict, grid: dict = GRID) -> dict:
    """Define a nominal *angular field* from external grid, not selected rays.

    Grid XY is nominal BASE XY; solve each target's BASE z from the exact
    world z=0 plane under the complete frozen T1 BASE rotation. Cone-filter
    the fixed nominal angular field; no endpoint-x rule is applied here.
    """
    xs = np.linspace(grid["x_start_m"], grid["x_stop_m"], grid["x_count"])
    ys = np.linspace(grid["y_start_m"], grid["y_stop_m"], grid["y_count"])
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    rot_wb = mount["base_rot_world"]
    x_base, y_base = xx.ravel(), yy.ravel()
    z_base = -(
        mount["base_pos_world"][2] + rot_wb[2, 0] * x_base
        + rot_wb[2, 1] * y_base
    ) / rot_wb[2, 2]
    # Solve the exact world-z=0 plane in the tilted frozen T1 BASE frame.
    ground_base = np.stack((x_base, y_base, z_base), axis=1)
    delta = ground_base - mount["mount_pos_base"]
    lengths = np.linalg.norm(delta, axis=1)
    rays_base = delta / lengths[:, None]
    nominal_cone = (rays_base @ cone_axis_body(mount)) >= np.sin(np.deg2rad(THETA_DEG))
    nominal_valid = (
        nominal_cone &
        (lengths > 0) & (lengths < MAX_RANGE_M)
    )
    if not np.any(nominal_valid):
        raise ValueError("independent grid contains no nominal cone ground targets")
    return {
        "nominal_points_base": ground_base[nominal_valid],
        "directions_base": rays_base[nominal_valid],
        "nominal_valid_count": int(np.count_nonzero(nominal_valid)),
        "grid_total_count": int(xx.size),
        "grid": dict(grid),
    }


def reproject_template(template: dict, mount: dict, *, pitch_deg: float,
                       roll_deg: float, height_delta_m: float = 0.0) -> dict:
    """Rotate *nominal target sight lines* rigidly; hit the unmoving z=0 floor.

    The target is NOT a rigidly rotated planar patch: that patch would float
    away from the physical ground. Instead the fixed angular aperture moves.
    """
    rotation = Rotation.from_euler("xyz", [roll_deg, pitch_deg, 0.0], degrees=True)
    r_wb = mount["base_rot_world"] @ rotation.as_matrix()
    unit_body = template["directions_base"]
    unit_world = (r_wb @ unit_body.T).T
    root_world = mount["base_pos_world"] + np.array([0.0, 0.0, height_delta_m])
    origin_world = root_world + r_wb @ mount["mount_pos_base"]
    dz = unit_world[:, 2]
    length = np.full(len(dz), np.inf, dtype=np.float64)
    np.divide(-origin_world[2], dz, out=length, where=np.abs(dz) > 1e-12)
    floor_hit = np.isfinite(length) & (length > 0) & (length < MAX_RANGE_M)
    # Keep invalid probes' coordinates finite; NEVER count them as success.
    safe_length = np.where(floor_hit, length, MAX_RANGE_M)
    endpoint_base = mount["mount_pos_base"] + safe_length[:, None] * unit_body
    # Keep the fixed angular field: do not reapply an actual endpoint BASE-x
    # threshold after the robot pose changes.
    near = floor_hit
    return {
        "points_base": endpoint_base,
        "floor_hit": floor_hit,
        "near": near,
        "valid_target_points_base": endpoint_base[near],
        "nominal_probe_total": len(unit_body),
        "floor_hit_count": int(floor_hit.sum()),
        "near_probe_count": int(near.sum()),
        "near_fraction_of_nominal": float(near.mean()),
        "nominal_region_world": {
            "pose_pitch_deg": pitch_deg, "pose_roll_deg": roll_deg,
            "root_height_delta_m": height_delta_m,
        },
    }


def distance_metrics(reference: np.ndarray, chosen: np.ndarray) -> dict:
    """Direct metric on fixed-count equal-area nominal cells; NO voxel reweight."""
    count = len(reference)
    if count == 0:
        return {"status": "no_valid_theoretical_floor_probes", "probes": 0}
    if len(chosen) == 0:
        return {"status": "no_selected_real_hits", "probes": count}
    nearest = cKDTree(chosen).query(reference, k=1, workers=1)[0]
    return {
        "status": "ok", "probes": count,
        "mean_nn_m": float(np.mean(nearest)),
        "p95_nn_m": float(np.percentile(nearest, 95)),
        "max_nn_m": float(np.max(nearest)),
        "within_0p20_fraction": float(np.mean(nearest <= 0.20)),
        "within_0p25_fraction": float(np.mean(nearest <= 0.25)),
    }


def evaluate_selection(scan: dict, source_ids: np.ndarray,
                       reference: np.ndarray) -> dict:
    """Separate candidate coverage, consumed near-pool and near FPS."""
    ids = np.asarray(source_ids, dtype=np.int64)
    in_near = ids[scan["near"][ids]]
    real_near = in_near[scan["hit_type"][in_near] == 1]
    real_any = ids[scan["hit_type"][ids] == 1]
    picked = fps_ids(scan["points"], in_near, NEAR_OUT)
    real_picked = picked[scan["hit_type"][picked] == 1]
    return {
        "fixed_near_candidates": int(len(in_near)),
        "fixed_near_real_floor_hits": int(len(real_near)),
        "fps_fixed_near_count": int(len(picked)),
        "fps_fixed_near_real_floor_hits": int(len(real_picked)),
        "all_raw_floor_candidate_coverage": distance_metrics(
            reference, scan["points"][real_any]),
        "consumed_fixed_near_candidate_coverage": distance_metrics(
            reference, scan["points"][real_near]),
        "consumed_fixed_near_fps256_coverage": distance_metrics(
            reference, scan["points"][real_picked]),
    }


def read_frozen_design(design: Path, rays: np.ndarray) -> tuple[dict, dict]:
    mf = design / "manifest.json"
    manifest = json.loads(mf.read_text(encoding="utf-8"))
    catalog = build_channel_catalog()
    if manifest.get("schema") != catalog.identity["schema"]:
        raise ValueError("incorrect fixed-channel design schema")
    if manifest.get("catalog_identity") != catalog.identity:
        raise ValueError("fixed-channel catalog identity differs from current baseline")
    choices = {}
    for name, entry in manifest["files"].items():
        source = design / entry["file"]
        if sha256(source) != entry["npz_sha256"]:
            raise ValueError(f"frozen recipe mismatch: {name}")
        ray_ids, _ = load_sparse_selection(source, catalog)
        if not np.array_equal(rays[ray_ids], catalog.directions_sensor[ray_ids]):
            raise ValueError(f"invalid original source IDs: {name}")
        choices[name] = ray_ids
    return manifest, choices


def resolve_recipe_names(recipes: dict, requested=None) -> list[str]:
    """Return every manifest recipe by default, or validate an explicit subset."""
    names = list(recipes) if requested is None else list(requested)
    if not names:
        raise ValueError("at least one recipe is required")
    missing = [name for name in names if name not in recipes]
    if missing:
        raise ValueError(f"unknown frozen recipe(s): {', '.join(missing)}")
    if len(set(names)) != len(names):
        raise ValueError("recipe names must be unique")
    return names


def recipe_p95_summary(sparse: dict) -> dict:
    """Compact per-recipe moving-footprint P95 summary, without assumed keys."""
    return {
        name: data["consumed_fixed_near_fps256_coverage"].get("p95_nn_m")
        for name, data in sparse.items()
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new, nonexisting directory")
    parser.add_argument("--recipes", nargs="+", help="optional manifest recipe subset; default: all")
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"evidence path already exists: {output}")
    design = args.design.expanduser().resolve()
    mount = nominal_mount()
    rays = airy_directions()
    manifest, choices = read_frozen_design(design, rays)
    recipe_names = resolve_recipe_names(choices, args.recipes)
    choices = {name: choices[name] for name in recipe_names}
    template = make_independent_template(mount)
    all_ids = np.arange(FULL_RAYS, dtype=np.int64)
    cases = {}
    for name, pitch, roll, height in CASES:
        targets = reproject_template(
            template, mount, pitch_deg=pitch, roll_deg=roll,
            height_delta_m=height
        )
        scan = intersect_scene(
            rays, mount, width=None, pitch_deg=pitch, roll_deg=roll,
            height_delta_m=height,
        )
        near_target = targets["valid_target_points_base"]
        cases[name] = {
            "pose": {
                "pitch_deg": pitch, "roll_deg": roll, "height_delta_m": height,
            },
            "nominal_probes": targets["nominal_probe_total"],
            "floor_hit_probes": targets["floor_hit_count"],
            "fixed_angular_field_floor_probes": targets["near_probe_count"],
            "fixed_angular_field_valid_fraction": targets["near_fraction_of_nominal"],
            "moving_footprint_bounds_base": (
                {"x_min": float(near_target[:, 0].min()),
                 "x_max": float(near_target[:, 0].max()),
                 "y_min": float(near_target[:, 1].min()),
                 "y_max": float(near_target[:, 1].max())}
                if len(near_target) else None
            ),
            "full_reference": evaluate_selection(scan, all_ids, near_target),
            "sparse": {
                label: evaluate_selection(scan, ids, near_target)
                for label, ids in choices.items()
            },
        }
    result = {
        "schema": "confined_locomotion_2.sparse_rays.moving_footprint/v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "design_manifest_sha256": sha256(design / "manifest.json"),
        "catalog_identity": manifest["catalog_identity"],
        "evaluator_sha256": sha256(Path(__file__)),
        "target_definition": {
            "independent_of_selected_source_ids": True,
            "nominal_equal_area_xy_grid": GRID,
            "nominal_cone_filtered": True,
            "nominal_valid_count": template["nominal_valid_count"],
            "total_grid_cells": template["grid_total_count"],
            "nominal_direction_from_real_sensor_mount": True,
            "pose_rule": "rotate fixed theoretical probe directions with BASE, then intersect fixed world z=0 ground",
            "active_rule": "fixed nominal angular probe directions re-intersected with world z=0; retain true floor hits within 15m without reapplying endpoint BASE-x cutoff",
            "invalid_out_of_range_excluded_but_counts_explicitly_reported": True,
        },
        "scope": [
            "Independent analytic, flat plane only; no real obstacle/occlusion validation",
            "Static, unchanged physical Airy source IDs; probes are virtual ONLY for evaluation",
            "All results measured in scan-time BASE XYZ metres",
            "Catalog fixed near/far ownership and XYZ FPS applied to candidate rays for consumed-network input",
            "Moving footprint quality is PRIMARY; full raw hit-cloud coverage from compare_sampling is AUXILIARY",
            "No inference about Warp total train throughput, latency or sim2real",
        ],
        "cases": cases,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    brief = {}
    for scene, values in cases.items():
        row = {
            "fixed_field_active_fraction": values["fixed_angular_field_valid_fraction"],
            "full_P95": values["full_reference"]["consumed_fixed_near_fps256_coverage"].get("p95_nn_m"),
            "per_recipe_P95": recipe_p95_summary(values["sparse"]),
        }
        # Preserve familiar aliases when their exact recipes are present.
        for key, alias in (("near_x4_far_x4", "space_x4_P95"),
                           ("near_x5_far_x4", "space_x5_P95")):
            if key in row["per_recipe_P95"]:
                row[alias] = row["per_recipe_P95"][key]
        brief[scene] = row
    print(json.dumps({"output": str(output), "nominal_valid_probes": template["nominal_valid_count"],
                      "summary": brief}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
