#!/usr/bin/env python3
"""A2: far-hit-only FPS distance ceilings (15, 12, 10, 8 m), read-only.

Fixed-T1 near/far membership and the original 15m REAL raycast are unchanged.
Raycast labels/first-hit geometry are computed ONCE per scene. The single A2
intervention is a ceiling on ORIGINAL sensor ray distance for the far hit-only
FPS candidate pool. The actual far FPS still uses BASE xyz and max-norm seed.

Includes far-away wall controls at 9/11/13m to reveal lost early observations.
No rollout, physical cutoff, noise, gap, temporal scan, or production changes.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from compare_sampling import fps_ids
from evaluate_obstacles import SCENES, scan_obstacles, surface_support
from evaluate_far_hit_only import evaluate_far, SCENE_NAMES
from evaluate_moving_footprint import read_frozen_design
from select_channels import FULL_RAYS, FAR_OUT, airy_directions, nominal_mount, sha256

DISTANCE_BINS_M = ((0., 8.), (8., 10.), (10., 12.), (12., 15.))
LIMITS_M = (15., 12., 10., 8.)
ADDED_SCENES = (
    {"name": "front_wall_9m", "pitch": 0, "roll": 0,
     "boxes": (("wall", (8.9, -2., 0.), (9., 2., 2.5)),)},
    {"name": "front_wall_11m", "pitch": 0, "roll": 0,
     "boxes": (("wall", (10.9, -2., 0.), (11., 2., 2.5)),)},
    {"name": "front_wall_13m", "pitch": 0, "roll": 0,
     "boxes": (("wall", (12.9, -2., 0.), (13., 2., 2.5)),)},
)
FLAT_SCENES = (
    {"name": "flat_ground", "pitch": 0, "roll": 0, "boxes": ()},
    {"name": "flat_ground_pitch_up_5", "pitch": -5, "roll": 0, "boxes": ()},
)


def get_scenes():
    cases = (*FLAT_SCENES, *(s for s in SCENES if s["name"] in SCENE_NAMES),
             *ADDED_SCENES)
    labels = [s["name"] for s in cases]
    if len(labels) != len(set(labels)):
        raise AssertionError("duplicate test scene")
    return cases


def evaluate_limit(scan: dict, catalog_ids: np.ndarray, scene: dict, limit_m: float) -> dict:
    """Measure both actual first-hit pool loss and FPS geometry at a ceiling."""
    if not np.isfinite(limit_m) or not (0 < limit_m <= 15.):
        raise ValueError("distance limit must be finite within (0,15]m")
    ids = np.sort(np.asarray(catalog_ids, dtype=np.int64))
    far_ids = ids[scan["far"][ids]]
    labels = scan["hit_label"]
    dists = scan["distances_m"]
    real_hit = labels[far_ids] >= 0
    in_limit = real_hit & (dists[far_ids] <= limit_m)
    eligible = far_ids[in_limit]
    picked = fps_ids(scan["points"], eligible, FAR_OUT)
    selected_labels = labels[picked]

    full_far = np.flatnonzero(scan["far"])
    real_ref = full_far[labels[full_far] >= 0]
    ground_ref = scan["points"][real_ref[labels[real_ref] == 0]]
    picked_ground = scan["points"][picked[selected_labels == 0]]

    all_real_ids = far_ids[real_hit]
    selected_d = dists[picked]

    objects = {}
    for obj_id, (name, _min, _max) in enumerate(scene["boxes"], start=1):
        # Surface reference is SAME object / full-raw fixed far, unchanged
        # when a candidate limit removes the object. This exposes recall loss.
        raw_obj = full_far[labels[full_far] == obj_id]
        ids_obj = far_ids[labels[far_ids] == obj_id]
        eligible_obj = eligible[labels[eligible] == obj_id]
        selected_obj = picked[selected_labels == obj_id]
        objects[name] = {
            "full_raw_fixed_far_hits": int(len(raw_obj)),
            "subset_raw_fixed_far_hits": int(len(ids_obj)),
            "subset_far_hits_within_limit": int(len(eligible_obj)),
            "subset_far_hits_discarded_by_limit": int(len(ids_obj) - len(eligible_obj)),
            "fps_kept": int(len(selected_obj)),
            "support_vs_full_raw_far": surface_support(
                scan["points"][raw_obj], scan["points"][selected_obj]
            ),
        }
    return {
        "distance_ceiling_m": float(limit_m),
        "raw_far_candidate_count": int(len(far_ids)),
        "raw_far_hit_count": int(real_hit.sum()),
        "raw_far_miss_count": int((~real_hit).sum()),
        "hit_within_distance_count": int(len(eligible)),
        "hit_removed_by_distance_count": int(real_hit.sum() - len(eligible)),
        "fps_valid_count": int(len(picked)),
        "fps_invalid_padding_count": int(max(0, FAR_OUT - len(picked))),
        "fps_ground_count": int(np.count_nonzero(selected_labels == 0)),
        "fps_obstacle_count": int(np.count_nonzero(selected_labels > 0)),
        "fps_no_hit_count": int(np.count_nonzero(selected_labels == -1)),
        "floor_surface_support_vs_full_raw_far": surface_support(
            ground_ref, picked_ground
        ),
        "raw_hit_distance_bins": {
            f"[{lo:g},{hi:g})": int(np.sum((dists[all_real_ids] >= lo) &
                                             (dists[all_real_ids] < hi)))
            for lo, hi in DISTANCE_BINS_M
        },
        "selected_hit_distance_bins": {
            f"[{lo:g},{hi:g})": int(np.sum((selected_d >= lo) & (selected_d < hi)))
            for lo, hi in DISTANCE_BINS_M
        },
        "selected_original_source_ids": picked.tolist(),
        "objects": objects,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--design", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"evidence output already exists: {output}")
    rays = airy_directions()
    mount = nominal_mount()
    _, candidates = read_frozen_design(args.design.expanduser().resolve(), rays)
    keys = ("near_x6_far_x8", "near_x6_far_x10")
    if any(k not in candidates for k in keys):
        raise ValueError("A2 requires frozen near6-far8 and near6-far10 selections")
    pools = {"full_86400": np.arange(FULL_RAYS, dtype=np.int64)}
    pools.update({name: candidates[name] for name in keys})
    records = {}
    for scene in get_scenes():
        scan = scan_obstacles(rays, mount, scene)
        budgets = {}
        for name, ids in pools.items():
            A0 = evaluate_far(scan, ids, scene, remove_no_hit=False)
            A1 = evaluate_far(scan, ids, scene, remove_no_hit=True)
            variants = {f"A2_{int(limit)}m": evaluate_limit(scan, ids, scene, limit)
                        for limit in LIMITS_M}
            # A2_15m is exactly A1 at unchanged range and must preserve the
            # precise source-ID sequence (candidate pool, FPS metric, seed).
            if variants["A2_15m"]["selected_original_source_ids"] != A1["selected_original_source_ids"]:
                raise AssertionError(f"A1/A2_15 ID mismatch: {scene['name']} {name}")
            if variants["A2_15m"]["fps_valid_count"] != A1["far_fps_selected_count"]:
                raise AssertionError("A1/A2 15m count mismatch")
            if any(variants[f"A2_{int(l)}m"]["fps_no_hit_count"] != 0
                   for l in LIMITS_M):
                raise AssertionError("A2 retained no-hit source")
            counts = [variants[f"A2_{int(l)}m"]["hit_within_distance_count"]
                      for l in LIMITS_M]
            if counts != sorted(counts, reverse=True):
                raise AssertionError("eligible hit pool must shrink monotonically")
            budgets[name] = {
                "A0_original": A0,
                "A1_15m_hit_only": A1,
                "A2_distance_ceiling": variants,
            }
        records[scene["name"]] = {
            "pitch_deg": scene["pitch"],
            "roll_deg": scene["roll"],
            "by_budget": budgets,
        }
        print("PASS", scene["name"], flush=True)

    output.mkdir(parents=True, exist_ok=False)
    payload = {
        "schema": "confined_locomotion_2.far_hit_only_cutoff_A2/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evaluator_sha256": sha256(Path(__file__)),
        "design_manifest_sha256": sha256(args.design / "manifest.json"),
        "raycast_max_range_m": 15.0,
        "fps_candidate_hit_distance_limits_m": list(LIMITS_M),
        "budgets": list(pools),
        "scope": [
            "Independent CPU analytic first-hit floor/box geometry, fixed T1 source groups, not GPU",
            "15m physical raycast and catalog are unchanged; only FPS candidate mask on real original hit distance changes",
            "BASE XYZ max-norm-seed FPS128, source ID stable, no random start, P6 noise or scan dropout",
            "Same-object and floor support references always full raw physical ray hits from the fixed far group",
            "A0 reference includes no-hit valid 15m endpoints; A1 and A2 omit them",
            "Distance ceilings can erase 8-15m real obstacles and cause invalid padding",
            "12/10/8m are illustrative, not final policy or automatically suitable for sim2real",
        ],
        "scenes": records,
    }
    (output / "report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "scenes": len(records),
                      "budgets": list(pools), "cutoffs_m": list(LIMITS_M)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
