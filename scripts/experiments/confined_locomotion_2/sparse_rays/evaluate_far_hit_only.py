#!/usr/bin/env python3
"""A1-only independent comparison: fixed far-channel FPS vs hit-only far FPS.

Never changes physical raycast range (15m), fixed T1 channel assignment,
near FPS256, sampled XYZ metric, output budget far128 or production sources.
Uses full 86400-ray analytic observations and two frozen far budget subsets.
No noise, no gap, no range cutoff, no random seed, no full-vs-sparse pointwise
matching. The sole A1 change is omitting actual no-hit rays from FAR FPS pool.

Scores same-object full-RAW far-channel surface 0.10m voxel probes;
this distinguishes detection and geometry support, rather than hit counts
alone. This is static analytic evidence, not a real GPU/control test.
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
from select_channels import FULL_RAYS, FAR_OUT, airy_directions, nominal_mount, sha256
from evaluate_moving_footprint import read_frozen_design

SCENE_NAMES = (
    "flat_ground", "flat_ground_pitch_up_5",
    "step_10cm", "step_25cm", "step_10cm_pitch_down_5",
    "step_25cm_pitch_up_5", "slim_curb_10cm", "front_wall_60cm",
    "corridor_1p6", "corridor_1p6_roll_left_5",
    "corridor_2p0_pitch_down_5", "overhead_beam",
)


def evaluate_far(scan: dict, catalog_ids: np.ndarray, scene: dict, *,
                 remove_no_hit: bool) -> dict:
    source_ids = np.sort(np.asarray(catalog_ids, dtype=np.int64))
    fixed_far = source_ids[scan["far"][source_ids]]
    hit_label = scan["hit_label"]
    hit = hit_label[fixed_far] >= 0
    eligible = fixed_far[hit] if remove_no_hit else fixed_far
    picked = fps_ids(scan["points"], eligible, FAR_OUT)
    selected_labels = hit_label[picked]
    objects = {}
    all_full_far = np.flatnonzero(scan["far"])
    for obj_id, (label, _min, _max) in enumerate(scene["boxes"], 1):
        reference = scan["points"][all_full_far[hit_label[all_full_far] == obj_id]]
        retained = scan["points"][picked[selected_labels == obj_id]]
        objects[label] = {
            "full_raw_far_hit_count": int(len(reference)),
            "source_far_hit_count": int(np.sum(hit_label[fixed_far] == obj_id)),
            "selected_far_count": int(np.sum(selected_labels == obj_id)),
            "surface_support_0p20": surface_support(reference, retained),
        }
    return {
        "far_source_count": int(len(fixed_far)),
        "far_real_hit_candidate_count": int(np.count_nonzero(hit)),
        "far_no_hit_candidate_count": int(np.count_nonzero(~hit)),
        "far_fps_eligible_count": int(len(eligible)),
        "far_fps_selected_count": int(len(picked)),
        "far_fps_padding_count": int(max(0, FAR_OUT - len(picked))),
        "far_fps_real_hit_count": int(np.sum(selected_labels >= 0)),
        "far_fps_no_hit_count": int(np.sum(selected_labels == -1)),
        "far_fps_ground_count": int(np.sum(selected_labels == 0)),
        "far_fps_obstacle_count": int(np.sum(selected_labels > 0)),
        "objects": objects,
        # Preserve the source-ID sample sequence for reproducibility and
        # future official CPU FPS equivalence checks; no invented 15m fillers.
        "selected_original_source_ids": picked.tolist(),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--design", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    out = args.output.expanduser().resolve()
    if out.exists():
        raise FileExistsError(f"refusing existing evidence output: {out}")
    rays = airy_directions()
    mount = nominal_mount()
    _, candidates = read_frozen_design(args.design.expanduser().resolve(), rays)
    names = ("near_x6_far_x8", "near_x6_far_x10")
    if any(k not in candidates for k in names):
        raise ValueError("selected recipes are missing from frozen design")
    recipes = {"full_86400": np.arange(FULL_RAYS, dtype=np.int64)}
    recipes.update((name, candidates[name]) for name in names)
    cases = {}
    ground_controls = (
        {"name": "flat_ground", "pitch": 0, "roll": 0, "boxes": ()},
        {"name": "flat_ground_pitch_up_5", "pitch": -5, "roll": 0, "boxes": ()},
    )
    for scene in (*ground_controls, *SCENES):
        if scene["name"] not in SCENE_NAMES:
            continue
        scan = scan_obstacles(rays, mount, scene)
        by_budget = {}
        for key, ids in recipes.items():
            by_budget[key] = {
                "A0_original": evaluate_far(scan, ids, scene, remove_no_hit=False),
                "A1_hit_only": evaluate_far(scan, ids, scene, remove_no_hit=True),
            }
            base = by_budget[key]["A0_original"]
            alt = by_budget[key]["A1_hit_only"]
            if (base["far_source_count"] != alt["far_source_count"] or
                base["far_real_hit_candidate_count"] != alt["far_real_hit_candidate_count"] or
                base["far_fps_selected_count"] != FAR_OUT):
                raise AssertionError(f"unexpected static candidate or baseline count: {scene['name']} / {key}")
            if alt["far_fps_no_hit_count"] != 0:
                raise AssertionError("A1 selected at least one no-hit ray")
            if alt["far_fps_selected_count"] != min(
                FAR_OUT, base["far_real_hit_candidate_count"]
            ):
                raise AssertionError("A1 failed exact hit-mask candidate budget")
        cases[scene["name"]] = {
            "pose": {"pitch_deg": scene["pitch"], "roll_deg": scene["roll"]},
            "by_budget": by_budget,
        }
        print(f"PASS: {scene['name']}", flush=True)

    if len(cases) != len(SCENE_NAMES):
        raise AssertionError("some representative scenes not evaluated")
    out.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": "confined_locomotion_2.far_hit_only/v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_manifest_sha256": sha256(args.design / "manifest.json"),
        "evaluator_sha256": sha256(Path(__file__)),
        "far_output_count": FAR_OUT,
        "comparisons": ("full_86400", *names),
        "modes": {
            "A0_original": "fixed far IDs, all rays incl 15m no-hit, max-norm-seed XYZ FPS",
            "A1_hit_only": "same fixed far IDs, real returns only (hit_label >= 0), same XYZ FPS",
        },
        "scope": [
            "NO change to physical 15m range, 10Hz, near FPS, fixed T1 channel grouping, or ray directions",
            "Only analytic floor + axis-aligned boxes; no GPU or ML policy validation",
            "Without noise/gap (P6): actual hit is based on raw raycast, not a perturbed range",
            "Surface probe reference uses same-object FULL raw fixed-far returns",
            "Hit-only may output <128 selected points: actual input must retain invalid padded slots if ever deployed",
            "No check here for information loss from directions with no detected returns",
            "Original-source source IDs stored for FPS provenance, not for network input",
        ],
        "scenes": cases,
    }
    (out / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "scenes": len(cases), "budgets": list(recipes)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
