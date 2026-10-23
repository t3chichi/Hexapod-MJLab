#!/usr/bin/env python3
"""A3 independent moving-pose far FPS evaluation on stairs and low-clearance spaces.

A synthetic kinematic pose SEQUENCE, not an MJLab locomotion simulation.
Real Airy full rays / fixed T1 classification / frozen sparse source IDs.
Physical raycast length stays 15m. Only far FPS filtering changes:
  A0: all fixed-far endpoints, including no-hit 15m points;
  A1: actual raw returns up to 15m;
  A2: actual raw returns up to 10m.
Near FPS256 remains untouched. Identical pose/first-hit geometry for all modes.

Physical world geometry is opaque AABB solids and z=0 ground, with proper
nearest intersection and torso-mounted rays rotating/translating with BASE.
Reports early observation of stair/roof/wall and grounded under-ceiling
clearance, with same-object raw-reference surface coverage and frame continuity.
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
from compare_sampling import fps_ids
from evaluate_obstacles import ray_aabb_entry, surface_support
from evaluate_moving_footprint import read_frozen_design
from select_channels import (
    FULL_RAYS, NEAR_OUT, FAR_OUT, MAX_RANGE_M, airy_directions,
    nominal_mount, sha256,
)

NEAR_FACTOR = 6
FAR_FACTORS = (8, 10)
DISTANCE_LIMIT_M = 10.0


def _solid(name, lo, hi, *, kind):
    return {"name": name, "lo": tuple(lo), "hi": tuple(hi), "kind": kind}


def _pose(x, pitch, height_delta=0., roll=0.):
    return {"x_m": float(x), "pitch_down_deg": float(pitch),
            "height_delta_m": float(height_delta), "roll_deg": float(roll)}


def make_scenarios():
    # 15cm risers, 35cm treads, 2m wide; connected boxes encode opaque
    # stair steps and a landing. Kinematic BASE poses approximate body climb,
    # not contact-consistent footfall trajectories.
    ascend = [
        _solid(f"up_step_{i + 1}", (1.15 + i * .35, -1., 0.),
               (1.50 + i * .35, 1., (i + 1) * .15), kind="stair")
        for i in range(3)
    ]
    ascend.append(_solid("up_landing", (2.20, -1., 0.), (6., 1., .45), kind="stair"))
    # Downstairs begin on an elevated landing then step down to floor.
    descend = [_solid("down_start", (-3., -1., 0.), (1.15, 1., .45), kind="stair")]
    descend.extend([
        _solid(f"down_step_{i + 1}", (1.15 + i * .35, -1., 0.),
               (1.50 + i * .35, 1., .30 - i * .15), kind="stair")
        for i in range(2)
    ])
    # Stair descent floor after x=1.85 remains nominal z=0.
    ceiling = [
        _solid("roof", (1.10, -.95, .73), (4.40, .95, .88), kind="roof")
    ]
    tunnel = [
        _solid("roof", (1.10, -.90, .73), (4.70, .90, .88), kind="roof"),
        _solid("left_wall", (1.10, .90, 0.), (4.70, .95, 1.7), kind="wall"),
        _solid("right_wall", (1.10, -.95, 0.), (4.70, -.90, 1.7), kind="wall"),
    ]
    tunnel_with_step = [
        *tunnel,
        _solid("inner_step", (2.05, -.85, 0.), (2.70, .85, .12), kind="stair")
    ]
    scenarios = [
        {"name": "flat_pitch_down", "solids": [], "poses": [
            _pose(x, pitch) for x, pitch in zip(
                (0., .2, .4, .6, .8, 1.0, 1.2),
                (0., 1., 2., 3., 5., 3., 2.)
            )]},
        {"name": "ascending_stairs", "solids": ascend, "poses": [
            _pose(x, pitch, h) for x, pitch, h in zip(
                (.0, .25, .5, .8, 1.1, 1.4, 1.7, 2.),
                (0., 1., 2., 4., 5., 4., 3., 2.),
                (0., 0., 0., .04, .10, .18, .28, .38),
            )]},
        {"name": "descending_stairs", "solids": descend, "poses": [
            _pose(x, pitch, h) for x, pitch, h in zip(
                (0., .25, .5, .8, 1.1, 1.4, 1.7, 2.),
                (0., 1., 2., 4., 5., 4., 3., 2.),
                (.45, .45, .45, .40, .30, .20, .10, 0.),
            )]},
        {"name": "low_ceiling", "solids": ceiling, "poses": [
            _pose(x, pitch) for x, pitch in zip(
                (0., .25, .5, .75, 1., 1.25, 1.5, 1.75),
                (0., 1., 2., 3., 5., 3., 2., 0.),
            )]},
        {"name": "low_tunnel", "solids": tunnel, "poses": [
            _pose(x, pitch) for x, pitch in zip(
                (0., .25, .5, .75, 1., 1.25, 1.5, 1.75),
                (0., 1., 2., 3., 5., 3., 2., 0.),
            )]},
        {"name": "tunnel_with_step", "solids": tunnel_with_step, "poses": [
            _pose(x, pitch, h) for x, pitch, h in zip(
                (0., .25, .5, .75, 1., 1.25, 1.5, 1.75),
                (0., 1., 2., 3., 5., 3., 2., 0.),
                (0., 0., 0., 0., .02, .04, .04, .06),
            )]},
        {"name": "pitch_up_boundary_2", "solids": [], "poses": [
            _pose(x, pitch) for x, pitch in zip(
                (0., .2, .4, .6), (0., -1., -2., -1.)
            )]},
    ]
    return scenarios


def interpolate_poses(poses, substeps: int):
    """Densify keyframe poses linearly, including each endpoint once.

    Substeps=5 maps the largest 0.30m keyframe move to <=0.06m per scan,
    representative of <=0.6m/s at 10Hz (not a claimed physical trajectory).
    """
    if not isinstance(substeps, int) or isinstance(substeps, bool) or not (1 <= substeps <= 20):
        raise ValueError("substeps must be in [1,20]")
    if len(poses) <= 1:
        return list(poses)
    out = []
    for a, b in zip(poses, poses[1:]):
        for j in range(substeps):
            alpha = j / substeps
            out.append({
                k: float(a[k] * (1.0 - alpha) + b[k] * alpha)
                for k in a
            })
    out.append(dict(poses[-1]))
    return out


def scan_pose(dirs_sensor, mount, pose, solids, *, fixed_near):
    """One first-return per real physical ray, in a stationary world scene."""
    r_delta = Rotation.from_euler(
        "xyz", [pose["roll_deg"], pose["pitch_down_deg"], 0], degrees=True
    ).as_matrix()
    body_rot_w = mount["base_rot_world"] @ r_delta
    base_pos_w = mount["base_pos_world"] + np.array(
        [pose["x_m"], 0., pose["height_delta_m"]], dtype=np.float64
    )
    sensor_o_w = base_pos_w + body_rot_w @ mount["mount_pos_base"]
    unit_b = (mount["mount_rot_base"] @ dirs_sensor.T).T
    unit_w = (body_rot_w @ unit_b.T).T

    dz = unit_w[:, 2]
    t_floor = np.full(len(dz), np.inf, dtype=np.float64)
    np.divide(-sensor_o_w[2], dz, out=t_floor, where=np.abs(dz) > 1e-12)
    t_floor[(t_floor <= 1e-8) | (t_floor >= MAX_RANGE_M)] = np.inf
    best = t_floor.copy()
    labels = np.where(np.isfinite(best), 0, -1).astype(np.int16)
    for index, solid in enumerate(solids, 1):
        t = ray_aabb_entry(sensor_o_w, unit_w, solid["lo"], solid["hi"])
        closer = t < best
        best[closer] = t[closer]
        labels[closer] = index
    measured = np.where(np.isfinite(best), best, MAX_RANGE_M)
    xyz_b = mount["mount_pos_base"][None, :] + measured[:, None] * unit_b
    xyz_w = sensor_o_w[None, :] + measured[:, None] * unit_w
    return {
        "points": xyz_b, "world": xyz_w, "distances_m": measured,
        "hit_label": labels, "near": fixed_near, "far": ~fixed_near,
        "sensor_origin_world": sensor_o_w,
    }


def choose(scan, ids, *, far_method):
    """Near selection stays standard; far changes only the candidate mask."""
    ids = np.asarray(ids, dtype=np.int64)
    near = ids[scan["near"][ids]]
    far = ids[scan["far"][ids]]
    near_selected = fps_ids(scan["points"], near, NEAR_OUT)
    far_eligible = far
    if far_method in ("hit15", "hit10"):
        far_eligible = far[scan["hit_label"][far] >= 0]
    if far_method == "hit10":
        far_eligible = far_eligible[
            scan["distances_m"][far_eligible] <= DISTANCE_LIMIT_M
        ]
    elif far_method not in ("original", "hit15"):
        raise ValueError(f"unrecognized method: {far_method}")
    far_selected = fps_ids(scan["points"], far_eligible, FAR_OUT)
    return {
        "near": near_selected, "far": far_selected,
        "far_pool": far, "far_eligible": far_eligible,
    }


def _support(scan, ref_ids, selected_ids):
    return surface_support(
        scan["world"][ref_ids], scan["world"][selected_ids]
    )


def score_frame(scan, ids, solids, picked):
    labels = scan["hit_label"]
    near, far = picked["near"], picked["far"]
    both = np.r_[near, far]
    full_far = np.flatnonzero(scan["far"])
    ground_far = full_far[labels[full_far] == 0]
    selected_far_ground = far[labels[far] == 0]
    classes = {}
    floor_under = []
    roof_under = []
    for i, solid in enumerate(solids, 1):
        ref = full_far[labels[full_far] == i]
        raw = picked["far_pool"][labels[picked["far_pool"]] == i]
        eligible = picked["far_eligible"][labels[picked["far_eligible"]] == i]
        selected_far = far[labels[far] == i]
        selected_both = both[labels[both] == i]
        classes[solid["name"]] = {
            "kind": solid["kind"],
            "full_far_raw_hits": int(len(ref)),
            "sparse_far_raw_hits": int(len(raw)),
            "far_eligible_hits": int(len(eligible)),
            "far_retained": int(len(selected_far)),
            "both_retained": int(len(selected_both)),
            "far_support": _support(scan, ref, selected_far),
        }
        if solid["kind"] == "roof":
            # Only roof UNDERSIDE, not arbitrary side/top surface, counts as
            # direct low-ceiling height evidence. Reference geometry uses
            # exact AABB bottom z.
            bottom_z = solid["lo"][2]
            sub = selected_both[
                np.abs(scan["world"][selected_both, 2] - bottom_z) <= .025
            ]
            roof_under.extend(sub.tolist())
    # A clearance estimate requires BOTH floor and roof underside evidence
    # in the same roof projected corridor, from near+far selected channels.
    roof_solids = [s for s in solids if s["kind"] == "roof"]
    if roof_solids:
        roof = roof_solids[0]
        floor_candidates = both[labels[both] == 0]
        ground_in_roof = floor_candidates[
            (scan["world"][floor_candidates, 0] >= roof["lo"][0]) &
            (scan["world"][floor_candidates, 0] <= roof["hi"][0]) &
            (np.abs(scan["world"][floor_candidates, 1]) <= roof["hi"][1])
        ]
        floor_under = ground_in_roof.tolist()
        roof_x_bins_0p5m = (len(np.unique(
            np.floor((scan["world"][roof_under, 0] - roof["lo"][0]) / .5).astype(np.int64)
        )) if roof_under else 0)
        roof_z_est = (float(np.median(scan["world"][roof_under, 2]))
                      if roof_under else None)
        floor_z_est = (float(np.median(scan["world"][floor_under, 2]))
                       if floor_under else None)
        clearance = (roof_z_est - floor_z_est
                     if roof_z_est is not None and floor_z_est is not None
                     else None)
        target_clearance = float(roof["lo"][2])  # world ground z=0
        clearance_abs_error = (abs(clearance - target_clearance)
                               if clearance is not None else None)
    else:
        roof_z_est = floor_z_est = clearance = clearance_abs_error = None
        roof_x_bins_0p5m = 0
        target_clearance = None

    hit_far = far[labels[far] >= 0]
    return {
        "near_fps_count": int(len(near)),
        "far_eligible_count": int(len(picked["far_eligible"])),
        "far_fps_count": int(len(far)),
        "far_padding_count": int(FAR_OUT - len(far)),
        "far_selected_miss_count": int(np.sum(labels[far] == -1)),
        "far_selected_ground_count": int(np.sum(labels[far] == 0)),
        "far_selected_obstacle_count": int(np.sum(labels[far] > 0)),
        "far_surface_ground_support": _support(scan, ground_far, selected_far_ground),
        "objects": classes,
        "roof_underside_retained_both": int(len(roof_under)),
        "roof_underside_longitudinal_bins_0p5m": int(roof_x_bins_0p5m),
        "floor_under_roof_retained_both": int(len(floor_under)),
        "roof_bottom_z_est": roof_z_est,
        "floor_z_est": floor_z_est,
        "clearance_m": clearance,
        "true_roof_floor_clearance_m": target_clearance,
        "clearance_abs_error_m": clearance_abs_error,
        "near_ids": near.tolist(),
        "far_ids": far.tolist(),
        # Transient, used ONLY to score adjacent scans; removed before writing.
        "_far_hit_world_points": scan["world"][hit_far].tolist(),
    }


def motion_summary(frames, method):
    effective = [int(f["scores"][method]["far_fps_count"]) for f in frames]
    pad = [int(f["scores"][method]["far_padding_count"]) for f in frames]
    temporal = []
    for a, b in zip(frames, frames[1:]):
        sa, sb = a["scores"][method], b["scores"][method]
        prev = np.asarray(sa["far_ids"], dtype=np.int64)
        curr = np.asarray(sb["far_ids"], dtype=np.int64)
        ja = len(np.intersect1d(prev, curr)) / max(1, len(np.union1d(prev, curr)))
        # World-frame symmetric nearest neighbour P95 on REAL hit-only
        # surfaces; excludes static 15m artificial endpoint jumps.
        point_a = np.asarray(sa["_far_hit_world_points"], dtype=np.float64).reshape(-1, 3)
        point_b = np.asarray(sb["_far_hit_world_points"], dtype=np.float64).reshape(-1, 3)
        if len(point_a) and len(point_b):
            n_a = cKDTree(point_a).query(point_b)[0]
            n_b = cKDTree(point_b).query(point_a)[0]
            p95 = float(np.percentile(np.r_[n_a, n_b], 95))
        else:
            p95 = None
        temporal.append({
            "frame_from": a["frame"], "frame_to": b["frame"],
            "valid_count_delta": int(sb["far_fps_count"] - sa["far_fps_count"]),
            "source_id_jaccard": float(ja),
            "real_world_symmetric_nn_p95_m": p95,
        })
    return {
        "frames": len(frames),
        "min_far_valid": int(min(effective)),
        "mean_far_valid": float(np.mean(effective)),
        "p10_far_valid": float(np.percentile(effective, 10)),
        "frames_with_padding": int(sum(x > 0 for x in pad)),
        "max_padding": int(max(pad)),
        "max_adjacent_valid_count_change": max(
            (abs(t["valid_count_delta"]) for t in temporal), default=0
        ),
        "max_real_world_symmetric_nn_p95_m": max(
            (t["real_world_symmetric_nn_p95_m"] for t in temporal
             if t["real_world_symmetric_nn_p95_m"] is not None),
            default=None,
        ),
        "consecutive": temporal,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--design", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--substeps", type=int, default=1,
                    help="linear intermediate scan poses between each adjacent keyframe")
    args = ap.parse_args(argv)
    if not 1 <= args.substeps <= 20:
        raise ValueError("--substeps must be in [1,20]")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"evidence directory already exists: {output}")
    dirs = airy_directions()
    mount = nominal_mount()
    from hexapod_mjlab.tasks.confined_locomotion_2.channels import build_channel_catalog
    catalog = build_channel_catalog()
    _, candidates = read_frozen_design(args.design.expanduser().resolve(), dirs)
    keys = ("near_x6_far_x8", "near_x6_far_x10")
    if any(x not in candidates for x in keys):
        raise ValueError("A3 requires frozen near6/far8 and near6/far10 candidates")
    modes = ("original", "hit15", "hit10")
    scenes = {}
    for scenario in make_scenarios():
        frames = []
        trajectory = interpolate_poses(scenario["poses"], args.substeps)
        for frame_idx, pose in enumerate(trajectory):
            scan = scan_pose(dirs, mount, pose, scenario["solids"],
                             fixed_near=catalog.near_mask)
            scores = {}
            for key in keys:
                ids = candidates[key]
                for mode in modes:
                    chosen = choose(scan, ids, far_method=mode)
                    score = score_frame(scan, ids, scenario["solids"], chosen)
                    if mode != "original" and score["far_selected_miss_count"]:
                        raise AssertionError("hit-only FPS retained no-hit point")
                    if mode == "hit10" and np.any(
                        scan["distances_m"][chosen["far"]] > DISTANCE_LIMIT_M
                    ):
                        raise AssertionError("10m FPS retained over-range hit")
                    scores[key + "/" + mode] = score
            frames.append({"frame": frame_idx, "pose": pose,
                           "scores": scores})
        summaries = {m: motion_summary(frames, m)
                     for m in frames[0]["scores"]}
        # Save bounded provenance and metrics without redundant XYZ arrays
        # duplicated across six samples/methods at each pose.
        for frame in frames:
            for score in frame["scores"].values():
                score.pop("_far_hit_world_points")
        scenes[scenario["name"]] = {
            "solids": scenario["solids"],
            "frames": frames, "summaries": summaries,
        }
        print("PASS", scenario["name"], flush=True)
    output.mkdir(parents=True, exist_ok=False)
    payload = {
        "schema": "confined_locomotion_2.far_motion_A3/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evaluator_sha256": sha256(Path(__file__)),
        "candidate_manifest_sha256": sha256(args.design / "manifest.json"),
        "raw_range_m": MAX_RANGE_M,
        "far_hit10_cutoff_m": DISTANCE_LIMIT_M,
        "near_factor": NEAR_FACTOR,
        "far_factors": FAR_FACTORS,
        "pose_interpolation_substeps": int(args.substeps),
        "nominal_scan_hz_for_motion_interpretation": 10,
        "methods": {
            "original": "15m raycast endpoints incl nohit, fixed far FPS128",
            "hit15": "real raw hits, max range 15m, fixed far FPS128",
            "hit10": "real raw hits <=10m, fixed far FPS128",
        },
        "scope": [
            "Analytic static opaque AABB + plane and APPROXIMATE kinematic robot poses, not MJLab physical gait",
            "Full catalog real Airy directions, frozen T1 fixed near/far grouping, sparse selected source IDs",
            "All modes use 15m physical first hit, BASE frame XYZ max-norm seed FPS (near256 unaffected)",
            "Far hit eligibility from original first returns, never from noisy/perturbed endpoint",
            "No self-occlusion, scan timing, collisions, leg/torso detailed mesh, P6 noise, dropout or recurrent ML model",
            "Clearance uses only observed SAME roof underside + floor points in its plan footprint",
            "Adjacent scans vary kinematic x, pitch, body height; no dynamics or feasibility guarantee",
            "No production scanner/checkpoint/network modification; diagnostic only",
        ],
        "scenes": scenes,
    }
    (output / "report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "scenes": len(scenes),
                      "frames_total": sum(len(s["frames"]) for s in scenes.values()),
                      "modes": len(modes), "budgets": len(keys)}))


if __name__ == "__main__":
    main()
