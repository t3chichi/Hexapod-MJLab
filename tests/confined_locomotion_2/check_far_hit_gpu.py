#!/usr/bin/env python3
"""Finite real-CUDA acceptance for the production ``far_sampling=hit_only_10m`` path.

This runs the official sparse ConfinedScanPipeline over the A3 synthetic pose
sequences and the same fixed source IDs in both legacy and hit-only modes. The
sequence and AABB scenes are kinematic diagnostics, not a dynamically feasible
MJLab gait. A real Warp mesh raycast and the production scanner/sampler are
used for every reported A3 scan; an explicitly labelled mask fixture separately
checks zero/short far-slot padding through the pipeline's native sampler.
No optimizer or training run is performed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import warp as wp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _json_value(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _pose_robot(robot, sim, original_pos, original_quat_wxyz, pose) -> None:
    """Set BASE pose through generalized coordinates and refresh kinematics."""
    from scipy.spatial.transform import Rotation

    desired_pos = original_pos.clone()
    desired_pos[0] += float(pose["x_m"])
    desired_pos[2] += float(pose["height_delta_m"])
    q = original_quat_wxyz.detach().cpu().numpy()
    base = Rotation.from_quat([q[1], q[2], q[3], q[0]])
    delta = Rotation.from_euler(
        "xyz", [pose["roll_deg"], pose["pitch_down_deg"], 0.0], degrees=True
    )
    xyzw = (base * delta).as_quat()
    desired_quat = torch.tensor(
        [xyzw[3], xyzw[0], xyzw[1], xyzw[2]],
        device=original_quat_wxyz.device,
        dtype=original_quat_wxyz.dtype,
    )
    robot.write_root_link_pose_to_sim(
        torch.cat((desired_pos, desired_quat)).view(1, 7), env_ids=slice(0, 1)
    )
    # EntityData.root_link_pose_w is a concatenated derived-state copy; writing
    # root_link_pos_w/root_link_quat_w silently edits a temporary. Update qpos
    # via the entity API and refresh mjWarp kinematics without a physics step.
    sim.forward()
    actual_pos = robot.data.root_link_pos_w[0]
    actual_quat = robot.data.root_link_quat_w[0]
    _check(torch.allclose(actual_pos, desired_pos, atol=2e-5, rtol=0.0),
           "A3 pose fixture position was not applied to simulation state")
    _check(torch.allclose(actual_quat, desired_quat, atol=2e-5, rtol=0.0),
           "A3 pose fixture quaternion was not applied to simulation state")


def _pose_ray_diagnostic(frame, raw_delta, analytic_d, ids_np, pose, previous_pose,
                         root_pos, root_quat_wxyz, original_pos, original_quat_wxyz,
                         pipeline, mount, directions_sensor):
    """Describe the worst mismatched ray and both acquisition poses."""
    from scipy.spatial.transform import Rotation

    flat_i = int(raw_delta.argmax().item())
    source_id = int(ids_np[flat_i])
    base_q = original_quat_wxyz.detach().cpu().numpy()
    base_rot = Rotation.from_quat([base_q[1], base_q[2], base_q[3], base_q[0]])
    delta_rot = Rotation.from_euler(
        "xyz", [pose["roll_deg"], pose["pitch_down_deg"], 0.0], degrees=True
    )
    desired_rot = base_rot * delta_rot
    desired_pos = original_pos.detach().cpu().numpy().astype(np.float64) + np.array(
        [pose["x_m"], 0.0, pose["height_delta_m"]], dtype=np.float64
    )
    actual_pos = root_pos[0].detach().cpu().numpy().astype(np.float64)
    actual_q = root_quat_wxyz[0].detach().cpu().numpy()
    actual_rot = Rotation.from_quat([actual_q[1], actual_q[2], actual_q[3], actual_q[0]])
    mount_t = np.asarray(pipeline.mount_pos, dtype=np.float64)
    mount_q = np.asarray(pipeline.mount_quat_xyzw, dtype=np.float64)
    mount_rot = Rotation.from_quat(mount_q)
    frame_sensor_q = frame.sensor_quat_xyzw.detach().cpu().numpy().astype(np.float64)
    frame_sensor_rot = Rotation.from_quat(frame_sensor_q)
    desired_sensor_rot = desired_rot * mount_rot
    frame_base_q_wxyz = frame.base_quat_wxyz.detach().cpu().numpy().astype(np.float64)
    frame_base_rot = Rotation.from_quat(
        [frame_base_q_wxyz[1], frame_base_q_wxyz[2], frame_base_q_wxyz[3], frame_base_q_wxyz[0]]
    )
    frame_composed_sensor_rot = frame_base_rot * mount_rot
    desired_sensor = desired_pos + desired_rot.apply(mount_t)
    actual_sensor = actual_pos + actual_rot.apply(mount_t)
    desired_direction = desired_sensor_rot.apply(
        np.asarray(directions_sensor, dtype=np.float64)[flat_i]
    )
    actual_direction = frame_sensor_rot.apply(
        np.asarray(directions_sensor, dtype=np.float64)[flat_i]
    )
    sensor_q_dot = float(np.clip(abs(np.dot(
        frame_sensor_q / np.linalg.norm(frame_sensor_q), desired_sensor_rot.as_quat()
    )), 0.0, 1.0))
    return {
        "worst_source_id": source_id,
        "raw_gpu_distance_m": float(frame.raw_dists[flat_i].item()),
        "analytic_distance_m": float(analytic_d[flat_i].item()),
        "absolute_error_m": float(raw_delta[flat_i].item()),
        "pose_definition": dict(pose),
        "previous_pose_definition": None if previous_pose is None else dict(previous_pose),
        "actual_base_position_world": actual_pos.tolist(),
        "desired_base_position_world": desired_pos.tolist(),
        "actual_base_quaternion_wxyz": actual_q.tolist(),
        "frame_captured_base_quaternion_wxyz": frame_base_q_wxyz.tolist(),
        "desired_base_quaternion_wxyz": [desired_rot.as_quat()[3], *desired_rot.as_quat()[:3].tolist()],
        "frame_base_vs_current_quaternion_error_deg": float(2.0 * np.degrees(np.arccos(np.clip(abs(np.dot(
            frame_base_q_wxyz / np.linalg.norm(frame_base_q_wxyz), actual_q / np.linalg.norm(actual_q)
        )), 0.0, 1.0)))),
        "actual_sensor_origin_world": actual_sensor.tolist(),
        "desired_sensor_origin_world": desired_sensor.tolist(),
        "sensor_origin_error_m": float(np.linalg.norm(actual_sensor - desired_sensor)),
        "actual_sensor_quaternion_xyzw": frame_sensor_q.tolist(),
        "sensor_quaternion_from_frame_base_xyzw": frame_composed_sensor_rot.as_quat().tolist(),
        "desired_sensor_quaternion_xyzw": desired_sensor_rot.as_quat().tolist(),
        "sensor_orientation_error_deg": float(2.0 * np.degrees(np.arccos(sensor_q_dot))),
        "sensor_vs_frame_composition_error_deg": float(2.0 * np.degrees(np.arccos(np.clip(abs(np.dot(
            frame_sensor_q / np.linalg.norm(frame_sensor_q), frame_composed_sensor_rot.as_quat()
        )), 0.0, 1.0)))),
        "ray_direction_world_actual": actual_direction.tolist(),
        "ray_direction_world_desired": desired_direction.tolist(),
        "mount_position_base": mount_t.tolist(),
        "mount_quaternion_xyzw": mount_q.tolist(),
    }


def _mesh_for_scene(device: str, solids):
    """Make a real Warp triangle mesh for the A3 z=0 floor and opaque AABBs."""
    vertices: list[list[float]] = [
        [-100.0, -100.0, 0.0], [100.0, -100.0, 0.0],
        [100.0, 100.0, 0.0], [-100.0, 100.0, 0.0],
    ]
    triangles: list[tuple[int, int, int]] = [(0, 1, 2), (0, 2, 3)]
    # Six quad faces, two triangles each. Warp's mesh ray query is two-sided.
    face_triangles = (
        (0, 2, 1, 0, 3, 2), (4, 5, 6, 4, 6, 7),
        (0, 1, 5, 0, 5, 4), (1, 2, 6, 1, 6, 5),
        (2, 3, 7, 2, 7, 6), (3, 0, 4, 3, 4, 7),
    )
    for solid in solids:
        lo, hi = np.asarray(solid["lo"], dtype=np.float64), np.asarray(solid["hi"], dtype=np.float64)
        first = len(vertices)
        vertices.extend([
            [lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]],
            [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
            [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]],
            [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]],
        ])
        for tri in face_triangles:
            triangles.extend(tuple(first + i for i in tri[j:j + 3]) for j in (0, 3))
    points_np = np.ascontiguousarray(vertices, dtype=np.float32)
    indices_np = np.ascontiguousarray(triangles, dtype=np.int32).reshape(-1)
    with wp.ScopedDevice(device):
        points_wp = wp.array(points_np, dtype=wp.vec3)
        indices_wp = wp.array(indices_np, dtype=wp.int32)
        mesh = wp.Mesh(points=points_wp, indices=indices_wp)
        mesh_ids = wp.array([mesh.id], dtype=wp.uint64)
    return mesh, points_wp, indices_wp, mesh_ids


def _install_mesh(pipelines, device, solids, mesh_bank):
    owned = _mesh_for_scene(device, solids)
    mesh_bank.append(owned)
    mesh, points, indices, mesh_ids = owned
    for pipeline in pipelines:
        pipeline._wp_mesh = mesh
        pipeline._wp_points = points
        pipeline._wp_indices = indices
        pipeline.mesh_ids = mesh_ids
        for bundle in pipeline._sensors:
            bundle.sensor.mesh_ids = mesh_ids


def _make_observation(OnlineObservation, device, step: int, time_s: float):
    z3 = torch.zeros((1, 3), dtype=torch.float32, device=device)
    z18 = torch.zeros((1, 18), dtype=torch.float32, device=device)
    return OnlineObservation(
        env_ids=torch.zeros(1, dtype=torch.long, device=device),
        episode_ids=torch.zeros(1, dtype=torch.long, device=device),
        step_ids=torch.full((1,), step, dtype=torch.long, device=device),
        time_s=torch.full((1,), time_s, dtype=torch.float32, device=device),
        lin_vel_b=z3.clone(), ang_vel_b=z3.clone(),
        gravity_b=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32, device=device),
        joint_pos_rel_ref=z18.clone(), joint_vel=z18.clone(), prev_action=z18.clone(),
        command=z3.clone(),
    )


def _encode_frame(network, frame):
    return network.encode_points(
        frame.near_points_base.unsqueeze(0), frame.near_valid.unsqueeze(0),
        frame.far_points_base.unsqueeze(0), frame.far_valid.unsqueeze(0),
        near_angle_keys=frame.near_angle_keys.unsqueeze(0),
        far_angle_keys=frame.far_angle_keys.unsqueeze(0),
    )


def _network_sequence(network, frame_feature, observation, hidden):
    net_output = network.forward_from_features(observation, frame_feature, hidden)
    _check(bool(torch.isfinite(net_output.action_mean).all()), "network action mean is non-finite")
    _check(bool(torch.isfinite(net_output.hidden).all()), "network temporal hidden is non-finite")
    return net_output


def _frame_global_ids(frame, name: str) -> torch.Tensor:
    return getattr(frame, name + "_indices")[getattr(frame, name + "_valid")]


def _scan_metrics(pipeline, frame, selection_ids: torch.Tensor, mode: str) -> dict:
    _check(torch.equal(frame.raw_source_ids, selection_ids), "raw_source_ids changed from selection IDs")
    _check(bool(torch.isfinite(frame.raw_dists).all()), "raw distances contain NaN/Inf")
    nohit_expected = frame.raw_dists == float(frame.raw_dists.new_tensor(15.0))
    _check(torch.equal(frame.no_hit_mask, nohit_expected), "no_hit_mask must exactly track the 15m raw convention")
    near_ids = _frame_global_ids(frame, "near")
    far_ids = _frame_global_ids(frame, "far")
    for name, sampled_ids in (("near", near_ids), ("far", far_ids)):
        sampled_pos = torch.searchsorted(frame.raw_source_ids, sampled_ids)
        _check(bool((frame.raw_source_ids.index_select(0, sampled_pos) == sampled_ids).all()),
               f"sampled {name} ID not present in compact raw frame")
        _check(torch.unique(sampled_ids).numel() == sampled_ids.numel(),
               f"sampled {name} source IDs repeat")
    positions = torch.searchsorted(frame.raw_source_ids, far_ids)
    _check(bool((frame.raw_source_ids.index_select(0, positions) == far_ids).all()),
           "sampled far ID not present in compact raw frame")
    valid_dists = frame.raw_dists.index_select(0, positions)
    valid_misses = frame.no_hit_mask.index_select(0, positions)
    if mode == "hit_only_10m":
        _check(not bool(valid_misses.any()), "hit_only_10m retained a no-hit ray")
        _check(not bool((valid_dists > 10.0).any()), "hit_only_10m retained a raw hit over 10m")
    pool_positions = torch.searchsorted(frame.raw_source_ids, frame.pool_raw_indices)
    _check(bool((frame.raw_source_ids.index_select(0, pool_positions) == frame.pool_raw_indices).all()),
           "candidate pool source IDs are unmappable")
    fixed_far = ~pipeline.fixed_near_mask.index_select(0, pool_positions)
    consumed_valid = (
        torch.ones_like(frame.no_hit_mask) if frame.consumed_valid is None else frame.consumed_valid
    ).index_select(0, pool_positions)
    expected_far_mask = fixed_far & consumed_valid
    if mode == "hit_only_10m":
        expected_far_mask &= ~frame.no_hit_mask.index_select(0, pool_positions)
        expected_far_mask &= frame.raw_dists.index_select(0, pool_positions) <= 10.0
    _check(torch.equal(frame.far_mask, expected_far_mask),
           "far candidate mask differs from the full formal fixed-far/valid/hit/10m contract")
    actual_pool_ids = frame.pool_raw_indices[frame.far_mask]
    actual_positions = torch.searchsorted(frame.raw_source_ids, actual_pool_ids)
    _check(bool((frame.raw_source_ids.index_select(0, actual_positions) == actual_pool_ids).all()),
           "far candidate source IDs are unmappable")
    return {
        "raw_rays": int(frame.raw_source_ids.numel()),
        "raw_hits": int((~frame.no_hit_mask).sum().item()),
        "raw_misses": int(frame.no_hit_mask.sum().item()),
        "raw_hits_over_10m": int(((~frame.no_hit_mask) & (frame.raw_dists > 10.0)).sum().item()),
        "near_candidate_count": int(frame.near_mask.sum().item()),
        "far_candidate_count": int(frame.far_mask.sum().item()),
        "near_valid_count": int(frame.near_valid.sum().item()),
        "far_valid_count": int(frame.far_valid.sum().item()),
        "far_padding_count": int((~frame.far_valid).sum().item()),
        "far_selected_nohit_count": int(valid_misses.sum().item()),
        "far_selected_over_10m_count": int((valid_dists > 10.0).sum().item()),
        "far_source_ids_sha256": hashlib.sha256(
            far_ids.detach().cpu().numpy().astype("<i8", copy=False).tobytes()
        ).hexdigest(),
        "far_source_id_min": int(far_ids.min().item()) if far_ids.numel() else None,
        "far_source_id_max": int(far_ids.max().item()) if far_ids.numel() else None,
    }


def _expected_raw_points_base(pipeline, frame):
    """Rebuild clean raw endpoints in acquisition-time BASE coordinates."""
    raw_sensor = torch.where(
        frame.no_hit_mask.unsqueeze(-1),
        float(pipeline.cfg.max_range_m) * frame.raw_ray_dirs,
        frame.raw_points_sensor,
    )
    q = pipeline._mount_q.expand(frame.raw_points_sensor.shape[0], 4)
    qv, qw = q[:, :3], q[:, 3:4]
    t = 2.0 * torch.linalg.cross(qv, raw_sensor, dim=-1)
    rotated = raw_sensor + qw * t + torch.linalg.cross(qv, t, dim=-1)
    return rotated + pipeline._mount_t.expand_as(rotated)


def _retained_class_counts(frame, selection_ids_np, hit_labels, solids) -> dict:
    """Map the final far FPS IDs to A3 analytic labels at the same selected source IDs."""
    far_ids = _frame_global_ids(frame, "far")
    selection_ids = torch.as_tensor(selection_ids_np, dtype=torch.long, device=far_ids.device)
    selected_pos = torch.searchsorted(selection_ids, far_ids)
    _check(bool((selection_ids.index_select(0, selected_pos) == far_ids).all()),
           "far FPS source ID is absent from A3 selected-label table")
    labels = torch.as_tensor(hit_labels, dtype=torch.int16, device=far_ids.device).index_select(0, selected_pos)
    labels_np = labels.detach().cpu().numpy()
    counts = {
        "ground": int(np.count_nonzero(labels_np == 0)),
        "miss": int(np.count_nonzero(labels_np == -1)),
        "stair": 0,
        "roof": 0,
        "wall": 0,
        "obstacle": int(np.count_nonzero(labels_np > 0)),
        "objects": {},
    }
    for i, solid in enumerate(solids, 1):
        count = int(np.count_nonzero(labels_np == i))
        counts["objects"][solid["name"]] = count
        counts[solid["kind"]] += count
    return counts


def _padding_fixtures(pipeline, raw_frame, results: dict, network, OnlineObservation, device) -> None:
    """Artificial eligibility masks over a real raycast frame exercise native pad logic."""
    for eligible_n in (0, 30, 60, 128):
        fixture = copy.copy(raw_frame)
        mask = torch.zeros_like(fixture.far_mask)
        candidate_slots = torch.nonzero(fixture.far_mask, as_tuple=False).flatten()
        if eligible_n > int(candidate_slots.numel()):
            # A frame with the 128-point legacy candidate pool is enough for the
            # largest fixture; use all pool slots only when they are available.
            raise AssertionError(
                f"real raycast frame has {candidate_slots.numel()} far candidates, cannot fixture {eligible_n}"
            )
        if eligible_n:
            mask[candidate_slots[:eligible_n]] = True
        fixture.far_mask = mask
        pipeline._sample_frames([fixture])
        count = int(fixture.far_valid.sum().item())
        _check(count == eligible_n, f"native sampler emitted {count} far slots, fixture requested {eligible_n}")
        invalid = ~fixture.far_valid
        _check(torch.equal(fixture.far_points_base[invalid], torch.zeros_like(fixture.far_points_base[invalid])),
               "native sampler did not zero-fill far padding")
        _check(bool((fixture.far_indices[invalid] == -1).all()), "native sampler padding source IDs are not -1")
        _check(torch.unique(fixture.far_indices[fixture.far_valid]).numel() == eligible_n,
               "native sampler repeated a real source ID")
        obs = _make_observation(OnlineObservation, device, eligible_n, eligible_n / 50.0)
        with torch.no_grad():
            feature = _encode_frame(network, fixture)
            net_output = network.forward_from_features(
                obs, feature, network.initial_hidden(1, device)
            )
        _check(bool(torch.isfinite(feature).all()), "padding fixture frame feature is non-finite")
        _check(bool(torch.isfinite(net_output.action_mean).all()), "padding fixture action mean is non-finite")
        _check(bool(torch.isfinite(net_output.hidden).all()), "padding fixture hidden is non-finite")
        results[str(eligible_n)] = {
            "fixture": "synthetic eligibility mask over actual Warp raw frame; not A3 scene coverage",
            "far_valid": count,
            "far_padding": 128 - count,
            "zero_filled": True,
            "source_id_sentinel_minus_one": True,
            "network_frame_feature_finite": True,
            "network_action_hidden_finite": True,
        }


def _make_network(device):
    from torch import nn
    from hexapod_mjlab.rl.confined_locomotion_2.network import ConfinedLocomotionNetwork

    network = ConfinedLocomotionNetwork(
        mlp_activation=nn.ELU(), init_fn=None, envelope_mode="raw", ground_samples=None
    ).to(device)
    network.eval()
    return network


def _persist_report(output_dir, report, original_exception, original_traceback):
    """Write diagnostics while retaining the real failure if rich JSON fails."""
    report_path = Path(output_dir) / "report.json"
    try:
        report_text = json.dumps(
            report, ensure_ascii=False, indent=2, default=_json_value, allow_nan=False
        )
        report_path.write_text(report_text, encoding="utf-8")
        return None
    except BaseException as report_error:
        report["report_write_error"] = repr(report_error)
        minimal = {
            "schema": report.get("schema"),
            "passed": False,
            "error": original_exception or report.get("error"),
            "traceback": original_traceback or report.get("traceback"),
            "report_write_error": repr(report_error),
            "elapsed_s": report.get("elapsed_s"),
        }
        try:
            report_path.write_text(json.dumps(minimal, ensure_ascii=False, indent=2), encoding="utf-8")
        except BaseException as fallback_error:
            print(f"FATAL: could not preserve report at {report_path}: {fallback_error!r}", file=sys.stderr)
            print(json.dumps(minimal, ensure_ascii=False), file=sys.stderr)
        return repr(report_error)


def main(argv=None) -> None:
    default_selection = (
        ROOT / "src/hexapod_mjlab/tasks/confined_locomotion_2/assets/fixed_channels/v1_hit10/near_x6_far_x10.npz"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="new evidence directory; must not exist")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--selection", default=str(default_selection),
                        help="fixed sparse source selection NPZ (default: v1_hit10 near_x6_far_x10)")
    parser.add_argument("--seed", type=int, default=20261023)
    args = parser.parse_args(argv)
    output_dir = Path(args.output).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"evidence directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    source_path = Path(__file__).resolve()
    source_copy = output_dir / "executed_probe.py"
    shutil.copy2(source_path, source_copy)
    script_sha256 = hashlib.sha256(source_copy.read_bytes()).hexdigest()
    report = {
        "schema": "confined_locomotion_2.far_hit_gpu_acceptance/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": False,
        "seed": args.seed,
        "gpu_index": args.gpu,
        "selection": str(Path(args.selection).expanduser().resolve()),
        "executed_script": str(source_copy),
        "executed_script_sha256": script_sha256,
        "scope": [
            "Real CUDA Warp raycast and production ConfinedScanPipeline._sample_frames over the fixed sparse candidate IDs.",
            "A3 poses are interpolated kinematic samples; they are not an MJLab dynamics rollout or feasible locomotion trajectory.",
            "Network uses deterministic-seed random initialization unless weights are supplied by a later extension; outputs are interface diagnostics, not evidence of training effect or stable learned control.",
            "Network diagnostics use zero linear/angular velocity, gravity [0,0,-1], zero joint state, zero previous action and zero command at every synthetic pose; they isolate point-cloud effects and are not trajectory-consistent proprioception or action dynamics.",
            "A3 raw ray distances are physically raycast against a triangulated z=0 floor and A3 opaque AABB boxes; only artificial eligibility masks are used for exact padding fixtures.",
            "Far padding is zero-filled with source ID -1; active recurrent hidden is measured, with no empirical learned-stability threshold asserted.",
        ],
    }
    started = time.monotonic()
    original_exception = None
    original_traceback = None
    env = runtime = None
    mesh_bank = []
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("real CUDA is required; CPU does not pass this acceptance")
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
        selection_path = Path(args.selection).expanduser().resolve()
        if not selection_path.is_file():
            raise FileNotFoundError(f"fixed far-hit selection does not exist: {selection_path}")

        # Import the A3 scenario producer instead of copying geometry/poses.
        a3_dir = ROOT / "scripts/experiments/confined_locomotion_2/sparse_rays"
        sys.path.insert(0, str(a3_dir))
        from evaluate_far_trajectory import make_scenarios, interpolate_poses, scan_pose
        from hexapod_mjlab.tasks.confined_locomotion_2.channels import build_channel_catalog, load_sparse_selection
        from hexapod_mjlab.tasks.confined_locomotion_2.flat_env_cfg import make_training_env_cfg
        from hexapod_mjlab.tasks.confined_locomotion_2.scanner import ConfinedLidarCfg, ConfinedScanPipeline
        from hexapod_mjlab.tasks.confined_locomotion_2.runtime import TrainingRuntime
        from hexapod_mjlab.common.perturbations import (
            GapPerturbationCfg, ObservationPerturbationCfg, ObservationPerturbations,
            RangePerturbationCfg,
        )
        from hexapod_mjlab.rl.confined_locomotion_2.contracts import OnlineObservation
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper

        catalog = build_channel_catalog()
        ids_np, selection_identity = load_sparse_selection(str(selection_path), catalog)
        ids = torch.as_tensor(ids_np, dtype=torch.long, device=device)
        _check(ids.numel() > 128, "selection needs >128 candidate IDs to exercise 128 far padding fixture")
        report["catalog_identity"] = catalog.identity
        report["selection_identity"] = selection_identity
        report["selection_source_id_count"] = int(ids.numel())
        report["selection_near_count"] = int(catalog.near_mask[ids_np].sum())
        report["selection_far_count"] = int((~catalog.near_mask[ids_np]).sum())
        report["gpu_name"] = torch.cuda.get_device_name(args.gpu)

        env = ManagerBasedRlEnv(
            make_training_env_cfg(num_envs=1, seed=args.seed), device=str(device)
        )
        wrapper = RslRlVecEnvWrapper(env, clip_actions=None)
        # Production runtime owns the hit-only pipeline and provides an actual
        # lifecycle/cache/reset seam. Its initial takeover scan is a real plane scan.
        hit_cfg = ConfinedLidarCfg(
            raycast_mode="sparse", sparse_selection_path=str(selection_path),
            far_sampling="hit_only_10m",
        )
        runtime = TrainingRuntime(
            wrapper, seed=args.seed, output_dir=output_dir / "runtime", mode="training_lifecycle_fixture",
            lidar_cfg=hit_cfg,
        )
        hit_pipeline = runtime.pipeline
        legacy_pipeline = ConfinedScanPipeline(
            env, ConfinedLidarCfg(
                raycast_mode="sparse", sparse_selection_path=str(selection_path),
                far_sampling="all_endpoints",
            )
        )
        _check(torch.equal(hit_pipeline.source_ray_ids, ids), "production hit-only source IDs mismatch selection")
        _check(torch.equal(legacy_pipeline.source_ray_ids, ids), "legacy source IDs mismatch same selection")
        pipelines = (hit_pipeline, legacy_pipeline)
        robot = env.scene["robot"]
        original_pos = robot.data.root_link_pos_w[0].detach().clone()
        original_quat = robot.data.root_link_quat_w[0].detach().clone()

        # Production cache repeatability and selected reset identity change.
        cached_a = runtime.prepare()
        launch_before_repeat = hit_pipeline._sensors[0].sensor.launch_count
        cached_b = runtime.prepare()
        _check(cached_a is cached_b, "repeat prepare did not return same cached object")
        _check(hit_pipeline._sensors[0].sensor.launch_count == launch_before_repeat,
               "cached prepare launched another physical scan")
        cache_record = {"same_object_on_repeat": True, "raycast_launches_on_repeat": 0}

        scenarios = make_scenarios()
        trajectories = [(scene, interpolate_poses(scene["poses"], 5)) for scene in scenarios]
        total_frames = sum(len(poses) for _, poses in trajectories)
        _check(total_frames == 227, f"A3 expected 227 interpolated poses, got {total_frames}")
        report["trajectory_definition"] = {
            "name": "A3 make_scenarios/interpolate_poses(substeps=5)",
            "frame_count": total_frames,
            "scan_rate_hz_interpretation": 10,
            "control_rate_hz_interpretation": 50,
            "dynamic_simulation_steps": 0,
            "is_dynamics_trajectory": False,
        }

        from scipy.spatial.transform import Rotation
        base_q = original_quat.detach().cpu().numpy()
        base_rotation = Rotation.from_quat([base_q[1], base_q[2], base_q[3], base_q[0]]).as_matrix()
        mount_pos = hit_pipeline.mount_pos.astype(np.float64)
        mount_rot = Rotation.from_quat(hit_pipeline.mount_quat_xyzw.astype(np.float64)).as_matrix()
        mount = {
            "base_pos_world": original_pos.detach().cpu().numpy().astype(np.float64),
            "base_rot_world": base_rotation,
            "mount_pos_base": mount_pos,
            "mount_rot_base": mount_rot,
        }

        torch.manual_seed(args.seed + 101)
        network = _make_network(device)
        hidden = {
            "all_endpoints": network.initial_hidden(1, device),
            "hit_only_10m": network.initial_hidden(1, device),
        }
        last_mean = {key: None for key in hidden}
        network_stats = {key: {"action_mean_norm": [], "delta_from_previous_control_norm": [],
                               "frame_feature_norm": [], "frame_feature_delta_norm": [],
                               "temporal_feature_norm": [], "hidden_norm": [], "hidden_max_abs": [],
                               "far_valid_count": [], "far_valid_delta": []} for key in hidden}
        scene_report = {}
        pad_recorded = False
        frame_idx_global = 0
        for scene, poses in trajectories:
            # Each named A3 scene is an independent synthetic trajectory.
            hidden = {key: network.initial_hidden(1, device) for key in hidden}
            last_mean = {key: None for key in last_mean}
            _install_mesh(pipelines, str(device), scene["solids"], mesh_bank)
            scene_frames = []
            previous_far_count = {key: None for key in hidden}
            previous_frame_feature = {key: None for key in hidden}
            previous_pose = None
            for frame_idx, pose in enumerate(poses):
                _pose_robot(robot, env.sim, original_pos, original_quat, pose)
                # Use both production paths over the identical physical mesh and pose.
                hit_frame = hit_pipeline.scan_rows([0])[0]
                legacy_frame = legacy_pipeline.scan_rows([0])[0]
                torch.cuda.synchronize(device)
                legacy_metrics = _scan_metrics(legacy_pipeline, legacy_frame, ids, "all_endpoints")
                hit_metrics = _scan_metrics(hit_pipeline, hit_frame, ids, "hit_only_10m")
                _check(torch.equal(hit_frame.raw_dists, legacy_frame.raw_dists),
                       f"{scene['name']} frame {frame_idx}: raw Warp distances differ between far sampling modes")
                _check(torch.equal(hit_frame.raw_source_ids, legacy_frame.raw_source_ids),
                       f"{scene['name']} frame {frame_idx}: raw source IDs differ by mode")
                for field in ("near_indices", "near_valid", "near_points_base", "near_angle_keys"):
                    _check(torch.equal(getattr(hit_frame, field), getattr(legacy_frame, field)),
                           f"{scene['name']} frame {frame_idx}: far mode changed near {field}")

                # Compare actual raw GPU distances with the same A3 analytic
                # solid intersection reference, only at selected IDs.
                # A3 geometry reference is evaluated only on this fixed sparse
                # source set; do not repeat its 86,400-ray full analytic scan.
                analytic = scan_pose(catalog.directions_sensor[ids_np], mount, pose, scene["solids"],
                                     fixed_near=catalog.near_mask[ids_np])
                analytic_d = torch.as_tensor(analytic["distances_m"], device=device, dtype=torch.float32)
                analytic_labels = np.asarray(analytic["hit_label"])
                raw_delta = (hit_frame.raw_dists - analytic_d).abs()
                _check(bool(torch.isfinite(raw_delta).all()), "analytic/GPU comparison is non-finite")
                max_delta = float(raw_delta.max().item())
                _check(max_delta <= 0.025,
                       f"{scene['name']} frame {frame_idx}: GPU/A3 first-hit max error {max_delta:.4f}m > 0.025m; "
                       f"diagnostic={json.dumps(_pose_ray_diagnostic(hit_frame, raw_delta, analytic_d, ids_np, pose, previous_pose, robot.data.root_link_pos_w, robot.data.root_link_quat_w, original_pos, original_quat, hit_pipeline, mount, catalog.directions_sensor[ids_np]), sort_keys=True)}")

                if not pad_recorded and scene["name"] == "ascending_stairs" and frame_idx == 0:
                    _padding_fixtures(legacy_pipeline, legacy_frame,
                                      report.setdefault("native_padding_fixtures", {}),
                                      network, OnlineObservation, device)
                    pad_recorded = True

                frames = {"all_endpoints": legacy_frame, "hit_only_10m": hit_frame}
                metrics = {"all_endpoints": legacy_metrics, "hit_only_10m": hit_metrics}
                labels_by_mode = {}
                feature_by_mode = {}
                feature_delta_by_mode = {}
                far_count_delta_by_mode = {
                    mode: (None if previous_far_count[mode] is None else
                           metrics[mode]["far_valid_count"] - previous_far_count[mode])
                    for mode in frames
                }
                for mode in frames:
                    if far_count_delta_by_mode[mode] is not None:
                        network_stats[mode]["far_valid_delta"].append(
                            far_count_delta_by_mode[mode]
                        )
                for mode, frame in frames.items():
                    labels_by_mode[mode] = _retained_class_counts(
                        frame, ids_np, analytic_labels, scene["solids"]
                    )
                    with torch.no_grad():
                        feature = _encode_frame(network, frame)
                    _check(bool(torch.isfinite(feature).all()), "frame feature is non-finite")
                    feature_by_mode[mode] = feature
                    feature_delta_by_mode[mode] = (
                        None if previous_frame_feature[mode] is None else
                        float(torch.linalg.vector_norm(
                            feature - previous_frame_feature[mode]
                        ).item())
                    )
                    network_stats[mode]["frame_feature_norm"].append(
                        float(torch.linalg.vector_norm(feature).item())
                    )
                    if feature_delta_by_mode[mode] is not None:
                        network_stats[mode]["frame_feature_delta_norm"].append(
                            feature_delta_by_mode[mode]
                        )
                    previous_frame_feature[mode] = feature.detach().clone()

                control_outputs = {}
                for mode, frame in (("all_endpoints", legacy_frame), ("hit_only_10m", hit_frame)):
                    for repeat in range(5):
                        control_step = frame_idx_global * 5 + repeat
                        observation = _make_observation(
                            OnlineObservation, device, control_step,
                            frame_idx_global / 10.0 + repeat / 50.0,
                        )
                        with torch.no_grad():
                            net_output = _network_sequence(
                                network, feature_by_mode[mode], observation, hidden[mode]
                            )
                        mean_norm = float(torch.linalg.vector_norm(net_output.action_mean).item())
                        delta_norm = (None if last_mean[mode] is None else
                                      float(torch.linalg.vector_norm(net_output.action_mean - last_mean[mode]).item()))
                        hnorm = float(torch.linalg.vector_norm(net_output.hidden).item())
                        hmax = float(net_output.hidden.abs().max().item())
                        network_stats[mode]["action_mean_norm"].append(mean_norm)
                        if delta_norm is not None:
                            network_stats[mode]["delta_from_previous_control_norm"].append(delta_norm)
                        network_stats[mode]["temporal_feature_norm"].append(
                            float(torch.linalg.vector_norm(net_output.temporal_feature).item()))
                        network_stats[mode]["hidden_norm"].append(hnorm)
                        network_stats[mode]["hidden_max_abs"].append(hmax)
                        network_stats[mode]["far_valid_count"].append(int(frame.far_valid.sum().item()))
                        hidden[mode] = net_output.hidden
                        last_mean[mode] = net_output.action_mean.detach().clone()
                        if repeat == 4:
                            control_outputs[mode] = {
                                "action_mean_norm": mean_norm,
                                "delta_from_previous_control_norm": delta_norm,
                                "hidden_norm": hnorm,
                                "hidden_max_abs": hmax,
                            }

                scene_frames.append({
                    "frame": frame_idx, "pose": pose,
                    "all_endpoints": {**legacy_metrics,
                                      "far_selected_class_counts": labels_by_mode["all_endpoints"]},
                    "hit_only_10m": {**hit_metrics,
                                     "far_selected_class_counts": labels_by_mode["hit_only_10m"]},
                    "raw_distance_max_abs_vs_A3_analytic_m": max_delta,
                    "analytic_selected_hit_counts_by_object": {
                        solid["name"]: int(np.count_nonzero(analytic_labels == i + 1))
                        for i, solid in enumerate(scene["solids"])
                    },
                    "frame_feature_norm": {
                        mode: float(torch.linalg.vector_norm(feature_by_mode[mode]).item())
                        for mode in frames
                    },
                    "frame_feature_delta_from_previous_scan_norm": feature_delta_by_mode,
                    "far_valid_count_delta_from_previous_scan": far_count_delta_by_mode,
                    "control_step_5_metrics": control_outputs,
                    "same_near_outputs": True,
                    "raw_hit_geometry_same": True,
                })
                # Raw frames own many GPU tensors. Keep the current pair only;
                # the evidence records compact metrics and source-ID hashes.
                hit_pipeline.scan_records.clear()
                legacy_pipeline.scan_records.clear()
                for mode in frames:
                    previous_far_count[mode] = metrics[mode]["far_valid_count"]
                frame_idx_global += 1
                previous_pose = dict(pose)

            class_visibility = {}
            for mode in ("all_endpoints", "hit_only_10m"):
                class_visibility[mode] = {}
                for class_name in ("ground", "roof", "stair", "wall", "obstacle"):
                    counts = [f[mode]["far_selected_class_counts"][class_name] for f in scene_frames]
                    class_visibility[mode][class_name] = {
                        "min_far_fps_count": min(counts),
                        "frames_with_visible_far_fps": int(sum(count > 0 for count in counts)),
                        "visible_frame_ratio": float(sum(count > 0 for count in counts) / len(counts)),
                    }
            transitions = []
            for prev, curr in zip(scene_frames, scene_frames[1:]):
                for mode in ("all_endpoints", "hit_only_10m"):
                    transitions.append({
                        "mode": mode,
                        "from_frame": prev["frame"], "to_frame": curr["frame"],
                        "far_valid_count_delta": curr[mode]["far_valid_count"] - prev[mode]["far_valid_count"],
                        "frame_feature_delta_norm": curr["frame_feature_delta_from_previous_scan_norm"][mode],
                    })
            top_feature_changes = sorted(
                transitions,
                key=lambda item: abs(item["frame_feature_delta_norm"] or 0.0),
                reverse=True,
            )[:5]
            feature_far_count_corr = {}
            for mode in ("all_endpoints", "hit_only_10m"):
                pairs = [
                    (abs(item["far_valid_count_delta"]), item["frame_feature_delta_norm"])
                    for item in transitions
                    if item["mode"] == mode and item["frame_feature_delta_norm"] is not None
                ]
                if len(pairs) >= 2 and np.std(np.asarray(pairs)[:, 0]) > 0 and np.std(np.asarray(pairs)[:, 1]) > 0:
                    corr = float(np.corrcoef(np.asarray(pairs, dtype=np.float64).T)[0, 1])
                else:
                    corr = None
                feature_far_count_corr[mode] = corr
            scene_report[scene["name"]] = {
                "solids": scene["solids"],
                "frames": scene_frames,
                "legacy_far_valid_min": min(x["all_endpoints"]["far_valid_count"] for x in scene_frames),
                "hit_only_far_valid_min": min(x["hit_only_10m"]["far_valid_count"] for x in scene_frames),
                "hit_only_frames_with_padding": sum(x["hit_only_10m"]["far_padding_count"] > 0 for x in scene_frames),
                "far_fps_class_visibility": class_visibility,
                "top_frame_feature_deltas_with_far_count_change": top_feature_changes,
                "abs_far_count_delta_vs_frame_feature_delta_pearson_r": feature_far_count_corr,
            }

        _check(pad_recorded, "native exact far padding fixture was not run")
        # Formal P6 consumption test: fixed gap schedule and range-noise streams
        # on the same representative A3 roof/tunnel raw returns in both modes.
        _install_mesh(pipelines, str(device), next(s["solids"] for s in scenarios if s["name"] == "low_tunnel"), mesh_bank)
        representative = next(s for s in scenarios if s["name"] == "low_tunnel")
        pose = interpolate_poses(representative["poses"], 1)[3]
        _pose_robot(robot, env.sim, original_pos, original_quat, pose)
        clean_hit = hit_pipeline.scan_rows([0])[0]
        clean_legacy = legacy_pipeline.scan_rows([0])[0]
        _check(torch.equal(clean_hit.raw_dists, clean_legacy.raw_dists),
               "clean representative raw scans differ across modes")
        expected_clean_raw_base = _expected_raw_points_base(hit_pipeline, clean_hit)
        _check(torch.allclose(clean_hit.raw_points_base, expected_clean_raw_base, atol=1e-6, rtol=0.0),
               "clean raw_points_base does not match raw range/endpoint transform")
        _check(torch.allclose(torch.linalg.vector_norm(clean_hit.raw_points_sensor, dim=-1),
                              clean_hit.raw_dists, atol=1e-3, rtol=1e-4),
               "clean raw sensor endpoints do not match raw ranges")
        noise_cfg = ObservationPerturbationCfg(
            gap=GapPerturbationCfg(jitter_enabled=False), range=RangePerturbationCfg(), proprio=None
        )
        noise_seed = args.seed + 19
        for _ in range(10000):
            p_hit = ObservationPerturbations(1, cfg=noise_cfg, seed=noise_seed, device=device)
            phase = p_hit.gap_schedule_record()[0]["phase"]
            if phase == 2:
                break
            noise_seed += 1
        else:
            raise RuntimeError("could not find deterministic P6 phase-2 seed")
        p_legacy = ObservationPerturbations(1, cfg=noise_cfg, seed=noise_seed, device=device)
        p_hit = ObservationPerturbations(1, cfg=noise_cfg, seed=noise_seed, device=device)
        noise_record = {"seed": noise_seed, "phase": phase, "raw_equal_across_modes": True,
                        "consumed_equal_across_modes": True, "gap_removed_observations": 0,
                        "range_noise_applied_hits": 0}
        for persistent_scan in (1, 2):
            hit_noisy = hit_pipeline.scan_rows(
                [0], perturbations=p_hit, persistent_scan_numbers={0: persistent_scan}
            )[0]
            legacy_noisy = legacy_pipeline.scan_rows(
                [0], perturbations=p_legacy, persistent_scan_numbers={0: persistent_scan}
            )[0]
            _scan_metrics(hit_pipeline, hit_noisy, ids, "hit_only_10m")
            _scan_metrics(legacy_pipeline, legacy_noisy, ids, "all_endpoints")
            for field in ("raw_dists", "raw_points_sensor", "raw_points_base", "no_hit_mask", "raw_source_ids"):
                _check(torch.equal(getattr(hit_noisy, field), getattr(legacy_noisy, field)),
                       f"P6 changed raw field {field} across mode or physical scan")
                _check(torch.equal(getattr(hit_noisy, field), getattr(clean_hit, field)),
                       f"P6 consumption noise mutated raw field {field}")
            expected_noisy_raw_base = _expected_raw_points_base(hit_pipeline, hit_noisy)
            _check(torch.allclose(hit_noisy.raw_points_base, expected_noisy_raw_base, atol=1e-6, rtol=0.0),
                   "P6 raw_points_base does not match untouched raw range/endpoint transform")
            for field in ("consumed_valid", "consumed_dists", "range_epsilon", "range_applied_mask"):
                _check(torch.equal(getattr(hit_noisy, field), getattr(legacy_noisy, field)),
                       f"P6 consumed field {field} differs for identical source stream")
            for field in ("near_indices", "near_valid", "near_points_base", "near_angle_keys"):
                _check(torch.equal(getattr(hit_noisy, field), getattr(legacy_noisy, field)),
                       f"P6 far sampling changed near output {field}")
            noise_record["gap_removed_observations"] += int((~hit_noisy.consumed_valid).sum().item())
            noise_record["range_noise_applied_hits"] += int(hit_noisy.range_applied_mask.sum().item())
            noise_record.setdefault("per_scan", []).append({
                "persistent_scan": persistent_scan,
                "gap_fired": bool(hit_noisy.gap_fired),
                "valid_raw_count": int(hit_noisy.consumed_valid.sum().item()),
                "range_applied_count": int(hit_noisy.range_applied_mask.sum().item()),
                "raw_dists_unchanged_by_consumption_noise": True,
            })
        _check(noise_record["gap_removed_observations"] > 0,
               "scheduled P6 gap excluded no selected raw observations")
        _check(noise_record["range_noise_applied_hits"] > 0,
               "P6 range noise was not applied to any hit")
        report["p6_noise_and_gap"] = noise_record

        # Reset the real runtime at an unchanged global control step: episode
        # identity changes, first scan is acquired, and cache is rebuilt once.
        before_reset_launch = hit_pipeline._sensors[0].sensor.launch_count
        reset_record = runtime.reset(torch.tensor([0], device=device))
        after_reset = runtime.prepare()
        _check(after_reset is not cached_a, "selected reset reused the pre-reset cached input")
        _check(int(runtime.episode_ids[0].item()) == 1, "runtime reset did not advance episode identity")
        _check(hit_pipeline._sensors[0].sensor.launch_count > before_reset_launch,
               "runtime reset did not produce a fresh first scan")
        _check(after_reset is runtime.prepare(), "post-reset prepare is not cache-stable")
        cache_record.update({
            "episode_after_reset": int(runtime.episode_ids[0].item()),
            "reset_created_new_first_scan": True,
            "post_reset_repeat_same_object": True,
            "reset_record_keys": sorted(reset_record.keys()),
        })
        report["runtime_cache_reset"] = cache_record
        report["network_diagnostics"] = {
            "weights": "random initialization; no checkpoint/learning claim",
            "control_steps": frame_idx_global * 5,
            "scans": frame_idx_global,
            "per_mode": {
                mode: {
                    "action_mean_norm": _summary(values["action_mean_norm"]),
                    "action_mean_delta_norm_per_50hz_step": _summary(values["delta_from_previous_control_norm"]),
                    "frame_feature_norm_per_10hz_scan": _summary(values["frame_feature_norm"]),
                    "frame_feature_delta_norm_per_adjacent_scan": _summary(values["frame_feature_delta_norm"]),
                    "temporal_feature_norm": _summary(values["temporal_feature_norm"]),
                    "hidden_norm": _summary(values["hidden_norm"]),
                    "hidden_max_abs": _summary(values["hidden_max_abs"]),
                    "far_valid_count": _summary(values["far_valid_count"]),
                    "far_valid_count_delta_per_10hz_scan": _summary(values["far_valid_delta"]),
                    "finite": True,
                    "stability_claim": "none; diagnostic statistics for random weights and this short input sequence",
                }
                for mode, values in network_stats.items()
            },
            "comparison_interpretation": "The two modes share identical near samples, raw returns, selection IDs and randomized initial weights; statistics quantify input/output changes only.",
        }
        report["scenes"] = scene_report
        report["passed"] = True
    except BaseException as error:
        original_exception = repr(error)
        original_traceback = traceback.format_exc()
        report["passed"] = False
        report["error"] = original_exception
        report["traceback"] = original_traceback
    finally:
        if runtime is not None:
            try:
                runtime.close()
            except BaseException as close_error:
                report["close_error"] = repr(close_error)
        if env is not None:
            try:
                env.close()
            except BaseException as env_close_error:
                report["env_close_error"] = repr(env_close_error)
        report["elapsed_s"] = time.monotonic() - started
        if torch.cuda.is_available():
            try:
                report["gpu_name"] = torch.cuda.get_device_name(args.gpu)
                report["peak_allocated_MiB"] = torch.cuda.max_memory_allocated(args.gpu) / 2**20
            except BaseException as gpu_info_error:
                report["gpu_info_error"] = repr(gpu_info_error)
        # Preserve the actual scan/network exception even if report
        # serialization or the output filesystem also fails.
        _persist_report(output_dir, report, original_exception, original_traceback)
        print(json.dumps({"output": str(output_dir), "passed": report["passed"],
                          "elapsed_s": report["elapsed_s"], "frames": report.get("trajectory_definition", {}).get("frame_count"),
                          "error": report.get("error")}, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


def _summary(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "min": None, "mean": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {"count": int(array.size), "min": float(array.min()), "mean": float(array.mean()),
            "p95": float(np.percentile(array, 95)), "max": float(array.max())}


if __name__ == "__main__":
    main()
