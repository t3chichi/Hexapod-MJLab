"""Finite CUDA acceptance check for fixed-ID full and sparse LiDAR scans.

This is an integration/throughput check, not a training run. It compares real
raycasts from one unchanged simulated pose, then checks fixed source identity,
grouping, consumption, FPS and runtime checkpoint identity.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _same_tensor(a: torch.Tensor, b: torch.Tensor, *, atol=0.0, rtol=0.0) -> bool:
    if a.shape != b.shape:
        return False
    a, b = a.detach().cpu(), b.detach().cpu()
    if a.dtype == torch.bool or not (a.is_floating_point() or b.is_floating_point()):
        return torch.equal(a, b)
    return bool(torch.allclose(a, b, atol=atol, rtol=rtol, equal_nan=False))


def _indices_in(frame, source_ids: torch.Tensor) -> torch.Tensor:
    """Map global source IDs into the frame's compact raw arrays."""
    ids = frame.raw_source_ids
    pos = torch.searchsorted(ids, source_ids)
    _check(bool((pos < ids.numel()).all()), "source ID absent from compact raw frame")
    _check(bool((ids.index_select(0, pos) == source_ids).all()),
           "source ID absent from compact raw frame")
    return pos


def _latest(pipeline, num_envs: int):
    by_env = {}
    for record in pipeline.scan_records:
        frame = record["frame"]
        by_env[int(record["env_row"])] = frame
    _check(set(by_env) == set(range(num_envs)), "missing scan frame for an env row")
    return [by_env[i] for i in range(num_envs)]


def _frame_source_ids(frame, kind: str) -> torch.Tensor:
    valid = getattr(frame, kind + "_valid")
    ids = getattr(frame, kind + "_indices")
    return ids[valid]


def _set_pose(root_pos: torch.Tensor, root_quat_wxyz: torch.Tensor,
              original_pos: torch.Tensor, original_quat: torch.Tensor,
              *, roll_deg: float, pitch_deg: float, height_m: float) -> None:
    from scipy.spatial.transform import Rotation

    root_pos.copy_(original_pos)
    root_pos[:, 2] += float(height_m)
    delta = Rotation.from_euler("xyz", [roll_deg, pitch_deg, 0.0], degrees=True)
    root_quat_wxyz.copy_(original_quat)
    for row, original in enumerate(original_quat.detach().cpu().numpy()):
        baseline = Rotation.from_quat([original[1], original[2], original[3], original[0]])
        q_xyzw = (baseline * delta).as_quat()
        root_quat_wxyz[row] = torch.tensor(
            [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]],
            device=root_quat_wxyz.device, dtype=root_quat_wxyz.dtype,
        )


def _run_obstacle_raycasts(torch, wp, device, catalog, selection_ids: np.ndarray) -> dict:
    """Exercise the raw Warp ray kernel on three independent obstacle meshes.

    Raw selected ray distances must match full resolution. FPS outputs are
    reported as separate geometric samples: the fixed sparse candidate budget
    can miss geometry that a full-resolution candidate/FPS pool observes.
    """
    experiment_dir = ROOT / "scripts/experiments/confined_locomotion_2/sparse_rays"
    sys.path.insert(0, str(experiment_dir))
    try:
        from benchmark import make_query
        from check_obstacles_gpu import mesh_for_scene
        from evaluate_obstacles import SCENES, scan_obstacles
        from hexapod_mjlab.tasks.confined_locomotion_2.components.lidar.sampling import (
            farthest_point_sampling_masked,
        )
        from hexapod_mjlab.tasks.confined_locomotion_2.components.lidar.channels import _t1_pose_and_mount

        rays = np.asarray(catalog.directions_sensor, dtype=np.float32)
        _pose, _base_pos, _base_rot, mount_flat = _t1_pose_and_mount()
        mount_pos = mount_flat[:3]
        mount_rot = mount_flat[3:].reshape(3, 3)
        mount = {
            "base_pos_world": np.asarray(catalog.identity["pose_t1"]["base_pos_world_m"]),
            "base_rot_world": np.asarray(catalog.identity["pose_t1"]["base_rot_world"]),
            "mount_pos_base": mount_pos,
            "mount_rot_base": mount_rot,
        }
        selected_ids = np.asarray(selection_ids, dtype=np.int64)
        selected_set = np.zeros(len(rays), dtype=bool)
        selected_set[selected_ids] = True
        dirs_base = (mount_rot @ rays.astype(np.float64).T).T
        mount_t = torch.as_tensor(mount_pos, device=device, dtype=torch.float32)
        dirs_base_t = torch.as_tensor(dirs_base, device=device, dtype=torch.float32)
        results = {}
        scene_by_name = {scene["name"]: scene for scene in SCENES}
        for name in ("step_25cm", "corridor_1p6", "overhead_beam"):
            scene = scene_by_name[name]
            mesh = mesh_for_scene(wp, str(device), scene)
            full_query = make_query(
                wp, torch, str(device), mesh, rays, mount,
                scene["pitch"], scene["roll"],
            )
            sparse_query = make_query(
                wp, torch, str(device), mesh, rays[selected_ids], mount,
                scene["pitch"], scene["roll"],
            )
            full_query["run"]()
            sparse_query["run"]()
            torch.cuda.synchronize(device)
            full_dist = full_query["distances"].reshape(-1)
            sparse_dist = sparse_query["distances"].reshape(-1)
            selected_t = torch.as_tensor(selected_ids, device=device, dtype=torch.long)
            raw_delta = (full_dist.index_select(0, selected_t) - sparse_dist).abs()
            _check(bool(torch.equal(
                full_dist.index_select(0, selected_t), sparse_dist
            )), f"obstacle {name}: raw full/sparse distances are not bit-exact")

            # Fixed source grouping is retained even when an obstacle changes
            # the current hit point or endpoint.
            near_catalog = torch.as_tensor(catalog.near_mask, device=device)
            near_full = near_catalog
            near_sparse = near_catalog.index_select(0, selected_t)
            _check(torch.equal(near_full.index_select(0, selected_t), near_sparse),
                   f"obstacle {name}: fixed source grouping changed")

            analytic = scan_obstacles(catalog.directions_sensor, mount, scene)
            hit_labels = np.asarray(analytic["hit_label"])
            hit_ids = np.flatnonzero(hit_labels > 0)
            selected_hit_ids = hit_ids[selected_set[hit_ids]]
            full_points = mount_t[None, :] + full_dist[:, None] * dirs_base_t
            sparse_points = mount_t[None, :] + sparse_dist[:, None] * dirs_base_t.index_select(0, selected_t)

            def sampled_ids(candidate_ids: torch.Tensor, points: torch.Tensor,
                            near: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                all_ids, all_hits = [], []
                for mask, count in ((near, 256), (~near, 128)):
                    chosen = farthest_point_sampling_masked(
                        points.unsqueeze(0), mask.unsqueeze(0), count
                    )
                    actual_count = int(mask.sum().item())
                    slots = chosen[0, :actual_count]
                    picked = candidate_ids.index_select(0, slots)
                    all_ids.append(picked)
                    all_hits.append(torch.as_tensor(
                        hit_labels[picked.cpu().numpy()] > 0,
                        device=device, dtype=torch.bool,
                    ))
                return torch.cat(all_ids), torch.cat(all_hits)

            full_candidate_ids = torch.arange(len(rays), device=device, dtype=torch.long)
            sparse_candidate_ids = selected_t
            full_fps_ids, full_fps_hits = sampled_ids(
                full_candidate_ids, full_points, near_full
            )
            sparse_fps_ids, sparse_fps_hits = sampled_ids(
                sparse_candidate_ids, sparse_points, near_sparse
            )
            results[name] = {
                "full_raw_rays": int(len(rays)),
                "sparse_raw_rays": int(len(selected_ids)),
                "raw_selected_distance_max_abs_m": float(raw_delta.max().item()),
                "full_raw_obstacle_hit_rays": int(len(hit_ids)),
                "sparse_raw_obstacle_hit_rays": int(len(selected_hit_ids)),
                "omitted_full_visible_hit_rays": int(len(hit_ids) - len(selected_hit_ids)),
                "full_candidate_fps_count": int(full_fps_ids.numel()),
                "sparse_candidate_fps_count": int(sparse_fps_ids.numel()),
                "full_candidate_fps_obstacle_hits": int(full_fps_hits.sum().item()),
                "sparse_candidate_fps_obstacle_hits": int(sparse_fps_hits.sum().item()),
                "fps_source_id_sets_equal": torch.equal(
                    full_fps_ids.sort().values, sparse_fps_ids.sort().values
                ),
                "obstacle_objects": {
                    str(label): {
                        "full_raw_hits": int(np.count_nonzero(hit_labels == label)),
                        "sparse_raw_hits": int(np.count_nonzero(
                            hit_labels[selected_ids] == label
                        )),
                        "full_fps_hits": int(np.count_nonzero(
                            hit_labels[full_fps_ids.cpu().numpy()] == label
                        )),
                        "sparse_fps_hits": int(np.count_nonzero(
                            hit_labels[sparse_fps_ids.cpu().numpy()] == label
                        )),
                    }
                    for label in sorted(set(hit_labels.tolist()) - {-1, 0})
                },
            }
        return results
    finally:
        try:
            sys.path.remove(str(experiment_dir))
        except ValueError:
            pass


def _compare_frames(full, sparse, sparse_ids: torch.Tensor, catalog, label: str,
                    far_sampling="all_endpoints") -> dict:
    """Compare raw products on selected IDs and fixed catalog grouping."""
    full_pos = _indices_in(full, sparse_ids)
    sparse_pos = _indices_in(sparse, sparse_ids)
    checks = {}
    for name in (
        "raw_ray_dirs", "raw_dists", "no_hit_mask",
        "raw_points_sensor", "raw_points_base",
    ):
        a = getattr(full, name).index_select(0, full_pos.to(getattr(full, name).device))
        b = getattr(sparse, name).index_select(0, sparse_pos.to(getattr(sparse, name).device))
        checks[f"{label}_{name}"] = torch.equal(a.detach().cpu(), b.detach().cpu())
        _check(checks[f"{label}_{name}"], f"{label}: {name} differs on identical source IDs")
    _check(torch.equal(full.raw_source_ids.index_select(0, full_pos).cpu(), sparse_ids.cpu()),
           f"{label}: full source IDs do not map to catalog IDs")
    _check(torch.equal(sparse.raw_source_ids.index_select(0, sparse_pos).cpu(), sparse_ids.cpu()),
           f"{label}: sparse raw source IDs are not the selected global IDs")

    # Group masks are indexed by each pipeline's candidate pool. Resolve them
    # through pool_raw_indices so fixed near/far identity cannot depend on pose.
    for group in ("near", "far"):
        full_mask = getattr(full, group + "_mask")
        sparse_mask = getattr(sparse, group + "_mask")
        full_pool = full.pool_raw_indices
        sparse_pool = sparse.pool_raw_indices
        fpos = torch.searchsorted(full_pool, sparse_ids)
        spos = torch.searchsorted(sparse_pool, sparse_ids)
        _check(bool((full_pool.index_select(0, fpos) == sparse_ids).all()),
               f"{label}: full candidate pool misses selected IDs")
        _check(bool((sparse_pool.index_select(0, spos) == sparse_ids).all()),
               f"{label}: sparse candidate pool misses selected IDs")
        actual = sparse_mask.index_select(0, spos).cpu()
        expected_np = catalog.near_mask[sparse_ids.cpu().numpy()]
        expected = torch.as_tensor(expected_np, dtype=torch.bool)
        if group == "far":
            expected = ~expected
            if far_sampling == "hit_only_10m":
                expected &= (~sparse.no_hit_mask.index_select(0, sparse_pos).cpu()
                             & (sparse.raw_dists.index_select(0, sparse_pos).cpu() <= 10.0))
        checks[f"{label}_{group}_fixed_group"] = torch.equal(actual, expected)
        _check(checks[f"{label}_{group}_fixed_group"],
               f"{label}: {group} group changed from fixed catalog membership")
        _check(torch.equal(full_mask.index_select(0, fpos).cpu(), actual),
               f"{label}: full and sparse {group} masks differ on selected IDs")
    return checks


def _fps_comparison(full_pipeline, full_frames, sparse_frames, sparse_ids, label):
    """Run reference FPS on the full scan with the exact sparse source pool."""
    results = {}
    # Full mode has one local raw slot per global source ID. Build an isolated
    # candidate-pool view and gather every pool-aligned field together; in
    # particular, keep the full frame's near/far masks so a fired gap remains
    # excluded in the noisy reference path.
    slots = sparse_ids.to(full_pipeline.device)
    for row, (full_frame, sparse_frame) in enumerate(zip(full_frames, sparse_frames)):
        full_pool = full_frame.pool_raw_indices
        positions = torch.searchsorted(full_pool, slots)
        _check(bool((full_pool.index_select(0, positions) == slots).all()),
               f"{label}: full pool misses a sparse source ID")
        reference_frame = replace(
            full_frame,
            pool_raw_indices=full_frame.pool_raw_indices.index_select(0, positions),
            pool_points_base=full_frame.pool_points_base.index_select(0, positions),
            pool_dists=full_frame.pool_dists.index_select(0, positions),
            pool_angle_keys=full_frame.pool_angle_keys.index_select(0, positions),
            near_mask=full_frame.near_mask.index_select(0, positions),
            far_mask=full_frame.far_mask.index_select(0, positions),
        )
        full_pipeline._sample_frames([reference_frame])
        for kind in ("near", "far"):
            a = _frame_source_ids(reference_frame, kind).cpu()
            b = _frame_source_ids(sparse_frame, kind).cpu()
            ok = torch.equal(a, b)
            results[f"{label}_env{row}_{kind}_fps_source_ids"] = ok
            _check(ok, f"{label}: {kind} FPS global IDs differ")
            valid_n = int(getattr(sparse_frame, kind + "_valid").sum())
            _check(int(torch.unique(b).numel()) == valid_n,
                   f"{label}: repeated {kind} source IDs")
            _check(torch.equal(
                getattr(reference_frame, kind + "_points_base")[getattr(reference_frame, kind + "_valid")].cpu(),
                getattr(sparse_frame, kind + "_points_base")[getattr(sparse_frame, kind + "_valid")].cpu(),
            ), f"{label}: {kind} FPS point values differ")
            keys_name = kind + "_angle_keys"
            ak, bk = getattr(reference_frame, keys_name), getattr(sparse_frame, keys_name)
            if ak is not None or bk is not None:
                _check(ak is not None and bk is not None and _same_tensor(ak, bk),
                       f"{label}: {kind} FPS angle keys differ")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="new evidence directory")
    parser.add_argument("--selection", required=True,
                        help="versioned selection manifest JSON or NPZ")
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-envs", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--far-sampling", choices=("all_endpoints", "hit_only_10m"), default="all_endpoints")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    checks: dict = {"passed": False, "num_envs": args.num_envs, "seed": args.seed}
    started = time.monotonic()
    env = None
    full_runtime = sparse_runtime = None
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("real CUDA is required; this gate does not substitute CPU")
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
        from hexapod_mjlab.tasks.confined_locomotion_2.components.lidar.channels import (
            build_channel_catalog, load_sparse_selection,
        )
        from hexapod_mjlab.tasks.confined_locomotion_2.flat_env_cfg import make_training_env_cfg
        from hexapod_mjlab.tasks.confined_locomotion_2.components.lidar.scanner import (
            ConfinedLidarCfg, ConfinedScanPipeline,
        )
        from hexapod_mjlab.tasks.confined_locomotion_2.runtime import TrainingRuntime

        catalog = build_channel_catalog()
        selected_np, selection_meta = load_sparse_selection(args.selection, catalog)
        selected = torch.as_tensor(selected_np, device=device, dtype=torch.long)
        _check(np.all(selected_np[1:] > selected_np[:-1]), "selection source IDs are not sorted unique")
        checks["catalog_identity"] = catalog.identity
        checks["selection"] = selection_meta["selection"]
        checks["selected_source_ray_count"] = int(selected.numel())
        checks["gpu"] = torch.cuda.get_device_name(args.gpu)

        env = ManagerBasedRlEnv(
            make_training_env_cfg(num_envs=args.num_envs, seed=args.seed), device=str(device)
        )
        full_cfg = ConfinedLidarCfg(raycast_mode="full", far_sampling=args.far_sampling)
        sparse_cfg = ConfinedLidarCfg(
            raycast_mode="sparse", sparse_selection_path=str(Path(args.selection).resolve()),
            far_sampling=args.far_sampling,
        )
        full_pipeline = ConfinedScanPipeline(env, full_cfg)
        sparse_pipeline = ConfinedScanPipeline(env, sparse_cfg)
        _check(full_pipeline.num_raw_rays == 86400, "full scan actual ray count is not 86400")
        _check(sparse_pipeline.num_raw_rays == int(selected.numel()),
               "sparse actual ray count differs from selected source count")
        _check(full_pipeline.full_ray_dirs.shape == (86400, 3), "full direction catalog shape changed")
        _check(torch.equal(full_pipeline.source_ray_ids.cpu(), torch.arange(86400)),
               "full source IDs are not the global catalog IDs")
        _check(torch.equal(sparse_pipeline.source_ray_ids.cpu(), selected.cpu()),
               "sparse source IDs are not sorted selected global IDs")
        catalog_dirs = torch.as_tensor(catalog.directions_sensor, dtype=torch.float32)
        full_selected_dirs = full_pipeline.full_ray_dirs.index_select(0, selected).cpu()
        sparse_raw_dirs = sparse_pipeline._sensor.get_ray_directions().cpu()
        _check(_same_tensor(full_selected_dirs, catalog_dirs[selected.cpu()]),
               "CPU catalog and full GPU direction table differ on selected IDs")
        _check(_same_tensor(sparse_raw_dirs, catalog_dirs[selected.cpu()]),
               "CPU catalog and sparse GPU ray directions differ")
        checks["raw_ray_counts"] = {
            "full_actual": full_pipeline.num_raw_rays,
            "full_catalog": int(full_pipeline.full_ray_dirs.shape[0]),
            "sparse_actual": sparse_pipeline.num_raw_rays,
            "selection": int(selected.numel()),
        }

        rows = list(range(args.num_envs))
        def measure(pipeline):
            for _ in range(2):
                pipeline.scan_records.clear()
                warm_frames = pipeline.scan_rows(rows)
                del warm_frames
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            samples = []
            for _ in range(5):
                pipeline.scan_records.clear()
                t0 = time.perf_counter()
                measured_frames = pipeline.scan_rows(rows)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                samples.append(time.perf_counter() - t0)
                del measured_frames
                pipeline.scan_records.clear()
            peak = torch.cuda.max_memory_allocated(device) / 2**20
            return float(np.median(samples)), samples, peak

        full_s, full_samples, full_peak_mib = measure(full_pipeline)
        sparse_s, sparse_samples, sparse_peak_mib = measure(sparse_pipeline)
        # Retain one fresh pair for parity assertions, then clear each pipeline's
        # bounded audit so the 86,400-ray records do not accumulate.
        full_pipeline.scan_records.clear()
        sparse_pipeline.scan_records.clear()
        full_frames = full_pipeline.scan_rows(rows)
        sparse_frames = sparse_pipeline.scan_rows(rows)
        checks["scan_benchmark"] = {
            "num_envs": args.num_envs,
            "full_wall_s": full_s,
            "sparse_wall_s": sparse_s,
            "samples_per_mode": 5,
            "warmups_per_mode": 2,
            "full_samples_s": full_samples,
            "sparse_samples_s": sparse_samples,
            "full_rays_per_scan_batch": args.num_envs * full_pipeline.num_raw_rays,
            "sparse_rays_per_scan_batch": args.num_envs * sparse_pipeline.num_raw_rays,
            "full_peak_allocated_MiB": full_peak_mib,
            "sparse_peak_allocated_MiB": sparse_peak_mib,
        }
        checks["same_pose_raw_fixed_group"] = {}
        for row, (full, sparse) in enumerate(zip(full_frames, sparse_frames)):
            checks["same_pose_raw_fixed_group"].update(
                _compare_frames(full, sparse, selected, catalog, f"clean_env{row}", args.far_sampling)
            )
        checks["same_pool_fps"] = _fps_comparison(
            full_pipeline, full_frames, sparse_frames, selected, "clean"
        )
        full_pipeline.scan_records.clear()
        sparse_pipeline.scan_records.clear()

        # Recreate identical gap/range streams for each implementation. These
        # are consumption-only perturbations; the raw sensor return must stay
        # bit-identical, and consumed validity/range/FPS must follow source IDs.
        from hexapod_mjlab.common.perturbations import (
            GAP_FIXED_JITTER_OFF_PRESET, ObservationPerturbations,
            RangePerturbationCfg,
        )
        noise_cfg = type(GAP_FIXED_JITTER_OFF_PRESET)(
            gap=GAP_FIXED_JITTER_OFF_PRESET.gap,
            range=RangePerturbationCfg(),
            proprio=None,
        )
        # Fix a deterministic schedule phase whose first event is scan 2.
        # The noise comparison needs the same real begin_scan/scan_rows calls
        # on both paths, with the first eligible frame explicitly non-firing.
        noise_seed = args.seed + 17
        for _ in range(10000):
            full_perturb = ObservationPerturbations(
                1, cfg=noise_cfg, seed=noise_seed, device=device
            )
            phase = full_perturb.gap_schedule_record()[0]["phase"]
            if phase == 2:
                break
            noise_seed += 1
        else:
            raise RuntimeError("could not derive a fixed scan-2 gap schedule seed")
        sparse_perturb = ObservationPerturbations(
            1, cfg=noise_cfg, seed=noise_seed, device=device
        )
        checks["gap_range_noise_seed"] = noise_seed
        checks["gap_range_noise_phase"] = phase
        noise_rows = [0]
        full_gap_scan1 = full_pipeline.scan_rows(
            noise_rows, perturbations=full_perturb,
            persistent_scan_numbers={0: 1},
        )
        sparse_gap_scan1 = sparse_pipeline.scan_rows(
            noise_rows, perturbations=sparse_perturb,
            persistent_scan_numbers={0: 1},
        )
        _check(not any(frame.gap_fired for frame in full_gap_scan1),
               "the first scheduled gap scan must not fire")
        _check(not any(frame.gap_fired for frame in sparse_gap_scan1),
               "the first sparse scheduled gap scan must not fire")
        del full_gap_scan1, sparse_gap_scan1
        full_noisy = full_pipeline.scan_rows(
            noise_rows, perturbations=full_perturb,
            persistent_scan_numbers={0: 2},
        )
        sparse_noisy = sparse_pipeline.scan_rows(
            noise_rows, perturbations=sparse_perturb,
            persistent_scan_numbers={0: 2},
        )
        _check(all(frame.gap_fired for frame in full_noisy),
               "the second scheduled full scan must fire the fixed gap")
        _check(all(frame.gap_fired for frame in sparse_noisy),
               "the second scheduled sparse scan must fire the fixed gap")
        _check(all(int((~frame.consumed_valid).sum()) > 0 for frame in full_noisy),
               "the fired full-frame gap removed no rays")
        _check(all(int((~frame.consumed_valid).sum()) > 0 for frame in sparse_noisy),
               "the fired sparse-frame gap removed no rays")
        checks["gap_range_noise_by_source"] = {}
        for row, (full, sparse) in enumerate(zip(full_noisy, sparse_noisy)):
            fpos = _indices_in(full, selected)
            spos = _indices_in(sparse, selected)
            for name in (
                "raw_ray_dirs", "raw_dists", "no_hit_mask", "raw_points_sensor",
                "raw_points_base", "consumed_valid", "consumed_dists", "range_epsilon",
            ):
                a, b = getattr(full, name), getattr(sparse, name)
                if a is None or b is None:
                    _check(a is None and b is None, f"noise env{row}: {name} presence differs")
                    continue
                a = a.index_select(0, fpos.to(a.device))
                b = b.index_select(0, spos.to(b.device))
                ok = torch.equal(a.detach().cpu(), b.detach().cpu())
                checks["gap_range_noise_by_source"][f"env{row}_{name}"] = ok
                _check(ok, f"noise env{row}: {name} differs for same source IDs")
            invalid_gap = ~sparse.consumed_valid
            checks["gap_range_noise_by_source"][f"env{row}_gap_excluded_count"] = int(invalid_gap.sum())
            selected_source_ids = _frame_source_ids(sparse, "near")
            far_source_ids = _frame_source_ids(sparse, "far")
            sampled = torch.cat((selected_source_ids, far_source_ids))
            sampled_pos = _indices_in(sparse, sampled)
            _check(bool(sparse.consumed_valid.index_select(0, sampled_pos).all()),
                   f"noise env{row}: FPS selected a gap-excluded ray")
            checks["gap_range_noise_by_source"].update(
                _fps_comparison(
                    full_pipeline, [full], [sparse], selected,
                    f"noise_env{row}",
                )
            )
        full_pipeline.scan_records.clear()
        sparse_pipeline.scan_records.clear()

        # Change pose without stepping physics: scanner grouping belongs to
        # source IDs, while actual world intersections and points can change.
        # EntityData's root_link_pose_w concatenates xpos/xquat into a copy.
        # This kinematic fixture must update the backing tensors the scanner
        # actually reads, without pretending a physics transition took place.
        robot_data = env.scene["robot"].data
        root_pos = robot_data.data.xpos[:, robot_data.indexing.root_body_id]
        root_quat = robot_data.data.xquat[:, robot_data.indexing.root_body_id]
        before_pos, before_quat = root_pos.clone(), root_quat.clone()
        pose_cases = (
            (5.0, 0.0, 0.0),
            (-5.0, 0.0, 0.04),
            (0.0, 5.0, 0.08),
            (0.0, -5.0, 0.0),
        )
        checks["pose_change_group_identity_stable"] = {}
        try:
            for roll_deg, pitch_deg, height_m in pose_cases:
                _set_pose(root_pos, root_quat, before_pos, before_quat,
                          roll_deg=roll_deg, pitch_deg=pitch_deg, height_m=height_m)
                full_pipeline.scan_records.clear()
                sparse_pipeline.scan_records.clear()
                shifted_full = full_pipeline.scan_rows(rows)
                shifted_sparse = sparse_pipeline.scan_rows(rows)
                pose_label = f"roll{roll_deg:g}_pitch{pitch_deg:g}_height{height_m:g}"
                pose_checks = {}
                for row, (full, sparse) in enumerate(zip(shifted_full, shifted_sparse)):
                    _check(torch.equal(full.base_pos_w, root_pos[row]),
                           f"{pose_label}: scanner did not consume the requested fixture position")
                    _check(torch.equal(full.base_quat_wxyz, root_quat[row]),
                           f"{pose_label}: scanner did not consume the requested fixture quaternion")
                    pose_checks.update(_compare_frames(
                        full, sparse, selected, catalog, f"{pose_label}_env{row}", args.far_sampling
                    ))
                    pose_checks.update(_fps_comparison(
                        full_pipeline, [full], [sparse], selected,
                        f"{pose_label}_env{row}",
                    ))
                checks["pose_change_group_identity_stable"][pose_label] = pose_checks
                del shifted_full, shifted_sparse
        finally:
            root_pos.copy_(before_pos)
            root_quat.copy_(before_quat)
            full_pipeline.scan_records.clear()
            sparse_pipeline.scan_records.clear()

        checks["obstacle_warp_comparison"] = _run_obstacle_raycasts(
            torch, __import__("warp"), device, catalog, selected_np
        )

        # Runtime checkpoint state is tied to the resolved raycast/selection
        # identity. Reusing it under a different sparse selection must fail.
        wrapper = RslRlVecEnvWrapper(env, clip_actions=None)
        full_runtime = TrainingRuntime(
            wrapper, seed=args.seed, output_dir=output / "runtime_full",
            mode="training_lifecycle_fixture", lidar_cfg=full_cfg,
        )
        state = full_runtime.state_dict()
        full_runtime.close()
        full_runtime = None
        sparse_runtime = TrainingRuntime(
            wrapper, seed=args.seed, output_dir=output / "runtime_sparse",
            mode="training_lifecycle_fixture", lidar_cfg=sparse_cfg,
        )
        try:
            sparse_runtime.load_state_dict(state)
        except RuntimeError as exc:
            checks["different_scan_identity_restore_rejected"] = "identity" in str(exc).lower()
        else:
            checks["different_scan_identity_restore_rejected"] = False
        _check(checks["different_scan_identity_restore_rejected"],
               "runtime accepted state from a different raycast identity")

        checks["passed"] = True
    except BaseException as error:
        checks["passed"] = False
        checks["error"] = repr(error)
        checks["traceback"] = traceback.format_exc()
    finally:
        if sparse_runtime is not None:
            try:
                sparse_runtime.close()
            except BaseException:
                pass
        if full_runtime is not None:
            try:
                full_runtime.close()
            except BaseException:
                pass
        if env is not None:
            try:
                env.close()
            except BaseException:
                pass
        checks["elapsed_s"] = time.monotonic() - started
        if torch.cuda.is_available():
            checks.setdefault("gpu", torch.cuda.get_device_name(args.gpu))
            checks["peak_allocated_MiB"] = torch.cuda.max_memory_allocated(args.gpu) / 2**20
        (output / "checks.json").write_text(json.dumps(checks, indent=2, ensure_ascii=False))
        print(json.dumps(checks, indent=2, ensure_ascii=False))
    if not checks["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
