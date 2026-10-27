"""Finite real-CUDA acceptance check for the fused point-sampling path.

The script does not train. It compares CUDA sampling against the pure reference
on synthetic edge cases and real runtime scan frames, then times one optimized
and one temporary rowwise-reference runtime scan per preset.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time
import traceback

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _assert(condition, message):
    if not condition:
        raise AssertionError(message)


def _as_indices(value):
    if torch.is_tensor(value):
        return value.detach().to(device="cpu", dtype=torch.long).reshape(-1)
    return torch.as_tensor(value, dtype=torch.long).reshape(-1)


def _reference_one(reference, points, mask, k):
    cpu_points = points.detach().to(device="cpu", dtype=torch.float32).contiguous()
    cpu_mask = mask.detach().to(device="cpu", dtype=torch.bool).contiguous()
    indices, count = reference(cpu_points, cpu_mask, int(k))
    return _as_indices(indices), int(count)


def _reference_masked(reference, points, masks, k, *, reference_device, output_device):
    """Fixed (B,k) result, rowwise pure-reference sampler, padded with -1."""
    reference_points = points.detach().to(
        device=reference_device, dtype=torch.float32
    ).contiguous()
    reference_masks = masks.detach().to(
        device=reference_device, dtype=torch.bool
    ).contiguous()
    batch, _count, _xyz = reference_points.shape
    result = torch.full((batch, int(k)), -1, dtype=torch.long)
    for row in range(batch):
        indices, count = reference(
            reference_points[row], reference_masks[row], int(k)
        )
        indices = _as_indices(indices)
        if count:
            result[row, :int(count)] = indices[:int(count)]
    return result.to(device=output_device)


def _compare(name, actual, expected):
    actual_cpu = actual.detach().cpu() if torch.is_tensor(actual) else torch.as_tensor(actual)
    expected_cpu = expected.detach().cpu() if torch.is_tensor(expected) else torch.as_tensor(expected)
    _assert(actual_cpu.dtype == expected_cpu.dtype, f"{name}: dtype {actual_cpu.dtype} != {expected_cpu.dtype}")
    _assert(tuple(actual_cpu.shape) == tuple(expected_cpu.shape), f"{name}: shape {tuple(actual_cpu.shape)} != {tuple(expected_cpu.shape)}")
    _assert(torch.equal(actual_cpu, expected_cpu), f"{name}: tensor values differ")


def _time_cuda(fn, *, repeats=3):
    # Warmup includes any first-call CUDA/Warp compilation; it is not timed.
    fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(int(repeats)):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / max(int(repeats), 1)


def _time_wall(fn):
    started = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - started


def _synthetic_cases(sampling, device, seed):
    reference = sampling.farthest_point_sampling_indices_reference
    fps_indices = sampling.farthest_point_sampling_indices
    fps_masked = sampling.farthest_point_sampling_masked
    fps_padded = sampling.farthest_point_sampling_padded
    generator = torch.Generator(device="cpu").manual_seed(int(seed))

    n = 192
    random_points = torch.randn((n, 3), generator=generator, dtype=torch.float32)
    repeated_points = torch.zeros((n, 3), dtype=torch.float32)
    repeated_points[::3] = torch.tensor([0.25, 0.0, 0.0])
    symmetric_points = torch.zeros((n, 3), dtype=torch.float32)
    basis = torch.tensor(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
         [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
         [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]], dtype=torch.float32
    )
    symmetric_points[:basis.shape[0]] = basis
    symmetric_points[basis.shape[0]:] = basis.repeat((n - basis.shape[0] + 5) // 6, 1)[:n - basis.shape[0]]
    random_mask = torch.rand((n,), generator=generator) > 0.23
    shortage_mask = torch.zeros(n, dtype=torch.bool)
    shortage_mask[[3, 17, 31, 59, 100]] = True
    empty_mask = torch.zeros(n, dtype=torch.bool)
    cases = (
        ("random_mask", random_points, random_mask, 48),
        ("duplicate_ties", repeated_points, torch.ones(n, dtype=torch.bool), 64),
        ("equidistant_ties", symmetric_points, torch.ones(n, dtype=torch.bool), 32),
        ("empty", random_points, empty_mask, 24),
        ("shortage", random_points, shortage_mask, 16),
    )
    case_checks = {}
    for name, points_cpu, mask_cpu, k in cases:
        points_gpu = points_cpu.to(device=device)
        mask_gpu = mask_cpu.to(device=device)
        expected, expected_count = reference(points_gpu, mask_gpu, int(k))
        cpu_expected, cpu_expected_count = reference(points_cpu, mask_cpu, int(k))
        actual, actual_count = fps_indices(points_gpu, mask_gpu, int(k))
        _compare(f"indices/{name}", _as_indices(actual), _as_indices(expected))
        _assert(int(actual_count) == int(expected_count), f"indices/{name}: count mismatch")
        _assert(int(cpu_expected_count) == int(expected_count), f"indices/{name}: CPU reference count mismatch")
        _assert(int(actual_count) == min(int(mask_cpu.sum()), int(k)), f"indices/{name}: unexpected valid count")
        case_checks[name] = {
            "count": int(actual_count),
            "cpu_vs_cuda_reference_slot_disagreements": int(
                (_as_indices(cpu_expected) != _as_indices(expected)).sum()
            ),
        }

    # Arbitrary masks and a variable number of members in one real CUDA batch.
    batch = torch.stack((random_points, repeated_points, symmetric_points, random_points[:n], random_points, random_points))
    masks = torch.stack((random_mask, torch.ones(n, dtype=torch.bool),
                         torch.ones(n, dtype=torch.bool), empty_mask,
                         shortage_mask, random_mask & (torch.arange(n) % 2 == 0)))
    masked_k = 40
    expected_masked = _reference_masked(
        reference, batch, masks, masked_k,
        reference_device=device, output_device=device,
    )
    actual_masked = fps_masked(batch.to(device), masks.to(device), masked_k)
    _compare("masked/mixed_arbitrary_masks", actual_masked, expected_masked)
    cpu_masked = _reference_masked(
        reference, batch, masks, masked_k,
        reference_device="cpu", output_device="cpu",
    )
    case_checks["masked_mixed_rows"] = {
        "rows": int(batch.shape[0]),
        "cpu_vs_cuda_reference_slot_disagreements": int(
            (cpu_masked != expected_masked.cpu()).sum()
        ),
    }

    # Lengths exercise empty, singleton, shortage, exact-k, and long rows.
    padded_n, padded_k = 96, 40
    padded_points = torch.randn((6, padded_n, 3), generator=generator, dtype=torch.float32)
    padded_points[1].zero_()
    padded_points[2].zero_()
    padded_points[2, :basis.shape[0]] = basis
    padded_points[2, basis.shape[0]:] = basis.repeat((padded_n - basis.shape[0] + 5) // 6, 1)[:padded_n - basis.shape[0]]
    lengths_cpu = torch.tensor([0, 1, 7, 39, 40, 96], dtype=torch.long)
    expected_padded = torch.full((6, padded_k), -1, dtype=torch.long, device=device)
    expected_padded_cpu = torch.full((6, padded_k), -1, dtype=torch.long)
    padded_points_gpu = padded_points.to(device)
    for row, length in enumerate(lengths_cpu.tolist()):
        mask = torch.arange(padded_n) < int(length)
        ids, count = reference(
            padded_points_gpu[row], mask.to(device), padded_k
        )
        ids = _as_indices(ids)
        if count:
            expected_padded[row, :int(count)] = ids[:int(count)]
        ids_cpu, count_cpu = reference(padded_points[row], mask, padded_k)
        ids_cpu = _as_indices(ids_cpu)
        if count_cpu:
            expected_padded_cpu[row, :int(count_cpu)] = ids_cpu[:int(count_cpu)]
    expected_padded = expected_padded.cpu()
    actual_padded = fps_padded(
        padded_points.to(device), lengths_cpu.to(device=device), padded_k
    )
    _compare("padded/mixed_lengths", actual_padded, expected_padded)
    _assert(torch.all(actual_padded[0] == -1), "padded empty row is not all -1")
    _assert(torch.all(actual_padded[1, 1:] == -1), "padded singleton tail is not -1")
    _assert(torch.all(actual_padded[2, 7:] == -1), "padded shortage tail is not -1")
    case_checks["padded_lengths"] = {
        "lengths": lengths_cpu.tolist(),
        "cpu_vs_cuda_reference_slot_disagreements": int(
            (expected_padded_cpu != expected_padded).sum()
        ),
    }

    # Finite isolated GPU timing on a representative 86,400-candidate pool.
    # The actual runtime pass below supplies the real masks and points.
    perf_n, perf_k, perf_batch = 8192, 128, 2
    perf_points = torch.randn((perf_batch, perf_n, 3), generator=generator, dtype=torch.float32).to(device)
    perf_masks = (torch.rand((perf_batch, perf_n), generator=generator) > 0.3).to(device)
    gpu_ms = _time_cuda(lambda: fps_masked(perf_points, perf_masks, perf_k), repeats=3)
    same_gpu_result = _reference_masked(
        reference, perf_points, perf_masks, perf_k,
        reference_device=device, output_device=device,
    )
    _compare("timing/synthetic_masked", fps_masked(perf_points, perf_masks, perf_k), same_gpu_result)
    cpu_result, cpu_s = _time_wall(lambda: _reference_masked(
        reference, perf_points, perf_masks, perf_k,
        reference_device="cpu", output_device="cpu"
    ))
    cpu_gpu_ref_differences = int((cpu_result != same_gpu_result.cpu()).sum())
    return {
        "cases": case_checks,
        "synthetic_masked_gpu_ms": gpu_ms,
        "synthetic_masked_cpu_reference_s": cpu_s,
        "synthetic_masked_cpu_vs_cuda_reference_slot_disagreements": cpu_gpu_ref_differences,
        "synthetic_shape": [perf_batch, perf_n, 3],
        "synthetic_k": perf_k,
    }


def _assert_frame_selection(frame, reference, pipeline, label, *, cpu_reference=None):
    raw_pool = frame.pool_raw_indices.detach().cpu().to(torch.long)
    _compare(f"{label}/pool_raw_ids", raw_pool, pipeline.pool_indices.detach().cpu().to(torch.long))
    pool_points = frame.pool_points_base
    # The pool consumes P6-noisy points; raw_points_base is unperturbed audit
    # geometry. Both use the same acquisition-time sensor-to-BASE transform.
    from hexapod_mjlab.tasks.confined_locomotion_2.scanner import _quat_apply_xyzw
    consumed_base = _quat_apply_xyzw(
        pipeline._mount_q.expand(frame.points_sensor.shape[0], 4), frame.points_sensor
    ) + pipeline._mount_t.expand(frame.points_sensor.shape[0], 3)
    expected_pool_points = consumed_base.index_select(0, frame.pool_raw_indices)
    _compare(f"{label}/pool_current_frame_xyz", pool_points, expected_pool_points)
    near_mask = frame.near_mask
    far_mask = frame.far_mask
    if frame.consumed_valid is None:
        pool_valid = torch.ones_like(near_mask)
    else:
        pool_valid = frame.consumed_valid.index_select(0, frame.pool_raw_indices)
    _assert(not bool((near_mask & far_mask).any()), f"{label}: near/far masks overlap")
    _compare(f"{label}/near_far_exhaustive", near_mask | far_mask, pool_valid)

    expected_counts = {}
    for kind, mask, k in (("near", near_mask, 256), ("far", far_mask, 128)):
        expected, count = reference(pool_points.contiguous(), mask.contiguous(), int(k))
        expected = _as_indices(expected)
        expected_raw = raw_pool.index_select(0, expected[:int(count)]) if count else torch.empty(0, dtype=torch.long)
        ids = getattr(frame, f"{kind}_indices").detach().cpu().to(torch.long)
        valid = getattr(frame, f"{kind}_valid").detach().cpu().to(torch.bool)
        xyz = getattr(frame, f"{kind}_points_base")
        _assert(int(valid.sum()) == int(count), f"{label}/{kind}: valid count {int(valid.sum())} != reference {count}")
        _compare(f"{label}/{kind}/source_raw_ids", ids[:int(count)], expected_raw)
        _assert(torch.all(ids[int(count):] == -1), f"{label}/{kind}: padding source id is not -1")
        _assert(not bool(valid[int(count):].any()), f"{label}/{kind}: padding marked valid")
        if int(count):
            expected_xyz = pool_points.index_select(0, expected[:int(count)].to(pool_points.device))
            _compare(f"{label}/{kind}/xyz", xyz[:int(count)], expected_xyz)
            # IDs, not XYZ, identify the original no-hit rays.
            no_hit = frame.no_hit_mask.index_select(0, expected_raw.to(frame.no_hit_mask.device))
            source_no_hit = frame.no_hit_mask.index_select(0, ids[:int(count)].to(frame.no_hit_mask.device))
            _compare(f"{label}/{kind}/no_hit_source_ids", source_no_hit, no_hit)
            raw_dist = frame.raw_dists.index_select(
                0, ids[:int(count)].to(frame.raw_dists.device)
            )
            _compare(
                f"{label}/{kind}/raw_no_hit_by_source_id",
                raw_dist == float(pipeline.cfg.max_range_m),
                source_no_hit,
            )
        _assert(torch.all(xyz[int(count):] == 0), f"{label}/{kind}: padded xyz is not zero")
        expected_counts[kind] = int(count)
        if cpu_reference is not None:
            cpu_expected, cpu_count = cpu_reference(
                pool_points.detach().to(device="cpu", dtype=torch.float32).contiguous(),
                mask.detach().to(device="cpu", dtype=torch.bool).contiguous(),
                int(k),
            )
            cpu_expected = _as_indices(cpu_expected)
            _assert(int(cpu_count) == int(count), f"{label}/{kind}: CPU reference count differs")
            expected_counts[f"{kind}_cpu_reference_disagreements"] = int(
                (cpu_expected[:int(count)] != expected[:int(count)].cpu()).sum()
            )
    return expected_counts


def _assert_frames_match(label, gpu_frames, ref_frames):
    _assert(len(gpu_frames) == len(ref_frames), f"{label}: runtime scan batch size changed")
    for gpu, ref in zip(gpu_frames, ref_frames):
        _assert(int(gpu.env_row) == int(ref.env_row), f"{label}: env-row order changed")
        for name in ("scan_id", "episode_id", "time_s", "gap_fired"):
            _assert(getattr(gpu, name) == getattr(ref, name),
                    f"{label}/env{gpu.env_row}: {name} differs")
        for name in (
            "pool_raw_indices", "pool_points_base", "near_mask", "far_mask",
            "near_indices", "near_valid", "near_points_base",
            "far_indices", "far_valid", "far_points_base", "no_hit_mask",
        ):
            _compare(f"{label}/frame{gpu.env_row}/{name}", getattr(gpu, name), getattr(ref, name))


def _latest_frames(pipeline, rows):
    latest = {}
    for record in pipeline.scan_records:
        frame = record["frame"]
        row = int(record["env_row"])
        if row in rows:
            if row not in latest or (frame.episode_id, frame.scan_id) >= (
                latest[row].episode_id, latest[row].scan_id
            ):
                latest[row] = frame
    _assert(set(latest) == set(rows), f"missing current scan frames for rows {sorted(set(rows) - set(latest))}")
    return [latest[row] for row in rows]


def _runtime_scan_once(runtime, scanner_module, rows, state, record_list,
                       *, reference_masked=None, reference_fn=None):
    """Re-run one real runtime scan from a restored idle boundary."""
    runtime.load_state_dict(copy.deepcopy(state))
    runtime.pipeline.scan_records = list(record_list)
    original_sampler = scanner_module.farthest_point_sampling_masked
    reference_ms = [0.0]

    def rowwise_reference_masked(points, masks, k):
        started = time.perf_counter()
        result = _reference_masked(
            reference_fn,
            points,
            masks,
            int(k),
            reference_device=points.device,
            output_device=points.device,
        )
        reference_ms[0] += (time.perf_counter() - started) * 1000.0
        return result

    if reference_masked is not None:
        scanner_module.farthest_point_sampling_masked = rowwise_reference_masked
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        writes = runtime._scan_rows_now(rows)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_s = time.perf_counter() - started
        write_keys = {(int(item["env_row"]), int(item["scan_id"])) for item in writes}
        frames = [
            record["frame"] for record in runtime.pipeline.scan_records
            if (int(record["env_row"]), int(record["frame"].scan_id)) in write_keys
        ]
        frames.sort(key=lambda frame: int(frame.env_row))
        _assert(len(frames) == len(rows), "runtime scan did not retain one frame per due row")
        return frames, writes, elapsed_s, reference_ms[0]
    finally:
        scanner_module.farthest_point_sampling_masked = original_sampler


def _check_runtime_mode(name, preset, env_factory, wrapper_factory, make_cfg,
                        runtime_type, sampling, scanner_module, seed, device,
                        num_envs, output_dir):
    env = wrapper = runtime = None
    try:
        cfg = make_cfg(num_envs=int(num_envs), seed=int(seed))
        env = env_factory(cfg, device=device)
        wrapper = wrapper_factory(env)
        runtime = runtime_type(
            wrapper,
            seed=int(seed),
            output_dir=Path(output_dir) / name,
            mode="engineering",
            perturbation_cfg=preset,
        )
        rows = list(range(int(num_envs)))

        # The constructor's first scan is stamped at t=0. Advance one real,
        # zero-action control step before any forced scan so the scan-window
        # time-order contract permits a subsequent write at t=0.02. The
        # confirmed cadence is five control steps, so this boundary must not
        # perform a natural scan; both later replays start from this same
        # post-step physics/runtime boundary without stepping again.
        from hexapod_mjlab.rl.confined_locomotion_2.contracts import JOINT_DIM

        transition = runtime.step(torch.zeros(
            (int(num_envs), JOINT_DIM), dtype=torch.float32, device=runtime.device
        ))
        _assert(runtime.control_steps_taken == 1, "runtime did not advance exactly one control step")
        _assert(not transition.scan_writes, "unexpected natural scan on the first control step")
        _assert(not runtime._stopped, "zero-action timing step stopped the runtime")

        forced_gap_frame = False
        timing_boundary = None
        timing_records = None
        if name == "gap_fixed0":
            perturb = runtime._perturbations
            _assert(perturb is not None and perturb.gap_enabled, "gap_fixed0 did not enable gap schedule")
            # Deterministic test fixture: scan 1 has already completed. Anchor
            # the fixed-ε=0 event at persistent scan 2 so this check exercises
            # actual gap-masked sampling without depending on a random phase.
            for state in perturb._gap_rows:
                _assert(state is not None, "gap schedule row is missing")
                state.phase = 2
                state.next_event_index = 0
                state.scheduled_epsilon = None
                state.events.clear()
            timing_boundary = copy.deepcopy(runtime.state_dict())
            timing_records = list(runtime.pipeline.scan_records)
            runtime._scan_rows_now(rows)
            forced_gap_frame = True

        frames = _latest_frames(runtime.pipeline, rows)
        cpu_reference_differences = []
        for frame in frames:
            counts = _assert_frame_selection(
                frame,
                sampling.farthest_point_sampling_indices_reference,
                runtime.pipeline,
                f"real/{name}/env{frame.env_row}/scan{frame.scan_id}",
                cpu_reference=sampling.farthest_point_sampling_indices_reference,
            )
            cpu_reference_differences.append({
                "env_row": int(frame.env_row),
                "scan_id": int(frame.scan_id),
                "near_slots": counts.get("near_cpu_reference_disagreements", 0),
                "far_slots": counts.get("far_cpu_reference_disagreements", 0),
            })
            _assert(int(frame.near_points_base.shape[0]) == 256, f"real/{name}: near output shape changed")
            _assert(int(frame.far_points_base.shape[0]) == 128, f"real/{name}: far output shape changed")
            if name == "gap_fixed0":
                _assert(frame.gap_fired, "gap_fixed0 fixture did not fire its fixed-zero gap")

        # Two-phase runtime scan correctness and finite timing. Both runs start
        # at the same boundary, sensor pose, perturbation streams and persistent
        # scan number. The rowwise reference patch is temporary and restored.
        boundary = copy.deepcopy(runtime.state_dict())
        original_records = list(runtime.pipeline.scan_records)
        if name == "gap_fixed0":
            # Replay the same fixed-zero scan-2 event from its pre-scan state.
            boundary = timing_boundary
            original_records = timing_records
        gpu_frames, gpu_writes, gpu_scan_s, _ = _runtime_scan_once(
            runtime, scanner_module, rows, boundary, original_records
        )
        ref_frames, ref_writes, ref_scan_s, ref_sampler_ms = _runtime_scan_once(
            runtime,
            scanner_module,
            rows,
            boundary,
            original_records,
            reference_masked=True,
            reference_fn=sampling.farthest_point_sampling_indices_reference,
        )
        _assert([(w["env_row"], w["near_valid_count"], w["far_valid_count"], w["gap_fired"])
                 for w in gpu_writes] ==
                [(w["env_row"], w["near_valid_count"], w["far_valid_count"], w["gap_fired"])
                for w in ref_writes], f"real/{name}: GPU/reference scan write metadata differs")
        _assert_frames_match(
            f"runtime_scan/{name}", gpu_frames, ref_frames,
        )

        # Isolated production masked-FPS timing on actual current-frame pools.
        # Keep both group calls in one small batch, as production does.
        pool_points = torch.stack([frame.pool_points_base for frame in frames])
        near_masks = torch.stack([frame.near_mask for frame in frames])
        far_masks = torch.stack([frame.far_mask for frame in frames])
        near_gpu_ms = _time_cuda(lambda: sampling.farthest_point_sampling_masked(
            pool_points, near_masks, 256
        ), repeats=3)
        far_gpu_ms = _time_cuda(lambda: sampling.farthest_point_sampling_masked(
            pool_points, far_masks, 128
        ), repeats=3)
        near_cpu_ref, near_cpu_ref_s = _time_wall(lambda: _reference_masked(
            sampling.farthest_point_sampling_indices_reference,
            pool_points, near_masks, 256,
            reference_device="cpu", output_device="cpu",
        ))
        far_cpu_ref, far_cpu_ref_s = _time_wall(lambda: _reference_masked(
            sampling.farthest_point_sampling_indices_reference,
            pool_points, far_masks, 128,
            reference_device="cpu", output_device="cpu",
        ))
        near_cuda_ref, near_cuda_ref_s = _time_wall(lambda: _reference_masked(
            sampling.farthest_point_sampling_indices_reference,
            pool_points, near_masks, 256,
            reference_device=pool_points.device, output_device=pool_points.device,
        ))
        far_cuda_ref, far_cuda_ref_s = _time_wall(lambda: _reference_masked(
            sampling.farthest_point_sampling_indices_reference,
            pool_points, far_masks, 128,
            reference_device=pool_points.device, output_device=pool_points.device,
        ))
        _compare("timing/real_near_same_gpu_reference", sampling.farthest_point_sampling_masked(
            pool_points, near_masks, 256
        ), near_cuda_ref)
        _compare("timing/real_far_same_gpu_reference", sampling.farthest_point_sampling_masked(
            pool_points, far_masks, 128
        ), far_cuda_ref)
        return {
            "preset": name,
            "env_rows": rows,
            "current_scan_ids": [int(frame.scan_id) for frame in frames],
            "current_episode_ids": [int(frame.episode_id) for frame in frames],
            "selected_counts": [
                {"near": int(frame.near_valid.sum()), "far": int(frame.far_valid.sum())}
                for frame in frames
            ],
            "gap_fired_in_correctness_frame": [bool(frame.gap_fired) for frame in frames],
            "cpu_vs_cuda_reference_slot_disagreements": cpu_reference_differences,
            "forced_gap_fixed0_fixture": forced_gap_frame,
            "gpu_runtime_scan_ms": gpu_scan_s * 1000.0,
            "same_gpu_reference_runtime_scan_ms": ref_scan_s * 1000.0,
            "same_gpu_reference_sampler_subtotal_ms": ref_sampler_ms,
            "real_pool_near_masked_fps_gpu_ms": near_gpu_ms,
            "real_pool_far_masked_fps_gpu_ms": far_gpu_ms,
            "real_pool_near_cpu_reference_s": near_cpu_ref_s,
            "real_pool_far_cpu_reference_s": far_cpu_ref_s,
            "real_pool_near_same_gpu_reference_s": near_cuda_ref_s,
            "real_pool_far_same_gpu_reference_s": far_cuda_ref_s,
            "real_pool_near_cpu_vs_cuda_reference_slot_disagreements": int(
                (near_cpu_ref != near_cuda_ref.cpu()).sum()
            ),
            "real_pool_far_cpu_vs_cuda_reference_slot_disagreements": int(
                (far_cpu_ref != far_cuda_ref.cpu()).sum()
            ),
            "runtime_gpu_vs_same_gpu_reference_indices_equal": True,
        }
    finally:
        if runtime is not None:
            runtime.close()
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="fresh output directory")
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=1, choices=(1, 2))
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    checks = {"passed": False, "seed": args.seed, "num_envs": args.num_envs}
    started = time.monotonic()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("real CUDA is required; this checker does not substitute CPU execution")
        torch.cuda.set_device(args.gpu)
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
        from hexapod_mjlab.tasks.confined_locomotion_2.flat_env_cfg import make_training_env_cfg
        from hexapod_mjlab.tasks.confined_locomotion_2.runtime import TrainingRuntime
        from hexapod_mjlab.tasks.confined_locomotion_2 import scanner, sampling
        from hexapod_mjlab.common.perturbations import preset_by_name

        required = (
            "farthest_point_sampling_indices",
            "farthest_point_sampling_indices_reference",
            "farthest_point_sampling_masked",
            "farthest_point_sampling_padded",
        )
        missing = [name for name in required if not callable(getattr(sampling, name, None))]
        _assert(not missing, f"sampling API is incomplete: {missing}")
        _assert(callable(getattr(scanner, "farthest_point_sampling_masked", None)),
                "scanner does not import the production masked FPS entry point")
        device = torch.device(f"cuda:{args.gpu}")
        checks["gpu"] = torch.cuda.get_device_name(args.gpu)
        checks["synthetic"] = _synthetic_cases(sampling, device, args.seed)

        def env_factory(cfg, *, device):
            return ManagerBasedRlEnv(cfg, device=str(device))

        def wrapper_factory(env):
            return RslRlVecEnvWrapper(env, clip_actions=None)

        runtime_modes = {}
        for preset_name in ("clean", "all", "gap_fixed0"):
            runtime_modes[preset_name] = _check_runtime_mode(
                preset_name,
                preset_by_name(preset_name),
                env_factory,
                wrapper_factory,
                make_training_env_cfg,
                TrainingRuntime,
                sampling,
                scanner,
                args.seed,
                device,
                args.num_envs,
                output,
            )
        checks["runtime_modes"] = runtime_modes
        checks["passed"] = all(
            value.get("runtime_gpu_vs_same_gpu_reference_indices_equal", False)
            for value in runtime_modes.values()
        ) and bool(checks["synthetic"]["cases"])
    except BaseException as error:
        checks["passed"] = False
        checks["error"] = repr(error)
        checks["traceback"] = traceback.format_exc()
    finally:
        checks["elapsed_s"] = time.monotonic() - started
        if torch.cuda.is_available():
            checks["gpu"] = torch.cuda.get_device_name(args.gpu)
            checks["peak_allocated_MiB"] = torch.cuda.max_memory_allocated(args.gpu) / 2**20
        (output / "checks.json").write_text(json.dumps(checks, indent=2, ensure_ascii=False, default=str))
        print(json.dumps(checks, indent=2, ensure_ascii=False, default=str))
    if not checks["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
