#!/usr/bin/env python3
"""Independent CUDA Warp raycast benchmark for frozen _2 sparse Airy channels.

Uses the existing production Warp ray-query kernel with the SAME mesh and
original sensor directions for full and sparse. Tests actual raycast reduction
and per-selected-ray equality; DOES NOT modify production scanner / FPS.
Not a full training step throughput benchmark.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from select_channels import MAX_RANGE_M, airy_directions, nominal_mount, sha256
from compare_sampling import intersect_scene
from evaluate_moving_footprint import read_frozen_design


def build_geometry(wp, device: str, width: float | None):
    """A diagnostic two-triangle floor + finite two-triangle wall per side."""
    vertices = []
    triangles = []

    def quad(a, b, c, d):
        k = len(vertices)
        vertices.extend((a, b, c, d))
        triangles.extend(((k, k + 1, k + 2), (k, k + 2, k + 3)))

    z0 = 0.0
    quad((-32, -32, z0), (32, -32, z0), (32, 32, z0), (-32, 32, z0))
    if width is not None:
        for y in (-width / 2, width / 2):
            quad((-2, y, 0), (6, y, 0), (6, y, 4), (-2, y, 4))

    v = wp.array(np.array(vertices, dtype=np.float32), dtype=wp.vec3, device=device)
    t = wp.array(np.array(triangles, dtype=np.int32).ravel(), dtype=wp.int32, device=device)
    mesh = wp.Mesh(points=v, indices=t)
    ids = wp.array([mesh.id], dtype=wp.uint64, device=device)
    return {"vertices": v, "triangles": t, "mesh": mesh, "mesh_ids": ids}


def sensor_pose(mount, pitch_deg, roll_deg):
    from scipy.spatial.transform import Rotation
    delta = Rotation.from_euler("xyz", [roll_deg, pitch_deg, 0], degrees=True)
    baseline = Rotation.from_matrix(mount["base_rot_world"])
    body_pos = mount["base_pos_world"]
    sensor_pos = body_pos + (baseline * delta).apply(mount["mount_pos_base"])
    sensor_rot = baseline * delta * Rotation.from_matrix(mount["mount_rot_base"])
    return sensor_pos.astype(np.float32), sensor_rot.as_quat().astype(np.float32)


def make_query(wp, torch, device, mesh, dirs, mount, pitch, roll):
    from hexapod_mjlab.utils.lidar.sensor_kernels.lidar_kernels_warp import LidarWarpKernels
    n = len(dirs)
    pos, quat = sensor_pose(mount, pitch, roll)
    pos_t = torch.tensor(pos.reshape(1, 1, 3), device=device)
    quat_t = torch.tensor(quat.reshape(1, 1, 4), device=device)
    ray_t = torch.tensor(dirs.reshape(n, 1, 3), dtype=torch.float32, device=device)
    points = torch.zeros((1, 1, n, 1, 3), dtype=torch.float32, device=device)
    distances = torch.zeros((1, 1, n, 1), dtype=torch.float32, device=device)
    inputs = [
        mesh["mesh_ids"],
        wp.from_torch(pos_t, dtype=wp.vec3),
        wp.from_torch(quat_t, dtype=wp.quat),
        wp.from_torch(ray_t, dtype=wp.vec3),
        float(MAX_RANGE_M),
        wp.from_torch(points, dtype=wp.vec3),
        wp.from_torch(distances, dtype=wp.float32),
        False,
    ]
    def run():
        wp.launch(
            kernel=LidarWarpKernels.draw_optimized_kernel_pointcloud,
            dim=(1, 1, n, 1),
            inputs=inputs,
            device=device,
        )
    # These are real owner tensors: Warp arrays alias the Torch allocations.
    return dict(run=run, points=points, distances=distances,
                owners=(pos_t, quat_t, ray_t), n=n)


def event_timing(torch, fn, *, warmup, iterations):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(float(start.elapsed_time(end)))
    return {"median_ms": float(np.median(times)), "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)), "samples_ms": times}


def load_reciped_ids(design):
    rays = airy_directions()
    manifest, recipes = read_frozen_design(Path(design), rays)
    return manifest, rays, recipes


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--design", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True, help="must not exist")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iterations", type=int, default=15)
    args = ap.parse_args(argv)
    if args.warmup < 1 or args.iterations < 3:
        raise ValueError("require >=1 warmup and >=3 measurements")
    design = args.design.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    import torch
    import warp as wp
    if not torch.cuda.is_available() or args.gpu >= torch.cuda.device_count() or args.gpu < 0:
        raise RuntimeError("real CUDA GPU required")
    torch.cuda.set_device(args.gpu)
    wp.init()
    device = f"cuda:{args.gpu}"
    manifest, rays, recipe_ids = load_reciped_ids(design)
    mount = nominal_mount()

    results = {}
    for name, width, pitch, roll in (
        ("flat", None, 0, 0),
        ("corridor_1p6", 1.6, 0, 0),
        # +Y Euler pitch rotates BASE +x toward world -z: nose DOWN.
        ("flat_pitch_up_5", None, -5, 0),
        ("flat_pitch_down_5", None, 5, 0),
    ):
        mesh = build_geometry(wp, device, width)
        full = make_query(wp, torch, device, mesh, rays, mount, pitch, roll)
        full["run"]()
        torch.cuda.synchronize()
        full_dist = full["distances"].reshape(-1).cpu().numpy()
        analytic = intersect_scene(rays, mount, width=width, pitch_deg=pitch, roll_deg=roll)
        expected = np.linalg.norm(
            analytic["points"] - mount["mount_pos_base"], axis=1
        )
        if not np.allclose(full_dist, expected, atol=3e-4, rtol=0):
            failures = int((np.abs(full_dist - expected) > 3e-4).sum())
            raise AssertionError(f"full Warp != analytic: scene={name}, {failures} rays")
        case = {
            "full_count": len(rays),
            "warp_vs_analytic_max_abs_m": float(np.max(np.abs(full_dist - expected))),
            "full_cuda_kernel": event_timing(torch, full["run"], warmup=args.warmup,
                                              iterations=args.iterations),
            "sparse": {},
        }
        for label, ids in recipe_ids.items():
            sparse = make_query(wp, torch, device, mesh, rays[ids], mount, pitch, roll)
            sparse["run"]()
            torch.cuda.synchronize()
            d = sparse["distances"].reshape(-1).cpu().numpy()
            error = np.abs(d - full_dist[ids])
            if np.max(error) > 1e-4:
                raise AssertionError(f"sparse Warp != full Warp for same physical rays: {name}/{label}")
            metrics = event_timing(torch, sparse["run"], warmup=args.warmup,
                                   iterations=args.iterations)
            case["sparse"][label] = {
                "raycasts": len(ids),
                "warp_vs_full_max_abs_m": float(np.max(error)),
                "kernel": metrics,
                "kernel_speedup_vs_full": float(
                    case["full_cuda_kernel"]["median_ms"] / metrics["median_ms"]
                ),
            }
        results[name] = case

    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "confined_locomotion_2.sparse_rays.cuda_kernel_benchmark/v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_design_manifest_sha256": sha256(design / "manifest.json"),
        "gpu": {"device": str(torch.cuda.get_device_name(args.gpu)),
                "torch": torch.__version__, "warp": wp.__version__},
        "timing": {"warmup": args.warmup, "iterations": args.iterations,
                   "method": "torch CUDA events, synchronized; kernel launch only"},
        "scope": "one env, static floor/wall diagnostic mesh, no FPS/no point grouping/no physics/no train; +pitch=nose-down",
        "cases": results,
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    brief = {scene: {
        "full_ms": v["full_cuda_kernel"]["median_ms"],
        "sparse": {name: {"ms": x["kernel"]["median_ms"],
                           "speedup": x["kernel_speedup_vs_full"]}
                   for name, x in v["sparse"].items()},
    } for scene, v in results.items()}
    print(json.dumps({"output": str(output), "kernel_only": brief},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
