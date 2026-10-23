#!/usr/bin/env python3
"""GPU witness for analytic obstacle scenes using the production Warp ray kernel.

One independent static triangle mesh per scene. Compare 86,400 full-ray
distances with analytic floor+box first hits; sparse source-ID distances must
equal the exact full-ray subset. No task/runtime/training source modifications.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from benchmark import make_query
from evaluate_obstacles import SCENES, scan_obstacles
from select_channels import MAX_RANGE_M, airy_directions, nominal_mount, sha256
from evaluate_moving_footprint import read_frozen_design, resolve_recipe_names

SCENE_NAMES = (
    "step_10cm", "step_25cm", "post_8cm_centre", "post_8cm_offset",
    "corridor_1p6", "corridor_1p6_roll_left_5", "overhead_beam",
)


def mesh_for_scene(wp, device: str, scene: dict):
    vertices: list = []
    triangles: list = []

    def quad(a, b, c, d):
        k = len(vertices)
        vertices.extend((a, b, c, d))
        triangles.extend(((k, k+1, k+2), (k, k+2, k+3)))

    quad((-32., -32., 0.), (32., -32., 0.), (32., 32., 0.), (-32., 32., 0.))
    for label, lo, hi in scene["boxes"]:
        ax, ay, az = lo
        bx, by, bz = hi
        # Six finite solid box faces, two triangles per face; Warp accepts
        # double-sided mesh ray intersections (also independently witnessed).
        quad((ax, ay, az), (bx, ay, az), (bx, by, az), (ax, by, az))  # bottom
        quad((ax, ay, bz), (ax, by, bz), (bx, by, bz), (bx, ay, bz))  # top
        quad((ax, ay, az), (ax, ay, bz), (bx, ay, bz), (bx, ay, az))
        quad((ax, by, az), (bx, by, az), (bx, by, bz), (ax, by, bz))
        quad((ax, ay, az), (ax, by, az), (ax, by, bz), (ax, ay, bz))
        quad((bx, ay, az), (bx, ay, bz), (bx, by, bz), (bx, by, az))
    v = wp.array(np.asarray(vertices, dtype=np.float32), dtype=wp.vec3, device=device)
    t = wp.array(np.asarray(triangles, dtype=np.int32).reshape(-1),
                 dtype=wp.int32, device=device)
    mesh = wp.Mesh(points=v, indices=t)
    ids = wp.array([mesh.id], dtype=wp.uint64, device=device)
    return {"vertices": v, "triangles": t, "mesh": mesh, "mesh_ids": ids}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new, absent directory")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--recipes", nargs="+", help="optional manifest recipe subset; default: all")
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"cannot overwrite verification output: {output}")
    import torch
    import warp as wp
    if not torch.cuda.is_available() or args.gpu < 0 or args.gpu >= torch.cuda.device_count():
        raise RuntimeError("real NVIDIA CUDA device required")
    torch.cuda.set_device(args.gpu)
    wp.init()
    device = f"cuda:{args.gpu}"
    rays = airy_directions()
    mount = nominal_mount()
    frozen, choices = read_frozen_design(args.design.expanduser().resolve(), rays)
    recipe_names = resolve_recipe_names(choices, args.recipes)
    choices = {name: choices[name] for name in recipe_names}
    results = {}
    for case in SCENES:
        if case["name"] not in SCENE_NAMES:
            continue
        geom = mesh_for_scene(wp, device, case)
        full = make_query(wp, torch, device, geom, rays, mount,
                          case["pitch"], case["roll"])
        full["run"]()
        torch.cuda.synchronize()
        observed = full["distances"].reshape(-1).cpu().numpy()
        expected = scan_obstacles(rays, mount, case)["distances_m"]
        residual = np.abs(observed - expected)
        # At a triangle edge / near-grazing hit, a raw exact-distance
        # cross-check is deliberately strict. Fail rather than silently hiding
        # geometry mismatch with a large tolerance.
        mismatches = int(np.count_nonzero(residual > 0.002))
        if mismatches:
            top = np.argsort(residual)[-8:][::-1]
            raise AssertionError(
                f"analytic/Warp scene {case['name']}: {mismatches} distances differ "
                f"above 0.002m; worst [(id, warp, analytic, abs)]: "
                f"{[(int(i), float(observed[i]), float(expected[i]), float(residual[i])) for i in top]}"
            )
        tests = {}
        for key in recipe_names:
            ids = choices[key]
            query = make_query(wp, torch, device, geom, rays[ids], mount,
                               case["pitch"], case["roll"])
            query["run"]()
            torch.cuda.synchronize()
            sparse_dist = query["distances"].reshape(-1).cpu().numpy()
            sparse_error = np.abs(sparse_dist - observed[ids])
            if np.max(sparse_error) > 0.0001:
                raise AssertionError(f"full/sparse Warp mismatch: {case['name']}/{key}")
            tests[key] = {
                "rays": int(len(ids)),
                "max_full_vs_sparse_warp_dist_m": float(sparse_error.max()),
            }
        results[case["name"]] = {
            "num_physical_rays": int(len(rays)),
            "analytic_vs_warp_max_dist_m": float(residual.max()),
            "analytic_vs_warp_bad_count_at_2mm": mismatches,
            "sparse": tests,
        }

    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "confined_locomotion_2.sparse_rays.obstacle_warp_parity/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_design_sha256": sha256(args.design / "manifest.json"),
        "gpu": torch.cuda.get_device_name(args.gpu),
        "scope": "static floor+solid box Warp ray intersections only; not production 50Hz runtime or training",
        "scenes": results,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "results": results},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
