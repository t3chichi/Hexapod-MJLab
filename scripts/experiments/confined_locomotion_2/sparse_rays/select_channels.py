#!/usr/bin/env python3
"""Freeze versioned static AIRY channel selections for confined_locomotion_2."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
MJCF = ROOT / "src/hexapod_mjlab/assets/robots/el_4090/mjcf/el_4090_lidar.xml"
from hexapod_mjlab.tasks.confined_locomotion_2.components.lidar.channels import (  # noqa: E402
    build_channel_catalog,
    airy_directions,
    _t1_pose_and_mount,
)

NEAR_BUDGET = 256
FAR_BUDGET = 128
FULL_RAYS = 900 * 96
MAX_RANGE_M = 15.0
NEAR_OUT = NEAR_BUDGET
FAR_OUT = FAR_BUDGET
THETA_DEG = 28.0
TILT_DEG = 56.0
NEAR_MAX_X_M = 3.5


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nominal_mount() -> dict:
    """Compiled T1 settled BASE pose and BASE→lidar_sensor transform."""
    pose, base_pos, base_rot, flat_mount = _t1_pose_and_mount()
    return {
        "mount_pos_base": flat_mount[:3].copy(),
        "mount_rot_base": flat_mount[3:].reshape(3, 3).copy(),
        "root_z_nominal": float(base_pos[2]),
        "base_pos_world": base_pos,
        "base_rot_world": base_rot,
        "pose_t1": pose,
    }


def nominal_scan(directions: np.ndarray, mount: dict, floor_z: float = 0.0) -> dict:
    """T1 baseline ray projection, with stable B8 ownership and range endpoints."""
    dirs_base = (mount["mount_rot_base"] @ np.asarray(directions, dtype=np.float64).T).T
    dirs_world = (mount["base_rot_world"] @ dirs_base.T).T
    origin_base = mount["mount_pos_base"]
    origin_world = mount["base_pos_world"] + mount["base_rot_world"] @ origin_base
    t = np.full(len(dirs_world), np.inf, dtype=np.float64)
    np.divide(floor_z - origin_world[2], dirs_world[:, 2], out=t,
              where=np.abs(dirs_world[:, 2]) > 1e-12)
    hit = (t > 0) & (t < MAX_RANGE_M)
    length = np.where(hit, t, MAX_RANGE_M)
    endpoint_world = origin_world + length[:, None] * dirs_world
    endpoints_base = (mount["base_rot_world"].T @
                      (endpoint_world - mount["base_pos_world"]).T).T
    axis = mount["mount_rot_base"] @ np.array([0., 0., 1.])
    delta = np.deg2rad(TILT_DEG)
    axis = np.array([np.cos(delta) * axis[0] + np.sin(delta) * axis[2],
                     axis[1], -np.sin(delta) * axis[0] + np.cos(delta) * axis[2]])
    axis /= np.linalg.norm(axis)
    cone = dirs_base @ axis >= np.sin(np.deg2rad(THETA_DEG))
    near = cone & (endpoints_base[:, 0] <= NEAR_MAX_X_M)
    return {"endpoints_base": endpoints_base, "directions_base": dirs_base,
            "hit": hit, "near": near, "far": ~near, "cone": cone}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spatial_order(points: np.ndarray, raw_ids: np.ndarray, count: int) -> np.ndarray:
    """Centre-seeded greedy Euclidean spread, deterministic by raw ID ties."""
    raw_ids = np.asarray(raw_ids, dtype=np.int64)
    if count < 1 or count > len(raw_ids):
        raise ValueError(f"requested {count} channels from pool of {len(raw_ids)}")
    xyz = np.asarray(points[raw_ids], dtype=np.float64)
    centre = xyz.mean(axis=0)
    seed = int(np.argmin(np.einsum("ij,ij->i", xyz - centre, xyz - centre)))
    chosen = np.empty(count, dtype=np.int64)
    d2 = np.full(len(raw_ids), np.inf, dtype=np.float64)
    selected = np.zeros(len(raw_ids), dtype=bool)
    current = seed
    for i in range(count):
        chosen[i] = raw_ids[current]
        selected[current] = True
        delta = xyz - xyz[current]
        np.minimum(d2, np.einsum("ij,ij->i", delta, delta), out=d2)
        d2[selected] = -1.0
        if i + 1 < count:
            current = int(np.argmax(d2))
    return chosen


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new directory for frozen design files")
    parser.add_argument("--near-multipliers", nargs="+", type=int, default=[4, 5, 6, 8])
    far_group = parser.add_mutually_exclusive_group()
    far_group.add_argument("--far-multiplier", type=int,
                           help="legacy single far multiplier (default: 4)")
    far_group.add_argument("--far-multipliers", nargs="+", type=int,
                           help="independent far candidate multipliers")
    args = parser.parse_args(argv)
    # The legacy spelling remains the default behavior for existing workflows.
    args.far_multipliers = ([args.far_multiplier] if args.far_multiplier is not None
                            else args.far_multipliers or [4])
    return args


def main(argv=None):
    args = parse_args(argv)
    near_multipliers = sorted(set(args.near_multipliers))
    far_multipliers = sorted(set(args.far_multipliers))
    if (not near_multipliers or not far_multipliers or
            any(x <= 0 for x in near_multipliers + far_multipliers)):
        raise ValueError("multipliers must be positive integers")
    out = args.output.expanduser().resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory {out}")
    catalog = build_channel_catalog()
    near_candidates = np.flatnonzero(catalog.near_mask)
    far_candidates = np.flatnonzero(~catalog.near_mask)
    # Preserve all nominal near candidates, including 15 m no-hit endpoints.
    near_order = spatial_order(catalog.endpoints_base, near_candidates,
                               NEAR_BUDGET * max(near_multipliers))
    far_order = spatial_order(catalog.endpoints_base, far_candidates,
                              FAR_BUDGET * max(far_multipliers))
    out.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "confined_locomotion_2.fixed_channels/v1",
        "catalog_identity": catalog.identity,
        "near_multipliers": near_multipliers,
        "far_multipliers": far_multipliers,
        "files": {},
        "method": "centre-seeded greedy Euclidean endpoint spread; near and far prefixes nested",
    }
    # Retain the scalar manifest field for old consumers when the far dimension
    # has exactly one choice. New multi-far manifests use the array field.
    if len(far_multipliers) == 1:
        manifest["far_multiplier"] = far_multipliers[0]
    for near_mult in near_multipliers:
        near_ids = near_order[: NEAR_BUDGET * near_mult]
        for far_mult in far_multipliers:
            name = f"near_x{near_mult}_far_x{far_mult}"
            far_ids = far_order[: FAR_BUDGET * far_mult]
            ids = np.sort(np.concatenate((near_ids, far_ids))).astype(np.int64)
            if len(np.unique(ids)) != len(ids):
                raise RuntimeError("near and far source IDs overlap")
            npz_path = out / f"{name}.npz"
            np.savez_compressed(
                npz_path,
                source_ray_ids=ids,
                source_ray_directions_sensor=catalog.directions_sensor[ids],
                designed_near_ids=near_ids.astype(np.int64),
                designed_far_ids=far_ids.astype(np.int64),
                catalog_identity_json=np.asarray(json.dumps(catalog.identity, sort_keys=True)),
            )
            recipe = {
                "file": npz_path.name,
                "npz_sha256": _sha256(npz_path),
                "catalog_identity": catalog.identity,
                "selection": {
                    "near_multiplier": near_mult,
                    "far_multiplier": far_mult,
                    "near_count": len(near_ids),
                    "far_count": len(far_ids),
                    "total_count": len(ids),
                    "raw_ids_sorted": True,
                },
            }
            (out / f"{name}.json").write_text(
                json.dumps({"schema": manifest["schema"], **recipe}, indent=2), encoding="utf-8")
            manifest["files"][name] = recipe
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "catalog_counts": catalog.identity["counts"], "files": list(manifest["files"])}, indent=2))


if __name__ == "__main__":
    main()
