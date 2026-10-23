"""Static raw-channel catalog and validated sparse selections for task _2.

The catalog is an offline design reference: actual AIRY directions are projected
from the T1 settled pose onto the world z=0 plane, then classified once using
the task's B8 cone and BASE-x cutoff. Runtime scans must still raycast these raw
directions against the current scene.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

_PACKAGE = Path(__file__).resolve().parents[2]
_MJCF = _PACKAGE / "assets/robots/el_4090/mjcf/el_4090_lidar.xml"
_GENERATOR = _PACKAGE / "utils/lidar/sensor_pattern/sensor_lidar/genera_lidar_scan_pattern.py"
_WITNESS = _PACKAGE / "tasks/el_4090_lidar/t0/witness.py"
_N_AZ = 900
_N_EL = 96
_N_RAYS = _N_AZ * _N_EL
_SCHEMA = "confined_locomotion_2.fixed_channels/v1"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def airy_directions() -> np.ndarray:
    """Return the real AIRY pattern in raw ID order ``azimuth*96+elevation``."""
    from hexapod_mjlab.utils.lidar.sensor_pattern.sensor_lidar.genera_lidar_scan_pattern import (
        SpinningLidarGenerator,
    )

    theta, phi = SpinningLidarGenerator.generate_AIRY(
        n_channels=_N_EL, horizontal_resolution_deg=0.4
    )
    dirs = np.stack(
        (np.cos(phi) * np.cos(theta), np.cos(phi) * np.sin(theta), np.sin(phi)),
        axis=-1,
    ).astype(np.float64)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    if dirs.shape != (_N_RAYS, 3) or not np.isfinite(dirs).all():
        raise RuntimeError("AIRY generator violates the 900x96 source ID contract")
    return dirs


def _t1_pose_and_mount() -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Load compiled MJCF at the frozen T1 qpos and resolve BASE/site transform."""
    import mujoco
    from hexapod_mjlab.tasks.el_4090_lidar.t0.witness import SETTLED_QPOS

    model = mujoco.MjModel.from_xml_path(str(_MJCF))
    data = mujoco.MjData(model)
    qpos = np.asarray(SETTLED_QPOS, dtype=np.float64)
    if qpos.shape != (model.nq,):
        raise ValueError(f"T1 SETTLED_QPOS length {qpos.size} != compiled MJCF nq {model.nq}")
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "BASE")
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "lidar_sensor")
    if bid < 0 or sid < 0:
        raise ValueError("compiled lidar MJCF must contain BASE and lidar_sensor")
    base_pos_w = np.asarray(data.xpos[bid], dtype=np.float64).copy()
    base_rot_w = np.asarray(data.xmat[bid], dtype=np.float64).reshape(3, 3).copy()
    site_pos_w = np.asarray(data.site_xpos[sid], dtype=np.float64).copy()
    site_rot_w = np.asarray(data.site_xmat[sid], dtype=np.float64).reshape(3, 3).copy()
    mount_pos_b = base_rot_w.T @ (site_pos_w - base_pos_w)
    mount_rot_b = base_rot_w.T @ site_rot_w
    pose = {
        "qpos": qpos.tolist(),
        "base_pos_world_m": base_pos_w.tolist(),
        "base_quat_wxyz": np.asarray(data.xquat[bid], dtype=np.float64).tolist(),
        "base_rot_world": base_rot_w.tolist(),
    }
    return pose, base_pos_w, base_rot_w, np.concatenate((mount_pos_b, mount_rot_b.reshape(-1)))


@dataclass(frozen=True)
class ChannelCatalog:
    directions_sensor: np.ndarray
    near_mask: np.ndarray
    endpoints_base: np.ndarray
    identity: dict[str, Any]

    @property
    def source_ids(self) -> np.ndarray:
        return np.arange(_N_RAYS, dtype=np.int64)

    @property
    def far_mask(self) -> np.ndarray:
        return ~self.near_mask


@lru_cache(maxsize=16)
def build_channel_catalog(
    theta_deg: float = 28,
    axis_tilt_deg: float = 56,
    forward_x_max_m: float = 3.5,
    max_range_m: float = 15,
) -> ChannelCatalog:
    """Build one immutable-by-convention full-resolution catalog.

    The source pose uses the complete frozen T1 ``SETTLED_QPOS``. Rays hit the
    hypothetical world plane z=0 only for 0<t<max_range; other endpoints sit
    at max range. Near is the fixed B8 cone membership intersected with
    endpoint BASE-x <= forward_x_max_m, including no-hit endpoints.
    """
    values = (theta_deg, axis_tilt_deg, forward_x_max_m, max_range_m)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("catalog parameters must all be finite")
    if theta_deg <= 0 or theta_deg >= 90 or max_range_m <= 0:
        raise ValueError("theta_deg must be in (0,90) and max_range_m positive")
    dirs = airy_directions()
    pose, base_pos_w, base_rot_w, mount_flat = _t1_pose_and_mount()
    mount_pos_b, mount_rot_b = mount_flat[:3], mount_flat[3:].reshape(3, 3)
    sensor_pos_w = base_pos_w + base_rot_w @ mount_pos_b
    sensor_rot_w = base_rot_w @ mount_rot_b
    dirs_world = (sensor_rot_w @ dirs.T).T
    dz = dirs_world[:, 2]
    distance = np.full(_N_RAYS, np.inf, dtype=np.float64)
    np.divide(-sensor_pos_w[2], dz, out=distance, where=np.abs(dz) > 1e-12)
    hit = (distance > 0.0) & (distance < max_range_m)
    lengths = np.where(hit, distance, max_range_m)
    endpoints_world = sensor_pos_w + lengths[:, None] * dirs_world
    endpoints_base = (base_rot_w.T @ (endpoints_world - base_pos_w).T).T

    dome_axis_b = mount_rot_b[:, 2]
    delta = np.deg2rad(axis_tilt_deg)
    axis_b = np.array([
        np.cos(delta) * dome_axis_b[0] + np.sin(delta) * dome_axis_b[2],
        dome_axis_b[1],
        -np.sin(delta) * dome_axis_b[0] + np.cos(delta) * dome_axis_b[2],
    ])
    axis_b /= np.linalg.norm(axis_b)
    dirs_base = (mount_rot_b @ dirs.T).T
    in_cone = dirs_base @ axis_b >= np.sin(np.deg2rad(theta_deg))
    near = in_cone & (endpoints_base[:, 0] <= forward_x_max_m)

    source_hashes = {
        "mjcf_sha256": _sha256_file(_MJCF),
        "airy_generator_sha256": _sha256_file(_GENERATOR),
        "t1_witness_sha256": _sha256_file(_WITNESS),
    }
    direction_hash = _sha256_bytes(np.ascontiguousarray(dirs, dtype="<f8").tobytes())
    mask_hash = _sha256_bytes(np.packbits(near, bitorder="little").tobytes())
    pose_hash = _sha256_bytes(np.asarray(pose["qpos"], dtype="<f8").tobytes())
    mount = {"position_base_m": mount_pos_b.tolist(), "rotation_base_sensor": mount_rot_b.tolist()}
    identity: dict[str, Any] = {
        "schema": _SCHEMA,
        "ray_count": _N_RAYS,
        "source_id_order": "azimuth*96+elevation",
        "pattern": "SpinningLidarGenerator.generate_AIRY(96, 0.4deg)",
        "directions_sha256": direction_hash,
        "near_mask_sha256": mask_hash,
        "pose_sha256": pose_hash,
        "pose_t1": pose,
        "mount_base_to_sensor": mount,
        "source_sha256": source_hashes,
        "classification": {
            "floor_world_z_m": 0.0,
            "theta_deg": float(theta_deg),
            "axis_tilt_deg": float(axis_tilt_deg),
            "forward_x_max_m": float(forward_x_max_m),
            "max_range_m": float(max_range_m),
            "hit_rule": "0 < t < max_range; otherwise endpoint at max_range",
        },
        "counts": {
            "near": int(near.sum()),
            "far": int((~near).sum()),
            "near_nominal_hits": int((near & hit).sum()),
            "near_nominal_nohits": int((near & ~hit).sum()),
        },
    }
    return ChannelCatalog(
        directions_sensor=np.ascontiguousarray(dirs),
        near_mask=np.ascontiguousarray(near),
        endpoints_base=np.ascontiguousarray(endpoints_base),
        identity=identity,
    )


def load_sparse_selection(path: str | Path, catalog: ChannelCatalog) -> tuple[np.ndarray, dict[str, Any]]:
    """Load sorted raw IDs and require exact catalog identity/direction parity.

    ``path`` may be a versioned NPZ or its JSON manifest. A manifest resolves
    the NPZ recorded by its ``file`` field. The returned IDs are sorted int64.
    """
    p = Path(path).expanduser().resolve()
    if p.suffix.lower() == ".json":
        manifest = json.loads(p.read_text(encoding="utf-8"))
        if manifest.get("schema") != _SCHEMA:
            raise ValueError("unsupported sparse channel manifest schema")
        if "file" not in manifest:
            if "files" in manifest:
                raise ValueError("index manifest contains multiple selections; pass a specific near_xN_far_xN.npz or its same-name JSON manifest")
            raise ValueError("selection manifest is missing its NPZ file field")
        selection_meta = manifest
        archive = p.parent / selection_meta["file"]
    else:
        archive = p
        sidecar = p.with_suffix(".json")
        selection_meta = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
    if not archive.is_file():
        raise FileNotFoundError(archive)
    expected_file_hash = selection_meta.get("npz_sha256")
    actual_file_hash = _sha256_file(archive)
    if expected_file_hash and expected_file_hash != actual_file_hash:
        raise ValueError("sparse selection NPZ hash mismatch")
    with np.load(archive, allow_pickle=False) as z:
        if "source_ray_ids" not in z or "source_ray_directions_sensor" not in z:
            raise ValueError("selection NPZ must contain IDs and source directions")
        raw_ids = np.asarray(z["source_ray_ids"])
        if raw_ids.dtype.kind not in "iu":
            raise ValueError("selection IDs must have an integer dtype")
        if raw_ids.dtype.kind == "u" and np.any(raw_ids > np.iinfo(np.int64).max):
            raise ValueError("selection IDs exceed int64 range")
        ids = raw_ids.astype(np.int64, copy=False)
        dirs = np.asarray(z["source_ray_directions_sensor"], dtype=np.float64)
        embedded_identity = json.loads(str(z["catalog_identity_json"].item())) if "catalog_identity_json" in z else None
    if ids.ndim != 1 or len(ids) == 0 or np.any(ids < 0) or np.any(ids >= len(catalog.directions_sensor)):
        raise ValueError("selection IDs must be a nonempty vector in catalog range")
    if np.any(np.diff(ids) <= 0):
        raise ValueError("selection IDs must be unique and already sorted ascending")
    if dirs.shape != (len(ids), 3) or not np.array_equal(dirs, catalog.directions_sensor[ids]):
        raise ValueError("saved directions do not match the actual AIRY catalog IDs")
    sidecar_identity = selection_meta.get("catalog_identity")
    if embedded_identity is None or sidecar_identity is None or embedded_identity != sidecar_identity:
        raise ValueError("embedded and manifest catalog identities disagree or are missing")
    if sidecar_identity != catalog.identity:
        raise ValueError("sparse selection was built against a different catalog identity")
    return ids.astype(np.int64, copy=False), {
        "schema": _SCHEMA,
        "npz_sha256": actual_file_hash,
        "catalog_identity": catalog.identity,
        "selection": selection_meta.get("selection", {}),
    }

