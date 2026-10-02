"""Airy sensor mounting and fixed 187-channel selection for envelope_adaptive_2.

This module selects the 187 channels used by the EA2 fixed ground-grid
observation:

* full Airy pattern: 900 azimuth x 96 elevation = 86,400 rays;
* body-frame ground region: x in [0.65, 3.65], y in [-1, 1];
* 11 rows (x) x 17 cols (y) = 187 cells;
* each cell keeps the ray whose ground hit is closest to the cell centre;
* the normalization divisor is the maximum slant distance among the selected
  channels, rounded up to 0.1 m.

The quaternion helpers are local reproductions of ``isaacgym.torch_utils``
formulas, so this module remains importable without Isaac Gym.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional, Union

import torch

from .contracts import (
    EA2_AIRY_N_AZIMUTH_FULL,
    EA2_AIRY_N_ELEVATION,
    EA2_BASE_HEIGHT_M,
    EA2_FULL_N_RAYS,
    EA2_GRID_COLS,
    EA2_GRID_ROWS,
    EA2_RANGE_DIM,
    EA2_REGION_X_MAX,
    EA2_REGION_X_MIN,
    EA2_REGION_Y_MAX,
    EA2_REGION_Y_MIN,
    EA2_SELECTED_CHANNELS_FILE,
    EA2_SENSOR_OFFSET_POS,
    EA2_SENSOR_OFFSET_RPY,
)

PathLike = Union[str, Path]

_DEG2RAD = math.pi / 180.0


def _quat_from_euler_xyz(roll: float, pitch: float, yaw: float) -> torch.Tensor:
    """Build an xyzw quaternion from XYZ Euler angles (same as torch_utils)."""
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp
    return torch.tensor([qx, qy, qz, qw], dtype=torch.float64)


def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Apply quaternion rotation(s) to vector(s), matching torch_utils."""
    xyz = q[..., :3]
    t = torch.cross(xyz, v, dim=-1) * 2.0
    return v + q[..., 3:] * t + torch.cross(xyz, t, dim=-1)


def generate_full_airy_directions() -> torch.Tensor:
    """Generate the full Airy sensor-frame ray directions.

    Returns:
        Float tensor of shape ``(86400, 3)``, C-order ``az * 96 + el``.
    """
    phi = (
        torch.arange(EA2_AIRY_N_ELEVATION, dtype=torch.float64)
        * (math.pi / 2.0)
        / (EA2_AIRY_N_ELEVATION - 1)
    )
    theta = (
        torch.arange(EA2_AIRY_N_AZIMUTH_FULL, dtype=torch.float64)
        * (2.0 * math.pi)
        / EA2_AIRY_N_AZIMUTH_FULL
    )[:, None]

    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    sensor_dirs = torch.stack(
        [
            cos_phi[None, :] * cos_theta,
            cos_phi[None, :] * sin_theta,
            sin_phi[None, :].expand(
                EA2_AIRY_N_AZIMUTH_FULL, EA2_AIRY_N_ELEVATION
            ),
        ],
        dim=-1,
    ).reshape(-1, 3)
    return sensor_dirs


def body_frame_ray_directions() -> torch.Tensor:
    """Return all full Airy ray directions expressed in the body frame."""
    sensor_dirs = generate_full_airy_directions()
    offset_q = _quat_from_euler_xyz(*EA2_SENSOR_OFFSET_RPY)
    body_dirs = _quat_apply(
        offset_q.view(1, 4).expand(EA2_FULL_N_RAYS, 4), sensor_dirs
    )
    return body_dirs.to(dtype=torch.float32)


def _ground_hits_in_region() -> tuple:
    """Compute body-frame ground hits for all downward full-Airy rays.

    Returns:
        ``(valid_indices, hit_xy, slant_ranges)`` where ``valid_indices`` are
        full ray indices with ``dir_z < 0``, ``hit_xy`` is ``(N, 2)`` and
        ``slant_ranges`` is ``(N,)``.
    """
    sensor_pos = torch.tensor(EA2_SENSOR_OFFSET_POS, dtype=torch.float64)
    body_dirs = body_frame_ray_directions().to(dtype=torch.float64)

    downward = body_dirs[:, 2] < 0.0
    valid_indices = torch.nonzero(downward, as_tuple=False).squeeze(1)
    dirs = body_dirs[valid_indices]

    # The body frame origin is at the robot base height, so the ground plane
    # in body coordinates is z = -EA2_BASE_HEIGHT_M.
    ground_z = -EA2_BASE_HEIGHT_M
    t = (ground_z - sensor_pos[2]) / dirs[:, 2]
    hit_xy = sensor_pos[:2].unsqueeze(0) + dirs[:, :2] * t.unsqueeze(-1)
    slant_ranges = t  # dirs are unit length
    return valid_indices, hit_xy, slant_ranges


def select_ground_grid_channels() -> Dict[str, object]:
    """Select the fixed 187 channels for the EA2 ground grid.

    Returns:
        Dict with ``ray_indices``, ``ray_directions``, ``cell_centers``,
        ``ground_hits``, ``slant_ranges`` and ``max_range``.
    """
    valid_indices, hit_xy, slant_ranges = _ground_hits_in_region()

    xs = torch.linspace(
        EA2_REGION_X_MIN, EA2_REGION_X_MAX, EA2_GRID_ROWS, dtype=torch.float64
    )
    ys = torch.linspace(
        EA2_REGION_Y_MIN, EA2_REGION_Y_MAX, EA2_GRID_COLS, dtype=torch.float64
    )
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    centers = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)  # (187, 2)

    # Distance from each cell centre to each valid ground hit.
    dist2 = (
        (centers[:, None, :] - hit_xy[None, :, :]) ** 2
    ).sum(dim=-1)  # (187, N_valid)
    nearest = torch.argmin(dist2, dim=1)  # (187,)

    selected_full_indices = valid_indices[nearest]
    selected_hits = hit_xy[nearest]
    selected_slant = slant_ranges[nearest]
    # The Warp raycast kernel expects SENSOR-frame ray directions.  Storing
    # body-frame directions here would cause a double rotation and misaligned
    # point clouds in the reduced-raycast path.
    selected_dirs = generate_full_airy_directions().to(dtype=torch.float32)[
        selected_full_indices
    ]

    max_slant = float(selected_slant.max().item())
    max_range = math.ceil(max_slant * 10.0) / 10.0

    return {
        "ray_indices": selected_full_indices.to(dtype=torch.long),
        "ray_directions": selected_dirs,  # sensor frame, (187, 3)
        "cell_centers": centers.to(dtype=torch.float32),
        "ground_hits": selected_hits.to(dtype=torch.float32),
        "slant_ranges": selected_slant.to(dtype=torch.float32),
        "max_range": float(max_range),
        "grid_rows": int(EA2_GRID_ROWS),
        "grid_cols": int(EA2_GRID_COLS),
    }


def save_selected_channels(
    data: Dict[str, object], path: Optional[PathLike] = None
) -> Path:
    """Save the selected-channel dictionary to ``.pt``."""
    if path is None:
        path = EA2_SELECTED_CHANNELS_FILE
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)
    return path


def load_selected_channels(
    path: Optional[PathLike] = None,
) -> Dict[str, object]:
    """Load the selected-channel dictionary from ``.pt``."""
    if path is None:
        path = EA2_SELECTED_CHANNELS_FILE
    return torch.load(Path(path), map_location="cpu")


def self_check_selected_channels(data: Dict[str, object]) -> Dict[str, object]:
    """Validate the selected-channel dictionary."""
    ray_indices = data["ray_indices"]
    ray_directions = data["ray_directions"]
    cell_centers = data["cell_centers"]
    ground_hits = data["ground_hits"]
    slant_ranges = data["slant_ranges"]
    max_range = float(data["max_range"])

    if ray_indices.shape != (EA2_RANGE_DIM,):
        raise AssertionError(
            f"ray_indices shape must be {(EA2_RANGE_DIM,)}, got {tuple(ray_indices.shape)}"
        )
    if ray_indices.dtype != torch.long:
        raise AssertionError(f"ray_indices dtype must be long, got {ray_indices.dtype}")
    if len(torch.unique(ray_indices)) != EA2_RANGE_DIM:
        raise AssertionError("selected ray indices are not unique")

    if ray_directions.shape != (EA2_RANGE_DIM, 3):
        raise AssertionError(
            f"ray_directions shape must be {(EA2_RANGE_DIM, 3)}, "
            f"got {tuple(ray_directions.shape)}"
        )
    if cell_centers.shape != (EA2_RANGE_DIM, 2):
        raise AssertionError(
            f"cell_centers shape must be {(EA2_RANGE_DIM, 2)}, "
            f"got {tuple(cell_centers.shape)}"
        )
    if ground_hits.shape != (EA2_RANGE_DIM, 2):
        raise AssertionError(
            f"ground_hits shape must be {(EA2_RANGE_DIM, 2)}, "
            f"got {tuple(ground_hits.shape)}"
        )
    if slant_ranges.shape != (EA2_RANGE_DIM,):
        raise AssertionError(
            f"slant_ranges shape must be {(EA2_RANGE_DIM,)}, "
            f"got {tuple(slant_ranges.shape)}"
        )

    # All selected ground hits must lie inside the requested region, with a
    # small tolerance for discrete ray coverage near the boundary.
    margin = 0.15
    in_region = (
        (ground_hits[:, 0] >= EA2_REGION_X_MIN - margin)
        & (ground_hits[:, 0] <= EA2_REGION_X_MAX + margin)
        & (ground_hits[:, 1] >= EA2_REGION_Y_MIN - margin)
        & (ground_hits[:, 1] <= EA2_REGION_Y_MAX + margin)
    )
    if not bool(in_region.all()):
        raise AssertionError("some selected ground hits fall outside the region")

    # max_range must be the max slant rounded up to 0.1 m.
    expected_max = math.ceil(float(slant_ranges.max().item()) * 10.0) / 10.0
    if abs(max_range - expected_max) > 1e-9:
        raise AssertionError(
            f"max_range {max_range} does not match expected {expected_max}"
        )

    # Stored ray_directions must be in SENSOR frame: rotating them by the
    # sensor offset must reproduce the body-frame directions of the selected
    # full-Airy rays.  This prevents the double-rotation bug in the reduced
    # raycast path.
    offset_q = _quat_from_euler_xyz(*EA2_SENSOR_OFFSET_RPY)
    expected_body = body_frame_ray_directions()[ray_indices]
    actual_body = _quat_apply(
        offset_q.view(1, 4).expand(ray_directions.shape[0], 4),
        ray_directions.to(dtype=torch.float64),
    ).to(dtype=torch.float32)
    if not torch.allclose(actual_body, expected_body, atol=1e-4, rtol=1e-4):
        raise AssertionError("ray_directions are not in sensor frame")

    return {
        "n_channels": int(ray_indices.shape[0]),
        "unique": True,
        "max_range": max_range,
        "grid_shape": (int(data["grid_rows"]), int(data["grid_cols"])),
        "all_hits_in_region": True,
    }
