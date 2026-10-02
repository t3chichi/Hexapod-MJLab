"""Build the fixed 187-channel range image for envelope_adaptive_2.

The 187 channels are pre-selected from the full Airy pattern and correspond
one-to-one to the cells of a body-frame ground region.  This module therefore
does not perform old-style multi-ray bucket aggregation or ``r_min``/``max_range``
filtering.  Instead:

* each channel keeps its raw slant distance;
* no-hit / out-of-region distances are replaced by ``max_range`` (the
  normalization divisor);
* the observation is normalized by ``max_range``.

``max_range`` is the maximum slant distance among the selected 187 channels,
rounded up to 0.1 m.
"""

from __future__ import annotations

import torch

from .contracts import EA2_LIDAR_FAR_PLANE_M


def _flatten_dists(dists: torch.Tensor) -> torch.Tensor:
    """Flatten LidarSensor-style dists to ``(E, N)``."""
    if dists.dim() == 4 and dists.shape[-1] == 1:
        return dists.reshape(dists.shape[0], -1)
    if dists.dim() != 2:
        raise ValueError(
            f"dists must be 2D (E, N) or 4D (E, 1, N, 1), got shape {tuple(dists.shape)}"
        )
    return dists


def build_selected_range_image(
    dists: torch.Tensor,
    max_range: float,
    far_plane: float = EA2_LIDAR_FAR_PLANE_M,
) -> torch.Tensor:
    """Convert the 187 selected-channel distances into a raw range image.

    Args:
        dists: Per-channel distances, shape ``(E, 187)`` or ``(E, 1, 187, 1)``.
        max_range: Normalization divisor / empty sentinel (max slant rounded up).
        far_plane: Sensor no-hit distance; kept for explicitness.

    Returns:
        Float tensor of shape ``(E, 187)``.  Valid hits keep their raw slant
        distance; no-hit and out-of-region distances are set to ``max_range``.
    """
    dists = _flatten_dists(dists)
    valid = (dists >= 0.0) & (dists <= max_range) & (dists < far_plane)
    return torch.where(valid, dists, torch.full_like(dists, max_range))


def extract_selected_range_image(
    full_dists: torch.Tensor,
    selected_indices: torch.Tensor,
    max_range: float,
    far_plane: float = EA2_LIDAR_FAR_PLANE_M,
) -> torch.Tensor:
    """Extract the 187 selected channels from a full Airy distance cloud.

    Used by the debug/full-raycast path to compute the same observation from
    the complete 86,400-ray point cloud.

    Args:
        full_dists: Full per-ray distances, shape ``(E, N_full)`` or
            ``(E, 1, N_full, 1)``.
        selected_indices: Integer tensor of shape ``(187,)`` selecting rays.
        max_range: Normalization divisor / empty sentinel.
        far_plane: Sensor no-hit distance.

    Returns:
        Float tensor of shape ``(E, 187)`` in the fixed row-major grid order.
    """
    full_dists = _flatten_dists(full_dists)
    selected = full_dists[:, selected_indices.to(device=full_dists.device)]
    return build_selected_range_image(selected, max_range, far_plane=far_plane)


def range_image_observation(
    range_image: torch.Tensor,
    max_range: float,
) -> torch.Tensor:
    """Normalize a raw range image by dividing every channel by ``max_range``.

    Args:
        range_image: Raw selected range image, shape ``(..., 187)``.
        max_range: Normalization divisor (max slant rounded up).

    Returns:
        Normalized range image of the same shape/dtype as the input.
    """
    return range_image / max_range
