"""Sequence-aware mirror helpers; tensor transforms reuse the v1 contracts."""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from hexapod_mjlab.rl.confined_locomotion.symmetry import (
    JointPermutation, MirrorMappingError, mirror_action_distribution,
    mirror_envelope_values, mirror_mean_mse, mirror_observation,
    mirror_point_block,
)


class SymmetryError(ValueError):
    pass


def mirror_height_targets(heights, valid):
    """Reflect root-yaw-frame targets across Y=0; preserve signed Z values."""
    if heights.shape[-2:] != (11, 17) or valid.shape != heights.shape or valid.dtype != torch.bool:
        raise SymmetryError("height targets and mask must have matching [...,11,17] shapes")
    return heights.flip(-2), valid.flip(-2)


def mirror_prepared(prepared: Any, joint_permutation: JointPermutation, *, mount_quat_xyzw: torch.Tensor) -> Any:
    """Mirror one prepared frame with source-ray-aware optical resorting.

    Inputs need their original sensor-frame ray directions; recomputing them from
    endpoints is invalid for zero-range returns. Acquisition-time BODY points are
    reflected and the optical keys are recalculated by the shared legacy helper.
    """
    kwargs = {}
    for prefix, count in (("near", 256), ("far", 128)):
        points = getattr(prepared, f"{prefix}_points")
        valid = getattr(prepared, f"{prefix}_valid")
        dirs = getattr(prepared, f"{prefix}_source_dirs_sensor", None)
        if dirs is None:
            if bool(valid.any()):
                raise SymmetryError(f"{prefix} source ray directions required for mirror")
            dirs = torch.zeros((*valid.shape, 3), dtype=torch.float32, device=points.device)
        block = mirror_point_block(
            points, valid, dirs, mount_quat_xyzw,
            scan_ids=getattr(prepared, "far_point_scan_ids", None) if prefix == "far" else None,
            times=getattr(prepared, "far_point_times", None) if prefix == "far" else None,
            source_indices=getattr(prepared, f"{prefix}_source_indices", None),
        )
        kwargs[f"{prefix}_points"] = block.points
        kwargs[f"{prefix}_valid"] = block.valid
        kwargs[f"{prefix}_angle_keys"] = block.angle_keys
        kwargs[f"{prefix}_source_dirs_sensor"] = block.source_dirs_sensor
        present = getattr(prepared, f"{prefix}_source_dirs_present", None)
        kwargs[f"{prefix}_source_dirs_present"] = (
            torch.gather(present, 1, block.permutation) if present is not None else None
        )
        if prefix == "far":
            kwargs["far_point_scan_ids"] = block.scan_ids
            kwargs["far_point_times"] = block.times
            kwargs["far_source_indices"] = block.source_indices
        else:
            kwargs["near_source_indices"] = block.source_indices
    kwargs["observation"] = mirror_observation(prepared.observation, joint_permutation)
    if hasattr(prepared, "near_scan_ids"):
        kwargs["near_scan_ids"] = prepared.near_scan_ids
    if hasattr(prepared, "near_times"):
        kwargs["near_times"] = prepared.near_times
    return replace(prepared, **{k: v for k, v in kwargs.items() if hasattr(prepared, k)})


__all__ = [
    "JointPermutation", "MirrorMappingError", "SymmetryError", "mirror_action_distribution",
    "mirror_envelope_values", "mirror_mean_mse", "mirror_observation", "mirror_point_block",
    "mirror_prepared", "mirror_height_targets",
]
