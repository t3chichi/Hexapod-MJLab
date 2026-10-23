"""Deterministic XYZ FPS for this task, with a batched CUDA implementation.

Distances use consumed acquisition-time BASE xyz in metres. Invalid slots are
excluded, and selected *indices* are excluded even when coordinates coincide.
CPU/reference execution remains available for independent verification. CUDA
uses no per-selection host scalar read or Python selection loop.
"""
from __future__ import annotations

import torch


FAR_SAMPLING_MODES = ("all_endpoints", "hit_only_10m")


def sampling_identity(far_sampling="all_endpoints"):
    """Input semantics recorded in recipes/checkpoints, independent of backend."""
    if far_sampling not in FAR_SAMPLING_MODES:
        raise ValueError(f"far_sampling must be one of {FAR_SAMPLING_MODES}")
    identity = {"version": "xyz_fps/v1", "metric": "base_xyz_m",
            "seed": "max_norm_first", "ties": "first_pool_index",
            "near_k": 256, "far_k": 128, "without_replacement": True}
    if far_sampling == "hit_only_10m":
        identity.update(version="xyz_fps/far_hit10/v1",
                        far_candidates="fixed_far & gap_valid & raw_hit & raw_range_le_10m",
                        far_range_origin="lidar_origin_raw_before_P6_noise",
                        far_range_limit_m=10.0,
                        shortage="valid_false_xyz_zero_source_minus_one")
    return identity


def far_candidate_mask(fixed_far_valid, raw_dists, no_hit_mask, far_sampling):
    """Filter FPS eligibility; never change physical returns or near ownership.

    All inputs follow the same candidate-pool order. Hit identity and the 10 m
    gate use original raycast data, independently of consumed noisy XYZ.
    """
    if far_sampling not in FAR_SAMPLING_MODES:
        raise ValueError(f"far_sampling must be one of {FAR_SAMPLING_MODES}")
    if far_sampling == "all_endpoints":
        return fixed_far_valid
    return fixed_far_valid & ~no_hit_mask & (raw_dists <= 10.0)


def _validate(points, mask, k, *, batched):
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    ndim = 3 if batched else 2
    if not torch.is_tensor(points) or points.ndim != ndim or points.shape[-1] != 3:
        raise ValueError("points must have shape (B,N,3)" if batched else "points must have shape (N,3)")
    if points.dtype != torch.float32 or points.device.type not in ("cpu", "cuda"):
        raise ValueError("points must be float32 on CPU or CUDA")
    if not torch.is_tensor(mask) or mask.dtype != torch.bool or mask.shape != points.shape[:-1]:
        raise ValueError("mask must be boolean and match the point dimensions")
    if mask.device != points.device:
        raise ValueError("mask and points must be on the same device")


@torch.no_grad()
def farthest_point_sampling_indices_reference(points, group_mask, k):
    """Unfused reference, preserving maximum-norm seed and first-index ties.

    Can execute on CPU or CUDA for verification; CUDA deliberately retains the
    original per-selection synchronization, and is not the production backend.
    Valid coordinates are required to be finite (the scanner validates them).
    """
    _validate(points, group_mask, k, batched=False)
    members = group_mask.nonzero(as_tuple=False).flatten()
    count = min(k, members.numel())
    if count == 0:
        return members.new_empty(0), 0
    pts = points.index_select(0, members)
    selected = int(pts.norm(dim=-1).argmax().item())
    chosen = [selected]
    nearest = (pts - pts[selected]).square().sum(dim=-1)
    nearest[selected] = -1.0
    for _ in range(1, count):
        selected = int(nearest.argmax().item())
        chosen.append(selected)
        nearest = torch.minimum(nearest, (pts - pts[selected]).square().sum(dim=-1))
        nearest[selected] = -1.0
    indices = torch.tensor(chosen, dtype=torch.long, device=points.device)
    return members.index_select(0, indices), count


@torch.no_grad()
def farthest_point_sampling_masked(points, mask, k):
    """Return original slot indices (B,k), with -1 after each row's valid count.

    Masks may have holes; stable compaction stays on the GPU, with no host
    counts or per-pick synchronization. Each row is independent, including empty rows. The scanner calls this
    twice for all due rows: once for near and once for far, using the same xyz.
    No sampling randomness or persistent workspace is introduced.
    """
    _validate(points, mask, k, batched=True)
    batch, size, _ = points.shape
    if batch == 0 or size == 0:
        return torch.full((batch, k), -1, dtype=torch.long, device=points.device)
    if points.device.type == "cpu":
        output = torch.full((batch, k), -1, dtype=torch.long, device=points.device)
        for row in range(batch):
            indices, count = farthest_point_sampling_indices_reference(points[row], mask[row], k)
            output[row, :count] = indices
        return output
    if size >= 2**31:
        raise ValueError("FPS candidate count exceeds int32 indexing")
    # Keep the existing PyTorch norm/argmax arithmetic for the seed. Unit-length
    # directions can be almost tied; changing its sqrt/reduction changed seeds
    # in the independent prototype. All of these tensors stay on the GPU.
    points = points.contiguous()
    mask = mask.contiguous()
    starts = points.norm(dim=-1).masked_fill(~mask, -1.0).argmax(dim=1)
    counts = mask.sum(dim=1, dtype=torch.int32)
    # Stable permutation: valid slots first, then invalid slots. Prefix ranks
    # preserve the original pool order, hence the same first-index tie rule.
    # Full fixed shapes avoid nonzero()/CPU count reads even for mixed lengths.
    valid_rank = mask.cumsum(dim=1, dtype=torch.int64) - 1
    invalid_rank = (~mask).cumsum(dim=1, dtype=torch.int64) - 1
    ranks = torch.where(mask, valid_rank, counts[:, None] + invalid_rank)
    order = torch.empty_like(ranks)
    source = torch.arange(size, device=points.device)[None, :].expand(batch, -1)
    order.scatter_(1, ranks, source)
    compact = points.gather(1, order[:, :, None].expand(-1, -1, 3))
    compact_starts = valid_rank.gather(1, starts[:, None]).flatten().clamp_min(0)
    from ._sampling_warp import launch_compact_fps
    selected = launch_compact_fps(compact, counts, compact_starts, k)
    original = order.gather(1, selected.clamp_min(0))
    return torch.where(selected >= 0, original, -1)


@torch.no_grad()
def farthest_point_sampling_padded(points, lengths, k):
    """Prefix-length batch convenience API; output slots beyond length are -1."""
    if not torch.is_tensor(points) or points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("points must have shape (B,N,3)")
    if points.dtype != torch.float32 or points.device.type not in ("cpu", "cuda"):
        raise ValueError("points must be float32 on CPU or CUDA")
    if (not torch.is_tensor(lengths) or lengths.dtype not in (torch.int32, torch.int64)
            or lengths.shape != points.shape[:1] or lengths.device != points.device):
        raise ValueError("lengths must be an integer (B,) tensor on the points device")
    if bool(((lengths < 0) | (lengths > points.shape[1])).any()):
        raise ValueError("lengths must be inside [0,N]")
    mask = torch.arange(points.shape[1], device=points.device)[None, :] < lengths[:, None]
    return farthest_point_sampling_masked(points, mask, k)


@torch.no_grad()
def farthest_point_sampling_indices(points, group_mask, k):
    """Single-cloud compatibility API, returning unpadded indices and an int count.

    Reading that count synchronizes once on CUDA. The scanner uses the fixed
    shape masked batch API instead, so this convenience read is not a hot path.
    """
    _validate(points, group_mask, k, batched=False)
    if points.device.type == "cpu":
        return farthest_point_sampling_indices_reference(points, group_mask, k)
    count = min(k, int(group_mask.sum().item()))
    if count == 0:
        return torch.empty(0, dtype=torch.long, device=points.device), 0
    indices = farthest_point_sampling_masked(points[None], group_mask[None], k)[0]
    return indices[:count], count
