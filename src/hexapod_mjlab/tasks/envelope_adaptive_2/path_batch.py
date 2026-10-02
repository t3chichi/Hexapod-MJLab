"""Padded device-side path batch for ``el4090_ea2``.

``PathBatch`` mirrors the Python ``PathData`` objects used by the environment
into padded GPU tensors.  This is the data layer required by the batched
kinematics rewrite; it does not by itself change any training behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .contracts import PathData

DEFAULT_MAX_POINTS = 768
DEFAULT_MAX_CORNERS = 256


@dataclass
class PathQueryResult:
    """Result of a batched path query for all envs."""

    idx: torch.Tensor
    xy: torch.Tensor
    tangent: torch.Tensor
    next_corner_idx: torch.Tensor
    next_corner: torch.Tensor
    next_target: torch.Tensor
    has_next_corner: torch.Tensor


class PathBatch:
    """Per-environment padded path tensors.

    The Python :class:`PathData` objects remain the environment's source of
    truth; this class is a device-side mirror that is written through
    :meth:`install` and read by the future batched kinematics code.
    """

    def __init__(
        self,
        num_envs: int,
        max_points: int,
        max_corners: int,
        device: torch.device,
    ):
        self.num_envs = int(num_envs)
        self.max_points = int(max_points)
        self.max_corners = int(max_corners)
        self.device = device

        self.valid = torch.zeros(
            self.num_envs, dtype=torch.bool, device=device
        )
        self.lengths = torch.zeros(
            self.num_envs, dtype=torch.long, device=device
        )
        self.corner_lengths = torch.zeros(
            self.num_envs, dtype=torch.long, device=device
        )
        self.has_seg = torch.zeros(
            self.num_envs, dtype=torch.bool, device=device
        )

        self.points = torch.zeros(
            self.num_envs, self.max_points, 2, dtype=torch.float32, device=device
        )
        self.arc = torch.full(
            (self.num_envs, self.max_points),
            float("inf"),
            dtype=torch.float32,
            device=device,
        )
        self.yaws = torch.zeros(
            self.num_envs, self.max_points, dtype=torch.float32, device=device
        )
        self.seg_dirs = torch.zeros(
            self.num_envs, self.max_points, dtype=torch.float32, device=device
        )

        self.corner_arcs = torch.full(
            (self.num_envs, self.max_corners),
            float("inf"),
            dtype=torch.float32,
            device=device,
        )
        self.corner_targets = torch.zeros(
            self.num_envs, self.max_corners, dtype=torch.float32, device=device
        )

    def ensure_capacity(self, need_points: int, need_corners: int) -> None:
        """Grow the padded capacity while preserving existing rows."""
        need_points = max(1, int(need_points))
        need_corners = max(0, int(need_corners))

        new_p = self.max_points
        new_k = self.max_corners
        if need_points > new_p:
            new_p = max(2 * self.max_points, need_points)
        if need_corners > new_k:
            new_k = max(2 * self.max_corners, need_corners)
        if new_p == self.max_points and new_k == self.max_corners:
            return

        old_p = self.max_points
        old_k = self.max_corners

        new_points = torch.zeros(
            self.num_envs, new_p, 2, dtype=torch.float32, device=self.device
        )
        new_points[:, :old_p].copy_(self.points)
        self.points = new_points

        new_arc = torch.full(
            (self.num_envs, new_p),
            float("inf"),
            dtype=torch.float32,
            device=self.device,
        )
        new_arc[:, :old_p].copy_(self.arc)
        self.arc = new_arc

        new_yaws = torch.zeros(
            self.num_envs, new_p, dtype=torch.float32, device=self.device
        )
        new_yaws[:, :old_p].copy_(self.yaws)
        self.yaws = new_yaws

        new_seg = torch.zeros(
            self.num_envs, new_p, dtype=torch.float32, device=self.device
        )
        new_seg[:, :old_p].copy_(self.seg_dirs)
        self.seg_dirs = new_seg

        new_corner_arcs = torch.full(
            (self.num_envs, new_k),
            float("inf"),
            dtype=torch.float32,
            device=self.device,
        )
        new_corner_arcs[:, :old_k].copy_(self.corner_arcs)
        self.corner_arcs = new_corner_arcs

        new_corner_targets = torch.zeros(
            self.num_envs, new_k, dtype=torch.float32, device=self.device
        )
        new_corner_targets[:, :old_k].copy_(self.corner_targets)
        self.corner_targets = new_corner_targets

        self.max_points = new_p
        self.max_corners = new_k

    def install(self, env_id: int, path: Optional[PathData]) -> None:
        """Mirror one :class:`PathData` into row ``env_id``."""
        if path is None:
            self.invalidate(env_id)
            return

        points = np.asarray(path.points, dtype=np.float32)
        arc = np.asarray(path.arc, dtype=np.float32)
        yaws = np.asarray(path.yaws, dtype=np.float32)
        n = int(points.shape[0])
        if n <= 0:
            self.invalidate(env_id)
            return

        need_corners = 0
        if path.corner_arcs is not None:
            need_corners = int(len(path.corner_arcs))
        self.ensure_capacity(n, need_corners)

        self.points[env_id, :n].copy_(torch.from_numpy(points).to(self.device))
        self.arc[env_id, :n].copy_(torch.from_numpy(arc).to(self.device))
        self.yaws[env_id, :n].copy_(torch.from_numpy(yaws).to(self.device))
        self.arc[env_id, n:].fill_(float("inf"))

        if path.segment_dirs is not None and len(path.segment_dirs) > 0:
            seg = np.asarray(path.segment_dirs, dtype=np.float32)
            self.seg_dirs[env_id, : len(seg)].copy_(
                torch.from_numpy(seg).to(self.device)
            )
            self.has_seg[env_id] = True
        else:
            self.has_seg[env_id] = False

        if (
            path.corner_arcs is not None
            and path.corner_targets is not None
            and len(path.corner_arcs) > 0
        ):
            ca = np.asarray(path.corner_arcs, dtype=np.float32)
            ct = np.asarray(path.corner_targets, dtype=np.float32)
            self.corner_arcs[env_id, : len(ca)].copy_(
                torch.from_numpy(ca).to(self.device)
            )
            self.corner_targets[env_id, : len(ct)].copy_(
                torch.from_numpy(ct).to(self.device)
            )
            self.corner_arcs[env_id, len(ca):].fill_(float("inf"))
            self.corner_lengths[env_id] = len(ca)
        else:
            self.corner_lengths[env_id] = 0

        self.lengths[env_id] = n
        self.valid[env_id] = True

    def invalidate(self, env_id: int) -> None:
        """Mark row ``env_id`` as not holding a path."""
        self.valid[env_id] = False
        self.lengths[env_id] = 0
        self.corner_lengths[env_id] = 0
        self.has_seg[env_id] = False

    def query(self, s: torch.Tensor) -> PathQueryResult:
        """Query interpolation and next-corner info for all envs.

        The caller is responsible for masking invalid rows / inactive envs;
        this function is safe (no crash / no out-of-bounds) even when some
        rows are invalid.
        """
        s = s.to(device=self.device, dtype=torch.float32)
        num_envs = self.num_envs
        arange = torch.arange(num_envs, device=self.device)

        # Interpolation segment index.  arc tails are inf, so this count is
        # exactly the number of finite arc entries <= s.
        idx = (self.arc <= s[:, None]).sum(dim=1) - 1
        idx = idx.clamp(0, max(0, self.max_points - 2))

        a0 = self.arc[arange, idx]
        a1 = self.arc[arange, idx + 1]
        p0 = self.points[arange, idx]
        p1 = self.points[arange, idx + 1]

        # Guard against invalid rows (arc all inf) and end-of-path rows
        # (idx at the last finite sample) so the query never propagates NaN.
        valid_row = self.valid
        finite_a1 = torch.isfinite(a1)
        safe_a0 = torch.where(valid_row, a0, torch.zeros_like(s))
        safe_a1 = torch.where(
            valid_row & finite_a1,
            a1,
            torch.where(valid_row, safe_a0 + 1e-3, torch.ones_like(s)),
        )
        safe_p1 = torch.where(
            (valid_row & finite_a1).unsqueeze(-1), p1, p0
        )

        ds = (safe_a1 - safe_a0).clamp_min(1e-12)
        t = ((s - safe_a0) / ds).clamp(0.0, 1.0)
        xy = p0 * (1.0 - t.unsqueeze(-1)) + safe_p1 * t.unsqueeze(-1)

        # Segment direction: prefer segment_dirs; fallback to yaws[idx].
        max_seg_idx = (self.lengths - 2).clamp(min=0)
        seg_idx = torch.minimum(idx, max_seg_idx)
        tangent = torch.where(
            self.has_seg,
            self.seg_dirs[arange, seg_idx],
            self.yaws[arange, idx],
        )

        # Next corner: count corners already crossed (<= s).
        corner_count = (self.corner_arcs <= s[:, None]).sum(dim=1)
        next_corner_idx = corner_count.clamp(0, max(0, self.max_corners - 1))
        next_corner = self.corner_arcs[arange, next_corner_idx]
        next_target = self.corner_targets[arange, next_corner_idx]
        has_next_corner = corner_count < self.corner_lengths

        return PathQueryResult(
            idx=idx,
            xy=xy,
            tangent=tangent,
            next_corner_idx=next_corner_idx,
            next_corner=next_corner,
            next_target=next_target,
            has_next_corner=has_next_corner,
        )

    def assert_invariants(self) -> None:
        """Debug consistency check (intended for tests/debug only)."""
        for i in range(self.num_envs):
            if not bool(self.valid[i]):
                if int(self.lengths[i]) != 0:
                    raise AssertionError(
                        f"env {i}: invalid row with non-zero length"
                    )
                continue

            n = int(self.lengths[i])
            if not (1 <= n <= self.max_points):
                raise AssertionError(
                    f"env {i}: bad path length {n}"
                )

            if not bool(torch.isfinite(self.arc[i, :n]).all()):
                raise AssertionError(f"env {i}: non-finite arc prefix")
            if not bool((self.arc[i, :n].diff() >= -1e-6).all()):
                raise AssertionError(f"env {i}: arc not monotonic")

            if not bool(torch.isinf(self.arc[i, n:]).all()):
                raise AssertionError(f"env {i}: arc tail is not inf")

            if bool(self.has_seg[i]) and n < 2:
                raise AssertionError(
                    f"env {i}: has_seg with path length {n}"
                )

            k = int(self.corner_lengths[i])
            if k > 0:
                if not bool(torch.isfinite(self.corner_arcs[i, :k]).all()):
                    raise AssertionError(
                        f"env {i}: non-finite corner prefix"
                    )
                if not bool(
                    (self.corner_arcs[i, :k].diff() >= -1e-6).all()
                ):
                    raise AssertionError(
                        f"env {i}: corner_arcs not monotonic"
                    )
                if not bool(torch.isinf(self.corner_arcs[i, k:]).all()):
                    raise AssertionError(
                        f"env {i}: corner tail is not inf"
                    )
