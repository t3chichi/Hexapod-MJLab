"""Contracts for the single-scan recurrent perception variant.

The observation and envelope-label types are shared with the original task so
rollout and label producers remain wire-compatible.  Only the scan-window
shape changes here: one current far scan of 128 points accompanies the current
near scan of 256 points.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..confined_locomotion.contracts import (
    COMMAND_DIM,
    EMPTY_SCAN_ID,
    ENVELOPE_DIM,
    FLOAT_DTYPE,
    INT_DTYPE,
    JOINT_DIM,
    NEAR_POINTS,
    POINT_CHANNELS,
    PROPRIO_DIM,
    PROPRIO_LAYOUT,
    ContractError,
    EnvelopeLabel,
    IdentityMismatchError,
    OnlineObservation,
    OrderedPoints,
    bind_envelope_label,
    bind_window_to_observation,
    sort_points_by_angle_keys,
    validate_point_block,
)

FAR_POINTS_PER_SCAN = 128
FAR_WINDOW_SLOTS = 1
FRAME_GRU_HIDDEN = 187
TEMPORAL_GRU_HIDDEN = 187
TEMPORAL_GRU_INPUT = FRAME_GRU_HIDDEN + 9
ACTION_DIM = 18
CONTROL_INPUT_DIM = TEMPORAL_GRU_HIDDEN + PROPRIO_DIM + COMMAND_DIM + ENVELOPE_DIM


@dataclass(frozen=True)
class HeightLabels:
    env_ids: torch.Tensor
    episode_ids: torch.Tensor
    step_ids: torch.Tensor
    heights: torch.Tensor             # [B,11,17], metres, signed
    valid: torch.Tensor               # [B,11,17], geometry only
    query_xy_w: torch.Tensor          # [B,11,17,2], offline audit
    root_pos_w: torch.Tensor          # [B,3], offline audit, never an observation
    grid_identity: dict
    terrain_identity: str

    def validate(self, observation):
        """Bind training-only labels to the exact pre-action observation."""
        b, device = len(observation.env_ids), observation.env_ids.device
        for name in ("env_ids", "episode_ids", "step_ids"):
            value = getattr(self, name)
            if (not isinstance(value, torch.Tensor) or value.dtype != torch.int64 or value.device != device or
                    value.shape != (b,) or not torch.equal(value, getattr(observation, name))):
                raise ContractError(f"height label {name} must match its pre-action observation")
        for name, shape, dtype in (
            ("heights", (b, 11, 17), torch.float32), ("valid", (b, 11, 17), torch.bool),
            ("query_xy_w", (b, 11, 17, 2), torch.float32), ("root_pos_w", (b, 3), torch.float32),
        ):
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor) or value.shape != shape or value.dtype != dtype or value.device != device:
                raise ContractError(f"height label {name} must be {shape} {dtype} on {device}")
            if dtype.is_floating_point and not bool(torch.isfinite(value).all()):
                raise ContractError(f"height label {name} contains NaN/Inf")
        if bool((self.heights[~self.valid] != 0).any()):
            raise ContractError("invalid height label slots must be zero")
        if not isinstance(self.terrain_identity, str) or not self.terrain_identity or not self.grid_identity:
            raise ContractError("height label requires terrain and grid identities")


@dataclass
class ScanWindowSnapshot:
    """Validated one-scan snapshot, with near and far data from the same scan."""

    env_ids: torch.Tensor
    episode_ids: torch.Tensor
    near_points: torch.Tensor
    near_valid: torch.Tensor
    near_filled: torch.Tensor
    near_scan_ids: torch.Tensor
    near_times: torch.Tensor
    far_points: torch.Tensor
    far_valid: torch.Tensor
    far_filled: torch.Tensor
    far_scan_ids: torch.Tensor
    far_times: torch.Tensor

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.env_ids, torch.Tensor) or self.env_ids.dim() != 1 \
                or self.env_ids.dtype is not INT_DTYPE or self.env_ids.numel() < 1:
            raise ContractError("env_ids must be a nonempty (B,) int64 tensor")
        batch = int(self.env_ids.shape[0])
        device = self.env_ids.device
        if bool((self.env_ids < 0).any()) or torch.unique(self.env_ids).numel() != batch:
            raise ContractError("snapshot env_ids must be nonnegative and unique")

        expected = (
            ("episode_ids", self.episode_ids, INT_DTYPE, (batch,)),
            ("near_points", self.near_points, FLOAT_DTYPE, (batch, NEAR_POINTS, POINT_CHANNELS)),
            ("near_valid", self.near_valid, torch.bool, (batch, NEAR_POINTS)),
            ("near_filled", self.near_filled, torch.bool, (batch,)),
            ("near_scan_ids", self.near_scan_ids, INT_DTYPE, (batch,)),
            ("near_times", self.near_times, FLOAT_DTYPE, (batch,)),
            ("far_points", self.far_points, FLOAT_DTYPE,
             (batch, FAR_WINDOW_SLOTS, FAR_POINTS_PER_SCAN, POINT_CHANNELS)),
            ("far_valid", self.far_valid, torch.bool,
             (batch, FAR_WINDOW_SLOTS, FAR_POINTS_PER_SCAN)),
            ("far_filled", self.far_filled, torch.bool, (batch, FAR_WINDOW_SLOTS)),
            ("far_scan_ids", self.far_scan_ids, INT_DTYPE, (batch, FAR_WINDOW_SLOTS)),
            ("far_times", self.far_times, FLOAT_DTYPE, (batch, FAR_WINDOW_SLOTS)),
        )
        for name, value, dtype, shape in expected:
            if not isinstance(value, torch.Tensor) or value.dtype is not dtype \
                    or tuple(value.shape) != shape or value.device != device:
                raise ContractError(f"{name} must be {shape} {dtype} on {device}")
        if bool((self.episode_ids < 0).any()):
            raise ContractError("episode_ids must be nonnegative")
        if not bool(torch.isfinite(self.near_times).all()) \
                or not bool(torch.isfinite(self.far_times).all()):
            raise ContractError("scan times must be finite")

        validate_point_block(self.near_points, self.near_valid,
                             n_points=NEAR_POINTS, name="near_points")
        validate_point_block(
            self.far_points.reshape(batch * FAR_WINDOW_SLOTS, FAR_POINTS_PER_SCAN, POINT_CHANNELS),
            self.far_valid.reshape(batch * FAR_WINDOW_SLOTS, FAR_POINTS_PER_SCAN),
            n_points=FAR_POINTS_PER_SCAN,
            name="far_points",
        )

        no_near = ~self.near_filled
        if bool(no_near.any()):
            if bool(self.near_valid[no_near].any()):
                raise ContractError("empty near rows must have no valid points")
            if bool((self.near_scan_ids[no_near] != EMPTY_SCAN_ID).any()):
                raise ContractError("empty near rows must use EMPTY_SCAN_ID")
            if bool((self.near_times[no_near] != 0).any()):
                raise ContractError("empty near rows must have time zero")
        if bool(self.near_filled.any()) and bool((self.near_scan_ids[self.near_filled] < 0).any()):
            raise ContractError("filled near rows require nonnegative scan ids")

        filled = self.far_filled[:, 0]
        if not torch.equal(filled, self.near_filled):
            raise ContractError("near and far fill state must describe the same scan")
        no_far = ~filled
        if bool(no_far.any()):
            if bool(self.far_valid[no_far].any()):
                raise ContractError("empty far rows must have no valid points")
            if bool((self.far_scan_ids[no_far] != EMPTY_SCAN_ID).any()):
                raise ContractError("empty far rows must use EMPTY_SCAN_ID")
            if bool((self.far_times[no_far] != 0).any()):
                raise ContractError("empty far rows must have time zero")
        if bool(filled.any()):
            if bool((self.far_scan_ids[:, 0][filled] < 0).any()):
                raise ContractError("filled far rows require nonnegative scan ids")
            if not torch.equal(self.near_scan_ids[filled], self.far_scan_ids[:, 0][filled]):
                raise ContractError("near and far scan ids must match")
            if not torch.equal(self.near_times[filled], self.far_times[:, 0][filled]):
                raise ContractError("near and far scan times must match")


def flatten_far_window(window: ScanWindowSnapshot) -> OrderedPoints:
    """Return the one current far scan as `(B,128,3)` with point identities."""
    window.validate()
    batch = int(window.far_points.shape[0])
    points = window.far_points.reshape(batch, FAR_POINTS_PER_SCAN, POINT_CHANNELS)
    valid = window.far_valid.reshape(batch, FAR_POINTS_PER_SCAN)
    scan_ids = window.far_scan_ids[:, 0, None].expand(batch, FAR_POINTS_PER_SCAN)
    times = window.far_times[:, 0, None].expand(batch, FAR_POINTS_PER_SCAN)
    return OrderedPoints(
        points=points,
        valid=valid,
        scan_ids=torch.where(valid, scan_ids, torch.full_like(scan_ids, EMPTY_SCAN_ID)),
        times=torch.where(valid, times, torch.zeros_like(times)),
    )


__all__ = [
    "ACTION_DIM", "COMMAND_DIM", "CONTROL_INPUT_DIM", "EMPTY_SCAN_ID",
    "ENVELOPE_DIM", "FAR_POINTS_PER_SCAN", "FAR_WINDOW_SLOTS",
    "FLOAT_DTYPE", "FRAME_GRU_HIDDEN", "IdentityMismatchError", "INT_DTYPE",
    "JOINT_DIM", "NEAR_POINTS", "OnlineObservation", "EnvelopeLabel",
    "OrderedPoints", "POINT_CHANNELS", "PROPRIO_DIM", "PROPRIO_LAYOUT",
    "ScanWindowSnapshot", "TEMPORAL_GRU_HIDDEN", "TEMPORAL_GRU_INPUT",
    "ContractError", "HeightLabels", "bind_envelope_label", "bind_window_to_observation",
    "flatten_far_window", "sort_points_by_angle_keys", "validate_point_block",
]
