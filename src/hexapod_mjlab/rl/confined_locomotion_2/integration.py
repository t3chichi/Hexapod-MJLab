"""Prepare current scan-window data for the recurrent controller."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .contracts import (
    EMPTY_SCAN_ID,
    FAR_POINTS_PER_SCAN,
    INT_DTYPE,
    NEAR_POINTS,
    ContractError,
    EnvelopeLabel,
    IdentityMismatchError,
    OnlineObservation,
    ScanWindowSnapshot,
    bind_envelope_label,
    bind_window_to_observation,
    flatten_far_window,
    sort_points_by_angle_keys,
)

__all__ = [
    "BoundTrainingLabels",
    "PreparedControlInputs",
    "bind_offline_labels",
    "prepare_control_inputs",
]


@dataclass(frozen=True)
class PreparedControlInputs:
    """Current near/far scan pair aligned with one control observation."""

    observation: OnlineObservation
    near_points: torch.Tensor
    near_valid: torch.Tensor
    far_points: torch.Tensor
    far_valid: torch.Tensor
    near_filled: torch.Tensor
    near_scan_ids: torch.Tensor
    near_times: torch.Tensor
    far_point_scan_ids: torch.Tensor
    far_point_times: torch.Tensor
    near_angle_keys: torch.Tensor | None = None
    far_angle_keys: torch.Tensor | None = None
    near_source_dirs_sensor: torch.Tensor | None = None
    far_source_dirs_sensor: torch.Tensor | None = None
    near_source_dirs_present: torch.Tensor | None = None
    far_source_dirs_present: torch.Tensor | None = None
    near_source_indices: torch.Tensor | None = None
    far_source_indices: torch.Tensor | None = None


def _lookup_window_rows(window: ScanWindowSnapshot, observation: OnlineObservation) -> torch.Tensor:
    equal = observation.env_ids[:, None] == window.env_ids[None, :]
    counts = equal.sum(dim=1)
    if bool((counts != 1).any()):
        raise IdentityMismatchError("every observation env_id must occur exactly once in the scan window")
    return equal.to(INT_DTYPE).argmax(dim=1)


def _identity_order(batch: int, count: int, device: torch.device) -> torch.Tensor:
    return torch.arange(count, dtype=INT_DTYPE, device=device).expand(batch, count)


def _validate_angle_keys(
    name: str, keys: torch.Tensor | None, *, batch: int, count: int, device: torch.device
) -> None:
    if keys is None:
        return
    if not isinstance(keys, torch.Tensor) or keys.dtype is not torch.float32 \
            or tuple(keys.shape) != (batch, count) or keys.device != device:
        raise ContractError(f"{name} must be ({batch}, {count}) float32 on {device}")


def _gather(order: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    if values.dim() == 3:
        return torch.gather(values, 1, order.unsqueeze(-1).expand(-1, -1, values.shape[-1]))
    return torch.gather(values, 1, order)


def _validate_source_metadata(
    *,
    label: str,
    directions: torch.Tensor,
    present: torch.Tensor,
    indices: torch.Tensor,
    valid: torch.Tensor,
    device: torch.device,
) -> None:
    batch, count = valid.shape
    if not isinstance(directions, torch.Tensor) or directions.dtype is not torch.float32 \
            or tuple(directions.shape) != (batch, count, 3) or directions.device != device:
        raise ContractError(f"{label}_source_dirs_sensor must be ({batch}, {count}, 3) float32 on {device}")
    if not isinstance(present, torch.Tensor) or present.dtype is not torch.bool \
            or tuple(present.shape) != (batch, count) or present.device != device:
        raise ContractError(f"{label}_source_dirs_present must be ({batch}, {count}) bool on {device}")
    if not isinstance(indices, torch.Tensor) or indices.dtype is not INT_DTYPE \
            or tuple(indices.shape) != (batch, count) or indices.device != device:
        raise ContractError(f"{label}_source_indices must be ({batch}, {count}) int64 on {device}")
    if bool((present & ~valid).any()) or not torch.equal(present, indices >= 0):
        raise ContractError(f"{label} source presence must match valid points and nonnegative ray ids")
    if bool(present.any()):
        selected = directions[present]
        if not bool(torch.isfinite(selected).all()):
            raise ContractError(f"{label} source directions contain NaN/Inf")
        norms = torch.linalg.vector_norm(selected, dim=-1)
        if not bool(torch.allclose(norms, torch.ones_like(norms), atol=1e-5, rtol=0.0)):
            raise ContractError(f"{label} source directions must be unit vectors")
    if bool((~present).any()) and not torch.equal(directions[~present], torch.zeros_like(directions[~present])):
        raise ContractError(f"{label} missing source directions must be exactly zero")


def prepare_control_inputs(
    *,
    window: ScanWindowSnapshot,
    observation: OnlineObservation,
    near_angle_keys: torch.Tensor | None,
    far_angle_keys: torch.Tensor | None,
    near_source_dirs_sensor: torch.Tensor | None = None,
    far_source_dirs_sensor: torch.Tensor | None = None,
    near_source_dirs_present: torch.Tensor | None = None,
    far_source_dirs_present: torch.Tensor | None = None,
    near_source_indices: torch.Tensor | None = None,
    far_source_indices: torch.Tensor | None = None,
) -> PreparedControlInputs:
    """Bind identities and sort near/far points together by stable angle keys.

    Both blocks are from the current scan. Angle keys, when supplied, are in
    observation row order; provenance arrays are in window row order, matching
    ``ScanWindow.read_source_directions()``. Passing both keys as ``None`` keeps
    their original slot order.
    """
    if (near_angle_keys is None) != (far_angle_keys is None):
        raise ContractError("near and far angle keys must be supplied together or both omitted")
    bind_window_to_observation(window, observation)
    rows = _lookup_window_rows(window, observation)
    batch = int(observation.env_ids.shape[0])
    device = observation.env_ids.device
    _validate_angle_keys("near_angle_keys", near_angle_keys, batch=batch,
                         count=NEAR_POINTS, device=device)
    _validate_angle_keys("far_angle_keys", far_angle_keys, batch=batch,
                         count=FAR_POINTS_PER_SCAN, device=device)

    near_raw = window.near_points.index_select(0, rows)
    near_valid = window.near_valid.index_select(0, rows)
    flat_far = flatten_far_window(window)
    far_raw = flat_far.points.index_select(0, rows)
    far_valid = flat_far.valid.index_select(0, rows)

    if near_angle_keys is None:
        near_order = _identity_order(batch, NEAR_POINTS, device)
        far_order = _identity_order(batch, FAR_POINTS_PER_SCAN, device)
        near_points = near_raw.clone()
        far_points = far_raw.clone()
        near_angle_sorted = None
        far_angle_sorted = None
    else:
        near_order = torch.argsort(
            torch.where(near_valid, near_angle_keys, torch.full_like(near_angle_keys, float("inf"))),
            dim=1, stable=True,
        )
        far_order = torch.argsort(
            torch.where(far_valid, far_angle_keys, torch.full_like(far_angle_keys, float("inf"))),
            dim=1, stable=True,
        )
        near_result = sort_points_by_angle_keys(near_raw, near_valid, near_angle_keys)
        far_result = sort_points_by_angle_keys(
            far_raw,
            far_valid,
            far_angle_keys,
            scan_ids=flat_far.scan_ids.index_select(0, rows),
            times=flat_far.times.index_select(0, rows),
        )
        near_points = near_result.points
        far_points = far_result.points
        near_valid = near_result.valid
        far_valid = far_result.valid
        near_angle_sorted = _gather(near_order, near_angle_keys)
        far_angle_sorted = _gather(far_order, far_angle_keys)
        far_scan_ids = far_result.scan_ids
        far_times = far_result.times

    if near_angle_keys is None:
        far_scan_ids = flat_far.scan_ids.index_select(0, rows)
        far_times = flat_far.times.index_select(0, rows)

    metadata = (
        near_source_dirs_sensor, far_source_dirs_sensor,
        near_source_dirs_present, far_source_dirs_present,
        near_source_indices, far_source_indices,
    )
    if any(value is None for value in metadata) and any(value is not None for value in metadata):
        raise ContractError("near/far source directions, presence and ray ids must be supplied together")
    near_dirs_sorted = far_dirs_sorted = None
    near_present_sorted = far_present_sorted = None
    near_indices_sorted = far_indices_sorted = None
    if near_source_dirs_sensor is not None:
        window_batch = int(window.env_ids.shape[0])
        expected_sources = (
            ("near_source_dirs_sensor", near_source_dirs_sensor,
             (window_batch, NEAR_POINTS, 3), torch.float32),
            ("far_source_dirs_sensor", far_source_dirs_sensor,
             (window_batch, FAR_POINTS_PER_SCAN, 3), torch.float32),
            ("near_source_dirs_present", near_source_dirs_present,
             (window_batch, NEAR_POINTS), torch.bool),
            ("far_source_dirs_present", far_source_dirs_present,
             (window_batch, FAR_POINTS_PER_SCAN), torch.bool),
            ("near_source_indices", near_source_indices,
             (window_batch, NEAR_POINTS), INT_DTYPE),
            ("far_source_indices", far_source_indices,
             (window_batch, FAR_POINTS_PER_SCAN), INT_DTYPE),
        )
        for name, value, shape, dtype in expected_sources:
            if not isinstance(value, torch.Tensor) or value.dtype is not dtype \
                    or tuple(value.shape) != shape or value.device != device:
                raise ContractError(f"{name} must be {shape} {dtype} on {device}")
        near_dirs = near_source_dirs_sensor.index_select(0, rows)
        far_dirs = far_source_dirs_sensor.index_select(0, rows)
        near_present = near_source_dirs_present.index_select(0, rows)
        far_present = far_source_dirs_present.index_select(0, rows)
        near_indices = near_source_indices.index_select(0, rows)
        far_indices = far_source_indices.index_select(0, rows)
        _validate_source_metadata(
            label="near", directions=near_dirs, present=near_present,
            indices=near_indices, valid=window.near_valid.index_select(0, rows), device=device,
        )
        _validate_source_metadata(
            label="far", directions=far_dirs, present=far_present,
            indices=far_indices, valid=flat_far.valid.index_select(0, rows), device=device,
        )
        # The source arrays are stored in window row order and original point
        # slot order, so gather the observation-aligned sort permutation.
        near_dirs_sorted = _gather(near_order, near_dirs)
        far_dirs_sorted = _gather(far_order, far_dirs)
        near_present_sorted = _gather(near_order, near_present)
        far_present_sorted = _gather(far_order, far_present)
        near_indices_sorted = _gather(near_order, near_indices)
        far_indices_sorted = _gather(far_order, far_indices)

    return PreparedControlInputs(
        observation=observation,
        near_points=near_points,
        near_valid=near_valid,
        far_points=far_points,
        far_valid=far_valid,
        near_filled=window.near_filled.index_select(0, rows).clone(),
        near_scan_ids=window.near_scan_ids.index_select(0, rows).clone(),
        near_times=window.near_times.index_select(0, rows).clone(),
        far_point_scan_ids=far_scan_ids,
        far_point_times=far_times,
        near_angle_keys=near_angle_sorted,
        far_angle_keys=far_angle_sorted,
        near_source_dirs_sensor=near_dirs_sorted,
        far_source_dirs_sensor=far_dirs_sorted,
        near_source_dirs_present=near_present_sorted,
        far_source_dirs_present=far_present_sorted,
        near_source_indices=near_indices_sorted,
        far_source_indices=far_indices_sorted,
    )


@dataclass(frozen=True)
class BoundTrainingLabels:
    envelope: EnvelopeLabel | None = None


def bind_offline_labels(
    observation: OnlineObservation, *, envelope_label: EnvelopeLabel | None = None
) -> BoundTrainingLabels:
    """Bind the only label used in this network, the six-dimensional envelope."""
    if envelope_label is None:
        raise ContractError("an envelope_label is required")
    return BoundTrainingLabels(envelope=bind_envelope_label(observation, envelope_label))
