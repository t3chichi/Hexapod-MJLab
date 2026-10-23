"""No-grad runtime cache for the point-cloud encoders.

The cache is deliberately an ordinary Python object, separate from the network
module and its checkpoint state.  It memoizes only near/far GRU embeddings;
observation-dependent morphology, action, value, and optional ground heads run
on every call.  The network's normal ``forward`` remains the fresh, differentiable
path used by PPO.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .contracts import ContractError, INT_DTYPE
from .integration import PreparedControlInputs
from .network import (
    FAR_GRU_HIDDEN,
    FAR_SEQUENCE_POINTS,
    NEAR_GRU_HIDDEN,
    ConfinedLocomotionNetwork,
    NetworkOutput,
    PointCloudFeatures,
)

__all__ = ["PointCloudFeatureCache"]


@dataclass
class _NearEntry:
    episode_id: int
    scan_id: int
    filled: bool
    points: torch.Tensor
    valid: torch.Tensor
    feature: torch.Tensor


@dataclass
class _FarEntry:
    episode_id: int
    scan_ids: torch.Tensor
    points: torch.Tensor
    valid: torch.Tensor
    feature: torch.Tensor


class PointCloudFeatureCache:
    """Cache point-cloud encoder outputs for no-grad control inference.

    Each real environment has at most one near and one far cache entry.  A
    cache match requires matching episode and scan identity *and* exact equality
    of the ordered point values and validity mask.  This catches callers that
    accidentally reuse a scan ID for changed coordinates, masking, or order.

    ``PreparedControlInputs`` is treated as an immutable per-call snapshot.  The
    cache never reads the live scanner/window/runtime, so a delayed bootstrap
    using an old episode snapshot is checked against that snapshot's own IDs and
    point tensors.
    """

    def __init__(self, network: ConfinedLocomotionNetwork) -> None:
        if not isinstance(network, ConfinedLocomotionNetwork):
            raise TypeError("network must be a ConfinedLocomotionNetwork")
        self.network = network
        self._near: dict[int, _NearEntry] = {}
        self._far: dict[int, _FarEntry] = {}
        self._parameter_signature = self._current_parameter_signature()
        self._stats: dict[str, int] = {
            "near_rows_encoded": 0,
            "far_rows_encoded": 0,
            "near_hits": 0,
            "far_hits": 0,
            "forward_calls": 0,
        }

    @property
    def stats(self) -> dict[str, int]:
        """Return a copy of cumulative cache counters."""
        return dict(self._stats)

    def reset_stats(self) -> None:
        """Reset counters without changing any cached features."""
        for name in self._stats:
            self._stats[name] = 0

    def clear(self) -> None:
        """Drop all cached rows while preserving cumulative counters."""
        self._near.clear()
        self._far.clear()
        self._parameter_signature = self._current_parameter_signature()

    def _encoder_signature(self, encoder: torch.nn.Module) -> tuple[tuple[object, ...], ...]:
        signature = []
        for parameter in encoder.parameters():
            signature.append(
                (
                    id(parameter),
                    int(parameter._version),
                    parameter.device,
                    parameter.dtype,
                )
            )
        return tuple(signature)

    def _current_parameter_signature(self) -> tuple[object, ...]:
        return (
            self._encoder_signature(self.network.near_encoder),
            self._encoder_signature(self.network.far_encoder),
        )

    def _clear_if_parameters_changed(self) -> None:
        signature = self._current_parameter_signature()
        if signature != self._parameter_signature:
            self._near.clear()
            self._far.clear()
            self._parameter_signature = signature

    @staticmethod
    def _validate_prepared_metadata(prepared: PreparedControlInputs) -> None:
        observation = prepared.observation
        batch = int(observation.env_ids.shape[0])
        device = observation.env_ids.device
        expected = (
            ("near_filled", prepared.near_filled, torch.bool, (batch,)),
            ("near_scan_ids", prepared.near_scan_ids, INT_DTYPE, (batch,)),
            (
                "far_point_scan_ids",
                prepared.far_point_scan_ids,
                INT_DTYPE,
                (batch, FAR_SEQUENCE_POINTS),
            ),
        )
        for name, tensor, dtype, shape in expected:
            if not isinstance(tensor, torch.Tensor):
                raise ContractError(f"prepared.{name} must be a tensor")
            if tensor.dtype is not dtype or tuple(tensor.shape) != shape:
                raise ContractError(
                    f"prepared.{name} must be {shape}/{dtype}, got "
                    f"{tuple(tensor.shape)}/{tensor.dtype}"
                )
            if tensor.device != device:
                raise ContractError(
                    f"prepared.{name} device {tensor.device} differs from observation {device}"
                )

    @staticmethod
    def _same_near(
        entry: _NearEntry | None,
        *,
        episode_id: int,
        scan_id: int,
        filled: bool,
        points: torch.Tensor,
        valid: torch.Tensor,
    ) -> bool:
        return (
            entry is not None
            and entry.episode_id == episode_id
            and entry.scan_id == scan_id
            and entry.filled == filled
            and torch.equal(entry.points, points)
            and torch.equal(entry.valid, valid)
        )

    @staticmethod
    def _same_far(
        entry: _FarEntry | None,
        *,
        episode_id: int,
        scan_ids: torch.Tensor,
        points: torch.Tensor,
        valid: torch.Tensor,
    ) -> bool:
        return (
            entry is not None
            and entry.episode_id == episode_id
            and torch.equal(entry.scan_ids, scan_ids)
            and torch.equal(entry.points, points)
            and torch.equal(entry.valid, valid)
        )

    def _near_features(
        self, prepared: PreparedControlInputs, env_ids: list[int], episode_ids: list[int]
    ) -> torch.Tensor:
        points = prepared.near_points
        valid = prepared.near_valid
        scan_ids = prepared.near_scan_ids.detach().cpu().tolist()
        filled = prepared.near_filled.detach().cpu().tolist()
        features = points.new_empty((len(env_ids), NEAR_GRU_HIDDEN))
        misses: list[int] = []
        hit_rows: list[int] = []
        for row, (env_id, episode_id, scan_id, is_filled) in enumerate(
            zip(env_ids, episode_ids, scan_ids, filled)
        ):
            entry = self._near.get(env_id)
            if self._same_near(
                entry,
                episode_id=episode_id,
                scan_id=int(scan_id),
                filled=bool(is_filled),
                points=points[row],
                valid=valid[row],
            ):
                features[row].copy_(entry.feature)
                hit_rows.append(row)
            else:
                misses.append(row)

        if misses:
            index = torch.tensor(misses, dtype=torch.long, device=points.device)
            encoded = self.network.encode_near_points(
                points.index_select(0, index), valid.index_select(0, index)
            )
            features.index_copy_(0, index, encoded)
            last_row_for_env: dict[int, int] = {}
            for row, env_id in enumerate(env_ids):
                last_row_for_env[env_id] = row
            encoded_offset = 0
            for row in misses:
                env_id = env_ids[row]
                if last_row_for_env[env_id] == row:
                    self._near[env_id] = _NearEntry(
                        episode_id=episode_ids[row],
                        scan_id=int(scan_ids[row]),
                        filled=bool(filled[row]),
                        points=points[row].detach().clone(),
                        valid=valid[row].detach().clone(),
                        feature=encoded[encoded_offset].detach().clone(),
                    )
                encoded_offset += 1
            self._stats["near_rows_encoded"] += len(misses)
        self._stats["near_hits"] += len(hit_rows)
        return features

    def _far_features(
        self, prepared: PreparedControlInputs, env_ids: list[int], episode_ids: list[int]
    ) -> torch.Tensor:
        points = prepared.far_points
        valid = prepared.far_valid
        scan_ids = prepared.far_point_scan_ids
        features = points.new_empty((len(env_ids), FAR_GRU_HIDDEN))
        misses: list[int] = []
        hit_rows: list[int] = []
        for row, (env_id, episode_id) in enumerate(zip(env_ids, episode_ids)):
            entry = self._far.get(env_id)
            row_scan_ids = scan_ids[row]
            if self._same_far(
                entry,
                episode_id=episode_id,
                scan_ids=row_scan_ids,
                points=points[row],
                valid=valid[row],
            ):
                features[row].copy_(entry.feature)
                hit_rows.append(row)
            else:
                misses.append(row)

        if misses:
            index = torch.tensor(misses, dtype=torch.long, device=points.device)
            encoded = self.network.encode_far_points(
                points.index_select(0, index), valid.index_select(0, index)
            )
            features.index_copy_(0, index, encoded)
            last_row_for_env: dict[int, int] = {}
            for row, env_id in enumerate(env_ids):
                last_row_for_env[env_id] = row
            encoded_offset = 0
            for row in misses:
                env_id = env_ids[row]
                if last_row_for_env[env_id] == row:
                    self._far[env_id] = _FarEntry(
                        episode_id=episode_ids[row],
                        scan_ids=scan_ids[row].detach().clone(),
                        points=points[row].detach().clone(),
                        valid=valid[row].detach().clone(),
                        feature=encoded[encoded_offset].detach().clone(),
                    )
                encoded_offset += 1
            self._stats["far_rows_encoded"] += len(misses)
        self._stats["far_hits"] += len(hit_rows)
        return features

    def forward(self, prepared: PreparedControlInputs) -> NetworkOutput:
        """Return a fresh head output, reusing only exact matching GRU rows.

        This API is inference-only.  Failing fast when autograd is enabled avoids
        silently detaching encoder features from PPO or other training losses.
        """
        if torch.is_grad_enabled():
            raise RuntimeError("PointCloudFeatureCache.forward requires torch.no_grad()")
        if not isinstance(prepared, PreparedControlInputs):
            raise TypeError("prepared must be PreparedControlInputs")

        self.network.validate_inputs(
            prepared.observation,
            prepared.near_points,
            prepared.near_valid,
            prepared.far_points,
            prepared.far_valid,
        )
        self._validate_prepared_metadata(prepared)
        self._clear_if_parameters_changed()

        env_ids = prepared.observation.env_ids.detach().cpu().tolist()
        episode_ids = prepared.observation.episode_ids.detach().cpu().tolist()
        self._stats["forward_calls"] += 1
        near_feature = self._near_features(prepared, env_ids, episode_ids)
        far_feature = self._far_features(prepared, env_ids, episode_ids)
        return self.network.forward_from_features(
            prepared.observation,
            PointCloudFeatures(near_feature=near_feature, far_feature=far_feature),
        )
