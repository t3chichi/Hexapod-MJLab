"""Frame and temporal recurrent actor/critic for confined_locomotion_2."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Sequence

import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

from ..confined_locomotion.mapping import H_REF_STARTUP, P0_STARTUP, EnvelopeSoftplusMapping
from ..confined_locomotion.network import (
    MLPHead,
    MaskedPointSequenceGRU,
    MorphologyHead,
    PROPRIO_FIELD_SCALE,
    build_proprio_scale,
)
from .contracts import (
    ACTION_DIM,
    COMMAND_DIM,
    CONTROL_INPUT_DIM,
    ENVELOPE_DIM,
    FAR_POINTS_PER_SCAN,
    FLOAT_DTYPE,
    FRAME_GRU_HIDDEN,
    NEAR_POINTS,
    PROPRIO_DIM,
    TEMPORAL_GRU_HIDDEN,
    TEMPORAL_GRU_INPUT,
    ContractError,
    OnlineObservation,
    sort_points_by_angle_keys,
    validate_point_block,
)

__all__ = [
    "ACTION_DIM", "COMMAND_SCALE", "CONTROL_HIDDEN_DIMS", "CONTROL_INPUT_DIM",
    "FRAME_GRU_HIDDEN", "ConfinedLocomotionNetwork", "MLPHead",
    "MORPHOLOGY_HIDDEN_DIMS", "MorphologyHead", "NetworkOutput",
    "PROPRIO_FIELD_SCALE", "TEMPORAL_GRU_HIDDEN", "TEMPORAL_GRU_INPUT",
    "build_proprio_scale", "TerrainMemoryNetwork",
]

MORPHOLOGY_HIDDEN_DIMS = (128, 64)
CONTROL_HIDDEN_DIMS = (512, 256, 128)
COMMAND_SCALE = (2.0, 2.0, 0.25)


@dataclass(frozen=True)
class NetworkOutput:
    frame_feature: torch.Tensor
    temporal_feature: torch.Tensor
    hidden: torch.Tensor
    envelope: torch.Tensor
    action_mean: torch.Tensor
    value: torch.Tensor
    height_prediction: torch.Tensor | None = None


class ConfinedLocomotionNetwork(nn.Module):
    """Single-frame point GRU plus caller-owned temporal hidden state.

    The point GRU always starts at zero for a scan. The temporal GRU has no
    stored module state: callers pass ``hidden`` into every control step and
    retain the returned ``NetworkOutput.hidden`` as needed.
    """

    def __init__(
        self,
        *,
        mlp_activation: Callable[[torch.Tensor], torch.Tensor] | nn.Module,
        init_fn: Callable[[nn.Module], None] | None,
        envelope_mode: str,
        ground_samples: None = None,
        p0: Sequence[float] | torch.Tensor = P0_STARTUP,
        h_ref: float = H_REF_STARTUP,
    ) -> None:
        super().__init__()
        if not callable(mlp_activation):
            raise ContractError("mlp_activation must be callable")
        if init_fn is not None and not callable(init_fn):
            raise ContractError("init_fn must be callable or None")
        if ground_samples is not None:
            raise ContractError("confined_locomotion_2 has no ground head; ground_samples must be None")

        self.mapping = EnvelopeSoftplusMapping(mode=envelope_mode, p0=p0, h_ref=h_ref)
        self.frame_encoder = MaskedPointSequenceGRU(hidden_size=FRAME_GRU_HIDDEN)
        self.temporal_encoder = nn.GRU(
            input_size=TEMPORAL_GRU_INPUT,
            hidden_size=TEMPORAL_GRU_HIDDEN,
            num_layers=1,
            batch_first=True,
        )
        self.morphology_head = MorphologyHead(
            input_dim=TEMPORAL_GRU_HIDDEN + PROPRIO_DIM,
            output_dim=ENVELOPE_DIM,
        )
        self.action_head = MLPHead(
            input_dim=CONTROL_INPUT_DIM,
            hidden_dims=CONTROL_HIDDEN_DIMS,
            output_dim=ACTION_DIM,
            activation=mlp_activation,
        )
        self.value_head = MLPHead(
            input_dim=CONTROL_INPUT_DIM,
            hidden_dims=CONTROL_HIDDEN_DIMS,
            output_dim=1,
            activation=mlp_activation,
        )
        self.register_buffer("proprio_scale", build_proprio_scale())
        self.register_buffer("command_scale", torch.tensor(COMMAND_SCALE, dtype=FLOAT_DTYPE))
        if init_fn is not None:
            self.apply(init_fn)
        # Initialization callbacks may initialize every Linear; restore the
        # required initial physical envelope after they have run.
        self.morphology_head.reset_terminal(self.mapping.head_terminal_bias)

    @staticmethod
    def initial_hidden(batch: int, device: torch.device | str) -> torch.Tensor:
        if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
            raise ContractError("batch must be a positive integer")
        return torch.zeros(batch, TEMPORAL_GRU_HIDDEN, dtype=FLOAT_DTYPE, device=device)

    @staticmethod
    def _batch_device(observation: OnlineObservation) -> tuple[int, torch.device]:
        if not isinstance(observation, OnlineObservation):
            raise ContractError("observation must be OnlineObservation")
        observation.validate()
        return int(observation.env_ids.shape[0]), observation.env_ids.device

    def _ordered_point_inputs(
        self,
        near: torch.Tensor,
        nvalid: torch.Tensor,
        far: torch.Tensor,
        fvalid: torch.Tensor,
        *,
        near_angle_keys: torch.Tensor | None = None,
        far_angle_keys: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Validate and merge near/far points in stable global angle order."""
        validate_point_block(near, nvalid, n_points=NEAR_POINTS, name="near_points")
        validate_point_block(far, fvalid, n_points=FAR_POINTS_PER_SCAN, name="far_points")
        if near.shape[0] != far.shape[0] or near.device != far.device:
            raise ContractError("near/far batches and devices must match")
        if (near_angle_keys is None) != (far_angle_keys is None):
            raise ContractError("near/far angle keys must be supplied together or both omitted")

        points = torch.cat((near, far), dim=1)
        valid = torch.cat((nvalid, fvalid), dim=1)
        if near_angle_keys is not None:
            batch = int(near.shape[0])
            if not isinstance(near_angle_keys, torch.Tensor) or near_angle_keys.dtype is not FLOAT_DTYPE \
                    or tuple(near_angle_keys.shape) != (batch, NEAR_POINTS) \
                    or near_angle_keys.device != near.device:
                raise ContractError(f"near_angle_keys must be ({batch}, {NEAR_POINTS}) float32 on {near.device}")
            if not isinstance(far_angle_keys, torch.Tensor) or far_angle_keys.dtype is not FLOAT_DTYPE \
                    or tuple(far_angle_keys.shape) != (batch, FAR_POINTS_PER_SCAN) \
                    or far_angle_keys.device != far.device:
                raise ContractError(
                    f"far_angle_keys must be ({batch}, {FAR_POINTS_PER_SCAN}) float32 on {far.device}"
                )
            keys = torch.cat((near_angle_keys, far_angle_keys), dim=1)
            ordered = sort_points_by_angle_keys(points, valid, keys)
            points, valid = ordered.points, ordered.valid
        return points, valid

    def encode_points(self, near, nvalid, far, fvalid, *, near_angle_keys=None,
                      far_angle_keys=None) -> torch.Tensor:
        points, valid = self._ordered_point_inputs(
            near, nvalid, far, fvalid, near_angle_keys=near_angle_keys,
            far_angle_keys=far_angle_keys)
        return self.frame_encoder(points, valid)

    def prepare_point_sequence(self, near, nvalid, far, fvalid, *, near_angle_keys=None,
                               far_angle_keys=None):
        """Prepare fixed rollout data once; retain no trainable frame features.

        Use the same stable compaction/packing as MaskedPointSequenceGRU. Only
        replay calls this path; the online encoder keeps its existing contract.
        """
        points, valid = self._ordered_point_inputs(
            near, nvalid, far, fvalid, near_angle_keys=near_angle_keys,
            far_angle_keys=far_angle_keys)
        lengths = valid.sum(dim=1)
        rows = (lengths > 0).nonzero(as_tuple=False).flatten()
        batch = len(points)
        if rows.numel() == 0:
            return None, rows, batch
        sub_points = points.index_select(0, rows)
        sub_valid = valid.index_select(0, rows)
        sub_lengths = lengths.index_select(0, rows)
        order = torch.argsort((~sub_valid).to(torch.int64), dim=1, stable=True)
        compacted = torch.gather(sub_points, 1, order.unsqueeze(-1).expand_as(sub_points)).contiguous()
        packed = pack_padded_sequence(
            compacted[:, :int(sub_lengths.max())].contiguous(), sub_lengths.cpu(),
            batch_first=True, enforce_sorted=False)
        # Plain tensors avoid PackedSequence namedtuple reconstruction by checkpoint.
        return tuple(packed), rows, batch

    def encode_point_sequence(self, sequence) -> torch.Tensor:
        """Recompute the existing point GRU at current weights, from fixed inputs."""
        packed, rows, batch = sequence
        features = self.frame_encoder.gru.weight_ih_l0.new_zeros(batch, FRAME_GRU_HIDDEN)
        if packed is None:
            return features
        _, hidden = self.frame_encoder.gru(PackedSequence(*packed))
        return features.index_copy(0, rows, hidden[0])

    def forward(
        self,
        observation: OnlineObservation,
        near: torch.Tensor,
        nvalid: torch.Tensor,
        far: torch.Tensor,
        fvalid: torch.Tensor,
        hidden: torch.Tensor,
        *,
        near_angle_keys: torch.Tensor | None = None,
        far_angle_keys: torch.Tensor | None = None,
    ) -> NetworkOutput:
        frame_feature = self.encode_points(
            near, nvalid, far, fvalid,
            near_angle_keys=near_angle_keys,
            far_angle_keys=far_angle_keys,
        )
        return self.forward_from_features(observation, frame_feature, hidden)

    def forward_from_features(
        self,
        observation: OnlineObservation,
        frame_feature: torch.Tensor,
        hidden: torch.Tensor,
    ) -> NetworkOutput:
        batch, device = self._batch_device(observation)
        for name, value, width in (
            ("frame_feature", frame_feature, FRAME_GRU_HIDDEN),
            ("hidden", hidden, TEMPORAL_GRU_HIDDEN),
        ):
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != (batch, width) \
                    or value.dtype is not FLOAT_DTYPE or value.device != device:
                raise ContractError(f"{name} must be ({batch}, {width}) float32 on {device}")
            if not bool(torch.isfinite(value).all()):
                raise ContractError(f"{name} contains NaN/Inf")

        proprio = observation.proprio_vector() * self.proprio_scale
        command = observation.command * self.command_scale
        # PROPRIO_LAYOUT starts [lin_vel_b, ang_vel_b, gravity_b], exactly the
        # nine state values appended after the current scan feature.
        temporal_input = torch.cat((frame_feature, proprio[:, :9]), dim=1).unsqueeze(1)
        temporal_sequence, hidden_out = self.temporal_encoder(temporal_input, hidden.unsqueeze(0))
        temporal_feature = temporal_sequence[:, 0]
        next_hidden = hidden_out[0]

        return self._output_from_temporal(
            frame_feature, temporal_feature, next_hidden, proprio, command)

    def _output_from_temporal(self, frame_feature, temporal_feature, next_hidden, proprio, command):
        """Shared stateless heads for online steps and batched training replay."""
        morphology_input = torch.cat((temporal_feature, proprio), dim=1)
        envelope = self.mapping(self.morphology_head(morphology_input))
        control = torch.cat((temporal_feature, proprio, command, envelope.detach()), dim=1)
        action_mean = self.action_head(control)
        value = self.value_head(control)
        return NetworkOutput(
            frame_feature=frame_feature,
            temporal_feature=temporal_feature,
            hidden=next_hidden,
            envelope=envelope,
            action_mean=action_mean,
            value=value,
        )

    def prepare_sequence_observations(self, observations):
        """Validate and scale immutable rollout observations once per replay plan."""
        proprio = torch.stack([obs.proprio_vector() for obs in observations]) * self.proprio_scale
        command = torch.stack([obs.command for obs in observations]) * self.command_scale
        return proprio, command

    def forward_sequence_from_features(self, observations, frame_features, hidden, layout,
                                       *, scaled_observations=None):
        """Training replay: run independent episode segments, then batch the heads."""
        time, batch, length = layout.shape
        device = hidden.device
        if (frame_features.shape != (time, batch, TEMPORAL_GRU_HIDDEN) or
                frame_features.dtype != FLOAT_DTYPE or frame_features.device != device or
                hidden.shape != (batch, TEMPORAL_GRU_HIDDEN) or hidden.dtype != FLOAT_DTYPE):
            raise ContractError("sequence features/hidden shape, dtype or device mismatch")
        if not bool(torch.isfinite(frame_features).all()) or not bool(torch.isfinite(hidden).all()):
            raise ContractError("sequence features/hidden contain NaN/Inf")
        proprio, command = (self.prepare_sequence_observations(observations)
                             if scaled_observations is None else scaled_observations)
        temporal_input = torch.cat((frame_features, proprio[:, :, :9]), dim=-1)
        segments = temporal_input.reshape(time * batch, TEMPORAL_GRU_INPUT)[layout.gather]
        h0 = hidden.detach().index_select(0, layout.rows)
        h0 = torch.where(layout.starts_at_zero[:, None], h0, torch.zeros_like(h0)).unsqueeze(0)
        packed = pack_padded_sequence(segments, layout.lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.temporal_encoder(packed, h0)
        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=length)
        temporal = output.reshape(-1, TEMPORAL_GRU_HIDDEN).index_select(0, layout.restore)
        decoded = self._output_from_temporal(
            frame_features.reshape(time * batch, TEMPORAL_GRU_HIDDEN), temporal, temporal,
            proprio.reshape(time * batch, PROPRIO_DIM), command.reshape(time * batch, COMMAND_DIM))
        temporal = temporal.reshape(time, batch, TEMPORAL_GRU_HIDDEN)
        envelope = decoded.envelope.reshape(time, batch, ENVELOPE_DIM)
        action = decoded.action_mean.reshape(time, batch, ACTION_DIM)
        value = decoded.value.reshape(time, batch, 1)
        height = (None if decoded.height_prediction is None else
                  decoded.height_prediction.reshape(time, batch, 11, 17))
        return [NetworkOutput(frame_features[t], temporal[t], temporal[t], envelope[t],
                              action[t], value[t], None if height is None else height[t])
                for t in range(time)]

    def unique_parameters(self) -> tuple[nn.Parameter, ...]:
        unique: dict[int, nn.Parameter] = {}
        for parameter in self.parameters():
            unique.setdefault(id(parameter), parameter)
        return tuple(unique.values())


class TerrainMemoryNetwork(ConfinedLocomotionNetwork):
    """Shared two-GRU policy with one training-only linear height decoder."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.height_head = nn.Linear(TEMPORAL_GRU_HIDDEN, 187)

    def _output_from_temporal(self, frame_feature, temporal_feature, next_hidden, proprio, command):
        output = super()._output_from_temporal(
            frame_feature, temporal_feature, next_hidden, proprio, command)
        return replace(output, height_prediction=self.height_head(output.temporal_feature).reshape(-1, 11, 17))
