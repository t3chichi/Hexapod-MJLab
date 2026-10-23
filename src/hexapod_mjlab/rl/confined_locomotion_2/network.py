"""Frame and temporal recurrent actor/critic for confined_locomotion_2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch import nn

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
    "build_proprio_scale",
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

    def encode_points(
        self,
        near: torch.Tensor,
        nvalid: torch.Tensor,
        far: torch.Tensor,
        fvalid: torch.Tensor,
        *,
        near_angle_keys: torch.Tensor | None = None,
        far_angle_keys: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Merge and encode current near/far points in stable global angle order."""
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
        return self.frame_encoder(points, valid)

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

    def unique_parameters(self) -> tuple[nn.Parameter, ...]:
        unique: dict[int, nn.Parameter] = {}
        for parameter in self.parameters():
            unique.setdefault(id(parameter), parameter)
        return tuple(unique.values())
