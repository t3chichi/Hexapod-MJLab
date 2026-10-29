"""Temporal local-memory confined locomotion training core."""

from .contracts import (
    EMPTY_SCAN_ID,
    FAR_POINTS_PER_SCAN,
    FAR_WINDOW_SLOTS,
    NEAR_POINTS,
    ContractError,
    EnvelopeLabel,
    IdentityMismatchError,
    OnlineObservation,
    ScanWindowSnapshot,
    bind_envelope_label,
    flatten_far_window,
    sort_points_by_angle_keys,
)
from .history import (
    EpisodeMismatchError,
    ScanWindow,
    StaleScanError,
    TimeOrderError,
    WindowAngleKeys,
    WindowSourceDirections,
)
from .integration import PreparedControlInputs, prepare_control_inputs
from .network import ConfinedLocomotionNetwork, NetworkOutput
from .distribution import DirectStdNormalProvider
from .ppo import TemporalPPOConfig, TemporalPPOTrainer, derive_shuffle_generator_seed
from .training_rollout import SequenceRolloutBuffer

__all__ = [
    "ConfinedLocomotionNetwork",
    "DirectStdNormalProvider",
    "ContractError",
    "EMPTY_SCAN_ID",
    "EnvelopeLabel",
    "EpisodeMismatchError",
    "FAR_POINTS_PER_SCAN",
    "FAR_WINDOW_SLOTS",
    "IdentityMismatchError",
    "NEAR_POINTS",
    "NetworkOutput",
    "OnlineObservation",
    "PreparedControlInputs",
    "ScanWindow",
    "ScanWindowSnapshot",
    "SequenceRolloutBuffer",
    "StaleScanError",
    "TemporalPPOConfig",
    "TemporalPPOTrainer",
    "TimeOrderError",
    "WindowAngleKeys",
    "WindowSourceDirections",
    "bind_envelope_label",
    "flatten_far_window",
    "prepare_control_inputs",
    "derive_shuffle_generator_seed",
    "sort_points_by_angle_keys",
]
