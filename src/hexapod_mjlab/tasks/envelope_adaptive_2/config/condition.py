"""Frozen source envelope bounds and pure morphology priors (no HAA network)."""
from dataclasses import dataclass
from typing import Tuple, Mapping
import torch
from torch import Tensor

@dataclass(frozen=True)
class EnvelopeConditionSpec:
    """Condition distribution read directly from ``spider_envelop`` config."""

    condition_names: Tuple[str, ...]
    low: Tuple[float, ...]
    high: Tuple[float, ...]
    morphology_prior_mode: str
    morphology_prior_weights: Mapping[str, Mapping[str, float]]
    morphology_middle_front_follow_weight: float

    def bounds(self, device: torch.device) -> Tuple[Tensor, Tensor]:
        return (
            torch.tensor(self.low, dtype=torch.float32, device=device),
            torch.tensor(self.high, dtype=torch.float32, device=device),
        )

def apply_env_morphology_priors(condition: Tensor, spec: EnvelopeConditionSpec) -> Tensor:
    """Apply the same prior derivation used by ``spider_envelop._set_morphology_prior_from_envelope``."""
    if condition.shape[-1] != len(spec.condition_names):
        raise ValueError(
            f"Expected condition width {len(spec.condition_names)}, got {condition.shape[-1]}"
        )
    condition = condition.clone()
    index = {name: idx for idx, name in enumerate(spec.condition_names)}
    low = condition.new_tensor(spec.low)
    high = condition.new_tensor(spec.high)

    def normalized(name: str) -> Tensor:
        value = condition[..., index[name]]
        value_low = low[index[name]]
        value_high = high[index[name]]
        if name == "backward_limit":
            value = -value
            value_low = -high[index[name]]
            value_high = -low[index[name]]
        return ((value - value_low) / (value_high - value_low).clamp_min(1e-6)).clamp(0.0, 1.0)

    if spec.morphology_prior_mode == "weighted_sum":
        prior_specs = {
            "morphology_front_prior": {"front_width": -0.5, "forward_limit": 0.5},
            "morphology_middle_prior": {"middle_width": -1.0},
            "morphology_back_prior": {"back_width": -0.5, "backward_limit": 0.5},
        }
        for prior_name, prior_weights in prior_specs.items():
            prior = torch.zeros_like(condition[..., 0])
            weight_sum = 0.0
            for name, weight in prior_weights.items():
                value = normalized(name)
                if weight < 0.0:
                    value = 1.0 - value
                prior = prior + abs(weight) * value
                weight_sum += abs(weight)
            condition[..., index[prior_name]] = (prior / max(weight_sum, 1e-6)).clamp(0.0, 1.0)
        return condition
    if spec.morphology_prior_mode != "directional_ratio":
        raise ValueError(f"Unknown morphology_prior_mode={spec.morphology_prior_mode!r}")

    morphology_weights = spec.morphology_prior_weights
    front_lateral = float(morphology_weights["front"].get("lateral", 0.5)) * normalized("front_width")
    front_longitudinal = float(morphology_weights["front"].get("longitudinal", 0.5)) * normalized("forward_limit")
    back_lateral = float(morphology_weights["back"].get("lateral", 0.5)) * normalized("back_width")
    back_longitudinal = float(morphology_weights["back"].get("longitudinal", 0.5)) * normalized("backward_limit")
    eps = 1e-6
    front_prior = front_longitudinal / (front_longitudinal + front_lateral).clamp_min(eps)
    back_prior = back_longitudinal / (back_longitudinal + back_lateral).clamp_min(eps)
    middle_prior = 1.0 - normalized("middle_width")
    if float(morphology_weights["middle"].get("lateral", 1.0)) <= 0.0:
        middle_prior = torch.full_like(middle_prior, 0.5)
    follow = min(max(spec.morphology_middle_front_follow_weight, 0.0), 1.0)
    middle_prior = follow * front_prior + (1.0 - follow) * middle_prior

    condition[..., index["morphology_front_prior"]] = front_prior.clamp(0.0, 1.0)
    condition[..., index["morphology_middle_prior"]] = middle_prior.clamp(0.0, 1.0)
    condition[..., index["morphology_back_prior"]] = back_prior.clamp(0.0, 1.0)
    return condition

ENVELOPE_SPEC = EnvelopeConditionSpec(('front_width', 'middle_width', 'back_width', 'forward_limit', 'backward_limit', 'morphology_front_prior', 'morphology_middle_prior', 'morphology_back_prior'), (0.3, 0.3, 0.3, 0.6, -0.9, 0.0, 0.0, 0.0), (0.6, 0.7, 0.6, 0.9, -0.6, 1.0, 1.0, 1.0), 'directional_ratio', {'front': {'lateral': 0.35, 'longitudinal': 0.5}, 'middle': {'lateral': 1.0}, 'back': {'lateral': 0.35, 'longitudinal': 0.5}}, 0.4)
