"""Optional auxiliary-loss extension point; no auxiliary model is created by default."""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any, Callable, Mapping

import torch

from .symmetry import mirror_height_targets


_HEIGHT_SHAPE = (11, 17)
_HEIGHT_CELLS = _HEIGHT_SHAPE[0] * _HEIGHT_SHAPE[1]


def masked_height_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Average valid-cell MSE per selected sample, then average those samples.

    ``valid`` describes geometry coverage and ``loss_mask`` selects PPO samples.
    A selected sample with no valid cells is omitted from the mean.
    """
    if not isinstance(prediction, torch.Tensor) or prediction.ndim != 4:
        raise ValueError("prediction must be a [T,B,11,17] tensor")
    if tuple(prediction.shape[-2:]) != _HEIGHT_SHAPE or prediction.shape[0] < 1 or prediction.shape[1] < 1:
        raise ValueError("prediction must have shape [T,B,11,17] with nonempty T and B")
    if prediction.dtype != torch.float32:
        raise TypeError("prediction must be float32")
    if not isinstance(target, torch.Tensor) or tuple(target.shape) != tuple(prediction.shape):
        raise ValueError("target must have the same [T,B,11,17] shape as prediction")
    if target.dtype != torch.float32:
        raise TypeError("target must be float32")
    if not isinstance(valid, torch.Tensor) or tuple(valid.shape) != tuple(prediction.shape):
        raise ValueError("valid must have the same [T,B,11,17] shape as prediction")
    if valid.dtype != torch.bool:
        raise TypeError("valid must be bool")
    if not isinstance(loss_mask, torch.Tensor) or tuple(loss_mask.shape) != tuple(prediction.shape[:2]):
        raise ValueError("loss_mask must have shape [T,B]")
    if loss_mask.dtype != torch.bool:
        raise TypeError("loss_mask must be bool")
    for name, value in (("target", target), ("valid", valid), ("loss_mask", loss_mask)):
        if value.device != prediction.device:
            raise ValueError(f"{name} must be on prediction's device")
    if not bool(torch.isfinite(prediction).all()):
        raise ValueError("prediction contains NaN/Inf")
    # Index first so invalid labels may safely contain arbitrary sentinels, including NaN.
    if not bool(torch.isfinite(target[valid]).all()):
        raise ValueError("valid target cells contain NaN/Inf")

    # Apply the PPO minibatch selection before any floating point arithmetic.
    # Otherwise a finite but extreme target in an unselected sample can overflow
    # while squaring and produce NaN gradients even though its loss is later sliced out.
    selected_valid = valid & loss_mask[:, :, None, None]
    expanded_valid = selected_valid.reshape(*prediction.shape[:2], _HEIGHT_CELLS)
    expanded_prediction = prediction.reshape(*prediction.shape[:2], _HEIGHT_CELLS)
    expanded_target = target.reshape(*target.shape[:2], _HEIGHT_CELLS)
    # Mask both operands before subtraction. In particular, NaN labels outside the
    # geometry mask never enter the subtraction or its backward pass.
    safe_prediction = torch.where(expanded_valid, expanded_prediction, torch.zeros_like(expanded_prediction))
    safe_target = torch.where(expanded_valid, expanded_target, torch.zeros_like(expanded_target))
    squared_error = (safe_prediction - safe_target).square()
    cell_counts = expanded_valid.sum(dim=-1)
    per_sample = squared_error.sum(dim=-1) / cell_counts.clamp_min(1).to(dtype=prediction.dtype)
    selected_with_labels = loss_mask & (cell_counts > 0)
    if bool(selected_with_labels.any()):
        return per_sample[selected_with_labels].mean()
    # A scalar view of prediction keeps the zero connected to autograd without
    # risking overflow from summing many otherwise finite predictions.
    return prediction.reshape(-1)[0] * 0.0


def _prediction_sequence(outputs: Any, name: str) -> list[torch.Tensor]:
    if not isinstance(outputs, (list, tuple)) or not outputs:
        raise ValueError(f"{name} must be a nonempty list of network outputs")
    predictions: list[torch.Tensor] = []
    for index, output in enumerate(outputs):
        prediction = getattr(output, "height_prediction", None)
        if not isinstance(prediction, torch.Tensor):
            raise TypeError(f"{name}[{index}].height_prediction must be a tensor")
        predictions.append(prediction)
    return predictions


@dataclass(frozen=True)
class PreparedHeightTargets:
    """Parameter-independent targets scoped to one rollout update."""

    targets: torch.Tensor
    valid: torch.Tensor
    mirror_targets: torch.Tensor | None
    mirror_valid: torch.Tensor | None
    valid_counts: torch.Tensor


@dataclass(frozen=True)
class HeightReconstructionAuxiliary:
    """Terrain Memory's optional height reconstruction loss.

    This callable owns only a scalar coefficient; all predictions come from the
    policy outputs supplied by PPO.
    """

    weight: float

    def __post_init__(self) -> None:
        if isinstance(self.weight, bool) or not isinstance(self.weight, Real):
            raise TypeError("weight must be a positive finite real number")
        if not math.isfinite(float(self.weight)) or float(self.weight) <= 0:
            raise ValueError("weight must be a positive finite real number")
        object.__setattr__(self, "weight", float(self.weight))

    def prepare_targets(self, steps, *, mirrored=False):
        """Validate aligned immutable labels once, before any optimizer step."""
        if not isinstance(steps, (list, tuple)) or not steps:
            raise ValueError("batch.steps must be a nonempty sequence")
        heights: list[torch.Tensor] = []
        validity: list[torch.Tensor] = []
        for index, step in enumerate(steps):
            label = getattr(step, "height_label", None)
            if label is None or not callable(getattr(label, "validate", None)):
                raise ValueError(f"batch.steps[{index}].height_label must be a validated HeightLabels")
            observation = getattr(step, "observation", None)
            if observation is None:
                raise ValueError(f"batch.steps[{index}].observation is required for height label validation")
            label.validate(observation)
            heights.append(label.heights)
            validity.append(label.valid)
        targets = torch.stack(heights)
        valid = torch.stack(validity)
        mirror_targets, mirror_valid = mirror_height_targets(targets, valid) if mirrored else (None, None)
        return PreparedHeightTargets(targets, valid, mirror_targets, mirror_valid,
                                     valid.reshape(*valid.shape[:2], _HEIGHT_CELLS).sum(dim=-1))

    def __call__(self, outputs: Any, batch: Any) -> AuxiliaryResult:
        if not isinstance(batch, Mapping):
            raise TypeError("batch must be a mapping")
        try:
            loss_mask = batch["loss_mask"]
            steps = batch["steps"]
            mirrored_outputs = batch.get("mirrored_outputs")
        except KeyError as exc:
            raise ValueError(f"batch is missing {exc.args[0]!r}") from exc
        if not isinstance(steps, (list, tuple)) or not steps:
            raise ValueError("batch.steps must be a nonempty sequence")
        predictions = _prediction_sequence(outputs, "outputs")
        if len(predictions) != len(steps):
            raise ValueError("outputs and batch.steps must have the same sequence length")

        prepared = batch.get("height_targets")
        if prepared is None:
            prepared = self.prepare_targets(steps, mirrored=mirrored_outputs is not None)
        if not isinstance(prepared, PreparedHeightTargets):
            raise TypeError("height_targets must be PreparedHeightTargets")
        targets, valid = prepared.targets, prepared.valid
        prediction = torch.stack(predictions)
        original_mse = masked_height_mse(prediction, targets, valid, loss_mask)

        mirror_mse: torch.Tensor | None = None
        if mirrored_outputs is not None:
            mirror_predictions = torch.stack(_prediction_sequence(mirrored_outputs, "mirrored_outputs"))
            if mirror_predictions.shape != prediction.shape:
                raise ValueError("mirrored outputs must match the original prediction shape")
            mirror_targets, mirror_valid = prepared.mirror_targets, prepared.mirror_valid
            if mirror_targets is None or mirror_valid is None:
                raise ValueError("prepared height targets lack the mirrored branch")
            mirror_mse = masked_height_mse(mirror_predictions, mirror_targets, mirror_valid, loss_mask)
            combined_mse = 0.5 * (original_mse + mirror_mse)
        else:
            combined_mse = original_mse

        weighted_loss = combined_mse * self.weight
        if not isinstance(loss_mask, torch.Tensor) or loss_mask.dtype != torch.bool \
                or tuple(loss_mask.shape) != tuple(prediction.shape[:2]) \
                or loss_mask.device != prediction.device:
            # Keep validation errors consistent even when every geometry label is empty.
            raise ValueError("loss_mask must be a [T,B] bool tensor on the prediction device")
        selected_count = loss_mask.sum()
        selected_denominator = selected_count.clamp_min(1).to(dtype=prediction.dtype)
        valid_counts = prepared.valid_counts
        eligible_count = (loss_mask & (valid_counts > 0)).sum()
        valid_cell_count = (valid_counts * loss_mask.to(dtype=valid_counts.dtype)).sum()
        valid_sample_fraction = eligible_count.to(dtype=prediction.dtype) / selected_denominator
        valid_cell_fraction = valid_cell_count.to(dtype=prediction.dtype) / (
            selected_denominator * _HEIGHT_CELLS
        )

        metrics: dict[str, torch.Tensor] = {
            "Height/mse_m2": combined_mse.detach(),
            "Height/original_mse_m2": original_mse.detach(),
            "Height/weighted_loss": weighted_loss.detach(),
            "Height/valid_sample_fraction": valid_sample_fraction.detach(),
            "Height/valid_cell_fraction": valid_cell_fraction.detach(),
        }
        if mirror_mse is not None:
            metrics["Height/mirrored_mse_m2"] = mirror_mse.detach()
        return AuxiliaryResult(weighted_loss, metrics)


@dataclass(frozen=True)
class AuxiliaryResult:
    loss: torch.Tensor
    metrics: Mapping[str, torch.Tensor | float]


class NoAuxiliary:
    """Explicit empty plugin. It owns no modules or parameters."""
    def __call__(self, output: Any, batch: Any) -> AuxiliaryResult:
        if isinstance(output, (list, tuple)):
            output = output[-1]
        reference = output.action_mean
        return AuxiliaryResult(reference.new_zeros(()), {})


def resolve_auxiliary(plugin: Callable[[Any, Any], AuxiliaryResult] | None) -> Callable[[Any, Any], AuxiliaryResult]:
    return NoAuxiliary() if plugin is None else plugin


__all__ = [
    "AuxiliaryResult", "HeightReconstructionAuxiliary", "NoAuxiliary",
    "masked_height_mse", "resolve_auxiliary",
]
