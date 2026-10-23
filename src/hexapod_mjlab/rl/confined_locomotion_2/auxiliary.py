"""Optional auxiliary-loss extension point; no auxiliary model is created by default."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch


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


__all__ = ["AuxiliaryResult", "NoAuxiliary", "resolve_auxiliary"]
