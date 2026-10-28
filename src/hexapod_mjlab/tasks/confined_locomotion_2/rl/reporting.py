"""Scalar aggregation and checkpoint scheduling shared by v2-family runners.

These functions only aggregate already-materialized metrics and decide the
periodic checkpoint boundary. They own no runner state, device access or
training lifecycle.
"""

from __future__ import annotations

import re

__all__ = [
    "checkpoint_due",
    "optimizer_scalar_summary",
]


def checkpoint_due(iteration, interval):
    """Use completed cumulative PPO iterations, including after a resume."""
    return iteration > 0 and iteration % interval == 0


def optimizer_scalar_summary(metrics):
    """Retain full minibatch JSON while plotting concise iteration means."""
    names = {"loss": "Loss/total", "policy": "Loss/policy", "value": "Loss/value",
             "envelope": "Loss/envelope", "kl": "Optimizer/kl", "lr": "Optimizer/learning_rate"}
    values = {name: [] for name in names}
    for key, value in metrics.items():
        match = re.fullmatch(r"epoch_\d+_minibatch_\d+_(loss|policy|value|envelope|kl|lr)", key)
        if match:
            values[match.group(1)].append(float(value))
    result = {name: value for name, value in metrics.items() if "/" in name}
    result.update({names[name]: sum(items) / len(items) for name, items in values.items() if items})
    return result
