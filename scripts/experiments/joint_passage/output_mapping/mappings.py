"""Pure-math output mappings for the joint-passage experiments.

All functions are side-effect free, depend only on numpy/torch/stdlib, accept
CPU floating-point tensors, support arbitrary leading batch axes (last dim = 6
for envelope parameters, 5 for EA2), and preserve dtype/autograd.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "Mapping",
    "physical_to_positive",
    "positive_to_envelope",
    "ea2_forward",
]

DEFAULT_P0 = (0.59, 0.69, 0.59, 0.89, 0.89, 0.46)

_FAMILIES = ("softplus", "elu", "rational", "linear", "sigmoid")


def _check_last_dim(x: torch.Tensor, size: int, name: str) -> None:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(x)!r}")
    if not x.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype")
    if x.ndim < 1 or x.shape[-1] != size:
        raise ValueError(f"{name} must have last dimension {size}, got shape {tuple(x.shape)}")


def _softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softplus(x)


def _softplus_inv(y: torch.Tensor) -> torch.Tensor:
    # y > 0; y + log(-expm1(-y)) is stable for both small and large y.
    return y + torch.log(-torch.expm1(-y))


def _softplus_deriv(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


def _elu(x: torch.Tensor) -> torch.Tensor:
    # Clamp the negative-branch input so the unselected branch cannot
    # overflow to inf and pollute gradients through torch.where.
    neg = torch.exp(x.clamp(max=0.0))
    return torch.where(x < 0, neg, 1.0 + x)


def _elu_inv(y: torch.Tensor) -> torch.Tensor:
    # y > 0; inactive logarithm uses a safe value.
    safe_y = torch.where(y < 1.0, y, torch.ones_like(y))
    return torch.where(y < 1.0, torch.log(safe_y), y - 1.0)


def _elu_deriv(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x < 0, torch.exp(x.clamp(max=0.0)), torch.ones_like(x))


def _rational(x: torch.Tensor) -> torch.Tensor:
    # Negative branch 1/(1-x): clamp input to <= 0 so the denominator stays
    # in [1, inf) and never hits the x = 1 division by zero.
    neg = 1.0 / (1.0 - x.clamp(max=0.0))
    return torch.where(x < 0, neg, 1.0 + x)


def _rational_inv(y: torch.Tensor) -> torch.Tensor:
    # y in (0, 1) uses 1 - 1/y; y >= 1 uses y - 1 (both finite for valid y).
    safe_y = torch.where(y < 1.0, y, torch.ones_like(y))
    return torch.where(y < 1.0, 1.0 - 1.0 / safe_y, y - 1.0)


def _rational_deriv(x: torch.Tensor) -> torch.Tensor:
    safe_x = x.clamp(max=0.0)
    neg = 1.0 / (1.0 - safe_x).pow(2)
    return torch.where(x < 0, neg, torch.ones_like(x))


class Mapping:
    """Dimension-wise positive (or linear/sigmoid control) mapping p = s * f(b + u).

    s = r * p0 and b = f_inverse(p0 / s), so u = 0 maps to p = p0. Parameters
    are rebuilt in the input's dtype/device on every call; the mapping holds no
    trainable tensors; it stores only a detached float64 p0 constant. `r` is unused for the sigmoid family (it only
    keeps the instance interface uniform).
    """

    def __init__(self, family: str, r: float, p0=DEFAULT_P0) -> None:
        if family not in _FAMILIES:
            raise ValueError(f"unknown family {family!r}; expected one of {_FAMILIES}")
        if not isinstance(r, (int, float)) or not math.isfinite(r) or r <= 0:
            raise ValueError(f"r must be a positive finite float, got {r!r}")
        p0_t = torch.as_tensor(p0, dtype=torch.float64).detach().clone()
        _check_last_dim(p0_t, 6, "p0")
        if p0_t.ndim != 1:
            raise ValueError("p0 must be a single six-dimensional vector")
        if not torch.isfinite(p0_t).all() or (p0_t <= 0).any():
            raise ValueError("p0 must be finite and strictly positive")
        self.family = family
        self.r = float(r)
        self.p0 = p0_t

    def _sb(self, ref: torch.Tensor):
        s = self.r * self.p0.to(device=ref.device, dtype=ref.dtype)
        if self.family == "sigmoid":
            return s, None
        if self.family == "linear":
            return s, None
        normalized = self.p0.to(device=ref.device, dtype=ref.dtype) / s  # = 1/r
        b = self._inv_shape(normalized)
        return s, b

    def _inv_shape(self, y: torch.Tensor) -> torch.Tensor:
        f = {
            "softplus": _softplus_inv,
            "elu": _elu_inv,
            "rational": _rational_inv,
        }[self.family]
        return f(y)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        _check_last_dim(u, 6, "u")
        s, b = self._sb(u)
        if self.family == "linear":
            return self.p0.to(u.dtype).to(u.device) + s * u
        if self.family == "sigmoid":
            return 2.0 * self.p0.to(u.dtype).to(u.device) * torch.sigmoid(u)
        return s * {
            "softplus": _softplus,
            "elu": _elu,
            "rational": _rational,
        }[self.family](b + u)

    def inverse(self, p: torch.Tensor) -> torch.Tensor:
        _check_last_dim(p, 6, "p")
        if not torch.isfinite(p).all():
            raise ValueError("inverse() input contains non-finite values")
        s, b = self._sb(p)
        p0 = self.p0.to(p.dtype).to(p.device)
        if self.family == "linear":
            return (p - p0) / s
        if self.family == "sigmoid":
            frac = p / (2.0 * p0)
            if (frac <= 0).any() or (frac >= 1).any():
                raise ValueError("sigmoid inverse requires 0 < p < 2*p0")
            return torch.log(frac) - torch.log1p(-frac)
        y = p / s
        if self.family == "softplus" and (y <= 0).any():
            raise ValueError("softplus inverse requires p > 0")
        if self.family == "elu" and (y <= 0).any():
            raise ValueError("elu inverse requires p > 0")
        if self.family == "rational" and (y <= 0).any():
            raise ValueError("rational inverse requires p > 0")
        return self._inv_shape(y) - b

    def derivative(self, u: torch.Tensor) -> torch.Tensor:
        """Analytic dp/du in meters (per component)."""
        _check_last_dim(u, 6, "u")
        s, b = self._sb(u)
        if self.family == "linear":
            return torch.broadcast_to(s, u.shape).clone()
        if self.family == "sigmoid":
            sig = torch.sigmoid(u)
            return 2.0 * self.p0.to(u.dtype).to(u.device) * sig * (1.0 - sig)
        return s * {
            "softplus": _softplus_deriv,
            "elu": _elu_deriv,
            "rational": _rational_deriv,
        }[self.family](b + u)


def physical_to_positive(E: torch.Tensor) -> torch.Tensor:
    """Convert physical envelope E to positive parameters: negate dim 5 (index 4).

    E's sixth dim (index 5) is an absolute target height and is left unchanged.
    Unit conversion only; no reachability or safety claim.
    """
    _check_last_dim(E, 6, "E")
    out = E.clone()
    out[..., 4] = -out[..., 4]
    return out


def positive_to_envelope(p: torch.Tensor, h_ref: float = 0.46) -> torch.Tensor:
    """Convert positive parameters to envelope: negate dim 5, subtract h_ref from dim 6."""
    _check_last_dim(p, 6, "p")
    out = p.clone()
    out[..., 4] = -out[..., 4]
    out[..., 5] = out[..., 5] - h_ref
    return out


def ea2_forward(
    actions: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    soft_limit: float = 0.95,
    action_max: float = 4.0,
) -> torch.Tensor:
    """Diagnostic EA2 envelope: clamp(mid + a * span * soft_limit / (2*action_max)).

    Applies to the last 5 dims; low/high are supplied by the caller.
    """
    _check_last_dim(actions, 5, "actions")
    _check_last_dim(low, 5, "low")
    _check_last_dim(high, 5, "high")
    if not isinstance(soft_limit, (int, float)) or not math.isfinite(soft_limit):
        raise ValueError("soft_limit must be a finite float")
    if not isinstance(action_max, (int, float)) or not math.isfinite(action_max) or action_max <= 0:
        raise ValueError("action_max must be a positive finite float")
    mid = (low + high) / 2.0
    span = (high - low).to(actions.dtype)
    scale = span * float(soft_limit) / (2.0 * float(action_max))
    raw = mid.to(actions.dtype) + actions * scale
    return torch.clamp(raw, low.to(actions.dtype), high.to(actions.dtype))
