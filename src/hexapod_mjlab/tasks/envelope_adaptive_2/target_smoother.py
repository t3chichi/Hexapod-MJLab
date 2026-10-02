"""Rate-limited smoothing of the oracle target (production).

Wraps the stateless grid oracle with the legacy rule computer's hysteresis
(shrink fast, grow slow, cooldown before growth) in normalised extent space,
so the supervised envelope target cannot jump between control steps when the
active-set binding sample switches.  A per-env safety check snaps any env
whose rate-limited candidate would itself be unsafe straight back to the raw
(safety-verified) oracle.

Rates are PHYSICAL (metres per second of envelope extent) and are converted
with the control ``dt`` at construction into per-call steps normalised by
each parameter's extent span.  The 10 Hz validation constants (shrink
0.12 extent/call, grow 0.03 extent/call, cooldown 5 calls) correspond to
~0.4 m/s, ~0.1 m/s and 0.5 s.  At the env's 50 Hz the approach demand is
~0.02 m/call, so the env defaults (shrink 2.0 m/s = 0.04 m/call) track
normal approach without lag while still rate-limiting the rare
binding-sample switching jumps (up to ~0.11 m).
"""

from __future__ import annotations

import torch

from hexapod_mjlab.tasks.envelope_adaptive_2.oracle import (
    _physical_min_max,
)

_SIGNED_SPAN_EPS = 1e-6


class RateLimitedOracle:
    """Batched rate limiter for the oracle envelope target.

    Works on ``(N, 5)`` parameter batches.  All elementwise ops keep one
    smoother state per env.  ``s`` is the normalised extent: 0 = fully
    shrunk, 1 = fully extended, for ALL five parameters (the signed span of
    ``backward_limit`` makes the direction uniform).
    """

    def __init__(
        self,
        num_envs: int,
        dt: float,
        device,
        low: torch.Tensor,
        high: torch.Tensor,
        shrink_rate: float = 2.0,
        grow_rate: float = 0.5,
        cooldown_seconds: float = 0.2,
        grow_tol_frac: float = 0.5,
        safety_check=None,
    ):
        self.num_envs = int(num_envs)
        self.device = device
        self.min_v, self.max_v = _physical_min_max(low, high)
        # Signed span: backward_limit is physically reversed (more negative =
        # larger rear extent), so its span is negative.  Division by the
        # signed span makes s=0 fully shrunk and s=1 fully extended for all
        # five parameters; rate steps divide by |span|.
        self.span = self.max_v - self.min_v
        abs_span = self.span.abs().clamp_min(_SIGNED_SPAN_EPS)
        self.shrink_n = (shrink_rate * dt) / abs_span
        self.grow_n = (grow_rate * dt) / abs_span
        self.grow_tol = grow_tol_frac * self.grow_n
        self.cooldown_calls = max(1, int(round(cooldown_seconds / max(dt, 1e-9))))
        self.safety_check = safety_check
        # Per-env state: prev_s None until the first update (lazily fully
        # open); counter counts consecutive "clear to grow" calls.
        self.prev_s: torch.Tensor | None = None
        self.counter: torch.Tensor | None = None
        self.snapped = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self.reset_ids(range(self.num_envs))

    def reset_ids(self, env_ids) -> None:
        """Reset smoother state for the given env indices (episode boundary)."""
        idx = torch.as_tensor(list(env_ids), dtype=torch.long, device=self.device)
        if idx.numel() == 0:
            return
        ones = torch.ones(len(idx), len(self.span), device=self.device)
        zeros = torch.zeros(len(idx), len(self.span), device=self.device)
        if self.prev_s is None:
            self.prev_s = torch.ones(self.num_envs, len(self.span), device=self.device)
            self.counter = torch.zeros(self.num_envs, len(self.span), device=self.device)
        self.prev_s[idx] = ones
        self.counter[idx] = zeros

    def _to_s(self, params: torch.Tensor) -> torch.Tensor:
        return ((params - self.min_v) / self.span).clamp(0.0, 1.0)

    def _from_s(self, s: torch.Tensor) -> torch.Tensor:
        return self.min_v + s * self.span

    def update(self, raw_params: torch.Tensor) -> torch.Tensor:
        """Advance one control step; returns the smoothed target (N, 5).

        Shrink is applied immediately at a bounded rate toward the raw
        target; growth waits ``cooldown_calls`` consecutive clear calls and
        then also advances at a bounded rate.  Envs whose candidate fails
        ``safety_check`` snap to the raw oracle for this call and their
        state is re-seeded from the raw target (raw is safety-verified by
        construction).
        """
        raw_s = self._to_s(raw_params)
        if self.prev_s is None:
            self.reset_ids(range(self.num_envs))
        needs_shrink = raw_s < self.prev_s - 1e-6
        clear = raw_s > self.prev_s + self.grow_tol
        self.counter = torch.where(
            needs_shrink,
            torch.zeros_like(self.counter),
            torch.where(clear, self.counter + 1, self.counter),
        )
        shrink_target = (self.prev_s - self.shrink_n).clamp(0.0, 1.0)
        grow_target = (self.prev_s + self.grow_n).clamp(0.0, 1.0)
        can_grow = clear & (self.counter >= self.cooldown_calls)
        new_s = torch.where(
            needs_shrink,
            torch.maximum(raw_s, shrink_target),
            torch.where(can_grow, torch.minimum(raw_s, grow_target), self.prev_s),
        )
        candidate = self._from_s(new_s)
        unsafe = None
        if self.safety_check is not None:
            unsafe = self.safety_check(candidate)
            if bool(unsafe.any()):
                candidate = torch.where(unsafe.unsqueeze(-1), raw_params, candidate)
                new_s = torch.where(unsafe.unsqueeze(-1), raw_s, new_s)
        self.prev_s = new_s
        self.snapped = (
            unsafe if unsafe is not None else torch.zeros_like(self.snapped)
        )
        return candidate
