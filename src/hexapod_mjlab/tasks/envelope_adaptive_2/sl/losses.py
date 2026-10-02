"""Pure recall and geometry safety losses, reusable without an SL runner or env."""
import torch
from ..config.condition import ENVELOPE_SPEC
from ..config.parameters import EnvelopeCfg
_LOW = torch.tensor(ENVELOPE_SPEC.low[:5])
_HIGH = torch.tensor(ENVELOPE_SPEC.high[:5])
_SAFE_MARGIN = EnvelopeCfg().margin
_SAFE_SOFT = EnvelopeCfg().soft_margin
_FLOOR_PINNED_VIOLATION = .05

def aux_memory_loss(
    pred: torch.Tensor, y: torch.Tensor, k: int, mode: str
) -> torch.Tensor:
    """Auxiliary memory loss for one probe horizon ``k``.

    ``pred`` is the probe's readout of the GRU hidden sequence ``h_t`` with the
    same ``(L, B, 5)`` layout as the target sequence ``y``.

    * ``mode="recall"``  -- ``pred[t]`` must equal ``y[t-k]``: reconstruct the
      envelope from ``k`` frames ago.  Pure memory demand: the past state
      cannot be derived from the current view, so the only path is retention
      in ``h``.
    * ``mode="forward"`` -- ``pred[t]`` must equal ``y[t+k]``: predict the
      envelope ``k`` frames ahead.  Mixed current-view extrapolation and
      retention; the gradient lands on ``h`` at the (last-sight) write moment.

    Frames without a valid partner are dropped (no padding, no wraparound).
    """
    L = y.shape[0]
    if not 0 < k < L:
        raise ValueError(f"aux horizon k={k} must be in (0, {L})")
    if mode == "recall":
        return torch.nn.functional.mse_loss(pred[k:], y[:-k])
    if mode == "forward":
        return torch.nn.functional.mse_loss(pred[: L - k], y[k:])
    raise ValueError(f"unknown aux_mode: {mode!r}")

def aux_head_weights(ks, seq_len: int, beta: float) -> dict:
    """Equal-budget per-horizon weights for the auxiliary memory probes.

    Each horizon's share follows its valid-sample count ``(L - k) / L`` (a
    longer horizon provides fewer training pairs and carries a higher
    irreducible floor), normalised so the *total* auxiliary budget equals
    ``beta`` regardless of how many heads are enabled -- keeping multi-k runs
    directly comparable with the single-head C recipe.
    """
    ks = sorted(int(k) for k in ks)
    shares = torch.tensor(
        [(seq_len - k) / seq_len for k in ks], dtype=torch.float64
    )
    total = float(shares.sum())
    return {k: float(beta) * float(s) / total for k, s in zip(ks, shares)}

def s_to_params(s: torch.Tensor) -> torch.Tensor:
    """Normalised extents -> envelope params (the deployment mapping)."""
    from hexapod_mjlab.tasks.envelope_adaptive_2.oracle import (
        _physical_min_max,
    )

    min_v, max_v = _physical_min_max(_LOW, _HIGH)
    return min_v.to(s.device) + s * (max_v - min_v).to(s.device)

def batch_safety_loss(
    s_pred: torch.Tensor,
    heading: torch.Tensor,
    pos: torch.Tensor,
    map_ids: torch.Tensor,
    dfs: torch.Tensor,
):
    """Differentiable collision loss for a batch of frames.

    Uses *bilinear* distance-field sampling: the env-reward lookup is nearest
    neighbour, i.e. piecewise-constant in the sample position, so its gradient
    w.r.t. the envelope parameters is exactly zero -- unusable as an SL loss.
    The bilinear readout is piecewise-linear and its gradient points down the
    local field slope (toward the obstacle).

    Frames of different maps are evaluated group by group: a violation frame
    can only be indexed against its *own* map's field.  Floor-pinned frames
    (the minimum envelope already violates; monitored by the env as
    ``oracle_floor_pinned``) are excluded from the loss -- there the gradient
    would be a constant collapse pressure with no escape direction.

    Returns ``(mean_violation, trigger_rate)`` over the 24 boundary samples of
    the non-floor-pinned frames (or zeros when every frame is floor-pinned).
    """
    from hexapod_mjlab.tasks.envelope_adaptive_2.geometry import (
        _hex_sample_violations,
    )

    params = s_to_params(s_pred.clamp(0.0, 1.0))
    losses, weights, triggers = [], [], []
    for m in map_ids.unique():
        sel = map_ids == m
        n = int(sel.sum())
        with torch.no_grad():  # mask only; must not carry gradient
            min_params = s_to_params(torch.zeros_like(s_pred[sel]))
            floor = (
                _hex_sample_violations(
                    min_params, heading[sel], pos[sel], dfs[m],
                    margin=_SAFE_MARGIN, soft_margin=_SAFE_SOFT,
                    sampling="bilinear",
                ).max(dim=-1).values
                > _FLOOR_PINNED_VIOLATION
            )
        v = _hex_sample_violations(
            params[sel], heading[sel], pos[sel], dfs[m],
            margin=_SAFE_MARGIN, soft_margin=_SAFE_SOFT,
            sampling="bilinear",
        )[..., :24]
        frame_loss = v.mean(dim=-1)
        keep = ~floor
        n_keep = int(keep.sum())
        if n_keep:
            losses.append(frame_loss[keep].sum())
            weights.append(n_keep)
            triggers.append(float((frame_loss > 0)[keep].float().sum()) / n_keep)
    if not losses:
        zero = params.sum() * 0.0
        return zero, 0.0
    total = torch.stack(losses).sum() / sum(weights)
    rate = sum(t * w for t, w in zip(triggers, weights)) / sum(weights)
    return total, rate

def frame_map_ids(window_map_ids, sequence_length):
    """Time-major (L,B,...) flatten needs [map0,map1,...] once per time step."""
    return window_map_ids.repeat(sequence_length)
