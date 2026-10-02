"""Pure EA2 action, motion, reward and observation mathematics."""
from __future__ import annotations
import math
from typing import Optional, Sequence, Tuple
import torch
from . import contracts as ea2c
from .range_image import range_image_observation
def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    """Wrap an angle/tensor to ``[-pi, pi)`` using PyTorch semantics."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi

def yaw_quat_from_heading(heading: torch.Tensor) -> torch.Tensor:
    """Build a z-up yaw quaternion ``[x, y, z, w]`` from a body heading.

    M1 has no pitch/roll, so the robot/body orientation is fully described by
    ``heading``.  This helper is the single source of truth for converting
    heading to the quaternion used by LiDAR raycasting and debug drawing.
    """
    half = heading * 0.5
    return torch.stack(
        [
            torch.zeros_like(half),
            torch.zeros_like(half),
            torch.sin(half),
            torch.cos(half),
        ],
        dim=-1,
    )

def height_step(
    height_filter: torch.Tensor,
    height_target: torch.Tensor,
    wobble_state: torch.Tensor,
    noise: torch.Tensor,
    min_m: float,
    max_m: float,
    dt: float,
    tau_s: float,
    wobble_fc_hz: float,
    wobble_amp: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance the time-varying carrier height model one control step.

    Implements README 2.2.7: a first-order low-pass filter toward a
    periodically resampled target plus a separate 1 Hz low-pass AR wobble.

    Returns ``(height_filter, wobble_state, height)``.  ``wobble_state`` is a
    normalized AR(1) state; the returned height is clipped to ``[min_m,
    max_m]``.
    """
    alpha = 1.0 - math.exp(-dt / max(tau_s, 1e-6))
    height_filter = height_filter + alpha * (height_target - height_filter)
    a = math.exp(-2.0 * math.pi * max(wobble_fc_hz, 0.0) * dt)
    wobble_state = a * wobble_state + math.sqrt(max(1.0 - a * a, 0.0)) * noise
    height = (height_filter + wobble_amp * wobble_state).clamp(min_m, max_m)
    return height_filter, wobble_state, height

def sway_update(
    state: torch.Tensor,
    amp: float,
    dt: float,
    fc_hz: float,
    noise: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Advance a first-order low-pass sway state.

    The returned offset is ``amp * state`` where ``state`` is an AR(1) process
    driven by unit-variance noise.  Used for both lateral position sway and
    heading sway (README 2.2.7).
    """
    a = math.exp(-2.0 * math.pi * max(fc_hz, 0.0) * dt)
    state = a * state + math.sqrt(max(1.0 - a * a, 0.0)) * noise
    return state, amp * state

def heading_update(
    heading: torch.Tensor,
    tangent: torch.Tensor,
    tangent_rate: torch.Tensor,
    delta_target: torch.Tensor,
    dt: float,
    k_p: float,
    omega_max: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tangent-relative heading controller (README 2.2.5).

    Returns ``(heading, omega, delta_actual)``.
    """
    delta_actual = wrap_to_pi(heading - tangent)
    omega_cmd = tangent_rate + k_p * wrap_to_pi(delta_target - delta_actual)
    omega = torch.clamp(
        torch.as_tensor(omega_cmd, dtype=torch.float32), -omega_max, omega_max
    )
    heading = wrap_to_pi(heading + omega * dt)
    delta_actual = wrap_to_pi(heading - tangent)
    return heading, omega, delta_actual

def ego_motion(
    v: torch.Tensor,
    heading: torch.Tensor,
    tangent: torch.Tensor,
    omega: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Body-frame ego motion from path tangent tracking (README 2.2.5/2.4)."""
    delta = heading - tangent
    return v * torch.cos(delta), v * torch.sin(delta), omega

def default_params(
    low: torch.Tensor,
    high: torch.Tensor,
) -> torch.Tensor:
    """Return the envelope midpoint (action=0 corresponds to this pose)."""
    return 0.5 * (low + high)

def envelope_action_scale(
    low: torch.Tensor,
    high: torch.Tensor,
    soft_dof_pos_limit: float,
    action_max: float,
) -> torch.Tensor:
    """Per-parameter linear action scale, mirroring standard legged_gym.

    ``action_max`` is the raw-action radius that reaches the soft bounds;
    ``soft_dof_pos_limit`` reserves a small band before the true hard clamp.
    """
    span = high - low
    return span * soft_dof_pos_limit / (2.0 * action_max)

def linear_params_target(
    actions: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    soft_dof_pos_limit: float,
    action_max: float,
) -> torch.Tensor:
    """Compute the unclamped linear envelope target from raw actions."""
    default = default_params(low, high)
    scale = envelope_action_scale(low, high, soft_dof_pos_limit, action_max)
    return default + actions * scale

def map_actions_to_params(
    actions: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    soft_dof_pos_limit: float = 0.9,
    action_max: float = 4.0,
) -> torch.Tensor:
    """Map raw actions to hard-clamped envelope parameters (linear + clamp).

    Thin wrapper around :func:`map_actions_to_params_with_target`; the
    unclamped target is available through that variant for the soft-limit
    reward.
    """
    params, _ = map_actions_to_params_with_target(
        actions, low, high, soft_dof_pos_limit, action_max
    )
    return params

def map_actions_to_params_with_target(
    actions: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    soft_dof_pos_limit: float = 0.9,
    action_max: float = 4.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(clamped_params, unclamped_target)`` for linear mapping."""
    target = linear_params_target(
        actions, low, high, soft_dof_pos_limit, action_max
    )
    return torch.clamp(target, low, high), target

def envelope_limit_violation(
    target: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    soft_dof_pos_limit: float = 0.9,
) -> torch.Tensor:
    """Soft out-of-limits penalty for the unclamped linear target.

    The soft bounds are the central ``soft_dof_pos_limit`` fraction of the
    ``[low, high]`` range.  Penalizing the unclamped target (rather than the
    clamped params) makes the reward keep providing a gradient when the policy
    tries to push beyond the working range.
    """
    margin = (1.0 - soft_dof_pos_limit) * (high - low) / 2.0
    soft_low = low + margin
    soft_high = high - margin

    low_over = torch.relu(soft_low - target)
    high_over = torch.relu(target - soft_high)
    return torch.sum(low_over ** 2 + high_over ** 2, dim=-1)

def normalized_envelope_params(
    params5: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
) -> torch.Tensor:
    """Return each envelope parameter normalized to ``[0, 1]``.

    ``backward_limit`` is normalized in its physical direction so that a
    more negative value (larger rear extent) maps to 1, matching
    :func:`potential_reward`.
    """
    low5 = low[:5]
    high5 = high[:5]
    norm = (params5 - low5) / (high5 - low5).clamp_min(1e-6)
    # backward_limit: value_low = -high, value_high = -low.
    norm = norm.clone()
    norm[..., 4] = (-params5[..., 4] - (-high5[4])) / (
        (-low5[4]) - (-high5[4])
    ).clamp_min(1e-6)
    return norm.clamp(0.0, 1.0)

def potential_reward(
    params5: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
) -> torch.Tensor:
    """Normalized-parameter mean potential (README 2.8).

    ``backward_limit`` is negated before normalization, matching the physical
    direction used by ``apply_env_morphology_priors``.
    """
    return normalized_envelope_params(params5, low, high).mean(dim=-1)

def raw_action_rate_term(
    actions: torch.Tensor,
    last_actions: torch.Tensor,
) -> torch.Tensor:
    """Sum squared change of raw actions, mirroring standard legged_gym.

    Raw-action rate is used because the mapped parameters saturate at the hard
    clamp; a mapped-parameter rate would become ~0 in the saturated region and
    would not prevent raw-action drift.
    """
    return torch.sum((actions - last_actions) ** 2, dim=-1)

def speed_mixture_draw(
    u_mode: torch.Tensor,
    u_speed: torch.Tensor,
    p_zero: float,
    max_speed: float,
) -> torch.Tensor:
    """Stage 2 speed schedule: 0 with probability ``p_zero``, else uniform.

    ``u_mode``/``u_speed`` are unit-interval draws.  The zero-speed mode
    provides stationary-hold states so the GRU learns "ego ~ 0 -> keep the
    memory", and the uniform part covers slow passes whose invisible tails
    are longer than the training window.
    """
    zero = u_mode < float(p_zero)
    return torch.where(zero, torch.zeros_like(u_speed), u_speed * float(max_speed))

def assemble_observation(
    range_image: torch.Tensor,
    ego: torch.Tensor,
    max_range: float = ea2c.EA2_RANGE_MAX_M,
    ego_scales: Sequence[float] = (1.5, 1.0, 1.5),
) -> torch.Tensor:
    """Concatenate normalized range image and normalized ego-motion."""
    normalized_range = range_image_observation(range_image, max_range)
    scales = normalized_range.new_tensor(ego_scales, dtype=torch.float32)
    normalized_ego = ego / scales
    return torch.cat([normalized_range, normalized_ego], dim=-1)

def empty_range_image(
    num_envs: int,
    max_range: float = ea2c.EA2_RANGE_MAX_M,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return an empty range-image buffer filled with ``max_range``."""
    return torch.full(
        (int(num_envs), ea2c.EA2_RANGE_DIM),
        max_range,
        dtype=torch.float32,
        device=device,
    )

def refresh_range_image_from_scan(
    range_image: torch.Tensor,
    fresh: torch.Tensor,
    stale: torch.Tensor,
) -> torch.Tensor:
    """Copy a global LiDAR scan into the live range-image buffer.

    README 2.4 empty-frame contract: an env reset between two global scans
    keeps the all-``max_range`` empty frame only until the next global scan.
    At that scan it must receive the freshly computed aggregate from its new
    pose.  Therefore this helper copies the whole fresh frame for every env
    (including reset-before-scan envs) and only then clears the stale marker;
    it deliberately does NOT replace ``fresh`` rows with ``max_range``.

    This is an Isaac-free helper so the contract can be unit-tested without
    creating a simulation.  It mutates and returns ``range_image``.
    """
    range_image.copy_(fresh.to(device=range_image.device))
    stale[:] = False
    return range_image

def selected_channel_mask(
    dists: torch.Tensor,
    selected_indices: torch.Tensor,
    is_reduced: bool,
    far_plane: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(red, green)`` boolean masks for debug visualization.

    In reduced mode the debug cloud contains only the 187 selected channels,
    so every valid hit is red.  In full mode the debug cloud contains the
    complete 86,400-ray Airy cloud, so only ``selected_indices`` are red and
    the remaining valid hits are green.

    Args:
        dists: Debug ray distances, shape ``(N,)``.
        selected_indices: Full Airy indices of the 187 selected channels.
        is_reduced: Whether ``dists`` is the reduced 187-channel cloud.
        far_plane: Sensor no-hit distance.

    Returns:
        ``(red, green)`` boolean masks with the same shape as ``dists``.
    """
    hit = dists < (float(far_plane) - 1e-3)
    if is_reduced or dists.shape[0] == ea2c.EA2_RANGE_DIM:
        selected = torch.ones_like(hit)
    else:
        selected = torch.zeros_like(hit)
        selected[selected_indices.to(device=dists.device)] = True
    red = selected & hit
    green = hit & ~red
    return red, green
