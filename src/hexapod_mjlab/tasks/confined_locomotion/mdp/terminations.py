"""Formal flat-training terminations bound to the TransitionContext.

Authority: docs/tasks/confined_locomotion/平地学习与基准测量.md §2
"训练episode结束与异常处理" (confirmed 2026-10-07):

* low-height fall: real root height h_actual < 0.20 m for 10 CONSECUTIVE
  control steps (0.20 s) → termination; the counter clears on interruption;
* sustained tilt: body +z vs world-up angle > 60° for 10 consecutive steps
  → termination; counter clears on interruption;
* flip: angle > 90° → IMMEDIATE termination;
* timeout: 20 s = 1000 control steps → truncation (``time_out=True``);
  a real fall in the SAME step suppresses the timeout flag (true fall has
  priority, both flags stored separately — never guessed from wrapper
  dones);
* collisions NEVER terminate; NaN faults stop the whole run elsewhere.

Angle convention (authority): cos(angle(body +z, world up)) = −g_b[2] with
g_b the unit projected gravity; thresholds compare STRICTLY
(up_dot < 0.5 for >60°, up_dot < 0.0 for >90°; h < 0.20 strict). All
conditions read the SAME sealed boundary record as the rewards (one
``ensure_ready`` per control step); the shared state holder updates once
per control id, so term dictionary order never matters and repeated
manager computes never double-count.

The counters/flags live on the env instance; the training runtime clears
the selected rows at every selected reset. An EXPLICIT, recorded preset
API exists for isolated lifecycle fixtures only (bring a counter next to a
gate to avoid waiting 20 s of real physics); presets are logged and are
never applied silently.
"""

from __future__ import annotations

import torch

from ..transition import get_transition_context
from .rewards import plane_world_z

__all__ = [
    "TRAINING_TERMINATION_TIME_OUT_FLAGS",
    "TrainingTerminationState",
    "advance_fall_state",
    "build_training_termination_cfgs",
    "episode_timeout",
    "fall_flipped",
    "fall_low_height",
    "fall_tilt_sustained",
    "get_termination_state",
]

#: Confirmed thresholds/durations (authority values).
LOW_HEIGHT_THRESHOLD_M = 0.20
LOW_HEIGHT_PERSISTENCE_STEPS = 10
TILT_THRESHOLD_COS = 0.5  # angle > 60°  ⇔  up_dot < cos60° (strict)
FLIP_THRESHOLD_COS = 0.0  # angle > 90°  ⇔  up_dot < 0 (strict)
EPISODE_TIMEOUT_CONTROL_STEPS = 1000  # 20 s at 50 Hz


def build_training_termination_cfgs() -> dict:
    """TerminationTermCfg dict (independent per call).

    time_out flags: only the 20 s episode budget truncates; the three fall
    terms are real terminations.
    """
    from mjlab.managers.termination_manager import TerminationTermCfg

    return {
        "fall_low_height": TerminationTermCfg(func=fall_low_height, time_out=False),
        "fall_tilt_sustained": TerminationTermCfg(func=fall_tilt_sustained, time_out=False),
        "fall_flipped": TerminationTermCfg(func=fall_flipped, time_out=False),
        "episode_timeout": TerminationTermCfg(func=episode_timeout, time_out=True),
    }


#: term name → time_out flag, for external contract checks.
TRAINING_TERMINATION_TIME_OUT_FLAGS = {
    "fall_low_height": False,
    "fall_tilt_sustained": False,
    "fall_flipped": False,
    "episode_timeout": True,
}


def advance_fall_state(
    low_height: torch.Tensor,
    tilted: torch.Tensor,
    flipped: torch.Tensor,
    low_height_count: torch.Tensor,
    tilt_count: torch.Tensor,
) -> dict:
    """Pure one-step fall-state advance (unit-testable without an env).

    Each persistence counter increments while its condition holds and
    clears on interruption; the flip condition is immediate. Returns new
    counters, per-condition done flags and ``any_fall``.
    """
    new_low_count = torch.where(low_height, low_height_count + 1, torch.zeros_like(low_height_count))
    new_tilt_count = torch.where(tilted, tilt_count + 1, torch.zeros_like(tilt_count))
    low_done = new_low_count >= LOW_HEIGHT_PERSISTENCE_STEPS
    tilt_done = new_tilt_count >= LOW_HEIGHT_PERSISTENCE_STEPS
    flip_done = flipped
    return {
        "low_height_count": new_low_count,
        "tilt_count": new_tilt_count,
        "low_height_done": low_done,
        "tilt_done": tilt_done,
        "flip_done": flip_done,
        "any_fall": low_done | tilt_done | flip_done,
    }


class TrainingTerminationState:
    """Per-env fall counters + the cached done flags of this control id."""

    def __init__(self, num_envs: int, device) -> None:
        self.low_height_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.tilt_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.low_height_done = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.tilt_done = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.flip_done = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.any_fall = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.control_id: int | None = None
        #: explicit, recorded fixture presets (never applied silently).
        self.presets: list[dict] = []

    def ensure_updated(self, env, control_id: int) -> None:
        """Idempotent per control id; conditions read the sealed record."""
        if self.control_id == int(control_id):
            return
        record = get_transition_context(env).ensure_ready()
        height = record["root_pos_w"][:, 2] - plane_world_z(env)
        up_dot = -record["gravity_b"][:, 2]
        advanced = advance_fall_state(
            height < LOW_HEIGHT_THRESHOLD_M,
            up_dot < TILT_THRESHOLD_COS,
            up_dot < FLIP_THRESHOLD_COS,
            self.low_height_count,
            self.tilt_count,
        )
        self.low_height_count = advanced["low_height_count"]
        self.tilt_count = advanced["tilt_count"]
        self.low_height_done = advanced["low_height_done"]
        self.tilt_done = advanced["tilt_done"]
        self.flip_done = advanced["flip_done"]
        self.any_fall = advanced["any_fall"]
        self.control_id = int(control_id)

    def reset_rows(self, env_ids: torch.Tensor) -> None:
        for tensor in (
            self.low_height_count,
            self.tilt_count,
            self.low_height_done,
            self.tilt_done,
            self.flip_done,
            self.any_fall,
        ):
            tensor[env_ids] = 0

    # ── explicit fixture presets (recorded; no silent defaults) ─────────────

    def preset_low_height_counts(self, counts: torch.Tensor) -> None:
        self.presets.append(
            {
                "field": "low_height_count",
                "values": counts.detach().cpu().tolist(),
            }
        )
        self.low_height_count = counts.detach().to(
            device=self.low_height_count.device, dtype=torch.long
        ).clone()

    def preset_tilt_counts(self, counts: torch.Tensor) -> None:
        self.presets.append(
            {
                "field": "tilt_count",
                "values": counts.detach().cpu().tolist(),
            }
        )
        self.tilt_count = counts.detach().to(
            device=self.tilt_count.device, dtype=torch.long
        ).clone()


def get_termination_state(env) -> TrainingTerminationState:
    state = getattr(env, "_confined_termination_state", None)
    if not isinstance(state, TrainingTerminationState):
        state = TrainingTerminationState(env.num_envs, env.device)
        env._confined_termination_state = state
    return state


# ── the terms ───────────────────────────────────────────────────────────────


def fall_low_height(env) -> torch.Tensor:
    context = get_transition_context(env)
    context.ensure_ready()
    state = get_termination_state(env)
    state.ensure_updated(env, context.control_id)
    return state.low_height_done


def fall_tilt_sustained(env) -> torch.Tensor:
    context = get_transition_context(env)
    context.ensure_ready()
    state = get_termination_state(env)
    state.ensure_updated(env, context.control_id)
    return state.tilt_done


def fall_flipped(env) -> torch.Tensor:
    context = get_transition_context(env)
    context.ensure_ready()
    state = get_termination_state(env)
    state.ensure_updated(env, context.control_id)
    return state.flip_done


def episode_timeout(env) -> torch.Tensor:
    """20 s budget as truncation, SUPPRESSED on rows with a real fall.

    The step count comes from the env's own episode counter (zeroed by
    every selected reset); the fall suppression reads the SAME control id's
    fall state, so this term does not depend on term ordering.
    """
    context = get_transition_context(env)
    context.ensure_ready()
    state = get_termination_state(env)
    state.ensure_updated(env, context.control_id)
    reached = env.episode_length_buf >= EPISODE_TIMEOUT_CONTROL_STEPS
    return reached & ~state.any_fall
