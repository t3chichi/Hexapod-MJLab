"""Formal flat-training reward terms bound to the TransitionContext.

Numerical authority: docs/tasks/confined_locomotion/平地学习与基准测量.md §2
("已确认的速度跟踪奖励", "已确认的平地启动基础惩罚", "足端奖励与步态引导阶段",
"平地固定高度验证"). Every term:

* FIRST calls ``get_transition_context(env).ensure_ready()`` — the clean
  post-four-substep boundary truth; each term is IDEMPOTENT per control
  identity (the context caches one ready record per control step, and this
  module's per-step state holders key on the context control id, so term
  dictionary ORDER never matters and repeated manager computes do not
  double-advance anything);
* returns the RAW per-term score/cost (B,); the RewardManager applies
  ``weight * dt`` exactly once. No total-reward clipping to non-negative —
  the confirmed total keeps negative values;
* checks its own raw value AND ``raw * weight * dt`` for finiteness BEFORE
  returning, i.e. before the manager's ``nan_to_num`` could hide an
  overflow (the weight is passed through ``params`` from the SAME table
  that builds ``RewardTermCfg.weight``); the raw values are cached per
  control id so the runtime can externally recompute the full expected
  reward and compare it with ``env.reward_buf``.

Differential terms use the OLD values REALLY captured by the context
``begin`` (previous clipped action, previous joint velocity) — never values
overwritten later in ``process_actions``, and never across an episode
boundary (resets clear the action rows and the fresh episode starts from
the reset state).

The foot air-time state machine advances ONCE per control id:
``air += 0.02`` first; ``landing = current_support & ~last_support``
consumes the accumulated value; support rows then reset their accumulator
to 0; a still-unsupported foot keeps accumulating. A reset initializes the
selected rows from the REAL post-reset support with air 0 and emits NO
landing event; the next genuine transition's real landing scores normally.
The same support control step never fires landing twice (the cached
``last_support`` is exactly the previous control step's support).
"""

from __future__ import annotations

import numpy as np
import torch

from hexapod_mjlab.common.transition import get_transition_context, require_finite
from ..robot import DECIMATION

__all__ = [
    "AIR_TIME_BASELINE_S",
    "CONTROL_DT_S",
    "FOOT_CLEARANCE_TARGET_M",
    "MOTION_GATE_OMEGA_RAD_S",
    "MOTION_GATE_TRANSLATION_M_S",
    "ROOT_HEIGHT_TARGET_M",
    "TRACKING_SIGMA_OMEGA",
    "TRACKING_SIGMA_V",
    "TRAINING_REWARD_WEIGHTS",
    "FootAirTimeState",
    "build_training_reward_cfgs",
    "foot_clearance_cost",
    "foot_slip_cost",
    "foot_stumble_mask",
    "get_foot_air_state",
    "landing_air_reward",
    "motion_gate",
    "plane_world_z",
    "ground_world_z",
    "read_current_foot_support",
    "reward_foot_air",
    "pen_action_rate",
    "pen_ang_vel_xy",
    "pen_foot_clearance",
    "pen_foot_slip",
    "pen_foot_stumble",
    "pen_gravity_xy",
    "pen_joint_acc",
    "pen_joint_vel",
    "pen_lin_vel_z",
    "pen_nonfoot_env_contact",
    "pen_root_height",
    "pen_self_contact",
    "pen_soft_joint_limit",
    "pen_torque",
    "root_height_above_plane",
    "track_ang_vel",
    "track_lin_vel",
]

#: Confirmed weights (authority tables; the ONE source for both the term
#: params' finite check and RewardTermCfg.weight — they cannot diverge).
TRAINING_REWARD_WEIGHTS = {
    "track_lin_vel": 5.0,
    "track_ang_vel": 2.5,
    "pen_lin_vel_z": -2.0,
    "pen_ang_vel_xy": -0.5,
    "pen_gravity_xy": -5.0,
    "pen_torque": -1e-6,
    "pen_joint_vel": -1e-6,
    "pen_joint_acc": -1e-8,
    "pen_action_rate": -0.005,
    "pen_soft_joint_limit": -1.0,
    "pen_nonfoot_env_contact": -1.0,
    "pen_self_contact": -1.0,
    "pen_foot_stumble": -0.2,
    "pen_root_height": -50.0,
    "pen_foot_slip": -0.05,
    "reward_foot_air": 3.0,
    "pen_foot_clearance": -5.0,
}

#: Confirmed scales/thresholds (authority values, single definition).
TRACKING_SIGMA_V = 0.25  # (m/s)^2 squared-error scale, NOT a std
TRACKING_SIGMA_OMEGA = 0.25  # (rad/s)^2
ROOT_HEIGHT_TARGET_M = 0.46
FOOT_CLEARANCE_TARGET_M = 0.12
AIR_TIME_BASELINE_S = 0.25
MOTION_GATE_TRANSLATION_M_S = 0.1
MOTION_GATE_OMEGA_RAD_S = 0.1
CONTROL_DT_S = 0.02  # == flat_env_cfg SIM_TIMESTEP_S * DECIMATION


def build_training_reward_cfgs() -> dict:
    """RewardTermCfg dict from the ONE weight table (independent per call)."""
    from mjlab.managers.reward_manager import RewardTermCfg

    terms = {}
    for name, weight in TRAINING_REWARD_WEIGHTS.items():
        func = globals()[name]
        terms[name] = RewardTermCfg(func=func, weight=float(weight), params={"weight": float(weight)})
    return terms


# ── shared helpers ──────────────────────────────────────────────────────────


def _plane_world_z(env) -> float:
    """World z of the ACTUAL compiled plane (static; cached once per env).

    Derived from the real compiled model exactly like the validated plane
    identity (single horizontal plane); never a hardcoded z=0.
    """
    cached = getattr(env, "_confined_training_plane_world_z", None)
    if cached is not None:
        return float(cached)
    from hexapod_mjlab.common.ground_validation import validate_single_horizontal_plane

    record = validate_single_horizontal_plane(env.sim.mj_model)
    value = float(record["world_pos"][2])
    env._confined_training_plane_world_z = value
    return value


#: Public read of the real plane world height (terminations/runtime reuse).
plane_world_z = _plane_world_z


def ground_world_z(env, positions_w: torch.Tensor):
    """MDP ground reference only; independent of the 11x17 auxiliary labels."""
    context = getattr(env, "_confined_transition_context", None)
    query = getattr(context, "terrain_query", None)
    if query is None:
        return _plane_world_z(env)
    rows = torch.arange(env.num_envs, dtype=torch.long, device=positions_w.device)
    heights, valid = query.world_heights(rows, positions_w[..., :2].reshape(env.num_envs, -1, 2))
    if not bool(valid.all()):
        raise RuntimeError("MDP ground query outside the valid Heightfield")
    return heights.reshape(positions_w.shape[:-1])


def root_height_above_plane(env, root_pos_w: torch.Tensor) -> torch.Tensor:
    """World-VERTICAL root clearance using the task's real ground reference.

    h_actual = (p_root − p_ground)·u_W with u_W the unit world-up vector
    (authority: 平地固定高度验证) — for the validated horizontal plane this
    is the z difference against the plane's real world position. Terrain
    Memory alone uses its bound Heightfield query at the root XY.
    """
    return root_pos_w[:, 2] - ground_world_z(env, root_pos_w)


def motion_gate(command: torch.Tensor) -> torch.Tensor:
    """G_move: translation norm > 0.1 m/s OR |ωz| > 0.1 rad/s (physical)."""
    translation = torch.linalg.vector_norm(command[:, :2], dim=1)
    return (translation > MOTION_GATE_TRANSLATION_M_S) | (
        command[:, 2].abs() > MOTION_GATE_OMEGA_RAD_S
    )


def foot_stumble_mask(foot_force_w: torch.Tensor) -> torch.Tensor:
    """ANY-foot stumble: ‖F_xy‖ > 5·|F_z| per control step (0/1 per env).

    ``foot_force_w`` (B, 6, 3) is the sealed boundary ANY-sensor world net
    force per foot — the confirmed judgement with NO force threshold and no
    per-foot accumulation.
    """
    f_xy = torch.linalg.vector_norm(foot_force_w[:, :, :2], dim=2)
    f_z = foot_force_w[:, :, 2].abs()
    return (f_xy > 5.0 * f_z).any(dim=1)


def foot_slip_cost(foot_vel_w: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """P_slip = Σ_support ‖world tangential velocity‖² (link-origin point)."""
    tangential_sq = foot_vel_w[:, :, :2].square().sum(dim=2)
    return (tangential_sq * support).sum(dim=1)


def foot_clearance_cost(
    h_foot: torch.Tensor, support: torch.Tensor
) -> torch.Tensor:
    """P_clear = Σ_swing max(0.12 − h_foot, 0)² (swing = non-support feet)."""
    deficit = (FOOT_CLEARANCE_TARGET_M - h_foot).clamp(min=0.0)
    return (deficit.square() * (~support)).sum(dim=1)


def landing_air_reward(
    landing: torch.Tensor, consumed_air_s: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    """R_air = Σ_landing (t_air − 0.25 s), zero on motion-gate-off rows.

    Linear landing-time reward per the authority (no clipping to zero was
    confirmed; a negative contribution from a very short hop is the literal
    formula). Gate uses the command in force for THIS control step.
    """
    per_foot = torch.where(landing, consumed_air_s - AIR_TIME_BASELINE_S, torch.zeros_like(consumed_air_s))
    total = per_foot.sum(dim=1)
    return torch.where(gate, total, torch.zeros_like(total))


class FootAirTimeState:
    """Per-env/per-foot air accumulators + cached landing of this control id."""

    def __init__(self, num_envs: int, device) -> None:
        self.air_time_s = torch.zeros(num_envs, 6, dtype=torch.float32, device=device)
        self.last_support = torch.zeros(num_envs, 6, dtype=torch.bool, device=device)
        self.control_id: int | None = None
        self.landing: torch.Tensor | None = None
        self.consumed_air_s: torch.Tensor | None = None

    def advance(self, control_id: int, support: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Idempotent per control id; returns (landing, consumed_air_s)."""
        if self.control_id == int(control_id):
            if self.landing is None or self.consumed_air_s is None:
                raise RuntimeError("foot air state cached control id without results")
            return self.landing, self.consumed_air_s
        # 1) every foot's accumulator ages by one control step first,
        # 2) landing = current support & previously unsupported (consumes),
        # 3) support rows reset to 0, unsupported keep accumulating.
        aged = self.air_time_s + CONTROL_DT_S
        landing = support & ~self.last_support
        consumed = torch.where(landing, aged, torch.zeros_like(aged))
        self.air_time_s = torch.where(support, torch.zeros_like(aged), aged)
        self.last_support = support.detach().clone()
        self.control_id = int(control_id)
        self.landing = landing
        self.consumed_air_s = consumed
        return landing, consumed

    def reset_rows(self, env_ids: torch.Tensor, real_support: torch.Tensor) -> None:
        """Selected rows: air = 0 and the REAL post-reset support; NO event."""
        self.air_time_s[env_ids] = 0.0
        self.last_support[env_ids] = real_support


def get_foot_air_state(env) -> FootAirTimeState:
    state = getattr(env, "_confined_foot_air_state", None)
    if not isinstance(state, FootAirTimeState):
        raise RuntimeError(
            "foot air-time state not initialized; the training runtime must "
            "initialize it from the real post-reset support before any "
            "formal reward can run"
        )
    return state


def read_current_foot_support(context) -> torch.Tensor:
    """REAL foot support from the CURRENT solver state (reset init path).

    Same qualification as the context boundary support (foot body vs the
    actual ground geom, per-contact dist <= 0 and force norm > 0.1 N), read
    from the live contact table after a reset's forward — used to seed the
    air-time state truthfully (the lifted T1 reset yields no contacts, but
    the value is measured, not assumed).
    """
    table = context.contact_reader.read_substep_table()
    support = torch.zeros(context.env.num_envs, 6, dtype=torch.bool, device=context.env.device)
    if int(table["n_contacts"]) == 0:
        return support
    geom = np.asarray(table["geom"])
    for j, bid in enumerate(context.foot_global_ids):
        has = (((table["body1"] == bid) & (geom[:, 1] == context.plane_geom_id)) |
               ((table["body2"] == bid) & (geom[:, 0] == context.plane_geom_id)))
        qualified = has & (np.asarray(table["dist"]) <= 0) & (
            np.linalg.norm(np.asarray(table["force6_world"])[:, :3], axis=1) > 0.1
        )
        support[torch.as_tensor(np.asarray(table["worldid"])[qualified], device=support.device, dtype=torch.long), j] = True
    return support


def _finish_term(env, name: str, raw: torch.Tensor, weight: float) -> torch.Tensor:
    """Finite checks (raw and weight*dt) BEFORE nan_to_num; cache the raw."""
    require_finite(f"reward_raw/{name}", raw)
    scaled = raw * float(weight) * CONTROL_DT_S
    require_finite(f"reward_weighted_dt/{name}", scaled)
    context = get_transition_context(env)
    state = getattr(env, "_confined_training_reward_terms", None)
    if not isinstance(state, dict) or state.get("control_id") != context.control_id:
        state = {"control_id": context.control_id, "raws": {}}
        env._confined_training_reward_terms = state
    state["raws"][name] = raw.detach().clone()
    weighted = state.setdefault("weighted", {})
    weighted[name] = scaled.detach().clone()
    require_finite("reward_running_total", torch.stack(list(weighted.values())).sum(0))
    return raw


# ── velocity tracking (scores, positive weights) ────────────────────────────


def track_lin_vel(env, *, weight: float) -> torch.Tensor:
    record = get_transition_context(env).ensure_ready()
    error = record["lin_vel_b"][:, :2] - record["command"][:, :2]
    raw = torch.exp(-error.square().sum(dim=1) / TRACKING_SIGMA_V)
    return _finish_term(env, "track_lin_vel", raw, weight)


def track_ang_vel(env, *, weight: float) -> torch.Tensor:
    record = get_transition_context(env).ensure_ready()
    error = record["ang_vel_b"][:, 2] - record["command"][:, 2]
    raw = torch.exp(-error.square() / TRACKING_SIGMA_OMEGA)
    return _finish_term(env, "track_ang_vel", raw, weight)


# ── base penalties ──────────────────────────────────────────────────────────


def pen_lin_vel_z(env, *, weight: float) -> torch.Tensor:
    record = get_transition_context(env).ensure_ready()
    raw = record["lin_vel_b"][:, 2].square()
    return _finish_term(env, "pen_lin_vel_z", raw, weight)


def pen_ang_vel_xy(env, *, weight: float) -> torch.Tensor:
    record = get_transition_context(env).ensure_ready()
    raw = record["ang_vel_b"][:, :2].square().sum(dim=1)
    return _finish_term(env, "pen_ang_vel_xy", raw, weight)


def pen_gravity_xy(env, *, weight: float) -> torch.Tensor:
    record = get_transition_context(env).ensure_ready()
    raw = record["gravity_b"][:, :2].square().sum(dim=1)
    return _finish_term(env, "pen_gravity_xy", raw, weight)


def pen_torque(env, *, weight: float) -> torch.Tensor:
    record = get_transition_context(env).ensure_ready()
    substeps = [entry["effort"] for entry in record["pd"]]
    if len(substeps) != DECIMATION:
        raise RuntimeError(f"expected {DECIMATION} substep effort tables")
    powers = torch.stack([tau.square().sum(dim=1) for tau in substeps])
    raw = powers.mean(dim=0)
    return _finish_term(env, "pen_torque", raw, weight)


def pen_joint_vel(env, *, weight: float) -> torch.Tensor:
    record = get_transition_context(env).ensure_ready()
    raw = record["joint_vel"].square().sum(dim=1)
    return _finish_term(env, "pen_joint_vel", raw, weight)


def pen_joint_acc(env, *, weight: float) -> torch.Tensor:
    record = get_transition_context(env).ensure_ready()
    delta = record["joint_vel"] - record["previous_qdot"]
    raw = (delta / CONTROL_DT_S).square().sum(dim=1)
    return _finish_term(env, "pen_joint_acc", raw, weight)


def pen_action_rate(env, *, weight: float) -> torch.Tensor:
    record = get_transition_context(env).ensure_ready()
    delta = record["clipped_action"] - record["previous_action"]
    raw = delta.square().sum(dim=1)
    return _finish_term(env, "pen_action_rate", raw, weight)


def pen_soft_joint_limit(env, *, weight: float) -> torch.Tensor:
    record = get_transition_context(env).ensure_ready()
    raw = (record["joint_pos"].abs() - 2.7).clamp(min=0.0).sum(dim=1)
    return _finish_term(env, "pen_soft_joint_limit", raw, weight)


def pen_nonfoot_env_contact(env, *, weight: float) -> torch.Tensor:
    record = get_transition_context(env).ensure_ready()
    raw = record["c_nonfoot_env"]
    return _finish_term(env, "pen_nonfoot_env_contact", raw, weight)


def pen_self_contact(env, *, weight: float) -> torch.Tensor:
    record = get_transition_context(env).ensure_ready()
    raw = record["c_self"]
    return _finish_term(env, "pen_self_contact", raw, weight)


def pen_foot_stumble(env, *, weight: float) -> torch.Tensor:
    record = get_transition_context(env).ensure_ready()
    raw = foot_stumble_mask(record["foot_force_w"]).to(torch.float32)
    return _finish_term(env, "pen_foot_stumble", raw, weight)


def pen_root_height(env, *, weight: float) -> torch.Tensor:
    record = get_transition_context(env).ensure_ready()
    height = root_height_above_plane(env, record["root_pos_w"])
    raw = (height - ROOT_HEIGHT_TARGET_M).square()
    return _finish_term(env, "pen_root_height", raw, weight)


# ── foot rewards (slip always on; air/clearance behind the motion gate) ─────


def pen_foot_slip(env, *, weight: float) -> torch.Tensor:
    record = get_transition_context(env).ensure_ready()
    raw = foot_slip_cost(record["foot_vel_w"], record["support"])
    return _finish_term(env, "pen_foot_slip", raw, weight)


def reward_foot_air(env, *, weight: float) -> torch.Tensor:
    context = get_transition_context(env)
    record = context.ensure_ready()
    landing, consumed = get_foot_air_state(env).advance(context.control_id, record["support"])
    raw = landing_air_reward(landing, consumed, motion_gate(record["command"]))
    return _finish_term(env, "reward_foot_air", raw, weight)


def pen_foot_clearance(env, *, weight: float) -> torch.Tensor:
    record = get_transition_context(env).ensure_ready()
    plane_z = ground_world_z(env, record["foot_pos_w"])
    h_foot = record["foot_pos_w"][:, :, 2] - plane_z
    raw = torch.where(
        motion_gate(record["command"]),
        foot_clearance_cost(h_foot, record["support"]),
        torch.zeros_like(record["command"][:, 0]),
    )
    return _finish_term(env, "pen_foot_clearance", raw, weight)
