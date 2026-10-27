"""18-dim residual PD action with exact recording.

Confirmed semantics (平地学习与基准测量.md §2; mirror of the pure-compute
reference ``rl/joint_passage/control.py``):

* raw action is clipped to ±100 FIRST (dimensionless);
* joint target = q_ref + 0.25 · clipped (rad; q_ref = T1 JOINT_TARGETS;
  NO position-target clipping — no ±3/±2.7 target clamp anywhere here);
* at EVERY 200 Hz physics substep (mjlab calls ``apply_actions()`` before
  each substep) recompute 130·(target − q) − 2·q̇ against the instantaneous
  joint state and clamp the torque to ±80 N·m;
* the action term stores, per control step: raw / clipped action and the
  joint target; per substep (when recording is enabled): q, q̇, pre-clip PD
  torque, applied torque and the saturation mask;
* ``prev_action`` exposed for the next control step's observation is the
  CLIPPED dimensionless action (no ×0.25, no q_ref offset);
* partial reset clears ONLY the target envs' previous-action and runtime
  buffers and restores the PD target to q_ref for those rows; retained
  diagnostic tables are NEVER erased (resets are logged instead).

Numerical contract (identical in spirit to ``rl/joint_passage/control.py``):
raw action, joint target, q, q̇ and the pre-clip torque are explicitly
checked for tensor type, shape (B, 18), float32, same-device and FULL
FINITENESS **before** any clamp. Non-finite input (e.g. Inf) is a hard
``RuntimeError`` — it is NEVER silently clamped into a ±80 torque. Only the
documented ±80 saturation of an otherwise finite PD torque is a clamp.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.envs.mdp.actions.actions import BaseAction, BaseActionCfg

_JOINT_DIM = 18


class ActionContractError(RuntimeError):
  """Raw action / PD input violated the finite-shape contract (no silent fix)."""


@dataclass(kw_only=True)
class ResidualPDActionCfg(BaseActionCfg):
  """Configuration for the residual PD-torque joint action."""

  raw_clip: float = 100.0
  """Clip applied to the RAW action before anything else (±100)."""
  residual_scale: float = 0.25
  """Target = q_ref + residual_scale · clipped [rad]."""
  stiffness: float = 130.0
  damping: float = 2.0
  effort_limit: float = 80.0
  record_substeps: bool = False
  """Record per-substep q/q̇/torque tables (explicit diagnostic mode only)."""
  max_recorded_control_steps: int = 120
  """Bound on retained control-step records (memory safety)."""

  def build(self, env):
    return ResidualPDAction(self, env)


class ResidualPDAction(BaseAction):
  """clip(±100) → q_ref + 0.25·clipped → per-substep PD → clamp ±80 N·m."""

  cfg: ResidualPDActionCfg

  def __init__(self, cfg: ResidualPDActionCfg, env):
    super().__init__(cfg=cfg, env=env)
    n = env.num_envs
    d = env.device
    # q_ref comes from the entity default joint pos (= T1 JOINT_TARGETS in
    # this task's robot cfg). Frozen at build: the fixed PD reference.
    self._q_ref = (
      self._entity.data.default_joint_pos[:, self._target_ids].clone().detach()
    )
    self.prev_action = torch.zeros(n, self._q_ref.shape[1], device=d)
    self.last_raw_action = torch.zeros_like(self.prev_action)
    self.last_clipped_action = torch.zeros_like(self.prev_action)
    self.last_target = self._q_ref.clone()
    # Zero-residual target before the first process_actions call.
    self._processed_actions = self._q_ref.clone()
    self.substep_records: list[dict] = []
    self.reset_log: list[dict] = []
    self.control_steps_seen = 0

  @property
  def q_ref(self) -> torch.Tensor:
    """Fixed PD joint reference (B, 18) — T1 JOINT_TARGETS."""
    return self._q_ref

  # ── contract checks (reject, never silently repair) ───────────────────────

  def _check_joint_tensor(self, name: str, value: torch.Tensor) -> torch.Tensor:
    """(B, 18) float32 on the env device and FULLY finite, or hard error."""
    if not isinstance(value, torch.Tensor):
      raise ActionContractError(
        f"{name} must be a torch.Tensor, got {type(value).__name__}"
      )
    if value.dim() != 2:
      raise ActionContractError(
        f"{name} must be 2-D (B, {_JOINT_DIM}), got shape {tuple(value.shape)}"
      )
    if int(value.shape[1]) != self._q_ref.shape[1]:
      raise ActionContractError(
        f"{name} must have {self._q_ref.shape[1]} columns, "
        f"got shape {tuple(value.shape)}"
      )
    if int(value.shape[0]) != int(self._q_ref.shape[0]):
      raise ActionContractError(
        f"{name} batch {int(value.shape[0])} != num_envs "
        f"{int(self._q_ref.shape[0])}"
      )
    if value.dtype is not torch.float32:
      raise ActionContractError(
        f"{name} dtype must be float32, got {value.dtype}"
      )
    if value.device != self._q_ref.device:
      raise ActionContractError(
        f"{name} device {value.device} != env device {self._q_ref.device}"
      )
    if not bool(torch.isfinite(value).all()):
      raise ActionContractError(
        f"{name} contains NaN/Inf; the execution chain requires fully "
        "finite inputs (no silent clamp of Inf into a ±80 torque)"
      )
    return value

  def _check_finite_computed(self, name: str, value: torch.Tensor) -> None:
    """Finite inputs must still produce finite intermediates (Inf−Inf guard)."""
    if not bool(torch.isfinite(value).all()):
      raise ActionContractError(
        f"{name} contains NaN/Inf computed from finite inputs "
        "(e.g. overflow / Inf−Inf); rejecting instead of silently clamping"
      )

  # ── control-step processing ────────────────────────────────────────────────

  def process_actions(self, action_tensor, *args, **kwargs) -> None:
    """clip raw ±100 → target = q_ref + 0.25·clipped, with hard contract
    checks on the raw action BEFORE any clipping."""
    raw = action_tensor.detach().clone()
    self._check_joint_tensor("raw_action", raw)
    clipped = torch.clamp(raw, -self.cfg.raw_clip, self.cfg.raw_clip)
    target = self._q_ref + self.cfg.residual_scale * clipped
    self._check_finite_computed("joint_target", target)
    self.last_raw_action = raw
    self.last_clipped_action = clipped
    self.last_target = target
    # Base introspection must see the ACTUAL raw action of this step.
    self._raw_actions[:] = raw
    # Observation contract: the NEXT control step's proprio reads THIS step's
    # clipped dimensionless action (never the raw sample, never the target).
    self.prev_action = clipped.clone()
    self._processed_actions = target
    self.control_steps_seen += 1
    if self.cfg.record_substeps:
      if len(self.substep_records) >= self.cfg.max_recorded_control_steps:
        self.substep_records.pop(0)  # bounded ring
      self.substep_records.append(
        {
          "raw": raw.detach().cpu().numpy().copy(),
          "clipped": clipped.detach().cpu().numpy().copy(),
          "target": target.detach().cpu().numpy().copy(),
          "q": [],
          "qdot": [],
          "tau_preclip": [],
          "tau_applied": [],
          "saturated": [],
          "reset_marks": [],
        }
      )

  def apply_actions(self) -> None:
    """Called before EVERY physics substep: recompute PD, clamp ±80, apply.

    q, q̇, the target and the pre-clip torque are finite-checked BEFORE the
    ±80 clamp; a fault aborts the step (it is never folded into ±80)."""
    target = self._processed_actions
    self._check_joint_tensor("joint_target", target)
    joint_pos = self._entity.data.joint_pos[:, self._target_ids]
    self._check_joint_tensor("joint_pos", joint_pos)
    joint_vel = self._entity.data.joint_vel[:, self._target_ids]
    self._check_joint_tensor("joint_vel", joint_vel)
    tau = self.cfg.stiffness * (target - joint_pos) - self.cfg.damping * joint_vel
    self._check_finite_computed("tau_preclip", tau)
    applied = torch.clamp(tau, -self.cfg.effort_limit, self.cfg.effort_limit)
    saturated = tau.abs() > (self.cfg.effort_limit + 1e-9)
    self._entity.set_joint_effort_target(applied, joint_ids=self._target_ids)
    if self.cfg.record_substeps and self.substep_records:
      rec = self.substep_records[-1]
      rec["q"].append(joint_pos.detach().cpu().numpy().copy())
      rec["qdot"].append(joint_vel.detach().cpu().numpy().copy())
      rec["tau_preclip"].append(tau.detach().cpu().numpy().copy())
      rec["tau_applied"].append(applied.detach().cpu().numpy().copy())
      rec["saturated"].append(saturated.detach().cpu().numpy().copy())

  # ── reset (partial, per-row) ───────────────────────────────────────────────

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """Partial reset: clear ONLY the target envs' runtime action rows and
    restore their PD target to q_ref. Retained diagnostic tables (past
    evidence) are never erased — resets are appended to ``reset_log`` and the
    in-flight record (if any) is annotated."""
    if env_ids is None:
      env_ids = slice(None)
    self._raw_actions[env_ids] = 0.0
    self.prev_action[env_ids] = 0.0
    self.last_raw_action[env_ids] = 0.0
    self.last_clipped_action[env_ids] = 0.0
    self.last_target[env_ids] = self._q_ref[env_ids]
    self._processed_actions[env_ids] = self._q_ref[env_ids]
    if isinstance(env_ids, torch.Tensor):
      ids = env_ids.detach().cpu().tolist()
    else:
      ids = "all"
    self.reset_log.append(
      {"control_steps_seen": self.control_steps_seen, "env_ids": ids}
    )
    if len(self.reset_log) > 64:
      self.reset_log.pop(0)
    if self.cfg.record_substeps and self.substep_records:
      rec = self.substep_records[-1]
      if rec["q"] and len(rec["q"]) != 4:
        # A reset landed inside an incomplete control step: mark it (evidence
        # is preserved, never rewritten).
        rec["reset_marks"].append(
          {"env_ids": ids, "substeps_seen": len(rec["q"])}
        )

  def reset_buffers(self, env_ids: torch.Tensor) -> None:
    """Reset-event entry point; identical per-row clearing as ``reset``."""
    self.reset(env_ids=env_ids)

  def clear_records(self) -> None:
    """Drop retained substep tables (does not touch live action state)."""
    self.substep_records.clear()
