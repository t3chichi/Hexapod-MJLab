"""Control-boundary observation snapshots for the clean training batch (P1).

This module owns the ONLINE observation identity only — it never touches
scans, the window, commands, rewards, or the network:

* ``build_control_observation`` reads the CURRENT clean robot state at a
  pre-action control boundary (the post-official-final-forward state the
  training runtime prepares inputs at) and returns an unscaled
  ``OnlineObservation``: body-frame root linear/angular velocity, unit
  projected gravity, q − q_ref, joint velocity and the previous CLIPPED
  dimensionless action (63 proprio dims), with the 3-dim body command kept
  SEPARATE. Every field is an independent fresh clone, so no later live
  state mutation can leak into a retained snapshot;
* the CONFIRMED observation multipliers (lin ×2, ang ×0.25, …) are applied
  exactly ONCE, inside the network forward — this module, the manager
  group and any storage stay unscaled (docs/research/训练架构对齐.md §1).
  This clean batch adds no noise, dropout, gap or mirror;
* ``clone_prepared`` produces a fully independent deep copy of a
  ``PreparedControlInputs`` (including a fresh ``OnlineObservation``), so a
  retained transition input can never alias the live cached input;
* ``ControlInputCache`` implements the repeatable-read contract: at the
  same control identity (global step counter + episode vector) repeated
  ``prepare`` reads return the SAME cached object — no new scan, no new
  command sampling, no new clones — and any identity change (a selected
  reset at the same counter, a completed step) invalidates it explicitly.

Each fresh point-sequence encoding starts with zero GRU hidden state. The
collector/play feature cache may reuse that encoding between scans; this
module carries neither GRU hidden state nor encoded features.
"""

from __future__ import annotations

import torch

from hexapod_mjlab.rl.confined_locomotion.contracts import (
    JOINT_DIM,
    OnlineObservation,
)
from hexapod_mjlab.rl.confined_locomotion.integration import (
    PreparedControlInputs,
)

__all__ = [
    "ControlInputCache",
    "build_control_observation",
    "clone_prepared",
]

_OBSERVATION_FIELDS = (
    "env_ids",
    "episode_ids",
    "step_ids",
    "time_s",
    "lin_vel_b",
    "ang_vel_b",
    "gravity_b",
    "joint_pos_rel_ref",
    "joint_vel",
    "prev_action",
    "command",
)


def build_control_observation(
    env,
    *,
    episode_ids: torch.Tensor,
    step_ids: torch.Tensor,
    control_time_s: float,
    command: torch.Tensor,
) -> OnlineObservation:
    """Unscaled ``OnlineObservation`` cloned from the CURRENT clean state.

    ``env`` is the unwrapped ``ManagerBasedRlEnv``; the read happens at a
    natural pre-action boundary (after the previous step's official final
    forward, or after a reset's refresh). ``command`` (B, 3) is embedded as
    given — the caller guarantees it is the same physical body-frame
    command that will drive this control step's rewards.
    """
    robot = env.scene["robot"]
    action = env.action_manager.get_term("residual_pd")
    num_envs = int(env.num_envs)
    device = torch.device(env.device)
    joint_pos = robot.data.joint_pos.detach()
    joint_vel = robot.data.joint_vel.detach()
    if tuple(joint_pos.shape) != (num_envs, JOINT_DIM) or tuple(joint_vel.shape) != (
        num_envs,
        JOINT_DIM,
    ):
        raise RuntimeError(
            f"robot joint state shapes {tuple(joint_pos.shape)}/"
            f"{tuple(joint_vel.shape)} != ({num_envs}, {JOINT_DIM})"
        )
    if tuple(command.shape) != (num_envs, 3) or command.device != device:
        raise ValueError(
            f"command must be ({num_envs}, 3) on {device}, got "
            f"{tuple(command.shape)} on {command.device}"
        )
    return OnlineObservation(
        env_ids=torch.arange(num_envs, dtype=torch.long, device=device),
        episode_ids=episode_ids.detach().to(device=device, dtype=torch.long).clone(),
        step_ids=step_ids.detach().to(device=device, dtype=torch.long).clone(),
        time_s=torch.full(
            (num_envs,), float(control_time_s), dtype=torch.float32, device=device
        ),
        lin_vel_b=robot.data.root_link_lin_vel_b.detach().clone(),
        ang_vel_b=robot.data.root_link_ang_vel_b.detach().clone(),
        gravity_b=robot.data.projected_gravity_b.detach().clone(),
        joint_pos_rel_ref=(joint_pos - action.q_ref.detach()).clone(),
        joint_vel=joint_vel.clone(),
        prev_action=action.prev_action.detach().clone(),
        command=command.detach().to(device=device, dtype=torch.float32).clone(),
    )


def clone_prepared(prepared: PreparedControlInputs) -> PreparedControlInputs:
    """Fully independent copy of a prepared input (no aliasing anywhere).

    The clone owns fresh storage for the observation (all fields), the
    near/far point blocks, validity masks and scan-identity metadata, so a
    retained transition input is immune to later window writes, resets or
    live-state mutation of the cached source object.
    """
    observation = prepared.observation
    cloned_observation = OnlineObservation(
        **{name: getattr(observation, name).detach().clone() for name in _OBSERVATION_FIELDS}
    )
    return PreparedControlInputs(
        observation=cloned_observation,
        near_points=prepared.near_points.detach().clone(),
        near_valid=prepared.near_valid.detach().clone(),
        far_points=prepared.far_points.detach().clone(),
        far_valid=prepared.far_valid.detach().clone(),
        near_filled=prepared.near_filled.detach().clone(),
        near_scan_ids=prepared.near_scan_ids.detach().clone(),
        near_times=prepared.near_times.detach().clone(),
        far_point_scan_ids=prepared.far_point_scan_ids.detach().clone(),
        far_point_times=prepared.far_point_times.detach().clone(),
        near_angle_keys=(
            None if prepared.near_angle_keys is None
            else prepared.near_angle_keys.detach().clone()
        ),
        far_angle_keys=(
            None if prepared.far_angle_keys is None
            else prepared.far_angle_keys.detach().clone()
        ),
        near_source_dirs_sensor=(
            None if prepared.near_source_dirs_sensor is None
            else prepared.near_source_dirs_sensor.detach().clone()
        ),
        far_source_dirs_sensor=(
            None if prepared.far_source_dirs_sensor is None
            else prepared.far_source_dirs_sensor.detach().clone()
        ),
        near_source_dirs_present=(
            None if prepared.near_source_dirs_present is None
            else prepared.near_source_dirs_present.detach().clone()
        ),
        far_source_dirs_present=(
            None if prepared.far_source_dirs_present is None
            else prepared.far_source_dirs_present.detach().clone()
        ),
        near_source_indices=(
            None if prepared.near_source_indices is None
            else prepared.near_source_indices.detach().clone()
        ),
        far_source_indices=(
            None if prepared.far_source_indices is None
            else prepared.far_source_indices.detach().clone()
        ),
    )


class ControlInputCache:
    """Identity-keyed single-slot cache of the current prepared inputs.

    Identity is (global control step counter, per-env episode vector): a
    selected reset at an unchanged counter changes the episode vector and
    therefore invalidates the cache explicitly. ``get`` returns the SAME
    object for repeated reads at one identity (no recompute, no rescan);
    ``store`` replaces the slot and returns what it stored.
    """

    def __init__(self) -> None:
        self._identity: int | None = None
        self._episodes: torch.Tensor | None = None
        self._prepared: PreparedControlInputs | None = None

    def matches(self, control_step: int, episode_ids: torch.Tensor) -> bool:
        if self._prepared is None or self._identity != int(control_step):
            return False
        return bool(torch.equal(self._episodes, episode_ids))

    def get(self) -> PreparedControlInputs | None:
        return self._prepared

    def store(
        self, control_step: int, episode_ids: torch.Tensor, prepared: PreparedControlInputs
    ) -> PreparedControlInputs:
        if not isinstance(prepared, PreparedControlInputs):
            raise TypeError("prepared must be a PreparedControlInputs")
        self._identity = int(control_step)
        self._episodes = episode_ids.detach().clone()
        self._prepared = prepared
        return prepared

    def invalidate(self) -> None:
        self._identity = None
        self._episodes = None
        self._prepared = None
