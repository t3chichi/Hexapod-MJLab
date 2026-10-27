"""Engine-independent execution semantics of ``el4090_tripod2_low``.

Sources: LeggedRobotRewMixin, EL_4090 and ElSpider in the reference checkout.
Foot arrays passed here MUST use alphabetical LB, LF, LM, RB, RF, RM order:
this is the order documented in the old gait implementation (the Isaac Gym
asset ordering could not be measured on this host). Joint arrays use the
current MJCF order; HFE/KFE indices can be supplied explicitly in the record.
All returned rewards are raw; weights and the 0.02 s multiplier are separate.
"""

from __future__ import annotations

from collections import deque
from itertools import combinations, product

import numpy as np
import torch

DT = 0.02
MAX_EPISODE_LENGTH = 1000
SPEED_MIN = 0.2
FOOT_ORDER = ("LB_FOOT", "LF_FOOT", "LM_FOOT", "RB_FOOT", "RF_FOOT", "RM_FOOT")
REWARD_WEIGHTS = {
    "tracking_lin_vel": (1., 1.), "tracking_ang_vel": (.5, .5),
    "lin_vel_z": (-8., -8.), "ang_vel_xy": (-.5, -.5),
    "orientation": (-8., -8.), "torques": (-.0001, -.0001),
    "dof_vel": (-.0001, -.0001), "dof_acc": (-5e-8, -5e-8),
    "base_height": (-2., -5.), "feet_slip": (0., -.2),
    "feet_air_time": (.5, .1), "collision": (-1., -1.),
    "feet_stumble": (0., -.2), "action_rate": (-.005, -.005),
    "stand_still2": (-.6, -.6), "dof_pos_limits": (-1., -1.),
    "feet_contact_forces": (-.05, -.25), "shank_perp2ground": (-.05, -.05),
    "gait_2_step": (-1., -.2),
}


def reward_scales(stage: int, *, dt: float = DT) -> dict[str, float]:
    """Effective scales; the reference multiplies each nonzero weight by dt."""
    if stage not in (0, 1):
        raise ValueError("reward stage must be 0 or 1")
    return {name: values[stage] * dt for name, values in REWARD_WEIGHTS.items()
            if values[stage] != 0.}


def weight_rewards(raw: dict[str, torch.Tensor], stage: int) -> tuple[torch.Tensor, dict]:
    weighted = {name: raw[name] * scale for name, scale in reward_scales(stage).items()}
    return torch.stack(list(weighted.values())).sum(0), weighted


class LegacyRewardState:
    """Control-boundary contact timers, including the old reset behavior."""

    def __init__(self, num_envs: int, device="cpu"):
        self.feet_air_time = torch.zeros(num_envs, 6, device=device)
        self.feet_contact_time = torch.zeros_like(self.feet_air_time)
        self.last_contacts = torch.zeros_like(self.feet_air_time, dtype=torch.bool)
        self.feet_first_contact = torch.zeros_like(self.last_contacts)
        self.feet_air_time_on_contact = torch.zeros_like(self.feet_air_time)

    def update(self, foot_force_w: torch.Tensor) -> None:
        contact = foot_force_w[:, :, 2] > 1.
        contact_filt = contact | self.last_contacts
        air_next = self.feet_air_time + DT
        contact_next = self.feet_contact_time + DT
        self.feet_first_contact.copy_((self.feet_air_time > 0.) & contact_filt)
        self.feet_air_time_on_contact.copy_(air_next)
        self.feet_air_time.copy_(air_next * ~contact_filt)
        self.feet_contact_time.copy_(contact_next * contact_filt)
        self.last_contacts.copy_(contact)

    def reset(self, env_ids) -> None:
        # The reference reset clears only these two timers. last_contacts is
        # deliberately retained, including the one-frame episode-boundary tail.
        self.feet_air_time[env_ids] = 0.
        self.feet_contact_time[env_ids] = 0.

    def state_dict(self) -> dict:
        return {name: value.clone() for name, value in vars(self).items()}

    def load_state_dict(self, state: dict) -> None:
        if set(state) != set(vars(self)):
            raise ValueError("contact timer state keys differ")
        for name, value in state.items():
            target = getattr(self, name)
            if target.shape != value.shape or target.dtype != value.dtype:
                raise ValueError(f"contact timer state mismatch: {name}")
            target.copy_(value.to(target.device))


def _gait_penalty(state: LegacyRewardState, commands: torch.Tensor) -> torch.Tensor:
    air, contact = state.feet_air_time, state.feet_contact_time

    def sync(i, j):
        return (air[:, i] - air[:, j]).square().clamp(max=4.) + \
            (contact[:, i] - contact[:, j]).square().clamp(max=4.)

    def anti(i, j):
        return (air[:, i] - contact[:, j]).square().clamp(max=4.) + \
            (contact[:, i] - air[:, j]).square().clamp(max=4.)

    g1, g2 = (0, 1, 5), (2, 3, 4)
    sync_g1 = sum(sync(i, j) for i, j in combinations(g1, 2)) / 3
    sync_g2 = sum(sync(i, j) for i, j in combinations(g2, 2)) / 3
    across = sum(anti(i, j) for i, j in product(g1, g2)) / 9
    # Preserve the source's grouping and summation order.
    sync_all = (sum(sync(i, j) for i, j in combinations(g1, 2)) +
                sum(sync(i, j) for i, j in combinations(g2, 2)) +
                sum(sync(i, j) for i, j in product(g1, g2))) / 15
    moving = (commands[:, :2].norm(dim=1) > SPEED_MIN) | (commands[:, 2].abs() > SPEED_MIN)
    return torch.where(moving, (sync_g1 + sync_g2) / 2 + across, sync_all)


def compute_legacy_rewards(record: dict, state: LegacyRewardState,
                           commands: torch.Tensor, stage: int = 0) -> dict[str, torch.Tensor]:
    """Evaluate one boundary exactly once; updates contact timers first.

    ``dof_pos_limits`` must already be the reference soft limits (0.9 of
    the URDF range). ``torques`` is the final substep effort, with ``pd``
    record fallback; never average the four substep squares. Plane height
    is zero unless ``measured_heights`` is supplied. The reward manager
    must cache this result so repeated term reads do not advance timers.
    """
    if stage not in (0, 1):
        raise ValueError("reward stage must be 0 or 1")
    q, qdot = record["joint_pos"], record["joint_vel"]
    force, vel = record["foot_force_w"], record["foot_vel_w"]
    state.update(force)
    limits = record["dof_pos_limits"]
    hfe = record.get("hfe_indices", (1, 4, 7, 10, 13, 16))
    kfe = record.get("kfe_indices", (2, 5, 8, 11, 14, 17))
    tau = record["torques"] if "torques" in record else record["pd"][-1]["effort"]
    base_height = record["root_pos_w"][:, 2]
    if "measured_heights" in record:
        base_height = (base_height[:, None] - record["measured_heights"]).mean(dim=1)
    small = (commands[:, :2].norm(dim=1) < SPEED_MIN) & (commands[:, 2].abs() < SPEED_MIN)
    # _update_feet_contact_timers already wrote last_contacts=current. Thus
    # the source's slip OR is the current contact mask, not contact_filt.
    slip_contact = (force[:, :, 2] > 1.) | state.last_contacts
    raw = {
        "tracking_lin_vel": torch.exp(-(commands[:, :2] - record["lin_vel_b"][:, :2]).square().sum(1) / .25),
        "tracking_ang_vel": torch.exp(-(commands[:, 2] - record["ang_vel_b"][:, 2]).square() / .25),
        "lin_vel_z": record["lin_vel_b"][:, 2].square(),
        "ang_vel_xy": record["ang_vel_b"][:, :2].square().sum(1),
        "orientation": record["gravity_b"][:, :2].square().sum(1),
        "torques": tau.square().sum(1),
        "dof_vel": qdot.square().sum(1),
        "dof_acc": ((record["previous_qdot"] - qdot) / DT).square().sum(1),
        "base_height": (base_height - .25).square(),
        "feet_slip": (slip_contact * vel[:, :, :2].norm(dim=2).square()).sum(1),
        "feet_air_time": ((state.feet_air_time_on_contact - .25) * state.feet_first_contact).sum(1) * (commands[:, :2].norm(dim=1) > .1),
        "collision": (record["penalized_contact_force_w"].norm(dim=-1) > .1).float().sum(1),
        "feet_stumble": (force[:, :, :2].norm(dim=2) > 5 * force[:, :, 2].abs()).any(1),
        "action_rate": (record["previous_action"] - record["clipped_action"]).square().sum(1),
        "stand_still2": 2 * q.abs().sum(1) * small,
        "dof_pos_limits": ((limits[..., 0] - q).clamp(min=0.) + (q - limits[..., 1]).clamp(min=0.)).sum(1),
        "feet_contact_forces": (force.norm(dim=-1) - 500.).clamp(min=0.).sum(1),
        "shank_perp2ground": (q[:, hfe] + q[:, kfe]).square().sum(1),
        "gait_2_step": _gait_penalty(state, commands),
    }
    return raw


class LegacyStageController:
    """Recent completed returns, advanced only AFTER a PPO update."""

    def __init__(self):
        self.stage = 0
        self.completed_returns = deque(maxlen=100)

    def add_completed(self, returns) -> None:
        if isinstance(returns, torch.Tensor):
            returns = returns.detach().cpu().flatten().tolist()
        self.completed_returns.extend(float(x) for x in returns)

    def after_update(self) -> bool:
        if self.stage == 0 and self.completed_returns and sum(self.completed_returns) / len(self.completed_returns) > 2.:
            self.stage = 1
            return True
        return False

    def state_dict(self) -> dict:
        return {"stage": self.stage, "completed_returns": list(self.completed_returns)}

    def load_state_dict(self, state: dict) -> None:
        if state["stage"] not in (0, 1) or len(state["completed_returns"]) > 100:
            raise ValueError("invalid reward stage state")
        self.stage = int(state["stage"])
        self.completed_returns = deque(state["completed_returns"], maxlen=100)


class LegacyCommands:
    """Independent uniform commands, episode-phased resampling, global pushes.

    Call ``tick`` after integration, before reward evaluation. Then process
    terminations and call ``reset``. The caller owns RNG checkpointing.
    """

    def __init__(self, num_envs: int, device="cpu"):
        self.commands = torch.zeros(num_envs, 3, device=device)
        self.episode_length = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.common_step_counter = 0
        self.ranges = {"lin_vel_x": [-1., 1.], "lin_vel_y": [-1., 1.], "ang_vel_yaw": [-1.2, 1.2]}

    def resample(self, env_ids: torch.Tensor) -> None:
        for j, name in enumerate(self.ranges):
            lo, hi = self.ranges[name]
            self.commands[env_ids, j] = (hi - lo) * torch.rand(len(env_ids), device=self.commands.device) + lo

    def reset(self, env_ids: torch.Tensor) -> None:
        self.resample(env_ids)
        self.episode_length[env_ids] = 0

    def randomize_initial_episode_length(self) -> None:
        self.episode_length.copy_(torch.randint_like(self.episode_length, high=MAX_EPISODE_LENGTH))

    def tick(self) -> bool:
        self.episode_length += 1
        self.common_step_counter += 1
        due = (self.episode_length % int(4. / DT) == 0).nonzero().flatten()
        self.resample(due)
        return self.common_step_counter % int(3. / DT) == 0

    def curriculum(self, env_ids: torch.Tensor, tracking_episode_sums: torch.Tensor,
                   tracking_scale: float = DT) -> bool:
        if len(env_ids) == 0 or self.common_step_counter % MAX_EPISODE_LENGTH:
            return False
        if tracking_episode_sums[env_ids].mean() / MAX_EPISODE_LENGTH > .8 * tracking_scale:
            lo, hi = self.ranges["lin_vel_x"]
            self.ranges["lin_vel_x"] = [max(-2.5, min(0., lo - .5)), min(2.5, max(0., hi + .5))]
            return True
        return False

    @property
    def timeout(self) -> torch.Tensor:
        return self.episode_length > MAX_EPISODE_LENGTH

    def state_dict(self) -> dict:
        return {"commands": self.commands.clone(), "episode_length": self.episode_length.clone(),
                "common_step_counter": self.common_step_counter,
                "ranges": {name: values.copy() for name, values in self.ranges.items()}}

    def load_state_dict(self, state: dict) -> None:
        for name in ("commands", "episode_length"):
            target, value = getattr(self, name), state[name]
            if target.shape != value.shape or target.dtype != value.dtype:
                raise ValueError(f"command state mismatch: {name}")
            target.copy_(value.to(target.device))
        self.common_step_counter = int(state["common_step_counter"])
        self.ranges = {name: list(values) for name, values in state["ranges"].items()}


class LegacyResetDRState:
    """Creation-time parameters; reset never redraws mass or friction.

    The old creation path uses CPU torch RNG for friction and numpy RNG for
    base mass. Store these realized parameters in the full checkpoint.
    Applying PhysX's average friction mixture and inertia recomputation in
    MuJoCo remains an explicit engine adapter responsibility.
    """

    def __init__(self, num_envs: int, device="cpu"):
        self.bucket_ids = torch.randint(0, 64, (num_envs, 1))
        self.friction_buckets = .3 + .95 * torch.rand(64, 1)
        self.friction = self.friction_buckets[self.bucket_ids].reshape(num_envs).to(device)
        self.added_mass = torch.as_tensor(np.random.uniform(-10., 10., num_envs), dtype=torch.float32, device=device)

    def reset_velocities(self, count: int) -> torch.Tensor:
        return torch.rand(count, 6, device=self.friction.device) - .5

    def push_velocities(self) -> torch.Tensor:
        return 2 * torch.rand(len(self.friction), 2, device=self.friction.device) - 1

    def state_dict(self) -> dict:
        return {name: value.clone() for name, value in vars(self).items()}

    def load_state_dict(self, state: dict) -> None:
        if set(state) != set(vars(self)):
            raise ValueError("DR state keys differ")
        for name, value in state.items():
            target = getattr(self, name)
            if target.shape != value.shape or target.dtype != value.dtype:
                raise ValueError(f"DR state mismatch: {name}")
            target.copy_(value.to(target.device))
