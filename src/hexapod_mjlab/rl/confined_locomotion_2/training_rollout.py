"""Compact sequence rollout storage for confined_locomotion_2 (pure Torch)."""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import torch

from hexapod_mjlab.rl.confined_locomotion.contracts import OnlineObservation, EnvelopeLabel

ROLLOUT_STEPS = 24
GAMMA = 0.99
GAE_LAMBDA = 0.95


class RolloutError(ValueError):
    pass


def _snapshot(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, OnlineObservation):
        return OnlineObservation(**{f.name: _snapshot(getattr(value, f.name)) for f in fields(value)})
    if isinstance(value, EnvelopeLabel):
        return EnvelopeLabel(**{f.name: _snapshot(getattr(value, f.name)) for f in fields(value)})
    if hasattr(value, "__dataclass_fields__"):
        return type(value)(**{f.name: _snapshot(getattr(value, f.name)) for f in fields(value)})
    return value


@dataclass(frozen=True)
class SequenceStep:
    observation: OnlineObservation
    near_points: torch.Tensor
    near_valid: torch.Tensor
    far_points: torch.Tensor
    far_valid: torch.Tensor
    action: torch.Tensor
    old_log_prob: torch.Tensor
    old_mean: torch.Tensor
    old_std: torch.Tensor
    old_value: torch.Tensor
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    next_value: torch.Tensor
    hidden_before: torch.Tensor
    envelope_label: EnvelopeLabel
    prepared: Any | None = None
    mirror_hidden_before: torch.Tensor | None = None
    near_angle_keys: torch.Tensor | None = None
    far_angle_keys: torch.Tensor | None = None
    near_source_dirs_sensor: torch.Tensor | None = None
    far_source_dirs_sensor: torch.Tensor | None = None
    near_source_dirs_present: torch.Tensor | None = None
    far_source_dirs_present: torch.Tensor | None = None


class SequenceRolloutBuffer:
    """Exactly 24 consecutive control steps, each containing all environment rows.

    The stored hidden states are pre-action states. The trainer replays each complete
    sequence from the detached initial state and clears rows after termination/timeout.
    """

    def __init__(self, num_envs: int, device: torch.device | str, *, rollout_steps: int = ROLLOUT_STEPS):
        if isinstance(num_envs, bool) or int(num_envs) < 1:
            raise RolloutError("num_envs must be >= 1")
        if int(rollout_steps) != ROLLOUT_STEPS:
            raise RolloutError(f"rollout_steps is fixed at {ROLLOUT_STEPS}")
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.rollout_steps = int(rollout_steps)
        self.steps: list[SequenceStep] = []
        self.consumed = False

    def append_step(
        self, *, observation: OnlineObservation, near_points: torch.Tensor,
        near_valid: torch.Tensor, far_points: torch.Tensor, far_valid: torch.Tensor,
        action: torch.Tensor, old_log_prob: torch.Tensor, old_mean: torch.Tensor,
        old_std: torch.Tensor, value: torch.Tensor, reward: torch.Tensor,
        terminated: torch.Tensor, truncated: torch.Tensor, next_value: torch.Tensor,
        hidden_before: torch.Tensor, envelope_label: EnvelopeLabel,
        prepared: Any | None = None,
        mirror_hidden_before: torch.Tensor | None = None,
        near_angle_keys: torch.Tensor | None = None, far_angle_keys: torch.Tensor | None = None,
        near_source_dirs_sensor: torch.Tensor | None = None,
        far_source_dirs_sensor: torch.Tensor | None = None,
        near_source_dirs_present: torch.Tensor | None = None,
        far_source_dirs_present: torch.Tensor | None = None,
    ) -> None:
        if self.consumed or len(self.steps) >= self.rollout_steps:
            raise RolloutError("buffer is consumed or already full")
        if not isinstance(observation, OnlineObservation) or not isinstance(envelope_label, EnvelopeLabel):
            raise RolloutError("observation and envelope must use their typed contracts")
        b = self.num_envs
        expected = {
            "near_points": ((b, 256, 3), torch.float32), "near_valid": ((b, 256), torch.bool),
            "far_points": ((b, 128, 3), torch.float32), "far_valid": ((b, 128), torch.bool),
            "action": ((b, 18), torch.float32), "old_log_prob": ((b,), torch.float32),
            "old_mean": ((b, 18), torch.float32), "old_std": ((b, 18), torch.float32),
            "value": ((b,), torch.float32), "reward": ((b,), torch.float32),
            "terminated": ((b,), torch.bool), "truncated": ((b,), torch.bool),
            "next_value": ((b,), torch.float32), "hidden_before": ((b, 187), torch.float32),
        }
        tensors = locals()
        for name, (shape, dtype) in expected.items():
            x = tensors[name]
            if not isinstance(x, torch.Tensor) or tuple(x.shape) != shape or x.dtype != dtype or x.device != self.device:
                raise RolloutError(f"{name} must be {shape} {dtype} on {self.device}")
            if dtype.is_floating_point and not bool(torch.isfinite(x).all()):
                raise RolloutError(f"{name} contains NaN/Inf")
        if observation.env_ids.numel() != b or envelope_label.env_ids.numel() != b:
            raise RolloutError("observation/envelope batch differs from num_envs")
        for name, x in (("mirror_hidden_before", mirror_hidden_before),):
            if x is not None and (tuple(x.shape) != (b, 187) or x.dtype != torch.float32 or x.device != self.device):
                raise RolloutError(f"{name} must be ({b},187) float32 on {self.device}")
        for name, x, shape in (
            ("near_angle_keys", near_angle_keys, (b, 256)), ("far_angle_keys", far_angle_keys, (b, 128)),
            ("near_source_dirs_sensor", near_source_dirs_sensor, (b, 256, 3)),
            ("far_source_dirs_sensor", far_source_dirs_sensor, (b, 128, 3)),
            ("near_source_dirs_present", near_source_dirs_present, (b, 256)),
            ("far_source_dirs_present", far_source_dirs_present, (b, 128)),
        ):
            if x is not None and (tuple(x.shape) != shape or x.device != self.device):
                raise RolloutError(f"{name} shape/device mismatch")
        observation.validate()
        envelope_label.validate()
        if not (torch.equal(observation.env_ids, envelope_label.env_ids)
                and torch.equal(observation.episode_ids, envelope_label.episode_ids)
                and torch.equal(observation.step_ids, envelope_label.step_ids)):
            raise RolloutError("envelope label identity must match its pre-action observation")
        if prepared is not None:
            for name, expected_value in (("near_points", near_points), ("near_valid", near_valid),
                                         ("far_points", far_points), ("far_valid", far_valid)):
                if not torch.equal(getattr(prepared, name), expected_value):
                    raise RolloutError(f"prepared.{name} must match append_step {name}")
            if not (torch.equal(prepared.observation.env_ids, observation.env_ids)
                    and torch.equal(prepared.observation.episode_ids, observation.episode_ids)
                    and torch.equal(prepared.observation.step_ids, observation.step_ids)
                    and torch.equal(prepared.observation.time_s, observation.time_s)):
                raise RolloutError("prepared observation identity mismatch")
        if self.steps:
            prev = self.steps[-1]
            prev_obs = prev.observation
            if not torch.equal(prev_obs.env_ids, observation.env_ids):
                raise RolloutError("environment row identities/order changed within rollout")
            same_episode = prev_obs.episode_ids == observation.episode_ids
            prev_done = prev.terminated | prev.truncated
            if bool((same_episode & (observation.step_ids != prev_obs.step_ids + 1)).any()):
                raise RolloutError("same-episode step identity did not advance by one")
            if bool((same_episode & prev_done).any()):
                raise RolloutError("episode continued after a terminal/truncated transition")
            changed_episode = ~same_episode
            if bool((changed_episode & ~prev_done).any()) or bool((changed_episode & (observation.step_ids != 0)).any()):
                raise RolloutError("episode identity may change only after done and must reset step_id to zero")
            dt = observation.time_s - prev_obs.time_s
            if bool((dt <= 0).any()):
                raise RolloutError("observation time must advance within the rollout")
        if bool((next_value[terminated] != 0).any()):
            raise RolloutError("true terminals must have zero bootstrap value")
        self.steps.append(SequenceStep(**{
            "observation": _snapshot(observation), "near_points": _snapshot(near_points),
            "near_valid": _snapshot(near_valid), "far_points": _snapshot(far_points),
            "far_valid": _snapshot(far_valid), "action": _snapshot(action),
            "old_log_prob": _snapshot(old_log_prob), "old_mean": _snapshot(old_mean),
            "old_std": _snapshot(old_std), "old_value": _snapshot(value),
            "reward": _snapshot(reward), "terminated": _snapshot(terminated),
            "truncated": _snapshot(truncated), "next_value": _snapshot(next_value),
            "hidden_before": _snapshot(hidden_before), "envelope_label": _snapshot(envelope_label),
            "prepared": _snapshot(prepared),
            "mirror_hidden_before": _snapshot(mirror_hidden_before),
            "near_angle_keys": _snapshot(near_angle_keys), "far_angle_keys": _snapshot(far_angle_keys),
            "near_source_dirs_sensor": _snapshot(near_source_dirs_sensor),
            "far_source_dirs_sensor": _snapshot(far_source_dirs_sensor),
            "near_source_dirs_present": _snapshot(near_source_dirs_present),
            "far_source_dirs_present": _snapshot(far_source_dirs_present),
        }))

    def _require_full(self) -> None:
        if len(self.steps) != self.rollout_steps:
            raise RolloutError(f"expected {self.rollout_steps} steps, got {len(self.steps)}")
        if self.consumed:
            raise RolloutError("buffer already consumed")

    def compute_gae(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._require_full()
        rewards = torch.stack([s.reward for s in self.steps])
        values = torch.stack([s.old_value for s in self.steps])
        next_values = torch.stack([s.next_value for s in self.steps])
        term = torch.stack([s.terminated for s in self.steps])
        done = term | torch.stack([s.truncated for s in self.steps])
        advantages = torch.zeros_like(values)
        carry = torch.zeros_like(values[0])
        for t in range(self.rollout_steps - 1, -1, -1):
            delta = rewards[t] + GAMMA * (~term[t]).float() * next_values[t] - values[t]
            carry = delta + GAMMA * GAE_LAMBDA * (~done[t]).float() * carry
            advantages[t] = carry
        return advantages, advantages + values

    def consume(self) -> None:
        self._require_full()
        self.consumed = True

    def clear(self) -> None:
        self.steps.clear()
        self.consumed = False

