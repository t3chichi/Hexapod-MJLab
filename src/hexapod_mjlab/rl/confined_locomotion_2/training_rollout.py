"""Compact sequence rollout storage for confined_locomotion_2 (pure Torch)."""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import torch

from hexapod_mjlab.rl.confined_locomotion.contracts import OnlineObservation, EnvelopeLabel
from .contracts import HeightLabels

ROLLOUT_STEPS = 24
GAMMA = 0.99
GAE_LAMBDA = 0.95


class RolloutError(ValueError):
    pass


def _snapshot(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
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
    height_label: HeightLabels | None = None


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
        height_label: HeightLabels | None = None,
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
        if height_label is not None:
            if not isinstance(height_label, HeightLabels):
                raise RolloutError("height_label must be HeightLabels or None")
            height_label.validate(observation)
        if self.steps:
            first = self.steps[0].height_label
            if (first is None) != (height_label is None):
                raise RolloutError("height label presence must stay constant within a rollout")
            if first is not None and (first.grid_identity != height_label.grid_identity or
                                      first.terrain_identity != height_label.terrain_identity):
                raise RolloutError("terrain/grid identity changed within a rollout")
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
            "height_label": _snapshot(height_label),
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

_POINT_FIELDS = ("near_points", "near_valid", "far_points", "far_valid",
                 "near_angle_keys", "far_angle_keys")
DEFAULT_POINT_REPLAY_CHUNK = 1024


def validate_point_replay_chunk(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("point_replay_chunk must be a positive integer")
    return value


@dataclass(frozen=True)
class TemporalReplayLayout:
    """Episode segments and time/environment mappings; no learned features."""

    gather: torch.Tensor
    restore: torch.Tensor
    lengths: torch.Tensor
    rows: torch.Tensor
    starts_at_zero: torch.Tensor
    shape: tuple[int, int, int]


def _temporal_layout(steps):
    time, batch = len(steps), steps[0].hidden_before.shape[0]
    done = torch.stack([s.terminated | s.truncated for s in steps]).cpu().tolist()
    segments = []
    for row in range(batch):
        start = 0
        for t in range(1, time):
            if done[t - 1][row]:
                segments.append((row, start, t))
                start = t
        segments.append((row, start, time))
    length = max(end - start for row, start, end in segments)
    gather, restore = [], [0] * (time * batch)
    for index, (row, start, end) in enumerate(segments):
        gather.append([t * batch + row for t in range(start, end)] + [0] * (length - end + start))
        for t in range(start, end):
            restore[t * batch + row] = index * length + t - start
    device = steps[0].hidden_before.device
    return TemporalReplayLayout(
        torch.tensor(gather, dtype=torch.long, device=device),
        torch.tensor(restore, dtype=torch.long, device=device),
        torch.tensor([end - start for row, start, end in segments], dtype=torch.long),
        torch.tensor([row for row, start, end in segments], dtype=torch.long, device=device),
        torch.tensor([start == 0 for row, start, end in segments], dtype=torch.bool, device=device),
        (time, batch, length),
    )


@dataclass(frozen=True)
class PreparedSequenceReplay:
    """Parameter-independent inputs, shared by this rollout's optimizer steps."""

    steps: list[SequenceStep]
    inputs: list[Any]
    point_inputs: tuple[torch.Tensor | None, ...]
    row_indices: list[torch.Tensor]
    mirrored: bool
    temporal_layout: TemporalReplayLayout
    point_sequences: tuple[Any, ...] | None = None
    scaled_observations: tuple[torch.Tensor, torch.Tensor] | None = None
    point_replay_chunk: int = DEFAULT_POINT_REPLAY_CHUNK


def prepare_sequence_replay(steps: list[SequenceStep], *, network=None, mirrored=False,
                            joint_permutation=None, mount_quat_xyzw=None,
                            point_replay_chunk=DEFAULT_POINT_REPLAY_CHUNK) -> PreparedSequenceReplay:
    """Validate snapshots and prepare unique scans once, without network features."""
    from .symmetry import mirror_prepared

    validate_point_replay_chunk(point_replay_chunk)
    if not steps:
        raise RolloutError("sequence replay requires at least one step")
    inputs = []
    for step in steps:
        if step.height_label is not None:
            step.height_label.validate(step.observation)
        if mirrored:
            if step.prepared is None:
                raise ValueError("symmetry sequence replay requires complete prepared metadata")
            inputs.append(mirror_prepared(
                step.prepared, joint_permutation, mount_quat_xyzw=mount_quat_xyzw))
        else:
            inputs.append(step.prepared if step.prepared is not None else step)

    batch = steps[0].hidden_before.shape[0]
    device = steps[0].hidden_before.device
    if all(getattr(p, "near_scan_ids", None) is not None and
           getattr(p, "far_point_scan_ids", None) is not None for p in inputs):
        # One host transfer instead of per-environment CUDA scalar/list reads.
        metadata = torch.stack([torch.cat((
            p.observation.env_ids[:, None], p.observation.episode_ids[:, None],
            p.near_scan_ids[:, None], p.far_point_scan_ids,
        ), dim=1) for p in inputs]).cpu().tolist()
    else:
        # Inputs without scan provenance cannot safely reuse frame features.
        metadata = [[(t, row) for row in range(batch)] for t in range(len(steps))]

    unique = {}
    representatives = []
    row_indices = []
    for t, rows in enumerate(metadata):
        indices = []
        for row, identity in enumerate(rows):
            key = tuple(identity)
            if key not in unique:
                unique[key] = len(representatives)
                representatives.append((t, row))
            indices.append(unique[key])
        row_indices.append(torch.tensor(indices, dtype=torch.long, device=device))
    point_inputs = []
    for name in _POINT_FIELDS:
        values = [getattr(p, name, None) for p in inputs]
        if all(value is None for value in values):
            point_inputs.append(None)
            continue
        if any(value is None for value in values):
            raise ValueError(f"{name} presence must stay constant within sequence replay")
        points = torch.stack([values[t][row] for t, row in representatives])
        # Keep the original scan identity/input check, once per immutable rollout.
        if any(not torch.equal(points.index_select(0, indices), value)
               for indices, value in zip(row_indices, values)):
            raise ValueError("same scan identity produced different frame input during PPO replay")
        point_inputs.append(points)
    sequences = None
    if network is not None and hasattr(network, "prepare_point_sequence"):
        sequences = []
        for start in range(0, len(point_inputs[0]), point_replay_chunk):
            near, nv, far, fv, nk, fk = (
                None if x is None else x[start:start + point_replay_chunk]
                for x in point_inputs)
            sequences.append(network.prepare_point_sequence(
                near, nv, far, fv, near_angle_keys=nk, far_angle_keys=fk))
        sequences = tuple(sequences)
    scaled = (network.prepare_sequence_observations([p.observation for p in inputs])
              if network is not None and hasattr(network, "prepare_sequence_observations") else None)
    return PreparedSequenceReplay(steps, inputs, tuple(point_inputs), row_indices, mirrored,
                                  _temporal_layout(steps), sequences, scaled, point_replay_chunk)


def replay_sequence(network, steps: list[SequenceStep] | PreparedSequenceReplay, *, mirrored=False,
                    joint_permutation=None, mount_quat_xyzw=None,
                    checkpoint_points=False, point_replay_chunk=DEFAULT_POINT_REPLAY_CHUNK) -> list[Any]:
    """Recompute differentiable features; retain the complete temporal gradient.

    A prepared plan owns its packing/chunk boundaries. The chunk argument is
    used when constructing a plan from raw steps.
    """
    plan = steps if isinstance(steps, PreparedSequenceReplay) else prepare_sequence_replay(
        steps, network=network, mirrored=mirrored, joint_permutation=joint_permutation,
        mount_quat_xyzw=mount_quat_xyzw, point_replay_chunk=point_replay_chunk)
    steps = plan.steps
    h = steps[0].mirror_hidden_before if plan.mirrored else steps[0].hidden_before
    if h is None:
        raise ValueError("mirror replay requires mirrored hidden initial state")
    h = h.detach().clone()
    encoded = hasattr(network, "forward_from_features") and hasattr(network, "encode_points")
    if encoded:
        from torch.utils.checkpoint import checkpoint

        features = []
        chunk = plan.point_replay_chunk
        for index, start in enumerate(range(0, len(plan.point_inputs[0]), chunk)):
            if plan.point_sequences is not None:
                encode = network.encode_point_sequence
                args, kwargs = (plan.point_sequences[index],), {}
            else:
                near, nv, far, fv, nk, fk = (
                    None if x is None else x[start:start + chunk]
                    for x in plan.point_inputs)
                encode = network.encode_points
                args = near, nv, far, fv
                kwargs = {"near_angle_keys": nk, "far_angle_keys": fk}
            if checkpoint_points and torch.is_grad_enabled():
                # Only the deterministic, weight-dependent GRU needs recomputation.
                feature = checkpoint(encode, *args, use_reentrant=False,
                                     preserve_rng_state=False, **kwargs)
            else:
                feature = encode(*args, **kwargs)
            features.append(feature)
        frame_features = torch.cat(features, dim=0)
        if hasattr(network, "forward_sequence_from_features"):
            frames = torch.stack([frame_features.index_select(0, indices)
                                  for indices in plan.row_indices])
            return network.forward_sequence_from_features(
                [p.observation for p in plan.inputs], frames, h, plan.temporal_layout,
                **({"scaled_observations": plan.scaled_observations}
                   if plan.scaled_observations is not None else {}))

    outputs = []
    for t, (step, p, indices) in enumerate(zip(steps, plan.inputs, plan.row_indices)):
        if t:
            previous = steps[t - 1]
            done = previous.terminated | previous.truncated
            h = torch.where(done[:, None], torch.zeros_like(h), h)
        if encoded:
            out = network.forward_from_features(
                p.observation, frame_features.index_select(0, indices), h)
        else:
            out = network(p.observation, p.near_points, p.near_valid, p.far_points,
                          p.far_valid, h, near_angle_keys=getattr(p, "near_angle_keys", None),
                          far_angle_keys=getattr(p, "far_angle_keys", None))
        outputs.append(out)
        h = out.hidden
    return outputs
