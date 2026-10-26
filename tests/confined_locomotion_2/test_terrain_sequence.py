"""CPU checks for Terrain Memory labels in the sequence collection path."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import torch

from hexapod_mjlab.rl.confined_locomotion.contracts import EnvelopeLabel, OnlineObservation
from hexapod_mjlab.rl.confined_locomotion.distribution import IndependentDiagonalNormalProvider
from hexapod_mjlab.rl.confined_locomotion_2.symmetry import mirror_height_targets
from hexapod_mjlab.rl.confined_locomotion_2.training_rollout import (
    RolloutError,
    SequenceRolloutBuffer,
    replay_sequence,
)
from hexapod_mjlab.tasks.confined_locomotion_2.collector import TrainingCollector
from hexapod_mjlab.tasks.confined_locomotion_2.terrain_geometry import HeightLabels


@contextmanager
def _raises(*exception_types):
    try:
        yield
    except exception_types:
        return
    raise AssertionError(f"expected one of {[item.__name__ for item in exception_types]}")


def _observation(*, step: int = 0, episode: int = 0) -> OnlineObservation:
    z3 = torch.zeros((1, 3), dtype=torch.float32)
    z18 = torch.zeros((1, 18), dtype=torch.float32)
    return OnlineObservation(
        env_ids=torch.tensor([0], dtype=torch.int64),
        episode_ids=torch.tensor([episode], dtype=torch.int64),
        step_ids=torch.tensor([step], dtype=torch.int64),
        time_s=torch.tensor([step * 0.02], dtype=torch.float32),
        lin_vel_b=z3.clone(), ang_vel_b=z3.clone(), gravity_b=z3.clone(),
        joint_pos_rel_ref=z18.clone(), joint_vel=z18.clone(), prev_action=z18.clone(), command=z3.clone(),
    )


def _height_label(*, step: int = 0, episode: int = 0) -> HeightLabels:
    heights = torch.arange(187, dtype=torch.float32).reshape(1, 11, 17) / 100.0 - 0.8
    valid = torch.ones((1, 11, 17), dtype=torch.bool)
    valid[:, 0, 0] = False
    heights[:, 0, 0] = 0
    xy = torch.zeros((1, 11, 17, 2), dtype=torch.float32)
    xy[..., 0] = torch.arange(17, dtype=torch.float32).reshape(1, 1, 17)
    xy[..., 1] = torch.arange(11, dtype=torch.float32).reshape(1, 11, 1)
    return HeightLabels(
        env_ids=torch.tensor([0], dtype=torch.int64),
        episode_ids=torch.tensor([episode], dtype=torch.int64),
        step_ids=torch.tensor([step], dtype=torch.int64),
        heights=heights,
        valid=valid,
        query_xy_w=xy,
        root_pos_w=torch.tensor([[1.0, 2.0, 0.5]], dtype=torch.float32),
        grid_identity={"nx": 17, "ny": 11, "order": "iy*17+ix"},
        terrain_identity="cpu-test-terrain",
    )


def _append(buffer: SequenceRolloutBuffer, observation: OnlineObservation, *, height_label=None) -> None:
    device = buffer.device
    b = buffer.num_envs
    zeros = lambda *shape: torch.zeros(shape, dtype=torch.float32, device=device)
    bools = lambda *shape: torch.zeros(shape, dtype=torch.bool, device=device)
    envelope = EnvelopeLabel(
        env_ids=observation.env_ids.clone(), episode_ids=observation.episode_ids.clone(),
        step_ids=observation.step_ids.clone(), values=zeros(b, 6), valid=torch.ones(b, dtype=torch.bool),
    )
    buffer.append_step(
        observation=observation,
        near_points=zeros(b, 256, 3), near_valid=bools(b, 256),
        far_points=zeros(b, 128, 3), far_valid=bools(b, 128),
        action=zeros(b, 18), old_log_prob=zeros(b), old_mean=zeros(b, 18), old_std=torch.ones(b, 18),
        value=zeros(b), reward=zeros(b), terminated=bools(b), truncated=bools(b),
        next_value=zeros(b), hidden_before=zeros(b, 187), envelope_label=envelope,
        height_label=height_label,
    )


def test_height_label_is_stored_with_identity_and_independent_snapshot():
    buffer = SequenceRolloutBuffer(1, "cpu")
    observation = _observation(step=3, episode=4)
    source = _height_label(step=3, episode=4)

    _append(buffer, observation, height_label=source)

    stored = buffer.steps[0].height_label
    assert torch.equal(stored.env_ids, observation.env_ids)
    assert torch.equal(stored.episode_ids, observation.episode_ids)
    assert torch.equal(stored.step_ids, observation.step_ids)
    assert torch.equal(stored.heights, source.heights)
    assert torch.equal(stored.valid, source.valid)
    assert torch.equal(stored.query_xy_w, source.query_xy_w)
    original_valid = source.valid.clone()
    source.heights.fill_(99)
    source.valid.zero_()
    source.query_xy_w.fill_(99)
    source.root_pos_w.fill_(99)
    assert not torch.equal(stored.heights, source.heights)
    assert torch.equal(stored.valid, original_valid)
    assert torch.all(stored.query_xy_w[..., 0] != 99)
    assert torch.equal(stored.root_pos_w, torch.tensor([[1.0, 2.0, 0.5]]))


def test_height_label_rejects_identity_mismatch():
    for identity_field in ("step_ids", "episode_ids"):
        buffer = SequenceRolloutBuffer(1, "cpu")
        observation = _observation(step=3, episode=4)
        label = _height_label(step=3, episode=4)
        label = replace(label, **{
            identity_field: torch.tensor([int(getattr(label, identity_field)[0]) + 1])
        })
        with _raises(RolloutError, ValueError):
            _append(buffer, observation, height_label=label)


def test_height_label_rejects_bad_shape_dtype_and_nonzero_invalid_cells():
    observation = _observation()
    for label in (
        HeightLabels(**{**vars(_height_label()), "heights": torch.zeros((1, 187))}),
        HeightLabels(**{**vars(_height_label()), "valid": torch.zeros((1, 11, 17), dtype=torch.float32)}),
        HeightLabels(**{**vars(_height_label()), "heights": torch.ones((1, 11, 17))}),
        replace(_height_label(), heights=torch.empty((1, 11, 17), device="meta")),
    ):
        with _raises(RolloutError, ValueError):
            _append(SequenceRolloutBuffer(1, "cpu"), observation, height_label=label)


def test_flat_baseline_without_height_labels_keeps_none_default():
    buffer = SequenceRolloutBuffer(1, "cpu")
    _append(buffer, _observation())
    assert buffer.steps[0].height_label is None


def test_y_mirror_reverses_rows_without_changing_signed_heights():
    label = _height_label()
    mirrored_heights, mirrored_valid = mirror_height_targets(label.heights, label.valid)
    assert torch.equal(mirrored_heights, label.heights.flip(-2))
    assert torch.equal(mirrored_valid, label.valid.flip(-2))
    # Reflection changes which y row a target occupies; height remains a signed scalar.
    assert torch.equal(mirrored_heights, label.heights.flip(-2))
    assert torch.equal(torch.sort(mirrored_heights.flatten()).values, torch.sort(label.heights.flatten()).values)
    assert torch.all(mirrored_heights[~mirrored_valid] == 0)


class _CollectorNetwork:
    @staticmethod
    def initial_hidden(batch, device):
        return torch.zeros(batch, 187, dtype=torch.float32, device=device)

    @staticmethod
    def encode_points(near, *args, **kwargs):
        return torch.zeros(near.shape[0], 187, dtype=torch.float32)

    @staticmethod
    def forward_from_features(observation, feature, hidden):
        return SimpleNamespace(hidden=hidden + 1, action_mean=torch.zeros(1, 18), value=hidden[:, :1] + 1)


class _HeightRuntime:
    device = torch.device("cpu")
    num_envs = 1

    def __init__(self, with_height: bool):
        self.t = 0
        self.with_height = with_height
        self.episode_ids = torch.zeros(1, dtype=torch.int64)

    def prepare(self):
        observation = _observation(step=self.t)
        self._prepared = SimpleNamespace(
            observation=observation,
            near_points=torch.zeros(1, 256, 3), near_valid=torch.zeros(1, 256, dtype=torch.bool),
            far_points=torch.zeros(1, 128, 3), far_valid=torch.zeros(1, 128, dtype=torch.bool),
            near_filled=torch.ones(1, dtype=torch.bool), near_scan_ids=torch.tensor([self.t + 1]),
            near_times=torch.tensor([self.t * 0.1]),
            far_point_scan_ids=torch.full((1, 128), -1, dtype=torch.int64),
            far_point_times=torch.zeros(1, 128),
            near_angle_keys=torch.zeros(1, 256), far_angle_keys=torch.zeros(1, 128),
        )
        return self._prepared

    def labels(self):
        obs = self._prepared.observation
        labels = {"envelope": EnvelopeLabel(
            obs.env_ids.clone(), obs.episode_ids.clone(), obs.step_ids.clone(),
            torch.zeros(1, 6), torch.ones(1, dtype=torch.bool),
        )}
        if self.with_height:
            labels["height"] = _height_label(step=self.t)
        return labels

    def step(self, action):
        self.t += 1
        return SimpleNamespace(
            final_inputs=self.prepare(), reward=torch.zeros(1),
            terminated=torch.zeros(1, dtype=torch.bool), truncated=torch.zeros(1, dtype=torch.bool),
            engineering_stopped=False,
        )


def test_collector_transfers_optional_height_label():
    for with_height in (False, True):
        runtime = _HeightRuntime(with_height)
        collector = TrainingCollector(runtime, _CollectorNetwork(), IndependentDiagonalNormalProvider(generator_seed=7))
        buffer = collector.collect()
        if with_height:
            assert all(step.height_label is not None for step in buffer.steps)
            assert [int(step.height_label.step_ids[0]) for step in buffer.steps] == list(range(24))
            assert torch.equal(buffer.steps[5].height_label.heights, _height_label(step=5).heights)
        else:
            assert all(step.height_label is None for step in buffer.steps)


def test_cpu_sequence_replay_rejects_height_label_from_another_step():
    runtime = _HeightRuntime(with_height=True)
    collector = TrainingCollector(runtime, _CollectorNetwork(), IndependentDiagonalNormalProvider(generator_seed=11))
    buffer = collector.collect()
    network = _CollectorNetwork()

    outputs = replay_sequence(network, buffer.steps)
    assert len(outputs) == 24

    wrong_label = replace(
        buffer.steps[7].height_label,
        step_ids=torch.tensor([8], dtype=torch.int64),
    )
    bad_steps = list(buffer.steps)
    bad_steps[7] = replace(bad_steps[7], height_label=wrong_label)
    with _raises(ValueError):
        replay_sequence(network, bad_steps)
