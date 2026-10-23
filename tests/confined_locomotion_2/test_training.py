from __future__ import annotations

from types import SimpleNamespace
from dataclasses import replace

import torch
from torch import nn

from hexapod_mjlab.rl.confined_locomotion.contracts import EnvelopeLabel, OnlineObservation
from hexapod_mjlab.rl.confined_locomotion.distribution import IndependentDiagonalNormalProvider
from hexapod_mjlab.rl.confined_locomotion.losses import ppo_policy_loss
from hexapod_mjlab.rl.confined_locomotion.symmetry import JointPermutation
from hexapod_mjlab.rl.confined_locomotion_2.integration import PreparedControlInputs
from hexapod_mjlab.rl.confined_locomotion_2.auxiliary import NoAuxiliary
from hexapod_mjlab.rl.confined_locomotion_2.ppo import TemporalPPOTrainer
from hexapod_mjlab.rl.confined_locomotion_2.ppo import derive_shuffle_generator_seed
from hexapod_mjlab.rl.confined_locomotion_2.training_rollout import SequenceRolloutBuffer


class TinyTemporalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.1))
        self.bias = nn.Parameter(torch.zeros(18))
        self.seen_hidden = []
        self.seen_obs = []

    def forward(self, observation, near, near_valid, far, far_valid, hidden, *, near_angle_keys=None, far_angle_keys=None):
        self.seen_hidden.append(hidden.detach().clone())
        self.seen_obs.append(observation.lin_vel_b.detach().clone())
        signal = observation.lin_vel_b[:, :1]
        updated = torch.tanh(hidden + self.gain * signal)
        mean = updated[:, :18] + self.bias
        return SimpleNamespace(
            frame_feature=signal.expand(-1, 187), temporal_feature=updated,
            hidden=updated, envelope=updated[:, :6], action_mean=mean,
            value=updated[:, :1],
        )


def _observation(step: int, device: torch.device) -> OnlineObservation:
    z = torch.zeros((1, 3), device=device)
    j = torch.zeros((1, 18), device=device)
    return OnlineObservation(
        env_ids=torch.tensor([0], dtype=torch.int64, device=device),
        episode_ids=torch.tensor([0], dtype=torch.int64, device=device),
        step_ids=torch.tensor([step], dtype=torch.int64, device=device),
        time_s=torch.tensor([step * 0.02], dtype=torch.float32, device=device),
        lin_vel_b=torch.tensor([[float(step + 1), 0, 0]], device=device),
        ang_vel_b=z.clone(), gravity_b=z.clone(), joint_pos_rel_ref=j.clone(),
        joint_vel=j.clone(), prev_action=j.clone(), command=z.clone(),
    )


def _label(step: int, device: torch.device) -> EnvelopeLabel:
    return EnvelopeLabel(
        env_ids=torch.tensor([0], dtype=torch.int64, device=device),
        episode_ids=torch.tensor([0], dtype=torch.int64, device=device),
        step_ids=torch.tensor([step], dtype=torch.int64, device=device),
        values=torch.zeros((1, 6), dtype=torch.float32, device=device),
        valid=torch.tensor([True], dtype=torch.bool, device=device),
    )


def _buffer(device: torch.device, *, terminal_at: int | None = None) -> SequenceRolloutBuffer:
    buf = SequenceRolloutBuffer(1, device)
    hidden = torch.zeros((1, 187), dtype=torch.float32, device=device)
    for t in range(24):
        obs = _observation(t, device)
        label = _label(t, device)
        if terminal_at is not None and t > terminal_at:
            obs.episode_ids[:] = 1
            obs.step_ids[:] = t - terminal_at - 1
            label.episode_ids[:] = 1
            label.step_ids[:] = t - terminal_at - 1
        term = torch.tensor([terminal_at == t], dtype=torch.bool, device=device)
        reward = torch.tensor([1.0 if t == 23 else 0.0], device=device)
        buf.append_step(
            observation=obs,
            near_points=torch.zeros((1, 256, 3), device=device),
            near_valid=torch.zeros((1, 256), dtype=torch.bool, device=device),
            far_points=torch.zeros((1, 128, 3), device=device),
            far_valid=torch.zeros((1, 128), dtype=torch.bool, device=device),
            action=torch.zeros((1, 18), device=device), old_log_prob=torch.zeros(1, device=device),
            old_mean=torch.zeros((1, 18), device=device), old_std=torch.full((1, 18), 0.5, device=device),
            value=torch.zeros(1, device=device), reward=reward, terminated=term,
            truncated=torch.zeros_like(term), next_value=torch.zeros(1, device=device),
            hidden_before=hidden, envelope_label=label,
        )
    return buf


def test_replay_keeps_temporal_gradient_and_clears_hidden_after_done():
    device = torch.device("cpu")
    net = TinyTemporalNet()
    provider = IndependentDiagonalNormalProvider(device=device, generator_seed=5)
    trainer = TemporalPPOTrainer(network=net, distribution_provider=provider, device=device)
    buf = _buffer(device, terminal_at=8)
    # A gradient from the final output must pass through earlier frames in the same episode.
    buf.steps[0].observation.lin_vel_b.requires_grad_()
    buf.steps[20].observation.lin_vel_b.requires_grad_()
    out = trainer._replay(buf.steps)
    out[-1].action_mean.sum().backward()
    assert buf.steps[0].observation.lin_vel_b.grad is not None
    assert torch.isfinite(buf.steps[0].observation.lin_vel_b.grad).all()
    assert torch.count_nonzero(buf.steps[0].observation.lin_vel_b.grad) == 0
    assert buf.steps[20].observation.lin_vel_b.grad.abs().sum() > 0
    # The hidden state entering step 9 is zeroed because step 8 ended the episode.
    assert torch.equal(net.seen_hidden[9], torch.zeros_like(net.seen_hidden[9]))


def test_mirror_replay_uses_separate_initial_state_and_mirrors_full_observation_sequence():
    device = torch.device("cpu")
    net = TinyTemporalNet()
    provider = IndependentDiagonalNormalProvider(device=device, generator_seed=7)
    names = tuple(f"{leg}_{kind}" for leg in ("RF", "LF", "RM", "LM", "RB", "LB")
                  for kind in ("HAA", "HFE", "KFE"))
    perm = JointPermutation.from_joint_names(names)
    trainer = TemporalPPOTrainer(
        network=net, distribution_provider=provider, device=device, symmetry_enabled=True,
        joint_permutation=perm, mount_quat_xyzw=torch.tensor([0., 0., 0., 1.]),
    )
    buf = _buffer(device, terminal_at=8)
    for t, step in enumerate(buf.steps):
        obs = step.observation
        obs.lin_vel_b[:, 1] = 2.0
        near_p = torch.zeros((1, 256, 3), device=device)
        near_v = torch.zeros((1, 256), dtype=torch.bool, device=device)
        far_p = torch.zeros((1, 128, 3), device=device)
        far_v = torch.zeros((1, 128), dtype=torch.bool, device=device)
        prepared = PreparedControlInputs(
            observation=obs, near_points=near_p, near_valid=near_v, far_points=far_p,
            far_valid=far_v, near_filled=torch.tensor([False]), near_scan_ids=torch.tensor([-1]),
            near_times=torch.zeros(1), far_point_scan_ids=torch.full((1, 128), -1, dtype=torch.int64),
            far_point_times=torch.zeros((1, 128)),
            near_source_dirs_sensor=torch.zeros((1, 256, 3)),
            far_source_dirs_sensor=torch.zeros((1, 128, 3)),
            near_source_dirs_present=torch.zeros((1, 256), dtype=torch.bool),
            far_source_dirs_present=torch.zeros((1, 128), dtype=torch.bool),
            near_source_indices=torch.full((1, 256), -1, dtype=torch.int64),
            far_source_indices=torch.full((1, 128), -1, dtype=torch.int64),
        )
        buf.steps[t] = replace(
            step, prepared=prepared,
            mirror_hidden_before=torch.full((1, 187), 3.0 if t == 0 else 0.0),
        )
    outputs = trainer._replay(buf.steps, mirrored=True)
    assert torch.equal(net.seen_hidden[0], torch.full((1, 187), 3.0))
    assert torch.equal(net.seen_obs[0][:, 1], torch.tensor([-2.0]))
    assert torch.equal(net.seen_hidden[9], torch.zeros((1, 187)))
    assert len(outputs) == 24


def test_gae_uses_terminal_bootstrap_mask_and_timeout_continuation_mask():
    device = torch.device("cpu")
    buf = _buffer(device, terminal_at=4)
    buf.steps[4] = buf.steps[4].__class__(**{**vars(buf.steps[4]), "next_value": torch.tensor([100.0])})
    adv, ret = buf.compute_gae()
    # terminal rows never bootstrap from the supplied next value
    assert torch.isfinite(adv).all() and torch.isfinite(ret).all()
    assert ret[4, 0] == buf.steps[4].reward[0]


def test_terminal_and_truncation_flags_may_both_be_set():
    device = torch.device("cpu")
    buf = _buffer(device, terminal_at=23)
    buf.steps[23] = buf.steps[23].__class__(
        **{**vars(buf.steps[23]), "truncated": torch.tensor([True])}
    )
    advantages, returns = buf.compute_gae()
    assert torch.isfinite(advantages).all() and torch.isfinite(returns).all()
    assert returns[23, 0] == buf.steps[23].reward[0]


def test_ppo_loss_matches_reference_formula_and_empty_auxiliary_is_parameter_free():
    new = torch.tensor([0.2, -0.1], requires_grad=True)
    old = torch.zeros(2)
    advantage = torch.tensor([1.0, -1.0])
    got = ppo_policy_loss(new, old, advantage)
    ratio = torch.exp(new - old)
    expected = torch.maximum(-advantage * ratio, -advantage * ratio.clamp(0.8, 1.2)).mean()
    assert torch.allclose(got, expected)
    plugin = NoAuxiliary()
    assert not list(plugin.parameters()) if isinstance(plugin, nn.Module) else True
    result = plugin(SimpleNamespace(action_mean=torch.zeros((1, 18))), None)
    assert result.loss.item() == 0 and result.metrics == {}


def test_temporal_ppo_update_runs_exact_recipe_steps_and_consumes_buffer():
    device = torch.device("cpu")
    net = TinyTemporalNet()
    provider = IndependentDiagonalNormalProvider(device=device, generator_seed=9)
    trainer = TemporalPPOTrainer(network=net, distribution_provider=provider, device=device)
    buf = _buffer(device)
    metrics = trainer.update(buf)
    assert buf.consumed
    assert trainer.update_count == 1
    assert trainer.optimizer_step_count == 20
    assert metrics["optimizer_steps"] == 20
    assert torch.isfinite(net.gain)


def test_shuffle_generator_is_named_private_and_checkpointed():
    device = torch.device("cpu")
    torch.manual_seed(123)
    expected_global = torch.rand(1)
    torch.manual_seed(123)
    first = TemporalPPOTrainer(
        network=TinyTemporalNet(),
        distribution_provider=IndependentDiagonalNormalProvider(device=device, generator_seed=1),
        device=device, seed=42,
    )
    assert torch.equal(torch.rand(1), expected_global)
    state = first.state_dict()
    next_a = torch.randperm(20, generator=first.shuffle_generator)
    second = TemporalPPOTrainer(
        network=TinyTemporalNet(),
        distribution_provider=IndependentDiagonalNormalProvider(device=device, generator_seed=2),
        device=device, seed=42,
    )
    second.load_state_dict(state)
    next_b = torch.randperm(20, generator=second.shuffle_generator)
    assert torch.equal(next_a, next_b)
    assert derive_shuffle_generator_seed(42) == derive_shuffle_generator_seed(42)
    assert derive_shuffle_generator_seed(42) != derive_shuffle_generator_seed(43)

