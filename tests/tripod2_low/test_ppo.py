"""Numerical tests for migrated legacy optimizer behavior (CPU, no training claim)."""
from dataclasses import replace
from contextlib import contextmanager
from types import SimpleNamespace

import torch
from torch import nn

from hexapod_mjlab.rl.confined_locomotion.contracts import EnvelopeLabel, OnlineObservation
from hexapod_mjlab.rl.confined_locomotion.symmetry import JointPermutation, mirror_mean_mse
from hexapod_mjlab.rl.confined_locomotion_2.training_rollout import SequenceRolloutBuffer
from hexapod_mjlab.tasks.confined_locomotion_2.tripod2_low.ppo import (
    LegacyNormalProvider, LegacyPPOTrainer, legacy_kl,
)


@contextmanager
def _raises(message):
    try:
        yield
    except ValueError as error:
        assert message in str(error)
    else:
        raise AssertionError("expected ValueError: " + message)


class TinyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(.1))
        self.bias = nn.Parameter(torch.zeros(18))

    def forward(self, observation, near, near_valid, far, far_valid, hidden, **kwargs):
        h = torch.tanh(hidden + self.gain * observation.lin_vel_b[:, :1])
        return SimpleNamespace(action_mean=h[:, :18] + self.bias, value=h[:, :1],
                               envelope=h[:, :6], hidden=h)


def _trainer(**kwargs):
    return LegacyPPOTrainer(
        network=TinyNetwork(), device="cpu", seed=19,
        distribution_provider=LegacyNormalProvider(device="cpu", generator_seed=3), **kwargs)


def _buffer():
    buffer = SequenceRolloutBuffer(1, "cpu")
    for t in range(24):
        identity = dict(env_ids=torch.tensor([0]), episode_ids=torch.tensor([0]), step_ids=torch.tensor([t]))
        z, j = torch.zeros(1, 3), torch.zeros(1, 18)
        obs = OnlineObservation(**identity, time_s=torch.tensor([t * .02]),
                                lin_vel_b=torch.tensor([[.01 * t, 0., 0.]]), ang_vel_b=z, gravity_b=z,
                                joint_pos_rel_ref=j, joint_vel=j, prev_action=j, command=z)
        envelope = EnvelopeLabel(**identity, values=torch.zeros(1, 6), valid=torch.tensor([True]))
        mean = torch.zeros(1, 18)
        old_lp = torch.distributions.Normal(mean, torch.ones_like(mean)).log_prob(mean).sum(-1)
        buffer.append_step(
            observation=obs, near_points=torch.zeros(1, 256, 3), near_valid=torch.zeros(1, 256, dtype=torch.bool),
            far_points=torch.zeros(1, 128, 3), far_valid=torch.zeros(1, 128, dtype=torch.bool),
            action=mean, old_log_prob=old_lp, old_mean=mean, old_std=torch.ones_like(mean),
            value=torch.tensor([.5 * t]), reward=torch.tensor([float(t == 23)]),
            terminated=torch.tensor([False]), truncated=torch.tensor([False]), next_value=torch.tensor([123.]),
            hidden_before=torch.zeros(1, 187), envelope_label=envelope)
    return buffer


def test_normal_optimizes_std_directly_and_preserves_raw_distribution_and_rng():
    provider = LegacyNormalProvider(generator_seed=7)
    assert tuple(dict(provider.named_parameters())) == ("std",)
    assert torch.equal(provider.std, torch.ones(18))
    mean = torch.zeros(2, 18)
    dist = provider(mean)
    state = provider.generator_state()
    raw = dist.sample()
    provider.load_generator_state(state)
    torch.testing.assert_close(provider(mean).sample(), raw, rtol=0, atol=0)
    ref = torch.distributions.Normal(mean, provider.std)
    torch.testing.assert_close(dist.log_prob(raw), ref.log_prob(raw).sum(-1))
    torch.testing.assert_close(dist.entropy(), ref.entropy().sum(-1))
    opt = torch.optim.SGD(provider.parameters(), lr=.2)
    provider(mean).entropy().mean().backward()
    opt.step()
    torch.testing.assert_close(provider.std, torch.full((18,), .8))
    with torch.no_grad():
        provider.std[0] = 0
    with _raises("positive"):
        provider(mean)


def test_timeout_adds_current_value_even_if_termination_is_simultaneous():
    for terminal in (False, True):
        trainer, buffer = _trainer(), _buffer()
        step = buffer.steps[-1]
        buffer.steps[-1] = replace(step, old_value=torch.tensor([2.]), reward=torch.tensor([3.]),
                                  next_value=torch.tensor([100.]), truncated=torch.tensor([True]),
                                  terminated=torch.tensor([terminal]))
        advantages, returns = trainer.compute_gae(buffer)
        torch.testing.assert_close(returns[-1], torch.tensor([3. + .99 * 2.]))
        torch.testing.assert_close(advantages[-1], torch.tensor([3. + .99 * 2. - 2.]))


def test_gae_reads_next_action_time_values_and_last_bootstrap():
    trainer, buffer = _trainer(), _buffer()
    adv, ret = trainer.compute_gae(buffer)
    # Explicit old storage recurrence, intentionally different from the supplied
    # next_value=123 at interior transitions.
    carry = 0.
    expected = []
    for t in reversed(range(24)):
        next_value = .5 * (t + 1) if t < 23 else 123.
        carry = float(t == 23) + .99 * next_value - .5 * t + .99 * .95 * carry
        expected.append(carry)
    torch.testing.assert_close(adv[:, 0], torch.tensor(list(reversed(expected))))
    torch.testing.assert_close(ret, adv + torch.arange(24)[:, None] * .5)


def test_legacy_kl_includes_epsilon_and_adapts_lr_before_optimizer_step():
    zeros, ones = torch.zeros(2, 18), torch.ones(2, 18)
    expected = torch.log(torch.tensor(1. + 1e-5)) * 18
    torch.testing.assert_close(legacy_kl(zeros, ones, zeros, ones), expected.expand(2))
    trainer, buffer = _trainer(), _buffer()
    # Force old and current means to match at the first minibatch to expose
    # the epsilon-induced positive KL and its immediate learning-rate increase.
    trainer.network.gain.data.zero_()
    lr_at_step = []
    original_step = trainer.optimizer.step
    def step(*args, **kwargs):
        lr_at_step.append(trainer.optimizer.param_groups[0]["lr"])
        return original_step(*args, **kwargs)
    trainer.optimizer.step = step
    rng = trainer.distribution_provider.generator_state()
    metrics = trainer.update(buffer)
    assert lr_at_step[0] == .0015
    assert metrics["epoch_0_minibatch_0_lr"] == lr_at_step[0]
    assert len(lr_at_step) == trainer.optimizer_step_count == 20
    assert buffer.consumed and trainer.update_count == 1
    assert torch.equal(rng, trainer.distribution_provider.generator_state())
    assert trainer.distribution_provider.std.grad is not None


def test_single_shuffle_is_reused_for_every_epoch_and_restored_strictly():
    trainer = _trainer()
    state = trainer.state_dict()
    batches = list(trainer._minibatch_indices(24))
    for epoch in range(5):
        got = torch.cat([selected for e, _, selected in batches if e == epoch])
        assert torch.equal(got.sort().values, torch.arange(24))
        assert torch.equal(got, torch.cat([row[2] for row in batches[:4]]))
    next_state = trainer.shuffle_generator.get_state()
    restored = _trainer()
    restored.load_state_dict(state)
    assert all(torch.equal(a[2], b[2]) for a, b in zip(batches, restored._minibatch_indices(24)))
    assert torch.equal(next_state, restored.shuffle_generator.get_state())
    state["schema"] = "confined_locomotion_2.temporal_ppo/v1"
    with _raises("incompatible"):
        restored.load_state_dict(state)


def test_mirror_loss_gradient_only_enters_mirrored_prediction():
    names = tuple(f"{leg}_{kind}" for leg in ("RF", "LF", "RM", "LM", "RB", "LB")
                  for kind in ("HAA", "HFE", "KFE"))
    permutation = JointPermutation.from_joint_names(names)
    original = torch.randn(4, 18, requires_grad=True)
    mirrored = torch.randn(4, 18, requires_grad=True)
    (.6 * mirror_mean_mse(mirrored, original, permutation)).backward()
    assert original.grad is None
    assert mirrored.grad.abs().sum() > 0


def test_mirrored_observations_do_not_enter_adaptive_kl():
    from hexapod_mjlab.tasks.confined_locomotion_2.tripod2_low import ppo
    names = tuple(f"{leg}_{kind}" for leg in ("RF", "LF", "RM", "LM", "RB", "LB")
                  for kind in ("HAA", "HFE", "KFE"))
    trainer = _trainer(symmetry_enabled=True,
                       joint_permutation=JointPermutation.from_joint_names(names),
                       mount_quat_xyzw=torch.tensor([0., 0., 0., 1.]))
    buffer = _buffer()
    for t, step in enumerate(buffer.steps):
        buffer.steps[t] = replace(step, prepared=object(), mirror_hidden_before=torch.zeros(1, 187))
    original_prepare = ppo.prepare_sequence_replay
    ppo.prepare_sequence_replay = lambda steps, **kwargs: kwargs.get("mirrored", False)
    def replay(mirrored):
        # Deliberately asymmetric outputs: averaging original and mirrored KL
        # would decrease the initial lr, whereas the old original-only rule
        # increases it. Both paths retain a network gradient.
        value = trainer.network.bias[:1].reshape(1, 1)
        mean = trainer.network.bias.reshape(1, 18) + (3. if mirrored else 0.)
        envelope = trainer.network.bias[:6].reshape(1, 6)
        return [SimpleNamespace(action_mean=mean, value=value, envelope=envelope) for _ in range(24)]
    trainer._replay = replay
    try:
        metrics = trainer.update(buffer)
    finally:
        ppo.prepare_sequence_replay = original_prepare
    assert metrics["epoch_0_minibatch_0_lr"] == .0015
    assert metrics["epoch_0_minibatch_0_kl"] < .005
    assert metrics["epoch_0_minibatch_0_symmetry_loss"] == 9.
    assert trainer.network.bias.grad.abs().sum() > 0
