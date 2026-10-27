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
    calls = []
    def record_batch(module, args):
        calls.append(tuple(args[0].shape))
    handle = provider.register_forward_pre_hook(record_batch)
    action_rng = provider.generator_state()
    metrics = trainer.update(buf)
    handle.remove()
    assert calls == [(24, 18)] * 20
    assert torch.equal(provider.generator_state(), action_rng)
    assert provider.log_std.grad is not None and torch.isfinite(provider.log_std.grad).all()
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



def test_prepared_replay_batches_scans_and_recomputes_trainable_features():
    from types import SimpleNamespace
    from hexapod_mjlab.rl.confined_locomotion_2.training_rollout import (
        prepare_sequence_replay, replay_sequence,
    )

    class FeatureNet(TinyTemporalNet):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(3, 187)
            self.encoded_rows = []

        def encode_points(self, near, *args, **kwargs):
            self.encoded_rows.append(len(near))
            return self.encoder(near[:, 0])

        def forward_from_features(self, observation, feature, hidden):
            updated = torch.tanh(0.5 * hidden + feature)
            return SimpleNamespace(hidden=updated, action_mean=updated[:, :18],
                                   value=updated[:, :1], envelope=updated[:, :6])

    net = FeatureNet()
    steps = _buffer(torch.device("cpu")).steps
    for t, step in enumerate(steps):
        near = torch.ones_like(step.near_points)
        prepared = SimpleNamespace(
            observation=step.observation, near_points=near, near_valid=torch.ones_like(step.near_valid),
            far_points=step.far_points, far_valid=step.far_valid,
            near_scan_ids=torch.zeros(1, dtype=torch.long),
            far_point_scan_ids=torch.full((1, 128), -1, dtype=torch.long),
        )
        steps[t] = replace(step, prepared=prepared)
    plan = prepare_sequence_replay(steps)
    assert plan.point_inputs[0].shape[0] == 1  # One immutable scan across all 24 steps.
    first = replay_sequence(net, plan, checkpoint_points=True)
    first[-1].action_mean.sum().backward()
    assert net.encoder.weight.grad.abs().sum() > 0
    assert net.encoded_rows == [1, 1]  # Forward plus backward recomputation.
    before = first[-1].action_mean.detach().clone()
    with torch.no_grad():
        net.encoder.bias.add_(0.1)
    second = replay_sequence(net, plan)
    assert not torch.allclose(before, second[-1].action_mean)
    assert net.encoded_rows[-1] == 1  # Input plan reused; features recomputed at new weights.
    bad = list(steps)
    bad[1] = replace(bad[1], prepared=SimpleNamespace(**{
        **vars(bad[1].prepared), "near_points": near + 1,
    }))
    try:
        prepare_sequence_replay(bad)
    except ValueError as error:
        assert "same scan identity" in str(error)
    else:
        raise AssertionError("conflicting scan snapshots were accepted")


def test_segmented_temporal_replay_preserves_resets_and_frame_gradients():
    from hexapod_mjlab.rl.confined_locomotion_2.network import TerrainMemoryNetwork
    from hexapod_mjlab.rl.confined_locomotion_2.training_rollout import _temporal_layout

    torch.manual_seed(73)
    net = TerrainMemoryNetwork(mlp_activation=nn.ELU(), init_fn=None, envelope_mode="raw")
    time, batch = 6, 3
    observations, steps = [], []
    hidden = torch.randn(batch, 187) * 0.1
    for t in range(time):
        obs = _observation(t, torch.device("cpu"))
        values = {name: value.repeat((batch,) + (1,) * (value.ndim - 1))
                  for name, value in vars(obs).items()}
        values["env_ids"] = torch.arange(batch)
        observations.append(OnlineObservation(**values))
        steps.append(SimpleNamespace(
            hidden_before=hidden, terminated=torch.tensor([t in (1, 2), False, t == 5]),
            truncated=torch.tensor([False, t == 3, False]),
        ))
    frames = torch.randn(time, batch, 187, requires_grad=True)
    outputs, h = [], hidden.detach().clone()
    for t in range(time):
        if t:
            h = torch.where((steps[t - 1].terminated | steps[t - 1].truncated)[:, None],
                            torch.zeros_like(h), h)
        output = net.forward_from_features(observations[t], frames[t], h)
        outputs.append(output)
        h = output.hidden
    fields = ("hidden", "envelope", "action_mean", "value", "height_prediction")
    reference = {name: torch.stack([getattr(o, name) for o in outputs]) for name in fields}
    sum(value.square().mean() for value in reference.values()).backward()
    reference_gradient = frames.grad.clone()
    net.zero_grad(set_to_none=True)
    frames.grad = None
    scaled = net.prepare_sequence_observations(observations)
    batched = net.forward_sequence_from_features(
        observations, frames, hidden, _temporal_layout(steps), scaled_observations=scaled)
    values = {name: torch.stack([getattr(o, name) for o in batched]) for name in fields}
    for name in fields:
        torch.testing.assert_close(values[name], reference[name], atol=2e-5, rtol=2e-4)
    sum(value.square().mean() for value in values.values()).backward()
    torch.testing.assert_close(frames.grad, reference_gradient, atol=2e-5, rtol=2e-4)


def test_collector_batches_only_new_scans_with_separate_mirror_cache():
    from hexapod_mjlab.tasks.confined_locomotion_2.collector import TrainingCollector

    encoded_batches = []
    def encode(near, *args, **kwargs):
        encoded_batches.append(len(near))
        return near[:, 0, :1].expand(-1, 187).clone()
    network = SimpleNamespace(initial_hidden=lambda b, device: torch.zeros(b, 187, device=device),
                              encode_points=encode)
    runtime = SimpleNamespace(device=torch.device("cpu"), num_envs=3,
                              episode_ids=torch.zeros(3, dtype=torch.long))
    collector = TrainingCollector(runtime, network, None)
    prepared = SimpleNamespace(
        observation=SimpleNamespace(episode_ids=torch.zeros(3, dtype=torch.long)),
        near_scan_ids=torch.zeros(3, dtype=torch.long),
        near_points=torch.arange(1, 4).float()[:, None, None].expand(-1, 256, 3).clone(),
        near_valid=torch.ones(3, 256, dtype=torch.bool),
        far_points=torch.zeros(3, 128, 3), far_valid=torch.zeros(3, 128, dtype=torch.bool),
        near_angle_keys=None, far_angle_keys=None,
    )
    first = collector._frame(prepared)
    torch.testing.assert_close(first[:, 0], torch.tensor([1., 2., 3.]))
    collector._frame(prepared)
    assert encoded_batches == [3]
    prepared.observation.episode_ids[1] += 1
    prepared.near_scan_ids[2] += 1
    prepared.near_points[1] = 7
    prepared.near_points[2] = 9
    torch.testing.assert_close(collector._frame(prepared)[:, 0], torch.tensor([1., 7., 9.]))
    assert encoded_batches == [3, 2]
    collector._frame(prepared, mirror=True)
    collector.clear_feature_cache()
    collector._frame(prepared)
    assert encoded_batches == [3, 2, 3, 3]
    assert collector.frame_encoding_rows == 11
    assert not collector.hidden.any()


def test_collector_bounds_large_scan_batches_and_preserves_row_order():
    from hexapod_mjlab.tasks.confined_locomotion_2.collector import TrainingCollector

    batches = []
    def encode(near, *args, **kwargs):
        batches.append(len(near))
        return near[:, 0, :1].expand(-1, 187).clone()
    count = 257
    collector = TrainingCollector(
        SimpleNamespace(device="cpu", num_envs=count, episode_ids=torch.zeros(count, dtype=torch.long)),
        SimpleNamespace(initial_hidden=lambda b, device: torch.zeros(b, 187, device=device), encode_points=encode), None)
    prepared = SimpleNamespace(
        observation=SimpleNamespace(episode_ids=torch.zeros(count, dtype=torch.long)),
        near_scan_ids=torch.zeros(count, dtype=torch.long),
        near_points=torch.arange(count).float()[:, None, None].expand(-1, 256, 3),
        near_valid=torch.ones(count, 256, dtype=torch.bool),
        far_points=torch.zeros(count, 128, 3), far_valid=torch.zeros(count, 128, dtype=torch.bool),
        near_angle_keys=None, far_angle_keys=None)
    result = collector._frame(prepared)
    assert batches == [128, 128, 1]
    torch.testing.assert_close(result[:, 0], torch.arange(count).float())
    collector._frame(prepared)
    assert batches == [128, 128, 1] and collector.frame_encoding_rows == count


def test_prepacked_point_inputs_match_online_encoder_and_recompute_gradients():
    from torch.utils.checkpoint import checkpoint
    from hexapod_mjlab.rl.confined_locomotion_2.network import ConfinedLocomotionNetwork

    torch.manual_seed(81)
    net = ConfinedLocomotionNetwork(mlp_activation=nn.ELU(), init_fn=None, envelope_mode="raw")
    near, far = torch.randn(4, 256, 3), torch.randn(4, 128, 3)
    nv, fv = torch.rand(4, 256) > .4, torch.rand(4, 128) > .6
    nv[0] = False
    fv[0] = False
    near[~nv] = 0
    far[~fv] = 0
    nk, fk = torch.randint(0, 9, (4, 256)).float(), torch.randint(0, 9, (4, 128)).float()
    nk[~nv] = float("nan")
    fk[~fv] = float("nan")
    reference = net.encode_points(near, nv, far, fv, near_angle_keys=nk, far_angle_keys=fk)
    reference.square().sum().backward()
    gradients = {name: p.grad.clone() for name, p in net.frame_encoder.named_parameters()}
    net.zero_grad(set_to_none=True)
    packed = net.prepare_point_sequence(near, nv, far, fv, near_angle_keys=nk, far_angle_keys=fk)
    output = checkpoint(net.encode_point_sequence, packed, use_reentrant=False, preserve_rng_state=False)
    torch.testing.assert_close(output, reference)
    output.square().sum().backward()
    for name, p in net.frame_encoder.named_parameters():
        torch.testing.assert_close(p.grad, gradients[name])
    with torch.no_grad():
        net.frame_encoder.gru.bias_ih_l0.add_(.1)
    assert not torch.allclose(net.encode_point_sequence(packed), output)
    empty = net.prepare_point_sequence(torch.zeros_like(near), torch.zeros_like(nv),
                                      torch.zeros_like(far), torch.zeros_like(fv))
    assert torch.equal(net.encode_point_sequence(empty), torch.zeros(4, 187))


def test_execution_mode_controls_only_recomputation_and_cross_mode_restore():
    import hexapod_mjlab.rl.confined_locomotion_2.ppo as ppo

    seen = []
    original = ppo.replay_sequence
    ppo.replay_sequence = lambda *args, **kwargs: seen.append(kwargs["checkpoint_points"])
    try:
        for mode, recompute in (("low_memory", True), ("speed", False)):
            trainer = TemporalPPOTrainer(
                network=TinyTemporalNet(), device="cpu", execution_mode=mode,
                distribution_provider=IndependentDiagonalNormalProvider(device="cpu", generator_seed=5))
            trainer._replay([])
            assert seen[-1] is recompute
            state = trainer.state_dict()
            assert state["execution_mode"] == mode
            trainer.load_state_dict(state)
            state["execution_mode"] = "speed" if recompute else "low_memory"
            trainer.load_state_dict(state)
            assert trainer.execution_mode == mode
            state["execution_mode"] = "bogus"
            try:
                trainer.load_state_dict(state)
            except ValueError as error:
                assert "execution mode" in str(error)
            else:
                raise AssertionError("invalid saved execution mode accepted")
    finally:
        ppo.replay_sequence = original


def test_batched_finite_checks_keep_optimizer_failure_boundaries():
    original_clip = torch.nn.utils.clip_grad_norm_
    for fault, expected, expected_steps in (
        ('gradient', 'non-finite PPO gradient', 0),
        ('clipped_gradient', 'non-finite PPO gradient after clipping', 0),
        ('parameter', 'non-finite PPO parameter after optimizer step', 1),
        ('adam', 'non-finite Adam state after optimizer step', 1),
    ):
        net = TinyTemporalNet()
        trainer = TemporalPPOTrainer(
            network=net, device='cpu',
            distribution_provider=IndependentDiagonalNormalProvider(device='cpu', generator_seed=5))
        if fault == 'gradient':
            net.gain.register_hook(lambda grad: torch.full_like(grad, float('nan')))
        elif fault == 'clipped_gradient':
            def corrupt_clip(params, *args, **kwargs):
                params = list(params)
                norm = original_clip(params, *args, **kwargs)
                params[0].grad.fill_(float('inf'))
                return norm
            torch.nn.utils.clip_grad_norm_ = corrupt_clip
        else:
            step = trainer.optimizer.step
            def corrupt_step(*args, **kwargs):
                result = step(*args, **kwargs)
                with torch.no_grad():
                    if fault == 'parameter':
                        net.gain.fill_(float('nan'))
                    else:
                        next(iter(trainer.optimizer.state.values()))['exp_avg'].fill_(float('inf'))
                return result
            trainer.optimizer.step = corrupt_step
        try:
            trainer.update(_buffer(torch.device('cpu')))
        except FloatingPointError as error:
            assert str(error) == expected
            assert trainer.optimizer_step_count == expected_steps and trainer.faulted
        else:
            raise AssertionError(f'{fault} was not detected at its optimizer boundary')
        finally:
            torch.nn.utils.clip_grad_norm_ = original_clip
