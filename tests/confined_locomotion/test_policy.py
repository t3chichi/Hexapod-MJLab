"""CPU contracts for online observations, the shared policy, and distributions."""
from __future__ import annotations

import dataclasses
import traceback
import unittest

import torch

from support.fixtures import build_adapter, build_network, build_provider
from hexapod_mjlab.rl.confined_locomotion.contracts import ContractError, JOINT_DIM, OnlineObservation
from hexapod_mjlab.rl.confined_locomotion.policy import (
    ACTION_CLIP_LIMIT,
    ConfinedPolicyAdapter,
    PolicyAdapterError,
)


def _inputs(batch: int = 2, seed: int = 7):
    generator = torch.Generator().manual_seed(seed)
    zeros = lambda shape: torch.zeros(shape, dtype=torch.float32)
    observation = OnlineObservation(
        env_ids=torch.arange(batch, dtype=torch.long),
        episode_ids=zeros((batch,)).long(),
        step_ids=zeros((batch,)).long(),
        time_s=zeros((batch,)),
        lin_vel_b=zeros((batch, 3)),
        ang_vel_b=zeros((batch, 3)),
        gravity_b=torch.tensor([[0.0, 0.0, -1.0]]).expand(batch, 3).clone(),
        joint_pos_rel_ref=zeros((batch, JOINT_DIM)),
        joint_vel=zeros((batch, JOINT_DIM)),
        prev_action=zeros((batch, JOINT_DIM)),
        command=torch.tensor([[0.4, -0.2, 0.1]]).expand(batch, 3).clone(),
    )
    near_valid = torch.zeros((batch, 256), dtype=torch.bool)
    near_valid[:, :32] = True
    near = torch.zeros((batch, 256, 3))
    near[:, :32] = torch.randn((batch, 32, 3), generator=generator)
    far_valid = torch.zeros((batch, 640), dtype=torch.bool)
    far_valid[:, :48] = True
    far = torch.zeros((batch, 640, 3))
    far[:, :48] = torch.randn((batch, 48, 3), generator=generator)
    return observation, near, near_valid, far, far_valid


def _same_output(left, right):
    for field in ("near_feature", "far_feature", "envelope", "action_mean", "value"):
        if not torch.equal(getattr(left, field), getattr(right, field)):
            return False
    if (left.ground_pred is None) != (right.ground_pred is None):
        return False
    return left.ground_pred is None or torch.equal(left.ground_pred, right.ground_pred)


def _prepared(observation, near, near_valid, far, far_valid, *, near_scan_ids=None,
              far_scan_ids=None):
    """Build an explicitly identified prepared block for cache-only CPU probes."""
    from hexapod_mjlab.rl.confined_locomotion.integration import PreparedControlInputs

    batch = int(observation.env_ids.shape[0])
    near_filled = near_valid.any(dim=1)
    if near_scan_ids is None:
        near_scan_ids = torch.where(
            near_filled, torch.ones(batch, dtype=torch.long), torch.full((batch,), -1)
        )
    far_point_scan_ids = torch.full(far_valid.shape, -1, dtype=torch.long)
    if far_scan_ids is None:
        far_point_scan_ids = torch.where(far_valid, 1, -1)
    else:
        far_point_scan_ids[far_valid] = far_scan_ids
    return PreparedControlInputs(
        observation=observation,
        near_points=near,
        near_valid=near_valid,
        far_points=far,
        far_valid=far_valid,
        near_filled=near_filled,
        near_scan_ids=near_scan_ids,
        near_times=observation.time_s.clone(),
        far_point_scan_ids=far_point_scan_ids,
        far_point_times=torch.zeros_like(far_valid, dtype=torch.float32),
    )


def _assert_output_close(left, right):
    for field in ("near_feature", "far_feature", "envelope", "action_mean", "value"):
        assert torch.allclose(
            getattr(left, field), getattr(right, field), atol=1e-6, rtol=1e-6
        ), field
    assert (left.ground_pred is None) == (right.ground_pred is None)
    if left.ground_pred is not None:
        assert torch.allclose(left.ground_pred, right.ground_pred, atol=1e-6, rtol=1e-6)


def _nonzero_morphology_network(seed=20261013):
    network = build_network(seed=seed, ground_samples=12)
    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for head in (network.horizontal_head, network.vertical_head):
            head.terminal.weight.uniform_(-0.05, 0.05, generator=generator)
            head.terminal.bias.uniform_(-0.1, 0.1, generator=generator)
    return network


def test_observation_keeps_proprio_and_command_separate_and_morphology_command_free():
    observation, near, near_valid, far, far_valid = _inputs()
    assert observation.proprio_vector().shape == (2, 63)
    assert observation.command.shape == (2, 3)
    network = build_network(seed=31, ground_samples=12)
    outputs = {}

    def save_input(name):
        def hook(_module, args):
            outputs[name] = args[0].detach().clone()
        return hook

    handles = [
        network.action_head.register_forward_pre_hook(save_input("actor_input")),
        network.value_head.register_forward_pre_hook(save_input("critic_input")),
    ]
    original = network(observation, near, near_valid, far, far_valid)
    for handle in handles:
        handle.remove()
    assert torch.equal(outputs["actor_input"], outputs["critic_input"])

    changed_command = dataclasses.replace(
        observation,
        command=torch.tensor([[-0.8, 0.7, -0.5], [-0.8, 0.7, -0.5]]),
    )
    changed = network(changed_command, near, near_valid, far, far_valid)
    assert torch.equal(original.envelope, changed.envelope)
    assert not torch.equal(original.action_mean, changed.action_mean)
    assert not hasattr(observation, "ground") and not hasattr(observation, "envelope_label")


def test_adapter_is_a_single_stateless_network_and_clips_only_its_mean_view():
    network = build_network(seed=41, ground_samples=12)
    adapter = build_adapter(network=network)
    observation, near, near_valid, far, far_valid = _inputs(seed=11)
    calls = {"network": 0}
    handle = network.register_forward_hook(
        lambda module, args, output: calls.__setitem__("network", calls["network"] + 1)
    )
    first = adapter(observation, near, near_valid, far, far_valid)
    assert calls["network"] == 1
    other_inputs = _inputs(seed=12)
    adapter(*other_inputs)
    repeated = adapter(observation, near, near_valid, far, far_valid)
    handle.remove()
    assert _same_output(first, repeated)
    assert adapter.network is network
    assert set(adapter.state_dict()) == {f"network.{name}" for name in network.state_dict()}
    assert tuple(map(id, adapter.unique_parameters())) == tuple(map(id, network.parameters()))

    mean_view = adapter.mean_actions(observation, near, near_valid, far, far_valid)
    assert torch.equal(mean_view.action_mean, first.action_mean)
    assert mean_view.clip_limit == ACTION_CLIP_LIMIT == 100.0
    assert torch.equal(mean_view.clipped_action, first.action_mean.clamp(-100.0, 100.0))
    with unittest.TestCase().assertRaises(PolicyAdapterError):
        adapter.sample(observation, near, near_valid, far, far_valid)


def test_provider_sample_and_raw_action_probability_keep_event_identity():
    network = build_network(seed=51, ground_samples=12)
    provider = build_provider(seed=51)
    adapter = build_adapter(network=network, provider=provider)
    observation, near, near_valid, far, far_valid = _inputs(seed=23)
    sample = adapter.sample(observation, near, near_valid, far, far_valid)
    assert sample.raw_action.shape == (2, JOINT_DIM)
    assert sample.log_prob.shape == (2,)
    assert torch.isfinite(sample.raw_action).all() and torch.isfinite(sample.log_prob).all()
    assert sample.identity_frozen == adapter.identity_frozen
    assert sample.distribution_identity == provider.distribution_identity()

    # A deterministic out-of-range probe distinguishes the probability of the
    # raw action from the probability of the clipped execution value.
    raw = torch.full((2, JOINT_DIM), 101.0)
    raw_log_prob = adapter.log_prob_actions(
        observation, near, near_valid, far, far_valid, raw_actions=raw
    )
    clipped_log_prob = adapter.log_prob_actions(
        observation, near, near_valid, far, far_valid,
        raw_actions=raw.clamp(-ACTION_CLIP_LIMIT, ACTION_CLIP_LIMIT),
    )
    assert not torch.allclose(raw_log_prob, clipped_log_prob)


def test_action_evaluation_keeps_gradient_to_network_and_provider_once():
    network = build_network(seed=61, ground_samples=12)
    provider = build_provider(seed=61)
    adapter = build_adapter(network=network, provider=provider)
    observation, near, near_valid, far, far_valid = _inputs(seed=29)
    raw_actions = torch.full((2, JOINT_DIM), 0.1)
    result = adapter.evaluate_actions(
        observation, near, near_valid, far, far_valid, raw_actions=raw_actions
    )
    assert result.action_mean.shape == (2, JOINT_DIM)
    assert result.log_prob.shape == result.entropy.shape == (2,)
    (result.log_prob + 0.5 * result.entropy).sum().backward()
    assert provider.log_std.grad is not None
    assert bool((provider.log_std.grad != 0).any())
    assert any(p.grad is not None and bool((p.grad != 0).any()) for p in network.action_head.parameters())
    parameters = list(adapter.parameters())
    unique = adapter.unique_parameters()
    assert len(parameters) == len(unique)
    assert len(unique) == len(network.unique_parameters()) + 1
    assert sum(parameter is provider.log_std for parameter in unique) == 1


def test_no_grad_feature_cache_reuses_only_cloud_rows_and_recomputes_heads():
    from hexapod_mjlab.rl.confined_locomotion.inference import PointCloudFeatureCache

    network = _nonzero_morphology_network()
    cache = PointCloudFeatureCache(network)
    observation, near, near_valid, far, far_valid = _inputs(batch=2, seed=123)
    prepared = _prepared(observation, near, near_valid, far, far_valid)
    with torch.no_grad():
        direct = network(observation, near, near_valid, far, far_valid)
        cached = cache.forward(prepared)
        changed_observation = dataclasses.replace(
            observation,
            joint_vel=observation.joint_vel + 0.25,
            command=observation.command - 0.35,
        )
        changed = cache.forward(
            _prepared(changed_observation, near, near_valid, far, far_valid)
        )
        direct_changed = network(changed_observation, near, near_valid, far, far_valid)

    _assert_output_close(cached, direct)
    _assert_output_close(changed, direct_changed)
    assert torch.equal(cached.near_feature, changed.near_feature)
    assert torch.equal(cached.far_feature, changed.far_feature)
    assert not torch.allclose(cached.envelope, changed.envelope)
    assert not torch.allclose(cached.action_mean, changed.action_mean)
    assert not torch.allclose(cached.value, changed.value)
    assert cache.stats == {
        "near_rows_encoded": 2,
        "far_rows_encoded": 2,
        "near_hits": 2,
        "far_hits": 2,
        "forward_calls": 2,
    }
    assert not hasattr(cached, "envelope_label") and not hasattr(cached, "ground_label")


def test_feature_cache_same_scan_input_mask_order_and_history_changes_miss():
    from hexapod_mjlab.rl.confined_locomotion.inference import PointCloudFeatureCache

    network = _nonzero_morphology_network(seed=20261014)
    observation, near, near_valid, far, far_valid = _inputs(batch=3, seed=456)
    base = _prepared(observation, near, near_valid, far, far_valid)

    def measure(candidate):
        cache = PointCloudFeatureCache(network)
        with torch.no_grad():
            cache.forward(base)
            output = cache.forward(candidate)
        return cache.stats, output

    changed_near = near.clone()
    changed_near[0, 0, 0] += 0.125  # same env/episode/scan id, different point value
    stats, _ = measure(_prepared(observation, changed_near, near_valid, far, far_valid))
    assert stats["near_rows_encoded"] == 4 and stats["far_rows_encoded"] == 3

    changed_valid = near_valid.clone()
    changed_points = near.clone()
    changed_valid[1, 1] = False
    changed_points[1, 1].zero_()  # keep the masked-zero input contract
    stats, _ = measure(_prepared(observation, changed_points, changed_valid, far, far_valid))
    assert stats["near_rows_encoded"] == 4 and stats["far_rows_encoded"] == 3

    reordered_near = near.clone()
    reordered_near[:, [0, 1]] = reordered_near[:, [1, 0]]
    stats, _ = measure(_prepared(observation, reordered_near, near_valid, far, far_valid))
    assert stats["near_rows_encoded"] == 6 and stats["far_rows_encoded"] == 3

    changed_far = far.clone()
    changed_far[2, 0, 0] += 0.125
    stats, _ = measure(_prepared(observation, near, near_valid, changed_far, far_valid))
    assert stats["near_rows_encoded"] == 3 and stats["far_rows_encoded"] == 4

    changed_far_valid = far_valid.clone()
    masked_far = far.clone()
    changed_far_valid[1, 1] = False
    masked_far[1, 1].zero_()
    stats, _ = measure(_prepared(observation, near, near_valid, masked_far, changed_far_valid))
    assert stats["near_rows_encoded"] == 3 and stats["far_rows_encoded"] == 4

    reordered_far = far.clone()
    reordered_far[:, [0, 1]] = reordered_far[:, [1, 0]]
    stats, _ = measure(_prepared(observation, near, near_valid, reordered_far, far_valid))
    assert stats["near_rows_encoded"] == 3 and stats["far_rows_encoded"] == 6

    changed_history_ids = base.far_point_scan_ids.clone()
    changed_history_ids[2, 0] = 9  # same cloud/mask, different far-window scan identity
    changed_history = dataclasses.replace(base, far_point_scan_ids=changed_history_ids)
    history_stats, history_output = measure(changed_history)
    with torch.no_grad():
        direct_history = network(observation, near, near_valid, far, far_valid)
    _assert_output_close(history_output, direct_history)
    assert history_stats["near_rows_encoded"] == 3
    assert history_stats["far_rows_encoded"] == 4


def test_feature_cache_supports_reordered_subset_and_repeated_env_rows():
    from hexapod_mjlab.rl.confined_locomotion.inference import PointCloudFeatureCache

    network = _nonzero_morphology_network(seed=20261015)
    cache = PointCloudFeatureCache(network)
    observation, near, near_valid, far, far_valid = _inputs(batch=3, seed=789)
    observation = dataclasses.replace(observation, env_ids=torch.tensor([10, 20, 30]))
    prepared = _prepared(observation, near, near_valid, far, far_valid)
    with torch.no_grad():
        cache.forward(prepared)

        # Reordered subset reuses by environment identity and still returns the requested order.
        indices = torch.tensor([2, 0])
        subset_observation = dataclasses.replace(
            observation,
            **{
                field.name: getattr(observation, field.name).index_select(0, indices)
                for field in dataclasses.fields(observation)
            },
        )
        subset = _prepared(
            subset_observation,
            near.index_select(0, indices), near_valid.index_select(0, indices),
            far.index_select(0, indices), far_valid.index_select(0, indices),
            near_scan_ids=prepared.near_scan_ids.index_select(0, indices),
            far_scan_ids=None,
        )
        subset_output = cache.forward(subset)
        subset_direct = network(
            subset_observation,
            subset.near_points, subset.near_valid, subset.far_points, subset.far_valid,
        )
        assert cache.stats["near_hits"] == 2
        assert cache.stats["far_hits"] == 2

        # Duplicate env identity is legal for OnlineObservation minibatch rows;
        # each output still reflects its own proprio/command values.
        repeated_ids = torch.tensor([30, 10, 30])
        duplicate_obs = dataclasses.replace(
            observation,
            env_ids=repeated_ids,
            command=observation.command[torch.tensor([2, 0, 2])].clone(),
            joint_vel=observation.joint_vel[torch.tensor([2, 0, 2])].clone(),
        )
        duplicate_points = _prepared(
            duplicate_obs,
            near[torch.tensor([2, 0, 2])], near_valid[torch.tensor([2, 0, 2])],
            far[torch.tensor([2, 0, 2])], far_valid[torch.tensor([2, 0, 2])],
        )
        duplicate_output = cache.forward(duplicate_points)
        duplicate_direct = network(
            duplicate_obs,
            duplicate_points.near_points, duplicate_points.near_valid,
            duplicate_points.far_points, duplicate_points.far_valid,
        )
    _assert_output_close(subset_output, subset_direct)
    _assert_output_close(duplicate_output, duplicate_direct)


def test_feature_cache_is_one_snapshot_per_env_and_never_enters_training_graph():
    from hexapod_mjlab.rl.confined_locomotion.inference import PointCloudFeatureCache

    network = _nonzero_morphology_network(seed=20261016)
    cache = PointCloudFeatureCache(network)
    observation, near, near_valid, far, far_valid = _inputs(batch=1, seed=991)
    first = _prepared(observation, near, near_valid, far, far_valid)
    changed_near = near.clone()
    changed_near[0, 0, 1] += 0.2
    second = _prepared(observation, changed_near, near_valid, far, far_valid)
    with torch.no_grad():
        cache.forward(first)
        cache.forward(second)
        cache.forward(first)
    # Returning to an older observation re-encodes it: the cache keeps only the
    # most recent near/far snapshot for this environment rather than rollout history.
    assert cache.stats["near_rows_encoded"] == 3
    assert cache.stats["far_rows_encoded"] == 1

    with unittest.TestCase().assertRaisesRegex(RuntimeError, "torch.no_grad"):
        cache.forward(first)
    assert cache.stats["forward_calls"] == 3  # rejected grad-mode calls do no cache work

    # A hit must still validate every point slot; the key is not permission to
    # accept malformed masked input.
    invalid_nan = near.clone()
    invalid_nan[0, 255, 0] = float("nan")
    invalid_masked = near.clone()
    invalid_masked[0, 255, 0] = 1.0
    with torch.no_grad():
        for invalid_points in (invalid_nan, invalid_masked):
            with unittest.TestCase().assertRaises(ContractError):
                cache.forward(_prepared(observation, invalid_points, near_valid, far, far_valid))
    assert cache.stats["forward_calls"] == 3

    # Fresh network.forward remains the differentiable path used by PPO.
    output = network(observation, near, near_valid, far, far_valid)
    (output.action_mean.square().mean() + output.value.square().mean()).backward()
    assert any(p.grad is not None and bool((p.grad != 0).any()) for p in network.near_encoder.parameters())
    assert any(p.grad is not None and bool((p.grad != 0).any()) for p in network.far_encoder.parameters())


def test_feature_cache_episode_and_partial_reset_rows_use_snapshot_identity():
    from hexapod_mjlab.rl.confined_locomotion.inference import PointCloudFeatureCache

    network = _nonzero_morphology_network(seed=20261018)
    observation, near, near_valid, far, far_valid = _inputs(batch=2, seed=654)
    observation = dataclasses.replace(observation, env_ids=torch.tensor([4, 9]))
    old_final = _prepared(observation, near, near_valid, far, far_valid)

    # Same scan ids and exact cloud, but one row has entered a new episode:
    # only that environment must miss both encoder entries.
    cache = PointCloudFeatureCache(network)
    with torch.no_grad():
        cache.forward(old_final)
        one_row_new_episode = dataclasses.replace(
            old_final,
            observation=dataclasses.replace(
                observation, episode_ids=torch.tensor([1, 0]), step_ids=torch.tensor([0, 1])
            ),
        )
        changed = cache.forward(one_row_new_episode)
        direct = network(
            one_row_new_episode.observation,
            one_row_new_episode.near_points, one_row_new_episode.near_valid,
            one_row_new_episode.far_points, one_row_new_episode.far_valid,
        )
    _assert_output_close(changed, direct)
    assert cache.stats["near_rows_encoded"] == 3 and cache.stats["near_hits"] == 1
    assert cache.stats["far_rows_encoded"] == 3 and cache.stats["far_hits"] == 1

    # Terminal bootstrap retains old-episode scan data. The next input has one
    # reset row with an empty scan and one unchanged row; it must not borrow the
    # old final's row-0 embedding. Re-reading the old final later is a fresh miss.
    cache = PointCloudFeatureCache(network)
    reset_near = near.clone()
    reset_near_valid = near_valid.clone()
    reset_near[0].zero_()
    reset_near_valid[0].zero_()
    reset_far = far.clone()
    reset_far_valid = far_valid.clone()
    reset_far[0].zero_()
    reset_far_valid[0].zero_()
    next_observation = dataclasses.replace(
        observation, episode_ids=torch.tensor([1, 0]), step_ids=torch.tensor([0, 1])
    )
    next_inputs = _prepared(
        next_observation, reset_near, reset_near_valid, reset_far, reset_far_valid
    )
    with torch.no_grad():
        cache.forward(old_final)  # final snapshot before reset
        next_cached = cache.forward(next_inputs)
        next_direct = network(
            next_observation, reset_near, reset_near_valid, reset_far, reset_far_valid
        )
        old_cached_again = cache.forward(old_final)
        old_direct_again = network(observation, near, near_valid, far, far_valid)
    _assert_output_close(next_cached, next_direct)
    _assert_output_close(old_cached_again, old_direct_again)
    assert cache.stats["near_rows_encoded"] == 4 and cache.stats["near_hits"] == 2
    assert cache.stats["far_rows_encoded"] == 4 and cache.stats["far_hits"] == 2


def test_feature_cache_empty_masks_encode_zero_features_and_hits_still_validate():
    from hexapod_mjlab.rl.confined_locomotion.inference import PointCloudFeatureCache

    network = _nonzero_morphology_network(seed=20261019)
    cache = PointCloudFeatureCache(network)
    observation, near, near_valid, far, far_valid = _inputs(batch=1, seed=777)
    near.zero_()
    near_valid.zero_()
    far.zero_()
    far_valid.zero_()
    prepared = _prepared(observation, near, near_valid, far, far_valid)
    with torch.no_grad():
        first = cache.forward(prepared)
        repeated = cache.forward(prepared)
    assert torch.equal(first.near_feature, torch.zeros_like(first.near_feature))
    assert torch.equal(first.far_feature, torch.zeros_like(first.far_feature))
    assert _same_output(first, repeated)
    assert cache.stats["near_rows_encoded"] == 1 and cache.stats["near_hits"] == 1
    assert cache.stats["far_rows_encoded"] == 1 and cache.stats["far_hits"] == 1


def test_feature_cache_invalidates_after_encoder_update_and_checkpoint_load():
    from hexapod_mjlab.rl.confined_locomotion.inference import PointCloudFeatureCache

    network = _nonzero_morphology_network(seed=20261017)
    cache = PointCloudFeatureCache(network)
    observation, near, near_valid, far, far_valid = _inputs(batch=2, seed=321)
    prepared = _prepared(observation, near, near_valid, far, far_valid)
    state_keys_before = set(network.state_dict())
    original_state = {name: value.clone() for name, value in network.state_dict().items()}

    with torch.no_grad():
        first = cache.forward(prepared)
        with torch.no_grad():
            network.near_encoder.gru.weight_ih_l0.add_(0.01)
        after_update = cache.forward(prepared)
        fresh_update = network(observation, near, near_valid, far, far_valid)
        assert cache.stats["near_rows_encoded"] == 4
        assert cache.stats["far_rows_encoded"] == 4
        _assert_output_close(after_update, fresh_update)

        network.load_state_dict(original_state)
        after_load = cache.forward(prepared)
        fresh_load = network(observation, near, near_valid, far, far_valid)
    _assert_output_close(after_load, fresh_load)
    assert cache.stats["near_rows_encoded"] == 6
    assert cache.stats["far_rows_encoded"] == 6
    assert set(network.state_dict()) == state_keys_before
    assert not any("cache" in name for name in network.state_dict())


def run_cpu_checks() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} CPU checks passed")
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(run_cpu_checks())

