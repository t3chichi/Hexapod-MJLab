"""Action sampling, training buffer, GAE, and collection contracts.

Two layers, deliberately separate:

* CPU contract tests (always run, no env construction): analytic diagonal-Normal
  log_prob/entropy, raw-vs-clipped probability counterexample, single learnable
  log_std + generator round-trip, hand-computed GAE (terminal / timeout /
  batch-end / both-flags), buffer lifecycle and identity guards, ground SOURCE
  dedup (duplicate key, empty frames, grid change, immutability conflict,
  cross-rollout reuse, bucket partition), collector construction rejections.
  CPU passing never claims the GPU path.
* An explicit CUDA recipe (``--output <new dir>`` or
  ``CONFINED_LOCOMOTION_CUDA_COLLECTION_OUTPUT``): real env → TrainingRuntime → ConfinedPolicyAdapter with
  the production provider → TrainingCollector, 2 envs, seed 20261008, full
  86,400-ray scans, temporary M12 training grid, initial std 0.5, 24 sampling
  steps in natural engineering mode, NO optimizer. Fixtures: the natural run
  (honest early stop if the fresh network falls — recorded, never continued),
  an isolated timeout fixture (declared episode-counter preset 999 → truncated
  row, natural-boundary V_next, partial reset, GAE continuation cut) and an
  isolated flip+timeout fixture (declared real upside-down pose + counter 999 →
  fall-priority termination). The recipe file is written BEFORE execution;
  reports keep raw evidence and never write PASS in place of it.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from pathlib import Path

import unittest
import torch

_TEST_DIR = str(Path(__file__).resolve().parent)
if _TEST_DIR not in sys.path:
    sys.path.insert(0, _TEST_DIR)

from support.fixtures import (
    build_collector as _build_collector,
    build_network as _build_network,
    build_provider as _provider,
    build_runtime as _build_runtime,
    m12_training_grid,
)
from support.oracles import generalized_advantage

from hexapod_mjlab.rl.confined_locomotion.contracts import (
    JOINT_DIM,
    EnvelopeLabel,
    OnlineObservation,
)
from hexapod_mjlab.rl.confined_locomotion.distribution import (
    INITIAL_STD_STARTUP,
    LOG_STD_INIT,
    DistributionContractError,
    derive_action_generator_seed,
)
from hexapod_mjlab.rl.confined_locomotion.policy import ConfinedPolicyAdapter
from hexapod_mjlab.rl.confined_locomotion.training_rollout import (
    GroundFrameStore,
    RolloutStepSample,
    TrainingRolloutBuffer,
    TrainingRolloutError,
    TrainingRolloutStateError,
)

RECIPE_SEED = 20261008
RECIPE_NUM_ENVS = 2
RECIPE_STEPS = 24

_CPU = torch.device("cpu")
_REFERENCE_GAMMA = 0.99
_REFERENCE_LAMBDA = 0.95


# ═══════════════════════════ CPU contract tests ═══════════════════════════


# ── 分布 ────────────────────────────────────────────────────────────────────


def test_distribution_single_learnable_log_std_and_identity():
    provider = _provider()
    parameters = list(provider.parameters())
    assert len(parameters) == 1 and parameters[0] is provider.log_std
    assert tuple(provider.log_std.shape) == (JOINT_DIM,)
    assert provider.log_std.requires_grad
    assert torch.allclose(provider.log_std.detach(), torch.full((JOINT_DIM,), LOG_STD_INIT))
    identity = provider.distribution_identity()
    assert identity["family"] == "independent_diagonal_normal"
    assert identity["action_dim"] == JOINT_DIM
    assert identity["initial_std"] == INITIAL_STD_STARTUP == 0.5
    frozen = json.dumps(identity, sort_keys=True, allow_nan=False)
    assert frozen.strip()
    # 学后仍是同一族：identity 不随当前 std 漂移；无任何凭据字段。
    with torch.no_grad():
        provider.log_std.fill_(0.123)
    assert provider.distribution_identity() == identity
    assert "credential" not in frozen and "token" not in frozen


def test_distribution_analytic_log_prob_entropy():
    provider = _provider()
    torch.manual_seed(7)
    mean = torch.randn(4, JOINT_DIM)
    raw = torch.randn(4, JOINT_DIM)
    distribution = provider(mean)
    assert torch.allclose(distribution.std, distribution.stddev)
    # 独立解析公式（torch.distributions.Normal 逐维）与 18 维求和比对。
    reference = torch.distributions.Normal(mean, torch.full_like(mean, 0.5))
    assert torch.allclose(distribution.log_prob(raw), reference.log_prob(raw).sum(-1), atol=1e-6)
    assert torch.allclose(distribution.entropy(), reference.entropy().sum(-1), atol=1e-5)
    # 手算常数：每维熵 0.5(1+ln2π)+lnσ，×18。
    expected_entropy = JOINT_DIM * (0.5 * (1.0 + math.log(2.0 * math.pi)) + math.log(0.5))
    assert torch.allclose(distribution.entropy(), torch.full((4,), expected_entropy), atol=1e-5)
    # log_std 与观测无关：不同均值共享同一 std。
    other = provider(torch.zeros(3, JOINT_DIM))
    assert torch.allclose(other.std[0], distribution.std[0])
    # log_prob/entropy 事件级 (B,)。
    assert distribution.log_prob(raw).shape == (4,) and distribution.entropy().shape == (4,)


def test_raw_not_clipped_probability_counterexample():
    provider = _provider()
    mean = torch.full((1, JOINT_DIM), 120.0)
    distribution = provider(mean)
    raw = mean.clone()                      # 概率峰值在 raw=mean（|raw|>100）
    clipped = raw.clamp(-100.0, 100.0)     # 执行链的 ±100 裁剪副本
    log_prob_raw = distribution.log_prob(raw)
    log_prob_clipped = distribution.log_prob(clipped)
    assert torch.isfinite(log_prob_raw).all() and torch.isfinite(log_prob_clipped).all()
    assert not torch.allclose(log_prob_raw, log_prob_clipped)
    assert float(log_prob_raw[0].detach()) > float(log_prob_clipped[0].detach())  # 概率对应 raw，不是 clip 值


def test_distribution_rejects_contract_violations():
    provider = _provider()
    with unittest.TestCase().assertRaises(DistributionContractError):
        provider(torch.zeros(2, 3))                       # 错误维数
    with unittest.TestCase().assertRaises(DistributionContractError):
        provider(torch.zeros(2, JOINT_DIM, dtype=torch.float64))  # 非 FP32
    bad = torch.zeros(2, JOINT_DIM)
    bad[0, 0] = float("inf")
    with unittest.TestCase().assertRaises(DistributionContractError):
        provider(bad)                                     # 非有限均值
    underflow = _underflow_provider()
    with unittest.TestCase().assertRaises(DistributionContractError):
        underflow(torch.zeros(2, JOINT_DIM))              # std 下溢为 0：拒绝，不 clamp
    distribution = provider(torch.zeros(2, JOINT_DIM))
    with unittest.TestCase().assertRaises(DistributionContractError):
        distribution.log_prob(torch.zeros(3, JOINT_DIM))  # 形状不符
    with unittest.TestCase().assertRaises(DistributionContractError):
        distribution.log_prob(torch.zeros(2, JOINT_DIM, dtype=torch.float64))


def _underflow_provider():
    provider = _provider()
    with torch.no_grad():
        provider.log_std.fill_(-200.0)   # exp 下溢为 0：必须拒绝，不 clamp
    return provider


def test_generator_roundtrip_and_derivation():
    assert derive_action_generator_seed(20261008) == derive_action_generator_seed(20261008)
    assert derive_action_generator_seed(20261008) != derive_action_generator_seed(20261007)
    with unittest.TestCase().assertRaises(DistributionContractError):
        derive_action_generator_seed(True)
    # 同 seed 两 provider 的采样序列一致（显式 generator，非全局流）。
    first, second = _provider(), _provider()
    mean = torch.zeros(4, JOINT_DIM)
    torch.randn(1000)  # 推进全局流：不得影响 provider 采样
    a = first(mean).sample()
    b = second(mean).sample()
    assert torch.equal(a, b)
    # State save/restore round-trip.
    state = first.generator_state()
    c1 = first(mean).sample()
    first.load_generator_state(state)
    c2 = first(mean).sample()
    assert torch.equal(c1, c2)
    again = first(mean).sample()
    assert not torch.equal(c2, again)   # 恢复后继续推进，不重播种


def test_log_std_receives_gradient_and_is_observation_independent():
    provider = _provider()
    mean_a = torch.zeros(1, JOINT_DIM, requires_grad=True)
    mean_b = torch.ones(1, JOINT_DIM, requires_grad=True)
    log_prob_a = provider(mean_a).log_prob(torch.zeros(1, JOINT_DIM))
    log_prob_a.sum().backward()
    assert provider.log_std.grad is not None
    assert tuple(provider.log_std.grad.shape) == (JOINT_DIM,)
    assert torch.isfinite(provider.log_std.grad).all()
    log_prob_b = provider(mean_b).log_prob(torch.ones(1, JOINT_DIM))
    log_prob_b.sum().backward()
    assert mean_a.grad is not None and mean_b.grad is not None
    # 解析梯度：z=0 处 d log_prob/d log_std_i = z_i²−1 = −1（逐维独立）。
    fresh = _provider()
    fresh(torch.zeros(1, JOINT_DIM)).log_prob(torch.zeros(1, JOINT_DIM)).sum().backward()
    assert torch.allclose(fresh.log_std.grad, torch.full((JOINT_DIM,), -1.0), atol=1e-6)


def test_adapter_with_and_without_provider():
    network = _build_network()
    provider = _provider()
    adapter = ConfinedPolicyAdapter(
        network=network,
        distribution_provider=provider,
        distribution_identity=provider.distribution_identity(),
    )
    assert len(adapter.unique_parameters()) == len(network.unique_parameters()) + 1
    observation, points = _tiny_inputs(2)
    sampled = adapter.sample(
        observation, points.near_points, points.near_valid, points.far_points, points.far_valid
    )
    assert tuple(sampled.raw_action.shape) == (2, JOINT_DIM)
    assert tuple(sampled.log_prob.shape) == (2,)
    assert sampled.identity_frozen == adapter.identity_frozen
    # 无 provider 的旧 mean_dev 行为保持：采样/评估显式拒绝。
    mean_only = ConfinedPolicyAdapter(network=_build_network(seed=3))
    with unittest.TestCase().assertRaises(Exception):
        mean_only.sample(
            observation, points.near_points, points.near_valid, points.far_points, points.far_valid
        )


class _PointBlocks:
    def __init__(self, near_points, near_valid, far_points, far_valid):
        self.near_points, self.near_valid = near_points, near_valid
        self.far_points, self.far_valid = far_points, far_valid


def _tiny_inputs(batch, device=_CPU):
    observation = OnlineObservation(
        env_ids=torch.arange(batch, device=device),
        episode_ids=torch.zeros(batch, dtype=torch.long, device=device),
        step_ids=torch.zeros(batch, dtype=torch.long, device=device),
        time_s=torch.zeros(batch, device=device),
        lin_vel_b=torch.zeros(batch, 3, device=device),
        ang_vel_b=torch.zeros(batch, 3, device=device),
        gravity_b=torch.zeros(batch, 3, device=device),
        joint_pos_rel_ref=torch.zeros(batch, JOINT_DIM, device=device),
        joint_vel=torch.zeros(batch, JOINT_DIM, device=device),
        prev_action=torch.zeros(batch, JOINT_DIM, device=device),
        command=torch.zeros(batch, 3, device=device),
    )
    points = _PointBlocks(
        torch.zeros(batch, 256, 3, device=device), torch.zeros(batch, 256, dtype=torch.bool, device=device),
        torch.zeros(batch, 640, 3, device=device), torch.zeros(batch, 640, dtype=torch.bool, device=device),
    )
    return observation, points


# ── GAE 手算（terminal / timeout / 批末 / 双 flag） ─────────────────────────


def _synthetic_sample(
    index,
    *,
    control_id,
    reward,
    value,
    v_next,
    terminated=False,
    truncated=False,
    episode=0,
    step_id=None,
    env=0,
    envelope_valid=True,
):
    batch = 1
    if step_id is None:
        step_id = control_id
    time_s = float(control_id) * 0.02

    def zeros(*shape, dtype=torch.float32):
        return torch.zeros(*shape, dtype=dtype)

    observation = OnlineObservation(
        env_ids=torch.tensor([env], dtype=torch.long),
        episode_ids=torch.tensor([episode], dtype=torch.long),
        step_ids=torch.tensor([step_id], dtype=torch.long),
        time_s=torch.full((batch,), time_s),
        lin_vel_b=zeros(batch, 3), ang_vel_b=zeros(batch, 3),
        gravity_b=zeros(batch, 3),
        joint_pos_rel_ref=zeros(batch, JOINT_DIM),
        joint_vel=zeros(batch, JOINT_DIM),
        prev_action=zeros(batch, JOINT_DIM),
        command=zeros(batch, 3),
    )
    envelope = EnvelopeLabel(
        env_ids=torch.tensor([env], dtype=torch.long),
        episode_ids=torch.tensor([episode], dtype=torch.long),
        step_ids=torch.tensor([step_id], dtype=torch.long),
        values=torch.tensor([[0.59, 0.69, 0.59, 0.89, -0.89, 0.0]]),
        valid=torch.tensor([envelope_valid]),
    )
    return RolloutStepSample(
        control_id=control_id,
        time_s=time_s,
        env_ids=torch.tensor([env], dtype=torch.long),
        episode_ids=torch.tensor([episode], dtype=torch.long),
        step_ids=torch.tensor([step_id], dtype=torch.long),
        observation=observation,
        command=observation.command.clone(),
        near_points=zeros(batch, 256, 3), near_valid=zeros(batch, 256, dtype=torch.bool),
        far_points=zeros(batch, 640, 3), far_valid=zeros(batch, 640, dtype=torch.bool),
        near_filled=torch.tensor([False]),
        near_scan_ids=torch.tensor([-1], dtype=torch.long),
        near_times=zeros(batch),
        far_point_scan_ids=torch.full((batch, 640), -1, dtype=torch.long),
        far_point_times=zeros(batch, 640),
        raw_action=zeros(batch, JOINT_DIM),
        old_log_prob=zeros(batch),
        old_mean=zeros(batch, JOINT_DIM),
        old_std=torch.full((batch, JOINT_DIM), 0.5),
        old_value=torch.tensor([value]),
        reward=torch.tensor([reward]),
        terminated=torch.tensor([terminated]),
        truncated=torch.tensor([truncated]),
        v_next=torch.tensor([v_next]),
        final_episode_ids=torch.tensor([episode], dtype=torch.long),
        final_step_ids=torch.tensor([step_id + 1], dtype=torch.long),
        final_control_id=control_id + 1,
        next_episode_ids=torch.tensor([episode + int(terminated or truncated)], dtype=torch.long),
        next_step_ids=torch.tensor([0 if terminated or truncated else step_id + 1], dtype=torch.long),
        reset_applied=torch.tensor([terminated or truncated]),
        envelope_label=envelope,
    )


def _gae_buffer(steps_data):
    buffer = TrainingRolloutBuffer(
        num_envs=1, ground_samples=12, grid_identity_hash="hash-test", device=_CPU,
        nominal_steps=max(1, len(steps_data)),
    )
    for sample in steps_data:
        buffer.append(sample)
    return buffer


def test_gae_hand_computed_normal_continuation():
    rewards, values, v_nexts = [1.0, 2.0, 3.0], [0.5, 0.6, 0.7], [0.8, 0.9, 1.0]
    buffer = _gae_buffer([
        _synthetic_sample(i, control_id=100 + i, reward=r, value=v, v_next=n)
        for i, (r, v, n) in enumerate(zip(rewards, values, v_nexts))
    ])
    advantage, returns = buffer.compute_gae()
    reference_advantage, reference_returns = generalized_advantage(
        torch.tensor([[1.0], [2.0], [3.0]]),
        torch.tensor([[0.5], [0.6], [0.7]]),
        torch.tensor([[0.8], [0.9], [1.0]]),
        torch.zeros((3, 1), dtype=torch.bool),
        torch.zeros((3, 1), dtype=torch.bool),
        gamma=_REFERENCE_GAMMA,
        lam=_REFERENCE_LAMBDA,
    )
    expected = reference_advantage[:, 0].tolist()
    for index, value in enumerate(expected):
        assert abs(float(advantage[index, 0]) - value) < 1e-5, (index, float(advantage[index, 0]), value)
        assert abs(float(returns[index, 0]) - float(reference_returns[index, 0])) < 1e-5
    # 原始 reward 原样保存；批末（最后一步）adv = δ（无下一步）。
    flat = buffer.replay(torch.arange(3))
    assert torch.allclose(flat.reward, torch.tensor(rewards))
    assert abs(float(advantage[2, 0]) - (3.0 + 0.99 * 1.0 - 0.7)) < 1e-6


def test_gae_terminal_vs_timeout_bootstrap():
    def run(flags):
        return _gae_buffer([
            _synthetic_sample(0, control_id=100, reward=1.0, value=0.5, v_next=0.8),
            _synthetic_sample(1, control_id=101, reward=2.0, value=0.6, v_next=0.9, **flags),
            _synthetic_sample(2, control_id=102, reward=3.0, value=0.7, v_next=1.0),
        ]).compute_gae()[0]

    adv_terminal = run({"terminated": True, "truncated": False})
    adv_timeout = run({"terminated": False, "truncated": True})
    adv_both = run({"terminated": True, "truncated": True})
    gl = _REFERENCE_GAMMA * _REFERENCE_LAMBDA
    # 终止（含双 flag 按真终止）：bootstrap 为 0，且递推切断。
    delta_t = 2.0 - 0.6
    adv_t = delta_t
    adv_0 = (1.0 + 0.99 * 0.8 - 0.5) + gl * adv_t
    for advantage in (adv_terminal, adv_both):
        assert abs(float(advantage[1, 0]) - adv_t) < 1e-5
        assert abs(float(advantage[0, 0]) - adv_0) < 1e-5
    # timeout：V_next 用终止前自然边界价值（非零 bootstrap），递推仍切断。
    delta_o = 2.0 + 0.99 * 0.9 - 0.6
    adv_0_o = (1.0 + 0.99 * 0.8 - 0.5) + gl * delta_o
    assert abs(float(adv_timeout[1, 0]) - delta_o) < 1e-5
    assert abs(float(adv_timeout[0, 0]) - adv_0_o) < 1e-5
    assert float(adv_timeout[1, 0]) != float(adv_terminal[1, 0])


def test_buffer_rejects_unnormalized_value_shapes():
    sample = _synthetic_sample(0, control_id=100, reward=1.0, value=0.5, v_next=0.8)
    broken = _synthetic_sample(0, control_id=100, reward=1.0, value=0.5, v_next=0.8)
    object.__setattr__(broken, "old_value", torch.tensor([[0.5]]))  # (B,1) 未 squeeze
    buffer = _gae_buffer([])
    with unittest.TestCase().assertRaises(TrainingRolloutError):
        buffer.append(broken)
    broken2 = _synthetic_sample(0, control_id=100, reward=1.0, value=0.5, v_next=0.8)
    object.__setattr__(broken2, "v_next", torch.tensor([[0.8]]))
    with unittest.TestCase().assertRaises(TrainingRolloutError):
        buffer.append(broken2)
    buffer.append(sample)  # 合法 (B,) 仍可入库


# ── buffer 生命周期与身份守卫 ────────────────────────────────────────────────


def test_buffer_lifecycle_and_replay():
    buffer = TrainingRolloutBuffer(
        num_envs=1, ground_samples=12, grid_identity_hash="hash-test", device=_CPU, nominal_steps=3
    )
    buffer.append(_synthetic_sample(0, control_id=100, reward=1.0, value=0.5, v_next=0.8))
    assert not buffer.is_full
    with unittest.TestCase().assertRaises(TrainingRolloutStateError):
        buffer.compute_gae()
    with unittest.TestCase().assertRaises(TrainingRolloutStateError):
        buffer.replay(torch.arange(1))
    buffer.append(_synthetic_sample(1, control_id=101, reward=2.0, value=0.6, v_next=0.9))
    buffer.append(_synthetic_sample(2, control_id=102, reward=3.0, value=0.7, v_next=1.0))
    assert buffer.is_full and buffer.steps_filled == 3
    with unittest.TestCase().assertRaises(TrainingRolloutStateError):
        buffer.append(_synthetic_sample(3, control_id=103, reward=1.0, value=0.5, v_next=0.8))
    batch = buffer.replay(torch.tensor([2, 0, 2]))  # 重复/乱序 indices
    assert tuple(batch.raw_action.shape) == (3, JOINT_DIM)
    assert batch.env_ids.tolist() == [0, 0, 0]
    assert batch.step_ids.tolist() == [102, 100, 102]
    assert batch.control_ids.tolist() == [102, 100, 102]
    assert batch.prepared.observation.step_ids.tolist() == [102, 100, 102]
    assert batch.prepared.near_points.shape == (3, 256, 3)
    assert batch.envelope_values.shape == (3, 6) and batch.envelope_valid.tolist() == [True] * 3
    flat = buffer.flat_identities()
    assert flat["env_ids"].tolist() == [0, 0, 0] and flat["control_ids"].tolist() == [100, 101, 102]
    buffer.consume()
    with unittest.TestCase().assertRaises(TrainingRolloutStateError):
        buffer.replay(torch.arange(1))
    buffer.clear()
    assert buffer.steps_filled == 0 and not buffer.consumed
    buffer.append(_synthetic_sample(0, control_id=200, reward=1.0, value=0.5, v_next=0.8))
    assert buffer.steps_filled == 1  # clear 后同 key/身份重新允许


def test_buffer_detaches_sample_and_validates_boundary():
    from dataclasses import replace
    sample = _synthetic_sample(0, control_id=0, reward=1.0, value=0.5, v_next=0.8)
    sample = replace(sample, old_value=torch.tensor([0.5], requires_grad=True))
    buffer = TrainingRolloutBuffer(num_envs=1, ground_samples=12,
        grid_identity_hash="hash-test", device=_CPU, nominal_steps=1)
    with unittest.TestCase().assertRaises(TrainingRolloutError):
        buffer.append(replace(sample, final_step_ids=torch.tensor([9])))
    assert buffer.steps_filled == 0
    buffer.append(sample)
    assert not buffer.replay(torch.tensor([0])).old_value.requires_grad


def test_distribution_rejects_probability_overflow():
    d = _provider()(torch.zeros(1, JOINT_DIM))
    with unittest.TestCase().assertRaises(DistributionContractError):
        d.log_prob(torch.full((1, JOINT_DIM), 1e30))


def test_buffer_identity_guards():
    def fresh():
        return TrainingRolloutBuffer(
            num_envs=1, ground_samples=12, grid_identity_hash="hash-test", device=_CPU, nominal_steps=4
        )
    # control_id 回跳/重复。
    buffer = fresh()
    buffer.append(_synthetic_sample(0, control_id=100, reward=1.0, value=0.5, v_next=0.8))
    with unittest.TestCase().assertRaises(TrainingRolloutError):
        buffer.append(_synthetic_sample(1, control_id=100, reward=1.0, value=0.5, v_next=0.8))
    with unittest.TestCase().assertRaises(TrainingRolloutError):
        buffer.append(_synthetic_sample(1, control_id=102, reward=1.0, value=0.5, v_next=0.8))
    # 重复 (env, episode, step) 身份。
    buffer = fresh()
    buffer.append(_synthetic_sample(0, control_id=100, reward=1.0, value=0.5, v_next=0.8))
    with unittest.TestCase().assertRaises(TrainingRolloutError):
        buffer.append(_synthetic_sample(1, control_id=101, reward=1.0, value=0.5, v_next=0.8, step_id=100))
    # 时间未按 dt 推进。
    stale = _synthetic_sample(1, control_id=101, reward=1.0, value=0.5, v_next=0.8)
    object.__setattr__(stale, "time_s", 100 * 0.02)
    with unittest.TestCase().assertRaises(TrainingRolloutError):
        buffer.append(stale)
    # 旧指令身份：command ≠ observation.command 拒绝。
    mismatched = _synthetic_sample(1, control_id=101, reward=1.0, value=0.5, v_next=0.8)
    object.__setattr__(mismatched, "command", torch.ones(1, 3))
    with unittest.TestCase().assertRaises(TrainingRolloutError):
        buffer.append(mismatched)
    # 包络标签身份与 pre-action 观测错位拒绝（未来 step）。
    future = _synthetic_sample(1, control_id=101, reward=1.0, value=0.5, v_next=0.8)
    object.__setattr__(
        future, "envelope_label",
        EnvelopeLabel(
            env_ids=torch.tensor([0]), episode_ids=torch.tensor([0]),
            step_ids=torch.tensor([999]), values=torch.zeros(1, 6), valid=torch.tensor([True]),
        ),
    )
    with unittest.TestCase().assertRaises(TrainingRolloutError):
        buffer.append(future)
    # env 行序改变拒绝。
    two_env = TrainingRolloutBuffer(
        num_envs=2, ground_samples=12, grid_identity_hash="hash-test", device=_CPU, nominal_steps=2
    )
    a = _synthetic_sample(0, control_id=100, reward=1.0, value=0.5, v_next=0.8, env=0)
    b = _synthetic_sample(0, control_id=100, reward=1.0, value=0.5, v_next=0.8, env=1)
    two_env.append(_merge_samples([a, b]))
    with unittest.TestCase().assertRaises(TrainingRolloutError):
        two_env.append(_merge_samples([b, a]))


def _merge_samples(samples):
    """按行合并多个单行样本为一批（复用首行标量身份）。"""
    first = samples[0]
    import dataclasses
    replacements = {
        "env_ids": torch.cat([s.env_ids for s in samples]),
        "episode_ids": torch.cat([s.episode_ids for s in samples]),
        "step_ids": torch.cat([s.step_ids for s in samples]),
        "observation": _stack_observation(samples),
        "command": torch.cat([s.command for s in samples]),
        "near_points": torch.cat([s.near_points for s in samples]),
        "near_valid": torch.cat([s.near_valid for s in samples]),
        "far_points": torch.cat([s.far_points for s in samples]),
        "far_valid": torch.cat([s.far_valid for s in samples]),
        "near_filled": torch.cat([s.near_filled for s in samples]),
        "near_scan_ids": torch.cat([s.near_scan_ids for s in samples]),
        "near_times": torch.cat([s.near_times for s in samples]),
        "far_point_scan_ids": torch.cat([s.far_point_scan_ids for s in samples]),
        "far_point_times": torch.cat([s.far_point_times for s in samples]),
        "raw_action": torch.cat([s.raw_action for s in samples]),
        "old_log_prob": torch.cat([s.old_log_prob for s in samples]),
        "old_mean": torch.cat([s.old_mean for s in samples]),
        "old_std": torch.cat([s.old_std for s in samples]),
        "old_value": torch.cat([s.old_value for s in samples]),
        "reward": torch.cat([s.reward for s in samples]),
        "terminated": torch.cat([s.terminated for s in samples]),
        "truncated": torch.cat([s.truncated for s in samples]),
        "v_next": torch.cat([s.v_next for s in samples]),
        "final_episode_ids": torch.cat([s.final_episode_ids for s in samples]),
        "final_step_ids": torch.cat([s.final_step_ids for s in samples]),
        "next_episode_ids": torch.cat([s.next_episode_ids for s in samples]),
        "next_step_ids": torch.cat([s.next_step_ids for s in samples]),
        "reset_applied": torch.cat([s.reset_applied for s in samples]),
        "envelope_label": EnvelopeLabel(
            env_ids=torch.cat([s.envelope_label.env_ids for s in samples]),
            episode_ids=torch.cat([s.envelope_label.episode_ids for s in samples]),
            step_ids=torch.cat([s.envelope_label.step_ids for s in samples]),
            values=torch.cat([s.envelope_label.values for s in samples]),
            valid=torch.cat([s.envelope_label.valid for s in samples]),
        ),
    }
    return dataclasses.replace(first, **replacements)


def _stack_observation(samples):
    return OnlineObservation(
        env_ids=torch.cat([s.observation.env_ids for s in samples]),
        episode_ids=torch.cat([s.observation.episode_ids for s in samples]),
        step_ids=torch.cat([s.observation.step_ids for s in samples]),
        time_s=torch.cat([s.observation.time_s for s in samples]),
        lin_vel_b=torch.cat([s.observation.lin_vel_b for s in samples]),
        ang_vel_b=torch.cat([s.observation.ang_vel_b for s in samples]),
        gravity_b=torch.cat([s.observation.gravity_b for s in samples]),
        joint_pos_rel_ref=torch.cat([s.observation.joint_pos_rel_ref for s in samples]),
        joint_vel=torch.cat([s.observation.joint_vel for s in samples]),
        prev_action=torch.cat([s.observation.prev_action for s in samples]),
        command=torch.cat([s.observation.command for s in samples]),
    )


# ── 地面 SOURCE 帧去重 ──────────────────────────────────────────────────────


def _frame_fields(m=12, seed=0, valid_count=6, near_valid_count=3):
    generator = torch.Generator().manual_seed(seed)
    near_points = torch.zeros(1, 256, 3)
    near_valid = torch.zeros(1, 256, dtype=torch.bool)
    if near_valid_count:
        near_points[0, :near_valid_count] = torch.randn(near_valid_count, 3, generator=generator)
        near_valid[0, :near_valid_count] = True
    heights = torch.zeros(m)
    positions = torch.zeros(m, 3)
    valid = torch.zeros(m, dtype=torch.bool)
    if valid_count:
        heights[:valid_count] = torch.linspace(-0.2, 0.3, valid_count)
        positions[:valid_count, :2] = torch.linspace(0.1, 0.8, valid_count, ).unsqueeze(1)
        valid[:valid_count] = True
    return near_points, near_valid, heights, positions, valid


def test_ground_dedup_immutability_and_stats():
    store = GroundFrameStore(ground_samples=12, grid_identity_hash="hash-A", device=_CPU)
    near, near_valid, heights, positions, valid = _frame_fields(seed=0, valid_count=6)
    common = dict(
        env_id=0, episode_id=0, scan_id=1,
        near_points=near, near_valid=near_valid,
        heights=heights, positions=positions, valid=valid,
        time_s=0.0, persistent_scan_count=1,
        scan_pos_w=torch.tensor([0.0, 0.0, 0.46]),
        scan_quat_wxyz=torch.tensor([1.0, 0.0, 0.0, 0.0]),
    )
    first = store.add_source_frame(control_id=0, **common)
    again = store.add_source_frame(control_id=5, **common)   # 同 key 一致重复
    assert again is first and len(store.frames()) == 1
    # 不可变冲突：near 变化 / 标签变化 / 身份变化分别报错并指明字段。
    near_b, near_valid_b, _, _, _ = _frame_fields(seed=1, valid_count=6)
    with unittest.TestCase().assertRaises(TrainingRolloutError) as raised:
        store.add_source_frame(control_id=6, **{**common, "near_points": near_b})
    assert "near_points" in str(raised.exception)
    heights_b = heights.clone(); heights_b[0] += 0.01
    with unittest.TestCase().assertRaises(TrainingRolloutError) as raised:
        store.add_source_frame(control_id=7, **{**common, "heights": heights_b})
    assert "heights" in str(raised.exception)
    with unittest.TestCase().assertRaises(TrainingRolloutError):
        store.add_source_frame(control_id=8, **{**common, "persistent_scan_count": 2})
    # 全 invalid 帧：保留身份、计入统计、不入 S。
    _, near_v0, heights0, positions0, valid0 = _frame_fields(seed=2, valid_count=0, near_valid_count=0)
    store.add_source_frame(
        control_id=9, env_id=0, episode_id=0, scan_id=2,
        near_points=torch.zeros(1, 256, 3), near_valid=near_v0,
        heights=heights0, positions=positions0, valid=valid0,
        time_s=0.1, persistent_scan_count=2,
    )
    stats = store.stats()
    assert stats["total_source_frames"] == 2
    assert stats["supervision_frames_S"] == 1
    assert stats["all_invalid_frames_excluded"] == 1
    assert stats["per_frame_valid_positions"] == [6, 0]
    # 空近端但几何有效仍监督。
    _, near_v1, heights1, positions1, valid1 = _frame_fields(seed=3, valid_count=3, near_valid_count=0)
    store.add_source_frame(
        control_id=10, env_id=1, episode_id=0, scan_id=1,
        near_points=torch.zeros(1, 256, 3), near_valid=near_v1,
        heights=heights1, positions=positions1, valid=valid1,
        time_s=0.1, persistent_scan_count=2,
    )
    assert [f.supervises for f in store.frames()] == [True, False, True]
    # grid 变更：不同 hash 的 store 是另一网格（同 env/episode/scan 重新登记）。
    other = GroundFrameStore(ground_samples=12, grid_identity_hash="hash-B", device=_CPU)
    other.add_source_frame(control_id=0, **common)
    assert len(other.frames()) == 1
    # cross-rollout：clear 后同 key 重新允许（无永久去重）。
    store.clear()
    assert len(store.frames()) == 0
    store.add_source_frame(control_id=0, **common)


def test_ground_bucket_partition_deterministic():
    store = GroundFrameStore(ground_samples=12, grid_identity_hash="hash-A", device=_CPU)
    for scan in range(9):
        _, near_valid, heights, positions, valid = _frame_fields(
            seed=scan, valid_count=0 if scan % 4 == 3 else 5, near_valid_count=1
        )
        store.add_source_frame(
            control_id=scan, env_id=0, episode_id=0, scan_id=scan,
            near_points=torch.zeros(1, 256, 3), near_valid=near_valid,
            heights=heights, positions=positions, valid=valid,
            time_s=float(scan) * 0.05, persistent_scan_count=scan,
        )
    supervision = {i for i, f in enumerate(store.frames()) if f.supervises}
    assert len(supervision) == 7  # 9 帧中 2 帧全 invalid
    buckets = store.bucket_partition()
    union = [index for bucket in buckets for index in bucket]
    assert sorted(union) == sorted(supervision)
    assert len(union) == len(set(union))          # 两两不相交
    sizes = [len(bucket) for bucket in buckets]
    assert max(sizes) - min(sizes) <= 1           # 均衡（4/|S| 统一权重）
    assert store.bucket_partition() == buckets    # 确定性
    rotated = store.bucket_partition(epoch=1)
    assert sorted(i for b in rotated for i in b) == sorted(supervision)
    assert rotated != buckets or len(supervision) <= 1
    # S 完全为空：合法跳过（全空桶）。
    empty = GroundFrameStore(ground_samples=12, grid_identity_hash="hash-C", device=_CPU)
    _, near_valid, heights, positions, valid = _frame_fields(valid_count=0, near_valid_count=0)
    empty.add_source_frame(
        control_id=0, env_id=0, episode_id=0, scan_id=1,
        near_points=torch.zeros(1, 256, 3), near_valid=near_valid,
        heights=heights, positions=positions, valid=valid,
        time_s=0.0, persistent_scan_count=1,
    )
    assert empty.supervision_frames() == ()
    assert all(bucket == [] for bucket in empty.bucket_partition())


# ── collector 构造契约（CPU；无需真实 env） ─────────────────────────────────


def _fake_runtime(m=12):
    from hexapod_mjlab.tasks.confined_locomotion.runtime import TrainingRuntime

    class _FakeGrid:
        num_points = m

        @staticmethod
        def identity_hash():
            return "grid-hash-test"

    class _Fake(TrainingRuntime):
        def __init__(self):  # 不构造真实 env：只做类型/属性契约检查
            self.device = "cpu"
            self.num_envs = 2
            self.mode = "engineering"
            self.grid_cfg = _FakeGrid()

    return _Fake()


def test_collector_construction_rejections():
    from hexapod_mjlab.tasks.confined_locomotion.collector import (
        TrainingCollector,
        model_state_fingerprint,
    )

    network = _build_network(ground_samples=12)
    provider = _provider()
    adapter = ConfinedPolicyAdapter(
        network=network,
        distribution_provider=provider,
        distribution_identity=provider.distribution_identity(),
    )
    with unittest.TestCase().assertRaises(TypeError):
        TrainingCollector(object(), adapter)               # runtime 类型
    with unittest.TestCase().assertRaises(TypeError):
        TrainingCollector(_fake_runtime(), ConfinedPolicyAdapter(network=_build_network(seed=5)))  # 无 provider
    with unittest.TestCase().assertRaises(ValueError):
        TrainingCollector(_fake_runtime(m=11), adapter)    # ground M 不一致
    collector = TrainingCollector(_fake_runtime(m=12), adapter)
    assert collector.steps == 24 and collector.ground_samples == 12
    buffer = collector.make_buffer()
    assert buffer.ground_frames.grid_identity_hash == "grid-hash-test"
    assert buffer.nominal_steps == 24 and buffer.num_envs == 2
    # 指纹：确定性；参数变化后改变。
    fingerprint = model_state_fingerprint(network)
    assert fingerprint == model_state_fingerprint(network)
    with torch.no_grad():
        network.ground_head.weight.add_(1e-3)
    assert model_state_fingerprint(network) != fingerprint


# ═══════════════════════════ 显式 CUDA recipe ═══════════════════════════


def _quat_mul_wxyz(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return torch.tensor(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=torch.float32,
    )


def _independent_gae(rewards, values, v_nexts, terminated, truncated):
    """Independent tensor oracle for (time, environment) data."""
    reward_t = torch.as_tensor(rewards, dtype=torch.float32)
    value_t = torch.as_tensor(values, dtype=torch.float32)
    next_t = torch.as_tensor(v_nexts, dtype=torch.float32)
    terminated_t = torch.as_tensor(terminated, dtype=torch.bool)
    truncated_t = torch.as_tensor(truncated, dtype=torch.bool)
    advantage, returns = generalized_advantage(
        reward_t,
        value_t,
        next_t,
        terminated_t,
        truncated_t,
        gamma=_REFERENCE_GAMMA,
        lam=_REFERENCE_LAMBDA,
    )
    return advantage.tolist(), returns.tolist()


def _verify_full_rollout(buffer, transitions, checks):
    """完整 24 步 buffer 的独立复算与统计核对（自然/夹具通用）。"""
    device = buffer.device
    total = buffer.nominal_steps * buffer.num_envs
    flat = buffer.replay(torch.arange(total, device=device))
    # 1) 概率独立复算（torch.distributions 公式 vs 存储的 old 概率/解析熵）。
    reference = torch.distributions.Normal(flat.old_mean, flat.old_std)
    checks["log_prob_recomputed_max_abs_diff"] = float(
        (flat.old_log_prob - reference.log_prob(flat.raw_action).sum(-1)).abs().max()
    )
    analytic_entropy = flat.old_std.log().sum(-1) + JOINT_DIM * (0.5 * (1.0 + math.log(2 * math.pi)))
    checks["entropy_formula_max_abs_diff"] = float(
        (analytic_entropy - reference.entropy().sum(-1)).abs().max()
    )
    assert checks["log_prob_recomputed_max_abs_diff"] < 1e-4
    assert checks["entropy_formula_max_abs_diff"] < 1e-3
    # 2) 采样一致性：raw−mean 的经验 std ≈ 0.5（初始 std，未退火）。
    deviation = (flat.raw_action - flat.old_mean).reshape(-1)
    checks["empirical_std_of_raw_minus_mean"] = float(deviation.std())
    assert 0.45 < checks["empirical_std_of_raw_minus_mean"] < 0.55
    checks["raw_all_finite"] = bool(torch.isfinite(flat.raw_action).all())
    # 3) 与转移记录一致：reward/flags 逐位、raw 与环境收到的 raw 一致（无预裁剪）。
    for step, transition in enumerate(transitions):
        rows = slice(step * buffer.num_envs, (step + 1) * buffer.num_envs)
        assert torch.equal(flat.reward[rows], transition.reward)
        assert torch.equal(flat.terminated[rows], transition.terminated)
        assert torch.equal(flat.truncated[rows], transition.truncated)
        assert torch.equal(flat.raw_action[rows], transition.context_record["raw_action"])
    checks["raw_matches_env_received_bitwise"] = True
    # 4) GAE 独立复算（Python float；与生产实现互不为镜像）。
    rewards = [t.reward.cpu().tolist() for t in transitions]
    values = _stacked(buffer, "old_value").cpu().tolist()
    v_nexts = _stacked(buffer, "v_next").cpu().tolist()
    advantage, returns = _independent_gae(
        rewards, values, v_nexts,
        [t.terminated.cpu().tolist() for t in transitions],
        [t.truncated.cpu().tolist() for t in transitions],
    )
    stored_adv = buffer.advantage.cpu().tolist()
    stored_ret = buffer.returns.cpu().tolist()
    max_adv_diff = max(abs(stored_adv[s][r] - advantage[s][r])
                       for s in range(buffer.nominal_steps) for r in range(buffer.num_envs))
    max_ret_diff = max(abs(stored_ret[s][r] - returns[s][r])
                       for s in range(buffer.nominal_steps) for r in range(buffer.num_envs))
    checks["gae_recomputed_max_adv_diff"] = max_adv_diff
    checks["gae_recomputed_max_return_diff"] = max_ret_diff
    assert max_adv_diff < 1e-4 and max_ret_diff < 1e-4
    # 5) 地面 SOURCE 帧统计与 4 桶划分。
    stats = buffer.ground_frames.stats()
    checks["ground_stats"] = stats
    buckets = buffer.ground_frames.bucket_partition()
    union = [i for bucket in buckets for i in bucket]
    supervision = {i for i, f in enumerate(buffer.ground_frames.frames()) if f.supervises}
    assert sorted(union) == sorted(supervision) and len(union) == len(set(union))
    checks["bucket_sizes"] = [len(bucket) for bucket in buckets]
    # 6) 有限性与 (B,) 规范化。
    checks["values_all_finite"] = bool(
        torch.isfinite(flat.old_value).all() and torch.isfinite(flat.v_next).all()
    )
    return checks


def _stacked(buffer, field):
    """从 replay 批取 (T,B) 等价视图（仅供复算取数）。"""
    total = buffer.nominal_steps * buffer.num_envs
    flat = buffer.replay(torch.arange(total, device=buffer.device))
    return getattr(flat, field).view(buffer.nominal_steps, buffer.num_envs)


def _fixture_natural(out: Path, device) -> dict:
    env, wrapper, runtime = _build_runtime(
        out, mode="engineering", device=device, ground_grid=m12_training_grid()
    )
    checks = {}
    try:
        collector, _adapter = _build_collector(runtime, device=device, steps=RECIPE_STEPS)
        checks["initial_prepare_repeatable"] = runtime.prepare() is runtime.prepare()
        # 见证包装：记录 collect 实际送给 env 的每一步 raw（不改变行为）。
        transitions = []
        original_step = runtime.step

        def _recording_step(raw_action):
            transition = original_step(raw_action)
            transitions.append(transition)
            return transition

        runtime.step = _recording_step
        try:
            result = collector.collect()
        finally:
            runtime.step = original_step
        buffer = result.buffer
        checks["complete"] = result.complete
        checks["completed_steps"] = result.completed_steps
        checks["mode"] = result.mode
        checks["model_fingerprint"] = result.model_fingerprint
        checks["distribution_identity_frozen"] = result.distribution_identity_frozen
        assert result.completed_steps == len(transitions)
        if result.complete:
            _verify_full_rollout(buffer, transitions, checks)
            # 工程模式批末无 reset；重复读取消费输入仍同对象。
            assert all(t.reset_env_ids is None for t in transitions)
            checks["final_prepare_repeatable"] = runtime.prepare() is runtime.prepare()
            scan_steps = sorted({entry["control_step"] for entry in runtime.scan_log})
            checks["scan_schedule_control_steps"] = scan_steps
            consumed = [entry for entry in runtime.scan_log if entry["control_step"] < RECIPE_STEPS]
            checks["scans_consumed_preaction"] = len(consumed)
            assert buffer.ground_frames.stats()["total_source_frames"] == len(consumed)
            # SOURCE 帧时间均早于最后边界（final 新扫描不当本步监督）。
            final_time = transitions[-1].final_inputs.observation.time_s.min().item()
            assert max(f.time_s for f in buffer.ground_frames.frames()) < final_time
            # 证据留存：完整 24 步 buffer + 小片段（PD/裁剪执行/final V）。
            torch.save(
                {
                    "buffer": buffer,
                    "first_last_context_excerpts": [
                        {
                            "control_id": t.control_id,
                            "clipped_action": t.context_record["clipped_action"].cpu(),
                            "pd_substeps": len(t.context_record["pd"]),
                            "pd_first_entry": (
                                t.context_record["pd"][0].detach().reshape(-1)[:18].cpu()
                                if torch.is_tensor(t.context_record["pd"][0])
                                else str(type(t.context_record["pd"][0]))
                            ),
                        }
                        for t in (transitions[0], transitions[-1])
                    ],
                    "v_next_first_last": _stacked(buffer, "v_next").cpu()[[0, -1]],
                },
                out / "evidence.pt",
            )
        else:
            # 自然跌倒/formal done：如实停止，不伪装完整、不 reset 续跑。
            checks["stopped_reason"] = result.stopped_reason
            assert result.stopped_reason == "engineering_stop"
            assert not buffer.is_full and buffer.steps_filled == result.completed_steps
            with unittest.TestCase().assertRaises(TrainingRolloutStateError):
                buffer.compute_gae()
            torch.save({"partial_buffer": buffer}, out / "evidence.pt")
        checks["summary"] = result.summary()
    finally:
        runtime.close()
        env.close()
    return checks


def _fixture_timeout(out: Path, device) -> dict:
    """Isolated timeout fixture: declared counter preset 999 mid-collection."""
    from hexapod_mjlab.tasks.confined_locomotion.collector import model_state_fingerprint

    env, wrapper, runtime = _build_runtime(
        out, mode="training_lifecycle_fixture", device=device, ground_grid=m12_training_grid()
    )
    checks = {}
    try:
        collector, _adapter = _build_collector(runtime, device=device, steps=RECIPE_STEPS)
        buffer = collector.make_buffer()
        buffer.bind_collection_identity(
            distribution_identity_frozen=collector.policy_adapter.identity_frozen,
            model_fingerprint=model_state_fingerprint(collector.policy_adapter),
        )
        transitions = []
        for step in range(RECIPE_STEPS):
            if step == 6:
                # DECLARED preset（记录在案，绝不静默）：env 1 到期 20 s 预算。
                env.episode_length_buf[1] = 999
                runtime.step_in_episode[1] = 999
                runtime._input_cache.invalidate()  # fixture changed the observation identity
                checks["preset_recorded"] = {"env": 1, "episode_length_buf": 999, "at_step": step}
            transitions.append(collector.collect_step(buffer))
        assert buffer.is_full
        _verify_full_rollout(buffer, transitions, checks)
        boundary = transitions[6]                      # 预置当步即触发
        assert bool(boundary.truncated.any())
        row = int(boundary.truncated.nonzero().flatten()[0])
        checks["timeout_row"] = row
        checks["timeout_step_index"] = 6
        assert not bool(boundary.terminated[row])          # 纯 timeout：非终止
        assert boundary.reset_env_ids is not None and row in boundary.reset_env_ids.tolist()
        # V_next 来自自然边界（终止前 final，非 reset 后状态）且参与 δ。
        step = 6
        stacked_v_next = _stacked(buffer, "v_next")
        checks["timeout_v_next"] = float(stacked_v_next[step, row])
        assert float(stacked_v_next[step, row]) != 0.0     # timeout bootstrap 非零
        # timeout 行 reward 原样（未用 oldV 修正）。
        flat = buffer.replay(torch.arange(buffer.nominal_steps * buffer.num_envs, device=device))
        base = step * buffer.num_envs + row
        assert torch.equal(flat.reward[base], boundary.reward[row])
        # reset 后 next 身份：episode+1、step 0；其余行不受影响。
        assert int(boundary.next_inputs.observation.episode_ids[row]) == \
            int(boundary.final_inputs.observation.episode_ids[row]) + 1
        assert int(boundary.next_inputs.observation.step_ids[row]) == 0
        # SOURCE 帧跨 episode 计入（新 episode 的首扫描是独立 key）。
        episodes_seen = {(f.env_id, f.episode_id) for f in buffer.ground_frames.frames()}
        checks["source_frame_episodes"] = sorted(episodes_seen)
        assert any(episode == 1 for _env, episode in episodes_seen)
        torch.save({"buffer": buffer}, out / "evidence.pt")
    finally:
        runtime.close()
        env.close()
    return checks


def _fixture_flip_timeout(out: Path, device) -> dict:
    """Isolated flip+timeout fixture: real pose write + counter 999 (fall priority)."""
    from hexapod_mjlab.tasks.confined_locomotion.collector import model_state_fingerprint

    env, wrapper, runtime = _build_runtime(
        out, mode="training_lifecycle_fixture", device=device, ground_grid=m12_training_grid()
    )
    checks = {}
    try:
        collector, _adapter = _build_collector(runtime, device=device, steps=RECIPE_STEPS)
        buffer = collector.make_buffer()
        buffer.bind_collection_identity(
            distribution_identity_frozen=collector.policy_adapter.identity_frozen,
            model_fingerprint=model_state_fingerprint(collector.policy_adapter),
        )
        transitions = []
        for step in range(RECIPE_STEPS):
            if step == 6:
                robot = env.scene["robot"]
                flip = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device)  # wxyz 180° about x
                positions = robot.data.root_link_pos_w[[0]].detach().clone()
                orientation = _quat_mul_wxyz(
                    flip.tolist(), robot.data.root_link_quat_w[0].detach().cpu().tolist()
                ).to(device)
                robot.write_root_link_pose_to_sim(
                    torch.cat([positions, orientation.unsqueeze(0)], dim=-1), env_ids=torch.tensor([0], device=device)
                )
                env.scene.write_data_to_sim()
                env.sim.forward()
                runtime._input_cache.invalidate()          # 声明的夹具缓存失效
                env.episode_length_buf[0] = 999
                runtime.step_in_episode[0] = 999
                checks["fixture_presets_recorded"] = {
                    "env": 0, "flip": True, "episode_length_buf": 999, "at_step": step,
                }
            transitions.append(collector.collect_step(buffer))
        assert buffer.is_full
        _verify_full_rollout(buffer, transitions, checks)
        boundary = transitions[6]                      # 翻转+预置当步即触发
        checks["terminated"] = boundary.terminated.cpu().tolist()
        checks["truncated"] = boundary.truncated.cpu().tolist()
        assert boundary.terminated.cpu().tolist()[0] is True       # 真实翻转终止
        assert boundary.truncated.cpu().tolist()[0] is False       # 压制同一步 timeout
        step = 6
        stacked_v_next = _stacked(buffer, "v_next")
        # 真终止 bootstrap 逻辑为 0：δ 中 (1−terminated) 项消去 V_next（存储值
        # 仍为自然边界证据），GAE 独立复算已在 _verify_full_rollout 覆盖。
        checks["terminal_v_next_stored_natural"] = float(stacked_v_next[step, 0])
        torch.save({"buffer": buffer}, out / "evidence.pt")
    finally:
        runtime.close()
        env.close()
    return checks


def run_cuda_recipe(output: Path, *, device="cuda:0") -> dict:
    """Write the recipe FIRST, then run the isolated fixtures sequentially."""
    output.mkdir(parents=True, exist_ok=False)
    recipe = {
        "task": "El-4090-Confined-Locomotion-Flat",
        "device": device,
        "num_envs": RECIPE_NUM_ENVS,
        "seed": RECIPE_SEED,
        "steps_natural": RECIPE_STEPS,
        "lidar_rays": 86400,
        "grid": "M12 temporary explicit engineering grid (geometry_only_flat_training)",
        "action_distribution": {
            "family": "independent_diagonal_normal",
            "action_dim": 18,
            "initial_std": 0.5,
            "generator_stream": "policy_action",
        },
        "optimizer": False,
        "gae": {"gamma": _REFERENCE_GAMMA, "lambda": _REFERENCE_LAMBDA,
                "normalized_in_collection": False},
        "natural_stop_rule": (
            "a natural engineering fall stops the run honestly; recorded, never "
            "continued, never padded to 24 steps"
        ),
        "declared_fixture_manipulations": [
            "timeout: episode_length_buf + runtime.step_in_episode preset 999 on env 1 "
            "at collected step 6 (training_lifecycle_fixture mode)",
            "flip_timeout: real upside-down root pose write + counter preset 999 on "
            "env 0 (fall-priority-over-timeout witness)",
        ],
        "claim": (
            "Finite engineering witness only: sampling, buffer, GAE, ground dedup. No "
            "optimizer, no update, no training/learning claim. CPU tests passing "
            "never claims the GPU path."
        ),
    }
    (output / "recipe.json").write_text(json.dumps(recipe, indent=2))

    fixtures = [
        ("natural24", _fixture_natural),
        ("timeout", _fixture_timeout),
        ("flip_timeout", _fixture_flip_timeout),
    ]
    summaries = {}
    failures = {}
    for name, function in fixtures:
        directory = output / name
        directory.mkdir(parents=True, exist_ok=True)
        try:
            summaries[name] = function(directory, device)
        except Exception:
            failures[name] = traceback.format_exc()
        (directory / "report.json").write_text(
            json.dumps({"checks": summaries.get(name), "error": failures.get(name)}, indent=2, default=str)
        )
    summary = {
        "checks": summaries,
        "failures": sorted(failures),
        "cuda_available": torch.cuda.is_available(),
        "peak_cuda_allocated": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    if failures:
        raise RuntimeError(f"CUDA collection recipe fixtures failed: {sorted(failures)}")
    return summaries


def _require_cuda() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("this recipe requires real CUDA (CPU passing claims nothing)")
    return "cuda:0"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    device = args.device or _require_cuda()
    run_cuda_recipe(args.output, device=device)
    print("CUDA collection recipe completed (sampling/buffer/GAE witness; no training claim)")


def test_cuda_recipe_explicit_opt_in():
    """Opt-in via CONFINED_LOCOMOTION_CUDA_COLLECTION_OUTPUT and a new directory."""
    value = os.environ.get("CONFINED_LOCOMOTION_CUDA_COLLECTION_OUTPUT")
    if not value:
        raise unittest.SkipTest(
            "CUDA recipe is opt-in: set CONFINED_LOCOMOTION_CUDA_COLLECTION_OUTPUT=<new dir> (CPU "
            "tests passing never claims the GPU path)"
        )
    run_cuda_recipe(Path(value), device=_require_cuda())


def run_cpu_checks() -> int:
    """Run every CPU contract check (explicitly executable; no pytest needed)."""
    functions = [(name, obj) for name, obj in sorted(globals().items())
                 if name.startswith("test_") and callable(obj)
                 and name != "test_cuda_recipe_explicit_opt_in"]
    failures = 0
    for name, function in functions:
        try:
            function()
            print(f"PASS {name}")
        except Exception:
            failures += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"{len(functions) - failures}/{len(functions)} CPU checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--cpu":
        sys.exit(run_cpu_checks())
    main()
