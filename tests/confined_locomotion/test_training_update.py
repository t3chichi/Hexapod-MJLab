"""PPO update contracts and an explicitly selected finite CUDA recipe.

Two layers, deliberately separate:

* CPU supplement (always runs; pure ``hexapod_mjlab.rl.confined_locomotion`` only —
  no task/mjlab import, no env): hand-computed PPO policy clip / value clip /
  entropy scale in the joint total, envelope six-dim **sum** scale with valid-row
  indexing, ground per-frame valid-position mean + independent-frame 4-bucket
  ``4/|S|`` denominators (four-bucket average == full-S equal-weight average),
  empty bucket / all-invalid frame / NaN-in-invalid-slot indexing, exact
  ``torch.distributions`` KL (rsl_rl 5.4.2 semantics) against an independent
  analytic formula, KL learning-rate schedule bounds (including the unclamped
  (1e-3, 1e-2] band and both clamp edges), full-rollout advantage normalization
  applied exactly once, frozen config binding, single-Adam unique-parameter
  construction, fault handling that keeps the buffer, and
  explicit gradient-isolation probes on **model copies with declared non-zero
  morphology terminal weights** (PPO-only / value-only leave the morphology
  heads untouched while the shared perception direct path stays live;
  ground-only touches only near encoder + ground head; a controlled non-zero
  envelope error updates the correct morphology heads and shared perception).
* An explicit CUDA recipe (``--output <new dir>`` or
  ``CONFINED_LOCOMOTION_CUDA_UPDATE_OUTPUT``): reuses test-only constructors
  from ``support.fixtures`` — 2 envs, full 86,400-ray scans,
  near 256 / far 64×10, explicit temporary M12 grid, seed 20261008, natural
  24-step collection → **20 joint updates** (5 epochs × 4 minibatches, one Adam)
  → one explicit post-update re-collection step. The recipe file is written
  BEFORE execution; the full pre-update buffer/model, post-update model/Adam/
  LR/RNG and per-minibatch diagnostic raw arrays (first and last minibatch) are
  saved for independent recomputation. A natural fall or fault stops honestly
  and reports NOT PASSED — no fixture mode is used to fake the main gate, and
  CPU passing never claims the GPU path. No learning/standing/walking claim.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import sys
import time
import traceback
import unittest
from pathlib import Path

import torch

_TEST_DIR = str(Path(__file__).resolve().parent)
if _TEST_DIR not in sys.path:
    sys.path.insert(0, _TEST_DIR)

from support.fixtures import (
    build_collector,
    build_network as _build_network,
    build_provider as _provider,
    build_runtime,
    m12_training_grid,
)
from support.oracles import (
    clipped_value_loss as oracle_value_clip_loss,
    diagonal_normal_kl as oracle_diagonal_normal_kl,
    envelope_sum_loss as oracle_envelope_sum_loss,
    ground_bucket_loss as oracle_ground_bucket_loss,
    ground_frame_loss as oracle_ground_frame_loss,
    joint_loss as oracle_joint_loss,
    kl_scheduled_learning_rate as oracle_adjust_learning_rate,
    normalize_advantage as oracle_normalize_rollout_advantage,
    ppo_policy_loss as oracle_ppo_policy_loss,
)

from hexapod_mjlab.rl.confined_locomotion.contracts import JOINT_DIM, EnvelopeLabel, OnlineObservation
from hexapod_mjlab.rl.confined_locomotion.distribution import (
    derive_action_generator_seed,
)
from hexapod_mjlab.rl.confined_locomotion.losses import (
    adjust_learning_rate,
    envelope_sum_loss,
    ground_bucket_loss,
    ground_frame_mean_loss,
    kl_divergence_old_new,
    normalize_rollout_advantage,
    ppo_policy_loss,
    total_joint_loss,
    value_clipped_loss,
)
from hexapod_mjlab.rl.confined_locomotion.policy import ConfinedPolicyAdapter
from hexapod_mjlab.rl.confined_locomotion.ppo import (
    PPO_CONFIG_SCHEMA_VERSION,
    ConfinedPPOConfig,
    ConfinedPPOTrainer,
    PPOUpdateError,
    PPOUpdateFault,
    derive_shuffle_generator_seed,
    policy_state_fingerprint,
)
from hexapod_mjlab.rl.confined_locomotion.training_rollout import (
    RolloutStepSample,
    TrainingRolloutBuffer,
)

RECIPE_SEED = 20261008
RECIPE_NUM_ENVS = 2
RECIPE_STEPS = 24
GROUND_M = 12
GRID_HASH = "cpu-ground-grid-fixture-hash"

_CPU = torch.device("cpu")


def _probe_network(seed):
    """梯度隔离探针的模型副本：显式非零形态末层权重/偏置扰动（不碰生产初始化）。

    生产形态末层为零权重+基准偏置（初始 Ê=标签、L_E=0 正常）；探针需要非零
    末层才能区分「梯度路径被截断」与「零权重掩盖」。扰动显式声明于 recipe，
    不篡改生产标签/初始化。
    """
    network = _build_network(seed=seed)
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for head in (network.horizontal_head, network.vertical_head):
            head.terminal.weight.uniform_(-0.05, 0.05, generator=generator)
            head.terminal.bias.fill_(0.5)
    return network


def _tiny_inputs(batch, *, seed=0, near_valid=16, far_valid=32):
    """小型合法在线输入：随机有效点 + 掩码 false 处精确 0。"""
    generator = torch.Generator().manual_seed(seed)
    near_points = torch.zeros(batch, 256, 3)
    near_valid_mask = torch.zeros(batch, 256, dtype=torch.bool)
    near_points[:, :near_valid] = torch.randn(batch, near_valid, 3, generator=generator)
    near_valid_mask[:, :near_valid] = True
    far_points = torch.zeros(batch, 640, 3)
    far_valid_mask = torch.zeros(batch, 640, dtype=torch.bool)
    far_points[:, :far_valid] = 0.5 * torch.randn(batch, far_valid, 3, generator=generator)
    far_valid_mask[:, :far_valid] = True
    observation = OnlineObservation(
        env_ids=torch.arange(batch),
        episode_ids=torch.zeros(batch, dtype=torch.long),
        step_ids=torch.zeros(batch, dtype=torch.long),
        time_s=torch.zeros(batch),
        lin_vel_b=torch.zeros(batch, 3), ang_vel_b=torch.zeros(batch, 3),
        gravity_b=torch.zeros(batch, 3),
        joint_pos_rel_ref=torch.zeros(batch, JOINT_DIM),
        joint_vel=torch.zeros(batch, JOINT_DIM),
        prev_action=torch.zeros(batch, JOINT_DIM),
        command=torch.zeros(batch, 3),
    )
    return observation, near_points, near_valid_mask, far_points, far_valid_mask


def _grad_all_zero(module) -> bool:
    """模块全部参数梯度为 None 或精确 0（区分「无连接」与「零值掩盖」需配合非零末层）。"""
    for parameter in module.parameters():
        if parameter.grad is not None and bool((parameter.grad != 0).any()):
            return False
    return True


def _grad_any_nonzero(module) -> bool:
    for parameter in module.parameters():
        if parameter.grad is not None and bool((parameter.grad != 0).any()):
            return True
    return False


def _clear_grads(module) -> None:
    for parameter in module.parameters():
        parameter.grad = None


# ═══════════════════════════ CPU：解析损失手算 ═══════════════════════════


def test_policy_loss_hand_computed():
    # ratio=0.5（低于 0.8 被 clip）、A=0.5：term=max(−0.25, −0.4)=−0.25
    # ratio=1（clip 区间内）、A=−1：term=max(1, 1)=1 → mean=(−0.25+1)/2
    new_log_prob = torch.tensor([math.log(0.5), 0.0])
    old_log_prob = torch.zeros(2)
    advantage = torch.tensor([0.5, -1.0])
    loss = ppo_policy_loss(new_log_prob, old_log_prob, advantage)
    assert torch.allclose(
        loss, oracle_ppo_policy_loss(new_log_prob, old_log_prob, advantage), atol=1e-7
    )
    assert abs(float(loss) - (-0.25 + 1.0) / 2.0) < 1e-6
    # 第三例：ratio=1.5（高于 1.2 被 clip）、A=0.5：max(−0.75, −0.6)=−0.6
    loss2 = ppo_policy_loss(torch.tensor([math.log(1.5)]), torch.zeros(1), torch.tensor([0.5]))
    assert abs(float(loss2) - (-0.6)) < 1e-6
    # exp 溢出立即拒绝（不 clamp、不删样本）。
    with unittest.TestCase().assertRaises(Exception):
        ppo_policy_loss(torch.tensor([1e30]), torch.zeros(1), torch.tensor([0.5]))


def test_value_clip_loss_hand_computed():
    returns = torch.tensor([1.0, 1.0])
    old_value = torch.zeros(2)
    new_value = torch.tensor([0.5, -0.5])
    # 0.5：unclipped (−0.5)²=0.25 vs clip 后 0.2→(−0.8)²=0.64 → max 0.64
    # −0.5：unclipped (−1.5)²=2.25 vs clip 后 −0.2→(−1.2)²=1.44 → max 2.25
    loss = value_clipped_loss(new_value, old_value, returns)
    assert torch.allclose(
        loss, oracle_value_clip_loss(new_value, old_value, returns), atol=1e-7
    )
    assert abs(float(loss) - (0.64 + 2.25) / 2.0) < 1e-6
    # clip 区间内（|V−oldV|<=0.2）：两支相等。
    inside = value_clipped_loss(torch.tensor([0.1]), torch.zeros(1), torch.tensor([1.0]))
    assert abs(float(inside) - 0.81) < 1e-6


def test_entropy_scale_in_total_joint_loss():
    policy = torch.tensor(1.0)
    value = torch.tensor(2.0)
    entropy = torch.tensor([3.0, 5.0])          # 批均值 4 → 贡献 −0.01·4
    envelope = torch.tensor(0.25)
    ground = torch.tensor(0.5)
    total = total_joint_loss(policy, value, entropy, envelope, ground)
    assert torch.allclose(
        total, oracle_joint_loss(policy, value, entropy, envelope, ground), atol=1e-7
    )
    assert abs(float(total) - (1.0 + 2.0 - 0.01 * 4.0 + 0.25 + 0.5)) < 1e-7


def test_envelope_six_dim_sum_scale_and_valid_rows():
    # 单位误差 6 维：每样本平方和 = 6·e²，批平均 = 6·e²（是六维取平均的 6 倍）。
    error = 0.1
    prediction = torch.zeros(3, 6)
    target = torch.full((3, 6), error)
    valid = torch.tensor([True, True, True])
    loss = envelope_sum_loss(prediction, target, valid)
    assert torch.allclose(loss, oracle_envelope_sum_loss(prediction, target, valid), atol=1e-7)
    assert abs(float(loss) - 6.0 * error * error) < 1e-7
    ratio_probe = envelope_sum_loss(prediction, target / 6.0, valid)
    assert abs(float(ratio_probe) - float(loss) / 36.0) < 1e-7
    # 有效行先索引：无效行含 NaN 也不进入算术（不乘 NaN×0）；批平均只在有效行上。
    poisoned = target.clone()
    poisoned[0] = float("nan")
    masked = torch.tensor([False, True, True])
    loss_masked = envelope_sum_loss(prediction, poisoned, masked)
    assert bool(torch.isfinite(loss_masked)) and abs(float(loss_masked) - 6 * error * error) < 1e-7
    # 有效空集：device 零。
    empty = envelope_sum_loss(prediction, target, torch.tensor([False, False, False]))
    assert float(empty) == 0.0 and empty.device == prediction.device


def test_ground_frame_mean_and_bucket_denominators():
    m = 6
    heights = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    prediction = torch.zeros(m)
    full = torch.ones(m, dtype=torch.bool)
    frame = ground_frame_mean_loss(prediction, heights, full)
    assert torch.allclose(frame, oracle_ground_frame_loss(prediction, heights, full), atol=1e-7)
    assert abs(float(frame) - torch.tensor([1.0, 4.0, 9.0, 16.0, 25.0, 36.0]).mean().item()) < 1e-6
    partial = torch.tensor([True, False, True, False, True, False])
    frame_partial = ground_frame_mean_loss(prediction, heights, partial)
    assert abs(float(frame_partial) - (1.0 + 9.0 + 25.0) / 3.0) < 1e-6

    # 桶：L_H,j = (4/|S|)·Σ_{s∈bucket_j} l_s，不除以桶局部帧数；四桶平均 ==
    # 全 S 帧等权平均（与各帧有效位置数无关）。
    frames_pred = torch.stack([torch.zeros(m) for _ in range(5)])
    frames_heights = torch.stack([
        torch.tensor([0.3, -0.2, 0.5, 0.0, 0.0, 0.0]),
        torch.tensor([0.1, 0.2, 0.0, 0.0, 0.0, 0.0]),
        torch.tensor([0.7, 0.0, 0.0, 0.0, 0.0, 0.0]),
        torch.tensor([-0.4, 0.9, 0.2, 0.6, 0.0, 0.0]),
        torch.tensor([0.5, -0.8, 0.0, 0.0, 0.0, 0.0]),
    ])
    frames_valid = torch.stack([
        torch.tensor([True, True, True, False, False, False]),
        torch.tensor([True, True, False, False, False, False]),
        torch.tensor([True, False, False, False, False, False]),
        torch.tensor([True, True, True, True, False, False]),
        torch.tensor([True, True, False, False, False, False]),
    ])
    per_frame = torch.stack([
        ground_frame_mean_loss(frames_pred[i], frames_heights[i], frames_valid[i])
        for i in range(5)
    ])
    s_size = 5
    buckets = [[0, 1], [2], [3], [4]]           # 非均衡桶（2/1/1/1）
    bucket_losses = [
        ground_bucket_loss(frames_pred[b], frames_heights[b], frames_valid[b], s_size)
        for b in buckets
    ]
    for b, loss in zip(buckets, bucket_losses):
        expected = 4.0 / s_size * float(per_frame[b].sum())
        assert abs(float(loss) - expected) < 1e-6
        assert torch.allclose(
            loss,
            oracle_ground_bucket_loss(
                frames_pred[b], frames_heights[b], frames_valid[b], s_size
            ),
            atol=1e-7,
        )
    four_bucket_average = sum(float(x) for x in bucket_losses) / 4.0
    full_s_mean = float(per_frame.mean())
    assert abs(four_bucket_average - full_s_mean) < 1e-6


def test_ground_empty_bucket_all_invalid_and_nan_indexing():
    m = 4
    # 空桶（K=0）与 |S|=0：device 零（合法跳过，不改 PPO/包络更新数）。
    empty = ground_bucket_loss(torch.zeros(0, m), torch.zeros(0, m), torch.zeros(0, m, dtype=torch.bool), 0)
    assert float(empty) == 0.0
    skipped = ground_bucket_loss(torch.zeros(3, m), torch.zeros(3, m), torch.ones(3, m, dtype=torch.bool), 0)
    assert float(skipped) == 0.0
    # 全无效帧：帧损失 device 零（该帧不入 S；这里验证解析函数的安全不变量）。
    all_invalid = ground_frame_mean_loss(torch.zeros(m), torch.zeros(m), torch.zeros(m, dtype=torch.bool))
    assert float(all_invalid) == 0.0
    # 无效槽含 NaN：先索引后平方，NaN 不进入算术（不乘 NaN×0）。
    heights = torch.tensor([0.5, float("nan"), float("nan"), float("nan")])
    valid = torch.tensor([True, False, False, False])
    loss = ground_frame_mean_loss(torch.zeros(m), heights, valid)
    assert bool(torch.isfinite(loss)) and abs(float(loss) - 0.25) < 1e-7


def test_kl_exact_formula():
    torch.manual_seed(3)
    old_mean, new_mean = torch.randn(5, JOINT_DIM), torch.randn(5, JOINT_DIM)
    old_std = 0.5 * torch.ones(5, JOINT_DIM)
    new_std = 0.7 * torch.ones(5, JOINT_DIM)
    kl = kl_divergence_old_new(old_mean, old_std, new_mean, new_std)
    assert torch.allclose(
        kl,
        oracle_diagonal_normal_kl(old_mean, old_std, new_mean, new_std),
        atol=1e-7,
    )
    # 与 torch.distributions 精确语义（rsl_rl 5.4.2 实际实现）一致。
    reference = torch.distributions.kl_divergence(
        torch.distributions.Normal(old_mean, old_std),
        torch.distributions.Normal(new_mean, new_std),
    ).sum(-1)
    assert torch.allclose(kl, reference, atol=1e-7)
    # 独立解析式：Σ[ln(σ2/σ1) + (σ1²+(μ1−μ2)²)/(2σ2²) − 1/2]。
    analytic = (
        torch.log(new_std / old_std)
        + (old_std ** 2 + (old_mean - new_mean) ** 2) / (2 * new_std ** 2)
        - 0.5
    ).sum(-1)
    assert torch.allclose(kl, analytic, atol=1e-6)
    # 相同分布 → 0；std 非正拒绝。
    zero = kl_divergence_old_new(old_mean, old_std, old_mean, old_std)
    assert float(zero.abs().max()) < 1e-7
    with unittest.TestCase().assertRaises(Exception):
        kl_divergence_old_new(old_mean, torch.zeros_like(old_std), new_mean, new_std)


def test_learning_rate_schedule_bounds():
    lr0 = 1e-3
    # 阈值边界（desired_kl=0.01：上界 0.02、下界 0.005）。
    assert adjust_learning_rate(0.03, lr0) == oracle_adjust_learning_rate(0.03, lr0) == lr0 / 1.5        # >0.02 → ÷1.5
    assert adjust_learning_rate(0.02 + 1e-12, lr0) == lr0 / 1.5
    assert adjust_learning_rate(0.02, lr0) == lr0              # 恰等上界：保持
    assert adjust_learning_rate(0.004, lr0) == lr0 * 1.5       # 0<KL<0.005 → ×1.5
    assert adjust_learning_rate(0.005, lr0) == lr0             # 恰等下界：保持
    assert adjust_learning_rate(0.0, lr0) == lr0               # 0<KL 严格：KL=0 保持
    assert adjust_learning_rate(0.01, lr0) == lr0              # 区间内保持
    assert adjust_learning_rate(-1.0, lr0) == lr0              # 负值（不应出现）不升 lr
    # 完整 clamp 边界。
    assert adjust_learning_rate(0.004, 0.009) == 0.01          # ×1.5 → 0.0135 → clamp 1e-2
    assert adjust_learning_rate(0.03, 1.2e-5) == 1e-5          # ÷1.5 → 8e-6 → clamp 1e-5
    # (1e-3, 1e-2] 带内直通（无 clamp；×1.5 的舍入用容差比较）。
    assert abs(adjust_learning_rate(0.004, 0.004) - 0.006) < 1e-12
    assert adjust_learning_rate(0.03, 0.002) == 0.002 / 1.5


def test_full_rollout_advantage_normalization_once():
    torch.manual_seed(5)
    advantage = torch.randn(48)
    normalized = normalize_rollout_advantage(advantage)
    assert torch.allclose(
        normalized, oracle_normalize_rollout_advantage(advantage), atol=1e-7
    )
    std_unbiased = torch.std(advantage, correction=1)
    expected = (advantage - advantage.mean()) / (std_unbiased + 1e-8)
    assert torch.allclose(normalized, expected, atol=1e-6)
    assert abs(float(normalized.mean())) < 1e-6
    # N<=1 拒绝；非有限拒绝。
    with unittest.TestCase().assertRaises(Exception):
        normalize_rollout_advantage(torch.zeros(1))
    bad = torch.randn(8)
    bad[3] = float("nan")
    with unittest.TestCase().assertRaises(Exception):
        normalize_rollout_advantage(bad)


def test_config_binds_confirmed_recipe():
    config = ConfinedPPOConfig()
    assert config.rollout_steps == 24 and config.num_epochs == 5 and config.num_minibatches == 4
    assert config.clip_epsilon == 0.2 and config.gamma == 0.99 and config.lam == 0.95
    assert config.value_loss_coeff == 1.0 and config.entropy_coeff == 0.01
    assert config.max_grad_norm == 1.0 and config.learning_rate == 1e-3
    assert (config.adam_beta1, config.adam_beta2) == (0.9, 0.999)
    assert config.adam_eps == 1e-8 and config.weight_decay == 0.0
    assert config.desired_kl == 0.01 and config.lr_min == 1e-5 and config.lr_max == 1e-2
    assert config.envelope_loss_coeff == 1.0 and config.ground_loss_coeff == 1.0
    summary = config.summary()
    assert summary["amp"] is False and summary["mirror"] is None and summary["std_clamp"] is None
    # 任何偏离已确认值都拒绝（不能经 config 隐藏改变配方）。
    for field, value in (
        ("rollout_steps", 12), ("num_epochs", 4), ("num_minibatches", 8),
        ("clip_epsilon", 0.3), ("gamma", 0.9), ("lam", 0.9),
        ("entropy_coeff", 0.005), ("max_grad_norm", 0.5), ("learning_rate", 5e-4),
        ("adam_eps", 1e-6), ("weight_decay", 1e-5), ("desired_kl", 0.02),
        ("lr_min", 1e-6), ("lr_max", 1e-3), ("envelope_loss_coeff", 0.5),
        ("ground_loss_coeff", 2.0),
    ):
        with unittest.TestCase().assertRaises(PPOUpdateError):
            dataclasses.replace(config, **{field: value})
    assert derive_shuffle_generator_seed(RECIPE_SEED) != derive_action_generator_seed(RECIPE_SEED)
    # The PPO state schema uses v2, while the mirror-off shuffle stream
    # retains the exact v1 seed derivation used by the accepted base recipe.
    material = (
        f"confined_locomotion.ppo/v1|optimization_shuffle|{RECIPE_SEED}"
    ).encode("utf-8")
    expected_shuffle_seed = int.from_bytes(hashlib.sha256(material).digest()[:4], "big")
    assert derive_shuffle_generator_seed(RECIPE_SEED) == expected_shuffle_seed


# ═══════════════════════════ CPU：合成 buffer 与联合更新 ═══════════════════════════


def _synthetic_two_env_sample(index, *, control_id):
    """两 env 单步样本（确定性奖励/价值梯度，使优势非退化）。"""
    batch = 2
    time_s = float(control_id) * 0.02
    rewards = [0.01 * (index + 1), -0.02 * (index + 1)]
    values = [0.5 + 0.01 * index, 0.4 + 0.01 * index]
    v_nexts = [0.6 + 0.01 * index, 0.3 + 0.01 * index]
    old_mean = torch.zeros(batch, JOINT_DIM)
    old_mean[0, 0] = 0.05 * math.sin(index)
    old_mean[1, 3] = 0.04 * math.cos(index)
    old_std = torch.full((batch, JOINT_DIM), 0.5)
    raw = old_mean + 0.1
    normal = torch.distributions.Normal(old_mean, old_std)
    observation = OnlineObservation(
        env_ids=torch.tensor([0, 1]),
        episode_ids=torch.zeros(batch, dtype=torch.long),
        step_ids=torch.tensor([control_id, control_id], dtype=torch.long),
        time_s=torch.tensor([time_s, time_s]),
        lin_vel_b=torch.zeros(batch, 3), ang_vel_b=torch.zeros(batch, 3),
        gravity_b=torch.zeros(batch, 3),
        joint_pos_rel_ref=torch.zeros(batch, JOINT_DIM),
        joint_vel=torch.zeros(batch, JOINT_DIM),
        prev_action=torch.zeros(batch, JOINT_DIM),
        command=torch.zeros(batch, 3),
    )
    envelope = EnvelopeLabel(
        env_ids=torch.tensor([0, 1]),
        episode_ids=torch.zeros(batch, dtype=torch.long),
        step_ids=torch.tensor([control_id, control_id], dtype=torch.long),
        values=torch.tensor([[0.59, 0.69, 0.59, 0.89, -0.89, 0.0]] * batch),
        valid=torch.tensor([True, True]),
    )
    return RolloutStepSample(
        control_id=control_id,
        time_s=time_s,
        env_ids=torch.tensor([0, 1]),
        episode_ids=torch.zeros(batch, dtype=torch.long),
        step_ids=torch.tensor([control_id, control_id], dtype=torch.long),
        observation=observation,
        command=observation.command.clone(),
        near_points=torch.zeros(batch, 256, 3),
        near_valid=torch.zeros(batch, 256, dtype=torch.bool),
        far_points=torch.zeros(batch, 640, 3),
        far_valid=torch.zeros(batch, 640, dtype=torch.bool),
        near_filled=torch.tensor([False, False]),
        near_scan_ids=torch.tensor([-1, -1], dtype=torch.long),
        near_times=torch.zeros(batch),
        far_point_scan_ids=torch.full((batch, 640), -1, dtype=torch.long),
        far_point_times=torch.zeros(batch, 640),
        raw_action=raw,
        old_log_prob=normal.log_prob(raw).sum(-1),
        old_mean=old_mean,
        old_std=old_std,
        old_value=torch.tensor(values),
        reward=torch.tensor(rewards),
        terminated=torch.tensor([False, False]),
        truncated=torch.tensor([False, False]),
        v_next=torch.tensor(v_nexts),
        final_episode_ids=torch.zeros(batch, dtype=torch.long),
        final_step_ids=torch.tensor([control_id + 1, control_id + 1], dtype=torch.long),
        final_control_id=control_id + 1,
        next_episode_ids=torch.zeros(batch, dtype=torch.long),
        next_step_ids=torch.tensor([control_id + 1, control_id + 1], dtype=torch.long),
        reset_applied=torch.tensor([False, False]),
        envelope_label=envelope,
    )


def _fill_buffer_with_ground_frames(buffer: TrainingRolloutBuffer) -> None:
    """5 个几何有效 SOURCE 帧 + 1 个全无效帧（S=5；近端随机有效点）。"""
    generator = torch.Generator().manual_seed(RECIPE_SEED)
    for scan in range(6):
        near_points = torch.zeros(1, 256, 3)
        near_valid = torch.zeros(1, 256, dtype=torch.bool)
        valid_positions = 0 if scan == 5 else (4 + scan % 3) * 2     # 8/10/12/8/10/0
        near_count = 24 if scan < 5 else 0
        if near_count:
            near_points[0, :near_count] = 0.2 * torch.randn(1, near_count, 3, generator=generator)
            near_valid[0, :near_count] = True
        heights = torch.zeros(GROUND_M)
        positions = torch.zeros(GROUND_M, 3)
        valid = torch.zeros(GROUND_M, dtype=torch.bool)
        if valid_positions:
            heights[:valid_positions] = torch.linspace(-0.2, 0.3, valid_positions)
            positions[:valid_positions, 0] = torch.linspace(0.1, 0.8, valid_positions)
            valid[:valid_positions] = True
        buffer.ground_frames.add_source_frame(
            env_id=scan % 2, episode_id=0, scan_id=scan,
            near_points=near_points, near_valid=near_valid,
            heights=heights, positions=positions, valid=valid,
            time_s=0.05 * scan, persistent_scan_count=scan, control_id=scan,
        )


def _filled_cpu_buffer() -> TrainingRolloutBuffer:
    buffer = TrainingRolloutBuffer(
        num_envs=RECIPE_NUM_ENVS, ground_samples=GROUND_M,
        grid_identity_hash=GRID_HASH, device=_CPU, nominal_steps=RECIPE_STEPS,
    )
    for index in range(RECIPE_STEPS):
        buffer.append(_synthetic_two_env_sample(index, control_id=100 + index))
    _fill_buffer_with_ground_frames(buffer)
    return buffer


def _cpu_trainer(adapter, seed=RECIPE_SEED):
    return ConfinedPPOTrainer(
        policy_adapter=adapter, grid_identity_hash=GRID_HASH, seed=seed,
        device=_CPU, allow_cpu=True,
    )


def _cpu_adapter(seed=RECIPE_SEED):
    network = _build_network(seed=seed)
    provider = _provider(seed=seed)
    return ConfinedPolicyAdapter(
        network=network,
        distribution_provider=provider,
        distribution_identity=provider.distribution_identity(),
    )


def test_trainer_single_adam_unique_params_and_construction_gates():
    adapter = _cpu_adapter()
    trainer = _cpu_trainer(adapter)
    # 唯一 Adam、单参数组、去重参数覆盖网络全部头 + log_std（每参数恰一次）。
    assert len(trainer.optimizer.param_groups) == 1
    group_params = trainer.optimizer.param_groups[0]["params"]
    ids = [id(p) for p in group_params]
    assert len(ids) == len(set(ids))
    assert trainer.unique_parameter_count == len(adapter.network.unique_parameters()) + 1
    assert len(group_params) == trainer.unique_parameter_count
    assert id(adapter.distribution_provider.log_std) in ids
    named_ids = {id(p) for _, p in adapter.named_parameters()}
    assert named_ids == set(ids)
    # 正式 update 必须 CUDA：默认（allow_cpu=False）在 CPU 上拒绝构造。
    with unittest.TestCase().assertRaises(PPOUpdateError):
        ConfinedPPOTrainer(
            policy_adapter=adapter, grid_identity_hash=GRID_HASH,
            seed=RECIPE_SEED, device=_CPU,
        )
    # 无 provider 的 mean-only adapter 拒绝；未建地面头的网络拒绝。
    with unittest.TestCase().assertRaises(PPOUpdateError):
        ConfinedPPOTrainer(
            policy_adapter=_mean_only_adapter(),
            grid_identity_hash=GRID_HASH, seed=RECIPE_SEED, device=_CPU, allow_cpu=True,
        )
    with unittest.TestCase().assertRaises(PPOUpdateError):
        ConfinedPPOTrainer(
            policy_adapter=ConfinedPolicyAdapter(
                network=_build_network(seed=9, ground_samples=None),
                distribution_provider=_provider(seed=9),
                distribution_identity=_provider(seed=9).distribution_identity(),
            ),
            grid_identity_hash=GRID_HASH, seed=RECIPE_SEED, device=_CPU, allow_cpu=True,
        )
    # 空/残缺 buffer 拒绝且不标 faulted（前置校验未触及优化器状态）。
    empty = TrainingRolloutBuffer(
        num_envs=RECIPE_NUM_ENVS, ground_samples=GROUND_M,
        grid_identity_hash=GRID_HASH, device=_CPU, nominal_steps=RECIPE_STEPS,
    )
    with unittest.TestCase().assertRaises(PPOUpdateError):
        trainer.update(empty)
    assert not trainer.faulted
    # grid 身份不匹配拒绝（其余身份正确绑定）。
    wrong_grid = TrainingRolloutBuffer(
        num_envs=RECIPE_NUM_ENVS, ground_samples=GROUND_M,
        grid_identity_hash="other-grid", device=_CPU, nominal_steps=RECIPE_STEPS,
    )
    for index in range(RECIPE_STEPS):
        wrong_grid.append(_synthetic_two_env_sample(index, control_id=100 + index))
    wrong_grid.bind_collection_identity(
        distribution_identity_frozen=str(adapter.identity_frozen),
        model_fingerprint=policy_state_fingerprint(adapter),
    )
    with unittest.TestCase().assertRaises(PPOUpdateError):
        trainer.update(wrong_grid)
    assert not trainer.faulted


def _mean_only_adapter():
    return ConfinedPolicyAdapter(network=_build_network(seed=13, ground_samples=GROUND_M))


def cpu_joint_update_smoke():
    """显式 CPU 解析构造（allow_cpu=True）跑完整 24 步合成 rollout 的 20 次联合更新。

    CPU 通过只作解析/契约补充，不冒充 CUDA 主门（正式 update 必须 CUDA）。
    普通 CPU 检查不运行；仅 ``--cpu-optimizer-smoke`` 显式调用此函数。
    """
    adapter = _cpu_adapter()
    trainer = _cpu_trainer(adapter)
    buffer = _filled_cpu_buffer()
    buffer.bind_collection_identity(
        distribution_identity_frozen=str(adapter.identity_frozen),
        model_fingerprint=policy_state_fingerprint(adapter),
    )
    parameters_before = [p.detach().clone() for p in adapter.unique_parameters()]
    behavior_rng_before = adapter.distribution_provider.generator_state()
    shuffle_before = trainer.shuffle_generator_state()
    advantage_before = buffer.advantage.reshape(-1).clone()
    expected_normalized = normalize_rollout_advantage(advantage_before)
    metrics = trainer.update(buffer)

    # 20 次联合 optimizer step；epoch/minibatch 索引全覆盖且不重复。
    assert metrics["optimizer_steps_this_update"] == 5 * 4 == 20
    assert trainer.optimizer_step_count == 20 and trainer.update_count == 1
    assert len(metrics["minibatches"]) == 20
    pairs = {(e["epoch"], e["minibatch"]) for e in metrics["minibatches"]}
    assert pairs == {(e, m) for e in range(5) for m in range(4)}
    # 每 epoch 的 4 个 minibatch 恰好构成 0..47 的互斥完整划分（不丢样本）。
    for epoch in range(5):
        merged = sorted(
            index for entry in metrics["minibatches"] if entry["epoch"] == epoch
            for index in entry["indices"]
        )
        assert merged == list(range(RECIPE_STEPS * RECIPE_NUM_ENVS))
    # 有限性与 LR 边界；联合梯度 norm 裁剪前后都记录且裁剪只减不增。
    for entry in metrics["minibatches"]:
        for key in ("loss_policy", "loss_value", "entropy_mean", "loss_envelope",
                    "loss_ground", "loss_total", "kl_mean", "grad_norm_pre_clip",
                    "grad_norm_post_clip"):
            assert math.isfinite(float(entry[key])), (entry["epoch"], entry["minibatch"], key)
        assert entry["kl_mean"] >= 0.0
        assert entry["grad_norm_post_clip"] <= entry["grad_norm_pre_clip"] + 1e-6
        assert entry["grad_norm_post_clip"] <= 1.0 + 1e-4
        assert 1e-5 <= entry["lr_after"] <= 1e-2
    assert 1e-5 <= metrics["learning_rate_final"] <= 1e-2
    # 完整 rollout 优势归一化只做一次：每个 minibatch 的切片与「单一完整归一化
    # 结果按 indices 取行」逐位一致（不是 minibatch 内重新归一化的产物）。
    assert metrics["advantage_normalization"]["applied_once_over_full_rollout"] is True
    for entry in metrics["minibatches"]:
        slice_expected = expected_normalized[torch.tensor(entry["indices"])]
        slice_recorded = torch.tensor(entry["normalized_advantage"])
        assert torch.allclose(slice_recorded, slice_expected, atol=1e-6)
    # 地面桶：每 epoch 4 桶并集 == S=5（互斥全覆盖，按 epoch 轮转）。
    assert metrics["ground"]["supervision_S"] == 5
    assert metrics["ground"]["total_source_frames"] == 6        # 含 1 个全无效帧
    for epoch, sizes in enumerate(metrics["ground"]["bucket_sizes_per_epoch"]):
        assert sum(sizes) == 5 and len(sizes) == 4
    executed = [e for e in metrics["minibatches"] if e["ground"]["ground_forward_executed"]]
    assert len(executed) == 20                                    # S=5 → 4 桶全非空
    # 参数真实变化（部分参数确有非零梯度并更新）。
    assert metrics["weights_changed"] is True
    changed = sum(
        1 for before, parameter in zip(parameters_before, adapter.unique_parameters())
        if not torch.equal(before, parameter.detach())
    )
    assert changed > 0
    # 行为 generator 优化期不变；shuffle generator 已推进。
    assert metrics["behavior_generator_unchanged"] is True
    assert torch.equal(behavior_rng_before, adapter.distribution_provider.generator_state())
    assert not torch.equal(shuffle_before, trainer.shuffle_generator_state())
    # 成功边界：consume + clear（buffer 为空，后续 update 显式拒绝）。
    assert buffer.steps_filled == 0 and not buffer.is_full
    with unittest.TestCase().assertRaises(PPOUpdateError):
        trainer.update(buffer)
    # state_dict 往返（算法局部状态；不声称完整环境恢复）。
    state = trainer.state_dict()
    assert state["schema"] == PPO_CONFIG_SCHEMA_VERSION
    assert state["optimizer_step_count"] == 20 and state["learning_rate"] == metrics["learning_rate_final"]


def cpu_small_S_optimizer_probe():
    """|S|=3 < 4 桶：每 epoch 恰 1 个空桶贡献 device 零，PPO/包络照常 20 次。"""
    adapter = _cpu_adapter(seed=41)
    trainer = _cpu_trainer(adapter, seed=41)
    buffer = TrainingRolloutBuffer(
        num_envs=RECIPE_NUM_ENVS, ground_samples=GROUND_M,
        grid_identity_hash=GRID_HASH, device=_CPU, nominal_steps=RECIPE_STEPS,
    )
    for index in range(RECIPE_STEPS):
        buffer.append(_synthetic_two_env_sample(index, control_id=100 + index))
    generator = torch.Generator().manual_seed(41)
    for scan in range(3):
        near_points = torch.zeros(1, 256, 3)
        near_valid = torch.zeros(1, 256, dtype=torch.bool)
        near_points[0, :16] = 0.2 * torch.randn(1, 16, 3, generator=generator)
        near_valid[0, :16] = True
        heights = torch.zeros(GROUND_M)
        valid = torch.zeros(GROUND_M, dtype=torch.bool)
        heights[:5] = torch.linspace(-0.1, 0.3, 5)
        valid[:5] = True
        buffer.ground_frames.add_source_frame(
            env_id=scan % 2, episode_id=0, scan_id=scan,
            near_points=near_points, near_valid=near_valid,
            heights=heights, positions=torch.zeros(GROUND_M, 3), valid=valid,
            time_s=0.05 * scan, persistent_scan_count=scan, control_id=scan,
        )
    buffer.bind_collection_identity(
        distribution_identity_frozen=str(adapter.identity_frozen),
        model_fingerprint=policy_state_fingerprint(adapter),
    )
    metrics = trainer.update(buffer)
    assert metrics["ground"]["supervision_S"] == 3
    for epoch, sizes in enumerate(metrics["ground"]["bucket_sizes_per_epoch"]):
        assert sum(sizes) == 3 and sorted(sizes) == [0, 1, 1, 1]
    empty_entries = [
        e for e in metrics["minibatches"]
        if e["ground"]["bucket_size"] == 0
    ]
    executed = [e for e in metrics["minibatches"] if e["ground"]["ground_forward_executed"]]
    assert len(empty_entries) == 5 and len(executed) == 15       # 每 epoch 恰 1 空桶
    for entry in empty_entries:
        assert entry["loss_ground"] == 0.0
        assert entry["ground"]["bucket_keys"] == []
    # 空桶不改 PPO/包络更新数：仍 20 次联合 step，PPO/包络损失照常记录。
    assert metrics["optimizer_steps_this_update"] == 20
    assert all(math.isfinite(e["loss_policy"]) and math.isfinite(e["loss_envelope"])
               for e in metrics["minibatches"])


def test_fault_marks_trainer_and_keeps_buffer():
    adapter = _cpu_adapter(seed=21)
    trainer = _cpu_trainer(adapter, seed=21)
    buffer = _filled_cpu_buffer()
    buffer.bind_collection_identity(
        distribution_identity_frozen=str(adapter.identity_frozen),
        model_fingerprint=policy_state_fingerprint(adapter),
    )
    old_log_prob_sample = buffer.replay(torch.arange(2)).old_log_prob.clone()
    with torch.no_grad():
        adapter.network.value_head.output.bias.fill_(float("nan"))
    with unittest.TestCase().assertRaises(PPOUpdateFault) as raised:
        trainer.update(buffer)
    assert raised.exception.stage == "preflight/finiteness"
    assert trainer.faulted and trainer.fault_stage == raised.exception.stage
    # 失败不清 buffer：数据完整、未消费；旧概率稳定不改。
    assert buffer.is_full and buffer.steps_filled == RECIPE_STEPS and not buffer.consumed
    replayed = buffer.replay(torch.arange(2))
    assert torch.equal(replayed.old_log_prob, old_log_prob_sample)
    # faulted 禁止继续 update。
    with unittest.TestCase().assertRaises(PPOUpdateFault):
        trainer.update(buffer)
    # on_fault 回调收到 (stage, error)。
    stages = []

    def on_fault(stage, error):
        stages.append((stage, type(error).__name__))

    adapter2 = _cpu_adapter(seed=22)
    trainer2 = ConfinedPPOTrainer(
        policy_adapter=adapter2, grid_identity_hash=GRID_HASH, seed=22,
        device=_CPU, allow_cpu=True, on_fault=on_fault,
    )
    buffer2 = _filled_cpu_buffer()
    buffer2.bind_collection_identity(
        distribution_identity_frozen=str(adapter2.identity_frozen),
        model_fingerprint=policy_state_fingerprint(adapter2),
    )
    with torch.no_grad():
        adapter2.network.ground_head.weight.fill_(float("inf"))
    with unittest.TestCase().assertRaises(PPOUpdateFault):
        trainer2.update(buffer2)
    assert stages and stages[0][0] == "preflight/finiteness"


def test_full_buffer_engineering_stop_still_refuses_update():
    adapter = _cpu_adapter(seed=25)
    trainer = _cpu_trainer(adapter, seed=25)
    buffer = _filled_cpu_buffer()
    buffer.bind_collection_identity(distribution_identity_frozen=str(adapter.identity_frozen), model_fingerprint=policy_state_fingerprint(adapter))
    buffer._engineering_stopped[-1] = True  # Explicit isolated final-step stop witness.
    with unittest.TestCase().assertRaisesRegex(PPOUpdateError, "工程跌倒停止"):
        trainer.update(buffer)
    assert trainer.optimizer_step_count == 0 and buffer.is_full and not buffer.consumed


def test_finite_gradient_overflow_is_fault_not_silent_zero_clip():
    adapter = _cpu_adapter(seed=26)
    trainer = _cpu_trainer(adapter, seed=26)
    buffer = _filled_cpu_buffer()
    buffer.bind_collection_identity(distribution_identity_frozen=str(adapter.identity_frozen), model_fingerprint=policy_state_fingerprint(adapter))
    handle = adapter.network.value_head.output.bias.register_hook(lambda gradient: torch.full_like(gradient, 1e30))
    try:
        with unittest.TestCase().assertRaises(PPOUpdateFault) as raised:
            trainer.update(buffer)
        assert raised.exception.stage == "epoch0/minibatch0/grad_pre_clip"
        assert trainer.optimizer_step_count == 0 and trainer.faulted
        assert buffer.is_full and not buffer.consumed
    finally:
        handle.remove()


# ═══════════════════════════ CPU：梯度路径显式隔离探针 ═══════════════════════════
# 探针均用模型副本 + 显式非零形态末层权重/偏置扰动（见 _probe_network 的声明）；
# 生产初始化（末层零权重、初始 Ê=固定标签、L_E=0）不被触碰。


def test_probe_ppo_only_keeps_morphology_cut_and_perception_live():
    network = _probe_network(seed=31)
    provider = _provider(seed=31)
    observation, near_points, near_valid, far_points, far_valid = _tiny_inputs(2, seed=3)
    output = network(observation, near_points, near_valid, far_points, far_valid)
    distribution = provider(output.action_mean)
    raw = torch.randn(2, JOINT_DIM, generator=torch.Generator().manual_seed(4))
    old_log_prob = distribution.log_prob(raw).detach()
    loss = ppo_policy_loss(distribution.log_prob(raw), old_log_prob, torch.tensor([0.3, -0.4]))
    loss.backward()
    # 形态头不被 PPO 更新（detach(Ê) 截断；非零末层下仍无梯度证明是路径截断）。
    assert _grad_all_zero(network.horizontal_head)
    assert _grad_all_zero(network.vertical_head)
    # 价值头不在 PPO-only 损失中。
    assert _grad_all_zero(network.value_head)
    # 动作头与 log_std 更新；共享感知直连路径活（近/远编码器收到梯度）。
    assert _grad_any_nonzero(network.action_head)
    assert provider.log_std.grad is not None and bool((provider.log_std.grad != 0).any())
    assert _grad_any_nonzero(network.near_encoder)
    assert _grad_any_nonzero(network.far_encoder)


def test_probe_value_only_keeps_morphology_cut_and_perception_live():
    network = _probe_network(seed=32)
    observation, near_points, near_valid, far_points, far_valid = _tiny_inputs(2, seed=5)
    output = network(observation, near_points, near_valid, far_points, far_valid)
    new_value = output.value.squeeze(-1)
    loss = value_clipped_loss(new_value, torch.zeros(2), torch.tensor([0.7, -0.2]))
    loss.backward()
    assert _grad_all_zero(network.horizontal_head)
    assert _grad_all_zero(network.vertical_head)
    assert _grad_all_zero(network.action_head)          # value-only 不含动作头
    assert _grad_any_nonzero(network.value_head)
    assert _grad_any_nonzero(network.near_encoder)
    assert _grad_any_nonzero(network.far_encoder)


def test_probe_ground_only_updates_near_encoder_and_ground_head():
    network = _probe_network(seed=33)
    near_points = torch.zeros(3, 256, 3)
    near_valid = torch.zeros(3, 256, dtype=torch.bool)
    generator = torch.Generator().manual_seed(6)
    near_points[:, :32] = 0.3 * torch.randn(3, 32, 3, generator=generator)
    near_valid[:, :32] = True
    feature = network.near_encoder(near_points, near_valid)
    predictions = network.ground_head(feature)
    heights = torch.zeros(3, GROUND_M)
    valid = torch.zeros(3, GROUND_M, dtype=torch.bool)
    for row in range(3):
        count = 4 + 2 * row
        heights[row, :count] = torch.linspace(-0.1, 0.4, count)
        valid[row, :count] = True
    loss = ground_bucket_loss(predictions, heights, valid, 3)
    loss.backward()
    assert _grad_any_nonzero(network.ground_head)
    assert _grad_any_nonzero(network.near_encoder)
    # ground-only：远编码器/形态/动作/价值头全无梯度。
    assert _grad_all_zero(network.far_encoder)
    assert _grad_all_zero(network.horizontal_head)
    assert _grad_all_zero(network.vertical_head)
    assert _grad_all_zero(network.action_head)
    assert _grad_all_zero(network.value_head)


def test_probe_envelope_nonzero_error_updates_morphology_and_perception():
    network = _probe_network(seed=34)
    observation, near_points, near_valid, far_points, far_valid = _tiny_inputs(2, seed=7)
    output = network(observation, near_points, near_valid, far_points, far_valid)
    # 受控非零误差：目标 = 初始输出 + 0.05（不伪造训练标签，仅探针副本上验证梯度）。
    target = output.envelope.detach() + 0.05
    loss = envelope_sum_loss(output.envelope, target, torch.tensor([True, True]))
    loss.backward()
    assert _grad_any_nonzero(network.horizontal_head)
    assert _grad_any_nonzero(network.vertical_head)
    # 共享感知经形态头获得梯度（近端经水平头、远端经垂向头）。
    assert _grad_any_nonzero(network.near_encoder)
    assert _grad_any_nonzero(network.far_encoder)
    # 动作/价值/地面头不在此辅助路径中。
    assert _grad_all_zero(network.action_head)
    assert _grad_all_zero(network.value_head)
    assert _grad_all_zero(network.ground_head)


# ═══════════════════════════ 显式 CUDA 主门 ═══════════════════════════


def _cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _cuda_observer(adapter, detail):
    """按阶段即时 clone 所需数组/梯度/state（首/末 minibatch 全量，供独立复算）。"""
    wanted = {(0, 0), (4, 3)}
    param_names = {id(p): name for name, p in adapter.named_parameters()}

    def observer(stage, payload):
        key = (payload.get("epoch"), payload.get("minibatch"))
        if key not in wanted:
            return
        bucket = detail.setdefault(f"epoch{key[0]}_minibatch{key[1]}", {})
        if stage == "minibatch_forward":
            output = payload["network_output"]
            batch = payload["batch"]
            bucket["forward"] = {
                "action_mean": output.action_mean.detach().cpu().clone(),
                "value": output.value.detach().cpu().clone(),
                "envelope": output.envelope.detach().cpu().clone(),
                "ground_pred": (
                    None if output.ground_pred is None
                    else output.ground_pred.detach().cpu().clone()
                ),
                "raw_action": batch.raw_action.detach().cpu().clone(),
                "old_log_prob": batch.old_log_prob.detach().cpu().clone(),
                "old_mean": batch.old_mean.detach().cpu().clone(),
                "old_std": batch.old_std.detach().cpu().clone(),
                "old_value": batch.old_value.detach().cpu().clone(),
                "returns": batch.returns.detach().cpu().clone(),
                "advantage": batch.advantage.detach().cpu().clone(),
                "envelope_values": batch.envelope_values.detach().cpu().clone(),
                "envelope_valid": batch.envelope_valid.detach().cpu().clone(),
                "loss_policy": payload["loss_policy"].detach().cpu().clone(),
                "loss_value": payload["loss_value"].detach().cpu().clone(),
                "loss_envelope": payload["loss_envelope"].detach().cpu().clone(),
                "loss_ground": payload["loss_ground"].detach().cpu().clone(),
                "loss_total": payload["loss_total"].detach().cpu().clone(),
                "kl_event": payload["kl_event"].detach().cpu().clone(),
            }
        elif stage == "grad_pre_clip":
            bucket["grad_pre_clip"] = {
                param_names.get(id(p), f"param_{i}"): p.grad.detach().cpu().clone()
                for i, p in enumerate(payload["parameters"]) if p.grad is not None
            }
        elif stage == "grad_post_clip":
            bucket["grad_post_clip"] = {
                param_names.get(id(p), f"param_{i}"): p.grad.detach().cpu().clone()
                for i, p in enumerate(payload["parameters"]) if p.grad is not None
            }
        elif stage == "adam_step_done":
            bucket["adam_state_after"] = _cpu_tree(payload["optimizer"].state_dict())
            bucket["learning_rate_after"] = float(payload["learning_rate"])
            bucket["parameters_after"] = {
                param_names.get(id(p), f"param_{i}"): p.detach().cpu().clone()
                for i, p in enumerate(payload["parameters"])
            }

    return observer


def run_cuda_recipe(output: Path, *, device="cuda:0") -> dict:
    """写 recipe 在先，然后执行自然 24 步 → 20 次联合更新 → 1 步重新采集。"""
    output.mkdir(parents=True, exist_ok=False)
    recipe = {
        "task": "El-4090-Confined-Locomotion-Flat",
        "device": device,
        "num_envs": RECIPE_NUM_ENVS,
        "seed": RECIPE_SEED,
        "lidar_rays": 86400,
        "near_points": 256,
        "far_window": "64x10",
        "grid": "M12 temporary explicit engineering grid (geometry_only_flat_training)",
        "mode": "engineering (natural; a fall saves and stops honestly → NOT PASSED)",
        "collection_steps": RECIPE_STEPS,
        "update": {
            "algorithm": "single-Adam joint PPO + envelope/ground auxiliaries",
            "epochs": 5, "minibatches": 4, "joint_steps_per_update": 20,
            "clip": 0.2, "value_clip": 0.2, "entropy": 0.01,
            "gamma": 0.99, "lam": 0.95,
            "adam": {"lr": 1e-3, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0},
            "kl_schedule": {"desired": 0.01, "down_above": 0.02, "up_below": 0.005,
                            "factor": 1.5, "clamp": [1e-5, 1e-2]},
            "auxiliary": {"envelope": 1.0, "ground": 1.0},
            "precision": "CUDA FP32 (no AMP/mirror/std-clamp)",
        },
        "post_update_recollect_steps": 1,
        "gradient_probes": (
            "CPU probes in this file use declared model copies with non-zero morphology "
            "terminal weights; Codex may re-run the same probes on GPU"
        ),
        "claim": (
            "Finite engineering witness only: one collect -> 20 joint updates -> one "
            "re-collection step. No learning/standing/walking claim; CPU passing "
            "never claims the GPU path."
        ),
    }
    (output / "recipe.json").write_text(json.dumps(recipe, indent=2))

    natural = output / "natural24_update"
    natural.mkdir(parents=True, exist_ok=True)
    checks: dict = {}
    env = None
    runtime = None
    try:
        env, _wrapper, runtime = build_runtime(
            natural, mode="engineering", device=device, ground_grid=m12_training_grid()
        )
        collector, adapter = build_collector(runtime, device=device, steps=RECIPE_STEPS)
        assert collector.grid_identity_hash and collector.ground_samples == GROUND_M

        # ── 自然 24 步采集（跌倒/故障保存后停止 → 如实未通过，不用夹具凑主门）。
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
        checks["collection"] = {
            "complete": result.complete,
            "completed_steps": result.completed_steps,
            "stopped_reason": result.stopped_reason,
            "runtime_stopped": bool(getattr(runtime, "_stopped", False)),
        }
        if not result.complete or bool(getattr(runtime, "_stopped", False)):
            # 第 24 步恰好收满但 engineering_stopped：同样不得随后优化。
            checks["gate_passed"] = False
            checks["reason"] = (
                "natural collection stopped (fall/fault): saved and stopped honestly; "
                "no update attempted"
            )
            torch.save({"partial_buffer": result.buffer}, natural / "evidence.pt")
            return checks
        buffer = result.buffer
        pre_fingerprint = policy_state_fingerprint(adapter)
        pre_provider_rng = adapter.distribution_provider.generator_state().cpu()
        # 更新前完整 buffer + 模型（供独立复算）。
        torch.save(
            {
                "buffer": buffer,
                "model_state": _cpu_tree(adapter.state_dict()),
                "model_fingerprint": pre_fingerprint,
                "provider_generator_state": pre_provider_rng.clone(),
            },
            natural / "pre_update.pt",
        )

        # ── 20 次联合更新（正式路径：默认 allow_cpu=False → 必须 CUDA）。
        detail: dict = {}
        trainer = ConfinedPPOTrainer(
            policy_adapter=adapter,
            grid_identity_hash=collector.grid_identity_hash,
            seed=RECIPE_SEED,
            device=torch.device(device),
            observer=_cuda_observer(adapter, detail),
        )
        control_steps_before_update = int(runtime.control_steps_taken)
        metrics = trainer.update(buffer)
        (natural / "metrics.json").write_text(json.dumps(metrics, indent=2))

        # ── 主门检查。
        checks["update"] = {
            "joint_steps": metrics["optimizer_steps_this_update"],
            "optimizer_step_count": trainer.optimizer_step_count,
            "update_count": trainer.update_count,
            "unique_parameter_count": trainer.unique_parameter_count,
            "adam_param_group_count": len(trainer.optimizer.param_groups),
            "learning_rate_final": metrics["learning_rate_final"],
            "weights_changed": metrics["weights_changed"],
            "behavior_generator_unchanged": metrics["behavior_generator_unchanged"],
        }
        assert metrics["optimizer_steps_this_update"] == 20
        assert trainer.optimizer_step_count == 20 and trainer.update_count == 1
        assert len(trainer.optimizer.param_groups) == 1
        assert trainer.unique_parameter_count == len(adapter.network.unique_parameters()) + 1
        assert metrics["weights_changed"] is True
        assert metrics["behavior_generator_unchanged"] is True
        assert torch.equal(pre_provider_rng, adapter.distribution_provider.generator_state().cpu())
        assert policy_state_fingerprint(adapter) != pre_fingerprint
        assert 1e-5 <= metrics["learning_rate_final"] <= 1e-2
        # 优化期间不触碰环境/传感（纯 rl 算法；runtime 步数不变、未停止）。
        assert int(runtime.control_steps_taken) == control_steps_before_update
        assert not bool(getattr(runtime, "_stopped", False))
        # 每 epoch minibatch 划分互斥完整；损失/梯度/KL 全有限。
        n_total = RECIPE_STEPS * RECIPE_NUM_ENVS
        for epoch in range(5):
            merged = sorted(
                index for entry in metrics["minibatches"] if entry["epoch"] == epoch
                for index in entry["indices"]
            )
            assert merged == list(range(n_total))
        for entry in metrics["minibatches"]:
            for key in ("loss_policy", "loss_value", "entropy_mean", "loss_envelope",
                        "loss_ground", "loss_total", "kl_mean",
                        "grad_norm_pre_clip", "grad_norm_post_clip"):
                assert math.isfinite(float(entry[key]))
            assert entry["grad_norm_post_clip"] <= entry["grad_norm_pre_clip"] + 1e-6
        # 成功边界：buffer 已 consume+clear（空 buffer 再 update 显式拒绝）。
        assert buffer.steps_filled == 0 and not buffer.is_full
        with unittest.TestCase().assertRaises(PPOUpdateError):
            trainer.update(buffer)

        # ── 更新后模型/Adam/LR/RNG 与按阶段诊断原始数据（独立复算用）。
        torch.save(
            {
                "model_state": _cpu_tree(adapter.state_dict()),
                "model_fingerprint": policy_state_fingerprint(adapter),
                "adam_state": _cpu_tree(trainer.optimizer.state_dict()),
                "learning_rate": trainer.learning_rate,
                "shuffle_generator_state": trainer.shuffle_generator_state().cpu(),
                "provider_generator_state": adapter.distribution_provider.generator_state().cpu(),
                "stage_detail_first_and_last_minibatch": detail,
            },
            natural / "post_update.pt",
        )

        # ── 更新后显式 1 步重新采集（证明 post-update 重跑；小批量即可）。
        fresh = collector.make_buffer()
        fresh.bind_collection_identity(
            distribution_identity_frozen=str(adapter.identity_frozen),
            model_fingerprint=policy_state_fingerprint(adapter),
        )
        transition = collector.collect_step(fresh)
        checks["post_update_recollect"] = {
            "steps": 1,
            "buffer_steps_filled": fresh.steps_filled,
            "engineering_stopped": bool(transition.engineering_stopped),
            "control_steps_taken": int(runtime.control_steps_taken),
        }
        assert not transition.engineering_stopped, "postupdate自然跌倒：已保存停止，完整门未通过"
        assert fresh.steps_filled == 1
        assert int(runtime.control_steps_taken) == control_steps_before_update + 1
        torch.save(
            {
                "raw_action": transition.context_record["raw_action"].cpu(),
                "clipped_action": transition.context_record["clipped_action"].cpu(),
                "reward": transition.reward.cpu(),
                "terminated": transition.terminated.cpu(),
                "truncated": transition.truncated.cpu(),
            },
            natural / "post_update_recollect.pt",
        )
        checks["gate_passed"] = True
    finally:
        if runtime is not None:
            runtime.close()
        if env is not None:
            env.close()
    return checks


def _require_cuda() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("this gate requires real CUDA (CPU passing claims nothing)")
    return "cuda:0"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    device = args.device or _require_cuda()
    start = time.monotonic()
    error_text = None
    try:
        checks = run_cuda_recipe(args.output, device=device)
    except BaseException:
        error_text = traceback.format_exc()
        checks = {"gate_passed": False, "error": error_text}
    summary = {
        "checks": checks,
        "elapsed_s": time.monotonic() - start,
        "cuda_available": torch.cuda.is_available(),
        "peak_cuda_allocated": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        "peak_cuda_reserved": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    if error_text is not None or not checks.get("gate_passed", False):
        raise RuntimeError("CUDA update gate NOT PASSED (honest stop; see summary.json)")
    print("CUDA update gate completed: 24-step collect -> 20 joint updates -> 1-step re-collect")


def test_cuda_recipe_explicit_opt_in():
    """Opt-in via CONFINED_LOCOMOTION_CUDA_UPDATE_OUTPUT pointing at a new directory."""
    value = os.environ.get("CONFINED_LOCOMOTION_CUDA_UPDATE_OUTPUT")
    if not value:
        raise unittest.SkipTest(
            "CUDA gate is opt-in: set CONFINED_LOCOMOTION_CUDA_UPDATE_OUTPUT=<new dir> (CPU tests "
            "passing never claims the GPU path)"
        )
    checks = run_cuda_recipe(Path(value), device=_require_cuda())
    assert checks.get("gate_passed", False), "natural stop / fault: NOT PASSED (honest)"


def run_cpu_checks() -> int:
    """Run pure CPU contracts; optimizer work has a separate explicit CLI flag."""
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


def run_cpu_optimizer_smokes() -> None:
    """Run optimizer probes only when explicitly selected on the command line."""
    cpu_joint_update_smoke()
    cpu_small_S_optimizer_probe()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--cpu-optimizer-smoke":
        run_cpu_optimizer_smokes()
    elif len(sys.argv) > 1 and sys.argv[1] == "--cpu":
        sys.exit(run_cpu_checks())
    else:
        main()
