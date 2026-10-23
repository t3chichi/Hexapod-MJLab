"""Independent reference equations used by feature tests.

These functions intentionally depend only on Python math and PyTorch. They do
not import the production loss, policy, or mirror modules, so comparisons can
catch a shared implementation mistake.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch


def diagonal_normal_log_prob(value: torch.Tensor, mean: torch.Tensor,
                             std: torch.Tensor) -> torch.Tensor:
    """Event-summed log probability for an independent diagonal Normal."""
    per_coordinate = -0.5 * ((value - mean) / std).square() - std.log() \
        - 0.5 * math.log(2.0 * math.pi)
    return per_coordinate.sum(dim=-1)


def diagonal_normal_entropy(std: torch.Tensor) -> torch.Tensor:
    """Event-summed differential entropy for an independent diagonal Normal."""
    per_coordinate = std.log() + 0.5 * (1.0 + math.log(2.0 * math.pi))
    return per_coordinate.sum(dim=-1)


def diagonal_normal_kl(old_mean: torch.Tensor, old_std: torch.Tensor,
                       new_mean: torch.Tensor, new_std: torch.Tensor) -> torch.Tensor:
    """Analytic KL(old || new), summed over the event dimension."""
    per_coordinate = torch.log(new_std / old_std) + (
        old_std.square() + (old_mean - new_mean).square()
    ) / (2.0 * new_std.square()) - 0.5
    return per_coordinate.sum(dim=-1)


def generalized_advantage(reward: torch.Tensor, value: torch.Tensor,
                          next_value: torch.Tensor, terminated: torch.Tensor,
                          truncated: torch.Tensor, *, gamma: float = 0.99,
                          lam: float = 0.95) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE on (time, environment) tensors with separate masks.

    Termination stops both bootstrap and recursion. Timeout keeps the natural
    next-value bootstrap but stops recursion into the reset episode.
    """
    horizon = int(reward.shape[0])
    advantage = torch.zeros_like(reward)
    running = torch.zeros_like(reward[0])
    for step in range(horizon - 1, -1, -1):
        bootstrap = (~terminated[step]).to(value.dtype)
        continue_trace = (~(terminated[step] | truncated[step])).to(value.dtype)
        delta = reward[step] + gamma * bootstrap * next_value[step] - value[step]
        running = delta + gamma * lam * continue_trace * running
        advantage[step] = running
    return advantage, advantage + value


def ppo_policy_loss(new_log_prob: torch.Tensor, old_log_prob: torch.Tensor,
                    advantage: torch.Tensor, *, clip_epsilon: float = 0.2) -> torch.Tensor:
    """Clipped PPO surrogate loss, with an independent hand-written equation."""
    ratio = torch.exp(new_log_prob - old_log_prob)
    plain = ratio * advantage
    clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantage
    return -torch.minimum(plain, clipped).mean()


def clipped_value_loss(new_value: torch.Tensor, old_value: torch.Tensor,
                       returns: torch.Tensor, *, clip_epsilon: float = 0.2) -> torch.Tensor:
    """PPO clipped value loss: mean of the per-row larger square (coefficient 1)."""
    clipped = old_value + (new_value - old_value).clamp(-clip_epsilon, clip_epsilon)
    plain_error = (new_value - returns).square()
    clipped_error = (clipped - returns).square()
    return torch.maximum(plain_error, clipped_error).mean()


def joint_loss(policy: torch.Tensor, value: torch.Tensor, entropy: torch.Tensor,
               envelope: torch.Tensor, ground: torch.Tensor, *,
               entropy_coefficient: float = 0.01,
               envelope_coefficient: float = 1.0,
               ground_coefficient: float = 1.0) -> torch.Tensor:
    """Joint PPO objective with the confirmed sign and scalar coefficients."""
    return policy + value - entropy_coefficient * entropy.mean() \
        + envelope_coefficient * envelope + ground_coefficient * ground


def envelope_sum_loss(prediction: torch.Tensor, target: torch.Tensor,
                      valid: torch.Tensor) -> torch.Tensor:
    """Six-coordinate squared error summed per sample, then averaged valid rows."""
    if not bool(valid.any()):
        return prediction.new_zeros(())
    difference = prediction[valid] - target[valid]
    return difference.square().sum(dim=-1).mean()


def ground_frame_loss(prediction: torch.Tensor, target: torch.Tensor,
                      valid: torch.Tensor) -> torch.Tensor:
    """One frame: average squared height error over valid grid points."""
    if not bool(valid.any()):
        return prediction.new_zeros(())
    difference = prediction[valid] - target[valid]
    return difference.square().mean()


def ground_bucket_loss(prediction: torch.Tensor, target: torch.Tensor,
                       valid: torch.Tensor, supervision_size: int, *,
                       bucket_count: int = 4) -> torch.Tensor:
    """Sum frame means in one bucket, weighted by the full source-frame count."""
    if prediction.shape[0] == 0 or supervision_size == 0:
        return prediction.new_zeros(())
    per_frame = [
        ground_frame_loss(prediction[i], target[i], valid[i])
        for i in range(int(prediction.shape[0]))
    ]
    return torch.stack(per_frame).sum() * (float(bucket_count) / float(supervision_size))


def normalize_advantage(advantage: torch.Tensor, *, epsilon: float = 1e-8) -> torch.Tensor:
    """Flattened-rollout normalization with sample standard deviation."""
    flat = advantage.reshape(-1)
    return (flat - flat.mean()) / (flat.std(correction=1) + epsilon)


def kl_scheduled_learning_rate(kl_mean: float, learning_rate: float, *,
                               desired_kl: float = 0.01,
                               minimum: float = 1e-5,
                               maximum: float = 1e-2) -> float:
    """One KL schedule decision, including its final learning-rate clamp."""
    next_rate = learning_rate
    if kl_mean > 2.0 * desired_kl:
        next_rate /= 1.5
    elif 0.0 < kl_mean < 0.5 * desired_kl:
        next_rate *= 1.5
    return min(maximum, max(minimum, next_rate))


def mirror_mean_mse(mirrored_mean: torch.Tensor, original_mean: torch.Tensor,
                    permutation_indices: Sequence[int]) -> torch.Tensor:
    """Batch×action mean square against a detached, caller-specified target map."""
    index = torch.as_tensor(permutation_indices, dtype=torch.long, device=original_mean.device)
    target = original_mean.detach().index_select(-1, index)
    return (mirrored_mean - target).square().mean()

