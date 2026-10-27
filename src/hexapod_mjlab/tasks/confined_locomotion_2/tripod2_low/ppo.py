"""Legacy tripod2-low PPO semantics with the frozen two-GRU replay contract.

Only the optimizer recipe differs from the existing temporal trainer. Point-cloud
preparation, episode-aware recurrent replay, and envelope supervision are shared.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from hexapod_mjlab.rl.confined_locomotion.distribution import (
    DiagonalNormalDistribution, DistributionContractError,
    IndependentDiagonalNormalProvider,
)
from hexapod_mjlab.rl.confined_locomotion.losses import (
    envelope_sum_loss, normalize_rollout_advantage, ppo_policy_loss, value_clipped_loss,
)
from hexapod_mjlab.rl.confined_locomotion.symmetry import (
    mirror_action_distribution, mirror_envelope_values, mirror_mean_mse,
)
from hexapod_mjlab.rl.confined_locomotion_2.ppo import (
    TemporalPPOTrainer, _require_finite_tensors,
)
from hexapod_mjlab.rl.confined_locomotion_2.training_rollout import (
    SequenceRolloutBuffer, prepare_sequence_replay,
)


@dataclass(frozen=True)
class LegacyPPOConfig:
    rollout_steps: int = 24
    epochs: int = 5
    minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coeff: float = 1.0
    entropy_coeff: float = 0.01
    envelope_coeff: float = 1.0
    learning_rate: float = 1e-3
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    desired_kl: float = 0.01
    lr_min: float = 1e-5
    lr_max: float = 1e-2
    symmetry_mean_coeff: float = 0.6


class _DirectStdNormal(DiagonalNormalDistribution):
    def __init__(self, *, mean, std, generator):
        super().__init__(mean=mean, log_std=std.log(), generator=generator)
        self._std_parameter = std

    @property
    def std(self):
        return self._std_parameter.expand_as(self._mean)

    stddev = std


class LegacyNormalProvider(IndependentDiagonalNormalProvider):
    """Original policy's directly optimized std, initialized to one, without clipping."""

    def __init__(self, *, device=None, generator_seed: int):
        # Reuse the existing explicit random-stream state interface, but do not
        # retain its log_std parameter or startup value.
        super().__init__(device=device, generator_seed=generator_seed)
        resolved = self.log_std.device
        del self.log_std
        self.std = nn.Parameter(torch.ones(18, dtype=torch.float32, device=resolved))

    def forward(self, action_mean):
        if (not isinstance(action_mean, torch.Tensor) or action_mean.ndim != 2
                or action_mean.shape[1] != 18 or action_mean.shape[0] < 1
                or action_mean.dtype != torch.float32 or action_mean.device != self.std.device):
            raise DistributionContractError("action_mean must be (B>=1,18) float32 on provider device")
        self._require_generator_device(action_mean.device)
        if (not bool(torch.isfinite(action_mean).all())
                or not bool(torch.isfinite(self.std).all()) or bool((self.std <= 0).any())):
            raise DistributionContractError("legacy Normal mean/std must be finite and std positive")
        return _DirectStdNormal(mean=action_mean, std=self.std, generator=self._generator)

    def distribution_identity(self):
        return {
            "schema": "confined_locomotion_flat_tripod2_low.action_normal/v1",
            "family": "independent_diagonal_normal", "action_dim": 18,
            "initial_std": 1.0, "parameterization": "direct_std",
            "std_observation_dependent": False, "sample_generator_stream": "policy_action",
        }


def legacy_kl(old_mean, old_std, new_mean, new_std):
    """The historical epsilon is inside log, including at identical distributions."""
    return (torch.log(new_std / old_std + 1.0e-5)
            + (old_std.square() + (old_mean - new_mean).square())
            / (2.0 * new_std.square()) - 0.5).sum(-1)


class LegacyPPOTrainer(TemporalPPOTrainer):
    """Keep sequence replay, migrate timeout, shuffle, and original-only KL behavior."""

    def __init__(self, *, config=None, **kwargs):
        super().__init__(config=config or LegacyPPOConfig(), **kwargs)

    def state_dict(self):
        state = super().state_dict()
        state["schema"] = "confined_locomotion_flat_tripod2_low.ppo/v1"
        return state

    def load_state_dict(self, state):
        if state.get("schema") != "confined_locomotion_flat_tripod2_low.ppo/v1":
            raise ValueError("incompatible legacy tripod2-low PPO state")
        super().load_state_dict({**state, "schema": "confined_locomotion_2.temporal_ppo/v1"})

    def compute_gae(self, buffer):
        """Old process_env_step adds gamma*V(s_t) to timeout rewards before GAE."""
        buffer._require_full()
        values = torch.stack([step.old_value for step in buffer.steps])
        advantages = torch.zeros_like(values)
        carry = torch.zeros_like(values[0])
        for t in range(len(buffer.steps) - 1, -1, -1):
            step = buffer.steps[t]
            not_done = (~(step.terminated | step.truncated)).to(values.dtype)
            reward = step.reward + self.config.gamma * values[t] * step.truncated.to(values.dtype)
            # Unlike per-transition next_value, the old storage reads the next
            # action-time value (and only the rollout's final bootstrap value).
            next_value = values[t + 1] if t + 1 < len(buffer.steps) else step.next_value
            delta = reward + self.config.gamma * not_done * next_value - values[t]
            carry = delta + self.config.gamma * self.config.gae_lambda * not_done * carry
            advantages[t] = carry
        return advantages.detach(), (advantages + values).detach()

    def _minibatch_indices(self, total):
        if total % self.config.minibatches:
            raise ValueError("T*B must divide evenly into minibatches")
        # Legacy RolloutStorage samples this once, outside the epoch loop.
        permutation = torch.randperm(total, device=self.device, generator=self.shuffle_generator)
        chunks = permutation.chunk(self.config.minibatches)
        for epoch in range(self.config.epochs):
            for minibatch, selected in enumerate(chunks):
                yield epoch, minibatch, selected

    def _adapt_learning_rate(self, kl):
        lr = self.optimizer.param_groups[0]["lr"]
        if kl > self.config.desired_kl * 2.0:
            lr = max(self.config.lr_min, lr / 1.5)
        elif 0.0 < kl < self.config.desired_kl / 2.0:
            lr = min(self.config.lr_max, lr * 1.5)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def _update_impl(self, buffer: SequenceRolloutBuffer):
        buffer._require_full()
        if buffer.device != self.device:
            raise ValueError("buffer and trainer devices differ")
        if self.symmetry_enabled and any(
                step.prepared is None or step.mirror_hidden_before is None for step in buffer.steps):
            raise ValueError("symmetry requires prepared inputs and shadow hidden states")
        replay = prepare_sequence_replay(buffer.steps, network=self.network)
        mirrored = prepare_sequence_replay(
            buffer.steps, network=self.network, mirrored=True, joint_permutation=self.joint_permutation,
            mount_quat_xyzw=self.mount_quat_xyzw) if self.symmetry_enabled else None
        advantages, returns = self.compute_gae(buffer)
        advantages = normalize_rollout_advantage(advantages.reshape(-1)).reshape_as(advantages)
        time, batch = advantages.shape
        total = time * batch
        stack = lambda name: torch.stack([getattr(step, name) for step in buffer.steps])
        actions, old_lp = stack("action").reshape(total, 18), stack("old_log_prob")
        old_means, old_stds, old_values = stack("old_mean"), stack("old_std"), stack("old_value")
        targets = torch.stack([step.envelope_label.values for step in buffer.steps])
        valid = torch.stack([step.envelope_label.valid for step in buffer.steps])
        if self.symmetry_enabled:
            mirror_actions = mirror_action_distribution(
                old_means.reshape(total, 18), old_stds.reshape(total, 18), actions,
                self.joint_permutation).raw_action
            mirror_targets = mirror_envelope_values(targets.reshape(total, 6)).reshape_as(targets)
        metrics = {"steps": time, "num_envs": batch}
        diagnostics = {}
        for epoch, minibatch, selected in self._minibatch_indices(total):
            mask = torch.zeros(total, device=self.device, dtype=torch.bool)
            mask[selected] = True
            mask = mask.reshape(time, batch)
            outputs = self._replay(replay)
            means = torch.stack([output.action_mean for output in outputs])
            values = torch.stack([output.value.squeeze(-1) for output in outputs])
            envelopes = torch.stack([output.envelope for output in outputs])
            dist = self.distribution_provider(means.reshape(total, 18))
            lp = dist.log_prob(actions).reshape(time, batch)
            entropy = dist.entropy().reshape(time, batch)[mask].mean()
            stds = dist.std.reshape_as(means)
            # The adaptive schedule sees only original observations and executes
            # before this minibatch's optimizer step, exactly as in old PPO.
            kl = legacy_kl(old_means[mask], old_stds[mask],
                           means[mask].detach(), stds[mask].detach()).mean()
            _require_finite_tensors([kl], "non-finite legacy PPO KL")
            lr = self._adapt_learning_rate(float(kl))
            policy = ppo_policy_loss(lp[mask], old_lp[mask], advantages[mask],
                                     clip_epsilon=self.config.clip_epsilon)
            value = value_clipped_loss(values[mask], old_values[mask], returns[mask],
                                       clip_epsilon=self.config.clip_epsilon)
            envelope = envelope_sum_loss(envelopes[mask], targets[mask], valid[mask])
            symmetry = means.new_zeros(())
            if mirrored is not None:
                mirror_outputs = self._replay(mirrored)
                mirror_means = torch.stack([output.action_mean for output in mirror_outputs])
                mirror_values = torch.stack([output.value.squeeze(-1) for output in mirror_outputs])
                mirror_envelopes = torch.stack([output.envelope for output in mirror_outputs])
                mirror_dist = self.distribution_provider(mirror_means.reshape(total, 18))
                mirror_lp = mirror_dist.log_prob(mirror_actions).reshape(time, batch)
                policy = .5 * (policy + ppo_policy_loss(
                    mirror_lp[mask], old_lp[mask], advantages[mask], clip_epsilon=self.config.clip_epsilon))
                value = .5 * (value + value_clipped_loss(
                    mirror_values[mask], old_values[mask], returns[mask], clip_epsilon=self.config.clip_epsilon))
                envelope = .5 * (envelope + envelope_sum_loss(
                    mirror_envelopes[mask], mirror_targets[mask], valid[mask]))
                symmetry = mirror_mean_mse(mirror_means[mask], means[mask], self.joint_permutation)
            loss = (policy + self.config.value_coeff * value - self.config.entropy_coeff * entropy
                    + self.config.envelope_coeff * envelope + self.config.symmetry_mean_coeff * symmetry)
            _require_finite_tensors([loss], "non-finite PPO loss")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            params = [p for group in self.optimizer.param_groups for p in group["params"] if p.grad is not None]
            _require_finite_tensors((p.grad for p in params), "non-finite PPO gradient")
            grad_norm = torch.nn.utils.clip_grad_norm_(params, self.config.max_grad_norm)
            _require_finite_tensors([grad_norm, *(p.grad for p in params)], "non-finite PPO gradient after clipping")
            self.optimizer.step()
            self.optimizer_step_count += 1
            _require_finite_tensors(params, "non-finite PPO parameter after optimizer step")
            _require_finite_tensors(
                (item for state in self.optimizer.state.values() for item in state.values()
                 if isinstance(item, torch.Tensor)), "non-finite Adam state after optimizer step")
            prefix = f"epoch_{epoch}_minibatch_{minibatch}_"
            metrics[prefix + "lr"] = lr
            for name, result in (("loss", loss), ("policy", policy), ("value", value),
                                 ("envelope", envelope), ("kl", kl), ("entropy", entropy),
                                 ("symmetry_loss", symmetry), ("preclip_grad_norm", grad_norm),
                                 ("actor_std_mean", stds[mask].mean())):
                diagnostics[prefix + name] = result.detach()
        keys = tuple(diagnostics)
        metrics.update(zip(keys, torch.stack([diagnostics[key] for key in keys]).cpu().tolist()))
        for name in ("entropy", "symmetry_loss", "preclip_grad_norm", "actor_std_mean"):
            rows = [result for key, result in metrics.items() if key.endswith("_" + name)]
            metrics["Optimizer/" + name] = sum(rows) / len(rows)
        self.updates += 1
        buffer.consume()
        metrics.update(updates=self.updates, optimizer_steps=self.optimizer_step_count)
        return metrics


__all__ = ["LegacyPPOConfig", "LegacyPPOTrainer", "LegacyNormalProvider", "legacy_kl"]
