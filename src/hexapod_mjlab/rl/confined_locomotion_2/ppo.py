"""Sequence-correct pure-Torch PPO for confined locomotion v2."""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Callable

import torch
from torch import nn

from hexapod_mjlab.rl.confined_locomotion.contracts import FLOAT_DTYPE
from hexapod_mjlab.rl.confined_locomotion.losses import (
    adjust_learning_rate, envelope_sum_loss, kl_divergence_old_new, normalize_rollout_advantage,
    ppo_policy_loss, value_clipped_loss,
)
from hexapod_mjlab.rl.confined_locomotion.symmetry import (
    JointPermutation, mirror_action_distribution, mirror_envelope_values,
    mirror_mean_mse,
)
from .auxiliary import AuxiliaryResult, HeightReconstructionAuxiliary, resolve_auxiliary
from .training_rollout import (
    PreparedSequenceReplay, prepare_sequence_replay, replay_sequence,
    SequenceRolloutBuffer, SequenceStep, ROLLOUT_STEPS,
)


def _require_finite_tensors(tensors, message):
    """Read one reduced flag per device, preserving optimizer-stage boundaries."""
    by_device = {}
    for tensor in tensors:
        by_device.setdefault(tensor.device, []).append(torch.isfinite(tensor).all())
    if any(not bool(torch.stack(flags).all()) for flags in by_device.values()):
        raise FloatingPointError(message)


@dataclass(frozen=True)
class TemporalPPOConfig:
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

    def __post_init__(self) -> None:
        fixed = {
            "rollout_steps": 24, "epochs": 5, "minibatches": 4,
            "gamma": 0.99, "gae_lambda": 0.95, "clip_epsilon": 0.2,
            "value_coeff": 1.0, "entropy_coeff": 0.01, "envelope_coeff": 1.0,
            "learning_rate": 1e-3,
            "adam_betas": (0.9, 0.999), "adam_eps": 1e-8, "weight_decay": 0.0,
            "max_grad_norm": 1.0, "desired_kl": 0.01, "lr_min": 1e-5,
            "lr_max": 1e-2, "symmetry_mean_coeff": 0.6,
        }
        for name, value in fixed.items():
            if getattr(self, name) != value:
                raise ValueError(f"{name} is fixed by the accepted recipe at {value!r}")


class TemporalPPOTrainer:
    """One Adam, sequence replay, and full-rollout normalized advantages.

    ``network`` follows the v2 explicit-hidden forward contract. ``distribution_provider``
    is the shared independent diagonal Normal provider from the original implementation.
    """
    def __init__(
        self, *, network: nn.Module, distribution_provider: nn.Module,
        device: torch.device | str, config: TemporalPPOConfig | None = None,
        symmetry_enabled: bool = False, joint_permutation: JointPermutation | None = None,
        mount_quat_xyzw: torch.Tensor | None = None, allow_cpu: bool = True,
        seed: int = 0, execution_mode: str = "low_memory",
        auxiliary: Callable[[Any, Any], AuxiliaryResult] | None = None,
    ) -> None:
        if not isinstance(network, nn.Module) or not isinstance(distribution_provider, nn.Module):
            raise TypeError("network and distribution_provider must be torch modules")
        self.device = torch.device(device)
        if self.device.type != "cuda" and not allow_cpu:
            raise ValueError("PPO update requires CUDA unless allow_cpu=True")
        if not isinstance(symmetry_enabled, bool):
            raise TypeError("symmetry_enabled must be bool")
        if symmetry_enabled and (joint_permutation is None or mount_quat_xyzw is None):
            raise ValueError("symmetry requires joint mapping and lidar optical mount quaternion")
        if execution_mode not in ("low_memory", "speed"):
            raise ValueError("execution_mode must be low_memory or speed")
        self.execution_mode = execution_mode
        self.network = network
        self.distribution_provider = distribution_provider
        self.config = config or TemporalPPOConfig()
        self.symmetry_enabled = symmetry_enabled
        self.joint_permutation = joint_permutation
        self.mount_quat_xyzw = mount_quat_xyzw
        self.auxiliary = resolve_auxiliary(auxiliary)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an int")
        self.seed = int(seed)
        self.shuffle_generator = torch.Generator(device=self.device)
        self.shuffle_generator.manual_seed(derive_shuffle_generator_seed(self.seed))
        params = []
        seen: set[int] = set()
        for parameter in list(network.parameters()) + list(distribution_provider.parameters()):
            if parameter.requires_grad and id(parameter) not in seen:
                params.append(parameter)
                seen.add(id(parameter))
        if not params:
            raise ValueError("no trainable parameters")
        if any(p.device != self.device or p.dtype is not FLOAT_DTYPE for p in params):
            raise ValueError("all trainable parameters must be float32 on the declared device")
        self.optimizer = torch.optim.Adam(
            params, lr=self.config.learning_rate, betas=self.config.adam_betas,
            eps=self.config.adam_eps, weight_decay=self.config.weight_decay,
        )
        self.updates = 0
        self.optimizer_step_count = 0

    @property
    def update_count(self) -> int:
        return self.updates

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": "confined_locomotion_2.temporal_ppo/v1",
            "optimizer": self.optimizer.state_dict(),
            "updates": self.updates,
            "optimizer_step_count": self.optimizer_step_count,
            "config": asdict(self.config),
            "symmetry_enabled": self.symmetry_enabled,
            "execution_mode": self.execution_mode,
            "seed": self.seed,
            "shuffle_generator_state": self.shuffle_generator.get_state().cpu(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict) or state.get("schema") != "confined_locomotion_2.temporal_ppo/v1":
            raise ValueError("incompatible temporal PPO state")
        if state.get("config") != asdict(self.config) or state.get("symmetry_enabled") != self.symmetry_enabled:
            raise ValueError("saved temporal PPO recipe differs from this trainer")
        if int(state.get("seed", -1)) != self.seed:
            raise ValueError("saved temporal PPO seed differs from this trainer")
        # execution_mode only selects activation recomputation; it does not
        # change the optimizer or rollout semantics, so cross-mode resume is
        # allowed. The current trainer keeps its requested mode.
        saved_mode = state.get("execution_mode")
        if saved_mode is not None and saved_mode not in ("low_memory", "speed"):
            raise ValueError(f"saved execution mode is invalid: {saved_mode!r}")
        self.optimizer.load_state_dict(state["optimizer"])
        self.updates = int(state["updates"])
        self.optimizer_step_count = int(state["optimizer_step_count"])
        self.shuffle_generator.set_state(state["shuffle_generator_state"].cpu())

    @property
    def faulted(self) -> bool:
        return getattr(self, "_faulted", False)

    @staticmethod
    def _obs_select(obs: Any, mask: torch.Tensor) -> Any:
        from hexapod_mjlab.rl.confined_locomotion.contracts import OnlineObservation
        return OnlineObservation(**{name: getattr(obs, name)[mask] for name in obs.__dataclass_fields__})

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if not bool(mask.any()):
            raise ValueError("empty loss minibatch")
        return values[mask]

    def _replay(self, steps: list[SequenceStep] | PreparedSequenceReplay, *, mirrored: bool = False) -> list[Any]:
        return replay_sequence(
            self.network, steps, mirrored=mirrored,
            joint_permutation=self.joint_permutation, mount_quat_xyzw=self.mount_quat_xyzw,
            checkpoint_points=self.execution_mode == "low_memory")

    def update(self, buffer: SequenceRolloutBuffer) -> dict[str, float | int]:
        if self.faulted:
            raise RuntimeError("trainer is faulted and cannot continue")
        try:
            return self._update_impl(buffer)
        except BaseException:
            self._faulted = True
            raise

    def _update_impl(self, buffer: SequenceRolloutBuffer) -> dict[str, float | int]:
        buffer._require_full()
        if buffer.device != self.device:
            raise ValueError("buffer and trainer devices differ")
        if self.symmetry_enabled and any(step.prepared is None or step.mirror_hidden_before is None for step in buffer.steps):
            raise ValueError("symmetry requires prepared inputs and shadow hidden state at every step")
        replay_inputs = prepare_sequence_replay(buffer.steps, network=self.network)
        mirror_inputs = prepare_sequence_replay(
            buffer.steps, network=self.network, mirrored=True, joint_permutation=self.joint_permutation,
            mount_quat_xyzw=self.mount_quat_xyzw) if self.symmetry_enabled else None
        height_targets = (self.auxiliary.prepare_targets(buffer.steps, mirrored=self.symmetry_enabled)
                          if isinstance(self.auxiliary, HeightReconstructionAuxiliary) else None)
        advantages, returns = buffer.compute_gae()
        flat_adv = normalize_rollout_advantage(advantages.reshape(-1))
        advantages = flat_adv.reshape_as(advantages).detach()
        returns = returns.detach()
        t_count, batch = advantages.shape
        total = t_count * batch
        if total % self.config.minibatches:
            raise ValueError("T*B must divide evenly into four disjoint minibatches")
        # Behavior data and supervision are immutable throughout all Adam steps.
        actions = torch.stack([s.action for s in buffer.steps]).reshape(total, -1)
        old_lp = torch.stack([s.old_log_prob for s in buffer.steps])
        old_means = torch.stack([s.old_mean for s in buffer.steps])
        old_stds = torch.stack([s.old_std for s in buffer.steps])
        old_values = torch.stack([s.old_value for s in buffer.steps])
        et = torch.stack([s.envelope_label.values for s in buffer.steps])
        ev = torch.stack([s.envelope_label.valid for s in buffer.steps])
        if self.symmetry_enabled:
            pushed = mirror_action_distribution(
                old_means.reshape(total, -1), old_stds.reshape(total, -1),
                actions, self.joint_permutation)
            old_m = pushed.mean.reshape_as(old_means)
            old_s = pushed.std.reshape_as(old_stds)
            mirror_actions = pushed.raw_action
            mirror_et = mirror_envelope_values(et.reshape(total, -1)).reshape_as(et)
        summaries: dict[str, float | int] = {"updates": self.updates, "steps": t_count, "num_envs": batch}
        # Keep diagnostics as detached device scalars through the whole update.  Converting
        # each parameter norm (or each metric) with .item() here would synchronize CUDA
        # repeatedly; materialize the compact scalar list once after optimization.
        diagnostic_values: dict[str, list[torch.Tensor]] = {}
        frame_parameter_ids = {id(p) for p in getattr(self.network, "frame_encoder", nn.Module()).parameters()}
        temporal_parameter_ids = {id(p) for p in getattr(self.network, "temporal_encoder", nn.Module()).parameters()}

        def record_diagnostic(name: str, value: torch.Tensor) -> None:
            diagnostic_values.setdefault(name, []).append(value.detach().reshape(()))

        for epoch in range(self.config.epochs):
            permutation = torch.randperm(total, device=self.device, generator=self.shuffle_generator)
            chunk = total // self.config.minibatches
            for mb in range(self.config.minibatches):
                selected = permutation[mb * chunk:(mb + 1) * chunk]
                loss_mask = torch.zeros(total, dtype=torch.bool, device=self.device)
                loss_mask[selected] = True
                loss_mask = loss_mask.reshape(t_count, batch)
                outputs = self._replay(replay_inputs)
                mirrored_outputs = self._replay(mirror_inputs) if mirror_inputs is not None else None
                # Distribution calls operate on independent action events. Flatten only
                # T and B; retain all sequence replay and minibatch loss masks unchanged.
                means = torch.stack([o.action_mean for o in outputs])
                values = torch.stack([o.value.squeeze(-1) for o in outputs])
                ep = torch.stack([o.envelope for o in outputs])
                dist = self.distribution_provider(means.reshape(total, -1))
                lp = dist.log_prob(actions).reshape(t_count, batch)
                ent = dist.entropy().reshape(t_count, batch)
                stds = dist.std.reshape_as(means)
                if not bool(loss_mask.any()):
                    raise AssertionError("empty disjoint minibatch")
                mask = loss_mask
                loss_pi = ppo_policy_loss(lp[mask], old_lp[mask], advantages[mask], clip_epsilon=self.config.clip_epsilon)
                loss_v = value_clipped_loss(values[mask], old_values[mask], returns[mask])
                loss_h = ent[mask].mean()
                loss_e = envelope_sum_loss(ep[mask], et[mask], ev[mask])
                ratios = torch.exp((lp[mask] - old_lp[mask]).detach())
                ratio_clip_fraction = ((ratios < 1.0 - self.config.clip_epsilon) |
                                       (ratios > 1.0 + self.config.clip_epsilon)).to(lp.dtype).mean()
                kl = kl_divergence_old_new(
                    old_means[mask], old_stds[mask], means[mask].detach(), stds[mask].detach()
                ).mean()
                if mirrored_outputs is not None:
                    mmean = torch.stack([o.action_mean for o in mirrored_outputs])
                    mvalue = torch.stack([o.value.squeeze(-1) for o in mirrored_outputs])
                    menv = torch.stack([o.envelope for o in mirrored_outputs])
                    mdist = self.distribution_provider(mmean.reshape(total, -1))
                    mlp = mdist.log_prob(mirror_actions).reshape(t_count, batch)
                    ment = mdist.entropy().reshape(t_count, batch)
                    # The event log probability is invariant under the same joint permutation.
                    mpi = ppo_policy_loss(mlp[mask], old_lp[mask], advantages[mask], clip_epsilon=self.config.clip_epsilon)
                    mvl = value_clipped_loss(mvalue[mask], old_values[mask], returns[mask])
                    mel = envelope_sum_loss(menv[mask], mirror_et[mask], ev[mask])
                    loss_pi = 0.5 * (loss_pi + mpi)
                    loss_v = 0.5 * (loss_v + mvl)
                    loss_e = 0.5 * (loss_e + mel)
                    loss_h = 0.5 * (loss_h + ment[mask].mean())
                    loss_sym = mirror_mean_mse(mmean[mask], means[mask], self.joint_permutation)
                    new_mirror_std = mdist.std.reshape_as(mmean)[mask].detach()
                    kl_m = kl_divergence_old_new(
                        old_m[mask], old_s[mask], mmean[mask].detach(), new_mirror_std
                    ).mean()
                    kl = 0.5 * (kl + kl_m)
                else:
                    loss_sym = means.new_zeros(())
                # Plugin sees deployable output features and a small batch view; empty default has no head.
                auxiliary_result = self.auxiliary(outputs, {
                    "loss_mask": mask, "advantages": advantages, "returns": returns,
                    "steps": buffer.steps, "mirrored_outputs": mirrored_outputs,
                    "height_targets": height_targets,
                })
                if not isinstance(auxiliary_result, AuxiliaryResult) or auxiliary_result.loss.ndim != 0:
                    raise TypeError("auxiliary plugin must return AuxiliaryResult with scalar loss")
                total_loss = (loss_pi + self.config.value_coeff * loss_v - self.config.entropy_coeff * loss_h
                              + self.config.envelope_coeff * loss_e
                              + self.config.symmetry_mean_coeff * loss_sym + auxiliary_result.loss)
                if not bool(torch.isfinite(total_loss)):
                    raise FloatingPointError("non-finite PPO loss")
                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                params = [p for g in self.optimizer.param_groups for p in g["params"] if p.grad is not None]
                _require_finite_tensors((p.grad for p in params), "non-finite PPO gradient")
                frame_grad_norm = torch.stack([p.grad.detach().norm(2) for p in params
                                               if id(p) in frame_parameter_ids]).norm(2) if any(
                    id(p) in frame_parameter_ids for p in params
                ) else total_loss.new_zeros(())
                temporal_grad_norm = torch.stack([p.grad.detach().norm(2) for p in params
                                                  if id(p) in temporal_parameter_ids]).norm(2) if any(
                    id(p) in temporal_parameter_ids for p in params
                ) else total_loss.new_zeros(())
                grad_norm = torch.nn.utils.clip_grad_norm_(params, self.config.max_grad_norm)
                _require_finite_tensors([grad_norm, *(p.grad for p in params)],
                                        "non-finite PPO gradient after clipping")
                lr = adjust_learning_rate(float(kl.detach().item()), self.optimizer.param_groups[0]["lr"], desired_kl=self.config.desired_kl, lr_min=self.config.lr_min, lr_max=self.config.lr_max)
                for group in self.optimizer.param_groups:
                    group["lr"] = lr
                self.optimizer.step()
                self.optimizer_step_count += 1
                _require_finite_tensors(params, "non-finite PPO parameter after optimizer step")
                _require_finite_tensors(
                    (value for state in self.optimizer.state.values() for value in state.values()
                     if isinstance(value, torch.Tensor)), "non-finite Adam state after optimizer step")
                diagnostic_items = {
                    "entropy": loss_h,
                    "symmetry_loss": loss_sym,
                    "auxiliary_loss": auxiliary_result.loss,
                    "actor_std_mean": stds[mask].mean(),
                    "actor_std_min": stds[mask].min(),
                    "actor_std_max": stds[mask].max(),
                    "ratio_clip_fraction": ratio_clip_fraction,
                    "preclip_grad_norm": grad_norm,
                    "frame_encoder_preclip_grad_norm": frame_grad_norm,
                    "temporal_encoder_preclip_grad_norm": temporal_grad_norm,
                }
                target_var = returns[mask].var(unbiased=False)
                ev_valid = target_var > 0
                safe_target_var = torch.where(ev_valid, target_var, torch.ones_like(target_var))
                explained_variance = 1.0 - (returns[mask] - values[mask]).var(unbiased=False) / safe_target_var
                diagnostic_items["value_explained_variance_valid"] = ev_valid.to(total_loss.dtype)
                diagnostic_items["value_explained_variance"] = explained_variance
                for name, value in diagnostic_items.items():
                    record_diagnostic(name, value)
                for name, value in auxiliary_result.metrics.items():
                    record_diagnostic(name, torch.as_tensor(value, device=self.device))
                for name, value in (("loss", total_loss), ("policy", loss_pi), ("value", loss_v),
                                    ("envelope", loss_e), ("kl", kl)):
                    record_diagnostic(name, value)
                summaries[f"epoch_{epoch}_minibatch_{mb}_lr"] = float(lr)
        diagnostic_names = tuple(diagnostic_values)
        if diagnostic_names:
            # One host transfer for all optimizer diagnostics, independent of parameter
            # count and minibatch count.
            packed = torch.stack([value for name in diagnostic_names for value in diagnostic_values[name]]).cpu()
            offset = 0
            host_by_name: dict[str, list[float]] = {}
            for name in diagnostic_names:
                values_for_name = diagnostic_values[name]
                count = len(values_for_name)
                host_by_name[name] = packed[offset:offset + count].tolist()
                offset += count
            valid_ev = host_by_name["value_explained_variance_valid"]
            for name in diagnostic_names:
                host_values = host_by_name[name]
                for index, value in enumerate(host_values):
                    # Recover epoch/minibatch coordinates in the same order as append.
                    epoch, mb = divmod(index, self.config.minibatches)
                    if name == "value_explained_variance" and not bool(valid_ev[index]):
                        continue
                    summaries[f"epoch_{epoch}_minibatch_{mb}_{name}"] = float(value)
                if name in ("loss", "policy", "value", "envelope", "kl"):
                    continue  # Existing minibatch-only keys; no new aggregate metrics.
                if name in ("actor_std_min", "actor_std_max"):
                    reducer = min if name == "actor_std_min" else max
                    summaries[f"Optimizer/{name}"] = float(reducer(host_values))
                elif name == "value_explained_variance_valid":
                    summaries[f"Optimizer/{name}"] = float(max(host_values))
                elif name == "value_explained_variance":
                    valid_values = [float(value) for index, value in enumerate(host_values) if bool(valid_ev[index])]
                    if valid_values:
                        summaries["Optimizer/value_explained_variance_mean"] = sum(valid_values) / len(valid_values)
                elif name.startswith("Height/"):
                    summaries[name] = float(sum(host_values) / count)
                elif name != "actor_std_mean":
                    summaries[f"Optimizer/{name}_mean"] = float(sum(host_values) / count)
                else:
                    summaries[f"Optimizer/{name}"] = float(sum(host_values) / count)
        self.updates += 1
        buffer.consume()
        summaries["updates"] = self.updates
        summaries["optimizer_steps"] = self.optimizer_step_count
        return summaries


def derive_shuffle_generator_seed(seed: int) -> int:
    """Stable named PRNG derivation isolated from behavior/action RNG streams."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an int")
    digest = hashlib.sha256(f"confined_locomotion_2|optimization_shuffle|{seed}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


__all__ = ["TemporalPPOConfig", "TemporalPPOTrainer", "derive_shuffle_generator_seed"]
