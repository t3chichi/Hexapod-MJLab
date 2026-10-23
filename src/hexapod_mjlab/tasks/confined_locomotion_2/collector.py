"""Real transitions with explicit recurrent state; bootstrap never advances it."""

from __future__ import annotations

import hashlib

import torch

from hexapod_mjlab.rl.confined_locomotion_2.symmetry import mirror_prepared
from hexapod_mjlab.rl.confined_locomotion_2.training_rollout import SequenceRolloutBuffer


def model_state_fingerprint(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class TrainingCollector:
    def __init__(self, runtime, network, provider, *, symmetry_enabled=False,
                 joint_permutation=None, mount_quat_xyzw=None, transition_observer=None):
        self.runtime, self.network, self.provider = runtime, network, provider
        self.device = torch.device(runtime.device)
        self.num_envs = runtime.num_envs
        self.symmetry_enabled = symmetry_enabled
        self.joint_permutation = joint_permutation
        self.mount_quat_xyzw = mount_quat_xyzw
        if transition_observer is not None and not callable(transition_observer):
            raise TypeError("transition_observer must be callable or None")
        self.transition_observer = transition_observer
        self.hidden = network.initial_hidden(self.num_envs, self.device)
        self.mirror_hidden = self.hidden.clone() if symmetry_enabled else None
        self.episode_ids = runtime.episode_ids.detach().clone()
        self._features = {}
        self.frame_encoding_rows = 0

    def clear_feature_cache(self):
        self._features.clear()

    def _frame(self, prepared, *, mirror=False):
        """Row cache is inference-only and bounded to the latest scan per row."""
        rows = []
        for row in range(self.num_envs):
            key = (mirror, row)
            identity = (int(prepared.observation.episode_ids[row]),
                        int(prepared.near_scan_ids[row]))
            entry = self._features.get(key)
            if entry is None or entry[0] != identity:
                feature = self.network.encode_points(
                    prepared.near_points[row:row+1], prepared.near_valid[row:row+1],
                    prepared.far_points[row:row+1], prepared.far_valid[row:row+1],
                    near_angle_keys=(None if prepared.near_angle_keys is None else
                                     prepared.near_angle_keys[row:row+1]),
                    far_angle_keys=(None if prepared.far_angle_keys is None else
                                    prepared.far_angle_keys[row:row+1]),
                )
                entry = (identity, feature.detach())
                self._features[key] = entry
                self.frame_encoding_rows += 1
            rows.append(entry[1])
        return torch.cat(rows, dim=0)

    def _mirror(self, prepared):
        return mirror_prepared(prepared, self.joint_permutation,
                               mount_quat_xyzw=self.mount_quat_xyzw)

    def _sync_episodes(self, prepared):
        changed = prepared.observation.episode_ids != self.episode_ids
        self.hidden[changed] = 0
        if self.mirror_hidden is not None:
            self.mirror_hidden[changed] = 0
        self.episode_ids.copy_(prepared.observation.episode_ids)

    @torch.no_grad()
    def collect(self):
        self.clear_feature_cache()  # parameters may have changed since last rollout
        buffer = SequenceRolloutBuffer(self.num_envs, self.device, rollout_steps=24)
        for _ in range(24):
            prepared = self.runtime.prepare()
            self._sync_episodes(prepared)
            labels = self.runtime.labels()
            hidden_before = self.hidden.clone()
            mirror_before = None if self.mirror_hidden is None else self.mirror_hidden.clone()
            output = self.network.forward_from_features(
                prepared.observation, self._frame(prepared), hidden_before)
            self.hidden = output.hidden.detach().clone()
            if self.symmetry_enabled:
                mirrored = self._mirror(prepared)
                mirrored_output = self.network.forward_from_features(
                    mirrored.observation, self._frame(mirrored, mirror=True), mirror_before)
                self.mirror_hidden = mirrored_output.hidden.detach().clone()
            distribution = self.provider(output.action_mean)
            action = distribution.sample()
            old_log_prob = distribution.log_prob(action)
            transition = self.runtime.step(action)
            if self.transition_observer is not None:
                self.transition_observer(transition, output, labels["envelope"])
            # Final observation is PRE-reset, including time-limit rows. Pure
            # fork: the resulting hidden is deliberately never committed.
            final = transition.final_inputs
            final_output = self.network.forward_from_features(
                final.observation, self._frame(final), self.hidden)
            next_value = final_output.value.squeeze(-1).clone()
            next_value[transition.terminated] = 0
            buffer.append_step(
                prepared=prepared, observation=prepared.observation,
                near_points=prepared.near_points, near_valid=prepared.near_valid,
                far_points=prepared.far_points, far_valid=prepared.far_valid,
                near_angle_keys=prepared.near_angle_keys,
                far_angle_keys=prepared.far_angle_keys,
                action=action, old_log_prob=old_log_prob,
                old_mean=output.action_mean, old_std=distribution.std,
                value=output.value.squeeze(-1), reward=transition.reward,
                terminated=transition.terminated, truncated=transition.truncated,
                next_value=next_value, hidden_before=hidden_before,
                mirror_hidden_before=mirror_before,
                envelope_label=labels["envelope"],
            )
            done = transition.terminated | transition.truncated
            self.hidden[done] = 0
            if self.mirror_hidden is not None:
                self.mirror_hidden[done] = 0
            if transition.engineering_stopped:
                raise RuntimeError("engineering collection stopped naturally; incomplete rollout retained")
        return buffer

    def state_dict(self):
        return {"hidden": self.hidden.detach().clone(),
                "mirror_hidden": None if self.mirror_hidden is None else self.mirror_hidden.detach().clone(),
                "episode_ids": self.episode_ids.detach().clone()}

    def load_state_dict(self, state):
        for name in ("hidden", "episode_ids"):
            saved = state[name].to(self.device)
            live = getattr(self, name)
            if saved.shape != live.shape or saved.dtype != live.dtype:
                raise ValueError(f"collector {name} shape/dtype mismatch")
            if saved.is_floating_point() and not torch.isfinite(saved).all():
                raise ValueError(f"collector {name} is non-finite")
        mirrored = state["mirror_hidden"]
        if (mirrored is None) != (self.mirror_hidden is None):
            raise ValueError("collector mirror state does not match recipe")
        if mirrored is not None and (mirrored.shape != self.hidden.shape or mirrored.dtype != self.hidden.dtype or
                                     not torch.isfinite(mirrored).all()):
            raise ValueError("invalid mirror hidden state")
        self.hidden.copy_(state["hidden"].to(self.device))
        self.episode_ids.copy_(state["episode_ids"].to(self.device))
        if mirrored is not None:
            self.mirror_hidden.copy_(mirrored.to(self.device))
        self.clear_feature_cache()
