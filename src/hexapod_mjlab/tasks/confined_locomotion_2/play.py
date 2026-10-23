"""Deterministic, weights-only play for the independent flat task.

The session owns one inference step: read the runtime's current prepared
observation, encode a scan only when its identity changes, advance the
explicit 187-wide temporal state once, and pass the action mean to the real
runtime. It creates no optimizer or action distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
from pathlib import Path

import torch
from torch import nn

from hexapod_mjlab.rl.confined_locomotion_2.network import ConfinedLocomotionNetwork

from .flat_env_cfg import make_training_env_cfg
from .runtime import TrainingRuntime, TrainingRuntimeFrozenError
from .scanner import ConfinedLidarCfg
from .rl.checkpoint import load_weights

__all__ = ["PlayConfigError", "PlayStep", "PlaySession", "make_play_env_cfg"]


class PlayConfigError(RuntimeError):
    """Invalid play input or incompatible checkpoint."""


@dataclass(frozen=True)
class PlayStep:
    prepared: object
    output: object
    transition: object


def make_play_env_cfg(*, num_envs: int = 1, seed: int | None = None):
    """Build a fresh flat-task cfg with formal falls and no timeout term."""
    cfg = make_training_env_cfg(
        num_envs=int(num_envs), episode_length_s=20.0, seed=seed
    )
    removed = cfg.terminations.pop("episode_timeout", None)
    if removed is None or not bool(getattr(removed, "time_out", False)):
        raise PlayConfigError("training cfg did not contain its episode_timeout term")
    if any(bool(getattr(term, "time_out", False)) for term in cfg.terminations.values()):
        raise PlayConfigError("play cfg still contains a time_out termination")
    return cfg


class PlaySession:
    """A fresh-episode, deterministic-mean session over a new-task runtime.

    Supply either an already wrapped environment or a play-mode runtime. The
    latter is useful for CPU fixtures. In both cases a current
    ``confined_locomotion_2`` full checkpoint is loaded as weights only;
    simulation, recurrent state, and episode clocks always start fresh.
    """

    def __init__(
        self,
        *,
        checkpoint,
        output_dir,
        seed: int,
        wrapper=None,
        runtime=None,
        network=None,
        lidar_cfg: ConfinedLidarCfg | dict | None = None,
        perturbation_cfg=None,
        own_environment: bool = False,
        allow_scan_override: bool = False,
        device=None,
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise PlayConfigError("seed must be an explicit integer")
        if (runtime is None) == (wrapper is None):
            raise PlayConfigError("supply exactly one of wrapper or runtime")
        if not isinstance(own_environment, bool):
            raise PlayConfigError("own_environment must be bool")
        if not isinstance(allow_scan_override, bool):
            raise PlayConfigError("allow_scan_override must be bool")
        if lidar_cfg is not None and not isinstance(lidar_cfg, ConfinedLidarCfg):
            if not isinstance(lidar_cfg, dict):
                raise PlayConfigError("lidar_cfg must be a mapping or ConfinedLidarCfg")
            try:
                lidar_cfg = ConfinedLidarCfg(**lidar_cfg)
            except (TypeError, ValueError) as error:
                raise PlayConfigError(f"invalid lidar_cfg: {error}") from error

        self.checkpoint_path = Path(checkpoint)
        if not self.checkpoint_path.is_file():
            raise PlayConfigError(f"checkpoint does not exist: {self.checkpoint_path}")
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._own_environment = own_environment
        self.allow_scan_override = allow_scan_override
        self._closed = False

        if runtime is not None and getattr(runtime, "mode", None) != "play":
            raise PlayConfigError("PlaySession requires a runtime in mode='play'")
        target_device = device
        if target_device is None:
            if runtime is not None:
                target_device = runtime.device
            else:
                target_device = getattr(getattr(wrapper, "unwrapped", None), "device", "cpu")
        self.device = torch.device(target_device)

        if network is None:
            network = ConfinedLocomotionNetwork(
                mlp_activation=nn.ELU(), init_fn=None, envelope_mode="raw"
            )
        if not isinstance(network, nn.Module):
            raise PlayConfigError("network must be a torch.nn.Module")
        self.network = network.to(self.device)
        try:
            # The checkpoint reader verifies current schema/task/architecture;
            # only network tensors are consumed. No training state is restored.
            far_sampling = getattr(
                getattr(runtime, "lidar_cfg", None),
                "far_sampling",
                getattr(lidar_cfg, "far_sampling", "all_endpoints"),
            )
            self.weight_load_report = load_weights(
                self.network, self.checkpoint_path,
                expected_far_sampling=far_sampling,
            )
        except Exception as error:
            # A caller-supplied runtime is already live at this point. Keep
            # checkpoint rejection cleanup consistent with scan-identity
            # rejection below, including owned environment teardown.
            if runtime is not None:
                runtime.close()
                if own_environment:
                    environment = getattr(runtime, "env", None)
                    if environment is not None:
                        environment.close()
            raise PlayConfigError(f"checkpoint cannot be used as current-task weights: {error}") from error
        self.network.eval()

        if runtime is None:
            runtime = TrainingRuntime(
                wrapper,
                seed=seed,
                output_dir=self.output_dir / "runtime",
                mode="play",
                lidar_cfg=lidar_cfg,
                perturbation_cfg=perturbation_cfg,
                scan_audit_bound_per_env=1,
            )
        self.runtime = runtime
        try:
            checkpoint_scan_identity = self.weight_load_report.get("identity", {}).get("scan_identity")
            runtime_scan_identity = runtime.scan_identity
            same_identity = checkpoint_scan_identity == runtime_scan_identity
            cross_configuration = False
            if not same_identity:
                cross_configuration = self._is_allowed_scan_override(
                    checkpoint_scan_identity, runtime_scan_identity
                )
            if not same_identity and not (allow_scan_override and cross_configuration):
                raise PlayConfigError("checkpoint scan identity differs from the play runtime")
            self.scan_evaluation_report = {
                "checkpoint_scan_identity": copy.deepcopy(checkpoint_scan_identity),
                "runtime_scan_identity": copy.deepcopy(runtime_scan_identity),
                "cross_configuration": bool(not same_identity),
                "override_enabled": allow_scan_override,
            }
        except BaseException:
            # A rejected play checkpoint must not leave the runtime's sensor
            # hooks or files live, including when the caller supplied runtime.
            runtime.close()
            if own_environment:
                environment = getattr(runtime, "env", None)
                if environment is not None:
                    environment.close()
            raise
        self.wrapper = getattr(runtime, "wrapper", wrapper)
        self.env = getattr(runtime, "env", getattr(self.wrapper, "unwrapped", None))
        self.device = torch.device(runtime.device)
        self.num_envs = int(runtime.num_envs)
        self.network.to(self.device)

        self.hidden = self.network.initial_hidden(self.num_envs, self.device)
        if tuple(self.hidden.shape) != (self.num_envs, 187):
            raise PlayConfigError("play network must expose explicit (B,187) temporal state")
        self._episode_ids = runtime.episode_ids.detach().clone()
        self._feature_identity = None
        self._frame_feature = None
        self._last_step: PlayStep | None = None
        self.frame_encoding_count = 0
        self.forward_count = 0
        self.preview_count = 0
        self._display_rows = None
        self._display_pool_cache = {}

    @staticmethod
    def _is_allowed_scan_override(checkpoint_identity, runtime_identity) -> bool:
        """Allow only fixed-catalog comparisons between supported raycast modes."""
        if not isinstance(checkpoint_identity, dict) or not isinstance(runtime_identity, dict):
            return False
        if "catalog" not in checkpoint_identity or "catalog" not in runtime_identity:
            return False
        fixed_grouping = "fixed_t1_channels/v1"
        checkpoint_catalog = checkpoint_identity.get("catalog")
        runtime_catalog = runtime_identity.get("catalog")
        if checkpoint_identity.get("grouping") != fixed_grouping or runtime_identity.get("grouping") != fixed_grouping:
            return False
        # Older fixed-channel identities omit a sampling field and retain the
        # legacy all-endpoints semantics. Any explicit sampling variant must
        # match exactly before a full/sparse or selection override is allowed.
        from .sampling import sampling_identity
        legacy_sampling = sampling_identity("all_endpoints")
        if (checkpoint_identity.get("sampling", legacy_sampling)
                != runtime_identity.get("sampling", legacy_sampling)):
            return False
        if (not isinstance(checkpoint_catalog, dict) or not isinstance(runtime_catalog, dict)
                or checkpoint_catalog.get("schema") != "confined_locomotion_2.fixed_channels/v1"
                or runtime_catalog.get("schema") != "confined_locomotion_2.fixed_channels/v1"
                or checkpoint_catalog != runtime_catalog):
            return False
        return checkpoint_identity.get("mode") in {"full", "sparse"} and runtime_identity.get("mode") in {"full", "sparse"}

    @torch.no_grad()
    def step_once(self) -> PlayStep:
        if self._closed:
            raise TrainingRuntimeFrozenError("play session is closed")
        runtime = self.runtime
        if bool(getattr(runtime, "_stopped", False)):
            raise TrainingRuntimeFrozenError(
                "play is paused after a formal fall; call manual_reset before stepping"
            )
        prepared = runtime.prepare()
        current_episodes = prepared.observation.episode_ids
        changed = current_episodes.to(self._episode_ids.device) != self._episode_ids
        if bool(changed.any()):
            self.hidden[changed.to(self.hidden.device)] = 0
            self._episode_ids.copy_(current_episodes.to(self._episode_ids.device))

        scan_ids = prepared.near_scan_ids.detach()
        identity = (prepared.observation.episode_ids.detach().clone(), scan_ids.clone())
        same = (
            self._feature_identity is not None
            and torch.equal(identity[0], self._feature_identity[0])
            and torch.equal(identity[1], self._feature_identity[1])
        )
        if not same:
            self._frame_feature = self.network.encode_points(
                prepared.near_points,
                prepared.near_valid,
                prepared.far_points,
                prepared.far_valid,
                near_angle_keys=prepared.near_angle_keys,
                far_angle_keys=prepared.far_angle_keys,
            ).detach()
            self._feature_identity = identity
            self.frame_encoding_count += 1

        from .display import capture_inputs, retain_display
        captured = capture_inputs(self, prepared) if self.env is not None else None
        try:
            output = self.network.forward_from_features(
                prepared.observation, self._frame_feature, self.hidden
            )
            action_mean = output.action_mean.detach()
            if tuple(action_mean.shape) != (self.num_envs, 18) or not bool(torch.isfinite(action_mean).all()):
                raise RuntimeError("network action mean must be finite with shape (B,18)")
            if tuple(output.hidden.shape) != tuple(self.hidden.shape) or not bool(torch.isfinite(output.hidden).all()):
                raise RuntimeError("network hidden output must be finite and preserve (B,187)")
            # Commit exactly once for this control transition. Read-only peek
            # never calls the network or advances this explicit state.
            self.hidden = output.hidden.detach().clone()
            self.forward_count += 1
            transition = runtime.step(action_mean)
            if captured is not None:
                retain_display(self, prepared, output, captured, preview=False,
                               reason="last action inputs; post-step pose")
        except BaseException as error:
            save_fault = getattr(runtime, "_save_fault", None)
            if callable(save_fault) and not bool(getattr(runtime, "_faulted", False)):
                save_fault(error, phase="play/inference")
            raise
        self._last_step = PlayStep(prepared, output, transition)
        return self._last_step

    def peek(self) -> PlayStep | None:
        """Return the retained snapshot without stepping or advancing hidden state."""
        return self._last_step

    @torch.no_grad()
    def refresh_idle_snapshot(self, *, reason="preview") -> None:
        """Explicit read-only policy preview; never commits temporal state.

        Called only on initial display, command changes or episode reset.
        Re-rendering an existing snapshot never calls this method.
        """
        if self._closed or self.runtime._faulted or self.runtime._stopped:
            raise TrainingRuntimeFrozenError("cannot preview a closed, fallen or faulted session")
        from .display import capture_inputs, retain_display
        prepared = self.runtime.prepare()
        identity = (prepared.observation.episode_ids.detach().clone(), prepared.near_scan_ids.detach().clone())
        same = (self._feature_identity is not None
                and torch.equal(identity[0], self._feature_identity[0])
                and torch.equal(identity[1], self._feature_identity[1]))
        if not same:
            self._frame_feature = self.network.encode_points(
                prepared.near_points, prepared.near_valid, prepared.far_points, prepared.far_valid,
                near_angle_keys=prepared.near_angle_keys, far_angle_keys=prepared.far_angle_keys).detach()
            self._feature_identity = identity
            self.frame_encoding_count += 1
        output = self.network.forward_from_features(prepared.observation, self._frame_feature, self.hidden)
        self.preview_count += 1
        captured = capture_inputs(self, prepared)
        retain_display(self, prepared, output, captured, preview=True, reason=reason)

    def build_display_snapshot(self, env_row=0, *, paused=True) -> dict:
        from .display import build_display_snapshot
        snapshot = build_display_snapshot(self, env_row, paused=paused)
        snapshot["scan_evaluation"] = copy.deepcopy(self.scan_evaluation_report)
        return snapshot

    def set_command(self, values) -> dict:
        """Set a validated manual body command for every play row."""
        try:
            command = tuple(float(x) for x in values)
        except (TypeError, ValueError, OverflowError) as error:
            raise PlayConfigError("command must be three numeric values") from error
        if len(command) != 3 or not all(torch.isfinite(torch.tensor(command)).tolist()):
            raise PlayConfigError("command must contain three finite values")
        ranges = (
            self.runtime._commands.RANGE_VX,
            self.runtime._commands.RANGE_VY,
            self.runtime._commands.RANGE_OMEGA,
        )
        if any(not (limits[0] <= value <= limits[1]) for value, limits in zip(command, ranges)):
            raise PlayConfigError("manual command is outside the runtime's legal ranges")
        rows = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        tensor = torch.tensor(command, dtype=torch.float32, device=self.device)
        record = self.runtime.set_manual_command(rows, tensor.expand(self.num_envs, 3).contiguous())
        self._display_rows = None
        return record

    def manual_reset(self) -> dict:
        """Start a new full play episode and clear recurrent/frame caches."""
        if self._closed:
            raise TrainingRuntimeFrozenError("play session is closed")
        rows = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        record = self.runtime.reset(rows)
        self.hidden.zero_()
        self._episode_ids.copy_(self.runtime.episode_ids)
        self._feature_identity = None
        self._frame_feature = None
        self._last_step = None
        self._display_rows = None
        self._display_pool_cache.clear()
        return record

    def close(self) -> None:
        if self._closed:
            return
        self.runtime.close()
        if self._own_environment and self.env is not None:
            self.env.close()
        self._closed = True
