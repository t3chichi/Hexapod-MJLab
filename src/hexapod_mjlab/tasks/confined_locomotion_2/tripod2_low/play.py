"""Tripod mean-policy playback, retaining the independent LegacyRuntime."""

import json
from pathlib import Path

import torch
from torch import nn

from hexapod_mjlab.rl.confined_locomotion_2.network import ConfinedLocomotionNetwork
from ..components.lidar.scanner import ConfinedLidarCfg
from ..play import PlayStep
from .checkpoint import read_checkpoint
from .env import make_env_cfg
from .runtime import LegacyRuntime


class TripodPlaySession:
    """Native/headless session; never creates an optimizer or browser capture."""

    COMMAND_LIMITS = (2.5, 1., 1.2)

    def __init__(self, *, checkpoint, output_dir, seed=1, device="cuda:0"):
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper

        payload = read_checkpoint(checkpoint)
        if not torch.cuda.is_available() or torch.device(device).type != "cuda":
            raise RuntimeError("tripod policy playback requires CUDA")
        self.device, self.seed = torch.device(device), seed
        self.checkpoint = Path(checkpoint).resolve()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.env = self.runtime = self._metrics = None
        self._closed = False
        self.num_envs, self.forward_count = 1, 0
        # Preserve the old diagnostic's initialization order and saved scanner.
        torch.manual_seed(seed)
        self.network = ConfinedLocomotionNetwork(mlp_activation=nn.ELU(), init_fn=None,
            envelope_mode="raw", ground_samples=None).to(self.device).eval()
        self.network.load_state_dict(payload["network"], strict=True)
        lidar = ConfinedLidarCfg(**payload["identity"]["recipe"]["lidar_config"])
        try:
            self.env = ManagerBasedRlEnv(make_env_cfg(num_envs=1, seed=seed, play=True), device=str(self.device))
            self.runtime = LegacyRuntime(RslRlVecEnvWrapper(self.env), seed=seed,
                output_dir=self.output_dir / "runtime", mode="play", lidar_cfg=lidar)
            if self.runtime.scan_identity != payload["identity"]["scan_identity"]:
                raise ValueError("play scanner differs from checkpoint")
            self.hidden = self.network.initial_hidden(1, self.device)
            self._metrics = (self.output_dir / "play_metrics.jsonl").open("x")
            self.command = (0., 0., 0.)
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup:
                if hasattr(error, "add_note"):
                    error.add_note(f"Tripod Play cleanup failed: {cleanup!r}")
            raise

    @torch.inference_mode()
    def step_once(self, *, capture_display=False, retain_debug=False):
        if capture_display or retain_debug:
            raise ValueError("Tripod Play currently supports native/headless rendering")
        prepared = self.runtime.prepare()
        output = self.network(prepared.observation, prepared.near_points, prepared.near_valid,
            prepared.far_points, prepared.far_valid, self.hidden,
            near_angle_keys=prepared.near_angle_keys, far_angle_keys=prepared.far_angle_keys)
        self.hidden = output.hidden
        transition = self.runtime.step(output.action_mean)
        self.forward_count += 1  # The runtime's one startup warmup is not a policy step.
        self._metrics.write(json.dumps({"step": self.forward_count, "reward": float(transition.reward[0]),
            "root_height_m": float(transition.context_record["root_pos_w"][0, 2]),
            "flipped": bool(transition.terminated[0])}) + "\n")
        return PlayStep(prepared, output, transition)

    def set_command(self, values):
        command = torch.as_tensor(values, dtype=torch.float32, device=self.device).reshape(1, 3)
        record = self.runtime.set_manual_command([0], command)
        self.command = tuple(float(v) for v in values)
        return record

    @torch.inference_mode()
    def manual_reset(self):
        # LegacyRuntime/ActionManager own the actuator LSTM, contact and scan
        # reset. Do not run another startup warmup or use the Flat T1 reset.
        record = self.runtime.reset(torch.arange(self.num_envs, device=self.device))
        self.hidden.zero_()
        self.command = (0., 0., 0.)
        return record

    def close(self):
        if self._closed:
            return
        try:
            if self.runtime is not None:
                self.runtime.close()
        finally:
            try:
                if self.env is not None:
                    self.env.close()
            finally:
                if self._metrics is not None:
                    self._metrics.close()
                self._closed = True
