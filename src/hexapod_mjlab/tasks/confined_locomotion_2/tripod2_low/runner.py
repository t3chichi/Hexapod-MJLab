"""Independent recipe/runner; the policy and scan implementation are shared."""
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from mjlab.rl import RslRlBaseRunnerCfg
from mjlab.utils.torch import configure_torch_backends

from hexapod_mjlab.rl.confined_locomotion.distribution import derive_action_generator_seed
from hexapod_mjlab.rl.confined_locomotion.symmetry import JointPermutation
from hexapod_mjlab.common.provenance import capture_code_provenance
from hexapod_mjlab.rl.confined_locomotion_2.network import ConfinedLocomotionNetwork
from ..collector import TrainingCollector
from ..rl.reporting import (
    format_iteration_summary,
    optimizer_scalar_summary,
)
from ..rl.runner import ConfinedLocomotionRunner, environment_semantics
from ..components.lidar.scanner import ConfinedLidarCfg
from .actuator import ACTUATOR_PATH
from .legacy import reward_scales
from .checkpoint import architecture_identity, load_full_checkpoint, save_full_checkpoint
from .env import TASK_ID
from .ppo import LegacyNormalProvider, LegacyPPOConfig, LegacyPPOTrainer
from .runtime import LegacyRuntime


@dataclass
class TripodRlCfg(RslRlBaseRunnerCfg):
    seed: int = 1
    class_name: str = "TripodRunner"
    experiment_name: str = "confined_locomotion_flat_tripod2_low"
    max_iterations: int = 0  # Registration does not launch training.
    num_steps_per_env: int = 24
    clip_actions: float | None = None
    save_interval: int = 50
    logger: str = "tensorboard"
    upload_model: bool = False
    formal_num_envs: int | None = None
    engineering_validation: bool = False
    execution_mode: str = "low_memory"
    ppo_config: LegacyPPOConfig = field(default_factory=LegacyPPOConfig)
    lidar_config: dict = field(default_factory=lambda: {"raycast_mode": "full", "far_sampling": "all_endpoints"})


def validate_recipe(cfg):
    for name in ("max_iterations", "formal_num_envs", "save_interval"):
        value = cfg.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be an explicit positive integer")
    seed = cfg.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be uint32")
    if cfg.get("num_steps_per_env") != 24 or cfg.get("clip_actions") is not None:
        raise ValueError("legacy recipe uses 24 steps and task-owned action clipping")
    if cfg.get("logger") != "tensorboard" or cfg.get("upload_model", False):
        raise ValueError("tripod runner records local tensorboard artifacts")
    if cfg.get("execution_mode") not in ("low_memory", "speed"):
        raise ValueError("invalid execution mode")
    ppo = cfg.get("ppo_config", {})
    ppo = ppo if isinstance(ppo, LegacyPPOConfig) else LegacyPPOConfig(**ppo)
    if ppo != LegacyPPOConfig():
        raise ValueError("tripod2-low uses the audited reference PPO recipe")
    if cfg.get("engineering_validation", False) and (cfg["formal_num_envs"] > 2 or cfg["max_iterations"] != 1):
        raise ValueError("engineering validation is limited to two envs and one update")
    return ppo, ConfinedLidarCfg(**cfg["lidar_config"])


class LegacyMetrics:
    def __init__(self, num_envs, device):
        self.num_envs, self.device = num_envs, device
        self.episode_length = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.episode_return = torch.zeros(num_envs, device=device)
        self.begin_iteration()

    def begin_iteration(self):
        self.sums, self.samples, self.completed = {}, 0, []

    def observe(self, transition, output, label):
        rec, command = transition.context_record, transition.command
        self.samples += self.num_envs
        actual = torch.cat((rec["lin_vel_b"][:, :2], rec["ang_vel_b"][:, 2:3]), -1)
        values = {"Reward/actual": transition.reward,
                  "Tracking/squared_error": (actual - command).square().sum(-1),
                  "Proprio/root_height_m": rec["root_pos_w"][:, 2],
                  "Proprio/root_height_error_abs_m": (rec["root_pos_w"][:, 2] - .25).abs(),
                  "Envelope/squared_error": (output.envelope - label.values).square().sum(-1)}
        values.update({"RewardContribution/" + name: raw * transition.reward_weights[name] * .02
                       for name, raw in transition.reward_raw_terms.items()})
        for name, value in values.items():
            self.sums[name] = self.sums.get(name, 0.) + value.detach().sum()
        self.episode_return += transition.reward
        self.episode_length += 1
        done = transition.terminated | transition.truncated
        if bool(done.any()):
            self.completed.extend(zip(self.episode_return[done].cpu().tolist(), self.episode_length[done].cpu().tolist()))
            self.episode_return[done], self.episode_length[done] = 0., 0

    def finish_iteration(self):
        result = {name: float(value) / self.samples for name, value in self.sums.items()}
        result["Episode/completed_count"] = len(self.completed)
        if self.completed:
            result["Episode/mean_return"] = sum(v[0] for v in self.completed) / len(self.completed)
            result["Episode/mean_length_steps"] = sum(v[1] for v in self.completed) / len(self.completed)
        return result

    def state_dict(self):
        return {"episode_return": self.episode_return.clone(), "episode_length": self.episode_length.clone()}

    def load_state_dict(self, state):
        if set(state) != {"episode_return", "episode_length"}:
            raise ValueError("metrics fields differ")
        for name, value in state.items():
            target = getattr(self, name)
            if target.shape != value.shape or target.dtype != value.dtype:
                raise ValueError("metrics layout differs")
            target.copy_(value.to(self.device))
        self.begin_iteration()


class TripodRunner(ConfinedLocomotionRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cuda:0", **kwargs):
        ppo, lidar = validate_recipe(train_cfg)
        if kwargs or log_dir is None:
            raise ValueError("explicit local log_dir and supported runner arguments required")
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("tripod training requires CUDA; CPU checks are not GPU acceptance")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.env, self.cfg, self.task_id = env, dict(train_cfg), TASK_ID
        self.num_envs, self.seed = int(env.num_envs), train_cfg["seed"]
        if self.num_envs != train_cfg["formal_num_envs"]:
            raise ValueError("constructed environment count differs")
        self.max_iterations, self.save_interval = train_cfg["max_iterations"], train_cfg["save_interval"]
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        configure_torch_backends(allow_tf32=False)
        rnn = getattr(torch.backends.cudnn, "rnn", None)
        if rnn is not None and hasattr(rnn, "fp32_precision"):
            rnn.fp32_precision = "ieee"
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.network = ConfinedLocomotionNetwork(mlp_activation=nn.ELU(), init_fn=None,
                                                envelope_mode="raw", ground_samples=None).to(self.device)
        self.provider = LegacyNormalProvider(device=self.device, generator_seed=derive_action_generator_seed(self.seed))
        self.runtime = LegacyRuntime(env, seed=self.seed, output_dir=self.log_dir / "runtime", lidar_cfg=lidar)
        self.recipe_identity = {
            "architecture": architecture_identity(lidar.far_sampling), "ppo": asdict(ppo),
            "symmetry": {"enabled": True, "mean_loss_coeff": .6},
            "execution_mode": train_cfg["execution_mode"], "scan_identity": self.runtime.scan_identity,
            "lidar_config": asdict(lidar),
            "distribution": self.provider.distribution_identity(),
            "actuator_sha256": hashlib.sha256(ACTUATOR_PATH.read_bytes()).hexdigest(),
            "env_semantics": environment_semantics(env),
            "legacy_reward_scales": {
                str(stage): reward_scales(stage) for stage in (0, 1)
            },
            "warmup_control_steps": 1, "proprio_noise_level": 1.5,
            "running_normalization": False, "lidar_perturbations": None,
            "precision": "FP32, TF32 off, cuDNN RNN IEEE, no AMP",
            "observability": {"save_interval": self.save_interval},
        }
        try:
            action = env.unwrapped.action_manager.get_term("residual_pd")
            self.joint_permutation = JointPermutation.from_joint_names(tuple(action._target_names))
            self.mount_quat_xyzw = self.runtime.pipeline._mount_q[0].detach().clone()
            self.trainer = LegacyPPOTrainer(network=self.network, distribution_provider=self.provider,
                device=self.device, config=ppo, execution_mode=train_cfg["execution_mode"],
                symmetry_enabled=True, joint_permutation=self.joint_permutation,
                mount_quat_xyzw=self.mount_quat_xyzw, seed=self.seed)
            self.learning_metrics = LegacyMetrics(self.num_envs, self.device)
            self.collector = TrainingCollector(self.runtime, self.network, self.provider,
                symmetry_enabled=True, joint_permutation=self.joint_permutation,
                mount_quat_xyzw=self.mount_quat_xyzw, transition_observer=self._observe_transition)
            self.writer = SummaryWriter(str(self.log_dir / "tensorboard"))
        except BaseException:
            self.runtime.close()
            raise
        self.current_learning_iteration = self.total_control_batches = self.total_env_samples = 0
        self._active_buffer = None
        self.phase, self.records = "fresh", []
        self.closed, self.close_error = False, None
        self.final_checkpoint = self.last_periodic_checkpoint = None
        self._official_close = env.unwrapped.close
        env.unwrapped.close = self._close_official_environment
        self.code_provenance = capture_code_provenance(__file__)
        self._write_json("run_recipe.json", {"task": TASK_ID, "seed": self.seed,
            "num_envs": self.num_envs, **self.recipe_identity,
            "code_provenance": self.code_provenance, "gpu_validation_passed": False})

    def learn(self, num_learning_iterations, init_at_random_ep_len=True):
        if (self.phase not in ("fresh", "ready") or self.closed or
                isinstance(num_learning_iterations, bool) or not isinstance(num_learning_iterations, int)
                or not 1 <= num_learning_iterations <= self.max_iterations):
            raise ValueError("learning budget/state invalid")
        if not init_at_random_ep_len:
            raise ValueError("reference training requires randomized initial episode lengths")
        target = self.current_learning_iteration + num_learning_iterations
        try:
            for _ in range(num_learning_iterations):
                self.learning_metrics.begin_iteration()
                torch.cuda.reset_peak_memory_stats(self.device)
                started = time.perf_counter()
                self.phase = "collecting"
                self._active_buffer = self.collector.collect()
                torch.cuda.synchronize(self.device)
                collected = time.perf_counter()
                self.phase = "updating"
                metrics = self.trainer.update(self._active_buffer)
                torch.cuda.synchronize(self.device)
                updated = time.perf_counter()
                # The next iteration uses the new stage. Checkpoint AFTER
                # changing it so a strict resume retains that decision.
                self.runtime.env._tripod_stage.after_update()
                self._active_buffer = None
                self.collector.clear_feature_cache()
                self.current_learning_iteration += 1
                self.total_control_batches += 24
                self.total_env_samples += 24 * self.num_envs
                self.phase = "ready"
                scalars = self.learning_metrics.finish_iteration()
                scalars.update(optimizer_scalar_summary(metrics))
                scalars.update({"Curriculum/reward_stage": self.runtime.env._tripod_stage.stage,
                    "Curriculum/max_command_x": self.runtime.env._tripod_commands.ranges["lin_vel_x"][1],
                    "Performance/collect_s": collected - started, "Performance/update_s": updated - collected,
                    "Performance/env_steps_per_s": 24 * self.num_envs / (updated - started),
                    "Memory/torch_peak_allocated_MiB": torch.cuda.max_memory_allocated(self.device) / 2**20})
                if self.current_learning_iteration % self.save_interval == 0:
                    self.last_periodic_checkpoint = str(self.save(
                        self.log_dir / f"checkpoint_full_{self.current_learning_iteration:06d}.pt"))
                for name, value in scalars.items():
                    self.writer.add_scalar(name, value, self.current_learning_iteration)
                self.writer.flush()
                with (self.log_dir / "learning_metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps({"iteration": self.current_learning_iteration, **scalars}, allow_nan=False) + "\n")
                self.records.append({"iteration": self.current_learning_iteration, **metrics})
                print(format_iteration_summary(self.current_learning_iteration, target, self.num_envs, scalars), flush=True)
            self.final_checkpoint = str(self.save(self.log_dir / f"final_checkpoint_full_{self.current_learning_iteration:06d}.pt"))
            self._write_summary()
        except BaseException as error:
            self.phase = "faulted"
            if not self.runtime._faulted:
                self.runtime._save_fault(error, phase="legacy_learn")
            raise

    def save(self, path, infos=None):
        return save_full_checkpoint(self, path, infos=infos)

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        if load_cfg is not None or strict is not True or map_location not in (None, str(self.device)):
            raise ValueError("tripod full resume is strict and device-bound")
        return load_full_checkpoint(self, path)

    def get_inference_policy(self, device=None):
        raise RuntimeError("use the tripod2-low play entry to preserve recurrent and scan clocks")
