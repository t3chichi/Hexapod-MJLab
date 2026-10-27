"""Guarded, local-only runner for the independent two-GRU task."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import time
import traceback

import torch
from torch import nn

from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.utils.torch import configure_torch_backends

from hexapod_mjlab.rl.confined_locomotion.distribution import (
    IndependentDiagonalNormalProvider, derive_action_generator_seed,
)
from hexapod_mjlab.rl.confined_locomotion.ppo import MirrorTrainingConfig
from hexapod_mjlab.rl.confined_locomotion.symmetry import JointPermutation
from hexapod_mjlab.rl.confined_locomotion_2.network import ConfinedLocomotionNetwork, TerrainMemoryNetwork
from hexapod_mjlab.rl.confined_locomotion_2.auxiliary import HeightReconstructionAuxiliary
from hexapod_mjlab.rl.confined_locomotion_2.ppo import TemporalPPOConfig, TemporalPPOTrainer
from hexapod_mjlab.common.perturbations import ObservationPerturbationCfg

from ..collector import TrainingCollector
from ..flat_env_cfg import TASK_ID
from ..runtime import TrainingRuntime
from ..scanner import ConfinedLidarCfg
from .checkpoint import architecture_identity, load_full_checkpoint, save_full_checkpoint


def checkpoint_due(iteration, interval):
    """Use completed cumulative PPO iterations, including after a resume."""
    return iteration > 0 and iteration % interval == 0


def optimizer_scalar_summary(metrics):
    """Retain full minibatch JSON while plotting concise iteration means."""
    names = {"loss": "Loss/total", "policy": "Loss/policy", "value": "Loss/value",
             "envelope": "Loss/envelope", "kl": "Optimizer/kl", "lr": "Optimizer/learning_rate"}
    values = {name: [] for name in names}
    for key, value in metrics.items():
        match = re.fullmatch(r"epoch_\d+_minibatch_\d+_(loss|policy|value|envelope|kl|lr)", key)
        if match:
            values[match.group(1)].append(float(value))
    result = {name: value for name, value in metrics.items() if "/" in name}
    result.update({names[name]: sum(items) / len(items) for name, items in values.items() if items})
    return result


def format_iteration_summary(iteration, target_iteration, num_envs, scalars):
    """Format already materialized iteration metrics without touching device state."""
    def metric(name, spec=".4f"):
        return format(scalars[name], spec) if name in scalars else "--"

    lines = [
        f"{'=' * 18} 训练轮次 {iteration}/{target_iteration} | 环境 {num_envs} {'=' * 18}",
        f"采集 {metric('Performance/collect_s', '.2f')} s | "
        f"更新 {metric('Performance/update_s', '.2f')} s | "
        f"吞吐 {metric('Performance/env_steps_per_s', '.0f')} 环境步/s",
        f"每步奖励 {metric('Reward/actual')}",
    ]
    if scalars.get("Episode/completed_count", 0) > 0:
        lines.append(
            f"完成 episode {metric('Episode/completed_count', '.0f')} | "
            f"平均回报 {metric('Episode/mean_return')} | "
            f"平均长度 {metric('Episode/mean_length_steps', '.1f')} 步")
    else:
        lines.append("完成 episode 0（本轮无结束，平均回报和长度暂缺）")
    lines.extend([
        f"损失：策略 {metric('Loss/policy')} | 价值 {metric('Loss/value')} | "
        f"包络 {metric('Loss/envelope')} | 总计 {metric('Loss/total')}",
        f"学习率 {metric('Optimizer/learning_rate', '.3e')} | "
        f"Torch 峰值显存 {metric('Memory/torch_peak_allocated_MiB', '.0f')} MiB",
    ])
    if "Height/mse_m2" in scalars:
        lines.append(f"高度：MSE {metric('Height/mse_m2', '.6f')} m² | "
                     f"加权损失 {metric('Height/weighted_loss', '.6f')}")
    if "Performance/checkpoint_s" in scalars:
        lines.append(f"周期保存 {metric('Performance/checkpoint_s', '.2f')} s")
    return "\n".join(lines)


def resolve_recipe(cfg):
    if not isinstance(cfg, dict):
        raise ValueError("runner requires an explicit recipe dictionary")
    formal = cfg.get("formal_training_enabled") is True
    engineering = cfg.get("engineering_enabled") is True
    if not formal and not engineering:
        raise ValueError("registered _2 task is guarded; use an explicit engineering/formal recipe")
    seed = cfg.get("seed")
    budget = cfg.get("max_iterations")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("explicit uint32 seed required")
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise ValueError("explicit positive iteration budget required")
    if cfg.get("num_steps_per_env") != 24 or cfg.get("clip_actions") is not None:
        raise ValueError("_2 requires 24-step rollouts and the raw-action wrapper seam")
    if cfg.get("upload_model", False):
        raise ValueError("_2 runner uses local artifacts only")
    interval = cfg.get("save_interval", 50)
    if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
        raise ValueError("save_interval must be a positive integer")
    if cfg.get("logger", "tensorboard") != "tensorboard":
        raise ValueError("this task records learning metrics locally with tensorboard")
    if engineering and budget != 1:
        raise ValueError("engineering validation is limited to one iteration")
    execution_mode = cfg.get("execution_mode", "low_memory")
    if execution_mode not in ("low_memory", "speed"):
        raise ValueError("execution_mode must be low_memory or speed")
    ppo = cfg.get("ppo_config", {})
    ppo = ppo if isinstance(ppo, TemporalPPOConfig) else TemporalPPOConfig(**ppo)
    symmetry = cfg.get("symmetry", {})
    symmetry = symmetry if isinstance(symmetry, MirrorTrainingConfig) else MirrorTrainingConfig.from_dict(symmetry)
    perturbations = cfg.get("observation_perturbations")
    if perturbations is not None and not isinstance(perturbations, ObservationPerturbationCfg):
        perturbations = ObservationPerturbationCfg.from_dict(perturbations)
    lidar_cfg = cfg.get("lidar_config", {"raycast_mode": "full", "sparse_selection_path": None})
    if isinstance(lidar_cfg, ConfinedLidarCfg):
        pass
    elif isinstance(lidar_cfg, dict):
        lidar_cfg = ConfinedLidarCfg(**lidar_cfg)
    else:
        raise ValueError("lidar_config must be a mapping or ConfinedLidarCfg")
    if formal:
        from hexapod_mjlab.common.perturbations import ALL_THREE_PRESET
        if not symmetry.enabled or symmetry.mean_loss_coeff != 0.6 or perturbations != ALL_THREE_PRESET:
            raise ValueError("formal _2 recipe retains all-three perturbations and mirror coefficient 0.6")
        count = cfg.get("formal_num_envs")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("formal_num_envs must be explicit and positive")
    terrain = cfg.get("terrain_memory")
    terrain_recipe = {}
    if terrain is not None:
        from ..terrain_geometry import HeightGridCfg
        if not isinstance(terrain, dict) or set(terrain) != {"x_range_m", "y_range_m", "height_loss_weight"}:
            raise ValueError("terrain_memory requires explicit grid bounds and height_loss_weight")
        height_grid = HeightGridCfg(tuple(terrain["x_range_m"]), tuple(terrain["y_range_m"]))
        auxiliary = HeightReconstructionAuxiliary(terrain["height_loss_weight"])
        terrain_recipe = {"height_grid": height_grid, "height_loss_weight": auxiliary.weight}
    return {"seed": seed, "budget": budget, "formal": formal,
            "engineering": engineering, "ppo": ppo, "symmetry": symmetry,
            "perturbations": perturbations, "lidar_cfg": lidar_cfg,
            "save_interval": interval, "execution_mode": execution_mode, **terrain_recipe}


class ConfinedLocomotionRunner(MjlabOnPolicyRunner):
    """Uses the official mjlab launch surface, without the stock stateless PPO."""

    def __init__(self, env, train_cfg, log_dir=None, device="cuda:0", **kwargs):
        resolved = resolve_recipe(train_cfg)  # refuse registration defaults first
        if kwargs:
            raise ValueError(f"unsupported runner options: {sorted(kwargs)}")
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("real task training requires CUDA; CPU tests are not a GPU acceptance")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if log_dir is None:
            raise ValueError("explicit log_dir required")
        self.env, self.cfg = env, dict(train_cfg)
        terrain = "height_grid" in resolved
        self.task_id = TASK_ID
        if terrain:
            from ..config.terrain_memory import TASK_ID as TERRAIN_TASK_ID
            self.task_id = TERRAIN_TASK_ID
        self.num_envs, self.seed = int(env.num_envs), resolved["seed"]
        self.max_iterations = resolved["budget"]
        self.save_interval = resolved["save_interval"]
        if resolved["engineering"] and self.num_envs > 2:
            raise ValueError("engineering validation allows at most two environments")
        if resolved["formal"] and self.num_envs != train_cfg["formal_num_envs"]:
            raise ValueError("constructed environment count differs from recipe")
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        configure_torch_backends(allow_tf32=False)
        rnn = getattr(torch.backends.cudnn, "rnn", None)
        if rnn is not None and hasattr(rnn, "fp32_precision"):
            rnn.fp32_precision = "ieee"
        torch.manual_seed(self.seed)
        network_cls = TerrainMemoryNetwork if terrain else ConfinedLocomotionNetwork
        self.network = network_cls(
            mlp_activation=nn.ELU(), init_fn=None, envelope_mode="raw",
            ground_samples=None).to(self.device)
        self.provider = IndependentDiagonalNormalProvider(
            device=self.device, generator_seed=derive_action_generator_seed(self.seed))
        self.symmetry_config = resolved["symmetry"]
        self.recipe_identity = {
            "architecture": architecture_identity(resolved["lidar_cfg"].far_sampling, terrain_memory=terrain),
            "ppo": asdict(resolved["ppo"]), "symmetry": asdict(self.symmetry_config),
            "execution_mode": resolved["execution_mode"],
            "perturbations": None if resolved["perturbations"] is None else resolved["perturbations"].identity(),
            "precision": "FP32, TF32 off, cuDNN RNN IEEE, no AMP",
            "observability": {"schema": "learning_metrics/v1", "save_interval": self.save_interval},
        }
        if terrain:
            self.recipe_identity["height_grid"] = resolved["height_grid"].identity()
            self.recipe_identity["height_loss_weight"] = resolved["height_loss_weight"]
        self.runtime = TrainingRuntime(
            env, seed=self.seed, output_dir=self.log_dir / "runtime",
            mode="training" if resolved["formal"] else "engineering",
            lidar_cfg=resolved["lidar_cfg"],
            perturbation_cfg=resolved["perturbations"],
            **({"height_grid": resolved["height_grid"]} if terrain else {}))
        # Store the resolved scanner identity, not the user's absolute path to
        # its selection file. Identical selection content remains resumable
        # after the asset is moved to another checkout or directory.
        self.recipe_identity["scan_identity"] = self.runtime.scan_identity
        try:
            from .metrics import LearningMetrics
            from hexapod_mjlab.common.mdp.rewards import plane_world_z
            self.learning_metrics = LearningMetrics(
                self.num_envs, self.device,
                **({"terrain_query": self.runtime.terrain_query} if terrain else
                   {"plane_world_z": plane_world_z(self.runtime.env)}))
            action = env.unwrapped.action_manager.get_term("residual_pd")
            self.joint_permutation = JointPermutation.from_joint_names(tuple(action._target_names))
            self.mount_quat_xyzw = self.runtime.pipeline._mount_q[0].detach().clone()
            self.trainer = TemporalPPOTrainer(
                network=self.network, distribution_provider=self.provider,
                device=self.device, config=resolved["ppo"], execution_mode=resolved["execution_mode"],
                symmetry_enabled=self.symmetry_config.enabled,
                joint_permutation=self.joint_permutation,
                mount_quat_xyzw=self.mount_quat_xyzw, seed=self.seed,
                **({"auxiliary": HeightReconstructionAuxiliary(resolved["height_loss_weight"])} if terrain else {}))
            self.collector = TrainingCollector(
                self.runtime, self.network, self.provider,
                symmetry_enabled=self.symmetry_config.enabled,
                joint_permutation=self.joint_permutation,
                mount_quat_xyzw=self.mount_quat_xyzw,
                transition_observer=self._observe_transition)
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))
        except BaseException:
            self.runtime.close()
            raise
        self.current_learning_iteration = 0
        self.total_control_batches = 0
        self.total_env_samples = 0
        self._active_buffer = None
        self.phase = "fresh"
        self.records = []
        self.closed = False
        self.close_error = None
        self.final_checkpoint = None
        self.last_periodic_checkpoint = None
        self._official_close = env.unwrapped.close
        env.unwrapped.close = self._close_official_environment
        self._write_json("run_recipe.json", {"task": self.task_id, "seed": self.seed,
                         "num_envs": self.num_envs, **self.recipe_identity,
                         "learning_effect_ready": False})

    def _write_json(self, name, data):
        (self.log_dir / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))

    def add_git_repo_to_log(self, path):
        # Official launch calls this; do not initialize remote logging.
        return None

    def _observe_transition(self, transition, output, envelope_label):
        # Diagnostic collectors may be called outside learn() by replay tests;
        # only canonical collection contributes to the learning dashboard.
        if self.phase == "collecting":
            self.learning_metrics.observe(transition, output, envelope_label)

    def learn(self, num_learning_iterations, init_at_random_ep_len=True):
        if self.phase not in ("fresh", "ready") or self.closed:
            raise RuntimeError("runner is not available for learning")
        if isinstance(num_learning_iterations, bool) or not isinstance(num_learning_iterations, int) or not 1 <= num_learning_iterations <= self.max_iterations:
            raise ValueError("learning iteration count exceeds explicit budget")
        target_iteration = self.current_learning_iteration + num_learning_iterations
        # init_at_random_ep_len is a generic launch argument. Fixed T1 reset
        # and episode lengths are deliberately retained by this task.
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
                self._active_buffer = None
                self.collector.clear_feature_cache()
                self.current_learning_iteration += 1
                self.total_control_batches += 24
                self.total_env_samples += 24 * self.num_envs
                self.phase = "ready"
                scalars = self.learning_metrics.finish_iteration()
                scalars.update(optimizer_scalar_summary(metrics))
                scalars.update({
                    "Performance/collect_s": collected - started,
                    "Performance/update_s": updated - collected,
                    "Performance/env_steps_per_s": 24 * self.num_envs / (updated - started),
                    "Memory/torch_allocated_MiB": torch.cuda.memory_allocated(self.device) / 2**20,
                    "Memory/torch_reserved_MiB": torch.cuda.memory_reserved(self.device) / 2**20,
                    "Memory/torch_peak_allocated_MiB": torch.cuda.max_memory_allocated(self.device) / 2**20,
                    "Progress/environment_transitions": self.total_env_samples,
                })
                checkpoint_started = time.perf_counter()
                if checkpoint_due(self.current_learning_iteration, self.save_interval):
                    self.last_periodic_checkpoint = str(self.save(
                        self.log_dir / f"checkpoint_full_{self.current_learning_iteration:06d}.pt"))
                    scalars["Performance/checkpoint_s"] = time.perf_counter() - checkpoint_started
                for name, value in scalars.items():
                    self.writer.add_scalar(name, value, self.current_learning_iteration)
                self.writer.flush()
                with (self.log_dir / "learning_metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps({"iteration": self.current_learning_iteration, **scalars},
                                            ensure_ascii=False, allow_nan=False) + "\n")
                record = {"iteration": self.current_learning_iteration, **metrics}
                self.records.append(record)
                with (self.log_dir / "iteration_records.jsonl").open("a") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                print(format_iteration_summary(
                    self.current_learning_iteration, target_iteration, self.num_envs, scalars), flush=True)
            final = self.log_dir / f"final_checkpoint_full_{self.current_learning_iteration:06d}.pt"
            periodic = self.log_dir / f"checkpoint_full_{self.current_learning_iteration:06d}.pt"
            if self.last_periodic_checkpoint == str(periodic) and periodic.is_file():
                # Same completed boundary: preserve the usual final filename
                # without serializing or allocating a second full checkpoint.
                os.link(periodic, final)
                self.final_checkpoint = str(final)
            else:
                self.final_checkpoint = str(self.save(final))
            self._write_summary()
        except BaseException as error:
            self.phase = "faulted"
            if not self.runtime._faulted:
                self.runtime._save_fault(error, phase="v2_learn")
            self._write_json("failure.json", {"error": repr(error), "traceback": traceback.format_exc(),
                                             "resumable": False})
            raise

    def save(self, path, infos=None):
        return save_full_checkpoint(self, path, infos=infos)

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        if load_cfg is not None or strict is not True or map_location not in (None, str(self.device)):
            raise ValueError("full _2 resume is strict and uses the original device")
        return load_full_checkpoint(self, path)

    def get_inference_policy(self, device=None):
        if self.task_id != TASK_ID:
            raise RuntimeError("Terrain Memory interactive Play is not implemented; use its finite training diagnostics")
        raise RuntimeError("use confined_locomotion_2.play.PlaySession so recurrent state and scan clocks stay aligned")

    def _write_summary(self):
        self._write_json("training_summary.json", {
            "task": self.task_id, "iteration": self.current_learning_iteration,
            "control_batches": self.total_control_batches, "env_samples": self.total_env_samples,
            "phase": self.phase, "environment_closed": self.closed,
            "close_error": self.close_error,
            "checkpoint": self.final_checkpoint, "learning_effect_ready": False,
            "save_interval": self.save_interval,
            "last_periodic_checkpoint": self.last_periodic_checkpoint,
            "last_update": self.records[-1] if self.records else None})

    def _close_official_environment(self):
        if self.closed:
            return
        error = None
        try:
            self.runtime.close()
        except BaseException as failure:
            error = failure
        try:
            self._official_close()
            self.closed = True
        except BaseException as failure:
            if error is None:
                error = failure
        try:
            writer = getattr(self, "writer", None)
            if writer is not None:
                writer.close()
        except Exception as failure:
            if error is None:
                error = failure
        if error is not None:
            self.phase = "faulted"
            self.close_error = repr(error)
        self._write_summary()
        if error is not None:
            raise error

    def close(self):
        self._close_official_environment()
