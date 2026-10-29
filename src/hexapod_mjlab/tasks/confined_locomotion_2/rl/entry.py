"""Preflight explicit recipes before the official mjlab environment launch."""

from dataclasses import asdict, replace
from pathlib import Path
import re
import uuid

from mjlab.scripts.train import TrainConfig, launch_training

from ..config.flat_baseline_config import confined_locomotion_formal_rl_cfg
from ..flat_env_cfg import TASK_ID
from .checkpoint import read_checkpoint
from hexapod_mjlab.rl.confined_locomotion_2.training_rollout import DEFAULT_POINT_REPLAY_CHUNK
from .launch import (
    LaunchResult,
    build_lidar_config,
    find_single_run_and_final_checkpoint,
    preflight_formal_train_config,
    require_single_gpu,
)


def build_formal_train_config(*, seed, num_envs, max_iterations,
                              log_root="logs/confined_locomotion_2", gpu_ids=(0,),
                              engineering_validation=False, resume_checkpoint=None,
                              raycast_mode="sparse", sparse_selection_path=None,
                              far_sampling="hit_only_10m", save_interval=50, execution_mode="low_memory",
                              point_replay_chunk=DEFAULT_POINT_REPLAY_CHUNK):
    require_single_gpu(gpu_ids)
    agent = confined_locomotion_formal_rl_cfg(
        seed=seed, num_envs=num_envs, max_iterations=max_iterations)
    agent.save_interval = save_interval
    agent.execution_mode = execution_mode
    agent.point_replay_chunk = point_replay_chunk
    # Let the task-local scanner own the mode/path contract, then keep the
    # serializable config as a plain mapping across mjlab's launch boundary.
    lidar_cfg = build_lidar_config(
        raycast_mode=raycast_mode,
        sparse_selection_path=sparse_selection_path,
        far_sampling=far_sampling,
    )
    agent.lidar_config = asdict(lidar_cfg)
    if not isinstance(engineering_validation, bool):
        raise TypeError("engineering_validation must be bool")
    if engineering_validation:
        if num_envs > 2 or max_iterations != 1:
            raise ValueError("engineering validation is limited to <=2 envs and one iteration")
        agent.engineering_enabled = True
    agent.run_name = f"formal_{uuid.uuid4().hex}"
    cfg = replace(TrainConfig.from_task(TASK_ID), agent=agent,
                  log_root=str(Path(log_root).expanduser()), gpu_ids=list(gpu_ids))
    cfg.env.seed = seed
    cfg.env.scene.num_envs = num_envs
    if resume_checkpoint is not None:
        target = Path(resume_checkpoint).expanduser().resolve()
        payload = read_checkpoint(target, expected_far_sampling=lidar_cfg.far_sampling)
        if payload["identity"]["seed"] != seed or payload["identity"]["num_envs"] != num_envs:
            raise ValueError("resume seed/environment count differs")
        if payload["runtime"]["mode"] != "training":
            raise ValueError("formal resume requires a checkpoint from training mode")
        agent.experiment_name = target.parent.parent.name
        agent.load_run = "^" + re.escape(target.parent.name) + "$"
        agent.load_checkpoint = "^" + re.escape(target.name) + "$"
        agent.resume = True
        cfg = replace(cfg, log_root=str(target.parent.parent.parent))
    preflight_formal_train_config(cfg)
    return cfg


def launch_formal_training(**kwargs):
    cfg = build_formal_train_config(**kwargs)
    launch_training(TASK_ID, args=cfg)
    run_dir, checkpoint = find_single_run_and_final_checkpoint(
        cfg.log_root, cfg.agent.experiment_name, cfg.agent.run_name,
        run_error="official launch did not create exactly one matching run directory",
        checkpoint_error="official launch returned without one final full checkpoint",
    )
    return LaunchResult(cfg, run_dir, checkpoint, True)
