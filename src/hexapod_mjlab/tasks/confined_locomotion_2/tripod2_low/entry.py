"""Explicit-budget single-GPU launch for the independent migrated recipe."""
from dataclasses import asdict, replace
from pathlib import Path
import re
import uuid

from mjlab.scripts.train import TrainConfig, launch_training

from ..rl.launch import (
    LaunchResult,
    build_lidar_config,
    find_single_run_and_final_checkpoint,
    require_gpu_id,
)
from .checkpoint import read_checkpoint
from .env import TASK_ID, make_env_cfg
from .runner import TripodRlCfg, validate_recipe


def build_train_config(*, seed, num_envs, max_iterations, gpu=0,
                       log_root="logs/confined_locomotion_2", save_interval=50,
                       execution_mode="low_memory", engineering_validation=False,
                       resume_checkpoint=None, raycast_mode="full",
                       sparse_selection_path=None, far_sampling="all_endpoints"):
    require_gpu_id(gpu)
    if not isinstance(engineering_validation, bool):
        raise TypeError("engineering_validation must be bool")
    lidar = build_lidar_config(raycast_mode=raycast_mode, sparse_selection_path=sparse_selection_path,
                           far_sampling=far_sampling)
    agent = TripodRlCfg(seed=seed, formal_num_envs=num_envs, max_iterations=max_iterations,
                       save_interval=save_interval, execution_mode=execution_mode,
                       engineering_validation=engineering_validation, lidar_config=asdict(lidar),
                       run_name="tripod_" + uuid.uuid4().hex)
    validate_recipe(asdict(agent))
    cfg = TrainConfig(env=make_env_cfg(num_envs=num_envs, seed=seed), agent=agent,
                      log_root=str(Path(log_root).expanduser().resolve()), gpu_ids=[gpu])
    if resume_checkpoint is not None:
        checkpoint = Path(resume_checkpoint).expanduser().resolve()
        saved = read_checkpoint(checkpoint, expected_far_sampling=far_sampling)
        if saved["identity"]["seed"] != seed or saved["identity"]["num_envs"] != num_envs:
            raise ValueError("resume seed/environment count differs")
        if saved["runtime"]["mode"] != "training":
            raise ValueError("training resume requires a training checkpoint")
        agent.resume = True
        agent.experiment_name = checkpoint.parent.parent.name
        agent.load_run = "^" + re.escape(checkpoint.parent.name) + "$"
        agent.load_checkpoint = "^" + re.escape(checkpoint.name) + "$"
        cfg = replace(cfg, log_root=str(checkpoint.parent.parent.parent))
    return cfg


def launch(**kwargs):
    cfg = build_train_config(**kwargs)
    launch_training(TASK_ID, args=cfg)
    run_dir, checkpoint = find_single_run_and_final_checkpoint(
        cfg.log_root, cfg.agent.experiment_name, cfg.agent.run_name,
        run_error="expected one official tripod run directory",
        checkpoint_error="training returned without one full final checkpoint",
    )
    return LaunchResult(cfg, run_dir, checkpoint, True)
