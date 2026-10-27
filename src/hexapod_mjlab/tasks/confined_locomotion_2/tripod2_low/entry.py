"""Explicit-budget single-GPU launch for the independent migrated recipe."""
from dataclasses import asdict, replace
from pathlib import Path
import re
import uuid

from mjlab.scripts.train import TrainConfig, launch_training

from ..rl.entry import LaunchResult
from ..scanner import ConfinedLidarCfg
from .checkpoint import read_checkpoint
from .env import TASK_ID, make_env_cfg
from .runner import TripodRlCfg, validate_recipe


def build_train_config(*, seed, num_envs, max_iterations, gpu=0,
                       log_root="logs/confined_locomotion_2", save_interval=50,
                       execution_mode="low_memory", engineering_validation=False,
                       resume_checkpoint=None, raycast_mode="full",
                       sparse_selection_path=None, far_sampling="all_endpoints"):
    if isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0:
        raise ValueError("gpu must be a nonnegative integer")
    if not isinstance(engineering_validation, bool):
        raise TypeError("engineering_validation must be bool")
    lidar = ConfinedLidarCfg(raycast_mode=raycast_mode, sparse_selection_path=sparse_selection_path,
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
    runs = [p for p in (Path(cfg.log_root) / cfg.agent.experiment_name).iterdir()
            if p.is_dir() and p.name.endswith(cfg.agent.run_name)]
    if len(runs) != 1:
        raise RuntimeError("expected one official tripod run directory")
    checkpoints = list(runs[0].glob("final_checkpoint_full_*.pt"))
    if len(checkpoints) != 1:
        raise RuntimeError("training returned without one full final checkpoint")
    return LaunchResult(cfg, runs[0], checkpoints[0], True)
