"""Explicit Terrain Memory launch using the existing runner and strict checkpoints."""
from dataclasses import asdict, replace
from pathlib import Path
import re
import uuid

from mjlab.scripts.train import TrainConfig, launch_training

from ..config.terrain_memory_config import TASK_ID, make_training_env_cfg, terrain_memory_formal_rl_cfg
from .checkpoint import read_checkpoint
from hexapod_mjlab.rl.confined_locomotion_2.training_rollout import DEFAULT_POINT_REPLAY_CHUNK
from .launch import (
    LaunchResult,
    build_lidar_config,
    find_single_run_and_final_checkpoint,
    preflight_formal_train_config,
    require_single_gpu,
)


def build_terrain_train_config(*, seed, num_envs, max_iterations, height_loss_weight,
                               log_root="logs/confined_locomotion_2", gpu_ids=(0,),
                               engineering_validation=False, resume_checkpoint=None,
                               raycast_mode="sparse", sparse_selection_path=None,
                               far_sampling="hit_only_10m", save_interval=50, execution_mode="low_memory",
                               point_replay_chunk=DEFAULT_POINT_REPLAY_CHUNK):
    require_single_gpu(gpu_ids)
    if not isinstance(engineering_validation,bool):
        raise TypeError("engineering_validation must be bool")
    if engineering_validation and (num_envs>2 or max_iterations!=1):
        raise ValueError("engineering validation is limited to <=2 envs and one iteration")
    agent=terrain_memory_formal_rl_cfg(seed=seed,num_envs=num_envs,max_iterations=max_iterations,
                                     height_loss_weight=height_loss_weight)
    agent.engineering_enabled=engineering_validation
    agent.save_interval=save_interval
    agent.execution_mode=execution_mode
    agent.point_replay_chunk=point_replay_chunk
    lidar=build_lidar_config(raycast_mode=raycast_mode,sparse_selection_path=sparse_selection_path,
                          far_sampling=far_sampling)
    agent.lidar_config=asdict(lidar)
    agent.run_name=f"terrain_{uuid.uuid4().hex}"
    cfg=TrainConfig(env=make_training_env_cfg(num_envs=num_envs,seed=seed),agent=agent,
                    log_root=str(Path(log_root).expanduser()),gpu_ids=list(gpu_ids))
    if resume_checkpoint is not None:
        target=Path(resume_checkpoint).expanduser().resolve()
        payload=read_checkpoint(target,expected_far_sampling=far_sampling,expected_task_id=TASK_ID)
        if payload['identity']['seed']!=seed or payload['identity']['num_envs']!=num_envs:
            raise ValueError("resume seed/environment count differs")
        if payload['runtime']['mode']!='training':
            raise ValueError("Terrain Memory entry requires a training-mode checkpoint")
        agent.experiment_name=target.parent.parent.name
        agent.load_run='^'+re.escape(target.parent.name)+'$'
        agent.load_checkpoint='^'+re.escape(target.name)+'$'
        agent.resume=True
        cfg=replace(cfg,log_root=str(target.parent.parent.parent))
    preflight_formal_train_config(cfg)
    return cfg


def launch_terrain_training(**kwargs):
    cfg=build_terrain_train_config(**kwargs)
    launch_training(TASK_ID,args=cfg)
    run_dir, checkpoint = find_single_run_and_final_checkpoint(
        cfg.log_root, cfg.agent.experiment_name, cfg.agent.run_name,
        run_error="official launch did not create exactly one Terrain Memory run",
        checkpoint_error="Terrain Memory launch returned without a final full checkpoint",
    )
    return LaunchResult(cfg, run_dir, checkpoint, True)
