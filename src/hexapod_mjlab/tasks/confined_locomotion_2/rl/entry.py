"""Preflight explicit recipes before the official mjlab environment launch."""

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import re
import uuid

from mjlab.scripts.train import TrainConfig, launch_training

from ..config.flat_baseline import TASK_ID, confined_locomotion_formal_rl_cfg
from .checkpoint import read_checkpoint
from .runner import resolve_recipe


@dataclass(frozen=True)
class LaunchResult:
    config: TrainConfig
    run_dir: Path
    checkpoint_path: Path
    close_completed: bool


def build_formal_train_config(*, seed, num_envs, max_iterations,
                              log_root="logs/confined_locomotion_2", gpu_ids=(0,),
                              engineering_validation=False, resume_checkpoint=None,
                              raycast_mode="full", sparse_selection_path=None,
                              far_sampling="all_endpoints", save_interval=50, execution_mode="low_memory"):
    if len(gpu_ids) != 1 or isinstance(gpu_ids[0], bool) or not isinstance(gpu_ids[0], int) or gpu_ids[0] < 0:
        raise ValueError("exactly one explicit nonnegative GPU id is required")
    agent = confined_locomotion_formal_rl_cfg(
        seed=seed, num_envs=num_envs, max_iterations=max_iterations)
    agent.save_interval = save_interval
    agent.execution_mode = execution_mode
    # Let the task-local scanner own the mode/path contract, then keep the
    # serializable config as a plain mapping across mjlab's launch boundary.
    from ..scanner import ConfinedLidarCfg
    lidar_cfg = ConfinedLidarCfg(
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


def preflight_formal_train_config(cfg):
    recipe = resolve_recipe(asdict(cfg.agent))
    if not recipe["formal"] or cfg.env.scene.num_envs != cfg.agent.formal_num_envs:
        raise ValueError("formal environment/agent recipe mismatch")
    if cfg.env.seed != cfg.agent.seed:
        raise ValueError("environment/agent seed mismatch")
    if cfg.video or cfg.wandb_run_path is not None or cfg.wandb_checkpoint_name is not None:
        raise ValueError("_2 entry uses local single-GPU training without video wrappers")
    if len(cfg.gpu_ids) != 1:
        raise ValueError("_2 supports one GPU")
    return recipe


def launch_formal_training(**kwargs):
    cfg = build_formal_train_config(**kwargs)
    launch_training(TASK_ID, args=cfg)
    experiment = Path(cfg.log_root) / cfg.agent.experiment_name
    matches = [path for path in experiment.iterdir() if path.is_dir() and path.name.endswith(cfg.agent.run_name)]
    if len(matches) != 1:
        raise RuntimeError("official launch did not create exactly one matching run directory")
    candidates = list(matches[0].glob("final_checkpoint_full_*.pt"))
    if len(candidates) != 1:
        raise RuntimeError("official launch returned without one final full checkpoint")
    return LaunchResult(cfg, matches[0], candidates[0], True)
