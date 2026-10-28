"""Shared launch helpers for the confined_locomotion_2 task family.

Only task-agnostic plumbing lives here: result shape, single-GPU validation,
lidar config construction, official run/checkpoint discovery and the formal
preflight shared by Flat and Terrain. Each task entry keeps its own env/RL
recipe, checkpoint reader, resume wiring, log-root handling and UUID run name.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..components.lidar.scanner import ConfinedLidarCfg
from .runner import resolve_recipe

if TYPE_CHECKING:
    from mjlab.scripts.train import TrainConfig

__all__ = [
    "LaunchResult",
    "build_lidar_config",
    "find_single_run_and_final_checkpoint",
    "preflight_formal_train_config",
    "require_gpu_id",
    "require_single_gpu",
]


@dataclass(frozen=True)
class LaunchResult:
    config: TrainConfig
    run_dir: Path
    checkpoint_path: Path
    close_completed: bool


def require_single_gpu(gpu_ids) -> int:
    if len(gpu_ids) != 1 or isinstance(gpu_ids[0], bool) or not isinstance(gpu_ids[0], int) or gpu_ids[0] < 0:
        raise ValueError("exactly one explicit nonnegative GPU id is required")
    return gpu_ids[0]


def require_gpu_id(gpu) -> int:
    if isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0:
        raise ValueError("gpu must be a nonnegative integer")
    return gpu


def build_lidar_config(*, raycast_mode, sparse_selection_path, far_sampling) -> ConfinedLidarCfg:
    return ConfinedLidarCfg(
        raycast_mode=raycast_mode,
        sparse_selection_path=sparse_selection_path,
        far_sampling=far_sampling,
    )


def find_single_run_and_final_checkpoint(
    log_root,
    experiment_name,
    run_name,
    *,
    run_error,
    checkpoint_error,
) -> tuple[Path, Path]:
    experiment = Path(log_root) / experiment_name
    matches = [path for path in experiment.iterdir() if path.is_dir() and path.name.endswith(run_name)]
    if len(matches) != 1:
        raise RuntimeError(run_error)
    checkpoints = list(matches[0].glob("final_checkpoint_full_*.pt"))
    if len(checkpoints) != 1:
        raise RuntimeError(checkpoint_error)
    return matches[0], checkpoints[0]


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
