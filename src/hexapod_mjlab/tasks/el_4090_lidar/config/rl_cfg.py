"""RL configuration for the EL4090 LiDAR task (LidarPPO5x wiring).

Extends the official ``RslRlOnPolicyRunnerCfg`` dataclasses so that
``asdict(cfg.agent)`` (train.py) produces the exact dict contract that
``LidarPPO5x.construct_algorithm`` consumes:

* ``actor`` carries the LidarPD actor constants (proximal/distal points,
  dims, proprio dim) alongside the hidden dims ([512, 256, 128], old policy);
* ``algorithm`` carries the legacy extras (``aux_loss_coef``, ``use_amp``,
  ``symmetry_cfg`` incl. ``symmetry_kwargs``) plus
  ``class_name="hexapod_mjlab.rl.ppo_lidar:LidarPPO5x"``;
* ``obs_groups`` maps actor/critic to the env group ``policy``
  (2754-dim); the privileged ``height_grid`` group reaches the algorithm
  through the obs TensorDict directly;
* ``multi_gpu: None`` is required by the LidarPPO5x constructor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

from hexapod_mjlab.tasks.el_4090_lidar.mdp.lidar import (
  DISTAL_HISTORY,
  DISTAL_POINTS,
  LIDAR_MAX_RANGE,
  POOL_NUM_POINTS,
  PROXIMAL_POINTS,
)
from hexapod_mjlab.tasks.el_4090_lidar.mdp.observations import (
  HEIGHT_GRID_DIM,
  MEASURED_GRID_X_COUNT,
  MEASURED_GRID_Y_COUNT,
  PROPRIO_DIM,
)

NUM_DOF = 18


@dataclass
class El4090LidarActorCfg(RslRlModelCfg):
  """LidarPD actor model config (old El4090LidarHeadCfgPPO.policy)."""

  proximal_points: int = PROXIMAL_POINTS
  distal_history_length: int = DISTAL_HISTORY
  distal_points: int = DISTAL_POINTS
  proximal_feature_dim: int = HEIGHT_GRID_DIM  # 187 (old proximal GRU size)
  distal_feature_dim: int = DISTAL_POINTS
  proprio_obs_dim: int = PROPRIO_DIM
  privileged_height_dim: int = HEIGHT_GRID_DIM
  gradient_checkpointing_proximal: bool = False
  gradient_checkpointing_distal: bool = True
  init_noise_std: float = 1.0
  noise_std_type: str = "scalar"


@dataclass
class El4090LidarPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """LidarPPO5x algorithm config (old algorithm + symmetry block)."""

  class_name: str = "hexapod_mjlab.rl.ppo_lidar:LidarPPO5x"
  aux_loss_coef: float = 1.0
  use_amp: bool = True
  symmetry_cfg: dict[str, Any] = field(
    default_factory=lambda: {
      "use_data_augmentation": True,
      "use_mirror_loss": True,
      "mirror_loss_coeff": 1.0,
      # Legacy runner sensor-binding convention: consumed by
      # LidarPPO5x.construct_algorithm, which binds the mount tensors from
      # the env and injects them into the mirror function.
      "symmetry_kwargs": dict(
        proprio_dim=PROPRIO_DIM,
        proximal_points=PROXIMAL_POINTS,
        distal_history_points=DISTAL_HISTORY * DISTAL_POINTS,  # 640
        num_dof=NUM_DOF,
        height_grid_x_count=MEASURED_GRID_X_COUNT,
        height_grid_y_count=MEASURED_GRID_Y_COUNT,
      ),
    }
  )
  num_mini_batches: int = 8


@dataclass
class El4090LidarRunnerCfg(RslRlOnPolicyRunnerCfg):
  """Runner config for the EL4090 LiDAR task."""

  class_name: str = "El4090LidarRunner"
  obs_groups: dict[str, tuple[str, ...]] = field(
    default_factory=lambda: {
      # LidarPD actor/critic consume the flat policy vector (2754); the
      # privileged height_grid reaches update() through the obs TensorDict.
      "actor": ("policy",),
      "critic": ("policy",),
    }
  )
  actor: El4090LidarActorCfg = field(
    default_factory=lambda: El4090LidarActorCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=False,
      distribution_cfg=None,
    )
  )
  critic: RslRlModelCfg = field(
    default_factory=lambda: RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=False,
      distribution_cfg=None,
    )
  )
  algorithm: El4090LidarPpoAlgorithmCfg = field(
    default_factory=El4090LidarPpoAlgorithmCfg
  )
  multi_gpu: dict | None = None
  clip_actions: float | None = 100.0  # old normalization.clip_actions
  num_steps_per_env: int = 24
  max_iterations: int = 4000
  save_interval: int = 50
  experiment_name: str = "el_4090_lidar"
  logger: str = "tensorboard"
  upload_model: bool = False


def el_4090_lidar_ppo_runner_cfg() -> El4090LidarRunnerCfg:
  """Factory for the registered task's rl_cfg."""
  return El4090LidarRunnerCfg()
