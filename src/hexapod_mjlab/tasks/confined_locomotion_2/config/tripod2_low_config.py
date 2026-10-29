"""Task-owned Tripod2 Low runner configuration."""
from dataclasses import dataclass, field

from mjlab.rl import RslRlBaseRunnerCfg

from hexapod_mjlab.rl.confined_locomotion_2.ppo import TemporalPPOConfig
from hexapod_mjlab.rl.confined_locomotion_2.training_rollout import DEFAULT_POINT_REPLAY_CHUNK
from ..components.lidar.scanner import training_lidar_defaults


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
    point_replay_chunk: int = DEFAULT_POINT_REPLAY_CHUNK
    ppo_config: TemporalPPOConfig = field(default_factory=TemporalPPOConfig)
    lidar_config: dict = field(default_factory=training_lidar_defaults)
