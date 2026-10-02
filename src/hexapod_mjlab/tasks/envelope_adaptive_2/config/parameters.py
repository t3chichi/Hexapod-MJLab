"""EA2 defaults frozen from source; instance dataclasses replace old config inheritance."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import math
ASSET_DIR = Path(__file__).resolve().parents[1] / "assets"

@dataclass
class EnvCfg:
    num_envs: int = 1024
    num_observations: int = 190
    num_privileged_obs: int | None = None
    num_actions: int = 5
    env_spacing: float = 0.0
    episode_length_s: float = 45.0
    send_timeouts: bool = True

@dataclass
class SimCfg:
    dt: float = 0.02
    substeps: int = 1
    gravity: list = field(default_factory=lambda: [0.0, 0.0, -9.81])

@dataclass
class MapCfg:
    size_m: float = 74.0
    resolution_m: float = 0.1
    grid_shape: list = field(default_factory=lambda: [740, 740])
    boundary_occupied: bool = True
    ground_margin_m: float = 2.0
    inflation_m: float = 0.4
    inflation_cells: int = 4
    n_tiles: int = 4
    tile_size_m: float = 16.0
    border_size_m: float = 5.0
    min_free_component_ratio: float = 0.95
    max_gen_attempts: int = 20
    n_validation_paths: int = 12
    min_solved_ratio: float = 0.8
    path_near_obstacle_ratio: float = 0.3
    near_obstacle_range: list = field(default_factory=lambda: [0.7, 1.5])
    require_constraint_primitive: bool = False

@dataclass
class ObstaclesCfg:
    pillar_count_min: int = 24
    pillar_count_max: int = 24
    pillar_size_x_min: float = 0.5
    pillar_size_x_max: float = 4.0
    pillar_size_y_min: float = 0.5
    pillar_size_y_max: float = 4.0
    pillar_height_min: float = 1.0
    pillar_height_max: float = 2.0
    pillar_min_separation: float = 2.6
    pillar_center_clear_radius: float = 1.0
    pillar_spawn_radius: float = 8.5
    pillar_allow_height_variation: bool = True

@dataclass
class PathCfg:
    speed_range: list = field(default_factory=lambda: [1.0, 1.0])
    resample_time_s: float = 4.0
    delta_target_deg_range: list = field(default_factory=lambda: [0.0, 0.0])
    omega_max: float = 1.5
    k_p: float = 5.0
    min_turn_radius: float = 1.0
    resample_dist: float = 0.2
    goal_min_obstacle_dist: float = 0.5
    min_path_len: float = 3.0
    noise_amp_range: list = field(default_factory=lambda: [0.0, 0.0])
    noise_fc_hz: float = 1.0
    noise_retries: int = 8
    speed_randomize: bool = True
    speed_resample_steps: int = 150
    speed_p_zero: float = 0.02
    path_plan_workers: int = 8
    path_plan_batch_threshold: int = 4

@dataclass
class SwayCfg:
    pos_amp_range: list = field(default_factory=lambda: [0.0, 0.0])
    heading_amp_range: list = field(default_factory=lambda: [0.0, 0.0])
    fc_hz: float = 1.0

@dataclass
class HeightCfg:
    min_m: float = 0.52
    max_m: float = 0.52
    resample_time_s: float = 4.0
    tau_s: float = 0.8
    wobble_amp_range: list = field(default_factory=lambda: [0.0, 0.0])
    wobble_fc_hz: float = 1.0

@dataclass
class LidarCfg:
    airy_n_azimuth: int = 900
    airy_n_elevation: int = 96
    airy_horizontal_resolution_deg: float = 0.4
    airy_vertical_fov_deg: list = field(default_factory=lambda: [0, 90.0])
    far_plane: float = 60.0
    min_range: float = 0.2
    update_frequency_hz: float = 10.0
    effective_max_range: float = 3.2
    use_reduced_raycast: bool = True
    offset_pos: list = field(default_factory=lambda: [0.7, 0.0, -0.05])
    sensor_offset_rpy: list = field(default_factory=lambda: [0.0, math.pi / 2.0 + 0.1, 0.0])
    pointcloud_in_world_frame: bool = False
    randomize_placement: bool = False
    debug_env_ids: list = field(default_factory=lambda: [0])
    debug_point_stride: int = 1
    debug_geom_budget: int = 4096
    enable_sensor_noise: bool = True
    pixel_std_dev_multiplier: float = 0.02
    pixel_dropout_prob: float = 0.02
    random_distance_noise: float = 0.0
    random_angle_noise: float = 0.0

@dataclass
class AttitudeCfg:
    enable: bool = False
    source: str = str(ASSET_DIR / "attitude_traj_source.npz")
    ratio_tau: float = 0.5
    trim_range: list = field(default_factory=lambda: [-2.0, 2.0])
    scale_range: list = field(default_factory=lambda: [0.85, 1.2])
    rate_range: list = field(default_factory=lambda: [0.9, 1.1])
    ar_std: float = 0.3
    fade_steps: int = 100
    std_law_k: float = 0.858
    std_law_b: float = 0.318

@dataclass
class EnvelopeCfg:
    margin: float = 0.1
    soft_margin: float = 0.1
    action_max: float = 4.0
    soft_dof_pos_limit: float = 0.95
    oracle_margin: float = 0.2
    oracle_step: float = 0.05
    oracle_max_dist: float = 5.0
    oracle_interp_crossing: bool = True
    target_rate_limit: bool = True
    target_shrink_rate: float = 2.0
    target_grow_rate: float = 0.5
    target_cooldown_seconds: float = 0.2
    oracle_group_mode: str = 'axis'
    debug_color: tuple = (0.0, 0.85, 1.0)
    debug_line_radius: float = 0.012
    debug_line_samples: int = 8
    debug_ground_z_offset: float = 0.02

@dataclass
class RewardsCfgScales:
    potential: float = 0.0
    collision: float = 0.0
    action_rate: float = -0.01
    envelope_limits: float = -0.8
    oracle_mse: float = -3.0

@dataclass
class RewardsCfg:
    scales: RewardsCfgScales = field(default_factory=RewardsCfgScales)

@dataclass
class NormalizationCfg:
    clip_observations: float = 100.0
    clip_actions: float = 100.0

@dataclass
class EA2Cfg:
    seed: int = 1
    env: EnvCfg = field(default_factory=EnvCfg)
    sim: SimCfg = field(default_factory=SimCfg)
    map: MapCfg = field(default_factory=MapCfg)
    obstacles: ObstaclesCfg = field(default_factory=ObstaclesCfg)
    path: PathCfg = field(default_factory=PathCfg)
    sway: SwayCfg = field(default_factory=SwayCfg)
    height: HeightCfg = field(default_factory=HeightCfg)
    lidar: LidarCfg = field(default_factory=LidarCfg)
    attitude: AttitudeCfg = field(default_factory=AttitudeCfg)
    envelope: EnvelopeCfg = field(default_factory=EnvelopeCfg)
    rewards: RewardsCfg = field(default_factory=RewardsCfg)
    normalization: NormalizationCfg = field(default_factory=NormalizationCfg)
