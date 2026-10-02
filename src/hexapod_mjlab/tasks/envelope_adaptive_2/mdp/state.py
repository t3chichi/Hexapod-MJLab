"""Batched scripted state and once-per-frame supervision, independent of mjlab."""
from __future__ import annotations
import math
from typing import Optional, Tuple
import numpy as np
import torch
from .. import contracts as ea2c
from ..config.condition import ENVELOPE_SPEC
from ..attitude_replay import AttitudeReplay
from ..geometry import _hex_sample_violations, compute_hex_vertices, hex_collision_terms
from ..oracle import compute_direct_oracle_params_with_stats
from ..map_generator import generate_map
from ..path_batch import DEFAULT_MAX_CORNERS, DEFAULT_MAX_POINTS, PathBatch
from ..path_planner import PathCfg, PathData, plan_path
from ..math_utils import (assemble_observation, ego_motion, empty_range_image,
                         envelope_limit_violation, map_actions_to_params_with_target,
                         normalized_envelope_params, potential_reward, raw_action_rate_term,
                         speed_mixture_draw, wrap_to_pi, yaw_quat_from_heading)
from ..lidar import EnvelopeLidar

class EnvelopeState:
    def __init__(self, cfg, device="cuda:0", capture_debug=False):
        self.cfg, self.device, self.num_envs = cfg, device, cfg.env.num_envs
        self.dt, self.num_actions = cfg.sim.dt, 5
        self.max_episode_length = math.ceil(cfg.env.episode_length_s / self.dt)
        self._spec = ENVELOPE_SPEC
        self._envelope_low = torch.tensor(ENVELOPE_SPEC.low[:5])
        self._envelope_high = torch.tensor(ENVELOPE_SPEC.high[:5])
        self.reward_scales = vars(cfg.rewards.scales).copy()
        n=self.num_envs
        self.episode_length_buf = torch.zeros(n,device=device,dtype=torch.long)
        self.reset_buf = torch.zeros(n,device=device,dtype=torch.bool)
        self.time_out_buf = torch.zeros_like(self.reset_buf)
        self.rew_buf = torch.zeros(n,device=device)
        self.obs_buf = torch.zeros((n,190),device=device)
        self.extras = {}
        self._build_map()
        self._init_buffers()
        self.sensor = EnvelopeLidar(cfg, self.map_data, n, device, capture_debug)
        self.range_max = self.sensor.range_max
        self.range_image = self.sensor.range_image
        self.range_image_stale = self.sensor.range_image_stale
        self._physical_max = self._envelope_high_dev.clone()
        self._physical_max[4] = self._envelope_low_dev[4]
        self.oracle_params = self._physical_max.expand(n,5).clone()
        self.oracle_s = torch.ones((n,5),device=device)
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        from ..path_parallel import init_worker
        self._path_executor = None
        if cfg.path.path_plan_workers > 0:
            self._path_executor = ProcessPoolExecutor(
                max_workers=cfg.path.path_plan_workers,
                mp_context=multiprocessing.get_context("fork"), initializer=init_worker,
                initargs=(self.occupancy,self.inflated,self._make_path_cfg()))

    @property
    def logical_step(self):
        return self.common_step_counter

    @property
    def observation(self):
        return self.obs_buf

    @property
    def reward(self):
        return self.rew_buf

    def advance(self, actions):
        self.extras.pop("episode", None)
        self.actions.copy_(actions.clamp(-self.cfg.normalization.clip_actions,
                                        self.cfg.normalization.clip_actions))
        self.actions_mapped, self.actions_target = map_actions_to_params_with_target(
            self.actions,self._envelope_low_dev,self._envelope_high_dev,
            self.cfg.envelope.soft_dof_pos_limit,self.cfg.envelope.action_max)
        self.common_step_counter += 1
        self.episode_length_buf += 1
        self._resample_path_speeds()
        self._step_kinematics()
        self.sensor.scan(self.base_pos,self.heading,
                         getattr(self,"attitude_pitch",None),getattr(self,"attitude_roll",None))
        self._compute_rewards()
        self._compute_observations()
        self.executed_frame = {
            "observation": self.obs_buf.clone(), "target": self.oracle_s.clone(),
            "mapped": self.actions_mapped.clone(), "heading": self.heading.clone(),
            "pos": self.base_pos[:, :2].clone(), "collision_hard": self._collision_hard.clone(),
            "geometric_valid": self.oracle_geometric_valid.clone(),
            "supervision_valid": self.oracle_supervision_valid.clone(),
            "emergency": self.oracle_emergency.clone(),
            "rate_checked": self.oracle_rate_checked.clone(),
            "rate_valid": self.oracle_rate_valid.clone(),
            "teacher_status": self.oracle_status.copy(),
            "teacher_reason": self.oracle_reason.copy(),
        }
        self.last_actions_raw.copy_(self.actions)

    def reset(self, env_ids):
        if len(env_ids) == 0:
            return
        self.reset_idx(env_ids)
        if self._teacher is not None:
            self._update_teacher(env_ids)
        elif self._oracle_smoother is not None:
            self.oracle_s[env_ids] = self._oracle_smoother.prev_s[env_ids]
        else:
            self.oracle_s[env_ids] = 1.
        if self._teacher is None:
            self.oracle_params[env_ids] = self._physical_max
        self._compute_observations()

    def carrier_pose(self):
        from ..quaternion import quat_from_euler_xyz, quat_mul
        q = yaw_quat_from_heading(self.heading)
        if self._attitude is not None:
            q = quat_mul(q, quat_from_euler_xyz(self.attitude_roll, self.attitude_pitch,
                                               torch.zeros_like(self.heading)))
        return torch.cat((self.base_pos, q[:, [3, 0, 1, 2]]), dim=-1)

    def close(self):
        if self._path_executor is not None:
            self._path_executor.shutdown(wait=True, cancel_futures=True)
            self._path_executor = None

    def _build_map(self):
        map_cfg = ea2c.MapGenCfg(size_m=float(self.cfg.map.size_m), resolution_m=float(self.cfg.map.resolution_m), grid_shape=tuple(self.cfg.map.grid_shape), boundary_occupied=bool(self.cfg.map.boundary_occupied), ground_margin_m=float(self.cfg.map.ground_margin_m), inflation_m=float(self.cfg.map.inflation_m), inflation_cells=int(self.cfg.map.inflation_cells), n_tiles=int(self.cfg.map.n_tiles), tile_size_m=float(self.cfg.map.tile_size_m), border_size_m=float(self.cfg.map.border_size_m), min_free_component_ratio=float(self.cfg.map.min_free_component_ratio), max_gen_attempts=int(self.cfg.map.max_gen_attempts), n_validation_paths=int(self.cfg.map.n_validation_paths), min_solved_ratio=float(self.cfg.map.min_solved_ratio), path_near_obstacle_ratio=float(self.cfg.map.path_near_obstacle_ratio), near_obstacle_range=tuple(self.cfg.map.near_obstacle_range), require_constraint_primitive=bool(self.cfg.map.require_constraint_primitive))
        pillar_cfg = ea2c.PillarFieldCfg(count_min=int(self.cfg.obstacles.pillar_count_min), count_max=int(self.cfg.obstacles.pillar_count_max), size_x_min=float(self.cfg.obstacles.pillar_size_x_min), size_x_max=float(self.cfg.obstacles.pillar_size_x_max), size_y_min=float(self.cfg.obstacles.pillar_size_y_min), size_y_max=float(self.cfg.obstacles.pillar_size_y_max), height_min=float(self.cfg.obstacles.pillar_height_min), height_max=float(self.cfg.obstacles.pillar_height_max), min_separation=float(self.cfg.obstacles.pillar_min_separation), center_clear_radius=float(self.cfg.obstacles.pillar_center_clear_radius), spawn_radius=float(self.cfg.obstacles.pillar_spawn_radius), allow_height_variation=bool(self.cfg.obstacles.pillar_allow_height_variation))
        map_seed = int(getattr(self.cfg, 'seed', 42))
        self.map_data = generate_map(map_cfg, pillar_cfg, seed=map_seed)
        self.occupancy = self.map_data.occupancy
        self.inflated = self.map_data.inflated
        self.distance_field = torch.as_tensor(self.map_data.distance_field, dtype=torch.float32, device=self.device)
        from scipy.ndimage import label as _label
        from scipy.ndimage import sum as _nd_sum
        free = self.inflated == 0
        labels, n_components = _label(free)
        if n_components == 0:
            raise RuntimeError('map has no inflated-free cells')
        sizes = _nd_sum(free, labels, index=range(1, n_components + 1))
        largest_label = int(np.argmax(sizes)) + 1
        self._free_cells = np.argwhere(labels == largest_label)
        half_tile = float(self.cfg.map.size_m) / 2.0 - float(self.cfg.map.border_size_m)
        spawn_xs = ea2c.EA2_WORLD_MIN_XY + (self._free_cells[:, 1] + 0.5) * ea2c.EA2_RESOLUTION_M
        spawn_ys = ea2c.EA2_WORLD_MIN_XY + (self._free_cells[:, 0] + 0.5) * ea2c.EA2_RESOLUTION_M
        spawn_mask = (spawn_xs >= -half_tile) & (spawn_xs <= half_tile) & (spawn_ys >= -half_tile) & (spawn_ys <= half_tile)
        self._spawn_cells = self._free_cells[spawn_mask]
        if self._spawn_cells.shape[0] == 0:
            raise RuntimeError('no spawn cells inside the 4x4 tile area')


    def _init_buffers(self):
        device = self.device
        n = self.num_envs
        self._lidar_decimation = max(1, int(round(1.0 / (self.dt * float(self.cfg.lidar.update_frequency_hz)))))
        self._lidar_timer = self._lidar_decimation - 1
        self.common_step_counter = 0
        self.paths = [None] * n
        self._path_batch = PathBatch(num_envs=n, max_points=DEFAULT_MAX_POINTS, max_corners=DEFAULT_MAX_CORNERS, device=device)
        self._envelope_low_dev = self._envelope_low.to(device)
        self._envelope_high_dev = self._envelope_high.to(device)
        self.s = torch.zeros(n, dtype=torch.float32, device=device)
        self.v = torch.zeros(n, dtype=torch.float32, device=device)
        self.heading = torch.zeros(n, dtype=torch.float32, device=device)
        self.tangent = torch.zeros(n, dtype=torch.float32, device=device)
        self.tangent_rate = torch.zeros(n, dtype=torch.float32, device=device)
        self.delta_target = torch.zeros(n, dtype=torch.float32, device=device)
        self.delta_actual = torch.zeros(n, dtype=torch.float32, device=device)
        self.omega = torch.zeros(n, dtype=torch.float32, device=device)
        self.base_pos = torch.zeros(n, 3, dtype=torch.float32, device=device)
        self.height_filter = torch.zeros(n, dtype=torch.float32, device=device)
        self.height_target = torch.zeros(n, dtype=torch.float32, device=device)
        self.height_wobble_state = torch.zeros(n, dtype=torch.float32, device=device)
        self.pos_sway_state = torch.zeros(n, dtype=torch.float32, device=device)
        self.heading_sway_state = torch.zeros(n, dtype=torch.float32, device=device)
        self.pos_sway_amp = torch.zeros(n, dtype=torch.float32, device=device)
        self.heading_sway_amp = torch.zeros(n, dtype=torch.float32, device=device)
        self.height_wobble_amp = torch.zeros(n, dtype=torch.float32, device=device)
        self.actions = torch.zeros(n, self.num_actions, dtype=torch.float32, device=device)
        self.actions_mapped = torch.zeros(n, self.num_actions, dtype=torch.float32, device=device)
        self.actions_target = torch.zeros(n, self.num_actions, dtype=torch.float32, device=device)
        self.last_actions_raw = torch.zeros(n, self.num_actions, dtype=torch.float32, device=device)
        self.range_image = empty_range_image(n, ea2c.EA2_RANGE_MAX_M, device)
        self.range_image_stale = torch.ones(n, dtype=torch.bool, device=device)
        self.sensor_pos_tensor = torch.zeros(n, 3, dtype=torch.float32, device=device)
        self.sensor_quat_tensor = torch.zeros(n, 4, dtype=torch.float32, device=device)
        self.ego_motion = torch.zeros(n, 3, dtype=torch.float32, device=device)
        self._replan_reset_buf = torch.zeros(n, dtype=torch.bool, device=device)
        self._turn_in_place = torch.zeros(n, dtype=torch.bool, device=device)
        self._turn_target = torch.zeros(n, dtype=torch.float32, device=device)
        self._debug_env_ids = [int(i) for i in getattr(self.cfg.lidar, 'debug_env_ids', [0])]
        self._debug_point_stride = int(getattr(self.cfg.lidar, 'debug_point_stride', 1))
        self._debug_points = None
        self._debug_dists = None
        self._debug_is_reduced = False
        self.episode_sums = {name: torch.zeros(n, dtype=torch.float32, device=device) for name in self.reward_scales.keys()}
        self.episode_metrics = {'collision_hard_max': torch.zeros(n, dtype=torch.float32, device=device), 'raw_action_abs_mean': torch.zeros(n, dtype=torch.float32, device=device), 'envelope_limits_active_ratio': torch.zeros(n, dtype=torch.float32, device=device), 'oracle_unsafe_ratio': torch.zeros(n, dtype=torch.float32, device=device), 'oracle_unsafe_before_ratio': torch.zeros(n, dtype=torch.float32, device=device), 'oracle_potential': torch.zeros(n, dtype=torch.float32, device=device), 'oracle_mse_front_width': torch.zeros(n, dtype=torch.float32, device=device), 'oracle_mse_middle_width': torch.zeros(n, dtype=torch.float32, device=device), 'oracle_mse_back_width': torch.zeros(n, dtype=torch.float32, device=device), 'oracle_mse_forward_limit': torch.zeros(n, dtype=torch.float32, device=device), 'oracle_mse_backward_limit': torch.zeros(n, dtype=torch.float32, device=device), 'oracle_snap_ratio': torch.zeros(n, dtype=torch.float32, device=device), 'oracle_floor_pinned_ratio': torch.zeros(n, dtype=torch.float32, device=device)}
        from hexapod_mjlab.tasks.envelope_adaptive_2.target_smoother import RateLimitedOracle
        if self.cfg.envelope.oracle_kind == "legacy" and bool(getattr(self.cfg.envelope, 'target_rate_limit', True)):

            def _target_safety_check(candidate):
                viol = _hex_sample_violations(candidate, self.heading, self.base_pos[:, :2], self.distance_field, margin=float(self.cfg.envelope.margin), soft_margin=float(self.cfg.envelope.soft_margin))
                return viol.max(dim=-1).values > 0.05
            self._oracle_smoother = RateLimitedOracle(num_envs=n, dt=float(self.cfg.sim.dt), device=device, low=self._envelope_low_dev, high=self._envelope_high_dev, shrink_rate=float(self.cfg.envelope.target_shrink_rate), grow_rate=float(self.cfg.envelope.target_grow_rate), cooldown_seconds=float(self.cfg.envelope.target_cooldown_seconds), safety_check=_target_safety_check)
        else:
            self._oracle_smoother = None
        from ..teachers.batch import BatchTeacher, teacher_identity
        kind = self.cfg.envelope.oracle_kind
        if kind not in {"legacy", "exact"}:
            raise ValueError(f"Unknown oracle kind: {kind}")
        if kind == "exact" and (self.cfg.envelope.soft_dof_pos_limit != .95
                                or self.cfg.envelope.margin != .1):
            raise ValueError("Validated exact teacher requires soft limit .95 and margin .10")
        self._teacher = BatchTeacher(self.map_data.rects) if kind == "exact" else None
        self.teacher_identity = teacher_identity() if self._teacher else {"kind": "legacy", "version": "migrated_v2", "safety_certified": False}
        self.oracle_geometric_valid = torch.zeros(n, device=device, dtype=torch.bool)
        self.oracle_supervision_valid = torch.ones(n, device=device, dtype=torch.bool)
        self.oracle_emergency = torch.zeros(n, device=device, dtype=torch.bool)
        self.oracle_rate_checked = torch.zeros(n, device=device, dtype=torch.bool)
        self.oracle_rate_valid = torch.zeros(n, device=device, dtype=torch.bool)
        self.oracle_status = ["legacy_uncertified"] * n
        self.oracle_reason = [""] * n
        self.teacher_elapsed_s = 0.
        for name in ["teacher_geometry_valid_ratio", "teacher_supervision_valid_ratio", "teacher_emergency_ratio"]:
            self.episode_metrics[name] = torch.zeros(n, device=device)
        from hexapod_mjlab.tasks.envelope_adaptive_2.oracle import _physical_min_max
        _min_v, _ = _physical_min_max(self._envelope_low_dev, self._envelope_high_dev)
        self._oracle_floor_pinned_env = _min_v
        self._rng = np.random.default_rng(int(getattr(self.cfg, 'seed', 42)) + 1)
        self._speed_rng = torch.Generator(device=device)
        self._speed_rng.manual_seed(int(getattr(self.cfg, 'seed', 42)) + 2)
        if bool(getattr(self.cfg, 'attitude', None) is not None and getattr(self.cfg.attitude, 'enable', False)):
            self._attitude = AttitudeReplay(num_envs=n, device=device, dt=self.dt, max_episode_length=self.max_episode_length, source_path=self.cfg.attitude.source, cfg=self.cfg.attitude, seed=int(getattr(self.cfg, 'seed', 42)) + 7)
            self.attitude_pitch = torch.zeros(n, dtype=torch.float32, device=device)
            self.attitude_roll = torch.zeros(n, dtype=torch.float32, device=device)
        else:
            self._attitude = None

    def _sample_start_xy(self) -> np.ndarray:
        """Sample a spawn position inside the 4x4 tile area (exclude 5m border)."""
        spawn = self._spawn_cells
        if spawn.shape[0] == 0:
            raise RuntimeError('map has no spawn cells inside the tile area')
        idx = spawn[self._rng.integers(0, spawn.shape[0])]
        return np.asarray((ea2c.EA2_WORLD_MIN_XY + (idx[1] + 0.5) * ea2c.EA2_RESOLUTION_M, ea2c.EA2_WORLD_MIN_XY + (idx[0] + 0.5) * ea2c.EA2_RESOLUTION_M), dtype=np.float64)

    def _sample_goal_xy(self) -> np.ndarray:
        """Sample a goal inside the 4x4 tile area with obstacle clearance.

            Goals use the same ``_spawn_cells`` as starts so the robot does not
            navigate into the 5m outer border / open area.
            """
        free = self._spawn_cells
        if free.shape[0] == 0:
            raise RuntimeError('map has no goal cells inside the tile area')
        goal_clearance = float(self.cfg.path.goal_min_obstacle_dist)
        for _ in range(200):
            goal_idx = free[self._rng.integers(0, free.shape[0])]
            goal_xy = np.asarray((ea2c.EA2_WORLD_MIN_XY + (goal_idx[1] + 0.5) * ea2c.EA2_RESOLUTION_M, ea2c.EA2_WORLD_MIN_XY + (goal_idx[0] + 0.5) * ea2c.EA2_RESOLUTION_M), dtype=np.float64)
            if self._min_obstacle_distance_world(goal_xy) >= goal_clearance:
                return goal_xy
        raise RuntimeError('could not sample a valid goal in 200 attempts')

    def _sample_free_start_goal(self) -> 'Tuple[np.ndarray, np.ndarray]':
        """Sample a start/goal pair from the largest safe connected component."""
        return (self._sample_start_xy(), self._sample_goal_xy())

    def _min_obstacle_distance_world(self, xy) -> float:
        """Return distance to the nearest obstacle using the precomputed field.

            The map's unsigned distance field is computed once at generation time,
            so this is an O(1) lookup instead of scanning all occupied cells.
            """
        res = ea2c.EA2_RESOLUTION_M
        wmin = ea2c.EA2_WORLD_MIN_XY
        ix = int(np.floor((float(xy[0]) - wmin) / res))
        iy = int(np.floor((float(xy[1]) - wmin) / res))
        if ix < 0 or ix >= self.occupancy.shape[1] or iy < 0 or (iy >= self.occupancy.shape[0]):
            return float('inf')
        return float(self.map_data.distance_field[iy, ix])

    def _make_path_cfg(self) -> 'PathCfg':
        """Build the current :class:`PathCfg` from the environment config."""
        return PathCfg(speed_range=tuple(self.cfg.path.speed_range), resample_time_s=float(self.cfg.path.resample_time_s), delta_target_deg_range=tuple(self.cfg.path.delta_target_deg_range), omega_max=float(self.cfg.path.omega_max), k_p=float(self.cfg.path.k_p), min_turn_radius=float(self.cfg.path.min_turn_radius), resample_dist=float(self.cfg.path.resample_dist), goal_min_obstacle_dist=float(self.cfg.path.goal_min_obstacle_dist), min_path_len=float(self.cfg.path.min_path_len), noise_amp_range=tuple(self.cfg.path.noise_amp_range), noise_fc_hz=float(self.cfg.path.noise_fc_hz), noise_retries=int(self.cfg.path.noise_retries))

    def _sample_new_path(self) -> PathData:
        """Sample a feasible start/goal and plan one noisy A* path."""
        path_cfg = self._make_path_cfg()
        last_err = None
        for _ in range(40):
            try:
                start_xy, goal_xy = self._sample_free_start_goal()
                return plan_path(self.occupancy, self.inflated, start_xy, goal_xy, path_cfg, self._rng)
            except (ValueError, RuntimeError) as err:
                last_err = err
        raise RuntimeError(f'failed to plan a path after 40 attempts: {last_err}')

    def _install_path(self, env_id: 'int', path: 'Optional[PathData]') -> 'None':
        """Install a path into both the Python list and the device batch."""
        self.paths[env_id] = path
        self._path_batch.install(env_id, path)

    def _replan_from_current(self, env_id) -> None:
        """Plan a new path from the current position to a new random goal.

            Called when the robot reaches a goal.  The robot keeps its current
            pose and naturally turns toward the new path tangent on the next
            control step; no episode/GRU reset happens here.
            """
        start_xy = np.asarray(self.paths[env_id].points[-1], dtype=np.float64)
        path_cfg = self._make_path_cfg()
        for _ in range(40):
            try:
                goal_xy = self._sample_goal_xy()
                path = plan_path(self.occupancy, self.inflated, start_xy, goal_xy, path_cfg, self._rng)
                self._install_path(env_id, path)
                self.s[env_id] = 0.0
                if path.segment_dirs is not None and len(path.segment_dirs) > 0:
                    first_dir = float(path.segment_dirs[0])
                else:
                    first_dir = float(path.yaws[0])
                self.tangent[env_id] = first_dir
                self.tangent_rate[env_id] = 0.0
                self._turn_target[env_id] = first_dir
                self._turn_in_place[env_id] = True
                return
            except (ValueError, RuntimeError):
                continue
        self._replan_reset_buf[env_id] = True

    def _apply_replanned_path(self, env_id: 'int', path: 'PathData') -> 'None':
        """Install a newly planned path and enter turn-in-place."""
        self._install_path(env_id, path)
        self.s[env_id] = 0.0
        first_dir = float(path.segment_dirs[0]) if path.segment_dirs is not None and len(path.segment_dirs) > 0 else float(path.yaws[0])
        self.tangent[env_id] = first_dir
        self.tangent_rate[env_id] = 0.0
        self._turn_target[env_id] = first_dir
        self._turn_in_place[env_id] = True

    def _plan_paths_parallel(self, env_ids) -> 'dict':
        """Plan new reset paths for a batch of envs, parallel when beneficial."""
        ids = [int(e) for e in env_ids]
        threshold = int(getattr(self.cfg.path, 'path_plan_batch_threshold', 4))
        if self._path_executor is None or len(ids) < threshold:
            return {env_id: self._sample_new_path() for env_id in ids}
        from concurrent.futures import as_completed
        from ..path_parallel import plan_path_task
        results = {}
        future_map = {}
        for env_id in ids:
            try:
                start_xy, goal_xy = self._sample_free_start_goal()
            except (ValueError, RuntimeError) as err:
                print(f'[el4090_ea2] env {env_id}: start/goal sampling failed ({err}); falling back to _sample_new_path')
                results[env_id] = self._sample_new_path()
                continue
            seed = int(self._rng.integers(0, 2147483647))
            future = self._path_executor.submit(plan_path_task, seed, start_xy, goal_xy)
            future_map[future] = env_id
        for future in as_completed(future_map):
            env_id = future_map[future]
            try:
                results[env_id] = future.result()
            except Exception:
                results[env_id] = self._sample_new_path()
        return results

    def _batch_replan(self, env_ids) -> 'None':
        """Replan a batch of reached-goal envs, parallel when beneficial."""
        ids = [int(e) for e in env_ids]
        threshold = int(getattr(self.cfg.path, 'path_plan_batch_threshold', 4))
        if self._path_executor is None or len(ids) < threshold:
            for env_id in ids:
                self._replan_from_current(env_id)
            return
        from concurrent.futures import as_completed
        from ..path_parallel import plan_path_task
        future_map = {}
        for env_id in ids:
            try:
                start_xy = np.asarray(self.paths[env_id].points[-1], dtype=np.float64)
                goal_xy = self._sample_goal_xy()
            except (ValueError, RuntimeError) as err:
                print(f'[el4090_ea2] env {env_id}: replan sampling failed ({err}); falling back to _replan_from_current')
                self._replan_from_current(env_id)
                continue
            seed = int(self._rng.integers(0, 2147483647))
            future = self._path_executor.submit(plan_path_task, seed, start_xy, goal_xy)
            future_map[future] = env_id
        for future in as_completed(future_map):
            env_id = future_map[future]
            try:
                path = future.result()
                self._apply_replanned_path(env_id, path)
            except Exception:
                self._replan_from_current(env_id)

    def _reset_one_env(self, env_id: 'int', path: 'Optional[PathData]'=None):
        if path is None:
            path = self._sample_new_path()
        self._install_path(env_id, path)
        start_xy = path.points[0]
        start_yaw = float(path.segment_dirs[0]) if path.segment_dirs is not None and len(path.segment_dirs) > 0 else float(path.yaws[0])
        self.s[env_id] = 0.0
        self.base_pos[env_id, 0] = float(start_xy[0])
        self.base_pos[env_id, 1] = float(start_xy[1])
        self.heading[env_id] = start_yaw
        self.tangent[env_id] = start_yaw
        self.tangent_rate[env_id] = 0.0
        self.delta_actual[env_id] = 0.0
        self.omega[env_id] = 0.0
        if bool(getattr(self.cfg.path, 'speed_randomize', False)):
            p_zero = float(self.cfg.path.speed_p_zero)
            u_mode, u_speed = (self._rng.random(), self._rng.random())
            self.v[env_id] = float(speed_mixture_draw(torch.tensor(u_mode), torch.tensor(u_speed), p_zero, float(self.cfg.path.speed_range[1])))
        else:
            self.v[env_id] = float(self.cfg.path.speed_range[0])
        self.delta_target[env_id] = math.radians(float(self.cfg.path.delta_target_deg_range[0]))
        self.height_target[env_id] = float(self.cfg.height.min_m)
        self.height_filter[env_id] = float(self.height_target[env_id])
        self.base_pos[env_id, 2] = float(self.height_target[env_id])
        self.height_wobble_state[env_id] = 0.0
        self.pos_sway_state[env_id] = 0.0
        self.heading_sway_state[env_id] = 0.0
        self.pos_sway_amp[env_id] = float(self.cfg.sway.pos_amp_range[0])
        self.heading_sway_amp[env_id] = float(self.cfg.sway.heading_amp_range[0])
        self.height_wobble_amp[env_id] = float(self.cfg.height.wobble_amp_range[0])
        self.episode_length_buf[env_id] = 0
        self._turn_in_place[env_id] = False
        self._turn_target[env_id] = 0.0
        self._replan_reset_buf[env_id] = False
        self.reset_buf[env_id] = 1
        zero = torch.zeros_like(self.actions_mapped[env_id])
        self.actions_mapped[env_id], self.actions_target[env_id] = map_actions_to_params_with_target(zero, self._envelope_low_dev, self._envelope_high_dev, float(self.cfg.envelope.soft_dof_pos_limit), float(self.cfg.envelope.action_max))
        self.actions[env_id] = 0.0
        self.last_actions_raw[env_id] = 0.0
        self.range_image_stale[env_id] = True
        self.range_image[env_id] = self.range_max
        if self._teacher is not None:
            self._teacher.reset([env_id])
        if self._oracle_smoother is not None:
            self._oracle_smoother.reset_ids([env_id])
        if getattr(self, '_attitude', None) is not None:
            self._attitude.reset(torch.tensor([env_id], device=self.device), self.v[env_id:env_id + 1].detach())

    def _log_segment(self, env_ids):
        """Log accumulated reward sums for a set of envs and reset the sums.

            This is used at full environment resets (including the 30s timeout
            reset), independently of whether the GRU memory is cleared.
            """
        if len(env_ids) == 0:
            return
        env_ids = env_ids.to(self.device)
        ep = {}
        for name in self.reward_scales.keys():
            ep[f'rew_{name}'] = self.episode_sums[name][env_ids].mean() / max(self.max_episode_length, 1)
        for name, buf in self.episode_metrics.items():
            ep[f'ep_{name}'] = buf[env_ids].mean() / max(self.max_episode_length, 1)
        self.extras['episode'] = ep
        for name in self.reward_scales.keys():
            self.episode_sums[name][env_ids] = 0.0
        for buf in self.episode_metrics.values():
            buf[env_ids] = 0.0

    def reset_idx(self, env_ids):
        """Reset selected envs and log episode metrics."""
        if len(env_ids) == 0:
            return
        env_ids = env_ids.to(self.device)
        ids = env_ids.tolist()
        self._log_segment(env_ids)
        preplanned = self._plan_paths_parallel(ids)
        for env_id in ids:
            self._reset_one_env(env_id, path=preplanned.get(env_id))
        for i in ids:
            vx, vy, omega_out = ego_motion(self.v[i:i + 1], self.heading[i:i + 1], self.tangent[i:i + 1], self.omega[i:i + 1])
            self.ego_motion[i, 0] = vx
            self.ego_motion[i, 1] = vy
            self.ego_motion[i, 2] = omega_out
        if self.cfg.env.send_timeouts:
            self.extras['time_outs'] = self.time_out_buf

    def _resample_path_speeds(self) -> 'None':
        """Redraw path speeds on their per-env schedule (stage 2).

            Fires on every env whose ``episode_length_buf`` is a multiple of
            ``speed_resample_steps`` -- per-env phases are staggered naturally and
            restart at each episode boundary.  No-op unless
            ``cfg.path.speed_randomize`` is set.
            """
        if not bool(getattr(self.cfg.path, 'speed_randomize', False)):
            return
        due = self.episode_length_buf % int(self.cfg.path.speed_resample_steps) == 0
        n = int(due.sum())
        if n == 0:
            return
        u_mode = torch.rand(n, device=self.device, generator=self._speed_rng)
        u_speed = torch.rand(n, device=self.device, generator=self._speed_rng)
        self.v[due] = speed_mixture_draw(u_mode, u_speed, float(self.cfg.path.speed_p_zero), float(self.cfg.path.speed_range[1]))

    def _step_kinematics(self):
        """Advance each env along its reference path (batched backend)."""
        self._step_kinematics_batched()

    def _step_kinematics_batched(self):
        """Batched equivalent of :meth:`_step_kinematics` (stage 1)."""
        n = self.num_envs
        dt = self.dt
        k_p = float(self.cfg.path.k_p)
        omega_max = float(self.cfg.path.omega_max)
        height_min = float(self.cfg.height.min_m)
        valid = self._path_batch.valid
        was_turn = self._turn_in_place & valid
        zero_rate = torch.zeros(n, dtype=torch.float32, device=self.device)
        arange = torch.arange(n, device=self.device)
        if bool(was_turn.any()):
            tg = self._turn_target
            delta = wrap_to_pi(self.heading - tg)
            omega_cmd = k_p * wrap_to_pi(self.delta_target - delta)
            omega = torch.clamp(omega_cmd, -omega_max, omega_max)
            heading_new = wrap_to_pi(self.heading + omega * dt)
            delta_new = wrap_to_pi(heading_new - tg)
            aligned = delta_new.abs() < 0.05
            self.heading.copy_(torch.where(was_turn, heading_new, self.heading))
            self.omega.copy_(torch.where(was_turn, omega, self.omega))
            self.delta_actual.copy_(torch.where(was_turn, delta_new, self.delta_actual))
            self.tangent.copy_(torch.where(was_turn, tg, self.tangent))
            self.tangent_rate.copy_(torch.where(was_turn, zero_rate, self.tangent_rate))
            self._turn_in_place.copy_(was_turn & ~aligned)
            self.ego_motion[was_turn, 0] = 0.0
            self.ego_motion[was_turn, 1] = 0.0
            self.ego_motion[was_turn, 2] = omega[was_turn]
        active = valid & ~was_turn
        q_old = self._path_batch.query(self.s)
        near_corner = active & q_old.has_next_corner & (self.s + self.v * dt >= q_old.next_corner)
        advance = active & ~near_corner
        s_after = self.s + self.v * dt
        last_idx = (self._path_batch.lengths - 1).clamp(0, self._path_batch.max_points - 1)
        last = self._path_batch.arc[arange, last_idx]
        reached = advance & (s_after >= last)
        interp = advance & ~reached
        s_corner = torch.minimum(q_old.next_corner + 0.0001, last)
        sample_s = torch.where(near_corner, s_corner, s_after)
        q = self._path_batch.query(sample_s)
        self.s.copy_(torch.where(near_corner, s_corner, torch.where(advance, s_after, self.s)))
        sample = interp | near_corner
        if bool(sample.any()):
            self.base_pos[sample, 0] = q.xy[sample, 0]
            self.base_pos[sample, 1] = q.xy[sample, 1]
            self.base_pos[sample, 2] = height_min
        if bool(interp.any()):
            tang = q.tangent
            delta = wrap_to_pi(self.heading - tang)
            omega_cmd = k_p * wrap_to_pi(self.delta_target - delta)
            omega = torch.clamp(omega_cmd, -omega_max, omega_max)
            heading_new = wrap_to_pi(self.heading + omega * dt)
            delta_new = wrap_to_pi(heading_new - tang)
            self.heading.copy_(torch.where(interp, heading_new, self.heading))
            self.omega.copy_(torch.where(interp, omega, self.omega))
            self.delta_actual.copy_(torch.where(interp, delta_new, self.delta_actual))
            self.tangent.copy_(torch.where(interp, tang, self.tangent))
            self.tangent_rate.copy_(torch.where(interp, zero_rate, self.tangent_rate))
            vx = self.v * torch.cos(delta_new)
            vy = self.v * torch.sin(delta_new)
            self.ego_motion[interp, 0] = vx[interp]
            self.ego_motion[interp, 1] = vy[interp]
            self.ego_motion[interp, 2] = omega[interp]
        if bool(near_corner.any()):
            target = q_old.next_target
            not_aligned = wrap_to_pi(self.heading - target).abs() >= 0.05
            self._turn_target.copy_(torch.where(near_corner, target, self._turn_target))
            self.tangent.copy_(torch.where(near_corner, target, self.tangent))
            self.tangent_rate.copy_(torch.where(near_corner, zero_rate, self.tangent_rate))
            self.ego_motion[near_corner, 0] = 0.0
            self.ego_motion[near_corner, 1] = 0.0
            self.ego_motion[near_corner, 2] = 0.0
            self._turn_in_place.copy_(self._turn_in_place | near_corner & not_aligned)
        if bool(reached.any()):
            pending = reached.nonzero(as_tuple=False).flatten().tolist()
            self._batch_replan(pending)
        if getattr(self, '_attitude', None) is not None:
            pitch, roll, height = self._attitude.step(self.v)
            self.base_pos[:, 2] = height
            self.attitude_pitch = pitch
            self.attitude_roll = roll

    def _update_teacher(self, env_ids=None):
        ids = list(range(self.num_envs)) if env_ids is None else env_ids.tolist()
        index = torch.as_tensor(ids, device=self.device, dtype=torch.long)
        result = self._teacher.evaluate(self.base_pos[index, :2], self.heading[index],
                                        self.ego_motion[index], ids)
        self.oracle_s[index] = result['s']
        self.oracle_params[index] = result['params']
        for name in ['geometric_valid', 'supervision_valid', 'emergency', 'rate_checked', 'rate_valid']:
            getattr(self, 'oracle_' + name)[index] = result[name]
        for i, env_id in enumerate(ids):
            self.oracle_status[env_id] = result['status'][i]
            self.oracle_reason[env_id] = result['reason'][i]
        self.teacher_elapsed_s = result['elapsed_s']

    def _compute_rewards(self):
        self.rew_buf[:] = 0.0
        low = self._envelope_low_dev
        high = self._envelope_high_dev
        potential = potential_reward(self.actions_mapped, low, high)
        violation, self._collision_hard = hex_collision_terms(self.actions_mapped, self.heading, self.base_pos[:, :2], self.distance_field, margin=float(self.cfg.envelope.margin), soft_margin=float(self.cfg.envelope.soft_margin))
        act_rate = raw_action_rate_term(self.actions, self.last_actions_raw)
        limit_violation = envelope_limit_violation(self.actions_target, low, high, float(self.cfg.envelope.soft_dof_pos_limit))
        if self._teacher is not None:
            self._update_teacher()
            oracle_params = self.oracle_params
            oracle_hard_raw = (~self.oracle_geometric_valid).float()
            oracle_hard = oracle_hard_raw
            self._oracle_floor_pinned = torch.tensor([x == "minimum_infeasible" for x in self.oracle_status], device=self.device)
            smoother = None
            oracle_mse = ((normalized_envelope_params(self.actions_mapped, low, high) - self.oracle_s) ** 2).mean(-1)
            oracle_mse = torch.where(self.oracle_supervision_valid, oracle_mse, 0.)
        else:
            oracle_params, oracle_hard_raw = compute_direct_oracle_params_with_stats(self.heading, self.base_pos[:, :2], self.distance_field, low, high, margin=float(self.cfg.envelope.oracle_margin), step=float(self.cfg.envelope.oracle_step), max_dist=float(self.cfg.envelope.oracle_max_dist), soft_dof_pos_limit=float(self.cfg.envelope.soft_dof_pos_limit), interp_crossing=bool(getattr(self.cfg.envelope, 'oracle_interp_crossing', True)), group_mode=str(getattr(self.cfg.envelope, 'oracle_group_mode', 'coupled')))
            self._oracle_floor_pinned = _hex_sample_violations(self._oracle_floor_pinned_env, self.heading, self.base_pos[:, :2], self.distance_field, margin=float(self.cfg.envelope.margin), soft_margin=float(self.cfg.envelope.soft_margin)).max(dim=-1).values > 0.05
            smoother = getattr(self, '_oracle_smoother', None)
            if smoother is not None:
                oracle_params = smoother.update(oracle_params)
            oracle_mse = ((normalized_envelope_params(self.actions_mapped, low, high) - normalized_envelope_params(oracle_params, low, high)) ** 2).mean(dim=-1)
            _, oracle_hard = hex_collision_terms(oracle_params, self.heading, self.base_pos[:, :2], self.distance_field, margin=float(self.cfg.envelope.margin), soft_margin=float(self.cfg.envelope.soft_margin))
        terms = {'potential': potential, 'collision': violation, 'action_rate': act_rate, 'envelope_limits': limit_violation, 'oracle_mse': oracle_mse}
        for name, scale in self.reward_scales.items():
            rew = terms[name] * scale
            self.rew_buf += rew
            self.episode_sums[name] += rew
        self.episode_metrics['collision_hard_max'] += self._collision_hard
        self.episode_metrics['raw_action_abs_mean'] += self.actions.abs().mean(dim=-1)
        self.episode_metrics['envelope_limits_active_ratio'] += (limit_violation > 0).to(torch.float32)
        self.episode_metrics['oracle_unsafe_before_ratio'] += (oracle_hard_raw > 0).to(torch.float32)
        self.episode_metrics['oracle_unsafe_ratio'] += (oracle_hard > 0).to(torch.float32)
        self.episode_metrics['oracle_potential'] += potential_reward(oracle_params, low, high)
        oracle_sq = (normalized_envelope_params(self.actions_mapped, low, high) - normalized_envelope_params(oracle_params, low, high)) ** 2
        oracle_sq = torch.where(self.oracle_supervision_valid[:, None], oracle_sq, 0.)
        for j, metric_name in enumerate(('oracle_mse_front_width', 'oracle_mse_middle_width', 'oracle_mse_back_width', 'oracle_mse_forward_limit', 'oracle_mse_backward_limit')):
            self.episode_metrics[metric_name] += oracle_sq[:, j]
        if smoother is not None:
            self.episode_metrics['oracle_snap_ratio'] += smoother.snapped.to(torch.float32)
        self.episode_metrics['oracle_floor_pinned_ratio'] += self._oracle_floor_pinned.to(torch.float32)
        self.episode_metrics['teacher_geometry_valid_ratio'] += self.oracle_geometric_valid.float()
        self.episode_metrics['teacher_supervision_valid_ratio'] += self.oracle_supervision_valid.float()
        self.episode_metrics['teacher_emergency_ratio'] += self.oracle_emergency.float()
        if self._teacher is None:
            self.oracle_params.copy_(oracle_params)
            self.oracle_s.copy_(normalized_envelope_params(oracle_params, low, high))


    def _compute_observations(self):
        self.obs_buf[:] = assemble_observation(self.range_image, self.ego_motion, max_range=self.range_max)
        self.obs_buf.clamp_(-self.cfg.normalization.clip_observations,
                           self.cfg.normalization.clip_observations)
