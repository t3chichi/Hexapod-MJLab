"""Airy raycasting with an explicit 10 Hz clock and packaged fixed channels."""
from __future__ import annotations
import numpy as np
import torch
from . import contracts as ea2c
from .math_utils import yaw_quat_from_heading, empty_range_image, refresh_range_image_from_scan
from .range_image import build_selected_range_image, extract_selected_range_image

class EnvelopeLidar:
    def __init__(self, cfg, map_data, num_envs, device, capture_debug=False):
        self.cfg, self.map_data = cfg, map_data
        self.num_envs, self.device, self.dt = num_envs, device, cfg.sim.dt
        self.capture_debug = capture_debug
        self._debug_env_ids = [i for i in cfg.lidar.debug_env_ids if 0 <= i < num_envs]
        self._lidar_decimation = max(1, round(1 / (self.dt * cfg.lidar.update_frequency_hz)))
        self._lidar_timer = self._lidar_decimation - 1
        self.range_image_stale = torch.ones(num_envs, device=device, dtype=torch.bool)
        self.sensor_pos_tensor = torch.zeros((num_envs,3), device=device)
        self.sensor_quat_tensor = torch.zeros((num_envs,4), device=device)
        self._attitude = None
        self._debug_points = self._debug_dists = None
        self._debug_is_reduced = cfg.lidar.use_reduced_raycast
        self._init_lidar()

    def scan(self, base_pos, heading, pitch=None, roll=None):
        self.base_pos, self.heading = base_pos, heading
        self._attitude = pitch
        self.attitude_pitch, self.attitude_roll = pitch, roll
        self._update_lidar()

    def reset(self, env_ids):
        self.range_image_stale[env_ids] = True
        self.range_image[env_ids] = self.range_max

    def _init_lidar(self):
        """Initialize the Warp mesh and the shared LidarSensor instance."""
        import warp as wp
        from .quaternion import quat_from_euler_xyz
        from hexapod_mjlab.utils.lidar import LidarConfig, LidarSensor, LidarType
        wp.init()
        vertices = torch.from_numpy(self.map_data.vertices).to(self.device)
        triangles = np.asarray(self.map_data.triangles, dtype=np.int32).flatten()
        self._wp_mesh = wp.Mesh(points=wp.from_torch(vertices, dtype=wp.vec3), indices=wp.from_numpy(triangles, dtype=wp.int32, device=self.device))
        self.mesh_ids = wp.array([self._wp_mesh.id], dtype=wp.uint64, device=self.device)
        use_full_lidar = not bool(getattr(self.cfg.lidar, 'use_reduced_raycast', True))
        if use_full_lidar:
            lidar_cfg = LidarConfig(sensor_type=LidarType.AIRY, dt=float(self.dt), update_frequency=float(self.cfg.lidar.update_frequency_hz), max_range=float(self.cfg.lidar.far_plane), min_range=float(self.cfg.lidar.min_range), num_sensors=1, horizontal_line_num=int(self.cfg.lidar.airy_n_azimuth), vertical_line_num=int(self.cfg.lidar.airy_n_elevation), horizontal_fov_deg_min=-180.0, horizontal_fov_deg_max=180.0, vertical_fov_deg_min=float(self.cfg.lidar.airy_vertical_fov_deg[0]), vertical_fov_deg_max=float(self.cfg.lidar.airy_vertical_fov_deg[1]), airy_horizontal_resolution_deg=float(self.cfg.lidar.airy_horizontal_resolution_deg), return_pointcloud=True, pointcloud_in_world_frame=False, randomize_placement=False, enable_sensor_noise=bool(self.cfg.lidar.enable_sensor_noise), pixel_std_dev_multiplier=float(self.cfg.lidar.pixel_std_dev_multiplier), pixel_dropout_prob=float(self.cfg.lidar.pixel_dropout_prob), random_distance_noise=float(self.cfg.lidar.random_distance_noise), random_angle_noise=float(self.cfg.lidar.random_angle_noise))
            lidar_env = {'device': self.device, 'num_envs': self.num_envs, 'num_sensors': 1, 'sensor_pos_tensor': self.sensor_pos_tensor, 'sensor_quat_tensor': self.sensor_quat_tensor, 'mesh_ids': self.mesh_ids}
            self.lidar_sensor = LidarSensor(lidar_env, None, lidar_cfg, num_sensors=1, device=self.device)
        else:
            self.lidar_sensor = None
        self._lidar_positions_wp = wp.from_torch(self.sensor_pos_tensor.view(self.num_envs, 1, 3), dtype=wp.vec3)
        self._lidar_quat_wp = wp.from_torch(self.sensor_quat_tensor.view(self.num_envs, 1, 4), dtype=wp.quat)
        from types import SimpleNamespace
        self._noise_ctx = SimpleNamespace(sensor_cfg=SimpleNamespace(enable_sensor_noise=bool(self.cfg.lidar.enable_sensor_noise), pixel_std_dev_multiplier=float(self.cfg.lidar.pixel_std_dev_multiplier), pixel_dropout_prob=float(self.cfg.lidar.pixel_dropout_prob)), far_plane=float(self.cfg.lidar.far_plane))
        self._apply_noise = LidarSensor.apply_noise
        offset_pos = list(self.cfg.lidar.offset_pos)
        self._sensor_translation = torch.tensor(offset_pos, dtype=torch.float32, device=self.device).view(1, 3).repeat(self.num_envs, 1)
        rpy = list(self.cfg.lidar.sensor_offset_rpy)
        offset_q = quat_from_euler_xyz(torch.tensor(float(rpy[0]), device=self.device), torch.tensor(float(rpy[1]), device=self.device), torch.tensor(float(rpy[2]), device=self.device))
        self._sensor_offset_quat = offset_q.view(1, 4).repeat(self.num_envs, 1)
        from hexapod_mjlab.tasks.envelope_adaptive_2.airy_mount import load_selected_channels
        self.selected_channels = load_selected_channels()
        self.range_max = float(self.selected_channels['max_range'])
        self.selected_ray_indices = self.selected_channels['ray_indices'].to(self.device)
        self.selected_ray_directions = self.selected_channels['ray_directions'].to(self.device)
        self.range_image = empty_range_image(self.num_envs, self.range_max, self.device)
        n_sel = ea2c.EA2_RANGE_DIM
        ray_dir_tensor = self.selected_ray_directions.unsqueeze(1).contiguous()
        self._reduced_ray_vectors = wp.from_torch(ray_dir_tensor, dtype=wp.vec3)
        self._reduced_lidar_tensor = torch.zeros((self.num_envs, 1, n_sel, 1, 3), dtype=torch.float32, device=self.device)
        self._reduced_dist_tensor = torch.zeros((self.num_envs, 1, n_sel, 1), dtype=torch.float32, device=self.device)
        self._reduced_lidar_warp = wp.from_torch(self._reduced_lidar_tensor, dtype=wp.vec3)
        self._reduced_dist_warp = wp.from_torch(self._reduced_dist_tensor, dtype=wp.float32)

    def _update_lidar(self):
        """Advance the global 10 Hz clock and refresh the 187-dim range image.

            Two paths are supported:

            * reduced path (training, headless): raycast only the 187 selected
              channels;
            * full path (debug/viewer): raycast the full Airy pattern and extract
              the 187 selected channels, keeping the full point cloud for
              comparison.
            """
        self._lidar_timer += 1
        if self._lidar_timer % self._lidar_decimation != 0:
            return
        import warp as wp
        from .quaternion import quat_apply, quat_from_euler_xyz, quat_mul
        from hexapod_mjlab.utils.lidar.sensor_kernels.lidar_kernels_warp import LidarWarpKernels
        current_q = yaw_quat_from_heading(self.heading)
        if getattr(self, '_attitude', None) is not None:
            zeros = torch.zeros_like(self.attitude_roll)
            tilt = quat_from_euler_xyz(self.attitude_roll, self.attitude_pitch, zeros)
            current_q = quat_mul(current_q, tilt)
        self.sensor_quat_tensor.copy_(quat_mul(current_q, self._sensor_offset_quat))
        self.sensor_pos_tensor.copy_(self.base_pos + quat_apply(current_q, self._sensor_translation))
        use_full = not bool(getattr(self.cfg.lidar, 'use_reduced_raycast', True))
        if use_full:
            lidar_points, lidar_dist = self.lidar_sensor.update()
            points = lidar_points.view(self.num_envs, -1, 3)
            dists = lidar_dist.view(self.num_envs, -1)
            n_total = int(points.numel() // 3)
            points_body = quat_apply(self._sensor_offset_quat[0:1].expand(n_total, 4), points.reshape(-1, 3)).reshape(self.num_envs, -1, 3) + self._sensor_translation.unsqueeze(1)
            fresh = extract_selected_range_image(dists, self.selected_ray_indices, self.range_max)
            refresh_range_image_from_scan(self.range_image, fresh, self.range_image_stale)
            if self.capture_debug and self._debug_env_ids:
                self._debug_is_reduced = False
                ids = torch.tensor(self._debug_env_ids, dtype=torch.long, device=self.device)
                self._debug_points = points_body[ids].clone()
                self._debug_dists = dists[ids].clone()
        else:
            wp.launch(kernel=LidarWarpKernels.draw_optimized_kernel_pointcloud, dim=(self.num_envs, 1, ea2c.EA2_RANGE_DIM, 1), inputs=[self.mesh_ids, self._lidar_positions_wp, self._lidar_quat_wp, self._reduced_ray_vectors, float(self.cfg.lidar.far_plane), self._reduced_lidar_warp, self._reduced_dist_warp, False], device=self.device)
            lidar_points = wp.to_torch(self._reduced_lidar_warp)
            lidar_dist = wp.to_torch(self._reduced_dist_warp)
            lidar_points, lidar_dist = self._apply_noise(self._noise_ctx, lidar_points, lidar_dist)
            dists = lidar_dist.view(self.num_envs, -1)
            fresh = build_selected_range_image(dists, self.range_max)
            refresh_range_image_from_scan(self.range_image, fresh, self.range_image_stale)
        if self.capture_debug and self._debug_env_ids:
            self._debug_is_reduced = not use_full
            points = lidar_points.view(self.num_envs, -1, 3)
            n_total = int(points.numel() // 3)
            points_body = quat_apply(self._sensor_offset_quat[0:1].expand(n_total, 4), points.reshape(-1, 3)).reshape(self.num_envs, -1, 3) + self._sensor_translation.unsqueeze(1)
            ids = torch.tensor(self._debug_env_ids, dtype=torch.long, device=self.device)
            self._debug_points = points_body[ids].clone()
            self._debug_dists = dists[ids].clone()
