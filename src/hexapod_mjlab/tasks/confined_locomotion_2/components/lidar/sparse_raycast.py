"""Task-local full/subset raycast using the existing physical Warp query.

Only the supplied real Airy directions are allocated and launched. Tensor
owners stay alive; Torch pose writes, the Warp kernel and consumers share the
current Torch CUDA stream. Scheduling remains entirely with TrainingRuntime.
"""
from __future__ import annotations

import torch
import warp as wp


class ChannelRaycaster:
    def __init__(self, *, directions, position, quaternion, mesh_ids, max_range):
        from hexapod_mjlab.utils.lidar.sensor_kernels.lidar_kernels_warp import LidarWarpKernels

        self.directions = directions.contiguous()
        self.position, self.quaternion = position, quaternion
        self.mesh_ids, self.max_range = mesh_ids, float(max_range)
        self.kernel = LidarWarpKernels.draw_optimized_kernel_pointcloud
        self.device = directions.device
        self.num_rays = int(directions.shape[0])
        self.capacity = int(position.shape[0])
        self.points = torch.empty((self.capacity, 1, self.num_rays, 1, 3), device=self.device)
        self.distances = torch.empty((self.capacity, 1, self.num_rays, 1), device=self.device)
        self.launch_count = 0

    def get_ray_directions(self):
        return self.directions

    def update(self, num_rows=None):
        """Launch the existing kernel only for the supplied pose prefix."""
        num_rows = self.capacity if num_rows is None else num_rows
        if not 1 <= num_rows <= self.capacity:
            raise ValueError("raycast row count outside allocated capacity")
        wp.launch(
            kernel=self.kernel,
            dim=(num_rows, 1, self.num_rays, 1),
            inputs=[self.mesh_ids,
                    wp.from_torch(self.position.view(self.capacity, 1, 3), dtype=wp.vec3),
                    wp.from_torch(self.quaternion.view(self.capacity, 1, 4), dtype=wp.quat),
                    wp.from_torch(self.directions.view(self.num_rays, 1, 3), dtype=wp.vec3),
                    self.max_range,
                    wp.from_torch(self.points, dtype=wp.vec3),
                    wp.from_torch(self.distances, dtype=wp.float32), False],
            device=str(self.device),
            stream=wp.stream_from_torch(torch.cuda.current_stream(self.device)),
        )
        self.launch_count += 1
        return self.points[:num_rows], self.distances[:num_rows]
