"""Private CUDA FPS kernel. Imported lazily; all arrays use PyTorch's stream."""
from __future__ import annotations

import torch
import warp as wp

# Module-local options: do not change global Warp configuration or old tasks.
wp.set_module_options({"enable_backward": False, "enable_fp_fma": False})
BLOCK_DIM = 1024


@wp.kernel(enable_backward=False)
def _compact_fps(points: wp.array3d[wp.float32], counts: wp.array[wp.int32], starts: wp.array[wp.int64],
                nearest: wp.array2d[wp.float32], output: wp.array2d[wp.int64], k: int):
    row, lane = wp.tid()
    size = counts[row]
    for p in range(lane, size, BLOCK_DIM):
        nearest[row, p] = float(1.0e30)
    if lane == 0:
        for j in range(k):
            output[row, j] = wp.int64(-1)
    if counts[row] > 0:
        selected = int(starts[row])
        if lane == 0:
            output[row, 0] = wp.int64(selected)
        for j in range(1, wp.min(k, counts[row])):
            sx = points[row, selected, 0]
            sy = points[row, selected, 1]
            sz = points[row, selected, 2]
            local_max = float(-1.0)
            local_idx = int(2147483647)
            for p in range(lane, size, BLOCK_DIM):
                dx = points[row, p, 0] - sx
                dy = points[row, p, 1] - sy
                dz = points[row, p, 2] - sz
                value = wp.min(nearest[row, p], (dx * dx + dy * dy) + dz * dz)
                if p == selected:
                    value = float(-1.0)
                nearest[row, p] = value
                if value > local_max or (value == local_max and p < local_idx):
                    local_max = value
                    local_idx = p
            maximum = wp.tile_max(wp.tile(local_max))[0]
            candidate = int(2147483647)
            if local_max == maximum:
                candidate = local_idx
            # Explicit first-source-index tie rule, including coincident rays.
            selected = wp.tile_min(wp.tile(candidate))[0]
            if lane == 0:
                output[row, j] = wp.int64(selected)


def launch_compact_fps(points, counts, starts, k):
    wp.init()
    output = torch.empty((points.shape[0], k), dtype=torch.long, device=points.device)
    nearest = torch.empty(points.shape[:2], dtype=torch.float32, device=points.device)
    # Allocation, producer ops and the Warp launch share the current Torch
    # stream. No global synchronization or cross-stream lifetime is required.
    stream = wp.stream_from_torch(torch.cuda.current_stream(points.device))
    wp.launch(
        _compact_fps, dim=(points.shape[0], BLOCK_DIM), block_dim=BLOCK_DIM,
        inputs=[wp.from_torch(points), wp.from_torch(counts),
                wp.from_torch(starts), wp.from_torch(nearest), wp.from_torch(output), k],
        stream=stream, device=str(points.device),
    )
    return output
