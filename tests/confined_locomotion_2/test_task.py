"""Task-local contracts for the single-frame far scan baseline."""

import torch

from hexapod_mjlab.rl.confined_locomotion_2.contracts import (
    FAR_POINTS_PER_SCAN,
    FAR_WINDOW_SLOTS,
    NEAR_POINTS,
)
from hexapod_mjlab.tasks.confined_locomotion_2.components.lidar.scanner import (
    ConfinedLidarCfg,
    farthest_point_sampling_indices,
)


def test_confined_locomotion_2_scan_dimensions_are_current_only():
    cfg = ConfinedLidarCfg()
    assert NEAR_POINTS == 256
    assert FAR_POINTS_PER_SCAN == 128
    assert FAR_WINDOW_SLOTS == 1
    assert cfg.near_points == 256
    assert cfg.far_points_per_scan == 128
    assert cfg.far_window_slots == 1


def test_far_xyz_fps_returns_128_distinct_sources():
    count = 512
    points = torch.arange(count, dtype=torch.float32)[:, None].expand(-1, 3).clone()
    pool = torch.ones(count, dtype=torch.bool)
    indices, selected = farthest_point_sampling_indices(points, pool, FAR_POINTS_PER_SCAN)
    assert selected == 128
    assert indices.shape == (128,)
    assert int(torch.unique(indices).numel()) == 128
    assert bool(((indices >= 0) & (indices < count)).all())
    assert indices[:2].tolist() == [511, 0]
