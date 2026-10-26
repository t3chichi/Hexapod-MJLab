"""CPU checks for the display-only supervision range preview."""

import importlib.util
from pathlib import Path

import numpy as np
import torch


def _cli():
    path = Path(__file__).resolve().parents[2] / "scripts/experiments/confined_locomotion_2/preview_supervision_grid.py"
    spec = importlib.util.spec_from_file_location("terrain_grid_preview", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preview_points_match_training_xy_and_grid_order():
    cli = _cli()
    grid = cli.configured_grid()
    root, quat = [1.7, -0.8, 0.65], [np.cos(0.3), 0, 0, np.sin(0.3)]
    points, edges = cli.preview_geometry(grid, root, quat)
    expected = cli.grid_world_xy(torch.tensor([root], dtype=torch.float64),
                                torch.tensor([quat], dtype=torch.float64), grid)[0].numpy()
    np.testing.assert_allclose(points[:, :2], expected, atol=2e-7)
    assert points.shape == (187, 3) and edges.shape == (4, 2, 3)
    np.testing.assert_array_equal(edges[:, 0], points[[0, 16, 186, 170]])
    np.testing.assert_allclose(points[93, :2], root[:2], atol=1e-7)
    np.testing.assert_allclose(points[:, 2], 0.015)


def test_preview_is_yaw_only_and_display_z_is_not_root_relative_height():
    cli = _cli()
    from scipy.spatial.transform import Rotation
    grid = cli.configured_grid()
    root = [0.2, -0.3, 0.5]
    flat = Rotation.from_euler("xyz", [0, 0, 0.7]).as_quat(scalar_first=True)
    tilted = Rotation.from_euler("xyz", [0.4, -0.3, 0.7]).as_quat(scalar_first=True)
    points, _ = cli.preview_geometry(grid, root, flat)
    other, _ = cli.preview_geometry(grid, [root[0], root[1], 1.3], tilted)
    np.testing.assert_array_equal(points, other)


def test_candidate_changes_do_not_mutate_configured_grid():
    cli = _cli()
    actual = cli.configured_grid()
    before = actual.identity()
    candidate = cli.HeightGridCfg((-0.8, 0.8), (-0.6, 0.6))
    points, _ = cli.preview_geometry(candidate, [0, 0, 0.5], [1, 0, 0, 0])
    np.testing.assert_allclose(points[0, :2], [-0.8, -0.6])
    assert actual.identity() == cli.configured_grid().identity() == before
    # Reset uses immutable actual bounds, not a candidate/config writeback.
    from types import SimpleNamespace
    import threading
    viewer = SimpleNamespace(actual=actual, half_x=SimpleNamespace(value=0.8),
                             half_y=SimpleNamespace(value=0.6), dirty=threading.Event())
    cli.GridPreview.reset_candidate(viewer)
    assert viewer.half_x.value == actual.x_range_m[1]
    assert viewer.half_y.value == actual.y_range_m[1] and viewer.dirty.is_set()


def test_preview_rejects_invalid_display_plane_and_bounds():
    cli = _cli()
    for x, y in ((0, 0.9), (-1, 0.9), (float("nan"), 0.9), (1.2, float("inf"))):
        try:
            cli.HeightGridCfg((-x, x), (-y, y))
        except ValueError:
            pass
        else:
            raise AssertionError("invalid candidate accepted")
    try:
        cli.preview_geometry(cli.configured_grid(), [0, 0, 1], [1, 0, 0, 0], display_z=float("nan"))
    except ValueError:
        pass
    else:
        raise AssertionError("nonfinite display plane accepted")
