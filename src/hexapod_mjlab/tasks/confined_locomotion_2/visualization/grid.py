"""Display-only 11x17 terrain-memory grid range controls and drawing."""

import numpy as np
import torch

from hexapod_mjlab.tasks.confined_locomotion_2.config.terrain_memory_config import TerrainMemoryRlCfg
from hexapod_mjlab.tasks.confined_locomotion_2.components.terrain_query import HeightGridCfg, grid_world_xy


def configured_grid():
    """Return the immutable display reference to the configured production grid."""
    cfg = TerrainMemoryRlCfg().terrain_memory
    return HeightGridCfg(tuple(cfg["x_range_m"]), tuple(cfg["y_range_m"]))


def preview_geometry(grid, root_pos, root_quat_wxyz, *, display_z=0.015):
    """Return 187 display points and their rectangular boundary.

    XY comes from the same yaw-only mapping used for training. Z is a display
    plane only; this function does not query terrain or predict height.
    """
    if not np.isfinite(display_z):
        raise ValueError("display_z must be finite")
    xy = grid_world_xy(
        torch.as_tensor(root_pos, dtype=torch.float64).reshape(1, 3),
        torch.as_tensor(root_quat_wxyz, dtype=torch.float64).reshape(1, 4), grid,
    )[0].numpy()
    points = np.column_stack((xy, np.full(len(xy), display_z))).astype(np.float32)
    corners = points[[0, 16, 186, 170]]
    return points, np.stack((corners, np.roll(corners, -1, axis=0)), axis=1)


class GridRangePreview:
    """Viser controls and display layers for actual and candidate grid bounds.

    GUI handlers only call ``on_change`` to request an owner-loop redraw. The
    candidate grid is local display state and never changes the config.
    """

    def __init__(self, server, actual_grid, on_change):
        self.server = server
        self.actual_grid = actual_grid
        self._on_change = on_change
        self.scene, gui = server.scene, server.gui
        with gui.add_folder("网格范围预览", order=2, expand_by_default=False):
            self.half_x = gui.add_number("X 半范围 (m)", actual_grid.x_range_m[1], min=0.05, max=5.0, step=0.05)
            self.half_y = gui.add_number("Y 半范围 (m)", actual_grid.y_range_m[1], min=0.05, max=5.0, step=0.05)
            self.show_actual = gui.add_checkbox("原始范围（绿）", True)
            self.show_candidate = gui.add_checkbox("预览范围（蓝）", True)
            self.reset_button = gui.add_button("恢复原始范围")
            gui.add_markdown("仅调整显示范围，不改变模型或训练配置。")
        for control in (self.half_x, self.half_y, self.show_actual, self.show_candidate):
            control.on_update(lambda _: self._request_redraw())
        self.reset_button.on_click(self.reset_candidate)
        self.layers = {}
        for name, color, size, width in (
            ("actual", (60, 210, 95), 0.035, 7),
            ("candidate", (55, 150, 255), 0.018, 3),
        ):
            points, edges = preview_geometry(actual_grid, (0, 0, 0), (1, 0, 0, 0))
            self.layers[name] = (
                self.scene.add_point_cloud(
                    f"/terrain_memory_grid/{name}/points", points,
                    np.tile(color, (len(points), 1)).astype(np.uint8), point_size=size,
                ),
                self.scene.add_line_segments(
                    f"/terrain_memory_grid/{name}/bounds", edges, color, line_width=width,
                ),
            )

    def _request_redraw(self):
        if self._on_change is not None:
            self._on_change()

    def reset_candidate(self, _=None):
        self.half_x.value = self.actual_grid.x_range_m[1]
        self.half_y.value = self.actual_grid.y_range_m[1]
        self._request_redraw()

    def candidate_grid(self):
        return HeightGridCfg(
            (-self.half_x.value, self.half_x.value),
            (-self.half_y.value, self.half_y.value),
        )

    def render(self, root_pos_w, root_quat_wxyz, world_to_display=None, display_z=None):
        """Redraw both grids at a horizontal display plane.

        ``world_to_display`` may convert the generated world-space point arrays
        to a viewer's display frame. Candidate bounds sit 1 cm above actual
        bounds to make coincident outlines visible.
        """
        candidate = self.candidate_grid()
        root_pos = np.asarray(root_pos_w, dtype=np.float64).reshape(3)
        base_z = float(root_pos[2] + 0.05 if display_z is None else display_z)
        if not np.isfinite(base_z):
            raise ValueError("display_z must be finite")
        for name, grid, visible, z in (
            ("actual", self.actual_grid, self.show_actual.value, base_z),
            ("candidate", candidate, self.show_candidate.value, base_z + 0.01),
        ):
            points, edges = preview_geometry(grid, root_pos, root_quat_wxyz, display_z=z)
            if world_to_display is not None:
                points = np.asarray(world_to_display(points), dtype=np.float32)
                edges = np.asarray(world_to_display(edges), dtype=np.float32)
                if points.shape != (187, 3) or edges.shape != (4, 2, 3):
                    raise ValueError("world_to_display must preserve the point and edge shapes")
            cloud, bounds = self.layers[name]
            cloud.points, bounds.points = points, edges
            cloud.visible = bounds.visible = visible
        return candidate
