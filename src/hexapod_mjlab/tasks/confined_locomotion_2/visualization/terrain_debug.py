"""Terrain-specific diagnostic layers over the shared read-only DebugViewer."""

import numpy as np

from .debug import DebugViewer, quat_wxyz_to_matrix
from .grid import GridRangePreview
from ..components.terrain_query import CompiledHeightfield, HeightGridCfg


def height_points_world(data, name):
    """Both prediction and target use ONE retained pre-action Root Z."""
    xy = np.asarray(data["query_xy_w"], dtype=np.float64)
    heights = np.asarray(data[name], dtype=np.float64)
    valid = np.asarray(data["valid"], dtype=bool)
    if xy.shape != (11, 17, 2) or heights.shape != (11, 17) or valid.shape != (11, 17):
        raise ValueError("terrain diagnostics must preserve the 11x17 grid")
    points = np.concatenate((xy, (heights + float(data["root_pos_w"][2]))[..., None]), axis=-1)
    if not np.isfinite(points[valid]).all():
        raise ValueError("valid height display points must be finite")
    return points[valid]


def heightfield_mesh(model):
    """Top triangles from actual compiled samples; same diagonal as ray labels."""
    field = CompiledHeightfield(model)
    nr, nc = field.samples.shape
    xx, yy = np.meshgrid(np.linspace(-field.size[0], field.size[0], nc),
                         np.linspace(-field.size[1], field.size[1], nr))
    local = np.stack((xx, yy, field.samples * field.size[2]), axis=-1).reshape(-1, 3)
    vertices = local @ field.rotation.T + field.position
    r, c = np.meshgrid(np.arange(nr-1), np.arange(nc-1), indexing="ij")
    a = (r*nc+c).ravel()
    faces = np.concatenate((np.stack((a, a+1, a+nc+1), axis=-1),
                            np.stack((a, a+nc+1, a+nc), axis=-1)))
    return vertices, faces.astype(np.int32)


class TerrainMemoryDebugViewer(DebugViewer):
    def _setup_gui(self):
        super()._setup_gui()
        gui = self.server.gui
        with self._layers_folder:
            self._envelope_target = gui.add_checkbox("监督包络（黄）", True)
            self._height_prediction = gui.add_checkbox("高度预测（橙）", True)
            self._height_target = gui.add_checkbox("高度真值（青）", True)
        with self._status_folder:
            self._terrain_panel = gui.add_markdown("")
        for control in (self._height_prediction, self._height_target, self._envelope_target):
            control.on_update(lambda _: self.pending.submit_rerender())
        self._grid_preview = None
        self._terrain_mesh = None

    def _world_to_display(self, points):
        points = np.asarray(points)
        if self.view_mode == "world":
            return points
        pose = self._last_snapshot["pose"]
        return (points - pose["root_pos_w"]) @ quat_wxyz_to_matrix(pose["root_quat_wxyz"])

    def _draw_ground(self, snapshot):
        if self._terrain_mesh is None:
            self._terrain_mesh = heightfield_mesh(self.mj_model)
        vertices, faces = self._terrain_mesh
        vertices = self._world_to_display(vertices).astype(np.float32)
        self._swap("/physical_terrain", lambda: self.server.scene.add_mesh_simple(
            "/physical_terrain", vertices, faces, color=(115, 125, 115), side="double"))

    def _draw_task_layers(self, snapshot):
        data = snapshot["terrain"]
        if data["identity"] != {key: snapshot["identity"][key] for key in data["identity"]}:
            raise ValueError("terrain diagnostics do not match the action input identity")
        for name, visible, color in (("prediction", self._height_prediction.value, (255, 145, 40)),
                                     ("target", self._height_target.value, (20, 220, 230))):
            path = "/height/" + name
            points = self._world_to_display(height_points_world(data, name)).astype(np.float32)
            if visible and len(points):
                self._swap(path, lambda path=path, points=points, color=color: self.server.scene.add_point_cloud(
                    path, points, colors=color, point_size=0.035))
            else:
                self._drop(path)
        if self._envelope_target.value and data["envelope_valid"]:
            self._draw_envelope_values(snapshot, data["envelope_target"],
                prefix="/envelope_supervision", color=(255, 220, 40))
        else:
            self._drop("/envelope_supervision/prism")
        if self._grid_preview is None:
            grid = HeightGridCfg(tuple(data["grid"]["x_range_m"]), tuple(data["grid"]["y_range_m"]))
            self._grid_preview = GridRangePreview(self.server, grid, self.pending.submit_rerender)
        try:
            candidate = self._grid_preview.render(data["root_pos_w"], data["root_quat_wxyz"], self._world_to_display)
            candidate_text = f"预览范围：X±{candidate.x_range_m[1]:.3f} / Y±{candidate.y_range_m[1]:.3f} m（仅显示）。\n\n"
        except ValueError:
            candidate_text = "预览范围无效，请输入正数；保留上次显示。\n\n"
        self._terrain_panel.content = (
            f"**地形监督** · 动作前控制步 {data['identity']['input_control_step']}\n\n"
            f"模型网格：X±{data['grid']['x_range_m'][1]:.3f} / Y±{data['grid']['y_range_m'][1]:.3f} m · "
            f"有效 {int(data['valid'].sum())}/187。\n\n"
            + candidate_text + "高度图不输入策略；六棱柱跟随机器人。"
        )

    def _clear_session_display(self):
        super()._clear_session_display()
        self._terrain_mesh = None
        if self._grid_preview is not None:
            for pair in self._grid_preview.layers.values():
                for handle in pair:
                    handle.visible = False
