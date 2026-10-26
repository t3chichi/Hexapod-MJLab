"""Local, display-only Terrain Memory grid preview; no checkpoint or simulation."""

import argparse
from pathlib import Path
import sys
import threading

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from hexapod_mjlab.tasks.confined_locomotion_2.terrain_geometry import HeightGridCfg, grid_world_xy
from hexapod_mjlab.tasks.confined_locomotion_2.visualization.grid import (
    GridRangePreview,
    configured_grid,
    preview_geometry,
)


class GridPreview:
    def __init__(self, port=8095):
        import mujoco
        import viser
        from hexapod_mjlab.tasks.confined_locomotion_2.flat_env_cfg import (
            JOINT_NAMES, get_robot_spec, lifted_witness_state, settled_reset_joint_pos,
        )
        from hexapod_mjlab.tasks.confined_locomotion.visualization.play import robot_meshes_at_qpos

        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise ValueError("port must be in 1..65535")
        self.actual = configured_grid()
        self.dirty = threading.Event()
        self.finished = threading.Event()
        model = get_robot_spec().compile()
        qpos = model.qpos0.copy()
        # Static reference posture: existing reset joints, horizontal root at
        # its witness height. No claim about the complete moving foot workspace.
        self.root_z = lifted_witness_state()[0][2]
        qpos[:7] = [0, 0, self.root_z, 1, 0, 0, 0]
        for name, value in zip(JOINT_NAMES, settled_reset_joint_pos(), strict=True):
            qpos[int(model.joint(name).qposadr[0])] = value
        meshes = robot_meshes_at_qpos(model, qpos, mj_data=mujoco.MjData(model))
        self.server = viser.ViserServer(host="127.0.0.1", port=port, label="监督网格预览")
        try:
            if self.server.get_port() != port:
                raise RuntimeError(f"requested port {port} is occupied")
            scene, gui = self.server.scene, self.server.gui
            scene.set_up_direction("+z")
            scene.add_grid("/reference", width=8, height=8, plane="xy", cell_size=0.1, section_size=1.0)
            self.robot = scene.add_frame("/robot", show_axes=False, position=(0, 0, self.root_z))
            for i, mesh in enumerate(meshes):
                scene.add_mesh_simple(
                    f"/robot/mesh_{i}",
                    (mesh["vertices"] - [0, 0, self.root_z]).astype(np.float32),
                    mesh["faces"].astype(np.int32), color=(168, 175, 185), side="double",
                )
            scene.add_frame("/robot/root_axes", axes_length=0.35, axes_radius=0.008)
            gui.add_markdown(
                "**监督范围预览 · 11×17 点**\n\n"
                "绿色为原始范围，蓝色为预览范围。水平参考面仅用于比较覆盖范围，不表示地形高度。",
                order=0,
            )
            self.range_preview = GridRangePreview(self.server, self.actual, self.dirty.set)
            self.half_x = self.range_preview.half_x
            self.half_y = self.range_preview.half_y
            self.show_actual = self.range_preview.show_actual
            self.show_candidate = self.range_preview.show_candidate
            with gui.add_folder("参考姿态", order=3):
                self.root_x = gui.add_slider("位置 X (m)", -2.0, 2.0, step=0.05, initial_value=0.0)
                self.root_y = gui.add_slider("位置 Y (m)", -2.0, 2.0, step=0.05, initial_value=0.0)
                self.yaw = gui.add_slider("朝向 (°)", -180.0, 180.0, step=1.0, initial_value=0.0)
            self.status = gui.add_markdown("", order=4)
            gui.add_button("退出", order=1).on_click(lambda _: self.finished.set())
            for control in (self.root_x, self.root_y, self.yaw):
                control.on_update(lambda _: self.dirty.set())
            self.layers = self.range_preview.layers

            @self.server.on_client_connect
            def camera(client):
                client.camera.position = (2.1, -2.5, 2.8)
                client.camera.look_at = (0, 0, 0.15)

            self.render()
        except BaseException:
            self.close()
            raise

    def reset_candidate(self, _=None):
        self.half_x.value = self.actual.x_range_m[1]
        self.half_y.value = self.actual.y_range_m[1]
        self.dirty.set()

    def render(self):
        """Owner-loop drawing only; GUI callbacks never touch model or training."""
        self.dirty.clear()
        try:
            candidate = self.range_preview.candidate_grid()
        except ValueError as exc:
            self.status.content = "范围无效，请输入正数；保留上次显示。"
            return
        yaw = np.deg2rad(self.yaw.value)
        quat = (np.cos(yaw/2), 0, 0, np.sin(yaw/2))
        root = (self.root_x.value, self.root_y.value, self.root_z)
        with self.server.atomic():
            self.robot.position, self.robot.wxyz = root, quat
            self.range_preview.render(root, quat, display_z=0.015)
            self.status.content = (
                f"**原始范围** X±{self.actual.x_range_m[1]:.3f} / Y±{self.actual.y_range_m[1]:.3f} m\n\n"
                f"**预览范围** X±{candidate.x_range_m[1]:.3f} / Y±{candidate.y_range_m[1]:.3f} m\n\n"
                f"采样间距：X {2*candidate.x_range_m[1]/16:.4f} / Y {2*candidate.y_range_m[1]/10:.4f} m。"
            )

    def close(self):
        self.server.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8095)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in 1..65535")
    viewer = GridPreview(args.port)
    print(f"监督网格预览：http://127.0.0.1:{args.port}（仅显示，Ctrl+C 或页面按钮退出）", flush=True)
    try:
        while not viewer.finished.wait(0.05):
            if viewer.dirty.is_set():
                viewer.render()
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
