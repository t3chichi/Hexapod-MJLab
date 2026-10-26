import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from hexapod_mjlab.tasks.confined_locomotion_2.visualization.play import (
    PLAY_SNAPSHOT_SCHEMA,
    PlayViewer,
    export_snapshot_manifest,
    resolve_point_pick,
)


class _Handle:
    def remove(self):
        pass


class _Control:
    def __init__(self, value=None):
        self.value = value
        self._click = None
        self._update = None

    def on_click(self, callback):
        self._click = callback

    def on_update(self, callback):
        self._update = callback

    def click(self, value=None):
        if value is not None:
            self.value = value
        self._click(None)

    def update(self, value):
        self.value = value
        if self._update is not None:
            self._update(None)


class _Folder:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class _Gui:
    def __init__(self):
        self.controls = {}

    def _add(self, name, value=None, **_kwargs):
        control = _Control(value)
        self.controls[name] = control
        return control

    def add_dropdown(self, name, options, initial_value=None):
        return self._add(name, initial_value if initial_value is not None else options[0])

    def add_checkbox(self, name, initial_value=False):
        return self._add(name, initial_value)

    def add_slider(self, name, _min, _max, *, initial_value=0.0, **_kwargs):
        return self._add(name, initial_value)

    def add_number(self, name, initial_value=0, **_kwargs):
        return self._add(name, initial_value)

    def add_folder(self, name, **kwargs):
        return _Folder()

    def add_button_group(self, name, options):
        return self._add(name, options[0])

    def add_button(self, name):
        return self._add(name, False)

    def add_markdown(self, name):
        return self._add(name, "")


class _Scene:
    def __init__(self):
        self.objects = {}

    def _add(self, name, *args, **kwargs):
        self.objects[name] = (args, kwargs)
        return _Handle()

    add_point_cloud = _add
    add_mesh_simple = _add
    add_line_segments = _add
    add_icosphere = _add


class _Server:
    def __init__(self, **_kwargs):
        self.port = _kwargs["port"]
        self.gui = _Gui()
        self.scene = _Scene()
        self.stopped = False
        self._connect_callback = None
        self.clients = {}

    def stop(self):
        self.stopped = True

    def on_client_connect(self, callback):
        self._connect_callback = callback

    def get_clients(self):
        return self.clients

    def get_port(self):
        return self.port


class _Pipeline:
    class Config:
        split_theta_deg = 28.0
        split_axis_tilt_deg = 56.0
        split_max_forward_x_m = 3.5
        max_range_m = 15.0

    fixed_boundary_segments = np.array([[[0.0, 0.0, 0.1], [1.0, 0.0, -0.1]]])
    cfg = Config()
    mount_pos = np.array([0.0, 0.0, 0.1])
    cone_axis_body = np.array([0.0, 0.0, 1.0])
    mount_quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0])
    mesh_record = {
        "z": 0.0,
        "vertices": [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]],
        "triangles": [[0, 1, 2], [0, 2, 3]],
    }

    def scan_rows(self, *_args, **_kwargs):
        raise AssertionError("viewer must not scan")


def _snapshot():
    near_ids = np.full(256, -1, dtype=np.int64)
    far_ids = np.full(128, -1, dtype=np.int64)
    near_ids[0] = 47
    far_ids[0] = 903
    near_valid = np.zeros(256, dtype=bool)
    far_valid = np.zeros(128, dtype=bool)
    near_valid[0] = True
    far_valid[0] = True
    near_pts = np.zeros((256, 3), dtype=np.float64)
    far_pts = np.zeros((128, 3), dtype=np.float64)
    near_pts[0] = [1.0, 0.2, 0.4]
    far_pts[0] = [2.0, -0.3, 0.1]
    return {
        "schema": PLAY_SNAPSHOT_SCHEMA,
        "env": 0,
        "identity": {
            "input_control_step": 4,
            "pose_control_step": 5,
            "episode_id": 2,
            "step_in_episode": 5,
            "input_episode_id": 2,
            "input_step_in_episode": 4,
            "input_time_s": 0.08,
            "pose_time_s": 0.1,
            "frame_encoding_count": 1,
            "forward_count": 5,
        },
        "pose": {
            "root_pos_w": np.array([3.0, 4.0, 0.46]),
            "root_quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0]),
            "qpos": np.asarray([0.0, 0.0, 0.0, 1.0] + [0.0] * 15, dtype=np.float64),
            "root_height_m": 0.46,
        },
        "input": {
            "near": {"points_base": near_pts, "valid": near_valid,
                     "scan_id": 1, "time_s": 0.0},
            "far": {"points_base": far_pts, "valid": far_valid,
                    "scan_id": 1, "time_s": 0.0},
        },
        "prediction": {
            "envelope": np.array([0.59, 0.69, 0.59, 0.89, -0.89, 0.0]),
            "action_mean": np.zeros(18), "preview": False,
        },
        "status": {"paused": False, "stopped": False, "faulted": False,
                   "command": np.zeros(3), "reason": "running"},
        "provenance": {
            "near_raw_ids": near_ids,
            "far_raw_ids": far_ids,
            "near_no_hit": np.zeros(256, dtype=bool),
            "far_no_hit": np.zeros(128, dtype=bool),
        },
        "pool": {"points_base": np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
                 "near_mask": np.array([True, False]),
                 "far_mask": np.array([False, True])},
        "checkpoint": {"path": "/tmp/current.pt"},
    }


def test_snapshot_point_picker_and_manifest_use_raw_source_identity(tmp_path):
    snapshot = _snapshot()
    pick = resolve_point_pick(snapshot, "far", 0)
    assert pick["raw_ray_id"] == 903
    assert pick["scan_id"] == 1
    assert pick["valid"] is True
    assert pick["no_hit"] is False
    assert np.array_equal(pick["xyz_base"], snapshot["input"]["far"]["points_base"][0])
    viewer = SimpleNamespace(view_mode="base", identity_header={"task": "fixture"},
                             _handles={"/points/far": object()})
    out = tmp_path / "manifest.json"
    manifest = export_snapshot_manifest(snapshot, viewer, out)
    assert manifest["counts"]["far_valid"] == 1
    assert "uncompensated" in manifest["source_coordinate_semantics"]
    assert "no ground" in manifest["ground_semantics"]
    assert json.loads(out.read_text())["snapshot_schema"] == PLAY_SNAPSHOT_SCHEMA


def test_viewer_callbacks_are_intents_and_render_has_only_task_layers():
    fake_viser = SimpleNamespace(ViserServer=_Server)
    previous = sys.modules.get("viser")
    sys.modules["viser"] = fake_viser
    try:
        import mujoco
        joint_bodies = "".join(
            f'<body name="j{i}" pos="0 0 0"><joint name="j{i}" type="hinge" axis="0 0 1"/><geom type="sphere" size="0.01" mass="0.01"/></body>'
            for i in range(12)
        )
        model = mujoco.MjModel.from_xml_string(
            f'<mujoco><worldbody><body name="root"><freejoint name="root_free"/><geom name="base" type="sphere" size="0.1" mass="1"/>{joint_bodies}</body></worldbody></mujoco>'
        )
        pipeline = _Pipeline()
        pipeline.scan_identity = {"mode": "sparse", "selection": {
            "selection": {"near_multiplier": 5, "far_multiplier": 4,
                          "total_count": 1920},
            "npz_sha256": "0123456789abcdef", "catalog_identity": {"large": "fixture"},
        }}
        selections = {
            "near x4 / far x4": "/selections/near_x4_far_x4.npz",
            "near x5 / far x4": "/selections/near_x5_far_x4.npz",
            "near x8 / far x6": "/selections/near_x8_far_x6.npz",
            "near x2 / far x3": "/selections/near_x2_far_x3.npz",
        }
        viewer = PlayViewer(1, model, pipeline, {"checkpoint": "sample"},
                            acquisition_options=selections)
        assert viewer._acquisition_mode.value == "稀疏"
        assert viewer._acquisition_near.value == "5"
        assert viewer._acquisition_far.value == "4"
        assert viewer._selection_status("sparse", pipeline.scan_identity) == "近端×5，远端×4"
        viewer._acquisition_near.update("4")
        assert viewer._acquisition_far.value == "4"
        assert viewer.pending.take()["acquisition"] is None
        viewer._acquisition_apply.click()
        intent = viewer.pending.take()
        assert intent["acquisition"] == {
            "mode": "sparse", "selection_path": "/selections/near_x4_far_x4.npz",
        }
        viewer._acquisition_near.update("8")
        assert viewer._acquisition_far.value == "4"
        assert viewer._acquisition_apply.disabled is True
        viewer._acquisition_apply.click()
        assert viewer.pending.take()["acquisition"] is None
        assert "未找到" in viewer._acquisition_status.content
        viewer._acquisition_far.update("6")
        assert viewer._acquisition_apply.disabled is False
        viewer._acquisition_apply.click()
        assert viewer.pending.take()["acquisition"] == {
            "mode": "sparse", "selection_path": "/selections/near_x8_far_x6.npz",
        }
        assert "catalog_identity" not in str(viewer._selection_status("sparse", pipeline.scan_identity))
        assert viewer._selection_status("sparse", {
            "selection": {"npz_sha256": "abcdefgh1234", "selection": {"total_count": 1536}}
        }) == "稀疏选集 · sha256=abcdefgh · 射线数=1536"
        viewer._acquisition_mode.value = "全量"
        viewer._acquisition_apply.click()
        assert viewer.pending.take()["acquisition"] == {
            "mode": "full", "selection_path": None,
        }
        viewer.set_acquisition_status("Preparing new full-raycast session…")
        assert "Preparing" in viewer._acquisition_status.content
        viewer.pending.submit_exit()
        viewer.pending.submit_step()
        viewer.pending.clear_session_intents()
        pending = viewer.pending.take()
        assert pending["exit"] is True
        assert pending["step"] is False
        viewer._command[0].value = 0.25
        viewer._command[1].value = -0.1
        viewer._command[2].value = 0.2
        viewer._command_actions.click("应用")
        intent = viewer.pending.take()
        assert intent["command"] == (0.25, -0.1, 0.2)
        viewer.set_command_inputs((0.3, 0.2, -0.4))
        assert [control.value for control in viewer._command] == [0.3, 0.2, -0.4]
        assert viewer.pending.take()["command"] is None
        viewer._playback.click("运行/暂停")
        assert viewer.pending.take()["pause_toggle"] is True

        snapshot = _snapshot()
        viewer.render(snapshot)
        assert "/points/near" in viewer._handles
        assert "/points/far" in viewer._handles
        assert viewer._cone.value is False
        assert "/cone/b8" not in viewer._handles
        viewer._cone.update(True)
        assert viewer.pending.take()["rerender"] is True
        viewer.render(snapshot)
        assert "/cone/b8" in viewer._handles
        assert "/envelope/prism" in viewer._handles
        assert "/physical_plane" in viewer._handles
        assert not any("ground" in name for name in viewer._handles)
        replacement = _Pipeline()
        replacement.scan_identity = {"mode": "sparse", "selection": {
            "selection": {"near_multiplier": 8, "far_multiplier": 6},
        }}
        replacement.cfg.sparse_selection_path = "/selections/near_x8_far_x6.npz"
        viewer.rebind_session(model, replacement, {"checkpoint": "replacement"})
        assert viewer.pipeline is replacement
        assert viewer._acquisition_mode.value == "稀疏"
        assert viewer._acquisition_near.value == "8"
        assert viewer._acquisition_far.value == "6"
        assert viewer.identity_header["checkpoint"] == "replacement"
        assert viewer._handles == {}
        assert viewer._last_snapshot is None
        viewer.render(snapshot)
        before = len(viewer._handles)
        viewer.set_pick_result(resolve_point_pick(snapshot, "near", 0))
        assert "/picker/highlight" in viewer._handles
        assert len(viewer._handles) == before + 1
        changed = _snapshot()
        changed["input"]["near"]["scan_id"] = 2
        viewer.render(changed)
        assert viewer._last_pick is None
        assert "/picker/highlight" not in viewer._handles
        viewer.set_pick_result(resolve_point_pick(snapshot, "near", 0))
        assert viewer._last_pick is None  # late result from the prior scan is discarded
        client = SimpleNamespace(camera=SimpleNamespace(position=None, look_at=None))
        viewer.server._connect_callback(client)
        assert np.allclose(client.camera.look_at, [0.0, 0.0, 0.1])
        assert np.allclose(client.camera.position, [2.4, -2.4, 1.4])
        viewer._view.value = "世界"
        world = viewer._display_points(np.array([[1.0, 0.0, 0.0]]), snapshot)
        assert np.allclose(world, [[4.0, 4.0, 0.46]])
        viewer.close()
        assert viewer.server.stopped
        full_pipeline = _Pipeline()
        full_pipeline.scan_identity = {"mode": "full"}
        full_pipeline.cfg.sparse_selection_path = "/selections/near_x8_far_x6.npz"
        full_viewer = PlayViewer(1, model, full_pipeline, {}, acquisition_options=selections)
        assert full_viewer._acquisition_near.value == "4"
        assert full_viewer._acquisition_far.value == "4"
        full_viewer._acquisition_apply.click()
        assert full_viewer.pending.take()["acquisition"] == {"mode": "full", "selection_path": None}
        full_viewer.close()
        uncommon_pipeline = _Pipeline()
        uncommon_pipeline.scan_identity = {"mode": "sparse", "selection": {
            "selection": {"near_multiplier": 2, "far_multiplier": 3},
        }}
        uncommon_pipeline.cfg.sparse_selection_path = "/selections/near_x2_far_x3.npz"
        uncommon_viewer = PlayViewer(1, model, uncommon_pipeline, {}, acquisition_options=selections)
        assert uncommon_viewer._acquisition_near.value == "2"
        assert uncommon_viewer._acquisition_far.value == "3"
        uncommon_viewer._acquisition_apply.click()
        assert uncommon_viewer.pending.take()["acquisition"] == {
            "mode": "sparse", "selection_path": "/selections/near_x2_far_x3.npz",
        }
        uncommon_viewer.close()
    finally:
        if previous is None:
            sys.modules.pop("viser", None)
        else:
            sys.modules["viser"] = previous


def test_occupied_port_does_not_silently_bind_another_port():
    created = []
    class OccupiedServer(_Server):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.port += 1
            created.append(self)
    previous = sys.modules.get("viser")
    sys.modules["viser"] = SimpleNamespace(ViserServer=OccupiedServer)
    try:
        try:
            PlayViewer(1, None, _Pipeline(), {}, port=8094)
        except OSError as error:
            assert "occupied" in str(error)
        else:
            raise AssertionError("unexpected port substitution must be rejected")
        assert len(created) == 1 and created[0].stopped
    finally:
        if previous is None:
            sys.modules.pop("viser", None)
        else:
            sys.modules["viser"] = previous
