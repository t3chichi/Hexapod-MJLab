"""Read-only local Viser rendering for confined_locomotion_2 play snapshots.

All scan xyz values are acquisition-time BASE coordinates. WORLD mode applies
the current pose for display only; it is an uncompensated projection and does
not reconstruct where old rays were acquired in world coordinates. GUI
callbacks only write :class:`PendingControls`; this module never accesses the
runtime, network, scanner, or environment state.
"""

from __future__ import annotations

import ipaddress
import json
import re
import threading
from pathlib import Path

import numpy as np

from hexapod_mjlab.common.visualization import (
    PendingControls,
    quat_wxyz_to_matrix,
    robot_meshes_at_qpos,
)

from .envelope import envelope_prism_base

PLAY_SNAPSHOT_SCHEMA = "confined_locomotion_2.play_snapshot/v1"

__all__ = [
    "PLAY_SNAPSHOT_SCHEMA",
    "PendingControls",
    "PlayViewer",
    "export_snapshot_manifest",
    "resolve_point_pick",
]


class _AcquisitionPendingControls(PendingControls):
    """Legacy intents plus a separate owner-loop acquisition request."""

    def __init__(self):
        super().__init__()
        self._acquisition_lock = threading.Lock()
        self._acquisition = None

    def submit_acquisition(self, mode, selection_path):
        with self._acquisition_lock:
            self._acquisition = {"mode": str(mode), "selection_path": selection_path}

    def take(self):
        pending = super().take()
        with self._acquisition_lock:
            pending["acquisition"] = self._acquisition
            self._acquisition = None
        return pending

    def clear_session_intents(self):
        """Drop controls queued for a session that is being replaced; keep exit."""
        with self._lock:
            self._command = None
            self._zero_command = False
            self._pause_toggle = False
            self._step = False
            self._reset = False
            self._rerender = False
            self._pick = None
            self._env_row = None
        with self._acquisition_lock:
            self._acquisition = None


def _array(value, shape, name, dtype=None):
    result = np.asarray(value, dtype=dtype)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    return result


def validate_snapshot(snapshot: dict) -> None:
    if not isinstance(snapshot, dict) or snapshot.get("schema") != PLAY_SNAPSHOT_SCHEMA:
        raise ValueError(f"expected snapshot schema {PLAY_SNAPSHOT_SCHEMA!r}")
    if not isinstance(snapshot.get("env"), int):
        raise ValueError("snapshot env must be an integer")
    pose = snapshot["pose"]
    _array(pose["root_pos_w"], (3,), "pose.root_pos_w", np.float64)
    _array(pose["root_quat_wxyz"], (4,), "pose.root_quat_wxyz", np.float64)
    _array(pose["qpos"], (np.asarray(pose["qpos"]).size,), "pose.qpos", np.float64)
    for name, count in (("near", 256), ("far", 128)):
        cloud = snapshot["input"][name]
        _array(cloud["points_base"], (count, 3), f"input.{name}.points_base", np.float64)
        _array(cloud["valid"], (count,), f"input.{name}.valid", bool)
    provenance = snapshot["provenance"]
    _array(provenance["near_raw_ids"], (256,), "provenance.near_raw_ids", np.int64)
    _array(provenance["far_raw_ids"], (128,), "provenance.far_raw_ids", np.int64)
    _array(provenance["near_no_hit"], (256,), "provenance.near_no_hit", bool)
    _array(provenance["far_no_hit"], (128,), "provenance.far_no_hit", bool)
    pool = snapshot.get("pool")
    if pool is not None:
        points = np.asarray(pool["points_base"])
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("pool.points_base must be (N,3)")
        _array(pool["near_mask"], (len(points),), "pool.near_mask", bool)
        _array(pool["far_mask"], (len(points),), "pool.far_mask", bool)


def resolve_point_pick(snapshot: dict, kind: str, index: int) -> dict:
    """Resolve an input slot to recorded raw-ray provenance, without XYZ matching."""
    validate_snapshot(snapshot)
    if kind not in ("near", "far"):
        raise ValueError("kind must be 'near' or 'far'")
    count = 256 if kind == "near" else 128
    if isinstance(index, bool) or not isinstance(index, (int, np.integer)) or not 0 <= int(index) < count:
        raise ValueError(f"{kind} index must be in [0,{count})")
    slot = int(index)
    cloud = snapshot["input"][kind]
    prov = snapshot["provenance"]
    valid = bool(cloud["valid"][slot])
    raw_ids = prov[f"{kind}_raw_ids"]
    no_hits = prov[f"{kind}_no_hit"]
    return {
        "kind": kind,
        "index": slot,
        "valid": valid and int(raw_ids[slot]) >= 0,
        "env": int(snapshot["env"]),
        "episode_id": int(snapshot["identity"]["input_episode_id"]),
        "scan_id": int(cloud["scan_id"]),
        "time_s": float(cloud["time_s"]),
        "raw_ray_id": int(raw_ids[slot]),
        "no_hit": bool(no_hits[slot]),
        "xyz_base": np.asarray(cloud["points_base"][slot], dtype=np.float64).copy(),
    }


class PlayViewer:
    """Local Viser viewer with display-only transforms and intent callbacks."""

    def __init__(
        self,
        num_envs,
        mj_model,
        pipeline,
        identity_header,
        host="127.0.0.1",
        port=8094,
        acquisition_options=None,
    ):
        if host != "127.0.0.1" or not ipaddress.ip_address(host).is_loopback:
            raise ValueError("the play viewer must bind exactly to 127.0.0.1")
        if isinstance(port, bool) or not 1 <= int(port) <= 65535:
            raise ValueError("port must be in 1..65535")
        if isinstance(num_envs, bool) or int(num_envs) < 1:
            raise ValueError("num_envs must be a positive integer")
        import viser

        self.num_envs = int(num_envs)
        self.mj_model = mj_model
        self.pipeline = pipeline
        self.identity_header = dict(identity_header)
        self.host = host
        self.port = int(port)
        self.pending = _AcquisitionPendingControls()
        self.acquisition_options = dict(acquisition_options or {})
        self._lock = threading.RLock()
        self._handles = {}
        self._mj_data = None
        self._last_snapshot = None
        self._last_pick = None
        self._last_render_identity = None
        self._last_pick_identity = None
        self._camera_focused = False
        self._closed = False
        self.server = viser.ViserServer(host=host, port=self.port, label="机器人运动预览")
        try:
            actual_port = int(self.server.get_port())
            if actual_port != self.port:
                raise OSError(f"requested viewer port {self.port} is occupied; choose another --port")
            self._setup_gui()
            on_connect = getattr(self.server, "on_client_connect", None)
            if callable(on_connect):
                on_connect(self._focus_client_camera)
        except BaseException:
            self.close()
            raise

    def _setup_gui(self):
        gui = self.server.gui
        with gui.add_folder("运行控制", order=0):
            choices = [f"环境 {i}" for i in range(self.num_envs)]
            self._env = gui.add_dropdown("环境", choices, initial_value=choices[0])
            self._playback = gui.add_button_group("播放", ["运行/暂停", "单步", "重置", "退出"])
            self._command_actions = gui.add_button_group("指令", ["应用", "清零"])
            self._command = [
                gui.add_slider("前进 (m/s)", -1.5, 1.5, step=0.05, initial_value=0.0),
                gui.add_slider("侧移 (m/s)", -1.0, 1.0, step=0.05, initial_value=0.0),
                gui.add_slider("转向 (rad/s)", -1.0, 1.0, step=0.05, initial_value=0.0),
            ]
        self._layers_folder = gui.add_folder("显示图层", order=1)
        with self._layers_folder:
            self._view = gui.add_dropdown("坐标系", ["机器人", "世界"], initial_value="机器人")
            self._near = gui.add_checkbox("近端点云（256）", True)
            self._far = gui.add_checkbox("远端点云（128）", True)
            self._envelope = gui.add_checkbox("预测包络（绿）", True)
            self._pool = gui.add_checkbox("采样候选点", False)
            self._cone = gui.add_checkbox("近端圆锥线", False)
        with gui.add_folder("点源查询", order=4, expand_by_default=False):
            self._pick_kind = gui.add_dropdown("点云", ["近端", "远端"], initial_value="近端")
            self._pick_near_index = gui.add_number("近端序号", 0, min=0, max=255, step=1)
            self._pick_far_index = gui.add_number("远端序号", 0, min=0, max=127, step=1)
            self._pick_button = gui.add_button("查询")
            self._pick_panel = gui.add_markdown("")
        self._status_folder = gui.add_folder("状态信息", order=5)
        with self._status_folder:
            self._status = gui.add_markdown("")
            with gui.add_folder("模型与扫描详情", expand_by_default=False):
                self._details = gui.add_markdown("")

        self._acquisition_mode = None
        self._acquisition_near = None
        self._acquisition_far = None
        self._acquisition_custom_selection = None
        self._acquisition_custom_active = False
        self._acquisition_apply = None
        with gui.add_folder("扫描设置", order=3, expand_by_default=False):
            gui.add_markdown("应用后重置回合，并保持暂停。")
            self._acquisition_status = gui.add_markdown("")
            if self.acquisition_options:
                scan_identity = getattr(self.pipeline, "scan_identity", {}) or {}
                mode = scan_identity.get("mode", "full")
                mode = mode if mode in ("full", "sparse") else "full"
                self._acquisition_mode = gui.add_dropdown(
                    "扫描模式", ["全量", "稀疏"], initial_value={"full": "全量", "sparse": "稀疏"}[mode]
                )
                current_path = getattr(getattr(self.pipeline, "cfg", None), "sparse_selection_path", None)
                near, far = self._initial_multipliers(
                    scan_identity, current_path if mode == "sparse" else None
                )
                near_choices = self._multiplier_choices(0)
                far_choices = self._multiplier_choices(1)
                self._acquisition_near = gui.add_dropdown(
                    "近端倍率", near_choices, initial_value=near
                )
                self._acquisition_far = gui.add_dropdown(
                    "远端倍率", far_choices, initial_value=far
                )
                custom_labels = self._custom_selection_labels()
                if custom_labels:
                    custom_initial = self._initial_custom_selection(scan_identity, current_path, custom_labels)
                    self._acquisition_custom_selection = gui.add_dropdown(
                        "自定义选集", custom_labels, initial_value=custom_initial
                    )
                self._acquisition_apply = gui.add_button("应用并重置")
                self._acquisition_apply.on_click(self._submit_acquisition)
                self._acquisition_mode.on_update(lambda _event: self._update_acquisition_controls())
                self._acquisition_near.on_update(lambda _event: self._multiplier_selection_changed())
                self._acquisition_far.on_update(lambda _event: self._multiplier_selection_changed())
                if self._acquisition_custom_selection is not None:
                    self._acquisition_custom_selection.on_update(lambda _event: self._custom_selection_changed())
                    self._acquisition_custom_active = (
                        self._acquisition_custom_selection.value in self._custom_selection_labels()
                        and self._path_is_custom(current_path)
                    )
                self._update_acquisition_controls()


        self._playback.on_click(self._submit_playback)
        self._command_actions.on_click(self._submit_command_action)
        self._pick_button.on_click(
            lambda _event: self.pending.submit_pick(
                {"近端": "near", "远端": "far"}[self._pick_kind.value],
                int(self._pick_near_index.value if self._pick_kind.value == "近端"
                    else self._pick_far_index.value),
            )
        )
        self._env.on_update(lambda _event: self.pending.submit_env_row(self.selected_env_row))
        for control in (self._view, self._near, self._far, self._pool, self._cone, self._envelope):
            control.on_update(lambda _event: self.pending.submit_rerender())

    def _submit_playback(self, _event):
        {"运行/暂停": self.pending.submit_pause_toggle, "单步": self.pending.submit_step,
         "重置": self.pending.submit_reset, "退出": self.pending.submit_exit}[self._playback.value]()

    def _submit_command_action(self, _event):
        if self._command_actions.value == "应用":
            self.pending.submit_command(*(float(x.value) for x in self._command))
        else:
            self.pending.submit_zero_command()

    def _selected_acquisition_mode(self):
        return {"全量": "full", "稀疏": "sparse"}[self._acquisition_mode.value]

    @staticmethod
    def _selection_pair(label):
        match = re.fullmatch(r"\s*near\s*x(\d+)\s*/\s*far\s*x(\d+)\s*", str(label), re.IGNORECASE)
        return (match.group(1), match.group(2)) if match else None

    def _custom_selection_labels(self):
        return [label for label in self.acquisition_options if self._selection_pair(label) is None]

    def _multiplier_choices(self, side):
        choices = {4, 5, 6, 8}
        for label in self.acquisition_options:
            pair = self._selection_pair(label)
            if pair is not None:
                value = int(pair[side])
                if value > 0:
                    choices.add(value)
        return [str(value) for value in sorted(choices)]

    def _initial_multipliers(self, scan_identity, current_path=None):
        pair = None
        if current_path is not None:
            try:
                current = Path(current_path).expanduser().resolve()
                pair = next((self._selection_pair(label) for label, candidate in self.acquisition_options.items()
                             if Path(candidate).expanduser().resolve() == current and self._selection_pair(label)), None)
            except (OSError, TypeError, ValueError):
                pass
        selection = scan_identity.get("selection") if isinstance(scan_identity, dict) else None
        if pair is None and isinstance(selection, dict):
            metadata = selection.get("selection", selection)
            near = metadata.get("near_multiplier") if isinstance(metadata, dict) else None
            far = metadata.get("far_multiplier") if isinstance(metadata, dict) else None
            if near is not None and far is not None:
                candidate = (str(near), str(far))
                if any(self._selection_pair(label) == candidate for label in self.acquisition_options):
                    pair = candidate
        return pair or ("4", "4")

    def _initial_custom_selection(self, scan_identity, current_path, labels):
        if current_path is not None:
            try:
                current = Path(current_path).expanduser().resolve()
                for label in labels:
                    if Path(self.acquisition_options[label]).expanduser().resolve() == current:
                        return label
            except (OSError, TypeError, ValueError):
                pass
        selection = scan_identity.get("selection") if isinstance(scan_identity, dict) else None
        def normalize(value):
            return re.sub(r"[^a-z0-9]+", "", str(value).lower())
        text = json.dumps(selection, sort_keys=True, default=str).lower()
        compact_text = normalize(text)
        for label in labels:
            compact = normalize(label)
            if compact and compact in compact_text:
                return label
        return labels[0]

    def _path_is_custom(self, path):
        if path is None:
            return False
        try:
            current = Path(path).expanduser().resolve()
            return any(Path(self.acquisition_options[label]).expanduser().resolve() == current
                       for label in self._custom_selection_labels())
        except (OSError, TypeError, ValueError):
            return False

    def _multiplier_selection_changed(self):
        self._acquisition_custom_active = False
        self._update_acquisition_controls()

    def _custom_selection_changed(self):
        self._acquisition_custom_active = True
        self._update_acquisition_controls()

    def _selected_sparse_path(self):
        if self._acquisition_custom_active and self._acquisition_custom_selection is not None:
            label = str(self._acquisition_custom_selection.value)
            if label in self._custom_selection_labels():
                return self.acquisition_options[label]
        expected = (str(self._acquisition_near.value), str(self._acquisition_far.value))
        for label, path in self.acquisition_options.items():
            if self._selection_pair(label) == expected:
                return path
        return None

    def _update_acquisition_controls(self):
        if self._acquisition_apply is None:
            return
        mode = self._selected_acquisition_mode()
        path = self._selected_sparse_path() if mode == "sparse" else None
        self._acquisition_apply.disabled = mode == "sparse" and path is None
        if mode == "sparse" and path is None:
            status = (f"未找到近端×{self._acquisition_near.value}、远端×{self._acquisition_far.value}的选集，请更换倍率。")
        elif mode == "sparse" and self._acquisition_custom_active:
            custom = str(self._acquisition_custom_selection.value)
            status = (f"自定义选集：{custom}" if custom in self._custom_selection_labels()
                      else f"稀疏扫描：近端×{self._acquisition_near.value}，远端×{self._acquisition_far.value}")
        elif mode == "sparse":
            status = f"稀疏扫描：近端×{self._acquisition_near.value}，远端×{self._acquisition_far.value}"
        else:
            status = "全量扫描。"
        self._acquisition_status.content = status

    def _submit_acquisition(self, _event):
        mode = self._selected_acquisition_mode()
        path = self._selected_sparse_path() if mode == "sparse" else None
        if mode == "sparse" and path is None:
            self._update_acquisition_controls()
            return
        self.pending.submit_acquisition(mode, path)

    def set_acquisition_status(self, status):
        """Show owner-loop acquisition progress or an error without side effects."""
        self._acquisition_status.content = str(status)

    def _clear_session_display(self):
        for name in list(self._handles):
            self._drop(name)
        self._last_snapshot = None
        self._last_pick = None
        self._last_render_identity = None
        self._last_pick_identity = None
        self._mj_data = None
        self._camera_focused = False

    def rebind_session(self, mj_model, pipeline, identity_header=None):
        """Bind a newly created owner session while preserving Viser and GUI callbacks."""
        with self._lock:
            self._clear_session_display()
            self.pending.clear_session_intents()
            self.mj_model = mj_model
            self.pipeline = pipeline
            if identity_header is not None:
                self.identity_header = dict(identity_header)
            if self._acquisition_mode is not None:
                scan_identity = getattr(pipeline, "scan_identity", {}) or {}
                mode = scan_identity.get("mode", "full")
                self._acquisition_mode.value = {"full": "全量", "sparse": "稀疏"}.get(mode, "全量")
                current_path = getattr(getattr(pipeline, "cfg", None), "sparse_selection_path", None)
                near, far = self._initial_multipliers(
                    scan_identity, current_path if self._selected_acquisition_mode() == "sparse" else None
                )
                self._acquisition_near.value = near
                self._acquisition_far.value = far
                if self._acquisition_custom_selection is not None:
                    self._acquisition_custom_selection.value = self._initial_custom_selection(
                        scan_identity, current_path, self._custom_selection_labels()
                    )
                    self._acquisition_custom_active = self._path_is_custom(current_path)
                self._update_acquisition_controls()

    def detach_session(self):
        """Release viewer references before the owner tears down the old session."""
        with self._lock:
            self._clear_session_display()
            self.pending.clear_session_intents()
            self.pipeline = None
            self.mj_model = None

    @property
    def selected_env_row(self):
        return int(str(self._env.value).split()[-1])

    @property
    def view_mode(self):
        return {"机器人": "base", "世界": "world"}[self._view.value]

    def set_command_inputs(self, values):
        """Update displayed sliders only; do not submit a command intent."""
        try:
            values = tuple(float(value) for value in values)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("command inputs must contain three numeric values") from error
        if len(values) != 3 or not np.isfinite(values).all():
            raise ValueError("command inputs must contain three finite values")
        ranges = ((-1.5, 1.5), (-1.0, 1.0), (-1.0, 1.0))
        if any(not lo <= value <= hi for value, (lo, hi) in zip(values, ranges)):
            raise ValueError("command inputs are outside the displayed slider ranges")
        for control, value in zip(self._command, values):
            control.value = value

    def _camera_target_position(self, snapshot):
        if snapshot is None or self.view_mode == "base":
            target = np.array([0.0, 0.0, 0.1])
        else:
            target = np.asarray(snapshot["pose"]["root_pos_w"], dtype=np.float64).copy()
            target += np.array([0.0, 0.0, 0.1])
        return target, target + np.array([2.4, -2.4, 1.3])

    def _focus_client_camera(self, client):
        target, position = self._camera_target_position(self._last_snapshot)
        try:
            client.camera.position = position
            client.camera.look_at = target
        except Exception:
            pass

    def _display_points(self, points_base, snapshot):
        points = np.asarray(points_base, dtype=np.float64)
        if self.view_mode == "base":
            return points
        pose = snapshot["pose"]
        rot = quat_wxyz_to_matrix(pose["root_quat_wxyz"])
        return points @ rot.T + np.asarray(pose["root_pos_w"], dtype=np.float64)

    def _drop(self, name):
        handle = self._handles.pop(name, None)
        if handle is not None:
            try:
                handle.remove()
            except Exception:
                pass

    def _swap(self, name, factory):
        self._drop(name)
        self._handles[name] = factory()

    def _render_cloud(self, name, points, color, size, mask):
        if not bool(np.asarray(mask, dtype=bool).any()):
            self._drop(name)
            return
        xyz = self._display_points(np.asarray(points)[mask], self._last_snapshot).astype(np.float32)
        self._swap(name, lambda: self.server.scene.add_point_cloud(
            name, xyz, colors=color, point_size=size, precision="float32"
        ))

    def render(self, snapshot):
        """Render only the supplied immutable snapshot; never queries live state."""
        if self.pipeline is None or self.mj_model is None:
            raise RuntimeError("viewer has no attached play session")
        validate_snapshot(snapshot)
        if not 0 <= int(snapshot["env"]) < self.num_envs:
            raise ValueError("snapshot env row is outside the viewer's batch")
        with self._lock:
            render_identity = (
                int(snapshot["env"]),
                int(snapshot["identity"]["input_episode_id"]),
                int(snapshot["input"]["near"]["scan_id"]),
                int(snapshot["input"]["far"]["scan_id"]),
            )
            if self._last_render_identity is not None and render_identity != self._last_render_identity:
                self._last_pick = None
                self._last_pick_identity = None
            self._last_render_identity = render_identity
            self._last_snapshot = snapshot
            scene = self.server.scene
            pose = snapshot["pose"]

            if self._mj_data is None:
                import mujoco
                self._mj_data = mujoco.MjData(self.mj_model)
            meshes = robot_meshes_at_qpos(self.mj_model, pose["qpos"], mj_data=self._mj_data)
            for key in [key for key in self._handles if key.startswith("/robot/")]:
                self._drop(key)
            if self.view_mode == "base":
                rot = quat_wxyz_to_matrix(pose["root_quat_wxyz"])
                origin = np.asarray(pose["root_pos_w"], dtype=np.float64)
                for mesh in meshes:
                    mesh["vertices"] = (mesh["vertices"] - origin) @ rot
            for mesh in meshes:
                name = f"/robot/{mesh['body']}/{mesh['geom']}"
                vertices = np.asarray(mesh["vertices"], dtype=np.float32)
                faces = np.asarray(mesh["faces"], dtype=np.int32)
                self._swap(name, lambda name=name, vertices=vertices, faces=faces: scene.add_mesh_simple(
                    name, vertices, faces, color=(168, 175, 185), side="double"
                ))

            self._draw_ground(snapshot)

            inp = snapshot["input"]
            provenance = snapshot["provenance"]
            near, far = inp["near"], inp["far"]
            near_valid = np.asarray(near["valid"], dtype=bool)
            far_valid = np.asarray(far["valid"], dtype=bool)
            near_hit = near_valid & ~np.asarray(provenance["near_no_hit"], dtype=bool)
            near_nohit = near_valid & np.asarray(provenance["near_no_hit"], dtype=bool)
            far_nohit = far_valid & np.asarray(provenance["far_no_hit"], dtype=bool)
            if self._near.value:
                self._render_cloud("/points/near", near["points_base"], (0, 170, 255), 0.055, near_hit)
                self._render_cloud("/points/near_nohit", near["points_base"], (255, 145, 0), 0.065, near_nohit)
            else:
                self._drop("/points/near")
                self._drop("/points/near_nohit")
            if self._far.value:
                self._render_cloud("/points/far", far["points_base"], (185, 105, 255), 0.045, far_valid & ~far_nohit)
                self._render_cloud("/points/far_nohit", far["points_base"], (255, 70, 70), 0.06, far_nohit)
            else:
                self._drop("/points/far")
                self._drop("/points/far_nohit")

            pool = snapshot.get("pool")
            if self._pool.value and pool is not None:
                self._render_cloud("/pool/near", pool["points_base"], (0, 230, 190), 0.018, pool["near_mask"])
                self._render_cloud("/pool/far", pool["points_base"], (205, 145, 255), 0.018, pool["far_mask"])
            else:
                self._drop("/pool/near")
                self._drop("/pool/far")

            cfg = self.pipeline.cfg
            if self._cone.value:
                segments = self.pipeline.fixed_boundary_segments
                world_lines = segments if self.view_mode == "base" else self._display_points(segments.reshape(-1, 3), snapshot).reshape(-1, 2, 3)
                self._swap("/cone/b8", lambda: scene.add_line_segments(
                    "/cone/b8", world_lines.astype(np.float32), colors=(255, 220, 50),
                    thickness=1.5, thickness_units="screen"
                ))
            else:
                self._drop("/cone/b8")

            if self._envelope.value:
                self._draw_envelope(snapshot)
            else:
                self._drop("/envelope/prism")
            self._draw_task_layers(snapshot)
            self._render_pick_highlight()
            self._update_panels(snapshot)
            self._focus_camera_once(snapshot)

    def _draw_ground(self, snapshot):
        scene = self.server.scene
        # Physical plane reference only; never a prediction/label surface.
        floor = np.asarray(self.pipeline.mesh_record["vertices"], dtype=np.float64).reshape(-1, 3)
        floor_faces = np.asarray(self.pipeline.mesh_record["triangles"], dtype=np.int32).reshape(-1, 3)
        if self.view_mode == "base":
            rot = quat_wxyz_to_matrix(snapshot["pose"]["root_quat_wxyz"])
            floor = (floor - np.asarray(snapshot["pose"]["root_pos_w"], dtype=np.float64)) @ rot
        floor = floor.astype(np.float32)
        self._swap("/physical_plane", lambda: scene.add_mesh_simple(
            "/physical_plane", floor, floor_faces, color=(115, 125, 115), side="double"
        ))

    def _draw_task_layers(self, snapshot):
        pass

    def _draw_envelope(self, snapshot):
        self._draw_envelope_values(snapshot, snapshot["prediction"]["envelope"])

    def _draw_envelope_values(self, snapshot, envelope, *, prefix="/envelope",
                              color=(60, 245, 160)):
        # Attach the retained six values to the DISPLAYED robot's full Root
        # frame, including roll/pitch. Height-map diagnostics use their own
        # pre-action yaw-only world frame and are not changed here.
        lines_base = envelope_prism_base(envelope)
        lines = self._display_points(lines_base.reshape(-1, 3), snapshot).reshape(18, 2, 3).astype(np.float32)
        self._swap(prefix + "/prism", lambda: self.server.scene.add_line_segments(
            prefix + "/prism", lines, colors=color, thickness=2.0,
            thickness_units="screen"
        ))

    def _render_pick_highlight(self):
        pick = self._last_pick
        if pick is None or not pick.get("valid"):
            self._drop("/picker/highlight")
            return
        xyz = self._display_points(np.asarray(pick["xyz_base"]).reshape(1, 3), self._last_snapshot)[0]
        self._swap("/picker/highlight", lambda: self.server.scene.add_icosphere(
            "/picker/highlight", radius=0.06, color=(255, 0, 255), subdivisions=2,
            position=xyz.astype(np.float32)
        ))

    def _focus_camera_once(self, snapshot):
        if self._camera_focused:
            return
        self._camera_focused = True
        target, position = self._camera_target_position(snapshot)
        try:
            initial = getattr(self.server, "initial_camera", None)
            if initial is not None:
                initial.position = position
                initial.look_at = target
            get_clients = getattr(self.server, "get_clients", None)
            if callable(get_clients):
                for client in get_clients().values():
                    client.camera.position = position
                    client.camera.look_at = target
        except Exception:
            # Camera focus is a display convenience and cannot affect play.
            pass

    def _update_panels(self, snapshot):
        ident, inp, status = snapshot["identity"], snapshot["input"], snapshot["status"]
        prediction = snapshot["prediction"]
        evaluation = snapshot.get("scan_evaluation", {}) or {}
        trained = evaluation.get("checkpoint_scan_identity") or {}
        actual = evaluation.get("runtime_scan_identity") or getattr(self.pipeline, "scan_identity", {}) or {}
        mode_names = {"full": "全量", "sparse": "稀疏"}
        trained_mode = mode_names.get(trained.get("mode"), "未记录")
        actual_mode = mode_names.get(actual.get("mode"), "未记录")
        sampling = getattr(getattr(self.pipeline, "cfg", None), "far_sampling", "all_endpoints")
        sampling_text = {"hit_only_10m": "仅命中且不超过10米", "all_endpoints": "全部端点"}.get(sampling, sampling)
        far_count = int(np.asarray(inp["far"]["valid"], dtype=bool).sum())
        height = snapshot["pose"]["root_height_m"]
        height_text = f"{height:.3f} m" if height is not None else "超出地形"
        stop_text = "场地边界暂停" if status.get("stop_kind") == "play_terrain_boundary_stop" else "跌倒暂停"
        state = ("故障停止" if status["faulted"] else stop_text if status["stopped"]
                 else "已暂停" if status["paused"] else "运行中")
        task_id = self.identity_header.get("task", "")
        task_name = "地形记忆" if "Terrain-Memory" in task_id else "平地基线"
        self._status.content = (
            f"**{state}** · {task_name} · 环境 {snapshot['env']}\n\n"
            f"回合 {ident['input_episode_id']} · 步 {ident['input_step_in_episode']} · 离地 {height_text}\n\n"
            f"指令 {np.asarray(status['command']).round(2).tolist()}\n\n"
            f"控制步：输入 {ident['input_control_step']} / 姿态 {ident['pose_control_step']}"
        )
        self._details.content = (
            f"**任务** {task_id}\n\n"
            f"**模型** `{snapshot.get('checkpoint', {}).get('path', '')}`\n\n"
            f"**扫描** 模型：{trained_mode}；当前：{actual_mode}；"
            f"配置变更：{'是' if evaluation.get('cross_configuration', False) else '否'}\n\n"
            f"{self._selection_status(actual.get('mode'), actual)}；远端规则：{sampling_text}；有效 {far_count}/128。\n\n"
            f"近端帧 {inp['near']['scan_id']} @ {inp['near']['time_s']:.3f} s；"
            f"远端帧 {inp['far']['scan_id']} @ {inp['far']['time_s']:.3f} s。\n\n"
            f"六维预测 {np.asarray(prediction['envelope']).round(3).tolist()}。\n\n"
            "点云保留采集时的机器人坐标；世界视图按当前姿态显示，不做运动补偿。"
        )
        pick = self._last_pick
        if pick is None:
            self._pick_panel.content = "选择点云和序号后点击“查询”。"
        else:
            kind = {"near": "近端", "far": "远端"}.get(pick['kind'], pick['kind'])
            if pick.get("valid"):
                self._pick_panel.content = (
                    f"**{kind}[{pick['index']}]** · 原始射线 {pick['raw_ray_id']}\n\n"
                    f"{'未命中' if pick['no_hit'] else '命中'} · 帧 {pick['scan_id']} @ {pick['time_s']:.3f} s · "
                    f"回合 {pick['episode_id']}\n\n"
                    f"机器人坐标 {np.asarray(pick['xyz_base']).round(4).tolist()}"
                )
            else:
                self._pick_panel.content = f"{kind}[{pick['index']}] 为填充点或无效点。"

    @staticmethod
    def _selection_status(mode, runtime_identity):
        if mode != "sparse" or not isinstance(runtime_identity, dict):
            return "全量扫描"
        selection_identity = runtime_identity.get("selection")
        if not isinstance(selection_identity, dict):
            return "稀疏选集未记录"
        metadata = selection_identity.get("selection", {})
        if isinstance(metadata, dict):
            near = metadata.get("near_multiplier")
            far = metadata.get("far_multiplier")
            if near is not None and far is not None:
                return f"近端×{near}，远端×{far}"
            ray_count = metadata.get("total_count")
        else:
            ray_count = None
        digest = str(selection_identity.get("npz_sha256", ""))[:8]
        parts = ["稀疏选集"]
        if digest:
            parts.append(f"sha256={digest}")
        if ray_count is not None:
            parts.append(f"射线数={ray_count}")
        return " · ".join(parts)

    def set_pick_result(self, pick):
        """Display a point-query result produced by the owner loop."""
        with self._lock:
            candidate = None if pick is None else dict(pick)
            if candidate is not None and self._last_snapshot is not None:
                kind = candidate.get("kind")
                if kind not in ("near", "far"):
                    candidate = None
                else:
                    expected = (
                        int(self._last_snapshot["env"]),
                        int(self._last_snapshot["identity"]["input_episode_id"]),
                        int(self._last_snapshot["input"][kind]["scan_id"]),
                    )
                    actual = (
                        int(candidate["env"]),
                        int(candidate["episode_id"]),
                        int(candidate["scan_id"]),
                    )
                    if actual != expected:
                        candidate = None
            self._last_pick = candidate
            self._last_pick_identity = (
                None if candidate is None else (
                    int(candidate["env"]), int(candidate["episode_id"]), int(candidate["scan_id"])
                )
            )
            if self._last_snapshot is not None:
                self._render_pick_highlight()
                self._update_panels(self._last_snapshot)

    def close(self):
        if self._closed:
            return
        self._closed = True
        server = getattr(self, "server", None)
        if server is not None:
            stop = getattr(server, "stop", None)
            if callable(stop):
                try:
                    stop()
                except Exception:
                    pass


def _clip_cone_x(segments, xmax):
    clipped = []
    for p0, p1 in np.asarray(segments, dtype=np.float64).reshape(-1, 2, 3):
        a, b = p0.copy(), p1.copy()
        if a[0] > xmax and b[0] > xmax:
            continue
        if a[0] > xmax or b[0] > xmax:
            t = (xmax - a[0]) / (b[0] - a[0])
            bound = a + t * (b - a)
            bound[0] = xmax
            if a[0] > xmax:
                a = bound
            else:
                b = bound
        clipped.append((a, b))
    return np.asarray(clipped, dtype=np.float64).reshape(-1, 2, 3)


def export_snapshot_manifest(snapshot, viewer, path):
    """Write a compact manifest with explicit source-frame and no-ground semantics."""
    validate_snapshot(snapshot)
    near = snapshot["input"]["near"]
    far = snapshot["input"]["far"]
    nv = np.asarray(near["valid"], dtype=bool)
    fv = np.asarray(far["valid"], dtype=bool)
    near_nohit = np.asarray(snapshot["provenance"]["near_no_hit"], dtype=bool)
    far_nohit = np.asarray(snapshot["provenance"]["far_no_hit"], dtype=bool)
    manifest = {
        "schema": "confined_locomotion_2.play_snapshot_manifest/v1",
        "snapshot_schema": snapshot["schema"],
        "env": int(snapshot["env"]),
        "identity": dict(snapshot["identity"]),
        "status": {**snapshot["status"], "command": np.asarray(snapshot["status"]["command"]).tolist()},
        "prediction": {"envelope": np.asarray(snapshot["prediction"]["envelope"]).tolist(),
                       "preview": bool(snapshot["prediction"]["preview"])},
        "checkpoint": dict(snapshot["checkpoint"]),
        "view_mode": None if viewer is None else viewer.view_mode,
        "viewer_identity_header": None if viewer is None else dict(viewer.identity_header),
        "scene_objects": None if viewer is None else sorted(viewer._handles),
        "counts": {
            "near_valid": int(nv.sum()),
            "near_hit": int((nv & ~near_nohit).sum()),
            "near_no_hit": int((nv & near_nohit).sum()),
            "far_valid": int(fv.sum()),
            "far_hit": int((fv & ~far_nohit).sum()),
            "far_no_hit": int((fv & far_nohit).sum()),
            "far_window_slots": 1,
            "pool_points": 0 if snapshot.get("pool") is None else int(len(snapshot["pool"]["points_base"])),
        },
        "provenance": {
            name: {
                "scan_id": int(snapshot["input"][name]["scan_id"]),
                "time_s": float(snapshot["input"][name]["time_s"]),
                "raw_ray_ids": np.asarray(snapshot["provenance"][f"{name}_raw_ids"], dtype=np.int64).tolist(),
                "valid": np.asarray(snapshot["input"][name]["valid"], dtype=bool).tolist(),
                "no_hit": np.asarray(snapshot["provenance"][f"{name}_no_hit"], dtype=bool).tolist(),
            }
            for name in ("near", "far")
        },
        "source_coordinate_semantics": (
            "all scan xyz are acquisition-time BASE coordinates; WORLD view applies the current pose as an uncompensated display projection"
        ),
        "ground_semantics": "no ground prediction, label, candidate rectangle, or ground panel is present",
        "data_source": "owner-loop snapshot only; viewer does not step runtime or run network",
        "envelope_display": "18-edge prism attached to displayed Root frame including roll/pitch; local bottom_z=0.02-0.46, top_z=delta_z; not collision clearance",
    }
    if "terrain" in snapshot:
        data = snapshot["terrain"]
        manifest["terrain"] = {key: (value.tolist() if isinstance(value, np.ndarray) else value)
                               for key, value in data.items()}
        manifest["ground_semantics"] = "pre-action yaw-only XY; world_z=root_z+signed_height; geometry mask independent of visibility; diagnostic only"
    if path is not None:
        Path(path).write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str))
    return manifest
