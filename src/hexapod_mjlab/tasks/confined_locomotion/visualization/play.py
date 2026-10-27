"""Live task-play visualization: provenance mapping + local Viser viewer.

LIVE, not replay: this module renders the display snapshots produced by the
task play owner loop (:mod:`.play`). It NEVER constructs an env, NEVER calls the
scanner, NEVER steps physics and NEVER runs the network — every tensor it
draws comes from the snapshot dict handed to ``PlayViewer.render`` (the play
session builds one snapshot per completed control step; repeated rendering of
the same snapshot re-computes nothing on the runtime side). The ONLY model
evaluation is a host-side ``mujoco.MjData.mj_forward`` on the live compiled
template model at the SNAPSHOT's qpos — kinematics for placing the real robot
meshes in one coherent display frame; no dynamics, no GPU integration, no
extra physics steps.

Two clearly separated parts:

* PURE provenance (:func:`resolve_point_sources`) — maps each prepared
  near/far network-input point back to its ORIGINAL raw ray id / no-hit flag
  / source scan WITHOUT guessing by XYZ equality. The P6 range-noise 0 m and
  15 m saturations make many points share identical coordinates, so equality
  matching is ambiguous BY CONSTRUCTION; instead the mapping reproduces the
  EXACT stable angle-key sort permutation that ``prepare_control_inputs``
  applied (window-carried direction keys when present, the same geometric
  key path for key-free clean rows) and reads the sampled raw ids from the
  runtime's bounded play source-scan metadata. Degenerate duplicate points
  still resolve to distinct, unique sources. A permutation mismatch is a
  hard error — the mapping never falls back to coordinate guessing.
* :class:`PlayViewer` — the live local Viser viewer (127.0.0.1 ONLY; no
  public tunnel, no upload). GUI callbacks only SUBMIT control intents
  through :class:`PendingControls`; the single owner loop (the CLI driver)
  takes them and is the only thread that touches the runtime/env/network.
  Layer toggles / env switching / picking request a RE-RENDER of the SAME
  cached snapshot — they never advance physics, scans or noise, and never
  re-run the network. Every layer is a NAMED scene object swapped/dropped
  through the same discipline as the accepted M3 replay viewer (no residue,
  view transforms never touch the online data).

Displayed per snapshot (all honest pairings labelled):

* the robot's real compiled-model meshes at the snapshot qpos (the post-step
  boundary pose t_(k+1) — or the current reset state before any step);
* the CURRENT near 256 points and the far-window 640 valid history points of
  the o_k inputs the LAST action consumed (pre-action snapshot, acquisition
  BASE coordinates, NO motion compensation, no-hit endpoints distinct);
* the actual sampling candidates of the retained current raw frame with the
  angular group masks (optional layer);
* the grouping cone (θ/δ of the live pipeline cfg) at the corrected mount;
* the predicted hexagon Ê and the separate root-height TARGET marker
  0.46+Δz (world-constructed, ground-referenced — never a prism/ceiling
  claim);
* ground_pred on the REAL grid xy (signed body-z heights) and the real
  ground TRUTH label as an independent diagnostic layer (display-only: the
  truth/prediction never feed the online network);
* the picker reports env/episode/source scan/time/raw ray id/no-hit/xyz.

Status panels show the command, actual world-vertical root height, scan
age/window fill, control/env/episode/scan identities, noise preset,
running/paused/fallen/faulted state and the checkpoint identity.
"""

from __future__ import annotations

import json
import ipaddress
import threading
from pathlib import Path

import numpy as np
import torch

from hexapod_mjlab.rl.confined_locomotion.contracts import (
    FAR_POINTS_PER_SCAN,
    FAR_WINDOW_SLOTS,
    NEAR_POINTS,
    flatten_far_window,
)
from hexapod_mjlab.rl.confined_locomotion.history import WindowAngleKeys
from hexapod_mjlab.rl.confined_locomotion.integration import PreparedControlInputs

from hexapod_mjlab.common.visualization import (
    H_REF_M,
    PendingControls,
    cone_axis_base,
    cone_wireframe,
    hex_vertices,
    quat_wxyz_to_matrix,
    robot_meshes_at_qpos,
)

from .geometry import (
    STARTUP_ENVELOPE,
    clip_segments_base_x,
    root_height_marker_world,
    world_to_base,
)

__all__ = [
    "FAR_INPUT_POINTS",
    "NEAR_INPUT_POINTS",
    "PLAY_SNAPSHOT_SCHEMA",
    "PendingControls",
    "PlayViewer",
    "export_snapshot_manifest",
    "resolve_point_sources",
    "robot_meshes_at_qpos",
]

#: Input point counts of the network consumption contract.
NEAR_INPUT_POINTS = NEAR_POINTS            # 256
FAR_INPUT_POINTS = FAR_WINDOW_SLOTS * FAR_POINTS_PER_SCAN  # 640

#: Schema stamp of the display snapshot dict this module renders.
PLAY_SNAPSHOT_SCHEMA = "confined_locomotion.play_snapshot/v2"


def preview_rectangle_geometry(
    near_m: float, far_m: float, half_width_m: float, *,
    z_base_m: float = 0.0, nx: int = 17, ny: int = 11,
) -> tuple[np.ndarray, np.ndarray]:
    """Read-only candidate BASE-xy box and sample locations, never labels.

    The selected bounds have no effect on the policy's checkpoint grid,
    observations, queries or ground-head semantics. The common z is only a
    display reference plane, NOT a terrain-surface estimate.
    """
    values = (near_m, far_m, half_width_m, z_base_m)
    if not all(np.isfinite(v) for v in values):
        raise ValueError("candidate rectangle bounds must be finite")
    if not 0 <= near_m < far_m or half_width_m <= 0:
        raise ValueError("require 0 ≤ nearest < farthest and half-width > 0")
    if nx < 2 or ny < 2:
        raise ValueError("candidate preview must have at least two rows and columns")
    corners = np.array([
        [near_m, -half_width_m, z_base_m],
        [far_m, -half_width_m, z_base_m],
        [far_m, half_width_m, z_base_m],
        [near_m, half_width_m, z_base_m],
    ], dtype=np.float64)
    segments = np.stack([corners, np.roll(corners, -1, axis=0)], axis=1)
    x = np.linspace(near_m, far_m, nx, dtype=np.float64)
    y = np.linspace(-half_width_m, half_width_m, ny, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)  # y outer, x inner, same count/order as the true grid
    pts = np.stack((xx.ravel(), yy.ravel(), np.full(nx * ny, z_base_m)), axis=1)
    return segments, pts


# Selected near/far source point: physical 3-D sphere instead of Viser's
# default square point-cloud sprite. Diameter=0.24 m, larger than normal
# near/far point sizes (0.06/0.045 m). Display-only: online data unchanged.
PICKER_HIGHLIGHT_SPHERE_RADIUS_M = 0.12


# ═════════════════════════ exact provenance mapping (pure) ═════════════════


def _sort_permutation(valid: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """The EXACT stable ascending permutation ``sort_points_by_angle_keys``
    applies: invalid slots go last (inf key, stable). Reproduced here so the
    prepared slot order can be inverted WITHOUT re-sorting any online data.
    """
    effective = torch.where(valid, keys, torch.full_like(keys, float("inf")))
    return torch.argsort(effective, dim=-1, stable=True)


def resolve_point_sources(
    *,
    snapshot,
    angle_keys: WindowAngleKeys | None,
    geometric_keys_fn,
    prepared: PreparedControlInputs,
    capture_entries: list,
    env_row: int,
    episode_id: int,
) -> dict:
    """Map the prepared o_k near/far points to their ORIGINAL raw ray sources.

    Inputs (CPU or device tensors; only read, never mutated):

    * ``snapshot`` — the ``ScanWindowSnapshot`` the prepared inputs were
      built from (retained by the play session at the input identity);
    * ``angle_keys`` — ``window.read_angle_keys()`` AT THAT SNAPSHOT, or
      None on the clean key-free path;
    * ``geometric_keys_fn`` — ``pipeline.angle_keys``-compatible callable
      (required iff the row lacks window keys — the same geometric fallback
      ``TrainingRuntime.prepare`` uses);
    * ``prepared`` — the actual prepared network inputs (o_k);
    * ``capture_entries`` — this env's bounded play source-scan metadata
      (runtime ``play_source_scans[env]``).

    Returns CPU tensors (slot-aligned with ``prepared``): raw ray ids (−1 =
    invalid/unresolved), ORIGINAL no-hit flags (from the frame's raw
    ``no_hit_mask`` at the sampled raw ids — never inferred from consumed
    distances), the far points' source scan ids, and match counts. The
    permutation identity is VERIFIED bit-exactly against the prepared
    points; any mismatch raises instead of guessing.
    """
    matches = (snapshot.env_ids == int(env_row)).nonzero(as_tuple=False).flatten()
    if int(matches.numel()) != 1:
        raise RuntimeError(f"env {env_row} not found exactly once in the window snapshot")
    wrow = int(matches[0])
    prow_matches = (
        prepared.observation.env_ids == int(env_row)
    ).nonzero(as_tuple=False).flatten()
    if int(prow_matches.numel()) != 1:
        raise RuntimeError(
            f"env {env_row} not found exactly once in the prepared inputs"
        )
    prow = int(prow_matches[0])
    has_keys = angle_keys is not None and bool(angle_keys.present[wrow])
    if not has_keys and geometric_keys_fn is None:
        raise ValueError(
            "key-free clean rows require the geometric-keys callable (the "
            "same fallback TrainingRuntime.prepare uses)"
        )

    entries: dict[tuple[int, int], dict] = {}
    for entry in capture_entries:
        entries[(int(entry["episode"]), int(entry["scan"]))] = entry

    # All row math happens on CPU (small rows; the caller may hold CUDA
    # tensors). Keys are computed on the ORIGINAL device first — exactly the
    # way TrainingRuntime.prepare computed them — then moved.
    near_keys_dev = (
        angle_keys.near[wrow]
        if has_keys
        else geometric_keys_fn(snapshot.near_points[wrow : wrow + 1])[0]
    )
    near_valid = snapshot.near_valid[wrow].detach().cpu()
    near_pts = snapshot.near_points[wrow].detach().cpu()
    near_order = _sort_permutation(near_valid, near_keys_dev.detach().cpu())
    if not torch.equal(near_pts.index_select(0, near_order),
                       prepared.near_points[prow].detach().cpu()):
        raise RuntimeError(
            "near permutation check failed: the reproduced angle-key order "
            "does not reproduce the prepared points bit-exactly (refusing "
            "to guess any raw ray id)"
        )
    near_valid_sorted = near_valid.index_select(0, near_order)
    near_raw = torch.full((NEAR_INPUT_POINTS,), -1, dtype=torch.long)
    near_no_hit = torch.zeros(NEAR_INPUT_POINTS, dtype=torch.bool)
    near_entry = entries.get((int(episode_id), int(snapshot.near_scan_ids[wrow])))
    if near_entry is not None:
        ids = near_entry["near_indices"].index_select(0, near_order)
        nh = near_entry["near_no_hit"].index_select(0, near_order)
        near_raw = torch.where(near_valid_sorted, ids, near_raw)
        near_no_hit = torch.where(near_valid_sorted, nh, near_no_hit)

    # ── far: flatten (oldest→newest), sort, then per-source-scan lookup ─────
    flat = flatten_far_window(snapshot)
    far_keys_dev = (
        angle_keys.far.reshape(int(snapshot.env_ids.shape[0]), -1)[wrow]
        if has_keys
        else geometric_keys_fn(snapshot.far_points[wrow].reshape(1, -1, 3))[0]
    )
    far_valid = flat.valid[wrow].detach().cpu()
    far_pts = flat.points[wrow].detach().cpu()
    far_keys = far_keys_dev.detach().cpu()
    far_order = _sort_permutation(far_valid, far_keys)
    if not torch.equal(far_pts.index_select(0, far_order),
                       prepared.far_points[prow].detach().cpu()):
        raise RuntimeError(
            "far permutation check failed: the reproduced angle-key order "
            "does not reproduce the prepared points bit-exactly (refusing "
            "to guess any raw ray id)"
        )
    far_valid_sorted = far_valid.index_select(0, far_order)
    scan_ids_sorted = flat.scan_ids[wrow].detach().cpu().index_select(0, far_order)
    slot = torch.div(far_order, FAR_POINTS_PER_SCAN, rounding_mode="floor")
    pos_in_scan = far_order % FAR_POINTS_PER_SCAN
    far_raw = torch.full((FAR_INPUT_POINTS,), -1, dtype=torch.long)
    far_no_hit = torch.zeros(FAR_INPUT_POINTS, dtype=torch.bool)
    resolved_scan = torch.full((FAR_INPUT_POINTS,), -1, dtype=torch.long)
    source_resolved = torch.zeros(FAR_INPUT_POINTS, dtype=torch.bool)
    unique_scans, inverse = torch.unique(scan_ids_sorted, return_inverse=True)
    for index, scan_value in enumerate(unique_scans.tolist()):
        entry = entries.get((int(episode_id), int(scan_value)))
        if entry is None:
            continue
        selected = inverse == index
        far_raw[selected] = entry["far_indices"].index_select(0, pos_in_scan[selected])
        far_no_hit[selected] = entry["far_no_hit"].index_select(0, pos_in_scan[selected])
        resolved_scan[selected] = int(scan_value)
        source_resolved[selected] = True
    unresolved = torch.full_like(far_raw, -1)
    far_raw = torch.where(far_valid_sorted & source_resolved, far_raw, unresolved)
    far_no_hit = far_no_hit & far_valid_sorted & source_resolved
    resolved_scan = torch.where(
        far_valid_sorted & source_resolved, resolved_scan, unresolved
    )
    near_valid_count = int(near_valid_sorted.sum())
    far_valid_count = int(far_valid_sorted.sum())
    return {
        "near_raw_ids": near_raw,
        "near_no_hit": near_no_hit,
        "near_valid": near_valid_sorted,
        "near_matched": int((near_raw[near_valid_sorted] >= 0).sum()),
        "near_valid_count": near_valid_count,
        "far_raw_ids": far_raw,
        "far_no_hit": far_no_hit,
        "far_valid": far_valid_sorted,
        "far_source_scan_ids": scan_ids_sorted,
        "far_source_slot": slot,
        "far_resolved_scan_ids": resolved_scan,
        "far_matched": int((far_raw[far_valid_sorted] >= 0).sum()),
        "far_valid_count": far_valid_count,
        "capture_entries_available": len(capture_entries),
        "method": "exact stable angle-key permutation + sampled raw-id "
                  "metadata (XYZ equality is never used)",
    }


# ═════════════════════════ the live viewer ═════════════════════════════════


class PlayViewer:
    """Local live Viser viewer for task play (127.0.0.1 only).

    The owner loop calls :meth:`render` with one display snapshot after each
    completed control step (or to re-render the SAME snapshot after pure-GUI
    changes). GUI callbacks only push intents into :attr:`pending`; nothing
    here advances physics, scans, noise or the network. No public tunnel, no
    upload; viser renders client-side so screenshots stay with the operator.
    """

    def __init__(
        self,
        *,
        num_envs: int,
        mj_model,
        pipeline,
        ground_grid,
        identity_header: dict,
        host: str = "127.0.0.1",
        port: int = 8093,
        cone_display_length_m: float = 3.0,
        initial_env: int = 0,
    ) -> None:
        import viser

        if host != "127.0.0.1" or not ipaddress.ip_address(host).is_loopback:
            raise ValueError("the task-play viewer must bind exactly to 127.0.0.1")
        if not 0 <= int(initial_env) < int(num_envs):
            raise ValueError(f"initial_env out of range 0..{num_envs - 1}")
        if not 1 <= int(port) <= 65535:
            raise ValueError(f"viewer port must be in 1..65535, got {port!r}")
        self.num_envs = int(num_envs)
        self.mj_model = mj_model
        self.pipeline = pipeline
        self.ground_grid = ground_grid  # matched to checkpoint before viewer creation
        self._reference_rectangle = (
            float(ground_grid.x_coords[0]), float(ground_grid.x_coords[-1]),
            float(ground_grid.y_coords[0]), float(ground_grid.y_coords[-1]),
        )
        self.identity_header = dict(identity_header)
        self.host = host
        self.port = int(port)
        self.pending = PendingControls()
        self._mj_data = None
        self._camera_focused = False
        self._lock = threading.Lock()
        self._handles: dict = {}
        self._last_pick: dict | None = None
        max_range = float(pipeline.cfg.max_range_m)
        self._cone_display_length_m = float(
            min(max(cone_display_length_m, 0.5), max_range)
        )
        self.server = viser.ViserServer(
            host=host, port=port, label="confined_locomotion 任务play查看器"
        )
        try:
            self._setup_gui(initial_env)
        except BaseException:
            self.close()
            raise

    def _setup_gui(self, initial_env: int) -> None:
        g = self.server.gui
        env_choices = [f"env {i}" for i in range(self.num_envs)]
        self._gui_env = g.add_dropdown(
            "环境", env_choices, initial_value=env_choices[int(initial_env)]
        )
        self._gui_view = g.add_dropdown(
            "视图", ["BASE（网络输入系）", "WORLD（当前位姿）"],
            initial_value="BASE（网络输入系）",
        )
        self._gui_near = g.add_checkbox("近端 256 点（o_k 输入）", True)
        self._gui_far = g.add_checkbox("远端历史 640 点（o_k 输入）", True)
        self._gui_pool = g.add_checkbox("采样候选点（当前帧角组掩码）", False)
        theta = float(self.pipeline.cfg.split_theta_deg)
        tilt = float(self.pipeline.cfg.split_axis_tilt_deg)
        cutoff = self.pipeline.cfg.split_max_forward_x_m
        cutoff_text = "" if cutoff is None else f"，BASE x≤{cutoff:g}m"
        self._gui_cone = g.add_checkbox(
            f"近端分组 θ{theta:g}°/δ{tilt:g}°（半角{90 - theta:g}°{cutoff_text}）", True
        )
        self._gui_cone_len = g.add_slider(
            "锥显示长度 (m)", 0.5, float(self.pipeline.cfg.max_range_m), step=0.1,
            initial_value=self._cone_display_length_m,
        )
        self._gui_env_hex = g.add_checkbox("包络六边形 + 根高目标", True)
        self._gui_ground_pred = g.add_checkbox("地面预测（真实grid xy）", True)
        self._gui_ground_truth = g.add_checkbox("地面真值（诊断图层）", False)
        x_min, x_max, y_min, y_max = self._reference_rectangle
        with g.add_folder("候选地面监督范围（只显示，不修改策略）"):
            self._gui_candidate_show = g.add_checkbox("显示候选矩形与真实范围边框", False)
            self._gui_candidate_near = g.add_number(
                "最近端 x (m)", x_min, min=0.0, max=float(self.pipeline.cfg.max_range_m),
                step=0.05,
            )
            self._gui_candidate_far = g.add_number(
                "最远端 x (m)", x_max, min=0.0, max=float(self.pipeline.cfg.max_range_m),
                step=0.05,
            )
            self._gui_candidate_halfwidth = g.add_number(
                "左右半宽 ±y (m)", max(abs(y_min), abs(y_max)),
                min=0.01, max=float(self.pipeline.cfg.max_range_m), step=0.05,
            )
            self._gui_candidate_points = g.add_checkbox(
                "显示候选采样位置（仅XY示意）", False
            )
            self._gui_candidate_reset = g.add_button("重置候选范围为Checkpoint原范围")
            self._gui_candidate_status = g.add_markdown(
                "候选蓝色线框默认隐藏；绿色边框是Checkpoint真实网格范围。"
                "调节不修改地面标签、网络或Checkpoint。"
            )
        with g.add_folder("手动控制（提交意图，主循环执行）"):
            self._gui_cmd_vx = g.add_slider(
                "指令 vx (m/s)", -1.5, 1.5, step=0.05, initial_value=0.0
            )
            self._gui_cmd_vy = g.add_slider(
                "指令 vy (m/s)", -1.0, 1.0, step=0.05, initial_value=0.0
            )
            self._gui_cmd_w = g.add_slider(
                "指令 ωz (rad/s)", -1.0, 1.0, step=0.05, initial_value=0.0
            )
            self._gui_cmd_apply = g.add_button("应用指令")
            self._gui_cmd_zero = g.add_button("指令清零")
            self._gui_pause = g.add_button("暂停 / 继续（由主循环同步状态）")
            self._gui_step = g.add_button("单步")
            self._gui_reset = g.add_button("重置（整episode手动reset）")
            self._gui_exit = g.add_button("退出")
        with g.add_folder("点选查询（原始射线溯源）"):
            self._gui_pick_kind = g.add_dropdown(
                "点选类别", ["近端索引 (0..255)", "远端输入索引 (0..639)"],
                initial_value="近端索引 (0..255)",
            )
            self._gui_pick_near = g.add_number(
                "近端索引", 0, min=0, max=NEAR_INPUT_POINTS - 1, step=1
            )
            self._gui_pick_far = g.add_number(
                "远端输入索引", 0, min=0, max=FAR_INPUT_POINTS - 1, step=1
            )
            self._gui_pick_btn = g.add_button("查询选中点")
        self._gui_header = g.add_markdown("")
        self._gui_status = g.add_markdown("")
        self._gui_pick = g.add_markdown("")

        # GUI → intents only (no runtime access from these threads).
        self._gui_cmd_apply.on_click(
            lambda _e: self.pending.submit_command(
                float(self._gui_cmd_vx.value),
                float(self._gui_cmd_vy.value),
                float(self._gui_cmd_w.value),
            )
        )
        self._gui_cmd_zero.on_click(lambda _e: self.pending.submit_zero_command())
        self._gui_pause.on_click(
            lambda _e: self.pending.submit_pause_toggle()
        )
        self._gui_step.on_click(lambda _e: self.pending.submit_step())
        self._gui_reset.on_click(lambda _e: self.pending.submit_reset())
        self._gui_exit.on_click(lambda _e: self.pending.submit_exit())
        self._gui_pick_btn.on_click(
            lambda _e: self.pending.submit_pick(
                "far" if str(self._gui_pick_kind.value).startswith("远端") else "near",
                int(
                    self._gui_pick_far.value
                    if str(self._gui_pick_kind.value).startswith("远端")
                    else self._gui_pick_near.value
                ),
            )
        )
        self._gui_env.on_update(
            lambda _e: self.pending.submit_env_row(self.selected_env_row)
        )

        def _reset_candidate(_event) -> None:
            # Only GUI display state is changed from this callback.
            self._gui_candidate_near.value = x_min
            self._gui_candidate_far.value = x_max
            self._gui_candidate_halfwidth.value = max(abs(y_min), abs(y_max))
            self.pending.submit_rerender()

        self._gui_candidate_reset.on_click(_reset_candidate)
        for handle in (
            self._gui_view,
            self._gui_near,
            self._gui_far,
            self._gui_pool,
            self._gui_cone,
            self._gui_cone_len,
            self._gui_env_hex,
            self._gui_ground_pred,
            self._gui_ground_truth,
            self._gui_candidate_show,
            self._gui_candidate_near,
            self._gui_candidate_far,
            self._gui_candidate_halfwidth,
            self._gui_candidate_points,
        ):
            handle.on_update(lambda _e: self.pending.submit_rerender())

    # ── named-layer scene management (toggles leave NO residue) ─────────────

    def _swap(self, name: str, factory) -> None:
        self._drop(name)
        self._handles[name] = factory()

    def _drop(self, name: str) -> None:
        handle = self._handles.pop(name, None)
        if handle is not None:
            try:
                handle.remove()
            except Exception:
                pass  # best-effort removal of a vanished server node

    # ── helpers ──────────────────────────────────────────────────────────────

    @property
    def selected_env_row(self) -> int:
        return int(str(self._gui_env.value).split()[-1])

    @property
    def view_mode(self) -> str:
        return "world" if "WORLD" in str(self._gui_view.value) else "base"

    def _display_points(self, pts_base: np.ndarray, snapshot: dict) -> np.ndarray:
        if self.view_mode == "world":
            r = quat_wxyz_to_matrix(snapshot["pose"]["root_quat_wxyz"])
            return pts_base @ r.T + snapshot["pose"]["root_pos_w"]
        return pts_base

    def _display_dir(self, v_base: np.ndarray, snapshot: dict) -> np.ndarray:
        if self.view_mode == "world":
            return quat_wxyz_to_matrix(snapshot["pose"]["root_quat_wxyz"]) @ v_base
        return v_base

    def _render_pick_highlight(self, scene, snapshot: dict) -> None:
        """Show a larger magenta 3-D sphere at the selected source point.

        Convert the selected BASE coordinate into the current display frame
        just like the normal point-cloud layer. Rerenders replace the prior
        marker; invalid/cleared selections remove it without residue.
        """
        pick = self._last_pick
        if pick is None or not pick.get("valid"):
            self._drop("/picker/highlight")
            return
        point = self._display_points(
            np.asarray(pick["xyz_base"], dtype=np.float64).reshape(1, 3),
            snapshot,
        )[0]
        self._swap(
            "/picker/highlight",
            lambda: scene.add_icosphere(
                "/picker/highlight",
                radius=PICKER_HIGHLIGHT_SPHERE_RADIUS_M,
                color=(255, 0, 255),
                subdivisions=3,
                position=np.asarray(point, dtype=np.float32),
            ),
        )

    def _render_ground_range_preview(self, scene, snapshot: dict) -> None:
        """Overlay candidate XY-only rectangle and immutable actual grid bounds.

        All geometry is derived from GUI numbers and the retained display
        snapshot; only Viser objects are touched. No policy/label/grid writes.
        """
        layers = (
            "/ground_preview/candidate_box",
            "/ground_preview/true_bounds",
            "/ground_preview/candidate_points",
        )
        if not self._gui_candidate_show.value:
            for name in layers:
                self._drop(name)
            self._gui_candidate_status.content = (
                "候选范围隐藏。开启后蓝色为候选XY矩形、绿色为真实监督范围；"
                "两者均为参考平面投影，不表示实际可观测性。"
            )
            return

        try:
            near = float(self._gui_candidate_near.value)
            far = float(self._gui_candidate_far.value)
            halfwidth = float(self._gui_candidate_halfwidth.value)
            labels = snapshot.get("labels") or {}
            heights = np.asarray(labels.get("ground_heights", []), dtype=np.float64)
            valid = np.asarray(labels.get("ground_valid", []), dtype=bool)
            # Use a single visual reference height from this scan's actual
            # labels. This does not query ground at candidate points.
            z_ref = (
                float(np.median(heights[valid]))
                if heights.size and valid.shape == heights.shape and bool(valid.any())
                else 0.0
            )
            if not np.isfinite(z_ref):
                z_ref = 0.0
            nx, ny = len(self.ground_grid.x_coords), len(self.ground_grid.y_coords)
            candidate_lines, candidate_pts = preview_rectangle_geometry(
                near, far, halfwidth, z_base_m=z_ref + 0.015,
                nx=nx, ny=ny,
            )
        except (TypeError, ValueError) as err:
            for name in layers:
                self._drop(name)
            self._gui_candidate_status.content = f"无效候选范围：{err}"
            return

        actual_near, actual_far, actual_ymin, actual_ymax = self._reference_rectangle
        actual_corners = np.asarray([
            [actual_near, actual_ymin, z_ref],
            [actual_far, actual_ymin, z_ref],
            [actual_far, actual_ymax, z_ref],
            [actual_near, actual_ymax, z_ref],
        ], dtype=np.float64)
        actual_lines = np.stack(
            (actual_corners, np.roll(actual_corners, -1, axis=0)), axis=1
        )
        self._swap(
            "/ground_preview/true_bounds",
            lambda: scene.add_line_segments(
                "/ground_preview/true_bounds",
                self._display_points(actual_lines.reshape(-1, 3), snapshot).reshape(-1, 2, 3),
                colors=(35, 190, 100), thickness=2.0, thickness_units="screen",
            ),
        )
        self._swap(
            "/ground_preview/candidate_box",
            lambda: scene.add_line_segments(
                "/ground_preview/candidate_box",
                self._display_points(candidate_lines.reshape(-1, 3), snapshot).reshape(-1, 2, 3),
                colors=(35, 210, 255), thickness=3.0, thickness_units="screen",
            ),
        )
        if self._gui_candidate_points.value:
            self._swap(
                "/ground_preview/candidate_points",
                lambda: scene.add_point_cloud(
                    "/ground_preview/candidate_points",
                    self._display_points(candidate_pts, snapshot),
                    colors=(35, 210, 255),
                    point_size=0.025, point_shape="circle",
                    precision="float32",
                ),
            )
        else:
            self._drop("/ground_preview/candidate_points")
        self._gui_candidate_status.content = (
            f"候选BASE x=[{near:.2f}, {far:.2f}]m、"
            f"y=±{halfwidth:.2f}m；示意点数={nx * ny}。"
            "仅几何参考平面投影，不是新地面标签或预测。"
        )

    def _hex_segments(
        self, hexv: dict, snapshot: dict, z_base: float, colors,
        name: str = "/envelope/hex",
    ) -> None:
        pts = np.asarray(
            [(*hexv[label], z_base) for label in ("B", "D", "F", "E", "C", "A")],
            dtype=np.float64,
        )
        pts = np.vstack([pts, pts[0]])
        arr = self._display_points(pts, snapshot)
        segs = np.stack([arr[:-1], arr[1:]], axis=1)
        self._swap(
            name,
            lambda: self.server.scene.add_line_segments(
                name, segs, colors, thickness=2.0, thickness_units="screen"
            ),
        )

    # ── the render (reads ONLY the snapshot; no runtime calls) ──────────────

    def render(self, snapshot: dict) -> None:
        """Draw one display snapshot (idempotent w.r.t. the online state)."""
        with self._lock:
            if snapshot.get("schema") != PLAY_SNAPSHOT_SCHEMA:
                raise ValueError(
                    f"unknown snapshot schema {snapshot.get('schema')!r}"
                )
            scene = self.server.scene
            pose = snapshot["pose"]

            # Robot: real compiled-model meshes at the snapshot qpos.
            if self._mj_data is None:
                import mujoco

                self._mj_data = mujoco.MjData(self.mj_model)
            meshes = robot_meshes_at_qpos(
                self.mj_model, pose["qpos"], mj_data=self._mj_data
            )
            if self.view_mode == "base":
                r = quat_wxyz_to_matrix(pose["root_quat_wxyz"])
                t = pose["root_pos_w"]
                for msh in meshes:
                    msh["vertices"] = (msh["vertices"] - t) @ r
            for stale in [n for n in list(self._handles) if n.startswith("/robot/")]:
                self._drop(stale)
            for msh in meshes:
                name = f"/robot/{msh['body']}/{msh['geom']}"
                self._swap(
                    name,
                    (
                        lambda msh=msh, name=name: scene.add_mesh_simple(
                            name,
                            np.asarray(msh["vertices"], dtype=np.float32),
                            np.asarray(msh["faces"], dtype=np.int32),
                            color=(168, 175, 185),
                            side="double",
                        )
                    ),
                )

            floor = np.asarray(
                self.pipeline.mesh_record["vertices"], dtype=np.float64
            )
            if self.view_mode == "base":
                r = quat_wxyz_to_matrix(pose["root_quat_wxyz"])
                floor = (floor - pose["root_pos_w"]) @ r
            floor_f32 = np.asarray(floor, dtype=np.float32)
            self._swap(
                "/floor",
                lambda: scene.add_mesh_simple(
                    "/floor",
                    floor_f32,
                    np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
                    color=(120, 130, 120),
                    side="double",
                ),
            )

            near = snapshot["input"]["near"]
            far = snapshot["input"]["far"]
            prov = snapshot["provenance"]
            near_pts = near["points_base"].astype(np.float64)
            near_valid = near["valid"]
            near_no_hit = prov["near_no_hit"].astype(bool)
            hit_sel = near_valid.copy()
            hit_sel[near_valid] = ~near_no_hit[near_valid]
            noh_sel = near_valid.copy()
            noh_sel[near_valid] = near_no_hit[near_valid]
            if self._gui_near.value and bool(hit_sel.any()):
                self._swap(
                    "/points/near",
                    lambda: scene.add_point_cloud(
                        "/points/near",
                        self._display_points(near_pts[hit_sel], snapshot),
                        colors=(0, 170, 255),
                        point_size=0.06,
                        precision="float32",
                    ),
                )
            else:
                self._drop("/points/near")
            if self._gui_near.value and bool(noh_sel.any()):
                self._swap(
                    "/points/near_nohit",
                    lambda: scene.add_point_cloud(
                        "/points/near_nohit",
                        self._display_points(near_pts[noh_sel], snapshot),
                        colors=(255, 140, 0),
                        point_size=0.06,
                        precision="float32",
                    ),
                )
            else:
                self._drop("/points/near_nohit")

            far_pts = far["points_base"].astype(np.float64)
            far_valid = far["valid"]
            if self._gui_far.value and bool(far_valid.any()):
                age = far["age_s"]
                finite = np.where(np.isfinite(age), age, 0.9)
                t01 = np.clip(finite / 0.9, 0.0, 1.0)
                colors = np.stack(
                    [60 + t01 * 195, 120 + t01 * 100, 255 - t01 * 255], axis=1
                )
                noh = prov["far_no_hit"].astype(bool)
                colors_valid = np.zeros((int(far_valid.sum()), 3))
                colors_valid[~noh[far_valid]] = colors[far_valid][~noh[far_valid]]
                colors_valid[noh[far_valid]] = (255, 60, 60)
                self._swap(
                    "/points/far",
                    lambda: scene.add_point_cloud(
                        "/points/far",
                        self._display_points(far_pts[far_valid], snapshot),
                        colors_valid.astype(np.uint8),
                        point_size=0.045,
                        precision="float32",
                    ),
                )
            else:
                self._drop("/points/far")

            pool = snapshot.get("pool")
            for layer, mask, color in (
                (
                    "/pool/near_group",
                    None if pool is None else pool["near_mask"],
                    (0, 230, 200),
                ),
                (
                    "/pool/far_group",
                    None if pool is None else pool["far_mask"],
                    (190, 120, 255),
                ),
            ):
                if (
                    self._gui_pool.value
                    and pool is not None
                    and bool(np.asarray(mask).any())
                ):
                    ppts = pool["points_base"].astype(np.float64)
                    sel = np.asarray(mask, dtype=bool)

                    def _factory(layer=layer, ppts=ppts, sel=sel, color=color):
                        return scene.add_point_cloud(
                            layer,
                            self._display_points(ppts[sel], snapshot),
                            colors=color,
                            point_size=0.02,
                            precision="float32",
                        )

                    self._swap(layer, _factory)
                else:
                    self._drop(layer)

            if self._gui_cone.value:
                origin_base = np.asarray(
                    self.pipeline.mount_pos, dtype=np.float64
                ).reshape(3)
                axis_base = cone_axis_base(
                    np.asarray(self.pipeline.mount_quat_xyzw, dtype=np.float64),
                    float(self.pipeline.cfg.split_axis_tilt_deg),
                )
                length = float(
                    min(
                        max(float(self._gui_cone_len.value), 0.1),
                        float(self.pipeline.cfg.max_range_m),
                    )
                )
                segments_base = cone_wireframe(
                    origin_base, axis_base,
                    90.0 - float(self.pipeline.cfg.split_theta_deg), length,
                )
                segments_base = clip_segments_base_x(
                    segments_base,
                    x_max_m=self.pipeline.cfg.split_max_forward_x_m,
                )
                segs = self._display_points(
                    segments_base.reshape(-1, 3), snapshot
                ).reshape(-1, 2, 3)
                if len(segs):
                    self._swap(
                        "/cone",
                        lambda: scene.add_line_segments(
                            "/cone", segs, colors=(255, 210, 0), thickness=2.0,
                            thickness_units="screen",
                        ),
                    )
                else:
                    self._drop("/cone")
            else:
                self._drop("/cone")

            network = snapshot.get("network")
            labels = snapshot.get("labels") or {}
            if self._gui_env_hex.value and network is not None:
                envelope = np.asarray(network["envelope"], dtype=np.float64)
                self._hex_segments(
                    hex_vertices(envelope), snapshot, 0.0, (30, 220, 30)
                )
                self._hex_segments(
                    hex_vertices(STARTUP_ENVELOPE), snapshot, 0.0, (60, 120, 60)
                )
                ground_z = float(
                    np.asarray(
                        self.pipeline.mesh_record["z"], dtype=np.float64
                    ).reshape(-1)[0]
                )
                marker_world = root_height_marker_world(
                    pose["root_pos_w"], ground_z, envelope, H_REF_M
                )
                s = 0.6
                corners_w = marker_world + np.array(
                    [[dx, dy, 0.0] for dx, dy in ((-s, -s), (s, -s), (s, s), (-s, s))]
                )
                if self.view_mode == "world":
                    pts = corners_w
                else:
                    pts = np.stack(
                        [
                            world_to_base(p, pose["root_pos_w"], pose["root_quat_wxyz"])
                            for p in corners_w
                        ]
                    )
                segs = np.stack(
                    [(pts[0], pts[1]), (pts[1], pts[2]), (pts[2], pts[3]), (pts[3], pts[0])]
                )
                self._swap(
                    "/envelope/root_height_target",
                    lambda: scene.add_line_segments(
                        "/envelope/root_height_target",
                        segs,
                        colors=(255, 80, 220),
                        thickness=2.0,
                        thickness_units="screen",
                    ),
                )
            else:
                for name in (
                    "/envelope/hex",
                    "/envelope/hex_startup",
                    "/envelope/root_height_target",
                ):
                    self._drop(name)

            grid_xy = labels.get("grid_xy_base")
            if (
                self._gui_ground_pred.value
                and network is not None
                and network.get("ground_pred") is not None
                and grid_xy is not None
            ):
                pts = np.column_stack(
                    [grid_xy, np.asarray(network["ground_pred"], dtype=np.float64)]
                )
                self._swap(
                    "/ground/pred",
                    lambda: scene.add_point_cloud(
                        "/ground/pred",
                        self._display_points(pts, snapshot),
                        colors=(0, 255, 120),
                        point_size=0.05,
                        precision="float32",
                    ),
                )
            else:
                self._drop("/ground/pred")

            if (
                self._gui_ground_truth.value
                and labels.get("ground_heights") is not None
                and grid_xy is not None
            ):
                valid = np.asarray(labels["ground_valid"], dtype=bool)
                if bool(valid.any()):
                    pts = np.column_stack(
                        [
                            np.asarray(grid_xy)[valid],
                            np.asarray(labels["ground_heights"], dtype=np.float64)[valid],
                        ]
                    )
                    self._swap(
                        "/ground/truth",
                        lambda: scene.add_point_cloud(
                            "/ground/truth",
                            self._display_points(pts, snapshot),
                            colors=(255, 215, 0),
                            point_size=0.05,
                            precision="float32",
                        ),
                    )
                else:
                    self._drop("/ground/truth")
            else:
                self._drop("/ground/truth")

            self._render_ground_range_preview(scene, snapshot)
            self._render_pick_highlight(scene, snapshot)

            if self.view_mode == "world":
                self._swap(
                    "/labels/history",
                    lambda: scene.add_label(
                        "/labels/history",
                        "远端历史未补偿（采集时机身坐标，仅施加当前位姿视图变换）",
                    ),
                )
            else:
                self._drop("/labels/history")

            self._update_panels(snapshot)
            self._focus_camera_once(snapshot)

    # ── panels / camera / picker ─────────────────────────────────────────────

    def set_pick_result(self, pick: dict | None) -> None:
        """Store the owner loop's picker result (pure display state)."""
        self._last_pick = pick

    def _focus_camera_once(self, snapshot: dict) -> None:
        if self._camera_focused:
            return
        self._camera_focused = True
        pose = snapshot["pose"]
        target = (
            np.asarray(pose["root_pos_w"], dtype=np.float64).copy()
            if self.view_mode == "world"
            else np.zeros(3)
        )
        target = target + np.array([0.0, 0.0, 0.1])
        position = target + np.array([2.4, -2.4, 1.3])
        try:
            cam = getattr(self.server, "initial_camera", None)
            if cam is not None:
                cam.position = position
                cam.look_at = target
                return

            def _set_client_camera(client):
                client.camera.position = position
                client.camera.look_at = target

            self.server.on_client_connect(_set_client_camera)
        except Exception as err:  # view-only convenience; never fatal
            print(f"[play-viewer] camera focus skipped ({err!r})")

    def _update_panels(self, snapshot: dict) -> None:
        identity = self.identity_header
        ident = snapshot["identity"]
        input_ident = {
            "episode_id": ident["input_episode_id"],
            "control_step": ident["input_control_step"],
            "step_in_episode": ident["input_step_in_episode"],
            "persistent_scan_count": ident["input_persistent_scan_count"],
        }
        pose_ident = {
            "episode_id": ident["pose_episode_id"],
            "control_step": ident["pose_control_step"],
            "step_in_episode": ident["pose_step_in_episode"],
            "persistent_scan_count": ident["pose_persistent_scan_count"],
        }
        metrics = snapshot["metrics"]
        state = snapshot["state"]
        network = snapshot.get("network")
        prov = snapshot["provenance"]

        def _fmt(vec, nd=4):
            return np.array2string(np.asarray(vec), precision=nd, separator=", ")

        state_labels = {
            "running": "运行中",
            "paused": "已暂停（时钟停止）",
            "fallen": "跌倒已保存并暂停（需手动重置）",
            "faulted": "故障已保存并停止（reset/继续均拒绝）",
        }
        height_target = 0.46 + (
            float(network["envelope"][5]) if network is not None else 0.0
        )
        ground_pred_note = (
            "无（本快照无前向输出）"
            if network is None or network.get("ground_pred") is None
            else f"[{int(np.asarray(network['ground_pred']).shape[0])}] "
            f"均值={float(np.mean(network['ground_pred'])):.4f} m"
        )
        self._gui_header.content = (
            f"**任务play（实时均值闭环）** · checkpoint `{identity.get('path')}`"
            f"（{identity.get('kind')}，M={identity.get('ground_samples')}，"
            f"grid `{str(identity.get('grid_identity_hash'))[:16]}…）`\n"
            f"噪声preset：`{identity.get('preset')}` · "
            f"权重加载：{'OK' if identity.get('weights_loaded') else '?'}"
            f"（仅权重，非续跑）"
        )
        pick = self._last_pick
        if pick is None:
            self._gui_pick.content = (
                "（未查询：选择点选类别与索引后点击“查询选中点”；来源按精确角键"
                "排序映射，不用XYZ相等猜测）"
            )
        else:
            self._gui_pick.content = (
                f"**选中点** 类型={pick['kind']} 索引={pick['index']} "
                f"有效={pick['valid']} · env={pick['env']} "
                f"episode={pick['episode_id']} 来源scan={pick['scan_id']} "
                f"扫描时刻={pick['scan_time_s']}s\n"
                f"原始射线ID={pick['raw_ray_id']} 无命中={pick['no_hit']} "
                f"xyz(BASE)={_fmt(pick['xyz_base'], 5)}"
            )
        self._gui_status.content = (
            f"**状态** {state_labels.get(state['kind'], state['kind'])} "
            f"{state.get('note', '')}\n\n"
            f"**身份** env={snapshot['env']} 输入o_k: 控制步="
            f"{input_ident['control_step']} episode={input_ident['episode_id']} "
            f"episode内步={input_ident['step_in_episode']} "
            f"扫描计数={input_ident['persistent_scan_count']}；"
            f"显示姿态: 控制步={pose_ident['control_step']} "
            f"episode={pose_ident['episode_id']} "
            f"episode内步={pose_ident['step_in_episode']} "
            f"扫描计数={pose_ident['persistent_scan_count']} "
            f"控制时间={ident['control_time_s']:.3f}s\n\n"
            f"**指令（o_k 配对）** {_fmt(snapshot['command'])} m/s, rad/s · "
            f"**当前手动指令** "
            f"{_fmt(snapshot.get('current_command', snapshot['command']))}"
            f"（随时可改，下一步生效）\n"
            f"**实际根竖直高度** {metrics['root_height_m']:.4f} m"
            f"（根高目标 0.46+Δz={height_target:.4f} m）\n\n"
            f"**扫描窗口** 近端scan={snapshot['input']['near']['scan_id']} @ "
            f"{snapshot['input']['near']['time_s']}s "
            f"年龄={metrics['scan_age_s']:.3f}s · "
            f"远端已填槽={metrics['far_window_fill']}/10\n\n"
            f"**姿态时刻** {snapshot['pose']['source']}\n"
            f"**输入/预测时刻** {snapshot['input']['source_label']}\n"
            + (
                f"**Ê** {_fmt(network['envelope'])} m · "
                f"ground_pred {ground_pred_note}"
                if network is not None
                else "**网络输出** 尚未执行任何控制步（无前向输出）"
            )
            + "\n\n"
            f"**点数** 近端有效={prov['near_valid_count']}"
            f"（溯源={prov['near_matched']}） "
            f"远端有效={prov['far_valid_count']}"
            f"（溯源={prov['far_matched']}，角键排序映射）\n"
            f"**碰撞（继续运行）** 非足环境={metrics['c_nonfoot_env']:.3f} "
            f"自碰撞={metrics['c_self']:.3f}（本控制步，仅显示）"
        )

    # ── lifecycle ────────────────────────────────────────────────────────────

    def close(self) -> None:
        stop = getattr(self.server, "stop", None)
        if callable(stop):
            try:
                stop()
            except Exception:
                pass


# ═════════════════════════ manifest export ═════════════════════════════════


def export_snapshot_manifest(snapshot: dict, viewer: PlayViewer | None, path) -> dict:
    """JSON-able audit manifest of one rendered snapshot (scene objects,
    point/grid/envelope counts, provenance stats and identities)."""
    prov = snapshot["provenance"]
    network = snapshot.get("network")
    labels = snapshot.get("labels") or {}
    manifest = {
        "schema": "confined_locomotion.play_snapshot_manifest/v1",
        "snapshot_schema": snapshot.get("schema"),
        "identity": dict(snapshot["identity"]),
        "state": dict(snapshot["state"]),
        "env": snapshot["env"],
        "view_mode": None if viewer is None else viewer.view_mode,
        "scene_objects": None if viewer is None else sorted(viewer._handles),
        "candidate_ground_rectangle": (
            None if viewer is None else {
                "enabled": bool(viewer._gui_candidate_show.value),
                "near_m": float(viewer._gui_candidate_near.value),
                "far_m": float(viewer._gui_candidate_far.value),
                "half_width_m": float(viewer._gui_candidate_halfwidth.value),
                "show_sample_positions": bool(viewer._gui_candidate_points.value),
                "actual_checkpoint_xy_bounds_m": list(viewer._reference_rectangle),
                "semantics": "display_only_BASE_XY_reference_plane_not_ground_labels",
            }
        ),
        "counts": {
            "near_valid": prov["near_valid_count"],
            "near_provenance_matched": prov["near_matched"],
            "far_valid": prov["far_valid_count"],
            "far_provenance_matched": prov["far_matched"],
            "far_window_fill": snapshot["metrics"]["far_window_fill"],
            "near_no_hit": int(
                np.asarray(prov["near_no_hit"])[
                    np.asarray(prov["near_valid"], dtype=bool)
                ].sum()
            ),
            "far_no_hit": int(
                np.asarray(prov["far_no_hit"])[
                    np.asarray(prov["far_valid"], dtype=bool)
                ].sum()
            ),
            "ground_grid_points": (
                None if labels.get("grid_xy_base") is None
                else int(np.asarray(labels["grid_xy_base"]).shape[0])
            ),
            "ground_truth_valid": (
                None if labels.get("ground_valid") is None
                else int(np.asarray(labels["ground_valid"], dtype=bool).sum())
            ),
            "envelope_dims": None if network is None else int(
                np.asarray(network["envelope"]).reshape(-1).shape[0]
            ),
            "pool_candidates": (
                0 if snapshot.get("pool") is None
                else int(np.asarray(snapshot["pool"]["points_base"]).shape[0])
            ),
        },
        "provenance_method": prov["method"],
        "capture_entries_available": prov["capture_entries_available"],
        "command": list(snapshot["command"]),
        "metrics": dict(snapshot["metrics"]),
        "checkpoint": dict(snapshot["checkpoint"]),
        "pose_source": snapshot["pose"]["source"],
        "input_source": snapshot["input"]["source_label"],
        "data_source": (
            "live play snapshot built by the single owner loop; rendering "
            "recomputes nothing on the runtime/network side"
        ),
    }
    if path is not None:
        Path(path).write_text(json.dumps(manifest, indent=2, default=str))
    return manifest
