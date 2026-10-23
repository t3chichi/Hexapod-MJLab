"""CPU contracts for shared display geometry and point-source provenance."""
from __future__ import annotations

import math
import traceback

import numpy as np
import torch

from hexapod_mjlab.rl.confined_locomotion.contracts import OnlineObservation
from hexapod_mjlab.rl.confined_locomotion.history import ScanWindow, WindowAngleKeys
from hexapod_mjlab.rl.confined_locomotion.integration import prepare_control_inputs
from hexapod_mjlab.tasks.confined_locomotion.visualization.geometry import (
    CONE_TILT_DEG,
    H_REF_M,
    HEX_VERTEX_ORDER,
    cone_axis_base,
    cone_wireframe,
    clip_segments_base_x,
    hex_vertices,
    normalize_mount,
    quat_wxyz_to_matrix,
    root_height_marker_world,
    root_height_target_m,
    rotation_matrix_to_wxyz,
    world_to_base,
)
from hexapod_mjlab.tasks.confined_locomotion.visualization.play import (
    PICKER_HIGHLIGHT_SPHERE_RADIUS_M,
    PlayViewer,
    preview_rectangle_geometry,
    resolve_point_sources,
)


def _observation(*, step: int = 1, time_s: float = 0.2) -> OnlineObservation:
    zeros = lambda shape: torch.zeros(shape, dtype=torch.float32)
    return OnlineObservation(
        env_ids=torch.tensor([0], dtype=torch.long),
        episode_ids=torch.tensor([0], dtype=torch.long),
        step_ids=torch.tensor([step], dtype=torch.long),
        time_s=torch.tensor([time_s], dtype=torch.float32),
        lin_vel_b=zeros((1, 3)),
        ang_vel_b=zeros((1, 3)),
        gravity_b=torch.tensor([[0.0, 0.0, -1.0]]),
        joint_pos_rel_ref=zeros((1, 18)),
        joint_vel=zeros((1, 18)),
        prev_action=zeros((1, 18)),
        command=zeros((1, 3)),
    )


def _source_fixture(*, keys_enabled: bool = True):
    window = ScanWindow(1, episode_ids=[0])
    near_points = torch.zeros((1, 256, 3))
    near_valid = torch.zeros((1, 256), dtype=torch.bool)
    near_valid[0, :3] = True
    near_points[0, :3] = torch.tensor([7.5, 0.0, 0.0])
    far_points = torch.zeros((1, 64, 3))
    far_valid = torch.zeros((1, 64), dtype=torch.bool)
    far_valid[0, :4] = True
    far_points[0, :4, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    near_keys = torch.arange(256, 0, -1, dtype=torch.float32).reshape(1, 256)
    far_keys = torch.arange(64, 0, -1, dtype=torch.float32).reshape(1, 64)
    window.write(
        env_ids=[0], episode_ids=[0], scan_ids=[7], times=[0.1],
        near_points=near_points, near_valid=near_valid,
        far_points=far_points, far_valid=far_valid,
        near_angle_keys=near_keys if keys_enabled else None,
        far_angle_keys=far_keys if keys_enabled else None,
    )
    snapshot = window.read()
    window_keys = window.read_angle_keys()
    observation = _observation()
    if keys_enabled:
        near_sort_keys = near_keys
        far_sort_keys = window_keys.far.reshape(1, -1)
    else:
        geometric = lambda points: torch.atan2(points[..., 1], points[..., 0])
        near_sort_keys = geometric(snapshot.near_points)
        far_sort_keys = geometric(snapshot.far_points.reshape(1, -1, 3))
    prepared = prepare_control_inputs(
        window=snapshot,
        observation=observation,
        near_angle_keys=near_sort_keys,
        far_angle_keys=far_sort_keys,
    )
    entry = {
        "episode": 0,
        "scan": 7,
        "near_indices": 3000 + torch.arange(256),
        "far_indices": 4000 + torch.arange(64),
        "near_no_hit": (torch.arange(256) % 2 == 0),
        "far_no_hit": (torch.arange(64) % 3 == 0),
    }
    return snapshot, window_keys, prepared, entry


def test_selected_point_is_larger_3d_sphere_and_cleans_up_on_rerender():
    """Exercise the real marker renderer without opening a Viser server."""
    from types import SimpleNamespace

    class FakeHandle:
        def __init__(self):
            self.removed = False

        def remove(self):
            self.removed = True

    class FakeScene:
        def __init__(self):
            self.calls = []
            self.handles = []

        def add_icosphere(self, name, **kwargs):
            self.calls.append((name, kwargs))
            handle = FakeHandle()
            self.handles.append(handle)
            return handle

    viewer = PlayViewer.__new__(PlayViewer)
    viewer._handles = {}
    viewer._last_pick = {"valid": True, "xyz_base": [1., 0.2, -0.5]}
    viewer._gui_view = SimpleNamespace(value="BASE（网络输入系）")
    scene = FakeScene()
    snapshot = {
        "pose": {
            "root_pos_w": np.array([4.0, -3.0, 0.8]),
            "root_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
    }

    viewer._render_pick_highlight(scene, snapshot)
    name, kwargs = scene.calls[-1]
    assert name == "/picker/highlight"
    assert kwargs["radius"] == PICKER_HIGHLIGHT_SPHERE_RADIUS_M == 0.12
    assert 2 * kwargs["radius"] > 0.06  # default near/FPS point visual size
    assert kwargs["color"] == (255, 0, 255)
    assert kwargs["subdivisions"] >= 2
    assert np.allclose(kwargs["position"], [1., 0.2, -0.5])
    assert list(viewer._handles) == ["/picker/highlight"]

    # WORLD view must transform the same BASE coordinate exactly once,
    # replacing the previous handle rather than leaving a second marker.
    old_handle = scene.handles[-1]
    viewer._gui_view.value = "WORLD（当前位姿）"
    viewer._render_pick_highlight(scene, snapshot)
    assert old_handle.removed
    assert np.allclose(scene.calls[-1][1]["position"], [5.0, -2.8, 0.3])
    assert len(viewer._handles) == 1

    # An invalid pick and an explicit cleared pick both remove the sphere.
    old_handle = scene.handles[-1]
    viewer._last_pick = {"valid": False, "xyz_base": [0., 0., 0.]}
    viewer._render_pick_highlight(scene, snapshot)
    assert old_handle.removed and "/picker/highlight" not in viewer._handles
    viewer._last_pick = None
    viewer._render_pick_highlight(scene, snapshot)
    assert len(scene.calls) == 2 and not viewer._handles


def test_ground_preview_geometry_is_readonly_and_validated():
    segments, points = preview_rectangle_geometry(
        0.8, 3.3, 0.6, z_base_m=-0.46, nx=17, ny=11,
    )
    assert segments.shape == (4, 2, 3)
    assert points.shape == (187, 3)
    assert np.allclose(points[0], [0.8, -0.6, -0.46])
    assert np.allclose(points[-1], [3.3, 0.6, -0.46])
    assert np.allclose(segments[0, 0], [0.8, -0.6, -0.46])
    assert np.allclose(segments[0, 1], [3.3, -0.6, -0.46])
    assert np.allclose(segments[-1, 1], segments[0, 0])
    for bounds in ((1., 1., .6), (2., 1., .6), (1., 3., 0.),
                   (float("nan"), 3., .6), (1., float("inf"), .6)):
        try:
            preview_rectangle_geometry(*bounds)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid preview bounds accepted: {bounds}")


def test_ground_preview_layers_toggle_transform_and_invalid_input():
    from types import SimpleNamespace
    from hexapod_mjlab.tasks.confined_locomotion.ground_presets import s1_187_ground_grid

    class FakeHandle:
        def __init__(self):
            self.removed = False
        def remove(self):
            self.removed = True

    class FakeScene:
        def __init__(self):
            self.calls = []
            self.handles = []
        def add_line_segments(self, name, pts, **kwargs):
            self.calls.append((name, np.asarray(pts).copy(), kwargs))
            handle = FakeHandle()
            self.handles.append(handle)
            return handle
        def add_point_cloud(self, name, pts, **kwargs):
            return self.add_line_segments(name, pts, **kwargs)

    grid = s1_187_ground_grid()
    viewer = PlayViewer.__new__(PlayViewer)
    viewer.ground_grid = grid
    viewer._reference_rectangle = (0.8, 3.2, -0.5, 0.5)
    viewer._handles = {}
    viewer._gui_candidate_show = SimpleNamespace(value=False)
    viewer._gui_candidate_near = SimpleNamespace(value=0.8)
    viewer._gui_candidate_far = SimpleNamespace(value=3.3)
    viewer._gui_candidate_halfwidth = SimpleNamespace(value=0.6)
    viewer._gui_candidate_points = SimpleNamespace(value=True)
    viewer._gui_candidate_status = SimpleNamespace(content="")
    viewer._gui_view = SimpleNamespace(value="BASE（网络输入系）")
    scene = FakeScene()
    snapshot = {
        "pose": {
            "root_pos_w": np.array([1., 2., 3.]),
            "root_quat_wxyz": [1., 0., 0., 0.],
        },
        "labels": {
            "ground_heights": np.full(187, -.45, dtype=np.float64),
            "ground_valid": np.ones(187, dtype=bool),
        },
    }

    viewer._render_ground_range_preview(scene, snapshot)
    assert not scene.calls and not viewer._handles  # default hidden

    viewer._gui_candidate_show.value = True
    viewer._render_ground_range_preview(scene, snapshot)
    assert set(viewer._handles) == {
        "/ground_preview/true_bounds",
        "/ground_preview/candidate_box",
        "/ground_preview/candidate_points",
    }
    actual = scene.calls[0][1]
    candidate = scene.calls[1][1]
    shown_points = scene.calls[2][1]
    assert np.allclose(actual[0, 0], [0.8, -0.5, -.45])
    assert np.allclose(candidate[0, 0], [0.8, -0.6, -.435])
    assert shown_points.shape == (187, 3)
    assert np.allclose(shown_points[-1], [3.3, 0.6, -.435])

    old_handles = list(viewer._handles.values())
    viewer._gui_view.value = "WORLD（当前位姿）"
    viewer._render_ground_range_preview(scene, snapshot)
    assert all(h.removed for h in old_handles)
    assert np.allclose(scene.calls[-2][1][0, 0], [1.8, 1.4, 2.565])
    assert len(viewer._handles) == 3

    old_handles = list(viewer._handles.values())
    viewer._gui_candidate_far.value = 0.7  # far <= near: no stale layers
    viewer._render_ground_range_preview(scene, snapshot)
    assert all(h.removed for h in old_handles)
    assert not viewer._handles
    assert "无效候选范围" in viewer._gui_candidate_status.content

    viewer._gui_candidate_far.value = 3.3
    viewer._gui_candidate_points.value = False
    viewer._render_ground_range_preview(scene, snapshot)
    assert len(viewer._handles) == 2
    viewer._gui_candidate_show.value = False
    old_handles = list(viewer._handles.values())
    viewer._render_ground_range_preview(scene, snapshot)
    assert not viewer._handles and all(h.removed for h in old_handles)


def test_hexagon_and_height_marker_use_documented_display_coordinates():
    envelope = (0.5, 0.7, 0.9, 0.4, -0.6, 0.03)
    vertices = hex_vertices(envelope)
    assert tuple(vertices) == HEX_VERTEX_ORDER == ("B", "D", "F", "E", "C", "A")
    assert vertices == {
        "B": (0.4, 0.5), "D": (0.0, 0.7), "F": (-0.6, 0.9),
        "E": (-0.6, -0.9), "C": (0.0, -0.7), "A": (0.4, -0.5),
    }
    assert root_height_target_m(envelope) == H_REF_M + 0.03
    assert np.allclose(
        root_height_marker_world((1.2, -0.4, 9.0), 0.1, envelope),
        np.array([1.2, -0.4, 0.1 + H_REF_M + 0.03]),
        atol=1e-12,
    )


def test_base_world_transform_and_quaternion_roundtrip_include_full_rotation():
    roll, pitch, yaw = 0.42, -0.31, 1.1
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    q = np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])
    rotation = quat_wxyz_to_matrix(q)
    base_point = np.array([0.3, -0.8, 1.1])
    root = np.array([2.0, -1.0, 0.6])
    world = root + rotation @ base_point
    assert np.allclose(world_to_base(world, root, q), base_point, atol=1e-12)
    recovered_rotation = quat_wxyz_to_matrix(rotation_matrix_to_wxyz(rotation))
    assert np.allclose(recovered_rotation, rotation, atol=1e-12)


def test_mount_and_cone_geometry_are_normalized_and_axis_relative():
    position, quaternion = normalize_mount([[0.1, 0.2, 0.3]], [[0.0, 0.0, 0.0, 2.0]])
    assert np.array_equal(position, np.array([0.1, 0.2, 0.3]))
    assert np.array_equal(quaternion, np.array([0.0, 0.0, 0.0, 1.0]))
    try:
        normalize_mount([0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
    except ValueError:
        pass
    else:
        raise AssertionError("malformed mount position was accepted")

    axis = cone_axis_base([0.0, 0.0, 0.0, 1.0], tilt_deg=CONE_TILT_DEG)
    expected_axis = np.array([math.sin(math.radians(CONE_TILT_DEG)), 0.0,
                              math.cos(math.radians(CONE_TILT_DEG))])
    assert np.allclose(axis, expected_axis, atol=1e-12)
    origin = np.array([0.4, -0.7, 1.2])
    length, half_angle, rim_count = 4.5, 58.0, 16
    segments = cone_wireframe(origin, axis, half_angle, length, n_rim=rim_count)
    assert segments.shape == (2 * rim_count, 2, 3)
    rim = segments[rim_count:, 1]
    relative = rim - origin
    assert np.allclose(relative @ axis, length, atol=1e-12)
    radial = relative - np.outer(relative @ axis, axis)
    assert np.allclose(np.linalg.norm(radial, axis=1),
                       length * math.tan(math.radians(half_angle)), atol=1e-12)
    generator = segments[0, 1] - segments[0, 0]
    cosine = float(np.dot(generator, axis) / np.linalg.norm(generator))
    assert abs(cosine - math.cos(math.radians(half_angle))) < 1e-12
    reversed_segments = cone_wireframe(origin, [0.0, 0.0, -1.0], 32.0, 1.0)
    assert np.isfinite(reversed_segments).all()


def test_B8_viewer_wireframe_is_clipped_in_base_before_world_projection():
    lines = np.array([
        [[0., 0., 0.], [4., 1., -1.]],
        [[4., 0., 0.], [2., 2., 2.]],
        [[5., 1., 0.], [6., 2., 0.]],
        [[1., -1., 0.], [2., -2., 0.]],
    ])
    result = clip_segments_base_x(lines, x_max_m=3.5)
    assert result.shape == (3, 2, 3)
    assert np.all(result[:, :, 0] <= 3.5 + 1e-12)
    assert np.allclose(result[0, 1], (3.5, 0.875, -0.875))
    assert np.allclose(result[1, 0], (3.5, 0.5, 0.5))
    assert np.array_equal(result[2], lines[3])
    assert np.array_equal(clip_segments_base_x(lines, x_max_m=None), lines)


def test_source_mapping_preserves_duplicate_points_invalid_slots_and_scan_identity():
    snapshot, keys, prepared, entry = _source_fixture()
    result = resolve_point_sources(
        snapshot=snapshot,
        angle_keys=keys,
        geometric_keys_fn=None,
        prepared=prepared,
        capture_entries=[entry],
        env_row=0,
        episode_id=0,
    )
    near_valid = result["near_valid"]
    assert int(near_valid.sum()) == 3
    assert int(torch.unique(result["near_raw_ids"][near_valid]).numel()) == 3
    assert result["near_raw_ids"][:3].tolist() == [3002, 3001, 3000]
    assert result["near_no_hit"][:3].tolist() == [True, False, True]
    assert torch.equal(result["near_raw_ids"][~near_valid],
                       torch.full_like(result["near_raw_ids"][~near_valid], -1))

    far_valid = result["far_valid"]
    assert int(far_valid.sum()) == 4
    assert result["far_raw_ids"][:4].tolist() == [4003, 4002, 4001, 4000]
    assert result["far_source_scan_ids"][:4].tolist() == [7, 7, 7, 7]
    assert result["far_source_slot"][:4].tolist() == [0, 0, 0, 0]
    assert result["far_no_hit"][:4].tolist() == [True, False, False, True]
    assert torch.equal(result["far_raw_ids"][~far_valid],
                       torch.full_like(result["far_raw_ids"][~far_valid], -1))


def test_source_mapping_refuses_a_wrong_sort_permutation_and_supports_clean_fallback():
    snapshot, keys, prepared, entry = _source_fixture()
    wrong = WindowAngleKeys(
        near=keys.near,
        far=torch.flip(keys.far, dims=[2]),
        present=keys.present,
    )
    try:
        resolve_point_sources(
            snapshot=snapshot, angle_keys=wrong, geometric_keys_fn=None,
            prepared=prepared, capture_entries=[entry], env_row=0, episode_id=0,
        )
    except RuntimeError as error:
        assert "permutation" in str(error)
    else:
        raise AssertionError("mismatched source order was guessed through")

    snapshot, keys, prepared, entry = _source_fixture(keys_enabled=False)
    result = resolve_point_sources(
        snapshot=snapshot,
        angle_keys=keys,
        geometric_keys_fn=lambda points: torch.atan2(points[..., 1], points[..., 0]),
        prepared=prepared,
        capture_entries=[entry],
        env_row=0,
        episode_id=0,
    )
    assert result["near_matched"] == 3
    assert result["far_matched"] == 4


def run_cpu_checks() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} CPU checks passed")
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(run_cpu_checks())
