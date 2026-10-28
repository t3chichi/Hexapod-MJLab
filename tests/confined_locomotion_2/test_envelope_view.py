"""Numerical checks for the display-only robot-frame envelope prism."""

from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from hexapod_mjlab.tasks.confined_locomotion_2.visualization.envelope import envelope_prism_base
from hexapod_mjlab.tasks.confined_locomotion_2.visualization.debug import DebugViewer


def test_prism_has_18_edges_with_identical_hexagons_and_reference_heights():
    values = np.array([.59, .69, .59, .89, -.89, .12])
    lines = envelope_prism_base(values)
    assert lines.shape == (18, 2, 3)
    np.testing.assert_allclose(lines[:6, :, 2], -.44)
    np.testing.assert_allclose(lines[6:12, :, 2], .12)
    np.testing.assert_array_equal(lines[:6, :, :2], lines[6:12, :, :2])
    np.testing.assert_array_equal(lines[12:, 0], lines[:6, 0])
    np.testing.assert_array_equal(lines[12:, 1], lines[6:12, 0])
    # At Root's reference height above ground, these are exactly the agreed
    # ground-relative planes; live Root-height changes translate the whole box.
    ground_z = .2
    np.testing.assert_allclose(lines[:6, :, 2] + ground_z + .46, ground_z + .02)
    np.testing.assert_allclose(lines[6:12, :, 2] + ground_z + .46, ground_z + .46 + values[5])
    changed = envelope_prism_base([.59, .69, .59, .89, -.89, -.6])
    np.testing.assert_array_equal(changed[:6], lines[:6])
    np.testing.assert_allclose(changed[6:12, :, 2], -.6)  # No prediction clamping.


def test_prism_follows_full_displayed_root_pose_and_base_world_views_agree():
    viewer = object.__new__(DebugViewer)
    viewer._view = SimpleNamespace(value="世界")
    viewer._handles = {}
    rendered = {}
    def add(name, points, **kwargs):
        rendered[name] = points.copy()
        return SimpleNamespace(remove=lambda: None)
    viewer.server = SimpleNamespace(scene=SimpleNamespace(add_line_segments=add))
    rotation = Rotation.from_euler("xyz", [.35, -.25, .7])
    root = np.array([1.2, -.3, .8])
    snapshot = {"pose": {"root_pos_w": root, "root_quat_wxyz": rotation.as_quat(scalar_first=True)}}
    envelope = [.4, .5, .3, .8, -.7, .1]
    local = envelope_prism_base(envelope)
    viewer._draw_envelope_values(snapshot, envelope)
    world = rendered["/envelope/prism"]
    np.testing.assert_allclose(world, local @ rotation.as_matrix().T + root, atol=1e-7)
    # Local vertical posts rotate with Roll/Pitch; they are not world vertical.
    direction = world[12, 1] - world[12, 0]
    assert abs(direction[0]) + abs(direction[1]) > .01
    viewer._view.value = "机器人"
    viewer._draw_envelope_values(snapshot, envelope)
    base = rendered["/envelope/prism"]
    np.testing.assert_allclose(base, local, atol=1e-7)
    np.testing.assert_allclose((world-root) @ rotation.as_matrix(), base, atol=1e-7)


def test_prism_five_shape_values_control_both_bottom_and_top():
    for dimension in range(5):
        original = [.4, .5, .3, .8, -.7, .1]
        changed = original.copy()
        changed[dimension] += .1
        before, after = envelope_prism_base(original), envelope_prism_base(changed)
        assert not np.array_equal(before[:6, :, :2], after[:6, :, :2])
        np.testing.assert_array_equal(after[:6, :, :2], after[6:12, :, :2])
