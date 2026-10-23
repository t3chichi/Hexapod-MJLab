"""Pure CPU contract checks for A3 moving-pose FAR hit-only experiment."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/experiments/confined_locomotion_2/sparse_rays/evaluate_far_trajectory.py"


def _load():
    key = str(SCRIPT.parent)
    added = key not in sys.path
    if added:
        sys.path.insert(0, key)
    try:
        spec = importlib.util.spec_from_file_location("far_motion_A3", SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        if added:
            sys.path.remove(key)


def test_far_motion_interpolation_has_bounded_kinematic_step():
    m = _load()
    poses = [m._pose(0., 0., 0.), m._pose(.25, 5., .1)]
    interpolated = m.interpolate_poses(poses, 5)
    assert len(interpolated) == 6
    assert interpolated[0] == poses[0] and interpolated[-1] == poses[-1]
    assert np.allclose(np.diff([p["x_m"] for p in interpolated]), .05)
    assert np.allclose(np.diff([p["pitch_down_deg"] for p in interpolated]), 1)
    assert np.allclose(np.diff([p["height_delta_m"] for p in interpolated]), .02)
    for invalid in (0, -1, 21, True):
        try:
            m.interpolate_poses(poses, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid interpolation step accepted")


def test_far_motion_physical_15m_ray_and_real_roof_vs_floor_hits():
    m = _load()
    dirs = np.array([
        [1.0, 0., 0.],
        [1.0, 0., -0.50],
        [1.0, 0., 0.25],
        [1.0, 0., 0.40],
    ], dtype=np.float64)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    mount = {
        "base_rot_world": np.eye(3),
        "base_pos_world": np.array([0., 0., .46]),
        "mount_pos_base": np.array([0., 0., -.164]),
        "mount_rot_base": np.eye(3),
    }
    roof = m._solid("roof", (.8, -.8, .73), (2.8, .8, .85), kind="roof")
    fixed_near = np.array([False, True, False, False])
    scan = m.scan_pose(dirs, mount, m._pose(0, 0), [roof], fixed_near=fixed_near)
    assert scan["hit_label"][0] == -1
    assert scan["distances_m"][0] == 15.0
    assert scan["hit_label"][1] == 0
    assert (scan["hit_label"][2:] == 1).all()
    assert np.array_equal(scan["near"], fixed_near)
    before = scan["distances_m"].copy()
    original = m.choose(scan, np.arange(4), far_method="original")
    hit15 = m.choose(scan, np.arange(4), far_method="hit15")
    hit10 = m.choose(scan, np.arange(4), far_method="hit10")
    assert len(original["far"]) == 3
    assert len(hit15["far"]) == 2
    assert len(hit10["far"]) == 2
    assert 0 not in hit10["far"]
    assert np.array_equal(hit15["far"], hit10["far"])
    assert np.array_equal(scan["distances_m"], before)
    # One meter forward rigid translation is a different sensor location,
    # yet near/far raw channel assignment must remain fixed.
    moved = m.scan_pose(dirs, mount, m._pose(1., 2.),
                        [roof], fixed_near=fixed_near)
    assert np.array_equal(moved["near"], fixed_near)
    assert np.isfinite(moved["points"]).all()


def test_far_motion_scenarios_focus_stairs_and_clearance_not_thin_objects():
    m = _load()
    cases = m.make_scenarios()
    assert len({c["name"] for c in cases}) == len(cases)
    by_name = {c["name"]: c for c in cases}
    assert set(("ascending_stairs", "descending_stairs", "low_ceiling",
                "low_tunnel", "tunnel_with_step")) <= set(by_name)
    assert sum(s["kind"] == "roof" for s in by_name["low_tunnel"]["solids"]) == 1
    assert any(s["kind"] == "stair" for s in by_name["tunnel_with_step"]["solids"])
    for scene in cases:
        assert all(s["hi"][0] > s["lo"][0] and
                   s["hi"][1] > s["lo"][1] and
                   s["hi"][2] > s["lo"][2] for s in scene["solids"])
