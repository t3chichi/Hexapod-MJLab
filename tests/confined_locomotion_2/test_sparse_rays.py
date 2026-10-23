"""Contract checks for the separate static sparse-ray experiment (no GPU/training)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SELECTOR = ROOT / "scripts/experiments/confined_locomotion_2/sparse_rays/select_channels.py"
COMPARE = ROOT / "scripts/experiments/confined_locomotion_2/sparse_rays/compare_sampling.py"
MOVING = ROOT / "scripts/experiments/confined_locomotion_2/sparse_rays/evaluate_moving_footprint.py"
OBSTACLES = ROOT / "scripts/experiments/confined_locomotion_2/sparse_rays/evaluate_obstacles.py"
ROBUSTNESS = ROOT / "scripts/experiments/confined_locomotion_2/sparse_rays/evaluate_random_robustness.py"


def _load_selector():
    spec = importlib.util.spec_from_file_location("confined_v2_sparse_select", SELECTOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_compare():
    import sys
    # Executed as standalone from its own scripts directory in production.
    key = str(COMPARE.parent)
    inserted = key not in sys.path
    if inserted:
        sys.path.insert(0, key)
    try:
        spec = importlib.util.spec_from_file_location("confined_v2_sparse_compare", COMPARE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if inserted:
            sys.path.remove(key)


def _load_moving():
    import sys
    key = str(MOVING.parent)
    inserted = key not in sys.path
    if inserted:
        sys.path.insert(0, key)
    try:
        spec = importlib.util.spec_from_file_location("confined_v2_moving_footprint", MOVING)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if inserted:
            sys.path.remove(key)


def _load_obstacles():
    import sys
    key = str(OBSTACLES.parent)
    inserted = key not in sys.path
    if inserted:
        sys.path.insert(0, key)
    try:
        spec = importlib.util.spec_from_file_location("confined_v2_sparse_obstacles", OBSTACLES)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if inserted:
            sys.path.remove(key)


def _load_robustness():
    import sys
    key = str(ROBUSTNESS.parent)
    inserted = key not in sys.path
    if inserted:
        sys.path.insert(0, key)
    try:
        spec = importlib.util.spec_from_file_location(
            "confined_v2_sparse_random_robustness", ROBUSTNESS
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if inserted:
            sys.path.remove(key)


def test_sparse_source_directions_are_original_airy():
    s = _load_selector()
    dirs = s.airy_directions()
    assert dirs.shape == (86400, 3)
    assert np.isfinite(dirs).all()
    assert np.allclose(np.linalg.norm(dirs, axis=1), 1, atol=2e-7)
    assert np.allclose(dirs[0], [1, 0, 0], atol=1e-7)
    assert np.allclose(dirs[95], [0, 0, 1], atol=1e-6)
    # Next azimuth has precisely 0.4 degree rotation in sensor frame.
    assert np.isclose(np.arctan2(dirs[96, 1], dirs[96, 0]), np.deg2rad(0.4), atol=1e-6)


def test_sparse_stand_mount_and_b8_cone():
    s = _load_selector()
    mount = s.nominal_mount()
    assert mount["root_z_nominal"] > 0
    assert np.isclose(np.linalg.det(mount["mount_rot_base"]), 1.0, atol=1e-6)
    dirs = s.airy_directions()
    scan = s.nominal_scan(dirs, mount)
    assert scan["near"].shape == (86400,)
    assert np.array_equal(scan["near"] | scan["far"], np.ones(86400, dtype=bool))
    assert not np.any(scan["near"] & scan["far"])
    assert np.all(scan["endpoints_base"][scan["near"], 0] <= 3.5 + 1e-8)
    assert 256 < np.count_nonzero(scan["near"] & scan["hit"])
    assert 128 < np.count_nonzero(scan["far"])


def test_sparse_frozen_assets_are_catalog_bound_nested_and_explicit():
    from hexapod_mjlab.tasks.confined_locomotion_2.channels import (
        build_channel_catalog,
        load_sparse_selection,
    )
    design = ROOT / "src/hexapod_mjlab/tasks/confined_locomotion_2/assets/fixed_channels/v1"
    catalog = build_channel_catalog()
    choices = {}
    for mult in (4, 5, 6, 8):
        path = design / f"near_x{mult}_far_x4.npz"
        ids, identity = load_sparse_selection(path, catalog)
        assert identity["catalog_identity"] == catalog.identity
        assert np.all(np.diff(ids) > 0)
        with np.load(path, allow_pickle=False) as archive:
            near_ids = archive["designed_near_ids"]
            far_ids = archive["designed_far_ids"]
            assert np.array_equal(archive["source_ray_ids"], ids)
            assert np.array_equal(archive["source_ray_directions_sensor"],
                                  catalog.directions_sensor[ids])
        assert np.all(catalog.near_mask[near_ids])
        assert np.all(~catalog.near_mask[far_ids])
        assert len(ids) == 256 * mult + 512
        choices[mult] = near_ids
    for smaller, larger in ((4, 5), (5, 6), (6, 8)):
        assert np.array_equal(choices[smaller], choices[larger][:len(choices[smaller])])
    try:
        load_sparse_selection(design / "manifest.json", catalog)
    except ValueError as exc:
        assert "pass a specific near_xN_far_xN" in str(exc)
    else:
        raise AssertionError("index manifest silently selected an undecided multiplier")


def test_sparse_selector_is_deterministic_nested_unique():
    s = _load_selector()
    pts = np.stack([np.arange(20), np.arange(20) ** 2, np.zeros(20)], axis=1)
    ids = np.array([1, 3, 5, 7, 9, 11, 13, 15], dtype=np.int64)
    seq_a = s.spatial_order(pts, ids, 6)
    seq_b = s.spatial_order(pts, ids, 6)
    assert np.array_equal(seq_a, seq_b)
    assert len(np.unique(seq_a)) == len(seq_a)
    assert set(seq_a).issubset(set(ids))
    assert np.array_equal(seq_a[:4], s.spatial_order(pts, ids, 4))


def test_sparse_recipe_summary_accepts_far_x8_only_manifest():
    moving = _load_moving()
    recipes = {"near_x4_far_x8": np.arange(8, dtype=np.int64)}
    selected = moving.resolve_recipe_names(recipes)
    assert selected == ["near_x4_far_x8"]
    assert moving.resolve_recipe_names(recipes, ["near_x4_far_x8"]) == selected
    summary = moving.recipe_p95_summary({
        "near_x4_far_x8": {
            "consumed_fixed_near_fps256_coverage": {"p95_nn_m": 0.42},
        },
    })
    assert summary == {"near_x4_far_x8": 0.42}
    try:
        moving.resolve_recipe_names(recipes, ["near_x4_far_x4"])
    except ValueError as exc:
        assert "unknown frozen recipe" in str(exc)
    else:
        raise AssertionError("an absent recipe was silently substituted")


def test_sparse_analytic_scene_keeps_catalog_membership_fixed():
    s = _load_selector()
    c = _load_compare()
    dirs = s.airy_directions()
    mount = s.nominal_mount()
    nominal = s.nominal_scan(dirs, mount)
    analytic = c.intersect_scene(dirs, mount, width=None, pitch_deg=0, roll_deg=0)
    assert np.array_equal(analytic["near"], nominal["near"])
    assert np.allclose(analytic["points"], nominal["endpoints_base"], atol=1e-6)
    corridor = c.intersect_scene(dirs, mount, width=1.6, pitch_deg=0, roll_deg=0)
    assert (corridor["hit_type"] == 2).any()
    # Static ownership remains fixed even though endpoint-derived B8 would move.
    assert np.array_equal(corridor["near"], nominal["near"])
    assert np.any(corridor["dynamic_near"] != nominal["near"])


def test_sparse_pitch_sign_labels_are_physically_correct():
    from scipy.spatial.transform import Rotation
    c = _load_compare()
    cases = {label: pitch for label, _, pitch, _ in c.CASES}
    assert cases["flat_pitch_up_5"] == -5.0
    assert cases["flat_pitch_down_5"] == 5.0
    for label in ("flat_pitch_up_5", "flat_pitch_down_5"):
        x_world = Rotation.from_euler(
            "xyz", [0, cases[label], 0], degrees=True
        ).apply([1, 0, 0])
        if "up" in label:
            assert x_world[2] > 0
        else:
            assert x_world[2] < 0


def test_sparse_independent_moving_footprint_nominal_identity():
    s = _load_selector()
    f = _load_moving()
    mount = s.nominal_mount()
    template = f.make_independent_template(mount)
    assert template["nominal_valid_count"] > 500
    assert template["grid_total_count"] > template["nominal_valid_count"]
    assert np.allclose(np.linalg.norm(template["directions_base"], axis=1), 1)
    nominal = f.reproject_template(template, mount, pitch_deg=0, roll_deg=0)
    assert nominal["floor_hit_count"] == template["nominal_valid_count"]
    assert nominal["near_probe_count"] == template["nominal_valid_count"]
    assert np.allclose(nominal["valid_target_points_base"],
                       template["nominal_points_base"], atol=1e-10)
    assert np.all(nominal["valid_target_points_base"][:, 0] <= s.NEAR_MAX_X_M)


def test_sparse_moving_footprint_shifts_with_pitch_and_reports_validity():
    s = _load_selector()
    f = _load_moving()
    mount = s.nominal_mount()
    template = f.make_independent_template(mount)
    level = f.reproject_template(template, mount, pitch_deg=0, roll_deg=0)
    down = f.reproject_template(template, mount, pitch_deg=5, roll_deg=0)
    up = f.reproject_template(template, mount, pitch_deg=-5, roll_deg=0)
    assert down["near_probe_count"] == template["nominal_valid_count"]
    assert np.max(down["valid_target_points_base"][:, 0]) < np.max(
        level["valid_target_points_base"][:, 0]) - 0.3
    assert up["near_probe_count"] < template["nominal_valid_count"]
    assert up["near_fraction_of_nominal"] < 1
    assert np.any(up["valid_target_points_base"][:, 0] > s.NEAR_MAX_X_M)
    # Reject the common bug: rigidly rotating a nominal ground patch without
    # re-intersecting world z=0 would leave targets at varying world heights.
    from scipy.spatial.transform import Rotation
    for pitch, frame in ((5, down), (-5, up)):
        rot = mount["base_rot_world"] @ Rotation.from_euler(
            "xyz", [0, pitch, 0], degrees=True
        ).as_matrix()
        root = mount["base_pos_world"]
        world = root + (rot @ frame["valid_target_points_base"].T).T
        assert np.max(np.abs(world[:, 2])) < 1e-10


def test_sparse_moving_footprint_height_and_empty_metric_guard():
    s = _load_selector()
    f = _load_moving()
    mount = s.nominal_mount()
    template = f.make_independent_template(mount)
    level = f.reproject_template(template, mount, pitch_deg=0, roll_deg=0)
    higher = f.reproject_template(template, mount, pitch_deg=0, roll_deg=0,
                                  height_delta_m=0.08)
    assert higher["near_probe_count"] <= level["near_probe_count"]
    assert not np.allclose(higher["valid_target_points_base"],
                           level["valid_target_points_base"])
    assert f.distance_metrics(np.empty((0, 3)), np.empty((0, 3)))["status"] == "no_valid_theoretical_floor_probes"
    assert f.distance_metrics(np.array([[1., 0, 0]]), np.empty((0, 3)))["status"] == "no_selected_real_hits"
    assert np.isclose(f.distance_metrics(np.array([[1., 0, 0]]),
                                         np.array([[1., 0, 0]]))["p95_nn_m"], 0)


def test_sparse_obstacle_ray_aabb_handles_hit_miss_parallel_and_inside():
    o = _load_obstacles()
    dirs = np.array([[1., 0, 0], [-1., 0, 0], [0, 1, 0],
                     [0, 0, 1], [0, 0, -1]], dtype=np.float64)
    lo, hi = np.array([1., -0.5, 0.0]), np.array([2., 0.5, 1.0])
    distance = o.ray_aabb_entry(np.array([0., 0., 0.5]), dirs, lo, hi)
    assert np.isclose(distance[0], 1)
    assert np.isinf(distance[1:]).all()
    # Optical origin inside solid must exit at positive length, not report t=0.
    inner = o.ray_aabb_entry(np.array([1.5, 0, 0.5]), dirs, lo, hi)
    assert np.isclose(inner[0], 0.5)
    assert np.isclose(inner[1], 0.5)


def test_sparse_obstacle_scene_first_hit_beats_floor_and_assignment_is_fixed():
    s = _load_selector()
    o = _load_obstacles()
    rays = s.airy_directions()
    mount = s.nominal_mount()
    case = next(x for x in o.SCENES if x["name"] == "step_25cm")
    scan = o.scan_obstacles(rays, mount, case)
    assert np.sum(scan["hit_label"] == 1) > 1000
    assert np.sum(scan["hit_label"] == 0) > 1000
    assert np.sum(scan["hit_label"] == -1) > 1000
    assert np.array_equal(scan["near"] | scan["far"],
                          np.ones(s.FULL_RAYS, dtype=bool))
    assert not np.any(scan["near"] & scan["far"])
    assert np.array_equal(scan["near"], s.build_channel_catalog().near_mask)
    full = o.make_full_reference(scan, case)
    assert full["objects"][0]["full_fps_near"] > 0
    assert full["objects"][0]["full_raw_visible_rays"] > full["objects"][0]["full_fps_near"]


def test_sparse_step_support_separates_raw_fps_and_full():
    # Use a representative 25cm step, not a sub-FPS-resolution thin post.
    s = _load_selector()
    o = _load_obstacles()
    mount = s.nominal_mount()
    rays = s.airy_directions()
    scenario = next(x for x in o.SCENES if x["name"] == "step_25cm")
    scan = o.scan_obstacles(rays, mount, scenario)
    full = o.make_full_reference(scan, scenario)
    ids = np.arange(s.FULL_RAYS, dtype=np.int64)
    trial = o.evaluate_scenario(scan, ids, full_reference=full)["objects"]["step"]
    assert trial["full_raw_visible_rays"] == trial["raw_visible"] > 0
    assert trial["raw_visible"] > trial["combined_fps_visible"] > 0
    assert trial["raw_surface_coverage"]["probes"] > 0


def test_sparse_main_robustness_scenes_exclude_micro_features():
    r = _load_robustness()
    scenes = r.stress_scenes()
    assert len(scenes) >= 8
    assert len({x["name"] for x in scenes}) == len(scenes)
    for scene in scenes:
        for _, lower, upper in scene["boxes"]:
            dx, dy, dz = np.asarray(upper) - np.asarray(lower)
            assert max(dx, dy) >= 0.08
            assert dz >= 0.10
    assert all("3cm" not in x["name"] and "5cm_far" not in x["name"] for x in scenes)


def test_sparse_overhead_fps_zero_even_on_full_scan():
    s = _load_selector()
    o = _load_obstacles()
    mount = s.nominal_mount()
    rays = s.airy_directions()
    scene = next(x for x in o.SCENES if x["name"] == "overhead_beam")
    scanned = o.scan_obstacles(rays, mount, scene)
    full = o.make_full_reference(scanned, scene)
    obstacle = full["objects"][0]
    assert obstacle["full_raw_visible_rays"] > 0
    assert obstacle["full_fps_near"] + obstacle["full_fps_far"] == 0
    ref = o.evaluate_scenario(scanned, np.arange(s.FULL_RAYS), full_reference=full)["objects"]["beam"]
    assert ref["raw_visible"] > 0 and ref["combined_fps_visible"] == 0
    assert ref["fps_surface_coverage"]["status"] == "missed"


def test_sparse_robust_fps_seed_and_exact_pool_identity():
    c = _load_compare()
    r = _load_robustness()
    points = np.array([[0, 0, 0], [4, 0, 0], [1, 0, 0],
                       [2, 0, 0], [0, 3, 0], [-2, 0, 0]], dtype=np.float32)
    ids = np.array([5, 1, 4, 0, 3, 2], dtype=np.int64)
    chosen = r.random_start_fps(points, ids, 4, uniform_start=1 / 6)
    assert np.array_equal(chosen, c.fps_ids(points, ids, 4))
    assert len(np.unique(chosen)) == len(chosen)
    assert len(r.random_start_fps(points, np.array([], dtype=np.int64), 4, 0.2)) == 0
    assert np.array_equal(r.random_start_fps(points, ids, 6, 0.7), np.sort(ids))


def test_sparse_robust_noise_does_not_distort_misses_or_source_ids():
    s = _load_selector()
    o = _load_obstacles()
    r = _load_robustness()
    source = s.airy_directions()
    mount = s.nominal_mount()
    scenario = next(x for x in o.SCENES if x["name"] == "step_10cm")
    scan = o.scan_obstacles(source, mount, scenario)
    rng_a = np.random.default_rng(np.random.SeedSequence([1234, 0, 7, 2]))
    rng_b = np.random.default_rng(np.random.SeedSequence([1234, 0, 7, 2]))
    obs_a, near_a = r.noisy_geometry(scan, source, mount, 0.015, rng_a)
    obs_b, near_b = r.noisy_geometry(scan, source, mount, 0.015, rng_b)
    assert np.array_equal(obs_a, obs_b) and np.array_equal(near_a, near_b)
    assert np.array_equal(near_a, scan["near"])
    miss = scan["hit_label"] == -1
    assert np.allclose(obs_a[miss], scan["points"][miss], atol=1e-9)
    hit = ~miss
    delta = obs_a[hit] - scan["points"][hit]
    ray_unit = (
        mount["mount_rot_base"] @ source.astype(np.float64).T
    ).T[hit]
    cross = np.cross(delta, ray_unit)
    assert np.max(np.linalg.norm(cross, axis=1)) < 1e-9
    assert np.any(np.linalg.norm(delta, axis=1) > 1e-4)


def test_sparse_robust_aggregation_reports_zero_failure_uncertainty():
    r = _load_robustness()
    series = [
        {"hits": 1, "support20": 0.75, "support10": 0.5,
         "near_candidates": 512, "far_candidates": 512},
        {"hits": 2, "support20": 1.0, "support10": 1.0,
         "near_candidates": 511, "far_candidates": 513},
    ]
    report = r.aggregate(series)
    assert report["zero_hit_count"] == 0
    assert report["hits_min"] == 1
    assert 0.0 < report["zero_failure_wilson95_upper_when_no_misses"] < 1.0
    series[1]["hits"] = 0
    assert r.aggregate(series)["zero_failure_wilson95_upper_when_no_misses"] is None
    assert r.wilson_upper_zero_failures(100) < 0.04


def test_sparse_fps_reference_not_repeated_and_counts():
    c = _load_compare()
    points = np.array([[0., 0, 0], [3., 0, 0], [0., 0, 0],
                       [1., 0, 0], [2., 0, 0]], dtype=np.float32)
    ids = np.arange(len(points), dtype=np.int64)
    picked = c.fps_ids(points, ids, 4)
    assert len(picked) == 4
    assert len(np.unique(picked)) == 4
    assert int(picked[0]) == 1


def test_sparse_bad_budget_and_missing_path_rejected(tmp_path):
    s = _load_selector()
    try:
        s.spatial_order(np.zeros((3, 3)), np.array([0, 1]), 3)
    except ValueError:
        pass
    else:
        raise AssertionError("requested candidates exceeding true sources accepted")
    existing = tmp_path / "occupied"
    existing.mkdir()
    try:
        s.main(["--output", str(existing), "--near-multipliers", "4"])
    except FileExistsError:
        pass
    else:
        raise AssertionError("existing evidence output was not rejected")
