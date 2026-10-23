"""CPU guards for the standalone far hit-only A1 comparison.

These tests do NOT change production grouping, sampling, or scanner contracts.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/experiments/confined_locomotion_2/sparse_rays/evaluate_far_hit_only.py"
A2 = ROOT / "scripts/experiments/confined_locomotion_2/sparse_rays/evaluate_far_range_threshold.py"


def _load(script=SCRIPT):
    key = str(script.parent)
    inserted = key not in sys.path
    if inserted:
        sys.path.insert(0, key)
    try:
        spec = importlib.util.spec_from_file_location("standalone_far_A1_or_A2_test", script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        if inserted:
            sys.path.remove(key)


def test_far_A1_excludes_raw_no_hit_and_preserves_real_surface():
    m = _load()
    n = 160
    pts = np.zeros((n, 3), dtype=np.float64)
    pts[:100, 0] = np.linspace(0.5, 4.9, 100)
    pts[:100, 1] = np.linspace(-0.6, 0.6, 100)
    pts[100:, 0] = 15.0
    pts[100:, 1] = np.linspace(-0.8, 0.8, 60)
    labels = np.full(n, -1, dtype=np.int16)
    labels[:80] = 0
    labels[80:100] = 1
    scene = {"name": "small_synthetic_floor_and_wall", "boxes": (
        ("wall", (0, -1, 0), (5, 1, 1)),)}
    scan = {"points": pts, "hit_label": labels,
            "far": np.ones(n, dtype=bool)}
    ids = np.arange(n, dtype=np.int64)
    old = m.evaluate_far(scan, ids, scene, remove_no_hit=False)
    new = m.evaluate_far(scan, ids, scene, remove_no_hit=True)
    assert old["far_fps_selected_count"] == 128
    assert old["far_fps_no_hit_count"] > 0
    assert new["far_fps_selected_count"] == 100
    assert new["far_fps_padding_count"] == 28
    assert new["far_fps_no_hit_count"] == 0
    assert set(new["selected_original_source_ids"]) <= set(range(100))
    assert new["objects"]["wall"]["full_raw_far_hit_count"] == 20


def test_far_A1_no_real_hit_has_all_128_slots_invalid():
    m = _load()
    n = 160
    xyz = np.zeros((n, 3), dtype=np.float64)
    xyz[:, 0] = 15.0
    scan = {"points": xyz, "hit_label": np.full(n, -1, dtype=np.int16),
            "far": np.ones(n, dtype=bool)}
    scene = {"name": "all_miss", "boxes": ()}
    old = m.evaluate_far(scan, np.arange(n), scene, remove_no_hit=False)
    new = m.evaluate_far(scan, np.arange(n), scene, remove_no_hit=True)
    assert old["far_fps_selected_count"] == 128
    assert old["far_fps_no_hit_count"] == 128
    assert new["far_fps_selected_count"] == 0
    assert new["far_fps_no_hit_count"] == 0
    assert new["far_fps_padding_count"] == 128


def test_far_A2_15m_is_A1_and_raw_hit_ceiling_excludes_distant_walls():
    a1 = _load()
    a2 = _load(A2)
    n = 180
    true_dist = np.r_[
        np.linspace(2.0, 13.2, 140), np.full(5, 9.0),
        np.full(5, 11.0), np.full(30, 15.0),
    ]
    labels = np.r_[
        np.zeros(140, dtype=np.int16),
        np.ones(5, dtype=np.int16),
        np.full(5, 2, dtype=np.int16),
        np.full(30, -1, dtype=np.int16),
    ]
    pts = np.zeros((n, 3), dtype=np.float64)
    pts[:, 0] = true_dist
    pts[:, 1] = np.linspace(-2, 2, n)
    scan = {
        "points": pts, "distances_m": true_dist, "hit_label": labels,
        "far": np.ones(n, dtype=bool),
    }
    scene = {"name": "two_distant_walls", "boxes": (
        ("wall_9m", (8.9, -2.0, 0), (9.1, 2, 2)),
        ("wall_11m", (10.9, -2.0, 0), (11.1, 2, 2)),
    )}
    ids = np.arange(n, dtype=np.int64)
    old = a1.evaluate_far(scan, ids, scene, remove_no_hit=False)
    same_15 = a1.evaluate_far(scan, ids, scene, remove_no_hit=True)
    d15 = a2.evaluate_limit(scan, ids, scene, 15.0)
    d10 = a2.evaluate_limit(scan, ids, scene, 10.0)
    d8 = a2.evaluate_limit(scan, ids, scene, 8.0)
    assert old["far_fps_no_hit_count"] > 0
    assert d15["selected_original_source_ids"] == same_15["selected_original_source_ids"]
    assert d15["fps_no_hit_count"] == d10["fps_no_hit_count"] == d8["fps_no_hit_count"] == 0
    assert d15["objects"]["wall_9m"]["subset_far_hits_within_limit"] == 5
    assert d10["objects"]["wall_9m"]["subset_far_hits_within_limit"] == 5
    assert d10["objects"]["wall_11m"]["subset_far_hits_within_limit"] == 0
    assert d8["objects"]["wall_9m"]["subset_far_hits_within_limit"] == 0
    assert d8["objects"]["wall_11m"]["subset_far_hits_within_limit"] == 0
    assert d15["hit_within_distance_count"] > d10["hit_within_distance_count"] > d8["hit_within_distance_count"]
    assert d8["fps_valid_count"] <= d10["fps_valid_count"] <= d15["fps_valid_count"]
    # A2 is a selection gate: it cannot mutate the underlying 15m raw scan.
    assert np.array_equal(scan["distances_m"], true_dist)


def test_far_A2_short_range_all_misses_and_invalid_limit():
    a2 = _load(A2)
    n = 180
    xyz = np.zeros((n, 3), dtype=np.float64)
    xyz[:, 0] = 15.0
    scan = {
        "points": xyz,
        "distances_m": np.full(n, 15.0),
        "hit_label": np.full(n, -1, dtype=np.int16),
        "far": np.ones(n, dtype=bool),
    }
    d8 = a2.evaluate_limit(scan, np.arange(n), {"name": "open", "boxes": ()}, 8.0)
    assert d8["raw_far_miss_count"] == n
    assert d8["fps_valid_count"] == 0
    assert d8["fps_invalid_padding_count"] == 128
    for bad in (0, -1, 15.1, float("nan")):
        try:
            a2.evaluate_limit(scan, np.arange(n), {"name": "open", "boxes": ()}, bad)
        except ValueError:
            continue
        raise AssertionError("invalid distance ceiling was accepted")
