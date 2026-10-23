"""CPU contract checks for the fixed raw-channel catalog and artifacts."""
from __future__ import annotations

import json
import hashlib
import importlib.util
from functools import lru_cache
from pathlib import Path

import numpy as np

from hexapod_mjlab.tasks.confined_locomotion_2.channels import (
    ChannelCatalog,
    build_channel_catalog,
    load_sparse_selection,
)

ROOT = Path(__file__).resolve().parents[2]
SELECTOR = ROOT / "scripts/experiments/confined_locomotion_2/sparse_rays/select_channels.py"


def _load_selector():
    spec = importlib.util.spec_from_file_location("confined_v2_fixed_channel_selector", SELECTOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def catalog():
    return build_channel_catalog()


def test_catalog_is_real_full_airy_and_identity_tracks_geometry():
    catalog = globals()["catalog"]()
    assert catalog.directions_sensor.shape == (86400, 3)
    assert catalog.near_mask.shape == (86400,)
    assert catalog.endpoints_base.shape == (86400, 3)
    assert np.isfinite(catalog.directions_sensor).all()
    assert np.isfinite(catalog.endpoints_base).all()
    assert np.allclose(np.linalg.norm(catalog.directions_sensor, axis=1), 1.0, atol=1e-12)
    assert np.allclose(catalog.directions_sensor[0], [1.0, 0.0, 0.0], atol=1e-12)
    assert catalog.identity["source_id_order"] == "azimuth*96+elevation"
    assert abs(catalog.identity["pose_t1"]["qpos"][2] - 0.4598590386845707) < 1e-12
    assert catalog.identity["classification"]["floor_world_z_m"] == 0.0
    assert catalog.identity["counts"]["near"] + catalog.identity["counts"]["far"] == 86400
    assert catalog.identity["counts"]["near_nominal_nohits"] >= 0
    assert len(catalog.identity["source_sha256"]["mjcf_sha256"]) == 64


def _write_selection(path: Path, catalog: ChannelCatalog, ids: np.ndarray):
    ids = np.sort(np.asarray(ids, dtype=np.int64))
    np.savez_compressed(
        path,
        source_ray_ids=ids,
        source_ray_directions_sensor=catalog.directions_sensor[ids],
        catalog_identity_json=np.asarray(json.dumps(catalog.identity, sort_keys=True)),
    )
    meta = {
        "schema": catalog.identity["schema"],
        "file": path.name,
        "npz_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "catalog_identity": catalog.identity,
        "selection": {"test": True},
    }
    path.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")


def test_selection_loads_sorted_ids_with_matching_directions(tmp_path):
    catalog = globals()["catalog"]()
    archive = tmp_path / "pick.npz"
    _write_selection(archive, catalog, np.array([95, 0, 1234], dtype=np.int64))
    ids, identity = load_sparse_selection(archive, catalog)
    assert ids.dtype == np.int64
    assert ids.tolist() == [0, 95, 1234]
    assert identity["catalog_identity"] == catalog.identity


def test_selection_rejects_changed_catalog_identity(tmp_path):
    catalog = globals()["catalog"]()
    archive = tmp_path / "pick.npz"
    _write_selection(archive, catalog, np.array([1, 2], dtype=np.int64))
    changed = ChannelCatalog(
        catalog.directions_sensor,
        catalog.near_mask,
        catalog.endpoints_base,
        {**catalog.identity, "pose_sha256": "0" * 64},
    )
    try:
        load_sparse_selection(archive, changed)
    except ValueError as exc:
        assert "different catalog identity" in str(exc)
    else:
        raise AssertionError("changed catalog identity was accepted")


def test_selection_rejects_tampered_directions(tmp_path):
    catalog = globals()["catalog"]()
    archive = tmp_path / "pick.npz"
    _write_selection(archive, catalog, np.array([1, 2], dtype=np.int64))
    with np.load(archive, allow_pickle=False) as z:
        payload = {key: z[key] for key in z.files}
    payload["source_ray_directions_sensor"] = payload["source_ray_directions_sensor"].copy()
    payload["source_ray_directions_sensor"][0, 0] += 1e-4
    np.savez_compressed(archive, **payload)
    # Hash validation catches tampering before the per-direction guard.
    try:
        load_sparse_selection(archive, catalog)
    except ValueError as exc:
        assert "hash mismatch" in str(exc)
    else:
        raise AssertionError("tampered NPZ was accepted")


def test_selection_requires_explicit_recipe_and_integer_sorted_ids(tmp_path):
    catalog = globals()["catalog"]()
    try:
        build_channel_catalog(theta_deg=float("nan"))
    except ValueError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("non-finite catalog parameter was accepted")
    index = tmp_path / "manifest.json"
    index.write_text(json.dumps({
        "schema": catalog.identity["schema"],
        "files": {"near_x4_far_x4": {}},
    }), encoding="utf-8")
    try:
        load_sparse_selection(index, catalog)
    except ValueError as exc:
        assert "specific near_xN_far_xN" in str(exc)
    else:
        raise AssertionError("multi-selection index silently chose a recipe")

    fractional = tmp_path / "fractional.npz"
    _write_selection(fractional, catalog, np.array([0, 1]))
    with np.load(fractional, allow_pickle=False) as z:
        payload = {key: z[key] for key in z.files}
    payload["source_ray_ids"] = payload["source_ray_ids"].astype(np.float64) + 0.25
    np.savez_compressed(fractional, **payload)
    meta = json.loads(fractional.with_suffix(".json").read_text(encoding="utf-8"))
    meta["npz_sha256"] = hashlib.sha256(fractional.read_bytes()).hexdigest()
    fractional.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")
    try:
        load_sparse_selection(fractional, catalog)
    except ValueError as exc:
        assert "integer dtype" in str(exc)
    else:
        raise AssertionError("fractional IDs were silently truncated")

    archive = tmp_path / "unsorted.npz"
    _write_selection(archive, catalog, np.array([0, 1]))
    with np.load(archive, allow_pickle=False) as z:
        payload = {key: z[key] for key in z.files}
    payload["source_ray_ids"] = payload["source_ray_ids"][::-1]
    payload["source_ray_directions_sensor"] = payload["source_ray_directions_sensor"][::-1]
    np.savez_compressed(archive, **payload)
    meta = json.loads(archive.with_suffix(".json").read_text(encoding="utf-8"))
    meta["npz_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")
    try:
        load_sparse_selection(archive, catalog)
    except ValueError as exc:
        assert "already sorted ascending" in str(exc)
    else:
        raise AssertionError("unsorted IDs were silently normalized")


def test_independent_fixed_channel_assets_are_cartesian_nested_prefixes():
    design = ROOT / "src/hexapod_mjlab/tasks/confined_locomotion_2/assets/fixed_channels/v1_independent"
    catalog = globals()["catalog"]()
    choices = {}
    for near_mult in (4, 5, 6, 8):
        for far_mult in (4, 5, 6, 8):
            name = f"near_x{near_mult}_far_x{far_mult}"
            path = design / f"{name}.npz"
            ids, identity = load_sparse_selection(path, catalog)
            assert identity["catalog_identity"] == catalog.identity
            with np.load(path, allow_pickle=False) as archive:
                near_ids = archive["designed_near_ids"]
                far_ids = archive["designed_far_ids"]
                assert np.array_equal(archive["source_ray_ids"], ids)
                assert np.array_equal(archive["source_ray_directions_sensor"], catalog.directions_sensor[ids])
            assert len(ids) == 256 * near_mult + 128 * far_mult
            assert np.all(catalog.near_mask[near_ids])
            assert np.all(~catalog.near_mask[far_ids])
            assert len(np.intersect1d(near_ids, far_ids)) == 0
            choices[(near_mult, far_mult)] = (near_ids, far_ids, ids)
    for small, large in ((4, 5), (5, 6), (6, 8)):
        for far_mult in (4, 5, 6, 8):
            assert np.array_equal(choices[(small, far_mult)][0], choices[(large, far_mult)][0][:256 * small])
        for near_mult in (4, 5, 6, 8):
            assert np.array_equal(choices[(near_mult, small)][1], choices[(near_mult, large)][1][:128 * small])
    for near_mult in (4, 5, 6, 8):
        for far_mult in (4, 5, 6, 8):
            near_ids, far_ids, ids = choices[(near_mult, far_mult)]
            assert np.array_equal(ids, np.sort(np.concatenate((near_ids, far_ids))))


def test_far_cli_is_mutually_exclusive_and_legacy_spelling_is_compatible(tmp_path):
    selector = _load_selector()
    legacy = selector.parse_args(["--output", str(tmp_path / "legacy"),
                                  "--near-multipliers", "4", "5", "--far-multiplier", "4"])
    assert legacy.far_multipliers == [4]
    legacy_default = selector.parse_args(["--output", str(tmp_path / "default")])
    assert legacy_default.far_multipliers == [4]
    try:
        selector.parse_args(["--output", str(tmp_path / "invalid"), "--far-multiplier", "4",
                             "--far-multipliers", "4", "5"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("mutually exclusive far multiplier arguments were accepted")

