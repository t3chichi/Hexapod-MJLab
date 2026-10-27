"""Focused checks for batched ScanWindow writes."""

from __future__ import annotations

import torch

from hexapod_mjlab.rl.confined_locomotion_2 import history
from hexapod_mjlab.rl.confined_locomotion_2.contracts import (
    FAR_POINTS_PER_SCAN,
    NEAR_POINTS,
    ContractError,
)
from hexapod_mjlab.rl.confined_locomotion_2.history import (
    EpisodeMismatchError,
    ScanWindow,
    StaleScanError,
    TimeOrderError,
)


def _blocks(markers: list[float]):
    batch = len(markers)
    near = torch.zeros(batch, NEAR_POINTS, 3)
    far = torch.zeros(batch, FAR_POINTS_PER_SCAN, 3)
    near_valid = torch.zeros(batch, NEAR_POINTS, dtype=torch.bool)
    far_valid = torch.zeros(batch, FAR_POINTS_PER_SCAN, dtype=torch.bool)
    near[:, 0, 0] = torch.tensor(markers)
    far[:, 0, 0] = torch.tensor(markers)
    near_valid[:, 0] = True
    far_valid[:, 0] = True
    return near, near_valid, far, far_valid


def _write(window: ScanWindow, envs, episodes, scans, times, markers, **metadata):
    near, near_valid, far, far_valid = _blocks(markers)
    window.write(
        env_ids=envs,
        episode_ids=episodes,
        scan_ids=scans,
        times=times,
        near_points=near,
        near_valid=near_valid,
        far_points=far,
        far_valid=far_valid,
        **metadata,
    )


def _assert_state_equal(left, right):
    assert left["batch_size"] == right["batch_size"]
    assert left["seen_episodes"] == right["seen_episodes"]
    assert left["tensors"].keys() == right["tensors"].keys()
    for key in left["tensors"]:
        assert torch.equal(left["tensors"][key], right["tensors"][key]), key


def test_write_batch_preserves_ring_order_for_multiple_slots():
    # Exercise the ring arithmetic beyond today's one-slot configuration.
    previous_slots = history.FAR_WINDOW_SLOTS
    history.FAR_WINDOW_SLOTS = 3
    try:
        window = ScanWindow(3, episode_ids=[0, 0, 0])
        _write(window, [2, 0], [0, 0], [1, 1], [1.0, 1.0], [12.0, 10.0])
        _write(window, [2], [0], [2], [2.0], [22.0])
        _write(window, [2], [0], [3], [3.0], [32.0])
        _write(window, [2], [0], [4], [4.0], [42.0])

        assert window._far_count.tolist() == [1, 0, 3]
        assert window._far_pos.tolist() == [1, 0, 1]
        # Physical ring storage wraps oldest slot 0 after scans 1..4.
        assert window._far_scan_ids[2].tolist() == [4, 2, 3]
        assert window._far_times[2].tolist() == [4.0, 2.0, 3.0]
        assert window._far_points[2, :, 0, 0].tolist() == [42.0, 22.0, 32.0]
        assert window._near_scan_ids.tolist() == [1, -1, 4]
    finally:
        history.FAR_WINDOW_SLOTS = previous_slots


def test_write_preserves_input_row_then_identity_category_error_order_atomically():
    window = ScanWindow(3, episode_ids=[7, 8, 9])
    _write(window, [0, 1, 2], [7, 8, 9], [4, 5, 6], [4.0, 5.0, 6.0], [1, 2, 3])
    before = window.state_dict()

    # Row 1's episode error wins over row 2's scan/time errors.
    try:
        _write(window, [2, 1], [9, 80], [7, 5], [7.0, 5.0], [30, 20])
    except EpisodeMismatchError as exc:
        assert "env 1" in str(exc)
    else:
        raise AssertionError("expected episode mismatch")
    _assert_state_equal(before, window.state_dict())

    # Input row order wins across categories: row 0 time precedes row 1 episode.
    try:
        _write(window, [0, 1], [7, 80], [5, 6], [4.0, 6.0], [10, 20])
    except TimeOrderError as exc:
        assert "env 0" in str(exc)
    else:
        raise AssertionError("expected time order error")
    _assert_state_equal(before, window.state_dict())

    # Within a row, stale scan precedes time order.
    try:
        _write(window, [0], [7], [4], [3.0], [10])
    except StaleScanError as exc:
        assert "env 0" in str(exc)
    else:
        raise AssertionError("expected stale scan error")
    _assert_state_equal(before, window.state_dict())


def test_write_clears_optional_metadata_and_keeps_returned_snapshots_owned():
    window = ScanWindow(1, episode_ids=[3])
    near, near_valid, far, far_valid = _blocks([1.0])
    near_keys = torch.zeros(1, NEAR_POINTS)
    far_keys = torch.zeros(1, FAR_POINTS_PER_SCAN)
    near_keys[:, 0] = 0.25
    far_keys[:, 0] = 0.5
    near_dirs = torch.zeros(1, NEAR_POINTS, 3)
    far_dirs = torch.zeros(1, FAR_POINTS_PER_SCAN, 3)
    near_dirs[:, 0, 0] = 1.0
    far_dirs[:, 0, 0] = 1.0
    near_indices = torch.full((1, NEAR_POINTS), -1, dtype=torch.int64)
    far_indices = torch.full((1, FAR_POINTS_PER_SCAN), -1, dtype=torch.int64)
    near_indices[:, 0] = 5
    far_indices[:, 0] = 7
    window.write(
        env_ids=[0], episode_ids=[3], scan_ids=[1], times=[1.0],
        near_points=near, near_valid=near_valid, far_points=far, far_valid=far_valid,
        near_angle_keys=near_keys, far_angle_keys=far_keys,
        near_source_dirs_sensor=near_dirs, far_source_dirs_sensor=far_dirs,
        near_source_indices=near_indices, far_source_indices=far_indices,
    )
    snapshot = window.read()
    old_keys = window.read_angle_keys()
    old_sources = window.read_source_directions()
    assert old_keys is not None and old_sources is not None

    _write(window, [0], [3], [2], [2.0], [2.0])
    new_keys = window.read_angle_keys()
    new_sources = window.read_source_directions()
    assert new_keys is None
    assert new_sources is not None
    assert not bool(new_sources.near_present.any())
    assert not bool(new_sources.far_present.any())
    assert torch.equal(new_sources.near_indices, torch.full_like(near_indices, -1))
    assert torch.equal(new_sources.far_indices[:, 0], torch.full_like(far_indices, -1))
    # Later writes neither mutate old snapshots nor alias their backing storage.
    assert snapshot.near_points[0, 0, 0].item() == 1.0
    assert old_keys.near[0, 0].item() == 0.25
    assert old_sources.near_indices[0, 0].item() == 5


def test_write_rejects_bad_optional_metadata_before_mutating():
    window = ScanWindow(1, episode_ids=[0])
    before = window.state_dict()
    near, near_valid, far, far_valid = _blocks([1.0])
    try:
        window.write(
            env_ids=[0], episode_ids=[0], scan_ids=[1], times=[1.0],
            near_points=near, near_valid=near_valid, far_points=far, far_valid=far_valid,
            near_angle_keys=torch.zeros(1, NEAR_POINTS),
        )
    except ContractError:
        pass
    else:
        raise AssertionError("expected paired angle-key validation")
    _assert_state_equal(before, window.state_dict())
