"""CPU reference and padded-boundary checks for near-cloud FPS sampling."""

import torch

from hexapod_mjlab.tasks.confined_locomotion_2.sampling import (
    farthest_point_sampling_indices,
    farthest_point_sampling_indices_reference,
    farthest_point_sampling_masked,
    farthest_point_sampling_padded,
)


def _raises_value_error(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_fps_empty_and_short_groups_have_no_repeated_sources():
    points = torch.tensor([[1.0, 0, 0], [0, 2.0, 0], [0, 0, 3.0]])

    empty, empty_count = farthest_point_sampling_indices_reference(
        points, torch.zeros(3, dtype=torch.bool), 4
    )
    assert empty_count == 0
    assert empty.shape == (0,)
    assert empty.dtype == torch.long

    short, short_count = farthest_point_sampling_indices_reference(
        points, torch.tensor([True, False, True]), 5
    )
    assert short_count == 2
    assert short.tolist() == [2, 0]  # max-norm seed, then the only other member
    assert short.unique().numel() == short_count


def test_fps_duplicate_coordinates_are_deterministic_and_tie_safe():
    points = torch.tensor([
        [2.0, 0, 0],  # max-norm tie: earliest member seeds
        [2.0, 0, 0],
        [-2.0, 0, 0],
        [0.0, 0, 0],
    ])
    mask = torch.ones(4, dtype=torch.bool)
    expected = [0, 2, 3, 1]
    for sampler in (farthest_point_sampling_indices_reference,
                    farthest_point_sampling_indices):
        first, count = sampler(points, mask, 4)
        second, count_again = sampler(points, mask, 4)
        assert count == count_again == 4
        assert first.tolist() == expected
        assert torch.equal(first, second)
        assert first.unique().numel() == 4


def test_fps_noncontiguous_mask_returns_original_source_indices():
    points = torch.tensor([
        [90.0, 0, 0], [1.0, 0, 0], [80.0, 0, 0], [0, 2.0, 0],
        [70.0, 0, 0], [0, 0, 3.0],
    ])
    source_ids = torch.tensor([900, 41, 800, 17, 700, 63])
    mask = torch.tensor([False, True, False, True, False, True])
    for sampler in (farthest_point_sampling_indices_reference,
                    farthest_point_sampling_indices):
        indices, count = sampler(points, mask, 3)
        assert count == 3
        assert indices.tolist() == [5, 3, 1]
        # The sampler's indices address the original pool, so source provenance
        # remains a direct gather; no coordinate-based recovery is needed.
        assert source_ids[indices].tolist() == [63, 17, 41]
        assert bool(mask[indices].all())


def test_fps_padded_mixed_lengths_use_minus_one_tail_padding():
    points = torch.tensor([
        [[10.0, 0, 0], [20.0, 0, 0], [30.0, 0, 0], [40.0, 0, 0]],
        [[1.0, 0, 0], [0, 2.0, 0], [99.0, 0, 0], [99.0, 0, 0]],
        [[1.0, 0, 0], [0, 2.0, 0], [0, 0, 3.0], [-4.0, 0, 0]],
    ])
    lengths = torch.tensor([0, 2, 4], dtype=torch.long)
    selected = farthest_point_sampling_padded(points, lengths, 3)
    assert selected.shape == (3, 3)
    assert selected.dtype == torch.long
    assert selected[0].tolist() == [-1, -1, -1]
    assert selected[1, :2].tolist() == [1, 0]
    assert selected[1, 2].item() == -1
    assert selected[1, :2].unique().numel() == 2
    assert bool(((selected[1, :2] >= 0) & (selected[1, :2] < 2)).all())
    assert selected[2].unique().numel() == 3
    assert bool(((selected[2] >= 0) & (selected[2] < 4)).all())


def test_fps_masked_empty_shapes_and_holes_exclude_invalid_nan():
    empty_batch = farthest_point_sampling_masked(
        torch.empty((0, 5, 3), dtype=torch.float32),
        torch.empty((0, 5), dtype=torch.bool), 3,
    )
    assert empty_batch.shape == (0, 3)
    empty_points = farthest_point_sampling_masked(
        torch.empty((2, 0, 3), dtype=torch.float32),
        torch.empty((2, 0), dtype=torch.bool), 3,
    )
    assert empty_points.shape == (2, 3)
    assert empty_points.tolist() == [[-1, -1, -1], [-1, -1, -1]]

    points = torch.tensor([
        [[float("nan"), 0, 0], [1.0, 0, 0], [0, 2.0, 0],
         [float("nan"), 0, 0], [0, 0, 3.0]],
    ])
    mask = torch.tensor([[False, True, False, False, True]])
    result = farthest_point_sampling_masked(points, mask, 3)
    assert result.tolist() == [[4, 1, -1]]
    assert bool((result[0, :2] >= 0).all())
    assert bool(mask[0, result[0, :2]].all())


def test_fps_invalid_dtype_device_shape_and_k_raise_value_error():
    points = torch.zeros((4, 3), dtype=torch.float32)
    mask = torch.ones(4, dtype=torch.bool)
    _raises_value_error(farthest_point_sampling_indices_reference,
                        points.to(torch.int64), mask, 2)
    _raises_value_error(farthest_point_sampling_indices_reference,
                        points, mask.to(torch.int64), 2)
    _raises_value_error(farthest_point_sampling_indices_reference,
                        points, torch.ones(3, dtype=torch.bool), 2)
    _raises_value_error(farthest_point_sampling_indices_reference,
                        torch.zeros((4, 2)), mask, 2)
    _raises_value_error(farthest_point_sampling_indices_reference,
                        points, mask, 0)
    _raises_value_error(farthest_point_sampling_indices_reference,
                        points, mask, True)

    # Meta tensors let the CPU suite verify device rejection without requiring
    # CUDA hardware or allocating accelerator memory.
    _raises_value_error(farthest_point_sampling_indices_reference,
                        torch.empty((4, 3), device="meta"), mask, 2)
    _raises_value_error(farthest_point_sampling_indices_reference,
                        points, torch.ones(4, dtype=torch.bool, device="meta"), 2)

    batch_points = torch.zeros((2, 4, 3), dtype=torch.float32)
    lengths = torch.tensor([4, 2], dtype=torch.long)
    _raises_value_error(farthest_point_sampling_padded,
                        batch_points.to(torch.int32), lengths, 2)
    _raises_value_error(farthest_point_sampling_padded,
                        batch_points, lengths.to(torch.float32), 2)
    _raises_value_error(farthest_point_sampling_padded,
                        batch_points, torch.tensor([4]), 2)
    _raises_value_error(farthest_point_sampling_padded,
                        torch.zeros((2, 4, 2)), lengths, 2)
    _raises_value_error(farthest_point_sampling_padded,
                        batch_points, lengths, 0)
    _raises_value_error(farthest_point_sampling_padded,
                        batch_points, lengths, True)
    _raises_value_error(farthest_point_sampling_padded,
                        batch_points, lengths.to("meta"), 2)


def test_checkpoint_identity_rejects_old_sampling_semantics(tmp_path):
    from hexapod_mjlab.tasks.confined_locomotion_2.rl.checkpoint import (
        SCHEMA,
        TASK_ID,
        architecture_identity,
        read_checkpoint,
    )
    from hexapod_mjlab.tasks.confined_locomotion_2.sampling import sampling_identity

    current = architecture_identity()
    expected = {
        "version": "xyz_fps/v1",
        "metric": "base_xyz_m",
        "seed": "max_norm_first",
        "ties": "first_pool_index",
        "near_k": 256,
        "far_k": 128,
        "without_replacement": True,
    }
    assert sampling_identity() == expected
    assert current["sampling"] == expected

    # An old same-task checkpoint without sampler semantics must not silently
    # pass weights-only play loading or full resume architecture validation.
    legacy = dict(current)
    del legacy["sampling"]
    path = tmp_path / "legacy-angular-sampler.pt"
    torch.save({
        "schema": SCHEMA,
        "task": TASK_ID,
        "kind": "full",
        "resumable": True,
        "identity": {"architecture": legacy},
    }, path)
    _raises_value_error(read_checkpoint, path)


def test_checkpoint_sampling_identity_selects_supported_mode(tmp_path):
    from hexapod_mjlab.tasks.confined_locomotion_2.rl.checkpoint import (
        SCHEMA, TASK_ID, architecture_identity, read_checkpoint,
    )

    path = tmp_path / "hit-only.pt"
    torch.save({
        "schema": SCHEMA,
        "task": TASK_ID,
        "kind": "full",
        "resumable": True,
        "identity": {"architecture": architecture_identity("hit_only_10m")},
    }, path)
    assert read_checkpoint(path)["identity"]["architecture"] == architecture_identity("hit_only_10m")
    _raises_value_error(read_checkpoint, path, expected_far_sampling="all_endpoints")
