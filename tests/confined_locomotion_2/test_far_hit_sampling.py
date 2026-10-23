"""Production far-candidate gates and shortage semantics, independent of CUDA."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from hexapod_mjlab.tasks.confined_locomotion_2.sampling import far_candidate_mask, sampling_identity
from hexapod_mjlab.tasks.confined_locomotion_2.scanner import ConfinedLidarCfg, ConfinedScanPipeline


def test_hit10_gate_uses_original_hit_distance_and_gap_validity():
    raw = torch.tensor([9.99, 10.0, 10.001, 15., 9., 0.])
    # Last slot represents a legitimate zero-distance hit under kernel rules;
    # no new minimum physical range is imposed by this candidate gate.
    no_hit = torch.tensor([False, False, False, True, False, False])
    fixed_far_valid = torch.tensor([True, True, True, True, False, True])
    legacy = far_candidate_mask(fixed_far_valid, raw, no_hit, "all_endpoints")
    assert torch.equal(legacy, fixed_far_valid)
    eligible = far_candidate_mask(fixed_far_valid, raw, no_hit, "hit_only_10m")
    assert eligible.tolist() == [True, True, False, False, False, True]
    # Neither noisy XYZ nor consumed ranges are inputs: e.g. raw 10.001m
    # remains excluded even if negative noise takes its observed range <10m.
    assert sampling_identity("hit_only_10m") != sampling_identity()
    cfg = ConfinedLidarCfg(far_sampling="hit_only_10m")
    assert cfg.max_range_m == 15.0 and cfg.near_points == 256
    try:
        ConfinedLidarCfg(far_sampling="unknown")
    except ValueError:
        pass
    else:
        raise AssertionError("unsupported far candidate semantics accepted")
    try:
        ConfinedLidarCfg(far_sampling="hit_only_10m", max_range_m=10.0)
    except ValueError:
        pass
    else:
        raise AssertionError("candidate cutoff changed physical raycast range")


def test_production_batched_fps_keeps_shortage_slots_and_near_unchanged():
    torch.manual_seed(7)
    points = torch.randn(400, 3)
    near = torch.arange(400) < 256
    frames = []
    for count in (0, 31, 60, 128):
        far = (torch.arange(400) >= 256) & (torch.arange(400) < 256 + count)
        frames.append(SimpleNamespace(
            pool_points_base=points, near_mask=near, far_mask=far,
            pool_raw_indices=torch.arange(400) * 2, pool_angle_keys=torch.arange(400).float(),
            near_angle_keys=torch.zeros(256), far_angle_keys=torch.zeros(128),
        ))
    pipe = object.__new__(ConfinedScanPipeline)
    pipe.cfg = ConfinedLidarCfg(far_sampling="hit_only_10m")
    pipe._sample_frames(frames)
    for frame, count in zip(frames, (0, 31, 60, 128)):
        assert frame.far_valid.sum() == count
        assert torch.all(frame.far_points_base[~frame.far_valid] == 0)
        assert torch.all(frame.far_indices[~frame.far_valid] == -1)
        assert len(torch.unique(frame.far_indices[frame.far_valid])) == count
        assert torch.equal(frame.near_indices, frames[0].near_indices)
        assert torch.equal(frame.near_points_base, frames[0].near_points_base)


def test_hit10_formal_selection_is_the_frozen_6x10_experiment():
    from hexapod_mjlab.tasks.confined_locomotion_2.channels import build_channel_catalog, load_sparse_selection
    root = Path(__file__).resolve().parents[2]
    current = root / "src/hexapod_mjlab/tasks/confined_locomotion_2/assets/fixed_channels/v1_hit10/near_x6_far_x10.npz"
    frozen = root / "logs/verification/confined_locomotion_2/sparse_rays/stage5_hit_only_select_20261023/near_x6_far_x10.npz"
    catalog = build_channel_catalog()
    ids, identity = load_sparse_selection(current, catalog)
    assert len(ids) == 2816 and catalog.near_mask[ids].sum() == 1536
    assert identity["selection"]["far_count"] == 1280
    # Compare frozen experiment if its optional local evidence is available.
    if frozen.is_file():
        assert current.read_bytes() == frozen.read_bytes()
    with np.load(current, allow_pickle=False) as data:
        near_old = root / "src/hexapod_mjlab/tasks/confined_locomotion_2/assets/fixed_channels/v1_independent/near_x6_far_x8.npz"
        with np.load(near_old, allow_pickle=False) as old:
            assert np.array_equal(data["designed_near_ids"], old["designed_near_ids"])
            assert np.array_equal(data["designed_far_ids"][:1024], old["designed_far_ids"])
