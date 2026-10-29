from dataclasses import asdict

from hexapod_mjlab.tasks.confined_locomotion_2.rl.entry import build_formal_train_config
from hexapod_mjlab.tasks.confined_locomotion_2.rl.runner import resolve_recipe


def test_training_defaults_use_published_sparse_hit10_selection():
    from pathlib import Path
    from hexapod_mjlab.tasks.confined_locomotion_2.rl.terrain_entry import build_terrain_train_config
    from hexapod_mjlab.tasks.confined_locomotion_2.rl.tripod_entry import build_train_config
    from hexapod_mjlab.tasks.confined_locomotion_2.components.lidar.channels import (
        build_channel_catalog, load_sparse_selection,
    )
    kwargs = dict(seed=17, num_envs=2, max_iterations=1, engineering_validation=True)
    configs = (build_formal_train_config(**kwargs),
               build_terrain_train_config(**kwargs, height_loss_weight=.1), build_train_config(**kwargs))
    for cfg in configs:
        lidar = cfg.agent.lidar_config
        assert lidar["raycast_mode"] == "sparse" and lidar["far_sampling"] == "hit_only_10m"
        assert Path(lidar["sparse_selection_path"]).is_file()
        assert cfg.agent.point_replay_chunk == 1024
    selection = configs[0].agent.lidar_config["sparse_selection_path"]
    ids, _ = load_sparse_selection(selection, build_channel_catalog())
    assert len(ids) == 2816
    full = build_formal_train_config(**kwargs, raycast_mode="full", far_sampling="all_endpoints",
                                     point_replay_chunk=128)
    recipe = resolve_recipe(asdict(full.agent))
    assert recipe["lidar_cfg"].raycast_mode == "full" and recipe["lidar_cfg"].sparse_selection_path is None
    assert recipe["lidar_cfg"].far_sampling == "all_endpoints" and recipe["point_replay_chunk"] == 128


def test_sparse_selection_is_serialized_through_formal_recipe(tmp_path):
    selection = tmp_path / "selected-rays.npz"
    selection.write_bytes(b"configuration path fixture")
    cfg = build_formal_train_config(
        seed=17,
        num_envs=2,
        max_iterations=1,
        engineering_validation=True,
        raycast_mode="sparse",
        sparse_selection_path=str(selection),
    )
    recipe = resolve_recipe(asdict(cfg.agent))
    assert recipe["lidar_cfg"].raycast_mode == "sparse"
    assert recipe["lidar_cfg"].sparse_selection_path == str(selection)


def test_raycast_mode_and_selection_path_contract():
    invalid = (("full", "selection.npz"), ("sparse", ""), ("adaptive", None))
    for raycast_mode, path in invalid:
        try:
            build_formal_train_config(
                seed=17,
                num_envs=2,
                max_iterations=1,
                engineering_validation=True,
                raycast_mode=raycast_mode,
                sparse_selection_path=path,
            )
        except (TypeError, ValueError):
            continue
        raise AssertionError(f"invalid raycast config accepted: {raycast_mode=}, {path=}")
