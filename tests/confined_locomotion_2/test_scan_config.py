from dataclasses import asdict

from hexapod_mjlab.tasks.confined_locomotion_2.rl.entry import build_formal_train_config
from hexapod_mjlab.tasks.confined_locomotion_2.rl.runner import resolve_recipe


def test_formal_recipe_defaults_to_full_raycast_mode():
    cfg = build_formal_train_config(
        seed=17, num_envs=2, max_iterations=1, engineering_validation=True
    )
    recipe = resolve_recipe(asdict(cfg.agent))
    assert recipe["lidar_cfg"].raycast_mode == "full"
    assert recipe["lidar_cfg"].sparse_selection_path is None


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
    invalid = (("full", "selection.npz"), ("sparse", None), ("adaptive", None))
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
