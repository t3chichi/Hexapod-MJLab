"""Registration and launch-isolation contracts."""
from dataclasses import asdict

from mjlab.tasks.registry import list_tasks
from hexapod_mjlab.tasks.confined_locomotion_2.rl.tripod_entry import build_train_config
from hexapod_mjlab.tasks.confined_locomotion_2.tripod_env_cfg import TASK_ID
from hexapod_mjlab.tasks.confined_locomotion_2.config.tripod2_low_config import TripodRlCfg
from hexapod_mjlab.tasks.confined_locomotion_2.rl.tripod_runner import validate_recipe
from hexapod_mjlab.tasks.confined_locomotion_2.config.flat_baseline_config import confined_locomotion_formal_rl_cfg
from hexapod_mjlab.tasks.confined_locomotion_2.config.terrain_memory_config import make_training_env_cfg as terrain_cfg


def test_registration_and_explicit_budget_do_not_rewrite_old_recipes():
    assert TASK_ID in list_tasks()
    original = asdict(confined_locomotion_formal_rl_cfg(seed=1, num_envs=2, max_iterations=1))
    cfg = build_train_config(seed=1, num_envs=2, max_iterations=1, engineering_validation=True)
    assert cfg.agent.experiment_name == "confined_locomotion_flat_tripod2_low"
    assert cfg.env.scene.num_envs == 2
    assert cfg.agent.lidar_config["raycast_mode"] == "sparse"
    assert cfg.agent.lidar_config["far_sampling"] == "hit_only_10m"
    old = build_train_config(seed=1, num_envs=2, max_iterations=1,
                            raycast_mode="full", far_sampling="all_endpoints", point_replay_chunk=128)
    assert old.agent.lidar_config["sparse_selection_path"] is None
    assert old.agent.lidar_config["far_sampling"] == "all_endpoints"
    assert old.agent.point_replay_chunk == 128
    assert all(value == 0. for value in cfg.env.scene.entities["robot"].init_state.joint_pos.values())
    assert asdict(confined_locomotion_formal_rl_cfg(seed=1, num_envs=2, max_iterations=1)) == original
    assert set(terrain_cfg().terminations) == {"fall_low_height", "fall_tilt_sustained", "fall_flipped", "episode_timeout", "terrain_boundary"}


def test_registration_defaults_and_oversized_engineering_budget_are_rejected():
    for kwargs in ({"num_envs": 3, "max_iterations": 1}, {"num_envs": 2, "max_iterations": 2}):
        try:
            build_train_config(seed=1, engineering_validation=True, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("oversized engineering budget accepted")
    try:
        validate_recipe(asdict(TripodRlCfg()))
    except ValueError:
        pass
    else:
        raise AssertionError("registration must not silently authorize training")
    for chunk in (0, -1, True, 1.5):
        try:
            build_train_config(seed=1, num_envs=2, max_iterations=1, point_replay_chunk=chunk)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid point replay chunk accepted")
