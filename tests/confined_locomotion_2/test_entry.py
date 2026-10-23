from dataclasses import asdict

from hexapod_mjlab.tasks.confined_locomotion_2.rl_cfg import (
    confined_locomotion_rl_cfg, confined_locomotion_formal_rl_cfg,
)
from hexapod_mjlab.tasks.confined_locomotion_2.rl.runner import resolve_recipe
from hexapod_mjlab.tasks.confined_locomotion_2.rl.entry import build_formal_train_config


def test_registry_keeps_original_and_new_task_separate_and_defaults_guarded():
    from mjlab.tasks.registry import list_tasks
    names = list_tasks()
    assert "El-4090-Confined-Locomotion-Flat" in names
    assert "El-4090-Confined-Locomotion-Flat_2" in names
    try:
        resolve_recipe(asdict(confined_locomotion_rl_cfg()))
    except ValueError:
        pass
    else:
        raise AssertionError("registration must not silently authorize learning")


def test_formal_recipe_retains_24_steps_mirror_and_perturbations_without_grid():
    cfg = build_formal_train_config(seed=9, num_envs=2, max_iterations=1,
                                    engineering_validation=True)
    resolved = resolve_recipe(asdict(cfg.agent))
    assert cfg.env.scene.num_envs == 2
    assert resolved["symmetry"].enabled
    assert resolved["ppo"].rollout_steps == 24
    assert resolved["ppo"].epochs == 5 and resolved["ppo"].minibatches == 4
    assert not hasattr(cfg.agent, "ground_grid")
    assert not hasattr(resolved["ppo"], "ground_loss_coeff")
    hit10 = build_formal_train_config(
        seed=9, num_envs=2, max_iterations=1, engineering_validation=True,
        raycast_mode="sparse", sparse_selection_path="explicit-selection.npz",
        far_sampling="hit_only_10m",
    )
    hit_recipe = resolve_recipe(asdict(hit10.agent))
    assert hit_recipe["lidar_cfg"].far_sampling == "hit_only_10m"
    assert hit_recipe["lidar_cfg"].max_range_m == 15.0


def test_finite_engineering_budget_is_checked_before_environment_construction():
    try:
        build_formal_train_config(seed=9, num_envs=2, max_iterations=2,
                                  engineering_validation=True)
    except ValueError:
        pass
    else:
        raise AssertionError("engineering path may not start a long run")


def test_close_failure_does_not_report_environment_closed():
    from types import SimpleNamespace
    from hexapod_mjlab.tasks.confined_locomotion_2.rl.runner import ConfinedLocomotionRunner
    runner = object.__new__(ConfinedLocomotionRunner)
    runner.closed, runner.close_error, runner.phase = False, None, "ready"
    runner.runtime = SimpleNamespace(close=lambda: None)
    runner._write_summary = lambda: None
    def fail():
        raise OSError("declared close fixture")
    runner._official_close = fail
    try:
        runner._close_official_environment()
    except OSError:
        pass
    else:
        raise AssertionError("close failure must propagate")
    assert runner.closed is False and runner.phase == "faulted"
    assert "declared close fixture" in runner.close_error
