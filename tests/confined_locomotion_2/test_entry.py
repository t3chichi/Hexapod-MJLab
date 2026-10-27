import copy
from dataclasses import asdict

from hexapod_mjlab.tasks.confined_locomotion_2.config.flat_baseline import (
    confined_locomotion_rl_cfg, confined_locomotion_formal_rl_cfg,
)
from hexapod_mjlab.tasks.confined_locomotion_2.flat_env_cfg import TASK_ID
from hexapod_mjlab.tasks.confined_locomotion_2.rl.runner import resolve_recipe
from hexapod_mjlab.tasks.confined_locomotion_2.rl.entry import build_formal_train_config


def _checkpoint_identity():
    return {
        "architecture": {"task": "fixture", "control_hz": 50},
        "seed": 1,
        "num_envs": 2,
        "scan_identity": {"catalog": "fixture", "mode": "full"},
        "compiled_model": {
            "dims": {"nq": 25, "nu": 18},
            "joint_names": ["RF_HAA"],
            "actuator_names": ["RF_HAA"],
            "actuator_gear": [1.0],
            "q_ref": [0.0],
            "options": {"timestep": 0.005},
            "compiled_binary_sha256": "binary",
        },
        "recipe": {
            "ppo": {"learning_rate": 0.001},
            "symmetry": {"enabled": True},
            "perturbations": {"mode": "all"},
            "env_semantics": {
                "action": {"stiffness": 130.0, "damping": 2.0},
                "rewards": {"track_lin_vel": 5.0},
                "terminations": {"names": ["fall"], "time_out": {"fall": False}},
                "sim": {"timestep": 0.005, "decimation": 4},
            },
        },
    }


def test_identity_compares_optional_semantics_without_legacy_breakage():
    from hexapod_mjlab.tasks.confined_locomotion_2.rl.checkpoint import compare_identity

    current = _checkpoint_identity()
    compare_identity(current, copy.deepcopy(current))

    changed = copy.deepcopy(current)
    changed["recipe"]["env_semantics"]["action"]["stiffness"] = 131.0
    try:
        compare_identity(current, changed)
    except ValueError as error:
        assert "training recipe" in str(error)
    else:
        raise AssertionError("action semantics change accepted")

    legacy = copy.deepcopy(current)
    legacy["recipe"].pop("env_semantics")
    compare_identity(current, legacy)

    low_memory = copy.deepcopy(current)
    low_memory["recipe"]["execution_mode"] = "low_memory"
    speed = copy.deepcopy(low_memory)
    speed["recipe"]["execution_mode"] = "speed"
    compare_identity(low_memory, speed)

    actuator = copy.deepcopy(current)
    actuator["recipe"]["actuator_sha256"] = "asset-a"
    compare_identity(actuator, copy.deepcopy(actuator))
    changed_actuator = copy.deepcopy(actuator)
    changed_actuator["recipe"]["actuator_sha256"] = "asset-b"
    try:
        compare_identity(actuator, changed_actuator)
    except ValueError as error:
        assert "actuator_sha256" in str(error)
    else:
        raise AssertionError("actuator identity change accepted")

    changed_model = copy.deepcopy(current)
    changed_model["compiled_model"]["q_ref"] = [1.0]
    try:
        compare_identity(current, changed_model)
    except ValueError as error:
        assert "compiled-model" in str(error)
    else:
        raise AssertionError("model structure change accepted")


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


def test_terrain_memory_registration_is_real_independent_and_guarded():
    from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
    from hexapod_mjlab.tasks.confined_locomotion_2.config import terrain_memory

    assert terrain_memory.TASK_ID != TASK_ID
    assert terrain_memory.TASK_ID in list_tasks()
    terrain = load_env_cfg(terrain_memory.TASK_ID)
    assert terrain.scene.terrain.terrain_type == "generator"
    assert terrain.rewards and terrain.terminations
    assert load_runner_cls(terrain_memory.TASK_ID) is load_runner_cls(TASK_ID)
    try:
        resolve_recipe(asdict(load_rl_cfg(terrain_memory.TASK_ID)))
    except ValueError:
        pass
    else:
        raise AssertionError("registration defaults must not authorize learning")


def test_flat_environment_factories_return_independent_configs():
    from hexapod_mjlab.tasks.confined_locomotion_2.flat_env_cfg import (
        make_flat_env_cfg, make_training_env_cfg,
    )

    generic_a, generic_b = make_flat_env_cfg(num_envs=1), make_flat_env_cfg(num_envs=1)
    assert generic_a is not generic_b
    assert generic_a.scene is not generic_b.scene
    assert generic_a.observations is not generic_b.observations
    assert generic_a.actions is not generic_b.actions
    assert generic_a.events is not generic_b.events
    generic_a.observations["policy"].terms.clear()
    assert generic_b.observations["policy"].terms

    train_a, train_b = make_training_env_cfg(num_envs=1), make_training_env_cfg(num_envs=1)
    assert train_a is not train_b and train_a.rewards is not train_b.rewards
    train_a.rewards.clear()
    assert train_b.rewards


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


def test_execution_modes_reach_both_task_recipes_and_reject_invalid_values():
    from hexapod_mjlab.tasks.confined_locomotion_2.rl.terrain_entry import build_terrain_train_config
    assert confined_locomotion_rl_cfg().execution_mode == "low_memory"
    for mode in ("low_memory", "speed"):
        flat = build_formal_train_config(seed=9, num_envs=2, max_iterations=1, execution_mode=mode)
        terrain = build_terrain_train_config(seed=9, num_envs=2, max_iterations=1,
                                            height_loss_weight=1., execution_mode=mode)
        for cfg in (flat, terrain):
            resolved = resolve_recipe(asdict(cfg.agent))
            assert resolved["execution_mode"] == mode
            assert resolved["ppo"].epochs == 5 and resolved["ppo"].minibatches == 4
    for mode in ("auto", "fast", None, True):
        try:
            build_formal_train_config(seed=9, num_envs=2, max_iterations=1, execution_mode=mode)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid execution mode accepted")
