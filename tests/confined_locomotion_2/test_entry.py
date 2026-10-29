import copy
from dataclasses import asdict

import torch

from hexapod_mjlab.tasks.confined_locomotion_2.config.flat_baseline_config import (
    confined_locomotion_rl_cfg, confined_locomotion_formal_rl_cfg,
)
from hexapod_mjlab.tasks.confined_locomotion_2.flat_env_cfg import TASK_ID
from hexapod_mjlab.tasks.confined_locomotion_2.rl.runner import resolve_recipe
from hexapod_mjlab.tasks.confined_locomotion_2.rl.entry import build_formal_train_config
from hexapod_mjlab.rl.confined_locomotion_2.ppo import ppo_semantics


def _checkpoint_identity():
    return {
        "architecture": {"task": "fixture", "control_hz": 50},
        "seed": 1,
        "num_envs": 2,
        "scan_identity": {"catalog": "fixture", "mode": "full"},
        "compiled_model": {
            "physics": {"schema": "hexapod_mjlab.model_physics/v1", "fields": {"body_mass": "mass"}},
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
            "ppo_semantics": ppo_semantics(),
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


def test_v2_initial_exploration_std_is_one_without_changing_v1_default():
    from hexapod_mjlab.rl.confined_locomotion.distribution import IndependentDiagonalNormalProvider
    v1 = IndependentDiagonalNormalProvider(generator_seed=1)
    from hexapod_mjlab.rl.confined_locomotion_2.distribution import DirectStdNormalProvider
    v2 = DirectStdNormalProvider(generator_seed=1)
    assert torch.equal(v1.log_std.detach(), torch.full((18,), torch.log(torch.tensor(0.5)).item()))
    assert torch.equal(v2.std.detach(), torch.ones(18))
    assert not hasattr(v2, "log_std")
    assert v1.distribution_identity()["initial_std"] == 0.5
    assert v2.distribution_identity()["initial_std"] == 1.0
    assert v2.distribution_identity()["parameterization"] == "direct_std"


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

    with_distribution = copy.deepcopy(current)
    with_distribution["recipe"]["distribution"] = {"initial_std": 1.0, "parameterization": "direct_std"}
    compare_identity(with_distribution, copy.deepcopy(with_distribution))
    changed_distribution = copy.deepcopy(with_distribution)
    changed_distribution["recipe"]["distribution"]["initial_std"] = 0.5
    try:
        compare_identity(with_distribution, changed_distribution)
    except ValueError as error:
        assert "distribution" in str(error)
    else:
        raise AssertionError("changed exploration recipe accepted")
    compare_identity(with_distribution, current)  # historical checkpoints omit distribution

    changed_model = copy.deepcopy(current)
    changed_model["compiled_model"]["q_ref"] = [1.0]
    try:
        compare_identity(current, changed_model)
    except ValueError as error:
        assert "compiled-model" in str(error)
    else:
        raise AssertionError("model structure change accepted")


def test_full_resume_rejects_unknown_or_changed_physics_before_any_apply():
    from types import SimpleNamespace
    from unittest.mock import Mock, patch
    from hexapod_mjlab.tasks.confined_locomotion_2.rl import checkpoint as cp

    current = _checkpoint_identity()
    runner = SimpleNamespace(phase="fresh", _active_buffer=None, task_id=TASK_ID,
                             runtime=SimpleNamespace(lidar_cfg=SimpleNamespace(far_sampling="all_endpoints")),
                             network=Mock())
    for case in ("mass", "missing_physics", "missing_ppo_semantics"):
        saved = copy.deepcopy(current)
        if case == "missing_ppo_semantics":
            saved["recipe"].pop("ppo_semantics")
        elif case == "missing_physics":
            saved["compiled_model"].pop("physics")
        else:
            saved["compiled_model"]["physics"]["fields"]["body_mass"] = "other mass"
        with patch.object(cp, "read_checkpoint", return_value={"identity": saved}), \
             patch.object(cp, "identity", return_value=current):
            try:
                cp.load_full_checkpoint(runner, "unused.pt")
            except ValueError as error:
                assert ("training recipe" if case == "missing_ppo_semantics" else "physical model") in str(error)
            else:
                raise AssertionError("incompatible physics accepted")
        assert runner.phase == "fresh"
        assert not runner.network.mock_calls


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
    from hexapod_mjlab.tasks.confined_locomotion_2.config import terrain_memory_config as terrain_memory

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


def test_runner_failures_release_resources_without_masking_original_error():
    from types import SimpleNamespace
    from unittest.mock import Mock, patch
    from hexapod_mjlab.tasks.confined_locomotion_2.rl.runner import ConfinedLocomotionRunner
    from hexapod_mjlab.tasks.confined_locomotion_2.rl.tripod_runner import TripodRunner

    for cls in (ConfinedLocomotionRunner, TripodRunner):
        # Construction can fail after hooks/writer allocation or much earlier.
        original = RuntimeError("construction fixture")
        env = SimpleNamespace(close=Mock())
        runtime = SimpleNamespace(close=Mock(side_effect=OSError("cleanup fixture")))
        logger = SimpleNamespace(close=Mock())
        def fail_init(self, *args, **kwargs):
            self.runtime, self.logger = runtime, logger
            raise original
        with patch.object(cls, "_initialize", fail_init):
            try:
                cls(env, {})
            except RuntimeError as error:
                assert error is original
            else:
                raise AssertionError("construction failure swallowed")
        assert runtime.close.call_count == logger.close.call_count == env.close.call_count == 1

        # Exercise the real hooked close path: every resource gets a close
        # attempt even when the first one fails, and repeat close is harmless.
        runner = object.__new__(cls)
        runner.closed, runner.close_error, runner.phase = False, None, "faulted"
        runner.runtime, runner.logger = runtime, logger
        runner._official_close, runner._write_summary = Mock(), Mock()
        runtime.close.reset_mock()
        logger.close.reset_mock()
        runner._close_after_error(original)
        assert runner.closed and runner.close_error is not None
        assert runtime.close.call_count == logger.close.call_count == runner._official_close.call_count == 1
        runner.close()
        assert runner._official_close.call_count == 1

        # A writer initialization or rollout failure must also close the normal
        # hooked environment, even if saving diagnostic evidence fails.
        for during_logging in (True, False):
            original = KeyboardInterrupt("training fixture")
            runner = object.__new__(cls)
            runner.phase, runner.closed, runner.max_iterations = "fresh", False, 1
            runner.current_learning_iteration = runner.total_env_samples = 0
            runner.device = "cuda:0"
            runner.logger = SimpleNamespace(begin_learn=Mock(side_effect=original if during_logging else None))
            runner.learning_metrics = SimpleNamespace(begin_iteration=Mock())
            runner.collector = SimpleNamespace(collect=Mock(side_effect=original))
            runner.runtime = SimpleNamespace(_faulted=False, _save_fault=Mock(side_effect=OSError("evidence fixture")))
            runner._official_close = Mock()
            runner.close = Mock(side_effect=OSError("close fixture"))
            with patch("torch.cuda.reset_peak_memory_stats"):
                try:
                    runner.learn(1)
                except KeyboardInterrupt as error:
                    assert error is original
                else:
                    raise AssertionError("training failure swallowed")
            assert runner.phase == "faulted" and runner.close.call_count == 1
            assert runner.collector.collect.call_count == (0 if during_logging else 1)


def test_execution_modes_reach_both_task_recipes_and_reject_invalid_values():
    from hexapod_mjlab.tasks.confined_locomotion_2.rl.terrain_entry import build_terrain_train_config
    assert confined_locomotion_rl_cfg().execution_mode == "low_memory"
    for mode in ("low_memory", "speed"):
        flat = build_formal_train_config(seed=9, num_envs=2, max_iterations=1, execution_mode=mode,
                                         point_replay_chunk=128)
        terrain = build_terrain_train_config(seed=9, num_envs=2, max_iterations=1,
                                            height_loss_weight=1., execution_mode=mode, point_replay_chunk=128)
        for cfg in (flat, terrain):
            resolved = resolve_recipe(asdict(cfg.agent))
            assert resolved["execution_mode"] == mode
            assert resolved["point_replay_chunk"] == 128
            assert resolved["ppo"].epochs == 5 and resolved["ppo"].minibatches == 4
    for mode in ("auto", "fast", None, True):
        try:
            build_formal_train_config(seed=9, num_envs=2, max_iterations=1, execution_mode=mode)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid execution mode accepted")
    for chunk in (0, -1, True, 1.5):
        try:
            build_formal_train_config(seed=9, num_envs=2, max_iterations=1, point_replay_chunk=chunk)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid point replay chunk accepted")
