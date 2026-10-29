"""CPU contract checks; these do not establish GPU simulation resumability."""

import copy
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from hexapod_mjlab.tasks.confined_locomotion_2.rl import tripod_checkpoint as cp

from hexapod_mjlab.rl.confined_locomotion_2.ppo import ppo_semantics

STRICT_REQUIRE_STABLE = cp.require_stable


class State:
    def __init__(self, **state):
        self.state = state or {"value": torch.zeros(1)}

    def state_dict(self):
        return self.state

    def load_state_dict(self, state, **kwargs):
        self.state = state


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.directory = Path(self.stack.enter_context(TemporaryDirectory()))
        tensor = lambda: torch.ones(2)
        action = SimpleNamespace(**{name: tensor() for name in cp.ACTION_FIELDS},
                                 sea_hidden_state=torch.ones(2, 36, 8),
                                 sea_cell_state=torch.ones(2, 36, 8), last_effort=tensor(),
                                 _tripod_checkpoint_fields=("sea_hidden_state", "sea_cell_state", "last_effort"),
                                 control_steps_seen=24)
        self.events = []
        sim = SimpleNamespace(model=SimpleNamespace(**{name: tensor() for name in cp.MODEL_FIELDS}),
                              recompute_constants=lambda level: self.events.append("constants"),
                              forward=lambda: self.events.append("forward"))
        env = SimpleNamespace(sim=sim, action_manager=SimpleNamespace(get_term=lambda name: action),
                              _tripod_model_fields=cp.MODEL_FIELDS, common_step_counter=24,
                              _sim_step_counter=96, **{name: tensor() for name in cp.ENV_FIELDS})
        runtime = State(legacy_reward_state={"timer": tensor()}, stage_controller={"stage": 1, "returns": [1.0]},
                        commands=tensor(), generators={"noise": torch.get_rng_state()})
        runtime.env = env
        runtime.lidar_cfg = SimpleNamespace(far_sampling="all_endpoints")
        runtime._context = SimpleNamespace(sensor=SimpleNamespace(_invalidate_cache=lambda: self.events.append("invalidate")))
        runtime._save_fault = lambda error, **kwargs: self.events.append("fault")
        provider = State()
        provider.generator_state = lambda: torch.get_rng_state()
        provider.load_generator_state = lambda state: None
        self.runner = SimpleNamespace(runtime=runtime, network=State(), provider=provider, trainer=State(schema="confined_locomotion_2.temporal_ppo/v2", semantics=ppo_semantics()),
                                      collector=State(), learning_metrics=State(), task_id=cp.TASK_ID,
                                      phase="ready", _active_buffer=None, current_learning_iteration=1,
                                      total_control_batches=24, total_env_samples=48, device=torch.device("cpu"))
        self.runner.code_provenance = {"status": "dirty", "commit": "fixture", "content_sha256": "code-a"}
        identity = {
            "architecture": cp.architecture_identity(),
            "seed": 1,
            "num_envs": 2,
            "scan_identity": {"scan": "fixture"},
            "recipe": {
                "ppo": {}, "ppo_semantics": ppo_semantics(), "symmetry": {}, "perturbations": None,
                "actuator_sha256": "asset-a",
                "env_semantics": {
                    "action": {"stiffness": 1.0},
                    "rewards": {"reward": 1.0},
                    "terminations": {"names": [], "time_out": {}},
                },
            },
            "compiled_model": {
                "physics": {"schema": "hexapod_mjlab.model_physics/v1", "fields": {"body_mass": "nominal mass"}},
                "dims": {}, "joint_names": [], "actuator_names": [],
                "actuator_gear": [], "q_ref": [], "options": {},
            },
            "warmup_control_steps": cp.WARMUP_CONTROL_STEPS,
            "source": "fixture",
        }
        self.identity = identity
        for target, name, value in (
            (cp, "require_stable", lambda runner: None),
            (cp, "identity", lambda runner: identity),
            (cp.base, "_physics_arrays", lambda env: ({"qpos": tensor()}, {"qpos": "fixture"})),
            (cp.base, "_manager_tensors", lambda env: {"manager": tensor()}),
            (cp.base, "_write_persistent_fields", lambda *args: self.events.append("physics")),
        ):
            self.stack.enter_context(patch.object(target, name, value))

    def save(self):
        return cp.save_full_checkpoint(self.runner, self.directory / "full.pt", infos={"finished": 1})

    def test_round_trip_restores_dr_and_actuator(self):
        path = self.save()
        saved = cp.read_checkpoint(path)
        self.assertEqual(saved["code_provenance"], self.runner.code_provenance)
        self.runner.code_provenance = {"status": "clean", "commit": "other-code"}
        self.assertEqual(saved["runtime"]["stage_controller"]["stage"], 1)
        self.assertEqual(saved["action"]["sea_hidden_state"].shape, (2, 36, 8))
        self.runner.runtime.env.sim.model.body_mass.zero_()
        self.runner.runtime.env.sim.model.pair_friction.zero_()
        self.runner.runtime.env.action_manager.get_term("residual_pd").sea_cell_state.zero_()
        self.assertEqual(cp.load_full_checkpoint(self.runner, path), {"finished": 1})
        self.assertEqual(self.runner.code_provenance["commit"], "other-code")
        self.assertTrue(bool(self.runner.runtime.env.sim.model.body_mass.eq(1).all()))
        self.assertTrue(bool(self.runner.runtime.env.sim.model.pair_friction.eq(1).all()))
        self.assertTrue(bool(self.runner.runtime.env.action_manager.get_term("residual_pd").sea_cell_state.eq(1).all()))
        self.assertEqual(self.events, ["constants", "physics", "forward", "physics", "invalidate"])
        with self.assertRaises(FileExistsError):
            self.save()

    def test_identity_ignores_source_and_device_metadata(self):
        saved = copy.deepcopy(self.identity)
        saved["source"] = {"changed.py": "digest"}
        saved["dependencies"] = {"torch": "9.9.9"}
        saved["device"] = "cuda:1"
        saved["recipe"]["execution_mode"] = "speed"
        saved["recipe"]["point_replay_chunk"] = 128
        current = copy.deepcopy(self.identity)
        current["recipe"]["lidar_config"] = {
            "raycast_mode": "sparse", "far_sampling": "hit_only_10m", "sparse_selection_path": "/new/selection.npz"}
        saved["recipe"]["lidar_config"] = {
            **current["recipe"]["lidar_config"], "sparse_selection_path": "/old/selection.npz"}
        cp.base.compare_identity(
            current, saved, extra_keys=("warmup_control_steps",)
        )
        changed = copy.deepcopy(saved)
        changed["scan_identity"] = {"scan": "different selection content"}
        with self.assertRaisesRegex(ValueError, "scan_identity"):
            cp.base.compare_identity(current, changed, extra_keys=("warmup_control_steps",))
        saved["recipe"]["ppo"] = {"lr": 0.1}
        with self.assertRaisesRegex(ValueError, "training recipe"):
            cp.base.compare_identity(
                self.identity, saved, extra_keys=("warmup_control_steps",)
            )

    def test_apply_failure_freezes_runner(self):
        path = self.save()
        with patch.object(cp.base, "_write_persistent_fields",
                          side_effect=RuntimeError("apply failed")):
            with self.assertRaisesRegex(RuntimeError, "apply failed"):
                cp.load_full_checkpoint(self.runner, path)
        self.assertEqual(self.runner.phase, "faulted")
        self.assertIn("fault", self.events)

    def test_strict_rejection_precedes_mutation(self):
        path = self.save()
        original = torch.load(path, weights_only=False)
        cases = (
            (lambda state: state.update(task=cp.base.TASK_ID), "not a tripod"),
            (lambda state: state.pop("random"), "incomplete"),
            (lambda state: state["identity"].update(warmup_control_steps=2), "warmup_control_steps"),
            (lambda state: state["identity"]["recipe"].update(ppo={"lr": 0.1}), "training recipe"),
            (lambda state: state["identity"]["recipe"].pop("ppo_semantics"), "training recipe"),
            (lambda state: state["trainer"].update(schema="confined_locomotion_2.temporal_ppo/v1"), "incompatible temporal PPO"),
            (lambda state: state["trainer"].pop("semantics"), "incompatible temporal PPO"),
            (lambda state: state["identity"]["recipe"].update(actuator_sha256="asset-b"), "actuator_sha256"),
            (lambda state: state["identity"]["compiled_model"].pop("physics"), "physical model identity"),
            (lambda state: state["identity"]["compiled_model"]["physics"]["fields"].update(body_mass="changed nominal"), "physical model differs"),
            (lambda state: state["action"].pop("sea_cell_state"), "action keys differ"),
            (lambda state: state["runtime"].pop("stage_controller"), "runtime keys differ"),
            (lambda state: state["random"].pop("numpy"), "random keys differ"),
            (lambda state: state["model"].update(body_mass=torch.ones(3)), "model.body_mass shape/dtype differs"),
            (lambda state: state["model"].pop("pair_friction"), "model keys differ"),
        )
        for mutation, message in cases:
            state = cp.base.cpu_tree(original)
            mutation(state)
            torch.save(state, path)
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                cp.load_full_checkpoint(self.runner, path)
            self.assertEqual(self.events, [])

    def test_cannot_select_old_task_or_bad_sampling(self):
        path = self.save()
        with self.assertRaisesRegex(ValueError, "unsupported"):
            cp.read_checkpoint(path, expected_task_id=cp.base.TASK_ID)
        with self.assertRaisesRegex(ValueError, "architecture"):
            cp.read_checkpoint(path, expected_far_sampling="hit_only_10m")

    def test_nonfinite_rejected_on_save(self):
        self.runner.runtime.env.sim.model.body_mass[0] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.save()
        self.assertFalse((self.directory / "full.pt").exists())

    def test_stable_boundary_accounts_for_exactly_one_warmup(self):
        runtime = self.runner.runtime
        runtime._faulted = runtime._stopped = runtime._closed = False
        runtime._context.phase = "idle"
        runtime.control_steps_taken = 25
        runtime.env._manual_reset_pending = torch.zeros(2, dtype=torch.bool)
        trainer = self.runner.trainer
        trainer.faulted, trainer.update_count, trainer.optimizer_step_count = False, 1, 20
        STRICT_REQUIRE_STABLE(self.runner)
        runtime.control_steps_taken = 24
        with self.assertRaisesRegex(RuntimeError, "startup warmup"):
            STRICT_REQUIRE_STABLE(self.runner)
        runtime.control_steps_taken = 25
        trainer.optimizer_step_count = 19
        with self.assertRaisesRegex(RuntimeError, "optimizer"):
            STRICT_REQUIRE_STABLE(self.runner)

    def real_runtime(self, *, populated):
        """Use the real CPU state owners/load path without constructing a simulator."""
        from hexapod_mjlab.tasks.confined_locomotion_2.runtime_tripod import LegacyRuntime
        from hexapod_mjlab.tasks.confined_locomotion_2.components.tripod_mdp import (
            LegacyCommands, LegacyRewardState, LegacyStageController, LegacyResetDRState)
        from hexapod_mjlab.tasks.confined_locomotion_2.observation import ControlInputCache
        from hexapod_mjlab.rl.confined_locomotion_2.history import ScanWindow
        from hexapod_mjlab.rl.confined_locomotion_2.contracts import OnlineObservation
        from hexapod_mjlab.rl.confined_locomotion_2.integration import prepare_control_inputs
        runtime = LegacyRuntime.__new__(LegacyRuntime)
        runtime.env = self.runner.runtime.env
        runtime.device = torch.device("cpu")
        runtime.seed, runtime.mode, runtime.scan_identity = 1, "training", {"scan": "fixture"}
        runtime._closed = runtime._faulted = runtime._stopped = False
        runtime._context = self.runner.runtime._context
        runtime._context.phase = "idle"
        runtime._save_fault = self.runner.runtime._save_fault
        runtime.lidar_cfg = self.runner.runtime.lidar_cfg
        runtime._control_steps = 24
        for name in ("episode_ids", "scan_counters", "sensor_scan_counts", "next_due_scan_step"):
            setattr(runtime, name, torch.zeros(2, dtype=torch.int64))
        runtime._noise_keys = torch.zeros(2, 2, dtype=torch.int64)
        runtime._noise = torch.zeros(2, 63)
        runtime._proprio_cached = {name: torch.ones(2, 3) for name in
                                  ("lin_vel_b", "ang_vel_b", "gravity_b")} if populated else None
        env = runtime.env
        env._tripod_commands = LegacyCommands(2)
        env._tripod_commands.episode_length.fill_(24)
        env._tripod_commands.common_step_counter = 24
        env.episode_length_buf = env._tripod_commands.episode_length.clone()
        runtime.step_in_episode = env._tripod_commands.episode_length
        env._tripod_reward_state = LegacyRewardState(2)
        env._tripod_stage = LegacyStageController()
        env._tripod_dr = LegacyResetDRState(2)
        for name, shape in (("_tripod_episode_return", (2,)), ("_tripod_tracking_sum", (2,)),
                            ("_tripod_previous_qdot", (2, 18)), ("_tripod_previous_action", (2, 18))):
            setattr(env, name, torch.zeros(shape))
        runtime.window = ScanWindow(2, episode_ids=[0, 0])
        runtime.window.write(env_ids=[0, 1], episode_ids=[0, 0], scan_ids=[0, 0], times=[0., 0.],
                             near_points=torch.zeros(2, 256, 3), near_valid=torch.zeros(2, 256, dtype=torch.bool),
                             far_points=torch.zeros(2, 128, 3), far_valid=torch.zeros(2, 128, dtype=torch.bool))
        runtime._input_cache = ControlInputCache()
        runtime._current_window_snapshot = None
        if populated:
            env._tripod_stage.add_completed([3., 4.])
            env._tripod_stage.after_update()
            snapshot = runtime.window.read()
            obs = OnlineObservation(
                env_ids=torch.arange(2), episode_ids=torch.zeros(2, dtype=torch.int64),
                step_ids=torch.full((2,), 24, dtype=torch.int64), time_s=torch.full((2,), .48),
                lin_vel_b=torch.zeros(2, 3), ang_vel_b=torch.zeros(2, 3),
                gravity_b=torch.tensor([[0., 0., -1.]]).expand(2, 3).clone(),
                joint_pos_rel_ref=torch.zeros(2, 18), joint_vel=torch.zeros(2, 18),
                prev_action=torch.zeros(2, 18), command=torch.zeros(2, 3))
            prepared = prepare_control_inputs(window=snapshot, observation=obs,
                                               near_angle_keys=None, far_angle_keys=None)
            runtime._input_cache.store(24, runtime.episode_ids, prepared)
            runtime._current_window_snapshot = snapshot
        return runtime

    def test_real_runtime_populated_cache_restores_into_fresh_optional_state(self):
        self.runner.runtime = self.real_runtime(populated=True)
        path = self.save()
        self.runner.runtime = self.real_runtime(populated=False)
        self.assertIsNone(self.runner.runtime._input_cache.get())
        self.assertEqual(self.runner.runtime.env._tripod_stage.stage, 0)
        cp.load_full_checkpoint(self.runner, path)
        runtime = self.runner.runtime
        self.assertEqual(runtime.env._tripod_stage.stage, 1)
        self.assertEqual(list(runtime.env._tripod_stage.completed_returns), [3., 4.])
        self.assertEqual(runtime._input_cache.get().near_points.shape, (2, 256, 3))
        self.assertEqual(runtime._current_window_snapshot.far_points.shape, (2, 1, 128, 3))
        self.assertTrue(bool(runtime._proprio_cached["lin_vel_b"].eq(1).all()))

    def test_real_runtime_bad_optional_cache_layout_rejected_before_apply(self):
        from dataclasses import replace
        self.runner.runtime = self.real_runtime(populated=True)
        path = self.save()
        payload = torch.load(path, weights_only=False)
        prepared = payload["runtime"]["cache"]["_prepared"]
        payload["runtime"]["cache"]["_prepared"] = replace(prepared, near_points=torch.zeros(2, 255, 3))
        torch.save(payload, path)
        self.runner.runtime = self.real_runtime(populated=False)
        with self.assertRaisesRegex(ValueError, "cache near_points shape/dtype differs"):
            cp.load_full_checkpoint(self.runner, path)
        self.assertEqual(self.events, [])


if __name__ == "__main__":
    unittest.main()
