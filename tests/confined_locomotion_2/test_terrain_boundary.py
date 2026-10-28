"""Focused CPU contracts for Terrain Memory's finite-field boundary."""
from types import SimpleNamespace
import unittest

import torch

from hexapod_mjlab.common.transition import TransitionContext
from hexapod_mjlab.common.mdp.rewards import (
    ROOT_HEIGHT_TARGET_M,
    pen_foot_clearance,
    pen_root_height,
    terrain_clearances,
)
from hexapod_mjlab.common.mdp.terminations import terrain_boundary


def bind_context(record, query=None):
    """Bind a minimal instance of the production context class."""
    env = SimpleNamespace(num_envs=record["root_pos_w"].shape[0], device="cpu")
    context = object.__new__(TransitionContext)
    context.env = env
    context.terrain_query = query
    context.control_id = 7
    context.ensure_ready = lambda: record
    env._confined_transition_context = context
    return env, context


def make_record(root_z=(0.46, 0.46), *, gravity=None):
    batch = len(root_z)
    root = torch.zeros(batch, 3)
    root[:, 2] = torch.tensor(root_z)
    feet = torch.zeros(batch, 6, 3)
    feet[:, :, 2] = 0.12
    return {
        "root_pos_w": root,
        "foot_pos_w": feet,
        "gravity_b": gravity if gravity is not None else torch.tensor([[0., 0., -1.]] * batch),
    }


class TerrainBoundaryTest(unittest.TestCase):
    def test_clearances_cache_query_and_mask_invalid_samples(self):
        record = make_record()
        valid = torch.tensor([[True, True, False, True, True, True, True],
                              [True, True, True, True, True, True, True]])
        heights = torch.full((2, 7), 0.02)

        class FakeQuery:
            calls = 0

            def world_heights(self, rows, xy):
                self.calls += 1
                self.rows, self.xy = rows.clone(), xy.clone()
                return heights, valid

        query = FakeQuery()
        env, _ = bind_context(record, query)
        first = terrain_clearances(env, record)
        again = terrain_clearances(env, record)

        self.assertIs(first, again)
        self.assertEqual(query.calls, 1)
        self.assertEqual(tuple(query.xy.shape), (2, 7, 2))
        self.assertEqual(tuple(first[0].shape), (2, 7))
        self.assertTrue(torch.equal(first[1], valid))
        self.assertEqual(float(first[0][0, 2]), 0.0)
        self.assertAlmostEqual(float(first[0][0, 1]), 0.10, places=6)

    def test_flat_clearances_keep_plane_path(self):
        record = make_record()
        env, _ = bind_context(record)
        self.assertIsNone(terrain_clearances(env, record))

    def test_boundary_truncates_missing_terrain_but_real_fall_wins(self):
        record = make_record(gravity=torch.tensor([[0., 0., -1.], [0., 0., 1.]]))
        valid = torch.ones(2, 7, dtype=torch.bool)
        valid[0, 3] = False
        valid[1, 0] = False

        class FakeQuery:
            def world_heights(self, rows, xy):
                return torch.zeros(2, 7), valid

        env, _ = bind_context(record, FakeQuery())
        self.assertEqual(terrain_boundary(env).tolist(), [True, False])

    def test_invalid_root_does_not_count_as_low_height_fall(self):
        record = make_record((0.01, 0.01))
        valid = torch.ones(2, 7, dtype=torch.bool)
        valid[0, 0] = False

        class FakeQuery:
            def world_heights(self, rows, xy):
                return torch.zeros(2, 7), valid

        env, _ = bind_context(record, FakeQuery())
        result = terrain_boundary(env)
        state = env._confined_termination_state
        self.assertEqual(result.tolist(), [True, False])
        self.assertEqual(state.low_height_count.tolist(), [0, 1])

    def test_root_height_reward_preserves_formula_on_valid_rows(self):
        record = make_record((ROOT_HEIGHT_TARGET_M - 0.06, ROOT_HEIGHT_TARGET_M))
        valid = torch.ones(2, 7, dtype=torch.bool)
        valid[1, 0] = False

        class FakeQuery:
            def world_heights(self, rows, xy):
                return torch.zeros(2, 7), valid

        env, _ = bind_context(record, FakeQuery())
        raw = pen_root_height(env, weight=-50.0)
        self.assertTrue(torch.allclose(raw, torch.tensor([0.06**2, 0.0])))

    def test_foot_clearance_reward_masks_only_invalid_feet(self):
        record = make_record((ROOT_HEIGHT_TARGET_M,))
        record["foot_pos_w"][:, :, 2] = 0.0
        record["support"] = torch.zeros(1, 6, dtype=torch.bool)
        record["command"] = torch.tensor([[0.2, 0.0, 0.0]])
        valid = torch.ones(1, 7, dtype=torch.bool)
        valid[0, 2:4] = False

        class FakeQuery:
            def world_heights(self, rows, xy):
                return torch.zeros(1, 7), valid

        env, _ = bind_context(record, FakeQuery())
        raw = pen_foot_clearance(env, weight=-5.0)
        self.assertTrue(torch.allclose(raw, torch.tensor([4 * 0.12**2])))

    def test_terrain_only_config_registers_boundary_timeout(self):
        from hexapod_mjlab.tasks.confined_locomotion_2.config.terrain_memory_config import (
            make_training_env_cfg,
        )
        from hexapod_mjlab.tasks.confined_locomotion_2.flat_env_cfg import (
            make_flat_env_cfg,
        )

        terrain_terms = make_training_env_cfg(num_envs=1).terminations
        flat_terms = make_flat_env_cfg(play=False, num_envs=1).terminations
        self.assertIn("terrain_boundary", terrain_terms)
        self.assertTrue(terrain_terms["terrain_boundary"].time_out)
        self.assertNotIn("terrain_boundary", flat_terms)


def test_terrain_boundary_contracts():
    """Include these unittest contracts in the dependency-free task runner."""
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TerrainBoundaryTest)
    result = unittest.TestResult()
    suite.run(result)
    assert result.wasSuccessful(), result.errors + result.failures


def test_command_records_restore_after_episode_resampling():
    from hexapod_mjlab.tasks.confined_locomotion_2.commands import TrainingCommandScheduler
    first = TrainingCommandScheduler(2, seed=19, device="cpu")
    first.sample_reset([0, 1])
    first.sample_reset([0])
    state = first.state_dict()
    restored = TrainingCommandScheduler(2, seed=19, device="cpu")
    restored.sample_reset([0, 1])
    restored.load_state_dict(state)
    assert restored.state_dict()["resample_records"] == state["resample_records"]
    assert restored.state_dict()["resample_records_count"] == 3
    assert torch.equal(restored.current_command(), first.current_command())
    first.sample_reset([1])
    restored.sample_reset([1])
    assert torch.equal(restored.current_command(), first.current_command())
    assert restored.resample_records == first.resample_records


if __name__ == "__main__":
    unittest.main()
