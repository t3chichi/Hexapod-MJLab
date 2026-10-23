"""CPU contracts for transition helpers, reward state, and fall state."""
from __future__ import annotations

import traceback
import unittest
from types import SimpleNamespace

import numpy as np

import torch

from hexapod_mjlab.tasks.confined_locomotion.mdp.rewards import (
    FootAirTimeState,
    TRAINING_REWARD_WEIGHTS,
    foot_clearance_cost,
    foot_slip_cost,
    foot_stumble_mask,
    landing_air_reward,
    motion_gate,
    read_current_foot_support,
)
from hexapod_mjlab.tasks.confined_locomotion.mdp.terminations import (
    TRAINING_TERMINATION_TIME_OUT_FLAGS,
    TrainingTerminationState,
    advance_fall_state,
)
from hexapod_mjlab.tasks.confined_locomotion.transition import require_finite


def test_finite_guard_returns_good_input_and_rejects_nonfinite_values():
    value = torch.tensor([1.0, -2.0])
    assert require_finite("transition input", value) is value
    for bad in (torch.tensor([1.0, float("nan")]), torch.tensor([float("inf")]),
                [1.0, float("nan")]):
        with unittest.TestCase().assertRaisesRegex(RuntimeError, "NaN/Inf"):
            require_finite("transition input", bad)


def test_foot_air_time_state_advances_once_per_control_and_resets_on_landing():
    state = FootAirTimeState(1, "cpu")
    unsupported = torch.zeros(1, 6, dtype=torch.bool)
    supported = torch.ones(1, 6, dtype=torch.bool)

    landing, consumed = state.advance(1, unsupported)
    assert not landing.any() and torch.equal(consumed, torch.zeros(1, 6))
    assert torch.allclose(state.air_time_s, torch.full((1, 6), 0.02))
    repeated_landing, repeated_consumed = state.advance(1, unsupported)
    assert torch.equal(landing, repeated_landing) and torch.equal(consumed, repeated_consumed)
    assert torch.allclose(state.air_time_s, torch.full((1, 6), 0.02))

    state.advance(2, unsupported)
    landing, consumed = state.advance(3, supported)
    assert bool(landing.all()) and torch.allclose(consumed, torch.full((1, 6), 0.06))
    assert torch.equal(state.air_time_s, torch.zeros(1, 6))
    landing, consumed = state.advance(4, supported)
    assert not landing.any() and torch.equal(consumed, torch.zeros(1, 6))

    state.reset_rows(torch.tensor([0]), unsupported)
    assert torch.equal(state.air_time_s, torch.zeros(1, 6))
    landing, _ = state.advance(5, supported)
    assert bool(landing.all())


def test_fall_counters_clear_on_interruption_and_flip_is_immediate():
    low = torch.tensor([True, False])
    tilted = torch.tensor([False, True])
    flipped = torch.tensor([False, False])
    counts = torch.zeros(2, dtype=torch.long)
    state = advance_fall_state(low, tilted, flipped, counts, counts.clone())
    assert state["low_height_count"].tolist() == [1, 0]
    assert state["tilt_count"].tolist() == [0, 1]
    assert not state["any_fall"].any()

    state = advance_fall_state(
        torch.tensor([False, True]), tilted, flipped,
        state["low_height_count"], state["tilt_count"],
    )
    assert state["low_height_count"].tolist() == [0, 1]
    assert state["tilt_count"].tolist() == [0, 2]

    state = advance_fall_state(
        torch.tensor([True, False]), torch.tensor([False, False]),
        torch.tensor([False, True]), torch.tensor([9, 0]), torch.zeros(2, dtype=torch.long),
    )
    assert state["low_height_done"].tolist() == [True, False]
    assert state["flip_done"].tolist() == [False, True]
    assert state["any_fall"].tolist() == [True, True]


def test_reward_support_formulas_and_timeout_identity_are_fixed():
    command = torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.2]])
    assert motion_gate(command).tolist() == [True, False, True]
    support = torch.tensor([[False, True]])
    foot_velocity = torch.tensor([[[1.0, 2.0, 9.0], [5.0, 5.0, 1.0]]])
    assert foot_slip_cost(foot_velocity, support).tolist() == [50.0]
    heights = torch.tensor([[0.05, 0.3]])
    assert torch.allclose(
        foot_clearance_cost(heights, support), torch.tensor([(0.12 - 0.05) ** 2]), atol=1e-7
    )
    landing = torch.tensor([[True, False]])
    consumed = torch.tensor([[0.30, 0.30]])
    assert torch.allclose(
        landing_air_reward(landing, consumed, torch.tensor([True])), torch.tensor([0.05]), atol=1e-7
    )
    assert landing_air_reward(landing, consumed, torch.tensor([False])).tolist() == [0.0]

    force = torch.zeros(1, 6, 3)
    force[0, 0] = torch.tensor([6.0, 0.0, 1.0])
    force[0, 1] = torch.tensor([1.0, 0.0, 1.0])
    assert foot_stumble_mask(force).tolist() == [True]

    assert set(TRAINING_REWARD_WEIGHTS) == {
        "track_lin_vel", "track_ang_vel", "pen_lin_vel_z", "pen_ang_vel_xy",
        "pen_gravity_xy", "pen_torque", "pen_joint_vel", "pen_joint_acc",
        "pen_action_rate", "pen_soft_joint_limit", "pen_nonfoot_env_contact",
        "pen_self_contact", "pen_foot_stumble", "pen_root_height", "pen_foot_slip",
        "reward_foot_air", "pen_foot_clearance",
    }
    assert TRAINING_TERMINATION_TIME_OUT_FLAGS == {
        "fall_low_height": False,
        "fall_tilt_sustained": False,
        "fall_flipped": False,
        "episode_timeout": True,
    }


def test_fixture_termination_presets_are_recorded():
    state = TrainingTerminationState(2, "cpu")
    state.preset_low_height_counts(torch.tensor([9, 0]))
    state.preset_tilt_counts(torch.tensor([0, 9]))
    assert state.low_height_count.tolist() == [9, 0]
    assert state.tilt_count.tolist() == [0, 9]
    assert [record["field"] for record in state.presets] == ["low_height_count", "tilt_count"]
    assert state.presets[0]["values"] == [9, 0]


def test_reset_support_reads_current_contact_reader_and_qualifies_real_contacts():
    # No legacy diagnostic object: reset must use the same current reader as
    # TransitionContext. Exercise body/plane pairing, distance and force gates.
    table = {
        "n_contacts": 4,
        "geom": np.array([[7, 99], [99, 8], [9, 99], [10, 99]]),
        "body1": np.array([10, 0, 12, 13]),
        "body2": np.array([0, 11, 0, 0]),
        "dist": np.array([0.0, -0.01, 0.01, 0.0]),
        "force6_world": np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0],
                                  [1, 0, 0, 0, 0, 0], [0.05, 0, 0, 0, 0, 0]]),
        "worldid": np.array([0, 1, 0, 1]),
    }
    context = SimpleNamespace(
        env=SimpleNamespace(num_envs=2, device="cpu"),
        contact_reader=SimpleNamespace(read_substep_table=lambda: table),
        foot_global_ids=[10, 11, 12, 13, 14, 15], plane_geom_id=99,
    )
    support = read_current_foot_support(context)
    assert support.tolist() == [[True, False, False, False, False, False],
                                [False, True, False, False, False, False]]


def run_cpu_checks() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} CPU checks passed")
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(run_cpu_checks())

