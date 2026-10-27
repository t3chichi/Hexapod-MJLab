"""CPU contracts for training configuration, command streams, and row resets."""
from __future__ import annotations

import traceback
import unittest
from types import SimpleNamespace

import torch

from support.fixtures import DEFAULT_SEED, build_env_cfg
from hexapod_mjlab.tasks.confined_locomotion.commands import (
    TrainingCommandScheduler,
    derive_command_generator_seed,
)


AUTHORITY_WEIGHTS = {
    "track_lin_vel": 5.0,
    "track_ang_vel": 2.5,
    "pen_lin_vel_z": -2.0,
    "pen_ang_vel_xy": -0.5,
    "pen_gravity_xy": -5.0,
    "pen_torque": -1e-6,
    "pen_joint_vel": -1e-6,
    "pen_joint_acc": -1e-8,
    "pen_action_rate": -0.005,
    "pen_soft_joint_limit": -1.0,
    "pen_nonfoot_env_contact": -1.0,
    "pen_self_contact": -1.0,
    "pen_foot_stumble": -0.2,
    "pen_root_height": -50.0,
    "pen_foot_slip": -0.05,
    "reward_foot_air": 3.0,
    "pen_foot_clearance": -5.0,
}


def test_training_environment_config_is_fresh_and_keeps_flat_defaults_separate():
    from hexapod_mjlab.tasks.confined_locomotion.flat_env_cfg import make_flat_env_cfg
    from hexapod_mjlab.common.mdp.terminations import (
        TRAINING_TERMINATION_TIME_OUT_FLAGS,
    )

    cfg = build_env_cfg(num_envs=2, seed=DEFAULT_SEED)
    assert cfg.auto_reset is False
    assert cfg.commands == {}
    assert cfg.observations["policy"].nan_policy == "error"
    assert cfg.episode_length_s == 20.0 and cfg.seed == DEFAULT_SEED
    weights = {name: float(term.weight) for name, term in cfg.rewards.items()}
    assert weights == AUTHORITY_WEIGHTS
    for name, term in cfg.rewards.items():
        assert term.params == {"weight": AUTHORITY_WEIGHTS[name]}
    timeouts = {name: bool(term.time_out) for name, term in cfg.terminations.items()}
    assert timeouts == TRAINING_TERMINATION_TIME_OUT_FLAGS
    assert set(cfg.observations["policy"].terms) == {
        "base_lin_vel", "base_ang_vel", "projected_gravity",
        "joint_pos_rel", "joint_vel_rel", "clipped_prev_action",
    }

    other = build_env_cfg(num_envs=2)
    cfg.rewards.clear()
    cfg.terminations.clear()
    assert set(other.rewards) == set(AUTHORITY_WEIGHTS)
    assert set(other.terminations) == set(TRAINING_TERMINATION_TIME_OUT_FLAGS)

    flat = make_flat_env_cfg(num_envs=2)
    assert flat.auto_reset is True
    assert flat.observations["policy"].nan_policy == "disabled"
    assert flat.rewards == {}
    assert flat.terminations == {}


def test_command_streams_are_per_environment_and_reproducible():
    assert derive_command_generator_seed(DEFAULT_SEED, 0) == derive_command_generator_seed(DEFAULT_SEED, 0)
    assert derive_command_generator_seed(DEFAULT_SEED, 0) != derive_command_generator_seed(DEFAULT_SEED, 1)
    assert derive_command_generator_seed(DEFAULT_SEED - 1, 0) != derive_command_generator_seed(DEFAULT_SEED, 0)
    assert 0 <= derive_command_generator_seed(DEFAULT_SEED, 3) < 2**32

    a = TrainingCommandScheduler(3, seed=123, device="cpu")
    b = TrainingCommandScheduler(3, seed=123, device="cpu")
    a.sample_reset([0, 1, 2])
    b.sample_reset([0, 1, 2])
    assert torch.equal(a.current_command(), b.current_command())

    # An extra reset consumes only the selected row's random stream.
    a.sample_reset([0])
    c = TrainingCommandScheduler(3, seed=123, device="cpu")
    c.sample_reset([0, 1, 2])
    assert torch.equal(a.current_command()[1:], c.current_command()[1:])
    assert not torch.equal(a.current_command()[0], c.current_command()[0])


def test_command_period_ranges_state_roundtrip_and_manual_mode():
    scheduler = TrainingCommandScheduler(1, seed=7, device="cpu", period=5)
    scheduler.sample_reset([0])
    records = []
    for _ in range(5):
        records.extend(scheduler.step())
    assert len(records) == 1 and records[0]["reason"] == "period"
    assert records[0]["env"] == 0
    assert "category" in records[0] and "component_draws" in records[0]

    held = TrainingCommandScheduler(1, seed=7, device="cpu", period=5)
    held.sample_reset([0])
    original = held.current_command().clone()
    for _ in range(4):
        held.step()
    assert torch.equal(held.current_command(), original)

    sampler = TrainingCommandScheduler(1, seed=99, device="cpu", period=1)
    sampler.sample_reset([0])
    for _ in range(64):
        sampler.step()
        vx, vy, omega = sampler.current_command()[0].tolist()
        assert -1.5 <= vx <= 1.5 and -1.0 <= vy <= 1.0 and -1.0 <= omega <= 1.0

    saved = TrainingCommandScheduler(2, seed=5, device="cpu", period=10)
    saved.sample_reset([0, 1])
    saved.step()
    state = saved.state_dict()
    restored = TrainingCommandScheduler(2, seed=5, device="cpu", period=10)
    restored.load_state_dict(state)
    for _ in range(7):
        saved.step()
        restored.step()
    assert torch.equal(saved.current_command(), restored.current_command())
    with unittest.TestCase().assertRaises(Exception):
        TrainingCommandScheduler(2, seed=6, device="cpu", period=10).load_state_dict(state)

    random_mode = TrainingCommandScheduler(1, seed=11, device="cpu")
    with unittest.TestCase().assertRaises(Exception):
        random_mode.set_command([0], (0.5, 0.0, 0.0))
    manual = TrainingCommandScheduler(1, seed=11, device="cpu", allow_manual=True)
    manual.sample_reset([0])
    manual.set_command([0], (1.5, -1.0, 1.0))
    for _ in range(300):
        manual.step()
    assert torch.equal(manual.current_command(), torch.tensor([[1.5, -1.0, 1.0]]))
    manual.sample_reset([0])
    assert torch.equal(manual.current_command(), torch.zeros(1, 3))
    with unittest.TestCase().assertRaises(Exception):
        manual.set_command([0], (1.6, 0.0, 0.0))


def test_command_audit_retains_new_records_when_history_is_bounded():
    import hexapod_mjlab.tasks.confined_locomotion.commands as module

    old_limit = module.MAX_RESAMPLE_RECORDS
    try:
        module.MAX_RESAMPLE_RECORDS = 2
        scheduler = TrainingCommandScheduler(2, seed=7, device="cpu", period=1)
        first = scheduler.sample_reset([0, 1])
        assert [r["sample_index"] for r in first] == [1, 2]
        next_records = scheduler.step(episode_steps=torch.tensor([200, 200]))
        assert [r["sample_index"] for r in next_records] == [3, 4]
        assert [r["episode_step"] for r in next_records] == [200, 200]
        assert len(scheduler.resample_records) == 2
        assert scheduler.sample_reset([0])[0]["sample_index"] == 5
    finally:
        module.MAX_RESAMPLE_RECORDS = old_limit


def test_partial_reset_restores_every_unselected_physics_row_without_reallocation():
    import numpy as np
    import warp as wp

    from hexapod_mjlab.tasks.confined_locomotion.runtime import (
        _reset_env_preserving_unselected_physics,
    )
    from hexapod_mjlab.common.transition import PERSISTENT_PHYSICS_FIELDS

    shapes = {
        "time": (2,), "qpos": (2, 3), "qvel": (2, 3),
        "qacc_warmstart": (2, 3), "ctrl": (2, 2), "qfrc_applied": (2, 3),
        "xfrc_applied": (2, 2, 6), "eq_active": (2, 1), "act": (2, 2),
        "act_dot": (2, 2), "mocap_pos": (2, 1, 3), "mocap_quat": (2, 1, 4),
        "userdata": (2, 2), "history": (2, 0), "plugin_state": (2, 2),
    }
    arrays, before, pointers = {}, {}, {}
    for index, (name, shape) in enumerate(shapes.items()):
        values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + index * 100
        arrays[name] = wp.array(values, dtype=wp.float32, device="cpu")
        before[name] = wp.to_torch(arrays[name]).clone()
        pointers[name] = int(arrays[name].ptr or 0)
    assert set(PERSISTENT_PHYSICS_FIELDS).issubset(arrays)

    class Env:
        num_envs = 2

        def __init__(self):
            self.sim = SimpleNamespace(wp_data=SimpleNamespace(**arrays))
            self.reset_calls = []

        def reset(self, *, env_ids):
            self.reset_calls.append(env_ids.clone())
            for array in vars(self.sim.wp_data).values():
                if array.size:
                    wp.to_torch(array).add_(10)
            wp.to_torch(self.sim.wp_data.qpos)[0].fill_(-1)
            return "reset-result"

    env = Env()
    result = _reset_env_preserving_unselected_physics(env, torch.tensor([0], dtype=torch.long))
    assert result == "reset-result"
    assert len(env.reset_calls) == 1
    assert torch.equal(env.reset_calls[0], torch.tensor([0]))
    for name, array in arrays.items():
        assert int(array.ptr or 0) == pointers[name], name
        if array.size:
            assert torch.equal(wp.to_torch(array)[1], before[name][1]), name
    assert torch.equal(wp.to_torch(arrays["qpos"])[0], torch.full((3,), -1.0))
    assert torch.equal(wp.to_torch(arrays["qacc_warmstart"])[0], before["qacc_warmstart"][0] + 10)


def test_full_reset_does_not_read_or_snapshot_physics():
    from hexapod_mjlab.tasks.confined_locomotion.runtime import (
        _reset_env_preserving_unselected_physics,
    )

    class Env:
        num_envs = 2

        @property
        def sim(self):
            raise AssertionError("full reset must not inspect physics arrays")

        def reset(self, *, env_ids):
            assert torch.equal(env_ids, torch.tensor([0, 1]))
            return env_ids

    result = _reset_env_preserving_unselected_physics(Env(), torch.tensor([0, 1]))
    assert torch.equal(result, torch.tensor([0, 1]))


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

