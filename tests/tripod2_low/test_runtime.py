"""CPU lifecycle checks at real adapter boundaries, without a fake GPU claim."""
from types import SimpleNamespace
from unittest.mock import patch

import torch

from hexapod_mjlab.common.transition import TransitionContext
from hexapod_mjlab.rl.confined_locomotion.contracts import OnlineObservation
from hexapod_mjlab.rl.confined_locomotion_2.integration import PreparedControlInputs
from hexapod_mjlab.tasks.confined_locomotion_2.runtime import TrainingRuntime
from hexapod_mjlab.tasks.confined_locomotion_2.tripod2_low.runtime import LegacyRuntime, LegacyContext
from hexapod_mjlab.tasks.confined_locomotion_2.tripod2_low.legacy import (
    LegacyCommands, LegacyRewardState, LegacyStageController)


def observation(episodes=(0, 0)):
    return OnlineObservation(
        env_ids=torch.arange(2), episode_ids=torch.tensor(episodes), step_ids=torch.zeros(2, dtype=torch.long),
        time_s=torch.zeros(2), lin_vel_b=torch.zeros(2, 3), ang_vel_b=torch.zeros(2, 3),
        gravity_b=torch.tensor([[0., 0., -1.]]).expand(2, 3).clone(), joint_pos_rel_ref=torch.zeros(2, 18),
        joint_vel=torch.zeros(2, 18), prev_action=torch.zeros(2, 18), command=torch.zeros(2, 3))


def prepared(episodes=(0, 0)):
    return SimpleNamespace(observation=observation(episodes))


def test_runtime_noise_is_cached_per_row_after_partial_reset_and_clipped_after_scaling():
    runtime = LegacyRuntime.__new__(LegacyRuntime)
    runtime.device = torch.device("cpu")
    runtime._noise_keys = torch.full((2, 2), -1, dtype=torch.long)
    runtime._noise = torch.zeros(2, 63)
    runtime._last_noised = None
    runtime._proprio_cached = {"lin_vel_b": torch.full((2, 3), 60.),
                              "ang_vel_b": torch.zeros(2, 3), "gravity_b": torch.zeros(2, 3)}
    first = prepared()
    with patch.object(TrainingRuntime, "prepare", return_value=first):
        runtime.prepare()
        noise = runtime._noise.clone()
        rng = torch.get_rng_state().clone()
        runtime.prepare()
        assert torch.equal(rng, torch.get_rng_state())
        assert torch.equal(first.observation.lin_vel_b, torch.full((2, 3), 50.))
    # A manual command edit rebuilds inputs but must reuse noise at the same identity.
    again = prepared()
    with patch.object(TrainingRuntime, "prepare", return_value=again):
        runtime.prepare()
        assert torch.equal(rng, torch.get_rng_state())
        assert torch.equal(first.observation.proprio_vector(), again.observation.proprio_vector())
    partial = prepared((1, 0))
    with patch.object(TrainingRuntime, "prepare", return_value=partial):
        runtime.prepare()
    assert torch.equal(runtime._noise[1], noise[1])
    assert not torch.equal(runtime._noise[0], noise[0])
    assert torch.equal(partial.observation.prev_action, torch.zeros(2, 18))
    assert torch.equal(partial.observation.command, torch.zeros(2, 3))


def test_context_resamples_before_reward_push_overwrites_and_repeated_terms_do_not_tick():
    commands = LegacyCommands(2)
    commands.episode_length.fill_(199)
    commands.common_step_counter = 149
    commands.commands.fill_(9.)
    env = SimpleNamespace(_tripod_commands=commands, _tripod_reward_state=LegacyRewardState(2),
        _tripod_stage=LegacyStageController(), episode_length_buf=torch.full((2,), 200),
        common_step_counter=150, num_envs=2, device="cpu",
        _tripod_previous_qdot=torch.zeros(2, 18), _tripod_previous_action=torch.zeros(2, 18),
        _tripod_episode_return=torch.zeros(2), _tripod_tracking_sum=torch.zeros(2),
        _tripod_dr=SimpleNamespace(push_velocities=lambda: torch.tensor([[.2, -.3], [.4, -.5]])))
    sensor = SimpleNamespace(data=SimpleNamespace(force=torch.zeros(2, 13, 3)), _invalidate_cache=lambda: None)
    env.scene = {"legacy_penalized_contact": sensor}
    written = []
    robot = SimpleNamespace(data=SimpleNamespace(root_link_vel_w=torch.ones(2, 6)),
                            write_root_link_velocity_to_sim=lambda value: written.append(value.clone()))
    record = {"joint_pos": torch.zeros(2, 18), "joint_vel": torch.zeros(2, 18),
              "clipped_action": torch.zeros(2, 18), "root_pos_w": torch.tensor([[0., 0., .45]]).expand(2, 3),
              "lin_vel_b": torch.zeros(2, 3), "ang_vel_b": torch.zeros(2, 3),
              "gravity_b": torch.tensor([[0., 0., -1.]]).expand(2, 3),
              "foot_force_w": torch.zeros(2, 6, 3), "foot_vel_w": torch.zeros(2, 6, 3),
              "foot_pos_w": torch.zeros(2, 6, 3), "support": torch.zeros(2, 6, dtype=torch.bool),
              "pd": [{"effort": torch.zeros(2, 18)}]}
    context = LegacyContext.__new__(LegacyContext)
    context.env, context.robot = env, robot
    context.phase = "capturing"
    context.runtime = SimpleNamespace(mode="training")

    def seal(_):
        context.phase = "ready"
        return record

    with patch.object(TransitionContext, "ensure_ready", seal):
        context.ensure_ready()
        assert commands.common_step_counter == 150
        assert bool((record["command"].abs() < 1.201).all())
        expected = torch.exp(-commands.commands[:, :2].square().sum(1) / .25)
        torch.testing.assert_close(context.raw_rewards["tracking_lin_vel"], expected)
        torch.testing.assert_close(written[0][:, :2], env._tripod_dr.push_velocities())
        assert torch.equal(written[0][:, 2:], torch.ones(2, 4))
        assert torch.equal(record["lin_vel_b"], torch.zeros(2, 3))
        timers = env._tripod_reward_state.feet_air_time.clone()
        context.ensure_ready()
        assert commands.common_step_counter == 150 and len(written) == 1
        assert torch.equal(timers, env._tripod_reward_state.feet_air_time)
