"""Numerical and lifecycle checks for the isolated legacy recipe."""

import torch

from hexapod_mjlab.tasks.confined_locomotion_2.tripod2_low.legacy import (
    LegacyCommands, LegacyRewardState, LegacyStageController,
    compute_legacy_rewards, reward_scales, weight_rewards,
)


def make_record(batch=2):
    return {
        "joint_pos": torch.zeros(batch, 18), "joint_vel": torch.zeros(batch, 18),
        "previous_qdot": torch.zeros(batch, 18),
        "previous_action": torch.zeros(batch, 18), "clipped_action": torch.zeros(batch, 18),
        "lin_vel_b": torch.zeros(batch, 3), "ang_vel_b": torch.zeros(batch, 3),
        "gravity_b": torch.tensor([[0., 0., -1.]]).repeat(batch, 1),
        "root_pos_w": torch.tensor([[0., 0., .25]]).repeat(batch, 1),
        "foot_force_w": torch.zeros(batch, 6, 3), "foot_vel_w": torch.zeros(batch, 6, 3),
        "penalized_contact_force_w": torch.zeros(batch, 13, 3),
        "dof_pos_limits": torch.tensor([[-1., 1.]]).repeat(18, 1),
        "torques": torch.ones(batch, 18),
    }


def test_reward_formula_scaling_and_last_substep_effort():
    record = make_record()
    record["joint_pos"][:] = 2.
    record["joint_vel"][:] = .5
    record["clipped_action"][:] = 3.
    record["lin_vel_b"][:] = torch.tensor([.1, .2, .3])
    raw = compute_legacy_rewards(record, LegacyRewardState(2), torch.zeros(2, 3))
    torch.testing.assert_close(raw["tracking_lin_vel"], torch.full((2,), float(torch.exp(torch.tensor(-.05 / .25)))))
    torch.testing.assert_close(raw["torques"], torch.full((2,), 18.))
    torch.testing.assert_close(raw["dof_acc"], torch.full((2,), 18 * 25. ** 2))
    torch.testing.assert_close(raw["dof_pos_limits"], torch.full((2,), 18.))
    torch.testing.assert_close(raw["stand_still2"], torch.full((2,), 72.))
    torch.testing.assert_close(raw["shank_perp2ground"], torch.full((2,), 96.))
    torch.testing.assert_close(raw["action_rate"], torch.full((2,), 162.))
    total0, weighted0 = weight_rewards(raw, 0)
    total1, weighted1 = weight_rewards(raw, 1)
    assert "feet_slip" not in weighted0 and "feet_slip" in weighted1
    torch.testing.assert_close(weighted0["torques"], raw["torques"] * (-.0001 * .02))
    assert (total0 < 0).all() and (total1 < 0).all()  # no positive reward clipping


def test_contact_timers_air_time_touchdown_and_reset_tail():
    state = LegacyRewardState(2)
    record = make_record()
    command = torch.tensor([[.15, 0., 0.], [0., 0., .8]])
    for _ in range(13):
        compute_legacy_rewards(record, state, command)
    record["foot_force_w"][:, :, 2] = 2.
    raw = compute_legacy_rewards(record, state, command)
    torch.testing.assert_close(raw["feet_air_time"], torch.tensor([6 * (.28 - .25), 0.]), atol=1e-6, rtol=0)
    # Filtered air timers keep contact for one extra frame; slip in the
    # actual EL_4090 call sequence uses only CURRENT contact instead.
    record["foot_force_w"].zero_()
    record["foot_vel_w"][:] = 1.
    raw = compute_legacy_rewards(record, state, command)
    assert not raw["feet_slip"].any()
    assert not state.feet_air_time.any()
    record["foot_force_w"][:, :, 2] = 2.
    compute_legacy_rewards(record, state, command)
    state.reset(torch.tensor([0]))
    assert state.last_contacts[0].all()
    assert not state.feet_contact_time[0].any()
    assert state.feet_contact_time[1].any()


def test_gait_groups_are_alphabetical_and_stationary_uses_all_pairs():
    state = LegacyRewardState(2)
    record = make_record()
    # Group (LB, LF, RM) airborne, (LM,RB,RF) grounded. Timer update
    # yields identical mode durations and exactly zero moving gait error.
    state.feet_air_time[:, (0, 1, 5)] = .18
    state.feet_contact_time[:, (2, 3, 4)] = .18
    record["foot_force_w"][:, (2, 3, 4), 2] = 2.
    commands = torch.tensor([[1., 0., 0.], [0., 0., 0.]])
    raw = compute_legacy_rewards(record, state, commands)
    torch.testing.assert_close(raw["gait_2_step"], torch.tensor([0., 9 / 15 * 2 * .2 ** 2]), atol=1e-7, rtol=0)


def test_command_phase_curriculum_push_and_strict_timeout():
    scheduler = LegacyCommands(3)
    scheduler.commands.fill_(9.)
    scheduler.episode_length[:] = torch.tensor([199, 0, 999])
    scheduler.common_step_counter = 149
    assert scheduler.tick()
    assert scheduler.commands[0].abs().max() <= 1.2
    assert torch.all(scheduler.commands[1] == 9.)
    assert not scheduler.timeout.any()  # length=1000 is still active
    scheduler.tick()
    assert scheduler.timeout[2]
    scheduler.common_step_counter = 1000
    assert scheduler.curriculum(torch.tensor([0]), torch.full((3,), 16.01))
    assert scheduler.ranges["lin_vel_x"] == [-1.5, 1.5]
    assert scheduler.ranges["lin_vel_y"] == [-1., 1.]
    scheduler.reset(torch.tensor([2]))
    assert scheduler.episode_length[2] == 0
    clone = LegacyCommands(3)
    clone.load_state_dict(scheduler.state_dict())
    torch.testing.assert_close(clone.commands, scheduler.commands, rtol=0, atol=0)
    assert clone.ranges == scheduler.ranges


def test_stage_only_advances_after_update_and_window_is_retained():
    schedule = LegacyStageController()
    schedule.add_completed([2.] * 100)
    assert not schedule.after_update()  # strict > 2
    schedule.add_completed([3.])
    assert schedule.stage == 0
    assert schedule.after_update()
    assert schedule.stage == 1 and len(schedule.completed_returns) == 100
    assert not schedule.after_update()
    clone = LegacyStageController()
    clone.load_state_dict(schedule.state_dict())
    assert clone.state_dict() == schedule.state_dict()
    assert reward_scales(0)["base_height"] == -.04
    assert reward_scales(1)["base_height"] == -.1
