"""CPU execution-chain checks, not a locomotion/training validation."""

from types import SimpleNamespace

import torch

from hexapod_mjlab.tasks.confined_locomotion_2.tripod2_low.actuator import (
    ACTUATOR_PATH, LegacyLSTMAction, LegacyLSTMActionCfg,
)


class FakeRobot:
    def __init__(self, batch):
        self.data = SimpleNamespace(
            default_joint_pos=torch.full((batch, 18), .9),
            joint_pos=torch.zeros(batch, 18),
            joint_vel=torch.zeros(batch, 18),
        )

    def find_joints_by_actuator_names(self, names):
        return list(range(18)), [f"joint{i}" for i in range(18)]

    def set_joint_effort_target(self, efforts, joint_ids):
        self.efforts = efforts.clone()


def make_action(batch=2):
    robot = FakeRobot(batch)
    env = SimpleNamespace(num_envs=batch, device="cpu", scene={"robot": robot})
    cfg = LegacyLSTMActionCfg(entity_name="robot", actuator_names=(".*",))
    return LegacyLSTMAction(cfg, env), robot


def test_lstm_four_substeps_matches_original_execution_chain():
    action, robot = make_action()
    reference = torch.jit.load(str(ACTUATOR_PATH)).eval()
    h, c = torch.zeros(2, 36, 8), torch.zeros(2, 36, 8)
    raw = torch.linspace(-120., 120., 36).reshape(2, 18)
    action.process_actions(raw)
    assert not action.q_ref.any()  # never the existing T1 reference
    torch.testing.assert_close(action.last_target, raw.clamp(-100., 100.) * .25)
    for substep in range(4):
        robot.data.joint_pos[:] = substep * .02
        robot.data.joint_vel[:] = torch.linspace(-2., 2., 36).reshape(2, 18)
        inp = torch.stack(((raw.clamp(-100., 100.) * .25 - robot.data.joint_pos).flatten(),
                           robot.data.joint_vel.flatten()), -1).unsqueeze(1)
        with torch.inference_mode():
            expected, (h, c) = reference(inp, (h, c))
        action.apply_actions()
        torch.testing.assert_close(robot.efforts, expected.reshape(2, 18), rtol=0, atol=0)
        torch.testing.assert_close(action.sea_hidden_state, h, rtol=0, atol=0)
        torch.testing.assert_close(action.sea_cell_state, c, rtol=0, atol=0)


def test_partial_reset_clears_only_selected_joint_hidden_rows():
    action, _ = make_action()
    action.process_actions(torch.ones(2, 18))
    action.apply_actions()
    retained = action.sea_hidden_state_per_env[:, 1].clone()
    retained_cell = action.sea_cell_state_per_env[:, 1].clone()
    action.reset(torch.tensor([0]))
    assert not action.sea_hidden_state_per_env[:, 0].any()
    assert not action.sea_cell_state_per_env[:, 0].any()
    torch.testing.assert_close(action.sea_hidden_state_per_env[:, 1], retained, rtol=0, atol=0)
    torch.testing.assert_close(action.sea_cell_state_per_env[:, 1], retained_cell, rtol=0, atol=0)
    assert not action.last_target[0].any()
    assert torch.all(action.last_target[1] == .25)


def test_network_output_is_not_pd_or_clipped_to_urdf_effort_limit():
    action, robot = make_action()

    class OversizedOutput:
        def __call__(self, inputs, states):
            return torch.full((36, 1), 123.), states

    action.actuator_network = OversizedOutput()
    action.process_actions(torch.zeros(2, 18))
    action.apply_actions()
    assert torch.all(robot.efforts == 123.)
