from __future__ import annotations

import torch
from torch import nn

from hexapod_mjlab.rl.confined_locomotion_2.contracts import OnlineObservation
from hexapod_mjlab.rl.confined_locomotion_2.network import ConfinedLocomotionNetwork


def _network() -> ConfinedLocomotionNetwork:
    return ConfinedLocomotionNetwork(
        mlp_activation=nn.ELU(),
        init_fn=None,
        envelope_mode="raw",
        ground_samples=None,
    )


def _observation(batch: int = 1) -> OnlineObservation:
    device = torch.device("cpu")
    return OnlineObservation(
        env_ids=torch.arange(batch, dtype=torch.int64, device=device),
        episode_ids=torch.zeros(batch, dtype=torch.int64, device=device),
        step_ids=torch.zeros(batch, dtype=torch.int64, device=device),
        time_s=torch.zeros(batch, dtype=torch.float32, device=device),
        lin_vel_b=torch.zeros(batch, 3),
        ang_vel_b=torch.zeros(batch, 3),
        gravity_b=torch.zeros(batch, 3),
        joint_pos_rel_ref=torch.zeros(batch, 18),
        joint_vel=torch.zeros(batch, 18),
        prev_action=torch.zeros(batch, 18),
        command=torch.zeros(batch, 3),
    )


def _empty_points(batch: int = 1):
    return (
        torch.zeros(batch, 256, 3),
        torch.zeros(batch, 256, dtype=torch.bool),
        torch.zeros(batch, 128, 3),
        torch.zeros(batch, 128, dtype=torch.bool),
    )


class _CaptureFrame(nn.Module):
    def __init__(self):
        super().__init__()
        self.points = None
        self.valid = None

    def forward(self, points, valid):
        self.points = points.detach().clone()
        self.valid = valid.detach().clone()
        return points.new_zeros(points.shape[0], 187)


def test_global_angle_sort_is_stable_across_near_and_far() -> None:
    model = _network()
    capture = _CaptureFrame()
    model.frame_encoder = capture
    near, nvalid, far, fvalid = _empty_points()
    near[0, 0] = torch.tensor([1.0, 0.0, 0.0])
    near[0, 1] = torch.tensor([2.0, 0.0, 0.0])
    far[0, 0] = torch.tensor([3.0, 0.0, 0.0])
    far[0, 1] = torch.tensor([4.0, 0.0, 0.0])
    nvalid[0, :2] = True
    fvalid[0, :2] = True
    near_keys = torch.zeros(1, 256)
    far_keys = torch.zeros(1, 128)
    near_keys[0, 0], near_keys[0, 1] = 0.5, 0.2
    far_keys[0, 0], far_keys[0, 1] = 0.2, 0.3

    model.encode_points(
        near, nvalid, far, fvalid,
        near_angle_keys=near_keys,
        far_angle_keys=far_keys,
    )

    # The equal key at 0.2 stays stable: near points precede far points.
    expected = torch.tensor([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0],
                             [4.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    torch.testing.assert_close(capture.points[0, :4], expected)
    assert capture.valid[0, :4].all()
    assert not capture.valid[0, 4:].any()


def test_frame_gru_resets_for_each_scan() -> None:
    model = _network()
    near, nvalid, far, fvalid = _empty_points()
    near[0, :3] = torch.tensor([[0.2, 0.1, 0.0], [0.4, 0.0, 0.1], [0.6, -0.1, 0.0]])
    nvalid[0, :3] = True
    first = model.encode_points(near, nvalid, far, fvalid)
    second = model.encode_points(near, nvalid, far, fvalid)
    torch.testing.assert_close(first, second)


def test_explicit_temporal_hidden_carries_memory() -> None:
    model = _network()
    with torch.no_grad():
        for parameter in model.temporal_encoder.parameters():
            parameter.zero_()
    frame = torch.zeros(1, 187)
    zero = torch.zeros(1, 187)
    ones = torch.ones(1, 187)
    out_zero = model.forward_from_features(_observation(), frame, zero)
    out_ones = model.forward_from_features(_observation(), frame, ones)
    torch.testing.assert_close(out_zero.temporal_feature, torch.zeros_like(frame))
    torch.testing.assert_close(out_ones.temporal_feature, torch.full_like(frame, 0.5))
    torch.testing.assert_close(out_ones.hidden, out_ones.temporal_feature)


def test_actor_detaches_envelope_but_envelope_loss_trains_its_head() -> None:
    model = _network()
    near, nvalid, far, fvalid = _empty_points()
    near[0, 0, 0] = 0.3
    near.requires_grad_()
    nvalid[0, 0] = True
    hidden = model.initial_hidden(1, "cpu").requires_grad_()
    out = model(
        _observation(), near, nvalid, far, fvalid,
        hidden,
    )
    out.envelope.retain_grad()
    out.action_mean.sum().backward()
    assert out.envelope.grad is None
    assert model.morphology_head.terminal.weight.grad is None
    assert any(p.grad is not None for p in model.frame_encoder.parameters())
    assert hidden.grad is not None

    model.zero_grad(set_to_none=True)
    near.grad = None
    hidden.grad = None
    out = model(
        _observation(), near, nvalid, far, fvalid,
        model.initial_hidden(1, "cpu"),
    )
    out.envelope.sum().backward()
    assert model.morphology_head.terminal.weight.grad is not None
    assert bool((model.morphology_head.terminal.weight.grad != 0).any())


def test_shape_and_initialization_contract() -> None:
    model = _network()
    assert model.morphology_head.fc1.in_features == 250
    assert model.action_head.hidden[0].in_features == 259
    assert model.value_head.output.out_features == 1
    assert torch.count_nonzero(model.morphology_head.terminal.weight) == 0
    expected_bias = model.mapping.head_terminal_bias
    torch.testing.assert_close(
        model.morphology_head.terminal.bias,
        torch.full((6,), expected_bias),
    )
    try:
        ConfinedLocomotionNetwork(
            mlp_activation=nn.ELU(), init_fn=None,
            envelope_mode="raw", ground_samples=8,
        )
    except ValueError as exc:
        assert "ground_samples" in str(exc)
    else:
        raise AssertionError("a ground head must not be constructible")
