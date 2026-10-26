"""CPU checks for the optional Terrain Memory height reconstruction loss."""
from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import torch

from hexapod_mjlab.rl.confined_locomotion_2.auxiliary import (
    HeightReconstructionAuxiliary,
    masked_height_mse,
)
from hexapod_mjlab.rl.confined_locomotion_2.contracts import HeightLabels


@contextmanager
def _raises(*exception_types):
    try:
        yield
    except exception_types:
        return
    raise AssertionError(f"expected one of {[item.__name__ for item in exception_types]}")


def _observation(*, step: int = 0, episode: int = 0, env: int = 0):
    return SimpleNamespace(
        env_ids=torch.tensor([env], dtype=torch.int64),
        episode_ids=torch.tensor([episode], dtype=torch.int64),
        step_ids=torch.tensor([step], dtype=torch.int64),
    )


def _label(*, step: int = 0, episode: int = 0, env: int = 0, heights=None, valid=None):
    heights = torch.zeros((1, 11, 17), dtype=torch.float32) if heights is None else heights
    valid = torch.ones((1, 11, 17), dtype=torch.bool) if valid is None else valid
    return HeightLabels(
        env_ids=torch.tensor([env], dtype=torch.int64),
        episode_ids=torch.tensor([episode], dtype=torch.int64),
        step_ids=torch.tensor([step], dtype=torch.int64),
        heights=heights,
        valid=valid,
        query_xy_w=torch.zeros((1, 11, 17, 2), dtype=torch.float32),
        root_pos_w=torch.zeros((1, 3), dtype=torch.float32),
        grid_identity={"nx": 17, "ny": 11},
        terrain_identity="height-aux-test",
    )


def _output(prediction):
    return SimpleNamespace(height_prediction=prediction)


def _step(label, observation=None):
    return SimpleNamespace(height_label=label, observation=observation or _observation())


def test_masked_mse_averages_per_sample_not_per_cell():
    prediction = torch.zeros((1, 2, 11, 17), dtype=torch.float32)
    target = torch.zeros_like(prediction)
    valid = torch.zeros_like(prediction, dtype=torch.bool)
    valid[0, 0, 0, 0] = True
    prediction[0, 0, 0, 0] = 2.0
    valid[0, 1] = True
    prediction[0, 1] = 1.0

    result = masked_height_mse(prediction, target, valid, torch.ones((1, 2), dtype=torch.bool))

    assert torch.allclose(result, torch.tensor(2.5))
    pooled_by_cell = (4.0 + 187.0) / 188.0
    assert not torch.isclose(result, torch.tensor(pooled_by_cell))


def test_loss_mask_excludes_unselected_samples():
    prediction = torch.zeros((1, 2, 11, 17), dtype=torch.float32)
    prediction[0, 1] = 100.0
    target = torch.zeros_like(prediction)
    valid = torch.ones_like(prediction, dtype=torch.bool)

    result = masked_height_mse(prediction, target, valid, torch.tensor([[True, False]]))

    assert result.item() == 0.0


def test_all_invalid_returns_differentiable_scalar_zero():
    prediction = torch.randn((2, 3, 11, 17), dtype=torch.float32, requires_grad=True)
    target = torch.zeros_like(prediction)
    valid = torch.zeros_like(prediction, dtype=torch.bool)

    result = masked_height_mse(prediction, target, valid, torch.ones((2, 3), dtype=torch.bool))
    result.backward()

    assert result.ndim == 0 and result.item() == 0.0 and prediction.grad is not None
    assert torch.count_nonzero(prediction.grad) == 0


def test_invalid_target_values_never_enter_loss_or_gradient():
    prediction = torch.zeros((1, 1, 11, 17), dtype=torch.float32, requires_grad=True)
    target = torch.full_like(prediction, 1.0e20)
    target[0, 0, 0, 0] = 2.0
    target[0, 0, 0, 1] = float("nan")
    valid = torch.zeros_like(prediction, dtype=torch.bool)
    valid[0, 0, 0, 0] = True

    result = masked_height_mse(prediction, target, valid, torch.ones((1, 1), dtype=torch.bool))
    result.backward()

    assert result.item() == 4.0
    assert prediction.grad[0, 0, 0, 0].item() == -4.0
    assert torch.count_nonzero(prediction.grad[~valid]) == 0


def test_minibatch_gradient_only_reaches_selected_valid_cells():
    prediction = torch.ones((1, 2, 11, 17), dtype=torch.float32, requires_grad=True)
    target = torch.zeros_like(prediction)
    valid = torch.zeros_like(prediction, dtype=torch.bool)
    valid[0, 0, 0, 0] = True
    valid[0, 1, 0, 0] = True

    masked_height_mse(prediction, target, valid, torch.tensor([[True, False]])).backward()

    assert prediction.grad[0, 0, 0, 0].item() == 2.0
    assert torch.count_nonzero(prediction.grad) == 1


def test_unselected_extreme_finite_labels_do_not_create_nan_gradients():
    prediction = torch.ones((1, 2, 11, 17), dtype=torch.float32, requires_grad=True)
    target = torch.zeros_like(prediction)
    target[0, 1] = 3.0e38
    valid = torch.ones_like(prediction, dtype=torch.bool)
    loss_mask = torch.tensor([[True, False]])

    result = masked_height_mse(prediction, target, valid, loss_mask)
    result.backward()

    assert result.item() == 1.0
    assert torch.isfinite(prediction.grad).all()
    assert torch.count_nonzero(prediction.grad[0, 1]) == 0


def test_masked_mse_rejects_invalid_shapes_dtypes_devices_and_values():
    prediction = torch.zeros((1, 1, 11, 17), dtype=torch.float32)
    target = torch.zeros_like(prediction)
    valid = torch.ones_like(prediction, dtype=torch.bool)
    loss_mask = torch.ones((1, 1), dtype=torch.bool)
    bad_inputs = (
        (prediction.to(torch.float64), target, valid, loss_mask, TypeError),
        (prediction, target.to(torch.float64), valid, loss_mask, TypeError),
        (prediction, target, valid.to(torch.float32), loss_mask, TypeError),
        (prediction, target, valid, loss_mask.to(torch.float32), TypeError),
        (prediction[..., :-1], target, valid, loss_mask, ValueError),
        (prediction, target[..., :-1], valid, loss_mask, ValueError),
        (prediction, target, valid[..., :-1], loss_mask, ValueError),
        (prediction, target, valid, torch.ones((1,), dtype=torch.bool), ValueError),
        (prediction, torch.empty((1, 1, 11, 17), dtype=torch.float32, device="meta"),
         valid, loss_mask, ValueError),
    )
    for bad_prediction, bad_target, bad_valid, bad_mask, error in bad_inputs:
        with _raises(error):
            masked_height_mse(bad_prediction, bad_target, bad_valid, bad_mask)

    invalid_target = target.clone()
    invalid_target[0, 0, 0, 0] = float("inf")
    with _raises(ValueError):
        masked_height_mse(prediction, invalid_target, valid, loss_mask)
    invalid_prediction = prediction.clone()
    invalid_prediction[0, 0, 0, 0] = float("nan")
    with _raises(ValueError):
        masked_height_mse(invalid_prediction, target, valid, loss_mask)


def test_auxiliary_averages_original_and_y_mirrored_branches():
    target = torch.zeros((1, 11, 17), dtype=torch.float32)
    target[0, 0, :] = 1.0
    label = _label(heights=target, valid=torch.ones_like(target, dtype=torch.bool))
    observation = _observation()
    original = target.unsqueeze(0).clone()
    mirrored_prediction = torch.zeros_like(original)
    plugin = HeightReconstructionAuxiliary(weight=3.0)

    result = plugin(
        [_output(original[0])],
        {
            "loss_mask": torch.ones((1, 1), dtype=torch.bool),
            "steps": [_step(label, observation)],
            "mirrored_outputs": [_output(mirrored_prediction[0])],
        },
    )

    original_mse = torch.tensor(0.0)
    mirrored_mse = torch.tensor(17.0 / 187.0)
    assert torch.allclose(result.metrics["Height/original_mse_m2"], original_mse)
    assert torch.allclose(result.metrics["Height/mirrored_mse_m2"], mirrored_mse)
    assert torch.allclose(result.metrics["Height/mse_m2"], 0.5 * (original_mse + mirrored_mse))
    assert torch.allclose(result.loss, torch.tensor(51.0 / 374.0))
    assert torch.allclose(result.metrics["Height/weighted_loss"], result.loss.detach())
    assert result.metrics["Height/valid_sample_fraction"].item() == 1.0
    assert result.metrics["Height/valid_cell_fraction"].item() == 1.0


def test_auxiliary_uses_loss_mask_and_handles_no_eligible_samples():
    outputs = [_output(torch.zeros((2, 11, 17))), _output(torch.full((2, 11, 17), 50.0))]
    # One sequence step contains B=2 rows; HeightLabels are built separately below
    # as a compact batch-compatible label matching that observation.
    batch_heights = torch.zeros((2, 11, 17), dtype=torch.float32)
    batch_valid = torch.ones((2, 11, 17), dtype=torch.bool)
    batch_label = HeightLabels(
        env_ids=torch.tensor([0, 1], dtype=torch.int64),
        episode_ids=torch.zeros(2, dtype=torch.int64),
        step_ids=torch.zeros(2, dtype=torch.int64),
        heights=batch_heights,
        valid=batch_valid,
        query_xy_w=torch.zeros((2, 11, 17, 2), dtype=torch.float32),
        root_pos_w=torch.zeros((2, 3), dtype=torch.float32),
        grid_identity={"nx": 17, "ny": 11},
        terrain_identity="height-aux-test",
    )
    batch_observation = SimpleNamespace(
        env_ids=torch.tensor([0, 1], dtype=torch.int64),
        episode_ids=torch.zeros(2, dtype=torch.int64),
        step_ids=torch.zeros(2, dtype=torch.int64),
    )
    plugin = HeightReconstructionAuxiliary(weight=2.0)
    result = plugin(
        outputs,
        {"loss_mask": torch.tensor([[True, False], [False, False]]),
         "steps": [_step(batch_label, batch_observation), _step(batch_label, batch_observation)]},
    )
    assert result.loss.item() == 0.0
    assert result.metrics["Height/valid_sample_fraction"].item() == 1.0

    all_invalid = HeightLabels(
        **{**vars(batch_label), "heights": torch.zeros_like(batch_heights),
           "valid": torch.zeros_like(batch_valid)}
    )
    no_valid = plugin(
        [_output(torch.zeros((2, 11, 17), requires_grad=True)),
         _output(torch.ones((2, 11, 17), requires_grad=True))],
        {"loss_mask": torch.ones((2, 2), dtype=torch.bool),
         "steps": [_step(all_invalid, batch_observation), _step(all_invalid, batch_observation)]},
    )
    no_valid.loss.backward()
    assert no_valid.loss.item() == 0.0


def test_auxiliary_validates_label_identity_and_owns_no_parameters():
    plugin = HeightReconstructionAuxiliary(weight=0.25)
    label = _label()
    output = [_output(torch.zeros((1, 11, 17), dtype=torch.float32, requires_grad=True))]
    batch = {
        "loss_mask": torch.ones((1, 1), dtype=torch.bool),
        "steps": [_step(label, _observation(step=1))],
    }
    with _raises(ValueError):
        plugin(output, batch)
    assert not isinstance(plugin, torch.nn.Module)
    assert vars(plugin) == {"weight": 0.25}
    for bad_weight in (0.0, -1.0, float("inf"), float("nan"), True):
        with _raises(TypeError, ValueError):
            HeightReconstructionAuxiliary(weight=bad_weight)



def test_prepared_height_targets_preserve_masked_mirror_loss_and_gradients():
    target = torch.randn(1, 11, 17)
    valid = torch.zeros_like(target, dtype=torch.bool)
    valid[:, 2, :3] = True
    target[~valid] = 0.  # HeightLabels requires zero in invalid geometry cells.
    steps = [_step(_label(heights=target, valid=valid)),
             _step(_label(step=1, heights=torch.full_like(target, 3.e38)), _observation(step=1))]
    plugin = HeightReconstructionAuxiliary(2.)
    prepared = plugin.prepare_targets(steps, mirrored=True)
    mask = torch.tensor([[True], [False]])
    original = torch.randn(2, 1, 11, 17)
    mirror = torch.randn_like(original)
    results, gradients = [], []
    for use_prepared in (False, True):
        prediction = original.clone().requires_grad_()
        mirrored_prediction = mirror.clone().requires_grad_()
        batch = {'steps': steps, 'loss_mask': mask,
                 'mirrored_outputs': [_output(x) for x in mirrored_prediction]}
        if use_prepared:
            batch['height_targets'] = prepared
        result = plugin([_output(x) for x in prediction], batch)
        result.loss.backward()
        results.append(result)
        gradients.append((prediction.grad, mirrored_prediction.grad))
    torch.testing.assert_close(results[0].loss, results[1].loss, rtol=0, atol=0)
    for name in results[0].metrics:
        torch.testing.assert_close(results[0].metrics[name], results[1].metrics[name], rtol=0, atol=0)
    for left, right in zip(*gradients):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        assert torch.isfinite(right).all() and torch.count_nonzero(right[1]) == 0
    bad = _step(_label(step=1), _observation(step=0))
    with _raises(ValueError):
        plugin.prepare_targets([bad], mirrored=True)
