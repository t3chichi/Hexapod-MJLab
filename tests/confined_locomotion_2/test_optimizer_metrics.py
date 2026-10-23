from __future__ import annotations

import math
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from hexapod_mjlab.rl.confined_locomotion.distribution import IndependentDiagonalNormalProvider
from hexapod_mjlab.rl.confined_locomotion_2.ppo import TemporalPPOTrainer

_training_spec = importlib.util.spec_from_file_location(
    "confined_locomotion_2_training_fixtures", Path(__file__).with_name("test_training.py")
)
_training_module = importlib.util.module_from_spec(_training_spec)
_training_spec.loader.exec_module(_training_module)
TinyTemporalNet = _training_module.TinyTemporalNet
_buffer = _training_module._buffer


class InstrumentedNet(TinyTemporalNet):
    def __init__(self):
        super().__init__()
        self.frame_encoder = nn.Linear(1, 1, bias=False)
        self.temporal_encoder = nn.Linear(1, 1, bias=False)
        nn.init.constant_(self.frame_encoder.weight, 0.5)
        nn.init.constant_(self.temporal_encoder.weight, 0.25)

    def forward(self, observation, near, near_valid, far, far_valid, hidden, *, near_angle_keys=None, far_angle_keys=None):
        self.seen_hidden.append(hidden.detach().clone())
        self.seen_obs.append(observation.lin_vel_b.detach().clone())
        signal = self.frame_encoder(observation.lin_vel_b[:, :1])
        updated = torch.tanh(hidden + self.gain * signal + self.temporal_encoder(signal))
        mean = updated[:, :18] + self.bias
        return SimpleNamespace(
            frame_feature=signal.expand(-1, 187), temporal_feature=updated,
            hidden=updated, envelope=updated[:, :6], action_mean=mean,
            value=updated[:, :1],
        )


def _trainer(seed: int) -> tuple[TemporalPPOTrainer, InstrumentedNet]:
    net = InstrumentedNet()
    provider = IndependentDiagonalNormalProvider(device="cpu", generator_seed=seed)
    trainer = TemporalPPOTrainer(network=net, distribution_provider=provider, device="cpu", seed=17)
    return trainer, net


class OptimizerMetricTests(unittest.TestCase):
    def test_optimizer_metrics_are_finite_and_grad_norms_match_preclip_reference(self):
        captured: list[tuple[float, float, float]] = []
        original = torch.nn.utils.clip_grad_norm_
        trainer, net = _trainer(5)

        def capture_before_clip(parameters, max_norm, *args, **kwargs):
            params = list(parameters)
            all_norm = torch.sqrt(sum(p.grad.detach().pow(2).sum() for p in params if p.grad is not None))

            def module_norm(module):
                return torch.sqrt(sum(p.grad.detach().pow(2).sum() for p in module.parameters()
                                      if p.grad is not None))

            frame = module_norm(net.frame_encoder)
            temporal = module_norm(net.temporal_encoder)
            captured.append((float(all_norm), float(frame), float(temporal)))
            return original(params, max_norm, *args, **kwargs)

        torch.nn.utils.clip_grad_norm_ = capture_before_clip
        try:
            metrics = trainer.update(_buffer(torch.device("cpu")))
        finally:
            torch.nn.utils.clip_grad_norm_ = original

        assert len(captured) == 20
        first = captured[0]
        self.assertTrue(math.isclose(metrics["epoch_0_minibatch_0_preclip_grad_norm"], first[0], rel_tol=1e-6))
        self.assertTrue(math.isclose(metrics["epoch_0_minibatch_0_frame_encoder_preclip_grad_norm"], first[1], rel_tol=1e-6))
        self.assertTrue(math.isclose(metrics["epoch_0_minibatch_0_temporal_encoder_preclip_grad_norm"], first[2], rel_tol=1e-6))
        names = (
            "entropy", "symmetry_loss", "auxiliary_loss", "actor_std_mean", "actor_std_min",
            "actor_std_max", "ratio_clip_fraction", "preclip_grad_norm",
            "frame_encoder_preclip_grad_norm", "temporal_encoder_preclip_grad_norm",
            "value_explained_variance_valid",
        )
        for name in names:
            self.assertTrue(torch.isfinite(torch.tensor(metrics[f"epoch_0_minibatch_0_{name}"])))
            aggregate_name = name if name in ("actor_std_mean", "actor_std_min", "actor_std_max",
                                               "value_explained_variance_valid") else f"{name}_mean"
            self.assertTrue(torch.isfinite(torch.tensor(metrics[f"Optimizer/{aggregate_name}"])))
        self.assertEqual(metrics["Optimizer/value_explained_variance_valid"], 1.0)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["Optimizer/value_explained_variance_mean"])))
        self.assertGreaterEqual(metrics["Optimizer/ratio_clip_fraction_mean"], 0.0)
        self.assertLessEqual(metrics["Optimizer/ratio_clip_fraction_mean"], 1.0)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in captured[0]))


    def test_value_explained_variance_is_omitted_for_constant_targets(self):
        trainer, _ = _trainer(8)
        buffer = _buffer(torch.device("cpu"))
        for step in buffer.steps:
            step.reward.zero_()
            step.next_value.zero_()
        metrics = trainer.update(buffer)
        self.assertEqual(metrics["Optimizer/value_explained_variance_valid"], 0.0)
        self.assertNotIn("Optimizer/value_explained_variance_mean", metrics)
        self.assertNotIn("epoch_0_minibatch_0_value_explained_variance", metrics)


    def test_diagnostics_do_not_change_fixed_input_training_result(self):
        first, net_a = _trainer(12)
        second, net_b = _trainer(12)
        # Copy all initialization exactly, including distribution parameters.
        net_b.load_state_dict(net_a.state_dict())
        second.distribution_provider.load_state_dict(first.distribution_provider.state_dict())
        first.update(_buffer(torch.device("cpu")))
        second.update(_buffer(torch.device("cpu")))
        for left, right in zip(net_a.parameters(), net_b.parameters()):
            self.assertTrue(torch.equal(left, right))
        for left, right in zip(first.distribution_provider.parameters(), second.distribution_provider.parameters()):
            self.assertTrue(torch.equal(left, right))


def test_optimizer_metrics_are_finite_and_grad_norms_match_preclip_reference():
    OptimizerMetricTests().test_optimizer_metrics_are_finite_and_grad_norms_match_preclip_reference()


def test_value_explained_variance_is_omitted_for_constant_targets():
    OptimizerMetricTests().test_value_explained_variance_is_omitted_for_constant_targets()


def test_diagnostics_do_not_change_fixed_input_training_result():
    OptimizerMetricTests().test_diagnostics_do_not_change_fixed_input_training_result()
if __name__ == "__main__":
    unittest.main()
