"""Shared v2-family action distribution, using the audited direct-std recipe."""

import torch
from torch import nn

from hexapod_mjlab.rl.confined_locomotion.distribution import (
    DiagonalNormalDistribution, DistributionContractError,
    IndependentDiagonalNormalProvider,
)


class _DirectStdNormal(DiagonalNormalDistribution):
    def __init__(self, *, mean, std, generator):
        super().__init__(mean=mean, log_std=std.log(), generator=generator)
        self._std_parameter = std

    @property
    def std(self):
        return self._std_parameter.expand_as(self._mean)

    stddev = std


class DirectStdNormalProvider(IndependentDiagonalNormalProvider):
    """18 independent Normal actions; optimize std directly, initialized to one.

    Keep the established named PRNG interface and finite/positive checks.
    There is no new clamping, squashing or observation-dependent variance.
    """

    def __init__(self, *, device=None, generator_seed: int):
        super().__init__(device=device, generator_seed=generator_seed, initial_std=1.0)
        resolved = self.log_std.device
        del self.log_std
        self.std = nn.Parameter(torch.ones(18, dtype=torch.float32, device=resolved))

    def forward(self, action_mean):
        if (not isinstance(action_mean, torch.Tensor) or action_mean.ndim != 2
                or action_mean.shape[1] != 18 or action_mean.shape[0] < 1
                or action_mean.dtype != torch.float32 or action_mean.device != self.std.device):
            raise DistributionContractError("action_mean must be (B>=1,18) float32 on provider device")
        self._require_generator_device(action_mean.device)
        if (not bool(torch.isfinite(action_mean).all())
                or not bool(torch.isfinite(self.std).all()) or bool((self.std <= 0).any())):
            raise DistributionContractError("Normal mean/std must be finite and std positive")
        return _DirectStdNormal(mean=action_mean, std=self.std, generator=self._generator)

    def distribution_identity(self):
        return {
            "schema": "confined_locomotion_2.action_normal/v2",
            "family": "independent_diagonal_normal", "action_dim": 18,
            "initial_std": 1.0, "parameterization": "direct_std",
            "std_observation_dependent": False, "sample_generator_stream": "policy_action",
        }


__all__ = ["DirectStdNormalProvider"]
