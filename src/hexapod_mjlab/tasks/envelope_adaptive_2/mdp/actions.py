"""Envelope actions drive one logical frame before rewards and observations."""
from dataclasses import dataclass

import torch
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg


@dataclass(kw_only=True)
class EnvelopeActionCfg(ActionTermCfg):
    def build(self, env):
        return EnvelopeAction(self, env)


class EnvelopeAction(ActionTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._raw = torch.zeros((env.num_envs, 5), device=env.device)

    @property
    def action_dim(self):
        return 5

    @property
    def raw_action(self):
        return self._raw

    def process_actions(self, actions):
        self._raw.copy_(actions.clamp(-100, 100))

    def apply_actions(self):
        self._env.ea2.advance(self._raw)
        if hasattr(self._env.ea2, "base_pos"):
            from ..env import write_carrier_pose
            write_carrier_pose(self._env)

    def reset(self, env_ids=None):
        self._raw[slice(None) if env_ids is None else env_ids] = 0
