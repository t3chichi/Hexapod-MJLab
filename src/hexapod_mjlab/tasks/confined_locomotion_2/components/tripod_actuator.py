"""Original AnyDrive LSTM, called once per 0.005 s physical substep.

The reference ElSpider._compute_torques returns the network output directly;
the PD branch's URDF effort clamp is not executed. Consequently this action
does not clamp to 80 N m. The environment adapter must explicitly configure
MuJoCo actuator/control limits and document the unresolved PhysX limit
semantics. An 80 N m limit in the URDF alone is not evidence of a Python clamp.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from hexapod_mjlab.common.mdp.actions import ResidualPDAction, ResidualPDActionCfg

ACTUATOR_PATH = Path(__file__).parents[1] / "assets" / "actuators" / "anydrive_v3_lstm.pt"


@dataclass(kw_only=True)
class LegacyLSTMActionCfg(ResidualPDActionCfg):
    """Keep the transition capture interface while replacing PD by the LSTM."""

    actuator_path: str = str(ACTUATOR_PATH)
    raw_clip: float = 100.
    residual_scale: float = .25
    effort_limit: float = float("inf")

    def build(self, env):
        return LegacyLSTMAction(self, env)


class LegacyLSTMAction(ResidualPDAction):
    cfg: LegacyLSTMActionCfg
    _tripod_checkpoint_fields = ("sea_hidden_state", "sea_cell_state", "last_effort")

    def __init__(self, cfg: LegacyLSTMActionCfg, env):
        if cfg.raw_clip != 100. or cfg.residual_scale != .25 or cfg.effort_limit != float("inf"):
            raise ValueError("legacy LSTM action must use clip=100, scale=.25 and no effort clamp")
        super().__init__(cfg, env)
        self._q_ref.zero_()
        self.last_target.zero_()
        self._processed_actions.zero_()
        self.actuator_network = torch.jit.load(cfg.actuator_path, map_location=env.device)
        self.actuator_network.eval()
        n, joints = self._q_ref.shape
        self.sea_input = torch.zeros(n * joints, 1, 2, device=env.device)
        self.sea_hidden_state = torch.zeros(2, n * joints, 8, device=env.device)
        self.sea_cell_state = torch.zeros_like(self.sea_hidden_state)
        self.last_effort = torch.zeros_like(self._q_ref)

    @property
    def sea_hidden_state_per_env(self):
        return self.sea_hidden_state.view(2, *self._q_ref.shape, 8)

    @property
    def sea_cell_state_per_env(self):
        return self.sea_cell_state.view(2, *self._q_ref.shape, 8)

    def apply_actions(self) -> None:
        target = self._check_joint_tensor("joint_target", self._processed_actions)
        q = self._check_joint_tensor("joint_pos", self._entity.data.joint_pos[:, self._target_ids])
        qdot = self._check_joint_tensor("joint_vel", self._entity.data.joint_vel[:, self._target_ids])
        # The original flattens env-major joint rows and presents a single
        # sequence element [target-q, qdot] to the shared per-joint LSTM.
        with torch.inference_mode():
            self.sea_input[:, 0, 0] = (target - q).flatten()
            self.sea_input[:, 0, 1] = qdot.flatten()
            efforts, (hidden, cell) = self.actuator_network(
                self.sea_input, (self.sea_hidden_state, self.sea_cell_state))
            self.sea_hidden_state.copy_(hidden)
            self.sea_cell_state.copy_(cell)
            self.last_effort.copy_(efforts.reshape_as(self._q_ref))
        self._check_finite_computed("lstm_effort", self.last_effort)
        self._entity.set_joint_effort_target(self.last_effort, joint_ids=self._target_ids)
        if self.cfg.record_substeps and self.substep_records:
            rec = self.substep_records[-1]
            rec["q"].append(q.detach().cpu().numpy().copy())
            rec["qdot"].append(qdot.detach().cpu().numpy().copy())
            # Preserve existing recording keys; here both denote the same
            # unclipped LSTM output, not a computed PD torque.
            rec["tau_preclip"].append(self.last_effort.detach().cpu().numpy().copy())
            rec["tau_applied"].append(self.last_effort.detach().cpu().numpy().copy())
            rec["saturated"].append(torch.zeros_like(self.last_effort, dtype=torch.bool).cpu().numpy())

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        # Views preserve env-major flattening without constructing a possibly
        # reordered list of flattened joints on partial resets.
        if hasattr(self, "sea_hidden_state"):
            self.sea_hidden_state_per_env[:, ids] = 0.
            self.sea_cell_state_per_env[:, ids] = 0.
            self.sea_input.view(*self._q_ref.shape, 1, 2)[ids] = 0.
            self.last_effort[ids] = 0.
