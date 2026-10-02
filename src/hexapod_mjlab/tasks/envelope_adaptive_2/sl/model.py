"""Temporal GRU(190→187) + MLP(256,128→5), with optional memory probes."""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from .config import SLModelConfig

def _make_activation(name: str) -> nn.Module:
    key = (name or 'elu').lower()
    if key == 'elu':
        return nn.ELU()
    if key == 'relu':
        return nn.ReLU()
    if key == 'tanh':
        return nn.Tanh()
    if key == 'lrelu' or key == 'leaky_relu':
        return nn.LeakyReLU()
    raise ValueError(f'unsupported activation: {name!r}')

class EnvelopeNet(nn.Module):
    """GRU + MLP mapping a 190-D observation sequence to 5 envelope params."""

    def __init__(self, cfg: Optional[SLModelConfig]=None, **overrides):
        super().__init__()
        cfg = cfg or SLModelConfig()
        self.cfg = cfg
        rnn_type = (cfg.rnn_type or 'gru').lower()
        if rnn_type != 'gru':
            raise ValueError(f"only 'gru' is supported for export parity, got {rnn_type!r}")
        self.gru = nn.GRU(input_size=cfg.obs_dim, hidden_size=cfg.rnn_hidden_dim, num_layers=cfg.rnn_num_layers, batch_first=False)
        dims: List[int] = [cfg.rnn_hidden_dim, *list(cfg.actor_hidden_dims), cfg.action_dim]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(_make_activation(cfg.activation))
        self.mlp = nn.Sequential(*layers)
        self.aux_ks: List[int] = sorted((int(k) for k in cfg.aux_ks or []))
        if self.aux_ks:
            if cfg.aux_mode not in ('recall', 'forward'):
                raise ValueError(f'unknown aux_mode: {cfg.aux_mode!r}')
            self.aux_heads = nn.ModuleDict({str(k): nn.Linear(cfg.rnn_hidden_dim, cfg.action_dim) for k in self.aux_ks})
        else:
            self.aux_heads = None

    def forward(self, obs: torch.Tensor, hidden=None) -> torch.Tensor:
        """Full-sequence forward.

        Args:
            obs: ``(seq, batch, 190)``
        Returns:
            ``(seq, batch, 5)``
        """
        out, _ = self.gru(obs, hidden)
        return self.mlp(out)

    def forward_with_aux(self, obs: torch.Tensor, hidden=None):
        """Sequence forward that also returns the GRU hidden states.

        Training-only helper for the auxiliary memory loss.  Args and the main
        output match :meth:`forward`; additionally returns the pre-MLP hidden
        sequence ``(seq, batch, rnn_hidden_dim)`` so the aux probes read the
        GRU memory directly (never the MLP features).
        """
        out, _ = self.gru(obs, hidden)
        return (self.mlp(out), out)

    def step(self, obs: torch.Tensor, hidden=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single-frame forward used for deployment.

        Args:
            obs: ``(1, batch, 190)`` -- a single frame with a leading time axis.
            hidden: previous hidden state or ``None``.
        Returns:
            ``(pred, new_hidden)`` where ``pred`` is ``(batch, 5)``.
        """
        if obs.dim() != 3 or obs.shape[0] != 1:
            raise ValueError(f'step expects (1, batch, obs_dim), got {tuple(obs.shape)}')
        out, new_hidden = self.gru(obs, hidden)
        return (self.mlp(out[-1]), new_hidden)

    def init_hidden(self, batch_size: int, device=None) -> torch.Tensor:
        device = device or next(self.parameters()).device
        return torch.zeros(self.cfg.rnn_num_layers, batch_size, self.cfg.rnn_hidden_dim, device=device)

    def param_count(self) -> int:
        return int(sum((p.numel() for p in self.parameters())))
