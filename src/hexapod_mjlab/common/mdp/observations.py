"""Task-local observation terms for confined locomotion."""

from __future__ import annotations

import torch


def clipped_prev_action(env, action_name: str = "residual_pd") -> torch.Tensor:
  """(B, 18) previous control step's CLIPPED dimensionless action.

  Read AFTER the previous step's action processing (the standard
  observation-computation point), so each observation row carries the
  clipped action of the step before it — per the confirmed prev_action
  semantics (clipped ±100 value; no ×0.25, no q_ref offset).
  """
  term = env.action_manager.get_term(action_name)
  return term.prev_action
