"""Task-specific runner, checkpointing, and official training entry."""

from .runner import ConfinedLocomotionRunner
from .entry import (
  FormalLaunchResult,
  FormalTrainingConfigError,
  build_formal_train_config,
  launch_formal_training,
  preflight_formal_train_config,
)

__all__ = [
  "ConfinedLocomotionRunner",
  "FormalLaunchResult",
  "FormalTrainingConfigError",
  "build_formal_train_config",
  "launch_formal_training",
  "preflight_formal_train_config",
]
