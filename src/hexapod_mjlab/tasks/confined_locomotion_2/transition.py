"""Shared sealed MDP transition boundary for flat task version 2.

The callbacks are unchanged and remain valid because they bind per environment
instance. This module keeps the task-local seam explicit while reusing the
verified transition capture implementation.
"""

from hexapod_mjlab.tasks.confined_locomotion.transition import (
    PERSISTENT_PHYSICS_FIELDS,
    TransitionContext,
    get_transition_context,
    physics_field_manifest,
    require_finite,
)

__all__ = [
    "PERSISTENT_PHYSICS_FIELDS",
    "TransitionContext",
    "get_transition_context",
    "physics_field_manifest",
    "require_finite",
]
