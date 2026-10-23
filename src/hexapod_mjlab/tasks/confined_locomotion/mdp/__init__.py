"""Task-local MDP for confined locomotion.

Stock mjlab mdp names are re-exported for config convenience; task-local
terms live in the submodules. No old-task lidar/commands/terrain symbols.
"""

from mjlab.envs.mdp import *  # noqa: F401,F403
from mjlab.envs.mdp import dr as dr  # noqa: F401
from mjlab.managers.scene_entity_config import SceneEntityCfg  # noqa: F401

from .actions import ResidualPDAction, ResidualPDActionCfg  # noqa: F401
from .events import (  # noqa: F401
  clear_action_runtime,
  reset_joints_witness,
  reset_root_witness,
)
from .observations import clipped_prev_action  # noqa: F401
