"""T0-A isolated flat-ground variant of the EL4090 LiDAR task.

Members:
* :mod:`witness` -- frozen T1 ``height_0.48_width_1.00`` witness + lift math;
* :mod:`flat_env_cfg` -- deterministic flat configuration
  (``El-4090-Lidar-T0-Flat``);
* :mod:`diagnostics` -- per-physics-substep contact / torque recorder and
  state probes used by the T0 gates and the evaluation entry point.
"""

from . import diagnostics, flat_env_cfg, witness  # noqa: F401
from .flat_env_cfg import (  # noqa: F401
    T0_EVAL_COMMANDS,
    T0_TASK_ID,
    T0_TERRAIN_FRICTION,
    T0_TRAIN_VX_RANGE,
    make_el_4090_t0_flat_env_cfg,
)
