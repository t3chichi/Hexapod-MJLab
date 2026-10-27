"""Frozen EL4090 T1 witness data shared by the formal locomotion tasks.

The values are copied verbatim from the T1 acceptance record. The fixed-channel
catalog identity hashes :data:`FROZEN_WITNESS_PATH`, a byte-identical snapshot
of the historical T0-A witness module; do not edit either copy without
reissuing every fixed-channel selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

WITNESS_ID = "height_0.48_width_1.00"

#: Root height the robot settled at after the 5 s T1 hold [m].
SETTLED_ROOT_HEIGHT_M = 0.4598590386845707

#: Full settled qpos (root pos xyz + quat wxyz + 18 joints, model order).
SETTLED_QPOS = (
    0.00016009242037980125,
    -0.004903351360774704,
    0.4598590386845707,
    0.9999986922984655,
    0.000487646644791755,
    0.00042902361577461544,
    -0.0014810607169152116,
    8.813684464595729e-05,
    -0.09481412023889756,
    -0.014639117731400381,
    8.605044304574133e-05,
    -0.09363550575120877,
    -0.014249306969756154,
    9.360685248402362e-05,
    -0.09310449096922903,
    -0.012301203213651196,
    0.00010591199444261626,
    -0.09444800737185956,
    -0.013547858358561376,
    0.00010549257907957825,
    -0.09234588175608953,
    -0.01283570359452122,
    8.848715324420229e-05,
    -0.08842676594644978,
    -0.012781334350785645,
)

#: PD targets held during the T1 hold (per joint, model order).
JOINT_TARGETS = (
    9.389576513876847e-09,
    -0.0046310019899056375,
    0.0032190703431012983,
    9.389576513876847e-09,
    -0.004631001989905635,
    0.0032190703431012975,
    7.090640413418786e-09,
    -0.0034315511220239557,
    0.002386943721151135,
    -9.691385120622772e-09,
    -0.0034311741225189502,
    0.002386681870373368,
    -9.755299049821659e-09,
    -0.003453816541361166,
    0.0024024224646846054,
    -1.2881559322243197e-08,
    -0.004647833558627842,
    0.0032307610315835825,
)

#: T1 hold PD constants (declared simulation assumptions, not hardware).
PD_KP = 130.0
PD_KD = 2.0
TORQUE_CLIP_NM = 80.0

#: Minimum foot geometry clearance before release [m] (plan: >= 1 mm).
FOOT_CLEARANCE_M = 0.002

#: Body names whose geometry counts as feet (model foot bodies).
FOOT_BODIES = (
    "RF_FOOT",
    "RM_FOOT",
    "RB_FOOT",
    "LF_FOOT",
    "LM_FOOT",
    "LB_FOOT",
)


@dataclass(frozen=True)
class T1Witness:
    settled_qpos: tuple
    joint_targets: tuple
    settled_root_height_m: float
    joint_names: tuple

    @property
    def root_pos(self) -> tuple:
        return tuple(self.settled_qpos[0:3])

    @property
    def root_quat_wxyz(self) -> tuple:
        return tuple(self.settled_qpos[3:7])

    @property
    def joint_pos(self) -> tuple:
        return tuple(self.settled_qpos[7:25])


#: Joint order of the robot MJCF (matches qpos[7:25] order).
JOINT_ORDER = (
    "RF_HAA",
    "RF_HFE",
    "RF_KFE",
    "RM_HAA",
    "RM_HFE",
    "RM_KFE",
    "RB_HAA",
    "RB_HFE",
    "RB_KFE",
    "LF_HAA",
    "LF_HFE",
    "LF_KFE",
    "LM_HAA",
    "LM_HFE",
    "LM_KFE",
    "LB_HAA",
    "LB_HFE",
    "LB_KFE",
)


def frozen_witness() -> T1Witness:
    return T1Witness(
        settled_qpos=SETTLED_QPOS,
        joint_targets=JOINT_TARGETS,
        settled_root_height_m=SETTLED_ROOT_HEIGHT_M,
        joint_names=JOINT_ORDER,
    )


#: Immutable identity anchor; byte-identical to the historical T0-A module.
FROZEN_WITNESS_PATH = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "witness"
    / "t1_witness_v1.py.frozen"
)
