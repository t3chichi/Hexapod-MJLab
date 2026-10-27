"""T1 `height_0.48_width_1.00` witness: frozen data + lifted-root computation.

Source of truth: ``docs/research/morphology/evidence/library.json`` and
``holds.json`` (record id ``height_0.48_width_1.00``, validation level
``scripted_slow_crawl`` — the highest tier in the library).

The witness distinguishes three things the T0 flat configuration must not
conflate:

* **joint PD targets** (``joint_targets``): the near-zero IK targets that the
  T1 hold commanded (max |value| 4.6e-3 rad); these become the T0 action
  default offset so a zero action reproduces the T1 hold controller;
* **settled joint state** (``measured_qpos``): where the robot actually came
  to rest after the 5 s hold (HFE sags to ≈ −0.095 rad under gravity);
* **actual hold root height** (``settled_root_height``): the T0 temporary
  height semantics for the base-height reward and evaluation (0.4599 m). It
  is a debugging reference, NOT a finalized paper z₀, and NOT the 0.48 m
  grid name's exact end state.

Reset protocol: start from the settled witness state, lift the root so the
LOWEST POINT of the actual foot collision geometry clears the ground by
``foot_clearance_m`` (≥ 1 mm per the plan; default 2 mm), then let it settle.
The lift is computed from compiled collision meshes, never from foot-origin
kinematics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

WITNESS_ID = "height_0.48_width_1.00"
T1_LIBRARY_REL = Path("docs/research/morphology/evidence/library.json")
T1_HOLDS_REL = Path("logs/morphology/acceptance-v3-20261003/holds.json")

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


def get_witness_summary() -> dict:
    w = frozen_witness()
    return {
        "id": WITNESS_ID,
        "hold_height_m": w.settled_root_height_m,
        "n_qpos": len(w.settled_qpos),
        "joint_targets_max_abs": max(abs(v) for v in w.joint_targets),
        "settled_joint_max_abs": max(abs(v) for v in w.joint_pos),
        "pd_kp": PD_KP,
        "pd_kd": PD_KD,
        "torque_clip_Nm": TORQUE_CLIP_NM,
        "foot_clearance_m": FOOT_CLEARANCE_M,
    }


def verify_against_library(repo_root: Path) -> dict:
    """Cross-check the frozen copy against the T1 acceptance library.

    Returns a report dict; ``match`` is False when the library is missing or
    disagrees with the frozen values.
    """
    from ..el_4090_lidar_env_cfg import _EL4090_XML  # noqa: PLC0415

    lib_path = repo_root / T1_LIBRARY_REL
    report: dict = {"library": str(lib_path), "exists": lib_path.exists()}
    if not lib_path.exists():
        report["match"] = False
        report["reason"] = "library.json not found"
        return report
    with open(lib_path, encoding="utf-8") as f:
        lib = json.load(f)
    rec = next((r for r in lib["records"] if r["id"] == WITNESS_ID), None)
    if rec is None:
        report["match"] = False
        report["reason"] = f"record {WITNESS_ID} missing"
        return report
    w = frozen_witness()
    qpos_ok = all(abs(a - b) <= 1e-15 for a, b in zip(rec["measured_qpos"], w.settled_qpos))
    tgt_ok = True
    for name in JOINT_ORDER:
        if abs(rec["joint_targets"][name] - w.joint_targets[JOINT_ORDER.index(name)]) > 1e-15:
            tgt_ok = False
    height_ok = abs(rec["measured_E"][5] - w.settled_root_height_m) <= 1e-12
    report.update(
        {
            "qpos_match": qpos_ok,
            "joint_targets_match": tgt_ok,
            "hold_height_match": height_ok,
            "validation_level": rec["validation_level"],
            "model_xml": str(_EL4090_XML),
        }
    )
    report["match"] = qpos_ok and tgt_ok and height_ok
    return report


def lowest_collision_point_z(model, data) -> float:
    """Lowest world-z over all collision geoms of the given bodies' subtree.

    Uses compiled mesh vertices (the real collision representation), not foot
    origins. ``model``/``data`` are classic mujoco objects with the witness
    state set.
    """
    foot_body_ids = []
    for name in FOOT_BODIES:
        bid = _body_id(model, name)
        if bid >= 0:
            foot_body_ids.append(bid)
    if not foot_body_ids:
        raise RuntimeError("no foot bodies found in model")
    z_min = np.inf
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] not in foot_body_ids:
            continue
        if model.geom_contype[gid] == 0 and model.geom_conaffinity[gid] == 0:
            continue  # visual-only
        gtype = model.geom_type[gid]
        xpos = data.geom_xpos[gid]
        xmat = data.geom_xmat[gid].reshape(3, 3)
        if gtype == 7:  # mjGEOM_MESH
            m = model.geom_dataid[gid]
            adr, num = model.mesh_vertadr[m], model.mesh_vertnum[m]
            # mesh vertices live in the geom frame (geom_xpos/geom_xmat already
            # include the mesh placement offset geom_pos)
            local = np.asarray(model.mesh_vert[adr : adr + num], dtype=np.float64)
            world = local @ xmat.T + xpos
            z_min = min(z_min, float(world[:, 2].min()))
        else:
            # spheres/capsules/etc: conservative - use geom rbound
            z_min = min(z_min, float(xpos[2]) - float(model.geom_rbound[gid]))
    return z_min


def _body_id(model, name: str) -> int:
    import mujoco

    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))


def compute_lifted_root(
    xml_path: Path,
    qpos: tuple | list,
    clearance_m: float = FOOT_CLEARANCE_M,
):
    """Compute the lifted root state for the drop-in reset.

    Returns ``(lifted_qpos, min_foot_z_before, min_foot_z_after)``. The root is
    raised so the lowest foot collision point sits exactly ``clearance_m``
    above the world z=0 plane of the flat terrain.
    """
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    z_min = lowest_collision_point_z(model, data)
    lift = clearance_m - z_min
    lifted = list(qpos)
    lifted[2] = qpos[2] + lift
    data.qpos[:] = lifted
    mujoco.mj_forward(model, data)
    z_after = lowest_collision_point_z(model, data)
    return lifted, float(z_min), float(z_after)
