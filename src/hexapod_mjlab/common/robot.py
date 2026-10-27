"""Base EL4090 robot constants and entity factory shared by formal tasks."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .lidar import EL4090_LIDAR_MJCF
from .witness import (
    FOOT_BODIES as WITNESS_FOOT_BODIES,
    FOOT_CLEARANCE_M as WITNESS_FOOT_CLEARANCE_M,
    PD_KD,
    PD_KP,
    TORQUE_CLIP_NM,
    frozen_witness,
)

if TYPE_CHECKING:
    import mujoco

#: Radar-variant MJCF (read-only asset; same file the old tasks compile).
ROBOT_XML = EL4090_LIDAR_MJCF

#: Single source of the joint order: the frozen T1 witness JOINT_ORDER.
JOINT_NAMES = tuple(frozen_witness().joint_names)
JOINT_NAME_SET = frozenset(JOINT_NAMES)
JOINT_DIM = len(JOINT_NAMES)

#: Confirmed action/PD constants (平地学习与基准测量.md §2).
ACTION_CLIP = 100.0
ACTION_SCALE = 0.25
PD_STIFFNESS = float(PD_KP)  # 130
PD_DAMPING = float(PD_KD)  # 2
TORQUE_LIMIT_NM = float(TORQUE_CLIP_NM)  # 80
SIM_TIMESTEP_S = 0.005
DECIMATION = 4  # 50 Hz control, 200 Hz physics
FOOT_CLEARANCE_M = float(WITNESS_FOOT_CLEARANCE_M)  # 0.002 (plan min 1 mm)

#: Nominal hinge limits stay ENABLED (contrast: T0-A disables them in memory).
NOMINAL_JOINT_LIMIT_RAD = 3.0
SOFT_JOINT_POS_LIMIT_FACTOR = 0.9  # soft penalty range ±2.7 rad

#: Feet (physical foot bodies of the asset; witness order).
FOOT_BODIES = tuple(WITNESS_FOOT_BODIES)


def verify_nominal_joint_limits(
  xml_path: Path = ROBOT_XML,
  atol: float = 1e-9,
) -> dict:
  """Compile the asset and verify the 18 nominal hinge limits are enabled ±3.

  Read-only revalidation of the CURRENT limited asset (old T0-A keeps its own
  in-memory-unlocked spec; this module never unlocks limits). Returns the raw
  per-joint table for independent checks; raises on any deviation.
  """
  import mujoco

  model = mujoco.MjModel.from_xml_path(str(xml_path))
  rows = {}
  for name in JOINT_NAMES:
    jid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
    if jid < 0:
      raise RuntimeError(f"joint {name!r} missing from {ROBOT_XML}")
    limited = int(model.jnt_limited[jid])
    lo, hi = (float(model.jnt_range[jid][0]), float(model.jnt_range[jid][1]))
    rows[name] = (limited, lo, hi)
    if limited != 1:
      raise RuntimeError(f"joint {name!r}: limits disabled (expected ±3 enabled)")
    if (
      abs(lo + NOMINAL_JOINT_LIMIT_RAD) > atol
      or abs(hi - NOMINAL_JOINT_LIMIT_RAD) > atol
    ):
      raise RuntimeError(f"joint {name!r}: range ({lo}, {hi}) != ±{NOMINAL_JOINT_LIMIT_RAD}")
  if len(rows) != 18:
    raise RuntimeError(f"expected 18 joints, found {len(rows)}")
  return rows


def get_robot_spec() -> mujoco.MjSpec:
  """Load the radar MJCF unchanged; verify the limits on the COMPILED model.

  The asset compiles with autolimit enabled, so raw spec joints carry
  ``mjLIMITED_AUTO`` (= 2) with an explicit ±3 range; ``limited == 1`` only
  exists AFTER compilation (independently verified on the compiled
  MjModel). Checking the raw enum therefore rejected a correctly limited
  asset (real construction failed at RF_HAA with limited=2). The spec is
  returned UNMODIFIED (±3 limits retained, no in-memory unlocking); instead
  a throwaway model compiled from THIS spec instance is checked against the
  same enabled-±3 contract as ``verify_nominal_joint_limits``.
  """
  import mujoco

  spec = mujoco.MjSpec.from_file(str(ROBOT_XML))
  seen = 0
  for joint in spec.joints:
    if joint.name in JOINT_NAME_SET:
      seen += 1
      lo, hi = float(joint.range[0]), float(joint.range[1])
      if (
        abs(lo + NOMINAL_JOINT_LIMIT_RAD) > 1e-9
        or abs(hi - NOMINAL_JOINT_LIMIT_RAD) > 1e-9
      ):
        raise RuntimeError(
          f"joint {joint.name!r} spec range ({lo}, {hi}) != "
          f"±{NOMINAL_JOINT_LIMIT_RAD}"
        )
  if seen != 18:
    raise RuntimeError(f"expected 18 hinge joints in spec, found {seen}")
  model = spec.compile()
  for name in JOINT_NAMES:
    jid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
    if jid < 0:
      raise RuntimeError(f"joint {name!r} missing from {ROBOT_XML}")
    limited = int(model.jnt_limited[jid])
    lo, hi = (float(model.jnt_range[jid][0]), float(model.jnt_range[jid][1]))
    if limited != int(mujoco.mjtLimited.mjLIMITED_TRUE) or (
      abs(lo + NOMINAL_JOINT_LIMIT_RAD) > 1e-9
      or abs(hi - NOMINAL_JOINT_LIMIT_RAD) > 1e-9
    ):
      raise RuntimeError(
        f"joint {name!r} compiled limits not enabled ±3 "
        f"(limited={limited}, range=({lo}, {hi}))"
      )
  return spec


def get_confined_robot_cfg() -> EntityCfg:
  """Robot entity with enabled nominal limits, soft factor 0.9, T1 q_ref.

  ``init_state.joint_pos`` IS the PD default (q_ref = T1 JOINT_TARGETS) and
  is deliberately DISTINCT from the reset state (settled witness qpos,
  written by the reset events). The nominal root height here is only the
  EntityCfg default; the real reset pose comes from the lifted witness.
  """
  from mjlab.actuator.xml_actuator import XmlActuatorCfg
  from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

  w = frozen_witness()
  return EntityCfg(
    spec_fn=get_robot_spec,
    articulation=EntityArticulationInfoCfg(
      actuators=(XmlActuatorCfg(target_names_expr=JOINT_NAMES),),
      soft_joint_pos_limit_factor=SOFT_JOINT_POS_LIMIT_FACTOR,
    ),
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, w.settled_root_height_m),
      joint_pos={name: float(v) for name, v in zip(w.joint_names, w.joint_targets)},
    ),
  )
