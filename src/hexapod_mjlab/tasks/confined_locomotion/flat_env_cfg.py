"""El-4090-Confined-Locomotion-Flat: real flat-ground mjlab task.

The task configuration uses:

* one horizontal INFINITE physical plane (single ``mjGEOM_PLANE``, friction
  (1.0, 0.005, 0.0001)); no old finite terrain field mesh and no LiDAR mesh;
* the radar-variant robot XML stays byte-identical (corrected optical
  coordinates); its 18 nominal hinge limits stay ENABLED at ±3 rad and the
  soft position-limit factor is 0.9 (penalty range ±2.7 rad);
* ``q_ref`` (PD default joint targets) = the frozen T1 ``JOINT_TARGETS``
  (nominal IK targets, max |v| ≈ 4.6e-3 rad), NOT the settled measured qpos;
* reset state = the complete settled T1 witness (root pos/quat + 18 joints),
  lifted so the lowest actual foot collision-geometry point sits 2 mm above
  the plane, with zero root/joint velocities;
* actions: 18-dim residual; clip raw ±100, target = q_ref + 0.25·clipped,
  and at EVERY 200 Hz physics substep recompute 130·(target−q) − 2·q̇ then
  clamp to ±80 N·m; 50 Hz control, 4 substeps; no position-target clipping;
* no observation noise / pushes / mass / friction randomization; generic
  registration has no commands, rewards, or curricula. Explicit training and
  play factories provide their own task terms.

This is the real registered mjlab environment configuration. The production
training pipeline is available through explicit typed recipes; the registered
runner defaults remain intentionally incomplete and are rejected before
network or optimizer construction. Runnable implementation does not by itself
establish learning-effect acceptance. Generic environment defaults remain
separate from the training and play factories below.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.terrain_generator import (
  SubTerrainCfg,
  TerrainGeneratorCfg,
  TerrainGeometry,
  TerrainOutput,
)

from hexapod_mjlab.tasks.el_4090_lidar.t0 import witness as t1_witness

from . import mdp
from .mdp.actions import ResidualPDActionCfg

TASK_ID = "El-4090-Confined-Locomotion-Flat"

#: Radar-variant MJCF (read-only asset; same file the old tasks compile).
_ROBOT_XML = (
  Path(__file__).parents[2] / "assets" / "robots" / "el_4090" / "mjcf" / "el_4090_lidar.xml"
)

#: Single source of the joint order: the frozen T1 witness JOINT_ORDER.
JOINT_NAMES = tuple(t1_witness.JOINT_ORDER)
_JOINT_NAME_SET = frozenset(JOINT_NAMES)

#: Fixed ground friction (sliding, torsional, rolling) — declared sim value.
FLAT_GROUND_FRICTION = (1.0, 0.005, 0.0001)

#: Confirmed action/PD constants (平地学习与基准测量.md §2).
ACTION_CLIP = 100.0
ACTION_SCALE = 0.25
PD_STIFFNESS = float(t1_witness.PD_KP)  # 130
PD_DAMPING = float(t1_witness.PD_KD)  # 2
TORQUE_LIMIT_NM = float(t1_witness.TORQUE_CLIP_NM)  # 80
SIM_TIMESTEP_S = 0.005
DECIMATION = 4  # 50 Hz control, 200 Hz physics
FOOT_CLEARANCE_M = float(t1_witness.FOOT_CLEARANCE_M)  # 0.002 (plan min 1 mm)

#: Nominal hinge limits stay ENABLED (contrast: T0-A disables them in memory).
NOMINAL_JOINT_LIMIT_RAD = 3.0
SOFT_JOINT_POS_LIMIT_FACTOR = 0.9  # soft penalty range ±2.7 rad

#: Generic registered defaults, separate from the explicit training recipe.
GENERIC_NUM_ENVS = 64
GENERIC_EPISODE_LENGTH_S = 20.0

#: Feet (physical foot bodies of the asset; witness order).
FOOT_BODIES = tuple(t1_witness.FOOT_BODIES)


def verify_nominal_joint_limits(
  xml_path: Path = _ROBOT_XML,
  atol: float = 1e-9,
) -> dict:
  """Compile the asset and verify the 18 nominal hinge limits are enabled ±3.

  Read-only revalidation of the CURRENT limited asset (old T0-A keeps its own
  in-memory-unlocked spec; this module never unlocks limits). Returns the raw
  per-joint table for independent checks; raises on any deviation.
  """
  model = mujoco.MjModel.from_xml_path(str(xml_path))
  rows = {}
  for name in JOINT_NAMES:
    jid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
    if jid < 0:
      raise RuntimeError(f"joint {name!r} missing from {_ROBOT_XML}")
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
  spec = mujoco.MjSpec.from_file(str(_ROBOT_XML))
  seen = 0
  for joint in spec.joints:
    if joint.name in _JOINT_NAME_SET:
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
      raise RuntimeError(f"joint {name!r} missing from {_ROBOT_XML}")
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
  w = t1_witness.frozen_witness()
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


@functools.lru_cache(maxsize=1)
def lifted_witness_state() -> tuple[tuple, tuple, float, float]:
  """Lifted reset state from the frozen T1 witness, computed once.

  Returns ``(root_pos_xyz, root_quat_wxyz, min_foot_z_before, min_foot_z_after)``.
  The lift is computed from compiled collision meshes via the read-only T1
  witness geometry utility (never from foot-origin kinematics), raising if
  the achieved clearance is not 2 mm within 1e-6 m.
  """
  lifted, z_before, z_after = t1_witness.compute_lifted_root(
    _ROBOT_XML, t1_witness.SETTLED_QPOS, clearance_m=FOOT_CLEARANCE_M
  )
  if abs(z_after - FOOT_CLEARANCE_M) > 1e-6:
    raise RuntimeError(
      f"witness lift failed: min foot z after lift = {z_after} "
      f"(expected {FOOT_CLEARANCE_M})"
    )
  return tuple(lifted[0:3]), tuple(lifted[3:7]), float(z_before), float(z_after)


def settled_reset_joint_pos() -> tuple:
  """Reset joint angles = the settled T1 witness qpos (NOT q_ref)."""
  return tuple(t1_witness.frozen_witness().joint_pos)


# ── Single infinite physical plane ──────────────────────────────────────────


@dataclass(kw_only=True)
class SinglePlaneSubTerrainCfg(SubTerrainCfg):
  """One ``mjGEOM_PLANE`` geom: a single horizontal INFINITE ground.

  MuJoCo plane collision is infinite in extent; the half-extent only sizes
  the rendered patch. No heightfield, no mesh, no old finite terrain field:
  the only environment geometry is this plane with the confirmed friction
  (1.0, 0.005, 0.0001). priority=1 makes the plane's friction win the
  max-mix combine against robot geoms (same mechanism as the old flat pin).
  """

  friction: tuple[float, float, float] = FLAT_GROUND_FRICTION
  plane_half_extent: float = 100.0
  _built: bool = False

  def function(self, difficulty, spec, rng):
    if self._built:
      raise RuntimeError(
        "SinglePlaneSubTerrainCfg.function called twice; the flat config "
        "must contain exactly one ground plane"
      )
    self._built = True
    geo = spec.body("terrain").add_geom(
      name="confined_flat_ground",
      type=mujoco.mjtGeom.mjGEOM_PLANE,
      size=[self.plane_half_extent, self.plane_half_extent, 0.1],
      priority=1,
      friction=list(self.friction),
    )
    return TerrainOutput(
      origin=np.array([self.size[0] / 2.0, self.size[1] / 2.0, 0.0]),
      geometries=[TerrainGeometry(geom=geo)],
      flat_patches=None,
    )


def make_flat_env_cfg(
  play: bool = False,
  num_envs: int = GENERIC_NUM_ENVS,
  episode_length_s: float = GENERIC_EPISODE_LENGTH_S,
) -> ManagerBasedRlEnvCfg:
  """Build the registered flat confined-locomotion environment config.

  Generic defaults only: play merely removes the episode budget; training
  explicitly installs its reward and termination terms through
  ``make_training_env_cfg``.
  """
  feet_any_contact = ContactSensorCfg(
    name="feet_any_contact",
    # Feet vs ANY contact partner (the plane here, but the match is not
    # restricted to the plane) for the world-coordinate net foot force.
    primary=ContactMatch(mode="body", pattern=FOOT_BODIES, entity="robot"),
    secondary=None,
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )

  # The registered generic observation set contains stock-mjlab proprio terms
  # plus the task-local clipped previous action. The training recipe installs
  # its strict observation transport contract independently.
  # No noise/corruption (this batch disables all artificial noise).
  policy_terms = {
    "base_lin_vel": ObservationTermCfg(func=envs_mdp.base_lin_vel),
    "base_ang_vel": ObservationTermCfg(func=envs_mdp.base_ang_vel),
    "projected_gravity": ObservationTermCfg(func=envs_mdp.projected_gravity),
    "joint_pos_rel": ObservationTermCfg(
      func=envs_mdp.joint_pos_rel,
      params={"asset_cfg": mdp.SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "joint_vel_rel": ObservationTermCfg(
      func=envs_mdp.joint_vel_rel,
      params={"asset_cfg": mdp.SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "clipped_prev_action": ObservationTermCfg(
      func=mdp.clipped_prev_action,
      params={"action_name": "residual_pd"},
    ),
  }
  observations = {
    "policy": ObservationGroupCfg(
      terms=policy_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  actions = {
    "residual_pd": ResidualPDActionCfg(
      entity_name="robot",
      actuator_names=JOINT_NAMES,
      raw_clip=ACTION_CLIP,
      residual_scale=ACTION_SCALE,
      stiffness=PD_STIFFNESS,
      damping=PD_DAMPING,
      effort_limit=TORQUE_LIMIT_NM,
    ),
  }

  # Deterministic reset: complete settled T1 witness (lifted 2 mm), zero
  # velocities; plus clearing the action runtime buffers of the reset envs.
  events = {
    "reset_root_witness": EventTermCfg(func=mdp.reset_root_witness, mode="reset"),
    "reset_joints_witness": EventTermCfg(func=mdp.reset_joints_witness, mode="reset"),
    "clear_action_runtime": EventTermCfg(
      func=mdp.clear_action_runtime,
      mode="reset",
      params={"action_name": "residual_pd"},
    ),
  }

  # Generic registration defines no reward terms. The training factory binds
  # the formal reward set explicitly.
  rewards = {}

  # Generic registration has no termination terms. Training installs its
  # budget and fall terms through the explicit recipe.
  terminations = {}

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
          size=(8.0, 8.0),
          num_rows=1,
          num_cols=1,
          border_width=0.0,
          curriculum=False,
          sub_terrains={
            "confined_flat_plane": SinglePlaneSubTerrainCfg(
              proportion=1.0,
              size=(8.0, 8.0),
            )
          },
        ),
        max_init_terrain_level=None,
      ),
      entities={"robot": get_confined_robot_cfg()},
      sensors=(feet_any_contact,),
      num_envs=num_envs,
      env_spacing=8.0,
    ),
    observations=observations,
    actions=actions,
    commands={},
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum={},
    sim=SimulationCfg(
      mujoco=MujocoCfg(timestep=SIM_TIMESTEP_S, gravity=(0.0, 0.0, -9.81)),
    ),
    decimation=DECIMATION,
    episode_length_s=episode_length_s,
  )

  if play:
    # Generic play default: drop the episode budget only; task play uses its
    # separate explicit factory.
    cfg.episode_length_s = int(1e9)
  return cfg


def make_training_env_cfg(
  *,
  num_envs: int = 2,
  seed: int | None = None,
  episode_length_s: float = GENERIC_EPISODE_LENGTH_S,
):
  """Build the independent training environment config.

  The registered defaults remain untouched. Training binds the formal
  reward/termination callbacks, Null commands, ``auto_reset=False`` and a
  strict observation transport group. Reward and termination imports stay
  local because both MDP modules depend on this flat config.
  """
  expected_episode_length_s = GENERIC_EPISODE_LENGTH_S
  if abs(float(episode_length_s) - expected_episode_length_s) > 1e-9:
    raise ValueError(
      f"episode_length_s must be {expected_episode_length_s} s (1000 "
      f"control steps) for the confirmed timeout; got {episode_length_s!r}"
    )
  cfg = make_flat_env_cfg(
    play=False, num_envs=int(num_envs), episode_length_s=float(episode_length_s)
  )
  cfg.auto_reset = False
  if seed is not None:
    cfg.seed = int(seed)
  cfg.observations["policy"].nan_policy = "error"
  from .mdp.rewards import build_training_reward_cfgs
  from .mdp.terminations import build_training_termination_cfgs

  cfg.rewards = build_training_reward_cfgs()
  cfg.terminations = build_training_termination_cfgs()
  cfg.commands = {}
  return cfg
