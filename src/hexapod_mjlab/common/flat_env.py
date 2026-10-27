"""Shared flat-ground environment construction for the confined tasks.

This module is the single implementation of the flat environment config used
by the initial Flat task and the v2 Flat Baseline. It owns only task-agnostic
construction: the infinite physical plane, robot entity, reset events,
proprio/previous-action observations, residual PD action, generic defaults,
and the explicit training variant with the shared formal reward/termination
bindings. Task IDs, registration, RL recipes and play lifecycles stay in the
task packages.

mjlab imports are function-local: importing this module must not trigger
mjlab's external task entry-point discovery while this module is still
initializing. The plane config lives in ``common.flat_terrain`` for the same
reason.

The environment configuration uses:

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
  construction has no commands, rewards, or curricula. Explicit training and
  play factories provide their own task terms.

Generic environment defaults remain separate from the explicit training
variant. Runnable implementation does not by itself establish learning-effect
acceptance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hexapod_mjlab.common.robot import (
  ACTION_CLIP,
  ACTION_SCALE,
  DECIMATION,
  FOOT_BODIES,
  JOINT_NAMES,
  PD_DAMPING,
  PD_STIFFNESS,
  SIM_TIMESTEP_S,
  TORQUE_LIMIT_NM,
  get_confined_robot_cfg,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnvCfg

#: Fixed ground friction (sliding, torsional, rolling) — declared sim value.
FLAT_GROUND_FRICTION = (1.0, 0.005, 0.0001)

#: Generic registered defaults, separate from the explicit training recipe.
GENERIC_NUM_ENVS = 64
GENERIC_EPISODE_LENGTH_S = 20.0


def make_flat_env_cfg(
  play: bool = False,
  num_envs: int = GENERIC_NUM_ENVS,
  episode_length_s: float = GENERIC_EPISODE_LENGTH_S,
) -> ManagerBasedRlEnvCfg:
  """Build a fresh generic flat confined-locomotion environment config.

  Generic defaults only: play merely removes the episode budget; training
  explicitly installs its reward and termination terms through
  ``make_training_env_cfg``.
  """
  from mjlab.envs import ManagerBasedRlEnvCfg
  from mjlab.envs import mdp as envs_mdp
  from mjlab.managers.event_manager import EventTermCfg
  from mjlab.managers.observation_manager import (
    ObservationGroupCfg,
    ObservationTermCfg,
  )
  from mjlab.managers.scene_entity_config import SceneEntityCfg
  from mjlab.scene import SceneCfg
  from mjlab.sensor import ContactMatch, ContactSensorCfg
  from mjlab.sim import MujocoCfg, SimulationCfg
  from mjlab.terrains import TerrainEntityCfg
  from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

  from hexapod_mjlab.common.flat_terrain import SinglePlaneSubTerrainCfg
  from hexapod_mjlab.common.mdp.actions import ResidualPDActionCfg
  from hexapod_mjlab.common.mdp.events import (
    clear_action_runtime,
    reset_joints_witness,
    reset_root_witness,
  )
  from hexapod_mjlab.common.mdp.observations import clipped_prev_action

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
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "joint_vel_rel": ObservationTermCfg(
      func=envs_mdp.joint_vel_rel,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "clipped_prev_action": ObservationTermCfg(
      func=clipped_prev_action,
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
    "reset_root_witness": EventTermCfg(func=reset_root_witness, mode="reset"),
    "reset_joints_witness": EventTermCfg(func=reset_joints_witness, mode="reset"),
    "clear_action_runtime": EventTermCfg(
      func=clear_action_runtime,
      mode="reset",
      params={"action_name": "residual_pd"},
    ),
  }

  # Generic construction defines no reward terms. The training factory binds
  # the formal reward set explicitly.
  rewards = {}

  # Generic construction has no termination terms. Training installs its
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
  """Build a fresh independent training environment config.

  The generic defaults remain untouched. Training binds the formal
  reward/termination callbacks, Null commands, ``auto_reset=False`` and a
  strict observation transport group. Reward and termination imports stay
  local because both MDP modules depend on the shared flat config.
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
  from hexapod_mjlab.common.mdp.rewards import build_training_reward_cfgs
  from hexapod_mjlab.common.mdp.terminations import build_training_termination_cfgs

  cfg.rewards = build_training_reward_cfgs()
  cfg.terminations = build_training_termination_cfgs()
  cfg.commands = {}
  return cfg
