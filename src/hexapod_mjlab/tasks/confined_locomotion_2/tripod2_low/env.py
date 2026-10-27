"""MuJoCo adapters for the reference recipe; existing assets stay unchanged."""
from functools import lru_cache

import mujoco
import numpy as np
import torch

from mjlab.managers.event_manager import EventTermCfg, RecomputeLevel, requires_model_fields
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from hexapod_mjlab.common.flat_env import make_flat_env_cfg
from hexapod_mjlab.common.robot import JOINT_NAMES, ROBOT_XML, get_robot_spec
from .actuator import LegacyLSTMActionCfg
from .legacy import (LegacyCommands, LegacyResetDRState, LegacyRewardState,
                     LegacyStageController, REWARD_WEIGHTS)

TASK_ID = "El-4090-Confined-Locomotion-Flat-Tripod2-Low_2"
MODEL_FIELDS = ("body_mass", "body_inertia", "geom_friction", "pair_friction")


def collision_min_z(model, data, geom_ids=None):
    """Exact vertex support height of compiled mesh collision geoms."""
    values = []
    for i in range(model.ngeom) if geom_ids is None else geom_ids:
        if not (model.geom_contype[i] or model.geom_conaffinity[i]):
            continue
        if model.geom_type[i] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh = int(model.geom_dataid[i])
        start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        vertices = model.mesh_vert[start:start + count]
        world = vertices @ data.geom_xmat[i].reshape(3, 3).T + data.geom_xpos[i]
        values.append(float(world[:, 2].min()))
    if not values:
        raise RuntimeError("radar robot has no mesh collision geometry")
    return min(values)


@lru_cache(maxsize=1)
def reset_height():
    """Lift zero-joint robot to the existing 2 mm collision clearance."""
    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    data = mujoco.MjData(model)
    data.qpos[:] = 0.
    data.qpos[3] = 1.
    mujoco.mj_forward(model, data)
    return -collision_min_z(model, data) + .002


def robot_spec():
    spec = get_robot_spec()
    # Let EntityCfg create the sole initial-state keyframe from this task's
    # geometry-derived height, rather than selecting the asset's stand key.
    for key in list(spec.keys):
        spec.delete(key)
    for i, geom in enumerate(spec.geoms):
        if not geom.name:
            geom.name = f"tripod_geom_{i}"
    # Match explicitly applied legacy LSTM efforts. URDF effort metadata is
    # not a clamp in its Python force path; physical-engine equivalence is
    # documented separately. Do not mutate the shared radar MJCF.
    for actuator in spec.actuators:
        actuator.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
        actuator.forcelimited = mujoco.mjtLimited.mjLIMITED_FALSE
    for joint in spec.joints:
        if joint.name in JOINT_NAMES:
            joint.actfrclimited = mujoco.mjtLimited.mjLIMITED_FALSE
    return spec


def contact_pairs(spec):
    """Explicit floor pairs implement the reference's stated average mixture.

    Robot self contacts retain their own randomized material. Spin/rolling
    friction have no PhysX counterpart here; condim=3 uses sliding only.
    """
    plane = [g for g in spec.geoms if g.type == mujoco.mjtGeom.mjGEOM_PLANE]
    if len(plane) != 1:
        raise RuntimeError("tripod2-low requires one plane")
    for i, geom in enumerate(list(spec.geoms)):
        if not geom.name.startswith("robot/") or not (geom.contype or geom.conaffinity):
            continue
        spec.add_pair(name=f"tripod_floor_{i}", geomname1=geom.name,
                      geomname2=plane[0].name, condim=3,
                      friction=[1., 1., 0., 0., 0.])


def owners(env):
    if not hasattr(env, "_tripod_commands"):
        env._tripod_commands = LegacyCommands(env.num_envs, env.device)
        env._tripod_reward_state = LegacyRewardState(env.num_envs, env.device)
        env._tripod_stage = LegacyStageController()
        env._tripod_dr = LegacyResetDRState(env.num_envs, env.device)
        env._tripod_episode_return = torch.zeros(env.num_envs, device=env.device)
        env._tripod_tracking_sum = torch.zeros_like(env._tripod_episode_return)
        env._tripod_previous_qdot = torch.zeros(env.num_envs, 18, device=env.device)
        env._tripod_previous_action = torch.zeros_like(env._tripod_previous_qdot)
        env._tripod_model_fields = MODEL_FIELDS
    return env._tripod_commands


@requires_model_fields(*MODEL_FIELDS, recompute=RecomputeLevel.set_const)
def randomize_model(env, env_ids=None):
    owners(env)
    mm, robot, dr = env.sim.mj_model, env.scene["robot"], env._tripod_dr
    base = next(i for i in range(mm.nbody) if mm.body(i).name == "robot/BASE")
    new_mass = float(mm.body_mass[base]) + dr.added_mass
    env.sim.model.body_mass[:, base] = new_mass
    # PhysX is asked to recompute inertia after changing base mass. MuJoCo's
    # explicit adapter preserves COM/principal axes and scales its tensor by
    # density; this is an approximation of PhysX shape-based recomputation.
    env.sim.model.body_inertia[:, base] = torch.as_tensor(
        mm.body_inertia[base].copy(), device=env.device) * (new_mass / float(mm.body_mass[base]))[:, None]
    geoms = robot.indexing.geom_ids
    env.sim.model.geom_friction[:, geoms, 0] = dr.friction[:, None]
    pairs = [i for i in range(mm.npair) if mm.pair(i).name.startswith("tripod_floor_")]
    env.sim.model.pair_friction[:, pairs, :2] = ((1. + dr.friction) / 2.)[:, None, None]


def reset_state(env, env_ids):
    commands = owners(env)
    commands.curriculum(env_ids, env._tripod_tracking_sum)
    robot = env.scene["robot"]
    root = robot.data.default_root_state[env_ids].clone()
    root[:, :3] += env.scene.env_origins[env_ids]
    root[:, 7:13] = env._tripod_dr.reset_velocities(len(env_ids))
    # Preserve the reference's RNG draw even though multiplying q_default=0
    # makes its joint reset deterministic.
    q = torch.zeros(len(env_ids), 18, device=env.device) * (
        .5 + torch.rand(len(env_ids), 18, device=env.device))
    robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
    robot.write_root_state_to_sim(root, env_ids=env_ids)
    commands.reset(env_ids)
    env._tripod_reward_state.reset(env_ids)
    env._tripod_episode_return[env_ids] = 0.
    env._tripod_tracking_sum[env_ids] = 0.
    env._tripod_previous_qdot[env_ids] = 0.
    env._tripod_previous_action[env_ids] = 0.


def reward_term(env, name):
    context = env._confined_transition_context
    context.ensure_ready()
    return context.raw_rewards[name] * REWARD_WEIGHTS[name][env._tripod_stage.stage]


def flipped(env):
    return env._confined_transition_context.ensure_ready()["gravity_b"][:, 2] > 0.


def timed_out(env):
    env._confined_transition_context.ensure_ready()
    # Both flags may be true, as in the legacy environment.
    return env.episode_length_buf > 1000


def make_env_cfg(*, num_envs=64, seed=1, play=False):
    cfg = make_flat_env_cfg(num_envs=num_envs)
    cfg.seed, cfg.auto_reset = seed, False
    robot = cfg.scene.entities["robot"]
    robot.spec_fn = robot_spec
    robot.init_state.pos = (0., 0., reset_height())
    robot.init_state.joint_pos = {name: 0. for name in JOINT_NAMES}
    cfg.actions = {"residual_pd": LegacyLSTMActionCfg(entity_name="robot", actuator_names=JOINT_NAMES)}
    cfg.scene.spec_fn = contact_pairs
    cfg.scene.sensors += (ContactSensorCfg(
        name="legacy_penalized_contact", primary=ContactMatch(
            mode="body", pattern=("BASE", ".*SHANK", ".*THIGH"), entity="robot"),
        secondary=None, fields=("force",), reduce="netforce", num_slots=1),)
    cfg.events = {
        "legacy_model": EventTermCfg(func=randomize_model, mode="startup"),
        "legacy_reset": EventTermCfg(func=reset_state, mode="reset"),
    }
    cfg.rewards = {name: RewardTermCfg(func=reward_term, weight=1., params={"name": name})
                   for name in REWARD_WEIGHTS}
    cfg.terminations = {"flipped": TerminationTermCfg(func=flipped)}
    if not play:
        cfg.terminations["episode_timeout"] = TerminationTermCfg(func=timed_out, time_out=True)
    cfg.observations["policy"].nan_policy = "error"
    return cfg
