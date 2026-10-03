"""El-4090-Lidar-T0-Flat: isolated, deterministic flat-ground configuration.

Scope (T0-A plan §2): one flat-ground variant for diagnostics, a fixed short
training pipeline and independent evaluation. The pillar-field task
``El-4090-Lidar`` keeps its default behavior untouched; every difference is
made explicit here.

Deterministic contract of this configuration:

* flat terrain through the SAME builder/view chain (zero-pillar config), so
  physics plane, LiDAR warp mesh and height-grid supervision all read one
  field;
* no observation noise, no LiDAR sensor/pixel noise or dropout, no
  point-mask/distance DR;
* no push, no mass/friction startup randomization, no reset pose/velocity
  randomization; terrain sliding friction pinned to μ=1.0 (declared
  simulation assumption; priority=1 makes it win the max-mix combine);
* reset = T1 ``height_0.48_width_1.00`` settled witness, root lifted so the
  actual foot collision geometry clears ≥ 2 mm, then released; zero initial
  velocities;
* action default joint offset = the witness PD targets (zero action
  reproduces the T1 hold controller: kp 130, kd 2, ±80 N·m clip);
* nominal ±3 rad hinge position limits disabled in the IN-MEMORY spec
  (asset bytes unchanged; T1 measurement assumption), and the
  ``dof_pos_limits`` soft-limit reward removed (it penalized the uncalibrated
  URDF values);
* ``base_height`` reward target = the witness settled height (0.4599 m), not
  the legacy fixed 0.25 m;
* commands: standing proportion 0.20, otherwise vx ∈ [0.25, 0.35] m/s,
  vy = 0, yaw rate = 0; heading conversion disabled; per-episode resampling
  only; the 0.2 m/s deadzone is inert for these ranges (documented);
* no reward-stage switching and no command curriculum.

Retained unchanged: 18-DoF PD-torque actions, 2754-dim policy obs, 187-dim
height_grid supervision, 2880-point pool, actor/PPO network contract,
terminations (time_out + old fall thresholds), the legacy contact sensors
and the legacy reward weights (stage-0).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import mujoco
import torch
from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import (
    ObservationGroupCfg,
    ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from hexapod_mjlab.terrains.builder import TerrainConfig

from .. import mdp
from ..el_4090_lidar_env_cfg import (
    ACTION_SCALE,
    DAMPING,
    DECIMATION,
    EPISODE_LENGTH_S,
    JOINT_NAMES,
    SIM_TIMESTEP,
    STIFFNESS,
    TORQUE_LIMIT,
    make_el_4090_lidar_env_cfg,
)
from ..mdp.commands import El4090VelocityCommandCfg
from ..mdp.lidar import El4090LidarCfg
from ..mdp.terrain import El4090PillarFieldCfg
from . import witness as t0_witness
from .witness import frozen_witness
from .point_order import t0_prox_obs, t0_distal_obs

T0_TASK_ID = "El-4090-Lidar-T0-Flat"

#: Standing-command proportion during training (plan: explicit 0.20).
T0_STANDING_PROPORTION = 0.20
#: Forward command range during training.
T0_TRAIN_VX_RANGE = (0.25, 0.35)
#: Evaluation commands (independent of the training range).
T0_EVAL_COMMANDS = {
    "stand": 0.0,
    "forward_025": 0.25,
}

#: Pinned terrain friction (sliding, torsional, rolling; declared sim
#: assumption — NOT calibrated hardware friction).
T0_TERRAIN_FRICTION = (1.0, 0.005, 0.0001)


def make_t0_flat_terrain_config() -> TerrainConfig:
    """Zero-pillar TerrainConfig: same builder chain, no obstacles.

    ``pillar_count_min=max=0`` makes ``pillar_field_terrain`` place nothing,
    leaving the whole 2x2 patch + border field at height 0 for every seed; the
    env_origins z (max over the clear center platform) is 0 as well.
    """
    return TerrainConfig(pillar_count_min=0, pillar_count_max=0)


@dataclass(kw_only=True)
class El4090FlatFieldCfg(El4090PillarFieldCfg):
    """Flat sibling of the pillar sub-terrain (same builder/view chain).

    Subclassing keeps :func:`get_el_4090_terrain_view` working unchanged (it
    isinstance-checks ``El4090PillarFieldCfg`` and looks the entry up under the
    ``"pillar_field"`` key), so the LiDAR raycast mesh and the height-grid
    supervision consume exactly this flat field.
    """

    terrain_seed: int = 42

    representation: str = "plane"

    def function(self, difficulty, spec, rng):
        """One infinite plane (not four overlapping planes), same flat view.

        The LiDAR/height view is finite [-13,13]^2. T0 evaluation stays in
        that declared domain; rays outside it miss, never invent an obstacle.
        The hfield option exists only for controlled diagnostic comparisons.
        """
        if self.representation == "hfield":
            out = super().function(difficulty, spec, rng)
            for geo in out.geometries:
                geo.geom.friction = list(T0_TERRAIN_FRICTION)
            return out
        if self.representation != "plane":
            raise ValueError(self.representation)
        import numpy as np
        from mjlab.terrains.terrain_generator import TerrainGeometry, TerrainOutput

        self._ensure_built()
        self._patch_count += 1
        geometries = []
        if self._patch_count == 1:
            geo = spec.body("terrain").add_geom(
                name="t0_flat_ground",
                type=mujoco.mjtGeom.mjGEOM_PLANE,
                size=[13.0, 13.0, 0.1],
                priority=1,
                friction=list(T0_TERRAIN_FRICTION),
            )
            geometries = [TerrainGeometry(geom=geo)]
        return TerrainOutput(
            origin=np.array([self.size[0] / 2, self.size[1] / 2, 0.0]),
            geometries=geometries,
            flat_patches=None,
        )


def _get_el_4090_t0_spec() -> mujoco.MjSpec:
    """Load the radar-variant MJCF and disable the 18 nominal hinge limits.

    The ±3 rad ``range="-3 3"`` attributes are uncalibrated URDF placeholders
    (uniform across all joints); T1 measurements already assumed them disabled.
    Disabling happens on the in-memory MjSpec only — the XML on disk is never
    rewritten.
    """
    from ..el_4090_lidar_env_cfg import _EL4090_XML

    spec = mujoco.MjSpec.from_file(str(_EL4090_XML))
    disabled = []
    for joint in spec.joints:
        if joint.name in JOINT_NAMES:
            joint.limited = mujoco.mjtLimited.mjLIMITED_FALSE
            disabled.append(joint.name)
    missing = sorted(set(JOINT_NAMES) - set(disabled))
    if missing:
        raise RuntimeError(f"T0 spec did not find joints to unlock: {missing}")
    return spec


def get_el_4090_t0_robot_cfg() -> EntityCfg:
    """Robot cfg on the unlocked spec with witness PD targets as defaults."""
    w = frozen_witness()
    return EntityCfg(
        spec_fn=_get_el_4090_t0_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=(XmlActuatorCfg(target_names_expr=JOINT_NAMES),),
        ),
        init_state=EntityCfg.InitialStateCfg(
            # Root height here is nominal; the real reset comes from the witness
            # event below (lifted settled state).
            pos=(0.0, 0.0, w.settled_root_height_m),
            joint_pos={
                name: float(w.joint_targets[i]) for i, name in enumerate(t0_witness.JOINT_ORDER)
            },
        ),
    )


# ── Deterministic reset events (T1 witness) ────────────────────────────────


@functools.lru_cache(maxsize=1)
def lifted_witness_root() -> tuple[tuple, tuple]:
    """Lifted witness root ((x, y, z), (qw, qx, qy, qz)); computed once.

    The lift guarantees the lowest actual foot collision point sits
    ``witness.FOOT_CLEARANCE_M`` (2 mm >= the 1 mm plan minimum) above the
    flat ground, computed from compiled collision meshes — never from foot
    origins.
    """
    from ..el_4090_lidar_env_cfg import _EL4090_XML

    lifted, z_before, z_after = t0_witness.compute_lifted_root(
        _EL4090_XML, frozen_witness().settled_qpos
    )
    if z_after < 1e-3:
        raise RuntimeError(f"witness lift failed: min foot z after lift = {z_after}")
    return tuple(lifted[0:3]), tuple(lifted[3:7])


def t0_reset_root_witness(
    env,
    env_ids: torch.Tensor | None,
    root_pos_local: tuple | None = None,
    root_quat_wxyz: tuple | None = None,
    asset_cfg_name: str = "robot",
) -> None:
    """Reset the root to the lifted T1 witness pose with zero velocity.

    ``root_pos_local``/``root_quat_wxyz`` default to the frozen lifted witness
    state. Env-origin offsets are applied the same way
    ``reset_root_state_uniform`` does.
    """
    from mjlab.envs.mdp.events import resolve_env_ids

    env_ids = resolve_env_ids(env, env_ids)
    asset = env.scene[asset_cfg_name]
    if root_pos_local is None or root_quat_wxyz is None:
        pos, quat = lifted_witness_root()
    else:
        pos, quat = root_pos_local, root_quat_wxyz
    n = len(env_ids)
    positions = torch.tensor(pos, dtype=torch.float32, device=env.device)
    positions = positions.view(1, 3).expand(n, 3) + env.scene.env_origins[env_ids]
    orientations = (
        torch.tensor(quat, dtype=torch.float32, device=env.device).view(1, 4).expand(n, 4)
    )
    asset.write_root_link_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_link_velocity_to_sim(
        torch.zeros(n, 6, dtype=torch.float32, device=env.device), env_ids=env_ids
    )


def t0_reset_joints_witness(
    env,
    env_ids: torch.Tensor | None,
    joint_pos: tuple | None = None,
    asset_cfg_name: str = "robot",
) -> None:
    """Reset the 18 joints to the settled witness angles with zero velocity."""
    from mjlab.envs.mdp.events import resolve_env_ids

    env_ids = resolve_env_ids(env, env_ids)
    asset = env.scene[asset_cfg_name]
    if joint_pos is None:
        joint_pos = frozen_witness().joint_pos
    n = len(env_ids)
    pos = torch.tensor(joint_pos, dtype=torch.float32, device=env.device)
    pos = pos.view(1, len(joint_pos)).expand(n, len(joint_pos))
    vel = torch.zeros(n, len(joint_pos), dtype=torch.float32, device=env.device)
    asset.write_joint_state_to_sim(pos, vel, env_ids=env_ids)


def make_t0_lidar_cfg() -> El4090LidarCfg:
    """Noise-free LiDAR configuration (deterministic evaluation contract)."""
    return El4090LidarCfg(
        enable_sensor_noise=False,
        pixel_std_dev_multiplier=0.0,
        pixel_dropout_prob=0.0,
        lidar_point_mask_ratio=0.0,
        lidar_distance_noise_ratio=0.0,
        mount_from_model=True,
    )


def outside_flat_domain(env):
    """Robot root must leave room for occupancy and height-grid samples."""
    return (env.scene["robot"].data.root_link_pos_w[:, :2].abs() > 11.0).any(dim=1)


def make_el_4090_t0_flat_env_cfg(
    play: bool = False,
    num_envs: int = 32,
    episode_length_s: float = EPISODE_LENGTH_S,
    terrain_representation: str = "plane",
) -> ManagerBasedRlEnvCfg:
    """Build the deterministic flat-ground EL4090 LiDAR environment config."""
    w = frozen_witness()

    feet_ground_contact = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="body", pattern=mdp.FEET_NAMES, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
    )
    # Legacy penalised_contact kept verbatim so the legacy ``collision``
    # reward keeps its exact old meaning in the T0 variant too.
    penalised_contact = ContactSensorCfg(
        name="penalised_contact",
        primary=ContactMatch(
            mode="body",
            pattern=("BASE", ".*_SHANK", ".*_THIGH"),
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
    )
    # T0 addition: EVERY non-foot robot collision body (BASE, HIP, THIGH,
    # SHANK, RADAR_BRACKET, LIDAR_AIRY) against ANYTHING (terrain or robot).
    # The legacy penalised_contact (BASE/SHANK/THIGH vs terrain only) missed
    # HIP and all self-collision; this sensor closes that gap for the T0
    # diagnostics. Feet are excluded as primaries so normal standing contacts
    # do not trigger it. Foot-vs-foot pairs (both sides feet) are invisible
    # here and are covered by the raw mujoco_warp contact classification in
    # t0.diagnostics.
    nonfoot_any_contact = ContactSensorCfg(
        name="nonfoot_any_contact",
        primary=ContactMatch(
            mode="body",
            pattern=(".*",),
            entity="robot",
            exclude=tuple(name for name in mdp.FEET_NAMES),
        ),
        secondary=None,
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
    )

    t0_lidar_cfg = make_t0_lidar_cfg()

    policy_terms = {
        "base_lin_vel": ObservationTermCfg(
            func=mdp.base_lin_vel,
            scale=mdp.OBS_SCALE_LIN_VEL,
        ),
        "base_ang_vel": ObservationTermCfg(
            func=mdp.base_ang_vel,
            scale=mdp.OBS_SCALE_ANG_VEL,
        ),
        "projected_gravity": ObservationTermCfg(
            func=envs_mdp.projected_gravity,
        ),
        "command": ObservationTermCfg(
            func=envs_mdp.generated_commands,
            params={"command_name": "twist"},
            scale=mdp.COMMANDS_SCALE,
        ),
        "joint_pos": ObservationTermCfg(
            func=envs_mdp.joint_pos_rel,
            params={"asset_cfg": mdp.SceneEntityCfg("robot", joint_names=(".*",))},
            scale=mdp.OBS_SCALE_DOF_POS,
        ),
        "joint_vel": ObservationTermCfg(
            func=envs_mdp.joint_vel_rel,
            params={"asset_cfg": mdp.SceneEntityCfg("robot", joint_names=(".*",))},
            scale=mdp.OBS_SCALE_DOF_VEL,
        ),
        "actions": ObservationTermCfg(func=envs_mdp.last_action),
        "lidar_prox": ObservationTermCfg(
            func=t0_prox_obs,
            params={"lidar_cfg": t0_lidar_cfg},
        ),
        "lidar_distal": ObservationTermCfg(
            func=t0_distal_obs,
            params={"lidar_cfg": t0_lidar_cfg},
        ),
    }
    observations = {
        "policy": ObservationGroupCfg(
            terms=policy_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),
        "height_grid": ObservationGroupCfg(
            terms={"height_grid": ObservationTermCfg(func=mdp.height_grid_obs)},
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }

    actions = {
        "joint_pd_torque": mdp.JointPDTorqueActionCfg(
            entity_name="robot",
            actuator_names=JOINT_NAMES,
            scale=ACTION_SCALE,
            stiffness=STIFFNESS,  # 130 (T1 hold protocol)
            damping=DAMPING,  # 2
            effort_limit=TORQUE_LIMIT,  # 80 N·m clip
            use_default_offset=True,  # default = witness PD targets
        )
    }

    commands = {
        "twist": El4090VelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(episode_length_s, episode_length_s),
            heading_command=False,  # disabled per plan (no heading conversion)
            rel_heading_envs=0.0,
            rel_standing_envs=T0_STANDING_PROPORTION,
            rel_world_envs=0.0,
            rel_forward_envs=0.0,
            ranges=El4090VelocityCommandCfg.Ranges(
                lin_vel_x=T0_TRAIN_VX_RANGE,
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),
                heading=None,
            ),
            # Inert for T0 ranges: commands are exactly 0 (standing) or in
            # [0.25, 0.35]; nothing falls inside (0, 0.2].
            cmd_deadzone=0.2,
        )
    }

    events = {
        "reset_base": EventTermCfg(
            func=t0_reset_root_witness,
            mode="reset",
        ),
        "reset_robot_joints": EventTermCfg(
            func=t0_reset_joints_witness,
            mode="reset",
        ),
        "reset_gait_state": EventTermCfg(
            func=mdp.reset_gait_state,
            mode="reset",
        ),
    }

    # Rewards: the pillar task's stage-0 set, except dof_pos_limits (removed —
    # the uncalibrated nominal limits are disabled in this variant) and with
    # the base_height target read from the T1 witness instead of 0.25 m.
    rewards: dict = {}
    for name, ref in make_el_4090_lidar_env_cfg().rewards.items():
        if name == "dof_pos_limits":
            continue
        if name == "base_height":
            rewards[name] = RewardTermCfg(
                func=mdp.base_height,
                weight=ref.weight,
                params={"target": w.settled_root_height_m},
            )
        else:
            rewards[name] = ref

    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "fell_over": TerminationTermCfg(func=mdp.fell_over),
        "outside_flat_domain": TerminationTermCfg(func=outside_flat_domain),
    }

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(
                terrain_type="generator",
                terrain_generator=TerrainGeneratorCfg(
                    size=(8.0, 8.0),
                    num_rows=2,
                    num_cols=2,
                    border_width=0.0,
                    curriculum=False,
                    sub_terrains={
                        # key must stay "pillar_field": get_el_4090_terrain_view looks
                        # the sub-terrain up under this name and isinstance-checks
                        # El4090PillarFieldCfg (El4090FlatFieldCfg subclasses it).
                        "pillar_field": El4090FlatFieldCfg(
                            representation=terrain_representation,
                            proportion=1.0,
                            size=(8.0, 8.0),
                            builder_cfg=make_t0_flat_terrain_config(),
                        )
                    },
                ),
                max_init_terrain_level=None,
            ),
            entities={"robot": get_el_4090_t0_robot_cfg()},
            sensors=(
                feet_ground_contact,
                penalised_contact,
                nonfoot_any_contact,
            ),
            num_envs=num_envs,
            env_spacing=8.0,
        ),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        curriculum={},
        sim=SimulationCfg(
            mujoco=MujocoCfg(timestep=SIM_TIMESTEP, gravity=(0.0, 0.0, -9.81)),
        ),
        decimation=DECIMATION,
        episode_length_s=episode_length_s,
    )

    if play:
        cfg.episode_length_s = int(1e9)
    return cfg
