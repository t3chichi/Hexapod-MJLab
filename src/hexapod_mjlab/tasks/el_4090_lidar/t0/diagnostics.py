"""Read-only native 200 Hz contact/state observer, without additional solves.

Solver contacts are from the pre-integration state; qpos/qvel are the
integrated state. Native boundary forward is observed separately. Manual
reset evaluation preserves the true terminal state.
"""

from __future__ import annotations

import torch
import warp as wp

from ..mdp.rewards import FEET_NAMES  # alphabetical order (legacy)

#: Penetration tolerance for "in contact" classification [m]. A contact slot
#: with dist > 0 is a near-but-not-touching broadphase entry (MuJoCo writes
#: margin contacts only when includemargin > 0; this model has none, but the
#: guard keeps the classifier honest if that ever changes).
CONTACT_DIST_MAX_M = 0.0


class T0ContactDiagnostics:
    """Per-substep contact + torque recorder for one live environment."""

    def __init__(self, env, record: bool = False):
        self.env = env
        self.device = env.device
        self.record = record
        self.active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self._orig_forward = None
        self._installed = False
        self._orig_scene_update = None
        self._orig_apply_action = None

        # ── Static classification tables from the host template model ──────────
        mm = env.sim.mj_model
        import mujoco

        ngeom = mm.ngeom
        geom_body = torch.tensor(mm.geom_bodyid, dtype=torch.long)
        body_names = [
            mujoco.mj_id2name(mm, mujoco.mjtObj.mjOBJ_BODY, int(b)) or "" for b in range(mm.nbody)
        ]
        # mjlab prefixes entity bodies with "<entity>/"; terrain bodies are
        # unprefixed children of the generated terrain entity.
        is_robot = torch.tensor([bn.startswith("robot/") for bn in body_names], dtype=torch.bool)
        foot_body_ids = {bn for bn in body_names if bn.split("/")[-1] in FEET_NAMES}
        is_foot = torch.tensor([bn in foot_body_ids for bn in body_names], dtype=torch.bool)
        self._geom_body = geom_body.to(env.device)
        self._geom_is_robot = is_robot[geom_body].to(env.device)
        self._geom_is_foot = is_foot[geom_body].to(env.device)
        self._body_names = body_names

        self.reset_accumulators()

    # ── accumulator lifecycle ────────────────────────────────────────────────

    def reset_accumulators(self) -> None:
        n = self.env.num_envs
        d = self.device
        self.substeps_seen = 0
        self.policy_steps_seen = 0
        self.apply_calls_seen = 0
        # contact event counts per env
        self.nonfoot_terrain_count = torch.zeros(n, dtype=torch.long, device=d)
        self.nonfoot_any_count = torch.zeros(n, dtype=torch.long, device=d)
        self.self_contact_count = torch.zeros(n, dtype=torch.long, device=d)
        self.foot_foot_count = torch.zeros(n, dtype=torch.long, device=d)
        self.foot_terrain_count = torch.zeros(n, dtype=torch.long, device=d)
        self.max_penetration = torch.zeros(n, device=d)
        self.nonfinite_state_count = torch.zeros(n, dtype=torch.long, device=d)
        # torque per env
        self.tau_preclip_max = torch.zeros(n, device=d)
        self.tau_saturation_count = torch.zeros(n, dtype=torch.long, device=d)
        self.tau_samples = torch.zeros(n, dtype=torch.long, device=d)
        self.first_events: dict[str, dict] = {}
        self.boundary_checks = 0
        self.overflow_count = torch.zeros(n, dtype=torch.long, device=d)
        self.overflow_bits = torch.zeros(n, dtype=torch.long, device=d)
        self.angle_domain_count = torch.zeros(n, dtype=torch.long, device=d)
        self.applied_tau_max = torch.zeros(n, device=d)
        self.active_samples = torch.zeros(n, dtype=torch.long, device=d)
        self.frames = []

    # ── install / uninstall (instance-level, read-only) ─────────────────────

    def install(self) -> None:
        if self._installed:
            raise RuntimeError("diagnostics already installed")
        scene = self.env.scene
        am = self.env.action_manager
        self._orig_scene_update = scene.update
        self._orig_apply_action = am.apply_action
        self._orig_forward = self.env.sim.forward
        diag = self

        def scene_update_wrapper(dt: float) -> None:
            diag._orig_scene_update(dt)
            diag._record_substep()

        def apply_action_wrapper() -> None:
            diag._orig_apply_action()
            diag._record_action()

        def forward_wrapper():
            diag._orig_forward()
            diag.boundary_checks += 1
            diag._classify_contacts()
            diag._check_state_finiteness()
            diag._check_overflow()

        self.env.sim.forward = forward_wrapper
        scene.update = scene_update_wrapper
        am.apply_action = apply_action_wrapper
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        self.env.sim.forward = self._orig_forward
        self.env.scene.update = self._orig_scene_update
        self.env.action_manager.apply_action = self._orig_apply_action
        self._installed = False

    # ── recorders ───────────────────────────────────────────────────────────

    def _record_action(self) -> None:
        """After apply_action: recompute the exact pre-clip PD torque."""
        self.apply_calls_seen += 1
        robot = self.env.scene["robot"]
        term = self.env.action_manager.get_term("joint_pd_torque")
        cfg = term.cfg
        target = term._processed_actions  # a * scale + default offset
        q = robot.data.joint_pos[:, term._target_ids]
        qd = robot.data.joint_vel[:, term._target_ids]
        tau = cfg.stiffness * (target - q) - cfg.damping * qd
        clip = cfg.effort_limit
        tau_abs = tau.abs()
        tau_abs = torch.nan_to_num(tau_abs)
        self.tau_preclip_max = torch.maximum(
            self.tau_preclip_max, tau_abs.max(dim=1).values * self.active
        )
        self.tau_saturation_count += (tau_abs > clip + 1e-9).sum(dim=1) * self.active
        self.tau_samples += tau.shape[1] * self.active

    def _check_overflow(self):
        bits = wp.to_torch(self.env.sim.wp_data.overflow).long()
        self.overflow_count += (bits != 0).long() * self.active
        self.overflow_bits |= bits * self.active

    def _record_substep(self) -> None:
        self.substeps_seen += 1
        self.active_samples += self.active.long()
        self._classify_contacts()
        self._check_state_finiteness()
        self._check_overflow()
        data = self.env.sim.wp_data
        qpos, qvel = wp.to_torch(data.qpos), wp.to_torch(data.qvel)
        self.angle_domain_count += (
            qpos[:, 7:].abs().amax(dim=1) > torch.pi + 1e-6
        ).long() * self.active
        tau = wp.to_torch(data.ctrl)
        self.applied_tau_max = torch.maximum(
            self.applied_tau_max, torch.nan_to_num(tau.abs()).amax(dim=1) * self.active
        )
        if self.record:
            self.frames.append(
                {
                    "qpos": qpos.detach().cpu().numpy().copy(),
                    "qvel": qvel.detach().cpu().numpy().copy(),
                    "ctrl": tau.detach().cpu().numpy().copy(),
                    "root_force_residual": (
                        wp.to_torch(data.qfrc_bias)
                        - wp.to_torch(data.qfrc_passive)
                        - wp.to_torch(data.qfrc_actuator)
                        - wp.to_torch(data.qfrc_constraint)
                    )[:, :6]
                    .detach()
                    .cpu()
                    .numpy()
                    .copy(),
                    "active": self.active.detach().cpu().numpy().copy(),
                    **{k: v.detach().cpu().numpy().copy() for k, v in self.counters().items()},
                }
            )

    def counters(self):
        return {
            "nonfoot_events": self.nonfoot_terrain_count,
            "self_events": self.self_contact_count,
            "nonfinite_events": self.nonfinite_state_count,
            "overflow_events": self.overflow_count,
            "overflow_bits": self.overflow_bits,
            "angle_domain_events": self.angle_domain_count,
            "preclip_peak": self.tau_preclip_max,
            "clip_count": self.tau_saturation_count,
            "tau_samples": self.tau_samples,
            "applied_peak": self.applied_tau_max,
            "substep_samples": self.active_samples,
        }

    def episode_safety(self, e):
        return {
            k: float(v[e]) if v.dtype.is_floating_point else int(v[e])
            for k, v in self.counters().items()
        }

    def save_physics(self, path):
        import numpy as np

        if path.exists():
            raise FileExistsError(path)
        np.savez_compressed(
            path, **{k: np.stack([f[k] for f in self.frames]) for k in self.frames[0]}
        )

    def _classify_contacts(self) -> None:
        wp_data = self.env.sim.wp_data
        n = int(wp.to_torch(wp_data.nacon).item())
        if n == 0:
            return
        geom = wp.to_torch(wp_data.contact.geom).view(-1, 2)[:n].long()
        dist = wp.to_torch(wp_data.contact.dist)[:n]
        worldid = wp.to_torch(wp_data.contact.worldid)[:n].long()
        touching = (dist <= CONTACT_DIST_MAX_M) & self.active[worldid]
        if not bool(touching.any()):
            return
        geom = geom[touching]
        worldid = worldid[touching]
        dist = dist[touching]

        b1 = self._geom_body[geom[:, 0]]
        b2 = self._geom_body[geom[:, 1]]
        r1 = self._geom_is_robot[geom[:, 0]]
        r2 = self._geom_is_robot[geom[:, 1]]
        f1 = self._geom_is_foot[geom[:, 0]]
        f2 = self._geom_is_foot[geom[:, 1]]

        num_envs = self.env.num_envs
        ones = torch.ones_like(worldid)

        foot_terrain = (r1 & f1 & ~r2) | (r2 & f2 & ~r1)
        nonfoot_terrain = (r1 & ~f1 & ~r2) | (r2 & ~f2 & ~r1)
        self_ = (r1 & r2) & (b1 != b2)
        foot_foot = self_ & f1 & f2
        # any violation-relevant non-foot involvement: non-foot body touching
        # the terrain, or a self contact that is not foot-vs-foot (e.g. a foot
        # swinging into its own or another leg's thigh/shank/hip)
        nonfoot_any = nonfoot_terrain | (self_ & ~(f1 & f2))

        self._acc(worldid, foot_terrain, self.foot_terrain_count, "foot_terrain")
        self._acc(
            worldid,
            nonfoot_terrain,
            self.nonfoot_terrain_count,
            "nonfoot_terrain",
            geom,
            dist,
            b1,
            b2,
        )
        self._acc(worldid, self_, self.self_contact_count, "self_contact", geom, dist, b1, b2)
        self._acc(worldid, foot_foot, self.foot_foot_count, "foot_foot", geom, dist, b1, b2)
        self._acc(worldid, nonfoot_terrain | self_, self.nonfoot_any_count, "nonfoot_any")

        # deepest penetration among violation classes (nonfoot + self)
        viol = nonfoot_terrain | self_
        if bool(viol.any()):
            pen = (-dist[viol]).clamp(min=0)
            wid_v = worldid[viol]
            pen_max = torch.zeros(num_envs, device=self.device)
            pen_max.index_reduce_(0, wid_v, pen, reduce="amax", include_self=True)
            self.max_penetration = torch.maximum(self.max_penetration, pen_max)

    def _acc(self, worldid, mask, counter, label, geom=None, dist=None, b1=None, b2=None) -> None:
        if not bool(mask.any()):
            return
        wid = worldid[mask]
        counter.index_add_(0, wid, torch.ones_like(wid))
        if label not in self.first_events:
            i = int(mask.nonzero()[0, 0])
            rec = {
                "substep": self.substeps_seen,
                "policy_step": self.policy_steps_seen,
                "env": int(worldid[i]),
                "dist_m": float(dist[i]) if dist is not None else None,
                "bodies": [
                    self._body_names[int(b1[i])] if b1 is not None else None,
                    self._body_names[int(b2[i])] if b2 is not None else None,
                ],
            }
            self.first_events[label] = rec

    def _check_state_finiteness(self) -> None:
        data = self.env.sim.wp_data
        ok = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.device)
        for key in ("qpos", "qvel", "qacc", "ctrl", "qfrc_actuator", "qfrc_constraint"):
            values = wp.to_torch(getattr(data, key))
            ok &= torch.isfinite(values).reshape(self.env.num_envs, -1).all(dim=1)
        self.nonfinite_state_count += (~ok).long() * self.active

    def note_policy_step(self) -> None:
        self.policy_steps_seen += 1

    # ── probes (policy-step level, read-only) ───────────────────────────────

    def probe_state(self) -> dict:
        """Per-env state snapshot for episode statistics."""
        robot = self.env.scene["robot"]
        pos = robot.data.root_link_pos_w
        quat = robot.data.root_link_quat_w  # wxyz
        grav = robot.data.projected_gravity_b
        tilt = torch.rad2deg(torch.acos(torch.clamp(-grav[:, 2], -1.0, 1.0)))
        lin_vel = robot.data.root_link_lin_vel_b
        ang_vel = robot.data.root_link_ang_vel_b
        return {
            "root_pos": pos.detach().clone(),
            "root_height": pos[:, 2].detach().clone(),
            "tilt_deg": tilt.detach().clone(),
            "lin_vel_b": lin_vel.detach().clone(),
            "ang_vel_b": ang_vel.detach().clone(),
            "joint_pos": robot.data.joint_pos.detach().clone(),
            "joint_vel": robot.data.joint_vel.detach().clone(),
            "actuator_force": (
                robot.data.actuator_force.detach().clone()
                if robot.data.actuator_force is not None
                else None
            ),
        }

    def foot_contacts(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(in_contact (B,6) bool alphabetical, force_z_up (B,6)) via the
        feet_ground_contact sensor.

        Reads the sensor directly (read-only); deliberately does NOT go through
        the reward step-state helper, which advances foot contact timers.
        """
        from mjlab.sensor import ContactSensor

        sensor: ContactSensor = self.env.scene["feet_ground_contact"]
        force = sensor.data.force
        assert force is not None
        n_prim = len(sensor.primary_names)
        force = force.view(self.env.num_envs, n_prim, 3).clone()
        perm = [sensor.primary_names.index(n) for n in FEET_NAMES]
        force = force[:, perm]
        force[..., 2] = -force[..., 2]  # old IsaacGym up-positive convention
        return force[..., 2] > 1.0, force[..., 2]

    # ── reporting ───────────────────────────────────────────────────────────

    def summary(self) -> dict:
        def cnt(t):
            return {
                "total": int(t.sum().item()),
                "envs_with_events": int((t > 0).sum().item()),
                "max_per_env": int(t.max().item()) if t.numel() else 0,
            }

        return {
            "substeps_seen": self.substeps_seen,
            "native_boundary_checks": self.boundary_checks,
            "overflow": cnt(self.overflow_count),
            "overflow_bits_per_env": self.overflow_bits.tolist(),
            "angle_domain": cnt(self.angle_domain_count),
            "policy_steps_seen": self.policy_steps_seen,
            "apply_calls_seen": self.apply_calls_seen,
            "coverage": {
                "physics_hz": 200,
                "recorded_substeps": self.substeps_seen,
                "note": (
                    "native solver contacts at every physics substep, no additional solve; "
                    "native boundary forward observes post-integration geometry; "
                    "manual reset is required to preserve terminal states"
                ),
            },
            "nonfoot_terrain": cnt(self.nonfoot_terrain_count),
            "nonfoot_any": cnt(self.nonfoot_any_count),
            "self_contact": cnt(self.self_contact_count),
            "foot_foot": cnt(self.foot_foot_count),
            "foot_terrain": cnt(self.foot_terrain_count),
            "nonfinite_state": cnt(self.nonfinite_state_count),
            "max_violation_penetration_m": float(self.max_penetration.max().item()),
            "first_events": self.first_events,
            "torque": {
                "preclip_max_abs_Nm": float(self.tau_preclip_max.max().item()),
                "saturation_fraction": (
                    float(self.tau_saturation_count.sum().item())
                    / max(int(self.tau_samples.sum().item()), 1)
                ),
                "samples": int(self.tau_samples.sum().item()),
            },
        }
