#!/usr/bin/env python
"""T0-A stage 2: flat-ground contract gate.

Verifies on the REAL El-4090-Lidar-T0-Flat environment (CUDA):

1. flat consistency — physics plane, LiDAR raycast hits and height-grid
   supervision all reference one flat plane (z=0): measured heights exactly
   zero, ground-hit world z ≈ 0, height_grid == clip(root_z-0.5)*5, feet
   actually standing on the physical ground;
2. clocks — control 50 Hz (0.02 s), physics 200 Hz (0.005 s, decimation 4),
   LiDAR 10 Hz (tick spacing 5 policy steps);
3. model contract — joint/actuator order, 18 nominal hinge limits disabled
   on the compiled env model (asset untouched), terrain friction pinned
   (μ=1.0, priority 1), actuator ctrlranges ±80 N·m;
4. observation contract — policy 2754 / height_grid 187 / actions 18, all
   finite across steps;
5. partial reset isolation — one env forced to fall+reset: only that env's
   root/joints return to the witness, only its LiDAR distal window and gait
   timers reset, all other envs continue undisturbed;
6. mount binding — pipeline mount == model site == runner exposure;
7. mirror contract — involution on live obs, identity-mount control, the
   empirical FK mirror-convention evidence (swap WITHOUT negation matches
   the geometric y-mirror within the model's own ~9.4 mm asymmetry), and
   the relative-obs/action default-offset conversion check;
8. commands — training resamples are exactly 0 (standing) or in
   [0.25, 0.35] with vy=0, yaw=0, standing fraction ≈ 0.20; heading
   conversion disabled; forced evaluation commands reach the observation;
9. collision-filter boundary — parent-child and welded robot body pairs
   that MuJoCo excludes by construction (recorded, not counted as
   self-collision).

Any required gate FAIL/NOT_VERIFIED exits non-zero.
"""

from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import t0lib  # noqa: E402
from t0lib import FAIL, NOT_VERIFIED, PASS, StageReport, new_run_dir  # noqa: E402


def fk_mirror_evidence() -> dict:
    """Empirical FK test: which joint swap/negate convention reproduces the
    geometric y-mirror of the robot. Pure classic-MuJoCo, no env needed."""
    import mujoco

    from hexapod_mjlab.tasks.el_4090_lidar.el_4090_lidar_env_cfg import (
        _EL4090_XML,
    )

    model = mujoco.MjModel.from_xml_path(str(_EL4090_XML))
    legs = ["RF", "RM", "RB", "LF", "LM", "LB"]
    parts = ["HIP", "THIGH", "SHANK", "FOOT"]
    pairs = [("RF", "LF"), ("RM", "LM"), ("RB", "LB")]
    mirror = np.diag([1.0, -1.0, 1.0])

    def fk(qmap):
        d = mujoco.MjData(model)
        mujoco.mj_resetData(model, d)
        d.qpos[2] = 0.5
        for n, q in qmap.items():
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
            d.qpos[model.jnt_qposadr[jid]] = q
        mujoco.mj_forward(model, d)
        return {
            b: d.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)].copy()
            for b in ["BASE"] + [f"{l}_{p}" for l in legs for p in parts]
        }

    rng = np.random.default_rng(0)
    trials = [
        {f"{l}_{p}": float(rng.uniform(-0.6, 0.6)) for l in legs for p in ("HAA", "HFE", "KFE")}
        for _ in range(5)
    ]

    def err(q0, nh, nf, nk):
        q1 = {}
        for rl, ll in pairs:
            for src, dst in ((rl, ll), (ll, rl)):
                for p, neg in (("HAA", nh), ("HFE", nf), ("KFE", nk)):
                    q1[f"{dst}_{p}"] = -q0[f"{src}_{p}"] if neg else q0[f"{src}_{p}"]
        P, P2 = fk(q0), fk(q1)
        errs = [
            np.abs(P2[f"{r}_{p}"] - mirror @ P[f"{l}_{p}"]).max() for r, l in pairs for p in parts
        ]
        errs.append(np.abs(P2["BASE"] - mirror @ P["BASE"]).max())
        return float(max(errs))

    rows = []
    for nh, nf, nk in product((False, True), repeat=3):
        es = [err(q0, nh, nf, nk) for q0 in trials]
        rows.append(
            {
                "negate_HAA": nh,
                "negate_HFE": nf,
                "negate_KFE": nk,
                "max_err_mm": round(max(es) * 1000, 3),
                "mean_err_mm": round(float(np.mean(es)) * 1000, 3),
            }
        )
    rows.sort(key=lambda r: r["max_err_mm"])
    best = rows[0]
    legacy_ok = not best["negate_HAA"] and not best["negate_HFE"] and not best["negate_KFE"]
    return {
        "conventions_sorted": rows,
        "best": best,
        "legacy_swap_no_negation_is_best": legacy_ok,
        "note": (
            "the model is not exactly mirror-symmetric (foot y differs ~9.5 mm at "
            "the zero pose), so the best possible residual is the model's own "
            "asymmetry; the legacy mirror convention (swap, no negation) is the "
            "geometric arg-min over all 8 swap/negate conventions"
        ),
    }


def collision_filter_boundary(env) -> dict:
    """Parent-child and welded robot body pairs excluded by MuJoCo."""
    import mujoco

    mm = env.sim.mj_model
    robot_bodies = []
    for b in range(mm.nbody):
        name = mujoco.mj_id2name(mm, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if name.startswith("robot/"):
            robot_bodies.append((b, name))
    parent_child, welded = [], []
    for i, (b1, n1) in enumerate(robot_bodies):
        for b2, n2 in robot_bodies[i + 1 :]:
            if mm.body_parentid[b2] == b1 or mm.body_parentid[b1] == b2:
                parent_child.append([n1, n2])
            elif mm.body_weldid[b1] == mm.body_weldid[b2]:
                welded.append([n1, n2])
    return {
        "robot_body_count": len(robot_bodies),
        "parent_child_pairs_excluded": parent_child,
        "same_weld_pairs_excluded": welded,
        "note": (
            "MuJoCo excludes same-body, direct parent-child and welded pairs by "
            "construction; every robot-robot contact that DOES appear in the raw "
            "arrays is a post-filter self collision (diagnostics counts those)"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    run_dir = args.run_dir or new_run_dir()
    rep = StageReport("contract_gate", run_dir)

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    from hexapod_mjlab.tasks.el_4090_lidar.mdp.lidar import (
        get_lidar_pipeline,
        resolve_lidar_mount,
    )
    from hexapod_mjlab.tasks.el_4090_lidar.mdp.observations import (
        measured_heights,
    )
    from hexapod_mjlab.tasks.el_4090_lidar.t0 import make_el_4090_t0_flat_env_cfg
    from hexapod_mjlab.tasks.el_4090_lidar.t0.diagnostics import T0ContactDiagnostics
    from hexapod_mjlab.tasks.el_4090_lidar.t0.flat_env_cfg import (
        T0_STANDING_PROPORTION,
        T0_TRAIN_VX_RANGE,
        lifted_witness_root,
    )
    from hexapod_mjlab.tasks.el_4090_lidar.t0.witness import frozen_witness

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        rep.gate(
            "cuda_available",
            NOT_VERIFIED,
            device=args.device,
            cuda_available=torch.cuda.is_available(),
            note="contract gate requires the real CUDA environment",
        )
        doc = rep.finalize()
        return doc["returncode"]
    rep.gate("cuda_available", PASS, device=torch.cuda.get_device_name(0))

    print(f"[contract] building T0 flat env ({args.num_envs} envs)...", flush=True)
    cfg = make_el_4090_t0_flat_env_cfg(num_envs=args.num_envs)
    env = ManagerBasedRlEnv(cfg, device=args.device)
    obs, _ = env.reset()
    robot = env.scene["robot"]
    w = frozen_witness()
    diag = T0ContactDiagnostics(env)
    diag.install()

    # ── settle a moment so the witness drop lands ────────────────────────────
    for _ in range(25):
        obs, rew, term, trunc, extras = env.step(torch.zeros(args.num_envs, 18, device=env.device))
        diag.note_policy_step()
    assert not bool(term.any()) and not bool(trunc.any()), "settling terminated"

    # ── 1: flat consistency ─────────────────────────────────────────────────
    heights = measured_heights(env)
    pipeline = get_lidar_pipeline(env)
    base_pos = robot.data.root_link_pos_w
    base_quat_wxyz = robot.data.root_link_quat_w

    def quat_apply_wxyz(q, v):
        qv, qw = q[..., :3], q[..., 3:4]
        t = 2.0 * torch.linalg.cross(qv, v, dim=-1)
        return v + qw * t + torch.linalg.cross(qv, t, dim=-1)

    pts_b = pipeline.lidar_points_base.clone()
    hits = pipeline.hit_mask
    # root_link_quat_w is wxyz (mjlab); rotate with the xyzw formula
    base_quat_xyzw = base_quat_wxyz[:, [1, 2, 3, 0]]
    pts_w = base_pos.unsqueeze(1) + quat_apply_wxyz(
        base_quat_xyzw.unsqueeze(1).expand(-1, pts_b.shape[1], -1), pts_b
    )
    hit_z = pts_w[hits][:, 2]
    frac_ground = (hit_z.abs() < 0.01).float().mean().item() if hits.any() else 0.0
    obs_hg = obs["height_grid"]
    expect_hg = (torch.clamp(base_pos[:, 2:3] - 0.5, -1.0, 1.0) * 5.0).expand(-1, 187)
    hg_err = (obs_hg - expect_hg).abs().max().item()
    in_contact, force_z = diag.foot_contacts()
    flat_ok = (
        bool((heights == 0).all())
        and frac_ground > 0.99
        and hg_err < 1e-5
        and int(in_contact.sum(dim=1).min().item()) >= 4
    )
    rep.gate(
        "flat_consistency",
        PASS if flat_ok else FAIL,
        measured_heights_all_zero=bool((heights == 0).all()),
        lidar_ground_hit_frac_within_1cm=round(frac_ground, 5),
        lidar_hits=int(hits.sum().item()),
        lidar_hit_world_z_absmax_m=round(float(hit_z.abs().max()) if hits.any() else -1, 5),
        height_grid_formula_max_err=hg_err,
        min_feet_in_contact=int(in_contact.sum(dim=1).min().item()),
        mean_root_height_m=round(float(base_pos[:, 2].mean()), 5),
        note="one flat field feeds physics plane, LiDAR warp mesh and 187-dim supervision",
    )

    # ── 2: clocks ───────────────────────────────────────────────────────────
    tick_log = pipeline.tick_log
    spacings = [b - a for a, b in zip(tick_log[:-1], tick_log[1:])]
    spacing_5 = sum(1 for s in spacings if s == 5)
    clocks_ok = (
        env.step_dt == 0.02
        and env.physics_dt == 0.005
        and env.cfg.decimation == 4
        and pipeline.pool_decimation == 5
        and len(spacings) >= 4
        and spacing_5 == len(spacings)
    )
    rep.gate(
        "clocks",
        PASS if clocks_ok else FAIL,
        control_hz=1.0 / env.step_dt,
        physics_hz=1.0 / env.physics_dt,
        decimation=env.cfg.decimation,
        lidar_hz=1.0 / (env.step_dt * pipeline.pool_decimation),
        tick_spacings_5_of=f"{spacing_5}/{len(spacings)}",
    )

    # ── 3: model contract ───────────────────────────────────────────────────
    import mujoco

    mm = env.sim.mj_model

    def strip(name: str) -> str:
        return name.split("/", 1)[-1] if name else name

    model_hinges = {
        j: strip(mujoco.mj_id2name(mm, mujoco.mjtObj.mjOBJ_JOINT, j))
        for j in range(mm.njnt)
        if mm.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
    }
    robot_hinge_ids = {j for j, n in model_hinges.items() if n in set(w.joint_names)}
    limits_disabled = (
        all(not bool(mm.jnt_limited[j]) for j in robot_hinge_ids) and len(robot_hinge_ids) == 18
    )
    # no non-robot hinge may exist (model = robot + terrain only)
    limits_disabled &= len(model_hinges) == 18
    terrain_friction = None
    terrain_priority = None
    for g in range(mm.ngeom):
        name = mujoco.mj_id2name(mm, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if mm.geom_bodyid[g] == mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_BODY, "terrain"):
            terrain_friction = [round(float(x), 5) for x in mm.geom_friction[g]]
            terrain_priority = int(mm.geom_priority[g])
            break
    act_names = [
        strip(mujoco.mj_id2name(mm, mujoco.mjtObj.mjOBJ_ACTUATOR, a)) for a in range(mm.nu)
    ]
    model_ok = (
        list(model_hinges.values()) == list(w.joint_names)
        and act_names == list(w.joint_names)
        and limits_disabled
        and terrain_friction is not None
        and abs(terrain_friction[0] - 1.0) < 1e-9
        and terrain_priority == 1
    )
    rep.gate(
        "model_contract",
        PASS if model_ok else FAIL,
        joint_order_matches=list(model_hinges.values()) == list(w.joint_names),
        actuator_order_matches=act_names == list(w.joint_names),
        nominal_hinge_limits_disabled=limits_disabled,
        terrain_friction=terrain_friction,
        terrain_priority=terrain_priority,
        actuator_ctrlrange_all_pm80=bool(
            (mm.actuator_ctrlrange[:, 0] == -80).all() and (mm.actuator_ctrlrange[:, 1] == 80).all()
        ),
    )

    # ── 4: observation contract ─────────────────────────────────────────────
    obs_ok = (
        obs["policy"].shape == (args.num_envs, 2754)
        and obs["height_grid"].shape == (args.num_envs, 187)
        and env.action_manager.total_action_dim == 18
        and bool(torch.isfinite(obs["policy"]).all())
        and bool(torch.isfinite(obs["height_grid"]).all())
    )
    rep.gate(
        "obs_contract",
        PASS if obs_ok else FAIL,
        policy_shape=list(obs["policy"].shape),
        height_grid_shape=list(obs["height_grid"].shape),
        action_dim=env.action_manager.total_action_dim,
        finite=bool(
            torch.isfinite(obs["policy"]).all() and torch.isfinite(obs["height_grid"]).all()
        ),
    )

    # ── 5: mount binding ────────────────────────────────────────────────────
    mount = resolve_lidar_mount()
    q_err = (pipeline._mount_q - torch.tensor(mount[1], device=env.device)).abs().max().item()
    t_err = (
        (pipeline._mount_t.view(-1) - torch.tensor(mount[0], device=env.device)).abs().max().item()
    )
    rep.gate(
        "mount_binding",
        PASS if q_err < 1e-6 and t_err < 1e-6 else FAIL,
        mount_pos=[round(x, 6) for x in mount[0]],
        mount_quat_xyzw=[round(x, 6) for x in mount[1]],
        quat_err=q_err,
        trans_err=t_err,
    )

    # ── 6: commands ─────────────────────────────────────────────────────────
    term_cmd = env.command_manager.get_term("twist")
    cmd_cfg = term_cmd.cfg
    # Sample resampling + the standing zeroing that _update_command applies
    # (the manager calls it every step); measure through the real obs pipeline.
    n_draws = 60
    standing = 0
    bad = 0
    vx_vals: list[float] = []
    env_ids = torch.arange(args.num_envs, device=env.device)
    for _ in range(n_draws):
        term_cmd._resample_command(env_ids)
        term_cmd._update_command()
        cmd = term_cmd.vel_command_b
        standing += int((cmd[:, :2].norm(dim=1) == 0).sum())
        vx_vals.extend(cmd[:, 0].tolist())
        bad += int(
            (
                (cmd[:, 0] != 0)
                & ~((cmd[:, 0] >= T0_TRAIN_VX_RANGE[0]) & (cmd[:, 0] <= T0_TRAIN_VX_RANGE[1]))
            ).sum()
        )
        bad += int((cmd[:, 1] != 0).sum()) + int((cmd[:, 2] != 0).sum())
    total = n_draws * args.num_envs
    standing_frac = standing / total
    cmd_ok = (
        bad == 0
        and abs(standing_frac - T0_STANDING_PROPORTION) < 0.05
        and cmd_cfg.heading_command is False
    )
    # forced evaluation command reaches the observation
    term_cmd.is_standing_env[:] = False
    term_cmd.is_world_env[:] = False
    term_cmd.vel_command_b[:] = torch.tensor([0.25, 0.0, 0.0], device=env.device).view(1, 3)
    obs_clean, *_ = env.step(torch.zeros(args.num_envs, 18, device=env.device))
    diag.note_policy_step()
    cmd_obs = obs_clean["policy"][:, 9:12]  # command term, scaled
    expected = torch.tensor([0.5, 0.0, 0.0], device=env.device).view(1, 3)
    forced_err = (cmd_obs - expected.expand(args.num_envs, 3)).abs().max().item()
    forced_ok = forced_err < 1e-5
    rep.gate(
        "commands_contract",
        PASS if cmd_ok and forced_ok else FAIL,
        draws=total,
        standing_fraction=round(standing_frac, 4),
        expected_standing_fraction=T0_STANDING_PROPORTION,
        out_of_range_or_nonzero_vy_yaw=bad,
        heading_command_disabled=cmd_cfg.heading_command is False,
        resampling_range_s=list(cmd_cfg.resampling_time_range),
        deadzone=cmd_cfg.cmd_deadzone,
        forced_eval_command_obs_err=forced_err,
        note="deadzone 0.2 is inert: commands are exactly 0 or in [0.25, 0.35]",
    )

    # ── 7: no violating contacts during the clean phase ─────────────────────
    s_clean = diag.summary()
    clean_ok = (
        s_clean["nonfoot_terrain"]["total"] == 0
        and s_clean["self_contact"]["total"] == 0
        and s_clean["nonfinite_state"]["total"] == 0
        and s_clean["substeps_seen"] == 4 * diag.policy_steps_seen
    )
    rep.gate(
        "no_violating_contacts_clean_phase",
        PASS if clean_ok else FAIL,
        substeps=s_clean["substeps_seen"],
        policy_steps=s_clean["policy_steps_seen"],
        substeps_per_policy_step=s_clean["substeps_seen"] / max(s_clean["policy_steps_seen"], 1),
        nonfoot_terrain=s_clean["nonfoot_terrain"]["total"],
        self_contact=s_clean["self_contact"]["total"],
        nonfinite=s_clean["nonfinite_state"]["total"],
        torque_preclip_max_Nm=round(s_clean["torque"]["preclip_max_abs_Nm"], 3),
    )

    # ── 8: mirror contract (clean obs, before the intentional flip) ─────────
    from tensordict import TensorDict

    from hexapod_mjlab.rl.symmetry_adapter import make_el_4090_symmetry_func

    sensor_trans = torch.tensor(mount[0], device=env.device).view(1, 3)
    sensor_quat = torch.tensor(mount[1], device=env.device).view(1, 4)
    from hexapod_mjlab.tasks.el_4090_lidar.t0.symmetry import t0_symmetry
    import functools

    mirror = functools.partial(t0_symmetry, env=env)
    obs_td = TensorDict(
        {
            "policy": obs_clean["policy"].clone(),
            "height_grid": obs_clean["height_grid"].clone(),
        },
        batch_size=[args.num_envs],
    )
    obs_m, _ = mirror(obs=obs_td, actions=None)
    obs_mm, _ = mirror(obs=obs_m[args.num_envs :], actions=None)
    inv_obs = (obs_mm["policy"][args.num_envs :] - obs_td["policy"]).abs().max().item()
    inv_hg = (obs_mm["height_grid"][args.num_envs :] - obs_td["height_grid"]).abs().max().item()
    actions_t = torch.randn(args.num_envs, 18, device=env.device)
    _, act_m = mirror(obs=None, actions=actions_t)
    _, act_mm = mirror(obs=None, actions=act_m[args.num_envs :])
    inv_act = (act_mm[args.num_envs :] - actions_t).abs().max().item()

    # identity-mount control: mirroring a point at the sensor translation must
    # move it (a wrong identity mount would not)
    pts = torch.zeros(args.num_envs, 256, 3, device=env.device)
    pts[:, 0] = sensor_trans.view(1, 3) + torch.tensor([0.3, 0.4, 0.1], device=env.device)
    td = TensorDict(
        {
            "policy": torch.cat(
                [
                    torch.zeros(args.num_envs, 66, device=env.device),
                    pts.reshape(args.num_envs, -1),
                    torch.zeros(args.num_envs, 1920, device=env.device),
                ],
                dim=-1,
            ),
            "height_grid": torch.zeros(args.num_envs, 187, device=env.device),
        },
        batch_size=[args.num_envs],
    )
    td_m, _ = mirror(obs=td, actions=None)
    moved = (td_m["policy"][args.num_envs :, 66:69] - pts[:, 0].reshape(-1, 3)).abs().max().item()

    fk_ev = fk_mirror_evidence()
    d = torch.tensor(w.joint_targets, device=env.device).view(1, 18)
    q = obs_td["policy"][:, 12:30] + d
    expected = torch.cat([q[:, 9:], q[:, :9]], dim=1) - d
    offset_invariance_err = (obs_m["policy"][args.num_envs :, 12:30] - expected).abs().max().item()
    absolute_target = d + 0.25 * actions_t
    expected_target = torch.cat([absolute_target[:, 9:], absolute_target[:, :9]], dim=1)
    action_target_err = (d + 0.25 * act_m[args.num_envs :] - expected_target).abs().max().item()

    mirror_ok = (
        inv_obs < 1e-4
        and inv_hg < 1e-4
        and inv_act < 1e-6
        and moved > 0.5
        and fk_ev["legacy_swap_no_negation_is_best"]
        and offset_invariance_err < 1e-6
        and action_target_err < 1e-6
    )
    rep.gate(
        "mirror_contract",
        PASS if mirror_ok else FAIL,
        involution_policy_obs=inv_obs,
        involution_height_grid=inv_hg,
        involution_actions=inv_act,
        sensor_mount_control_move_m=round(moved, 4),
        fk_convention_best=fk_ev["best"],
        legacy_convention_is_geometric_argmin=fk_ev["legacy_swap_no_negation_is_best"],
        relative_obs_offset_conversion_err=offset_invariance_err,
        absolute_action_target_err=action_target_err,
        note=fk_ev["note"],
    )

    # ── 9: partial reset isolation (intentionally creates violations in ONE
    #    env; contact events from this section are a positive control for the
    #    diagnostics and are gated in contact_gate.py, not here) ─────────────
    window_before = pipeline.distal_window.clone()
    frames_before = pipeline.distal_frame_count.clone()
    pos_before = robot.data.root_link_pos_w.clone()
    k = 3
    origin_k = env.scene.env_origins[k]
    lifted_pos, lifted_quat = lifted_witness_root()
    # 180 deg roll (upside down) above the ground: fell_over must fire
    flipped_pose = torch.tensor(
        [origin_k[0], origin_k[1], 0.35, 0.0, 1.0, 0.0, 0.0],
        dtype=torch.float32,
        device=env.device,
    ).view(1, 7)
    robot.write_root_link_pose_to_sim(flipped_pose, env_ids=torch.tensor([k], device=env.device))
    obs2, rew2, term2, trunc2, _ = env.step(torch.zeros(args.num_envs, 18, device=env.device))
    diag.note_policy_step()
    others = [i for i in range(args.num_envs) if i != k]
    wpos = robot.data.root_link_pos_w
    wj = robot.data.joint_pos
    env_k_reset_ok = (
        bool(term2[k])
        and not bool(term2[others].any())
        and (wpos[k, 0] - (origin_k[0] + lifted_pos[0])).abs() < 0.02
        and (wpos[k, 1] - (origin_k[1] + lifted_pos[1])).abs() < 0.02
        and (wpos[k, 2] - (origin_k[2] + lifted_pos[2])).abs() < 0.02
        and (wj[k] - torch.tensor(w.joint_pos, device=env.device)).abs().max() < 1e-5
    )
    window_after = pipeline.distal_window
    frames_after = pipeline.distal_frame_count
    env_k_window_zero = bool((window_after[k] == 0).all())
    others_window_nonzero = bool((window_after[others].abs().sum(dim=(1, 2)) > 0).all())
    others_frames_advanced = bool((frames_after[others] == frames_before[others] + 1).all())
    reset_ok = (
        env_k_reset_ok and env_k_window_zero and others_window_nonzero and others_frames_advanced
    )
    s_after_flip = diag.summary()
    flip_nonfoot = s_after_flip["nonfoot_terrain"]["total"] - s_clean["nonfoot_terrain"]["total"]
    rep.gate(
        "partial_reset_isolation",
        PASS if reset_ok else FAIL,
        forced_env=k,
        forced_env_terminated=bool(term2[k]),
        others_terminated=bool(term2[others].any()),
        forced_env_root_back_at_witness=env_k_reset_ok,
        forced_env_distal_window_zeroed=env_k_window_zero,
        others_distal_windows_nonzero=others_window_nonzero,
        others_frame_counts_advanced_by_one=others_frames_advanced,
        others_root_continuity_max_jump_m=round(
            float((wpos[others] - pos_before[others]).norm(dim=1).max()),
            5,
        ),
        flip_induced_nonfoot_events=flip_nonfoot,
        note=(
            "the flipped env terminates on ORIENTATION within one policy step "
            "(4 substeps of free fall do not reach the ground from z=0.35), so "
            "no contact event is expected here; real ground-contact positive "
            "controls run in contact_gate.py"
        ),
    )

    # ── 10: collision-filter boundary ───────────────────────────────────────
    boundary = collision_filter_boundary(env)
    rep.gate("collision_filter_boundary_recorded", PASS, **boundary)

    # no contact violations may leak into other envs during the flip step
    flip_events_env_k_only = bool((diag.nonfoot_terrain_count[others] == 0).all())
    rep.gate(
        "no_contact_leak_into_other_envs",
        PASS if flip_events_env_k_only else FAIL,
        per_env_nonfoot_counts=diag.nonfoot_terrain_count.cpu().tolist(),
    )

    s = diag.summary()
    diag.uninstall()
    rep.meta["diag_summary"] = s
    rep.meta["vx_values_summary"] = {
        "min": round(min(vx_vals), 4),
        "max": round(max(vx_vals), 4),
    }

    env.close()
    doc = rep.finalize()
    return doc["returncode"]


if __name__ == "__main__":
    raise SystemExit(main())
