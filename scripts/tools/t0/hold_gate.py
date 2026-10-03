#!/usr/bin/env python
"""32 worlds x 5 seconds, all-substep T1 thresholds and offline static checks."""

import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch, warp as wp, mujoco
from t0lib import PASS, FAIL, NOT_VERIFIED, StageReport, new_run_dir, write_json
from morphology.meter import MorphologyMeter
from morphology.statics import equilibrium
from morphology.verify_hold import POLICY as T1_THRESHOLDS


def hull_margin(points, point):
    from scipy.spatial import ConvexHull, QhullError

    if len(points) < 3:
        return float("-inf")
    try:
        h = ConvexHull(points)
    except QhullError:
        return float("-inf")
    return float(
        np.min(
            -(h.equations[:, :2] @ point + h.equations[:, 2])
            / np.linalg.norm(h.equations[:, :2], axis=1)
        )
    )


def verify_hold_record(run):
    from hexapod_mjlab.tasks.el_4090_lidar.t0.witness import frozen_witness

    meta = __import__("t0lib").read_json(run / "hold_metadata.json")
    raw = np.load(run / "hold_physics.npz", allow_pickle=False)
    qp = raw["qpos"]
    qv = raw["qvel"]
    n = qp.shape[1]
    target = np.array(frozen_witness().joint_targets)
    rms = np.sqrt(np.mean((qp[:, :, 7:] - target) ** 2, axis=0))
    q = qp[:, :, 3:7]
    tilt = np.degrees(np.arccos(np.clip(1 - 2 * (q[:, :, 1] ** 2 + q[:, :, 2] ** 2), -1, 1)))
    drift = np.linalg.norm(qp[:, :, :2] - np.array(meta["initial_qpos"])[:, :2], axis=-1)
    height = frozen_witness().settled_root_height_m
    tail = qv[-100:]
    sf = meta["safety"]
    checks = dict(
        full_32_world_1000_substeps=qp.shape == (1000, 32, 25) and raw["active"].all(),
        finite=all(np.isfinite(v).all() for v in raw.values()),
        no_termination=not any(meta["terminated"]),
        no_violation=all(
            all(
                s[k] == 0
                for k in (
                    "nonfoot_events",
                    "self_events",
                    "nonfinite_events",
                    "overflow_events",
                    "angle_domain_events",
                )
            )
            for s in sf
        ),
        coverage=all(s["substep_samples"] == 1000 and s["tau_samples"] == 18000 for s in sf),
        mean_tracking=float(np.max(rms.mean(axis=1))) < 0.15,
        worst_tracking=float(rms.max()) < 0.2,
        tilt=float(tilt.max()) < 5,
        xy_drift=float(drift.max()) < 0.03,
        height_deviation=float(np.abs(qp[:, :, 2] - np.array(meta["initial_qpos"])[:, 2]).max())
        < 0.05,
        tail_linear=float(np.linalg.norm(tail[:, :, :3], axis=-1).max()) < 0.01,
        tail_angular=float(np.linalg.norm(tail[:, :, 3:6], axis=-1).max()) < 0.02,
        tail_joint=float(np.abs(tail[:, :, 6:]).max()) < 0.05,
        tail_height=float(np.abs(qp[-100:, :, 2].mean(axis=0) - height).max()) < 0.035,
        unactuated_force_balance=float(np.abs(raw["root_force_residual"][-100:, :, :3]).max())
        <= 2.0,
        unactuated_moment_balance=float(np.abs(raw["root_force_residual"][-100:, :, 3:]).max())
        <= 1.0,
        applied_torque=float(np.abs(raw["ctrl"]).max()) <= 80 + 1e-6,
        nominal_headroom=max(s["preclip_peak"] for s in sf) <= 64,
        angle_domain=float(np.abs(qp[:, :, 7:]).max()) <= np.pi + 1e-6,
    )
    return dict(
        checks={k: bool(v) for k, v in checks.items()},
        metrics=dict(
            mean_tracking_rmse_worst_env=float(rms.mean(axis=1).max()),
            worst_joint_time_rmse=float(rms.max()),
            max_tilt_deg=float(tilt.max()),
            max_drift_m=float(drift.max()),
            tail_joint_speed=float(np.abs(tail[:, :, 6:]).max()),
            tail_root_linear_speed=float(np.linalg.norm(tail[:, :, :3], axis=-1).max()),
            tail_root_angular_speed=float(np.linalg.norm(tail[:, :, 3:6], axis=-1).max()),
            peak_preclip_torque=max(s["preclip_peak"] for s in sf),
            tail_unactuated_force_max=float(np.abs(raw["root_force_residual"][-100:, :, :3]).max()),
            tail_unactuated_moment_max=float(
                np.abs(raw["root_force_residual"][-100:, :, 3:]).max()
            ),
        ),
    )


def offline_static(model, qpos, qvel):
    """Same compiled model, actual contact points and body IDs; never live sim."""
    from types import SimpleNamespace

    d = mujoco.MjData(model)
    d.qpos[:] = qpos
    d.qvel[:] = qvel
    mujoco.mj_forward(model, d)
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(model.nbody)
    ]
    ids = [i for i, n in enumerate(names) if n.startswith("robot/")]
    masses = model.body_mass[ids]
    com = (d.xipos[ids] * masses[:, None]).sum(axis=0) / masses.sum()
    points = []
    bearing = set()
    rows = []
    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = map(int, c.geom)
        b1, b2 = map(int, model.geom_bodyid[c.geom])
        a, b = names[b1], names[b2]
        f = np.zeros(6)
        mujoco.mj_contactForce(model, d, i, f)
        if c.dist <= 0 and f[0] > 1e-5 and a.startswith("robot/") != b.startswith("robot/"):
            rb = b1 if a.startswith("robot/") else b2
            if names[rb].endswith("_FOOT"):
                points.append(c.pos[:2].copy())
                bearing.add(rb)
                rows.append(
                    dict(geom1=g1, geom2=g2, point=c.pos.tolist(), normal_force=float(f[0]))
                )
    margin = hull_margin(np.array(points), com[:2])
    footgeoms = {
        g for g in range(model.ngeom) if names[int(model.geom_bodyid[g])].endswith("_FOOT")
    }
    jid = [i for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
    proxy = SimpleNamespace(
        model=model,
        data=d,
        contacts=lambda: {"foot_ground": rows},
        foot_geom_ids=footgeoms,
        signed_ground_distance=lambda p: float(p[2]),
        ground_normal=np.array([0.0, 0.0, 1.0]),
        force_basis=np.eye(3),
        joint_dof_adr=model.jnt_dofadr[jid],
        friction=1.0,
    )
    static = equilibrium(proxy, actual_contacts=True)
    inverse = mujoco.MjData(model)
    inverse.qpos[:] = qpos
    mujoco.mj_forward(model, inverse)
    inverse.qacc[:] = 0
    mujoco.mj_inverse(model, inverse)
    root = inverse.qfrc_inverse[:6]
    return dict(
        com_w=com.tolist(),
        body_ids=ids,
        body_mass_sum=float(masses.sum()),
        contact_points=[p.tolist() for p in points],
        bearing_feet=len(bearing),
        support_margin_m=margin if np.isfinite(margin) else None,
        static_certificate=static,
        inverse_root_wrench=root.tolist(),
        warnings=[
            mujoco.mjtWarning(i).name
            for i in range(mujoco.mjtWarning.mjNWARNING)
            if d.warning[i].number
        ],
        checks=dict(
            actual_support=len(bearing) >= 4 and margin >= 0.03,
            rigid_equilibrium=bool(static["valid"]),
            no_CPU_warning=not any(w.number for w in d.warning),
        ),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    run = a.run_dir or new_run_dir()
    rep = StageReport("hold_gate", run)
    if a.num_envs != 32:
        rep.gate("frozen_32_worlds", FAIL)
        return rep.finalize()["returncode"]
    if not a.device.startswith("cuda") or not torch.cuda.is_available():
        rep.gate("cuda", NOT_VERIFIED)
        return rep.finalize()["returncode"]
    from mjlab.envs import ManagerBasedRlEnv
    from hexapod_mjlab.tasks.el_4090_lidar.t0 import make_el_4090_t0_flat_env_cfg
    from hexapod_mjlab.tasks.el_4090_lidar.t0.diagnostics import T0ContactDiagnostics

    cfg = make_el_4090_t0_flat_env_cfg(num_envs=32)
    cfg.auto_reset = False
    env = ManagerBasedRlEnv(cfg, device=a.device)
    env.reset()
    initial = wp.to_torch(env.sim.wp_data.qpos).cpu().numpy().copy()
    diag = T0ContactDiagnostics(env, record=True)
    diag.install()
    ended = torch.zeros(32, dtype=torch.bool, device=a.device)
    for i in range(250):
        _, _, term, trunc, _ = env.step(torch.zeros(32, 18, device=a.device))
        diag.note_policy_step()
        ended |= term | trunc
        if (term | trunc).any():
            break
    diag.save_physics(run / "hold_physics.npz")
    mujoco.mj_saveModel(env.sim.mj_model, str(run / "hold_model.mjb"))
    meta = dict(
        initial_qpos=initial.tolist(),
        terminated=ended.tolist(),
        safety=[diag.episode_safety(e) for e in range(32)],
        diagnostics=diag.summary(),
        thresholds=T1_THRESHOLDS,
    )
    write_json(run / "hold_metadata.json", meta)
    measured = verify_hold_record(run)
    rep.gate("all_substep_T1_hold", PASS if all(measured["checks"].values()) else FAIL, **measured)
    finals = wp.to_torch(env.sim.wp_data.qpos).cpu().numpy()
    vels = wp.to_torch(env.sim.wp_data.qvel).cpu().numpy()
    static = [offline_static(env.sim.mj_model, finals[e], vels[e]) for e in range(32)]
    write_json(run / "hold_static.json", static)
    rep.gate(
        "actual_contact_COM_static_balance",
        PASS if all(all(s["checks"].values()) and not s["warnings"] for s in static) else FAIL,
        min_support_margin=min(
            s["support_margin_m"] for s in static if s["support_margin_m"] is not None
        ),
        body_mass_sum=static[0]["body_mass_sum"],
        worst_inverse_force=max(max(abs(x) for x in s["inverse_root_wrench"][:3]) for s in static),
    )
    diag.uninstall()
    env.close()
    return rep.finalize()["returncode"]


if __name__ == "__main__":
    raise SystemExit(main())
