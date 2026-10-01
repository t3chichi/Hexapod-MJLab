#!/usr/bin/env python3
"""M1 gate G3 — CPU MuJoCo single-env stance verification for el_4090.

Loads the converted MJCF (src/hexapod_mjlab/assets/robots/el_4090/mjcf/el_4090.xml) plus a
ground plane (friction 1.0, matching the old terrain config), resets to the
`stand` keyframe, and runs the OLD legged_gym PD law in the loop for 400 steps
(2 s at the old sim dt 0.005 s):

    torque = kp * (qdes - q) - kd * qd,   clipped to ±effort (80 N*m)
    ctrl   = torque (motors: gear 1, unit gain)

with kp/kd from the old config (HAA/HFE/KFE = 130 N*m/rad / 2 N*m*s/rad) and
qdes = the keyframe joint angles (old default_dof_pos, all 0). This mirrors
legged_gym's `_compute_torques` P-control including the torque clip. The PD
stays in Python at migration time; MuJoCo only sees motor torques.

PASS criteria (tolerances justified inline):
  - all qpos/qvel finite at every step,
  - no MuJoCo warnings (bad constraints / integrator instability / etc.),
  - base height deviation from its initial value < 0.05 m (no collapse; the
    robot starts 5 mm above contact, so a few mm of settle + PD deflection is
    expected — 0.05 m catches falling-through or exploded states with margin),
  - mean joint tracking RMSE < 0.15 rad (gravity-loaded PD sag at kp=130 for a
    ~39 kg robot on 6 legs is expected well below this; 0.15 rad would indicate
    a wrong torque convention, sign flip, or joint mapping mismatch).
"""

from __future__ import annotations

import json
import pathlib

import mujoco
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
XML = REPO / "src/hexapod_mjlab/assets/robots/el_4090/mjcf/el_4090.xml"
JOINT_TABLE = REPO / "tests/robot/joint_table.json"

N_STEPS = 400           # 2 s at dt = 0.005
BASE_HEIGHT_TOL = 0.05  # m, deviation from initial base height
JOINT_RMSE_TOL = 0.15   # rad, mean joint tracking RMSE


def main() -> None:
    cfg = json.loads(JOINT_TABLE.read_text())["old_config"]
    kp_map, kd_map = cfg["stiffness"], cfg["damping"]
    torque_clip = cfg["torque_clip"]

    # --- scene = robot MJCF + ground plane ------------------------------------
    spec = mujoco.MjSpec.from_file(str(XML))
    spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE, name="floor", size=[0, 0, 0.1],
        friction=[1.0, 0.005, 0.0001])  # static friction 1.0 (old terrain cfg)
    model = spec.compile()
    assert model.opt.timestep == cfg["sim_dt"], (
        f"timestep {model.opt.timestep} != old sim dt {cfg['sim_dt']}")
    data = mujoco.MjData(model)

    # --- reset to the stand keyframe -------------------------------------------
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    assert kid >= 0, "stand keyframe missing"
    mujoco.mj_resetDataKeyframe(model, data, kid)
    qdes = model.key_qpos[kid, 7:].copy()  # old default_dof_pos
    base_z0 = float(data.qpos[2])
    print(f"[scene] nq={model.nq} nu={model.nu} dt={model.opt.timestep} "
          f"| keyframe stand: base z={base_z0:.5f} m, qdes={np.unique(qdes)}")

    # per-dof gains from joint-name suffix (HAA/HFE/KFE), actuator -> dof map
    jnames = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
              for i in range(model.njnt)]
    hinge_ids = [i for i, n in enumerate(jnames) if n != "root"]
    kp = np.array([kp_map[n.split("_")[1]] for n in jnames if n != "root"])
    kd = np.array([kd_map[n.split("_")[1]] for n in jnames if n != "root"])
    ctrl_of_dof = []  # actuator index driving each hinge dof
    for jid in hinge_ids:
        dof = model.jnt_dofadr[jid] - 6  # hinge dofs follow the 6 freejoint dofs
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid))
        ctrl_of_dof.append((dof, aid))
    ctrl_of_dof = np.array(ctrl_of_dof)

    # --- PD stance loop ---------------------------------------------------------
    base_z = np.empty(N_STEPS)
    sq_err = np.empty((N_STEPS, len(kp)))
    for step in range(N_STEPS):
        q = data.qpos[7:]
        qd = data.qvel[6:]
        tau = kp * (qdes - q) - kd * qd
        tau = np.clip(tau, -torque_clip, torque_clip)
        data.ctrl[ctrl_of_dof[:, 1]] = tau
        mujoco.mj_step(model, data)

        assert np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel)), (
            f"non-finite state at step {step}")
        assert all(w.number == 0 for w in data.warning), (
            f"mujoco warning at step {step}: "
            f"{[str(w.number) for w in data.warning if w.number]}")
        base_z[step] = data.qpos[2]
        sq_err[step] = (q - qdes) ** 2

    # --- gate assertions ---------------------------------------------------------
    max_dev = float(np.max(np.abs(base_z - base_z0)))
    rmse_per_joint = np.sqrt(sq_err.mean(axis=0))
    mean_rmse = float(rmse_per_joint.mean())
    max_joint_err = float(np.max(np.abs(np.sqrt(sq_err.mean(axis=0)))))

    print(f"[metrics] base z: initial={base_z0:.5f} final={base_z[-1]:.5f} "
          f"max|dz|={max_dev:.5f} m")
    print(f"[metrics] joint tracking: mean RMSE={mean_rmse:.4f} rad "
          f"(worst joint RMSE={rmse_per_joint.max():.4f} rad at "
          f"{jnames[1 + int(np.argmax(rmse_per_joint))]}), "
          f"max |q-qdes| over time={max_joint_err:.4f} rad")
    print(f"[metrics] contacts final step: {data.ncon}, "
          f"feet in contact: {sum(1 for c in range(data.ncon) for g in (data.contact[c].geom1, data.contact[c].geom2) if 'FOOT' in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, model.geom_dataid[g]) or ''))}")

    assert max_dev < BASE_HEIGHT_TOL, f"base height deviation {max_dev:.4f} >= {BASE_HEIGHT_TOL}"
    assert mean_rmse < JOINT_RMSE_TOL, f"joint RMSE {mean_rmse:.4f} >= {JOINT_RMSE_TOL}"
    print(f"[G3] PASS — stance stable for {N_STEPS} steps "
          f"({N_STEPS * model.opt.timestep:.1f} s), tolerances "
          f"(dz<{BASE_HEIGHT_TOL} m, RMSE<{JOINT_RMSE_TOL} rad) held")


if __name__ == "__main__":
    main()
