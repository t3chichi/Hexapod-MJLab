#!/usr/bin/env python
"""Actual GPU positive/negative contacts, raw-state CPU reconstruction."""

import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
import numpy as np
import torch, warp as wp, mujoco
from t0lib import PASS, FAIL, NOT_VERIFIED, StageReport, new_run_dir, write_json


def classify_cpu_contacts(qpos_list, with_ground=False, *, model=None, qvel=None):
    if model is None:
        from hexapod_mjlab.tasks.el_4090_lidar.t0.flat_env_cfg import (
            _get_el_4090_t0_spec,
            T0_TERRAIN_FRICTION,
        )

        spec = _get_el_4090_t0_spec()
        if with_ground:
            spec.worldbody.add_geom(
                name="recon_ground",
                type=mujoco.mjtGeom.mjGEOM_PLANE,
                size=[13, 13, 0.1],
                priority=1,
                friction=list(T0_TERRAIN_FRICTION),
            )
        model = spec.compile()
    d = mujoco.MjData(model)
    d.qpos[:] = qpos_list
    if qvel is not None:
        d.qvel[:] = qvel
    mujoco.mj_forward(model, d)

    def name(b):
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(b)) or "world"

    def robot(b):
        return (
            name(b).startswith("robot/")
            if any(name(i).startswith("robot/") for i in range(model.nbody))
            else b != 0
        )

    pairs = []
    nf = []
    sf = []
    feet = []
    for i in range(d.ncon):
        c = d.contact[i]
        if c.dist > 0:
            continue
        a, b = map(int, model.geom_bodyid[c.geom])
        na, nb = name(a), name(b)
        pair = [na, nb]
        pairs.append(pair)
        if robot(a) and robot(b) and a != b:
            sf.append(pair)
        elif robot(a) != robot(b):
            r = na if robot(a) else nb
            (feet if r.endswith("_FOOT") else nf).append(pair)
    return dict(
        ncon=int(d.ncon),
        touching_pairs=pairs,
        nonfoot_involved=nf,
        robot_robot_pairs=sf,
        foot_ground_pairs=feet,
        distinct_body_pairs=sorted({tuple(p) for p in pairs}),
        cpu_warnings=[
            mujoco.mjtWarning(i).name
            for i in range(mujoco.mjtWarning.mjNWARNING)
            if d.warning[i].number > 0
        ],
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    run = a.run_dir or new_run_dir()
    rep = StageReport("contact_gate", run)
    if a.num_envs != 8:
        rep.gate("frozen_8_worlds", FAIL)
        return rep.finalize()["returncode"]
    if not a.device.startswith("cuda") or not torch.cuda.is_available():
        rep.gate("cuda", NOT_VERIFIED)
        return rep.finalize()["returncode"]
    from mjlab.envs import ManagerBasedRlEnv
    from hexapod_mjlab.tasks.el_4090_lidar.t0 import make_el_4090_t0_flat_env_cfg
    from hexapod_mjlab.tasks.el_4090_lidar.t0.diagnostics import T0ContactDiagnostics
    from hexapod_mjlab.tasks.el_4090_lidar.t0.witness import frozen_witness, JOINT_ORDER

    cfg = make_el_4090_t0_flat_env_cfg(num_envs=8)
    cfg.auto_reset = False
    env = ManagerBasedRlEnv(cfg, device=a.device)
    env.reset()
    diag = T0ContactDiagnostics(env)
    diag.install()
    zero = torch.zeros(8, 18, device=a.device)
    k = 1
    ids = torch.tensor([k], device=a.device)
    others = [i for i in range(8) if i != k]
    w = frozen_witness()
    for _ in range(50):
        env.step(zero)
        diag.note_policy_step()
    good = diag.summary()
    cpu = classify_cpu_contacts(
        wp.to_torch(env.sim.wp_data.qpos)[k].cpu().numpy(),
        model=env.sim.mj_model,
        qvel=wp.to_torch(env.sim.wp_data.qvel)[k].cpu().numpy(),
    )
    rep.gate(
        "normal_feet_allowed",
        (
            PASS
            if good["nonfoot_any"]["total"] == 0
            and good["nonfinite_state"]["total"] == 0
            and good["overflow"]["total"] == 0
            and good["foot_terrain"]["total"] > 0
            and not cpu["robot_robot_pairs"]
            and not cpu["nonfoot_involved"]
            else FAIL
        ),
        cpu=cpu,
        summary=good,
    )
    # Observer invocation must not mutate ANY active dynamics/sensor/contact buffer.
    keys = ("qpos", "qvel", "qacc", "ctrl", "qfrc_constraint", "qfrc_actuator", "sensordata")
    before = {key: wp.to_torch(getattr(env.sim.wp_data, key)).clone() for key in keys}
    diag._record_substep()
    error = {
        key: float((wp.to_torch(getattr(env.sim.wp_data, key)) - v).abs().max()) if v.numel() else 0
        for key, v in before.items()
    }
    rep.gate(
        "observer_read_only",
        PASS if all(v == 0 for v in error.values()) else FAIL,
        buffer_max_difference=error,
    )
    cases = []

    def reset():
        env.reset()
        diag.reset_accumulators()

    def pose(z):
        origin = env.scene.env_origins[k]
        return torch.tensor([[float(origin[0]), float(origin[1]), z, 1, 0, 0, 0]], device=a.device)

    def capture(label):
        q = wp.to_torch(env.sim.wp_data.qpos)[k].cpu().numpy().copy()
        v = wp.to_torch(env.sim.wp_data.qvel)[k].cpu().numpy().copy()
        cpu = classify_cpu_contacts(q, model=env.sim.mj_model, qvel=v)
        cases.append(
            dict(label=label, qpos=q.tolist(), qvel=v.tolist(), cpu=cpu, gpu=diag.summary())
        )
        return cpu

    reset()
    jp = np.array(w.joint_pos)
    jp[[i for i, n in enumerate(JOINT_ORDER) if n.endswith("_HFE")]] = -1.2
    jp[[i for i, n in enumerate(JOINT_ORDER) if n.endswith("_KFE")]] = 2.0
    env.scene["robot"].write_joint_state_to_sim(
        torch.tensor(jp, dtype=torch.float32, device=a.device).view(1, 18),
        torch.zeros(1, 18, device=a.device),
        env_ids=ids,
    )
    env.scene["robot"].write_root_link_pose_to_sim(pose(0.30), env_ids=ids)
    env.sim.forward()
    cpu = capture("HIP_ground")
    hip_gpu = any(
        "HIP" in " ".join(e["bodies"])
        for name, e in diag.first_events.items()
        if name == "nonfoot_terrain"
    )
    hip_cpu = any("HIP" in " ".join(pair) for pair in cpu["nonfoot_involved"])
    rep.gate(
        "hip_actual_gpu_cpu",
        (
            PASS
            if hip_gpu
            and hip_cpu
            and bool((diag.nonfoot_terrain_count[others] == 0).all())
            and not cpu["cpu_warnings"]
            else FAIL
        ),
    )
    reset()
    jp = np.array(w.joint_pos)
    jp[JOINT_ORDER.index("RM_HAA")] = -2.2
    jp[JOINT_ORDER.index("RF_HAA")] = 2.2
    env.scene["robot"].write_joint_state_to_sim(
        torch.tensor(jp, dtype=torch.float32, device=a.device).view(1, 18),
        torch.zeros(1, 18, device=a.device),
        env_ids=ids,
    )
    env.sim.forward()
    cpu = capture("self_collision")
    rep.gate(
        "self_actual_gpu_cpu",
        (
            PASS
            if diag.self_contact_count[k] > 0
            and bool((diag.self_contact_count[others] == 0).all())
            and bool(cpu["robot_robot_pairs"])
            else FAIL
        ),
        cpu_robot_pairs=cpu["robot_robot_pairs"],
    )
    reset()
    env.scene["robot"].write_root_link_pose_to_sim(pose(1.5), env_ids=ids)
    env.sim.forward()
    cpu = capture("airborne")
    rep.gate(
        "airborne",
        (
            PASS
            if diag.foot_terrain_count[k] == 0
            and diag.nonfoot_terrain_count[k] == 0
            and not cpu["touching_pairs"]
            else FAIL
        ),
    )
    # Root velocity/quaternion and joint NaNs are read directly, without
    # asking a solver to operate on a deliberately corrupted state.
    nanchecks = {}
    for field, index in [("qvel", 0), ("qpos", 3), ("qpos", 7)]:
        reset()
        arr = wp.to_torch(getattr(env.sim.wp_data, field))
        arr[k, index] = float("nan")
        diag._check_state_finiteness()
        nanchecks[field + str(index)] = bool(
            diag.nonfinite_state_count[k] > 0 and (diag.nonfinite_state_count[others] == 0).all()
        )
    reset()
    diag._check_state_finiteness()
    rep.gate(
        "nan_root_and_joints_confined_reset",
        (
            PASS
            if all(nanchecks.values()) and diag.summary()["nonfinite_state"]["total"] == 0
            else FAIL
        ),
        probes=nanchecks,
    )
    mujoco.mj_saveModel(env.sim.mj_model, str(run / "contact_model.mjb"))
    diag.uninstall()
    env.close()
    # Actual backend warning/overflow positive example on the original hfield
    # representation. A finite state with truncated contacts is invalid.
    cfg = make_el_4090_t0_flat_env_cfg(num_envs=8, terrain_representation="hfield")
    cfg.auto_reset = False
    env = ManagerBasedRlEnv(cfg, device=a.device)
    env.reset()
    diag = T0ContactDiagnostics(env)
    origin = env.scene.env_origins[k]
    env.scene["robot"].write_root_link_pose_to_sim(
        torch.tensor([[float(origin[0]), float(origin[1]), 0.10, 1, 0, 0, 0]], device=a.device),
        env_ids=ids,
    )
    env.sim.forward()
    diag._check_overflow()
    diag._check_state_finiteness()
    rep.gate(
        "actual_gpu_overflow_detected",
        (
            PASS
            if diag.overflow_count[k] > 0 and bool((diag.overflow_count[others] == 0).all())
            else FAIL
        ),
        overflow_bits=diag.overflow_bits.tolist(),
        finite_events=diag.nonfinite_state_count.tolist(),
    )
    env.close()
    write_json(run / "contact_cases.json", cases)
    return rep.finalize()["returncode"]


if __name__ == "__main__":
    raise SystemExit(main())
