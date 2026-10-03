#!/usr/bin/env python
"""64 independent first episodes, manual reset, full 200 Hz raw evidence."""

import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
import numpy as np
import torch
import t0lib
from t0lib import PASS, FAIL, NOT_VERIFIED, StageReport, new_run_dir, write_json, sha256_file
from hexapod_mjlab.tasks.el_4090_lidar.t0.evaluation import (
    validate_episode,
    summarize,
    SAFETY_KEYS,
    TOTAL_STEPS,
    SETTLE_STEPS,
    T0B_THRESHOLDS,
)

CONTROL_KEYS = (
    "root_pos",
    "root_quat",
    "lin_vel_b",
    "ang_vel_b",
    "joint_pos",
    "joint_vel",
    "actuator_force",
    "actions",
    "cmd",
    "tilt_deg",
    "foot_contact",
    "foot_body_origin_speed",
    "safety_samples",
)


def collect_batch(env, actor, command, seed_base, run_dir):
    """Freeze first terminal sample before resetting dead worlds to step peers."""
    from hexapod_mjlab.tasks.el_4090_lidar.t0.diagnostics import T0ContactDiagnostics
    from hexapod_mjlab.tasks.el_4090_lidar.t0.witness import frozen_witness
    from hexapod_mjlab.tasks.el_4090_lidar.t0.flat_env_cfg import T0_EVAL_COMMANDS

    env.reset(seed=seed_base)
    robot = env.scene["robot"]
    term = env.command_manager.get_term("twist")
    vx = T0_EVAL_COMMANDS[command]
    term.is_standing_env[:] = False
    term.is_world_env[:] = False
    term.is_heading_env[:] = False
    term.vel_command_b[:] = torch.tensor([vx, 0.0, 0.0], device=env.device)
    obs = env.get_observations()  # updated command before the FIRST action
    initial_pos = robot.data.root_link_pos_w.cpu().numpy().copy()
    initial_q = robot.data.root_link_quat_w.cpu().numpy().copy()
    diag = T0ContactDiagnostics(env, record=True)
    diag.install()
    n = env.num_envs
    data = {k: [] for k in CONTROL_KEYS}
    ends = np.full(n, TOTAL_STEPS)
    reasons = [None] * n
    safeties = [None] * n
    for step in range(TOTAL_STEPS):
        with torch.no_grad():
            actions = actor(obs)
        obs, _, term, trunc, _ = env.step(actions)
        diag.note_policy_step()
        st = diag.probe_state()
        feet, _ = diag.foot_contacts()
        foot_ids = [
            robot.body_names.index(name)
            for name in ("LB_FOOT", "LF_FOOT", "LM_FOOT", "RB_FOOT", "RF_FOOT", "RM_FOOT")
        ]
        frame = {
            k: st[k].cpu().numpy().copy()
            for k in (
                "root_pos",
                "lin_vel_b",
                "ang_vel_b",
                "joint_pos",
                "joint_vel",
                "actuator_force",
                "tilt_deg",
            )
        }
        frame.update(
            root_quat=robot.data.root_link_quat_w.cpu().numpy().copy(),
            actions=env.action_manager.action.cpu().numpy().copy(),
            cmd=(obs["policy"][:, 9:12] / torch.tensor([2.0, 2.0, 0.25], device=env.device))
            .cpu()
            .numpy()
            .copy(),
            foot_contact=feet.cpu().numpy().copy(),
            foot_body_origin_speed=robot.data.body_link_lin_vel_w[:, foot_ids]
            .norm(dim=-1)
            .cpu()
            .numpy()
            .copy(),
            safety_samples=np.array(
                [[diag.episode_safety(e)[k] for k in SAFETY_KEYS] for e in range(n)]
            ),
        )
        for k in data:
            data[k].append(frame[k])
        newly = (term | trunc) & diag.active
        # Post-integration physical violations that termination does not cover
        # also stop this first attempt; failure is saved, never tried again.
        bad = diag.nonfinite_state_count + diag.overflow_count
        newly |= (bad > 0) & diag.active
        for e in torch.nonzero(newly).flatten().tolist():
            ends[e] = step + 1
            reasons[e] = dict(
                terminated=bool(term[e]), truncated=bool(trunc[e]), physical_anomaly=bool(bad[e])
            )
            safeties[e] = diag.episode_safety(e)
        diag.active[newly] = False
        if not bool(diag.active.any()):
            break
        if newly.any():
            env.reset(env_ids=torch.nonzero(newly).flatten())
    diag.uninstall()
    stacked = {k: np.stack(v) for k, v in data.items()}
    physics = {k: np.stack([f[k] for f in diag.frames]) for k in diag.frames[0]}
    tag = f"{command}_{seed_base}"
    physics_file = run_dir / (tag + "_physics.npz")
    diag.save_physics(physics_file)
    episodes = []
    for e in range(n):
        length = int(ends[e])
        length = min(length, len(data["root_pos"]))
        sf = safeties[e] or diag.episode_safety(e)
        qw, qx, qy, qz = initial_q[e]
        yaw = float(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))
        ep = dict(
            episode_id=f"{command}_seed{seed_base+e}",
            command_name=command,
            requested_vx=vx,
            seed=seed_base + e,
            n_steps=length,
            early_terminated=reasons[e] is not None,
            termination_step=length - 1 if reasons[e] is not None else None,
            termination_reason=reasons[e],
            witness_height=frozen_witness().settled_root_height_m,
            initial_root_pos=initial_pos[e].tolist(),
            initial_yaw_rad=yaw,
            safety=sf,
            any_violation=any(
                sf[k] != 0
                for k in (
                    "nonfoot_events",
                    "self_events",
                    "nonfinite_events",
                    "overflow_events",
                    "angle_domain_events",
                )
            ),
            physics_file=physics_file.name,
            physics_env_index=e,
        )
        ep.update({k: v[:length, e] for k, v in stacked.items()})
        ep["physics"] = {k: v[: length * 4, e] for k, v in physics.items()}
        episodes.append(ep)
    return episodes


def load_episodes(run_dir):
    doc = t0lib.read_json(run_dir / "eval_summary.json")
    raw = np.load(run_dir / "episodes.npz", allow_pickle=False)
    cache = {}
    episodes = []
    for meta in doc["episodes_meta"]:
        ep = dict(meta)
        tag = meta["episode_id"]
        ep.update({k: raw[tag + "__" + k] for k in CONTROL_KEYS})
        file = meta["physics_file"]
        if file not in cache:
            cache[file] = np.load(run_dir / file, allow_pickle=False)
        ep["physics"] = {
            k: v[: ep["n_steps"] * 4, meta["physics_env_index"]] for k, v in cache[file].items()
        }
        episodes.append(ep)
    return episodes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--num-envs", type=int, default=32, choices=(16, 32))
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    run = a.run_dir or new_run_dir()
    rep = StageReport("eval_episodes", run)
    if not a.device.startswith("cuda") or not torch.cuda.is_available():
        rep.gate("cuda", NOT_VERIFIED)
        return rep.finalize()["returncode"]
    from dataclasses import asdict
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
    from hexapod_mjlab.tasks.el_4090_lidar.t0 import make_el_4090_t0_flat_env_cfg
    from hexapod_mjlab.tasks.el_4090_lidar.t0.flat_env_cfg import T0_TASK_ID

    cfg = make_el_4090_t0_flat_env_cfg(num_envs=a.num_envs, play=True)
    cfg.auto_reset = False
    cfg.commands["twist"].resampling_time_range = (1e9, 1e9)
    env = ManagerBasedRlEnv(cfg, device=a.device)
    wrapped = RslRlVecEnvWrapper(env)
    runner = load_runner_cls(T0_TASK_ID)(
        wrapped, asdict(load_rl_cfg(T0_TASK_ID)), str(run), a.device
    )
    runner.load(str(a.checkpoint))
    runner.alg.actor.eval()
    episodes = []
    for command in ("stand", "forward_025"):
        for start in range(0, 32, a.num_envs):
            episodes += collect_batch(env, runner.alg.actor, command, 1001 + start, run)
    raw = {ep["episode_id"] + "__" + k: ep[k] for ep in episodes for k in CONTROL_KEYS}
    np.savez_compressed(run / "episodes.npz", **raw)
    summary = summarize(episodes)
    write_json(
        run / "eval_summary.json",
        dict(
            schema=2,
            checkpoint_sha256=sha256_file(a.checkpoint),
            settle_steps=100,
            measure_steps=900,
            total_steps=1000,
            summary=summary,
            episodes_meta=[
                {k: v for k, v in ep.items() if k not in (*CONTROL_KEYS, "physics")}
                for ep in episodes
            ],
        ),
    )
    rep.gate(
        "64_first_episodes_saved",
        (
            PASS
            if len(episodes) == 64
            and all(summary["per_command"][c]["n"] == 32 for c in ("stand", "forward_025"))
            else FAIL
        ),
    )
    # A data validity failure blocks the pipeline; legitimate physics/walking
    # failures remain valid evidence, not a prerequisite of this short run.
    rep.gate(
        "raw_data_replayable",
        PASS if all(v["data_valid"] for v in summary["verdicts"]) else FAIL,
        summary=summary["per_command"],
    )
    replay = summarize(load_episodes(run))
    rep.gate("summary_roundtrip", PASS if replay == summary else FAIL)
    rep.meta["walking_summary"] = summary["per_command"]
    rep.meta["learned_walking_passed"] = all(
        summary["per_command"][c]["passed"] >= 29 for c in ("stand", "forward_025")
    )
    env.close()
    return rep.finalize()["returncode"]


if __name__ == "__main__":
    raise SystemExit(main())
