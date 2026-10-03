#!/usr/bin/env python
"""Fail closed on source/model/config/raw data identity before executing replay."""

import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
import torch
from t0lib import PASS, FAIL, NOT_VERIFIED, StageReport, new_run_dir, read_json
from provenance import verify_inputs, verify_outputs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path)
    a = p.parse_args()
    out = a.out_dir or new_run_dir(prefix="t0a-replay")
    rep = StageReport("replay_verify", out)
    try:
        issues = verify_inputs(a.run_dir) + verify_outputs(a.run_dir)
    except Exception as exc:
        issues = [repr(exc)]
    rep.gate("complete_frozen_identity", FAIL if issues else PASS, issues=issues)
    if issues:
        return rep.finalize()["returncode"]
    from eval_episodes import load_episodes, summarize
    from hold_gate import verify_hold_record, offline_static

    summary = summarize(load_episodes(a.run_dir))
    saved = read_json(a.run_dir / "eval_summary.json")["summary"]
    rep.gate(
        "summary_independent_raw_recompute",
        PASS if summary == saved else FAIL,
        counts=summary["per_command"],
    )
    measured = verify_hold_record(a.run_dir)
    hold = read_json(a.run_dir / "hold_gate.json")
    original = next(g["evidence"] for g in hold["gates"] if g["name"] == "all_substep_T1_hold")
    rep.gate("hold_raw_recompute", PASS if measured == original else FAIL)
    import mujoco, numpy as np

    model = mujoco.MjModel.from_binary_path(str(a.run_dir / "hold_model.mjb"))
    raw = np.load(a.run_dir / "hold_physics.npz", allow_pickle=False)
    recalculated = [
        offline_static(model, raw["qpos"][-1, e], raw["qvel"][-1, e]) for e in range(32)
    ]
    old = read_json(a.run_dir / "hold_static.json")
    close = all(
        s["checks"] == o["checks"]
        and np.allclose(s["com_w"], o["com_w"], atol=1e-9, rtol=0)
        and abs(s["support_margin_m"] - o["support_margin_m"]) < 1e-9
        for s, o in zip(recalculated, old)
    )
    rep.gate("COM_support_recompute", PASS if close else FAIL)
    if not torch.cuda.is_available():
        rep.gate("checkpoint_forward", NOT_VERIFIED)
        return rep.finalize()["returncode"]
    from dataclasses import asdict
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
    from hexapod_mjlab.tasks.el_4090_lidar.t0 import make_el_4090_t0_flat_env_cfg
    from hexapod_mjlab.tasks.el_4090_lidar.t0.flat_env_cfg import T0_TASK_ID

    env = ManagerBasedRlEnv(make_el_4090_t0_flat_env_cfg(num_envs=2, play=True), device="cuda:0")
    wrapped = RslRlVecEnvWrapper(env)
    runner = load_runner_cls(T0_TASK_ID)(
        wrapped, asdict(load_rl_cfg(T0_TASK_ID)), str(out), "cuda:0"
    )
    runner.load(str(a.run_dir / "t0_short_model.pt"))
    runner.alg.actor.eval()
    obs = wrapped.get_observations()
    with torch.no_grad():
        actions = runner.alg.actor(obs)
    step = env.step(actions)
    good = (
        actions.shape == (2, 18)
        and bool(torch.isfinite(actions).all())
        and bool(torch.isfinite(step[0]["policy"]).all())
    )
    rep.gate("checkpoint_forward", PASS if good else FAIL)
    env.close()
    return rep.finalize()["returncode"]


if __name__ == "__main__":
    raise SystemExit(main())
