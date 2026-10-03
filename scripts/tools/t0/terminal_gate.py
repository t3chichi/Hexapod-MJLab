#!/usr/bin/env python
"""Actual early fall through the production recorder, without auto-reset stitching."""

import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
import torch, numpy as np
from t0lib import PASS, FAIL, NOT_VERIFIED, StageReport, new_run_dir, write_json
from eval_episodes import collect_batch, CONTROL_KEYS, validate_episode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path)
    a = p.parse_args()
    run = a.run_dir or new_run_dir()
    rep = StageReport("terminal_gate", run)
    if not torch.cuda.is_available():
        rep.gate("cuda", NOT_VERIFIED)
        return rep.finalize()["returncode"]
    from mjlab.envs import ManagerBasedRlEnv
    from hexapod_mjlab.tasks.el_4090_lidar.t0 import make_el_4090_t0_flat_env_cfg

    cfg = make_el_4090_t0_flat_env_cfg(num_envs=8, play=True)
    cfg.auto_reset = False
    cfg.commands["twist"].resampling_time_range = (1e9, 1e9)
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    calls = 0

    def actor(obs):
        nonlocal calls
        if calls == 0:
            orig = env.scene.env_origins[1]
            env.scene["robot"].write_root_link_pose_to_sim(
                torch.tensor(
                    [[float(orig[0]), float(orig[1]), 0.35, 0, 1, 0, 0]], device=env.device
                ),
                env_ids=torch.tensor([1], device=env.device),
            )
        calls += 1
        return torch.zeros(8, 18, device=env.device)

    episodes = collect_batch(env, actor, "stand", 9001, run)
    victim = episodes[1]
    verdict = validate_episode(victim)
    peer = episodes[0]
    good = (
        victim["early_terminated"]
        and victim["n_steps"] < 100
        and victim["root_pos"][-1, 2] < 0.4
        and not verdict["passed"]
        and verdict["data_valid"]
        and peer["n_steps"] == 1000
        and not peer["early_terminated"]
    )
    np.savez_compressed(
        run / "terminal_episodes.npz",
        **{e["episode_id"] + "__" + k: e[k] for e in episodes for k in CONTROL_KEYS},
    )
    write_json(
        run / "terminal_metadata.json",
        [{k: v for k, v in e.items() if k not in (*CONTROL_KEYS, "physics")} for e in episodes],
    )
    rep.gate(
        "true_terminal_and_peer_isolation",
        PASS if good else FAIL,
        victim_verdict=verdict,
        victim_steps=victim["n_steps"],
        terminal_height=float(victim["root_pos"][-1, 2]),
        peer_steps=peer["n_steps"],
    )
    env.close()
    return rep.finalize()["returncode"]


if __name__ == "__main__":
    raise SystemExit(main())
