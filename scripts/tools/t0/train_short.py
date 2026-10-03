#!/usr/bin/env python
"""T0-A stage 5: fixed-budget REAL CUDA short-PPO pipeline.

Budget (frozen, not expandable): 32 envs × 24 steps/env × 5 iterations,
seed 1 — at most 3,840 environment control steps — with a 20-minute wall
clock budget including all compilation. No dummy training, no CPU stand-in,
no checkpoint loading: a real environment + real LidarPPO5x update loop on
the registered ``El-4090-Lidar-T0-Flat`` task.

Checks:
* device is CUDA and network parameters live on the GPU;
* per-iteration loss dicts (PPO surrogate/value/entropy + symmetry mirror +
  auxiliary height MSE) are all finite;
* gradients are finite immediately before every optimizer step;
* actor parameters actually changed from their pre-learn snapshot;
* exactly num_envs × num_steps_per_env × iters env control steps were taken
  (the wrapper counts steps);
* checkpoint saved (with SHA-256), then RELOADED into a fresh runner on the
  same env: deterministic forward outputs match the original actor on a
  fixed obs batch (same-hardware float tolerance) and the env
  common_step_counter is restored from the checkpoint infos.

On exceeding the wall clock the run is marked NOT_VERIFIED with logs saved.
Any failure exits non-zero.
"""

from __future__ import annotations

import argparse
import gc
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

import t0lib  # noqa: E402
from t0lib import (  # noqa: E402
    FAIL,
    NOT_VERIFIED,
    PASS,
    StageReport,
    new_run_dir,
    sha256_file,
    write_json,
)

WALL_BUDGET_S = 20 * 60


class WallBudgetExceeded(RuntimeError):
    pass


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--budget-min", type=float, default=20.0)
    parser.add_argument("--oom-evidence", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir or new_run_dir()
    rep = StageReport("train_short", run_dir)
    global _ACTIVE_REPORT
    _ACTIVE_REPORT = rep
    budget_s = args.budget_min * 60
    recipe_ok = (
        args.iters == 5 and args.seed == 1 and args.num_envs in (16, 32) and 0 < budget_s <= 1200
    )
    if args.num_envs == 16:
        try:
            prior = t0lib.read_json(args.oom_evidence)
            recipe_ok &= prior["returncode"] != 0 and any(
                "out of memory" in str(g).lower() for g in prior["gates"]
            )
        except Exception:
            recipe_ok = False
    rep.gate(
        "frozen_recipe",
        PASS if recipe_ok else FAIL,
        num_envs=args.num_envs,
        iters=args.iters,
        seed=args.seed,
        budget_s=budget_s,
    )
    from provenance import verify_inputs

    prerequisites = []
    for stage in ("contract_gate", "contact_gate", "hold_gate", "terminal_gate"):
        path = run_dir / (stage + ".json")
        prerequisites.append(
            path.is_file() and t0lib.read_json(path).get("pipeline_passed") is True
        )
    try:
        identity_ok = not verify_inputs(run_dir)
    except Exception:
        identity_ok = False
    ready = all(prerequisites) and identity_ok
    rep.gate("prepared_gates_and_identity", PASS if ready else FAIL)
    if not recipe_ok or not ready:
        return rep.finalize()["returncode"]

    from dataclasses import asdict

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import (
        load_env_cfg,
        load_rl_cfg,
        load_runner_cls,
    )

    from hexapod_mjlab.tasks.el_4090_lidar.t0.flat_env_cfg import T0_TASK_ID

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        rep.gate(
            "cuda_available",
            NOT_VERIFIED,
            note="short training gate REQUIRES real CUDA (no CPU/dummy path)",
        )
        return rep.finalize()["returncode"]
    rep.gate(
        "cuda_available",
        PASS,
        device=torch.cuda.get_device_name(0),
        free_mb=t0lib.gpu_smi_free_mb(),
    )

    t_start = time.monotonic()

    def on_alarm(signum, frame):
        raise WallBudgetExceeded(f"wall budget {budget_s:.0f}s exceeded (includes compilation)")

    signal.signal(signal.SIGALRM, on_alarm)
    signal.setitimer(signal.ITIMER_REAL, budget_s)

    torch.cuda.reset_peak_memory_stats()
    env_cfg = load_env_cfg(T0_TASK_ID)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    agent_cfg = load_rl_cfg(T0_TASK_ID)
    agent_cfg.max_iterations = args.iters
    agent_cfg.seed = args.seed
    agent_cfg.num_steps_per_env = 24
    agent_cfg.experiment_name = f"t0_{run_dir.name}"
    agent_cfg.logger = "tensorboard"
    runner_cls = load_runner_cls(T0_TASK_ID)

    # resolved-config freeze (before the runner mutates its dict in-place)
    write_json(
        run_dir / "resolved_env_cfg.json",
        {"task": T0_TASK_ID, "seed": args.seed, "num_envs": args.num_envs, "cfg": asdict(env_cfg)},
    )
    write_json(
        run_dir / "resolved_agent_cfg.json",
        {
            "max_iterations": args.iters,
            "num_steps_per_env": 24,
            "seed": args.seed,
            "cfg": asdict(agent_cfg),
        },
    )

    print(f"[train] building env ({args.num_envs} envs, seed {args.seed})...", flush=True)
    env = ManagerBasedRlEnv(env_cfg, device=args.device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    torch.manual_seed(args.seed)
    runner = runner_cls(wrapped, asdict(agent_cfg), str(run_dir), args.device)
    alg = runner.alg

    # explicit seeding beyond mjlab's env seed (recorded honestly)
    import numpy as np

    np.random.seed(args.seed)

    params_before = {k: v.detach().clone() for k, v in alg.actor.state_dict().items()}

    # instrument update() and optimizer.step() (read-only wrappers)
    loss_dicts: list[dict] = []
    orig_update = alg.update

    def update_wrapper():
        out = orig_update()
        loss_dicts.append({k: float(v) for k, v in out.items()})
        return out

    alg.update = update_wrapper
    grad_finite_all = True
    grad_checks = 0
    grad_max_abs = 0.0
    optimizer = alg.optimizer
    orig_step = optimizer.step

    def step_wrapper(*a, **kw):
        nonlocal grad_finite_all, grad_checks, grad_max_abs
        gmax = 0.0
        ok = True
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    ok &= bool(torch.isfinite(p.grad).all())
                    gmax = max(gmax, float(p.grad.abs().max()))
        grad_finite_all &= ok
        grad_checks += 1
        grad_max_abs = max(grad_max_abs, gmax)
        return orig_step(*a, **kw)

    optimizer.step = step_wrapper

    step_counter = {"n": 0}
    orig_env_step = wrapped.step

    def step_count_wrapper(actions):
        step_counter["n"] += 1
        return orig_env_step(actions)

    wrapped.step = step_count_wrapper

    # network really on GPU?
    on_gpu = all(p.is_cuda for p in alg.actor.parameters()) and all(
        p.is_cuda for p in alg.critic.parameters()
    )
    rep.gate("network_on_cuda", PASS if on_gpu else FAIL, params_on_gpu=on_gpu)

    ckpt_path = run_dir / "t0_short_model.pt"
    timed_out = False
    error: Exception | None = None
    try:
        print("[train] learn() starting", flush=True)
        runner.learn(num_learning_iterations=args.iters, init_at_random_ep_len=True)
        runner.save(str(ckpt_path))
    except WallBudgetExceeded as exc:
        timed_out = True
        error = exc
    except Exception as exc:  # noqa: BLE001 - record and fail honestly
        error = exc
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)

    train_time_s = time.monotonic() - t_start
    peak_mb = torch.cuda.max_memory_allocated() / 1e6

    if error is not None:
        rep.gate(
            "training_completed",
            NOT_VERIFIED if timed_out else FAIL,
            error=repr(error),
            wall_time_s=round(train_time_s, 1),
            note=(
                "wall budget exceeded (logs preserved)"
                if timed_out
                else "training raised before completion"
            ),
        )
        doc = rep.finalize()
        return doc["returncode"]

    # ── post-training checks ────────────────────────────────────────────────
    step_calls = step_counter["n"]
    env_control_steps = step_calls * args.num_envs
    expected_calls = 24 * args.iters
    expected_steps = expected_calls * args.num_envs
    steps_ok = (
        step_calls == expected_calls
        and env_control_steps <= 3840
        and len(loss_dicts) == 5
        and train_time_s <= budget_s
    )
    rep.gate(
        "training_completed",
        PASS if steps_ok else FAIL,
        wall_time_s=round(train_time_s, 1),
        wall_budget_s=budget_s,
        env_step_calls=step_calls,
        expected_step_calls=expected_calls,
        env_control_steps=env_control_steps,
        expected_env_control_steps=expected_steps,
        budget_cap_env_control_steps=3840,
        within_frozen_budget=env_control_steps <= 3840,
        iterations_completed=len(loss_dicts),
        torch_peak_mb=round(peak_mb, 0),
    )

    keys_expected = ("value_function", "surrogate", "entropy")
    losses_finite = len(loss_dicts) == args.iters and all(
        all((k not in d) or (d[k] == d[k] and abs(d[k]) < float("inf")) for k in d)
        for d in loss_dicts
    )
    has_symmetry = all("symmetry" in d for d in loss_dicts)
    has_aux = all("aux" in d for d in loss_dicts)
    rep.gate(
        "losses_finite",
        (
            PASS
            if losses_finite
            and has_symmetry
            and has_aux
            and all(all(k in d for k in keys_expected) for d in loss_dicts)
            else FAIL
        ),
        per_iteration=loss_dicts,
        symmetry_reported=has_symmetry,
        aux_reported=has_aux,
    )
    rep.gate(
        "gradients_finite",
        PASS if grad_finite_all and grad_checks > 0 else FAIL,
        optimizer_steps=grad_checks,
        grad_max_abs=grad_max_abs,
    )

    params_after = alg.actor.state_dict()
    max_change = max(float((params_after[k] - params_before[k]).abs().max()) for k in params_before)
    rep.gate(
        "parameters_changed",
        PASS if max_change > 0 else FAIL,
        max_abs_param_change=max_change,
    )

    saved = ckpt_path.exists()
    rep.gate(
        "checkpoint_saved",
        PASS if saved else FAIL,
        path=str(ckpt_path),
        sha256=sha256_file(ckpt_path) if saved else None,
        size_bytes=ckpt_path.stat().st_size if saved else 0,
    )

    # ── reload + deterministic forward parity ───────────────────────────────
    if saved:
        obs_batch = wrapped.get_observations()
        with torch.no_grad():
            # deterministic forward returns the distribution mean directly
            ref_mean = alg.actor(obs_batch).detach().clone()
        step_before = wrapped.unwrapped.common_step_counter

        runner2 = runner_cls(wrapped, asdict(agent_cfg), str(run_dir / "reload"), args.device)
        infos = runner2.load(str(ckpt_path))
        with torch.no_grad():
            alg2 = runner2.alg
            new_mean = alg2.actor(obs_batch).detach().clone()
        fwd_err = float((new_mean - ref_mean).abs().max())
        step_restored = wrapped.unwrapped.common_step_counter == step_before
        it_saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)["iter"]
        reload_ok = fwd_err < 1e-5 and step_restored and it_saved == args.iters - 1
        rep.gate(
            "checkpoint_reload_forward_parity",
            PASS if reload_ok else FAIL,
            deterministic_forward_max_abs_err=fwd_err,
            common_step_counter_restored=step_restored,
            checkpoint_iter=it_saved,
            expected_iter=args.iters - 1,
            infos_keys=sorted(infos.keys()) if isinstance(infos, dict) else None,
            tolerance="same-hardware float32, max-abs < 1e-5",
        )
        del runner2
        gc.collect()
        torch.cuda.empty_cache()

    # rsl_rl logger artifacts (TB events + per-iteration model_*.pt) were
    # written directly into run_dir (logger log_dir == run_dir); register them.
    ckpts = {}
    for pt in sorted(run_dir.glob("model_*.pt")):
        ckpts[pt.name] = {"sha256": sha256_file(pt), "size_bytes": pt.stat().st_size}
    rep.meta["iteration_checkpoints"] = ckpts
    rep.meta["tb_event_files"] = [
        str(p.relative_to(run_dir)) for p in sorted(run_dir.glob("events.out.tfevents.*"))
    ]
    rep.meta["loss_dicts"] = loss_dicts

    env.close()
    doc = rep.finalize()
    return doc["returncode"]


_ACTIVE_REPORT = None


def main():
    try:
        return _main()
    except Exception as exc:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if _ACTIVE_REPORT is None:
            raise
        _ACTIVE_REPORT.gate(
            "construction_or_reload_error",
            NOT_VERIFIED if isinstance(exc, WallBudgetExceeded) else FAIL,
            error=repr(exc),
        )
        return _ACTIVE_REPORT.finalize()["returncode"]
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    raise SystemExit(main())
