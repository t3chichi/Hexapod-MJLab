#!/usr/bin/env python
"""M5a throughput benchmark: new mjlab stack vs old G5 (Isaac Gym) numbers.

Port of the old G5 bench
(``Lidar_legged_gym/legged_gym/legged_gym/tests/pd_gru_lidar_head/throughput_bench.py``,
read-only reference).  Old G5 methodology, run headless with zero actions at
``num_envs=16`` (WARMUP=10 / MEASURE=100 control steps):

* ``tick_ms_mean``     -- ``env.post_physics_step()`` wall time, which contains
  the decimation-gated 10 Hz LiDAR tick (pose transform + warp raycast +
  domain rand + pool gather); averaged over ALL steps, i.e. 1-in-5 steps carry
  the real tick.  ``tick_ms_per_control_step`` divided by ``pool_decimation``
  again (documented here for interpretation; see report).
* ``wrap_obs_ms_mean`` -- runner ``LidarWrapper.wrap_obs`` (FPS 256 + angular
  sort + 64-point downsample + 10-frame ring) per control step.
* ``step_ms_mean``     -- full ``env.step(zero_action)`` wall time.
* comparison: old low-res task (SIMPLE_GRID 1000 rays) vs new full-Airy task
  (86400 rays) => the LiDAR-resolution cost ratio (``step_time_ratio`` 1.122).

New-stack ON/OFF mapping (the honest equivalent chosen here):

* **ON**  = the M3b production path: 10 Hz tick with the full Airy 86400-ray
  warp raycast + pool sampling, plus per-policy-step prox/distal production
  (``LidarWrapper.wrap_obs``) inside the observation manager.
* **OFF** = the M3a zero-placeholder path: obs terms return pre-zeroed
  prox/distal blocks (same 2754-dim policy obs, no tick, no wrap).  This is
  literally the committed M3a class semantics (``git show 6ba9edf:
  src/.../mdp/lidar.py``), so the ON/OFF step-time delta = the whole LiDAR
  sensing stack cost -- the new-stack analogue of the old bench's
  low-res-vs-full-Airy step-time delta.  The old bench never ran a truly
  LiDAR-less task either; both sides keep the full 2754-dim obs pipeline.

Sub-metrics (like the old bench's per-component timing):

* ``tick_raw_ms`` -- a forced ``pipeline._tick()`` (transform + 3 syncs + warp
  raycast + DR + pool write), median of >=10 forced ticks.
* ``wrap_ms``     -- ``pipeline.wrapper.wrap_obs(...)`` per policy step.

Differences vs old G5 kept honest:

* timing bracketed by ``torch.cuda.synchronize()`` (mjlab steps are async on
  cuda; the old bench relied on isaacgym's internal sync points);
* ``num_envs`` swept over {16, 128, 256, 512, 1024} -- 16 for a direct
  apples-to-apples row against the old G5 numbers (same GPU, same env count),
  the rest per the M5a spec; OOMs are caught cleanly and skipped;
* policy steps here are env-only (zero actions, no PPO update), exactly like
  the old G5 step metric; training throughput is lower (adds PPO update).

Results -> ``tests/data/m5/throughput_new.json``.

Usage (hp_mjlab conda env)::

    python scripts/bench/throughput_bench.py            # full sweep
    python scripts/bench/throughput_bench.py --num-envs 128
    python scripts/bench/throughput_bench.py --worker 128  # internal (subprocess)
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

OUT_DIR = REPO / "tests" / "data" / "m5"
TASK_ID = "El-4090-Lidar"
WARMUP, MEASURE_STEPS = 10, 60  # >= 50 policy steps per point (spec)
NUM_ENVS_SWEEP = (16, 128, 256, 512, 1024)

# Old G5 recorded results (same laptop GPU, RTX 4050 6 GB, num_envs=16) for
# the comparison table; source file read-only.  Optional input — gracefully
# skipped when absent.  Point LIDAR_LEGGED_GYM at your old-repo checkout.
OLD_G5_JSON = Path(
  os.environ.get("LIDAR_LEGGED_GYM", "/home/t3chichi/Lidar_legged_gym")
) / (
  "legged_gym/legged_gym/tests/"
  "pd_gru_lidar_head/logs/acceptance/g5_throughput.json"
)


def _sync(device: str) -> None:
  if device.startswith("cuda"):
    import torch

    torch.cuda.synchronize()


class _ZeroLidarPipeline:
  """M3a zero-placeholder shim (obs terms read ``_prox_block``/``_distal_block``).

  Reproduces the committed M3a semantics: zero prox/distal blocks, no tick,
  no wrap.  Installed by swapping ``env._el_4090_lidar_pipeline``.
  """

  def __init__(self, env):
    import torch

    self.num_envs = env.num_envs
    self.device = env.device
    self.total_ticks = 0
    self._prox_block = torch.zeros(self.num_envs, 768, device=self.device)
    self._distal_block = torch.zeros(self.num_envs, 1920, device=self.device)

  def ensure_current(self) -> None:  # no-op: blocks stay zero
    return None


def smi_process_mb() -> float | None:
  try:
    out = subprocess.run(
      ["nvidia-smi", "--query-compute-apps=pid,used_memory",
       "--format=csv,noheader,nounits"],
      capture_output=True, text=True, timeout=20,
    ).stdout
    for line in out.strip().splitlines():
      parts = [p.strip() for p in line.split(",")]
      if len(parts) == 2 and int(parts[0]) == os.getpid():
        return float(parts[1])
  except Exception:
    pass
  return None


def bench_worker(num_envs: int, out_path: str) -> int:
  import torch

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  from hexapod_mjlab.tasks.el_4090_lidar.mdp.lidar import get_lidar_pipeline

  device = "cuda:0"
  result: dict = {"num_envs": num_envs, "status": "ok"}
  try:
    torch.cuda.reset_peak_memory_stats()
    cfg = load_env_cfg(TASK_ID)
    cfg.scene.num_envs = num_envs
    t0 = time.perf_counter()
    env = ManagerBasedRlEnv(cfg, device=device)
    env_build_s = time.perf_counter() - t0
    obs, _ = env.reset()
    assert obs["policy"].shape == (num_envs, 2754)
    pipeline = get_lidar_pipeline(env)
    zero_actions = torch.zeros(num_envs, 18, device=env.device)

    # ── ON: full LiDAR stack (tick + wrap inside step) ──────────────────
    for _ in range(WARMUP):
      env.step(zero_actions)
    _sync(device)
    step_times_on = []
    for _ in range(MEASURE_STEPS):
      _sync(device)
      t0 = time.perf_counter()
      env.step(zero_actions)
      _sync(device)
      step_times_on.append(time.perf_counter() - t0)
    n_ticks_during = pipeline.total_ticks

    # ── sub-metrics (ON): forced tick + direct wrap_obs ─────────────────
    for _ in range(3):
      pipeline._tick()
    _sync(device)
    tick_times = []
    for _ in range(12):
      _sync(device)
      t0 = time.perf_counter()
      pipeline._tick()
      _sync(device)
      tick_times.append(time.perf_counter() - t0)

    dones = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    empty_proprio = torch.zeros(num_envs, 0, device=env.device)
    for _ in range(5):
      pipeline.wrapper.wrap_obs(empty_proprio, pipeline.lidar_points_base, dones)
    _sync(device)
    wrap_times = []
    for _ in range(MEASURE_STEPS):
      _sync(device)
      t0 = time.perf_counter()
      pipeline.wrapper.wrap_obs(empty_proprio, pipeline.lidar_points_base, dones)
      _sync(device)
      wrap_times.append(time.perf_counter() - t0)

    hit_rate = pipeline.hit_mask.float().mean().item()

    # ── OFF: M3a zero-placeholder path (no tick, no wrap) ───────────────
    env._el_4090_lidar_pipeline = _ZeroLidarPipeline(env)
    for _ in range(WARMUP):
      env.step(zero_actions)
    _sync(device)
    step_times_off = []
    for _ in range(MEASURE_STEPS):
      _sync(device)
      t0 = time.perf_counter()
      env.step(zero_actions)
      _sync(device)
      step_times_off.append(time.perf_counter() - t0)

    mean = statistics.mean
    step_on_ms = mean(step_times_on) * 1e3
    step_off_ms = mean(step_times_off) * 1e3
    tick_raw_ms = statistics.median(tick_times) * 1e3
    wrap_ms = mean(wrap_times) * 1e3
    lidar_delta_ms = step_on_ms - step_off_ms

    result.update({
      "status": "ok",
      "device": torch.cuda.get_device_name(0),
      "raw_rays": 86400,
      "pool_points": 2880,
      "env_build_s": round(env_build_s, 1),
      "hit_rate": round(hit_rate, 4),
      "ticks_during_on_measure": n_ticks_during,
      "on": {
        "step_ms_mean": round(step_on_ms, 3),
        "policy_steps_per_s": round(1e3 / step_on_ms, 1),
        "env_steps_per_s": round(num_envs * 1e3 / step_on_ms, 1),
      },
      "off": {
        "step_ms_mean": round(step_off_ms, 3),
        "policy_steps_per_s": round(1e3 / step_off_ms, 1),
        "env_steps_per_s": round(num_envs * 1e3 / step_off_ms, 1),
      },
      "lidar_stack": {
        "delta_step_ms": round(lidar_delta_ms, 3),
        "cost_share_of_on_step": round(lidar_delta_ms / step_on_ms, 4),
        "tick_raw_ms": round(tick_raw_ms, 3),
        "tick_amortized_ms_per_policy_step": round(tick_raw_ms / 5.0, 3),
        "wrap_obs_ms_mean": round(wrap_ms, 3),
        "step_time_ratio_on_over_off": round(step_on_ms / step_off_ms, 4),
      },
      "vram": {
        "torch_peak_mb": round(torch.cuda.max_memory_allocated() / 1e6, 0),
        "nvidia_smi_mb": smi_process_mb(),
      },
    })
  except torch.cuda.OutOfMemoryError as e:
    result = {"num_envs": num_envs, "status": "oom", "error": str(e)[:300]}
  except RuntimeError as e:
    # warp/mujoco-warp raise plain RuntimeError("Failed to allocate ...")
    # on device exhaustion — same condition as torch OOM.
    msg = str(e).lower()
    if "out of memory" in msg or "failed to allocate" in msg:
      result = {"num_envs": num_envs, "status": "oom", "error": str(e)[:300]}
    else:
      result = {"num_envs": num_envs, "status": "error", "error": str(e)[:300]}
  Path(out_path).write_text(json.dumps(result, indent=2))
  print(json.dumps(result, indent=2)[:2000])
  return 0 if result["status"] == "ok" else 1


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--num-envs", type=int, nargs="+",
                      default=list(NUM_ENVS_SWEEP))
  parser.add_argument("--worker", type=int, default=None,
                      help="internal: run a single env count in-process")
  parser.add_argument("--out", type=str, default=None,
                      help="internal: worker partial-json output path")
  args = parser.parse_args()

  OUT_DIR.mkdir(parents=True, exist_ok=True)
  tmp = OUT_DIR / "tmp"
  tmp.mkdir(exist_ok=True)

  if args.worker is not None:
    return bench_worker(args.worker, args.out or str(tmp / f"partial.json"))

  rows: dict[str, dict] = {}
  for n in args.num_envs:
    partial = tmp / f"partial_{n}.json"
    print(f"\n=== num_envs={n} ===", flush=True)
    r = subprocess.run(
      [sys.executable, os.path.abspath(__file__),
       "--worker", str(n), "--out", str(partial)],
      timeout=1800, capture_output=True, text=True,
    )
    if partial.exists():
      rows[str(n)] = json.loads(partial.read_text())
    else:
      rows[str(n)] = {"num_envs": n, "status": "failed",
                      "error": (r.stderr or r.stdout)[-500:]}
    status = rows[str(n)]["status"]
    print(f"[m5-bench] num_envs={n}: {status}")
    if status != "ok":
      print((r.stderr or "")[-1200:])
    partial.unlink(missing_ok=True)

  old_g5 = json.loads(OLD_G5_JSON.read_text()) if OLD_G5_JSON.exists() else None
  results = {
    "task": TASK_ID,
    "benchmark": "throughput_bench (new stack, mjlab 1.6.0, mujoco-warp)",
    "methodology": {
      "warmup_steps": WARMUP,
      "measure_steps": MEASURE_STEPS,
      "actions": "zero (env-only, no PPO update — same as old G5)",
      "timing": "wall clock bracketed by torch.cuda.synchronize()",
      "on": "M3b full LiDAR stack (86400-ray Airy tick @10 Hz + wrap_obs "
            "per policy step)",
      "off": "M3a zero-placeholder (zero prox/distal blocks, no tick, no "
             "wrap; obs stays 2754-dim)",
      "old_g5_mapping": "old bench compared low-res(1000-ray) vs "
                        "full-Airy(86400-ray) tasks; new ON/OFF delta = "
                        "whole LiDAR stack cost. tick_raw_ms/wrap_ms match "
                        "the old tick_ms_mean/wrap_obs_ms_mean sub-metrics.",
    },
    "num_envs_sweep": rows,
    "old_g5_reference": old_g5,
  }
  out = OUT_DIR / "throughput_new.json"
  out.write_text(json.dumps(results, indent=2, ensure_ascii=False))

  # Console summary table.
  print("\n=== M5a throughput summary (new stack) ===")
  print(f"{'envs':>5} {'on ms':>8} {'off ms':>8} {'ratio':>6} "
        f"{'pol/s':>8} {'envsteps/s':>10} {'lidar%':>7} {'VRAM MB':>8}")
  for n, row in rows.items():
    if row.get("status") != "ok":
      print(f"{n:>5} {row.get('status', '?'):>8}")
      continue
    print(f"{n:>5} {row['on']['step_ms_mean']:>8.2f} "
          f"{row['off']['step_ms_mean']:>8.2f} "
          f"{row['lidar_stack']['step_time_ratio_on_over_off']:>6.3f} "
          f"{row['on']['policy_steps_per_s']:>8.1f} "
          f"{row['on']['env_steps_per_s']:>10.0f} "
          f"{100 * row['lidar_stack']['cost_share_of_on_step']:>6.1f}% "
          f"{row['vram']['torch_peak_mb']:>8.0f}")
  if old_g5:
    g5 = old_g5["new_lidar_head"]
    print(f"\nold G5 reference (Isaac Gym, 16 envs): step_ms={g5['step_ms_mean']} "
          f"tick_ms={g5['tick_ms_mean']} (1-in-5-gated mean) "
          f"wrap_ms={g5['wrap_obs_ms_mean']}")
    if rows.get("16", {}).get("status") == "ok":
      r = rows["16"]
      print(f"new stack 16 envs: step_ms={r['on']['step_ms_mean']} "
            f"tick_raw_ms={r['lidar_stack']['tick_raw_ms']} "
            f"wrap_ms={r['lidar_stack']['wrap_obs_ms_mean']}")
  print(f"\nresults -> {out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
