#!/usr/bin/env python
"""M5 baseline comparison: matched scalars from a new-stack (mjlab) and an
old-stack (Isaac Gym) baseline run directory.

Inputs are the run directories created by ``run_baseline_mjlab.sh`` /
``run_baseline_isaacgym.sh`` (each contains the training TensorBoard event
files — the machine-readable per-iteration log — plus ``run_meta.json`` and
the full ``train_stdout.log``).  Both stacks log rsl_rl-style scalars:

* ``Train/mean_reward``, ``Train/mean_episode_length`` (identical names)
* ``Loss/<name>`` (value_function / surrogate / symmetry / aux / ...)
* ``Loss/learning_rate``, ``Perf/total_fps``, ``Perf/collection_time``,
  ``Perf/learning_time``
* actor noise std is named ``Policy/mean_std`` (new rsl_rl) vs
  ``Policy/mean_noise_std`` (old rsl_rl) — aliased here.

Outputs ``tests/data/m5/baseline_comparison.{json,md}`` with:
* per-scalar final value + mean of the last 10% of iterations,
* wall time, env steps, iterations, env counts (from run_meta.json when
  present),
* a fair-comparison note (same env count / iterations / seed required;
  hardware + engine differences stated; friction-combination semantics
  differ — see ``tests/data/m5/friction_semantics.md``).

Usage (mjlab conda env — needs tensorboard)::

    python scripts/baseline_compare.py --new logs/m5/baseline_new/<ts> \\
        --old logs/m5/baseline_old/<ts>
    python scripts/baseline_compare.py --extract <run_dir>   # TB -> per_iteration.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "tests" / "data" / "m5"

# canonical name -> (new-stack TB tag, old-stack TB tag)
SCALAR_MAP = {
  "mean_reward": ("Train/mean_reward", "Train/mean_reward"),
  "mean_episode_length": ("Train/mean_episode_length", "Train/mean_episode_length"),
  "loss_value_function": ("Loss/value_function", "Loss/value_function"),
  "loss_surrogate": ("Loss/surrogate", "Loss/surrogate"),
  "loss_learning_rate": ("Loss/learning_rate", "Loss/learning_rate"),
  "loss_symmetry": ("Loss/symmetry", "Loss/symmetry"),
  "loss_aux": ("Loss/aux", "Loss/aux"),
  "policy_noise_std": ("Policy/mean_std", "Policy/mean_noise_std"),
  "perf_total_fps": ("Perf/total_fps", "Perf/total_fps"),
  "perf_collection_time": ("Perf/collection_time", "Perf/collection time"),
  "perf_learning_time": ("Perf/learning_time", "Perf/learning_time"),
}


def find_event_dirs(run_dir: Path) -> list[Path]:
  """Directories containing tfevents files (searched recursively).

  New-stack harness runs keep their TB events in the mjlab logger's own run
  dir (``logs/rsl_rl/{experiment}/{ts}/``) rather than inside the harness
  metadata dir; those are located via the ``mjlab_log_dir`` field recorded
  in ``run_meta.json`` (run_baseline_mjlab.sh).
  """
  dirs = sorted({p.parent for p in run_dir.rglob("events.out.tfevents*")})
  if not dirs:
    ref = read_meta(run_dir).get("mjlab_log_dir")
    if ref:
      ref_dir = Path(ref)
      if not ref_dir.is_absolute():
        ref_dir = run_dir.parents[3] / ref  # repo-root-relative path
      dirs = sorted({p.parent for p in ref_dir.rglob("events.out.tfevents*")})
  return dirs


def load_scalars(run_dir: Path, tag_map: dict[str, str]) -> dict[str, list[tuple[int, float]]]:
  """Load TB scalars for the tags in tag_map (canonical -> TB tag)."""
  from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
  )

  scalars: dict[str, list[tuple[int, float]]] = {}
  for ev_dir in find_event_dirs(run_dir):
    acc = EventAccumulator(str(ev_dir), size_guidance={"scalars": 0})
    acc.Reload()
    available = set(acc.Tags().get("scalars", []))
    for canon, tag in tag_map.items():
      if tag in available and canon not in scalars:
        scalars[canon] = [(s.step, s.value) for s in acc.Scalars(tag)]
  return scalars


def stats(series: list[tuple[int, float]]) -> dict:
  vals = [v for _, v in series]
  n_tail = max(1, len(vals) // 10)
  return {
    "iterations": len(vals),
    "final": round(vals[-1], 6) if vals else None,
    "mean_last10pct": round(sum(vals[-n_tail:]) / n_tail, 6) if vals else None,
    "first": round(vals[0], 6) if vals else None,
  }


def read_meta(run_dir: Path) -> dict:
  for candidate in (run_dir / "run_meta.json",
                    *run_dir.rglob("run_meta.json")):
    if candidate.exists():
      return json.loads(candidate.read_text())
  return {}


def extract_csv(run_dir: Path) -> Path:
  """TB events of a run dir -> per_iteration.csv (machine-readable log)."""
  meta = read_meta(run_dir)
  stack = meta.get("stack", "unknown")
  tag_map = {k: v[0] if stack == "new" else v[1] for k, v in SCALAR_MAP.items()}
  scalars = load_scalars(run_dir, tag_map)
  if not scalars:
    raise SystemExit(f"no TB scalars found under {run_dir}")
  all_steps = sorted({s for series in scalars.values() for s, _ in series})
  by_step = {c: dict(series) for c, series in scalars.items()}
  out = run_dir / "per_iteration.csv"
  cols = ["step"] + [c for c in SCALAR_MAP if c in scalars]
  with out.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(cols)
    for s in all_steps:
      w.writerow([s] + [by_step[c].get(s, "") for c in cols[1:]])
  print(f"[m5-compare] wrote {out} ({len(all_steps)} iterations, "
        f"{len(cols) - 1} scalars)")
  return out


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--new", type=str, help="new-stack run dir")
  parser.add_argument("--old", type=str, help="old-stack run dir")
  parser.add_argument("--extract", type=str,
                      help="run dir: convert TB events to per_iteration.csv")
  parser.add_argument("--out-prefix", type=str,
                      default=str(OUT_DIR / "baseline_comparison"))
  args = parser.parse_args()

  if args.extract:
    extract_csv(Path(args.extract))
    return 0
  if not (args.new and args.old):
    parser.error("need --new and --old, or --extract")

  new_dir, old_dir = Path(args.new), Path(args.old)
  new_scalars = load_scalars(new_dir, {k: v[0] for k, v in SCALAR_MAP.items()})
  old_scalars = load_scalars(old_dir, {k: v[1] for k, v in SCALAR_MAP.items()})
  new_meta, old_meta = read_meta(new_dir), read_meta(old_dir)

  matched = {}
  for canon in SCALAR_MAP:
    if canon in new_scalars and canon in old_scalars:
      matched[canon] = {
        "new": stats(new_scalars[canon]),
        "old": stats(old_scalars[canon]),
      }
  missing = [c for c in SCALAR_MAP
             if c not in new_scalars or c not in old_scalars]

  comparison = {
    "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    "run_dirs": {"new": str(new_dir), "old": str(old_dir)},
    "meta": {"new": new_meta, "old": old_meta},
    "matched_scalars": matched,
    "unmatched_scalars": missing,
    "env_steps": {
      "new": new_meta.get("iters", 0) * new_meta.get("num_envs", 0) * 24,
      "old": old_meta.get("iters", 0) * old_meta.get("num_envs", 0) * 24,
    },
    "fair_comparison_note": (
      "Fair only if both runs used the SAME num_envs, iterations and seed "
      "(see run_meta.json in each dir). Hardware: both launched on this "
      "laptop (RTX 4050 6 GB) unless stated otherwise in run_meta. Engine "
      "semantics still differ by design (MuJoCo vs PhysX contact/friction "
      "combination — see tests/data/m5/friction_semantics.md; new-stack "
      "effective foot friction [1.00, 1.25] vs old [0.65, 1.125]); reward "
      "scales and obs/action contracts are ported 1:1 (MIGRATION.md M3)."
    ),
  }

  OUT_DIR.mkdir(parents=True, exist_ok=True)
  json_path = Path(f"{args.out_prefix}.json")
  json_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False))

  md = [
    "# M5 baseline comparison (new mjlab stack vs old Isaac Gym stack)",
    "",
    f"* new run: `{new_dir}`" + (f" ({new_meta.get('num_envs')} envs, "
      f"{new_meta.get('iters')} iters, seed {new_meta.get('seed')})"
      if new_meta else ""),
    f"* old run: `{old_dir}`" + (f" ({old_meta.get('num_envs')} envs, "
      f"{old_meta.get('iters')} iters, seed {old_meta.get('seed')})"
      if old_meta else ""),
    f"* env steps: new {comparison['env_steps']['new']:,} / "
    f"old {comparison['env_steps']['old']:,} "
    "(iters x 24 steps/env x num_envs)",
    "",
    "> **Fair-comparison note**: " + comparison["fair_comparison_note"],
    "",
    "| scalar | new final | new last10% | old final | old last10% |",
    "|---|---|---|---|---|",
  ]
  for canon, m in matched.items():
    md.append(
      f"| {canon} | {m['new']['final']} | {m['new']['mean_last10pct']} "
      f"| {m['old']['final']} | {m['old']['mean_last10pct']} |"
    )
  if missing:
    md += ["", f"unmatched (tag missing on one side): {', '.join(missing)}"]
  md_path = Path(f"{args.out_prefix}.md")
  md_path.write_text("\n".join(md) + "\n")

  print("\n".join(md))
  print(f"-> {json_path}\n-> {md_path}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
