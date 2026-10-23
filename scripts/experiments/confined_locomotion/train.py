#!/usr/bin/env python
"""Run the formal confined-locomotion recipe with the current S1-187 task grid.

B8 is the scanner default; the grid defaults to the confirmed S1-187
coordinates. Seed, environment count, budget and GPU are separately
validated. Custom grids still require two explicit ordered axes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hexapod_mjlab.tasks.confined_locomotion.ground import DEFAULT_PARALLEL_TOL
from hexapod_mjlab.tasks.confined_locomotion.ground_presets import (
  GROUND_GRID_PRESET_NAMES, resolve_task_ground_grid,
)
from hexapod_mjlab.tasks.confined_locomotion.rl.entry import (
  launch_formal_training,
)


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description=(
      "Train confined locomotion with the current B8 scanner and S1-187 "
      "ground-grid baseline; seed/env/budget must be explicit. An optional "
      "grid preset or both axes override the baseline."
    )
  )
  parser.add_argument("--seed", type=int, required=True)
  parser.add_argument("--num-envs", type=int, required=True)
  parser.add_argument("--max-iterations", type=int, required=True)
  parser.add_argument("--grid-x", type=float, nargs="+",
                      help="explicit ordered BODY-x coordinates (requires --grid-y)")
  parser.add_argument("--grid-y", type=float, nargs="+",
                      help="explicit ordered BODY-y coordinates (requires --grid-x)")
  parser.add_argument("--grid-preset", choices=GROUND_GRID_PRESET_NAMES,
                      help="named grid override (default: current s1_187; incompatible with --grid-x/y)")
  parser.add_argument(
    "--parallel-tol", type=float, default=DEFAULT_PARALLEL_TOL,
    help=f"near-parallel tolerance recorded in the grid identity (default {DEFAULT_PARALLEL_TOL:g})",
  )
  parser.add_argument("--resume-checkpoint")
  parser.add_argument("--log-root", default="logs/confined_locomotion")
  parser.add_argument("--gpu-id", type=int, default=0)
  parser.add_argument(
    "--engineering-validation",
    action="store_true",
    help="permit only the bounded <=2-env, exactly 1-update real training-mode check",
  )
  return parser


def main() -> None:
  parser = _parser()
  args = parser.parse_args()
  if args.grid_preset is None and ((args.grid_x is None) != (args.grid_y is None)):
    parser.error("custom grid requires BOTH --grid-x and --grid-y")
  if args.grid_preset is not None and (args.grid_x is not None or args.grid_y is not None):
    parser.error("--grid-preset must not be mixed with --grid-x/--grid-y")
  grid = resolve_task_ground_grid(
    grid_preset=args.grid_preset,
    grid_x=args.grid_x,
    grid_y=args.grid_y,
    parallel_tol=args.parallel_tol,
  )
  result = launch_formal_training(
    seed=args.seed,
    ground_grid=grid,
    num_envs=args.num_envs,
    max_iterations=args.max_iterations,
    resume_checkpoint=args.resume_checkpoint,
    log_root=args.log_root,
    gpu_ids=(args.gpu_id,),
    engineering_validation=args.engineering_validation,
  )
  print(
    json.dumps(
      {
        "run_dir": str(result.run_dir),
        "checkpoint_path": str(result.checkpoint_path),
        "close_completed": result.close_completed,
        "global_learning_iteration": result.summary.get("current_learning_iteration"),
        "global_learning_iteration_source": "persisted_summary_or_null",
        "summary_close_record_persisted": result.summary[
          "formal_entry_verification"
        ]["close_record_persisted"],
        "summary_read_error": result.summary["formal_entry_verification"][
          "summary_read_error"
        ],
        "requested_new_iterations": args.max_iterations,
      },
      indent=2,
    )
  )


if __name__ == "__main__":
  main()
