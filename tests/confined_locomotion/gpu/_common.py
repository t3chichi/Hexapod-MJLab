"""Shared CLI plumbing for feature-named real GPU checks."""
from __future__ import annotations

import argparse
import traceback
import time
from pathlib import Path
from typing import Callable

DEFAULT_SEED = 20261008
DEFAULT_NUM_ENVS = 2
ROLLOUT_STEPS = 24
OPTIMIZER_STEPS = 20
RAW_RAY_COUNT = 86_400


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
  parser.add_argument("--output", required=True, type=Path,
                      help="new, exclusive directory for this evidence bundle")
  parser.add_argument("--gpu", type=int, default=0,
                      help="explicit CUDA device index (default: 0)")
  parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                      help=f"explicit engineering seed (default: {DEFAULT_SEED})")
  parser.add_argument("--num-envs", type=int, choices=(1, 2), default=DEFAULT_NUM_ENVS,
                      help="explicit bounded environment count (default: 2)")


def configure_imports() -> Path:
  root = Path(__file__).resolve().parents[3]
  if not (root / ".git").exists() or not (root / "src").is_dir():
    raise RuntimeError(f"could not resolve repository root from {__file__}: {root}")
  import sys

  source = str(root / "src")
  tests = str(root / "tests")
  for entry in (source, tests):
    if entry not in sys.path:
      sys.path.insert(0, entry)
  return root


def require_cuda(gpu: int) -> str:
  import torch

  if isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0:
    raise ValueError(f"GPU id must be a non-negative integer, got {gpu!r}")
  if not torch.cuda.is_available():
    raise RuntimeError("this check requires real CUDA; CPU execution cannot pass the GPU gate")
  if gpu >= torch.cuda.device_count():
    raise RuntimeError(f"requested cuda:{gpu}, but only {torch.cuda.device_count()} CUDA device(s) are visible")
  torch.cuda.set_device(gpu)
  return f"cuda:{gpu}"


def finite_recipe(feature: str, mode: str, *, seed: int, gpu: int, num_envs: int,
                  optimizer: bool = False, grid=None, extra: dict | None = None) -> dict:
  recipe = {
    "schema": "confined_locomotion.gpu_recipe/v1",
    "feature": feature,
    "mode": mode,
    "seed": int(seed),
    "device": f"cuda:{gpu}",
    "num_envs": int(num_envs),
    "scan": {
      "sensor_raw_rays_per_scan": RAW_RAY_COUNT,
      "near_points": 256,
      "far_points_per_scan": 64,
      "far_distinct_scan_window": 10,
      "schedule_control_steps": {"near": 5, "far": 1},
    },
    "control": {"control_dt_s": 0.02, "physics_substeps_per_control": 4},
    "rollout_steps": ROLLOUT_STEPS,
    "optimizer_steps": OPTIMIZER_STEPS if optimizer else 0,
    "optimizer_enabled": bool(optimizer),
    "claim": "bounded engineering evidence only; no learning-effect claim",
  }
  if grid is not None:
    recipe["grid_identity"] = grid.identity()
  if extra:
    recipe.update(extra)
  return recipe


def execute_gpu_check(args, *, feature: str, mode: str, runner: Callable,
                      optimizer: bool = False, grid=None,
                      extra_recipe: dict | None = None) -> int:
  """Create an exclusive evidence bundle and run one explicitly selected GPU check."""
  root = configure_imports()
  from confined_locomotion.support.evidence import fresh_directory, write_json

  output = fresh_directory(args.output)
  recipe = finite_recipe(
    feature, mode, seed=args.seed, gpu=args.gpu, num_envs=args.num_envs,
    optimizer=optimizer, grid=grid, extra=extra_recipe,
  )
  recipe["repository_root"] = str(root)
  recipe["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
  write_json(output / "run_recipe.json", recipe)
  status = {"state": "running", "passed": False, "checks": None, "error": None}
  started = time.monotonic()
  try:
    device = require_cuda(args.gpu)
    status["device_name"] = __import__("torch").cuda.get_device_name(args.gpu)
    status["checks"] = runner(output, device, recipe)
    status["passed"] = bool(status["checks"].get("passed", False))
    status["state"] = "passed" if status["passed"] else "not_passed"
  except BaseException as error:
    status.update(state="failed", error=repr(error), traceback=traceback.format_exc())
  status["elapsed_s"] = time.monotonic() - started
  try:
    import torch

    status["peak_cuda_allocated_bytes"] = (
      torch.cuda.max_memory_allocated(args.gpu) if torch.cuda.is_available() else None
    )
    status["peak_cuda_reserved_bytes"] = (
      torch.cuda.max_memory_reserved(args.gpu) if torch.cuda.is_available() else None
    )
  except BaseException as error:
    status["resource_snapshot_error"] = repr(error)
  write_json(output / "report.json", status)
  print(f"{feature}/{mode}: {status['state']} ({output / 'report.json'})", flush=True)
  return 0 if status["passed"] else 1


def close_runtime(env, runtime) -> None:
  """Close task hooks and the official environment, preserving the first error."""
  error = None
  if runtime is not None:
    try:
      runtime.close()
    except BaseException as caught:
      error = caught
  if env is not None:
    try:
      env.close()
    except BaseException as caught:
      if error is None:
        error = caught
  if error is not None:
    raise error

