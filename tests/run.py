#!/usr/bin/env python3
"""Run functional CPU/GPU checks in isolated processes, without training a task."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Check:
    name: str
    script: str
    args: tuple[str, ...] = ()
    output: str | None = None
    timeout_s: int = 300


CPU = (
    Check("verification_runner", "tests/test_runner.py"),
    Check("checkpoint_physics", "tests/test_checkpoint_physics.py"),
    Check(
        "confined_partial_reset_cpu",
        "tests/confined_locomotion/test_runtime.py",
        timeout_s=120,
    ),
    Check("lidar_mount", "tests/lidar_sensor/mount_check.py"),
    Check("lidar_optical_alignment", "tests/lidar/alignment/test_alignment.py"),
    Check("lidar_mesh_geometry", "tests/lidar/alignment/test_mesh_relations.py"),
    Check("lidar_sampling", "tests/lidar_sensor/pool_sampler_check.py"),
    Check("robot_stance", "tests/robot/verify_stance.py"),
    Check("morphology_geometry", "scripts/tools/morphology/run_checks.py", output="--out"),
)
GPU = (
    Check("robot_warp", "tests/robot/warp_check.py"),
)
CONFINED_CPU = tuple(
    Check(
        "confined_" + name,
        "tests/confined_locomotion/test_" + name + ".py",
        ("--cpu",) if name in {"training_collection", "training_update"} else (),
    )
    for name in (
        "runtime", "transition", "scanner", "ground", "visualization", "policy",
        "training_collection", "training_update", "checkpoint", "perturbations",
        "play", "symmetry", "symmetry_update", "training_entry",
    )
)
CONFINED_GPU = tuple(
    Check(
        "confined_" + feature + "_" + mode,
        "tests/confined_locomotion/gpu/check_" + feature + ".py",
        ("--mode", mode, "--gpu", "0", "--seed", "20261008", "--num-envs", "2"),
        "--output",
        timeout_s=1800 if mode in {"cycle", "command_period", "schedule"} else 600,
    )
    for feature, modes in (
        ("runtime", ("transition", "partial_reset", "persistent_low", "persistent_tilt",
                     "input_fault", "action_clip", "command_period")),
        ("collection", ("natural_rollout", "raw_action_log_prob", "value_fault", "timeout_bootstrap",
                        "feature_cache")),
        ("update", ("joint_update", "mirror_update")),
        ("entry", ("official_formal",)),
        ("checkpoint", ("cycle",)),
        ("perturbations", ("observation_reuse", "schedule")),
        ("symmetry", ("mapping", "window_rotation")),
        ("lifecycle", ("runtime_failure", "ppo_failure", "final_save_failure",
                       "record_log_failure", "postcommit", "close_failure")),
    )
    for mode in modes
)
PROFILES = {
    "cpu": CPU, "gpu": GPU,
    "confined_cpu": CONFINED_CPU,
    "confined_gpu": CONFINED_GPU,
}


def validate_checks(checks):
    names = [check.name for check in checks]
    if not checks or len(set(names)) != len(names):
        raise ValueError("Empty or duplicate verification selection")
    missing = [check.script for check in checks if not (ROOT / check.script).is_file()]
    if missing:
        raise FileNotFoundError(missing)


def run_checks(checks, output):
    validate_checks(checks)
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": 1, "started": datetime.now(timezone.utc).isoformat(), "checks": []}
    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "MPLCONFIGDIR": str(output / "matplotlib"),
        "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), os.environ.get("PYTHONPATH", "")]),
    }
    for check in checks:
        command = [sys.executable, check.script, *check.args]
        if check.output:
            destination = output / check.name
            if check.output == "--out":
                destination = destination.with_suffix(".json")
            command.extend([check.output, str(destination)])
        log = output / (check.name + ".log")
        env["HEXAPOD_TEST_OUTPUT"] = str(output / check.name / "artifacts")
        start = time.monotonic()
        print("RUN", check.name, flush=True)
        with log.open("x") as stream:
            try:
                rc = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=check.timeout_s,
                ).returncode
            except subprocess.TimeoutExpired:
                rc = 124
        report["checks"].append(
            {
                "name": check.name,
                "state": "PASS" if rc == 0 else "FAIL",
                "returncode": rc,
                "wall_time_s": round(time.monotonic() - start, 3),
                "command": command,
                "log": log.name,
            }
        )
        print(report["checks"][-1]["state"], check.name, flush=True)
    report["passed"] = len(report["checks"]) == len(checks) and all(
        row["returncode"] == 0 for row in report["checks"]
    )
    with (output / "summary.json").open("x") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return 0 if report["passed"] else 1


def main():
    if not __debug__:
        raise RuntimeError("Verification requires assertions; do not use python -O")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="cpu")
    parser.add_argument("--only", nargs="+", help="Select check names from the chosen profile")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    checks = PROFILES[args.profile]
    if args.only:
        unknown = set(args.only) - {check.name for check in checks}
        if unknown:
            parser.error("Unknown checks: " + ", ".join(sorted(unknown)))
        checks = tuple(check for check in checks if check.name in args.only)
    validate_checks(checks)
    if args.list:
        for check in checks:
            print(check.name, check.script)
        return 0
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output = (
        args.out_dir or ROOT / "logs" / "verification" / (args.profile + "-" + timestamp)
    ).resolve()
    return run_checks(checks, output)


if __name__ == "__main__":
    raise SystemExit(main())
