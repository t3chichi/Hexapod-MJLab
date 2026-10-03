#!/usr/bin/env python
"""Frozen T0-A task flow. Prepare -> train -> evaluate -> sealed replay."""

import argparse, os, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t0lib
from t0lib import PASS, FAIL, BLOCKED, StageReport, new_run_dir, read_json
from provenance import freeze, verify_inputs, seal_outputs

REQUIRED_STAGES = (
    "audit_identity",
    "test_units",
    "contract_gate",
    "contact_gate",
    "hold_gate",
    "terminal_gate",
    "gate_a",
    "gate_b",
    "train_short",
    "eval_episodes",
    "replay_verify",
)
EXTERNAL_STAGES = ("test_units", "gate_a", "gate_b")


def stages_passed(records):
    return (
        len(records) == len(REQUIRED_STAGES)
        and {r["name"] for r in records} == set(REQUIRED_STAGES)
        and all(r["state"] == PASS and r["returncode"] == 0 for r in records)
    )


def run_stage(name, cmd, run, idx, timeout_s=1200):
    path = run / "stage_logs" / f"{idx:02}_{name}.log"
    path.parent.mkdir(exist_ok=True)
    started = time.monotonic()
    print("[T0]", name, flush=True)
    with path.open("x") as stream:
        try:
            r = subprocess.run(
                cmd,
                cwd=t0lib.REPO_ROOT,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=timeout_s,
            )
            rc = r.returncode
        except subprocess.TimeoutExpired:
            rc = 124
    report = (
        (run / "replay" / "replay_verify.json")
        if name == "replay_verify"
        else run / (name + ".json")
    )
    doc = read_json(report) if report.is_file() else {}
    ok = rc == 0 and (
        name in EXTERNAL_STAGES
        or doc.get("pipeline_passed") is True
        and bool(doc.get("gates"))
        and all(g["state"] == PASS for g in doc["gates"])
    )
    rec = dict(
        name=name,
        argv=cmd,
        returncode=rc,
        state=PASS if ok else FAIL,
        wall_time_s=time.monotonic() - started,
        log=str(path.relative_to(run)),
    )
    print("[T0]", name, rec["state"], round(rec["wall_time_s"], 1), "s", flush=True)
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--prepare-only", action="store_true", help="Diagnostic run: never qualifies full T0-A"
    )
    p.add_argument("--num-envs", type=int, choices=(16, 32), default=32)
    p.add_argument("--oom-evidence", type=Path)
    a = p.parse_args()
    run = new_run_dir(prefix="t0a-strict")
    rep = StageReport("acceptance", run)
    print("RUN_DIR", run, flush=True)
    freeze(run)
    py = sys.executable
    records = []
    prepare = [
        ("audit_identity", "scripts/tools/t0/audit_identity.py", []),
        ("test_units", "tests/t0/test_units.py", []),
        ("contract_gate", "scripts/tools/t0/contract_gate.py", ["--num-envs", "16"]),
        ("contact_gate", "scripts/tools/t0/contact_gate.py", ["--num-envs", "8"]),
        ("hold_gate", "scripts/tools/t0/hold_gate.py", ["--num-envs", "32"]),
        ("terminal_gate", "scripts/tools/t0/terminal_gate.py", []),
        ("gate_a", "tests/env/gate_a.py", ["--num-envs", "8", "--steps", "40"]),
        ("gate_b", "tests/env/gate_b.py", ["--num-envs", "8", "--steps", "40"]),
    ]
    for name, file, args in prepare:
        cmd = [py, file, *args]
        if name not in EXTERNAL_STAGES:
            cmd += ["--run-dir", str(run)]
        records.append(run_stage(name, cmd, run, len(records) + 1))
    ready = all(r["state"] == PASS for r in records) and not verify_inputs(run)
    if ready and not a.prepare_only:
        extra = ["--oom-evidence", str(a.oom_evidence)] if a.oom_evidence else []
        record = run_stage(
            "train_short",
            [
                py,
                "scripts/tools/t0/train_short.py",
                "--run-dir",
                str(run),
                "--num-envs",
                str(a.num_envs),
                *extra,
            ],
            run,
            8,
            timeout_s=1200,
        )
        records.append(record)
        if record["state"] == PASS:
            records.append(
                run_stage(
                    "eval_episodes",
                    [
                        py,
                        "scripts/tools/t0/eval_episodes.py",
                        "--run-dir",
                        str(run),
                        "--checkpoint",
                        str(run / "t0_short_model.pt"),
                        "--num-envs",
                        str(a.num_envs),
                    ],
                    run,
                    9,
                )
            )
            if records[-1]["state"] == PASS:
                seal_outputs(run)
                records.append(
                    run_stage(
                        "replay_verify",
                        [
                            py,
                            "scripts/tools/t0/replay_verify.py",
                            "--run-dir",
                            str(run),
                            "--out-dir",
                            str(run / "replay"),
                        ],
                        run,
                        10,
                    )
                )
    present = {r["name"] for r in records}
    for name in REQUIRED_STAGES:
        if name not in present:
            records.append(
                dict(
                    name=name,
                    returncode=1,
                    state=BLOCKED,
                    reason="preparation/dependency failed or prepare-only",
                    argv=[],
                )
            )
    issues = verify_inputs(run)
    rep.gate("source_inputs_unchanged", FAIL if issues else PASS, issues=issues)
    rep.gate("complete_required_flow", PASS if stages_passed(records) else FAIL, stages=records)
    rep.meta.update(
        run_dir=str(run),
        prepare_only=a.prepare_only,
        stages=records,
        learned_walking_passed=(
            read_json(run / "eval_episodes.json").get("meta", {}).get("learned_walking_passed")
            if (run / "eval_episodes.json").exists()
            else None
        ),
    )
    return rep.finalize()["returncode"]


if __name__ == "__main__":
    raise SystemExit(main())
