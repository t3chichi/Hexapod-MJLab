#!/usr/bin/env python3
"""Independently reduce raw demands with scalar arithmetic, without analyze.py."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]


def read(path):
    return json.loads(path.read_text())


def verify(summary):
    for name, expected in summary["input_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    assert hashlib.sha256(Path(__file__).with_name("analyze.py").read_bytes()).hexdigest() == summary["analyzer_sha256"]
    tree = ast.parse((ROOT / "src/hexapod_mjlab/tasks/el_4090_lidar/t0/witness.py").read_text())
    constants = {target.id: ast.literal_eval(node.value) for node in tree.body
                 if isinstance(node, ast.Assign) for target in node.targets
                 if isinstance(target, ast.Name) and target.id in ("JOINT_ORDER", "JOINT_TARGETS")}
    names, ref = constants["JOINT_ORDER"], constants["JOINT_TARGETS"]
    assert list(names) == summary["joint_order"] and list(ref) == summary["q_ref_rad"]
    assert summary["action_scale_rad"] == .25 and not summary["range_adopted"]
    root = ROOT / "logs/morphology/acceptance-v3-20261003"
    holds, gaits = read(root / "holds.json"), read(root / "gaits.json")
    cases = {c["id"]: c for c in holds["cases"]}
    good_h = [r for r in holds["results"] if r["verdict"] == "PASS"]
    good_g = [r for r in gaits["results"] if r["verdict"] == "PASS"]
    checks = 0

    def reduce(rows, group, targets=True):
        nonlocal checks
        minima, maxima = [math.inf]*18, [-math.inf]*18
        frame_counts = {c: 0 for c in (1, 2, 3, 4, 100)}
        cell_counts = dict(frame_counts)
        peak, count = 0., 0
        for row in rows:
            assert len(row) == 18
            values = [(float(x)-ref[i])/.25 if targets else float(x) for i, x in enumerate(row)]
            assert all(math.isfinite(x) for x in values)
            count += 1
            for i, x in enumerate(values):
                minima[i], maxima[i] = min(minima[i], x), max(maxima[i], x)
                peak = max(peak, abs(x))
            for c in frame_counts:
                cells = sum(abs(x) > c for x in values)
                cell_counts[c] += cells
                frame_counts[c] += bool(cells)
        assert count == group["rows"]
        assert math.isclose(peak, group["peak_abs_action"], rel_tol=0, abs_tol=1e-12)
        assert math.isclose(peak*.25, group["peak_abs_target_offset_rad"], rel_tol=0, abs_tol=1e-12)
        checks += 3
        for i, name in enumerate(names):
            g = group["per_joint"][name]
            for actual, expected in ((minima[i], g["min_action"]), (maxima[i], g["max_action"]),
                                     (max(abs(minima[i]), abs(maxima[i])), g["peak_abs_action"])):
                assert math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12)
                checks += 1
        for c in frame_counts:
            g = group["clip_diagnostics"][str(c)]
            assert frame_counts[c] == g["rows_with_any_exceedance"]
            assert cell_counts[c] == g["coordinate_exceedances"]
            assert math.isclose(max(0., peak-c)*.25, g["largest_target_change_rad"], rel_tol=0, abs_tol=1e-12)
            checks += 3

    groups = summary["groups"]
    reduce(([cases[r["id"]]["joint_angles"][n] for n in names] for r in good_h), groups["T1_accepted_hold_targets"])
    reduce((t["q_target"] for r in good_g for t in r["trajectory"]), groups["T1_accepted_crawl_targets"])
    for r, group in zip(good_g, groups["T1_accepted_crawl_targets"]["per_case"], strict=True):
        assert r["id"] == group["id"]
        reduce((t["q_target"] for t in r["trajectory"]), group)
    states = [q for r in good_h for q in r["trajectory"]["joint_angles"]]
    states.extend(t["q"] for r in good_g for t in r["trajectory"])
    reduce(states, groups["T1_actual_state_diagnostic"])
    t0 = ROOT / "logs/t0/t0a-strict-20261003-223137"
    meta = read(t0 / "eval_summary.json")["episodes_meta"]
    with np.load(t0 / "episodes.npz", allow_pickle=False) as arrays:
        for command in ("stand", "forward_025"):
            rows = (a for e in meta if e["command_name"] == command
                    for a in arrays[e["episode_id"]+"__actions"])
            reduce(rows, groups["T0_"+command+"_actions"], targets=False)
    return {"passed": True, "scalar_numeric_checks": checks,
            "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scope": "frozen input hashes and independent extrema/exceedance reductions; no dynamic safety or range adoption"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = verify(read(args.summary))
    result["summary_sha256"] = hashlib.sha256(args.summary.read_bytes()).hexdigest()
    with args.out.open("x") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    print(json.dumps(result))
