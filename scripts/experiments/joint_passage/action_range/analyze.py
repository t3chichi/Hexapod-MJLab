#!/usr/bin/env python3
"""Inspect frozen T1/T0-A residual demands; no simulation or range adoption."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
T1 = ROOT / "logs/morphology/acceptance-v3-20261003"
T0 = ROOT / "logs/t0/t0a-strict-20261003-223137"


def read(path):
    return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def constants(path):
    wanted = {"JOINT_ORDER", "JOINT_TARGETS", "WITNESS_ID"}
    result = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    result[target.id] = ast.literal_eval(node.value)
    assert set(result) == wanted
    return result


def summarize(rows, names, scale):
    a = np.asarray(rows, dtype=np.float64)
    assert a.ndim == 2 and a.shape[1] == 18 and len(a) and np.isfinite(a).all()
    peak_index = np.unravel_index(np.argmax(np.abs(a)), a.shape)
    return {
        "rows": len(a),
        "peak_abs_action": float(np.abs(a).max()),
        "peak_abs_target_offset_rad": float(np.abs(a).max() * scale),
        "peak_row": int(peak_index[0]),
        "peak_joint": names[peak_index[1]],
        "per_joint": {
            n: {"min_action": float(a[:, i].min()),
                "max_action": float(a[:, i].max()),
                "peak_abs_action": float(np.abs(a[:, i]).max())}
            for i, n in enumerate(names)
        },
        "clip_diagnostics": {
            str(c): {"rows_with_any_exceedance": int((np.abs(a) > c).any(axis=1).sum()),
                     "coordinate_exceedances": int((np.abs(a) > c).sum()),
                     "largest_target_change_rad": float(max(0., np.abs(a).max() - c) * scale)}
            for c in (1, 2, 3, 4, 100)
        },
    }


def analyze():
    sources = {}

    def checked(path, expected):
        actual = digest(path)
        assert actual == expected, f"frozen hash mismatch: {path}"
        sources[str(path.relative_to(ROOT))] = actual
        return read(path) if path.suffix == ".json" else path

    release_path = ROOT / "docs/research/morphology/evidence/release.json"
    release = read(release_path)
    assert release["acceptance"]["verdict"] == "PASS"
    hashes = release["acceptance"]["artifacts_sha256"]
    library = checked(T1 / "library.json", hashes["library.json"])
    checked(ROOT / "docs/research/morphology/evidence/library.json", hashes["library.json"])
    holds = checked(T1 / "holds.json", hashes["holds.json"])
    gaits = checked(T1 / "gaits.json", hashes["gaits.json"])
    output_path = T0 / "output_identity.json"
    t0_release_path = ROOT / "docs/tasks/el_4090_lidar/evidence/release.json"
    t0_release = read(t0_release_path)
    # Anchor the output manifest to the small tracked release record.
    checked(output_path, t0_release["artifacts"][str(output_path.relative_to(ROOT))])
    sources[str(release_path.relative_to(ROOT))] = digest(release_path)
    sources[str(t0_release_path.relative_to(ROOT))] = digest(t0_release_path)
    manifest = read(output_path)["files"]
    cfg = checked(T0 / "resolved_env_cfg.json", manifest["resolved_env_cfg.json"])["cfg"]
    agent = checked(T0 / "resolved_agent_cfg.json", manifest["resolved_agent_cfg.json"])["cfg"]
    contract = checked(T0 / "contract_gate.json", manifest["contract_gate.json"])
    evaluation = checked(T0 / "eval_summary.json", manifest["eval_summary.json"])
    episodes = checked(T0 / "episodes.npz", manifest["episodes.npz"])
    witness_path = ROOT / "src/hexapod_mjlab/tasks/el_4090_lidar/t0/witness.py"
    witness = constants(witness_path)
    sources[str(witness_path.relative_to(ROOT))] = digest(witness_path)
    names = list(witness["JOINT_ORDER"])
    q_ref = np.asarray(witness["JOINT_TARGETS"], dtype=float)
    ref_record = next(r for r in library["records"] if r["id"] == witness["WITNESS_ID"])
    ref_copy_error = float(np.max(np.abs(q_ref - [ref_record["joint_targets"][n] for n in names])))
    # Same 1e-15 rad copy-check tolerance as T0 witness.verify_against_library.
    assert ref_copy_error <= 1e-15
    action_cfg = cfg["actions"]["joint_pd_torque"]
    scale = float(action_cfg["scale"])
    assert scale == .25 and action_cfg["actuator_names"] == names
    assert action_cfg["clip"] is None and action_cfg["use_default_offset"]
    default = cfg["scene"]["entities"]["robot"]["init_state"]["joint_pos"]
    assert np.array_equal(q_ref, [default[n] for n in names])
    model_gate = next(g for g in contract["gates"] if g["name"] == "model_contract")
    assert model_gate["state"] == "PASS"
    assert model_gate["evidence"]["joint_order_matches"]
    assert model_gate["evidence"]["actuator_order_matches"]
    assert list(library["joint_observed_demands"]) == names
    cases = {r["id"]: r for r in holds["cases"]}
    accepted_holds = [r for r in holds["results"] if r["verdict"] == "PASS"]
    accepted_gaits = [r for r in gaits["results"] if r["verdict"] == "PASS"]
    assert len(accepted_holds) == 35 and len(accepted_gaits) == 7
    assert {r["id"] for r in accepted_holds} == {r["id"] for r in library["records"]}
    assert {r["id"] for r in accepted_gaits} == {
        r["id"] for r in library["records"] if r["validation_level"] == "scripted_slow_crawl"}
    groups = {}
    hold_targets = np.array([[cases[r["id"]]["joint_angles"][n] for n in names]
                             for r in accepted_holds])
    groups["T1_accepted_hold_targets"] = summarize((hold_targets - q_ref) / scale, names, scale)
    groups["T1_accepted_hold_targets"]["row_case_ids"] = [r["id"] for r in accepted_holds]
    per_case = []
    for r in accepted_gaits:
        q = np.array([t["q_target"] for t in r["trajectory"]])
        assert q.shape == (1200, 18)
        per_case.append(dict(id=r["id"], **summarize((q-q_ref)/scale, names, scale)))
    gait_targets = np.array([t["q_target"] for r in accepted_gaits for t in r["trajectory"]])
    groups["T1_accepted_crawl_targets"] = summarize((gait_targets-q_ref)/scale, names, scale)
    groups["T1_accepted_crawl_targets"]["per_case"] = per_case
    states = np.concatenate([np.array(r["trajectory"]["joint_angles"]) for r in accepted_holds]
                            + [np.array([t["q"] for t in r["trajectory"]]) for r in accepted_gaits])
    state_diagnostic = summarize((states-q_ref)/scale, names, scale)
    state_diagnostic["semantics"] = "actual q minus reference divided by scale; NOT commanded action"
    groups["T1_actual_state_diagnostic"] = state_diagnostic
    meta = evaluation["episodes_meta"]
    verdicts = evaluation["summary"]["verdicts"]
    assert len(meta) == 64 and all(v["data_valid"] and v["complete"] for v in verdicts)
    with np.load(episodes, allow_pickle=False) as arrays:
        for command in ("stand", "forward_025"):
            selected = [r for r in meta if r["command_name"] == command]
            assert len(selected) == 32 and all(r["n_steps"] == 1000 for r in selected)
            actions = np.concatenate([arrays[r["episode_id"] + "__actions"] for r in selected])
            assert actions.shape == (32000, 18)
            groups["T0_" + command + "_actions"] = summarize(actions, names, scale)
    # Keep original failures visible; they are not mixed into the accepted-demand evidence.
    return {
        "schema": 1, "range_adopted": False, "simulation_or_training_run": False,
        "reference_id": witness["WITNESS_ID"], "joint_order": names,
        "q_ref_rad": q_ref.tolist(), "reference_copy_error_rad": ref_copy_error,
        "action_scale_rad": scale,
        "old_runner_clip_abs": agent["clip_actions"],
        "formula": "a_equivalent = (q_target - q_ref) / 0.25",
        "input_sha256": sources, "analyzer_sha256": digest(Path(__file__)),
        "source_verdict_counts": {
            "T1_holds": dict(Counter(r["verdict"] for r in holds["results"])),
            "T1_gaits": dict(Counter(r["verdict"] for r in gaits["results"])),
            "T0_per_command": evaluation["summary"]["per_command"],
        }, "groups": groups,
        "limitations": [
            "T1 crawl uses gravity/contact feedforward; residual representability is not a new-task PD performance certificate.",
            "Crawl targets sampled at 50Hz in crawl phase only; no continuous transition or swept safety certificate.",
            "Actual joint position is distinct from the commanded PD target.",
            "T0 forward policy did not learn locomotion; its small actions cannot bound future action demand.",
            "Static/crawl witnesses and per-joint marginals do not certify every configuration in a clipped action box.",
            "No hardware limits, future learned gait, six-dimensional teacher or arbitrary transitions certified.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = analyze()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    for name, group in result["groups"].items():
        print(name, "rows=", group["rows"], "peak_abs=", group["peak_abs_action"])


if __name__ == "__main__":
    main()
