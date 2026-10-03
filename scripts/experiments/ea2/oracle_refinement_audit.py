"""Derived comparisons: production legacy, first safety QP and each candidate.

All dynamic comparisons use exactly the candidate/current-geometric-feasibility
mask. Unsafe legacy outputs stay IN the trajectory comparison. Shape comparisons
also expose the same independently-safe legacy subset to avoid rewarding legacy
obstacle penetration. Source evidence files are hashed and never changed.
"""

import _bootstrap
import argparse, glob, hashlib, json
from pathlib import Path
import numpy as np
from oracle_refinement_compare import dynamics
from static_oracle_compare import area, mean
from hexapod_mjlab.tasks.envelope_adaptive_2.experiments.static_qp import MIN, SPAN, NOMINAL_S


def change(new, old):
    return (new / old - 1) * 100 if old is not None and old > 1e-12 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    root = Path("logs/ea2/oracle_refinement/20261003")
    out = root / f"derived_audit_{args.tag}.json"
    if out.exists():
        raise FileExistsError(out)
    records = []
    for p in sorted(root.glob("*.npz")):
        if not p.with_suffix(".json").exists():
            continue
        d = json.loads(p.with_suffix(".json").read_text())
        v = dict(np.load(p))
        valid = np.isfinite(v["s"]).all(-1)
        safe = valid & (v["legacy_clearance"] >= d["margin"] - 1e-6)
        now = dynamics(v["s"], v["time"], v["env"], valid)
        old = dynamics(v["legacy_s"], v["time"], v["env"], valid)
        current_shape = np.mean((v["s"][valid] - NOMINAL_S) ** 2)
        shape_safe_new = np.mean((v["s"][safe] - NOMINAL_S) ** 2) if safe.any() else None
        shape_safe_old = np.mean((v["legacy_s"][safe] - NOMINAL_S) ** 2) if safe.any() else None
        area_new = mean(area(MIN + SPAN * v["s"][safe]))
        area_old = mean(area(MIN + SPAN * v["legacy_s"][safe]))
        compare = {}
        for name, key in [
            ("jump", "normalised_jump"),
            ("speed", "physical_speed_max_m_s"),
            ("acceleration", "physical_acceleration_max_m_s2"),
        ]:
            compare[name] = {
                stat: change(now[key][stat], old[key][stat]) for stat in ["mean", "p95", "max"]
            }
        compare["tv_percent"] = change(
            now["normalised_total_variation_per_pair"], old["normalised_total_variation_per_pair"]
        )
        records.append(
            {
                "run": p.stem,
                "seed": d["seed"],
                "margin": d["margin"],
                "kind": d["kind"],
                "npz_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "json_sha256": hashlib.sha256(p.with_suffix(".json").read_bytes()).hexdigest(),
                "frames": len(valid),
                "candidate_valid": int(valid.sum()),
                "candidate_supervision_valid_if_emergencies_masked": int(
                    (valid & ~v["emergency"]).sum()
                ),
                "candidate_geom_unsafe_count": int(
                    (valid & (v["clearance"] < d["margin"] - 1e-6)).sum()
                ),
                "legacy_safe_all_frames": int((v["legacy_clearance"] >= d["margin"] - 1e-6).sum()),
                "legacy_safe_current_feasible_frames": int(safe.sum()),
                "current_feasible_legacy_dynamics": old,
                "candidate_dynamics": now,
                "relative_to_production_legacy_percent": compare,
                "candidate_nominal_mse_all_valid": float(current_shape),
                "legacy_safe_shape": {
                    "frames": int(safe.sum()),
                    "legacy_mse": shape_safe_old,
                    "candidate_mse": shape_safe_new,
                    "legacy_area_m2": area_old,
                    "candidate_area_m2": area_new,
                    "mse_change_percent": (
                        change(shape_safe_new, shape_safe_old) if safe.any() else None
                    ),
                    "area_change_percent": change(area_new, area_old) if safe.any() else None,
                },
                "relative_to_first_qp": d.get("paired_with_first_qp"),
                "emergencies": d["emergencies"],
                "correctness_failures": d["correctness_failures"],
                "metadata_clarification": (
                    "common shape at 0,h/2,h; no mix=.5 or min-prev trigger, static cache ignored; see source snapshot"
                    if d["kind"] == "lookahead" and d["tag"].endswith(("v10", "v11"))
                    else "see exact per-run config"
                ),
                "old_horizon_field_correction": (
                    d["causal_config"]["horizon_s"]
                    if d["kind"] == "causal" and d["tag"].endswith(("v4", "v5"))
                    else None
                ),
                "direction_issue": d["kind"] == "causal" and d["tag"].endswith(("v4", "v5")),
            }
        )
    out.write_text(
        json.dumps({"comparison_rule": __doc__, "records": records}, indent=2, allow_nan=False)
        + "\n"
    )
    print(out, len(records))


if __name__ == "__main__":
    main()
