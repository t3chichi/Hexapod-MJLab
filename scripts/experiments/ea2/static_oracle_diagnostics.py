"""Audit saved static evidence without rerunning the full candidate experiment."""

import _bootstrap
import argparse
import shutil
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from hexapod_mjlab.tasks.envelope_adaptive_2 import contracts
from hexapod_mjlab.tasks.envelope_adaptive_2.geometry import _hex_sample_violations
from hexapod_mjlab.tasks.envelope_adaptive_2.map_generator import generate_map
from hexapod_mjlab.tasks.envelope_adaptive_2.sl.dataset import load_map
from hexapod_mjlab.tasks.envelope_adaptive_2.experiments.static_qp import MIN, SPAN, vertices
from static_oracle_compare import old_raw, paired_metrics, variation, EA2Cfg, ENVELOPE_SPEC
from source_reference import load_reference

OUT = Path("logs/ea2/oracle_static/20261003")
DATA = Path("logs/ea2/data/ea2-migration-small-20261002")


def dense_axis_box_distance(params, pos, yaw, rects):
    """Independent dense boundary/box probe, with explicit contained-corner test.

    Axis-aligned rectangles only. The dense distance overestimates true edge
    distance by no more than half the largest sample spacing. Corner containment
    uses the shape's piecewise linear half-width in body coordinates, avoiding
    the exact validator's edge intersections and point-in-polygon algorithm.
    """
    fw, mw, bw, f, b = params
    outline = vertices(params, pos, yaw)
    centers = np.array([r.center for r in rects])
    half = np.array([r.size for r in rects]) / 2
    assert all(abs(r.yaw) < 1e-12 for r in rects)
    # All possible nearest obstacles to this bounded shape are retained.
    center_gap = np.linalg.norm(centers - pos, axis=-1) - np.linalg.norm(half, axis=-1)
    close = center_gap <= np.linalg.norm(outline - pos, axis=-1).max() + 1.5
    centers, half = centers[close], half[close]
    corners = centers[:, None, :] + half[:, None, :] * np.array(
        [[-1, -1], [1, -1], [1, 1], [-1, 1]]
    )
    c, s = np.cos(yaw), np.sin(yaw)
    local = (corners - pos) @ np.array([[c, -s], [s, c]])
    x, y = local[..., 0], local[..., 1]
    width = np.where(x >= 0, mw + (fw - mw) * x / f, mw + (bw - mw) * x / b)
    if np.any((x >= b) & (x <= f) & (np.abs(y) <= width)):
        return 0.0, 0.0
    t = np.linspace(0, 1, 2001)
    edge = np.roll(outline, -1, axis=0) - outline
    samples = (outline[:, None, :] + t[None, :, None] * edge[:, None, :]).reshape(-1, 2)
    delta = np.maximum(np.abs(samples[:, None, :] - centers) - half, 0)
    gap = float(np.linalg.norm(delta, axis=-1).min())
    bound = float(np.linalg.norm(edge, axis=-1).max() / 4000)
    return gap, bound


def main():
    global OUT, DATA
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=OUT)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    for name in ("comparison.json", "map_seed1.npz", "map_seed7.npz"):
        shutil.copy2(args.input_dir / name, args.out_dir / name)
    OUT, DATA = args.out_dir, args.data_dir
    torch.set_num_threads(1)
    reference = load_reference()["envelope_oracle"]
    path = OUT / "comparison.json"
    report = json.loads(path.read_text())
    original = OUT / "comparison.before_statistics_clarification.json"
    if not original.exists():
        original.write_text(path.read_text())
    audit = {
        "result": "running",
        "fixed_probe_selection": "32 evenly spaced indices per map, including endpoints",
        "probe_note": "Additional read-only baseline/geometry checks; no new candidate solving, training, map, or margin",
        "maps": {},
        "npz_sha256_before": {},
        "npz_sha256_after": {},
    }
    for seed in [1, 7]:
        file = OUT / f"map_seed{seed}.npz"
        audit["npz_sha256_before"][str(seed)] = hashlib.sha256(file.read_bytes()).hexdigest()
        a = dict(np.load(file))
        n = len(a["pos"])
        m = load_map(str(DATA / f"map_seed{seed}.pt"))
        cfg = m.meta["environment_config"]
        geometry = generate_map(
            contracts.MapGenCfg(**cfg["map"]),
            contracts.PillarFieldCfg(
                **{k.removeprefix("pillar_"): v for k, v in cfg["obstacles"].items()}
            ),
            seed,
        )
        index = {(int(t), int(e)): i for i, (t, e) in enumerate(zip(a["time"], a["env"]))}
        for margin in [0.1, 0.2]:
            report["maps"][str(seed)]["paired"][str(margin)] = paired_metrics(a, margin, index)
        for name in report["maps"][str(seed)]["algorithms"]:
            report["maps"][str(seed)]["algorithms"][name]["adjacent_max_s_change"] = variation(
                a[name + "_s"], index
            )
        ids = np.linspace(0, n - 1, 32, dtype=int)
        record = {
            "frames": n,
            "probe_indices": ids.tolist(),
            "baselines": {},
            "geometry_probes": [],
        }
        specs = [
            ("old_original", 0.2, 0.1),
            ("old_matched_0.10", 0.1, 0.1),
            ("old_matched_0.20", 0.2, 0.2),
        ]
        for name, ray, check in specs:
            params = MIN + SPAN * a[name + "_s"]
            violation = _hex_sample_violations(
                torch.tensor(params, dtype=torch.float32),
                torch.tensor(a["heading"], dtype=torch.float32),
                torch.tensor(a["pos"], dtype=torch.float32),
                m.distance_field,
                check,
                0.1,
            ).numpy()
            batch, _ = old_raw(a["heading"][ids], a["pos"][ids], m.distance_field, ray, check)
            batch_error = float(np.max(np.abs(batch - params[ids])))
            single = np.stack(
                [
                    old_raw(
                        a["heading"][i : i + 1], a["pos"][i : i + 1], m.distance_field, ray, check
                    )[0][0]
                    for i in ids
                ]
            )
            single_error = float(np.max(np.abs(single - params[ids])))
            envelope = EA2Cfg().envelope
            with torch.no_grad():
                source, _ = reference.compute_direct_oracle_params_with_stats(
                    torch.tensor(a["heading"][ids], dtype=torch.float32),
                    torch.tensor(a["pos"][ids], dtype=torch.float32),
                    m.distance_field,
                    torch.tensor(ENVELOPE_SPEC.low[:5]),
                    torch.tensor(ENVELOPE_SPEC.high[:5]),
                    margin=ray,
                    collision_margin=check,
                    collision_soft_margin=envelope.soft_margin,
                    step=envelope.oracle_step,
                    max_dist=envelope.oracle_max_dist,
                    soft_dof_pos_limit=envelope.soft_dof_pos_limit,
                    interp_crossing=envelope.oracle_interp_crossing,
                    group_mode=envelope.oracle_group_mode,
                )
            source_error = float(np.max(np.abs(source.numpy() - params[ids])))
            assert max(batch_error, single_error, source_error) <= 1e-6, (
                seed,
                name,
                batch_error,
                single_error,
                source_error,
            )
            bad = a[name + "_clearance"] < check - 1e-6
            record["baselines"][name] = {
                "margin_m": check,
                "exact_unsafe_count": int(bad.sum()),
                "exact_intersection_count": int((a[name + "_clearance"] == 0).sum()),
                "unsafe_gap_deficit_gt_0.02m_count": int(
                    (a[name + "_clearance"] < check - 0.02).sum()
                ),
                "old_24_boundary_violation_gt_0.05_count": int(
                    (violation[:, :24].max(1) > 0.05).sum()
                ),
                "old_34_sample_violation_gt_0.05_count": int((violation.max(1) > 0.05).sum()),
                "batch32_vs_full_max_error_m": batch_error,
                "single_vs_full_max_error_m": single_error,
                "hash_checked_original_source_max_error_m": source_error,
            }
        for name in [
            "old_original",
            "old_matched_0.10",
            "old_matched_0.20",
            "new_0.10",
            "new_0.20",
        ]:
            for i in ids:
                if not np.isfinite(a[name + "_s"][i]).all():
                    continue
                dense, bound = dense_axis_box_distance(
                    MIN + SPAN * a[name + "_s"][i], a["pos"][i], a["heading"][i], geometry.rects
                )
                exact = float(a[name + "_clearance"][i])
                error = dense - exact
                assert -1e-8 <= error <= bound + 1e-8, (seed, name, int(i), exact, dense, bound)
                record["geometry_probes"].append(
                    {
                        "algorithm": name,
                        "index": int(i),
                        "time": int(a["time"][i]),
                        "env": int(a["env"][i]),
                        "exact_m": exact,
                        "dense_m": dense,
                        "approximation_bound_m": bound,
                        "error_m": error,
                    }
                )
        audit["maps"][str(seed)] = record
        audit["npz_sha256_after"][str(seed)] = hashlib.sha256(file.read_bytes()).hexdigest()
    assert audit["npz_sha256_before"] == audit["npz_sha256_after"]
    audit["result"] = "passed"
    report["statistics_clarification"] = {
        "source": "saved map_seed1/7.npz; candidate outputs unchanged",
        "backup": str(original),
        "ambiguous_joint_fraction_replaced_by_explicit_counts_and_joint_conditional_fractions": True,
        "paired_adjacent_change": "both endpoints in common safe mask; identical pairs for old/new",
        "npz_sha256": audit["npz_sha256_after"],
    }
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (OUT / "diagnostics.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    print(
        "STATIC DIAGNOSTICS PASS: frozen outputs unchanged; batch/single consistency and fixed dense geometry probes; explicit paired denominators"
    )


if __name__ == "__main__":
    main()
