"""Render the approved static results; never train or change the corpus."""

import _bootstrap
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from hexapod_mjlab.tasks.envelope_adaptive_2 import contracts
from hexapod_mjlab.tasks.envelope_adaptive_2.geometry import hex_body_sample_points
from hexapod_mjlab.tasks.envelope_adaptive_2.map_generator import _rasterize_occupancy, generate_map
from hexapod_mjlab.tasks.envelope_adaptive_2.sl.dataset import load_map
from hexapod_mjlab.tasks.envelope_adaptive_2.experiments.static_qp import (
    StaticOracle,
    MIN,
    SPAN,
    NOMINAL_S,
    vertices,
    rect_polygon,
)
from static_oracle_compare import old_raw, describe, area, variation
from test_oracle import fixtures
from whole_polygon_reference import WholePolygonChecker

OUT = Path("logs/ea2/oracle_static/20261003")
DATA = Path("logs/ea2/data/ea2-migration-small-20261002")
plt.rcParams.update(
    {"font.family": "Noto Sans CJK SC", "axes.unicode_minus": False, "font.size": 10}
)
COLORS = {
    "safe": "#267963",
    "unsafe": "#bd503a",
    "infeasible": "#adb4bc",
    "old": "#bd503a",
    "new": "#256caf",
}


def aggregate(report, arrays):
    summary = {
        "correctness": report["result"],
        "total_frames": sum(m["frames"] for m in report["maps"].values()),
        "data_sha256": report["data_sha256"],
        "margins": {},
        "note": "Every margin uses all 12,320 frozen valid poses. Shape-quality metrics use only common-safe poses.",
    }
    n = summary["total_frames"]
    for margin in [0.1, 0.2]:
        old = f"old_matched_{margin:.2f}"
        new = f"new_{margin:.2f}"
        statuses = np.concatenate([a[new + "_status"] for a in arrays.values()])
        new_gap = np.concatenate([a[new + "_clearance"] for a in arrays.values()])
        floor = np.concatenate([a["minimum_clearance"] >= margin - 1e-6 for a in arrays.values()])
        entry = {
            "minimum_feasible_frames": int(floor.sum()),
            "minimum_infeasible_frames": int((~floor).sum()),
            "new_feasible_frames": int((statuses == "feasible").sum()),
            "new_solver_or_region_failures": int(
                np.sum((statuses != "feasible") & (statuses != "minimum_infeasible"))
            ),
            "new_returned_unsafe_frames": int(
                np.sum(
                    (statuses == "feasible") & (~np.isfinite(new_gap) | (new_gap < margin - 1e-6))
                )
            ),
            "coverage": {},
            "paired": {},
            "raw_variation": {},
        }
        for name in ["old_original", old, new]:
            gap = np.concatenate([a[name + "_clearance"] for a in arrays.values()])
            safe = np.isfinite(gap) & (gap >= margin - 1e-6)
            assert not np.any(safe & ~floor)
            entry["coverage"][name] = {
                "safe_frames": int(safe.sum()),
                "safe_fraction_of_all": float(safe.mean()),
                "unsafe_while_minimum_feasible_frames": int((~safe & floor).sum()),
                "minimum_infeasible_frames": int((~floor).sum()),
            }
        common = []
        os = []
        ns = []
        old_changes = []
        new_changes = []
        for seed, a in arrays.items():
            mask = np.isfinite(a[new + "_s"]).all(-1) & (a[old + "_clearance"] >= margin - 1e-6)
            common.append(mask)
            os.append(a[old + "_s"][mask])
            ns.append(a[new + "_s"][mask])
            index = {(int(t), int(e)): i for i, (t, e) in enumerate(zip(a["time"], a["env"]))}
            for (t, e), i in index.items():
                before = index.get((t - 1, e))
                if before is not None and mask[i] and mask[before]:
                    old_changes.append(
                        float(np.max(np.abs(a[old + "_s"][i] - a[old + "_s"][before])))
                    )
                    new_changes.append(
                        float(np.max(np.abs(a[new + "_s"][i] - a[new + "_s"][before])))
                    )
            entry["raw_variation"][str(seed)] = {
                name: variation(a[name + "_s"], index) for name in [old, new]
            }
        os = np.concatenate(os)
        ns = np.concatenate(ns)
        od = np.mean((os - NOMINAL_S) ** 2, axis=-1)
        nd = np.mean((ns - NOMINAL_S) ** 2, axis=-1)
        oa = area(MIN + SPAN * os)
        na = area(MIN + SPAN * ns)
        entry["paired"] = {
            "frames": len(os),
            "fraction_of_all": len(os) / n,
            "old_deviation_mse": float(od.mean()),
            "new_deviation_mse": float(nd.mean()),
            "old_area_m2": float(oa.mean()),
            "new_area_m2": float(na.mean()),
            "area_change_percent": float((na.mean() / oa.mean() - 1) * 100),
            "new_deviation_better_frames": int((nd < od - 1e-8).sum()),
            "adjacent_change": {"old": describe(old_changes), "new": describe(new_changes)},
        }
        summary["margins"][str(margin)] = entry
    summary["old_original_at_0.10"] = {
        "exact_unsafe_frames": sum(
            int((a["old_original_clearance"] < 0.1 - 1e-6).sum()) for a in arrays.values()
        ),
        "exact_intersection_frames": sum(
            int((a["old_original_clearance"] == 0).sum()) for a in arrays.values()
        ),
    }
    return summary


def summary_figure(summary):
    n = summary["total_frames"]
    fig, axs = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    for col, margin in enumerate([0.1, 0.2]):
        entry = summary["margins"][str(margin)]
        ax = axs[0, col]
        names = ["old_original", f"old_matched_{margin:.2f}", f"new_{margin:.2f}"]
        labels = ["旧原设置", "旧同间隙", "新方案"]
        for y, (name, label) in enumerate(zip(names, labels)):
            v = entry["coverage"][name]
            left = 0
            for key, color, title in [
                ("safe_frames", "safe", "安全输出"),
                ("unsafe_while_minimum_feasible_frames", "unsafe", "最小可行但输出未达间隙"),
                ("minimum_infeasible_frames", "infeasible", "最小形态已不可行"),
            ]:
                width = v[key] / n * 100
                ax.barh(y, width, left=left, color=COLORS[color], label=title if y == 0 else None)
                if width > 6:
                    ax.text(
                        left + width / 2,
                        y,
                        f"{width:.1f}%",
                        va="center",
                        ha="center",
                        color="white" if key != "minimum_infeasible_frames" else "#222222",
                        fontsize=9,
                    )
                left += width
        ax.set(
            yticks=range(3),
            yticklabels=labels,
            xlim=(0, 100),
            xlabel="全部有效位姿占比 (%)",
            title=f"连续几何间隙 {margin:.2f} m · n={n:,}",
        )
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.15)
        ax.set_axisbelow(True)
    axs[0, 0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1)
    x = np.arange(2)
    margins = [0.1, 0.2]
    ax = axs[1, 0]
    changes = [summary["margins"][str(m)]["paired"]["area_change_percent"] for m in margins]
    ax.bar(x, changes, color=COLORS["new"])
    ax.axhline(0, color="#777", lw=1)
    ax.set(
        xticks=x,
        xticklabels=["0.10 m", "0.20 m"],
        ylabel="新方案相对旧同间隙面积变化 (%)",
        title="共同安全位姿：平均面积未增加",
    )
    for i, value in enumerate(changes):
        ax.text(i, value / 2, f"{value:+.3f}%", ha="center", va="center", color="white")
    ax = axs[1, 1]
    old = [summary["margins"][str(m)]["paired"]["old_deviation_mse"] for m in margins]
    new = [summary["margins"][str(m)]["paired"]["new_deviation_mse"] for m in margins]
    ax.bar(x - 0.17, old, 0.34, color=COLORS["old"], label="旧同间隙")
    ax.bar(x + 0.17, new, 0.34, color=COLORS["new"], label="新方案")
    ax.set(
        xticks=x,
        xticklabels=[
            f'{m:.2f} m\n共同安全 {summary["margins"][str(m)]["paired"]["fraction_of_all"]*100:.1f}%'
            for m in margins
        ],
        ylabel="到名义形态的归一化 MSE（越低越好）",
        title="共同安全位姿：展开程度未整体改善",
    )
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.legend(fontsize=9)
    fig.suptitle("首轮静态 oracle 对照：安全覆盖改善，形态质量预期尚未成立", fontsize=14)
    fig.savefig(OUT / "summary.png", dpi=150)
    plt.close(fig)


def draw_outline(ax, params, yaw, color, label, style="-", pos=(0.0, 0.0)):
    v = vertices(params, pos, yaw)
    v = np.vstack([v, v[0]])
    ax.plot(v[:, 0], v[:, 1], style, color=color, lw=1.7, label=label)


def synthetic_figures():
    records = []
    case_titles = {
        "open": "空旷",
        "front_wall": "正面墙",
        "corridor": "窄通道",
        "oblique": "斜角障碍",
        "turn_0": "转向 yaw=0°",
        "turn_45": "转向 yaw=45°",
        "turn_90": "转向 yaw=90°",
        "minimum_infeasible": "最小形态不可行",
    }
    for margin in [0.1, 0.2]:
        fig, axs = plt.subplots(2, 4, figsize=(13, 7), layout="constrained")
        for ax, (name, rects, yaw, expected) in zip(axs.flat, fixtures(margin)):
            occupancy = _rasterize_occupancy(rects, contracts.MapGenCfg())
            field = torch.tensor(
                distance_transform_edt(occupancy == 0, sampling=0.1).astype(np.float32)
                if rects
                else np.full((740, 740), 100, np.float32)
            )
            old, _ = old_raw(np.array([yaw]), np.zeros((1, 2)), field, margin, margin)
            original, _ = old_raw(np.array([yaw]), np.zeros((1, 2)), field, 0.2, 0.1)
            result = StaticOracle(rects, margin).solve(np.zeros(2), yaw)
            checker = WholePolygonChecker(rects)
            record = {
                "name": name,
                "margin_m": margin,
                "new_status": result.status,
                "old_original_params": original[0].tolist(),
                "old_matched_params": old[0].tolist(),
            }
            for tag, params in [("old_original", original[0]), ("old_matched", old[0])]:
                gap = checker.clearance(params, np.zeros(2), yaw)
                record[tag + "_clearance_m"] = float(gap) if np.isfinite(gap) else None
                record[tag + "_meets_margin"] = bool(gap >= margin - 1e-6)
            if expected is not None:
                assert (
                    result.status == "feasible" and np.max(np.abs(result.params - expected)) <= 1e-4
                )
                record.update(
                    new_params=result.params.tolist(),
                    new_clearance_m=(
                        float(checker.clearance(result.params, np.zeros(2), yaw)) if rects else None
                    ),
                )
            else:
                assert result.status == "minimum_infeasible"
            for rect in rects:
                ax.add_patch(Polygon(rect_polygon(rect), color="#555555", alpha=0.35))
            draw_outline(ax, MIN + SPAN * NOMINAL_S, yaw, "#888888", "名义形态", ":")
            draw_outline(ax, MIN, yaw, "#444444", "最小形态", ":")
            draw_outline(ax, original[0], yaw, "#be8a22", "旧原设置", "--")
            draw_outline(ax, old[0], yaw, COLORS["old"], "旧同间隙", "-.")
            if result.status == "feasible":
                draw_outline(ax, result.params, yaw, COLORS["new"], "新方案")
            else:
                ax.text(
                    0,
                    0,
                    "新方案报告不可行",
                    ha="center",
                    va="center",
                    fontsize=9,
                    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
                )
            ax.set(
                xlim=(-1.4, 1.4),
                ylim=(-1.4, 1.4),
                aspect="equal",
                title=case_titles[name],
                xlabel="x (m)",
                ylabel="y (m)",
            )
            ax.grid(alpha=0.2)
            records.append(record)
        handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=5, fontsize=9)
        fig.suptitle(f"人工静态案例 · 间隙 {margin:.2f} m · 本轮无时间平滑", fontsize=14)
        fig.savefig(OUT / f"synthetic_{margin:.2f}.png", dpi=150)
        plt.close(fig)
    (OUT / "synthetic_comparison.json").write_text(
        json.dumps(records, indent=2, allow_nan=False) + "\n"
    )


def fixed_probe_figure(arrays, diagnostics):
    # Illustrative view only: first intersecting old-original pose among the
    # predetermined 32 evenly spaced probes; aggregate claims use all frames.
    for seed, a in arrays.items():
        indices = diagnostics["maps"][str(seed)]["probe_indices"]
        ids = [
            i
            for i in indices
            if a["old_original_clearance"][i] == 0 and np.isfinite(a["new_0.10_s"][i]).all()
        ]
        if ids:
            break
    assert ids
    i = ids[0]
    pos = a["pos"][i]
    yaw = a["heading"][i]
    data = load_map(str(DATA / f"map_seed{seed}.pt"))
    cfg = data.meta["environment_config"]
    geometry = generate_map(
        contracts.MapGenCfg(**cfg["map"]),
        contracts.PillarFieldCfg(
            **{k.removeprefix("pillar_"): v for k, v in cfg["obstacles"].items()}
        ),
        int(seed),
    )
    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    for rect in geometry.rects:
        if np.linalg.norm(np.asarray(rect.center) - pos) <= 3 + np.linalg.norm(rect.size) / 2:
            ax.add_patch(
                Polygon(rect_polygon(rect), facecolor="#888888", edgecolor="#444444", alpha=0.45)
            )
    old = MIN + SPAN * a["old_original_s"][i]
    new = MIN + SPAN * a["new_0.10_s"][i]
    draw_outline(ax, old, yaw, COLORS["old"], "旧原设置", pos=pos)
    draw_outline(ax, new, yaw, COLORS["new"], "新方案 0.10 m", pos=pos)
    body = hex_body_sample_points(torch.tensor(old, dtype=torch.float32)[None]).numpy()[0]
    c, s = np.cos(yaw), np.sin(yaw)
    points = body @ np.array([[c, s], [-s, c]]) + pos
    ax.scatter(points[:, 0], points[:, 1], s=14, color="#267963", label="旧检查的34个样点")
    ax.scatter(*pos, color="#222", s=15)
    ax.set(
        xlim=(pos[0] - 1.8, pos[0] + 1.8),
        ylim=(pos[1] - 1.8, pos[1] + 1.8),
        aspect="equal",
        xlabel="世界 x (m)",
        ylabel="世界 y (m)",
        title=f'固定几何核验样本 · map={seed}, t={a["time"][i]}, env={a["env"][i]}\n旧完整包络相交；新间隙={a["new_0.10_clearance"][i]:.3f} m',
    )
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9, loc="upper left")
    fig.savefig(OUT / "fixed_probe.png", dpi=160)
    plt.close(fig)
    return {
        "seed": int(seed),
        "index": int(i),
        "time": int(a["time"][i]),
        "env": int(a["env"][i]),
        "selection": "first intersecting old-original pose with new feasible output among fixed 32 evenly spaced probes",
        "purpose": "illustration only, not used to select or evaluate the corpus",
    }


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=OUT)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    OUT = args.out_dir
    source = args.input_dir
    torch.set_num_threads(1)
    report = json.loads((source / "comparison.json").read_text())
    assert report["result"] == "passed"
    diagnostics = json.loads((source / "diagnostics.json").read_text())
    assert diagnostics["result"] == "passed"
    arrays = {str(seed): dict(np.load(source / f"map_seed{seed}.npz")) for seed in [1, 7]}
    summary = aggregate(report, arrays)
    summary_figure(summary)
    synthetic_figures()
    summary["illustration"] = fixed_probe_figure(arrays, diagnostics)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(
        "STATIC REPORT PASS: all-frame safety, paired quality/variation, all synthetic cases, and fixed diagnostic illustration"
    )


if __name__ == "__main__":
    main()
