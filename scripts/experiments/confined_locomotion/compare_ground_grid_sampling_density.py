#!/usr/bin/env python
"""Fixed rectangular ground-supervision point-set comparisons (offline only).

Does NOT alter production GroundGridCfg, model ground_head, runner, PPO or
training recipe. Reuses measured near-field 15-pose TRUE CUDA raycast evidence
from analyze_ground_sampling_support.py. The historical labels B8/S1 are kept
in evidence schemas and candidate IDs only; they denote the widened near-field
grouping and the fixed rectangular supervision range, not current stages.

Metrics are true positive ground-hit/FPS 3D-neighbor distances (r in
0.15,0.20,0.25,0.30 m), NOT terrain prediction accuracy or actual observation
sufficiency on occluded/rough terrain. Compare S1 x=[.6,3.4], y=[-.7,.7]
across several grid resolutions; permit test-only 5-column symmetric
non-uniform y samples to show distribution effects.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

X8 = tuple(round(.6+.4*i, 2) for i in range(8))
X15 = tuple(round(.6+.2*i, 2) for i in range(15))
X29 = tuple(round(.6+.1*i, 2) for i in range(29))
Y5 = (-.7, -.4, 0.0, .4, .7)  # symmetric, nonuniform 0.3/0.4m gaps
Y8 = (-.7, -.5, -.3, -.1, .1, .3, .5, .7)  # symmetric 0.2m
CANDIDATES = (
    ("C40", X8, Y5),
    ("C64", X8, Y8),
    ("C75", X15, Y5),
    ("C120", X15, Y8),
    ("C232", X29, Y8),
)
RADIUS_M = (.15, .20, .25, .30)
PRIMARY = .20
ROUTINE_COUNT = 9


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path,
                   default=ROOT/"logs/verification/stageA_B8_ground_support_20261019_v3")
    p.add_argument("--output", type=Path, required=True, help="new, non-existing directory")
    return p.parse_args(argv)


def main(argv=None):
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hexapod_mjlab.tasks.confined_locomotion.ground import (
        GroundGridCfg,VALIDITY_GEOMETRY_ONLY_FLAT_TRAINING
    )
    from hexapod_mjlab.rl.confined_locomotion.symmetry import GridPermutation

    arg=parse(argv)
    source=arg.source.resolve()
    outfile=source/"probes_support_matrix.npz"
    srcreport=source/"report.json"
    out=arg.output.expanduser().resolve()
    out.mkdir(parents=True,exist_ok=False)
    dataset=np.load(outfile,allow_pickle=False)
    meta=json.loads(srcreport.read_text())
    if meta["state"]!="completed" or meta["probe_count"]!=1833 or meta["pose_checks"]!=15:
        raise AssertionError("B8 measured source identity not accepted")
    if "B8" not in meta["rule"] or meta["primary_distance_m"]!=.25:
        raise AssertionError("source was not B8 real-scan pose study")
    xx=np.asarray(dataset["x"])
    yy=np.asarray(dataset["y"])
    probe_xyz=np.asarray(dataset["grid_xyz"])
    pose_names=dataset["pose_names"].tolist()
    if len(pose_names)!=15 or pose_names[0]!="nominal":
        raise AssertionError("unexpected pose ordering")
    poolD=np.asarray(dataset["pool_distances_m"])
    fpsD=np.asarray(dataset["fps_distances_m"])
    if poolD.shape!=(15,1833) or fpsD.shape!=(15,1833):
        raise AssertionError("source distance dimensions changed")
    if not np.array_equal(probe_xyz[:,0],np.tile(xx,len(yy))):
        raise AssertionError("source probes not y-major x-minor")
    if not np.array_equal(probe_xyz[:,1],np.repeat(yy,len(xx))):
        raise AssertionError("source probes y ordering mismatch")

    report={
        "schema":"confined_locomotion.stageA.S1_ground_grid_layout_compare/v1",
        "state":"started","created_utc":datetime.now(timezone.utc).isoformat(),
        "measured_source":str(source.relative_to(ROOT)),
        "source_sha256":{
            str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (outfile,srcreport,Path(__file__))
        },
        "constraint":"all cells inside S1 x=[.6,3.4] y=[-.7,.7]; fixed symmetric XY Cartesian products",
        "source_actual_raycasts":15,
        "uses_previous_real_86k_GPU_raycasts":True,
        "this_program_runs_new_cuda_raycasts":False,
        "ground_provider_contract":"GroundGridCfg y-outer x-inner; mirrored y map exact",
        "evidence_definition":"Euclidean distance in 3D from provider ground intersection to TRUE near-floor hit, using B8 FPS256 (or full pool)",
        "limits":"nearest-hit proxy on flat open plane; not true terrain height estimation, not collision/occlusion proof, not supervised optimization",
        "radii_m":RADIUS_M,"routine_pose_names":pose_names[:ROUTINE_COUNT],
        "stress_pose_names":pose_names[ROUTINE_COUNT:],
    }
    (out/"recipe.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    result=[]
    try:
        fig,axes=plt.subplots(1,len(CANDIDATES),figsize=(20,5.4),layout="constrained")
        im=None
        for (name,xs,ys),ax in zip(CANDIDATES,axes):
            config=GroundGridCfg(
                x_coords=xs,y_coords=ys,
                validity_mode=VALIDITY_GEOMETRY_ONLY_FLAT_TRAINING,
            )
            mirror=GridPermutation.from_axes(config.x_coords,config.y_coords)
            assert config.num_points==int(name.removeprefix("C"))
            coords=config.grid_xy(dtype=torch.float64).numpy()
            # Existing preflight probe lattice is a 0.1m mesh. No interpolation
            # or synthetic points: exact nearest source probe identity.
            source_ids=[]
            for xpos,ypos in coords:
                xi=int(np.argmin(np.abs(xx-xpos)))
                yi=int(np.argmin(np.abs(yy-ypos)))
                if abs(xx[xi]-xpos)>1e-7 or abs(yy[yi]-ypos)>1e-7:
                    raise AssertionError(f"{name} node off real measured probe lattice")
                source_ids.append(yi*len(xx)+xi)
            ids=np.asarray(source_ids,dtype=np.int64)
            assert len(np.unique(ids))==len(ids)
            # Exact double mirror: a permutation of indices (not just sums).
            perm=np.asarray(mirror.source_indices)
            assert np.array_equal(perm[perm],np.arange(config.num_points))
            assert all(
                abs(coords[k,0]-coords[perm[k],0])<1e-6 and
                abs(coords[k,1]+coords[perm[k],1])<1e-6
                for k in range(config.num_points)
            )

            per_pose=[]
            for i,pose in enumerate(pose_names):
                item={
                    "pose":pose,
                    "class":"routine" if i<ROUTINE_COUNT else "stress",
                }
                for prefix,sourceD in (("pool",poolD),("fps",fpsD)):
                    d=sourceD[i,ids]
                    for radius in RADIUS_M:
                        item[f"{prefix}_r{radius:.2f}m"]=float(np.mean(d<=radius))
                    item[f"{prefix}_p50_distance_m"]=float(np.median(d))
                    item[f"{prefix}_p95_distance_m"]=float(np.percentile(d,95))
                per_pose.append(item)
            fdr=fpsD[:,ids]
            pdr=poolD[:,ids]
            rec={
                "name":name,"num_points":config.num_points,"nx":config.nx,
                "ny":config.ny,
                "min_delta_x_m":min(np.diff(xs)),"max_delta_x_m":max(np.diff(xs)),
                "min_delta_y_m":min(np.diff(ys)),"max_delta_y_m":max(np.diff(ys)),
                "x_coords":list(xs),"y_coords":list(ys),
                "point_order":"y_outer_x_inner",
                "mirror_bijection":True,
                "ground_identity_sha256":config.identity_hash(),
                "source_probe_ids":ids.tolist(),
                "metrics":{
                    "nominal_fps_0.15m":float(np.mean(fdr[0]<=.15)),
                    "nominal_fps_0.20m":float(np.mean(fdr[0]<=.20)),
                    "nominal_fps_0.25m":float(np.mean(fdr[0]<=.25)),
                    "routine_worst_case_fps_0.20m":min(r["fps_r0.20m"] for r in per_pose[:ROUTINE_COUNT]),
                    "routine_worst_case_fps_0.25m":min(r["fps_r0.25m"] for r in per_pose[:ROUTINE_COUNT]),
                    "routine_average_fps_0.20m":float(np.mean(fdr[:ROUTINE_COUNT]<=.20)),
                    "routine_average_fps_0.25m":float(np.mean(fdr[:ROUTINE_COUNT]<=.25)),
                    "pitch_up_5_fps_0.20m":per_pose[1]["fps_r0.20m"],
                    "pitch_up_5_fps_0.25m":per_pose[1]["fps_r0.25m"],
                    "pitch_up_10_fps_0.25m":per_pose[9]["fps_r0.25m"],
                    "stress_average_fps_0.25m":float(np.mean(fdr[ROUTINE_COUNT:]<=.25)),
                    "fps_never_0.25m_supported_at_any_pose":int(np.sum(~np.any(fdr<=.25,axis=0))),
                    "fps_all_routine_0.25m_count":int(np.sum(np.all(fdr[:ROUTINE_COUNT]<=.25,axis=0))),
                    "fps_all_15_pose_0.25m_count":int(np.sum(np.all(fdr<=.25,axis=0))),
                    "pool_all_15_pose_0.25m_count":int(np.sum(np.all(pdr<=.25,axis=0))),
                    "fps_routine_8_of_9_at_0.25m_count":int(np.sum(np.sum(fdr[:ROUTINE_COUNT]<=.25,axis=0)>=8)),
                },
                "per_pose":per_pose,
                "status":"measured-S1-grid-candidate-only; no production grid change",
            }
            result.append(rec)
            group=np.asarray(dataset["fps_routine_fraction"]).reshape(len(yy),len(xx))
            im=ax.imshow(group,extent=(xx[0]-.05,xx[-1]+.05,yy[0]-.05,yy[-1]+.05),
                         cmap="viridis",vmin=0,vmax=1,origin="lower",
                         interpolation="nearest")
            ax.scatter(coords[:,0],coords[:,1],s=(8 if len(coords)>120 else 20),
                       c="#fb7185",alpha=.85,marker=".",edgecolors="none")
            ax.set_xlim(.35,3.6);ax.set_ylim(-.85,.85)
            ax.set_aspect("equal",adjustable="box")
            ax.set_title(f"{name}: {config.nx} x {config.ny} = {config.num_points} pts",fontsize=11)
            ax.set_xlabel("BASE x (m)")
            ax.set_ylabel("BASE y (m)")
            ax.grid(alpha=.09)
            print(f"{name} m={config.num_points} regular@0.20="
                  f"{rec['metrics']['routine_average_fps_0.20m']:.4f}"
                  f" up5@0.25={rec['metrics']['pitch_up_5_fps_0.25m']:.4f}"
                  f" up10@0.25={rec['metrics']['pitch_up_10_fps_0.25m']:.4f}",
                  flush=True)
        fig.colorbar(im,ax=axes,shrink=.64,
                     label="Fraction of 9 routine poses with <=0.25m 3D near FPS support")
        fig.suptitle("Measured B8 near-ground support and FIVE fixed S1 supervised grid candidates\n"
                     "All candidates cover x=[0.6,3.4], |y|<=0.7 m; points mark candidate"
                     " labels only, not the robot ground sensor",fontsize=12)
        fig.savefig(out/"S1_candidate_grid_overlay.png",dpi=165)
        plt.close(fig)

        # Min/mean case by array, easy for external audits.
        flat=[]
        for r in result:
            for p in r["per_pose"]:
                flat.append({"candidate":r["name"],"num_points":r["num_points"],**p})
        with (out/"grid_pose_support.csv").open("w",newline="") as fh:
            w=csv.DictWriter(fh,fieldnames=list(flat[0]))
            w.writeheader();w.writerows(flat)
        with (out/"grid_summary.csv").open("w",newline="") as fh:
            fields=("candidate","num_points","nx","ny",
                    "nominal_0.20m","routine_avg_0.20m",
                    "routine_worst_0.20m","pitch_up_5_0.20m",
                    "pitch_up_5_0.25m","pitch_up_10_0.25m",
                    "all_15_pose_0.25m")
            w=csv.DictWriter(fh,fieldnames=fields);w.writeheader()
            for r in result:
                m=r["metrics"]
                w.writerow({
                    "candidate":r["name"],"num_points":r["num_points"],
                    "nx":r["nx"],"ny":r["ny"],
                    "nominal_0.20m":m["nominal_fps_0.20m"],
                    "routine_avg_0.20m":m["routine_average_fps_0.20m"],
                    "routine_worst_0.20m":m["routine_worst_case_fps_0.20m"],
                    "pitch_up_5_0.20m":m["pitch_up_5_fps_0.20m"],
                    "pitch_up_5_0.25m":m["pitch_up_5_fps_0.25m"],
                    "pitch_up_10_0.25m":m["pitch_up_10_fps_0.25m"],
                    "all_15_pose_0.25m":m["fps_all_15_pose_0.25m_count"],
                })
        report.update(state="completed",grid_candidates=result)
    except BaseException as err:
        report.update(state="failed",error=repr(err))
        raise
    finally:
        (out/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print(f"[S1-grid] {report['state']} {out/'report.json'}",flush=True)


if __name__=="__main__":
    raise SystemExit(main())
