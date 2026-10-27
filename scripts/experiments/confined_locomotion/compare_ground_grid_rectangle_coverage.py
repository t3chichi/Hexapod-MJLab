#!/usr/bin/env python
"""Select fixed robust rectangular ground labels from measured pose scans.

Strict geometry membership is a necessary cone/range condition checked on all
points in 9 routine poses. The second, independent criterion is a sampled
near-FPS TRUE FLOOR return within r metres in 3D. An in-cone position is not
automatically observed, especially on non-flat or occluded terrain.

This is deliberately NOT an installation of GroundGridCfg/MLP/PPO. Uses
artifact from analyze_ground_sampling_support.py v4 (15 poses, 1833 probes,
static real CUDA 86400-ray returns) and validates grid order/mirror against
production GroundGridCfg and GridPermutation. B8/S1 labels remain only in
historical evidence fields and candidate IDs.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"src"))

def seq(a,b,d):
    n=round((b-a)/d)
    assert abs(a+n*d-b)<1e-8
    return tuple(round(a+i*d,3) for i in range(n+1))

# Rectangles use real measured 0.1m source probe nodes, not interpolated.
# Keep first and last boundary positions exactly; y axes are mirror pairs.
Y7=seq(-.6,.6,.2)
Y6=seq(-.5,.5,.2)
CASES=(
    ("R56_area_coarse",seq(1.0,3.1,.3),Y7),
    ("R77_area_medium",seq(1.0,3.0,.2),Y7),
    ("R154_area_dense",seq(1.0,3.1,.1),Y7),
    ("R78_near_medium",seq(.8,3.2,.2),Y6),
    ("R?S1_prior_unchecked",seq(.6,3.4,.2),seq(-.7,.7,.2)),
)
def main():
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from hexapod_mjlab.tasks.confined_locomotion.ground import (
        GroundGridCfg,VALIDITY_GEOMETRY_ONLY_FLAT_TRAINING
    )
    from hexapod_mjlab.rl.confined_locomotion.symmetry import GridPermutation
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--source",type=Path,default=ROOT/"logs/verification/current/ground_height/01_ground_visibility_by_pose_20261019")
    cli.add_argument("--output",type=Path,required=True)
    arg=cli.parse_args()
    src=arg.source.expanduser().resolve()
    out=arg.output.expanduser().resolve()
    out.mkdir(parents=True,exist_ok=False)
    srcfile=src/"probes_support_matrix.npz"
    metadata=src/"report.json"
    r=json.loads(metadata.read_text())
    assert r["state"]=="completed" and r["pose_checks"]==15
    npz=np.load(srcfile,allow_pickle=False)
    mask=npz["strict_ground_query_cone_membership"]
    measured=npz["fps_distances_m"]
    pool=npz["pool_distances_m"]
    xy=npz["grid_xyz"]; xx=npz["x"];yy=npz["y"]
    assert mask.shape==measured.shape==(15,1833)
    output=[]
    for name,xs,ys in CASES:
        g=GroundGridCfg(x_coords=xs,y_coords=ys,validity_mode=VALIDITY_GEOMETRY_ONLY_FLAT_TRAINING)
        pi=GridPermutation.from_axes(xs,ys)
        coords=g.grid_xy(dtype=torch.float64).numpy()
        indices=[]
        for x,y in coords:
            u=int(np.abs(xx-x).argmin());v=int(np.abs(yy-y).argmin())
            assert abs(xx[u]-x)<1e-8 and abs(yy[v]-y)<1e-8
            indices.append(v*len(xx)+u)
        ids=np.array(indices,dtype=np.int64)
        assert len(set(ids.tolist()))==g.num_points
        perm=np.asarray(pi.source_indices)
        assert np.array_equal(perm[perm],np.arange(g.num_points))
        strict=mask[:,ids]
        f=measured[:,ids]
        p=pool[:,ids]
        # Check full rectangle, not just selected sparse locations: the
        # original 0.1m lattice is the independent geometry oracle.
        area_mask=(xy[:,0]>=xs[0]-1e-9)&(xy[:,0]<=xs[-1]+1e-9)&(
            np.abs(xy[:,1])<=max(abs(ys[0]),abs(ys[-1]))+1e-9
        )
        area_all9=bool(mask[:9,area_mask].all())
        corners=(
            (xs[0],ys[0]),(xs[0],ys[-1]),(xs[-1],ys[0]),(xs[-1],ys[-1])
        )
        corner_ids=[]
        for xc,yc in corners:
            k=np.argmin(np.sum((xy-[xc,yc])**2,axis=1))
            assert np.linalg.norm(xy[k]-[xc,yc])<1e-8
            corner_ids.append(k)
        corner_all9=bool(mask[:9,corner_ids].all())
        # On a flat horizontal world plane transformed rigidly into BASE,
        # target z(x,y) is affine. A circular cone with <90° half-angle and
        # sphere range cut are convex, so all four corners within both cuts
        # prove interior cone feasibility for those nine static poses.
        assert corner_all9==area_all9
        q={
            "candidate":name,"M":g.num_points,"nx":g.nx,"ny":g.ny,
            "x_min_m":xs[0],"x_max_m":xs[-1],
            "y_halfwidth_m":max(abs(ys[0]),abs(ys[-1])),
            "rectangle_area_m2":round((xs[-1]-xs[0])*(ys[-1]-ys[0]),4),
            "mirror_bijection":True,"grid_identity_sha256":g.identity_hash(),
            "all_routine_9_pose_strict_cone_all_probes":area_all9,
            "all_routine_9_pose_strict_cone_corners":corner_all9,
            "num_full_area_probes":int(area_mask.sum()),
            "num_full_area_bad_nominal":int(np.sum(~mask[0,area_mask])),
            "num_full_area_bad_routine":int(np.sum(~mask[:9,area_mask])),
            "nominal_fps_within_0.15m":float(np.mean(f[0]<=.15)),
            "nominal_fps_within_0.20m":float(np.mean(f[0]<=.20)),
            "routine_min_fps_within_0.20m":float(np.mean(f[:9]<=.20,axis=1).min()),
            "routine_mean_fps_within_0.20m":float(np.mean(f[:9]<=.20)),
            "routine_min_fps_within_0.25m":float(np.mean(f[:9]<=.25,axis=1).min()),
            "pitch_up5_fps_within_0.20m":float(np.mean(f[1]<=.20)),
            "pitch_up10_fps_within_0.25m":float(np.mean(f[9]<=.25)),
            "all_15_poses_strict_cone_fraction":float(np.mean(strict)),
            "all_15_poses_fps_within_0.25m":float(np.mean(f<=.25)),
            "all_15_poses_pool_within_0.25m":float(np.mean(p<=.25)),
            "pointwise_fps_all_9_routine_within_0.20m":int(np.all(f[:9]<=.20,axis=0).sum()),
            "pointwise_fps_all_15_within_0.25m":int(np.all(f<=.25,axis=0).sum()),
            "pose_fps_within_0.20m":{
                str(pose):float(np.mean(f[i]<=.20)) for i,pose in enumerate(npz["pose_names"])
            },
            "x_coords":xs,"y_coords":ys,
        }
        output.append(q)
        print(name,"M",q["M"],"area",q["rectangle_area_m2"],
              "all9strict",area_all9,
              "routine_min_0.20",round(q["routine_min_fps_within_0.20m"],3),
              "up10_0.25",round(q["pitch_up10_fps_within_0.25m"],3),flush=True)
    report={
        "schema":"confined_locomotion.stageA.S1_strict_rectangles/v1",
        "created_utc":datetime.now(timezone.utc).isoformat(),
        "source_sha256":{str(s.relative_to(ROOT)):hashlib.sha256(s.read_bytes()).hexdigest()
                         for s in (metadata,srcfile,Path(__file__))},
        "ground_truth":"actual B8 15 static pose CUDA floor raycasts and provider ground-query points",
        "use":"compare only; NO production grid change",
        "routine_poses":npz["pose_names"][:9].tolist(),
        "strict_geometry":"9 static routine pose all-probe cone inclusion for flat queried floor; rectangle corners also checked. Query point continuous cone feasibility, not full discrete scan observability.",
        "sampling":"Near FPS256 floor-hit support threshold 0.20m 3D, plus 0.25m stress. 15-pose matrix includes pitch_up10 and mixed poses.",
        "unknowns":"Actual step/wall occlusion, camera/robot own occlusion, motion/collision safety, scale of true prediction error, learning and total GPU memory",
        "candidates":output,
    }
    (out/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    fields=[
        "candidate","M","nx","ny","rectangle_area_m2",
        "all_routine_9_pose_strict_cone_all_probes","num_full_area_bad_nominal",
        "routine_min_fps_within_0.20m","routine_mean_fps_within_0.20m",
        "pitch_up5_fps_within_0.20m","pitch_up10_fps_within_0.25m",
        "all_15_poses_strict_cone_fraction",
    ]
    with (out/"summary.csv").open("w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader()
        for rec in output:writer.writerow({key:rec[key] for key in fields})
    inside=mask[:9].all(axis=0).reshape(len(yy),len(xx))
    fig,ax=plt.subplots(figsize=(12,6.1),layout="constrained")
    image=ax.imshow(inside,extent=(xx[0]-.05,xx[-1]+.05,yy[0]-.05,yy[-1]+.05),
        origin="lower",vmin=0,vmax=1,cmap="Greys",interpolation="nearest")
    for candidate,color,linestyle in (
        ("R?S1_prior_unchecked","#c64b38","--"),
        ("R154_area_dense","#287cad","-"),
        ("R78_near_medium","#3b9955","-."),
    ):
        x=next(x for x in output if x["candidate"]==candidate)
        ax.add_patch(Rectangle((x["x_min_m"],-x["y_halfwidth_m"]),
           x["x_max_m"]-x["x_min_m"],2*x["y_halfwidth_m"],
           fill=False,edgecolor=color,linewidth=2.1,linestyle=linestyle,
           label=f"{candidate}: {x['rectangle_area_m2']:.2f} m²"))
    ax.legend(loc="upper left",fontsize=9)
    ax.set(xlim=(.35,3.55),ylim=(-1.45,1.45),
           xlabel="BASE x (m)",ylabel="BASE y (m)",
           title="Actual B8 cone membership INTERSECTION of nine static routine poses")
    ax.set_aspect("equal",adjustable="box")
    fig.savefig(out/"strict_S1_rectangle_9poses.png",dpi=165)
    plt.close(fig)
    print("[S1 strict] report ready:",out/"report.json",flush=True)
if __name__=="__main__":
    main()
