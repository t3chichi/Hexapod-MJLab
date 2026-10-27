#!/usr/bin/env python
"""Real near-cloud ground support under rigid body pose variations.

The historical label B8 is retained only in evidence schemas/filenames and
denotes the current widened near-field grouping (theta=28, tilt=56, BASE
x <= 3.5m); it is not a current stage or entry name.

This is an ENGINEERING DIAGNOSTIC, not ground-grid implementation or training.
Use the live confined task's compiled floor plane, actual Airy 86,400-ray
pattern, calibrated mount, real Warp CUDA LidarSensor, B8 near partition and
production metric FPS256. The ground-query evaluator uses the task's
FlatGroundProvider to compute signed body-z labels at a DENSE PROBE grid.
The probe is NOT the training grid, and no change to GroundGridCfg is made.

A grid probe is 'geometrically supported' if a true B8 near FLOOR hit lies
within a declared metric 3D nearest-neighbor radius of its actual ground
intersection. Miss endpoints and wall returns never count as ground support.
We report BOTH full near-hit pool and near FPS256 support, separating sensor
coverage from sampling capacity. This distance is a proxy, NOT observability
proof (occlusion and multimodal ambiguity must be checked on real terrain).

Fixed B8 cones are BASE relative; tilt changes ground intersections, not the
raw angular selection. Pose changes here are sensor/robot rigid transforms,
NOT physically simulated gaits, joint articulation, self-occlusion or 10Hz
moving trajectories. Yaw checks are scene-invariant sanity witnesses only.

The 40-point runtime ground grid is a bootstrap fixture; not used in the
analysis, proposal, ranking or visualization.

Usage:
  python scripts/experiments/confined_locomotion/analyze_ground_sampling_support.py \\
      --output logs/verification/ground_sampling_support_YYYYMMDD
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

# Grid of PROBES for regional coverage, independent of training grid shape/M.
PROBE_X=(-1.2,3.4,47)   # 0.10m x spacing
PROBE_Y=(-1.9,1.9,39)   # 0.10m y spacing
SUPPORT_THRESHOLDS_M=(0.15,0.20,0.25,0.30)
PRIMARY_THRESHOLD_M=0.25
POSES=(
    ("nominal",0.,0.,0.,0.,"routine"),
    ("pitch_up_5",0.,5.,0.,0.,"routine"),
    ("pitch_down_5",0.,-5.,0.,0.,"routine"),
    ("roll_left_5",5.,0.,0.,0.,"routine"),
    ("roll_right_5",-5.,0.,0.,0.,"routine"),
    ("height_up_8cm",0.,0.,0.,0.08,"routine"),
    ("height_down_8cm",0.,0.,0.,-0.08,"routine"),
    ("yaw_left_12",0.,0.,12.,0.,"routine"),
    ("yaw_right_12",0.,0.,-12.,0.,"routine"),
    ("pitch_up_10",0.,10.,0.,0.,"stress"),
    ("pitch_down_10",0.,-10.,0.,0.,"stress"),
    ("roll_left_10",10.,0.,0.,0.,"stress"),
    ("roll_right_10",-10.,0.,0.,0.,"stress"),
    ("pitch_up_5_roll_left_5",5.,5.,0.,0.,"stress"),
    ("pitch_up_5_roll_right_5",-5.,5.,0.,0.,"stress"),
)
FLOOR_RANGE=15.0


def args_parser(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True,help="new evidence directory")
    p.add_argument("--gpu-id",type=int,default=0)
    p.add_argument("--seed",type=int,default=20261014)
    p.add_argument("--grid-preset",choices=("s1_187",),default=None,
                   help="evaluate exact 187 formal-grid query positions rather than 1833 broad probes")
    return p.parse_args(argv)


def to_np(t):
    return t.detach().cpu().numpy().copy()


def floor_distances(origin,world_dirs,floor_z):
    """Independent analytic ray-ground intersection in the actual world plane."""
    import numpy as np
    z=np.asarray(world_dirs)[:,2]
    d=np.full(z.shape,np.inf,dtype=np.float64)
    good=np.abs(z)>1e-10
    np.divide(floor_z-float(origin[2]),z,out=d,where=good)
    valid=good&(d>=0)&(d<FLOOR_RANGE)
    return np.where(valid,d,FLOOR_RANGE).astype(np.float32)


def get_probe_labels(model,xy,root_pos,rot,*,probe_x=None,probe_y=None):
    import numpy as np
    import torch
    from hexapod_mjlab.tasks.confined_locomotion.ground import (
        GroundGridCfg, FlatGroundProvider,VALIDITY_GEOMETRY_ONLY_FLAT_TRAINING
    )
    from scipy.spatial.transform import Rotation
    cfg=GroundGridCfg(
        x_coords=tuple(np.linspace(*PROBE_X) if probe_x is None else probe_x),
        y_coords=tuple(np.linspace(*PROBE_Y) if probe_y is None else probe_y),
        validity_mode=VALIDITY_GEOMETRY_ONLY_FLAT_TRAINING,
    )
    provider=FlatGroundProvider(model,cfg)
    quat=rot.as_quat()  # xyzw -> wxyz
    result=provider.query(torch.as_tensor(root_pos,dtype=torch.float64),
                          torch.as_tensor(quat[[3,0,1,2]],dtype=torch.float64))
    actual=to_np(result.positions)
    valid=to_np(result.valid)
    assert actual.shape==(len(xy),3) and valid.shape==(len(xy),)
    assert valid.all(),"all current probe poses must be non-parallel to plane"
    assert np.allclose(actual[:,:2],xy,atol=1e-6)
    # True-body-z target is physical ground, not arbitrary (xy,z=constant).
    return actual


def measure(rows,xy,thresholds):
    """Per-case threshold occupancy; a ground-neighbor is never a 15m miss."""
    import numpy as np
    cases={p["pose"]:p for p in rows}
    cases_count=len(rows)
    heat={}
    for kind in ("pool","fps"):
        dist=np.stack([p[f"{kind}_nearest_m"] for p in rows],axis=0)
        heat[kind]={
            "dist":dist,
            "fractions":{str(t):np.mean(dist<=t,axis=0) for t in thresholds},
        }
    routine=[i for i,r in enumerate(rows) if r["class"]=="routine"]
    stress=[i for i,r in enumerate(rows) if r["class"]=="stress"]
    for kind in ("pool","fps"):
        dist=heat[kind]["dist"]
        heat[kind]["routine_fraction"]=np.mean(dist[routine]<=PRIMARY_THRESHOLD_M,axis=0)
        heat[kind]["stress_fraction"]=np.mean(dist[stress]<=PRIMARY_THRESHOLD_M,axis=0)
        heat[kind]["all_fraction"]=np.mean(dist<=PRIMARY_THRESHOLD_M,axis=0)
        heat[kind]["nominal"]=dist[0]<=PRIMARY_THRESHOLD_M
        # Parameter-free hard intersection and a documented soft-coverage
        # candidate: the latter is NOT an acceptance threshold.
        heat[kind]["hard_all"]=np.all(dist<=PRIMARY_THRESHOLD_M,axis=0)
        heat[kind]["routine_80"]=np.mean(dist[routine]<=PRIMARY_THRESHOLD_M,axis=0)>=0.80
        heat[kind]["routine_90"]=np.mean(dist[routine]<=PRIMARY_THRESHOLD_M,axis=0)>=0.90
    return heat


def recommend_rectangles(xy,heat):
    """Rank fixed-BODY symmetric rectangles from the probe mask (NOT adopt).

    Ranking is exploratory, with explicit fixed, debatable filter choices:
    every point supported in nominal near-hit pool; >=95% routine *pool*
    support; >=85% routine FPS support (not pointwise). An actual supervision
    design may need nonrectangular masks or nonuniform point coordinates.
    """
    import numpy as np
    from itertools import product
    pool_nom=heat["pool"]["nominal"]
    routine_pool=heat["pool"]["routine_fraction"]
    routine_fps=heat["fps"]["routine_fraction"]
    candidates=[]
    for xlo,xhi,yhalf in product(
        (-.6,-.4,-.2,0.,.2,.4,.6),
        (1.2,1.6,2.,2.4,2.8,3.2,3.4),
        (.3,.4,.5,.6,.7,.8,1.,1.2,1.4),
    ):
        if xhi-xlo<1.0: continue
        inside=(xy[:,0]>=xlo-1e-8)&(xy[:,0]<=xhi+1e-8)&(np.abs(xy[:,1])<=yhalf+1e-8)
        idx=np.flatnonzero(inside)
        if not len(idx):continue
        s0=float(pool_nom[idx].mean())
        sp=float((routine_pool[idx]>=.80).mean())
        sf=float((routine_fps[idx]>=.80).mean())
        hard=float(heat["pool"]["hard_all"][idx].mean())
        if s0<0.98 or sp<0.95 or sf<0.85:continue
        candidates.append({
            "x_min_m":xlo,"x_max_m":xhi,"y_half_width_m":yhalf,
            "nominal_pool_0.25m_point_fraction":round(s0,4),
            "routine_pool_80pct_pose_point_fraction":round(sp,4),
            "routine_fps_80pct_pose_point_fraction":round(sf,4),
            "all_pose_pool_point_fraction":round(hard,4),
            "probe_points":int(len(idx)),
            "geometric_rectangle_area_m2":round((xhi-xlo)*2*yhalf,3),
            "bound_coverage_routine_fps_avg":round(float(routine_fps[idx].mean()),3),
        })
    candidates.sort(key=lambda x:(x["geometric_rectangle_area_m2"],
                                   x["bound_coverage_routine_fps_avg"]),reverse=True)
    return candidates


def named_region_comparison(xy,heat):
    """Two visual candidate shapes for discussion, NOT ground-grid activation."""
    import numpy as np
    xx=xy[:,0]
    yy=np.abs(xy[:,1])
    rect=(xx>=.6-1e-9)&(xx<=3.4+1e-9)&(yy<=.7+1e-9)
    trapez=(xx>=.6-1e-9)&(xx<=3.4+1e-9)&(
        yy<=np.interp(xx,[.6,1.2,2.0,3.2,3.4],[.5,.9,1.2,1.4,1.4])+1e-9
    )
    out={}
    for key,mask in (("S1_safe_rectangle",rect),("S2_expanding_trapezoid",trapez)):
        pool=heat["pool"]["dist"][:,mask]
        fps=heat["fps"]["dist"][:,mask]
        out[key]={
            "description":("x∈[0.6,3.4], |y|<=0.7"
                           if key.startswith("S1") else
                           "x∈[0.6,3.4], width per knot x=[.6,1.2,2.0,3.2,3.4], halfwidth=[.5,.9,1.2,1.4,1.4]"),
            "probe_points":int(mask.sum()),
            "diagnostic_probe_area_m2":round(float(mask.sum())*.01,3),
            "nominal_pool_near_support":float(np.mean(pool[0]<=PRIMARY_THRESHOLD_M)),
            "nominal_fps_near_support":float(np.mean(fps[0]<=PRIMARY_THRESHOLD_M)),
            "routine_fps_pose_coverage":{
                POSES[i][0]:float(np.mean(fps[i]<=PRIMARY_THRESHOLD_M))
                for i in range(9)
            },
            "stress_fps_pose_coverage":{
                POSES[i][0]:float(np.mean(fps[i]<=PRIMARY_THRESHOLD_M))
                for i in range(9,len(POSES))
            },
            "all_15_pose_fps_pointwise_fraction":float(np.mean(np.all(fps<=PRIMARY_THRESHOLD_M,axis=0))),
            "all_pose_fps_mean_fraction":float(np.mean(fps<=PRIMARY_THRESHOLD_M)),
        }
    return out


def plot_heatmaps(output,xy,heat,candidates):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon,Rectangle
    xax=np.linspace(PROBE_X[0],PROBE_X[1],PROBE_X[2])
    yax=np.linspace(PROBE_Y[0],PROBE_Y[1],PROBE_Y[2])
    ref=np.array([[-.89,-.59],[0,-.69],[.89,-.59],
                  [.89,.59],[0,.69],[-.89,.59]])
    def decorate(ax):
        ax.add_patch(Polygon(ref,closed=True,fill=False,edgecolor="#161616",
                             linestyle="--",linewidth=1.2,zorder=4))
        ax.scatter([.645],[0],marker="o",s=37,c="#fbab26",edgecolors="white",zorder=5)
        ax.set_xlim(-1.2,3.4);ax.set_ylim(-1.65,1.65);ax.set_aspect("equal")
        ax.set_xlabel("BASE x (m)");ax.set_ylabel("BASE y (m)")
    fields=[
        ("pool","nominal","Near hit-pool ground support (nominal)",None),
        ("pool","routine_fraction","Near hit-pool pose support (routine)",None),
        ("pool","all_fraction","Near hit-pool pose support (all)",None),
        ("fps","nominal","FPS256 ground support (nominal)",None),
        ("fps","routine_fraction","FPS256 pose support (routine)",None),
        ("fps","all_fraction","FPS256 pose support (all)",None),
    ]
    fig,axes=plt.subplots(2,3,figsize=(16,9.2),layout="constrained")
    im=None
    for ax,(kind,key,label,_) in zip(axes.flat,fields):
        vals=np.asarray(heat[kind][key],dtype=float).reshape(len(yax),len(xax))
        im=ax.imshow(vals,extent=(xax[0]-.05,xax[-1]+.05,yax[0]-.05,yax[-1]+.05),
                     origin="lower",vmin=0,vmax=1,cmap="viridis",aspect="equal",
                     interpolation="nearest")
        ax.contour(xax,yax,vals,levels=[.5,.8],colors=["white","#e2af38"],
                   linewidths=[.7,1.0],alpha=.85)
        decorate(ax);ax.set_title(label,fontsize=11)
    fig.colorbar(im,ax=axes.ravel().tolist(),label=f"Fraction of pose cases with 3D-nearest distance <= {PRIMARY_THRESHOLD_M:.2f}m",shrink=.76)
    fig.suptitle(
        "B8 (28°/56°/x<=3.5m) ground-hit SUPPORT, not arbitrary supervision mask\n"
        "Floor plane real CUDA raycasts; each label independently solved along BODY-z. "
        "Dashes = startup morphology (not robot mesh).",fontsize=12.5)
    fig.savefig(output/"B8_ground_pose_support_6panels.png",dpi=160)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(10.7,6.2),layout="constrained")
    pool=heat["pool"]["routine_fraction"].reshape(len(yax),len(xax))
    fps=heat["fps"]["routine_fraction"].reshape(len(yax),len(xax))
    ax.imshow(pool,origin="lower",extent=(xax[0]-.05,xax[-1]+.05,yax[0]-.05,yax[-1]+.05),
              cmap="Blues",vmin=0,vmax=1,interpolation="nearest")
    ax.contour(xax,yax,pool,levels=[.8],colors=["#2457a6"],linewidths=1.7)
    ax.contour(xax,yax,fps,levels=[.8],colors=["#c35e23"],linewidths=1.9)
    if candidates:
        c=candidates[0]
        ax.add_patch(Rectangle((c["x_min_m"],-c["y_half_width_m"]),
                               c["x_max_m"]-c["x_min_m"],2*c["y_half_width_m"],
                               ec="#e11d48",lw=2,ls="--",fc="none"))
        ax.text(.02,.97,
                f"Exploratory top rectangular candidate: x=[{c['x_min_m']},{c['x_max_m']}], |y|<={c['y_half_width_m']} m",
                transform=ax.transAxes,fontsize=10,va="top",color="#a11b41")
    decorate(ax)
    ax.set_title("Routine pose stability (pool vs FPS, 80% contour); exploratory rectangle only")
    fig.savefig(output/"B8_ground_boundary_comparison.png",dpi=165)
    plt.close(fig)

    # A second diagram combines the two human-discussion choices with the
    # actual coverage map, clearly preserving the original 3D NN evidence.
    fig,axs=plt.subplots(1,2,figsize=(15,6.5),layout="constrained")
    xknots=np.array([.6,1.2,2.0,3.2,3.4])
    half=np.array([.5,.9,1.2,1.4,1.4])
    poly=np.concatenate([
        np.c_[xknots,half],np.c_[xknots[::-1],-half[::-1]]
    ],axis=0)
    rect=np.array([[.6,-.7],[3.4,-.7],[3.4,.7],[.6,.7]])
    for ax, (key,title) in zip(
        axs, (("nominal","Nominal pose"),("routine_fraction","9 routine pose cases"))
    ):
        vals=heat["fps"][key].reshape(len(yax),len(xax)).astype(float)
        im=ax.imshow(vals,
            extent=(xax[0]-.05,xax[-1]+.05,yax[0]-.05,yax[-1]+.05),
            origin="lower",vmin=0,vmax=1,cmap="viridis",
            interpolation="nearest")
        ax.add_patch(Polygon(poly,closed=True,fill=False,
            edgecolor="#fb923c",linewidth=2.4,zorder=5))
        ax.add_patch(Polygon(rect,closed=True,fill=False,
            edgecolor="#f43f5e",linewidth=2.1,linestyle="--",zorder=6))
        decorate(ax)
        ax.set_title(f"{title} | orange solid=S2 trapezoid, red dashed=S1 rectangle")
    fig.colorbar(im,ax=axs,shrink=.7,label="Near FPS256 ground support (0.25m 3D proximity)")
    fig.suptitle(
        "B8 ground-supervision SHAPE candidates on measured scan support\n"
        "S1: x=0.6..3.4, y=±0.7m; S2: x=0.6..3.4, symmetric widening "
        "half-width 0.5→1.4m. Neither is installed in training.",fontsize=12)
    fig.savefig(output/"B8_supervision_S1_S2_overlay.png",dpi=165)
    plt.close(fig)


def main(argv=None):
    import numpy as np
    import torch
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
    from hexapod_mjlab.tasks.confined_locomotion.flat_env_cfg import make_training_env_cfg
    from hexapod_mjlab.tasks.confined_locomotion.ground import GroundGridCfg,VALIDITY_GEOMETRY_ONLY_FLAT_TRAINING
    from hexapod_mjlab.tasks.confined_locomotion.runtime import TrainingRuntime
    from hexapod_mjlab.tasks.confined_locomotion.scanner import ConfinedLidarCfg,_quat_apply_xyzw,partition_cone_forward_x,farthest_point_sampling_indices
    from hexapod_mjlab.utils.lidar import LidarSensor,LidarConfig,LidarType

    args=args_parser(argv)
    if args.gpu_id<0 or not torch.cuda.is_available() or args.gpu_id>=torch.cuda.device_count():
        raise RuntimeError("explicit CUDA device required")
    output=args.output.expanduser().resolve()
    output.mkdir(parents=True,exist_ok=False)
    paths=(Path(__file__),ROOT/"src/hexapod_mjlab/tasks/confined_locomotion/scanner.py",
           ROOT/"src/hexapod_mjlab/tasks/confined_locomotion/ground.py",
           ROOT/"src/hexapod_mjlab/assets/robots/el_4090/mjcf/el_4090_lidar.xml")
    report={
        "schema":"confined_locomotion.stageA_B8_ground_support/v1",
        "state":"started","created_utc":datetime.now(timezone.utc).isoformat(),
        "seed":args.seed,"device":f"cuda:{args.gpu_id}",
        "rule":"near=original B8 cone theta28° tilt56° AND observed endpoint BASE x<=3.5m",
        "probe_grid_m":({"x":PROBE_X,"y":PROBE_Y,"resolution_m":0.1,
                           "probe_only_not_training":True}
                          if args.grid_preset is None else
                          {"preset":args.grid_preset,"nodes_are_actual_s1_187_grid":True}),
        "poses":POSES,"thresholds_m":SUPPORT_THRESHOLDS_M,
        "primary_distance_m":PRIMARY_THRESHOLD_M,
        "support_definition":"3D nearest euclidean distance from genuine near FLOOR hit to a provider ground label in scan-time BASE xyz",
        "limitations":"one initial T1 pose, independent rigid transforms (not gaits/joint actuation), clean plane only, no self occlusion/terrain variation; distance proxy is not proof of terrain observability",
        "production_ground_grid_unchanged":True,
        "sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    }
    (output/"recipe.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    env=runtime=None
    try:
        torch.cuda.set_device(args.gpu_id)
        device=f"cuda:{args.gpu_id}"
        # Engineering bootstrap ONLY; replaced by dense independent probes.
        grid=GroundGridCfg(x_coords=(-.8,-.4,0.,.4,.8,1.2,1.6,2.),
                           y_coords=(-.8,-.4,0.,.4,.8),
                           validity_mode=VALIDITY_GEOMETRY_ONLY_FLAT_TRAINING)
        env=ManagerBasedRlEnv(make_training_env_cfg(num_envs=1,seed=args.seed),device=device)
        vec=RslRlVecEnvWrapper(env,clip_actions=None)
        runtime=TrainingRuntime(vec,grid_cfg=grid,seed=args.seed,mode="engineering",
                                output_dir=output/"runtime",
                                lidar_cfg=ConfinedLidarCfg(),perturbation_cfg=None)
        frame=next(v["frame"] for v in runtime.pipeline.scan_records if "frame" in v)
        assert frame.scan_id==1 and frame.episode_id==0
        assert runtime.pipeline.cfg.split_max_forward_x_m==3.5
        root=to_np(frame.base_pos_w)
        quat=to_np(frame.base_quat_wxyz)
        rootrot=Rotation.from_quat(quat[[1,2,3,0]])
        mount=to_np(runtime.pipeline.mount_t).reshape(3)
        mount_q=to_np(runtime.pipeline.mount_q).reshape(4)
        sensorpos=frame.sensor_pos_w.detach().clone().reshape(1,3)
        sensorquat=frame.sensor_quat_xyzw.detach().clone().reshape(1,4)
        cfg=LidarConfig(sensor_type=LidarType.AIRY,dt=.02,update_frequency=10,
                        max_range=FLOOR_RANGE,min_range=.2,
                        num_sensors=1,horizontal_line_num=900,vertical_line_num=96,
                        airy_horizontal_resolution_deg=.4,return_pointcloud=True,
                        pointcloud_in_world_frame=False,randomize_placement=False,
                        enable_sensor_noise=False,pixel_dropout_prob=0.,
                        pixel_std_dev_multiplier=0.)
        sensor=LidarSensor({"device":device,"num_envs":1,"num_sensors":1,
                            "sensor_pos_tensor":sensorpos,"sensor_quat_tensor":sensorquat,
                            "mesh_ids":runtime.pipeline.mesh_ids},
                           None,cfg,num_sensors=1,device=device)
        directions=sensor.get_ray_directions()
        assert float((directions-frame.raw_ray_dirs).abs().max())<1e-6
        base_dirs=_quat_apply_xyzw(runtime.pipeline.mount_q.expand(86400,4),directions)
        alignment=base_dirs@runtime.pipeline.cone_axis_body
        if args.grid_preset == "s1_187":
            from hexapod_mjlab.tasks.confined_locomotion.ground_presets import s1_187_ground_grid
            configured=s1_187_ground_grid()
            eval_x=np.asarray(configured.x_coords)
            eval_y=np.asarray(configured.y_coords)
            report["selected_grid_identity"] = configured.identity()
            report["selected_grid_hash"] = configured.identity_hash()
        else:
            eval_x=np.linspace(*PROBE_X)
            eval_y=np.linspace(*PROBE_Y)
        xx,yy=np.meshgrid(eval_x,eval_y)
        xy=np.stack([xx.ravel(),yy.ravel()],axis=-1)
        pose_rows=[];vectors={}
        # Independent cone membership of the exact ground-query point, NOT
        # a nearest-neighbor proxy. This is a necessary geometric condition;
        # discrete AIRY support must still be assessed via true ray hits.
        cone_axis=to_np(runtime.pipeline.cone_axis_body).reshape(3)
        mount_point=to_np(runtime.pipeline.mount_t).reshape(3)
        cone_sin=float(np.sin(np.deg2rad(28)))
        in_cone_cases=[]
        plane_z=float(runtime.pipeline.plane_record["world_pos"][2])
        for pname,roll,pitch,yaw,dz,pclass in POSES:
            rot=rootrot*Rotation.from_euler("xyz",[roll,pitch,yaw],degrees=True)
            newroot=root.copy();newroot[2]+=dz
            spos=newroot+rot.apply(mount)
            sq=rot*Rotation.from_quat(mount_q)
            with torch.no_grad():
                sensorpos.copy_(torch.tensor(spos,dtype=torch.float32,device=device).view(1,3))
                sensorquat.copy_(torch.tensor(sq.as_quat(),dtype=torch.float32,device=device).view(1,4))
            cloud,dists=sensor.update()
            measured=to_np(dists).reshape(-1)
            expected=floor_distances(spos,sq.apply(to_np(directions)),plane_z)
            err=float(np.max(np.abs(measured-expected)))
            mismatches=int(np.count_nonzero((measured==FLOOR_RANGE)!=(expected==FLOOR_RANGE)))
            if mismatches or err>0.001:
                raise AssertionError(f"{pname} ground analytic mismatch count={mismatches}, maxerr={err}")
            raw_p=to_np(cloud).reshape(-1,3)
            miss=measured==FLOOR_RANGE
            raw_p[miss]=to_np(directions)[miss]*FLOOR_RANGE
            point_tensor=torch.as_tensor(raw_p,device=device,dtype=torch.float32)
            points=_quat_apply_xyzw(runtime.pipeline.mount_q.expand(86400,4),point_tensor)
            points+=runtime.pipeline.mount_t.expand(86400,3)
            near,far=partition_cone_forward_x(alignment,points,
                theta_deg=28.,forward_x_max_m=3.5)
            ni,n=farthest_point_sampling_indices(points,near,256)
            assert n==256
            p=to_np(points)
            group=to_np(near)
            inds=to_np(ni)
            pool_hits=p[group&~miss]
            fps_hits=p[inds[~miss[inds]]]
            if pname=="nominal":
                assert np.array_equal(group,to_np(frame.near_mask)), "nominal near group deviates from official scanner"
                assert len(pool_hits)==37412, "nominal B8 ground hits deviated from locked evidence"
            targets=get_probe_labels(env.sim.mj_model,xy,newroot,rot,
                                     probe_x=eval_x,probe_y=eval_y)
            query_rel=targets-mount_point[None,:]
            query_dist=np.linalg.norm(query_rel,axis=1)
            strict_geometry=(
                ((query_rel@cone_axis)/(query_dist+1e-12)>=cone_sin)
                & (targets[:,0]<=3.5+1e-8)
                & (query_dist<=FLOOR_RANGE)
            )
            in_cone_cases.append(strict_geometry)
            dist_pool=cKDTree(pool_hits).query(targets,k=1)[0]
            dist_fps=cKDTree(fps_hits).query(targets,k=1)[0]
            assert np.isfinite(dist_pool).all() and np.isfinite(dist_fps).all()
            assert np.all(dist_pool<=dist_fps+1e-5)
            row={
                "pose":pname,"class":pclass,"pitch_deg":pitch,"roll_deg":roll,
                "yaw_deg":yaw,"height_offset_m":dz,"near_hit_pool":len(pool_hits),
                "near_fps_hit":len(fps_hits),
                "max_distance_analytic_error_m":err,
                "analytic_hit_mask_mismatches":mismatches,
                "geometry_cone_membership_count":int(strict_geometry.sum()),
                "pool_nearest_m":dist_pool,
                "fps_nearest_m":dist_fps,
            }
            for radius in SUPPORT_THRESHOLDS_M:
                row[f"pool_support_{radius:.2f}m"]=round(float(np.mean(dist_pool<=radius)),4)
                row[f"fps_support_{radius:.2f}m"]=round(float(np.mean(dist_fps<=radius)),4)
            pose_rows.append(row)
            print(f"{pname}: pool={len(pool_hits):,}, FPS={len(fps_hits)}"
                  f", radius0.25m pool={row['pool_support_0.25m']:.3f}"
                  f", FPS={row['fps_support_0.25m']:.3f}",flush=True)
        heat=measure(pose_rows,xy,SUPPORT_THRESHOLDS_M)
        rectangles=[] if args.grid_preset is not None else recommend_rectangles(xy,heat)
        named={} if args.grid_preset is not None else named_region_comparison(xy,heat)
        strict_cone=np.asarray(in_cone_cases,dtype=bool)
        xprobe=xy[:,0];yprobe=np.abs(xy[:,1])
        named_masks={
            "S1_safe_rectangle":(
                (xprobe>=.6-1e-9)&(xprobe<=3.4+1e-9)&(yprobe<=.7+1e-9)
            ),
            "S2_expanding_trapezoid":(
                (xprobe>=.6-1e-9)&(xprobe<=3.4+1e-9)&(
                    yprobe<=np.interp(
                        xprobe,[.6,1.2,2.0,3.2,3.4],[.5,.9,1.2,1.4,1.4]
                    )+1e-9
                )
            ),
        }
        for name,mask in (named_masks.items() if args.grid_preset is None else ()):
            named[name]["strict_cone_nominal_fraction"]=float(
                strict_cone[0,mask].mean()
            )
            named[name]["strict_cone_routine_worst_fraction"]=float(
                np.mean(strict_cone[:9,mask],axis=1).min()
            )
            named[name]["strict_cone_all_15_pose_fraction"]=float(
                np.mean(strict_cone[:,mask])
            )
        if args.grid_preset is None:
            plot_heatmaps(output,xy,heat,rectangles)
        else:
            assert len(xy)==187
            assert bool(strict_cone[:9].all()), "S1 187 true query positions violate routine B8 geometry"
            assert bool(np.all(heat["fps"]["dist"][:9] <= 0.20 + 1e-8)), (
                "S1 187 expected true-floor near FPS support failed in routine poses"
            )
            report["routine_strict_cone_all_points"] = True
            report["routine_fps_all_187_nearest_0p20m"] = True
        report.update(
            state="completed",ray_count=86400,pose_checks=len(POSES),
            max_analytic_error_m=max(row["max_distance_analytic_error_m"] for row in pose_rows),
            input_identity={"grid_hash":runtime.grid_cfg.identity_hash(),
                            "scan_id":frame.scan_id,"episode_id":frame.episode_id},
            probe_count=int(len(xy)),
            coverage_overview={
                k:{
                    "pool_hard_all_count":int(heat["pool"]["hard_all"].sum()),
                    "fps_hard_all_count":int(heat["fps"]["hard_all"].sum()),
                    "pool_routine_80_count":int(heat["pool"]["routine_80"].sum()),
                    "fps_routine_80_count":int(heat["fps"]["routine_80"].sum()),
                } for k in ("summary",)
            },
            candidate_rectangles_top_10=rectangles[:10],
            named_shape_candidates=named,
            rows=[{k:v for k,v in row.items() if not k.endswith("_nearest_m")}
                  for row in pose_rows],
        )
        with (output/"pose_summary.csv").open("w",newline="") as fh:
            keys=list(report["rows"][0])
            cw=csv.DictWriter(fh,fieldnames=keys);cw.writeheader();cw.writerows(report["rows"])
        import numpy as np
        np.savez_compressed(
            output/"probes_support_matrix.npz",
            x=eval_x,y=eval_y,grid_xyz=xy,
            pose_names=np.array([row["pose"] for row in pose_rows],dtype=str),
            pool_distances_m=heat["pool"]["dist"],
            fps_distances_m=heat["fps"]["dist"],
            pool_routine_fraction=heat["pool"]["routine_fraction"],
            fps_routine_fraction=heat["fps"]["routine_fraction"],
            pool_all_fraction=heat["pool"]["all_fraction"],
            fps_all_fraction=heat["fps"]["all_fraction"],
            strict_ground_query_cone_membership=strict_cone,
        )
    except BaseException as err:
        report.update(state="failed",error=repr(err))
        raise
    finally:
        try:
            if runtime is not None:runtime.close()
        finally:
            if env is not None:env.close()
            (output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
            print(f"[B8 ground support] {report['state']}: {output/'report.json'}",flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
