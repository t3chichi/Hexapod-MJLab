#!/usr/bin/env python
"""Stage-A B8 pose and observation-perturbation gates (STATIC diagnostic).

One production T1 reset provides the calibrated lidar mount and 86,400 RAW
directions. For each prescribed rigid-body root pose, the real CUDA LidarSensor
raycasts a fixed diagnostic ground + finite wall mesh; independent analytical
intersections must match every raw range and hit mask. Production 256-point
metric FPS and 64-point angular far sampling are reused for A0 and B8.

The scene includes NO robot self-occlusion, walking dynamics or joint-motion
geometry. Pose changes are external rigid-root changes, NOT physics execution.
GroundGridCfg is a runtime placeholder and is not scored. Perturbation replay
uses the official P6 gap/range API on an actual nominal corridor raw frame, but
does not claim the modified B8 grouping has been installed into production.

Result files are diagnostic evidence, NOT a formal-training acceptance claim.
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
from compare_B8_corridors import (
    build_mesh,analytic,xy_cells,MAX_RANGE_M,
    WALL_X_START_M,WALL_X_END_M,
)

POSES=(
    ("nominal",0,0,0,0),
    ("pitch_up_2",0,2,0,0),
    ("pitch_up_4",0,4,0,0),
    ("pitch_up_6",0,6,0,0),
    ("pitch_up_8",0,8,0,0),
    ("pitch_up_10",0,10,0,0),
    ("pitch_down_10",0,-10,0,0),
    ("roll_left_10",10,0,0,0),
    ("roll_right_10",-10,0,0,0),
    ("yaw_left_12",0,0,12,0),
    ("yaw_right_12",0,0,-12,0),
    ("root_up_8cm",0,0,0,.08),
    ("root_down_8cm",0,0,0,-.08),
)
WIDTHS=(1.6,2.0)
CANDIDATES=(("A0",32.,None),("B8",28.,3.5))
RAYS=86400


def opts(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--gpu-id",type=int,default=0)
    p.add_argument("--seed",type=int,default=20261014)
    return p.parse_args(argv)


def array(t):
    return t.detach().cpu().numpy().copy()


def select(points,frame,pipeline,valid=None,*,direction_based_cone=False):
    """Near/far complete disjoint masks and production sampler source indices."""
    import torch
    from hexapod_mjlab.tasks.confined_locomotion.scanner import (
        farthest_point_sampling_indices,uniform_angular_sample_indices
    )
    raw=torch.as_tensor(points,device=pipeline.device,dtype=torch.float32)
    if valid is None:
        valid=torch.ones(RAYS,dtype=torch.bool,device=pipeline.device)
    if direction_based_cone:
        # P6 contract: angle membership comes from raw SENSOR unit directions
        # even when a noisy measurement saturates to zero range.
        from hexapod_mjlab.tasks.confined_locomotion.scanner import _quat_apply_xyzw
        dirs=_quat_apply_xyzw(pipeline.mount_q.expand(RAYS,4),frame.raw_ray_dirs)
        angle=dirs@pipeline.cone_axis_body
    else:
        r=raw-pipeline.mount_t[0]
        angle=(r@pipeline.cone_axis_body)/(torch.linalg.vector_norm(r,dim=-1)+1e-9)
    out={}
    for name,theta,xmax in CANDIDATES:
        inside=angle>=__import__("math").sin(__import__("math").radians(theta))
        if xmax is not None:
            inside=inside&(raw[:,0]<=xmax)
        near=valid&inside
        far=valid&~inside
        assert not bool((near&far).any()) and torch.equal(near|far,valid)
        ni,n=farthest_point_sampling_indices(raw,near,256)
        fi,f=uniform_angular_sample_indices(frame.pool_angle_keys,far,64)
        assert len(torch.unique(ni))==n and len(torch.unique(fi))==f
        out[name]=dict(near=array(near),far=array(far),
                       near_i=array(ni),far_i=array(fi),near_k=n,far_k=f)
    return out


def scene_records(points,classes,source_dist,frame,pipeline,width,pose):
    import numpy as np
    items=select(points,frame,pipeline)
    records=[]
    for name,s in items.items():
        near=s["near"]; ni=s["near_i"]; fi=s["far_i"]
        cls=np.asarray(classes)
        hit=(cls!=3)
        pts=points[near&hit]
        cells=xy_cells(pts)
        fpcells=xy_cells(points[ni[cls[ni]!=3]]) if len(ni) else set()
        def hist(labels,ids):
            return {key:int(np.count_nonzero(labels[ids]==val))
                    for key,val in (("floor",0),("left",1),("right",2),("miss",3))}
        near_counts=hist(cls,np.flatnonzero(near))
        fps_counts=hist(cls,ni)
        far_counts=hist(cls,fi)
        record={
            "scene":f"channel_{width:.1f}","pose":pose,"candidate":name,
            "near_count":int(near.sum()),"far_count":int(s["far"].sum()),
            "near_surface":near_counts,"fps256_surface":fps_counts,
            "far64_surface":far_counts,
            "fps_valid":s["near_k"],"far_valid":s["far_k"],
            "near_xy_cells":len(cells),"fps_xy_cells":len(fpcells),
            "cell_retention":len(fpcells)/len(cells) if cells else None,
            "both_walls_in_fps":fps_counts["left"]>0 and fps_counts["right"]>0,
        }
        records.append(record)
    return records


def perturbation_replay(points,source_dists,labels,frame,pipeline,seed):
    """P6 real gap clock + range streams, same nominal raw corridor scene.

    Reuse one clean frame 25 times as a diagnostic-only controlled experiment.
    It is NOT 25 new physical raycasts nor an implementation of P6 in B8.
    """
    import torch
    import numpy as np
    from hexapod_mjlab.common.perturbations import (
        ObservationPerturbations,ObservationPerturbationCfg,
        GapPerturbationCfg,RangePerturbationCfg,apply_range_noise
    )
    from hexapod_mjlab.tasks.confined_locomotion.scanner import _quat_apply_xyzw

    p=ObservationPerturbations(1,cfg=ObservationPerturbationCfg(
        gap=GapPerturbationCfg(),range=RangePerturbationCfg()
    ),seed=seed,device=pipeline.device)
    d=torch.as_tensor(source_dists,device=pipeline.device,dtype=torch.float32)
    immutable_raw=d.clone()
    miss=d==MAX_RANGE_M
    raw_dirs=frame.raw_ray_dirs
    sector=p.gap_ray_mask(raw_dirs)
    sector_count=int(sector.sum())
    # Actual raw-direction angle-sector membership need not equal the
    # historical azimuth-column interval. Record both source-ID masks;
    # no assumptions about the discrepancy's physical cause.
    print(f"[P6] actual production gap-sector rays: {sector_count} (nominal 7680)",flush=True)
    # Strict P6 membership is defined by gap_sector_mask, not by
    # the historical nominal count (currently 7681 versus 7680).
    assert 0<sector_count<RAYS
    # Contrast with the frozen historical column count. Do not change P6.
    legacy=torch.zeros(RAYS,dtype=torch.bool,device=pipeline.device)
    legacy[185*96:265*96]=True
    extra=(sector&~legacy).nonzero(as_tuple=False).flatten().cpu().tolist()
    missing=(~sector&legacy).nonzero(as_tuple=False).flatten().cpu().tolist()
    print(f"[P6] actual-vs-legacy sector: extra={len(extra)}, missing={len(missing)}",flush=True)
    base=select(points,frame,pipeline)["B8"]
    old_mask=base["near"]
    records=[];gap_sweeps=0
    for scan in range(1,26):
        fired=p.begin_scan(0,scan)
        enabled=(~sector) if fired else torch.ones_like(sector)
        eps=p.draw_range_epsilon(0,RAYS)
        consumed,applied=apply_range_noise(d,miss,enabled,eps)
        # Preserve raw ray direction, even for a zero-distance noisy hit.
        consumed_sensor=consumed[:,None]*raw_dirs
        pts=_quat_apply_xyzw(pipeline.mount_q.expand(RAYS,4),consumed_sensor)
        pts+=pipeline.mount_t.expand(RAYS,3)
        selected=select(array(pts),frame,pipeline,enabled,
                        direction_based_cone=True)["B8"]
        clean_near=old_mask & array(enabled)
        changed=int(np.count_nonzero(selected["near"]!=clean_near))
        if fired:gap_sweeps+=1
        nohit_identity=bool(torch.equal(consumed[miss],d[miss]))
        raw_unchanged=bool(torch.equal(d,immutable_raw))
        assert nohit_identity and raw_unchanged
        assert np.all(selected["near"]<=array(enabled))
        assert selected["near_k"]==256 and selected["far_k"]==64
        assert len(set(selected["near_i"].tolist()))==256
        assert len(set(selected["far_i"].tolist()))==64
        assert not np.intersect1d(selected["near_i"],selected["far_i"]).size
        assert len(set(selected["near_i"]).intersection(
            set(np.flatnonzero(~array(enabled)).tolist())))==0
        label=np.asarray(labels)
        fps_wall_left=int((label[selected["near_i"]]==1).sum())
        fps_wall_right=int((label[selected["near_i"]]==2).sum())
        margin=np.abs(points[:,0]-3.5)<=0.05
        rec={
            "scan":scan,"gap_fired":bool(fired),"excluded_rays":int((~enabled).sum()),
            "range_noisy_hits":int(applied.sum()),
            "near_group_changed_vs_clean_valid":changed,
            "near_group_changed_in_xmax_5cm_band":int(np.count_nonzero(
                (selected["near"]!=clean_near)&margin)),
            "near_candidates":int(selected["near"].sum()),
            "far_candidates":int(selected["far"].sum()),
            "near_fps_left_wall":fps_wall_left,
            "near_fps_right_wall":fps_wall_right,
            "far_selected_real_hits":int((label[selected["far_i"]]!=3).sum()),
            "source_valid_exhaustive":int(np.count_nonzero(
                selected["near"]|selected["far"]))==int(enabled.sum()),
        }
        assert rec["source_valid_exhaustive"]
        records.append(rec)
    return records,dict(
        gap_scans=gap_sweeps,sector_rays=sector_count,
        nominal_sector_rays=7680,
        nominal_sector_count_matches_actual=(sector_count==7680),
        vs_nominal_column_interval={
            "extra_count":len(extra),"missing_count":len(missing),
            "extra_raw_ids_first20":extra[:20],"missing_raw_ids_first20":missing[:20],
        },
        schedule=p.gap_schedule_record(),stream_draws=p.draw_counts()
    )


def plot_summary(out,records,noise_records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,2,figsize=(17,5.8),layout="constrained")
    poses=[p[0] for p in POSES]
    w=.37
    for k,width in enumerate(WIDTHS):
        sub=[r for r in records if r["scene"]==f"channel_{width:.1f}"]
        x=np.arange(len(poses))
        a=[next(r for r in sub if r["pose"]==p and r["candidate"]=="A0") for p in poses]
        b=[next(r for r in sub if r["pose"]==p and r["candidate"]=="B8") for p in poses]
        valsA=[r["fps256_surface"]["left"]+r["fps256_surface"]["right"] for r in a]
        valsB=[r["fps256_surface"]["left"]+r["fps256_surface"]["right"] for r in b]
        ax=axes[k]
        ax.bar(x-w/2,valsA,width=w,label="A0 original cone",color="#879aaf")
        ax.bar(x+w/2,valsB,width=w,label="B8 28/56 + x<=3.5m",color="#2e77a3")
        ax.set_xticks(x,poses,rotation=50,ha="right",fontsize=8)
        ax.set(ylabel="Wall points in FPS256",title=f"Static corridor width={width:.1f} m")
        ax.grid(axis="y",alpha=.2);ax.legend(fontsize=9)
    fig.suptitle("Real CUDA raycasts across rigid robot/sensor poses; no joint-body self-occlusion")
    fig.savefig(out/"pose_wall_fps_comparison.png",dpi=170)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(10,4.8),layout="constrained")
    xs=[r["scan"] for r in noise_records]
    ax.bar(xs,[r["near_group_changed_vs_clean_valid"] for r in noise_records],
           color=["#d88f37" if r["gap_fired"] else "#4778ad" for r in noise_records])
    ax.set(xlabel="Persistent scan number (same static raw frame replayed)",
           ylabel="B8 group membership changes",
           title="P6 range noise and scheduled gap impact on B8")
    ax.grid(axis="y",alpha=.2)
    fig.savefig(out/"perturbation_group_changes.png",dpi=160)
    plt.close(fig)


def main(argv=None):
    import numpy as np
    import torch
    import warp as wp
    from scipy.spatial.transform import Rotation
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
    from hexapod_mjlab.tasks.confined_locomotion.flat_env_cfg import make_training_env_cfg
    from hexapod_mjlab.tasks.confined_locomotion.ground import (
        GroundGridCfg,VALIDITY_GEOMETRY_ONLY_FLAT_TRAINING
    )
    from hexapod_mjlab.tasks.confined_locomotion.runtime import TrainingRuntime
    from hexapod_mjlab.tasks.confined_locomotion.scanner import (
        ConfinedLidarCfg,_quat_apply_xyzw
    )
    from hexapod_mjlab.utils.lidar import LidarSensor,LidarConfig,LidarType

    args=opts(argv)
    if args.gpu_id<0 or not torch.cuda.is_available() or args.gpu_id>=torch.cuda.device_count():
        raise RuntimeError("explicit CUDA device required")
    out=args.output.expanduser().resolve()
    out.mkdir(parents=True,exist_ok=False)
    paths=(Path(__file__),
           ROOT/"scripts/experiments/confined_locomotion/compare_B8_corridors.py",
           ROOT/"src/hexapod_mjlab/tasks/confined_locomotion/scanner.py",
           ROOT/"src/hexapod_mjlab/tasks/confined_locomotion/perturbations.py",
           ROOT/"src/hexapod_mjlab/utils/lidar/lidar_sensor.py")
    report={
        "schema":"confined_locomotion.B8_pose_perturb/v1",
        "state":"started",
        "created_utc":datetime.now(timezone.utc).isoformat(),
        "gpu":args.gpu_id,"seed":args.seed,
        "selected_candidate":dict(theta_deg=28,tilt_deg=56,x_max_body_m=3.5),
        "reference":dict(theta_deg=32,tilt_deg=56,x_max_body_m=None),
        "rigid_poses":POSES,
        "widths_m":WIDTHS,
        "grid":"temporary TrainingRuntime fixture, excluded from metrics",
        "scope":"STATIC root/sensor transformed true CUDA raycasts; P6 range+gap offline controlled repeated-source replay",
        "limitations":"No robot geometry in raycast, no joint poses, no locomotion, no temporal scanning motion distortion, no formal scanner B8 adoption",
        "source_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    }
    (out/"recipe.json").write_text(json.dumps(report,indent=2,default=str))
    env=runtime=None
    try:
        torch.cuda.set_device(args.gpu_id)
        device=f"cuda:{args.gpu_id}"
        grid=GroundGridCfg(x_coords=(-.8,-.4,0,.4,.8,1.2,1.6,2.),
                           y_coords=(-.8,-.4,0,.4,.8),
                           validity_mode=VALIDITY_GEOMETRY_ONLY_FLAT_TRAINING)
        env=ManagerBasedRlEnv(make_training_env_cfg(num_envs=1,seed=args.seed),device=device)
        vec=RslRlVecEnvWrapper(env,clip_actions=None)
        runtime=TrainingRuntime(vec,grid_cfg=grid,seed=args.seed,
                                output_dir=out/"runtime",mode="engineering",
                                lidar_cfg=ConfinedLidarCfg(),perturbation_cfg=None)
        scans=[r["frame"] for r in runtime.pipeline.scan_records if "frame" in r]
        assert len(scans)==1
        frame=scans[0]
        root=array(frame.base_pos_w)
        quat=array(frame.base_quat_wxyz)
        rootRot=Rotation.from_quat(quat[[1,2,3,0]])
        initial_yaw=rootRot.as_euler("zyx")[0]
        mount_pos=np.asarray(runtime.pipeline.mount_pos,dtype=float)
        mount_q=np.asarray(runtime.pipeline.mount_quat_xyzw,dtype=float)
        pose_pos=frame.sensor_pos_w.detach().clone().reshape(1,3)
        pose_q=frame.sensor_quat_xyzw.detach().clone().reshape(1,4)
        p_cfg=LidarConfig(sensor_type=LidarType.AIRY,dt=.02,update_frequency=10.,
                         max_range=MAX_RANGE_M,min_range=.2,
                         num_sensors=1,horizontal_line_num=900,vertical_line_num=96,
                         airy_horizontal_resolution_deg=.4,return_pointcloud=True,
                         pointcloud_in_world_frame=False,randomize_placement=False,
                         enable_sensor_noise=False,pixel_dropout_prob=0,
                         pixel_std_dev_multiplier=0)
        results=[]
        checks=[]
        replay_input=None
        wp.init()
        for width in WIDTHS:
            with wp.ScopedDevice(device):
                owns=build_mesh(wp,device,root[:2],initial_yaw,width)
                sensor=LidarSensor({
                    "device":device,"num_envs":1,"num_sensors":1,
                    "sensor_pos_tensor":pose_pos,"sensor_quat_tensor":pose_q,
                    "mesh_ids":owns["ids"]
                },None,p_cfg,num_sensors=1,device=device)
                pattern=sensor.get_ray_directions()
                assert float((pattern-frame.raw_ray_dirs).abs().max())<1e-6
                for pname,roll,pitch,yaw,dz in POSES:
                    # R_WB=R_WB0 * R_Bdelta, preserve original BASE->sensor mount.
                    rot=rootRot*Rotation.from_euler("xyz",[roll,pitch,yaw],degrees=True)
                    root_new=root.copy()
                    root_new[2]+=dz
                    sensor_pos=root_new+rot.apply(mount_pos)
                    sensor_rot=rot*Rotation.from_quat(mount_q)
                    with torch.no_grad():
                        pose_pos.copy_(torch.tensor(sensor_pos,dtype=torch.float32,device=device).view(1,3))
                        pose_q.copy_(torch.tensor(sensor_rot.as_quat(),dtype=torch.float32,device=device).view(1,4))
                    cloud,distance=sensor.update()
                    dist=array(distance).reshape(-1)
                    local=array(cloud).reshape(-1,3)
                    world_dirs=sensor_rot.apply(array(pattern))
                    truth,surface=analytic(sensor_pos,world_dirs,root[:2],initial_yaw,width)
                    error=float(np.max(np.abs(dist-truth)))
                    mismatch=int(np.count_nonzero((dist==MAX_RANGE_M)!=(truth==MAX_RANGE_M)))
                    print(f"{width:.1f}/{pname} check: maxdist={error:.6g}, mismatch={mismatch}",flush=True)
                    if mismatch or error>0.001:
                        raise AssertionError(f"independent raycheck {width}/{pname} failed")
                    # Reconstruct canonical miss representation using original ray
                    # direction; transform by the exact task mount into local BASE.
                    p=torch.tensor(local,device=device,dtype=torch.float32)
                    miss=torch.as_tensor(dist==MAX_RANGE_M,device=device)
                    p=torch.where(miss[:,None],frame.raw_ray_dirs*MAX_RANGE_M,p)
                    pb=_quat_apply_xyzw(runtime.pipeline.mount_q.expand(RAYS,4),p)
                    pb+=runtime.pipeline.mount_t.expand(RAYS,3)
                    pts=array(pb)
                    records=scene_records(pts,surface,dist,frame,runtime.pipeline,width,pname)
                    results.extend(records)
                    checks.append(dict(width_m=width,pose=pname,
                                       hit_mismatch=mismatch,max_distance_error_m=error))
                    if pname=="nominal" and width==1.6:
                        replay_input=(pts,dist,surface)
                        b8=next(r for r in records if r["candidate"]=="B8")
                        # Cross-run frozen sanity invariant for nominal pose.
                        assert b8["fps256_surface"]["left"]+b8["fps256_surface"]["right"]>=45
        assert replay_input is not None
        noise_records,noise_meta=perturbation_replay(*replay_input,frame,runtime.pipeline,args.seed)
        plot_summary(out,results,noise_records)
        with (out/"pose_summary.csv").open("w",newline="") as fh:
            fields=("scene","pose","candidate","near_wall_pool","fps_wall_left",
                    "fps_wall_right","fps_floor","fps_misses","far_wall_left",
                    "far_wall_right","retained_cells","occupied_cells")
            wr=csv.DictWriter(fh,fieldnames=fields);wr.writeheader()
            for r in results:
                n=r["near_surface"];f=r["fps256_surface"];z=r["far64_surface"]
                wr.writerow(dict(
                    scene=r["scene"],pose=r["pose"],candidate=r["candidate"],
                    near_wall_pool=n["left"]+n["right"],
                    fps_wall_left=f["left"],fps_wall_right=f["right"],
                    fps_floor=f["floor"],fps_misses=f["miss"],
                    far_wall_left=z["left"],far_wall_right=z["right"],
                    retained_cells=r["fps_xy_cells"],occupied_cells=r["near_xy_cells"]
                ))
        with (out/"perturbation_summary.csv").open("w",newline="") as fh:
            fields=list(noise_records[0])
            wr=csv.DictWriter(fh,fieldnames=fields);wr.writeheader();wr.writerows(noise_records)
        report.update(state="completed",checks=checks,measurements=results,
                      perturbation_records=noise_records,perturbation_metadata=noise_meta)
    except BaseException as err:
        report.update(state="failed",error=repr(err))
        raise
    finally:
        try:
            if runtime is not None:runtime.close()
        finally:
            if env is not None:env.close()
            (out/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str))
            print(f"[B8 pose] {report['state']} {out/'report.json'}",flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
