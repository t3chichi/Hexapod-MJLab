#!/usr/bin/env python
"""Static corridor wall-sampling check using REAL CUDA LidarSensor raycasts.

This diagnostic does not alter production scanner, physical MuJoCo environment,
runtime, or PPO. It takes the real T1 reset's lidar pose + all 86,400 original
sensor-ray directions, constructs independent Warp meshes of the physical
floor and *diagnostic* corridor walls, and raycasts those meshes with the same
LidarSensor backend. It then reapplies EXACT production near FPS and far64
sampling to candidate grouping rules. The historical candidate IDs A0/B8/B9
are retained in report schemas and filenames; B8 denotes the selected widened
near-field grouping (theta=28, tilt=56, observed BASE x <= 3.5m).
Ground supervision grid is an unchanged placeholder solely to instantiate the
official TrainingRuntime, never used for scoring. This first pass uses exactly
one settled robot pose per scene; no robot self-occlusion, collision physics,
training, movement or deployment claims.

The independent plane/vertical-wall analytic intersections must match every
raycast distance AND ray hit/miss status or the experiment fails.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"src"))

# World geometry is a finite diagnostic surface, not a collision-enabled task.
CHANNEL_WIDTHS_M=(None,1.6,2.0)
WALL_X_START_M=-2.0
WALL_X_END_M=6.0
WALL_HEIGHT_M=4.0
MAX_RANGE_M=15.0
CANDIDATES=(("A0",32.0,None),("B8",28.0,3.5),("B9",28.0,4.0))
X_SEGMENTS=((0.0,1.0),(1.0,2.0),(2.0,3.5),(3.5,6.0))
ROOT_CELLS_X=(0.5,3.5)
ROOT_CELLS_Y=(-1.6,1.6)
CELL_SIZE_M=0.20


def _args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True,help="new, absent evidence dir")
    p.add_argument("--seed",type=int,default=20261014)
    p.add_argument("--gpu-id",type=int,default=0)
    return p.parse_args(argv)


def npcopy(t):
    return t.detach().cpu().numpy().copy()


def build_mesh(wp,device,root_xy,yaw,width):
    """Double-sided ray cast diagnostic surfaces: floor + two tall side walls.

    Mesh coords are world; corridor center line follows initial robot yaw.
    The mesh arrays must be kept alive for warp's Mesh device pointers.
    """
    import numpy as np
    verts=[]
    tri=[]
    def quad(p):
        k=len(verts)
        verts.extend(p)
        tri.extend(((k,k+1,k+2),(k,k+2,k+3)))
    x0,y0=root_xy
    def world(x,y,z):
        return [x0+np.cos(yaw)*x-np.sin(yaw)*y,
                y0+np.sin(yaw)*x+np.cos(yaw)*y,z]
    # The finite floor covers every reachable 15m ray for this one reset pose.
    quad([world(-32,-32,0),world(32,-32,0),
          world(32,32,0),world(-32,32,0)])
    if width is not None:
        for yy in (-width/2,width/2):
            quad([world(WALL_X_START_M,yy,0),
                  world(WALL_X_END_M,yy,0),
                  world(WALL_X_END_M,yy,WALL_HEIGHT_M),
                  world(WALL_X_START_M,yy,WALL_HEIGHT_M)])
    points=wp.array(np.asarray(verts,dtype=np.float32),dtype=wp.vec3,device=device)
    faces=wp.array(np.asarray(tri,dtype=np.int32).reshape(-1),dtype=wp.int32,device=device)
    mesh=wp.Mesh(points=points,indices=faces)
    meshids=wp.array([mesh.id],dtype=wp.uint64,device=device)
    return dict(points=points,faces=faces,mesh=mesh,ids=meshids,vert_count=len(verts))


def analytic(origin,world_dirs,root_xy,yaw,width):
    """Independent intersect all rays with floor z=0 and finite vertical walls."""
    import numpy as np
    o=np.asarray(origin,dtype=np.float64)
    d=np.asarray(world_dirs,dtype=np.float64)
    # compute in yaw-aligned ROOT xy; z remains world vertical
    c,s=np.cos(yaw),np.sin(yaw)
    v=o[:2]-np.asarray(root_xy,dtype=np.float64)
    ox=c*v[0]+s*v[1]
    oy=-s*v[0]+c*v[1]
    dx=c*d[:,0]+s*d[:,1]
    dy=-s*d[:,0]+c*d[:,1]
    dist=np.full((3,len(d)),np.inf,dtype=np.float64)
    good_floor=np.abs(d[:,2])>1e-10
    np.divide(-o[2],d[:,2],out=dist[0],where=good_floor)
    dist[0]=np.where(good_floor & (dist[0]>=0)&(dist[0]<MAX_RANGE_M),
                     dist[0],np.inf)
    if width is not None:
        # BASE +y is robot LEFT. Label 1=+y(left), 2=-y(right).
        for ix,yy in enumerate((width/2,-width/2),start=1):
            t=np.full(len(d),np.inf)
            usable=np.abs(dy)>1e-10
            np.divide(yy-oy,dy,out=t,where=usable)
            x=ox+t*dx
            z=o[2]+t*d[:,2]
            valid=usable&(t>=0)&(t<MAX_RANGE_M)&(x>=WALL_X_START_M)&(x<=WALL_X_END_M)&(z>=0)&(z<=WALL_HEIGHT_M)
            dist[ix]=np.where(valid,t,np.inf)
    surf=dist.argmin(axis=0)
    best=dist.min(axis=0)
    miss=~np.isfinite(best)
    surf[miss]=3
    best[miss]=MAX_RANGE_M
    return best.astype(np.float32),surf.astype(np.int8)


def xy_cells(points):
    import numpy as np
    if not len(points):
        return set()
    xmin,xmax=ROOT_CELLS_X
    ymin,ymax=ROOT_CELLS_Y
    q=points[(points[:,0]>=xmin)&(points[:,0]<xmax)&(points[:,1]>=ymin)&(points[:,1]<ymax)]
    cell=np.floor((q[:,:2]-[xmin,ymin])/CELL_SIZE_M).astype(int)
    return set(map(tuple,cell.tolist()))


def repartition(scene_id,frame,pts_body,source_ranges,surfaces,pipeline):
    """No synthetic/downsampled input: entire raw 86,400 sample with source IDs."""
    import numpy as np
    import torch
    from hexapod_mjlab.tasks.confined_locomotion.scanner import (
        farthest_point_sampling_indices,uniform_angular_sample_indices
    )

    points=torch.as_tensor(pts_body,device=frame.pool_points_base.device,dtype=torch.float32)
    dist=torch.as_tensor(source_ranges,device=points.device)
    miss=dist==MAX_RANGE_M
    if points.shape != (86400,3):
        raise AssertionError("unexpected source ray count")
    a=pipeline.cone_axis_body
    rel=points-pipeline.mount_t[0]
    cos=(rel@a)/(torch.linalg.vector_norm(rel,dim=-1)+1e-9)
    angular=frame.pool_angle_keys
    original_miss=npcopy(frame.no_hit_mask)
    if scene_id=="floor":
        assert np.array_equal(npcopy(miss),original_miss),"floor hit/miss parity failed"
        assert float(np.max(np.abs(pts_body-npcopy(frame.raw_points_base)))) < 0.002,"floor endpoint parity failed"
    result=[]
    figure={}
    source_surfaces=np.asarray(surfaces)
    for name,theta,xmax in CANDIDATES:
        cone=cos>=float(np.sin(np.deg2rad(theta)))
        near=cone if xmax is None else cone&(points[:,0]<=xmax)
        far=~near
        assert not bool((near&far).any()) and bool((near|far).all())
        near_i,n=farthest_point_sampling_indices(points,near,256)
        far_i,k=uniform_angular_sample_indices(angular,far,64)
        assert (n,k)==(256,64)
        floor_baseline_overlap=None
        if scene_id=="floor" and name=="A0":
            assert torch.equal(near,frame.near_mask),"current production cone parity failed"
            # A separate Warp raycast differs by ~1e-4 m at some endpoints;
            # metric FPS has near-ties and need not select the exact same
            # source indices for numerically equivalent geometry. Record
            # the overlap instead of declaring the second sensor defective.
            original=set(npcopy(frame.near_indices).tolist())
            selected=set(npcopy(near_i).tolist())
            floor_baseline_overlap=len(original & selected)
            assert torch.equal(far_i,frame.far_indices),"current far64 parity failed"
        m=npcopy(near)
        mi=npcopy(near_i)
        fi=npcopy(far_i)
        total={str(label):int(np.sum(source_surfaces[m]==label)) for label in range(4)}
        fps={str(label):int(np.sum(source_surfaces[mi]==label)) for label in range(4)}
        farcount={str(label):int(np.sum(source_surfaces[fi]==label)) for label in range(4)}
        wall_source=source_surfaces[m]
        near_points=pts_body[m]
        wall_points=near_points[(wall_source==1)|(wall_source==2)]
        fps_wall=pts_body[mi[(source_surfaces[mi]==1)|(source_surfaces[mi]==2)]]
        fps_near_real=pts_body[mi[source_surfaces[mi]!=3]]
        near_real=near_points[wall_source!=3]
        all_cells=xy_cells(near_real)
        fps_cells=xy_cells(fps_near_real)
        segs=[]
        for xlo,xhi in X_SEGMENTS:
            block=(near_points[:,0]>=xlo)&(near_points[:,0]<xhi)
            mask_l=wall_source==1
            mask_r=wall_source==2
            fpspts=pts_body[mi]
            fsl=source_surfaces[mi]
            fseg=(fpspts[:,0]>=xlo)&(fpspts[:,0]<xhi)
            segs.append({
                "x_range_m":[xlo,xhi],
                "near_wall_left_hits":int(np.sum(block&mask_l)),
                "near_wall_right_hits":int(np.sum(block&mask_r)),
                "fps_wall_left_hits":int(np.sum(fseg&(fsl==1))),
                "fps_wall_right_hits":int(np.sum(fseg&(fsl==2))),
            })
        record={
            "scene":scene_id,"candidate":name,"theta_deg":theta,"xmax_m":xmax,
            "near_candidates":int(m.sum()),"far_candidates":int((~m).sum()),
            "independent_floor_fps_source_overlap_256":floor_baseline_overlap,
            "near_surface_counts":total,
            "fps256_surface_counts":fps,
            "far64_surface_counts":farcount,
            "near_real_hit_xy_cells":len(all_cells),
            "fps_real_hit_xy_cells":len(fps_cells),
            "fps_cell_retention":(len(fps_cells)/len(all_cells)) if all_cells else None,
            "wall_segments":segs,
            "near_wall_hit_x_min_m":float(wall_points[:,0].min()) if len(wall_points) else None,
            "near_wall_hit_x_max_m":float(wall_points[:,0].max()) if len(wall_points) else None,
            "max_observed_near_x_m":float(near_real[:,0].max()) if len(near_real) else None,
        }
        assert (sum(total.values())==int(m.sum())
                and sum(fps.values())==256 and sum(farcount.values())==64)
        result.append(record)
        figure[name]={
            "pts":near_points,"labels":wall_source,"fps_pts":pts_body[mi],
            "fps_labels":source_surfaces[mi],
        }
        print(f"{scene_id} {name}: near walls={total['1']}/{total['2']}"
              f" fps walls={fps['1']}/{fps['2']} far walls={farcount['1']}/{farcount['2']}"
              f" retained={len(fps_cells)}/{len(all_cells)}",flush=True)
    return result,figure


def render(out,scenes,results,data,widths,origin_xy):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    # Nominal reference morphology is a geometry marker, NOT physical robot mesh.
    hexagon=np.array([[-.89,-.59],[0,-.69],[.89,-.59],
                      [.89,.59],[0,.69],[-.89,.59]])
    fig,axs=plt.subplots(len(scenes),len(CANDIDATES),figsize=(16,11),layout="constrained")
    for row,scene in enumerate(scenes):
        for col,(name,_,xmax) in enumerate(CANDIDATES):
            ax=axs[row,col]
            v=data[scene][name]
            labels=v["labels"];p=v["pts"]
            for code,color in ((0,"#adb4bd"),(1,"#2079ba"),(2,"#dc7134")):
                a=p[labels==code]
                stride=max(1,len(a)//4000)
                a=a[::stride]
                if len(a):
                    ax.scatter(a[:,0],a[:,1],s=(0.9 if code==0 else 1.7),
                               c=color,alpha=(0.18 if code==0 else 0.42),rasterized=True)
            q=v["fps_pts"];k=v["fps_labels"]
            for code,c in ((0,"#4c566a"),(1,"#184b91"),(2,"#b5481b")):
                s=q[k==code]
                if len(s): ax.scatter(s[:,0],s[:,1],s=9,c=c,alpha=.7)
            ax.add_patch(Polygon(hexagon,fill=False,edgecolor="#333",ls="--",lw=1.3))
            ax.scatter([0],[0],color="black",marker="+",s=55)
            ax.scatter([.645],[0],color="#efb820",s=35,edgecolors="white")
            if widths[scene] is not None:
                h=widths[scene]/2
                for yy in (-h,h):
                    ax.plot([WALL_X_START_M,WALL_X_END_M],[yy,yy],
                            c="#802a7a",lw=1.5,alpha=.9)
            if xmax is not None:
                ax.axvline(xmax,color="#b34c3e",ls="--",lw=1.2)
            ax.set(xlim=(-1.0,4.6),ylim=(-1.4,1.4))
            ax.set_aspect("equal",adjustable="box")
            ax.grid(alpha=.14)
            ax.set_xlabel("BASE x [m]")
            if col==0: ax.set_ylabel("BASE y [m]")
            hits=results[(scene,name)]["fps256_surface_counts"]
            ax.set_title(f"{scene} / {name}: FPS left={hits['1']}, right={hits['2']}",fontsize=10.4)
    fig.suptitle(
        "Independent real LidarSensor raycasts: floor and finite corridor wall meshes\n"
        "A0 = current production cone; B8 = 28/56 + x<=3.5m; B9 = 28/56 + x<=4m\n"
        "Light gray=floor pool hits, blue/orange=wall pool hits; darker dots=FPS256. "
        "Dashed hex=nominal envelope only (not robot mesh).",fontsize=11.8
    )
    fig.savefig(out/"corridors_B8_A0_B9_bev.png",dpi=160)
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
    from hexapod_mjlab.tasks.confined_locomotion.scanner import ConfinedLidarCfg,_quat_apply_xyzw
    from hexapod_mjlab.utils.lidar import LidarSensor,LidarConfig,LidarType

    args=_args(argv)
    if args.gpu_id<0 or not torch.cuda.is_available() or args.gpu_id>=torch.cuda.device_count():
        raise RuntimeError("real chosen CUDA device required")
    out=args.output.expanduser().resolve()
    out.mkdir(parents=True,exist_ok=False)
    devicestr=f"cuda:{args.gpu_id}"
    paths=[Path(__file__),ROOT/"src/hexapod_mjlab/tasks/confined_locomotion/scanner.py",
           ROOT/"src/hexapod_mjlab/utils/lidar/lidar_sensor.py",
           ROOT/"src/hexapod_mjlab/utils/lidar/sensor_kernels/lidar_kernels_warp.py",
           ROOT/"src/hexapod_mjlab/tasks/confined_locomotion/runtime.py"]
    report={
        "schema":"confined_locomotion.B8_corridor_raycheck/v1",
        "state":"started","created_utc":datetime.now(timezone.utc).isoformat(),
        "seed":args.seed,"gpu":devicestr,
        "near_candidate_baseline":{
            "id":"B8","theta_deg":28.0,"tilt_deg":56.0,"x_max_body_m":3.5,
            "rule":"near=cone(axis,ray)<=62deg half angle AND observed endpoint BASE-x<=3.5m",
            "status":"user-selected candidate, pending obstacle validation; production scanner unchanged"
        },
        "scenes_m":[{"scene":"floor","channel_width_m":None},
                    {"scene":"channel_1.6","channel_width_m":1.6},
                    {"scene":"channel_2.0","channel_width_m":2.0}],
        "wall_geometry":{
            "start_x_body_m":WALL_X_START_M,"end_x_body_m":WALL_X_END_M,
            "height_world_m":WALL_HEIGHT_M,
            "alignment":"world yaw aligned with T1 BASE at scanning time"
        },
        "candidate_names":[n for n,_,_ in CANDIDATES],
        "ground_grid":"TEMPORARY runtime fixture ONLY: not used in any score/plot",
        "limitations":"static clean one T1 pose, diagnostic non-colliding walls, no robot-self-occlusion, no time/dynamic passes, no real radar test",
        "source_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    }
    (out/"recipe.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    env=runtime=None
    try:
        torch.cuda.set_device(args.gpu_id)
        grid=GroundGridCfg(x_coords=(-.8,-.4,0,.4,.8,1.2,1.6,2.0),
                           y_coords=(-.8,-.4,0,.4,.8),
                           validity_mode=VALIDITY_GEOMETRY_ONLY_FLAT_TRAINING)
        env=ManagerBasedRlEnv(make_training_env_cfg(num_envs=1,seed=args.seed),device=devicestr)
        wrapper=RslRlVecEnvWrapper(env,clip_actions=None)
        runtime=TrainingRuntime(wrapper,grid_cfg=grid,seed=args.seed,
                                output_dir=out/"runtime",mode="engineering",
                                lidar_cfg=ConfinedLidarCfg(),perturbation_cfg=None)
        scans=[x["frame"] for x in runtime.pipeline.scan_records if "frame" in x]
        assert len(scans)==1 and scans[0].scan_id==1 and scans[0].episode_id==0
        frame=scans[0]
        origin=npcopy(frame.sensor_pos_w).reshape(3)
        root=npcopy(frame.base_pos_w).reshape(3)
        quatroot=npcopy(frame.base_quat_wxyz).reshape(4)
        yaw=Rotation.from_quat([quatroot[1],quatroot[2],quatroot[3],quatroot[0]]).as_euler("zyx")[0]
        directions=npcopy(frame.raw_ray_dirs)
        quat_sensor=npcopy(frame.sensor_quat_xyzw).reshape(4)
        dirs_world=Rotation.from_quat(quat_sensor).apply(directions)
        pos_tensor=frame.sensor_pos_w.detach().clone().reshape(1,3)
        quat_tensor=frame.sensor_quat_xyzw.detach().clone().reshape(1,4)
        cfg=LidarConfig(sensor_type=LidarType.AIRY,dt=0.02,update_frequency=10.0,
                        max_range=MAX_RANGE_M,min_range=0.2,
                        num_sensors=1,horizontal_line_num=900,vertical_line_num=96,
                        airy_horizontal_resolution_deg=0.4,return_pointcloud=True,
                        pointcloud_in_world_frame=False,randomize_placement=False,
                        enable_sensor_noise=False,pixel_dropout_prob=0.0,pixel_std_dev_multiplier=0.0)
        records=[];figures={};summary={}
        wp.init()
        for width in CHANNEL_WIDTHS_M:
            scene="floor" if width is None else f"channel_{width:.1f}"
            with wp.ScopedDevice(devicestr):
                own=build_mesh(wp,devicestr,root[:2],yaw,width)
                sensor=LidarSensor({
                    "device":devicestr,"num_envs":1,"num_sensors":1,
                    "sensor_pos_tensor":pos_tensor,"sensor_quat_tensor":quat_tensor,
                    "mesh_ids":own["ids"],
                },None,cfg,num_sensors=1,device=devicestr)
                raydirs=sensor.get_ray_directions()
                diff=float((raydirs-frame.raw_ray_dirs).abs().max())
                if diff>1e-6:
                    raise AssertionError(f"sensor pattern changed from production: {diff}")
                cloud,dists=sensor.update()
            distances=npcopy(dists).reshape(-1)
            rawpts=npcopy(cloud).reshape(-1,3)
            expect,surfaces=analytic(origin,dirs_world,root[:2],yaw,width)
            actualmiss=distances==MAX_RANGE_M
            expectedmiss=expect==MAX_RANGE_M
            bad=int(np.count_nonzero(actualmiss!=expectedmiss))
            error=float(np.max(np.abs(distances-expect)))
            print(f"{scene}: ray analytic vs actual max error={error:.6g}m, miss mismatch={bad}",flush=True)
            if bad or error>0.001:
                raise AssertionError(f"{scene}: analytic check failed ({bad} miss differences, {error:.6g}m max)")
            if scene=="floor":
                assert np.array_equal(npcopy(frame.no_hit_mask),actualmiss)
                err=float(np.max(np.abs(distances-npcopy(frame.raw_dists))))
                assert err<=0.001,f"production floor distance parity exceeded: {err}"
            endpoints=torch.as_tensor(rawpts,dtype=torch.float32,device=runtime.device)
            true_dirs=frame.raw_ray_dirs
            endpoints=torch.where(
                torch.as_tensor(actualmiss,device=runtime.device)[:,None],
                true_dirs*MAX_RANGE_M,endpoints
            )
            pbase=_quat_apply_xyzw(runtime.pipeline.mount_q.expand(86400,4),endpoints)
            pbase+=runtime.pipeline.mount_t.expand(86400,3)
            records_scene,data_scene=repartition(
                scene,frame,npcopy(pbase),distances,surfaces,runtime.pipeline
            )
            records.extend(records_scene)
            figures[scene]=data_scene
            summary[scene]={
                "channel_width_m":width,
                "raw_hit_labels":{
                    "floor":int((surfaces==0).sum()),
                    "left_wall":int((surfaces==1).sum()),
                    "right_wall":int((surfaces==2).sum()),
                    "miss":int((surfaces==3).sum()),
                },
                "actual_ray_distance_max_error_m":error,
                "ray_hit_mask_mismatch":bad,
                "ray_count":int(len(distances)),
                "sensor_pattern_diff_max":diff,
            }
        result_lookup={(x["scene"],x["candidate"]):x for x in records}
        widths={n:(None if width is None else width) for n,width in
                (("floor",None),("channel_1.6",1.6),("channel_2.0",2.0))}
        render(out,tuple(widths),result_lookup,figures,widths,root[:2])
        with (out/"summary.csv").open("w",newline="") as fh:
            fields=("scene","candidate","near_candidates","far_candidates",
                    "near_floor","near_left_wall","near_right_wall",
                    "near_misses","fps_floor","fps_left_wall","fps_right_wall",
                    "fps_misses","far_left_wall","far_right_wall","far_misses",
                    "near_real_hit_xy_cells","fps_real_hit_xy_cells",
                    "fps_cell_retention")
            writer=csv.DictWriter(fh,fieldnames=fields);writer.writeheader()
            for row in records:
                nr=row["near_surface_counts"];fp=row["fps256_surface_counts"]
                far=row["far64_surface_counts"]
                writer.writerow({
                    "scene":row["scene"],"candidate":row["candidate"],
                    "near_candidates":row["near_candidates"],"far_candidates":row["far_candidates"],
                    "near_floor":nr["0"],"near_left_wall":nr["1"],"near_right_wall":nr["2"],
                    "near_misses":nr["3"],
                    "fps_floor":fp["0"],"fps_left_wall":fp["1"],"fps_right_wall":fp["2"],
                    "fps_misses":fp["3"],
                    "far_left_wall":far["1"],"far_right_wall":far["2"],"far_misses":far["3"],
                    "near_real_hit_xy_cells":row["near_real_hit_xy_cells"],
                    "fps_real_hit_xy_cells":row["fps_real_hit_xy_cells"],
                    "fps_cell_retention":row["fps_cell_retention"]
                })
        report.update(state="completed",scan_origin_world_m=origin.tolist(),
                      robot_root_world_m=root.tolist(),root_yaw_rad=float(yaw),
                      raw_sensor_rays=86400,raycheck=summary,measurements=records)
    except BaseException as error:
        report.update(state="failed",error=repr(error))
        raise
    finally:
        try:
            if runtime is not None:runtime.close()
        finally:
            if env is not None:env.close()
            (out/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
            print(f"[B8 corridors] {report['state']}: {out/'report.json'}",flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
