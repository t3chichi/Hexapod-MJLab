#!/usr/bin/env python
"""Approved static experiment: 35 saved poses x 5 surfaces x 4 selection recipes.

Actual production CPU sensor and wrapper. No physics, model edits, training,
automatic candidate adoption or claim of passability. Scene geometry is diagnostic.
"""
import argparse,hashlib,json,math,time
from pathlib import Path
import numpy as np
import torch,mujoco,warp as wp
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from compare_flat import ROOT,MODEL,LIBRARY,load_model,frame,pattern
from hexapod_mjlab.utils.lidar import LidarSensor,LidarConfig,LidarType
from hexapod_mjlab.rl.lidar_wrapper import LidarWrapper

SCENES=[('floor',None,None),('channel_1.6',.8,None),('channel_2.0',1.,None),('ceiling_0.6',None,.6),('ceiling_0.8',None,.8)]
RECIPES=[('C0',32,56,30,96),('C1',24,56,30,96),('C2',32,48,30,96),('C3',32,56,60,48)]
# Surface names are diagnostic metadata, not network input or reward categories.
NAMES=['ground','left_wall','right_wall','ceiling','normal_nohit']


def analytic(origin,directions,halfwidth,ceiling):
    o=np.asarray(origin);v=np.asarray(directions);n=len(v);lengths=np.full((4,n),np.inf)
    specifications=[(2,0,None)]
    if halfwidth is not None:specifications += [(1,halfwidth,1),(1,-halfwidth,2)]
    if ceiling is not None:specifications += [(2,ceiling,3)]
    for axis,position,label in specifications:
        kind=0 if label is None else label
        with np.errstate(divide='ignore',invalid='ignore'):t=(position-o[axis])/v[:,axis]
        good=(np.abs(v[:,axis])>1e-12)&(t>=0)&(t<15)
        if kind!=0:
            x=o[0]+t*v[:,0];good &= (x>=-2)&(x<=6)
            if kind in [1,2]:good &= o[2]+t*v[:,2]>=0
        lengths[kind]=np.where(good,t,np.inf)
    labels=lengths.argmin(0);distance=lengths.min(0);nohit=~np.isfinite(distance)
    labels[nohit]=4;distance[nohit]=15
    return distance,labels


def surface_mesh(halfwidth,ceiling,height):
    vertices=[];faces=[]
    def quad(p):
        start=len(vertices);vertices.extend(p);faces.extend([start,start+1,start+2,start,start+2,start+3])
    quad([[-32,-32,0],[32,-32,0],[32,32,0],[-32,32,0]])
    if halfwidth is not None:
        for y in [halfwidth,-halfwidth]:quad([[-2,y,0],[6,y,0],[6,y,height],[-2,y,height]])
    if ceiling is not None:quad([[-2,-32,ceiling],[6,-32,ceiling],[6,32,ceiling],[-2,32,ceiling]])
    return wp.Mesh(points=wp.array(np.asarray(vertices,np.float32),dtype=wp.vec3,device='cpu'),indices=wp.array(faces,dtype=wp.int32,device='cpu'))


def counts(labels):return {name:int((labels==i).sum()) for i,name in enumerate(NAMES)}


def stats(indices,points,labels,occluded,t):
    p=points[indices];lab=labels[indices];oc=occluded[indices]
    walls=(lab==1)|(lab==2)
    return {'points':len(indices),'surfaces':counts(lab),'occluded':int(oc.sum()),
        'occluded_surfaces':counts(lab[oc]),
        'wall_ahead_of_sensor':int((walls&(p[:,0]>t[0])).sum()),
        'wall_behind_body_root':int((walls&(p[:,0]<0)).sum()),
        'wall_in_startup_reference_x':int((walls&(p[:,0]>=-.89)&(p[:,0]<=.89)).sum()),
        'wall_above_sensor_body_z':int((walls&(p[:,2]>t[2])).sum()),
        'wall_body_xyz_bounds_m':np.stack([p[walls].min(0),p[walls].max(0)]).tolist() if walls.any() else None}


def nearest_distances(queries,samples):
    if len(queries)==0 or len(samples)==0:return None
    dist=cKDTree(samples).query(queries)[0]
    return {'p05_p50_p95_max_m':np.r_[np.percentile(dist,[5,50,95]),dist.max()].tolist()}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    torch.set_num_threads(1);wp.config.kernel_cache_dir=str(out/'warp_cache');wp.init()
    model=load_model(MODEL);records=json.loads(LIBRARY.read_text())['records'];assert len(records)==35
    directions=pattern();states=[frame(model,r['measured_qpos']) for r in records]
    pos=torch.tensor(np.array([s[4] for s in states]),dtype=torch.float32)
    quat=torch.tensor(np.array([Rotation.from_matrix(s[3]).as_quat() for s in states]),dtype=torch.float32)
    # Robot visibility is queried once per saved pose; re-used for every surface/config.
    robot_dist=[];robot_gid=[]
    for d,rb,pb,rs,ps,r,t in states:
        dist=np.empty(86400);gid=np.empty(86400,dtype=np.int32)
        mujoco.mj_multiRay(model,d,ps,np.ascontiguousarray((directions@rs.T).ravel()),np.array([1,0,0,0,0,0],np.uint8),True,-1,gid,dist,None,86400,15.)
        robot_dist.append(dist);robot_gid.append(gid)
    robot_dist=np.array(robot_dist);robot_gid=np.array(robot_gid)
    indices={}
    for name,theta,delta,az,el in RECIPES:
        azix=np.arange(0,900,900//az);elix=np.rint(np.linspace(0,95,el)).astype(int)
        ix=(azix[:,None]*96+elix).ravel();assert len(ix)==2880 and len(np.unique(ix))==2880
        assert elix[0]==0 and elix[-1]==95
        indices[name]=ix
    x,y=np.meshgrid(np.linspace(1.2,3.8,17),np.linspace(-.9,.9,11));oldxy=np.c_[x.ravel(),y.ravel(),np.zeros(187)]
    arrays={'qpos':np.array([r['measured_qpos'] for r in records]),'case_ids':np.array([r['id'] for r in records]),'pattern_dirs':directions,'robot_distance':robot_dist,'robot_geomid':robot_gid}
    for name,ix in indices.items():arrays[name+'/pool_indices']=ix
    rows=[];raychecks=[]
    for scene,halfwidth,ceiling in SCENES:
        scene_start=time.monotonic();mesh=surface_mesh(halfwidth,ceiling,float(pos[:,2].max())+16)
        cfg=LidarConfig(sensor_type=LidarType.AIRY,airy_horizontal_resolution_deg=.4,max_range=15,enable_sensor_noise=False,randomize_placement=False,update_frequency=10)
        sensor=LidarSensor({'num_envs':35,'mesh_ids':wp.array([mesh.id],dtype=wp.uint64,device='cpu'),'sensor_pos_tensor':pos,'sensor_quat_tensor':quat},None,cfg,device='cpu')
        local,ranges=sensor.update();local=local.reshape(35,86400,3).numpy();ranges=ranges.reshape(35,86400).numpy()
        labels=[];maxerr=0.;maskdiff=0
        packs={name:{key:[] for key in ['pool_body','pool_labels','pool_occluded','near_mask','final_near_body','final_far_body','final_near_raw_index','final_far_raw_index']} for name,*_ in RECIPES}
        for k,(rec,state) in enumerate(zip(records,states)):
            d,rb,pb,rs,ps,r,t=state
            expected,lab=analytic(ps,directions@rs.T,halfwidth,ceiling)
            err=np.abs(expected-ranges[k]);maxerr=max(maxerr,float(err.max()))
            diff=(expected<15)!=(ranges[k]<15);maskdiff+=int(diff.sum())
            if diff.any() or err.max()>1e-3:
                np.savez_compressed(out/f'failure_{scene}_{k}.npz',expected=expected,actual=ranges[k],label=lab,diff=diff)
                raise AssertionError(f'Raycast disagreement {scene}/{rec["id"]}: {err.max()}, {diff.sum()} masks')
            points=local[k].astype(float)@r.T+t
            blocked=(robot_dist[k]>=0)&(robot_dist[k]<expected-1e-6)
            labels.append(lab)
            yaw=math.atan2(rb[1,0],rb[0,0]);grid_world=oldxy@Rotation.from_euler('z',yaw).as_matrix().T+pb;grid_world[:,2]=0
            grid_body=(grid_world-pb)@rb;gridrel=grid_world-ps;griddist=np.linalg.norm(gridrel,axis=1);gridv=gridrel/griddist[:,None]
            gd,glab=analytic(ps,gridv,halfwidth,ceiling)
            geometry_visible=(glab==0)&(np.abs(gd-griddist)<1e-6)
            dome=(gridrel/griddist[:,None])@(rs[:,2])>=0
            for name,theta,delta,az,el in RECIPES:
                ix=indices[name];p=points[ix]
                axis=Rotation.from_euler('y',delta,degrees=True).apply(r[:,2])
                wrapper=LidarWrapper(1,2880,1,256,64,theta,0,torch.device('cpu'),torch.tensor(Rotation.from_matrix(r).as_quat()[None,:],dtype=torch.float32),torch.tensor(t[None,:],dtype=torch.float32),torch.tensor(axis,dtype=torch.float32))
                pt=torch.tensor(p[None,:,:],dtype=torch.float32)
                rel=pt-wrapper._sensor_t
                norms=torch.norm(wrapper._to_sensor_frame(pt),dim=-1)
                near=((rel*wrapper._cone_axis).sum(-1)/(norms+1e-9)>=wrapper._cone_sin_thr)[0].numpy()
                assert near.sum()>=256 and (~near).sum()>=64
                with torch.no_grad():selected=wrapper.wrap_obs(torch.empty(1,0),pt,torch.zeros(1,dtype=torch.bool)).reshape(320,3).numpy()
                # Constrain source matching to the expected branch; no metadata mixing.
                selected_indices=[]
                for chunk,mask in [(selected[:256],near),(selected[256:],~near)]:
                    sub=np.flatnonzero(mask);dist,nearest=cKDTree(p[mask]).query(chunk)
                    assert dist.max()<5e-5,(scene,name,rec['id'],dist.max())
                    selected_indices.append(sub[nearest])
                ni,fi=selected_indices;raw_ni,raw_fi=ix[ni],ix[fi]
                fullnear=((directions@r.T)@axis)>=np.sin(np.deg2rad(theta))
                gn=((grid_body-t)@axis/griddist)>=np.sin(np.deg2rad(theta));visible=geometry_visible&dome&gn
                row={'scene':scene,'case':rec['id'],'recipe':name,'theta_deg':theta,'delta_deg':delta,
                     'full':{'near':stats(np.flatnonzero(fullnear),points,lab,blocked,t),'far':stats(np.flatnonzero(~fullnear),points,lab,blocked,t)},
                     'pool':{'near':stats(ix[near],points,lab,blocked,t),'far':stats(ix[~near],points,lab,blocked,t)},
                     'final':{'near':stats(raw_ni,points,lab,blocked,t),'far':stats(raw_fi,points,lab,blocked,t)},
                     'legacy_grid':{'queries':187,'scene_unoccluded_ground_queries':int(geometry_visible.sum()),'near_angular_scene_visible_queries':int(visible.sum()),
                        'pool_ground_distance_all_queries':nearest_distances(grid_body,p[near&(lab[ix]==0)]),
                        'final_ground_distance_all_queries':nearest_distances(grid_body,selected[:256][lab[raw_ni]==0]),
                        'final_ground_distance_visible_queries':nearest_distances(grid_body[visible],selected[:256][lab[raw_ni]==0])}}
                rows.append(row)
                for key,val in [('pool_body',p),('pool_labels',lab[ix]),('pool_occluded',blocked[ix]),('near_mask',near),('final_near_body',selected[:256]),('final_far_body',selected[256:]),('final_near_raw_index',raw_ni),('final_far_raw_index',raw_fi)]:packs[name][key].append(val)
        arrays[scene+'/raw_ranges']=ranges;arrays[scene+'/raw_labels']=np.array(labels)
        for name,pack in packs.items():
            for key,vals in pack.items():arrays[scene+'/'+name+'/'+key]=np.array(vals)
        raychecks.append({'scene':scene,'states':35,'rays_per_state':86400,'max_range_error_m':maxerr,'hit_mask_differences':maskdiff,'elapsed_s':time.monotonic()-scene_start})
        print(scene, 'done',round(time.monotonic()-scene_start,2),'s',flush=True)
    np.savez_compressed(out/'raw.npz',**arrays)
    sources=[Path(__file__),MODEL,LIBRARY,ROOT/'src/hexapod_mjlab/rl/lidar_wrapper.py',ROOT/'src/hexapod_mjlab/utils/lidar/lidar_sensor.py',ROOT/'src/hexapod_mjlab/utils/lidar/sensor_kernels/lidar_kernels_warp.py']
    report={'status':'diagnostic_completed_no_candidate_adoption','scope':__doc__,'scenes':SCENES,'recipes':RECIPES,'raw_scans':175,'selection_cases':700,'ray_checks':raychecks,'rows':rows,'elapsed_s':time.monotonic()-start,
            'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (out/'compare_coverage.py').write_text(Path(__file__).read_text())
    return 0

if __name__=='__main__':raise SystemExit(main())
