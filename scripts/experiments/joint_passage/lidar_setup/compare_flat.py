#!/usr/bin/env python
"""Before/after optical-frame diagnostic; fixed 35 saved states, no physics steps.

No robot occlusion in analytic floor intersections. Reuses production scan,
angle_stride pool and near/far selection; a one-frame wrapper only, not the
joint task's ten-independent-scan history. Nothing here selects training settings.
"""
from __future__ import annotations
import argparse, ast, functools, hashlib, json, math, sys
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
import torch
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'src'))
from hexapod_mjlab.utils.lidar.sensor_pattern.sensor_lidar.genera_lidar_scan_pattern import SpinningLidarGenerator
from hexapod_mjlab.utils.lidar.pool_samplers import AngleStrideSampler
from hexapod_mjlab.rl.lidar_wrapper import LidarWrapper
MODEL = ROOT/'src/hexapod_mjlab/assets/robots/el_4090/mjcf/el_4090_lidar.xml'
URDF = MODEL.parent.parent/'urdf/el_4090_lidar.urdf'
LIBRARY = ROOT/'docs/research/morphology/evidence/library.json'
MOUNT_SOURCE = ROOT/'src/hexapod_mjlab/tasks/el_4090_lidar/mdp/lidar.py'


def load_model(path, legacy_optical_frame=False):
    # Old snapshots retain meshdir relative to their original location.
    meshes = {p.name:p.read_bytes() for p in (MODEL.parent.parent/'meshes').glob('*.STL')}
    text=Path(path).read_text()
    if legacy_optical_frame:
        tree=ET.fromstring(text)
        tree.find(".//body[@name='LIDAR_SENSOR']").attrib.pop('quat',None)
        text=ET.tostring(tree,encoding='unicode')
    return mujoco.MjModel.from_xml_string(text, assets=meshes)


def mount_resolver():
    """Execute the real production function, avoiding unrelated task registration.

    Tests cover its actual compiled-site extraction; this is not a whole-env test.
    """
    tree = ast.parse(MOUNT_SOURCE.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='resolve_lidar_mount')
    scope = {'functools':functools, 'Path':Path, 'EL4090_LIDAR_MJCF':MODEL}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(MOUNT_SOURCE),'exec'),scope)
    return scope['resolve_lidar_mount']


def urdf_mount():
    tree = ET.parse(URDF);t=np.eye(4)
    for name in ['RADAR_BRACKET_FIX','LIDAR_AIRY_FIX','LIDAR_SENSOR_FIX']:
        origin=tree.find(f".//joint[@name='{name}']/origin")
        step=np.eye(4);step[:3,:3]=Rotation.from_euler('xyz',np.fromstring(origin.attrib['rpy'],sep=' ')).as_matrix()
        step[:3,3]=np.fromstring(origin.attrib['xyz'],sep=' ');t=t@step
    return t


def frame(model,qpos=None):
    d=mujoco.MjData(model)
    if qpos is None:mujoco.mj_resetDataKeyframe(model,d,0)
    else:d.qpos[:]=qpos
    mujoco.mj_forward(model,d)
    b=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'BASE')
    s=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'lidar_sensor')
    rb=d.xmat[b].reshape(3,3).copy();rs=d.site_xmat[s].reshape(3,3).copy()
    pb=d.xpos[b].copy();ps=d.site_xpos[s].copy()
    return d,rb,pb,rs,ps,rb.T@rs,rb.T@(ps-pb)


def pattern():
    theta,phi=SpinningLidarGenerator.generate_AIRY(horizontal_resolution_deg=.4)
    return np.column_stack([np.cos(phi)*np.cos(theta),np.cos(phi)*np.sin(theta),np.sin(phi)])


def cone(mount):
    pole=mount[:,2];angle=math.radians(56)
    return Rotation.from_rotvec([0,angle,0]).apply(pole)


def floor_scan(directions,rs,ps,rb,pb):
    wd=directions@rs.T
    distances=np.full(len(wd),15.)
    down=wd[:,2]<-1e-12
    possible=np.full(len(wd),np.inf);possible[down]=-ps[2]/wd[down,2]
    hit=down&(possible>=0)&(possible<15.)
    distances[hit]=possible[hit]
    world=ps+wd*distances[:,None]
    return (world-pb)@rb,hit,distances


def summary(points,hit,sensor_x):
    p=points[hit]
    return {'points':len(points),'ground_hits':int(hit.sum()),'normal_nohit':int((~hit).sum()),
            'ground_xy_bounds_m':np.stack([p[:,:2].min(0),p[:,:2].max(0)]).tolist() if len(p) else None,
            'ground_xy_percentiles_05_50_95_m':np.percentile(p[:,:2],[5,50,95],axis=0).tolist() if len(p) else None,
            'ground_ahead_of_sensor':int((p[:,0]>sensor_x).sum()),
            'ground_behind_or_at_sensor':int((p[:,0]<=sensor_x).sum())}


def source_hit(selected,pool,pool_hit):
    # Original ray identity can repeat (pole); identical endpoints have identical hits.
    sq=((selected[:,None,:]-pool[None,:,:])**2).sum(-1)
    index=sq.argmin(1)
    if np.sqrt(sq[np.arange(len(index)),index]).max()>5e-5:
        raise AssertionError('Selection is not from the original pool')
    return pool_hit[index]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--before',type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    models=[load_model(args.before),load_model(MODEL)]
    records=json.loads(LIBRARY.read_text())['records']
    if len(records)!=35:raise ValueError('Frozen diagnostic expects 35 saved states')
    directions=pattern();assert directions.shape==(86400,3)
    sampler=AngleStrideSampler(900,96,86400,seed=0,device='cpu',azimuth_stride=30)
    ix=sampler.sample(torch.zeros(1,86400,3),torch.zeros(1,86400),torch.zeros(1,dtype=torch.long))[0].numpy()
    assert len(ix)==2880
    rows=[];arrays={};nominal={}
    for label,model in zip(['before','after'],models):
        pools=[];hits=[];nears=[];finalns=[];finalfs=[];rotations=[];translations=[];worldposes=[]
        for rec in records:
            d,rb,pb,rs,ps,mount,t=frame(model,rec['measured_qpos'])
            pts,hit,dist=floor_scan(directions,rs,ps,rb,pb)
            near=(directions@mount.T@cone(mount))>=math.sin(math.radians(32))
            p,h,n=pts[ix],hit[ix],near[ix]
            q=Rotation.from_matrix(mount).as_quat()
            wrapper=LidarWrapper(1,2880,1,256,64,32,0,torch.device('cpu'),
                torch.tensor(q[None,:],dtype=torch.float32),torch.tensor(t[None,:],dtype=torch.float32),
                torch.tensor(cone(mount),dtype=torch.float32))
            with torch.no_grad():
                wrapped=wrapper.wrap_obs(torch.empty(1,0),torch.tensor(p[None,:],dtype=torch.float32),torch.zeros(1,dtype=torch.bool))
            selected=wrapped.reshape(-1,3).numpy();sn=selected[:256];sf=selected[256:]
            hn=source_hit(sn,p,h);hf=source_hit(sf,p,h)
            stats={'case':rec['id'],'version':label,'full':{'near':summary(pts[near],hit[near],t[0]),'far':summary(pts[~near],hit[~near],t[0])},
                'pool':{'near':summary(p[n],h[n],t[0]),'far':summary(p[~n],h[~n],t[0])},
                'final':{'near':summary(sn,hn,t[0]),'far':summary(sf,hf,t[0])},
                'sensor_origin_BASE_m':t.tolist(),'scan_pole_BASE':mount[:,2].tolist(),'cone_axis_BASE':cone(mount).tolist()}
            rows.append(stats);pools.append(p);hits.append(h);nears.append(n);finalns.append(sn);finalfs.append(sf)
            rotations.append(mount);translations.append(t);worldposes.append(np.r_[pb,Rotation.from_matrix(rb).as_quat()])
            if rec['id']=='height_0.48_width_1.00':nominal[label]=(p,h,n,sn,sf)
        for key,val in [('pool',pools),('hit',hits),('near_mask',nears),('final_near',finalns),('final_far',finalfs),('site_R_BASE',rotations),('site_t_BASE',translations),('base_pose_world_xyzw',worldposes)]:
            arrays[label+'/'+key]=np.asarray(val)
    arrays['case_ids']=np.array([r['id'] for r in records]);arrays['qpos']=np.array([r['measured_qpos'] for r in records])
    arrays['pool_indices']=ix;arrays['pattern_dirs']=directions
    np.savez_compressed(args.out/'raw.npz',**arrays)
    sources=[MODEL,URDF,LIBRARY,MOUNT_SOURCE,Path(__file__),ROOT/'src/hexapod_mjlab/rl/lidar_wrapper.py',ROOT/'src/hexapod_mjlab/utils/lidar/pool_samplers.py',ROOT/'src/hexapod_mjlab/utils/lidar/sensor_pattern/sensor_lidar/genera_lidar_scan_pattern.py']
    report={'scope':'Fixed 35 saved T1 states; analytic infinite z=0 floor, no robot occlusion or physics stepping. Actual pattern, fixed angle_stride30 and production 256/64 selection. This does not accept physical mounting, terrain grid or task settings.',
        'config':{'theta_deg':32,'delta_deg':56,'max_range_m':15,'raw_rays':86400,'pool_points':2880,'near_points':256,'far_points':64,'noise':False,'robot_occlusion':False,'num_states':35},
        'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'before_model_sha256':hashlib.sha256(args.before.read_bytes()).hexdigest(),'rows':rows}
    (args.out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for row,label in enumerate(['before','after']):
        p,h,n,sn,sf=nominal[label]
        for col,(i,j,title) in enumerate([(0,1,'Full BEV'),(0,1,'Local BEV'),(0,2,'Side view')]):
            ax=axes[row,col]
            ax.scatter(p[~n,i],p[~n,j],s=3,c='#4187bb',alpha=.4,label='Far pool')
            ax.scatter(p[n,i],p[n,j],s=3,c='#d55445',alpha=.5,label='Near pool')
            if col==1:
                ax.scatter(sn[:,i],sn[:,j],s=8,facecolors='none',edgecolors='black',linewidths=.4,label='Selected near 256')
                ax.set_xlim(-1,5);ax.set_ylim(-3,3)
            else:ax.set_xlim(-16,16);ax.set_ylim((-16,16) if col==0 else (-1,16))
            ax.set_title(label+' / '+title);ax.set_xlabel('Body x (m)');ax.set_ylabel('Body '+('y' if j==1 else 'z')+' (m)');ax.grid(alpha=.2)
            if col!=2:ax.set_aspect('equal')
        axes[row,1].legend(fontsize=8)
    fig.suptitle('Same nominal settled state; optical axes corrected, hardware unchanged\nAnalytic floor; genuine misses kept at 15 m; no robot occlusion')
    fig.savefig(args.out/'comparison.png',dpi=150);plt.close(fig)
    print(json.dumps([r for r in rows if r['case']=='height_0.48_width_1.00'],ensure_ascii=False,indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
