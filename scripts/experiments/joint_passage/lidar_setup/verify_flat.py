#!/usr/bin/env python
"""Independent saved-output checks + actual LidarSensor/Warp CPU raycasts.

Visibility probes use MuJoCo rays against group-0 robot geoms, with the radome
visual excluded. They diagnose possible occlusion, not physical mesh clearance.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
import mujoco
import warp as wp
from scipy.spatial.transform import Rotation
from compare_flat import ROOT,MODEL,LIBRARY,load_model,frame
from hexapod_mjlab.utils.lidar import LidarSensor,LidarConfig,LidarType


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path,required=True);ap.add_argument('--before',type=Path,required=True)
    args=ap.parse_args();args.run=args.run.resolve();args.before=args.before.resolve()
    out=args.run/'verification';out.mkdir(exist_ok=False)
    raw=np.load(args.run/'raw.npz');report=json.loads((args.run/'report.json').read_text())
    torch.set_num_threads(1);wp.config.kernel_cache_dir=str(out/'warp_cache');wp.init()
    verts=np.array([[-32,-32,0],[32,-32,0],[32,32,0],[-32,32,0]],dtype=np.float32)
    mesh=wp.Mesh(points=wp.array(verts,dtype=wp.vec3,device='cpu'),indices=wp.array([0,1,2,0,2,3],dtype=wp.int32,device='cpu'))
    ids=wp.array([mesh.id],dtype=wp.uint64,device='cpu')
    checks=[];occlusion=[];rawextra={}
    for label,path in [('before',args.before),('after',MODEL)]:
        model=load_model(path);pss=[];qss=[];analytic=[];sensor_frames=[]
        for k,qpos in enumerate(raw['qpos']):
            d,rb,pb,rs,ps,r,t=frame(model,qpos);pss.append(ps);qss.append(Rotation.from_matrix(rs).as_quat());sensor_frames.append((d,rb,pb,rs,ps))
            # Independent line-plane intersection and spherical rotation, not floor_scan().
            directions=raw['pattern_dirs']@rs.T
            with np.errstate(divide='ignore',invalid='ignore'):
                lengths=-ps[2]/directions[:,2]
            valid=(directions[:,2]<0)&(lengths>=0)&(lengths<15)
            expected=np.where(valid,lengths,15);analytic.append(expected)
            ix=raw['pool_indices'];points=((ps+expected[ix,None]*directions[ix])-pb)@rb
            np.testing.assert_allclose(points,raw[label+'/pool'][k],atol=1e-10,rtol=0)
            np.testing.assert_array_equal(valid[ix],raw[label+'/hit'][k])
            pole=r[:,2];c,s=np.cos(np.deg2rad(56)),np.sin(np.deg2rad(56))
            axis=np.array([c*pole[0]+s*pole[2],pole[1],-s*pole[0]+c*pole[2]])
            pred=((points-t)@axis/np.linalg.norm(points-t,axis=1))>=np.sin(np.deg2rad(32))
            np.testing.assert_array_equal(pred,raw[label+'/near_mask'][k])
            stat=next(row for row in report['rows'] if row['version']==label and row['case']==str(raw['case_ids'][k]))
            for group,mask in [('near',pred),('far',~pred)]:
                h=valid[ix]&mask
                assert stat['pool'][group]['points']==int(mask.sum())
                assert stat['pool'][group]['ground_hits']==int(h.sum())
                assert stat['pool'][group]['ground_ahead_of_sensor']==int((points[h,0]>t[0]).sum())
            for key,mask in [('final_near',pred),('final_far',~pred)]:
                selected=raw[label+'/'+key][k];pool=points[mask]
                assert np.sqrt(((selected[:,None,:]-pool[None,:,:])**2).sum(-1).min(1)).max()<5e-5
            # Independently ask the compiled robot whether a ray would hit it first.
            raydirs=np.ascontiguousarray(directions[ix].ravel())
            distance=np.empty(len(ix));geomid=np.empty(len(ix),dtype=np.int32)
            mujoco.mj_multiRay(model,d,ps,raydirs,np.array([1,0,0,0,0,0],np.uint8),True,-1,geomid,distance,None,len(ix),15.)
            blocked=(distance>=0)&(distance<expected[ix]-1e-6)
            bybody={}
            for gid in geomid[blocked]:
                name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[gid]))
                bybody[name]=bybody.get(name,0)+1
            occlusion.append({'version':label,'case':str(raw['case_ids'][k]),'blocked_pool':int(blocked.sum()),'near_pool':int(pred.sum()),'blocked_near':int((blocked&pred).sum()),'far_pool':int((~pred).sum()),'blocked_far':int((blocked&~pred).sum()),'first_hit_bodies':bybody})
            rawextra[f'{label}/occlusion_distance/{k}']=distance;rawextra[f'{label}/occlusion_geomid/{k}']=geomid
        cfg=LidarConfig(sensor_type=LidarType.AIRY,airy_horizontal_resolution_deg=.4,max_range=15,enable_sensor_noise=False,randomize_placement=False,update_frequency=10)
        env={'num_envs':len(pss),'mesh_ids':ids,'sensor_pos_tensor':torch.tensor(np.array(pss),dtype=torch.float32),'sensor_quat_tensor':torch.tensor(np.array(qss),dtype=torch.float32)}
        sensor=LidarSensor(env,None,cfg,device='cpu');pixels,dist=sensor.update()
        wd=dist.reshape(len(pss),-1).numpy();expected=np.array(analytic)
        err=np.abs(wd-expected);hit_match=np.array_equal(wd<15,expected<15)
        checks.append({'version':label,'actual_sensor_device':'cpu','states':len(pss),'rays_per_state':sensor.num_rays,'max_range_error_m':float(err.max()),'hit_mask_identical':hit_match,'all_points_finite':bool(torch.isfinite(pixels).all()),'raw_saved_pool_and_statistics_recomputed':True})
        rawextra[label+'/warp_ranges']=wd
        assert hit_match and err.max()<1e-3 and bool(torch.isfinite(pixels).all())
    np.savez_compressed(out/'raw_checks.npz',**rawextra)
    nominal=[r for r in occlusion if r['case']=='height_0.48_width_1.00']
    result={'scope':'Independent saved-output recomputation, actual LidarSensor and production Warp kernel on CPU, 35 states per version. No GPU/physics/terrain-grid/motion-clearance acceptance. MuJoCo robot visibility probes are separate from terrain-only sensor rays and exclude group1 visual radome.',
        'checks':checks,'occlusion':occlusion,'nominal_occlusion':nominal,
        'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),args.run/'raw.npz',args.run/'report.json',ROOT/'src/hexapod_mjlab/utils/lidar/lidar_sensor.py',ROOT/'src/hexapod_mjlab/utils/lidar/sensor_kernels/lidar_kernels_warp.py']},'status':'PASS'}
    (out/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'checks':checks,'nominal_occlusion':nominal},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
