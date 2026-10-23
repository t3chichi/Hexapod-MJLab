#!/usr/bin/env python
"""Postprocess saved flat scans: angular support vs actual retained points.

No new raycasts, physics steps, scan settings or training choices. An old grid
is a diagnostic query set, not a proposed new auxiliary grid. Sensor visibility
here ignores occluders; actual foot-height support and passability are not claimed.
"""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from compare_flat import ROOT,MODEL,load_model,frame,cone


def query_support(points_body,r,t):
    rel=points_body-t;distance=np.linalg.norm(rel,axis=1);directions=rel/distance[:,None]
    dome=(directions@r[:,2])>=-1e-12
    inside=(distance<=15)&dome
    near=inside&((directions@cone(r))>=np.sin(np.deg2rad(32)))
    return inside,near


def ground_body(xy,rb,pb):
    z=(-pb[2]-xy[:,0]*rb[2,0]-xy[:,1]*rb[2,1])/rb[2,2]
    return np.column_stack([xy,z])


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();args.run=args.run.resolve();args.out.mkdir(exist_ok=False,parents=True)
    raw=np.load(args.run/'raw.npz');model=load_model(MODEL);rows=[];nominal=None
    # Exactly the legacy measured grid: yaw frame, x=1.2..3.8, y=-.9...9, 17x11.
    x,y=np.meshgrid(np.linspace(1.2,3.8,17),np.linspace(-.9,.9,11));legacy=np.column_stack([x.ravel(),y.ravel()])
    for k,qpos in enumerate(raw['qpos']):
        d,rb,pb,rs,ps,r,t=frame(model,qpos)
        np.testing.assert_allclose(r,raw['after/site_R_BASE'][k],atol=1e-12)
        np.testing.assert_allclose(t,raw['after/site_t_BASE'][k],atol=1e-12)
        yaw=math.atan2(rb[1,0],rb[0,0]);ryaw=Rotation.from_euler('z',yaw).as_matrix()
        world=np.c_[legacy,np.zeros(len(legacy))]@ryaw.T+pb;world[:,2]=0
        grid=(world-pb)@rb;inside,near=query_support(grid,r,t)
        pool=raw['after/pool'][k];ph=raw['after/hit'][k]&raw['after/near_mask'][k]
        final=raw['after/final_near'][k]
        pooldist=np.sqrt(((grid[:,None,:]-pool[ph][None,:,:])**2).sum(-1).min(1))
        finaldist=np.sqrt(((grid[:,None,:]-final[None,:,:])**2).sum(-1).min(1))
        row={'case':str(raw['case_ids'][k]),'legacy_ground_queries':len(grid),'within_full_angular_fov':int(inside.sum()),'within_near_angular_cone':int(near.sum()),
             'nearest_near_pool_distance_m_p05_p50_p95_max':np.r_[np.percentile(pooldist,[5,50,95]),pooldist.max()].tolist(),
             'nearest_final_near_distance_m_p05_p50_p95_max':np.r_[np.percentile(finaldist,[5,50,95]),finaldist.max()].tolist()}
        rows.append(row)
        if row['case']=='height_0.48_width_1.00':nominal=(grid,inside,near,pool,ph,final,r,t,rb,pb)
    grid,inside,near,pool,ph,final,r,t,rb,pb=nominal
    maxz=float(cone(r)[2]*np.sin(np.deg2rad(32))+np.sqrt(1-cone(r)[2]**2)*np.cos(np.deg2rad(32)))
    # Independent extremum over actual recorded 86,400 directions.
    bodydirs=raw['pattern_dirs']@r.T;mask=(bodydirs@cone(r))>=np.sin(np.deg2rad(32))
    actualmax=float(bodydirs[mask,2].max());assert actualmax<=maxz+1e-12 and maxz<0
    result={'scope':'Postprocessing of saved after-correction scans only. No new sensor settings, physics, obstacles or training. Angular support is continuous ideal pattern geometry, not discrete-sample coverage; no occlusion or safety certificate.',
        'legacy_query_definition':'Exactly existing 17x11 yaw-frame grid, x=1.2..3.8m, y=-.9...9m; not the new auxiliary-grid choice.',
        'near_direction_upper_bound_body_z':maxz,'actual_raw_near_max_direction_body_z':actualmax,
        'near_direction_top_elevation_body_deg':float(np.degrees(np.arcsin(maxz))),
        'nominal':next(v for v in rows if v['case']=='height_0.48_width_1.00'),'rows':rows,
        'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.run/'raw.npz',MODEL,Path(__file__),ROOT/'src/hexapod_mjlab/tasks/el_4090_lidar/mdp/observations.py']}}
    (args.out/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (args.out/'summarize_support.py').write_text(Path(__file__).read_text())
    xy=np.stack(np.meshgrid(np.linspace(-1.1,5.2,253),np.linspace(-1.3,1.3,105)),axis=-1).reshape(-1,2)
    pts=ground_body(xy,rb,pb);fov,n=query_support(pts,r,t);labels=np.where(~fov,0,np.where(n,2,1)).reshape(105,253)
    fig,ax=plt.subplots(figsize=(12,5),layout='constrained')
    from matplotlib.colors import ListedColormap
    ax.imshow(labels,origin='lower',extent=[-1.1,5.2,-1.3,1.3],cmap=ListedColormap(['#dddddd','#dce9f4','#f9e1d9']),alpha=.8,aspect='equal',vmin=0,vmax=2)
    ax.scatter(pool[ph,0],pool[ph,1],s=5,c='#c4583f',alpha=.4,label='Near ground pool')
    ax.scatter(final[:,0],final[:,1],s=16,facecolors='none',edgecolors='black',linewidths=.5,label='Final near 256')
    ax.scatter(grid[near,0],grid[near,1],marker='+',s=30,c='#167153',label='Legacy grid: within near cone')
    ax.scatter(grid[~near,0],grid[~near,1],marker='x',s=30,c='#743589',label='Legacy grid: outside near cone')
    ax.add_patch(Rectangle((-.89,-.69),1.78,1.38,fill=False,edgecolor='#777',linestyle='--',lw=1.5,label='Startup reference bounding box (not actual robot)'))
    ax.scatter(t[0],t[1],marker='*',s=110,c='black',label='Sensor origin')
    ax.set_xlim(-1.1,5.2);ax.set_ylim(-1.3,1.3);ax.set_xlabel('Body x (m), forward right');ax.set_ylabel('Body y (m)')
    ax.set_title('Nominal settled state: ground observation support\nGray: outside full angular FoV | pale blue: far cone | pale red: near cone; no occlusion')
    ax.legend(loc='upper right',fontsize=7);ax.grid(alpha=.15)
    fig.savefig(args.out/'ground_support.png',dpi=180);plt.close(fig)
    print(json.dumps({k:result[k] for k in ['near_direction_upper_bound_body_z','near_direction_top_elevation_body_deg','nominal']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
