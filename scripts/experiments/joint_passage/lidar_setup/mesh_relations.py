#!/usr/bin/env python
"""Read-only original STL surface relations for 35 poses, bracket vs two HIPs.

No collision-filter changes. Broad phase conservatively bounds triangles using
centroid balls and exact AABBs. Narrow phase tests all six segment/triangle
relations plus coplanar 2D SAT. Non-closed HIP exports cannot certify solid volume.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import mujoco,trimesh
from scipy.spatial import cKDTree
from compare_flat import ROOT,MODEL,LIBRARY,load_model,frame
EPS=1e-8


def candidates(a,b,padding=0.):
    ca=a.mean(1);cb=b.mean(1);ra=np.linalg.norm(a-ca[:,None,:],axis=2).max(1);rb=np.linalg.norm(b-cb[:,None,:],axis=2).max(1)
    amin=a.min(1);amax=a.max(1);bmin=b.min(1);bmax=b.max(1)
    choose=np.flatnonzero(((amax>=bmin.min(0)-padding-EPS)&(amin<=bmax.max(0)+padding+EPS)).all(1))
    pairs=[]
    bins=np.floor(np.log2(np.maximum(rb,1e-15))).astype(int)
    for category in np.unique(bins):
        bi=np.flatnonzero(bins==category);tree=cKDTree(cb[bi]);neighbor=tree.query_ball_point(ca[choose],ra[choose]+rb[bi].max()+padding+EPS)
        lengths=np.array([len(x) for x in neighbor],dtype=int);aa=np.repeat(choose,lengths)
        if len(aa)==0:continue
        bb=bi[np.concatenate([np.asarray(x,dtype=int) for x in neighbor if len(x)])]
        sep=np.maximum(0,np.maximum(amin[aa]-bmax[bb],bmin[bb]-amax[aa]))
        keep=(np.linalg.norm(sep,axis=1)<=padding+EPS)
        pairs.append(np.column_stack([aa[keep],bb[keep]]))
    return np.concatenate(pairs) if pairs else np.empty((0,2),int)


def segment_face(start,end,triangle):
    e1=triangle[:,1]-triangle[:,0];e2=triangle[:,2]-triangle[:,0];direction=end-start
    p=np.cross(direction,e2);det=(e1*p).sum(1)
    scale=np.linalg.norm(direction,axis=1)*np.linalg.norm(np.cross(e1,e2),axis=1)
    nonparallel=np.abs(det)>1e-10*scale
    inv=np.divide(1.,det,out=np.zeros_like(det),where=nonparallel)
    tv=start-triangle[:,0];u=(tv*p).sum(1)*inv;q=np.cross(tv,e1)
    v=(direction*q).sum(1)*inv;alpha=(e2*q).sum(1)*inv
    tol=1e-9
    hit=nonparallel&(alpha>=-tol)&(alpha<=1+tol)&(u>=-tol)&(v>=-tol)&(u+v<=1+tol)
    proper=hit&(alpha>tol)&(alpha<1-tol)&(u>tol)&(v>tol)&(u+v<1-tol)
    return hit,proper,start+alpha[:,None]*direction


def relations(a,b):
    size=len(a);hit=np.zeros(size,bool);proper=np.zeros(size,bool);point=np.zeros((size,3))
    for source,face in [(a,b),(b,a)]:
        for e in range(3):
            h,p,w=segment_face(source[:,e],source[:,(e+1)%3],face)
            point[p|(~hit&h)]=w[p|(~hit&h)];hit|=h;proper|=p
    na=np.cross(a[:,1]-a[:,0],a[:,2]-a[:,0]);nb=np.cross(b[:,1]-b[:,0],b[:,2]-b[:,0])
    la=np.linalg.norm(na,axis=1);lb=np.linalg.norm(nb,axis=1)
    assert np.all(la>0)&np.all(lb>0),'Degenerate original triangles; classification not supported'
    parallel=np.linalg.norm(np.cross(na,nb),axis=1)<=1e-10*la*lb
    plane=np.abs(((b[:,0]-a[:,0])*na).sum(1))/la
    cop=np.flatnonzero(parallel&(plane<=EPS));cop_hit=np.zeros(size,bool)
    for axis in range(3):
        sub=cop[np.argmax(np.abs(na[cop]),axis=1)==axis]
        if len(sub)==0:continue
        dims=[i for i in range(3) if i!=axis];ta=a[sub][:,:,dims];tb=b[sub][:,:,dims];overlap=np.ones(len(sub),bool)
        for tri in [ta,tb]:
            for e in range(3):
                edge=tri[:,(e+1)%3]-tri[:,e];normal=np.stack([-edge[:,1],edge[:,0]],axis=1)
                pa=(ta*normal[:,None,:]).sum(2);pb=(tb*normal[:,None,:]).sum(2)
                tolerance=EPS*np.linalg.norm(normal,axis=1)
                overlap &= (pa.max(1)>=pb.min(1)-tolerance)&(pb.max(1)>=pa.min(1)-tolerance)
        cop_hit[sub]=overlap
    return hit|cop_hit,proper,cop_hit,point


def winding(points,triangles):
    values=[]
    for start in range(0,len(points),8):
        vec=triangles[None,:,:,:]-points[start:start+8,None,None,:]
        a,b,c=vec[:,:,0],vec[:,:,1],vec[:,:,2];la=np.linalg.norm(a,axis=2);lb=np.linalg.norm(b,axis=2);lc=np.linalg.norm(c,axis=2)
        numerator=np.einsum('...i,...i',a,np.cross(b,c))
        denominator=la*lb*lc+np.einsum('...i,...i',a,b)*lc+np.einsum('...i,...i',b,c)*la+np.einsum('...i,...i',c,a)*lb
        values.extend((2*np.arctan2(numerator,denominator)).sum(1)/(4*np.pi))
    return np.array(values)


def segment_distance(a0,a1,b0,b1):
    u=a1-a0;v=b1-b0;w=a0-b0
    aa=(u*u).sum(1);bb=(u*v).sum(1);cc=(v*v).sum(1);dd=(u*w).sum(1);ee=(v*w).sum(1)
    den=aa*cc-bb*bb
    s=np.divide(bb*ee-cc*dd,den,out=np.zeros_like(den),where=den>1e-12*aa*cc);s=np.clip(s,0,1)
    t=(bb*s+ee)/cc;below=t<0;above=t>1;t=np.clip(t,0,1)
    s[below]=np.clip(-dd[below]/aa[below],0,1);s[above]=np.clip((bb[above]-dd[above])/aa[above],0,1)
    return np.linalg.norm(a0+s[:,None]*u-b0-t[:,None]*v,axis=1)


def distances(a,b):
    result=np.full(len(a),np.inf)
    for source,target in [(a,b),(b,a)]:
        for vertex in range(3):
            closest=trimesh.triangles.closest_point(target,source[:,vertex])
            result=np.minimum(result,np.linalg.norm(closest-source[:,vertex],axis=1))
    for e in range(3):
        for f in range(3):result=np.minimum(result,segment_distance(a[:,e],a[:,(e+1)%3],b[:,f],b[:,(f+1)%3]))
    return result


def inspect(a,b):
    pairs=candidates(a,b);contacts=[];proper=[];coplanar=[]
    for start in range(0,len(pairs),8192):
        p=pairs[start:start+8192];hit,cross,cop,w=relations(a[p[:,0]],b[p[:,1]])
        contacts.extend(p[hit].tolist());coplanar.extend(p[cop].tolist())
        for j in np.flatnonzero(cross):
            if len(proper)<8:proper.append({'triangle_a':int(p[j,0]),'triangle_b':int(p[j,1]),'point_BASE_m':w[j].tolist()})
    if len(contacts):return {'candidate_pairs':len(pairs),'contact_triangle_pairs':len(contacts),'coplanar_pairs':len(coplanar),'proper_crossing_witnesses':proper,'surface_distance_m':0.}
    # A valid vertex-vertex upper bound lets the broad phase retain every pair
    # that could improve the global surface minimum.
    upper=float(cKDTree(b.reshape(-1,3)).query(a.reshape(-1,3))[0].min())
    pairs=candidates(a,b,upper);best=upper
    for start in range(0,len(pairs),8192):
        p=pairs[start:start+8192];best=min(best,float(distances(a[p[:,0]],b[p[:,1]]).min()))
    return {'candidate_pairs':len(pairs),'contact_triangle_pairs':0,'coplanar_pairs':0,'proper_crossing_witnesses':[],'surface_distance_m':best}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();model=load_model(MODEL);records=json.loads(LIBRARY.read_text())['records'];assert len(records)==35
    names=['RADAR_BRACKET','RF_HIP','LF_HIP'];meshpaths={n:MODEL.parent.parent/'meshes'/(n+'.STL') for n in names}
    meshes={n:trimesh.load(p,process=True) for n,p in meshpaths.items()};metadata={n:{'faces':len(m.faces),'vertices':len(m.vertices),'watertight':bool(m.is_watertight),'winding_consistent':bool(m.is_winding_consistent),'components':len(m.split(only_watertight=False))} for n,m in meshes.items()}
    assert meshes['RADAR_BRACKET'].is_watertight
    bids={n:mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,n) for n in names}
    rows=[];arrays={};cache={}
    for k,rec in enumerate(records):
        d,rb,pb,*_=frame(model,rec['measured_qpos']);transformed={};transforms={}
        for n,m in meshes.items():
            bid=bids[n];r=rb.T@d.xmat[bid].reshape(3,3);t=rb.T@(d.xpos[bid]-pb)
            transforms[n]=(r,t);transformed[n]=(m.vertices@r.T+t)[m.faces]
        for n in names[1:]:
            key=tuple(np.round(np.r_[transforms[n][0].ravel(),transforms[n][1]],12))
            if key in cache:result=json.loads(json.dumps(cache[key]))
            else:
                result=inspect(transformed[names[0]],transformed[n])
                # Surface crossings are definite geometric witnesses; only test
                # component containment when surfaces are disjoint. Open HIPs
                # remain uncertified as solids even with positive surface distance.
                inside=[]
                if result['contact_triangle_pairs']==0:
                    components=meshes[n].split(only_watertight=False);sample=np.array([c.vertices[0] for c in components]);r,t=transforms[n]
                    w=winding(sample@r.T+t,transformed['RADAR_BRACKET']);inside=np.flatnonzero(np.abs(w)>.5).tolist()
                result['hip_component_representatives_inside_bracket']=inside
                result['solid_relation_certified']=False
                result['classification']=('original_surfaces_cross' if result['proper_crossing_witnesses'] else 'surface_contact_or_coplanar_overlap' if result['contact_triangle_pairs'] else 'surface_disjoint_containment_not_fully_certified')
                cache[key]=result
            row={'case':rec['id'],'pair':['RADAR_BRACKET',n],**result};rows.append(row)
            if result['proper_crossing_witnesses']:
                witness=result['proper_crossing_witnesses'][0];aidx=witness['triangle_a'];bidx=witness['triangle_b']
                arrays[f'{k}/{n}/triangle_a']=transformed['RADAR_BRACKET'][aidx];arrays[f'{k}/{n}/triangle_b']=transformed[n][bidx];arrays[f'{k}/{n}/point']=np.array(witness['point_BASE_m'])
        print(k+1,rec['id'],rows[-2]['classification'],rows[-1]['classification'],flush=True)
    np.savez_compressed(out/'witnesses.npz',**arrays)
    report={'status':'static_original_surface_audit_completed','scope':__doc__,'epsilon_m':EPS,'mesh_metadata':metadata,'states':35,'pairs':70,'rows':rows,'elapsed_s':time.monotonic()-start,
        'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [MODEL,LIBRARY,Path(__file__),*meshpaths.values()]}}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');(out/'mesh_relations.py').write_text(Path(__file__).read_text())

if __name__=='__main__':main()
