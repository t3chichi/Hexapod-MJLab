#!/usr/bin/env python3
"""Bounded whole-configuration witness search on true soles and compiled collision hulls.

All accepted configurations, including extrema witnesses and unsuccessful
search cells, are saved. No bounded search is labelled proof of infeasibility
or a rectangular reachable set; limits are not imported from URDF.
"""
from __future__ import annotations
import argparse,heapq,itertools,sys,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import mujoco
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from morphology.meter import MorphologyMeter,JOINT_NAMES,LEGS,DEFAULT_XML,polygon_metrics
from morphology.hexfit import fit_collision_hexagon,PARAM_KEYS
from morphology.statics import equilibrium
from morphology.provenance import identity,write_new

WORK={}


def initialize(xml,xy_step,search_clearance,seed_count):
    m=MorphologyMeter(xml,floor=False);m.set_state((0,0,.5),(1,0,0,0),np.zeros(18));legids={leg:{mujoco.mj_name2id(m.model,mujoco.mjtObj.mjOBJ_BODY,f'{leg}_{x}') for x in ('HIP','THIGH','SHANK','FOOT')} for leg in LEGS};alllegs=set().union(*legids.values());rig={b for g,b,v in m._occ_geoms}-alllegs
    hip={leg:m.to_base_frame(m.data.xpos[mujoco.mj_name2id(m.model,mujoco.mjtObj.mjOBJ_BODY,f'{leg}_HIP')]).copy() for leg in LEGS}
    reach={}
    for leg in LEGS:
        thigh=mujoco.mj_name2id(m.model,mujoco.mjtObj.mjOBJ_BODY,f'{leg}_THIGH');shank=mujoco.mj_name2id(m.model,mujoco.mjtObj.mjOBJ_BODY,f'{leg}_SHANK');foot=m.foot_body_ids[leg]
        sole_radius=max(float(np.linalg.norm(m.world_geom(g,v)-m.data.xpos[foot],axis=1).max()) for g,v in m._body_geoms[foot])
        reach[leg]=sum(np.linalg.norm(m.model.body_pos[b]) for b in (thigh,shank,foot))+sole_radius+1e-5
    WORK.update(meter=m,legids=legids,rig=rig,hips=hip,reach=reach,step=xy_step,clearance=search_clearance,seeds=seed_count)


def cell_sample(task):
    leg,height=task;m=WORK['meter'];my=WORK['legids'][leg];rig=WORK['rig'];foot=m.foot_body_ids[leg];q=np.zeros(18);offset=JOINT_NAMES.index(f'{leg}_HAA');stats={'targets':0,'ik_targets':0,'ik_solutions':0,'rejected_collision':0,'rejected_ground':0,'retained':0};candidates=[]
    m.set_state((0,0,height),(1,0,0,0),q)
    for x in np.arange(-1.2,1.20001,WORK['step']):
        for y in np.arange(-1.,1.00001,WORK['step']):
            stats['targets']+=1
            # Conservative triangle bound with foot-radius allowance, not a
            # sampled maximum or nominal URDF range.
            if np.linalg.norm(np.array([x,y,-height])-WORK['hips'][leg])>WORK['reach'][leg]:continue
            stats['ik_targets']+=1
            sols=m.leg_ik(leg,[x,y,-height],sole_on_ground=True,seed_count=WORK['seeds']);stats['ik_solutions']+=len(sols)
            for sol in sols:
                q[:]=0;q[offset:offset+3]=[sol[f'{leg}_{k}'] for k in ('HAA','HFE','KFE')];m.set_state((0,0,height),(1,0,0,0),q)
                if any((int(m.model.geom_bodyid[r['geom1']]) in my and int(m.model.geom_bodyid[r['geom2']]) in my|rig) or (int(m.model.geom_bodyid[r['geom2']]) in my and int(m.model.geom_bodyid[r['geom1']]) in my|rig) for r in m.self_collision()):stats['rejected_collision']+=1;continue
                polys=[];clear=np.inf
                for g,b,v in m._occ_geoms:
                    if b in my|rig:
                        world=m.world_geom(g,v);polys.append(m.to_base_frame(world)[:,:2])
                        if b!=foot:clear=min(clear,float(m.signed_ground_distance(world).min()))
                if clear<WORK['clearance']:stats['rejected_ground']+=1;continue
                shape=fit_collision_hexagon(polys);candidates.append({'q':q[offset:offset+3].tolist(),'target_xy':[float(x),float(y)],'shape':shape,'nonfoot_clearance':clear})
    stats['retained']=len(candidates);return leg,height,stats,candidates


def combinations(rank_lists,cap):
    sizes=[len(x) for x in rank_lists]
    if any(n==0 for n in sizes):return
    seen={(0,)*6};heap=[(0,(0,)*6)];count=0
    while heap and count<cap:
        score,index=heapq.heappop(heap);count+=1;yield index
        for i in range(6):
            if index[i]+1<sizes[i]:
                nxt=index[:i]+(index[i]+1,)+index[i+1:]
                if nxt not in seen:seen.add(nxt);heapq.heappush(heap,(sum(nxt),nxt))


def search_height(m,cell,height,k,cap,static_budget,clearance):
    directions=[('stand_like','min'),*itertools.product(PARAM_KEYS,('min','max'))];seen={};records=[];reports={}
    m.set_state((0,0,height),(1,0,0,0),np.zeros(18));standxy={l:m.to_base_frame(m.data.xpos[b])[:2].copy() for l,b in m.foot_body_ids.items()};statics=0
    for direction_index,(key,mode) in enumerate(directions):
        quota=static_budget//len(directions)+int(direction_index<static_budget%len(directions));statics=0
        lists=[]
        for leg in LEGS:
            cs=cell[leg]
            if key=='stand_like':cs=sorted(cs,key=lambda c:(np.linalg.norm(np.array(c['target_xy'])-standxy[leg]),np.linalg.norm(c['q'])))
            else:cs=sorted(cs,key=lambda c:((1 if mode=='min' else -1)*c['shape'][key],np.linalg.norm(c['q'])))
            lists.append(cs[:k])
        counters={'combinations':0,'collision_reject':0,'ground_reject':0,'balance_reject':0,'support_reject':0,'static_budget_exhausted':0,'accepted_ids':[]}
        for index in combinations(lists,cap):
            counters['combinations']+=1;chosen=[lists[i][index[i]] for i in range(6)];q=np.array([a for c in chosen for a in c['q']]);tag=tuple(np.round(q,9))
            if tag in seen:
                if seen[tag] is not None:counters['accepted_ids'].append(seen[tag])
                continue
            seen[tag]=None;m.set_state((0,0,height),(1,0,0,0),q)
            if m.self_collision():counters['collision_reject']+=1;continue
            if m.clearance(True)<clearance:counters['ground_reject']+=1;continue
            points=np.array([m._lowest_point_of_body(b) for b in m.foot_body_ids.values()]);margin,area=polygon_metrics(points,m.data.subtree_com[m.base_body_id][:2])
            if margin is None or margin<.05:counters['support_reject']+=1;continue
            if statics>=quota:
                counters['static_budget_exhausted']+=1;seen.pop(tag,None);continue
            statics+=1;eq=equilibrium(m,torque_cap=None)
            if not eq['valid']:counters['balance_reject']+=1;continue
            shape=fit_collision_hexagon(m.collision_polygons());identifier=f'h{height:.3f}-w{len(records):04d}'
            record={'id':identifier,'root_height':height,'joint_angles':dict(zip(JOINT_NAMES,map(float,q))),'hex_params':shape,'nonfoot_clearance':m.clearance(True),'ideal_support_margin':margin,'ideal_support_area':area,'static_certificate':eq,'source':{'direction':key+'_'+mode,'candidate_indices':list(index)},'validity':'geometry_and_ideal_static_witness; PD/gait validity separate'}
            seen[tag]=identifier;records.append(record);counters['accepted_ids'].append(identifier)
        reports[key+'_'+mode]={**counters,'allocated_static_budget':quota,'static_solves':statics,'status':'found' if counters['accepted_ids'] else 'budget_not_found','global_extremum_certified':False}
    return records,reports


def aggregate(records):
    if not records:return {}
    out={}
    for key in PARAM_KEYS:
        a=min(records,key=lambda r:r['hex_params'][key]);b=max(records,key=lambda r:r['hex_params'][key]);out[key]={'observed_min':a['hex_params'][key],'min_witness_id':a['id'],'observed_max':b['hex_params'][key],'max_witness_id':b['id'],'is_global_bound':False}
    return out


def representatives(records,per_height):
    """Representative selection keeps unique actual witnesses, not mixed extrema."""
    ids=[]
    for info in aggregate(records).values():ids.extend([info['min_witness_id'],info['max_witness_id']])
    for h,recs in per_height.items():
        if recs:
            # Baseline near stand feet plus low required torque, ranked over
            # entire observed configurations rather than first per-leg rank.
            r=min(recs,key=lambda r:sum(v*v for v in r['joint_angles'].values()));ids.append(r['id'])
    lookup={r['id']:r for r in records};return [lookup[i] for i in dict.fromkeys(ids)]


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--xml',type=Path,default=DEFAULT_XML);ap.add_argument('--h-min',type=float,default=.20);ap.add_argument('--h-max',type=float,default=.64);ap.add_argument('--h-step',type=float,default=.02);ap.add_argument('--xy-step',type=float,default=.05);ap.add_argument('--k',type=int,default=8);ap.add_argument('--combo-cap',type=int,default=256);ap.add_argument('--static-budget',type=int,default=256);ap.add_argument('--ik-seeds',type=int,default=10);ap.add_argument('--clearance',type=float,default=.005);ap.add_argument('--workers',type=int,default=12);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    t=time.perf_counter();heights=np.round(np.arange(a.h_min,a.h_max+a.h_step*.1,a.h_step),4);tasks=[(l,float(h)) for h in heights for l in LEGS];cells={};counts={};totals={}
    with ProcessPoolExecutor(max_workers=a.workers,initializer=initialize,initargs=(str(a.xml),a.xy_step,a.clearance,a.ik_seeds)) as pool:
        for i,(l,h,s,cs) in enumerate(pool.map(cell_sample,tasks)):
            cells[(l,h)]=cs;counts[f'{l}@{h:.3f}']=s
            for key,n in s.items():totals[key]=totals.get(key,0)+n
            if (i+1)%6==0:print('IK',i+1,'/',len(tasks),'height',h,'retained',totals.get('retained',0),'seconds',round(time.perf_counter()-t,1),flush=True)
    print('IK COMPLETE',totals,flush=True);m=MorphologyMeter(a.xml,floor=False);allrecords=[];per_h={};search={}
    for h in heights:
        h=float(h);rs,reports=search_height(m,{l:cells[(l,h)] for l in LEGS},h,a.k,a.combo_cap,a.static_budget,a.clearance);allrecords+=rs;per_h[str(h)]=rs;search[str(h)]=reports
        print('COMBO',h,'witnesses',len(rs),'elapsed',round(time.perf_counter()-t,1),flush=True)
    data={'provenance':identity(a.xml,{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}),'semantics':{'height':'root vertical height above actual ground plane','joint_domain':[-np.pi,np.pi],'joint_stops':'URDF stops disabled; numerical period and configured collisions only','limits':'observed witness extrema, no global bound or reachable Cartesian box','equilibrium':'ideal rigid contacts with measured required torque; no URDF torque limit used to remove candidates','budget_allocation':'per-height static budget distributed equally across 11 declared directions; remainder allocated in listed order; untested candidates not cached as infeasible'},'sampling_totals':totals,'cell_counts':counts,'search_reports':search,'witnesses':allrecords,'observed_bounds':aggregate(allrecords),'representatives':representatives(allrecords,per_h),'duration_seconds':time.perf_counter()-t}
    # Small reproducible joint/target candidate arrays, no discarded hull or
    # arbitrary cloud reduction; whole geometry reconstructed from witnesses.
    candidate_path=a.out.with_suffix('.candidates.npz')
    if candidate_path.exists():raise FileExistsError(candidate_path)
    arrays={f'{l}_h{h:.3f}':np.array([[*c['q'],*c['target_xy']] for c in cs],dtype=float).reshape(-1,5) for (l,h),cs in cells.items()};np.savez_compressed(candidate_path,**arrays);data['candidate_arrays']=str(candidate_path);write_new(a.out,data);print('DONE',len(allrecords),'witnesses',flush=True)
    return 0

if __name__=='__main__':sys.exit(main())
