"""Frozen-corpus refinement audit, including causally replayed temporal modes."""
import argparse
from collections import Counter
import hashlib,json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
import scipy
from hexapod_mjlab.tasks.envelope_adaptive_2 import contracts
from hexapod_mjlab.tasks.envelope_adaptive_2.map_generator import generate_map
from hexapod_mjlab.tasks.envelope_adaptive_2.sl.dataset import load_map
from hexapod_mjlab.tasks.envelope_adaptive_2.experiments.static_qp import MIN,SPAN,NOMINAL_S
from hexapod_mjlab.tasks.envelope_adaptive_2.experiments.refined_oracle import RefinedOracle
from hexapod_mjlab.tasks.envelope_adaptive_2.experiments.causal_oracle import CausalOracle
from hexapod_mjlab.tasks.envelope_adaptive_2.experiments.lookahead_oracle import LookaheadOracle
from hexapod_mjlab.tasks.envelope_adaptive_2.experiments.hysteretic_oracle import HystereticOracle
from whole_polygon_reference import WholePolygonChecker
from static_oracle_compare import area,describe,mean

SOURCE=Path('logs/ea2/oracle_static/20261003')
DATA=Path('logs/ea2/data/ea2-migration-small-20261002')
OUT=Path('logs/ea2/oracle_refinement/20261003')
RUN_PROVENANCE={}


def dynamics(s,time,env,mask):
    pairs=[];triples=[];index={(int(t),int(e)):i for i,(t,e) in enumerate(zip(time,env))}
    for (t,e),i in index.items():
        j=index.get((t-1,e));k=index.get((t-2,e))
        if j is not None and mask[i] and mask[j]:
            pairs.append(s[i]-s[j])
            if k is not None and mask[k]:triples.append(s[i]-2*s[j]+s[k])
    ds=np.asarray(pairs).reshape(-1,5);dds=np.asarray(triples).reshape(-1,5)
    return {'adjacent_pairs':len(ds),'triple_count':len(dds),
        'normalised_jump':describe(np.max(np.abs(ds),axis=-1)),
        'normalised_total_variation_per_pair':float(np.abs(ds).sum(-1).mean()) if len(ds) else None,
        'physical_speed_max_m_s':describe(np.max(np.abs(ds*SPAN)/.02,axis=-1)),
        'physical_acceleration_max_m_s2':describe(np.max(np.abs(dds*SPAN)/.02**2,axis=-1))}


def run(seed,margin,kind,weight,stride,tag,horizon,deadband,continuous_preview,soft_deadband,cooldown,yaw_reserve=0.,headroom=0.):
    if kind!='lookahead' and (headroom or yaw_reserve):raise ValueError('headroom/yaw_reserve are only implemented by lookahead; refusing misleading config')
    prefix=f'{kind}_w{weight:g}_seed{seed}_m{margin:.2f}_stride{stride}_{tag}'
    if (OUT/f'{prefix}.npz').exists() or (OUT/f'{prefix}.json').exists():
        raise FileExistsError(f'{prefix}: choose a fresh --tag to preserve previous evidence')
    data=load_map(str(DATA/f'map_seed{seed}.pt'));cfg=data.meta['environment_config']
    geometry=generate_map(contracts.MapGenCfg(**cfg['map']),contracts.PillarFieldCfg(**{k.removeprefix('pillar_'):v for k,v in cfg['obstacles'].items()}),seed)
    base=dict(np.load(SOURCE/f'map_seed{seed}.npz'));pick=np.arange(0,len(base['time']),stride)
    base={k:v[pick] for k,v in base.items()};n=len(pick)
    index={(int(t),int(e)):i for i,(t,e) in enumerate(zip(base['time'],base['env']))}
    legacy=data.target.numpy()[base['time'],base['env']].astype(float)
    solver=RefinedOracle(geometry.rects,margin);checker=WholePolygonChecker(geometry.rects)
    causal=CausalOracle(geometry.rects,margin,horizon,weight,cooldown=cooldown,deadband=deadband,continuous_preview=continuous_preview,soft_deadband=soft_deadband) if kind=='causal' else LookaheadOracle(geometry.rects,margin,horizon,weight,cooldown=cooldown,yaw_reserve=yaw_reserve,headroom=headroom) if kind=='lookahead' else HystereticOracle(geometry.rects,margin,horizon,weight,cooldown=cooldown) if kind=='hysteretic' else None
    static_file=OUT/f'static_w0_seed{seed}_m{margin:.2f}_stride1.npz'
    static_saved=dict(np.load(static_file)) if causal and static_file.exists() else None
    static_index={(int(t),int(e)):i for i,(t,e) in enumerate(zip(static_saved['time'],static_saved['env']))} if static_saved else {}
    outputs=np.full((n,5),np.nan);gaps=np.full(n,np.nan);status=np.full(n,'not_run',dtype='<U32')
    emergencies=np.zeros(n,bool);reasons=[];timings=[];iterations=[];failures=[]
    raw=base[f'new_{margin:.2f}_s'];legacy_gap=np.array([checker.clearance(MIN+SPAN*s,p,h) for s,p,h in zip(legacy,base['pos'],base['heading'])])
    start=perf_counter();previous={};last_t={};preview_used=0;preview_infeasible=0
    sequence=((t,e) for t in range(data.obs.shape[0]) for e in range(data.obs.shape[1])) if kind in ['temporal','causal','lookahead','hysteretic'] else index.keys()
    for t,e in sequence:
        i=index.get((t,e));pos=data.pos[t,e].numpy().astype(float);yaw=float(data.heading[t,e])
        if kind in ['temporal','causal','lookahead','hysteretic'] and (bool(data.done[t,e]) or last_t.get(e,t-1)!=t-1):
            previous.pop(e,None)
            if causal:causal.reset([e])
        seed_s=raw[i] if i is not None and np.isfinite(raw[i]).all() else None
        prev=previous.get(e) if kind=='temporal' else None
        if causal:
            ref_i=static_index.get((t,e));ref=static_saved['s'][ref_i] if ref_i is not None and np.isfinite(static_saved['s'][ref_i]).all() else None
            result=causal.step(pos,yaw,data.obs[t,e,-3:].numpy()*[1.5,1.,1.5],e,seed=seed_s,reference=ref)
            if i is not None:
                preview_used+=int(causal.last_debug['preview_used']);preview_infeasible+=int(causal.last_debug['preview_infeasible'])
        else:result=solver.solve(pos,yaw,seed=seed_s,prev=prev,weight=weight)
        if kind in ['temporal','causal','lookahead','hysteretic']:
            if result.status=='feasible':previous[e]=result.s.copy()
            else:previous.pop(e,None)
            last_t[e]=t
        if i is None:continue
        status[i]=result.status;emergencies[i]=result.emergency;reasons.append(result.reason)
        timings.append(result.elapsed_s);iterations.append(result.iterations)
        if result.status=='feasible':
            outputs[i]=result.s;gaps[i]=checker.clearance(MIN+SPAN*result.s,pos,yaw)
            if gaps[i]<margin-1e-6 or np.any(result.s<-1e-6) or np.any(result.s>NOMINAL_S+1e-6):failures.append({'time':t,'env':e,'type':'safety_or_bounds','gap':float(gaps[i])})
            if causal and getattr(causal,'headroom',0.)>0:
                residual=float(np.max(causal.solver.extra_a@result.s-causal.solver.extra_b))
                if residual>1e-8:failures.append({'time':t,'env':e,'type':'headroom_constraint','residual':residual})
        elif result.status!='minimum_infeasible' or base['minimum_clearance'][i]>=margin+1e-6:
            failures.append({'time':t,'env':e,'type':result.status})
        if (i+1)%1000==0:print(seed,margin,kind,weight,i+1,n,flush=True)
    finite=np.isfinite(outputs).all(-1);common=finite&np.isfinite(raw).all(-1)
    lost=np.flatnonzero(np.isfinite(raw).all(-1)&~finite)
    gained=np.flatnonzero(finite&~np.isfinite(raw).all(-1))
    for j in lost:failures.append({'time':int(base['time'][j]),'env':int(base['env'][j]),'type':'coverage_regression'})
    new_d=np.mean((outputs[common]-NOMINAL_S)**2,axis=-1);old_d=np.mean((raw[common]-NOMINAL_S)**2,axis=-1)
    new_a=area(MIN+SPAN*outputs[common]);old_a=area(MIN+SPAN*raw[common])
    if kind=='static':
        worse=np.flatnonzero(new_d>old_d+1e-8)
        for i in np.flatnonzero(common)[worse]:failures.append({'index':int(i),'time':int(base['time'][i]),'env':int(base['env'][i]),'type':'static_objective_regression'})
    legacy_safe=common&(legacy_gap>=margin-1e-6)
    report={'seed':seed,'margin':margin,'kind':kind,'weight':weight,'stride':stride,'tag':tag,'frames':n,
        'selection':'every valid frozen pose' if stride==1 else f'pilot uniform stride {stride}, not final evidence',
        'solver_config':{'max_refinement_iters':solver.max_iters,'dt_s':.02,'shrink_m_s':2.,'grow_m_s':.5,
            'nominal_s':NOMINAL_S,'bounds_s':[0,NOMINAL_S],'normalised_weights':'equal',
            'projection_ftol':1e-11,'projection_max_iters':100,'temporal_weight':weight,'prediction_horizon_s':horizon if causal else 0,
            'emergency_policy':'preserve growth bound; relax shrink with weight max(49,temporal_weight)',
            'max_emergency_refinement_stages':1,'lookahead_common_shape_samples':3 if kind in ['lookahead','hysteretic'] else None},
        'run_provenance':RUN_PROVENANCE,
        'input_sha256':{'frozen_map':hashlib.sha256((DATA/f'map_seed{seed}.pt').read_bytes()).hexdigest(),
            'first_qp_npz':hashlib.sha256((SOURCE/f'map_seed{seed}.npz').read_bytes()).hexdigest()},
        'coverage_gained_over_first_qp':int(len(gained)),'coverage_lost_to_first_qp':int(len(lost)),
        'correctness_failures':failures,'status_counts':dict(Counter(status.tolist())),
        'reason_counts':dict(Counter(reasons)),'emergencies':int(emergencies.sum()),'iterations':describe(iterations),
        'causal_config':{'horizon_s':horizon,'desired_deadband_normalised':deadband,'soft_desired_deadband_normalised':soft_deadband,'continuous_preview':continuous_preview,'preview_mix':.5,'trigger_prev_min_s':.5,'growth_cooldown_s':cooldown,
            'preview_model':'constant world velocity and yaw rate from current observable ego; legacy right-positive lateral decoded',
            'preview_used_frames':preview_used,'preview_minimum_infeasible_frames':preview_infeasible,
            'current_static_reference_sha256':hashlib.sha256(static_file.read_bytes()).hexdigest() if causal and static_file.exists() else None} if causal else None,
        'candidate_seconds':describe(timings),'wall_seconds':perf_counter()-start,
        'legacy_smoothed_safe_frames':int((legacy_gap>=margin-1e-6).sum()),
        'paired_with_first_qp':{'frames':int(common.sum()),'old_mse':float(old_d.mean()),'new_mse':float(new_d.mean()),
            'deviation_change_percent':float((new_d.mean()/old_d.mean()-1)*100),'old_area_m2':float(old_a.mean()),
            'new_area_m2':float(new_a.mean()),'area_change_percent':float((new_a.mean()/old_a.mean()-1)*100),
            'objective_better_frames':int((new_d<old_d-1e-8).sum()),'objective_worse_frames':int((new_d>old_d+1e-8).sum()),
            'old_dynamics':dynamics(raw,base['time'],base['env'],common),'new_dynamics':dynamics(outputs,base['time'],base['env'],common)},
        'paired_legacy_safe':{'frames':int(legacy_safe.sum()),'legacy_area':mean(area(MIN+SPAN*legacy[legacy_safe])),
            'candidate_area':mean(area(MIN+SPAN*outputs[legacy_safe])),
            'legacy_mse':mean(np.mean((legacy[legacy_safe]-NOMINAL_S)**2,axis=-1)),
            'candidate_mse':mean(np.mean((outputs[legacy_safe]-NOMINAL_S)**2,axis=-1)),
            'legacy_dynamics':dynamics(legacy,base['time'],base['env'],legacy_safe),
            'candidate_dynamics':dynamics(outputs,base['time'],base['env'],legacy_safe)}}
    if kind in ['lookahead','hysteretic']:
        report['causal_config']={'horizon_s':horizon,'growth_cooldown_s':cooldown,'near_floor_headroom':headroom,'floor_cap':.05,'headroom_constraints':'fwd<=slope*fw+.05; back<=slope*bw+.05; slope=(.975-.05)/headroom if enabled','yaw_reserve_max_rad':yaw_reserve,'yaw_reserve_minimum_clearance_trigger_m':[margin+.05,margin+.15],'forecast_sample_times':[0,horizon/2,horizon],'additional_heading_sample_times':[0,horizon] if yaw_reserve else [],'additional_heading_offsets':'plus/minus reserve * clip((margin+.15-minimum_gap)/.1,0,1)','preview_model':'current observable right-positive legacy ego decoded into constant world velocity/yaw rate','preview_used_frames':preview_used,'preview_minimum_infeasible_frames':preview_infeasible,'goal':'common-shape nominal optimum' if kind=='lookahead' else 'safe mode monotonic growth; nearest shrink-only common shape otherwise','output_safety':'current geometry only; future goal is heuristic; no swept guarantee','static_cache_used':False}
    prefix=f'{kind}_w{weight:g}_seed{seed}_m{margin:.2f}_stride{stride}_{tag}'
    np.savez_compressed(OUT/f'{prefix}.npz',time=base['time'],env=base['env'],pos=base['pos'],heading=base['heading'],
        s=outputs,clearance=gaps,status=status,emergency=emergencies,first_qp_s=raw,legacy_s=legacy,legacy_clearance=legacy_gap)
    (OUT/f'{prefix}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('RESULT',prefix,'mse',report['paired_with_first_qp']['deviation_change_percent'],'area',report['paired_with_first_qp']['area_change_percent'],'emergency',int(emergencies.sum()),'failures',len(failures),flush=True)
    assert not failures,failures[:5]
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--kind',choices=['static','temporal','causal','lookahead','hysteretic'],default='static')
    ap.add_argument('--weight',type=float,default=0);ap.add_argument('--seeds',type=int,nargs='+',default=[1,7]);ap.add_argument('--margins',type=float,nargs='+',default=[.1,.2]);ap.add_argument('--stride',type=int,default=1)
    ap.add_argument('--tag',default='rate_seed_v2',help='versioned run label; existing evidence is never overwritten')
    ap.add_argument('--horizon',type=float,default=.24)
    ap.add_argument('--headroom',type=float,default=0.)
    ap.add_argument('--yaw-reserve',type=float,default=0.)
    ap.add_argument('--cooldown',type=float,default=.1)
    ap.add_argument('--soft-deadband',type=float,default=0.)
    ap.add_argument('--continuous-preview',action='store_true')
    ap.add_argument('--deadband',type=float,default=0.)
    args=ap.parse_args();torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=True)
    snapshot=OUT/'source_snapshots';snapshot.mkdir(exist_ok=True)
    for file in [Path(__file__),*[Path('src/hexapod_mjlab/tasks/envelope_adaptive_2/experiments')/f for f in ['refined_oracle.py','causal_oracle.py','lookahead_oracle.py','hysteretic_oracle.py','static_qp.py']],*Path('src/hexapod_mjlab/tasks/envelope_adaptive_2/teachers').glob('*.py')]:
        content=file.read_bytes();digest=hashlib.sha256(content).hexdigest();(snapshot/f'{digest}.py').write_bytes(content)
        RUN_PROVENANCE[str(file)]=digest
    RUN_PROVENANCE['versions']={'numpy':np.__version__,'scipy':scipy.__version__,'torch':torch.__version__}
    provenance=json.loads((SOURCE/'comparison.json').read_text())['data_sha256']
    for seed in args.seeds:
        path=DATA/f'map_seed{seed}.pt';assert hashlib.sha256(path.read_bytes()).hexdigest()==provenance[str(path.resolve())]
        for margin in args.margins:run(seed,margin,args.kind,args.weight,args.stride,args.tag,args.horizon,args.deadband,args.continuous_preview,args.soft_deadband,args.cooldown,args.yaw_reserve,args.headroom)


if __name__=='__main__':main()
