"""Approved whole-corpus, static, paired oracle comparison (CPU only)."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from hexapod_mjlab.tasks.envelope_adaptive_2 import contracts
from hexapod_mjlab.tasks.envelope_adaptive_2.config.parameters import EA2Cfg
from hexapod_mjlab.tasks.envelope_adaptive_2.config.condition import ENVELOPE_SPEC
from hexapod_mjlab.tasks.envelope_adaptive_2.map_generator import generate_map
from hexapod_mjlab.tasks.envelope_adaptive_2.oracle import compute_direct_oracle_params_with_stats
from hexapod_mjlab.tasks.envelope_adaptive_2.sl.dataset import load_map
from hexapod_mjlab.tasks.envelope_adaptive_2.experiments.static_qp import StaticOracle, MIN, SPAN, NOMINAL_S
from whole_polygon_reference import WholePolygonChecker


def area(params):
    return ((params[:,0]+params[:,1])*params[:,3]
            +(params[:,2]+params[:,1])*np.abs(params[:,4]))


def old_raw(heading,pos,field,ray_margin,check_margin):
    cfg=EA2Cfg().envelope
    start=perf_counter()
    with torch.no_grad():
        p,_=compute_direct_oracle_params_with_stats(
            torch.tensor(heading,dtype=torch.float32),torch.tensor(pos,dtype=torch.float32),field,
            torch.tensor(ENVELOPE_SPEC.low[:5]),torch.tensor(ENVELOPE_SPEC.high[:5]),
            margin=ray_margin,collision_margin=check_margin,
            collision_soft_margin=cfg.soft_margin,step=cfg.oracle_step,max_dist=cfg.oracle_max_dist,
            soft_dof_pos_limit=cfg.soft_dof_pos_limit,interp_crossing=cfg.oracle_interp_crossing,
            group_mode=cfg.oracle_group_mode)
    return p.numpy().astype(float),perf_counter()-start


def mean(values):
    return float(np.mean(values)) if len(values) else None


def describe(values):
    a=np.asarray(values)
    return {'count':len(a),'mean':mean(a),'median':float(np.median(a)) if len(a) else None,
            'p95':float(np.quantile(a,.95)) if len(a) else None,
            'max':float(a.max()) if len(a) else None}


def variation(s,index_map,mask=None):
    values=[]
    for (t,env),i in index_map.items():
        before=index_map.get((t-1,env))
        if before is not None and (mask is None or (mask[i] and mask[before])) and np.isfinite(s[[i,before]]).all():
            values.append(float(np.max(np.abs(s[i]-s[before]))))
    return describe(values)


def paired_metrics(arrays,margin,index_map):
    """Same pose/margin and same adjacent pair denominators for both methods."""
    name=f'new_{margin:.2f}';old=f'old_matched_{margin:.2f}'
    s=arrays[name+'_s'];old_s=arrays[old+'_s'];old_clearance=arrays[old+'_clearance']
    finite=np.isfinite(s).all(-1);common=finite & (old_clearance>=margin-1e-6)
    floor_ok=arrays['minimum_clearance']>=margin-1e-6
    unsafe_floor_ok=(old_clearance<margin-1e-6)&floor_ok
    n=len(s);floor_count=int(floor_ok.sum());unsafe_count=int(unsafe_floor_ok.sum())
    return {'paired_common_safe_frames':int(common.sum()),'total_frames':n,
            'paired_common_safe_fraction':float(common.mean()),
            'minimum_feasible_frames':floor_count,'minimum_infeasible_fraction':float((~floor_ok).mean()),
            'old_unsafe_minimum_feasible_frames':unsafe_count,
            'old_unsafe_and_minimum_feasible_fraction_of_all':unsafe_count/n,
            'old_unsafe_fraction_given_minimum_feasible':unsafe_count/floor_count if floor_count else None,
            'old_deviation':mean(np.mean((old_s[common]-NOMINAL_S)**2,axis=-1)),
            'new_deviation':mean(np.mean((s[common]-NOMINAL_S)**2,axis=-1)),
            'old_area_m2':mean(area((MIN+SPAN*old_s)[common])),
            'new_area_m2':mean(area((MIN+SPAN*s)[common])),
            'new_deviation_better_fraction':mean(np.mean((s[common]-NOMINAL_S)**2,axis=-1)<np.mean((old_s[common]-NOMINAL_S)**2,axis=-1)-1e-8),
            'paired_adjacent_change':{'old':variation(old_s,index_map,common),'new':variation(s,index_map,common)}}


def evaluate_map(data,map_geometry,out):
    t_ids,e_ids=np.nonzero(~data.done[30:].numpy());t_ids=t_ids+30
    index_map={(int(t),int(e)):i for i,(t,e) in enumerate(zip(t_ids,e_ids))}
    heading=data.heading.numpy()[t_ids,e_ids].astype(float)
    pos=data.pos.numpy()[t_ids,e_ids].astype(float)
    n=len(pos);checker=WholePolygonChecker(map_geometry.rects)
    floor_clearance=np.array([checker.clearance(MIN,p,h) for p,h in zip(pos,heading)])
    print(f'map {data.seed}: {n} frames, minimum geometry checked',flush=True)
    arrays={'time':t_ids,'env':e_ids,'pos':pos,'heading':heading,'minimum_clearance':floor_clearance}
    metadata={};violations=[]
    old_specs=[('old_original',.2,.1),('old_matched_0.10',.1,.1),('old_matched_0.20',.2,.2)]
    for name,ray,check in old_specs:
        params,seconds=old_raw(heading,pos,data.distance_field,ray,check)
        assert np.isfinite(params).all()
        s=(params-MIN)/SPAN
        clearance=np.array([checker.clearance(x,p,h) for x,p,h in zip(params,pos,heading)])
        arrays[name+'_s']=s;arrays[name+'_clearance']=clearance
        metadata[name]={'ray_margin':ray,'check_margin':check,'raw_batch_seconds':seconds,
                        'normalised_deviation':mean(np.mean((s-NOMINAL_S)**2,axis=-1)),
                        'area_m2':mean(area(params)),'clearance_m':describe(clearance),
                        'adjacent_max_s_change':variation(s,index_map),
                        'unsafe_fraction_by_margin':{str(m):float((clearance<m-1e-6).mean()) for m in [.1,.2]}}
        print(f'map {data.seed}: {name} complete',flush=True)
    comparisons={}
    for margin in [.1,.2]:
        name=f'new_{margin:.2f}';candidate=StaticOracle(map_geometry.rects,margin)
        s=np.full((n,5),np.nan);clearance=np.full(n,np.nan);construction=[];solve=[];check_times=[];residual=[];planes=[];statuses=[]
        for i,(p,h) in enumerate(zip(pos,heading)):
            result=candidate.solve(p,h);statuses.append(result.status)
            construction.append(result.construction_s);solve.append(result.solve_s);planes.append(result.plane_count)
            if result.status=='feasible':
                start=perf_counter();gap=checker.clearance(result.params,p,h);check_times.append(perf_counter()-start)
                s[i]=result.s;clearance[i]=gap;residual.append(result.constraint_residual_m)
                if gap<margin-1e-6 or np.any(result.s < -1e-6) or np.any(result.s > NOMINAL_S+1e-6):
                    violations.append({'seed':data.seed,'time':int(t_ids[i]),'env':int(e_ids[i]),'margin':margin,'type':'output_safety_or_bounds','clearance':gap})
            elif result.status=='minimum_infeasible':
                if floor_clearance[i]>=margin+1e-6:
                    violations.append({'seed':data.seed,'time':int(t_ids[i]),'env':int(e_ids[i]),'margin':margin,'type':'false_minimum_infeasible','clearance':floor_clearance[i]})
            else:
                violations.append({'seed':data.seed,'time':int(t_ids[i]),'env':int(e_ids[i]),'margin':margin,'type':result.status})
            if (i+1)%1000==0:print(f'map {data.seed}: {name} {i+1}/{n}',flush=True)
        finite=np.isfinite(s).all(-1);arrays[name+'_s']=s;arrays[name+'_clearance']=clearance;arrays[name+'_status']=np.asarray(statuses)
        params=MIN+SPAN*s
        metadata[name]={'margin':margin,'status_counts':dict(Counter(statuses)),
                        'normalised_deviation_on_returned':mean(np.mean((s[finite]-NOMINAL_S)**2,axis=-1)),
                        'area_m2_on_returned':mean(area(params[finite])),
                        'constraint_residual_m':describe(residual),'planes':describe(planes),
                        'construction_seconds':describe(construction),'solve_seconds':describe(solve),
                        'independent_check_seconds_on_returned':describe(check_times),
                        'adjacent_max_s_change':variation(s,index_map)}
        comparisons[str(margin)]=paired_metrics(arrays,margin,index_map)
        print(f'map {data.seed}: {name} complete, statuses={Counter(statuses)}',flush=True)
    np.savez_compressed(out/f'map_seed{data.seed}.npz',**arrays)
    return {'frames':n,'algorithms':metadata,'paired':comparisons,'correctness_violations':violations}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir',type=Path,default=Path('logs/ea2/data/ea2-migration-small-20261002'))
    ap.add_argument('--out-dir',type=Path,default=Path('logs/ea2/oracle_static/20261003'))
    args=ap.parse_args();torch.set_num_threads(1);args.out_dir.mkdir(parents=True,exist_ok=True)
    pipeline=json.loads(Path('logs/ea2/acceptance/ea2-migration-small-20261002/pipeline.json').read_text())
    report={'result':'running','nominal_s':NOMINAL_S,'objective':'0.5*sum((s-nominal_s)^2)',
            'temporal_terms':False,'new_margins_m':[.1,.2],'data_sha256':{},'maps':{},
            'limitations':['Conservative seed-dependent convex region, not global optimal free shape',
                           'Continuous geometry vs grid oracle; differences cannot be attributed solely to QP',
                           'CPU prototype timings, no GPU or real-time deployment claim']}
    path=args.out_dir/'comparison.json'
    save=lambda:path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()
    try:
        for seed in [1,7]:
            file=args.data_dir/f'map_seed{seed}.pt';digest=hashlib.sha256(file.read_bytes()).hexdigest()
            assert digest==pipeline['data_sha256'][str(file.resolve())]
            report['data_sha256'][str(file.resolve())]=digest
            data=load_map(str(file));assert data.obs.shape==(800,8,190)
            cfg=data.meta['environment_config']
            mapcfg=contracts.MapGenCfg(**cfg['map'])
            pillar=contracts.PillarFieldCfg(**{k.removeprefix('pillar_'):v for k,v in cfg['obstacles'].items()})
            geometry=generate_map(mapcfg,pillar,seed)
            assert np.array_equal(geometry.distance_field,data.distance_field.numpy())
            report['maps'][str(seed)]=evaluate_map(data,geometry,args.out_dir);save()
        assert sum(m['frames'] for m in report['maps'].values())==12320
        failures=[v for m in report['maps'].values() for v in m['correctness_violations']]
        report['result']='passed' if not failures else 'failed';save()
        if failures:raise AssertionError(f'{len(failures)} correctness failures; evidence preserved')
        print(f'FULL STATIC COMPARISON PASS {path}',flush=True)
    except Exception as error:
        report['result']='failed';report['error']=repr(error);save();raise


if __name__=='__main__':main()
