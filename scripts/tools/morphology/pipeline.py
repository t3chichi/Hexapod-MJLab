#!/usr/bin/env python3
"""Frozen end-to-end T1 measurement acceptance; never certifies a Cartesian range.

A PASS certifies the measurement implementation and explicitly listed witnesses
under declared simulation assumptions. Shape-library static/dynamic coverage
is reported separately; a rejected motion is never relabelled reachable.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from morphology.meter import DEFAULT_XML
from morphology.provenance import TOOLS, identity, sha, write_new

# Predeclared coverage, including deliberately aggressive counterexamples.
GAIT_IDS = ['height_0.40_width_1.00','height_0.44_width_1.00','height_0.48_width_1.00','height_0.52_width_1.00','height_0.56_width_1.00',
            'contract_0.65_h0.48','contract_0.85_h0.48','contract_0.95_h0.48','asym_+0.15_+0.05_h0.48','asym_+0.25_+0.05_h0.48','asym_-0.05_-0.15_h0.48','asym_-0.05_-0.25_h0.48']


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--xml',type=Path,default=DEFAULT_XML)
    ap.add_argument('--legacy',type=Path,default=TOOLS.parents[2]/'docs/research/morphology/evidence/hold_results.json')
    ap.add_argument('--workers',type=int,default=12)
    ap.add_argument('--out-dir',type=Path,required=True)
    a=ap.parse_args()
    a.out_dir.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();initial=identity(a.xml,dict(workers=a.workers,gait_ids=GAIT_IDS),[a.legacy])
    write_new(a.out_dir/'manifest.json',initial)
    snapshot=a.out_dir/'source_snapshot/morphology';snapshot.mkdir(parents=True)
    for p in TOOLS.glob('*.py'):shutil.copy2(p,snapshot/p.name)
    model=a.out_dir/'model_snapshot';(model/'mjcf').mkdir(parents=True)
    shutil.copy2(a.xml,model/'mjcf'/a.xml.name)
    shutil.copytree(a.xml.parent.parent/'meshes',model/'meshes')
    shutil.copy2(a.legacy,a.out_dir/'legacy_hold_input.json')
    commands=[
        ('core','run_checks.py',[]),
        ('local_ranges','analyze_ranges.py',['--step','.005']),
        ('sampling','sample_reachability.py',['--h-min','.20','--h-max','.64','--h-step','.02','--xy-step','.05','--k','8','--combo-cap','256','--static-budget','256','--ik-seeds','10','--clearance','.005','--workers',str(a.workers)]),
        ('replay','replay.py',['--sampling',str(a.out_dir/'sampling.json')]),
        ('families','build_cases.py',['--branch-budget','256']),
        ('holds','verify_hold.py',['--sampling',str(a.out_dir/'sampling.json'),'--cases',str(a.out_dir/'families.json'),'--legacy',str(a.legacy),'--heights','','--steps','1000']),
        ('gaits','verify_gait.py',['--cases',str(a.out_dir/'families.json'),'--case-ids',','.join(GAIT_IDS),'--cycles','2']),
        ('library','export_library.py',['--holds',str(a.out_dir/'holds.json'),'--gaits',str(a.out_dir/'gaits.json'),'--sampling',str(a.out_dir/'sampling.json')]),
    ]
    executions=[]
    for name,script,args in commands:
        command=[sys.executable,str(TOOLS/script),'--xml',str(a.xml),*args,'--out',str(a.out_dir/f'{name}.json')]
        t=time.perf_counter();print('START',name,flush=True)
        with (a.out_dir/f'{name}.stdout.txt').open('x') as log:
            process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env={**os.environ,'OMP_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'})
            for line in process.stdout:
                print(line,end='',flush=True);log.write(line);log.flush()
            code=process.wait()
        executions.append(dict(stage=name,argv=command,returncode=code,duration_s=time.perf_counter()-t))
        if code:
            write_new(a.out_dir/'execution_failure.json',dict(executions=executions,reason='stage exited nonzero; incomplete run, no acceptance'))
            return 1
    files={name:json.loads((a.out_dir/f'{name}.json').read_text()) for name,_,_ in commands}
    final=identity(a.xml,initial['config'],[a.legacy]);unchanged=initial['sources_sha256']==final['sources_sha256'] and initial['model']==final['model'] and initial['inputs_sha256']==final['inputs_sha256']
    artifact_hashes=all(data['provenance']['sources_sha256']==initial['sources_sha256'] and data['provenance']['model']==initial['model'] for data in files.values())
    holds=files['holds'];gaits=files['gaits'];families={c['id']:c for c in files['families']['cases']};hold_pass=[r['id'] for r in holds['results'] if r['verdict']=='PASS'];gait_pass=[r['id'] for r in gaits['results'] if r['verdict']=='PASS']
    countercases=all(any(r['id']=='legacy_'+k and r['verdict']=='FAIL' for r in holds['results']) for k in ('b1','c'))
    static_categories=sorted({families[i]['category'] for i in hold_pass if i in families})
    dynamic_categories=sorted({families[i]['category'] for i in gait_pass if i in families})
    checks=dict(core_adversarial=files['core']['passed'],every_witness_independently_replayed=files['replay']['passed'],source_model_inputs_unchanged=unchanged,all_artifacts_match_frozen_sources=artifact_hashes,known_unsafe_legacy_cases_rejected=countercases,static_family_coverage=set(static_categories)=={'height','lateral_contract','front_rear_asymmetry'},nominal_slow_crawl='height_0.48_width_1.00' in gait_pass,dynamic_family_coverage=set(dynamic_categories)=={'height','lateral_contract','front_rear_asymmetry'},validated_library_nonempty=files['library']['counts']['accepted_slow_crawl_witnesses']>0)
    result=dict(schema_version=2,verdict='PASS' if all(checks.values()) else 'FAIL',checks=checks,executions=executions,total_duration_s=time.perf_counter()-start,
                sampling_witnesses=len(files['sampling']['witnesses']),sampling_not_global_bounds=True,
                hold_pass_ids=hold_pass,hold_total=holds['total'],hold_verdict_counts={v:sum(r['verdict']==v for r in holds['results']) for v in ('PASS','FAIL','NOT_FOUND')},
                gait_pass_ids=gait_pass,gait_total=gaits['total'],gait_verdict_counts={v:sum(r['verdict']==v for r in gaits['results']) for v in ('PASS','FAIL','NOT_FOUND')},
                static_categories=static_categories,dynamic_categories=dynamic_categories,
                all_family_slow_motion_coverage_complete=set(dynamic_categories)=={'height','lateral_contract','front_rear_asymmetry'},
                acceptance_scope='T1 geometry/contact/statics implementation and listed witnesses under explicit model/friction/actuator assumptions; not global mechanical ranges, untested E combinations, transitions, real hardware, or T0 learned locomotion',
                artifacts_sha256={p.name:sha(p) for p in a.out_dir.glob('*.json')},provenance=final)
    write_new(a.out_dir/'acceptance.json',result)
    print('ACCEPTANCE',result['verdict'],'holds',len(hold_pass),'/',holds['total'],'gaits',len(gait_pass),'/',gaits['total'],flush=True)
    return 0 if result['verdict']=='PASS' else 1


if __name__=='__main__':sys.exit(main())
