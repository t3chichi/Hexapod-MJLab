#!/usr/bin/env python3
"""Validate frozen artifacts and re-run checks from the archived source/model."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from morphology.provenance import sha, write_new


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir',type=Path,required=True)
    ap.add_argument('--out-dir',type=Path,required=True)
    a=ap.parse_args();a.out_dir.mkdir(parents=True,exist_ok=False)
    run=a.run_dir.resolve();manifest=json.loads((run/'manifest.json').read_text());release=json.loads((run/'acceptance.json').read_text())
    source=run/'source_snapshot/morphology';xml=run/'model_snapshot/mjcf'/Path(manifest['model']['xml']).name
    artifact_matches={name:sha(run/name)==expected for name,expected in release['artifacts_sha256'].items()}
    sources_match={name:sha(source/name)==expected for name,expected in manifest['sources_sha256'].items()}
    meshes_match={name:sha(run/'model_snapshot/meshes'/name)==expected for name,expected in manifest['model']['meshes_sha256'].items()}
    xml_match=sha(xml)==manifest['model']['xml_sha256'];legacy_match=sha(run/'legacy_hold_input.json')==next(iter(manifest['inputs_sha256'].values()))
    initial_ok=all(artifact_matches.values()) and all(sources_match.values()) and all(meshes_match.values()) and xml_match and legacy_match and release['verdict']=='PASS' and all(release['checks'].values())
    if not initial_ok:
        write_new(a.out_dir/'verification.json',dict(passed=False,artifacts=artifact_matches,sources=sources_match,meshes=meshes_match,xml_match=xml_match,legacy_match=legacy_match))
        return 1
    runs=[]
    for name,script,args in [('core','run_checks.py',[]),('replay','replay.py',['--sampling',str(run/'sampling.json')])]:
        command=[sys.executable,str(source/script),'--xml',str(xml),*args,'--out',str(a.out_dir/f'{name}.json')]
        p=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env={**os.environ,'OMP_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'})
        (a.out_dir/f'{name}.stdout.txt').write_text(p.stdout);print(p.stdout,flush=True)
        runs.append(dict(stage=name,returncode=p.returncode,argv=command))
        if p.returncode:
            write_new(a.out_dir/'verification.json',dict(passed=False,executions=runs,reason='frozen-source execution failed'))
            return 1
    # Import only the archived implementation for exported-state and all
    # accepted motion-frame reconstruction. Model path changes are permitted
    # only after the byte identity checks above.
    package_parent=source.parent;sys.path.insert(0,str(package_parent))
    for key in list(sys.modules):
        if key=='morphology' or key.startswith('morphology.'):del sys.modules[key]
    from morphology.meter import MorphologyMeter
    from morphology.replay import filled_face_violation, force_errors
    from morphology.hexfit import PARAM_KEYS
    meter=MorphologyMeter(xml,floor=False)
    library=json.loads((run/'library.json').read_text());holds=json.loads((run/'holds.json').read_text());gaits=json.loads((run/'gaits.json').read_text())
    hpass={r['id']:r for r in holds['results'] if r['verdict']=='PASS'};gpass={r['id']:r for r in gaits['results'] if r['verdict']=='PASS'}
    failures=[];motion_frames=0;maximum=dict(static_E_m=0.,motion_geometry_m=0.,force_certificate_error=0.)
    for record in library['records']:
        p=record['measured_qpos'];meter.set_state(p[:3],p[3:7],p[7:]);state=meter.measure();E=[*[state.hex_params[k] for k in PARAM_KEYS],state.body_height]
        error=max(abs(x-y) for x,y in zip(E,record['measured_E']));maximum['static_E_m']=max(maximum['static_E_m'],error)
        hold=hpass.get(record['id']);bad=[]
        if hold is None or error>1e-7 or meter.self_collision() or meter.clearance(True)<-1e-6:bad.append('static_export')
        if hold:
            residual=max(force_errors(meter,hold['static_certificate']).values());maximum['force_certificate_error']=max(maximum['force_certificate_error'],residual)
            if residual>1e-5 or not all(hold['checks'].values()):bad.append('hold_certificate')
        if record['validation_level']=='scripted_slow_crawl':
            gait=gpass.get(record['id'])
            if not gait or not all(gait['checks'].values()):bad.append('motion_source')
            else:
                for row in gait['trajectory']:
                    meter.set_state((0,0,.5),(1,0,0,0),row['q']);v=filled_face_violation(meter,record['motion_envelope']);maximum['motion_geometry_m']=max(maximum['motion_geometry_m'],v);motion_frames+=1
                    if v>1e-7 or meter.self_collision():bad.append('motion_geometry');break
                p=gait['end_qpos'];meter.set_state(p[:3],p[3:7],p[7:]);residual=max(force_errors(meter,gait['end_static_certificate']).values());maximum['force_certificate_error']=max(maximum['force_certificate_error'],residual)
                if residual>1e-5:bad.append('motion_tail_certificate')
        if bad:failures.append(dict(id=record['id'],checks=bad))
    verdict_consistency=all((r['verdict']=='PASS')==all(r.get('checks',{}).values()) for data in (holds,gaits) for r in data['results'] if 'checks' in r)
    counts_correct=library['counts']['accepted_hold_witnesses']==len(hpass) and library['counts']['accepted_slow_crawl_witnesses']==len(gpass)
    passed=not failures and verdict_consistency and counts_correct
    write_new(a.out_dir/'verification.json',dict(passed=passed,frozen_source_and_model_reexecuted=True,executions=runs,artifact_hashes_valid=all(artifact_matches.values()),source_hashes_valid=all(sources_match.values()),model_hashes_valid=xml_match and all(meshes_match.values()),library_records=len(library['records']),accepted_motion_frames_reconstructed=motion_frames,maximum_errors=maximum,failures=failures,verdict_consistency=verdict_consistency,library_counts_correct=counts_correct))
    print('BUNDLE VERIFICATION','PASS' if passed else 'FAIL',maximum,flush=True)
    return 0 if passed else 1


if __name__=='__main__':sys.exit(main())
