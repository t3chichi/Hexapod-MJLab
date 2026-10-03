#!/usr/bin/env python3
"""Export actual accepted witnesses, their validation levels and observed demands."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from morphology.meter import DEFAULT_XML, JOINT_NAMES
from morphology.hexfit import PARAM_KEYS, fit_collision_hexagon
from morphology.provenance import identity, write_new


def motion_shape(rows):
    pieces=[]
    for r in rows:
        p=r['hex'];fw,mw,bw,fl,bl=[p[k] for k in PARAM_KEYS]
        pieces.extend([np.array([[0,-mw],[fl,-fw],[fl,fw],[0,mw]]),np.array([[bl,-bw],[0,-mw],[0,mw],[bl,bw]])])
    return fit_collision_hexagon(pieces)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--xml',type=Path,default=DEFAULT_XML)
    ap.add_argument('--holds',type=Path,required=True)
    ap.add_argument('--gaits',type=Path,required=True)
    ap.add_argument('--sampling',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    h=json.loads(a.holds.read_text());g=json.loads(a.gaits.read_text());s=json.loads(a.sampling.read_text())
    cases={c['id']:c for c in h['cases']};accepted={r['id']:r for r in h['results'] if r['verdict']=='PASS'}
    motions={r['id']:r for r in g['results'] if r['verdict']=='PASS'}
    records=[];angles=[];speed=[];torque=[]
    for name,r in accepted.items():
        qpos=r['final_qpos'];record=dict(id=name,category=cases[name].get('category','sampled_or_legacy'),validation_level='static_hold',
            measured_E=[*[r['final_hex'][k] for k in PARAM_KEYS],qpos[2]],height_semantics='root vertical height above explicit zero ground plane',
            measured_qpos=qpos,joint_targets=cases[name]['joint_angles'],actual_support_margin=r['final_actual_support_margin'],
            static_required_peak_torque_Nm=r['static_certificate']['max_joint_torque'],hold_applied_peak_torque_Nm=r['peak_torque'],
            source_result=dict(file=a.holds.name,id=name),global_or_hardware_certificate=False)
        angles.extend(r['trajectory']['joint_angles']);speed.extend(r['trajectory']['joint_velocity']);torque.extend(r['trajectory']['torques'])
        if name in motions:
            motion=motions[name];record.update(validation_level='scripted_slow_crawl',motion_envelope=motion_shape(motion['trajectory']),motion_envelope_sample_dt_s=.02,
                motion_envelope_is_continuous_certificate=False,motion_actual_progress_m=motion['forward_progress'],motion_required_peak_torque_Nm=motion['peak_torque'],motion_source_result=dict(file=a.gaits.name,id=name))
            angles.extend(t['q'] for t in motion['trajectory']);speed.extend(t['qd'] for t in motion['trajectory']);torque.extend(t['applied_torque'] for t in motion['trajectory'])
        records.append(record)
    if not records:raise RuntimeError('no accepted static witness')
    E=np.array([r['measured_E'] for r in records]);q=np.array(angles);v=np.array(speed);tau=np.array(torque)
    joint_demands={n:dict(observed_min_angle_rad=float(q[:,i].min()),observed_max_angle_rad=float(q[:,i].max()),observed_peak_speed_rad_s=float(np.abs(v[:,i]).max()),observed_peak_applied_torque_Nm=float(np.abs(tau[:,i]).max()),is_mechanical_limit=False) for i,n in enumerate(JOINT_NAMES)}
    minimum=E.min(axis=0);maximum=E.max(axis=0)
    # Signed rear coordinate decreases with expansion, matching EA2's 0/1 convention.
    expansion_minimum=minimum.copy();expansion_maximum=maximum.copy();expansion_minimum[4],expansion_maximum[4]=maximum[4],minimum[4]
    out=dict(schema_version=2,records=records,joint_observed_demands=joint_demands,
        normalization=dict(order=[*PARAM_KEYS,'root_height'],expansion_minimum=expansion_minimum.tolist(),expansion_maximum=expansion_maximum.tolist(),formula='(E-expansion_minimum)/(expansion_maximum-expansion_minimum)',semantics='dataset scaling only; rear 0=less extension, 1=more extension; height 0=lower, 1=higher; do not clip safety errors',height_reference_z0=None),
        observed_marginals=dict(minimum=minimum.tolist(),maximum=maximum.tolist(),is_reachable_cartesian_box=False,is_global_bound=False),
        counts=dict(ideal_geometric_witnesses=len(s['witnesses']),accepted_hold_witnesses=len(records),accepted_slow_crawl_witnesses=sum(r['validation_level']=='scripted_slow_crawl' for r in records)),
        assumptions='configured collision bodies/exclusions, friction 1.0, numerical [-pi,pi] search, uncalibrated actuator clip 80Nm with 64Nm peak acceptance; no hardware constraints inferred',
        lookup_contract='Full joint/root witnesses may be selected and revalidated for a target E; marginal extrema cannot be mixed into a synthetic configuration. Motion requires its swept envelope and transition verification, not just final static E.',
        provenance=identity(a.xml,{'purpose':'validated witness library; z0 and final task ranges not predetermined'},[a.holds,a.gaits,a.sampling]))
    write_new(a.out,out);print('LIBRARY',out['counts'],flush=True)


if __name__=='__main__':main()
