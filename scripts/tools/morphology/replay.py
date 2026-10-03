#!/usr/bin/env python3
"""Independently replay every geometric witness and rigid force certificate."""
import argparse
import json
import sys
from pathlib import Path
import mujoco
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from morphology.meter import MorphologyMeter, DEFAULT_XML, JOINT_NAMES
from morphology.hexfit import PARAM_KEYS
from morphology.provenance import identity, write_new, model_content_matches


def filled_face_violation(m, p):
    worst = 0.
    # Enumerate original compiled triangle edges, independently of hexfit's
    # projection hull clipping and LP constraint construction.
    for geom, triangles in m._triangles.items():
        world = triangles.reshape(-1, 3) @ m.data.geom_xmat[geom].reshape(3, 3).T + m.data.geom_xpos[geom]
        local = m.to_base_frame(world).reshape(-1, 3, 3)
        for a, b in ((local[:, 0], local[:, 1]), (local[:, 1], local[:, 2]), (local[:, 2], local[:, 0])):
            cross = a[:, 0] * b[:, 0] < 0
            ya = a[cross, 1] - a[cross, 0] * (b[cross, 1] - a[cross, 1]) / (b[cross, 0] - a[cross, 0])
            if len(ya):
                worst = max(worst, float(np.max(np.abs(ya) - p['middle_width'])))
        pts = local.reshape(-1, 3)
        for positive in (True, False):
            part = pts[pts[:, 0] >= 0] if positive else pts[pts[:, 0] < 0]
            limit, width = (p['forward_limit'], p['front_width']) if positive else (p['backward_limit'], p['back_width'])
            if len(part):
                if limit == 0 and np.any(part[:, 0] != 0):
                    return float('inf')
                t = part[:, 0] / limit if limit != 0 else np.zeros(len(part))
                hw = (1-t)*p['middle_width'] + t*width
                worst = max(worst, float(np.max(np.abs(part[:, 1])-hw)), float(np.max(part[:, 0]-p['forward_limit'])), float(np.max(p['backward_limit']-part[:, 0])))
    return worst


def force_errors(m, cert):
    points = np.array(cert['contact_points']); forces = np.array(cert['forces_world'])
    com = m.data.subtree_com[m.base_body_id]
    f_error = np.linalg.norm(forces.sum(axis=0) + m.model.body_mass.sum()*m.model.opt.gravity)
    moment_error = np.linalg.norm(np.cross(points-com, forces).sum(axis=0))
    joint_force = np.zeros(m.model.nv)
    jp = np.zeros((3, m.model.nv)); jr = np.zeros_like(jp)
    for point, force, body in zip(points, forces, cert['body_ids']):
        mujoco.mj_jac(m.model, m.data, jp, jr, point, body)
        joint_force += jp.T @ force
    tau = (m.data.qfrc_bias-m.data.qfrc_passive-joint_force)[m.joint_dof_adr]
    tau_error = np.max(np.abs(tau-np.array(cert['joint_torques'])))
    local = forces @ m.force_basis
    friction_violation = max(float(np.max(np.abs(local[:, :2]).sum(axis=1)-m.friction*local[:, 2])), float(np.max(-local[:, 2])))
    return dict(force_N=float(f_error), moment_Nm=float(moment_error), torque_Nm=float(tau_error), friction_N=max(0.,friction_violation))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--xml',type=Path,default=DEFAULT_XML)
    ap.add_argument('--sampling',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    raw=json.loads(a.sampling.read_text());m=MorphologyMeter(a.xml,floor=False)
    arrays=np.load(a.sampling.with_suffix('.candidates.npz'),allow_pickle=False)
    failures=[];maximum=dict(face_m=0.,force_N=0.,moment_Nm=0.,torque_Nm=0.,friction_N=0.)
    for rec in raw['witnesses']:
        m.set_state((0,0,rec['root_height']),(1,0,0,0),rec['joint_angles'])
        errors=force_errors(m,rec['static_certificate']);face=filled_face_violation(m,rec['hex_params']);bad=[]
        if face>1e-7:bad.append('filled_geometry')
        if m.self_collision() or m.clearance(True)<.005-1e-8:bad.append('collision_or_clearance')
        if any(v>1e-5 for v in errors.values()):bad.append('force_certificate')
        if max(abs(m.signed_ground_distance(m._lowest_point_of_body(b))) for b in m.foot_body_ids.values())>1e-4:bad.append('sole_to_plane')
        for leg in m.foot_body_ids:
            q=np.array([rec['joint_angles'][f'{leg}_{k}'] for k in ('HAA','HFE','KFE')]);cs=arrays[f'{leg}_h{rec["root_height"]:.3f}']
            if len(cs)==0 or np.min(np.max(np.abs(cs[:,:3]-q),axis=1))>1e-8:bad.append('missing_saved_leg_candidate')
        if bad:failures.append(dict(id=rec['id'],failed_checks=bad,errors=errors,face_m=face))
        maximum['face_m']=max(maximum['face_m'],face)
        for k,v in errors.items():maximum[k]=max(maximum[k],v)
    bounds_correct=bool(raw['witnesses'])
    for key in PARAM_KEYS:
        if not raw['witnesses']:break
        lo=min(r['hex_params'][key] for r in raw['witnesses']);hi=max(r['hex_params'][key] for r in raw['witnesses']);bound=raw['observed_bounds'][key]
        lookup={r['id']:r for r in raw['witnesses']}
        bounds_correct &= (abs(lo-bound['observed_min'])<1e-9 and abs(hi-bound['observed_max'])<1e-9 and abs(lookup[bound['min_witness_id']]['hex_params'][key]-lo)<1e-9 and abs(lookup[bound['max_witness_id']]['hex_params'][key]-hi)<1e-9 and not bound['is_global_bound'])
    current=identity(a.xml,{'purpose':'independent witness replay'},[a.sampling])
    hashes_match=raw['provenance']['sources_sha256']==current['sources_sha256'] and model_content_matches(raw['provenance']['model'],current['model'])
    passed=not failures and bounds_correct and hashes_match
    write_new(a.out,dict(passed=passed,total=len(raw['witnesses']),failures=failures,maximum_errors=maximum,bounds_correct=bool(bounds_correct),source_and_model_hash_match=hashes_match,provenance=current))
    print('replay',len(raw['witnesses']),'PASS' if passed else 'FAIL',maximum,flush=True)
    return 0 if passed else 1


if __name__=='__main__':sys.exit(main())
