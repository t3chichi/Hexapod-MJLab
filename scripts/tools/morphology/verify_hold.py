#!/usr/bin/env python3
"""All-step contact-aware static holds, preserving every rejected case."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import mujoco
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from morphology.meter import MorphologyMeter,JOINT_NAMES,LEGS,DEFAULT_XML,COLLISION_TOL,polygon_metrics
from morphology.statics import equilibrium
from morphology.provenance import identity,write_new

# Simulation acceptance tolerances. These are not inferred actuator ratings.
POLICY={'duration_s':5.,'dt':.005,'kp':130.,'kd':2.,'torque_clip_Nm':80.,'torque_nominal_peak_cap_Nm':64.,
        'height_deviation_m':.05,'mean_tracking_rmse_rad':.15,'worst_tracking_rmse_rad':.2,
        'max_tilt_deg':5.,'max_xy_drift_m':.03,'minimum_end_foot_contacts':4,'minimum_support_margin_m':.03,
        'tail_root_linear_speed_m_s':.01,'tail_root_angular_speed_rad_s':.02,'tail_joint_speed_rad_s':.05,
        'tail_height_error_m':.035,'self_penetration_tolerance_m':COLLISION_TOL,
        'nonfoot_penetration_tolerance_m':COLLISION_TOL,'contact_force_rejection_N':.05,
        'settled_unactuated_force_residual_N':2.,'settled_unactuated_moment_residual_Nm':1.}


def stance_case(meter,height=.4):
    meter.set_state((0,0,height),(1,0,0,0),np.zeros(18));xy={l:meter.to_base_frame(meter.data.xpos[b])[:2].copy() for l,b in meter.foot_body_ids.items()};q={n:0. for n in JOINT_NAMES}
    for l in LEGS:
        sols=meter.leg_ik(l,[*xy[l],-height],sole_on_ground=True)
        if not sols:raise RuntimeError(f'stance IK unavailable for {l}@{height}')
        s=min(sols,key=lambda s:sum(v*v for v in s.values()));q.update(s)
    return {'id':f'stand_like_h{height:.3f}','root_height':height,'joint_angles':q,'source':'true sole-to-plane IK at q0 foot xy','initial_mode':'sole_clear_drop'}


def run_hold(meter,case,steps=1000):
    m,d=meter.model,meter.data
    if not np.isclose(m.opt.timestep,POLICY['dt']):raise ValueError('hold protocol requires 200Hz physics')
    q=np.array([case['joint_angles'][n] for n in JOINT_NAMES]);height=float(case['root_height']);meter.set_state((0,0,height),(1,0,0,0),q)
    # Begin with every physical collision sole at least 1mm clear. No arbitrary
    # penetrating start and no rigid-lift inverse-dynamics shortcut.
    lowest=min(float(meter.signed_ground_distance(meter._lowest_point_of_body(b))) for b in meter.foot_body_ids.values())
    lift=max(0.,(.001-lowest)/meter.ground_normal[2]);meter.set_state((0,0,height+lift),(1,0,0,0),q)
    initial=meter.measure();start=d.qpos[:3].copy();initq=d.qpos.copy()
    acts=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_ACTUATOR,n) for n in JOINT_NAMES]
    tracking=[];heights=[];tilts=[];drifts=[];vel=[];torques=[];self_events=[];nonfoot_events=[];finite=True;qdomain=True;sole_clearance=[]
    failed_initial_geometry=bool(meter.self_collision() or initial.nonfoot_clearance < -COLLISION_TOL)
    for t in range(steps):
        tau=np.clip(POLICY['kp']*(q-d.qpos[meter.joint_qpos_adr])-POLICY['kd']*d.qvel[meter.joint_dof_adr],-POLICY['torque_clip_Nm'],POLICY['torque_clip_Nm']);d.ctrl[acts]=tau
        mujoco.mj_step(m,d)
        if not np.isfinite(np.r_[d.qpos,d.qvel]).all():finite=False;break
        # Read fresh contacts at this integrated state; this is part of the
        # declared v2 protocol. Warm-start fields are reset between cases.
        mujoco.mj_forward(m,d);contacts=meter.contacts()
        for category,events in (('self',self_events),('nonfoot_ground',nonfoot_events)):
            for r in contacts[category]:
                if r['distance'] < -COLLISION_TOL or r['normal_force']>POLICY['contact_force_rejection_N']:
                    events.append({'step':t,**r})
        tracking.append(d.qpos[meter.joint_qpos_adr]-q);heights.append(d.qpos[2]);vel.append(d.qvel.copy());torques.append(tau.copy());qdomain &= bool(np.max(np.abs(d.qpos[meter.joint_qpos_adr]))<=np.pi+1e-6)
        R=d.xmat[meter.base_body_id].reshape(3,3);tilts.append(float(np.degrees(np.arccos(np.clip(R[:,2]@meter.ground_normal,-1,1)))));drifts.append(float(np.linalg.norm(d.qpos[:2]-start[:2])))
    if not heights:raise RuntimeError('no finite trajectory')
    tracking=np.array(tracking);vel=np.array(vel);torques=np.array(torques);tail=min(100,len(heights));rms=np.sqrt(np.mean(tracking**2,axis=0));final=meter.measure();static=equilibrium(meter,actual_contacts=True)
    diagnostic=meter.inverse_diagnostic();root=np.array(diagnostic['root_required_wrench']);angles=d.qpos[meter.joint_qpos_adr].copy()
    contactfeet=len(final.contact_report['feet']);warning=[mujoco.mjtWarning(i).name for i,w in enumerate(d.warning) if w.number]
    checks={'minimum_protocol_duration':len(heights)*m.opt.timestep>=POLICY['duration_s'],'complete_duration':len(heights)==steps,'finite':finite,'no_warning':not warning,'initial_geometry_clear':not failed_initial_geometry,
            'no_self_collision_all_steps':not self_events,'no_nonfoot_ground_all_steps':not nonfoot_events,'within_measured_search_period':qdomain,
            'height_deviation':max(abs(np.array(heights)-start[2]))<POLICY['height_deviation_m'],'mean_tracking':float(rms.mean())<POLICY['mean_tracking_rmse_rad'],'worst_tracking':float(rms.max())<POLICY['worst_tracking_rmse_rad'],
            'tilt':max(tilts)<POLICY['max_tilt_deg'],'xy_drift':max(drifts)<POLICY['max_xy_drift_m'],'end_contact_feet':contactfeet>=POLICY['minimum_end_foot_contacts'],
            'end_actual_support':final.com_margin is not None and final.com_margin>=POLICY['minimum_support_margin_m'],
            'tail_linear_speed':float(np.max(np.linalg.norm(vel[-tail:,:3],axis=1)))<POLICY['tail_root_linear_speed_m_s'],
            'tail_angular_speed':float(np.max(np.linalg.norm(vel[-tail:,3:6],axis=1)))<POLICY['tail_root_angular_speed_rad_s'],
            'tail_joint_speed':float(np.max(np.abs(vel[-tail:,meter.joint_dof_adr])))<POLICY['tail_joint_speed_rad_s'],
            'target_height_preserved':abs(float(np.mean(heights[-tail:]))-height)<POLICY['tail_height_error_m'],
            'nominal_actuator_headroom':float(np.max(np.abs(torques)))<=POLICY['torque_nominal_peak_cap_Nm'],
            'rigid_contact_equilibrium':bool(static['valid']),
            'settled_unactuated_balance':float(np.max(np.abs(root[:3])))<=POLICY['settled_unactuated_force_residual_N'] and float(np.max(np.abs(root[3:])))<=POLICY['settled_unactuated_moment_residual_Nm'],
            'complete_collision_hex_containment':initial.hex_violation<=1e-7 and final.hex_violation<=1e-7}
    return {'id':case['id'],'verdict':'PASS' if all(checks.values()) else 'FAIL','checks':{k:bool(v) for k,v in checks.items()},'failed_checks':[k for k,v in checks.items() if not v],
            'source':case.get('source'),'initial_qpos':initq.tolist(),'final_qpos':d.qpos.tolist(),'initial_root_height':float(start[2]),'target_root_height':height,'settled_root_height':float(np.mean(heights[-tail:])),
            'initial_hex':initial.hex_params,'final_hex':final.hex_params,'hex_max_violation':max(initial.hex_violation,final.hex_violation),'final_actual_support_margin':final.com_margin,'final_support_feet':final.contact_report['feet'],
            'mean_tracking_rmse':float(rms.mean()),'worst_tracking_rmse':float(rms.max()),'max_tilt_deg':max(tilts),'max_xy_drift':max(drifts),'peak_torque':float(np.max(np.abs(torques))),
            'self_collision_events':{'count':len(self_events),'first':self_events[:3],'last':self_events[-3:]},'nonfoot_ground_events':{'count':len(nonfoot_events),'first':nonfoot_events[:3],'last':nonfoot_events[-3:]},
            'static_certificate':static,'inverse_diagnostic':diagnostic,'steps':len(heights),'warnings':warning,
            'tail':{'linear_speed_max':float(np.max(np.linalg.norm(vel[-tail:,:3],axis=1))),'angular_speed_max':float(np.max(np.linalg.norm(vel[-tail:,3:6],axis=1))),'joint_speed_max':float(np.max(np.abs(vel[-tail:,meter.joint_dof_adr])))},
            'trajectory':{'root_height':list(map(float,heights)),'tilt_deg':tilts,'xy_drift':drifts,'joint_angles':(tracking+q).tolist(),'root_velocity':vel[:,:6].tolist(),'joint_velocity':vel[:,meter.joint_dof_adr].tolist(),'torques':torques.tolist()}}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--xml',type=Path,default=DEFAULT_XML);ap.add_argument('--sampling',type=Path);ap.add_argument('--cases',type=Path);ap.add_argument('--legacy',type=Path);ap.add_argument('--heights',default='.34,.36,.38,.40,.42,.44,.46,.48,.50,.52,.54');ap.add_argument('--steps',type=int,default=1000);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    m=MorphologyMeter(a.xml);cases=[];inputs=[]
    if a.sampling:
        s=json.loads(a.sampling.read_text());cases+=s['representatives'];inputs.append(a.sampling)
    if a.cases:
        s=json.loads(a.cases.read_text());cases+=s['cases'];inputs.append(a.cases)
    if a.legacy:
        s=json.loads(a.legacy.read_text());inputs.append(a.legacy)
        for k,v in s['cases'].items():cases.append({'id':'legacy_'+k,'root_height':v.get('h'),'joint_angles':v['joint_angles'],'source':'frozen GLM case, old contact/height certificate not inherited'})
    for h in map(float,filter(None,a.heights.split(','))):
        try:cases.append(stance_case(m,h))
        except RuntimeError as e:cases.append({'id':f'stand_like_h{h:.3f}','error':str(e)})
    results=[]
    for case in cases:
        if 'error' in case:r={'id':case['id'],'verdict':'NOT_FOUND','reason':case['error']}
        else:r=run_hold(m,case,a.steps)
        results.append(r);print(r['id'],r['verdict'],r.get('failed_checks'),flush=True)
    data={'provenance':identity(a.xml,{'policy':POLICY,'steps':a.steps,'height_semantics':'root vertical height above explicit ground, not foot-origin plane','joint_domain':[-np.pi,np.pi],'URDF_stops_disabled':True},inputs),
          'cases':cases,'results':results,'passed':sum(r['verdict']=='PASS' for r in results),'total':len(results),'stage':'static_only; slow gait not tested'}
    write_new(a.out,data);print('complete',data['passed'],data['total'])
    return 0


if __name__=='__main__':sys.exit(main())
