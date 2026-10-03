#!/usr/bin/env python3
"""Slow scripted six-leg crawl checks; deliberately not a trained-policy T0 gate.

One foot swings at a time, 50Hz sole IK + 200Hz bounded joint PD, two complete
cycles, no external root forces. All failures and geometry envelopes are
saved. Demonstrates only this controller/terrain/speed, not learned locomotion.
"""
from pathlib import Path
import argparse,json,sys,time
import mujoco
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from morphology.meter import MorphologyMeter,JOINT_NAMES,LEGS,DEFAULT_XML,polygon_metrics
from morphology.verify_hold import POLICY,stance_case
from morphology.hexfit import fit_collision_hexagon,containment_violation,PARAM_KEYS
from morphology.provenance import identity,write_new
from morphology.statics import equilibrium

GAIT={'cycles':2,'phase_s':2.,'stride_m':.03,'swing_height_m':.04,'control_dt':.02,'physics_dt':.005,'settle_s':2.,'tail_s':2.,'minimum_actual_stance_feet':4,'support_margin_m':.03,
      'tilt_limit_deg':5.,'height_error_m':.035,'lateral_drift_m':.03,'minimum_forward_progress_m':.03,'max_tracking_rmse_rad':.2,'nominal_torque_peak_Nm':64.,'kp':130.,'kd':2.,'feedforward':'planned rigid-contact gravity balance','lateral_feedback_gain':2.}


def run_gait(xml,case,cycles=2):
    physical=MorphologyMeter(xml);planner=MorphologyMeter(xml,floor=False);m,d=physical.model,physical.data;h=case['root_height'];q=np.array([case['joint_angles'][n] for n in JOINT_NAMES]);physical.set_state((0,0,h),(1,0,0,0),q)
    if not np.isclose(m.opt.timestep,GAIT['physics_dt']):raise ValueError('gait protocol requires 200Hz physics')
    low=physical.clearance();physical.set_state((0,0,h+max(0,.001-low)),(1,0,0,0),q)
    planner.set_state((0,0,h),(1,0,0,0),q)
    certificate=equilibrium(planner,torque_cap=None);feedforward=np.array(certificate.get('joint_torques',[0.]*18));balance_failed=[]
    acts=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_ACTUATOR,n) for n in JOINT_NAMES];qdes=q.copy();rows=[];events=[];tracking=[];peak=0;lastcycle=None;ik_failed=[];minsupport=np.inf;minimumfeet=6;maxviolation=0;steady_support=0
    def advance(t,phase):
        nonlocal peak,minimumfeet,minsupport,steady_support
        torque=np.clip(feedforward+GAIT['kp']*(qdes-d.qpos[physical.joint_qpos_adr])-GAIT['kd']*d.qvel[physical.joint_dof_adr],-80,80);d.ctrl[acts]=torque;mujoco.mj_step(m,d);mujoco.mj_forward(m,d)
        peak=max(peak,float(np.max(np.abs(torque))));tracking.append(d.qpos[physical.joint_qpos_adr]-qdes);c=physical.contacts()
        for cat in ('self','nonfoot_ground'):
            for r in c[cat]:
                if r['distance'] < -1e-6 or r['normal_force']>.05:events.append({'time':t,'phase':phase,'category':cat,**r})
        if phase=='crawl':
            nf=len(c['feet']);minimumfeet=min(minimumfeet,nf);points=[r['point'][:2] for r in c['foot_ground'] if r['normal_force']>1e-5];margin,_=polygon_metrics(points,d.subtree_com[physical.base_body_id][:2])
            if margin is not None:minsupport=min(minsupport,margin)
            else:minsupport=-np.inf
            if nf<4:steady_support+=1
    for t in range(400):advance(t*.005,'settle')
    origin=d.qpos[:3].copy();planner.set_state((0,0,h),(1,0,0,0),q)
    anchors={leg:np.r_[origin[:2]+planner.to_base_frame(planner.data.xpos[b])[:2],0.] for leg,b in planner.foot_body_ids.items()};old={l:p.copy() for l,p in anchors.items()}
    duration=cycles*len(LEGS)*GAIT['phase_s'];frames=int(round(duration/.02));order=('RF','LB','RM','LF','RB','LM')
    for f in range(frames):
        t=f*.02;phase_index=int(t/GAIT['phase_s']);swing=order[phase_index%6];u=(t/GAIT['phase_s'])%1;rootgoal=np.array([origin[0]+GAIT['stride_m']*t/(6*GAIT['phase_s']),origin[1],h]);cycle=phase_index//6
        rootgoal[1]+=GAIT['lateral_feedback_gain']*(origin[1]-d.qpos[1])
        if lastcycle!=phase_index:
            if lastcycle is not None:
                leg=order[lastcycle%6];anchors[leg][0]=old[leg][0]+GAIT['stride_m'];old[leg]=anchors[leg].copy()
            lastcycle=phase_index
        planner.set_state(rootgoal,(1,0,0,0),qdes)
        for leg in LEGS:
            foot=anchors[leg].copy();height=0.
            if leg==swing:
                ease=u-np.sin(2*np.pi*u)/(2*np.pi);foot[0]=old[leg][0]+GAIT['stride_m']*ease;height=GAIT['swing_height_m']*np.sin(np.pi*u)**2
            target=foot-rootgoal;sol=planner.leg_ik(leg,target,sole_on_ground=True,sole_target_offset=height,seed_count=1)
            if not sol:ik_failed.append({'frame':f,'leg':leg});continue
            for kind in ('HAA','HFE','KFE'):qdes[JOINT_NAMES.index(f'{leg}_{kind}')]=sol[0][f'{leg}_{kind}']
        planner.set_state(rootgoal,(1,0,0,0),qdes);certificate=equilibrium(planner,torque_cap=None)
        if certificate['valid']:feedforward=np.array(certificate['joint_torques'])
        else:balance_failed.append(f)
        for sub in range(4):advance(t+sub*.005,'crawl')
        R=d.xmat[physical.base_body_id].reshape(3,3);polys=physical.collision_polygons();shape=fit_collision_hexagon(polys);maxviolation=max(maxviolation,containment_violation(polys,shape));rows.append({'time':t,'root':d.qpos[:3].tolist(),'tilt_deg':float(np.degrees(np.arccos(np.clip(R[2,2],-1,1)))),'hex':shape,'q_target':qdes.tolist(),'applied_torque':d.ctrl[acts].tolist(),'q':d.qpos[physical.joint_qpos_adr].tolist(),'qd':d.qvel[physical.joint_dof_adr].tolist(),'foot_contacts':physical.contacts()['feet']})
    # Complete the last swing at zero lift before final settling.
    leg=order[lastcycle%6];anchors[leg][0]=old[leg][0]+GAIT['stride_m'];rootgoal=np.array([origin[0]+GAIT['stride_m']*cycles,origin[1],h]);planner.set_state(rootgoal,(1,0,0,0),qdes)
    for leg in LEGS:
        sol=planner.leg_ik(leg,anchors[leg]-rootgoal,sole_on_ground=True,seed_count=1)
        if sol:
            for kind in ('HAA','HFE','KFE'):qdes[JOINT_NAMES.index(f'{leg}_{kind}')]=sol[0][f'{leg}_{kind}']
    planner.set_state(rootgoal,(1,0,0,0),qdes);certificate=equilibrium(planner,torque_cap=None)
    if certificate['valid']:feedforward=np.array(certificate['joint_torques'])
    else:balance_failed.append(frames)
    for t in range(400):advance(duration+t*.005,'tail')
    final=physical.measure();final_balance=equilibrium(physical,actual_contacts=True);tracking=np.array(tracking);rms=np.sqrt(np.mean(tracking**2,axis=0));end=d.qpos[:3].copy();tilts=[r['tilt_deg'] for r in rows];height_error=max(abs(r['root'][2]-h) for r in rows);ydrift=max(abs(r['root'][1]-origin[1]) for r in rows);warning=[mujoco.mjtWarning(i).name for i,w in enumerate(d.warning) if w.number]
    checks={'minimum_protocol_cycles':cycles>=2,'tail_actual_contact_balance':final_balance['valid'],'no_ik_failure':not ik_failed,'planned_contact_balance':not balance_failed,'no_self_or_nonfoot_contact_all_steps':not events,'no_warning':not warning,'finite':bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()),'actual_stance_feet':minimumfeet>=4,'actual_support_margin':minsupport>=.03,'nominal_torque_headroom':peak<=64.,'tracking':float(rms.max())<.2,'tilt':max(tilts)<5.,'height_tracking':height_error<.035,'lateral_drift':ydrift<.03,'forward_progress':end[0]-origin[0]>=.03,'complete_geometry_containment':maxviolation<=1e-7,'joint_search_period':all(max(abs(x) for x in r['q'])<=np.pi+1e-6 for r in rows),'tail_static':np.linalg.norm(d.qvel[:3])<.01 and np.linalg.norm(d.qvel[3:6])<.02}
    return {'id':case['id'],'verdict':'PASS' if all(checks.values()) else 'FAIL','checks':{k:bool(v) for k,v in checks.items()},'failed_checks':[k for k,v in checks.items() if not v],'root_height_target':h,'root_start':origin.tolist(),'root_end':end.tolist(),'forward_progress':float(end[0]-origin[0]),'max_height_error':height_error,'max_tilt_deg':max(tilts),'minimum_actual_stance_feet':minimumfeet,'minimum_support_margin':None if not np.isfinite(minsupport) else minsupport,'peak_torque':peak,'tracking_worst_rmse':float(rms.max()),'contact_events':{'count':len(events),'first':events[:3],'last':events[-3:]},'IK_failures':ik_failed,'planned_balance_failures':balance_failed,'geometry_max_violation':maxviolation,'end_qpos':d.qpos.tolist(),'end_qvel':d.qvel.tolist(),'end_static_certificate':final_balance,'end_hex':final.hex_params,'trajectory':rows,'duration_s':duration,'scope':'scripted slow crawl only; does not certify PPO policy or T0 baseline'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--xml',type=Path,default=DEFAULT_XML);ap.add_argument('--height',type=float,default=.48);ap.add_argument('--cases',type=Path);ap.add_argument('--case-ids',default='');ap.add_argument('--cycles',type=int,default=2);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    if a.cycles<1:ap.error('--cycles must be positive')
    if a.out.exists():raise FileExistsError(a.out)
    cases=[stance_case(MorphologyMeter(a.xml),a.height)];inputs=[]
    if a.cases:
        raw=json.loads(a.cases.read_text());ids=set(a.case_ids.split(','));cases=[r for r in raw['cases'] if not a.case_ids or r['id'] in ids];inputs.append(a.cases)
    result=[]
    for c in cases:
        r=run_gait(a.xml,c,a.cycles);result.append(r);print(r['id'],r['verdict'],r['failed_checks'],'progress',r['forward_progress'],flush=True)
    write_new(a.out,{'provenance':identity(a.xml,{'gait':GAIT,'cycles':a.cycles},inputs),'cases':cases,'results':result,'passed':sum(r['verdict']=='PASS' for r in result),'total':len(result)})
    return 0

if __name__=='__main__':sys.exit(main())
