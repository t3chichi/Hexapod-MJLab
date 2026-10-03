#!/usr/bin/env python3
"""Adversarial acceptance gates independent of reported range/hold summaries."""
from pathlib import Path
import argparse,sys,time
import mujoco
import numpy as np
from scipy.spatial import ConvexHull
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from morphology.meter import MorphologyMeter,JOINT_NAMES,DEFAULT_XML
from morphology.hexfit import fit_collision_hexagon,containment_violation,normalize_params,denormalize_params,PARAM_KEYS
from morphology.statics import equilibrium
from morphology.verify_hold import stance_case
from morphology.provenance import identity,write_new


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--xml',type=Path,default=DEFAULT_XML);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();checks={}
    def check(k,v):checks[k]=bool(v);print(k,'PASS' if v else 'FAIL',flush=True)
    m=MorphologyMeter(a.xml);m.set_state((0,0,.48601),(1,0,0,0),np.zeros(18));state=m.measure()
    check('FK_anchor_RF',np.max(np.abs(state.feet_base['RF']-[.4499,-.3764,-.4729]))<1e-3)
    check('actual_compiled_hulls_not_render_cloud',len(state.occupancy_base)<2000)
    # Independent MuJoCo native distance supports validate the graph decoder.
    floor=next(iter(m.ground_geom_ids));distance_errors=[]
    rng=np.random.default_rng(771)
    for i in range(5):
        m.set_state((0,0,1.3),rng.normal(size=4),rng.uniform(-1,1,18))
        for g,b,v in m._occ_geoms:
            api=mujoco.mj_geomDistance(m.model,m.data,floor,g,10,None)
            own=float(m.signed_ground_distance(m.world_geom(g,v)).min());distance_errors.append(abs(api-own))
    check('compiled_hull_vs_native_distance_135_probes',max(distance_errors)<1e-6)
    # Fill, not just vertices: an entire rectangle crossing the waist.
    box=np.array([[-1,-.5],[-1,.5],[1,-.5],[1,.5]])
    p=fit_collision_hexagon([box]);check('rectangle_waist_not_underapproximated',p['middle_width']>=.5-1e-9)
    # Disconnected side limbs may retain a nonconvex waist; no fictitious hull.
    left=np.array([[-1,-.6],[-1,.6],[-.6,.2],[-.6,-.2]])
    right=left*np.array([-1,1]);middle=np.array([[-.1,-.1],[-.1,.1],[.1,.1],[.1,-.1]])
    p=fit_collision_hexagon([left,right,middle]);check('disconnected_nonconvex_shape_preserved',p['middle_width']<.2)
    # Face/edge sampling is independently inside for randomized real bodies.
    violations=[]
    for i in range(15):
        m.set_state((0,0,.6),(1,0,0,0),rng.uniform(-1.5,1.5,18));p=fit_collision_hexagon(m.collision_polygons())
        for g,b,v in m._occ_geoms:
            tri=m._triangles[g];w=tri.reshape(-1,3)@m.data.geom_xmat[g].reshape(3,3).T+m.data.geom_xpos[g];w=m.to_base_frame(w).reshape(-1,3,3)
            pts=np.vstack([w.mean(axis=1),w[:,0]*.7+w[:,1]*.3,w[:,1]*.5+w[:,2]*.5])
            for x,y,z in pts:
                t=x/p['forward_limit'] if x>=0 else x/p['backward_limit'];end=p['front_width'] if x>=0 else p['back_width'];violations.append(abs(y)-((1-t)*p['middle_width']+t*end))
            # Explicit triangle edges crossing x=0 independently catch the old defect.
            for t3 in w:
                for va,vb in zip(t3,np.roll(t3,-1,axis=0)):
                    if va[0]*vb[0]<0:violations.append(abs(va[1]-va[0]*(vb[1]-va[1])/(vb[0]-va[0]))-p['middle_width'])
    check('real_filled_faces_waist_and_boundary',max(violations)<1e-6)
    m.set_state((0,0,2),(1,0,0,0),np.zeros(18));z=m.measure();check('airborne_no_fake_ground_support',z.n_stance_feet==0 and z.com_margin is None and not z.ground_flags['valid'] and abs(z.body_height-2)<1e-9)
    # Measurement must not reset controls or velocities of a rollout.
    m.data.qvel[:]=.01;m.data.ctrl[:]=.03;before=(m.data.qpos.copy(),m.data.qvel.copy(),m.data.ctrl.copy());m.measure();check('measurement_preserves_live_state',all(np.array_equal(x,y) for x,y in zip(before,(m.data.qpos,m.data.qvel,m.data.ctrl))))
    c=stance_case(m,.4);m.set_state((0,0,.4),(1,0,0,0),c['joint_angles']);eq=equilibrium(m)
    check('true_sole_to_ground_IK',max(abs(m.signed_ground_distance(m._lowest_point_of_body(b))) for b in m.foot_body_ids.values())<1e-5)
    check('unactuated_root_force_balance',eq['valid'] and max(abs(v) for v in eq['root_residual'])<1e-6)
    if eq['valid']:
        forces=np.array(eq['forces_world']);pts=np.array(eq['contact_points']);com=m.data.subtree_com[m.base_body_id]
        gravity=np.array([0,0,-m.model.body_mass.sum()*9.81]);forceerr=np.linalg.norm(forces.sum(axis=0)+gravity);momenterr=np.linalg.norm(np.cross(pts-com,forces).sum(axis=0))
        check('independent_global_force_and_moment_sum',forceerr<1e-5 and momenterr<1e-5)
    else:check('independent_global_force_and_moment_sum',False)
    check('zero_torque_budget_not_false_positive',not equilibrium(m,torque_cap=0)['valid'])
    m.set_state((0,0,2),(1,0,0,0),np.zeros(18));check('no_geometric_contacts_static_infeasible',not equilibrium(m)['valid'])
    # Asymmetric loaded contact set distinguishes floor, robot contact and missing foot.
    m.set_state((0,0,.479),(1,0,0,0),np.zeros(18));z=m.measure();check('actual_loaded_contacts_produce_support',z.n_stance_feet>=4 and z.ground_flags['valid'] and z.com_margin>0)
    c=stance_case(m,.4);m.set_state((0,0,.4),(1,0,0,0),c['joint_angles'])
    target=m.to_base_frame(m.data.xpos[m.foot_body_ids['RF']]);sol=m.leg_ik('RF',target,sole_on_ground=True,sole_target_offset=.05)
    partial=dict(c['joint_angles']);partial.update(sol[0]);m.set_state((0,0,.398),(1,0,0,0),partial);z=m.measure()
    check('raised_foot_not_counted_as_support','RF' not in z.contact_report['feet'] and z.n_stance_feet==5 and z.ground_flags['support_valid'])
    pmin=dict(zip(PARAM_KEYS,[.3,.3,.3,.65,-.6]));pmax=dict(zip(PARAM_KEYS,[.6,.7,.6,.9,-.9]));check('rear_normalization_minmax_roundtrip',np.allclose(normalize_params(pmin,pmin,pmax),0) and np.allclose(normalize_params(pmax,pmin,pmax),1) and denormalize_params([0,0,0,0,1],pmin,pmax)['backward_limit']==-.9)
    check('URDF_stops_disabled_not_safety_certificate',not m.model.jnt_limited[m.joint_ids].any() and m.declared_joint_ranges[0,0]==-3)
    # Explicit tilted floor; geometric IK is not a claimed support certificate.
    n=np.array([np.sin(np.deg2rad(5)),0,np.cos(np.deg2rad(5))]);slope=MorphologyMeter(a.xml,ground_normal=n);slope.set_state((0,0,.4),(1,0,0,0),np.zeros(18));q={n:0. for n in JOINT_NAMES}
    for l in slope.foot_body_ids:
        foot=slope.to_base_frame(slope.data.xpos[slope.foot_body_ids[l]]);sol=slope.leg_ik(l,foot,sole_on_ground=True)
        if sol:q.update(sol[0])
    slope.set_state((0,0,.4),(1,0,0,0),q);check('sloped_ground_sole_IK_and_clearance',max(abs(slope.signed_ground_distance(slope._lowest_point_of_body(b))) for b in slope.foot_body_ids.values())<1e-5)
    result={'passed':all(checks.values()),'checks':checks,'native_distance_max_error':max(distance_errors),'independent_face_violation_max':max(violations),'static_ideal_certificate':eq,'provenance':identity(a.xml,{'random_seed':771,'purpose':'adversarial core gate'})};write_new(a.out,result)
    return 0 if result['passed'] else 1

if __name__=='__main__':sys.exit(main())
