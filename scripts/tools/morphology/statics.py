"""Rigid-contact static equilibrium LP; no invented 5mm force certificate."""
import mujoco
import numpy as np
from scipy.optimize import linprog
from .meter import COLLISION_TOL


def equilibrium(meter,torque_cap=64.,sole_tolerance=1e-4,actual_contacts=False):
    m,d=meter.model,meter.data;points=[]
    if actual_contacts:
        for r in meter.contacts()['foot_ground']:
            if r['normal_force']>1e-5:
                g=r['geom1'] if r['geom1'] in meter.foot_geom_ids else r['geom2'];point=np.array(r['point']);point-=meter.signed_ground_distance(point)*meter.ground_normal;points.append((int(m.geom_bodyid[g]),point))
    else:
        for b in meter.foot_body_ids.values():
            for g,v in meter._body_geoms[b]:
                w=meter.world_geom(g,v);dist=meter.signed_ground_distance(w)
                for p,z in zip(w,dist):
                    if abs(z)<=sole_tolerance:points.append((b,p-z*meter.ground_normal))
    if not points:return {'valid':False,'reason':'no_foot_ground_candidates','contact_count':0}
    tmp=mujoco.MjData(m);tmp.qpos[:]=d.qpos;mujoco.mj_forward(m,tmp);bias=tmp.qfrc_bias.copy()-tmp.qfrc_passive
    matrices=[];jp=np.zeros((3,m.nv));jr=np.zeros_like(jp)
    for b,p in points:
        mujoco.mj_jac(m,tmp,jp,jr,p,b);matrices.append(jp.T@meter.force_basis)
    J=np.hstack(matrices);nj=len(points);n=3*nj+1
    eq=np.zeros((6,n));eq[:,:-1]=J[:6];beq=bias[:6]
    rows=[];bs=[]
    for row,b in zip(J[meter.joint_dof_adr],bias[meter.joint_dof_adr]):
        a=np.r_[row,-1.];rows.append(a);bs.append(b)
        a=np.r_[-row,-1.];rows.append(a);bs.append(-b)
    for i in range(nj):
        for x,y in ((1,1),(1,-1),(-1,1),(-1,-1)):
            a=np.zeros(n);a[3*i:3*i+3]=[x,y,-meter.friction];rows.append(a);bs.append(0.)
    objective=np.zeros(n);objective[-1]=1.
    bounds=[(None,None),(None,None),(0,None)]*nj+[(0,None if torque_cap is None else float(torque_cap))]
    r=linprog(objective,A_ub=np.array(rows),b_ub=np.array(bs),A_eq=eq,b_eq=beq,bounds=bounds,method='highs',options={'primal_feasibility_tolerance':1e-8,'dual_feasibility_tolerance':1e-8})
    if not r.success:return {'valid':False,'reason':'force_balance_or_torque_infeasible','contact_count':nj,'solver_status':int(r.status)}
    f=r.x[:-1];tau=bias[meter.joint_dof_adr]-J[meter.joint_dof_adr]@f;res=eq@r.x-beq
    constraints=np.array(rows)@r.x-np.array(bs);normals=f[2::3]
    valid=bool(np.max(np.abs(res))<=1e-6 and np.max(constraints)<=1e-6 and np.min(normals)>=-1e-8 and (torque_cap is None or np.max(np.abs(tau))<=torque_cap+1e-6))
    return {'valid':valid,'reason':'balanced' if valid else 'residual_failure','contact_count':nj,'root_residual':res.tolist(),'joint_torques':tau.tolist(),'max_joint_torque':float(np.max(np.abs(tau))),'contact_points':[p.tolist() for b,p in points],'body_ids':[b for b,p in points],'forces_world':[(meter.force_basis@f[3*i:3*i+3]).tolist() for i in range(nj)],'max_inequality_violation':float(max(0.,np.max(constraints))),'assumptions':'rigid point support, conservative L1 friction pyramid, nominal uncalibrated actuator cap; not soft-contact inverse diagnostic or hardware certificate'}
