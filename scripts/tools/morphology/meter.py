"""Collision-model geometry, real contact measurements and multi-branch leg IK.

URDF joint stops are not treated as verified hardware constraints. The
measurement model disables hinge stops and explores one principal angular
period [-pi,pi]; recorded free-of-collision states remain witnesses, not a
rectangular mechanical range. Collision exclusions and actuator limits are
explicit simulation assumptions, not hardware certification.
"""
from __future__ import annotations
from dataclasses import dataclass,field
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull,QhullError
from scipy.spatial.transform import Rotation
from .hexfit import fit_collision_hexagon,containment_violation

REPO_ROOT=Path(__file__).resolve().parents[3]
DEFAULT_XML=REPO_ROOT/'src/hexapod_mjlab/assets/robots/el_4090/mjcf/el_4090_lidar.xml'
LEGS=('RF','RM','RB','LF','LM','LB')
JOINT_TYPES=('HAA','HFE','KFE')
JOINT_NAMES=tuple(f'{leg}_{kind}' for leg in LEGS for kind in JOINT_TYPES)
JOINT_RANGE_LIMIT=np.pi  # numerical search domain, never called a mechanical stop
TORQUE_LIMIT=80.         # uncalibrated simulation assumption
COLLISION_TOL=1e-6
IK_POS_TOL=1e-5
IK_SEEDS=((0,0,0),(0,.5,-1),(0,-.5,1),(0,1,-2),(0,-1,2),(.8,.6,-1.2),(-.8,-.6,1.2),(2.,.3,-.6),(-2.,-.3,.6))


def compiled_hull(model,mesh):
    """Read the actual compiled hull used by collision detection.

    Graph triangle layout follows MuJoCo render_context.c. Validate all
    addresses/indices and retain faces for independent projection clipping.
    Unsupported layouts fail closed, rather than silently using render
    vertices. Tested against native plane/mesh distances in run_checks.
    """
    adr=int(model.mesh_graphadr[mesh]);va=int(model.mesh_vertadr[mesh]);vn=int(model.mesh_vertnum[mesh])
    if adr<0:raise RuntimeError('compiled collision hull required')
    nv,nf=map(int,model.mesh_graph[adr:adr+2]);start=adr+2+3*nv+3*nf
    faces=model.mesh_graph[start:start+3*nf].reshape(nf,3)
    ids=np.unique(faces)
    if len(ids)!=nv or np.min(ids)<0 or np.max(ids)>=vn:raise RuntimeError('unsupported convex graph layout')
    raw=np.array(model.mesh_vert[va:va+vn],dtype=float)
    return raw[ids],raw[faces]


@dataclass
class MorphologyMeasurement:
    occupancy_base:np.ndarray=field(repr=False)
    collision_polygons:list=field(repr=False)
    hex_params:dict
    hex_violation:float
    feet_base:dict
    body_height:float
    foot_origin_height:float|None
    n_stance_feet:int
    support_polygon_area:float|None
    com_xy:np.ndarray
    com_margin:float|None
    ground_flags:dict
    bottom_clearance:float
    top_height:float
    nonfoot_clearance:float
    contact_report:dict
    joint_angles:np.ndarray=field(repr=False)


def polygon_metrics(points,com):
    p=np.asarray(points)
    if len(p)<3:return None,None
    try:h=ConvexHull(p[:,:2])
    except QhullError:return None,None
    v=p[h.vertices,:2];edges=np.roll(v,-1,axis=0)-v
    n=np.stack([-edges[:,1],edges[:,0]],axis=1);n/=np.linalg.norm(n,axis=1)[:,None]
    return float(np.min(np.sum(n*(np.asarray(com)-v),axis=1))),float(h.volume)


class MorphologyMeter:
    def __init__(self,xml_path=DEFAULT_XML,floor=True,floor_friction=1.,ground_normal=(0,0,1),ground_height=0.):
        self.xml_path=Path(xml_path).resolve();spec=mujoco.MjSpec.from_file(str(self.xml_path))
        normal=np.asarray(ground_normal,dtype=float);normal/=np.linalg.norm(normal)
        if normal[2]<.2:raise ValueError('ground must have upward normal')
        self.ground_normal=normal;self.ground_height=float(ground_height);self.has_floor=bool(floor)
        self.friction=float(floor_friction)
        if floor:
            rot,_=Rotation.align_vectors([normal],[[0,0,1]]);q=rot.as_quat()[[3,0,1,2]]
            spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE,name='morphology_floor',size=[0,0,.1],pos=[0,0,ground_height],quat=q,friction=[floor_friction,.005,.0001])
        self.model=spec.compile();self.data=mujoco.MjData(self.model);m=self.model
        self.base_body_id=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'BASE')
        self.foot_body_ids={leg:mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,f'{leg}_FOOT') for leg in LEGS}
        self.joint_ids=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in JOINT_NAMES])
        self.joint_qpos_adr=m.jnt_qposadr[self.joint_ids].copy();self.joint_dof_adr=m.jnt_dofadr[self.joint_ids].copy()
        self.declared_joint_ranges=m.jnt_range[self.joint_ids].copy()
        m.jnt_limited[self.joint_ids]=False
        self.ground_geom_ids={g for g in range(m.ngeom) if m.geom_bodyid[g]==0 and m.geom_type[g]==mujoco.mjtGeom.mjGEOM_PLANE}
        self._occ_geoms=[];self._body_geoms={};self._triangles={}
        cache={}
        for g in range(m.ngeom):
            b=int(m.geom_bodyid[g])
            if b==0 or (m.geom_contype[g]==0 and m.geom_conaffinity[g]==0):continue
            if m.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH:raise RuntimeError('mesh robot expected')
            mesh=int(m.geom_dataid[g])
            if mesh not in cache:cache[mesh]=compiled_hull(m,mesh)
            verts,tri=cache[mesh];self._occ_geoms.append((g,b,verts));self._body_geoms.setdefault(b,[]).append((g,verts));self._triangles[g]=tri
        self.foot_geom_ids={g for g,b,v in self._occ_geoms if b in self.foot_body_ids.values()}
        axis=np.array([1.,0,0]) if abs(normal[0])<.8 else np.array([0,1.,0])
        tangent=np.cross(normal,axis);tangent/=np.linalg.norm(tangent)
        self.force_basis=np.stack([tangent,np.cross(normal,tangent),normal],axis=1)

    def set_state(self,base_pos,base_quat_wxyz,joint_angles):
        pos=np.asarray(base_pos,dtype=float);quat=np.asarray(base_quat_wxyz,dtype=float)
        angles=np.array([joint_angles[n] for n in JOINT_NAMES]) if hasattr(joint_angles,'keys') else np.asarray(joint_angles,dtype=float)
        if pos.shape!=(3,) or quat.shape!=(4,) or angles.shape!=(18,) or not np.isfinite(np.r_[pos,quat,angles]).all():raise ValueError('finite complete state required')
        if np.linalg.norm(quat)<1e-12:raise ValueError('nonzero quaternion required')
        mujoco.mj_resetData(self.model,self.data)
        self.data.qpos[:3]=pos;self.data.qpos[3:7]=quat/np.linalg.norm(quat);self.data.qpos[self.joint_qpos_adr]=angles
        mujoco.mj_forward(self.model,self.data)

    def to_base_frame(self,points_world):
        return (np.asarray(points_world)-self.data.xpos[self.base_body_id])@self.data.xmat[self.base_body_id].reshape(3,3)

    def world_geom(self,g,vertices=None):
        if vertices is None:vertices=next(v for a,b,v in self._occ_geoms if a==g)
        return vertices@self.data.geom_xmat[g].reshape(3,3).T+self.data.geom_xpos[g]

    def world_occupancy(self):return np.vstack([self.world_geom(g,v) for g,b,v in self._occ_geoms])

    def collision_polygons(self):return [self.to_base_frame(self.world_geom(g,v))[:,:2] for g,b,v in self._occ_geoms]

    def signed_ground_distance(self,points):
        return (np.asarray(points)-np.array([0,0,self.ground_height]))@self.ground_normal

    def ground_z(self,xy):
        xy=np.asarray(xy);return self.ground_height-(xy@self.ground_normal[:2])/self.ground_normal[2]

    def _lowest_point_of_body(self,b):
        points=np.vstack([self.world_geom(g,v) for g,v in self._body_geoms[b]])
        return points[np.argmin(self.signed_ground_distance(points))]

    def clearance(self,exclude_feet=False):
        points=np.vstack([self.world_geom(g,v) for g,b,v in self._occ_geoms if not exclude_feet or g not in self.foot_geom_ids])
        return float(self.signed_ground_distance(points).min())

    def contacts(self,forces=True):
        m,d=self.model,self.data;out={'foot_ground':[],'nonfoot_ground':[],'self':[],'feet':[]};feet=set()
        for i,c in enumerate(d.contact):
            g1,g2=int(c.geom1),int(c.geom2);b1,b2=int(m.geom_bodyid[g1]),int(m.geom_bodyid[g2])
            force=np.zeros(6)
            if forces and c.efc_address>=0:mujoco.mj_contactForce(m,d,i,force)
            r={'geom1':g1,'geom2':g2,'body1':mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,b1),'body2':mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,b2),'distance':float(c.dist),'point':c.pos.tolist(),'normal_force':float(force[0])}
            if g1 in self.ground_geom_ids or g2 in self.ground_geom_ids:
                robot=g2 if g1 in self.ground_geom_ids else g1
                if robot in self.foot_geom_ids:
                    out['foot_ground'].append(r)
                    if force[0]>1e-5 and c.dist<=COLLISION_TOL:feet.add(mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[robot]))[:2])
                else:out['nonfoot_ground'].append(r)
            elif b1!=b2 and m.body_weldid[b1]!=m.body_weldid[b2] and m.body_parentid[b1]!=b2 and m.body_parentid[b2]!=b1:
                out['self'].append(r)
        out['feet']=sorted(feet);return out

    def self_collision(self,exclude_floor=True):
        return [r for r in self.contacts(forces=False)['self'] if r['distance'] < -COLLISION_TOL]

    def measure(self):
        # Does not zero velocities/controls or perform inverse dynamics on live data.
        m,d=self.model,self.data;mujoco.mj_forward(m,d)
        polys=self.collision_polygons();occupancy=self.to_base_frame(self.world_occupancy());params=fit_collision_hexagon(polys)
        contacts=self.contacts();points=[r['point'][:2] for r in contacts['foot_ground'] if r['normal_force']>1e-5 and r['distance']<=COLLISION_TOL]
        com=d.subtree_com[self.base_body_id][:2].copy();margin,area=polygon_metrics(points,com)
        feet={l:self.to_base_frame(d.xpos[b]).copy() for l,b in self.foot_body_ids.items()}
        origin_height=-float(np.mean([v[2] for v in feet.values()])) if np.ptp([v[2] for v in feet.values()])<.005 else None
        base=d.xpos[self.base_body_id];height=float(base[2]-self.ground_z(base[:2]));world=self.world_occupancy()
        basepoints=np.vstack([self.world_geom(g,v) for g,v in self._body_geoms[self.base_body_id]])
        grounded=len(contacts['feet'])>=3 and margin is not None
        return MorphologyMeasurement(occupancy,polys,params,containment_violation(polys,params),feet,height,origin_height,len(contacts['feet']),area,com,margin,
          {'valid':grounded,'height_valid':self.has_floor,'support_valid':grounded,'feet_airborne':not bool(contacts['feet']),'slope_deg':float(np.degrees(np.arccos(self.ground_normal[2]))),'source':'explicit_ground_plane_and_actual_force_bearing_foot_contacts'},
          float(np.min(basepoints[:,2]-self.ground_z(basepoints[:,:2]))),float(np.max(world[:,2]-self.ground_z(world[:,:2]))),self.clearance(True),contacts,d.qpos[self.joint_qpos_adr].copy())

    def inverse_diagnostic(self):
        d=mujoco.MjData(self.model);d.qpos[:]=self.data.qpos;d.qvel[:]=0;mujoco.mj_forward(self.model,d);d.qacc[:]=0;mujoco.mj_inverse(self.model,d)
        return {'joint_loads':d.qfrc_inverse[self.joint_dof_adr].tolist(),'root_required_wrench':d.qfrc_inverse[:6].tolist(),'certifies_equilibrium':False}

    def leg_ik(self,leg,foot_target_base,sole_on_ground=False,seed_count=None,sole_target_offset=0.):
        if leg not in LEGS:raise ValueError(leg)
        target=np.asarray(foot_target_base,dtype=float);bid=self.foot_body_ids[leg];indices=np.array([JOINT_NAMES.index(f'{leg}_{k}') for k in JOINT_TYPES]);qa=self.joint_qpos_adr[indices];da=self.joint_dof_adr[indices]
        snapshot=self.data.qpos.copy();starts=[snapshot[qa],*map(np.array,IK_SEEDS)];starts=starts if seed_count is None else starts[:seed_count]
        last=None;cached=None
        def evaluate(q):
            nonlocal last,cached
            if last is not None and np.array_equal(last,q):return cached
            d=self.data;d.qpos[qa]=q;mujoco.mj_kinematics(self.model,d);mujoco.mj_comPos(self.model,d)
            foot=d.xpos[bid];R=d.xmat[self.base_body_id].reshape(3,3);footbase=self.to_base_frame(foot)
            jp=np.zeros((3,self.model.nv));jr=np.zeros_like(jp);mujoco.mj_jac(self.model,d,jp,jr,foot,bid)
            jac=(R.T@jp)[:,da];error=footbase-target
            if sole_on_ground:
                low=self._lowest_point_of_body(bid);mujoco.mj_jac(self.model,d,jp,jr,low,bid)
                error[2]=float(self.signed_ground_distance(low))-sole_target_offset;jac[2]=self.ground_normal@jp[:,da]
            last=q.copy();cached=(error,jac);return cached
        sols=[]
        try:
            for start in starts:
                last=None
                r=least_squares(lambda q:evaluate(q)[0],np.clip(start,-np.pi+1e-8,np.pi-1e-8),jac=lambda q:evaluate(q)[1],bounds=(-np.pi,np.pi),xtol=1e-10,ftol=1e-10,gtol=1e-10,max_nfev=80)
                if np.max(np.abs(evaluate(r.x)[0]))<IK_POS_TOL and not any(np.max(np.abs(r.x-s))<1e-3 for s in sols):sols.append(r.x.copy())
        finally:
            self.data.qpos[:]=snapshot;mujoco.mj_forward(self.model,self.data)
        sols.sort(key=lambda q:float(np.linalg.norm(q)))
        return [dict(zip((f'{leg}_{k}' for k in JOINT_TYPES),map(float,s))) for s in sols]
