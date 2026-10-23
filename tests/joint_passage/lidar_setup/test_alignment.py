#!/usr/bin/env python
"""Actual optical site and unchanged physical assembly; no environment/physics steps."""
import hashlib,importlib.util,json,sys,unittest
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('flat_diagnostic',ROOT/'scripts/experiments/joint_passage/lidar_setup/compare_flat.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

class Alignment(unittest.TestCase):
    def test_full_optical_frame_and_actual_resolver(self):
        model=mod.load_model(mod.MODEL);_,_,_,_,_,r,t=mod.frame(model)
        expected=np.array([[0,np.sin(.1),np.cos(.1)],[1,0,0],[0,np.cos(.1),-np.sin(.1)]])
        np.testing.assert_allclose(r,expected,atol=1e-6,rtol=0)
        np.testing.assert_allclose(r.T@r,np.eye(3),atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(r),1.,places=12)
        urdf=mod.urdf_mount();np.testing.assert_allclose(r,urdf[:3,:3],atol=1e-6,rtol=0)
        np.testing.assert_allclose(t,urdf[:3,3],atol=1e-6,rtol=0)
        pos,q=mod.mount_resolver()(mod.MODEL)
        np.testing.assert_allclose(pos,t,atol=1e-12)
        np.testing.assert_allclose(Rotation.from_quat(q).as_matrix(),r,atol=1e-12)

    def test_old_error_is_detected_and_origin_unchanged(self):
        old=mod.load_model(mod.MODEL,legacy_optical_frame=True);new=mod.load_model(mod.MODEL)
        *_,ro,to=mod.frame(old);*_,rn,tn=mod.frame(new)
        np.testing.assert_allclose(to,tn,atol=1e-12)
        self.assertLess(ro[2,2],-.99);self.assertGreater(rn[0,2],.99)
        self.assertAlmostEqual(ro[:,2]@rn[:,2],0.,places=12)
        with self.assertRaises(AssertionError):
            np.testing.assert_allclose(ro[:,2],[np.cos(.1),0,-np.sin(.1)],atol=1e-6)

    def test_physical_assembly_and_dynamics_unchanged(self):
        baseline=json.loads((ROOT/'tests/data/lidar_alignment_baseline.json').read_text())
        for label,path in [('mjcf',mod.MODEL),('urdf',mod.URDF)]:
            tree=ET.fromstring(path.read_text())
            if label=='mjcf':tree.find(".//body[@name='LIDAR_SENSOR']").attrib.pop('quat',None)
            else:tree.find(".//joint[@name='LIDAR_SENSOR_FIX']/origin").set('rpy','0 0 0')
            canonical=ET.canonicalize(ET.tostring(tree,encoding='unicode'),strip_text=True)
            self.assertEqual(hashlib.sha256(canonical.encode()).hexdigest(),baseline[label+'_physical_tree_sha256'])
        old=mod.load_model(mod.MODEL,legacy_optical_frame=True);new=mod.load_model(mod.MODEL)
        self.assertEqual((old.nq,old.nv,old.nbody,old.ngeom,old.nmesh),(new.nq,new.nv,new.nbody,new.ngeom,new.nmesh))
        sensor=mujoco.mj_name2id(new,mujoco.mjtObj.mjOBJ_BODY,'LIDAR_SENSOR')
        self.assertEqual(new.body_mass[sensor],0.)
        np.testing.assert_array_equal(new.body_inertia[sensor],[0,0,0])
        self.assertFalse(np.any(new.geom_bodyid==sensor))
        physical=np.arange(new.nbody)!=sensor
        np.testing.assert_array_equal(old.body_iquat[physical],new.body_iquat[physical])
        for name in ['body_mass','body_inertia','body_ipos','geom_pos','geom_quat','geom_size','geom_contype','geom_conaffinity','geom_group','geom_bodyid','geom_dataid','mesh_vert','mesh_face','jnt_range','jnt_pos','jnt_axis','jnt_limited','dof_damping','dof_armature','actuator_gear','actuator_ctrlrange','actuator_forcerange','key_qpos','key_qvel','key_ctrl']:
            np.testing.assert_array_equal(getattr(old,name),getattr(new,name),err_msg=name)
        records=json.loads(mod.LIBRARY.read_text())['records']
        for rec in records:
            do,*_=mod.frame(old,rec['measured_qpos']);dn,*_=mod.frame(new,rec['measured_qpos'])
            np.testing.assert_allclose(do.geom_xpos,dn.geom_xpos,atol=1e-12)
            np.testing.assert_allclose(do.geom_xmat,dn.geom_xmat,atol=1e-12)
            mo=np.empty((old.nv,old.nv));mn=np.empty((new.nv,new.nv))
            mujoco.mj_fullM(old,do,mo);mujoco.mj_fullM(new,dn,mn)
            np.testing.assert_allclose(mo,mn,atol=1e-12,rtol=0)
            np.testing.assert_allclose(do.qfrc_bias,dn.qfrc_bias,atol=1e-12)
            np.testing.assert_allclose(do.qfrc_passive,dn.qfrc_passive,atol=1e-12)
            self.assertEqual(do.ncon,dn.ncon)
            np.testing.assert_allclose(do.contact.dist[:do.ncon],dn.contact.dist[:dn.ncon],atol=1e-12)

    def test_nonidentity_base_pose_and_roundtrip(self):
        model=mod.load_model(mod.MODEL);q=model.key_qpos[0].copy()
        q[:3]=[1.2,-.7,1.1];q[3:7]=Rotation.from_euler('xyz',[.31,-.22,.67]).as_quat()[[3,0,1,2]]
        _,rb,pb,rs,ps,r,t=mod.frame(model,q)
        pos,quat=mod.mount_resolver()(mod.MODEL)
        np.testing.assert_allclose(rs,rb@Rotation.from_quat(quat).as_matrix(),atol=1e-12)
        np.testing.assert_allclose(ps,pb+rb@pos,atol=1e-12)
        dirs=mod.pattern()[::151];world=ps+(dirs@rs.T)*3.1
        recovered=((world-pb)@rb-t)@r
        np.testing.assert_allclose(recovered,dirs*3.1,atol=1e-12)
        self.assertGreater(float(mod.cone(r)[0]),.47)
        self.assertLess(float(mod.cone(r)[2]),-.88)

if __name__=='__main__':unittest.main(verbosity=2)
