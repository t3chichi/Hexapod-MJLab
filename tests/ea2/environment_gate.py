"""Real mjlab sensor, clock, reset, motion and supervision checks."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from hexapod_mjlab.tasks.envelope_adaptive_2.env import make_env, SLAdapter
from hexapod_mjlab.tasks.envelope_adaptive_2.config.parameters import EA2Cfg
from hexapod_mjlab.tasks.envelope_adaptive_2.lidar import EnvelopeLidar
from hexapod_mjlab.tasks.envelope_adaptive_2.math_utils import wrap_to_pi, selected_channel_mask
from hexapod_mjlab.tasks.envelope_adaptive_2.quaternion import quat_from_euler_xyz


def main():
    torch.set_num_threads(1)
    cfg=EA2Cfg();cfg.envelope.oracle_kind="legacy";cfg.env.num_envs=2;cfg.lidar.enable_sensor_noise=False
    env=make_env(cfg);adapter=SLAdapter(env)
    try:
        obs=adapter.reset();s=env.ea2
        assert obs.shape==(2,190) and s.sensor._lidar_timer==5
        # Actual geometry once per frame, reads never change smoother/counters.
        prev=s._oracle_smoother.prev_s.clone();counter=s._oracle_smoother.counter.clone()
        env.get_observations();env.get_observations()
        assert torch.equal(prev,s._oracle_smoother.prev_s) and torch.equal(counter,s._oracle_smoother.counter)
        z=torch.zeros((2,5),device=env.device)
        adapter.step(z)
        ids=torch.tensor([0],device=env.device);other=s.base_pos[1].clone()
        env.reset(env_ids=ids)
        assert torch.equal(other,s.base_pos[1]) and (s.range_image[0]==s.range_max).all()
        for _ in range(3):
            adapter.step(z);assert (s.range_image[0]==s.range_max).all()
        adapter.step(z)
        assert not s.range_image_stale.any() and (s.range_image[0]<s.range_max).any()
        assert env.ea2.logical_step==6
        # Compare source motion function on identical tensor inputs, away from goals.
        source=Path('/home/t3chichi/el4090_legged_gym/legged_gym/legged_gym/envs/el_4090/envelope_adaptive_2/el_4090_ea2_env.py')
        tree=ast.parse(source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
        fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_step_kinematics_batched')
        ns={'torch':torch,'wrap_to_pi':wrap_to_pi}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),ns)
        attrs=['s','v','heading','tangent','tangent_rate','delta_target','delta_actual','omega','base_pos',
               '_turn_in_place','_turn_target','ego_motion']
        ref=SimpleNamespace(**{name:getattr(s,name).clone() for name in attrs})
        ref.num_envs=s.num_envs;ref.device=s.device;ref.dt=s.dt;ref.cfg=s.cfg
        ref._path_batch=s._path_batch;ref._attitude=None
        ref._batch_replan=lambda ids: (_ for _ in ()).throw(AssertionError('probe reached path end'))
        for _ in range(20):
            s._step_kinematics_batched();ns['_step_kinematics_batched'](ref)
            for name in attrs:assert torch.allclose(getattr(s,name),getattr(ref,name),atol=1e-6,rtol=0),name
        # Full/reduced raw rays, including miss sentinel, at fixed nongrazing poses.
        fullcfg=copy.deepcopy(cfg);fullcfg.lidar.use_reduced_raycast=False
        full=EnvelopeLidar(fullcfg,s.map_data,2,env.device,capture_debug=True)
        reduced=EnvelopeLidar(cfg,s.map_data,2,env.device,capture_debug=True)
        pos=torch.tensor([[0,0,.52],[10,0,.52]],dtype=torch.float32,device=env.device)
        heading=torch.tensor([.31,-.42],device=env.device)
        full.scan(pos,heading);reduced.scan(pos,heading)
        a=full.lidar_sensor.lidar_dist_tensor.view(2,-1)[:,full.selected_ray_indices]
        b=reduced._reduced_dist_tensor.view(2,-1)
        assert torch.equal(a<60,b<60)
        assert torch.allclose(a,b,atol=1e-4,rtol=1e-5),(a-b).abs().max()
        assert torch.allclose(full.range_image,reduced.range_image,atol=1e-4,rtol=1e-5)
        assert not full._debug_is_reduced and reduced._debug_is_reduced
        red,green=selected_channel_mask(full._debug_dists[0],full.selected_ray_indices,False,60)
        assert red.any() and green.any() and not (red&green).any()
        # Sensor pose matches explicit xyzw rotation; entity carrier matches wxyz.
        q=quat_from_euler_xyz(torch.zeros_like(s.heading),torch.zeros_like(s.heading),s.heading)
        assert torch.allclose(s.carrier_pose()[:,3:],q[:,[3,0,1,2]],atol=1e-6)
        # Short episode covers the exact old >= boundary and a forced path failure.
        env.cfg.episode_length_s=.1
        env.reset()
        for i in range(5):
            _,_,done,_=adapter.step(z)
            assert bool(done.all()) == (i==4)
        s._replan_reset_buf[0]=True
        _,_,done,_=adapter.step(z)
        assert done.tolist()==[True,False]
        print('T2 GPU PASS: source motion, full/reduced rays, miss sentinel, 10Hz/reset stale, cache, timeout and path failure')
    finally:env.close()
    # Attitude smoke uses the packaged source and produces finite tilted ray frames.
    cfg.attitude.enable=True;env=make_env(cfg)
    try:
        adapter=SLAdapter(env);adapter.reset()
        for _ in range(6):adapter.step(torch.zeros((2,5),device=env.device))
        for t in [env.ea2.attitude_pitch,env.ea2.attitude_roll,env.ea2.base_pos,env.ea2.observation]:
            assert torch.isfinite(t).all()
        print('T2 ATTITUDE PASS: packaged replay, finite pitch/roll/height/ranges')
    finally:env.close()


if __name__=='__main__':main()
