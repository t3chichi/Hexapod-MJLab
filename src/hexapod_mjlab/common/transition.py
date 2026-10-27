"""Instance-local control-boundary seam for the ordinary mjlab environment.

This provides clean physical data, not an optimizer or a training-ready flag.
Runtime owns begin/finish, scanning, commands and selected-row reset. Four
solver contact tables describe pre-integration substeps; boundary net force
is read separately after forward and explicit contact-cache invalidation.
"""
from __future__ import annotations

import numpy as np
import torch
import warp as wp

from .diagnostics import ContactTableReader
from .robot import FOOT_BODIES, SIM_TIMESTEP_S, DECIMATION

PERSISTENT_PHYSICS_FIELDS = (
    'time', 'qpos', 'qvel', 'qacc_warmstart', 'ctrl', 'qfrc_applied',
    'xfrc_applied', 'eq_active', 'act', 'act_dot', 'mocap_pos', 'mocap_quat', 'userdata',
)


def require_finite(name, value):
    """Reject faults before managers can sanitize or clip them."""
    if isinstance(value, torch.Tensor):
        valid = bool(torch.isfinite(value).all())
    else:
        valid = bool(np.isfinite(value).all())
    if not valid:
        raise RuntimeError(f'{name} contains NaN/Inf')
    return value


def physics_field_manifest(env):
    """Record actual storage; empty fields remain declared, not invented."""
    result = {}
    for name in PERSISTENT_PHYSICS_FIELDS:
        array = getattr(env.sim.wp_data, name, None)
        if array is None:
            raise RuntimeError(f'backend persistent field absent: {name}')
        result[name] = dict(shape=list(array.shape), dtype=str(array.dtype),
                            ptr=int(array.ptr or 0), empty=not array.size)
    return result


def get_transition_context(env):
    context = getattr(env, '_confined_transition_context', None)
    if not isinstance(context, TransitionContext):
        raise RuntimeError('formal MDP called without a bound TransitionContext')
    return context


class TransitionContext:
    """One step at a time, bounded evidence and idempotent ready callbacks.

    close restores exact original bound methods. A fault freezes the context;
    reset is a separate runtime operation, never an implicit recovery.
    """
    def __init__(self, env, *, terrain_query=None):
        if env.cfg.auto_reset or env.cfg.decimation != DECIMATION:
            raise ValueError('training seam requires auto_reset=False and four substeps')
        if abs(env.physics_dt - SIM_TIMESTEP_S) > 1e-12:
            raise ValueError('unexpected physics timestep')
        if hasattr(env, '_confined_transition_context'):
            raise RuntimeError('transition context already bound')
        self.env = env
        self.robot = env.scene['robot']
        self.action = env.action_manager.get_term('residual_pd')
        self.sensor = env.scene['feet_any_contact']
        if not callable(getattr(self.sensor, '_invalidate_cache', None)):
            raise RuntimeError('contact-cache invalidation API unavailable')
        if self.sensor.cfg.reduce != 'netforce' or self.sensor.cfg.secondary is not None:
            raise RuntimeError('foot sensor must measure ANY world netforce')
        self.contact_reader = ContactTableReader(env)
        names = self.robot.body_names
        self.foot_local_ids = [next(i for i,n in enumerate(names)
                                   if n.split('/')[-1] == foot) for foot in FOOT_BODIES]
        mm = env.sim.mj_model
        self.foot_global_ids = [next(i for i,n in enumerate(self.contact_reader.body_names)
                                    if n.split('/')[-1] == foot) for foot in FOOT_BODIES]
        self.terrain_query = terrain_query
        if terrain_query is None:
            planes = np.flatnonzero(np.asarray(mm.geom_type) == 0)
            if len(planes) != 1:
                raise RuntimeError('expected exactly one actual ground plane')
            self.plane_geom_id = int(planes[0])
        else:
            if terrain_query.env is not env:
                raise ValueError('terrain query must own the same environment')
            # Existing support bookkeeping uses this field for the actual
            # ground collider; Terrain Memory supplies the compiled HField.
            self.plane_geom_id = terrain_query.surface.geom_id
        self.actuator_ids = []
        self.actuator_gears = []
        for name in self.action._target_names:
            full = 'robot/' + name.split('/')[-1]
            joint = next(i for i in range(mm.njnt) if mm.joint(i).name == full)
            ids = np.flatnonzero(np.asarray(mm.actuator_trnid)[:,0] == joint)
            if len(ids) != 1:
                raise RuntimeError(f'expected one torque actuator for {full}')
            aid = int(ids[0]); self.actuator_ids.append(aid)
            self.actuator_gears.append(float(mm.actuator_gear[aid,0]))
        self.phase = 'idle'
        self.fault = None
        self.record = None
        self.tables = []
        self.pd = []
        self.forward_count = 0
        self.closed = False
        self._orig_update = env.scene.update
        self._orig_apply = self.action.apply_actions
        self._orig_step = env.step
        env._confined_transition_context = self
        env.scene.update = self._update
        self.action.apply_actions = self._apply
        env.step = self._step

    def _check_overflow(self):
        bits = wp.to_torch(self.env.sim.wp_data.overflow).clone()
        if bool((bits != 0).any()):
            raise RuntimeError(f'physics/contact-match overflow: {bits.cpu().tolist()}')
        return bits

    def _freeze(self, exc):
        self.fault = {'phase': self.phase, 'control_id': getattr(self, 'control_id', None),
                      'error': repr(exc)}
        self.phase = 'fault'

    def begin(self, command):
        if self.closed or self.phase != 'idle' or self.fault:
            raise RuntimeError('transition cannot begin: closed, active or faulted')
        if tuple(command.shape) != (self.env.num_envs, 3) or command.device != self.action.q_ref.device:
            raise ValueError('command must have shape (envs,3) on the environment device')
        require_finite('command', command)
        if bool(self.env._manual_reset_pending.any()):
            raise RuntimeError('pending terminal rows require explicit selected reset')
        self.control_id = int(self.env.common_step_counter) + 1
        self.command = command.detach().clone()
        self.previous_action = self.action.prev_action.clone()
        self.previous_qdot = self.robot.data.joint_vel.clone()
        self.previous_time = wp.to_torch(self.env.sim.wp_data.time).clone()
        self.tables = []; self.pd = []; self.record = None; self.forward_count = 0
        self.phase = 'capturing'

    def _step(self, actions):
        if self.phase != 'capturing':
            raise RuntimeError('step requires an explicit begin; no reset/fault bypass')
        try:
            return self._orig_step(actions)
        except Exception as exc:
            self._freeze(exc)
            raise

    def _apply(self):
        if self.phase != 'capturing':
            raise RuntimeError('unbracketed physical action')
        try:
            ids = self.action._target_ids
            q = self.robot.data.joint_pos[:,ids].clone()
            qdot = self.robot.data.joint_vel[:,ids].clone()
            target = self.action.last_target.clone()
            self._orig_apply()
            effort = self.robot.data.joint_effort_target[:,ids].clone()
            for key,value in [('q',q),('qdot',qdot),('target',target),('effort',effort)]:
                require_finite(key,value)
            self.pd.append({'q':q,'qdot':qdot,'target':target,'effort':effort})
        except Exception as exc:
            self._freeze(exc)
            raise

    def _update(self, dt):
        self._orig_update(dt)
        if self.phase != 'capturing':
            return
        try:
            if abs(dt-self.env.physics_dt) > 1e-12 or len(self.pd) != len(self.tables)+1:
                raise RuntimeError('substep bracket/time mismatch')
            self._check_overflow()
            table = self.contact_reader.read_substep_table()
            for name in ('dist','pos','frame','force6_world'):
                require_finite('contact/'+name,table[name])
            ctrl = wp.to_torch(self.env.sim.wp_data.ctrl)[:,self.actuator_ids].clone()
            applied = ctrl * torch.tensor(self.actuator_gears,device=ctrl.device)
            if not torch.allclose(applied,self.pd[-1]['effort'],atol=2e-4,rtol=1e-5):
                raise RuntimeError('actual ctrl/gear differs from applied joint effort')
            self.pd[-1].update(ctrl=ctrl, gear=list(self.actuator_gears),
                               solver_state_semantics='pre-integration',
                               post_qpos=wp.to_torch(self.env.sim.wp_data.qpos).clone())
            self.tables.append(table)
            if len(self.tables)>DECIMATION:
                raise RuntimeError('extra physical substep')
        except Exception as exc:
            self._freeze(exc)
            raise

    def ensure_ready(self):
        if self.closed or self.fault:
            raise RuntimeError('transition context closed/frozen')
        if self.phase == 'ready':
            if self.env.common_step_counter != self.control_id:
                raise RuntimeError('stale transition identity')
            return self.record
        try:
            if self.phase != 'capturing' or len(self.tables) != DECIMATION or len(self.pd) != DECIMATION:
                raise RuntimeError('MDP needs exactly four completed physical substeps')
            if self.env.common_step_counter != self.control_id:
                raise RuntimeError('control identity mismatch')
            cached_force = self.sensor.data.force.detach().clone()
            self.env.sim.forward()
            self.forward_count += 1
            overflow = self._check_overflow()
            self.sensor._invalidate_cache()
            force = self.sensor.data.force.detach().clone()
            # Freeze this forward's readings before mjlab's retained final
            # forward can run the solver again with a new warm start.
            boundary_table = self.contact_reader.read_substep_table()
            sensor_data = wp.to_torch(self.env.sim.wp_data.sensordata).clone()
            primary = [n.split('/')[-1] for n in self.sensor.primary_names]
            order = [primary.index(n) for n in FOOT_BODIES]
            force = force[:,order]
            counts = [self.contact_reader.substep_counts(t) for t in self.tables]
            root_pos = self.robot.data.root_link_pos_w.clone()
            root_quat = self.robot.data.root_link_quat_w.clone()
            foot_pos = self.robot.data.body_link_pos_w[:,self.foot_local_ids].clone()
            foot_vel = self.robot.data.body_link_lin_vel_w[:,self.foot_local_ids].clone()
            support = torch.zeros((self.env.num_envs,6),device=force.device,dtype=torch.bool)
            last = self.tables[-1]
            for j,bid in enumerate(self.foot_global_ids):
                geometry = np.asarray(last['geom'])
                has = (((last['body1']==bid)&(geometry[:,1]==self.plane_geom_id)) |
                       ((last['body2']==bid)&(geometry[:,0]==self.plane_geom_id)))
                qualified = has & (last['dist']<=0) & (np.linalg.norm(last['force6_world'][:,:3],axis=1)>0.1)
                support[torch.as_tensor(last['worldid'][qualified],device=force.device,dtype=torch.long),j] = True
            self.record = {'control_id':self.control_id,'command':self.command,
                'previous_action':self.previous_action,'previous_qdot':self.previous_qdot,
                'raw_action':self.action.last_raw_action.clone(),
                'clipped_action':self.action.last_clipped_action.clone(),
                'root_pos_w':root_pos,'root_quat_w':root_quat,
                'lin_vel_b':self.robot.data.root_link_lin_vel_b.clone(),
                'ang_vel_b':self.robot.data.root_link_ang_vel_b.clone(),
                'gravity_b':self.robot.data.projected_gravity_b.clone(),
                'joint_pos':self.robot.data.joint_pos.clone(),
                'joint_vel':self.robot.data.joint_vel.clone(),
                'qpos':wp.to_torch(self.env.sim.wp_data.qpos).clone(),
                'qvel':wp.to_torch(self.env.sim.wp_data.qvel).clone(),
                'time':wp.to_torch(self.env.sim.wp_data.time).clone(),
                'foot_pos_w':foot_pos,'foot_vel_w':foot_vel,'support':support,
                'foot_force_w':force,'cached_force_before_forward':cached_force,
                'boundary_raw_contacts':boundary_table,'boundary_sensor_data':sensor_data,
                'c_nonfoot_env':torch.stack([c['nonfoot_env'] for c in counts]).mean(0),
                'c_self':torch.stack([c['self_pairs'] for c in counts]).mean(0),
                'overflow':overflow,'substep_tables':self.tables,'pd':self.pd}
            for name,value in self.record.items():
                if isinstance(value,torch.Tensor):require_finite('boundary/'+name,value)
            self.phase = 'ready'
            return self.record
        except Exception as exc:
            self._freeze(exc)
            raise

    def finish(self):
        record = self.ensure_ready()
        require_finite('manager reward',self.env.reward_buf)
        record.update(reward=self.env.reward_buf.clone(),
                      terminated=self.env.reset_terminated.clone(),
                      truncated=self.env.reset_time_outs.clone())
        self.phase = 'idle'
        return record

    def close(self):
        if self.closed:return
        self.env.scene.update = self._orig_update
        self.action.apply_actions = self._orig_apply
        self.env.step = self._orig_step
        if getattr(self.env,'_confined_transition_context',None) is self:
            del self.env._confined_transition_context
        self.closed = True
