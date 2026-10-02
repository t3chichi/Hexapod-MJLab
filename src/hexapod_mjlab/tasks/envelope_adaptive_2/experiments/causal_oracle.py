"""Current-motion preview, canonical shape tracking and growth hysteresis.

Uses current legacy ego velocity/yaw rate only (legacy lateral is right-positive); never accepts future pose
arrays. Preview is a heuristic goal, not a claim of guaranteed future motion.
The current output is still checked against exact current obstacle geometry.
"""
from time import perf_counter
import numpy as np
from .refined_oracle import RefinedOracle,RefinedResult
from .static_qp import SPAN


class CausalOracle:
    def __init__(self,rects,margin=.1,horizon=.24,weight=1.,cooldown=.1,dt=.02,deadband=0.,continuous_preview=False,soft_deadband=0.):
        self.solver=RefinedOracle(rects,margin)
        self.horizon=float(horizon);self.weight=float(weight);self.dt=float(dt)
        self.soft_deadband=float(soft_deadband)
        self.continuous_preview=bool(continuous_preview)
        self.cooldown_calls=max(0,int(round(cooldown/dt)))
        self.shrink_rate=2.;self.grow_rate=.5;self.deadband=float(deadband)
        self.prev={};self.counter={};self.last_debug={}

    def reset(self,ids):
        for e in ids:self.prev.pop(int(e),None);self.counter.pop(int(e),None)

    def step(self,pos,yaw,ego,env_id=0,seed=None,reference=None):
        start=perf_counter();e=int(env_id);prev=self.prev.get(e)
        if reference is not None and self.solver.clearance(reference,pos,yaw)>=self.solver.margin-1e-8:
            raw=RefinedResult('feasible',np.asarray(reference).copy(),0.,0,0.)
        else:raw=self.solver.solve(pos,yaw,seed=seed)
        self.last_debug={'preview_used':False,'preview_infeasible':False}
        if raw.status!='feasible':
            self.reset([e]);return raw
        desired=raw.s.copy()
        if self.horizon>0 and prev is not None and np.min(prev)<(.6 if self.continuous_preview else .5):
            c,s=np.cos(yaw),np.sin(yaw)
            world=np.asarray(ego[:2])*[1.,-1.]
            world=world@np.array([[c,s],[-s,c]])
            future=self.solver.solve(np.asarray(pos)+world*self.horizon,yaw+float(ego[2])*self.horizon)
            self.last_debug['preview_used']=True
            mix=.5*np.clip((.6-np.min(prev))/.2,0.,1.) if self.continuous_preview else .5
            if future.status=='feasible':desired=(1-mix)*desired+mix*future.s
            else:
                self.last_debug['preview_infeasible']=True
                # Current safety remains the hard constraint. Preparing the
                # minimum does not pretend an impossible future pose is safe.
                desired=(1-mix)*desired
        if prev is None:
            result=raw;self.counter[e]=np.zeros(5,dtype=int)
        else:
            delta=desired-prev
            desired=prev+np.sign(delta)*np.maximum(np.abs(delta)-self.soft_deadband,0.)
            desired=np.where(np.abs(desired-prev)<self.deadband,prev,desired)
            grow_step=self.grow_rate*self.dt/np.abs(SPAN)
            clear=desired>prev+.5*grow_step
            self.counter[e]=np.where(clear,self.counter.get(e,np.zeros(5,dtype=int))+1,0)
            allowed=clear&(self.counter[e]>=self.cooldown_calls)
            result=self.solver.solve(pos,yaw,seed=raw.s,prev=prev,weight=self.weight,dt=self.dt,
                shrink_rate=self.shrink_rate,grow_rate=self.grow_rate,desired=desired,grow_allowed=allowed)
        if result.status=='feasible':self.prev[e]=result.s.copy()
        else:self.reset([e])
        result.elapsed_s=perf_counter()-start
        return result
