"""Causal common-shape preview region; sampled future poses only, no swept-volume guarantee."""
from time import perf_counter
import numpy as np
from ..contracts import RectPrimitive
from .static_seed import NOMINAL_S,SPAN
from .refined import RefinedOracle


class LookaheadOracle:
    def __init__(self,rects,margin=.1,horizon=.12,weight=4.,cooldown=.1,dt=.02,yaw_reserve=0.,headroom=0.):
        self.headroom=float(headroom)
        self.yaw_reserve=float(yaw_reserve)
        self.rects=rects;self.solver=RefinedOracle(rects,margin,headroom=self.headroom)
        self.horizon=horizon;self.weight=weight;self.dt=dt
        self.prev={};self.counter={};self.last_debug={}
        self.cooldown_calls=max(0,int(round(cooldown/dt)))

    def reset(self,ids):
        for e in ids:self.prev.pop(int(e),None);self.counter.pop(int(e),None)

    def step(self,pos,yaw,ego,env_id=0,seed=None,reference=None):
        start=perf_counter();e=int(env_id);prev=self.prev.get(e)
        self.last_debug={'preview_used':False,'preview_infeasible':False}
        raw=self.solver.solve(pos,yaw,seed=seed)
        if raw.status!='feasible':self.reset([e]);return raw
        c,s=np.cos(yaw),np.sin(yaw)
        world=(np.asarray(ego[:2])*[1.,-1.])@np.array([[c,s],[-s,c]])
        omega=float(ego[2]);pos=np.asarray(pos)
        combined=list(self.rects)
        if np.linalg.norm(world)>1e-6 or abs(omega)>1e-6:
            for tau in [self.horizon/2,self.horizon]:
                predicted=pos+world*tau;angle=omega*tau
                if self.solver.clearance(np.zeros(5),predicted,yaw+angle)<self.solver.margin-1e-8:
                    self.last_debug['preview_infeasible']=True;break
                # Express future obstacles in the CURRENT world frame so one
                # parameter vector has the same shape at every sampled pose.
                ca,sa=np.cos(angle),np.sin(angle)
                inverse=np.array([[ca,-sa],[sa,ca]])
                for index in self.solver._nearby(predicted):
                    rect=self.rects[index]
                    center=pos+(np.asarray(rect.center)-predicted)@inverse
                    combined.append(RectPrimitive(tuple(center),rect.size,rect.yaw-angle))
                self.last_debug['preview_used']=True
        # Low minimum-footprint clearance warns that the available shape
        # reserve is nearly exhausted. Sample a small heading uncertainty band
        # proactively, including while translation has stopped before a turn.
        # Impossible forecast poses are omitted, never relabel current poses.
        minimum_gap=self.solver.clearance(np.zeros(5),pos,yaw)
        reserve=self.yaw_reserve*np.clip((self.solver.margin+.15-minimum_gap)/.1,0.,1.)
        self.last_debug['yaw_reserve_rad']=float(reserve)
        if reserve>1e-8:
            for tau in [0.,self.horizon]:
                predicted=pos+world*tau
                for offset in [-reserve,reserve]:
                    angle=omega*tau+offset
                    if self.solver.clearance(np.zeros(5),predicted,yaw+angle)<self.solver.margin-1e-8:
                        self.last_debug['preview_infeasible']=True;continue
                    ca,sa=np.cos(angle),np.sin(angle);inverse=np.array([[ca,-sa],[sa,ca]])
                    for index in self.solver._nearby(predicted):
                        rect=self.rects[index]
                        center=pos+(np.asarray(rect.center)-predicted)@inverse
                        combined.append(RectPrimitive(tuple(center),rect.size,rect.yaw-angle))
                    self.last_debug['preview_used']=True
        region=RefinedOracle(combined,self.solver.margin,headroom=self.headroom)
        preview=region.solve(pos,yaw,seed=raw.s)
        desired=preview.s if preview.status=='feasible' else raw.s
        if prev is None:result=raw;self.counter[e]=np.zeros(5,int)
        else:
            clear=desired>prev+.5*.5*self.dt/np.abs(SPAN)
            self.counter[e]=np.where(clear,self.counter[e]+1,0)
            allowed=clear&(self.counter[e]>=self.cooldown_calls)
            result=self.solver.solve(pos,yaw,seed=raw.s,prev=prev,weight=self.weight,dt=self.dt,desired=desired,grow_allowed=allowed)
        if result.status=='feasible':self.prev[e]=result.s.copy()
        else:self.reset([e])
        result.elapsed_s=perf_counter()-start
        return result
