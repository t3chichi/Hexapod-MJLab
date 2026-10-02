"""Preserve safe shape modes; grow without reallocating already-safe extents.

Forecast geometry guides changes before a mode reaches its minimum. Actual
outputs always pass current geometry, but not necessarily future geometry.
"""
from time import perf_counter
import numpy as np
from ..contracts import RectPrimitive
from .static_qp import SPAN,NOMINAL_S
from .refined_oracle import RefinedOracle
from .lookahead_oracle import LookaheadOracle


class HystereticOracle(LookaheadOracle):
    def step(self,pos,yaw,ego,env_id=0,seed=None,reference=None):
        start=perf_counter();e=int(env_id);prev=self.prev.get(e);pos=np.asarray(pos)
        self.last_debug={'preview_used':False,'preview_infeasible':False}
        if prev is None:
            result=self.solver.solve(pos,yaw,seed=seed)
            if result.status=='feasible':self.prev[e]=result.s.copy();self.counter[e]=np.zeros(5,int)
            return result
        if self.solver.clearance(np.zeros(5),pos,yaw)<self.solver.margin-1e-8:
            result=self.solver.solve(pos,yaw);self.reset([e]);return result
        c,s=np.cos(yaw),np.sin(yaw)
        world=(np.asarray(ego[:2])*[1.,-1.])@np.array([[c,s],[-s,c]])
        omega=float(ego[2]);combined=list(self.rects)
        if np.linalg.norm(world)>1e-6 or abs(omega)>1e-6:
            for tau in [self.horizon/2,self.horizon]:
                predicted=pos+world*tau;angle=omega*tau
                if self.solver.clearance(np.zeros(5),predicted,yaw+angle)<self.solver.margin-1e-8:
                    self.last_debug['preview_infeasible']=True;break
                ca,sa=np.cos(angle),np.sin(angle);inverse=np.array([[ca,-sa],[sa,ca]])
                for index in self.solver._nearby(predicted):
                    rect=self.rects[index]
                    center=pos+(np.asarray(rect.center)-predicted)@inverse
                    combined.append(RectPrimitive(tuple(center),rect.size,rect.yaw-angle))
                self.last_debug['preview_used']=True
        region=RefinedOracle(combined,self.solver.margin)
        if region.clearance(prev,pos,yaw)>=region.margin-1e-8:
            # Safe mode remains unchanged during growth: no parameter is
            # shrunk merely to trade its room for another parameter.
            clear=prev<NOMINAL_S-.5*.5*self.dt/np.abs(SPAN)
            self.counter[e]=np.where(clear,self.counter[e]+1,0)
            allowed=clear&(self.counter[e]>=self.cooldown_calls)
            result=region.solve(pos,yaw,seed=prev,prev=prev,weight=self.weight,dt=self.dt,
                shrink_rate=0.,grow_allowed=allowed)
        else:
            # Nearest shrink-only common shape (no nominal reallocation).
            forecast=region.solve(pos,yaw,seed=prev,prev=prev,desired=prev,
                limit_shrink=False,grow_rate=0.,weight=0.)
            desired=forecast.s if forecast.status=='feasible' else np.zeros(5)
            result=self.solver.solve(pos,yaw,seed=prev,prev=prev,desired=desired,
                weight=self.weight,dt=self.dt,grow_allowed=np.zeros(5,bool))
            self.counter[e]=np.zeros(5,int)
        if result.status=='feasible':self.prev[e]=result.s.copy()
        else:self.reset([e])
        result.elapsed_s=perf_counter()-start
        return result
