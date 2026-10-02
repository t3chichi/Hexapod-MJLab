"""Whole-polygon shape-specific safe-region refinement and causal regularisation.

The polygon is the exact union of two convex trapezoids. Each gets its own
separating plane from each nearby rectangle; an accepted feasible iterate
seeds the next local convex problem. This is a local method, not a global
nonconvex shape optimum. The independent test checker is never imported here.
"""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from scipy.optimize import minimize, linprog
from .static_seed import MIN, SPAN, NOMINAL_S, vertices, rect_polygon, _convex_nearest_pair, StaticOracle

PIECES=(np.array([0,1,4,5]),np.array([1,2,3,4]))


@dataclass
class RefinedResult:
    status: str
    s: np.ndarray | None
    clearance: float
    iterations: int
    elapsed_s: float
    emergency: bool = False
    reason: str = ''


class RefinedOracle:
    def __init__(self,rects,margin=.1,max_iters=8,headroom=0.,floor_cap=.05):
        self.margin=float(margin);self.max_iters=int(max_iters)
        self.headroom=float(headroom);self.floor_cap=float(floor_cap)
        self.extra_a=np.empty((0,5));self.extra_b=np.empty(0)
        if self.headroom>0:
            slope=(NOMINAL_S-self.floor_cap)/self.headroom
            rows=[]
            for i,j in [(3,0),(4,2)]:
                row=np.zeros(5);row[i]=1.;row[j]=-slope;rows.append(row)
            self.extra_a=np.array(rows);self.extra_b=np.full(len(rows),self.floor_cap)
        self.polygons=np.array([rect_polygon(r) for r in rects]).reshape(-1,4,2)
        self.centers=self.polygons.mean(1)
        self.radii=np.linalg.norm(self.polygons-self.centers[:,None],axis=-1).max(1) if len(rects) else np.empty(0)
        self.reach=np.linalg.norm(vertices(MIN+SPAN*NOMINAL_S),axis=-1).max()
        self.fallback=StaticOracle(rects,margin)

    def shape_valid(self,s):
        return bool(np.all(self.extra_a@np.asarray(s)<=self.extra_b+1e-8))

    def _nearby(self,pos):
        return np.flatnonzero(np.linalg.norm(self.centers-pos,axis=-1)<=self.reach+self.radii+self.margin)

    def _gap(self,s,pos,yaw,nearby):
        shape=vertices(MIN+SPAN*s,pos,yaw)
        best=float('inf')
        for index in nearby:
            for ids in PIECES:
                distance,_,_=_convex_nearest_pair(shape[ids],self.polygons[index])
                best=min(best,distance)
                if best==0:return 0.
        return best

    def clearance(self,s,pos,yaw):
        """Exact within the reach filter; far obstacles are safely beyond margin.

        Values beyond the required margin are only local clearance diagnostics,
        not claimed as the true global nearest obstacle distance.
        """
        return self._gap(s,np.asarray(pos),yaw,self._nearby(np.asarray(pos)))

    def _planes(self,seed,pos,yaw,nearby,require_seed_margin=True):
        v0=vertices(MIN,pos,yaw)
        basis=np.stack([vertices(MIN+SPAN*np.eye(5)[j],pos,yaw)-v0 for j in range(5)],axis=-1)
        shape=vertices(MIN+SPAN*seed,pos,yaw)
        rows=[];rhs=[]
        for index in nearby:
            obstacle=self.polygons[index]
            for ids in PIECES:
                distance,pa,pb=_convex_nearest_pair(shape[ids],obstacle)
                if distance<=1e-10 or (require_seed_margin and distance<self.margin-1e-7):return None,None
                normal=(pb-pa)/distance
                offset=(obstacle@normal).min()-self.margin
                rows.append(np.einsum('d,vdk->vk',normal,basis[ids]))
                rhs.append(offset-v0[ids]@normal)
        if not rows:return self.extra_a.copy(),self.extra_b.copy()
        a=np.concatenate(rows);b=np.concatenate(rhs)
        keep=np.linalg.norm(a,axis=-1)>1e-12
        if np.any(b[~keep]<-1e-7):return None,None
        return np.concatenate([a[keep],self.extra_a]),np.concatenate([b[keep],self.extra_b])

    @staticmethod
    def _project(target,seed,a,b,lo,hi):
        bounds=list(zip(lo,hi))
        if np.all(target>=lo) and np.all(target<=hi) and np.all(a@target<=b):return target.copy(),'ok'
        result=minimize(lambda x:.5*np.sum((x-target)**2),np.clip(seed,lo,hi),
            jac=lambda x:x-target,method='SLSQP',bounds=bounds,
            constraints=[{'type':'ineq','fun':lambda x:b-a@x,'jac':lambda x:-a}],
            options={'ftol':1e-11,'maxiter':100})
        s=np.clip(result.x,lo,hi)
        residual=max(0.,float((a@s-b).max())) if len(a) else 0.
        if result.success and residual<=1e-7:return s,'ok'
        feasibility=linprog(np.zeros(5),A_ub=a if len(a) else None,b_ub=b if len(a) else None,bounds=bounds,method='highs')
        return None,'rate_region_infeasible' if feasibility.status==2 else 'solver_failure'

    def solve(self,pos,yaw,seed=None,prev=None,weight=0.,dt=.02,shrink_rate=2.,grow_rate=.5,limit_shrink=True,desired=None,grow_allowed=None):
        start=perf_counter();pos=np.asarray(pos,dtype=float);nearby=self._nearby(pos)
        nominal=np.full(5,NOMINAL_S);zero=np.zeros(5)
        floor_gap=self._gap(zero,pos,yaw,nearby)
        if floor_gap<self.margin-1e-8:
            return RefinedResult('minimum_infeasible',None,floor_gap,0,perf_counter()-start)
        goal=nominal if desired is None else np.asarray(desired,dtype=float)
        target=goal if prev is None else (goal+float(weight)*np.asarray(prev))/(1+float(weight))
        lo=zero.copy();hi=nominal.copy()
        if prev is not None:
            if limit_shrink:lo=np.maximum(lo,np.asarray(prev)-shrink_rate*dt/np.abs(SPAN))
            hi=np.minimum(hi,np.asarray(prev)+grow_rate*dt/np.abs(SPAN))
            if grow_allowed is not None:hi=np.where(grow_allowed,hi,np.minimum(nominal,np.asarray(prev)))
        bounded=np.clip(target,lo,hi)
        gap=self._gap(bounded,pos,yaw,nearby)
        if gap>=self.margin-1e-8 and self.shape_valid(bounded):
            return RefinedResult('feasible',bounded,gap,0,perf_counter()-start,reason='safe_target')
        candidates=[zero]
        for point in [seed,prev]:
            if point is not None and self.shape_valid(point) and self._gap(point,pos,yaw,nearby)>=self.margin-1e-8:candidates.append(np.asarray(point).copy())
        if prev is not None:
            # A disjoint previous footprint can define obstacle support planes
            # even when it now lacks the requested margin. Shifting those
            # planes by margin certifies the NEW solution, not the old seed.
            # This preserves its geometric mode instead of forcing a cold
            # minimum/nominal seed to choose another width-vs-length branch.
            a,b=self._planes(np.asarray(prev),pos,yaw,nearby,require_seed_margin=False)
            if a is not None:
                projected,_=self._project(target,np.asarray(prev),a,b,lo,hi)
                if projected is not None and self._gap(projected,pos,yaw,nearby)>=self.margin-1e-7:
                    candidates.append(projected)
            if not any(np.all(x>=lo-1e-8) and np.all(x<=hi+1e-8) for x in candidates):
                # Shrinking all extents uniformly toward the constant-width
                # minimum rectangle is nested. Use a verified safe iterate;
                # this backup is not a certificate of global rate feasibility.
                left,right=0.,1.
                for _ in range(20):
                    alpha=(left+right)/2
                    if self._gap(np.asarray(prev)*alpha,pos,yaw,nearby)>=self.margin:left=alpha
                    else:right=alpha
                candidates.append(np.asarray(prev)*left)
        # Include first-round solution when no supplied static seed exists.
        if seed is None:
            raw=self.fallback.solve(pos,yaw)
            if raw.status=='feasible' and self.shape_valid(raw.s):candidates.append(raw.s)
        candidates=[x for x in candidates if self.shape_valid(x)]
        in_bounds=[x for x in candidates if np.all(x>=lo-1e-8) and np.all(x<=hi+1e-8)]
        best=min(in_bounds or candidates,key=lambda x:float(np.sum((x-target)**2)))
        best_value=float(np.sum((best-target)**2));accepted=0;reason='stationary'
        bounded_valid=bool(np.all(best>=lo-1e-8) and np.all(best<=hi+1e-8))
        for iteration in range(self.max_iters):
            a,b=self._planes(best,pos,yaw,nearby)
            if a is None:reason='plane_failure';break
            point,reason=self._project(target,best,a,b,lo,hi)
            if point is None:break
            point_gap=self._gap(point,pos,yaw,nearby)
            if point_gap<self.margin-1e-7:reason='geometry_rejected';break
            if not self.shape_valid(point):reason='headroom_rejected';break
            value=float(np.sum((point-target)**2))
            # First bounded solution may be worse than a seed outside rate box.
            if bounded_valid and value>best_value+1e-9:reason='objective_rejected';break
            delta=np.max(np.abs(point-best));improvement=best_value-value
            best=point;best_value=value;accepted+=1;bounded_valid=True
            if delta<1e-6 or (accepted>1 and improvement<1e-10):break
        emergency=prev is not None and not bounded_valid
        if emergency:
            # Safety dominates rate tracking. This is explicitly counted, not
            # presented as a certificate of global rate-box infeasibility.
            relaxed=self.solve(pos,yaw,seed=seed,prev=prev,weight=max(49.,weight),dt=dt,
                shrink_rate=shrink_rate,grow_rate=grow_rate,limit_shrink=False,desired=goal,grow_allowed=grow_allowed)
            return RefinedResult(relaxed.status,relaxed.s,relaxed.clearance,relaxed.iterations,
                perf_counter()-start,True,reason)
        gap=self._gap(best,pos,yaw,nearby)
        if not self.shape_valid(best):
            return RefinedResult('solver_failure',None,gap,accepted,perf_counter()-start,False,'headroom_rejected')
        return RefinedResult('feasible',best,gap,accepted,perf_counter()-start,False,reason)
