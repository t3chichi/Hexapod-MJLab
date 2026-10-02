"""Analytic geometry, legacy ego direction and causal reset isolation checks."""
from pathlib import Path
import json
import numpy as np
from hexapod_mjlab.tasks.envelope_adaptive_2.experiments.refined_oracle import RefinedOracle
from hexapod_mjlab.tasks.envelope_adaptive_2.experiments.causal_oracle import CausalOracle
from hexapod_mjlab.tasks.envelope_adaptive_2.experiments.lookahead_oracle import LookaheadOracle
from hexapod_mjlab.tasks.envelope_adaptive_2.experiments.static_qp import MIN,SPAN
from hexapod_mjlab.tasks.envelope_adaptive_2.contracts import RectPrimitive
from static_oracle_gate import fixtures
from whole_polygon_reference import WholePolygonChecker


def main():
    records=[]
    for margin in [.1,.2]:
        for name,rects,yaw,expected in fixtures(margin):
            result=RefinedOracle(rects,margin).solve(np.zeros(2),yaw)
            if expected is None:assert result.status=='minimum_infeasible'
            else:
                assert result.status=='feasible'
                error=np.max(np.abs(MIN+SPAN*result.s-expected))
                assert error<1e-4,(name,margin,error)
                assert WholePolygonChecker(rects).clearance(MIN+SPAN*result.s,[0,0],yaw)>=margin-1e-6
            records.append({'case':name,'margin':margin,'status':result.status})
    # A very narrow corridor has an analytic near-floor rule solution.
    # Width cap .315m => fw/bw_s=.05, so extension_s<=.05+9.25*.05.
    narrow=[RectPrimitive((0,y),(8,.6)) for y in [-.715,.715]]
    capped=RefinedOracle(narrow,.1,headroom=.1)
    r=capped.solve([0,0],0)
    expected=np.array([.05,.0375,.05,.5125,.5125])
    assert r.status=='feasible' and np.max(np.abs(r.s-expected))<1e-4,(r,expected)
    assert capped.shape_valid(r.s) and capped.shape_valid(np.zeros(5)) and capped.shape_valid(np.full(5,.975))
    for scale in np.linspace(0,1,11):assert capped.shape_valid(scale*r.s)
    assert WholePolygonChecker(narrow).clearance(MIN+SPAN*r.s,[0,0],0)>=.1-1e-6
    # Legacy observable semantics: heading=pi/2,tangent=0 => ego=(0,+v)
    # while physical motion follows +world-x. Capture actual preview queries.
    oracle=CausalOracle([],horizon=.24)
    oracle.prev[0]=np.full(5,.4);oracle.counter[0]=np.zeros(5,int)
    queries=[];original=oracle.solver.solve
    def capture(pos,yaw,**kwargs):
        queries.append((np.asarray(pos).copy(),float(yaw)))
        return original(pos,yaw,**kwargs)
    oracle.solver.solve=capture
    oracle.step([2,3],np.pi/2,[0,.4,.2])
    assert any(np.allclose(p,[2.096,3],atol=1e-12) and abs(h-(np.pi/2+.048))<1e-12 for p,h in queries),queries
    # Partial reset must neither erase another env nor carry reset history into
    # a fresh episode. This corpus is separate from frozen data (done==0).
    for cls in [CausalOracle,LookaheadOracle]:
        walls=[RectPrimitive((1.22,0),(.6,4))]
        multi=cls(walls);fresh=cls(walls)
        for e in [0,1]:
            for t in range(12):multi.step([-.1,0],0,[.2,0,0],e)
        other=multi.prev[1].copy();multi.reset([0]);assert np.array_equal(multi.prev[1],other)
        a=multi.step([-.1,0],0,[0,0,0],0);b=fresh.step([-.1,0],0,[0,0,0],0)
        assert np.array_equal(a.s,b.s)
        # Pause -> motion -> pause, with current full-polygon checks each frame.
        checker=WholePolygonChecker(walls)
        for t in range(60):
            velocity=.2 if 20<=t<40 else 0.
            pos=[-.1+.004*min(max(t-20,0),20),0]
            r=multi.step(pos,0,[velocity,0,0],0)
            assert r.status=='feasible'
            assert checker.clearance(MIN+SPAN*r.s,pos,0)>=.1-1e-6
        blocked=[RectPrimitive((0,y),(8,.6)) for y in [-.65,.65]]
        invalid=cls(blocked);r=invalid.step([0,0],0,[0,0,0],7)
        assert r.status=='minimum_infeasible' and 7 not in invalid.prev
    # Shared future pose constraints should have the exact front-wall cutoff.
    predictive=LookaheadOracle([RectPrimitive((1.22,0),(.6,4))],horizon=.12,weight=4)
    for _ in range(120):r=predictive.step([0,0],0,[.3,0,0])
    expected_fwd=.92-.1-.3*.12
    assert abs((MIN+SPAN*r.s)[3]-expected_fwd)<1e-4,((MIN+SPAN*r.s)[3],expected_fwd)
    # Rigid transform identity, using independently constructed polygons.
    # Nonzero yaw rate and an oblique obstacle catch transpose/sign mistakes
    # that the axis-aligned front-wall fixture cannot catch.
    from whole_polygon_reference import polygon_distance
    center=np.array([1.7,-.3]);pos=np.array([.2,.1]);heading=.7
    original=RectPrimitive(tuple(center),(.4,1.1),.37)
    params=MIN+SPAN*np.array([.8,.3,.7,.6,.5])
    def outline(p,h):
        fw,mw,bw,f,b=params
        poly=np.array([[f,-fw],[0,-mw],[b,-bw],[b,bw],[0,mw],[f,fw]])
        co,si=np.cos(h),np.sin(h)
        return poly@np.array([[co,si],[-si,co]])+p
    for angle in [-.24,.12]:
        future_pos=pos+np.array([.07,-.03]);co,si=np.cos(angle),np.sin(angle)
        translated=pos+(center-future_pos)@np.array([[co,-si],[si,co]])
        changed=RectPrimitive(tuple(translated),original.size,original.yaw-angle)
        first=WholePolygonChecker([original]);second=WholePolygonChecker([changed])
        d1=polygon_distance(outline(future_pos,heading+angle),first.polygons[0])
        d2=polygon_distance(outline(pos,heading),second.polygons[0])
        assert abs(d1-d2)<1e-12,(d1,d2)
    result={'analytic_cases':records,'legacy_velocity_direction':'pass',
        'partial_reset_isolation':'pass','stop_start_stop_safety':'pass',
        'minimum_infeasible_resets_history':'pass','common_shape_front_wall':'pass','rotated_forecast_distance_identity':'pass','near_floor_rule_analytic_corridor':'pass'}
    out=Path('logs/ea2/oracle_refinement/20261003');out.mkdir(parents=True,exist_ok=True)
    (out/'analytic_gate_production_v1.json').write_text(json.dumps(result,indent=2)+'\n')
    print('REFINEMENT GATE PASS',json.dumps(result))


if __name__=='__main__':main()
