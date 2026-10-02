"""Analytic fixtures for the opt-in static oracle and independent geometry."""
from pathlib import Path
import json
import numpy as np
from hexapod_mjlab.tasks.envelope_adaptive_2.contracts import RectPrimitive
from hexapod_mjlab.tasks.envelope_adaptive_2.experiments.static_qp import StaticOracle, MIN, SPAN, NOMINAL_S, vertices, _convex_nearest_pair, rect_polygon
from whole_polygon_reference import WholePolygonChecker, polygon_distance


def fixtures(margin):
    nominal = MIN + SPAN*NOMINAL_S
    front = RectPrimitive((1.22,0),(.6,4))
    corridor = [RectPrimitive((0,y),(8,.6)) for y in [-.9,.9]]
    n = np.array([1.,1.])/np.sqrt(2)
    oblique = RectPrimitive(tuple(n*1.25),(.6,6),np.pi/4)
    turning = [RectPrimitive((0,y),(8,.6)) for y in [-1.2,1.2]]
    blocked = [RectPrimitive((0,y),(8,.6)) for y in [-.65,.65]]
    expected_front = nominal.copy(); expected_front[3] = .92-margin
    expected_corridor = nominal.copy(); expected_corridor[:3] = .6-margin
    expected_oblique = nominal.copy()
    delta = (nominal[0]+nominal[3]-np.sqrt(2)*(.95-margin))/2
    expected_oblique[[0,3]] -= delta
    expected_turn45 = nominal.copy()
    delta = (nominal[0]+nominal[3]-np.sqrt(2)*(.9-margin))/2
    expected_turn45[[0,2,3]] -= delta; expected_turn45[4] += delta
    expected_turn90 = nominal.copy(); expected_turn90[3] = .9-margin; expected_turn90[4] = -(.9-margin)
    return [
        ('open',[],0.,nominal), ('front_wall',[front],0.,expected_front),
        ('corridor',corridor,0.,expected_corridor), ('oblique',[oblique],0.,expected_oblique),
        ('turn_0',turning,0.,nominal), ('turn_45',turning,np.pi/4,expected_turn45),
        ('turn_90',turning,np.pi/2,expected_turn90), ('minimum_infeasible',blocked,0.,None),
    ]


def main():
    evidence=[]
    for margin in [.1,.2]:
        for name,rects,yaw,expected in fixtures(margin):
            result=StaticOracle(rects,margin).solve(np.zeros(2),yaw)
            record={'name':name,'margin':margin,'status':result.status}
            if expected is None:
                assert result.status=='minimum_infeasible',record
            else:
                assert result.status=='feasible',record
                error=float(np.max(np.abs(result.params-expected)))
                assert error<=1e-4,(record,error,result.params,expected)
                clearance=WholePolygonChecker(rects).clearance(result.params,np.zeros(2),yaw)
                assert clearance>=margin-1e-6,(record,clearance)
                assert result.constraint_residual_m<=1e-6
                record.update(error_m=error,clearance_m=clearance if np.isfinite(clearance) else None)
            evidence.append(record)
    # Independent validator must detect containment/crossings, while retaining
    # genuinely clear concave waists which a convex hull would falsely reject.
    square=np.array([[-1,-1],[1,-1],[1,1],[-1,1]],dtype=float)
    assert polygon_distance(square,square*.02)==0
    assert polygon_distance(square,square+np.array([1.5,0]))==0
    assert abs(polygon_distance(square,square+np.array([3,0]))-1)<1e-12
    concave=np.array([.6,.3,.6,.6,-.6])
    checker=WholePolygonChecker([RectPrimitive((0,.5),(.04,.04))])
    assert checker.clearance(concave,np.zeros(2),0)>0
    # Corner-to-corner distance requires a diagonal normal, not edge normals.
    rect=RectPrimitive((.88,.58),(.4,.4))
    d,pa,pb=_convex_nearest_pair(vertices(MIN)[[0,2,3,5]],rect_polygon(rect))
    assert abs(d-np.sqrt(2)*.08)<1e-12 and np.allclose(pb-pa,[.08,.08])
    for margin in [.1,.2]:
        result=StaticOracle([rect],margin).solve(np.zeros(2),0)
        assert result.status==('feasible' if margin==.1 else 'minimum_infeasible')
        if result.params is not None:
            assert WholePolygonChecker([rect]).clearance(result.params,np.zeros(2),0)>=margin-1e-6
    out=Path('logs/ea2/oracle_static/20261003');out.mkdir(parents=True,exist_ok=True)
    (out/'synthetic.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print('STATIC GATE PASS: six scenario classes, 16 analytic solutions/status checks; independent concavity/containment/crossing/corner distance probes')


if __name__=='__main__':main()
