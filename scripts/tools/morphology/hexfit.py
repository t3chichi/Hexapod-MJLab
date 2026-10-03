"""Enclose complete convex collision projections in a possibly nonconvex EA2 hexagon."""
from __future__ import annotations
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, QhullError

PARAM_KEYS = ('front_width','middle_width','back_width','forward_limit','backward_limit')
CONTAIN_TOL = 1e-7


def convex_projection(points):
    p=np.unique(np.asarray(points,dtype=float)[:,:2],axis=0)
    if not len(p) or not np.isfinite(p).all():
        raise ValueError('finite nonempty collision projection required')
    if len(p)<3:return p
    try:return p[ConvexHull(p).vertices]
    except QhullError:return p[[np.argmin(p[:,0]),np.argmax(p[:,0]),np.argmin(p[:,1]),np.argmax(p[:,1])]]


def split_vertices(polygons):
    """Every polygon vertex and intersection of each convex projection with x=0.

    On each half h(x) is affine, so these vertices certify the *filled*
    intersection, rather than just the input mesh vertices. Disconnected
    bodies must remain separate; a whole-robot hull adds fictitious volume.
    """
    points=[]
    for poly in polygons:
        p=convex_projection(poly);points.extend(p)
        for a,b in zip(p,np.roll(p,-1,axis=0)):
            if a[0]*b[0]<0:
                points.append(np.array([0.,a[1]-a[0]*(b[1]-a[1])/(b[0]-a[0])]))
    if not points:raise ValueError('at least one collision body required')
    return np.unique(np.array(points),axis=0)


def _segment_weights(x,fl,bl):
    x=np.atleast_1d(x);w=np.zeros((len(x),3));pos=x>=0
    if fl>0:w[pos,0]=x[pos]/fl;w[pos,1]=1-w[pos,0]
    else:w[pos,1]=1
    if bl<0:w[~pos,2]=x[~pos]/bl;w[~pos,1]=1-w[~pos,2]
    else:w[~pos,1]=1
    return w


def hexagon_half_width(x,params):
    a=np.asarray(x,dtype=float);v=_segment_weights(a,params['forward_limit'],params['backward_limit'])@np.array([params[k] for k in PARAM_KEYS[:3]])
    return float(v[0]) if a.ndim==0 else v


def containment_violation(polygons,params):
    """Independent certificate: each clipped convex piece must be inside its trapezoid.

    Enumerates original edges and their waist cuts without LP constraint
    reduction. Width and longitudinal errors are in metres.
    """
    worst=-np.inf
    for poly in polygons:
        p=convex_projection(poly)
        probes=list(p)
        for a,b in zip(p,np.roll(p,-1,axis=0)):
            if a[0]*b[0]<0:probes.append(np.array([0.,a[1]-a[0]*(b[1]-a[1])/(b[0]-a[0])]))
        for x,y in probes:
            if x>=0:
                t=x/params['forward_limit'] if params['forward_limit']>0 else 0.
                hw=(1-t)*params['middle_width']+t*params['front_width']
            else:
                t=x/params['backward_limit'] if params['backward_limit']<0 else 0.
                hw=(1-t)*params['middle_width']+t*params['back_width']
            worst=max(worst,abs(y)-hw,x-params['forward_limit'],params['backward_limit']-x)
    return max(0.,float(worst))


def fit_collision_hexagon(polygons):
    polygons=[convex_projection(p) for p in polygons]
    p=split_vertices(polygons);fl=max(0.,float(p[:,0].max()));bl=min(0.,float(p[:,0].min()))
    if fl-bl<1e-12:raise ValueError('degenerate longitudinal extent')
    w=_segment_weights(p[:,0],fl,bl)
    res=linprog(np.ones(3),A_ub=-w,b_ub=-np.abs(p[:,1]),bounds=[(0,None)]*3,method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    if not res.success:raise RuntimeError(res.message)
    params=dict(zip(PARAM_KEYS,[*map(float,res.x),fl,bl]))
    violation=containment_violation(polygons,params)
    if violation>CONTAIN_TOL:raise RuntimeError(f'complete collision projection not contained: {violation}')
    return params


def fit_ea2_hexagon(points_xy):
    """Compatibility: a plain cloud denotes ONE filled convex collision body.

    Use fit_collision_hexagon for disconnected mesh groups. Treating a cloud
    as isolated points is deliberately no longer a safety contract.
    """
    return fit_collision_hexagon([points_xy])


def normalize_params(params,minimum,maximum):
    """Minimum/maximum describe expansion, not sorted signed coordinate bounds."""
    lo=np.array([minimum[k] for k in PARAM_KEYS]);hi=np.array([maximum[k] for k in PARAM_KEYS]);p=np.array([params[k] for k in PARAM_KEYS])
    if np.any(np.abs(hi-lo)<1e-12):raise ValueError('degenerate normalization range')
    return (p-lo)/(hi-lo)


def denormalize_params(s,minimum,maximum):
    lo=np.array([minimum[k] for k in PARAM_KEYS]);hi=np.array([maximum[k] for k in PARAM_KEYS])
    return dict(zip(PARAM_KEYS,map(float,lo+np.asarray(s)*(hi-lo))))
