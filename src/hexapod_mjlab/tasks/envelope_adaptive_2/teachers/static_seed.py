"""Conservative feasible seed for the shape-specific production teacher."""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from scipy.optimize import minimize
from ..config.condition import ENVELOPE_SPEC
from ..config.parameters import EnvelopeCfg

MIN = np.array([*ENVELOPE_SPEC.low[:4], ENVELOPE_SPEC.high[4]], dtype=float)
MAX = np.array([*ENVELOPE_SPEC.high[:4], ENVELOPE_SPEC.low[4]], dtype=float)
SPAN = MAX - MIN
NOMINAL_S = (1 + EnvelopeCfg().soft_dof_pos_limit) / 2


def vertices(params, pos=(0., 0.), heading=0.):
    fw, mw, bw, f, b = params
    v = np.array([[f, fw], [0, mw], [b, bw], [b, -bw], [0, -mw], [f, -fw]])
    c, s = np.cos(heading), np.sin(heading)
    return v @ np.array([[c, s], [-s, c]]) + np.asarray(pos)


def rect_polygon(rect):
    hx, hy = np.asarray(rect.size) / 2
    c, s = np.cos(rect.yaw), np.sin(rect.yaw)
    return np.array([[hx, hy], [-hx, hy], [-hx, -hy], [hx, -hy]]) @ np.array([[c, s], [-s, c]]) + np.asarray(rect.center)


def _nearest_vertices_to_edges(a, b):
    starts = b[None, :, :]
    edges = (np.roll(b, -1, axis=0) - b)[None, :, :]
    delta = a[:, None, :] - starts
    t = np.clip(np.sum(delta * edges, axis=-1) / np.sum(edges * edges, axis=-1), 0, 1)
    projected = starts + t[..., None] * edges
    sq = np.sum((a[:, None, :] - projected) ** 2, axis=-1)
    i, j = np.unravel_index(np.argmin(sq), sq.shape)
    return float(sq[i, j]), a[i], projected[i, j]


def _convex_nearest_pair(a, b):
    """True nearest pair for two convex polygons, with a separating-axis overlap test."""
    edges = np.concatenate([np.roll(a, -1, axis=0) - a, np.roll(b, -1, axis=0) - b])
    normals = np.stack([-edges[:, 1], edges[:, 0]], axis=-1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    ap, bp = a @ normals.T, b @ normals.T
    gap = np.maximum(ap.min(0) - bp.max(0), bp.min(0) - ap.max(0))
    if gap.max() <= 1e-12:
        return 0., None, None
    dab, pa, pb = _nearest_vertices_to_edges(a, b)
    dba, qb, qa = _nearest_vertices_to_edges(b, a)
    if dba < dab:
        dab, pa, pb = dba, qa, qb
    return np.sqrt(dab), pa, pb


@dataclass
class Result:
    status: str
    s: np.ndarray | None
    params: np.ndarray | None
    construction_s: float
    solve_s: float
    plane_count: int
    constraint_residual_m: float
    iterations: int
    normals: np.ndarray
    offsets: np.ndarray


class StaticOracle:
    def __init__(self, rects, margin):
        self.margin = float(margin)
        self.polygons = np.array([rect_polygon(r) for r in rects]).reshape(-1, 4, 2)
        self.centers = self.polygons.mean(1)
        self.radii = np.linalg.norm(self.polygons - self.centers[:, None], axis=-1).max(1) if len(rects) else np.empty(0)
        self.reach = np.linalg.norm(vertices(MIN + SPAN * NOMINAL_S), axis=-1).max()

    def solve(self, pos, heading):
        start = perf_counter()
        v0 = vertices(MIN, pos, heading)
        # Current lower bounds form a rectangle containing the body origin.
        seed = v0[[0, 2, 3, 5]]
        nearby = np.flatnonzero(np.linalg.norm(self.centers - pos, axis=-1) <= self.reach + self.radii + self.margin)
        ns, bs = [], []
        for index in nearby:
            obstacle = self.polygons[index]
            distance, pa, pb = _convex_nearest_pair(seed, obstacle)
            if distance < self.margin - 1e-9:
                return Result('minimum_infeasible', None, None, perf_counter()-start, 0., len(ns), 0., 0, np.empty((0, 2)), np.empty(0))
            normal = (pb-pa) / distance
            offset = float((obstacle @ normal).min()) - self.margin
            ns.append(normal); bs.append(offset)
        normals = np.asarray(ns).reshape(-1, 2); offsets = np.asarray(bs)
        basis = np.stack([vertices(MIN + SPAN * np.eye(5)[j], pos, heading)-v0 for j in range(5)], axis=-1)
        a = np.einsum('nd,vdk->nvk', normals, basis).reshape(-1, 5)
        b = (offsets[:, None] - normals @ v0.T).reshape(-1)
        keep = np.linalg.norm(a, axis=-1) > 1e-12
        if np.any(b[~keep] < -1e-8):
            return Result('region_failure', None, None, perf_counter()-start, 0., len(ns), float(-b.min()), 0, normals, offsets)
        a, b = a[keep], b[keep]
        construction = perf_counter()-start
        if not len(a) or np.all(a @ np.full(5, NOMINAL_S) <= b):
            s = np.full(5, NOMINAL_S); nit = 0; success = True
        else:
            result = minimize(lambda x: .5*np.sum((x-NOMINAL_S)**2), np.zeros(5),
                              jac=lambda x: x-NOMINAL_S, method='SLSQP',
                              bounds=[(0., NOMINAL_S)]*5,
                              constraints=[{'type':'ineq', 'fun':lambda x: b-a@x, 'jac':lambda x: -a}],
                              options={'ftol':1e-12, 'maxiter':100})
            s = np.clip(result.x, 0., NOMINAL_S); nit = result.nit; success = result.success
        residual = max(0., float((a@s-b).max())) if len(a) else 0.
        status = 'feasible' if success and residual <= 1e-7 else 'solver_failure'
        return Result(status, s, MIN+SPAN*s, construction, perf_counter()-start-construction, len(ns), residual, nit, normals, offsets)
