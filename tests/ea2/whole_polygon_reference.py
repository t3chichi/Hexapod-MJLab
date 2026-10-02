"""Independent simple-polygon clearance oracle; no task/solver imports.

Handles concave polygons, containment, crossing and collinear contact. Broad
phase uses AABB lower bounds only; each selected pair gets complete geometry.
"""
import numpy as np


def _cross(a, b):
    return a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]


def _point_segment(p, a, b):
    edge = b-a
    t = np.clip(np.dot(p-a, edge)/np.dot(edge, edge), 0, 1)
    return float(np.linalg.norm(p-(a+t*edge)))


def _contains(point, polygon):
    inside = False
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        if _point_segment(point, a, b) <= 1e-12:
            return True
        if (a[1]>point[1]) != (b[1]>point[1]):
            hit = a[0]+(point[1]-a[1])*(b[0]-a[0])/(b[1]-a[1])
            if hit > point[0]: inside = not inside
    return inside


def polygon_distance(a, b):
    """Euclidean distance of two complete filled simple polygons."""
    if _contains(a[0], b) or _contains(b[0], a):
        return 0.
    best = float('inf')
    for p, q in zip(a, np.roll(a, -1, axis=0)):
        for r, s in zip(b, np.roll(b, -1, axis=0)):
            u, v = q-p, s-r
            denom = _cross(u, v)
            if abs(denom) > 1e-14:
                t, w = _cross(r-p, v)/denom, _cross(r-p, u)/denom
                if -1e-12 <= t <= 1+1e-12 and -1e-12 <= w <= 1+1e-12:
                    return 0.
            distance = min(_point_segment(p,r,s), _point_segment(q,r,s),
                           _point_segment(r,p,q), _point_segment(s,p,q))
            best = min(best, distance)
    return best


class WholePolygonChecker:
    def __init__(self, rects):
        polygons = []
        for rect in rects:
            # Construct separately from the QP module's rectangle helper.
            corners = np.array([[-1,-1],[1,-1],[1,1],[-1,1]],dtype=float)*np.asarray(rect.size)/2
            rotation = np.array([[np.cos(rect.yaw),-np.sin(rect.yaw)],[np.sin(rect.yaw),np.cos(rect.yaw)]])
            polygons.append(corners@rotation.T+rect.center)
        self.polygons = np.array(polygons).reshape(-1,4,2)
        self.low = self.polygons.min(1); self.high = self.polygons.max(1)

    def clearance(self, params, pos, heading):
        fw,mw,bw,f,b = params
        outline = np.array([[f,-fw],[0,-mw],[b,-bw],[b,bw],[0,mw],[f,fw]])
        rotation = np.array([[np.cos(heading),-np.sin(heading)],[np.sin(heading),np.cos(heading)]])
        outline = outline@rotation.T+pos
        gaps = np.maximum(np.maximum(self.low-outline.max(0),outline.min(0)-self.high),0)
        lower = np.linalg.norm(gaps,axis=-1)
        best = float('inf')
        for i in np.argsort(lower):
            if lower[i] >= best: break
            best = min(best,polygon_distance(outline,self.polygons[i]))
            if best == 0.: break
        return best
