"""No environment state, simulation writes, tensor devices or task imports."""
import numpy as np


def _points(value):
    points=np.asarray(value,dtype=np.float64)
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():
        raise ValueError("World points must be finite (N,3)")
    return points


def draw_points(visualizer, points_world, color=(1.,0.,0.,1.), radius=.025, stride=1, max_points=None):
    if stride<1 or radius<=0:
        raise ValueError("stride and radius must be positive")
    points=_points(points_world)[::stride]
    if max_points is not None:
        if max_points<0:raise ValueError("max_points cannot be negative")
        if len(points)>max_points:
            points=points[np.linspace(0,len(points)-1,max_points,dtype=int)]
    for point in points:
        visualizer.add_sphere(point,radius,color)


class DrawingBudget:
    """Bound primitive calls through the public viewer API, without scene access."""
    def __init__(self,visualizer,max_geoms):
        self.visualizer=visualizer
        self.remaining=int(max_geoms)

    def _draw(self,name,*args,**kwargs):
        if self.remaining>0:
            getattr(self.visualizer,name)(*args,**kwargs)
            self.remaining-=1

    def add_sphere(self,*args,**kwargs):self._draw('add_sphere',*args,**kwargs)
    def add_cylinder(self,*args,**kwargs):self._draw('add_cylinder',*args,**kwargs)
    def add_arrow(self,*args,**kwargs):self._draw('add_arrow',*args,**kwargs)


def draw_polyline(visualizer, points_world, color=(0.,.85,1.,1.), radius=.012, closed=False):
    points=_points(points_world)
    if radius<=0:raise ValueError("radius must be positive")
    for start,end in zip(points[:-1],points[1:]):
        if np.linalg.norm(end-start)>1e-9:
            visualizer.add_cylinder(start,end,radius,color)
    if closed and len(points)>2:
        visualizer.add_cylinder(points[-1],points[0],radius,color)
