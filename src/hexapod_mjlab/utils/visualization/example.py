"""Independent example data: a square outline and three colored world points."""
import numpy as np
from . import draw_points,draw_polyline


def draw_example(visualizer):
    draw_polyline(visualizer,np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]]),closed=True)
    draw_points(visualizer,np.array([[.2,.2,.1],[.5,.5,.2],[.8,.8,.3]]),color=(1.,.6,0.,1.))
