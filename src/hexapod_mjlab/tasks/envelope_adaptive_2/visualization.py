"""EA2-specific world geometry and coloring on top of public drawing helpers."""
import math
import mujoco
import numpy as np
import torch
from hexapod_mjlab.utils.visualization import draw_points,draw_polyline
from hexapod_mjlab.utils.visualization.primitives import DrawingBudget
from .geometry import compute_hex_vertices
from .math_utils import selected_channel_mask, yaw_quat_from_heading
from .quaternion import quat_apply,quat_mul,quat_from_euler_xyz


def map_spec(map_data):
    """Visual boxes from exactly the same map primitives as the raycast mesh."""
    spec=mujoco.MjSpec();body=spec.worldbody.add_body(name="map")
    body.add_geom(name="ground",type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=[39.,39.,.025],pos=[0.,0.,-.025],rgba=[.35,.38,.4,1.],contype=0,conaffinity=0)
    for i,rect in enumerate(map_data.rects):
        body.add_geom(name=f"obstacle{i}",type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[rect.size[0]/2,rect.size[1]/2,rect.height/2],
                      pos=[*rect.center,rect.height/2],
                      quat=[math.cos(rect.yaw/2),0,0,math.sin(rect.yaw/2)],
                      rgba=[.55,.57,.6,1.],contype=0,conaffinity=0)
    for i,pillar in enumerate(map_data.pillars):
        body.add_geom(name=f"pillar{i}",type=mujoco.mjtGeom.mjGEOM_BOX if pillar.square else mujoco.mjtGeom.mjGEOM_CYLINDER,
                      size=[pillar.radius,pillar.radius,pillar.height/2] if pillar.square else [pillar.radius,pillar.height/2,0],
                      pos=[*pillar.center,pillar.height/2],rgba=[.55,.57,.6,1.],contype=0,conaffinity=0)
    return spec


def hex_world(params,state,index):
    p=params[index]
    body=compute_hex_vertices(*p.unbind(-1))
    h=state.heading[index];c,s=torch.cos(h),torch.sin(h)
    rotation=torch.stack((torch.stack((c,-s)),torch.stack((s,c))))
    xy=body@rotation.T+state.base_pos[index,:2]
    z=torch.full((6,1),state.cfg.envelope.debug_ground_z_offset,device=xy.device)
    return torch.cat((xy,z),dim=-1).cpu().numpy()


def draw_state(state,visualizer):
    sensor=state.sensor
    indices=list(visualizer.get_env_indices(state.num_envs))
    budget=state.cfg.lidar.debug_geom_budget
    visualizer=DrawingBudget(visualizer,budget)
    per_env_points=max(0,(budget-13*len(indices))//max(1,len(indices)))
    for index in indices:
        draw_polyline(visualizer,hex_world(state.actions_mapped,state,index),closed=True,
                      color=(*state.cfg.envelope.debug_color,1.),radius=state.cfg.envelope.debug_line_radius)
        draw_polyline(visualizer,hex_world(state.oracle_params,state,index),closed=True,
                      color=(1.,.75,0.,1.),radius=.008)
        start=state.base_pos[index].cpu().numpy()
        end=start+np.array([math.cos(float(state.heading[index])),math.sin(float(state.heading[index])),0.])
        visualizer.add_arrow(start,end,color=(.2,.5,1.,1.),width=.02)
        if sensor._debug_points is None or index not in sensor._debug_env_ids:
            continue
        row=sensor._debug_env_ids.index(index)
        body=sensor._debug_points[row];q=yaw_quat_from_heading(state.heading[index:index+1])
        if state._attitude is not None:
            q=quat_mul(q,quat_from_euler_xyz(state.attitude_roll[index:index+1],state.attitude_pitch[index:index+1],torch.zeros_like(state.heading[index:index+1])))
        # Preserve the source's body-attached display of the cached range frame.
        points=quat_apply(q.expand(len(body),4),body)+state.base_pos[index]
        red,green=selected_channel_mask(sensor._debug_dists[row],sensor.selected_ray_indices,
                                         sensor._debug_is_reduced,state.cfg.lidar.far_plane)
        stride=state.cfg.lidar.debug_point_stride
        red_count=min(per_env_points,len(points[red][::stride]))
        draw_points(visualizer,points[red].cpu().numpy(),color=(1.,0.,0.,1.),stride=stride,max_points=red_count)
        draw_points(visualizer,points[green].cpu().numpy(),color=(0.,.8,.1,1.),stride=stride,
                    max_points=per_env_points-red_count)
