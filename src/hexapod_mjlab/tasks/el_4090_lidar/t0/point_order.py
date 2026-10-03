"""Canonical angular ordering with deterministic ties for the T0 variant."""

import torch, math
from hexapod_mjlab.utils.pointcloud_geometry import to_sensor_frame, cartesian_to_spherical


def canonical_points(points, quat, translation):
    sensor = to_sensor_frame(points, quat, translation)
    _, azimuth, phi = cartesian_to_spherical(sensor)
    angular = phi * (2 * math.pi) + azimuth
    # Lexicographic exact coordinates break equal-angle/ray ties. Stable
    # sorting prevents a second mirror from shuffling different history points.
    order = torch.arange(points.shape[1], device=points.device).expand(points.shape[:2])
    for key in (points[:, :, 2], points[:, :, 1], points[:, :, 0], angular):
        rank = torch.argsort(torch.gather(key, 1, order), dim=1, stable=True)
        order = torch.gather(order, 1, rank)
    return torch.gather(points, 1, order.unsqueeze(-1).expand_as(points))


def _ordered(env, cfg, which, count):
    from ..mdp.lidar import get_lidar_pipeline

    pipe = get_lidar_pipeline(env, cfg)
    pipe.ensure_current()
    block = getattr(pipe, which).reshape(env.num_envs, count, 3)
    return canonical_points(
        block, pipe._sensor_offset_quat[:1], pipe._sensor_translation[:1]
    ).reshape(env.num_envs, -1)


def t0_prox_obs(env, lidar_cfg):
    return _ordered(env, lidar_cfg, "_prox_block", 256)


def t0_distal_obs(env, lidar_cfg):
    return _ordered(env, lidar_cfg, "_distal_block", 640)
