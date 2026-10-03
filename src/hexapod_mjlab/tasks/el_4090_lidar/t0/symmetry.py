"""Task-local symmetry of absolute targets under the fixed T1 default pose."""

import torch
from hexapod_mjlab.rl.symmetry_adapter import make_el_4090_symmetry_func
from .witness import JOINT_TARGETS


def offset_correction(device, dtype):
    d = torch.as_tensor(JOINT_TARGETS, device=device, dtype=dtype)
    return torch.cat([d[9:], d[:9]]) - d


def t0_symmetry(env=None, obs=None, actions=None):
    if env is None:
        raise ValueError("T0 mirror requires the actual env sensor mount")
    if hasattr(env, "_sensor_offset_quat"):
        quat, translation = env._sensor_offset_quat[:1], env._sensor_translation[:1]
    else:
        from ..mdp.lidar import get_lidar_pipeline

        pipeline = get_lidar_pipeline(env)
        quat, translation = pipeline._sensor_offset_quat[:1], pipeline._sensor_translation[:1]
    mirror = make_el_4090_symmetry_func(sensor_quat=quat, sensor_trans=translation)
    augmented, act = mirror(env=env, obs=obs, actions=actions)
    if augmented is not None:
        b = obs.batch_size[0]
        delta = offset_correction(augmented["policy"].device, augmented["policy"].dtype)
        # joint_pos_rel scale=1; previous actions scale=0.25 rad/action.
        augmented["policy"][b:, 12:30] += delta
        augmented["policy"][b:, 48:66] += delta / 0.25
        from .point_order import canonical_points

        for start, count in ((66, 256), (834, 640)):
            block = augmented["policy"][b:, start : start + count * 3].reshape(b, count, 3)
            augmented["policy"][b:, start : start + count * 3] = canonical_points(
                block, quat, translation
            ).reshape(b, -1)
    if act is not None:
        act[len(actions) :] += offset_correction(act.device, act.dtype) / 0.25
    return augmented, act
