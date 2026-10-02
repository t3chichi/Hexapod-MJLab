"""Explicit xyzw sensor rotations; mjlab entity poses use wxyz instead."""
import torch
from hexapod_mjlab.utils.pointcloud_geometry import quaternion_apply as quat_apply


def quat_from_euler_xyz(roll, pitch, yaw):
    cr, sr = torch.cos(roll*.5), torch.sin(roll*.5)
    cp, sp = torch.cos(pitch*.5), torch.sin(pitch*.5)
    cy, sy = torch.cos(yaw*.5), torch.sin(yaw*.5)
    return torch.stack((cy*sr*cp-sy*cr*sp, cy*cr*sp+sy*sr*cp,
                        sy*cr*cp-cy*sr*sp, cy*cr*cp+sy*sr*sp), dim=-1)


def quat_mul(a, b):
    av, bv = a[..., :3], b[..., :3]
    aw, bw = a[..., 3:], b[..., 3:]
    return torch.cat((aw*bv + bw*av + torch.cross(av,bv,dim=-1),
                      aw*bw - (av*bv).sum(-1,keepdim=True)),dim=-1)
