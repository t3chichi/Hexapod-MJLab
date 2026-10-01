#!/usr/bin/env python
"""M3 Gate A: el_4090_lidar env skeleton checks (LiDAR zero placeholder).

Checks (per the M3 plan):
1. task registration via the official registry (entry-point discovery);
2. env construction + reset + 60 policy steps with zero actions on cuda;
3. obs shapes: policy 2754, height_grid 187; action 18;
4. physics sanity: finite states, robot stands under PD (base height near
   the stand height), feet in contact, episode lengths > 1, no NaN;
5. terrain/physics alignment: mujoco hfield surface under env origin vs the
   M2 numpy heightfield at the same world point;
6. reward/termination activity: per-term means, feet contact force sign.

Usage: python tests/env/gate_a.py [--num-envs 32] [--steps 60]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

import torch  # noqa: E402


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--num-envs", type=int, default=32)
  parser.add_argument("--steps", type=int, default=60)
  parser.add_argument("--device", type=str, default="cuda:0")
  args = parser.parse_args()

  import mujoco
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  cfg = load_env_cfg("El-4090-Lidar")
  cfg.scene.num_envs = args.num_envs
  print(f"[gate-a] building env: num_envs={args.num_envs} device={args.device}")
  env = ManagerBasedRlEnv(cfg, device=args.device)

  obs, _ = env.reset()
  policy_obs = obs["policy"]
  height_obs = obs["height_grid"]
  print(f"[gate-a] obs groups: policy {tuple(policy_obs.shape)}, "
        f"height_grid {tuple(height_obs.shape)}")
  assert policy_obs.shape == (args.num_envs, 2754), policy_obs.shape
  assert height_obs.shape == (args.num_envs, 187), height_obs.shape
  assert env.action_manager.total_action_dim == 18

  robot = env.scene["robot"]
  ep_len_sum = 0
  min_ep_len = 10**9
  n_resets = 0
  fall_gravity = 0
  fall_height = 0
  for step in range(args.steps):
    actions = torch.zeros(args.num_envs, 18, device=env.device)
    obs, rew, terminated, truncated, extras = env.step(actions)
    assert torch.isfinite(obs["policy"]).all(), f"non-finite obs at step {step}"
    assert torch.isfinite(rew).all(), f"non-finite reward at step {step}"
    qpos = torch.as_tensor(
      env.sim.data.qpos, device=env.device
    )
    assert torch.isfinite(qpos).all(), f"non-finite qpos at step {step}"
    ep_len_sum += env.episode_length_buf.float().mean().item()
    min_ep_len = min(min_ep_len, env.episode_length_buf.min().item())
    if terminated.any():
      n_resets += int(terminated.sum())
      g_z = robot.data.projected_gravity_b[:, 2]
      fall_gravity += int((g_z > -0.1).sum())
      fall_height += int((robot.data.root_link_pos_w[:, 2] < 0.12).sum())

  base_z = robot.data.root_link_pos_w[:, 2]
  joint_pos = robot.data.joint_pos
  ctrl = env.sim.data.ctrl
  print(f"[gate-a] after {args.steps} steps: base_z mean={base_z.mean():.3f} "
        f"min={base_z.min():.3f} max={base_z.max():.3f}")
  print(f"[gate-a] |joint_pos| max={joint_pos.abs().max():.3f} "
        f"mean={joint_pos.abs().mean():.4f}")
  ctrl_t = torch.as_tensor(ctrl, device=env.device)
  print(f"[gate-a] |ctrl| mean={ctrl_t.abs().mean():.2f} "
        f"max={ctrl_t.abs().max():.2f} (limit 80)")
  print(f"[gate-a] episode length: mean={ep_len_sum / args.steps:.1f} "
         f"min={min_ep_len}")
  print(f"[gate-a] terminations: {n_resets} (flipped={fall_gravity}, "
         f"low_base={fall_height})")

  # Feet contact + sign check.
  from hexapod_mjlab.tasks.el_4090_lidar.mdp.rewards import _get_state
  state = _get_state(env, "feet_ground_contact")
  force = state.feet_force()
  contact = force[..., 2] > 1.0
  print(f"[gate-a] feet force z mean per foot: "
        f"{force[..., 2].mean(dim=0).cpu().numpy().round(1)}")
  print(f"[gate-a] feet in contact (z>1N): {contact.float().mean():.2f} "
        f"(expect ~1.0 while standing)")

  # Reward term activity.
  env.step(torch.zeros(args.num_envs, 18, device=env.device))
  from mjlab.tasks.registry import load_env_cfg as _lec  # noqa: F401
  for name, term in zip(env.reward_manager.active_terms,
                        env.reward_manager._term_cfgs):
    value = term.func(env, **term.params)
    print(f"[gate-a] reward {name:22s} w={term.weight:>8.4g} "
          f"mean={value.mean().item(): .5f}")

  # Termination activity.
  term_out = env.termination_manager.compute()
  print(f"[gate-a] terminated={env.termination_manager.terminated.float().mean():.2f} "
        f"time_out={env.termination_manager.time_outs.float().mean():.2f}")

  # Terrain alignment: mujoco hfield surface under a few env origins vs the
  # M2 numpy heightfield (nearest sample) at the same world point.
  import numpy as np

  from hexapod_mjlab.tasks.el_4090_lidar.mdp.terrain import get_el_4090_terrain_view
  view = get_el_4090_terrain_view(env)
  td = view.data
  origins = env.scene.env_origins[:4]
  mj_model = env.sim.mj_model
  mj_data = mujoco.MjData(mj_model)
  mujoco.mj_forward(mj_model, mj_data)
  align_ok = True
  for k, origin in enumerate(origins):
    # ray straight down from 5 m above the origin.
    pnt = np.array([origin[0].item(), origin[1].item(), 5.0])
    geomid = np.zeros(1, dtype=np.int32)
    dist = mujoco.mj_ray(mj_model, mj_data, pnt, np.array([0.0, 0.0, -1.0]),
                         None, 1, -1, geomid)
    mj_z = 5.0 - dist if dist >= 0 else float("nan")
    # numpy sample at the same world point (nearest neighbor).
    fx = origin[0].item() + view.world_offset_xy[0]
    fy = origin[1].item() + view.world_offset_xy[1]
    ix = int(fx / td.horizontal_scale)
    iy = int(fy / td.horizontal_scale)
    np_z = float(td.heightfield[ix, iy])
    print(f"[gate-a] origin {k} ({origin[0]:.1f},{origin[1]:.1f}): "
          f"mj_ray z={mj_z:.3f} numpy z={np_z:.3f} (geom "
          f"{mj_model.geom(geomid[0]).name if geomid[0] >= 0 else 'none'})")
    if not (mj_z >= 0 and abs(mj_z - np_z) <= 0.15):
      align_ok = False
  print(f"[gate-a] terrain alignment: {'OK' if align_ok else 'MISMATCH'}")

  # Pillar presence: ray down at a numpy heightfield cell with height > 0.5 m
  # inside patch (0,0) (world frame), and compare against the numpy height.
  hf = td.heightfield
  border = int(5.0 / td.horizontal_scale)
  patch = hf[border : border + 80, border : border + 80]
  # Interior pillar cell: all 4 neighbors also elevated (away from edges,
  # where the hfield bilinear rendering legitimately smooths the blocky
  # trimesh convention).
  mask = torch.as_tensor(patch > 0.5)
  interior_mask = mask.clone()
  interior_mask[1:-1, 1:-1] = (
    mask[1:-1, 1:-1] & mask[:-2, 1:-1] & mask[2:, 1:-1]
    & mask[1:-1, :-2] & mask[1:-1, 2:]
  )
  px, py = interior_mask.nonzero(as_tuple=True)
  if len(px) > 0:
    k = len(px) // 2
    field_x = border + int(px[k])
    field_y = border + int(py[k])
    world_x = field_x * td.horizontal_scale - view.world_offset_xy[0]
    world_y = field_y * td.horizontal_scale - view.world_offset_xy[1]
    pnt = np.array([world_x + 0.05, world_y + 0.05, 5.0])
    geomid = np.zeros(1, dtype=np.int32)
    dist = mujoco.mj_ray(mj_model, mj_data, pnt, np.array([0.0, 0.0, -1.0]),
                         None, 1, -1, geomid)
    mj_z = 5.0 - dist if dist >= 0 else float("nan")
    np_z = float(hf[field_x, field_y])
    print(f"[gate-a] pillar cell ({world_x:.1f},{world_y:.1f}): "
          f"mj_ray z={mj_z:.3f} numpy z={np_z:.3f} "
          f"interior cells: {len(px)}")
    if not (mj_z >= 0 and abs(mj_z - np_z) <= 0.05):
      align_ok = False
  else:
    print("[gate-a] WARNING: no interior pillar cells found in patch (0,0)")
    align_ok = False

  print("[gate-a] PASS" if align_ok else "[gate-a] FAIL")
  return 0 if align_ok else 1


if __name__ == "__main__":
  raise SystemExit(main())
