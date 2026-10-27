"""Deterministic T1-witness reset events + action-buffer clearing.

Reset state = the COMPLETE settled T1 witness: root pose from the lifted
settled qpos (lowest actual foot collision point 2 mm above the plane),
settled joint angles, zero root/joint velocities. The PD reference q_ref
(T1 JOINT_TARGETS) is deliberately NOT written here — default q_ref and
reset state stay distinct.
"""

from __future__ import annotations

import torch


def reset_root_witness(
  env,
  env_ids: torch.Tensor | None,
  asset_cfg_name: str = "robot",
) -> None:
  """Write the lifted settled-witness root pose with zero velocity."""
  from mjlab.envs.mdp.events import resolve_env_ids

  from hexapod_mjlab.common.reset import lifted_witness_state

  env_ids = resolve_env_ids(env, env_ids)
  asset = env.scene[asset_cfg_name]
  pos, quat, _z_before, _z_after = lifted_witness_state()
  n = len(env_ids)
  positions = torch.tensor(pos, dtype=torch.float32, device=env.device)
  positions = positions.view(1, 3).expand(n, 3) + env.scene.env_origins[env_ids]
  orientations = (
    torch.tensor(quat, dtype=torch.float32, device=env.device).view(1, 4).expand(n, 4)
  )
  asset.write_root_link_pose_to_sim(
    torch.cat([positions, orientations], dim=-1), env_ids=env_ids
  )
  asset.write_root_link_velocity_to_sim(
    torch.zeros(n, 6, dtype=torch.float32, device=env.device), env_ids=env_ids
  )


def reset_joints_witness(
  env,
  env_ids: torch.Tensor | None,
  joint_pos: tuple | None = None,
  asset_cfg_name: str = "robot",
) -> None:
  """Write the settled witness joint angles with zero velocity."""
  from mjlab.envs.mdp.events import resolve_env_ids

  from hexapod_mjlab.common.reset import settled_reset_joint_pos

  env_ids = resolve_env_ids(env, env_ids)
  asset = env.scene[asset_cfg_name]
  if joint_pos is None:
    joint_pos = settled_reset_joint_pos()
  n = len(env_ids)
  pos = torch.tensor(joint_pos, dtype=torch.float32, device=env.device)
  pos = pos.view(1, len(joint_pos)).expand(n, len(joint_pos))
  vel = torch.zeros(n, len(joint_pos), dtype=torch.float32, device=env.device)
  asset.write_joint_state_to_sim(pos, vel, env_ids=env_ids)


def clear_action_runtime(
  env,
  env_ids: torch.Tensor | None,
  action_name: str = "residual_pd",
) -> None:
  """Partial reset: clear ONLY the target envs' previous-action buffers."""
  from mjlab.envs.mdp.events import resolve_env_ids

  env_ids = resolve_env_ids(env, env_ids)
  term = env.action_manager.get_term(action_name)
  term.reset_buffers(env_ids)
