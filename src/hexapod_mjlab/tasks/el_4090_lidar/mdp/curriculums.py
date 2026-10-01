"""Curriculum terms — port of the old multi-stage rewards + command curriculum.

Old semantics:

* ``multi_stage_rewards`` (runner-driven): when the mean episode reward
  exceeds ``reward_stage_threshold`` (2.0), the reward scales switch from
  stage 0 to stage 1 (per-term weights below); the stage only advances
  (never returns).
* ``commands.curriculum`` (env ``update_command_curriculum`` at reset): when
  the mean episodic ``tracking_lin_vel`` sum exceeds
  ``tracking_lin_vel_curriculum_threshold`` (0.8) of its (dt-scaled) weight,
  the lin_vel_x range expands by ``command_curriculum_step`` (0.5) on each
  side, clamped to ``max_curriculum`` (2.5).

mjlab curriculum terms run at reset (``CurriculumManager.compute(env_ids)``
in ``_reset_idx``, BEFORE the reward manager's episode sums are cleared), so
both checks read the completed episodes' sums there — a close equivalent of
the old per-iteration runner check (deviation: checked per reset batch
instead of once per training iteration; monotonic quantities are unaffected).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

#: Stage-dependent reward weights (old scales lists [stage0, stage1]).
STAGE_WEIGHTS: dict[str, tuple[float, float]] = {
  "base_height": (-2.0, -5.0),
  "feet_slip": (-0.0, -0.2),
  "feet_air_time": (0.5, 0.1),
  "feet_stumble": (-0.0, -0.2),
  "feet_contact_forces": (-0.05, -0.25),
  "gait_2_step": (-1.0, -0.2),
}

REWARD_STAGE_THRESHOLD = 2.0
REWARD_MAX_STAGE = 1

COMMAND_CURRICULUM_THRESHOLD = 0.8
COMMAND_CURRICULUM_STEP = 0.5
COMMAND_MAX_CURRICULUM = 2.5


def reward_stage(env: ManagerBasedRlEnv, env_ids) -> dict:
  """Advance the multi-stage reward weights when mean episode reward is high.

  Returns the curriculum state (``{"stage": int}``).
  """
  stage = env.__dict__.get("_el_4090_reward_stage", 0)
  if stage < REWARD_MAX_STAGE:
    sums = env.reward_manager._episode_sums
    episode_total = torch.zeros(env.num_envs, device=env.device)
    for value in sums.values():
      episode_total += value
    # mjlab _episode_sums are dt-scaled; the old runner compared the RAW
    # per-step-summed episode total (cur_reward_sum += rewards) against the
    # threshold, so undo the dt scaling here (unit parity, else stage flips
    # ~50x too early).
    done_totals = episode_total[env_ids] / env.step_dt
    if done_totals.numel() > 0 and done_totals.mean() > REWARD_STAGE_THRESHOLD:
      stage += 1
      env._el_4090_reward_stage = stage
      for name, weights in STAGE_WEIGHTS.items():
        env.reward_manager.get_term_cfg(name).weight = weights[stage]
      print(f"[el_4090] reward stage -> {stage}")
  return {"stage": stage}


def commands_vel_el_4090(
  env: ManagerBasedRlEnv,
  env_ids,
  command_name: str = "twist",
  reward_name: str = "tracking_lin_vel",
) -> dict:
  """Expand the lin_vel_x command range when tracking is good (old semantics).

  Old check: ``mean(episode_sums['tracking_lin_vel'][env_ids]) / max_ep_len
  > 0.8 * reward_scale`` with the dt-scaled weight (1.0 * dt).
  """
  command_term = env.command_manager.get_term(command_name)
  cfg = command_term.cfg
  lo, hi = cfg.ranges.lin_vel_x
  sums = env.reward_manager._episode_sums.get(reward_name)
  scale = env.reward_manager.get_term_cfg(reward_name).weight * env.step_dt
  if sums is not None and sums[env_ids].numel() > 0:
    if (sums[env_ids].mean() / env.max_episode_length) > (
      COMMAND_CURRICULUM_THRESHOLD * scale
    ):
      lo = float(max(lo - COMMAND_CURRICULUM_STEP, -COMMAND_MAX_CURRICULUM))
      hi = float(min(hi + COMMAND_CURRICULUM_STEP, COMMAND_MAX_CURRICULUM))
      cfg.ranges.lin_vel_x = (lo, hi)
  return {"lin_vel_x_min": cfg.ranges.lin_vel_x[0], "lin_vel_x_max": cfg.ranges.lin_vel_x[1]}
