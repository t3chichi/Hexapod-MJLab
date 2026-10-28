"""RSL-RL logging with project metric names and resumable episode windows.

The upstream Logger owns the writer, episode accumulation, rolling means,
iteration output and ETA. Task metrics share its episode tensors and keep
their existing per-iteration diagnostics. No optimizer or simulator lives here.
"""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import sys

from rsl_rl.utils.logger import Logger


class TrainingLogger(Logger):
    """Small adapter for local custom runners; use upstream formatting intact."""

    # Keep the terminal summary bounded; the writer still receives all metrics.
    CONSOLE_METRICS = (
        "Progress/completed_iterations", "Reward/actual", "Episode/completed_count",
        "Episode/mean_return", "Episode/mean_length_steps", "Optimizer/kl",
        "Memory/torch_peak_allocated_MiB", "Height/mse_m2", "Height/weighted_loss",
        "Curriculum/reward_stage", "Curriculum/max_command_x", "Performance/checkpoint_s",
    )
    IDENTITY = {
        "implementation": "rsl_rl.utils.logger.Logger",
        "tensorboard_step": "zero_based_iteration_index",
        "episode_means": "last_100_completed_episodes",
        "Episode/mean_return": "episodes_completed_in_this_iteration",
        "elapsed_and_eta": "current_learn_call_collect_and_update_only",
    }

    def __init__(self, *, log_dir, cfg, env_cfg, num_envs, device, source_path):
        logging_cfg = deepcopy(cfg)
        # Logger only needs the RND switch, not a replacement PPO configuration.
        logging_cfg["algorithm"] = {"rnd_cfg": None}
        logging_cfg["run_name"] = " / ".join(
            str(cfg[name]) for name in ("experiment_name", "run_name") if cfg.get(name))
        super().__init__(
            log_dir=str(Path(log_dir) / "tensorboard"), cfg=logging_cfg,
            env_cfg=env_cfg, num_envs=num_envs, device=str(device),
            is_distributed=False, gpu_world_size=1, gpu_global_rank=0,
        )
        self.git_status_repos.append(str(source_path))
        self.init_logging_writer()

    def bind_episode_tensors(self, episode_return, episode_length):
        """Reuse existing checkpointed storage, including each task's dtype."""
        self.cur_reward_sum = episode_return
        self.cur_episode_length = episode_length

    def episode_state_dict(self):
        # Partial episodes are already stored by the task metrics. Times are
        # session diagnostics, never causal checkpoint state.
        return {"rewards": list(self.rewbuffer), "lengths": list(self.lenbuffer)}

    def load_episode_state_dict(self, state):
        # Older complete checkpoints retain their partial episodes but have no
        # historical rolling window. Start that window empty, without guessing.
        state = state if state is not None else {"rewards": [], "lengths": []}
        if not isinstance(state, dict) or set(state) != {"rewards", "lengths"}:
            raise ValueError("invalid logger episode window")
        rewards, lengths = state["rewards"], state["lengths"]
        if (not isinstance(rewards, (list, tuple)) or not isinstance(lengths, (list, tuple))
                or len(rewards) != len(lengths) or len(rewards) > self.rewbuffer.maxlen
                or any(not math.isfinite(float(v)) for v in (*rewards, *lengths))
                or any(float(v) < 1 or float(v) != int(v) for v in lengths)):
            raise ValueError("invalid logger episode window values")
        self.rewbuffer.clear()
        self.lenbuffer.clear()
        self.rewbuffer.extend(rewards)
        self.lenbuffer.extend(lengths)
        self.ep_extras.clear()

    def begin_learn(self, *, completed_iterations, environment_transitions):
        self.start_iteration = completed_iterations
        self.tot_timesteps = environment_transitions
        self.tot_time = 0.0

    def log_iteration(self, *, completed_iteration, target_iteration, scalars,
                      learning_rate, action_std):
        # Official log() consumes a zero-based iteration index. Project
        # checkpoint and JSONL counters remain completed, one-based counts.
        index = completed_iteration - 1
        losses = {name.removeprefix("Loss/"): value for name, value in scalars.items()
                  if name.startswith("Loss/")}
        extras = {name: scalars[name] for name in self.CONSOLE_METRICS if name in scalars}
        extras["Progress/completed_iterations"] = completed_iteration
        extras["Progress/num_envs"] = self.num_envs
        self.ep_extras.append(extras)
        # Upstream writes losses and console extras itself; do not write them
        # a second time through the same SummaryWriter.
        for name, value in scalars.items():
            if not name.startswith("Loss/") and name not in extras:
                self.writer.add_scalar(name, value, index)
        self.log(
            it=index, start_it=self.start_iteration, total_it=target_iteration,
            collect_time=scalars["Performance/collect_s"],
            learn_time=scalars["Performance/update_s"], loss_dict=losses,
            learning_rate=learning_rate, action_std=action_std.detach(), rnd_weight=None,
        )
        sys.stdout.flush()
        self.writer.flush()

    def close(self):
        self.stop_logging_writer()
        if self.writer is not None:
            self.writer.close()
