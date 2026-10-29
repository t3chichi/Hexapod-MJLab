"""Legacy MDP lifecycle on the unchanged scan, input and recurrent contracts."""
from dataclasses import fields, is_dataclass
from pathlib import Path
import copy
import json

import torch

from hexapod_mjlab.common.reset import reset_env_preserving_unselected_physics
from hexapod_mjlab.common.transition import TransitionContext, require_finite
from hexapod_mjlab.common.robot import FOOT_BODIES
from hexapod_mjlab.rl.confined_locomotion_2.history import ScanWindow
from .runtime import TrainingRuntime, TrainingTransition
from .observation import ControlInputCache, clone_prepared
from .components.lidar.scanner import ConfinedLidarCfg, ConfinedScanPipeline
from .tripod_env_cfg import TASK_ID, owners
from .components.tripod_mdp import DT, FOOT_ORDER, REWARD_WEIGHTS, compute_legacy_rewards


def _move(value, device):
    if torch.is_tensor(value):
        return value.to(device).clone()
    if is_dataclass(value):
        return type(value)(**{f.name: _move(getattr(value, f.name), device) for f in fields(value)})
    if isinstance(value, dict):
        return {k: _move(v, device) for k, v in value.items()}
    return copy.deepcopy(value)


class CommandView:
    def __init__(self, commands):
        self.commands = commands

    def current_command(self):
        return self.commands.commands.clone()

    def state_dict(self):
        return self.commands.state_dict()


class LegacyContext(TransitionContext):
    def ensure_ready(self):
        if self.phase == "ready":
            return super().ensure_ready()
        record = super().ensure_ready()
        env = self.env
        commands = env._tripod_commands
        manual_command = commands.commands.clone() if self.runtime.mode == "play" else None
        push_due = commands.tick()
        if manual_command is not None:
            commands.commands.copy_(manual_command)
        if not torch.equal(commands.episode_length, env.episode_length_buf):
            raise RuntimeError("legacy episode clocks differ")
        if commands.common_step_counter != env.common_step_counter:
            raise RuntimeError("legacy global clocks differ")
        # Reference resamples commands AFTER integration and BEFORE scoring.
        record["command"] = commands.commands.clone()
        order = [FOOT_BODIES.index(name) for name in FOOT_ORDER]
        for name in ("foot_force_w", "foot_pos_w", "foot_vel_w", "support"):
            record[name] = record[name][:, order]
        sensor = env.scene["legacy_penalized_contact"]
        sensor._invalidate_cache()
        record["penalized_contact_force_w"] = sensor.data.force.clone()
        record["previous_qdot"] = env._tripod_previous_qdot.clone()
        record["previous_action"] = env._tripod_previous_action.clone()
        record["dof_pos_limits"] = torch.tensor([[-2.7, 2.7]], device=env.device).expand(18, 2)
        self.raw_rewards = compute_legacy_rewards(
            record, env._tripod_reward_state, commands.commands, env._tripod_stage.stage)
        total = torch.zeros(env.num_envs, device=env.device)
        for name, raw in self.raw_rewards.items():
            require_finite("legacy_reward/" + name, raw)
            total += raw * REWARD_WEIGHTS[name][env._tripod_stage.stage] * DT
        env._tripod_episode_return += total
        env._tripod_tracking_sum += self.raw_rewards["tracking_lin_vel"] * DT
        # Legacy computes body velocities before this velocity overwrite.
        # Reward therefore uses the captured pre-push body-frame velocity.
        if push_due and self.runtime.mode != "play":
            velocity = self.robot.data.root_link_vel_w.clone()
            velocity[:, :2] = env._tripod_dr.push_velocities()
            self.robot.write_root_link_velocity_to_sim(velocity)
        return record


class LegacyRuntime(TrainingRuntime):
    """Reuse perception methods; keep legacy commands/rewards/reset independent.

    The base constructor/step are deliberately not used: their sealed reward
    dictionary, T1 reset and fall gates describe a different experiment.
    """
    def __init__(self, wrapper, *, seed, output_dir, mode="training", lidar_cfg=None):
        from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
        if not isinstance(wrapper, RslRlVecEnvWrapper) or wrapper.clip_actions is not None:
            raise TypeError("legacy runtime requires the raw-action mjlab wrapper")
        self.wrapper, self.env = wrapper, wrapper.unwrapped
        env = self.env
        if env.cfg.auto_reset or env.cfg.decimation != 4 or abs(env.step_dt - DT) > 1e-12:
            raise ValueError("legacy runtime requires explicit reset at 50 Hz / four substeps")
        if mode not in ("training", "play") or env.common_step_counter != 0:
            raise ValueError("legacy runtime requires a fresh environment")
        self.device, self.num_envs, self.seed, self.mode = env.device, env.num_envs, seed, mode
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.lidar_cfg = lidar_cfg or ConfinedLidarCfg()
        self.perturbation_cfg = self._perturbations = None
        self.height_grid = self.terrain_query = self._height_label = None
        self._commands = CommandView(owners(env))
        self._control_steps = 0
        self.episode_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.step_in_episode = env._tripod_commands.episode_length
        if mode == "play":
            env._tripod_commands.commands.zero_()
        env.episode_length_buf.copy_(self.step_in_episode)
        self.scan_counters = torch.zeros_like(self.episode_ids)
        self.sensor_scan_counts = torch.zeros_like(self.episode_ids)
        self.next_due_scan_step = torch.full_like(self.episode_ids, 5)
        self.window = ScanWindow(self.num_envs, episode_ids=[0] * self.num_envs, device=self.device)
        self.pipeline = ConfinedScanPipeline(env, self.lidar_cfg)
        self.scan_identity = copy.deepcopy(self.pipeline.scan_identity)
        self.scan_audit_bound = 2 * self.num_envs
        self._input_cache = ControlInputCache()
        self._current_window_snapshot = self._play_source_scans = None
        self.transitions, self.scan_log = [], []
        self._closed = self._faulted = self._stopped = False
        self.last_fault = self.stop_record = None
        self._evidence_count = 0
        self._noise_keys = torch.full((self.num_envs, 2), -1, dtype=torch.long, device=self.device)
        self._noise = torch.zeros(self.num_envs, 63, device=self.device)
        self._last_noised = None
        self._proprio_cached = None
        self._context = LegacyContext(env)
        self._context.runtime = self
        try:
            # BaseTask.reset in the reference performs a zero-action step
            # before learn randomizes the initial episode counters. Retain
            # the physical warmup and its actuator/contact state explicitly.
            self._context.begin(env._tripod_commands.commands)
            wrapper.step(torch.zeros(self.num_envs, 18, device=self.device))
            record = self._context.finish()
            if bool((record["terminated"] | record["truncated"]).any()):
                raise RuntimeError("legacy initialization warmup terminated")
            self._control_steps = int(env.common_step_counter)
            env._tripod_episode_return.zero_()  # runner's return window starts after reset()
            env._tripod_previous_qdot.copy_(env.scene["robot"].data.joint_vel)
            self._proprio_cached = {name: record[name].clone() for name in ("lin_vel_b", "ang_vel_b", "gravity_b")}
            if mode == "training":
                env._tripod_commands.randomize_initial_episode_length()
                env.episode_length_buf.copy_(self.step_in_episode)
            self.next_due_scan_step.fill_(self._control_steps + 5)
            self._scan_rows_now(list(range(self.num_envs)))
        except BaseException:
            self.close()
            raise
        (self.output_dir / "runtime_recipe.json").write_text(json.dumps({
            "task": TASK_ID, "mode": mode, "seed": seed, "scan_identity": self.scan_identity,
            "reset_height_m": float(env.scene["robot"].data.default_root_state[0, 2]),
            "lidar_perturbations": None, "proprio_noise_level": 1.5,
            "running_normalization": False, "warmup_control_steps": 1}, indent=2))

    def prepare(self):
        prepared = super().prepare()
        if prepared is self._last_noised:
            return prepared
        obs = prepared.observation
        # Reference reset/push refreshes q/qdot but retains the body-frame
        # velocity/gravity computed before those writes until the next step.
        if self._proprio_cached is not None:
            for name, value in self._proprio_cached.items():
                getattr(obs, name).copy_(value)
        keys = torch.stack((obs.episode_ids, obs.step_ids), -1)
        changed = (keys != self._noise_keys).any(-1)
        # Unscaled physical half-amplitudes; the frozen network applies its
        # existing multipliers once. Commands/previous actions have no noise.
        amplitude = obs.lin_vel_b.new_tensor([.15] * 3 + [.3] * 3 + [.075] * 3
                                            + [.075] * 18 + [2.25] * 18 + [0.] * 18)
        self._noise[changed] = (2 * torch.rand(int(changed.sum()), 63, device=self.device) - 1) * amplitude
        self._noise_keys.copy_(keys)
        fields = (("lin_vel_b", 3, 2.), ("ang_vel_b", 3, .25), ("gravity_b", 3, 1.),
                  ("joint_pos_rel_ref", 18, 1.), ("joint_vel", 18, .05), ("prev_action", 18, 1.))
        offset = 0
        for name, size, scale in fields:
            value = getattr(obs, name)
            value.add_(self._noise[:, offset:offset + size])
            value.copy_((value * scale).clamp(-100., 100.) / scale)
            offset += size
        self._last_noised = prepared
        return prepared

    def _reset_selected(self, env_ids):
        reset_env_preserving_unselected_physics(self.env, env_ids)
        if self.mode == "play":
            self.env._tripod_commands.commands[env_ids] = 0.
        self.episode_ids[env_ids] += 1
        self.window.reset_envs(env_ids=env_ids, new_episode_ids=self.episode_ids[env_ids])
        self.scan_counters[env_ids] = 0
        writes = self._scan_rows_now(env_ids.cpu().tolist())
        self.next_due_scan_step[env_ids] = self._control_steps + 5
        self._input_cache.invalidate()
        return {"scan_writes": writes}

    def set_manual_command(self, env_ids, values):
        self._guard_usable("set_manual_command")
        if self.mode != "play":
            raise RuntimeError("manual commands are available only in play")
        rows = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        command = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        if (rows.ndim != 1 or rows.numel() == 0 or len(rows.unique()) != len(rows)
                or bool(((rows < 0) | (rows >= self.num_envs)).any())):
            raise ValueError("invalid manual command rows")
        if command.shape != (len(rows), 3) or not bool(torch.isfinite(command).all()):
            raise ValueError("command must be finite (rows,3)")
        if bool((command.abs() > command.new_tensor([2.5, 1., 1.2])).any()):
            raise ValueError("command outside legacy curriculum limits")
        self.env._tripod_commands.commands[rows] = command
        self._input_cache.invalidate()
        return {"command": self.env._tripod_commands.commands.cpu().tolist()}

    def step(self, raw_action):
        self._guard_usable("step")
        self._assert_counter()
        prepared = clone_prepared(self.prepare())
        episodes, steps = self.episode_ids.clone(), self.step_in_episode.clone()
        self._context.begin(prepared.observation.command)
        try:
            self.wrapper.step(raw_action)
            record = self._context.finish()
            self._control_steps += 1
            self._proprio_cached = {name: record[name].clone() for name in ("lin_vel_b", "ang_vel_b", "gravity_b")}
            due = self._due_scan_rows()
            writes = self._scan_rows_now(due)
            if due:
                self.next_due_scan_step[due] = self._control_steps + 5
            final_inputs, final_labels = clone_prepared(self.prepare()), self.labels()
            terminated, truncated = record["terminated"], record["truncated"]
            done = terminated | truncated
            ids = done.nonzero().flatten()
            weights = {name: values[self.env._tripod_stage.stage] for name, values in REWARD_WEIGHTS.items()}
            expected = sum(self._context.raw_rewards[name] * weight * DT for name, weight in weights.items())
            if not torch.allclose(record["reward"], expected, atol=1e-6, rtol=1e-5):
                raise RuntimeError("legacy reward manager reconstruction differs")
            next_inputs, next_labels = final_inputs, final_labels
            if len(ids) and self.mode == "training":
                self.env._tripod_stage.add_completed(self.env._tripod_episode_return[ids])
                self._reset_selected(ids)
            elif len(ids):
                self._stopped = True
            # Old post_physics_step assigns these AFTER reset_idx, including
            # the terminal action. Retain this otherwise easy-to-miss detail.
            action = self.env.action_manager.get_term("residual_pd")
            self.env._tripod_previous_action.copy_(record["clipped_action"])
            self.env._tripod_previous_qdot.copy_(self.env.scene["robot"].data.joint_vel)
            if len(ids) and self.mode == "training":
                action.prev_action[ids] = record["clipped_action"][ids]
                self._input_cache.invalidate()
                next_inputs, next_labels = clone_prepared(self.prepare()), self.labels()
            transition = TrainingTransition(
                control_id=record["control_id"], env_ids=prepared.observation.env_ids,
                episode_ids=episodes, step_ids=steps, time_s=self.control_time_s() - DT,
                input=prepared, command=record["command"], reward=record["reward"],
                reward_raw_terms={k: v.clone() for k, v in self._context.raw_rewards.items()},
                reward_weights=weights, reward_expected=expected, terminated=terminated,
                truncated=truncated, final_inputs=final_inputs, final_labels=final_labels,
                reset_env_ids=ids if len(ids) else None, next_inputs=next_inputs, next_labels=next_labels,
                context_record=record, mode=self.mode, engineering_stopped=self._stopped,
                scan_writes=writes, diagnostics={"termination_reasons": {
                    "flipped": terminated, "timeout": truncated}})
            self.transitions[:] = [transition]
            return transition
        except BaseException as error:
            self._save_fault(error, phase="legacy_step")
            raise

    def state_dict(self):
        self._guard_usable("state_dict")
        if self._context.phase != "idle":
            raise RuntimeError("runtime save requires idle boundary")
        env = self.env
        return {"schema": "tripod2_low.runtime/v1", "seed": self.seed, "mode": self.mode,
                "warmup_control_steps": 1, "proprio_cached": _move(self._proprio_cached, self.device),
                "scan_identity": copy.deepcopy(self.scan_identity), "control_steps": self._control_steps,
                "tensors": {name: getattr(self, name).clone() for name in (
                    "episode_ids", "scan_counters", "sensor_scan_counts", "next_due_scan_step", "_noise_keys", "_noise")},
                "window": self.window.state_dict(), "commands": env._tripod_commands.state_dict(),
                "reward_state": env._tripod_reward_state.state_dict(), "stage": env._tripod_stage.state_dict(),
                "dr": env._tripod_dr.state_dict(),
                "episode": {name: getattr(env, name).clone() for name in (
                    "_tripod_episode_return", "_tripod_tracking_sum", "_tripod_previous_qdot", "_tripod_previous_action")},
                "cache": copy.deepcopy(self._input_cache.__dict__),
                "window_snapshot": copy.deepcopy(self._current_window_snapshot)}

    def load_state_dict(self, state):
        live = self.state_dict()
        if set(state) != set(live) or any(state[k] != live[k] for k in ("schema", "seed", "mode", "scan_identity", "warmup_control_steps")):
            raise ValueError("legacy runtime state identity/fields differ")
        self._control_steps = int(state["control_steps"])
        for saved, targets in ((state["tensors"], self), (state["episode"], self.env)):
            expected = live["tensors"] if targets is self else live["episode"]
            if set(saved) != set(expected):
                raise ValueError("runtime tensor fields differ")
            for name, value in saved.items():
                target = getattr(targets, name)
                if value.shape != target.shape or value.dtype != target.dtype:
                    raise ValueError("runtime tensor layout differs: " + name)
                target.copy_(value.to(self.device))
        env = self.env
        for name, owner in (("commands", env._tripod_commands), ("reward_state", env._tripod_reward_state),
                            ("stage", env._tripod_stage), ("dr", env._tripod_dr), ("window", self.window)):
            owner.load_state_dict(state[name])
        self._input_cache.__dict__.update({name: _move(value, self.device)
                                          for name, value in state["cache"].items()})
        self._current_window_snapshot = _move(state["window_snapshot"], self.device)
        self._last_noised = self._input_cache.get()
        self._proprio_cached = _move(state["proprio_cached"], self.device)
        if not torch.equal(env.episode_length_buf, self.step_in_episode):
            raise ValueError("restored legacy episode clocks differ")
