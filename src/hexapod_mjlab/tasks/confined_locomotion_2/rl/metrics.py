"""Bounded, detached diagnostics for real confined_locomotion_2 transitions.

Call :meth:`begin_iteration`, then once per real control step call
:meth:`observe(transition, output, envelope_label)`, then :meth:`finish_iteration`.
Reported step averages are over env-steps; episode return/length averages are
over episodes completed in that iteration. Episode accumulators survive rollout
boundaries and are included in the versioned state dict. Contributions use the
actual reward manager weight and the task's 0.02 s control interval.

Velocity uses the sealed post-action body-frame root velocity and the command
that scored the transition. Foot diagnostics use recorded foot link origins
and support, matching the reward definitions; they do not infer sole or radius
positions. Raw foot reward terms have units slip m²/s², clearance m², and
landing-air event score seconds (not flight duration).
"""

from __future__ import annotations

from collections.abc import Mapping
import math

import torch

from hexapod_mjlab.common.mdp.rewards import CONTROL_DT_S, ROOT_HEIGHT_TARGET_M


class LearningMetrics:
    """Aggregate learning diagnostics without retaining autograd graphs."""

    SCHEMA = 1

    def __init__(self, num_envs: int, device, plane_world_z: float = 0.0, *, terrain_query=None):
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        if not torch.isfinite(torch.tensor(float(plane_world_z))):
            raise ValueError("plane_world_z must be finite")
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.plane_world_z = float(plane_world_z)
        self.terrain_query = terrain_query
        self.episode_return = torch.zeros(num_envs, device=self.device, dtype=torch.float64)
        self.episode_length = torch.zeros(num_envs, device=self.device, dtype=torch.float64)
        self.logger = None
        self._active = False
        self._iteration: dict[str, torch.Tensor] = {}
        self._steps_this_iteration = 0
        self._action_dim = 0

    def bind_logger(self, logger):
        self.logger = logger
        logger.bind_episode_tensors(self.episode_return, self.episode_length)

    def begin_iteration(self) -> None:
        self._iteration = {}
        self._steps_this_iteration = 0
        self._active = True

    def _add(self, name: str, value: torch.Tensor) -> None:
        value = value.detach().to(device=self.device, dtype=torch.float64).sum()
        old = self._iteration.get(name)
        self._iteration[name] = value if old is None else old + value

    def _batch(self, value, shape, name, *, dtype=torch.float32) -> torch.Tensor:
        result = torch.as_tensor(value, device=self.device, dtype=dtype).detach()
        if tuple(result.shape) != tuple(shape):
            raise ValueError(f"{name} shape {tuple(result.shape)} != {tuple(shape)}")
        return result

    @staticmethod
    def _envelope(value, device, name):
        if not isinstance(value, torch.Tensor):
            value = getattr(value, "values", value)
        out = torch.as_tensor(value, device=device, dtype=torch.float32).detach()
        if out.ndim != 2 or out.shape[1] != 6:
            raise ValueError(f"{name} must have shape (num_envs, 6), got {tuple(out.shape)}")
        return out

    def observe(self, transition, output, envelope_label) -> None:
        """Consume one actual 20 ms control transition and its policy output."""
        if not self._active:
            raise RuntimeError("begin_iteration() must be called before observe()")
        self._steps_this_iteration += 1
        rec = transition.context_record
        b = self.num_envs
        cmd = self._batch(transition.command, (b, 3), "command")
        vel = self._batch(rec["lin_vel_b"], (b, 3), "lin_vel_b")
        omega = self._batch(rec["ang_vel_b"], (b, 3), "ang_vel_b")
        target = torch.cat((cmd[:, :2], cmd[:, 2:3]), dim=1)
        actual = torch.stack((vel[:, 0], vel[:, 1], omega[:, 2]), dim=1)
        error = actual - target
        for i, axis in enumerate(("vx", "vy", "wz")):
            self._add(f"Proprio/{axis}_squared_error", error[:, i].square())
            self._add(f"Proprio/{axis}_absolute_error", error[:, i].abs())
            self._add(f"Proprio/{axis}_actual", actual[:, i])
            self._add(f"Proprio/{axis}_target", target[:, i])

        if "gravity_b" in rec:
            gravity = self._batch(rec["gravity_b"], (b, 3), "gravity_b")
            tilt_deg = torch.acos((-gravity[:, 2]).clamp(-1.0, 1.0)) * (180.0 / torch.pi)
            self._add("Proprio/tilt_deg", tilt_deg)
            self._add("Proprio/angular_velocity_xy_norm_radps", torch.linalg.vector_norm(omega[:, :2], dim=1))

        root_pos = self._batch(rec["root_pos_w"], (b, 3), "root_pos_w")
        if self.terrain_query is None:
            root_ground = self.plane_world_z
        else:
            from hexapod_mjlab.common.mdp.rewards import terrain_clearances
            terrain_height, terrain_valid = terrain_clearances(self.terrain_query.env, rec)
        if self.terrain_query is None:
            height = root_pos[:, 2] - root_ground
            self._add("Proprio/root_height_error_abs_m", (height - ROOT_HEIGHT_TARGET_M).abs())
            self._add("Proprio/root_height_m", height)
        else:
            height = terrain_height[:, 0][terrain_valid[:, 0]]
            self._add("Proprio/root_height_valid_count", terrain_valid[:, 0])
            self._add("Proprio/root_height_error_abs_m", (height - ROOT_HEIGHT_TARGET_M).abs())
            self._add("Proprio/root_height_m", height)

        raw_action = torch.as_tensor(rec["raw_action"], device=self.device, dtype=torch.float32).detach()
        clipped = torch.as_tensor(rec["clipped_action"], device=self.device, dtype=torch.float32).detach()
        if raw_action.ndim != 2 or raw_action.shape[0] != b or clipped.shape != raw_action.shape:
            raise ValueError("raw/clipped action shapes must be matching (num_envs, action_dim)")
        self._action_dim = int(raw_action.shape[1])
        self._add("Action/raw_abs", raw_action.abs())
        self._add("Action/clipped_abs", clipped.abs())
        self._add("Action/saturated_count", (raw_action != clipped).to(torch.float32))
        self._add("Action/element_count", torch.ones_like(raw_action))

        raws = transition.reward_raw_terms
        weights = transition.reward_weights
        if set(raws) != set(weights):
            raise ValueError("reward raw term and weight names differ")
        reward = self._batch(transition.reward, (b,), "reward")
        contribution_total = torch.zeros_like(reward)
        for term, raw in raws.items():
            raw = self._batch(raw, (b,), f"reward_raw/{term}")
            contribution = raw * float(weights[term]) * CONTROL_DT_S
            self._add(f"RewardRaw/{term}", raw)
            self._add(f"RewardContribution/{term}", contribution)
            raw_unit_key = {
                "pen_foot_slip": "RewardRaw/pen_foot_slip_m2_per_s2",
                "pen_foot_clearance": "RewardRaw/pen_foot_clearance_m2",
                "reward_foot_air": "RewardRaw/reward_foot_air_s",
            }.get(term)
            if raw_unit_key is not None:
                self._add(raw_unit_key, raw)
            contribution_total = contribution_total + contribution
        self._add("Reward/actual", reward)
        self._add("Reward/reconstructed", contribution_total)

        support = rec.get("support")
        if support is not None:
            support = self._batch(support, (b, 6), "support", dtype=torch.bool)
            self._add("Feet/support_count", support.to(torch.float32))
            foot_pos = self._batch(rec["foot_pos_w"], (b, 6, 3), "foot_pos_w")
            foot_vel = self._batch(rec["foot_vel_w"], (b, 6, 3), "foot_vel_w")
            swing = ~support
            foot_ground = self.plane_world_z
            if self.terrain_query is not None:
                foot_height = terrain_height[:, 1:]
                swing = swing & terrain_valid[:, 1:]
            else:
                foot_height = foot_pos[:, :, 2] - foot_ground
            tangential_speed = torch.linalg.vector_norm(foot_vel[:, :, :2], dim=2)
            self._add("Feet/swing_link_origin_clearance_m", foot_height * swing)
            self._add("Feet/swing_foot_count", swing.to(torch.float32))
            self._add("Feet/supported_link_origin_tangential_speed_mps", tangential_speed * support)
            self._add("Feet/supported_foot_count", support.to(torch.float32))

        pred = self._envelope(output.envelope, self.device, "predicted envelope")
        truth = self._envelope(envelope_label, self.device, "envelope label")
        if tuple(pred.shape) != (b, 6) or tuple(truth.shape) != (b, 6):
            raise ValueError("envelope batch size must match num_envs")
        label_valid = getattr(envelope_label, "valid", None)
        valid = (torch.ones(b, dtype=torch.bool, device=self.device) if label_valid is None
                 else self._batch(label_valid, (b,), "envelope_label.valid", dtype=torch.bool))
        valid_f = valid.to(torch.float32)
        error = (pred - truth).abs() * valid[:, None]
        self._add("Envelope/valid_count", valid_f)
        self._add("Envelope/absolute_error", error)
        for d in range(6):
            self._add(f"Envelope/absolute_error_{d}", error[:, d])
            self._add(f"Envelope/predicted_{d}", pred[:, d] * valid_f)
            self._add(f"Envelope/target_{d}", truth[:, d] * valid_f)

        for label, mask in (("near", transition.input.near_valid), ("far", transition.input.far_valid)):
            mask = torch.as_tensor(mask, device=self.device, dtype=torch.bool).detach()
            if mask.ndim != 2 or mask.shape[0] != b:
                raise ValueError(f"{label}_valid must have shape (num_envs, point_count)")
            self._add(f"Scan/{label}_valid_count", mask.to(torch.float32))

        terminated = self._batch(transition.terminated, (b,), "terminated", dtype=torch.bool)
        truncated = self._batch(transition.truncated, (b,), "truncated", dtype=torch.bool)
        done = terminated | truncated
        reward64 = reward.to(torch.float64)
        episode_return = self.episode_return + reward64
        episode_length = self.episode_length + 1.0
        self._add("Episode/terminated_count", terminated.to(torch.float32))
        self._add("Episode/truncated_count", truncated.to(torch.float32))
        done_f = done.to(torch.float64)
        self._add("Episode/completed_count", done_f)
        self._add("Episode/completed_return", episode_return * done_f)
        self._add("Episode/completed_length_steps", episode_length * done_f)
        self._add("Episode/completed_length_seconds", episode_length * (CONTROL_DT_S * done_f))
        if self.logger is not None:
            self.logger.process_env_step(reward64, done, {})
        else:
            # Standalone diagnostics retain the same public metrics contract.
            self.episode_return.copy_(episode_return * (~done))
            self.episode_length.copy_(episode_length * (~done))

        # Preserve actual reason flags when a runtime adds them; absent reasons
        # are omitted rather than fabricated from a generic done flag.
        termination_reasons = {"fall_low_height", "fall_tilt_sustained", "fall_flipped"}
        truncation_reasons = {"episode_timeout"}
        for key, value in rec.items():
            if key.startswith("termination_reason_") or key.startswith("truncation_reason_"):
                flag = self._batch(value, (b,), key, dtype=torch.bool)
                self._add(f"Episode/{key}_count", flag.to(torch.float32))
            elif key in termination_reasons | truncation_reasons:
                flag = self._batch(value, (b,), key, dtype=torch.bool)
                kind = "termination" if key in termination_reasons else "truncation"
                self._add(f"Episode/{kind}_reason_{key}_count", flag.to(torch.float32))

        diagnostics = getattr(transition, "diagnostics", {}) or {}
        for key, value in diagnostics.get("termination_reasons", {}).items():
            flag = self._batch(value, (b,), f"termination_reasons.{key}", dtype=torch.bool)
            kind = "truncation" if key in {"timeout", "terrain_boundary"} else "termination"
            self._add(f"Episode/{kind}_reason_{key}_count", flag.to(torch.float32))
        landing = diagnostics.get("foot_landing")
        landing_air = diagnostics.get("foot_landing_air_time_s")
        if landing is not None or landing_air is not None:
            if landing is None or landing_air is None:
                raise ValueError("landing diagnostics must include both event mask and consumed air time")
            landing = self._batch(landing, (b, 6), "foot_landing", dtype=torch.bool)
            landing_air = self._batch(landing_air, (b, 6), "foot_landing_air_time_s")
            self._add("Feet/landing_event_count", landing.to(torch.float32))
            self._add("Feet/landing_air_time_s", landing_air * landing)
            self._add("Feet/landing_air_time_sample_count", landing.to(torch.float32))

    def finish_iteration(self) -> dict[str, float]:
        if not self._active:
            raise RuntimeError("begin_iteration() must be called before finish_iteration()")
        self._active = False
        keys = list(self._iteration)
        if not keys:
            return {"Episode/completed_count": 0.0}
        values = torch.stack([self._iteration[k] for k in keys]).detach().cpu().tolist()
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError("learning metrics contain non-finite transition values")
        sums = dict(zip(keys, values))
        count = getattr(self, "_steps_this_iteration", 0)
        env_samples = count * self.num_envs
        action_dim = int(getattr(self, "_action_dim", 0))
        out: dict[str, float] = {}
        for key, val in sums.items():
            if key == "Proprio/root_height_valid_count":
                out[key] = float(val)
                continue
            if key.startswith("Episode/completed_"):
                continue
            if key == "Action/element_count":
                continue
            if key == "Feet/swing_foot_count" or key == "Feet/supported_foot_count" or key == "Feet/landing_air_time_sample_count":
                continue
            if key == "Envelope/valid_count":
                out[key] = float(val)
                continue
            if key.startswith("Envelope/") and sums.get("Envelope/valid_count", 0.0) == 0:
                continue
            denom = env_samples
            if key in {"Proprio/root_height_m", "Proprio/root_height_error_abs_m"}:
                denom = sums.get("Proprio/root_height_valid_count", env_samples)
                if not denom:
                    continue
            if key.startswith("Action/"):
                denom = denom * action_dim
            elif key == "Envelope/absolute_error":
                denom = sums.get("Envelope/valid_count", 0.0) * 6
            elif key == "Feet/support_count":
                denom = env_samples
            elif key == "Feet/swing_link_origin_clearance_m":
                denom = sums.get("Feet/swing_foot_count", 0.0)
            elif key == "Feet/supported_link_origin_tangential_speed_mps":
                denom = sums.get("Feet/supported_foot_count", 0.0)
            elif key == "Feet/landing_air_time_s":
                denom = sums.get("Feet/landing_air_time_sample_count", 0.0)
            if key.startswith("Proprio/") and ("squared_error" in key):
                axis = key.split("/")[-1].split("_")[0]
                unit = "mps" if axis in ("vx", "vy") else "radps"
                out[key.replace("squared_error", "rmse") + f"_{unit}"] = float(max(val, 0.0) ** 0.5 / max(env_samples, 1) ** 0.5)
            elif key.startswith("Proprio/") and ("absolute_error" in key):
                axis = key.split("/")[-1].split("_")[0]
                unit = "mps" if axis in ("vx", "vy") else "radps"
                out[key.replace("absolute_error", "mae") + f"_{unit}"] = float(val / max(env_samples, 1))
            elif key.startswith("Envelope/absolute_error_"):
                out[key.replace("absolute_error", "mae") + "_m"] = float(val / max(sums.get("Envelope/valid_count", 0.0), 1.0))
            elif key == "Envelope/absolute_error":
                out["Envelope/mae_m"] = float(val / max(sums.get("Envelope/valid_count", 0.0) * 6, 1.0))
            elif key.startswith("Episode/") and key.endswith("_count"):
                out[key] = float(val)
            elif key == "Feet/landing_event_count":
                out[key] = float(val)
            elif denom is not None:
                den = float(denom)
                if key == "Action/saturated_count":
                    out["Action/saturation_rate"] = float(val / den) if den else 0.0
                elif key == "Feet/support_count":
                    out["Feet/support_count_mean"] = float(val / den) if den else 0.0
                elif key == "Feet/swing_link_origin_clearance_m":
                    if den:
                        out["Feet/swing_link_origin_clearance_mean_m"] = float(val / den)
                elif key == "Feet/supported_link_origin_tangential_speed_mps":
                    if den:
                        out["Feet/supported_link_origin_tangential_speed_mean_mps"] = float(val / den)
                elif key == "Feet/landing_air_time_s":
                    if den:
                        out["Feet/landing_air_time_mean_s"] = float(val / den)
                elif key.startswith("Envelope/") and key != "Envelope/absolute_error":
                    suffix = "_m" if key.startswith(("Envelope/predicted_", "Envelope/target_")) else ""
                    out[key + suffix] = float(val / max(sums.get("Envelope/valid_count", 0.0), 1.0))
                elif key.startswith("Proprio/") and key.endswith("_actual"):
                    axis = key.split("/")[-1].split("_")[0]
                    unit = "mps" if axis in ("vx", "vy") else "radps"
                    out[key.replace("_actual", f"_actual_mean_{unit}")] = float(val / max(env_samples, 1))
                elif key.startswith("Proprio/") and key.endswith("_target"):
                    axis = key.split("/")[-1].split("_")[0]
                    unit = "mps" if axis in ("vx", "vy") else "radps"
                    out[key.replace("_target", f"_target_mean_{unit}")] = float(val / max(env_samples, 1))
                else:
                    out[key] = float(val / den) if den else 0.0
        n_done = sums.get("Episode/completed_count", 0.0)
        out["Episode/completed_count"] = float(n_done)
        if n_done:
            out["Episode/mean_return"] = sums["Episode/completed_return"] / n_done
            out["Episode/mean_length_steps"] = sums["Episode/completed_length_steps"] / n_done
            out["Episode/mean_length_seconds"] = sums["Episode/completed_length_seconds"] / n_done
        self._steps_this_iteration = 0
        return out

    def state_dict(self) -> dict:
        return {"schema": self.SCHEMA, "num_envs": self.num_envs,
                "plane_world_z": self.plane_world_z,
                "episode_return": self.episode_return.detach().cpu().clone(),
                "episode_length": self.episode_length.detach().cpu().clone(),
                **({"logger": self.logger.episode_state_dict()} if self.logger is not None else {})}

    def load_state_dict(self, state: Mapping) -> None:
        if not isinstance(state, Mapping) or state.get("schema") != self.SCHEMA or state.get("num_envs") != self.num_envs:
            raise ValueError("learning metrics checkpoint schema/num_envs mismatch")
        if float(state.get("plane_world_z", float("nan"))) != self.plane_world_z:
            raise ValueError("learning metrics checkpoint plane_world_z mismatch")
        loaded = []
        for name in ("episode_return", "episode_length"):
            value = torch.as_tensor(state[name], dtype=torch.float64)
            if tuple(value.shape) != (self.num_envs,) or not torch.isfinite(value).all():
                raise ValueError(f"invalid learning metrics {name} checkpoint")
            loaded.append(value.to(self.device))
        if (loaded[1] < 0).any():
            raise ValueError("episode lengths cannot be negative")
        if self.logger is not None:
            self.logger.load_episode_state_dict(state.get("logger"))
        self.episode_return.copy_(loaded[0])
        self.episode_length.copy_(loaded[1])
        self._iteration = {}
        self._active = False
