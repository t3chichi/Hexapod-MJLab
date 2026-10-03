"""Strict, simulator-independent episode verdicts, replayed from raw arrays."""

import numpy as np

SETTLE_STEPS = 100
TOTAL_STEPS = 1000
T0B_THRESHOLDS = {
    "tilt_deg_max": 15.0,
    "height_err_m_max": 0.05,
    "forward_disp_along_heading_m_min": 3.0,
    "forward_vx_mae_max": 0.10,
    "forward_lateral_drift_m_max": 0.30,
    "stand_measure_disp_m_max": 0.05,
    "applied_torque_Nm_max": 80.0,
}
SAFETY_KEYS = (
    "nonfoot_events",
    "self_events",
    "nonfinite_events",
    "overflow_events",
    "overflow_bits",
    "angle_domain_events",
    "preclip_peak",
    "clip_count",
    "tau_samples",
    "applied_peak",
    "substep_samples",
)


def validate_episode(ep, thresholds=T0B_THRESHOLDS):
    checks = {
        k: False
        for k in (
            "data_valid",
            "complete",
            "no_violation",
            "tilt_ok",
            "height_ok",
            "command_ok",
            "torque_ok",
        )
    }
    metrics = {}
    errors = []
    try:
        n = ep["n_steps"]
        if not isinstance(n, int) or n < 0 or n > TOTAL_STEPS:
            raise ValueError("invalid step count")
        arrays = {
            k: np.asarray(ep[k])
            for k in (
                "root_pos",
                "root_quat",
                "lin_vel_b",
                "ang_vel_b",
                "joint_pos",
                "joint_vel",
                "actuator_force",
                "actions",
                "cmd",
                "tilt_deg",
                "safety_samples",
            )
        }
        shapes = {
            "root_pos": (3,),
            "root_quat": (4,),
            "lin_vel_b": (3,),
            "ang_vel_b": (3,),
            "joint_pos": (18,),
            "joint_vel": (18,),
            "actuator_force": (18,),
            "actions": (18,),
            "cmd": (3,),
            "tilt_deg": (),
            "safety_samples": (len(SAFETY_KEYS),),
        }
        if any(v.shape != (n, *shapes[k]) for k, v in arrays.items()):
            raise ValueError("array lengths/shapes mismatch")
        if any(not np.isfinite(v).all() for v in arrays.values()):
            raise ValueError("nonfinite control record")
        physics = ep["physics"]
        if any(
            np.asarray(physics[k]).shape != (4 * n, *shape)
            for k, shape in [("qpos", (25,)), ("qvel", (24,)), ("ctrl", (18,)), ("active", ())]
        ):
            raise ValueError("physics coverage/shape mismatch")
        if any(not np.isfinite(np.asarray(v)).all() for v in physics.values()):
            raise ValueError("nonfinite physics record")
        if not np.asarray(physics["active"]).all():
            raise ValueError("inactive sample within episode")
        if np.abs(physics["qpos"][:, 7:]).max(initial=0) > np.pi + 1e-6:
            raise ValueError("numeric angle domain exceeded")
        for field, part in [
            ("root_pos", slice(0, 3)),
            ("root_quat", slice(3, 7)),
            ("joint_pos", slice(7, 25)),
        ]:
            if not np.allclose(arrays[field], physics["qpos"][3::4, part], rtol=0, atol=1e-5):
                raise ValueError("control/physics position mismatch")
        requested = float(ep["requested_vx"])
        if requested not in (0.0, 0.25):
            raise ValueError("unsupported requested command")
        if ep["command_name"] != ("stand" if requested == 0 else "forward_025"):
            raise ValueError("command identity mismatch")
        if n == 0:
            raise ValueError("empty episode")
        safety = dict(zip(SAFETY_KEYS, arrays["safety_samples"][-1]))
        if any(float(ep["safety"][k]) != float(safety[k]) for k in SAFETY_KEYS):
            raise ValueError("safety metadata mismatch")
        for key in SAFETY_KEYS:
            v = arrays["safety_samples"][:, SAFETY_KEYS.index(key)]
            if (v < 0).any() or (np.diff(v) < -1e-6).any():
                raise ValueError("invalid safety counter " + key)
        if safety["substep_samples"] != 4 * n or safety["tau_samples"] != 4 * n * 18:
            raise ValueError("substep sample counts mismatch")
        # Counters include native boundary refreshes. Raw physical frames must
        # never disagree with final cumulative control-frame records.
        for key in SAFETY_KEYS:
            v = np.asarray(physics[key])
            if (
                v.shape != (4 * n,)
                or (v < 0).any()
                or (np.diff(v) < -1e-6).any()
                or v[-1] > safety[key] + 1e-6
            ):
                raise ValueError("invalid physics counter " + key)
        violations = any(
            safety[k] != 0
            for k in (
                "nonfoot_events",
                "self_events",
                "nonfinite_events",
                "overflow_events",
                "angle_domain_events",
            )
        )
        if bool(ep["any_violation"]) != violations:
            raise ValueError("violation metadata mismatch")
        checks["data_valid"] = True
        checks["complete"] = (
            n == TOTAL_STEPS and not ep["early_terminated"] and ep["termination_step"] is None
        )
        checks["no_violation"] = not violations and bool(
            (np.abs(arrays["root_pos"][:, :2]) <= 11.0).all()
        )
        cmd_expected = np.array([requested, 0.0, 0.0])
        matches = bool(np.allclose(arrays["cmd"], cmd_expected, atol=1e-6, rtol=0))
        tau_peak = max(
            float(safety["applied_peak"]),
            float(np.abs(physics["ctrl"]).max()),
            float(np.abs(arrays["actuator_force"]).max()),
        )
        checks["torque_ok"] = tau_peak <= thresholds["applied_torque_Nm_max"] + 1e-6
        metrics.update(
            steps=n,
            command_matches=matches,
            applied_torque_max_Nm=tau_peak,
            preclip_peak_Nm=float(safety["preclip_peak"]),
            clip_fraction=float(safety["clip_count"] / safety["tau_samples"]),
        )
        if n > SETTLE_STEPS:
            r = arrays["root_pos"]
            mz = slice(SETTLE_STEPS, n)
            height = float(np.abs(r[mz, 2] - ep["witness_height"]).max())
            tilt = float(arrays["tilt_deg"][mz].max())
            checks["height_ok"] = height <= thresholds["height_err_m_max"]
            checks["tilt_ok"] = tilt <= thresholds["tilt_deg_max"]
            yaw = float(ep["initial_yaw_rad"])
            heading = np.array([np.cos(yaw), np.sin(yaw)])
            side = np.array([-np.sin(yaw), np.cos(yaw)])
            initial = np.asarray(ep["initial_root_pos"])
            disp = r[-1, :2] - initial[:2]
            lateral = float(np.abs((r[mz, :2] - initial[:2]) @ side).max())
            stand = float(np.linalg.norm(r[mz, :2] - r[SETTLE_STEPS - 1, :2], axis=1).max())
            along = float(disp @ heading)
            mae = float(np.abs(arrays["lin_vel_b"][mz, 0] - arrays["cmd"][mz, 0]).mean())
            checks["command_ok"] = matches and (
                stand <= thresholds["stand_measure_disp_m_max"]
                if requested == 0
                else along >= thresholds["forward_disp_along_heading_m_min"]
                and mae <= thresholds["forward_vx_mae_max"]
                and lateral <= thresholds["forward_lateral_drift_m_max"]
            )
            metrics.update(
                max_height_error_m=height,
                max_tilt_deg=tilt,
                stand_excursion_m=stand,
                lateral_excursion_m=lateral,
                disp_along_heading_m=along,
                vx_mae_mps=mae,
            )
        else:
            errors.append("terminated/truncated before measurement zone")
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        checks["data_valid"] = False
        errors.append(str(exc))
    return {**checks, "passed": all(checks.values()), "errors": errors, "metrics": metrics}


def summarize(episodes):
    by = {}
    rows = []
    for ep in episodes:
        verdict = validate_episode(ep)
        c = by.setdefault(ep["command_name"], {"n": 0, "passed": 0, "fail_reasons": {}})
        c["n"] += 1
        c["passed"] += int(verdict["passed"])
        for key, v in verdict.items():
            if isinstance(v, bool) and not v and key != "passed":
                c["fail_reasons"][key] = c["fail_reasons"].get(key, 0) + 1
        rows.append(
            dict(
                episode_id=ep["episode_id"], command=ep["command_name"], seed=ep["seed"], **verdict
            )
        )
    return dict(
        thresholds=T0B_THRESHOLDS, total_episodes=len(episodes), per_command=by, verdicts=rows
    )
