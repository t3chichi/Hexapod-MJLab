"""Finite real-CUDA acceptance check for the confined_locomotion_2 play viewer.

This is an inference/simulation smoke check, not a training entry point. It
starts one play environment, keeps the Viser page available for a bounded
optional UI inspection, then checks scan cadence, explicit recurrent state,
reset behavior, point provenance, and coexistence with the legacy viewer.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import urllib.request
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _differences(a, b, prefix=""):
    if torch.is_tensor(a):
        return [] if torch.is_tensor(b) and a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.cpu(), b.cpu()) else [prefix]
    if isinstance(a, np.ndarray):
        return [] if isinstance(b, np.ndarray) and np.array_equal(a, b) else [prefix]
    if hasattr(a, "__dataclass_fields__"):
        return _differences(vars(a), vars(b), prefix)
    if isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            return [prefix + ".keys"]
        return [path for key in a for path in _differences(a[key], b[key], f"{prefix}.{key}")]
    if isinstance(a, (tuple, list)):
        if not isinstance(b, type(a)) or len(a) != len(b):
            return [prefix]
        return [path for i, (x, y) in enumerate(zip(a, b)) for path in _differences(x, y, f"{prefix}[{i}]")]
    return [] if a == b else [prefix]


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert(condition, message):
    if not condition:
        raise AssertionError(message)


def _wait_http(url: str, timeout_s: float = 8.0) -> int:
    deadline = time.monotonic() + timeout_s
    last_error = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                return int(response.status)
        except Exception as error:  # server startup can lag the constructor
            last_error = error
            time.sleep(0.1)
    raise RuntimeError(f"viewer endpoint did not become reachable: {url}: {last_error}")


def _causal_state(session):
    """Copy every task-owned causal state plus physics/managers for pure UI checks."""
    from hexapod_mjlab.tasks.confined_locomotion.rl.checkpoint import (
        _manager_tensors,
        _physics_arrays,
    )

    physics, _manifest = _physics_arrays(session.env)
    manager = _manager_tensors(session.env)
    return {
        "runtime": session.runtime.state_dict(),
        "physics": physics,
        "manager": {name: tensor.detach().cpu().clone() for name, tensor in manager.items()},
        "hidden": session.hidden.detach().cpu().clone(),
        "frame_feature": None if session._frame_feature is None else session._frame_feature.detach().cpu().clone(),
        "feature_identity": copy.deepcopy(session._feature_identity),
        "counts": (session.forward_count, session.frame_encoding_count, session.preview_count),
        "network": {name: value.detach().cpu().clone() for name, value in session.network.state_dict().items()},
    }


def _assert_causal_unchanged(session, before, why):
    after = _causal_state(session)
    changed = _differences(before, after)
    _assert(not changed, f"{why} changed causal state: {changed[:12]}")


def _snapshot_checks(snapshot, viewer_module):
    _assert(snapshot.get("schema") == "confined_locomotion_2.play_snapshot/v1", "wrong snapshot schema")
    _assert(snapshot["input"]["near"]["points_base"].shape == (256, 3), "near cloud is not 256x3")
    _assert(snapshot["input"]["far"]["points_base"].shape == (128, 3), "far cloud is not 128x3")
    _assert(snapshot["prediction"].get("preview") is True, "initial snapshot is not marked as a preview")
    _assert(snapshot["status"].get("paused") is True, "initial display is not paused")
    _assert("ground_prediction" not in snapshot and "ground_truth" not in snapshot, "snapshot contains a ground prediction/label")
    _assert(set(snapshot["prediction"]) == {"envelope", "action_mean", "preview"}, "prediction contains unexpected heads")

    for kind, count in (("near", 256), ("far", 128)):
        cloud = snapshot["input"][kind]
        valid = np.asarray(cloud["valid"], dtype=bool)
        raw_ids = np.asarray(snapshot["provenance"][f"{kind}_raw_ids"], dtype=np.int64)
        _assert(raw_ids.shape == (count,), f"{kind} provenance shape mismatch")
        _assert(np.all(raw_ids[valid] >= 0), f"a valid {kind} point lacks a source-ray id")

    _assert(snapshot["input"]["near"]["scan_id"] == snapshot["input"]["far"]["scan_id"], "far cloud is not from the current scan")
    _assert(snapshot["identity"]["input_episode_id"] == snapshot["identity"]["pose_episode_id"], "idle preview has mismatched input/pose episode")

    # Prove pick provenance follows the explicit slot->raw-id map, not XYZ.
    valid_slots = np.flatnonzero(np.asarray(snapshot["input"]["far"]["valid"], dtype=bool))
    _assert(valid_slots.size > 0, "no valid far points are available for provenance testing")
    slot = int(valid_slots[0])
    original = viewer_module.resolve_point_pick(snapshot, "far", slot)
    changed_xyz = copy.deepcopy(snapshot)
    changed_xyz["input"]["far"]["points_base"][slot] += np.asarray([17.0, -11.0, 5.0])
    remapped = viewer_module.resolve_point_pick(changed_xyz, "far", slot)
    _assert(remapped["raw_ray_id"] == original["raw_ray_id"], "raw source mapping depends on XYZ")
    _assert(original["valid"], "selected valid far point did not resolve")
    return {"far_pick_slot": slot, "far_pick_raw_id": int(original["raw_ray_id"])}


def _drive_display_intents(viewer, session, snapshot, hold_seconds, checks, require_gui_input):
    """Serve the page without autoplay; only explicit Single step/reset/command intents mutate state."""
    deadline = time.monotonic() + max(0.0, hold_seconds)
    paused = True
    seen = {"rerender": 0, "pick": 0, "step": 0, "reset": 0, "command": 0}
    while time.monotonic() < deadline:
        pending = viewer.pending.take()
        display_only = bool(pending["rerender"] or pending["pick"] is not None or pending["env_row"] is not None)
        if display_only:
            before = _causal_state(session)
            if pending["env_row"] is not None:
                _assert(int(pending["env_row"]) == 0, "single-environment viewer selected a nonzero row")
            if pending["pick"] is not None:
                kind, index = pending["pick"]
                from hexapod_mjlab.tasks.confined_locomotion_2.visualization.play import resolve_point_pick
                viewer.set_pick_result(resolve_point_pick(snapshot, kind, int(index)))
                seen["pick"] += 1
            viewer.render(snapshot)
            _assert_causal_unchanged(session, before, "render/layer/pick interaction")
            seen["rerender"] += int(bool(pending["rerender"]))

        if pending["pause_toggle"]:
            before = _causal_state(session)
            paused = not paused
            snapshot["status"]["paused"] = paused
            viewer.render(snapshot)
            _assert_causal_unchanged(session, before, "pause display toggle")

        if pending["command"] is not None or pending["zero_command"]:
            values = (0.0, 0.0, 0.0) if pending["zero_command"] else pending["command"]
            session.set_command(values)
            session.refresh_idle_snapshot(reason="command change")
            snapshot = session.build_display_snapshot(0, paused=paused)
            viewer.render(snapshot)
            seen["command"] += 1

        if pending["reset"]:
            session.manual_reset()
            session.refresh_idle_snapshot(reason="command/reset")
            snapshot = session.build_display_snapshot(0, paused=True)
            viewer.set_pick_result(None)
            viewer.render(snapshot)
            paused = True
            seen["reset"] += 1

        if pending["step"]:
            before_steps = session.runtime.control_steps_taken
            before_forwards = session.forward_count
            step = session.step_once()
            _assert(session.runtime.control_steps_taken == before_steps + 1, "single-step intent did not advance exactly one control step")
            _assert(session.forward_count == before_forwards + 1, "single-step intent did not advance exactly one hidden state")
            _assert(torch.equal(session.hidden, step.output.hidden), "single-step hidden state does not match the network output")
            snapshot = session.build_display_snapshot(0, paused=True)
            viewer.render(snapshot)
            paused = True
            seen["step"] += 1

        if pending["exit"]:
            checks["ui_exit_requested"] = True
            break
        time.sleep(0.05)

    if require_gui_input:
        _assert(seen["rerender"] > 0, "no browser layer/render event was observed")
        _assert(seen["pick"] > 0, "no browser point-pick event was observed")
    checks["ui_events"] = seen
    return snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--new-port", type=int, default=8094)
    parser.add_argument("--legacy-port", type=int, default=8093)
    parser.add_argument("--hold-seconds", type=float, default=0.0,
                        help="keep the paused browser view open for manual CUA inspection")
    parser.add_argument("--require-gui-input", action="store_true",
                        help="require at least one browser layer/render and point-pick event during the hold")
    parser.add_argument("--raycast-mode", choices=("full", "sparse"), default="full")
    parser.add_argument("--sparse-selection")
    args = parser.parse_args()
    if args.hold_seconds < 0:
        parser.error("--hold-seconds must be nonnegative")
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    root = REPO_ROOT
    legacy_files = (
        root / "src/hexapod_mjlab/tasks/confined_locomotion/play.py",
        root / "src/hexapod_mjlab/tasks/confined_locomotion/visualization/play.py",
        root / "src/hexapod_mjlab/tasks/confined_locomotion/visualization/geometry.py",
    )
    hashes_before = {path.relative_to(root).as_posix(): _hash(path) for path in legacy_files}
    checks = {"passed": False, "legacy_source_hashes_before": hashes_before}
    started = time.monotonic()
    env = session = viewer = legacy_viewer = None
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("real CUDA is required; this script is not a CPU acceptance check")
        torch.cuda.set_device(args.gpu)
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
        from mjlab.utils.torch import configure_torch_backends
        from hexapod_mjlab.tasks.confined_locomotion_2.play import PlaySession, make_play_env_cfg
        from hexapod_mjlab.tasks.confined_locomotion_2.visualization.play import (
            PlayViewer,
            export_snapshot_manifest,
            resolve_point_pick,
        )
        from hexapod_mjlab.tasks.confined_locomotion.visualization.play import (
            PlayViewer as LegacyPlayViewer,
        )

        configure_torch_backends(allow_tf32=False)
        rnn_backend = getattr(torch.backends.cudnn, "rnn", None)
        if rnn_backend is not None and hasattr(rnn_backend, "fp32_precision"):
            rnn_backend.fp32_precision = "ieee"

        device = f"cuda:{args.gpu}"
        env = ManagerBasedRlEnv(make_play_env_cfg(num_envs=1, seed=args.seed), device=device)
        session = PlaySession(
            wrapper=RslRlVecEnvWrapper(env, clip_actions=None),
            checkpoint=args.checkpoint,
            lidar_cfg=__import__("hexapod_mjlab.tasks.confined_locomotion_2.scanner", fromlist=["ConfinedLidarCfg"]).ConfinedLidarCfg(raycast_mode=args.raycast_mode, sparse_selection_path=args.sparse_selection),
            output_dir=out / "play",
            seed=args.seed,
            own_environment=False,
        )
        runtime = session.runtime
        checks["no_optimizer_or_ground_head"] = not hasattr(session, "optimizer") and not hasattr(session.network, "ground_head")
        checks["initial_runtime_control_steps"] = int(runtime.control_steps_taken)
        _assert(runtime.control_steps_taken == 0, "play did not start at control step zero")
        _assert(session.forward_count == 0, "play session ran a control forward before the initial paused preview")
        hidden_before_preview = session.hidden.detach().clone()
        snapshot = session.build_display_snapshot(0, paused=True)
        _assert(torch.equal(session.hidden, hidden_before_preview), "initial preview committed temporal hidden state")
        _assert(session.forward_count == 0, "initial preview counted as a control transition")
        _assert(runtime.control_steps_taken == 0, "initial preview advanced the control clock")
        checks.update(_snapshot_checks(snapshot, sys.modules[PlayViewer.__module__]))
        checks["initial_snapshot_preview_hidden_unchanged"] = True

        # Compare cached snapshot rebuilds, scene rendering, view/layer changes,
        # and point resolution against a full runtime/physics/command RNG copy.
        initial_identity = snapshot["identity"]
        viewer = PlayViewer(
            num_envs=1,
            mj_model=session.env.sim.mj_model,
            pipeline=runtime.pipeline,
            identity_header={"task": "confined_locomotion_2", "checkpoint": str(Path(args.checkpoint).resolve())},
            host="127.0.0.1",
            port=args.new_port,
        )
        viewer.render(snapshot)
        _assert(len(viewer._handles) > 0, "new Viser render created no scene handles")
        _assert(not any("ground" in name.lower() for name in viewer._handles), "new viewer rendered a ground prediction/label layer")

        legacy_viewer = LegacyPlayViewer(
            num_envs=1,
            mj_model=session.env.sim.mj_model,
            pipeline=runtime.pipeline,
            ground_grid=SimpleNamespace(x_coords=(0.2, 1.2), y_coords=(-0.8, 0.8)),
            identity_header={"task": "confined_locomotion legacy coexistence check"},
            host="127.0.0.1",
            port=args.legacy_port,
        )
        checks["new_http_status"] = _wait_http(f"http://127.0.0.1:{args.new_port}/")
        checks["legacy_http_status"] = _wait_http(f"http://127.0.0.1:{args.legacy_port}/")

        before = _causal_state(session)
        for _ in range(3):
            snapshot = session.build_display_snapshot(0, paused=True)
            viewer.render(snapshot)
        for view in ("世界", "机器人"):
            viewer._view.value = view
            viewer.render(snapshot)
        for control in (viewer._near, viewer._far, viewer._pool, viewer._cone, viewer._envelope):
            original = bool(control.value)
            control.value = not original
            viewer.render(snapshot)
            control.value = original
            viewer.render(snapshot)
        valid_near = np.flatnonzero(np.asarray(snapshot["input"]["near"]["valid"], dtype=bool))
        _assert(valid_near.size > 0, "no valid near points are available for pick testing")
        viewer.set_pick_result(resolve_point_pick(snapshot, "near", int(valid_near[0])))
        viewer.render(snapshot)
        _assert_causal_unchanged(session, before, "cached render/layer/pick sequence")
        checks["repeated_cached_render_preserved_control_and_command_state"] = True
        checks["initial_snapshot_identity_stable"] = snapshot["identity"] == initial_identity

        manifest_path = out / "initial_snapshot_manifest.json"
        manifest = export_snapshot_manifest(snapshot, viewer, manifest_path)
        checks["manifest_has_no_ground_head"] = "no ground" in manifest["ground_semantics"].lower()
        _assert(checks["manifest_has_no_ground_head"], "manifest does not state that the model has no ground prediction")

        snapshot = _drive_display_intents(
            viewer, session, snapshot, args.hold_seconds, checks, args.require_gui_input
        )

        # Return to a clean episode after optional UI inspection. Verify reset
        # clears explicit state/cache and takes one fresh scan with a new ID.
        old_episode = int(runtime.episode_ids[0])
        old_sensor_count = int(runtime.sensor_scan_counts[0])
        old_control_clock = int(runtime.control_steps_taken)
        pre_reset_frames = session.frame_encoding_count
        reset_record = session.manual_reset()
        _assert(torch.count_nonzero(session.hidden) == 0, "manual reset did not zero recurrent hidden state")
        _assert(session._feature_identity is None and session._frame_feature is None, "manual reset did not clear scan/frame caches")
        _assert(session.peek() is None, "manual reset retained the previous action transition")
        _assert(int(runtime.episode_ids[0]) == old_episode + 1, "manual reset did not advance the episode identity")
        _assert(int(runtime.scan_counters[0]) == 1, "manual reset did not take a fresh episode scan")
        _assert(int(runtime.sensor_scan_counts[0]) == old_sensor_count + 1, "manual reset did not advance the persistent sensor scan id once")
        _assert(int(runtime.next_due_scan_step[0]) == old_control_clock + 5, "reset did not anchor the next scan five control steps later")
        checks["reset_scan_writes"] = reset_record["scan_writes"]
        checks["reset_cleared_hidden_and_feature_cache"] = True

        hidden_at_reset = session.hidden.detach().clone()
        session.refresh_idle_snapshot(reason="command/reset")
        snapshot = session.build_display_snapshot(0, paused=True)
        _assert(torch.equal(session.hidden, hidden_at_reset), "reset preview committed hidden state")
        expected_identity = (
            runtime.episode_ids.detach().clone(),
            runtime.scan_counters.detach().clone(),
        )
        _assert(
            session._feature_identity is not None
            and torch.equal(session._feature_identity[0], expected_identity[0])
            and torch.equal(session._feature_identity[1], expected_identity[1]),
            "reset preview did not bind the fresh episode scan identity",
        )
        viewer.set_pick_result(None)
        viewer.render(snapshot)

        # The reset preview encodes its current frame once; 50 control actions
        # then consume the ten 10-Hz frames at the 5:1 control/scan cadence.
        episode_steps_before = int(runtime.step_in_episode[0])
        controls_before = runtime.control_steps_taken
        forwards_before = session.forward_count
        encodings_before_reset = pre_reset_frames
        terminated = False
        for action_index in range(50):
            previous_control = runtime.control_steps_taken
            previous_forward = session.forward_count
            step = session.step_once()
            _assert(runtime.control_steps_taken == previous_control + 1, f"action {action_index} advanced the control clock other than once")
            _assert(session.forward_count == previous_forward + 1, f"action {action_index} advanced recurrent state other than once")
            _assert(torch.equal(session.hidden, step.output.hidden), f"action {action_index} committed an unexpected hidden state")
            if bool(step.transition.engineering_stopped):
                terminated = True
                break
        controls_delta = runtime.control_steps_taken - controls_before
        forwards_delta = session.forward_count - forwards_before
        encoding_delta = session.frame_encoding_count - encodings_before_reset
        checks["post_reset_50_control_steps"] = controls_delta
        checks["post_reset_50_control_forwards"] = forwards_delta
        checks["post_reset_50_frame_encodings_including_preview"] = encoding_delta
        _assert(not terminated and controls_delta == 50, f"play stopped before 50 finite control steps (done={terminated}, count={controls_delta})")
        _assert(forwards_delta == 50, f"50 control steps produced {forwards_delta} recurrent forwards")
        _assert(encoding_delta == 10, f"50 controls plus the reset preview should encode ten scan frames; got {encoding_delta}")
        _assert(int(runtime.step_in_episode[0]) == episode_steps_before + 50, "episode step counter did not advance exactly 50")
        checks["50_controls_10_frame_encodings"] = True

        # Reset from a nonzero hidden state as a final lifecycle check, then
        # retain a fresh idle preview for the paused display.
        episode_before_final_reset = int(runtime.episode_ids[0])
        sensor_before_final_reset = int(runtime.sensor_scan_counts[0])
        clock_before_final_reset = runtime.control_steps_taken
        session.manual_reset()
        _assert(torch.count_nonzero(session.hidden) == 0, "post-rollout reset did not clear temporal hidden")
        _assert(session._feature_identity is None and session._frame_feature is None, "post-rollout reset retained frame cache")
        _assert(int(runtime.episode_ids[0]) == episode_before_final_reset + 1, "post-rollout reset did not advance episode id")
        _assert(int(runtime.scan_counters[0]) == 1, "post-rollout reset did not start a fresh scan id")
        _assert(int(runtime.sensor_scan_counts[0]) == sensor_before_final_reset + 1, "post-rollout reset did not take one new persistent scan")
        _assert(int(runtime.next_due_scan_step[0]) == clock_before_final_reset + 5, "post-rollout reset did not schedule the next fresh scan")
        hidden_before_final_preview = session.hidden.detach().clone()
        session.refresh_idle_snapshot(reason="command/reset")
        _assert(torch.equal(session.hidden, hidden_before_final_preview), "post-reset idle preview committed hidden")

        # The test owns only this viewer layer; ensure both bound servers close
        # cleanly before comparing immutable legacy source files.
        final_snapshot = session.build_display_snapshot(0, paused=True)
        viewer.render(final_snapshot)
        checks["final_scan_shapes"] = (
            tuple(final_snapshot["input"]["near"]["points_base"].shape) == (256, 3)
            and tuple(final_snapshot["input"]["far"]["points_base"].shape) == (128, 3)
        )
        checks["passed"] = all(
            value for key, value in checks.items()
            if key != "passed" and isinstance(value, bool)
        )
    except BaseException as error:
        checks["passed"] = False
        checks["error"] = repr(error)
        checks["traceback"] = traceback.format_exc()
    finally:
        if legacy_viewer is not None:
            legacy_viewer.close()
        if viewer is not None:
            viewer.close()
        if session is not None:
            session.close()
        if env is not None:
            env.close()
        hashes_after = {path.relative_to(root).as_posix(): _hash(path) for path in legacy_files}
        checks["legacy_source_hashes_after"] = hashes_after
        checks["legacy_source_hashes_unchanged"] = hashes_before == hashes_after
        if not checks["legacy_source_hashes_unchanged"]:
            checks["passed"] = False
        checks["elapsed_s"] = time.monotonic() - started
        if torch.cuda.is_available():
            checks["gpu"] = torch.cuda.get_device_name(args.gpu)
            checks["peak_allocated_MiB"] = torch.cuda.max_memory_allocated(args.gpu) / 2**20
        (out / "checks.json").write_text(json.dumps(checks, indent=2, ensure_ascii=False, default=str))
        print(json.dumps(checks, indent=2, ensure_ascii=False, default=str))
    if not checks["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
