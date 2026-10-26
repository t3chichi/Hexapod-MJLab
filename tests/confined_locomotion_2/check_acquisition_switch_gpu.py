"""Real-CUDA lifecycle acceptance check for live play acquisition switching.

The check starts one full-resolution play owner and repeatedly replaces its
simulation session while keeping the same checkpoint and Viser server alive.
It is deliberately finite and writes all measurements into a fresh evidence
directory.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback
import weakref

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

SELECTION_DIR = ROOT / "src/hexapod_mjlab/tasks/confined_locomotion_2/assets/fixed_channels/v1"


def _check(condition, message):
    if not condition:
        raise AssertionError(message)


def _network_state(session):
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in session.network.state_dict().items()
    }


def _assert_weights(session, expected, context):
    actual = session.network.state_dict()
    _check(actual.keys() == expected.keys(), f"{context}: network state keys changed")
    changed = [
        name for name, tensor in actual.items()
        if tensor.dtype != expected[name].dtype
        or tensor.shape != expected[name].shape
        or not torch.equal(tensor.detach().cpu(), expected[name])
    ]
    _check(not changed, f"{context}: checkpoint tensors changed: {changed[:8]}")


def _identity(session):
    return session.runtime.scan_identity


def _memory(device):
    torch.cuda.synchronize(device)
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
    }


def _submit_viewer_request(viewer, mode, selection_path):
    """Run the same method registered as the Viser button's click callback."""
    if mode not in ("full", "sparse"):
        raise ValueError(mode)
    viewer._acquisition_mode.value = {"full": "全量", "sparse": "稀疏"}[mode]
    if selection_path is not None:
        requested = Path(selection_path).expanduser().resolve()
        label = next(
            (name for name, path in viewer.acquisition_options.items()
             if Path(path).expanduser().resolve() == requested),
            None,
        )
        _check(label is not None, f"viewer has no option for {requested}")
        pair = viewer._selection_pair(label)
        if pair is not None:
            old_far = str(viewer._acquisition_far.value)
            viewer._acquisition_near.value = pair[0]
            viewer._multiplier_selection_changed()
            _check(str(viewer._acquisition_far.value) == old_far,
                   "changing near multiplier changed far multiplier")
            old_near = str(viewer._acquisition_near.value)
            viewer._acquisition_far.value = pair[1]
            viewer._multiplier_selection_changed()
            _check(str(viewer._acquisition_near.value) == old_near,
                   "changing far multiplier changed near multiplier")
        else:
            viewer._acquisition_custom_selection.value = label
            viewer._custom_selection_changed()
        _check(viewer.pending.take().get("acquisition") is None,
               "changing a multiplier submitted acquisition without Apply")
    viewer._submit_acquisition(None)
    request = viewer.pending.take().get("acquisition")
    _check(request == {"mode": mode, "selection_path": (
        None if mode == "full" else str(Path(selection_path).expanduser().resolve())
    )}, f"viewer callback submitted unexpected request: {request!r}")
    return request


def _step_segment(run_play, owner, count, viewer):
    """Run a bounded segment and query a displayed point through the viewer intent."""
    session = owner.session
    reason, _snapshot = run_play(
        session, viewer, steps=count, start_running=True,
        switch_acquisition=owner.switch_acquisition,
    )
    _check(reason == "step_budget", f"{count}-control segment ended as {reason}")
    _check(session.runtime.control_steps_taken == count,
           f"segment ran {session.runtime.control_steps_taken} controls, expected {count}")
    _check(session.forward_count == count,
           f"segment ran {session.forward_count} forwards, expected {count}")
    expected_scan_counter = 1 + count // 5
    _check(int(session.runtime.scan_counters[0]) == expected_scan_counter,
           f"{count}-control segment scan counter is {int(session.runtime.scan_counters[0])}, "
           f"expected {expected_scan_counter}")
    # Refresh the paused preview at the runtime's latest identity before
    # selecting a slot. This frame can be the post-action next input, while
    # the last action itself consumed the preceding pre-action frame.
    session.refresh_idle_snapshot(reason="acquisition switch acceptance source query")
    current_snapshot = session.build_display_snapshot(0, paused=True)
    viewer.render(current_snapshot)
    valid_far = np.flatnonzero(np.asarray(current_snapshot["input"]["far"]["valid"], dtype=bool))
    _check(valid_far.size > 0, "current acquisition produced no valid far point to query")
    pick_index = int(valid_far[0])
    viewer.pending.submit_pick("far", pick_index)
    # At the completed budget run_play still drains pending UI intents before
    # its budget check, so this exercises the owner-loop source query path.
    run_play(session, viewer, steps=count, start_running=True,
             switch_acquisition=owner.switch_acquisition)
    pick = viewer._last_pick
    queried_snapshot = viewer._last_snapshot
    _check(pick is not None, "viewer did not retain the latest source query")
    _check(queried_snapshot is not None, "viewer lost the snapshot used by the source query")
    _check(int(pick["env"]) == 0, "source query has the wrong env row")
    _check(int(pick["index"]) == pick_index, "source query returned the wrong displayed slot")
    _check(int(pick["episode_id"]) == int(queried_snapshot["identity"]["input_episode_id"]),
           "source query episode metadata differs from the displayed input frame")
    _check(int(pick["scan_id"]) == int(queried_snapshot["input"]["far"]["scan_id"]),
           "source query scan metadata differs from the displayed input frame")
    _check(bool((session.runtime.pipeline.source_ray_ids == int(pick["raw_ray_id"])).any()),
           "source query raw ID is not in the active acquisition")
    return {
        "requested_controls": count,
        "actual_controls": int(session.runtime.control_steps_taken),
        "forward_count": int(session.forward_count),
        "frame_encoding_count": int(session.frame_encoding_count),
        "scan_counter": int(session.runtime.scan_counters[0]),
        "sensor_scan_count": int(session.runtime.sensor_scan_counts[0]),
        "query_input_scan_id": int(queried_snapshot["input"]["far"]["scan_id"]),
        "runtime_scan_counter_after_segment": int(session.runtime.scan_counters[0]),
        "source_query": {key: pick[key] for key in (
            "kind", "index", "raw_ray_id", "scan_id", "episode_id", "env", "valid"
        )},
    }


def _assert_fresh_session(owner, mode, selection_path, checkpoint, weights, expected_count):
    session = owner.session
    runtime = session.runtime
    pipeline = runtime.pipeline
    _check(Path(session.checkpoint_path).resolve() == checkpoint,
           "switch changed the checkpoint path")
    _assert_weights(session, weights, f"{mode} session")
    expected_raw = 86400 if mode == "full" else expected_count
    _check(int(pipeline.num_raw_rays) == expected_raw,
           f"{mode} runtime has {pipeline.num_raw_rays} raw rays; expected {expected_raw}")
    _check(int(pipeline.source_ray_ids.numel()) == expected_raw,
           f"{mode} source ID count differs from raw ray count")
    identity = _identity(session)
    _check(identity.get("mode") == mode, f"runtime scan identity mode is {identity.get('mode')!r}")
    if mode == "sparse":
        from hexapod_mjlab.tasks.confined_locomotion_2.channels import (
            build_channel_catalog, load_sparse_selection,
        )
        selected_np, _selection_meta = load_sparse_selection(selection_path, build_channel_catalog())
        ids = torch.as_tensor(pipeline.source_ray_ids, dtype=torch.long).detach().cpu()
        _check(ids.numel() == expected_count and bool((ids[1:] > ids[:-1]).all()),
               "sparse source IDs are not the expected sorted unique selection")
        _check(np.array_equal(ids.numpy(), selected_np),
               "active sparse raw IDs do not match the chosen fixed-T1 selection membership")
        report = session.scan_evaluation_report
        _check(report.get("cross_configuration") is True and report.get("override_enabled") is True,
               "sparse runtime did not report checkpoint cross-configuration evaluation")
        actual_path = Path(pipeline.cfg.sparse_selection_path).expanduser().resolve()
        _check(actual_path == selection_path, "runtime sparse selection does not match request")
    else:
        _check(torch.equal(
            pipeline.source_ray_ids.detach().cpu(), torch.arange(86400, dtype=torch.long)
        ), "full source IDs are not the full global catalog")
    _check(torch.count_nonzero(session.hidden).item() == 0,
           f"{mode} session did not start with zero recurrent state")
    _check(int(runtime.control_steps_taken) == 0 and int(session.forward_count) == 0,
           f"{mode} session did not restart at control step zero")
    _check(int(runtime.scan_counters[0]) == 1,
           f"{mode} session did not take exactly its initial scan")
    command = runtime.current_command().detach().cpu()
    _check(torch.equal(command, torch.zeros_like(command)),
           f"{mode} session command did not reset to zero")
    return identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True,
                        help="explicit compatible checkpoint; historical local logs are not shipped")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--port", type=int, default=8196)
    parser.add_argument("--settle-seconds", type=float, default=0.15)
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    near4 = (SELECTION_DIR / "near_x4_far_x4.npz").resolve()
    near8 = (SELECTION_DIR / "near_x8_far_x4.npz").resolve()
    independent = SELECTION_DIR.parent / "v1_independent"
    near4_far8 = (independent / "near_x4_far_x8.npz").resolve()
    near8_far8 = (independent / "near_x8_far_x8.npz").resolve()
    checks = {"passed": False, "checkpoint": str(checkpoint), "port": args.port}
    owner = viewer = None
    old_session_refs = []
    started = time.monotonic()
    device = f"cuda:{args.gpu}"
    try:
        _check(torch.cuda.is_available(), "real CUDA is required for this acceptance check")
        _check(checkpoint.is_file(), f"checkpoint does not exist: {checkpoint}")
        _check(all(p.is_file() for p in (near4, near8, near4_far8, near8_far8)), "independent near/far selection files are required")
        torch.cuda.set_device(args.gpu)
        play_path = ROOT / "scripts/experiments/confined_locomotion_2/play_flat.py"
        spec = importlib.util.spec_from_file_location("confined_locomotion_2_play_gpu_check", play_path)
        play_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(play_module)
        PlayOwner, run_play = play_module.PlayOwner, play_module.run_play
        from hexapod_mjlab.tasks.confined_locomotion_2.visualization.play import PlayViewer

        # PlayOwner(args) is the same production owner used by the play entry
        # point. The test selects a dedicated localhost port and fresh output.
        owner_args = argparse.Namespace(
            checkpoint=str(checkpoint), output=str(output / "play"), seed=args.seed,
            gpu=args.gpu, num_envs=1, command=(0.0, 0.0, 0.0), preset="clean",
            no_viewer=False, start_running=True, port=args.port, steps=0,
            raycast_mode="full", sparse_selection_path=None,
            allow_scan_override=False,
        )
        owner = PlayOwner(owner_args)
        viewer = PlayViewer(
            num_envs=owner.session.num_envs,
            mj_model=owner.env.sim.mj_model,
            pipeline=owner.session.runtime.pipeline,
            identity_header={"task": "acquisition-switch-acceptance", "checkpoint": str(checkpoint)},
            port=args.port,
            acquisition_options=owner.acquisition_options,
        )
        owner.viewer = viewer
        viewer.set_command_inputs((0, 0, 0))
        _check(viewer is owner.viewer, "PlayOwner did not retain the browser viewer")
        _check(int(viewer.server.get_port()) == args.port,
               "owner did not retain the requested viewer port")
        opener = owner.session
        original_checkpoint = Path(opener.checkpoint_path).resolve()
        reference_weights = _network_state(opener)
        _check(int(opener.runtime.pipeline.num_raw_rays) == 86400,
               "initial full session does not use all 86400 rays")
        del opener
        checks["gpu"] = torch.cuda.get_device_name(args.gpu)
        checks["checkpoint_path"] = str(original_checkpoint)
        checks["initial_memory"] = _memory(device)
        checks["segments"] = []
        checks["switches"] = []

        # Sequence intentionally returns to prior modes and repeats x4 to
        # expose retained env/runtime allocations across rebuilds.
        sequence = [
            ("sparse", near4, 1536, 5),
            ("sparse", near4_far8, 2048, 5),
            ("sparse", near8_far8, 3072, 5),
            ("sparse", near8, 2560, 5),
            ("full", None, 86400, 5),
            ("sparse", near4, 1536, 5),
        ]
        segment_plan = [5]
        previous_mode = "full"
        previous_selection = None
        count_index = 0
        # First full control segment validates periodic scan update before any
        # replacement, then the four acquisition changes follow the UI path.
        checks["segments"].append(_step_segment(run_play, owner, segment_plan[0], viewer))
        checks["segments"][-1]["mode"] = "full"
        for mode, selection_path, raw_count, _post_switch_steps in sequence:
            old = owner.session
            old_env = owner.env
            old_runtime = old.runtime
            old_session_refs.append(weakref.ref(old))
            old_env_ref = weakref.ref(old_env)
            old_runtime_ref = weakref.ref(old_runtime)
            request = _submit_viewer_request(viewer, mode, selection_path)
            mem_before = _memory(device)
            replacement = owner.switch_acquisition(request)
            _check(replacement is owner.session, "owner did not return its replacement session")
            del replacement
            # Release all references owned by this test before checking GC.
            del old, old_env, old_runtime
            gc.collect()
            torch.cuda.empty_cache()
            _check(old_session_refs[-1]() is None,
                   "old PlaySession remains strongly referenced after switch")
            _check(old_env_ref() is None and old_runtime_ref() is None,
                   "old environment/runtime remains strongly referenced after switch")
            report_identity = _assert_fresh_session(
                owner, mode, selection_path, original_checkpoint,
                reference_weights, raw_count,
            )
            mem_after = _memory(device)
            checks["switches"].append({
                "request": request,
                "identity": report_identity,
                "raw_ray_count": int(owner.session.runtime.pipeline.num_raw_rays),
                "memory_before": mem_before,
                "memory_after": mem_after,
                "old_objects_collected": True,
            })
            count = _post_switch_steps
            segment = _step_segment(run_play, owner, count, viewer)
            segment["mode"] = mode
            segment["selection_path"] = None if selection_path is None else str(selection_path)
            checks["segments"].append(segment)
            previous_mode, previous_selection = mode, selection_path
            count_index += 1
            time.sleep(max(0.0, args.settle_seconds))

        # Recheck server liveness and stable memory for the repeated sparse-x4
        # same-step session, allowing allocator warm-up noise of a few MiB.
        import urllib.request
        with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/", timeout=3.0) as response:
            checks["viewer_http_status"] = int(response.status)
        torch.cuda.synchronize(device)
        final_memory = _memory(device)
        first_after = checks["switches"][0]["memory_after"]["allocated_bytes"]
        final_allocated = final_memory["allocated_bytes"]
        checks["final_memory"] = final_memory
        checks["live_allocated_growth_bytes"] = final_allocated - first_after
        _check(final_allocated <= first_after + 8 * 2**20,
               "live allocated CUDA memory accumulated across repeated switches")
        checks["same_mode_same_step_allocated_delta_bytes"] = abs(
            checks["switches"][0]["memory_after"]["allocated_bytes"]
            - checks["switches"][-1]["memory_after"]["allocated_bytes"]
        )
        _check(checks["same_mode_same_step_allocated_delta_bytes"] <= 8 * 2**20,
               "repeated sparse-x4 sessions differ by more than 8 MiB live allocated memory")
        checks["switch_order"] = ["full"] + [("full" if mode == "full" else selection.stem) for mode, selection, _, _ in sequence]
        checks["old_session_weakrefs_dead"] = all(reference() is None for reference in old_session_refs)
        _check(checks["old_session_weakrefs_dead"], "a replaced PlaySession remains alive")
        checks["passed"] = True
    except BaseException as error:
        checks["error"] = repr(error)
        checks["traceback"] = traceback.format_exc()
    finally:
        if owner is not None:
            try:
                owner.close(reason="acquisition_switch_gpu_check")
            except Exception as error:
                checks["close_error"] = repr(error)
                checks["passed"] = False
        gc.collect()
        if torch.cuda.is_available():
            try:
                checks["final_memory_after_close"] = _memory(device)
            except Exception:
                pass
        checks["elapsed_s"] = time.monotonic() - started
        (output / "checks.json").write_text(
            json.dumps(checks, indent=2, ensure_ascii=False, default=str)
        )
        print(json.dumps(checks, indent=2, ensure_ascii=False, default=str))
    if not checks["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
