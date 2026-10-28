"""Read-only display retention for the recurrent play owner loop.

Source references are retained BEFORE stepping, because a scan-record slot
may be replaced at the next physical boundary. Debug events materialize these
inputs on CPU and pair them with the post-step pose. Inline capture remains
available for finite diagnostics; native playback does not call either path.
"""

from __future__ import annotations

import copy

import numpy as np
import torch

PLAY_SNAPSHOT_SCHEMA = "confined_locomotion_2.play_snapshot/v1"


def array(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().copy()
    return np.asarray(value).copy()


def point_sources(prepared, capture_entries, row):
    """Associate source-ray IDs, never coordinates, with original no-hit flags."""
    episode = int(prepared.observation.episode_ids[row])
    scan_id = int(prepared.near_scan_ids[row])
    matches = [entry for entry in capture_entries
               if int(entry["episode"]) == episode and int(entry["scan"]) == scan_id]
    result = {}
    for prefix in ("near", "far"):
        valid = array(getattr(prepared, f"{prefix}_valid")[row])
        indices = getattr(prepared, f"{prefix}_source_indices", None)
        if indices is None and valid.any():
            raise ValueError("display requires original source-ray indices")
        raw_ids = np.full(valid.shape, -1, dtype=np.int64) if indices is None else array(indices[row])
        no_hit = np.zeros(valid.shape, dtype=bool)
        if valid.any():
            if len(matches) != 1:
                raise ValueError("current input source scan metadata is missing or ambiguous")
            entry = matches[0]
            source = array(entry[f"{prefix}_indices"])
            flags = array(entry[f"{prefix}_no_hit"])
            lookup = {int(index): bool(flag) for index, flag in zip(source, flags) if index >= 0}
            for slot in np.flatnonzero(valid):
                raw_id = int(raw_ids[slot])
                if raw_id not in lookup:
                    raise ValueError("display point has no matching original source ray")
                no_hit[slot] = lookup[raw_id]
        result[f"{prefix}_raw_ids"] = raw_ids
        result[f"{prefix}_no_hit"] = no_hit
    return result


def retain_input_context(session):
    """Keep bounded source references without GPU-to-CPU debug extraction."""
    runtime = session.runtime
    return {"control_step": runtime.control_steps_taken,
            "sources": {row: tuple(entries) for row, entries in runtime._play_source_scans.items()},
            "records": tuple(runtime.pipeline.scan_records)}


def capture_inputs(session, prepared, *, context=None):
    """Retain CPU-only input products at the pre-action identity."""
    runtime = session.runtime
    captured = []
    for row in range(session.num_envs):
        obs = prepared.observation
        episode, scan_id = int(obs.episode_ids[row]), int(prepared.near_scan_ids[row])
        scan_time = float(prepared.near_times[row])
        inputs = {}
        for prefix in ("near", "far"):
            inputs[prefix] = {
                "points_base": array(getattr(prepared, f"{prefix}_points")[row]),
                "valid": array(getattr(prepared, f"{prefix}_valid")[row]),
                "scan_id": scan_id, "time_s": scan_time,
            }
        entries = (runtime._play_source_scans if context is None else context["sources"])[row]
        provenance = point_sources(prepared, entries, row)
        cache_key = (episode, scan_id)
        cached = session._display_pool_cache.get(row)
        if cached is None or cached[0] != cache_key:
            pool = None
            for record in reversed(runtime.pipeline.scan_records if context is None else context["records"]):
                frame = record["frame"]
                if int(record["env_row"]) == row and int(frame.scan_id) == scan_id:
                    pool = {
                        "points_base": array(frame.pool_points_base),
                        "near_mask": array(frame.near_mask), "far_mask": array(frame.far_mask),
                        "raw_indices": array(frame.pool_raw_indices), "scan_id": scan_id,
                    }
                    break
            session._display_pool_cache[row] = (cache_key, pool)
        else:
            pool = cached[1]
        captured.append({
            "identity": {"input_control_step": int(runtime.control_steps_taken if context is None else context["control_step"]),
                         "input_episode_id": episode, "input_step_in_episode": int(obs.step_ids[row]),
                         "input_time_s": float(obs.time_s[row])},
            "input": inputs, "provenance": provenance, "pool": pool,
            "command": array(obs.command[row]),
        })
    return captured


def retain_display(session, prepared, output, captured, *, preview, reason, ground_z_w=None):
    """Pair retained action inputs with the current real boundary pose."""
    import warp as wp

    runtime, env = session.runtime, session.env
    robot = env.scene["robot"]
    positions = array(robot.data.root_link_pos_w)
    quaternions = array(robot.data.root_link_quat_w)
    qpos = array(wp.to_torch(env.sim.wp_data.qpos)).reshape(session.num_envs, -1)
    if ground_z_w is None:
        ground_z_w = np.full(session.num_envs, float(np.asarray(runtime.pipeline.mesh_record["z"]).reshape(-1)[0]))
    envelope, action = array(output.envelope), array(output.action_mean)
    rows = []
    for row, data in enumerate(captured):
        ground_z = float(ground_z_w[row])
        identity = dict(data["identity"])
        identity.update({
            "pose_control_step": int(runtime.control_steps_taken),
            "episode_id": int(runtime.episode_ids[row]),
            "step_in_episode": int(runtime.step_in_episode[row]),
            "pose_episode_id": int(runtime.episode_ids[row]),
            "pose_step_in_episode": int(runtime.step_in_episode[row]),
            "pose_time_s": float(runtime.control_time_s()),
            "frame_encoding_count": session.frame_encoding_count,
            "forward_count": session.forward_count,
        })
        rows.append({
            "schema": PLAY_SNAPSHOT_SCHEMA, "env": row, "identity": identity,
            "pose": {"root_pos_w": positions[row], "root_quat_wxyz": quaternions[row],
                     "qpos": qpos[row], "root_height_m": float(positions[row, 2] - ground_z),
                     "ground_z_m": ground_z},
            "input": data["input"], "provenance": data["provenance"], "pool": data["pool"],
            "prediction": {"envelope": envelope[row], "action_mean": action[row], "preview": preview},
            "status": {"paused": True, "stopped": bool(runtime._stopped),
                       "stop_kind": (getattr(runtime, "stop_record", None) or {}).get("kind"),
                       "faulted": bool(runtime._faulted), "command": data["command"],
                       "reason": str(reason)},
            "checkpoint": {"path": str(session.checkpoint_path.resolve())},
        })
    session._display_rows = rows


def build_display_snapshot(session, env_row, *, paused):
    if isinstance(env_row, bool) or not isinstance(env_row, int) or not 0 <= env_row < session.num_envs:
        raise ValueError("display environment index is out of range")
    if session._display_rows is None:
        session.refresh_idle_snapshot(reason="initial preview")
    snapshot = copy.deepcopy(session._display_rows[env_row])
    snapshot["status"].update(paused=bool(paused), stopped=bool(session.runtime._stopped),
                              faulted=bool(session.runtime._faulted))
    return snapshot

