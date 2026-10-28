"""Finite real-terrain play checks: alignment, read-only rendering and reset."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import torch
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8097)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"passed": False, "optimizer": False}
    env = session = viewer = None
    try:
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
        from mjlab.utils.torch import configure_torch_backends
        from hexapod_mjlab.tasks.confined_locomotion_2.config.terrain_memory_config import make_play_env_cfg
        from hexapod_mjlab.tasks.confined_locomotion_2.play_terrain_memory import TerrainMemoryPlaySession
        from hexapod_mjlab.tasks.confined_locomotion_2.components.lidar.scanner import ConfinedLidarCfg
        from hexapod_mjlab.tasks.confined_locomotion_2.visualization.play_cli import PlayOwner
        from hexapod_mjlab.tasks.confined_locomotion_2.visualization.terrain_debug import TerrainMemoryDebugViewer, height_points_world
        from hexapod_mjlab.tasks.confined_locomotion_2.visualization.debug import export_snapshot_manifest
        configure_torch_backends(allow_tf32=False)
        torch.backends.cudnn.rnn.fp32_precision = "ieee"
        root = Path(__file__).resolve().parents[2]
        selection = root / "src/hexapod_mjlab/tasks/confined_locomotion_2/assets/fixed_channels/v1_hit10/near_x6_far_x10.npz"
        env = ManagerBasedRlEnv(make_play_env_cfg(num_envs=2, seed=20261025), device="cuda:0")
        session = TerrainMemoryPlaySession(
            checkpoint=args.checkpoint, output_dir=args.output / "session", seed=20261025,
            wrapper=RslRlVecEnvWrapper(env, clip_actions=None), own_environment=True,
            lidar_cfg=ConfinedLidarCfg(raycast_mode="sparse", sparse_selection_path=str(selection), far_sampling="hit_only_10m"))
        viewer = TerrainMemoryDebugViewer(2, env.sim.mj_model, session.runtime.pipeline,
                                    {"task": session.task_id}, port=args.port)
        weights_before = PlayOwner.weight_digest(session.network)
        session.refresh_idle_snapshot()
        assert not bool(session.hidden.any())
        errors = []
        first = None
        for step in range(6):
            # Compare deferred event capture to the original pre-action CPU
            # capture, including a step that replaces the scan-record slot.
            prepared = session.runtime.prepare()
            captured = session.display.capture_inputs(session, prepared)
            result = session.step_once(capture_display=False, retain_debug=True)
            session.display.retain_display(session, result.prepared, result.output, captured,
                preview=False, reason="last action inputs; post-step pose")
            expected = copy.deepcopy(session._display_rows)
            hidden = session.hidden.clone()
            session._display_pool_cache.clear()  # Exercise retained old scan references.
            session.capture_last_step_display()
            from check_gpu import differences
            assert not differences(expected, session._display_rows), differences(expected, session._display_rows)
            assert torch.equal(hidden, session.hidden) and session.forward_count == step + 1
            for row in range(2):
                snapshot = session.build_display_snapshot(row)
                data = snapshot["terrain"]
                assert data["identity"]["input_control_step"] == step
                assert snapshot["identity"]["pose_control_step"] == step+1
                points = height_points_world(data, "target")
                reference, valid = session.runtime.terrain_query.surface.reference_heights(points[:, :2])
                assert valid.all()
                errors.append(float(np.max(np.abs(points[:, 2]-reference))))
                viewer.render(snapshot)
                if first is None:
                    first = snapshot
                    first_target = data["target"].copy()
        report["deferred_capture_matches_pre_action_capture"] = True
        assert max(errors) < 1e-5
        np.testing.assert_array_equal(first["terrain"]["target"], first_target)
        snapshot = session.build_display_snapshot(0)
        before = {
            "hidden": session.hidden.clone(), "qpos": wp.to_torch(env.sim.wp_data.qpos).clone(),
            "scans": session.runtime.scan_counters.clone(), "steps": session.forward_count,
            "preview": session.preview_count, "encodings": session.frame_encoding_count,
        }
        label = session.runtime.labels()["height"]
        for mode in ("世界", "机器人"):
            viewer._view.value = mode
            viewer._grid_preview.half_x.value = .75
            viewer._grid_preview.half_y.value = .65
            viewer._height_target.value = not viewer._height_target.value
            viewer.render(snapshot)
        assert torch.equal(before["hidden"], session.hidden)
        assert torch.equal(before["qpos"], wp.to_torch(env.sim.wp_data.qpos))
        assert torch.equal(before["scans"], session.runtime.scan_counters)
        assert before["steps"] == session.forward_count
        assert before["preview"] == session.preview_count
        assert before["encodings"] == session.frame_encoding_count
        assert session.runtime.labels()["height"] is label
        assert session.runtime.height_grid.x_range_m == tuple(snapshot["terrain"]["grid"]["x_range_m"])
        assert weights_before == PlayOwner.weight_digest(session.network)
        assert all(p.grad is None for p in session.network.parameters())
        export_snapshot_manifest(snapshot, viewer, args.output / "snapshot_manifest.json")
        session.manual_reset()
        assert not bool(session.hidden.any())
        session.refresh_idle_snapshot(reason="reset check")
        for row in range(2):
            reset = session.build_display_snapshot(row)
            assert reset["terrain"]["identity"]["input_step_in_episode"] == 0
            assert reset["terrain"]["identity"]["input_episode_id"] == 1
            viewer.render(reset)
        report.update(passed=True, controls=6, num_envs=2, max_height_error_m=max(errors),
                      rerender_preserves_physics_scans_gru=True, label_cache_preserved=True,
                      weights_unchanged=True, reset_zeroes_gru=True, no_gradients=True,
                      grid=session.height_grid.identity(), checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest())
    except BaseException as error:
        report.update(error=repr(error), traceback=traceback.format_exc())
    finally:
        if viewer is not None:
            viewer.close()
        if session is not None:
            session.close()
        elif env is not None:
            env.close()
        (args.output / "checks.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
