"""Source association and interactive owner-loop regression checks."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def test_original_ray_flags_follow_sorted_source_ids_including_padding():
    from hexapod_mjlab.tasks.confined_locomotion_2.display import point_sources
    prepared = SimpleNamespace(
        observation=SimpleNamespace(episode_ids=torch.tensor([2])), near_scan_ids=torch.tensor([7]),
        near_valid=torch.tensor([[True, True, False]]), far_valid=torch.tensor([[True, False]]),
        near_source_indices=torch.tensor([[9, 4, -1]]), far_source_indices=torch.tensor([[12, -1]]))
    source = [{"episode": 2, "scan": 7,
               "near_indices": np.array([4, 9, -1]), "near_no_hit": np.array([True, False, False]),
               "far_indices": np.array([12, -1]), "far_no_hit": np.array([True, False])}]
    result = point_sources(prepared, source, 0)
    assert result["near_raw_ids"].tolist() == [9, 4, -1]
    assert result["near_no_hit"].tolist() == [False, True, False]
    assert result["far_no_hit"].tolist() == [True, False]
    try:
        point_sources(prepared, [{**source[0], "episode": 1}], 0)
    except ValueError:
        pass
    else:
        raise AssertionError("stale episode source metadata must be rejected")


def test_cli_owner_loop_paused_rerender_single_step_and_reset():
    spec = importlib.util.spec_from_file_location("v2_play_cli", ROOT / "scripts/experiments/confined_locomotion_2/play_flat.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    from hexapod_mjlab.common.visualization import PendingControls

    class Session:
        def __init__(self):
            self.runtime = SimpleNamespace(_faulted=False, _stopped=False)
            self.forward_count = self.hidden = self.resets = self.previews = 0
        def refresh_idle_snapshot(self, **_):
            self.previews += 1
        def build_display_snapshot(self, row, *, paused):
            return {"row": row, "paused": paused, "controls": self.forward_count, "hidden": self.hidden}
        def capture_last_step_display(self):
            pass
        def step_once(self, **kwargs):
            self.forward_count += 1
            self.hidden += 1
            return SimpleNamespace(transition=SimpleNamespace(engineering_stopped=False))
        def manual_reset(self):
            self.resets += 1
            self.hidden = 0
        def set_command(self, values):
            assert tuple(values) == (0, 0, 0)

    class Viewer:
        def __init__(self):
            self.intents = iter(("rerender", "step", "reset", "step", "exit"))
            self.pending = self
            self.frames = []
        def take(self):
            pending = PendingControls()
            getattr(pending, "submit_" + next(self.intents))()
            return pending.take()
        def render(self, snapshot):
            self.frames.append(dict(snapshot))
        def set_pick_result(self, value):
            assert value is None
        def set_command_inputs(self, values):
            assert tuple(values) == (0, 0, 0)

    session, viewer = Session(), Viewer()
    reason, _ = cli.run_play(session, viewer, steps=0)
    assert reason == "viewer_exit"
    assert session.forward_count == 2 and session.hidden == 1
    assert session.resets == 1 and session.previews == 2
    assert viewer.frames[0]["controls"] == 0
    assert all(frame["paused"] for frame in viewer.frames)


def test_cli_acquisition_switch_preserves_total_budget_and_drops_old_intents():
    spec = importlib.util.spec_from_file_location("v2_play_switch_cli", ROOT / "scripts/experiments/confined_locomotion_2/play_flat.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    from hexapod_mjlab.common.visualization import PendingControls

    class Session:
        def __init__(self):
            self.runtime = SimpleNamespace(_faulted=False, _stopped=False)
            self.forward_count = 0
        def refresh_idle_snapshot(self, **_):
            pass
        def build_display_snapshot(self, row, *, paused):
            return {"controls": self.forward_count, "paused": paused}
        def capture_last_step_display(self):
            pass
        def step_once(self, **kwargs):
            self.forward_count += 1
            return SimpleNamespace(transition=SimpleNamespace(engineering_stopped=False))
        def set_command(self, *_):
            raise AssertionError("old queued command reached replacement")
        def manual_reset(self):
            raise AssertionError("old queued reset reached replacement")

    class Viewer:
        def __init__(self):
            self.pending = self
            self.calls = 0
        def take(self):
            p = PendingControls().take()
            if self.calls == 2:
                p.update(acquisition={"mode": "sparse", "selection_path": "fixture.npz"},
                         command=(1, 0, 0), reset=True, step=True, pause_toggle=True)
            elif self.calls == 3:
                p["pause_toggle"] = True
            self.calls += 1
            return p
        def render(self, _):
            pass
        def set_pick_result(self, _):
            pass

    original, replacement, viewer = Session(), Session(), Viewer()
    requests = []
    def switch(request):
        requests.append(request)
        return replacement
    reason, _ = cli.run_play(original, viewer, steps=4, start_running=True,
                             switch_acquisition=switch)
    assert reason == "step_budget"
    assert original.forward_count == 2
    assert replacement.forward_count == 2
    assert requests == [{"mode": "sparse", "selection_path": "fixture.npz"}]


def test_cli_discovers_independent_offline_combinations_without_replacing_v1():
    spec = importlib.util.spec_from_file_location("v2_play_discovery", ROOT / "scripts/experiments/confined_locomotion_2/play_flat.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    asset = ROOT / "src/hexapod_mjlab/tasks/confined_locomotion_2/assets/fixed_channels"
    options = cli.discover_acquisition_options([asset / "v1", asset / "v1_independent"])
    assert len(options) == 16
    assert Path(options["near x4 / far x4"]).parent == asset / "v1"
    assert Path(options["near x4 / far x8"]).parent == asset / "v1_independent"
    explicit = asset / "v1_independent/near_x4_far_x4.npz"
    override = cli.discover_acquisition_options([asset / "v1"], explicit)
    assert Path(override["near x4 / far x4"]) == explicit
    with_hit10 = cli.discover_acquisition_options(
        [asset / "v1", asset / "v1_independent", asset / "v1_hit10"]
    )
    assert len(with_hit10) == 17
    assert Path(with_hit10["near x6 / far x10"]).parent == asset / "v1_hit10"


def test_cli_discovery_rejects_incorrect_multiplier_metadata(tmp_path):
    import json
    import shutil
    spec = importlib.util.spec_from_file_location("v2_play_discovery_reject", ROOT / "scripts/experiments/confined_locomotion_2/play_flat.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    source = ROOT / "src/hexapod_mjlab/tasks/confined_locomotion_2/assets/fixed_channels/v1/near_x4_far_x4.npz"
    target = tmp_path / source.name
    shutil.copyfile(source, target)
    metadata = json.loads(source.with_suffix(".json").read_text())
    metadata["selection"]["far_multiplier"] = 8
    target.with_suffix(".json").write_text(json.dumps(metadata))
    try:
        cli.discover_acquisition_options([tmp_path])
    except ValueError as error:
        assert "counts disagree" in str(error)
    else:
        raise AssertionError("incorrect declared candidate count accepted")


def test_running_debug_only_materializes_snapshots_on_explicit_events():
    from hexapod_mjlab.common.visualization import PendingControls
    from hexapod_mjlab.tasks.confined_locomotion_2.visualization.play_cli import run_play
    class Session:
        forward_count = captures = previews = retained = 0
        runtime = SimpleNamespace(_faulted=False, _stopped=False)
        def refresh_idle_snapshot(self, **_):
            self.previews += 1
            self.retained = self.forward_count
        def capture_last_step_display(self):
            self.captures += 1
            self.retained = self.forward_count
        def build_display_snapshot(self, row, *, paused):
            return {"controls": self.retained, "paused": paused}
        def step_once(self, *, capture_display, retain_debug):
            assert not capture_display and retain_debug
            self.forward_count += 1
            return SimpleNamespace(transition=SimpleNamespace(engineering_stopped=False))
    class Viewer:
        def __init__(self):
            self.pending = self
            self.events = iter((None, None, "rerender", "refresh", "pause_toggle", "step", "exit"))
            self.frames = []
        def take(self):
            pending = PendingControls()
            event = next(self.events)
            if event == "refresh":
                return {**pending.take(), "refresh": True}
            if event is not None:
                getattr(pending, "submit_" + event)()
            return pending.take()
        def render(self, snapshot):
            self.frames.append(snapshot)
    session, viewer = Session(), Viewer()
    reason, _ = run_play(session, viewer, steps=0, start_running=True)
    assert reason == "viewer_exit" and session.forward_count == 5
    assert session.previews == 1 and session.captures == 3
    assert [frame["controls"] for frame in viewer.frames] == [0, 0, 3, 4, 5]


def test_native_buffers_are_independent_and_rebind_checks_actual_model():
    from contextlib import nullcontext
    from copy import copy
    from unittest.mock import Mock, patch
    import mujoco
    from mjlab.viewer import ViewerConfig
    from hexapod_mjlab.tasks.confined_locomotion_2.visualization.native import NativePlayViewer
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint/>'
        '<geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>')
    def runtime(model):
        sim = SimpleNamespace(mj_model=model, model=None, expanded_fields=set(), data=SimpleNamespace(
            qpos=torch.tensor(model.qpos0).reshape(1, -1), qvel=torch.zeros(1, model.nv)))
        return SimpleNamespace(num_envs=1, env=SimpleNamespace(sim=sim, cfg=SimpleNamespace(viewer=ViewerConfig())),
                               control_time_s=lambda: .02)
    source = runtime(model)
    handle = SimpleNamespace(lock=nullcontext, is_running=lambda: True, cam=SimpleNamespace(),
                             set_texts=Mock(), sync=Mock(), close=Mock())
    with patch('mujoco.viewer.launch_passive', return_value=handle):
        viewer = NativePlayViewer(source)
    assert viewer.model is not model
    viewer.data.qpos[0] = 100
    assert float(source.env.sim.data.qpos[0, 0]) == 0
    viewer.sync(paused=False)
    np.testing.assert_array_equal(viewer.data.qpos, source.env.sim.data.qpos[0])
    replacement = runtime(copy(model))
    replacement.env.sim.data.qpos[0, 0] = .25
    viewer.detach_session()
    viewer.rebind_session(replacement)
    viewer.sync(paused=True)
    assert viewer.handle is handle and viewer.data.qpos[0] == .25
    changed = copy(model)
    changed.body_mass[1] *= 2  # Same dimensions and names, different model.
    try:
        viewer.rebind_session(runtime(changed))
    except ValueError:
        pass
    else:
        raise AssertionError('incompatible display model accepted')
    assert viewer.runtime is replacement
    replacement._faulted = True
    replacement.env.sim.data.qpos[:] = float('nan')
    viewer.set_status(paused=True)  # Status must not read a faulted pose.
    assert 'FAULTED' in handle.set_texts.call_args.args[0][-1]
    assert viewer.data.qpos[0] == .25
    viewer.close()
    viewer.close()
    assert handle.close.call_count == 1
