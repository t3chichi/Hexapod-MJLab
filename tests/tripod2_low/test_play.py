"""Native play adapters keep the legacy policy and reset boundaries."""
from io import StringIO
from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import patch

import torch
import glfw

from hexapod_mjlab.common.visualization import PendingControls
from hexapod_mjlab.tasks.confined_locomotion_2.tripod2_low.play import TripodPlaySession
from hexapod_mjlab.tasks.confined_locomotion_2.visualization.native import NativePlayViewer


def test_tripod_play_counts_policy_steps_separately_and_resets_recurrent_state():
    session = TripodPlaySession.__new__(TripodPlaySession)
    session.num_envs, session.device = 1, torch.device("cpu")
    session.hidden, session.forward_count = torch.zeros(1, 187), 0
    session._metrics = StringIO()
    prepared = SimpleNamespace(observation=object(), near_points=None, near_valid=None,
        far_points=None, far_valid=None, near_angle_keys=None, far_angle_keys=None)
    transition = SimpleNamespace(reward=torch.ones(1), terminated=torch.tensor([False]),
        context_record={"root_pos_w": torch.tensor([[0., 0., .5]])})
    session.runtime = SimpleNamespace(prepare=lambda: prepared, step=Mock(return_value=transition),
                                     reset=Mock(return_value={"reset": True}), control_steps_taken=1)
    def forward(*args, **kwargs):
        return SimpleNamespace(action_mean=torch.full((1, 18), .25), hidden=args[5] + 1)
    session.network = forward
    step = session.step_once()
    assert step.transition is transition and session.forward_count == 1
    assert torch.equal(session.hidden, torch.ones(1, 187))
    torch.testing.assert_close(session.runtime.step.call_args.args[0], torch.full((1, 18), .25))
    assert '"step": 1' in session._metrics.getvalue()
    assert session.manual_reset() == {"reset": True}
    assert not bool(session.hidden.any()) and session.forward_count == 1
    assert session.runtime.control_steps_taken == 1  # Reset adds no extra warmup.
    torch.testing.assert_close(session.runtime.reset.call_args.args[0], torch.tensor([0]))


def test_native_tripod_command_keys_use_legacy_limits():
    viewer = NativePlayViewer.__new__(NativePlayViewer)
    import threading
    viewer._command_lock = threading.Lock()
    viewer._command = (2.4, 0., 1.1)
    viewer.command_limits = TripodPlaySession.COMMAND_LIMITS
    viewer.pending = PendingControls()
    viewer.selected_env_row = 2
    import mujoco
    native_display_keys = {ord(shortcut) for _, _, shortcut in
                           (*mujoco.mjVISSTRING, *mujoco.mjRNDSTRING) if shortcut}
    assert set(viewer._COMMAND_KEYS).isdisjoint(native_display_keys)
    # Native display shortcuts must no longer enqueue any project controls.
    for key in (glfw.KEY_W, glfw.KEY_S, glfw.KEY_A, glfw.KEY_D, glfw.KEY_J,
                glfw.KEY_L, glfw.KEY_X, glfw.KEY_Q, glfw.KEY_COMMA, glfw.KEY_PERIOD):
        viewer._on_key(key)
    assert not any(value is not None and value is not False for value in viewer.pending.take().values())
    assert viewer._command == (2.4, 0., 1.1)
    viewer._on_key(glfw.KEY_UP)
    viewer._on_key(glfw.KEY_HOME)
    command = viewer.pending.take()["command"]
    assert command[0] == 2.5 and abs(command[2] - 1.2) < 1e-7
    for key, expected in ((glfw.KEY_DOWN, (2.4, 0., 1.2)),
                          (glfw.KEY_LEFT, (2.4, .1, 1.2)),
                          (glfw.KEY_RIGHT, (2.4, 0., 1.2)),
                          (glfw.KEY_END, (2.4, 0., 1.1))):
        viewer._on_key(key)
        torch.testing.assert_close(torch.tensor(viewer.pending.take()["command"]), torch.tensor(expected))
    for key, field, expected in ((glfw.KEY_SPACE, "pause_toggle", True),
                                 (glfw.KEY_F8, "step", True),
                                 (glfw.KEY_ENTER, "reset", True),
                                 (glfw.KEY_DELETE, "zero_command", True),
                                 (glfw.KEY_F9, "env_row", 1),
                                 (glfw.KEY_F10, "env_row", 3),
                                 (glfw.KEY_F12, "exit", True)):
        viewer._on_key(key)
        assert viewer.pending.take()[field] == expected


def test_native_close_waits_for_renderer_to_release_model():
    viewer = NativePlayViewer.__new__(NativePlayViewer)
    viewer.runtime = object()
    handle = Mock()
    from unittest.mock import PropertyMock
    type(handle).m = PropertyMock(side_effect=[object(), None])
    viewer.handle = handle
    with patch("hexapod_mjlab.tasks.confined_locomotion_2.visualization.native.time.sleep") as sleep:
        viewer.close()
    handle.close.assert_called_once()
    sleep.assert_called_once_with(.01)
    assert viewer.handle is None and viewer.runtime is None
