"""Passive native rendering; the PlaySession alone advances physics and policy.

Uses the same MuJoCo window and model-sync helpers as MJLab's native viewer,
without its independent obs/policy/env.step loop or mouse-force feedback.
"""

from copy import copy
import hashlib
import threading
import time

import glfw
import mujoco
import numpy as np

from mjlab.viewer.model_sync import disable_model_sameframe_shortcuts, sync_model_fields, VIEWER_MODEL_FIELDS
from hexapod_mjlab.common.checkpoint import model_physics_identity
from hexapod_mjlab.common.visualization import PendingControls


def require_native_display():
    import os
    import sys
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise RuntimeError("Native Play requires a desktop display; use --no-viewer or a debug_* entry with --no-native")


def display_model_identity(model):
    """Check mapping, physical geometry and appearance once per env rebuild."""
    visual = hashlib.sha256()
    for name in ("geom_matid", "geom_rgba", "mat_rgba", "mat_texid", "mesh_normal", "mesh_texcoord", "tex_data"):
        value = getattr(model, name)
        visual.update(str((name, value.shape, value.dtype)).encode())
        visual.update(value.tobytes())
    return (model.nq, model.nv, model.nu, model.nmocap, bytes(model.names),
            model_physics_identity(model), visual.hexdigest())


class NativePlayViewer:
    """Persistent CPU display buffers, detached from rebuildable GPU sessions."""

    KEY_HELP = (
        "Space pause/run | F8 step | Enter reset | F12 exit\n"
        "Up/Down vx | Left/Right vy | Home/End yaw | Delete zero\n"
        "F9/F10 previous/next env"
    )
    # Leave MuJoCo's rendering shortcuts intact. Its Python key callback is
    # invoked after native key handling and cannot consume the event.
    _COMMAND_KEYS = {
        glfw.KEY_UP: (0, .1), glfw.KEY_DOWN: (0, -.1),
        glfw.KEY_LEFT: (1, .1), glfw.KEY_RIGHT: (1, -.1),
        glfw.KEY_HOME: (2, .1), glfw.KEY_END: (2, -.1),
    }

    def __init__(self, runtime, *, command=(0, 0, 0), command_limits=(1.5, 1., 1.)):
        import mujoco.viewer

        self.pending = PendingControls()
        self._command_lock = threading.Lock()
        self._command = tuple(command)
        self.command_limits = tuple(command_limits)
        self.selected_env_row = 0
        self.runtime = None
        self.handle = None
        self._status = None
        source = runtime.env.sim.mj_model
        self._identity = display_model_identity(source)
        self.model = copy(source)
        self.data = mujoco.MjData(self.model)
        disable_model_sameframe_shortcuts(self.model)
        self.rebind_session(runtime)
        try:
            self.handle = mujoco.viewer.launch_passive(
                self.model, self.data, key_callback=self._on_key,
                show_left_ui=False, show_right_ui=False)
            cfg = runtime.env.cfg.viewer
            with self.handle.lock():
                self.handle.cam.distance = cfg.distance
                self.handle.cam.azimuth = cfg.azimuth
                self.handle.cam.elevation = cfg.elevation
                free = np.flatnonzero(self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
                if len(free):
                    self.handle.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                    self.handle.cam.trackbodyid = int(self.model.jnt_bodyid[free[0]])
            self.sync(paused=True)
        except BaseException:
            self.close()
            raise

    def rebind_session(self, runtime):
        if display_model_identity(runtime.env.sim.mj_model) != self._identity:
            raise ValueError("native display model differs after acquisition rebuild; open a new Play process")
        self.runtime = runtime
        self.selected_env_row = min(self.selected_env_row, runtime.num_envs - 1)
        self._model_row = None
        self._status = None

    def detach_session(self):
        # Keep the window and its independent CPU buffers; release GPU owners.
        self.runtime = None

    def set_command_inputs(self, values):
        with self._command_lock:
            self._command = tuple(float(v) for v in values)

    def _on_key(self, key):
        # MuJoCo invokes this on its GUI thread. Only enqueue owner-loop intents.
        action = {glfw.KEY_SPACE: self.pending.submit_pause_toggle,
                  glfw.KEY_F8: self.pending.submit_step,
                  glfw.KEY_ENTER: self.pending.submit_reset,
                  glfw.KEY_F12: self.pending.submit_exit,
                  glfw.KEY_DELETE: self.pending.submit_zero_command}.get(key)
        if action is not None:
            action()
        elif key in (glfw.KEY_F9, glfw.KEY_F10):
            self.pending.submit_env_row(self.selected_env_row + (-1 if key == glfw.KEY_F9 else 1))
        elif key in self._COMMAND_KEYS:
            axis, delta = self._COMMAND_KEYS[key]
            with self._command_lock:
                values = list(self._command)
                limit = self.command_limits[axis]
                values[axis] = float(np.clip(values[axis] + delta, -limit, limit))
                self._command = tuple(values)
                self.pending.submit_command(*self._command)

    def is_running(self):
        return self.handle is not None and self.handle.is_running()

    def sync(self, *, paused):
        """Read post-control qpos/qvel directly; never build a debug snapshot."""
        if self.runtime is None or not self.is_running():
            return
        sim = self.runtime.env.sim
        row = self.selected_env_row
        # Stage small copies before taking the render lock. qpos contains Root
        # position/quaternion and all joint positions, including the final pose.
        qpos = sim.data.qpos[row].detach().cpu().numpy()
        qvel = sim.data.qvel[row].detach().cpu().numpy()
        with self.handle.lock():
            if self._model_row != row:
                sync_model_fields(self.model, sim.model, sim.expanded_fields & VIEWER_MODEL_FIELDS, row)
                self._model_row = row
            self.data.qpos[:] = qpos
            self.data.qvel[:] = qvel
            if self.model.nmocap:
                self.data.mocap_pos[:] = sim.data.mocap_pos[row].cpu().numpy()
                self.data.mocap_quat[:] = sim.data.mocap_quat[row].cpu().numpy()
            self.data.time = self.runtime.control_time_s()
            mujoco.mj_forward(self.model, self.data)
        self.set_status(paused=paused)
        self.handle.sync(state_only=True)

    def set_status(self, *, paused):
        """Update the overlay without reading possibly faulted GPU state."""
        if not self.is_running():
            return
        faulted = bool(getattr(self.runtime, "_faulted", False))
        stopped = bool(getattr(self.runtime, "_stopped", False))
        row = self.selected_env_row
        status = (paused, faulted, stopped, row, self._command)
        if status != self._status:
            state = "FAULTED" if faulted else "STOPPED" if stopped else "PAUSED" if paused else "RUNNING"
            self.handle.set_texts((
                mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                "Play\nEnv\nCommand\nKeys",
                f"{state}\n{row}\n{np.round(self._command, 2)}\n" + self.KEY_HELP))
            self._status = status

    def close(self):
        self.detach_session()
        handle, self.handle = self.handle, None
        if handle is not None:
            handle.close()
            # close() only requests exit. Let MuJoCo's daemon render thread
            # release its model before Python's atexit handler terminates GLFW.
            deadline = time.monotonic() + 5.
            while handle.m is not None:
                if time.monotonic() >= deadline:
                    raise RuntimeError("MuJoCo viewer did not finish closing within 5 seconds")
                time.sleep(.01)
