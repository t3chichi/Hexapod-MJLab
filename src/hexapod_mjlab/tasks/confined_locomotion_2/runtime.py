"""Runtime for real flat-task transitions, sensing, and episode lifecycle.

``flat_env_cfg.make_training_env_cfg`` builds an INDEPENDENT training configuration from
the registered flat config (the registered defaults are untouched): formal
reward/termination callbacks bound to the P0 ``TransitionContext``, Null
commands (the runtime owns the command streams), ``auto_reset=False`` and a
``nan_policy='error'`` observation group. The 63-dim manager group stays
the official transport/shape-probe surface only — construction probes build
no lidar and draw no random numbers.

``TrainingRuntime`` takes over a freshly constructed, officially reset
environment (``RslRlVecEnvWrapper(env, clip_actions=None)``; the wrapper's
initial reset already happened): it binds the instance TransitionContext,
takes over the CURRENT T1 state WITHOUT resetting again, performs exactly
one initial first scan, and then owns the whole control-step lifecycle:

1. ``prepare()`` at a pre-action control identity returns the SAME cached
   ``PreparedControlInputs`` for repeated reads (no new scan, no command
   sampling); identity = (global step counter, per-env episode vector);
2. ``step(raw_action)``: begin(c_k) → ``wrapper.step(raw)`` →
   ``context.finish()`` — the real four-substep MDP truth; the actual
   reward is externally re-derived from the cached per-term raw values and
   compared with the manager buffer; terminated/truncated are INDEPENDENT
   clones of ``unwrapped.reset_terminated/reset_time_outs`` taken BEFORE
   any selected reset (never guessed from wrapper dones);
3. at the NATURAL next boundary the runtime advances the command streams
   and the due 10 Hz scans and forms ``final_inputs`` (terminated rows
   included — the timeout bootstrap observation is the natural pre-reset
   one; no forced non-due scan);
4. mode dispatch — 'engineering' (default) stops the WHOLE run on a
   natural independent fall (root height < 0.20 m or up·world-up < 0.5)
   or on any formal done, saving the terminal evidence; NO partial reset
   ever continues a fallen natural run. 'training_lifecycle_fixture' is
   the explicit isolated mode that applies the selected partial reset for
   terminated/truncated rows and forms fresh ``next_inputs``;
   'play' is the task-play seam: the scheduler is constructed with
   ``allow_manual=True`` and ALL rows are set manual-zero BEFORE the
   takeover ``sample_reset`` (no random command is ever drawn and then
   overwritten), the play env cfg removes the ``episode_timeout``
   truncation term (the 20 s/1000-step values stay as RECORDED metadata
   and the runtime refuses any truncation flag), and a step fall decided
   ONLY by the confirmed formal gates (10-step persisted low height /
   10-step persisted tilt / immediate flip — never the engineering
   instantaneous gate) saves the stop scene and PAUSES: ``step`` refuses
   until the explicit full-episode manual reset (real T1 reset, history/
   prev-action/fall-counter cleanup, unique first scan, zeroed manual
   command), while cached read-only ``prepare``/``labels`` stay available
   for the paused display;
5. any fault in any mode saves the scene (explicitly NOT a resumable
   checkpoint) and freezes the runtime — later step/reset refuse
   (including play: a fault is never recoverable through reset).

No network, optimizer, PPO sampling or value bootstrap is built here.
Terrain Memory collection explicitly supplies ``height_grid``: only that
path uses the compiled Heightfield scanner/MDP query and training-only labels.
The default Flat path keeps its existing scanner and plane reference.
``close()`` restores the instance hooks via the context. The scan window,
near-input identities are kept per env with the episode
scan id and the PERSISTENT sensor scan count kept SEPARATE (the latter
survives resets; observation perturbations consume it).

Observation perturbations (``perturbation_cfg``; default None = the clean
path untouched): the ``ObservationPerturbations`` owner is built BEFORE
the takeover first scan (per-env gap phase drawn once at "creation"), the
gap/range consumption happens inside ``_scan_rows_now``/``pipeline.scan_rows``
on the CONSUMED copy only, and the proprio noise is applied in ``prepare`` to
the freshly built observation copy with a per-row (env, episode, step)
identity cache — repeated reads/replays never redraw and unselected rows of a
selected reset keep their noise bit-exact. A selected reset never rephases
the gap schedule or redraws a scheduled offset; its first scan just advances
the persistent count. True state, PD, rewards, terminations and labels never
read the noisy values.

Play provenance: in play mode every REAL scan additionally appends a
bounded per-env metadata entry (sampled raw ray ids, validity masks, slot
direction keys when present, the ORIGINAL frame no-hit flags read at the
sampled raw ids, and the scan identity) to the per-env ``play_source_scans``
ring — at most the current-frame metadata entry per env, cleared only for
the reset rows. This is display/provenance metadata only; it never feeds the
network.
"""

from __future__ import annotations

import copy as copylib
import json
import time
import traceback
from dataclasses import dataclass, field, fields as dataclass_fields
from pathlib import Path
from functools import wraps

import torch

from hexapod_mjlab.rl.confined_locomotion_2.contracts import (
    JOINT_DIM,
    EnvelopeLabel,
    bind_envelope_label,
)
from hexapod_mjlab.rl.confined_locomotion_2.history import ScanWindow
from hexapod_mjlab.rl.confined_locomotion_2.integration import (
    PreparedControlInputs,
    prepare_control_inputs,
)

from .commands import TrainingCommandScheduler
from .flat_env_cfg import DECIMATION, SIM_TIMESTEP_S
from hexapod_mjlab.common.mdp.rewards import (
    CONTROL_DT_S,
    TRAINING_REWARD_WEIGHTS,
    FootAirTimeState,
    get_foot_air_state,
    root_height_above_plane,
    read_current_foot_support,
)
from hexapod_mjlab.common.mdp.terminations import get_termination_state
from .observation import ControlInputCache, build_control_observation, clone_prepared
from hexapod_mjlab.common.perturbations import (
    ObservationPerturbationCfg,
    ObservationPerturbations,
)
from hexapod_mjlab.common.transition import (
    PERSISTENT_PHYSICS_FIELDS,
    TransitionContext,
    require_finite,
)
from .scanner import ConfinedLidarCfg, ConfinedScanPipeline
from .sampling import sampling_identity

__all__ = [
    "ENGINEERING_FALL_HEIGHT_M",
    "ENGINEERING_FALL_UP_DOT",
    "TrainingRuntime",
    "TrainingRuntimeFrozenError",
    "TrainingTransition",
]

#: Engineering (development-stop) instantaneous fall gate — deliberately
#: distinct from the formal 10-step training gates; it stops the whole run.
ENGINEERING_FALL_HEIGHT_M = 0.20
ENGINEERING_FALL_UP_DOT = 0.5

#: Confirmed cadence constants the runtime requires of the env config.
_SCAN_PERIOD_CONTROL_STEPS = 5
_EXPECTED_EPISODE_LENGTH_S = 20.0

#: The three confirmed formal fall terms (mdp.terminations). Play keeps
#: exactly these; engineering/fixture additionally require episode_timeout.
_FORMAL_FALL_TERM_NAMES = frozenset(
    {"fall_low_height", "fall_tilt_sustained", "fall_flipped"}
)

#: P7 play provenance keeps only the current REAL scan metadata per env; full raw frames are NOT
#: retained here (the pipeline keeps its own bounded audit).
_PLAY_SOURCE_SCAN_HISTORY = 1

#: Fixed flat envelope label E* (authority: 形态输出与初始化 table; Δz = 0,
#: NOT 0.46 — 0.46 is the world-vertical height-reward target).
_FLAT_ENVELOPE_LABEL_VALUES = (0.59, 0.69, 0.59, 0.89, -0.89, 0.0)

#: Bounded retained transition audit list (memory safety).
_MAX_RETAINED_TRANSITIONS = 1  # rollout ownership belongs to P2, not a second trajectory store
_MAX_RETAINED_SCAN_LOG = 128

#: Bound on retained raw scan audit records inside the pipeline (None keeps
#: the full P0-style audit; the default keeps the most recent scans only —
#: the raw 86,400-ray frames are diagnostic, not rollout state).
_DEFAULT_SCAN_AUDIT_BOUND_PER_ENV = 2

_MODES = ("engineering", "training_lifecycle_fixture", "training", "play")

# These two MuJoCo-Warp fields extend PERSISTENT_PHYSICS_FIELDS in the full
# checkpoint manifest. Keep them here so partial reset protects the same
# next-integration state without importing checkpoint.py into this runtime.
_EXTRA_PERSISTENT_PHYSICS_FIELDS = ("history", "plugin_state")


def _reset_env_preserving_unselected_physics(env, env_ids: torch.Tensor):
    """Run the real env reset while preserving other rows' persistent physics.

    mjlab's selected reset still calls a batch-wide ``sim.forward()`` and
    ``sim.sense()``. MuJoCo may rewrite persistent solver state such as
    ``qacc_warmstart`` during that forward, so preserve the unselected rows of
    every non-empty checkpoint physics field around the ordinary reset. The
    selected rows still receive the real reset, forward, sensing and manager
    updates. We deliberately do not forward again after restoring warmstart:
    that would overwrite the value we are restoring.
    """
    num_envs = int(env.num_envs)
    selected_rows = env_ids.detach().to(dtype=torch.long)
    selected_mask = torch.ones(
        num_envs, dtype=torch.bool, device=selected_rows.device
    )
    selected_mask[selected_rows] = False
    unselected_rows = torch.nonzero(selected_mask, as_tuple=False).flatten()
    if unselected_rows.numel() == 0:
        # Whole-batch resets need no snapshots or post-reset copies.
        return env.reset(env_ids=env_ids)

    import warp as wp

    data = env.sim.wp_data
    required_fields = set(PERSISTENT_PHYSICS_FIELDS)
    field_names = (*PERSISTENT_PHYSICS_FIELDS, *_EXTRA_PERSISTENT_PHYSICS_FIELDS)
    snapshots = []
    for name in field_names:
        array = getattr(data, name, None)
        if array is None:
            if name in required_fields:
                raise RuntimeError(
                    f"backend persistent physics field absent during partial reset: {name}"
                )
            continue
        if not array.size:
            continue
        tensor = wp.to_torch(array)
        if tensor.ndim == 0 or tensor.shape[0] != num_envs:
            raise RuntimeError(
                f"persistent physics field {name} has no env-row axis: "
                f"shape={tuple(tensor.shape)}, num_envs={num_envs}"
            )
        field_rows = unselected_rows.to(device=tensor.device)
        snapshots.append(
            (tensor, field_rows, tensor.index_select(0, field_rows).clone())
        )

    try:
        return env.reset(env_ids=env_ids)
    finally:
        # Copy into the existing Warp-backed tensors to retain their storage
        # identity (including any captured CUDA-graph references).
        with torch.no_grad():
            for tensor, rows, saved in snapshots:
                tensor.index_copy_(0, rows, saved)


def _freeze_failures(phase):
    """Public lifecycle faults save and freeze, including read/reset failures."""
    def decorate(fn):
        @wraps(fn)
        def call(self, *args, **kwargs):
            self._guard_usable(phase)
            try:
                return fn(self, *args, **kwargs)
            except BaseException as err:
                if not self._faulted:
                    self._save_fault(err, phase=phase)
                raise
        return call
    return decorate


class TrainingRuntimeFrozenError(RuntimeError):
    """The runtime froze after a fault or finished an engineering stop."""


@dataclass(frozen=True)
class TrainingTransition:
    """Named real transition record (all tensors independent clones).

    ``input`` is the o_k snapshot this step consumed; ``command`` is c_k
    (the OLD command that scored this transition, never c_(k+1));
    ``reward`` is the ACTUAL manager reward; ``reward_raw_terms`` the raw
    per-term values; ``terminated``/``truncated`` the independent pre-reset
    flag clones; ``final_inputs``/``final_labels`` the NATURAL boundary
    o_(k+1) (terminated rows included); ``next_inputs``/``next_labels`` the
    post-reset inputs (identical values to final on non-reset rows);
    ``context_record`` the sealed TransitionContext record of this step.
    """

    control_id: int
    env_ids: torch.Tensor
    episode_ids: torch.Tensor
    step_ids: torch.Tensor
    time_s: float
    input: PreparedControlInputs
    command: torch.Tensor
    reward: torch.Tensor
    reward_raw_terms: dict
    reward_weights: dict
    reward_expected: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    final_inputs: PreparedControlInputs
    final_labels: dict
    reset_env_ids: torch.Tensor | None
    next_inputs: PreparedControlInputs
    next_labels: dict | None
    context_record: dict
    mode: str
    engineering_stopped: bool = False
    scan_writes: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


class TrainingRuntime:
    """Owns the training lifecycle of one officially-reset wrapped env."""

    def __init__(
        self,
        wrapper,
        *,
        seed: int,
        output_dir,
        mode: str = "engineering",
        lidar_cfg: ConfinedLidarCfg | None = None,
        scan_audit_bound_per_env: int | None = _DEFAULT_SCAN_AUDIT_BOUND_PER_ENV,
        perturbation_cfg: ObservationPerturbationCfg | None = None,
        height_grid=None,
    ) -> None:
        from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper

        if not isinstance(wrapper, RslRlVecEnvWrapper):
            raise TypeError("wrapper must be an RslRlVecEnvWrapper (official reset done)")
        if wrapper.clip_actions is not None:
            raise ValueError(
                "the training seam requires RslRlVecEnvWrapper(clip_actions=None) "
                "(raw actions are clipped by the action term itself)"
            )
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES!r}, got {mode!r}")
        if perturbation_cfg is not None and not isinstance(
            perturbation_cfg, ObservationPerturbationCfg
        ):
            raise TypeError(
                "perturbation_cfg must be an ObservationPerturbationCfg or None "
                "(None = the clean P0–P5 path; never a generic noise switch)"
            )
        env = wrapper.unwrapped
        cfg = env.cfg
        if cfg.auto_reset is not False or int(cfg.decimation) != DECIMATION:
            raise RuntimeError("training runtime requires auto_reset=False and decimation=4")
        if abs(float(cfg.sim.mujoco.timestep) - SIM_TIMESTEP_S) > 1e-12:
            raise RuntimeError("unexpected physics timestep")
        if abs(float(env.step_dt) - CONTROL_DT_S) > 1e-12:
            raise RuntimeError("control step dt != 0.02 s")
        if abs(float(cfg.episode_length_s) - _EXPECTED_EPISODE_LENGTH_S) > 1e-9 or int(
            env.max_episode_length
        ) != 1000:
            raise RuntimeError("confirmed 20 s / 1000-control-step timeout required")
        # P7 play seam: play runs WITHOUT the 20 s timeout (the dedicated play
        # env-cfg factory removes that ONE term from its own independent
        # object; the 20 s/1000-step values above stay as RECORDED metadata).
        # Engineering/fixture/formal training keep the confirmed four-term set. Any other term
        # set is a config fault, never a silent behavior change.
        term_names = set(cfg.terminations)
        if mode == "play":
            if term_names != _FORMAL_FALL_TERM_NAMES:
                raise RuntimeError(
                    "play mode requires exactly the three formal fall terms "
                    f"{sorted(_FORMAL_FALL_TERM_NAMES)} (the episode_timeout "
                    "truncation is removed by the dedicated play env-cfg "
                    f"factory); got {sorted(term_names)}"
                )
            for name, term in cfg.terminations.items():
                if bool(getattr(term, "time_out", False)):
                    raise RuntimeError(
                        f"play mode forbids time_out terms ({name!r}); the 20 s "
                        "truncation must never fire in play"
                    )
        elif term_names != _FORMAL_FALL_TERM_NAMES | {"episode_timeout"}:
            raise RuntimeError(
                "engineering/fixture/training modes require the confirmed four-term "
                f"termination set {sorted(_FORMAL_FALL_TERM_NAMES | {'episode_timeout'})}; "
                f"got {sorted(term_names)}"
            )
        if int(env.common_step_counter) != 0 or bool(env.episode_length_buf.count_nonzero()):
            raise RuntimeError(
                "training runtime must take over the fresh post-initial-reset state "
                "(common_step_counter == 0, episode_length_buf all zero); construct "
                "the wrapper immediately before the runtime and do not step in between"
            )
        if getattr(env, "_confined_transition_context", None) is not None:
            raise RuntimeError("a transition context is already bound to this env")

        self.wrapper = wrapper
        self.env = env
        self.device = env.device
        self.num_envs = int(env.num_envs)
        self.mode = mode
        self.seed = int(seed)
        self.lidar_cfg = lidar_cfg or ConfinedLidarCfg()
        self.scan_audit_bound = (
            None
            if scan_audit_bound_per_env is None
            else int(scan_audit_bound_per_env) * self.num_envs
        )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=False)
        # P6: the perturbation owner is built BEFORE the takeover first scan
        # (the per-env gap phase is drawn exactly once at this "creation"
        # point; the first scan is persistent scan number 1). None keeps the
        # clean path untouched — no streams, no draws, no extra copies. An
        # all-None cfg normalizes to clean so the checkpoint identity and the
        # window key-presence parity stay consistent.
        self.perturbation_cfg = (
            None
            if perturbation_cfg is None or perturbation_cfg.is_clean
            else perturbation_cfg
        )
        self._perturbations = (
            None
            if self.perturbation_cfg is None
            else ObservationPerturbations(
                self.num_envs,
                cfg=self.perturbation_cfg,
                seed=self.seed,
                device=self.device,
            )
        )
        # ── bind the P0 seam and take over the current T1 state ────────────
        self._control_steps = int(env.common_step_counter)  # == 0

        self._commands = TrainingCommandScheduler(
            self.num_envs,
            seed=self.seed,
            device=self.device,
            allow_manual=(mode == "play"),
        )
        #: P7 play per-env provenance rings (None outside play mode).
        self._play_source_scans: dict[int, list] | None = (
            {row: [] for row in range(self.num_envs)} if mode == "play" else None
        )
        if mode == "play":
            # Confirmed play command contract: ALL rows are manual-zero
            # BEFORE the takeover sample_reset below — no random command is
            # ever sampled first and overwritten. The manual flags survive
            # resets; a reset zeroes the value without consuming the stream.
            self._commands.set_command(
                torch.arange(self.num_envs, dtype=torch.long, device=self.device),
                torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device),
            )
        self.episode_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.step_in_episode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        #: episode scan ids (reset to 0 on selected reset; first scan = 1).
        self.scan_counters = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        #: PERSISTENT sensor scan count (never reset; P6 will consume it).
        self.sensor_scan_counts = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.next_due_scan_step = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )

        self.window = ScanWindow(
            self.num_envs, episode_ids=[0] * self.num_envs, device=self.device
        )
        self.height_grid = height_grid
        self.terrain_query = None
        self._height_label = None
        if height_grid is not None:
            from .terrain_geometry import HeightGridCfg, TerrainGeometryQuery
            if not isinstance(height_grid, HeightGridCfg):
                raise TypeError("height_grid must be HeightGridCfg or None")
            self.terrain_query = TerrainGeometryQuery(env)
        self.pipeline = (ConfinedScanPipeline(env, self.lidar_cfg) if self.terrain_query is None else
                         ConfinedScanPipeline(env, self.lidar_cfg, terrain_query=self.terrain_query))
        self.scan_identity = copylib.deepcopy(self.pipeline.scan_identity)
        (self.output_dir / "runtime_recipe.json").write_text(
            json.dumps(
                {
                    "task": ("El-4090-Confined-Locomotion-Flat_2/TrainingRuntime" if height_grid is None else
                             "El-4090-Confined-Locomotion-Terrain-Memory_2/CollectionRuntime"),
                    "mode": self.mode,
                    "seed": self.seed,
                    "num_envs": self.num_envs,
                    "device": str(self.device),
                    "sampling": sampling_identity(self.lidar_cfg.far_sampling),
                    "lidar_rays": self.pipeline.num_raw_rays,
                    "scan_identity": self.scan_identity,
                    **({} if height_grid is None else {"height_grid": height_grid.identity()}),
                    "observation_perturbations": (
                        None if perturbation_cfg is None
                        else perturbation_cfg.identity()
                    ),
                    "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "optimizer": False,
                    "checkpoint": "none (runtime saves fault/stop evidence only)",
                },
                indent=2,
            )
        )


        self._input_cache = ControlInputCache()
        self._current_window_snapshot = None
        self.transitions: list[TrainingTransition] = []
        self.scan_log: list[dict] = []

        self._closed = False
        self._faulted = False
        self._stopped = False
        self.last_fault: dict | None = None
        self.stop_record: dict | None = None
        self._evidence_count = 0

        # Episode-0 takeover: reset-time command sampling, exactly one first
        # scan, REAL foot-support initialization (no landing events), zero
        # termination counters. No env reset happens here — the wrapper's
        # initial reset already wrote the T1 state.
        self._context = (TransitionContext(env) if self.terrain_query is None else
                         TransitionContext(env, terrain_query=self.terrain_query))
        try:
            self._initialize_rows(
                torch.arange(self.num_envs, dtype=torch.long, device=self.device),
                reason="takeover",
            )
        except BaseException as err:
            self._save_fault(err, phase="initialize")
            self.close()
            raise

    # ── identities / clocks ────────────────────────────────────────────────

    @property
    def control_steps_taken(self) -> int:
        return self._control_steps

    def control_time_s(self) -> float:
        """Global monotonic control time t(k) = k · 0.02 s (NOT env sim time)."""
        return self._control_steps * CONTROL_DT_S

    def state_dict(self) -> dict:
        """Return task-owned causal runtime state at an idle boundary.

        Environment physics and manager tensors are checkpointed by the outer
        runner. This payload owns the remaining task clocks, scan window,
        command/perturbation streams, identity-keyed current input cache, and
        lifecycle counters. It contains no ground-label provider/cache.
        """
        self._guard_usable("state_dict")
        if self._context.phase != "idle":
            raise RuntimeError("runtime state can only be saved at an idle control boundary")
        cache_prepared = self._input_cache.get()
        foot = get_foot_air_state(self.env)
        termination = get_termination_state(self.env)
        clone = lambda value: value.detach().clone()
        return {
            "schema": "confined_locomotion_2.runtime/v1",
            "scan_identity": copylib.deepcopy(self.scan_identity),
            "sampling": sampling_identity(self.lidar_cfg.far_sampling),
            "seed": self.seed,
            "num_envs": self.num_envs,
            "mode": self.mode,
            **({} if self.height_grid is None else {"height_grid": self.height_grid.identity()}),
            "control_steps": int(self._control_steps),
            "episode_ids": clone(self.episode_ids),
            "step_in_episode": clone(self.step_in_episode),
            "scan_counters": clone(self.scan_counters),
            "sensor_scan_counts": clone(self.sensor_scan_counts),
            "next_due_scan_step": clone(self.next_due_scan_step),
            "window": self.window.state_dict(),
            "commands": self._commands.state_dict(),
            "perturbations": (
                None if self._perturbations is None else self._perturbations.state_dict()
            ),
            "input_cache": {
                "identity": self._input_cache._identity,
                "episodes": (
                    None if self._input_cache._episodes is None
                    else clone(self._input_cache._episodes)
                ),
                "prepared": (
                    None if cache_prepared is None else clone_prepared(cache_prepared)
                ),
                "window_snapshot": copylib.deepcopy(self._current_window_snapshot),
            },
            "foot_air": {
                "air_time_s": clone(foot.air_time_s),
                "last_support": clone(foot.last_support),
                "control_id": foot.control_id,
                "landing": None if foot.landing is None else clone(foot.landing),
                "consumed_air_s": (
                    None if foot.consumed_air_s is None else clone(foot.consumed_air_s)
                ),
            },
            "termination": {
                name: clone(getattr(termination, name))
                for name in (
                    "low_height_count", "tilt_count", "low_height_done",
                    "tilt_done", "flip_done", "any_fall",
                )
            } | {"control_id": termination.control_id},
            "stopped": bool(self._stopped),
            "play_source_scans": copylib.deepcopy(self._play_source_scans),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore runtime-owned state without resetting, stepping or scanning."""
        self._guard_usable("load_state_dict")
        if self._context.phase != "idle":
            raise RuntimeError("runtime state can only be loaded at an idle boundary")
        if not isinstance(state, dict) or state.get("schema") != "confined_locomotion_2.runtime/v1":
            raise RuntimeError("unsupported confined_locomotion_2 runtime state schema")
        if state.get("seed") != self.seed or state.get("num_envs") != self.num_envs:
            raise RuntimeError("runtime checkpoint seed/environment identity mismatch")
        if state.get("scan_identity") != self.scan_identity:
            raise RuntimeError("runtime checkpoint scan identity mismatch")
        if state.get("sampling") != sampling_identity(self.lidar_cfg.far_sampling):
            raise RuntimeError("runtime checkpoint sampling identity mismatch")
        if state.get("mode") != self.mode:
            raise RuntimeError("runtime checkpoint mode mismatch")
        if state.get("height_grid") != (None if self.height_grid is None else self.height_grid.identity()):
            raise RuntimeError("runtime checkpoint height grid mismatch")
        if (state.get("perturbations") is None) != (self._perturbations is None):
            raise RuntimeError("runtime checkpoint perturbation mode mismatch")

        def copy_tensor(name: str, target: torch.Tensor) -> None:
            value = state[name]
            if not isinstance(value, torch.Tensor) or value.shape != target.shape or value.dtype != target.dtype:
                raise RuntimeError(f"runtime state field {name!r} shape/dtype mismatch")
            target.copy_(value.to(device=target.device))

        self._control_steps = int(state["control_steps"])
        for name in (
            "episode_ids", "step_in_episode", "scan_counters",
            "sensor_scan_counts", "next_due_scan_step",
        ):
            copy_tensor(name, getattr(self, name))
        self.window.load_state_dict(state["window"])
        self._commands.load_state_dict(state["commands"])
        if self._perturbations is not None:
            self._perturbations.load_state_dict(state["perturbations"])

        cache = state["input_cache"]
        self._input_cache._identity = cache["identity"]
        self._input_cache._episodes = (
            None if cache["episodes"] is None
            else cache["episodes"].to(device=self.device).clone()
        )
        self._input_cache._prepared = (
            None if cache["prepared"] is None
            else self._move_dataclass_tensors(cache["prepared"], self.device)
        )
        self._current_window_snapshot = (
            None if cache["window_snapshot"] is None
            else self._move_dataclass_tensors(cache["window_snapshot"], self.device)
        )

        foot = get_foot_air_state(self.env)
        foot_state = state["foot_air"]
        for name in ("air_time_s", "last_support", "landing", "consumed_air_s"):
            value = foot_state[name]
            if value is None:
                setattr(foot, name, None)
            else:
                target = getattr(foot, name)
                if target is None:
                    setattr(foot, name, value.to(device=self.device).clone())
                else:
                    target.copy_(value.to(device=target.device))
        foot.control_id = foot_state["control_id"]
        termination = get_termination_state(self.env)
        term_state = state["termination"]
        for name in (
            "low_height_count", "tilt_count", "low_height_done",
            "tilt_done", "flip_done", "any_fall",
        ):
            getattr(termination, name).copy_(term_state[name].to(device=self.device))
        termination.control_id = term_state["control_id"]
        self._stopped = bool(state["stopped"])
        self._play_source_scans = copylib.deepcopy(state["play_source_scans"])
        self._height_label = None  # geometry labels are regenerated from restored physics

    @staticmethod
    def _move_dataclass_tensors(value, device):
        """Rebuild a frozen dataclass with tensor fields moved to the live device."""
        if not hasattr(value, "__dataclass_fields__"):
            raise TypeError(f"expected a dataclass checkpoint value, got {type(value).__name__}")
        kwargs = {}
        for item in dataclass_fields(value):
            field_value = getattr(value, item.name)
            if torch.is_tensor(field_value):
                kwargs[item.name] = field_value.to(device=device).clone()
            elif hasattr(field_value, "__dataclass_fields__"):
                kwargs[item.name] = TrainingRuntime._move_dataclass_tensors(
                    field_value, device
                )
            else:
                kwargs[item.name] = copylib.deepcopy(field_value)
        return type(value)(**kwargs)

    def _assert_counter(self) -> int:
        actual = int(self.env.common_step_counter)
        if actual != self._control_steps:
            raise RuntimeError(
                f"env.common_step_counter ({actual}) desynced from the runtime "
                f"clock ({self._control_steps}); the env was stepped outside the "
                "training runtime"
            )
        return actual

    # ── inputs and labels (repeatable reads at one identity) ───────────────

    @_freeze_failures("prepare")
    def prepare(self) -> PreparedControlInputs:
        """Cached prepared inputs at the CURRENT control identity.

        Repeated calls at the same (counter, episodes) return the SAME
        object — no new scan, no command sampling, no new clones. A changed
        identity (a completed step, a selected reset) rebuilds once.

        P6: when the proprio perturbation is active, the freshly built clean
        observation copy gets its per-row cached/fresh noise HERE (one draw
        per (env, episode, step) identity; unselected rows after a selected
        reset reuse their cached noise bit-exactly); when the lidar
        perturbations are active, the window-carried direction angle keys are
        consumed instead of recomputing keys from the stored (possibly noisy)
        points. With no perturbations the path is exactly the clean one.
        """
        self._guard_usable("prepare")
        self._assert_counter()
        if self._input_cache.matches(self._control_steps, self.episode_ids):
            cached = self._input_cache.get()
            assert cached is not None
            return cached
        observation = build_control_observation(
            self.env,
            episode_ids=self.episode_ids,
            step_ids=self.step_in_episode,
            control_time_s=self.control_time_s(),
            command=self._commands.current_command(),
        )
        if self._perturbations is not None and self._perturbations.proprio_enabled:
            # Mutates the observation COPY only (build_control_observation
            # returned fresh clones of the true state); the true state, the
            # transition context, rewards and terminations never read it.
            self._perturbations.apply_proprio_noise(
                observation, control_step=self._control_steps
            )
        snapshot = self.window.read()
        angle_keys = self.window.read_angle_keys()
        source_directions = self.window.read_source_directions()
        if angle_keys is not None:
            near_keys = angle_keys.near
            far_keys = angle_keys.far.reshape(self.num_envs, -1)
            if not bool(angle_keys.present.all()):
                # Legacy key-free rows retain their old geometric key path.
                near_keys = torch.where(angle_keys.present[:, None], near_keys,
                    self.pipeline.angle_keys(snapshot.near_points))
                far_keys = torch.where(angle_keys.present[:, None], far_keys,
                    self.pipeline.angle_keys(snapshot.far_points.reshape(self.num_envs, -1, 3)))
        else:
            near_keys = self.pipeline.angle_keys(snapshot.near_points)
            far_keys = self.pipeline.angle_keys(
                snapshot.far_points.reshape(self.num_envs, -1, 3)
            )
        prepared = prepare_control_inputs(
            window=snapshot,
            observation=observation,
            near_angle_keys=near_keys,
            far_angle_keys=far_keys,
            near_source_dirs_sensor=(
                None if source_directions is None else source_directions.near
            ),
            far_source_dirs_sensor=(
                None if source_directions is None
                else source_directions.far.reshape(self.num_envs, -1, 3)
            ),
            near_source_dirs_present=(
                None if source_directions is None else source_directions.near_present
            ),
            far_source_dirs_present=(
                None if source_directions is None
                else source_directions.far_present.reshape(self.num_envs, -1)
            ),
            near_source_indices=(
                None if source_directions is None else source_directions.near_indices
            ),
            far_source_indices=(
                None if source_directions is None
                else source_directions.far_indices.reshape(self.num_envs, -1)
            ),
        )
        self._current_window_snapshot = snapshot
        return self._input_cache.store(self._control_steps, self.episode_ids, prepared)

    @_freeze_failures("labels")
    def labels(self) -> dict:
        """Current fixed envelope and optional cached, training-only height labels."""
        self._guard_usable("labels")
        prepared = self.prepare()
        observation = prepared.observation
        batch = self.num_envs
        values = torch.tensor(
            _FLAT_ENVELOPE_LABEL_VALUES, dtype=torch.float32, device=self.device
        )
        envelope = EnvelopeLabel(
            env_ids=observation.env_ids.clone(),
            episode_ids=observation.episode_ids.clone(),
            step_ids=observation.step_ids.clone(),
            values=values.view(1, 6).expand(batch, 6).clone(),
            valid=torch.ones(batch, dtype=torch.bool, device=self.device),
        )
        bind_envelope_label(observation, envelope)
        labels = {"envelope": envelope}
        if self.terrain_query is not None:
            cached = self._height_label
            if cached is None or not (torch.equal(cached.episode_ids, observation.episode_ids) and
                                      torch.equal(cached.step_ids, observation.step_ids)):
                self._height_label = self.terrain_query.height_labels(
                    observation.env_ids, observation.episode_ids, observation.step_ids, self.height_grid)
            labels["height"] = self._height_label
        return labels

    # ── scanning ───────────────────────────────────────────────────────────

    def _scan_rows_now(self, rows: list[int]) -> list[dict]:
        """Raycast the given rows NOW; stamp identity and update the window.

        P6: the runtime-owned PERSISTENT per-env scan numbers (1-based, never
        cleared by episode resets — a selected reset's first scan just
        advances them) are handed to the scanner seam so the gap schedule
        advances in scan order; the window write carries the direction angle
        keys whenever the consumed frames provide them.
        """
        time_s = float(self.control_time_s())
        persistent_numbers = (
            None
            if self._perturbations is None
            else {int(row): int(self.sensor_scan_counts[int(row)]) + 1 for row in rows}
        )
        frames = self.pipeline.scan_rows(
            rows,
            perturbations=self._perturbations,
            persistent_scan_numbers=persistent_numbers,
        )
        writes = []
        for frame in frames:
            row = int(frame.env_row)
            scan_id = int(self.scan_counters[row]) + 1
            episode_id = int(self.episode_ids[row])
            persistent = int(self.sensor_scan_counts[row]) + 1
            if persistent_numbers is not None and int(persistent_numbers[row]) != persistent:
                raise RuntimeError(
                    f"persistent scan numbering changed during the scan of env "
                    f"{row} ({persistent_numbers[row]} -> {persistent})"
                )
            self.pipeline.stamp_scan_record(
                frame, scan_id=scan_id, episode_id=episode_id, time_s=time_s
            )
            self.scan_counters[row] = scan_id
            self.sensor_scan_counts[row] = persistent
            if self._play_source_scans is not None:
                self._capture_play_source_scan(
                    frame,
                    episode_id=episode_id,
                    scan_id=scan_id,
                    persistent=persistent,
                    time_s=time_s,
                )
            self.scan_log.append(
                {
                    "env": row,
                    "episode": episode_id,
                    "scan": scan_id,
                    "persistent_scan_count": persistent,
                    "control_step": self._control_steps,
                    "time_s": time_s,
                    "gap_fired": bool(frame.gap_fired),
                    "gap_event": (
                        None
                        if self._perturbations is None
                        else self._perturbations.last_event(row)
                    ),
                }
            )
            del self.scan_log[:-_MAX_RETAINED_SCAN_LOG]
            writes.append(
                {
                    "env_row": row,
                    "scan_id": scan_id,
                    "episode_id": episode_id,
                    "persistent_scan_count": persistent,
                    "time_s": time_s,
                    "near_valid_count": int(frame.near_valid.sum().item()),
                    "far_valid_count": int(frame.far_valid.sum().item()),
                    "gap_fired": bool(frame.gap_fired),
                }
            )
        if frames:
            rows_t = torch.tensor(
                [w["env_row"] for w in writes], dtype=torch.long, device=self.device
            )
            near_keys_batch = None
            far_keys_batch = None
            if frames[0].near_angle_keys is not None:
                if any(f.near_angle_keys is None for f in frames):
                    raise RuntimeError(
                        "mixed angle-key presence within one scan batch "
                        "(all frames of a batch share the same mode)"
                    )
                near_keys_batch = torch.stack([f.near_angle_keys for f in frames])
                far_keys_batch = torch.stack([f.far_angle_keys for f in frames])
            near_dirs_batch = torch.stack([
                torch.where(
                    frame.near_indices[:, None] >= 0,
                    self.pipeline.full_ray_dirs[frame.near_indices.clamp(min=0)],
                    torch.zeros_like(frame.near_points_base),
                )
                for frame in frames
            ])
            far_dirs_batch = torch.stack([
                torch.where(
                    frame.far_indices[:, None] >= 0,
                    self.pipeline.full_ray_dirs[frame.far_indices.clamp(min=0)],
                    torch.zeros_like(frame.far_points_base),
                )
                for frame in frames
            ])
            self.window.write(
                env_ids=rows_t,
                episode_ids=self.episode_ids.index_select(0, rows_t),
                scan_ids=self.scan_counters.index_select(0, rows_t),
                times=torch.tensor(
                    [w["time_s"] for w in writes], dtype=torch.float32, device=self.device
                ),
                near_points=torch.stack([f.near_points_base for f in frames]),
                near_valid=torch.stack([f.near_valid for f in frames]),
                far_points=torch.stack([f.far_points_base for f in frames]),
                far_valid=torch.stack([f.far_valid for f in frames]),
                near_angle_keys=near_keys_batch,
                far_angle_keys=far_keys_batch,
                near_source_dirs_sensor=near_dirs_batch,
                far_source_dirs_sensor=far_dirs_batch,
                near_source_indices=torch.stack([f.near_indices for f in frames]),
                far_source_indices=torch.stack([f.far_indices for f in frames]),
            )
            if self.scan_audit_bound is not None:
                overflow = len(self.pipeline.scan_records) - self.scan_audit_bound
                for _ in range(max(0, overflow)):
                    self.pipeline.scan_records.pop(0)
        return writes

    def _capture_play_source_scan(
        self, frame, *, episode_id: int, scan_id: int, persistent: int, time_s: float
    ) -> None:
        """Bounded per-env provenance of one REAL scan (play display only).

        Small CPU clones of the sampled slot metadata — source raw ray ids,
        validity masks, slot direction keys when the consumed frames carry
        them, and the ORIGINAL frame ``no_hit_mask`` read AT the sampled raw
        ids (never inferred from consumed distances: the P6 0 m/15 m
        saturations make XYZ equality ambiguous). At most
        ``_PLAY_SOURCE_SCAN_HISTORY`` entries per env are kept, in the same
        order the window's far ring holds them; the full 86,400-ray raw
        frame itself is NOT retained here.
        """
        entry = {
            "env": int(frame.env_row),
            "episode": int(episode_id),
            "scan": int(scan_id),
            "persistent_scan_count": int(persistent),
            "time_s": float(time_s),
            "near_indices": frame.near_indices.detach().cpu().clone(),
            "near_valid": frame.near_valid.detach().cpu().clone(),
            "far_indices": frame.far_indices.detach().cpu().clone(),
            "far_valid": frame.far_valid.detach().cpu().clone(),
            "near_angle_keys": (
                None if frame.near_angle_keys is None
                else frame.near_angle_keys.detach().cpu().clone()
            ),
            "far_angle_keys": (
                None if frame.far_angle_keys is None
                else frame.far_angle_keys.detach().cpu().clone()
            ),
        }
        for key, indices in (
            ("near_no_hit", frame.near_indices),
            ("far_no_hit", frame.far_indices),
        ):
            entry[key] = torch.where(
                indices >= 0,
                frame.no_hit_mask[torch.searchsorted(frame.raw_source_ids, indices.clamp(min=0)).clamp(max=frame.no_hit_mask.numel() - 1)],
                torch.zeros_like(indices, dtype=torch.bool),
            ).detach().cpu().clone()
        ring = self._play_source_scans[int(frame.env_row)]
        ring.append(entry)
        del ring[:-_PLAY_SOURCE_SCAN_HISTORY]

    def current_command(self) -> torch.Tensor:
        """(B, 3) clone of the current command values (read-only convenience)."""
        return self._commands.current_command()

    def _due_scan_rows(self) -> list[int]:
        return [
            int(row)
            for row in range(self.num_envs)
            if int(self.next_due_scan_step[row]) >= 0
            and self._control_steps >= int(self.next_due_scan_step[row])
        ]

    # ── selected reset / row initialization ────────────────────────────────

    def _initialize_rows(self, env_ids: torch.Tensor, *, reason: str) -> dict:
        """Common takeover/reset row initialization.

        For ``takeover`` the env is ALREADY in the reset state (the wrapper
        reset wrote the T1 pose); for ``reset`` the caller ran
        ``env.reset(env_ids)`` first. Clears ONLY the selected rows: window,
        scan counters, command stream (reset-time sampling),
        episode/step counters, REAL foot support (no landing events),
        termination counters; then takes the FIRST scan and re-anchors the
        10 Hz phase at the current counter + 5. The PERSISTENT sensor scan
        count and the P6 gap schedule/streams are NOT cleared — the first
        scan advances the persistent count by exactly one and the gap phase/
        scheduled offsets survive the reset untouched.
        """
        rows = [int(r) for r in env_ids.detach().cpu().tolist()]
        rows_t = torch.tensor(rows, dtype=torch.long, device=self.device)
        # REAL post-reset foot support (measured, never assumed; no events).
        support = read_current_foot_support(self._context)
        get_termination_state(self.env).reset_rows(rows_t)
        if reason == "takeover":
            state = FootAirTimeState(self.num_envs, self.device)
            state.last_support = support.clone()
            self.env._confined_foot_air_state = state
        else:
            get_foot_air_state(self.env).reset_rows(rows_t, support[rows_t])
        new_episodes = self.episode_ids.index_select(0, rows_t) + (
            1 if reason == "reset" else 0
        )
        if reason == "reset":
            # A selected reset swaps the episode identity; at takeover the
            # window is freshly constructed with episode 0 already
            # registered — resetting to the same id would be rejected.
            self.window.reset_envs(env_ids=rows_t, new_episode_ids=new_episodes)
            if self._play_source_scans is not None:
                # P7 provenance: reset clears ONLY the selected rows' rings.
                for row in rows:
                    self._play_source_scans[row].clear()
        self.episode_ids.index_copy_(0, rows_t, new_episodes)
        self.scan_counters.index_copy_(
            0, rows_t, torch.zeros(len(rows), dtype=torch.long, device=self.device)
        )
        self.step_in_episode.index_copy_(
            0, rows_t, torch.zeros(len(rows), dtype=torch.long, device=self.device)
        )
        command_records = self._commands.sample_reset(rows_t)
        writes = self._scan_rows_now(rows)
        self.next_due_scan_step.index_copy_(
            0,
            rows_t,
            torch.full(
                (len(rows),),
                self._control_steps + _SCAN_PERIOD_CONTROL_STEPS,
                dtype=torch.long,
                device=self.device,
            ),
        )
        self._input_cache.invalidate()
        return {
            "reason": reason,
            "reset_rows": rows,
            "new_episodes": new_episodes.detach().cpu().tolist(),
            "command_records": command_records,
            "scan_writes": writes,
            "initial_support": support.detach().cpu().tolist(),
        }

    def _reset_selected(self, env_ids: torch.Tensor) -> dict:
        """Real env reset of the selected rows + confirmed row init."""
        _reset_env_preserving_unselected_physics(self.env, env_ids)
        return self._initialize_rows(env_ids, reason="reset")

    @_freeze_failures("reset")
    def reset(self, env_ids) -> dict:
        """EXPLICIT selected reset (fixture mode / explicit play callers).

        Runs the real env reset events for the selected rows, performs the
        confirmed row initialization (window/envelope labels/commands/first scan,
        REAL support re-init, counter clears) and returns the record; the
        caller reads the fresh next inputs via ``prepare()``/``labels()``.
        A faulted runtime refuses in every mode (a fault is never
        recoverable through reset). Play mode additionally REQUIRES the
        full row set (a play reset is a whole-episode manual restart) and
        it is the ONLY way past a saved play fall: the pause is cleared
        here after the confirmed reset succeeded.
        """
        self._guard_usable("reset")
        if self._context.phase != "idle":
            raise RuntimeError("reset requires an idle transition context")
        if not torch.is_tensor(env_ids):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        env_ids = env_ids.detach().to(device=self.device, dtype=torch.long).clone()
        if env_ids.dim() != 1 or env_ids.numel() == 0:
            raise ValueError("env_ids must be a non-empty 1-D tensor")
        if env_ids.numel() != torch.unique(env_ids).numel():
            raise ValueError("env_ids must not repeat")
        if bool(((env_ids < 0) | (env_ids >= self.num_envs)).any()):
            raise ValueError(f"env_ids out of range [0, {self.num_envs})")
        if self.mode == "play":
            all_rows = torch.arange(
                self.num_envs, dtype=torch.long, device=self.device
            )
            if not torch.equal(env_ids, all_rows):
                raise ValueError(
                    "play resets the WHOLE episode: env_ids must select every "
                    f"row 0..{self.num_envs - 1} (manual full reset)"
                )
        self._assert_counter()
        record = self._reset_selected(env_ids)
        if self.mode == "play" and self._stopped:
            # The explicit manual reset is the ONLY path past a saved play
            # fall; the fresh episode (T1, zeroed manual command, cleared
            # history/fall counters, unique first scan) is ready above.
            self._stopped = False
        return record

    def set_manual_command(self, env_ids, values) -> dict:
        """Manual body command override (play mode only; validated FIRST).

        Ordinary input errors (out-of-range/non-finite/shape) are rejected by
        the scheduler's validate-then-write order BEFORE any mutation — they
        are plain usage errors, never physics faults, and a rejected value
        changes nothing. A successful change invalidates the current input
        cache so the next ``prepare`` rebuilds the observation with the new
        command at the SAME identity: no new scan, no command random draw,
        and the per-row proprio noise stays cached (same (env, episode,
        step) identity — P6 semantics). Refused after a saved play fall; the
        explicit full reset is the only way past a fall. A fault is permanent.
        """
        self._guard_usable("set_manual_command")
        if self.mode != "play":
            raise RuntimeError(
                "set_manual_command exists only in play mode (engineering/"
                "fixture schedulers never accept manual overrides)"
            )
        self._commands.set_command(env_ids, values)
        self._input_cache.invalidate()
        return {
            "command": self._commands.current_command().detach().cpu().tolist(),
            "manual_mask": self._commands.manual_mask().detach().cpu().tolist(),
        }

    # ── one control step ───────────────────────────────────────────────────

    @_freeze_failures("step")
    def step(self, raw_action: torch.Tensor) -> TrainingTransition:
        """One clean real transition at the current control identity."""
        self._guard_usable("step")
        if not isinstance(raw_action, torch.Tensor):
            raise TypeError("raw_action must be a torch.Tensor")
        raw_action = raw_action.detach().to(device=self.device, dtype=torch.float32)
        if tuple(raw_action.shape) != (self.num_envs, JOINT_DIM):
            raise ValueError(f"raw_action must be ({self.num_envs}, {JOINT_DIM})")
        require_finite("raw_action", raw_action)
        self._assert_counter()
        k = self._control_steps
        time_s = float(self.control_time_s())
        try:
            # 1) input o_k and labels at the pre-action identity.
            prepared = self.prepare()
            input_snapshot = clone_prepared(prepared)
            # Binding validation at the consumed input identity
            # (envelope identity must match o_k before the action).
            self.labels()
            command = prepared.observation.command.detach().clone()
            episodes_before = self.episode_ids.clone()
            steps_before = self.step_in_episode.clone()

            # 2) begin → official wrapper step (the real 4-substep MDP).
            self._context.begin(command)
            self.wrapper.step(raw_action)
            if int(self.env.common_step_counter) != k + 1:
                raise RuntimeError("wrapper.step did not advance the counter by exactly one")
            self._control_steps += 1
            record = self._context.finish()
            self.step_in_episode += 1
            if not torch.equal(self.env.episode_length_buf, self.step_in_episode):
                raise RuntimeError(
                    "env.episode_length_buf diverged from the runtime episode step "
                    "counter (bypass step or missing reset?)"
                )
            if not torch.equal(record["command"], command):
                raise RuntimeError("context command differs from the input command")
            if not torch.equal(
                record["previous_action"], input_snapshot.observation.prev_action
            ):
                raise RuntimeError(
                    "context previous_action differs from the input snapshot's "
                    "previous clipped action"
                )

            # 3) external reward audit: recompute Σ weight·raw·dt from the
            #    cached raw terms and compare with the actual manager buffer.
            terms_state = getattr(self.env, "_confined_training_reward_terms", None)
            if not isinstance(terms_state, dict) or terms_state.get(
                "control_id"
            ) != record["control_id"]:
                raise RuntimeError("reward term cache missing for this control id")
            raws = terms_state["raws"]
            if set(raws) != set(TRAINING_REWARD_WEIGHTS):
                raise RuntimeError(
                    f"reward term set {sorted(raws)} != configured "
                    f"{sorted(TRAINING_REWARD_WEIGHTS)}"
                )
            expected = torch.zeros_like(record["reward"])
            for name, weight in TRAINING_REWARD_WEIGHTS.items():
                contribution = raws[name] * float(weight) * CONTROL_DT_S
                require_finite(f"reward_audit/{name}", contribution)
                expected = expected + contribution
            require_finite("reward_audit/total", expected)
            if not torch.allclose(record["reward"], expected, atol=1e-6, rtol=1e-5):
                raise RuntimeError(
                    f"actual reward {record['reward'].cpu().tolist()} != expected "
                    f"{expected.cpu().tolist()} from per-term raws"
                )

            # 4) Independent flags (cloned by finish BEFORE any reset). The
            #    runtime still completes the natural t_(k+1) boundary below:
            #    ordinary command cadence, only naturally-due scans, then
            #    the actual final observation/labels. The PlaySession display
            #    separately retains the consumed pre-action o_k and its
            #    predictions; after this method returns, read-only viewer
            #    work never calls prepare again while fallen.
            terminated = record["terminated"]
            truncated = record["truncated"]
            # Snapshot existing reward/termination state before partial reset
            # clears it. Diagnostics never advance either state machine.
            term_state = get_termination_state(self.env)
            diagnostics = {"termination_reasons": {
                "fall_low_height": (term_state.low_height_done & terminated).clone(),
                "fall_tilt": (term_state.tilt_done & terminated).clone(),
                "fall_flipped": (term_state.flip_done & terminated).clone(),
                "timeout": truncated.clone(),
            }}
            air_state = get_foot_air_state(self.env)
            if air_state.landing is not None and air_state.consumed_air_s is not None:
                diagnostics["foot_landing"] = air_state.landing.clone()
                diagnostics["foot_landing_air_time_s"] = air_state.consumed_air_s.clone()
            height = root_height_above_plane(self.env, record["root_pos_w"])
            up_dot = -record["gravity_b"][:, 2]
            engineering_fall = (height < ENGINEERING_FALL_HEIGHT_M) | (
                up_dot < ENGINEERING_FALL_UP_DOT
            )

            reset_env_ids = None
            scan_writes = []
            engineering_stopped = False
            play_fall = self.mode == "play" and bool(terminated.any())
            play_truncation = self.mode == "play" and bool(truncated.any())
            # Normal boundary for every mode, including a terminal play
            # step: no forced scan is added, but a scan already due at this
            # control identity is part of the accepted transition semantics.
            self._commands.step(episode_steps=self.step_in_episode)
            scan_writes = self._scan_rows_now(self._due_scan_rows())
            for write in scan_writes:
                self.next_due_scan_step[write["env_row"]] = (
                    self._control_steps + _SCAN_PERIOD_CONTROL_STEPS
                )
            final_inputs = clone_prepared(self.prepare())
            final_labels = self.labels()
            next_inputs = final_inputs
            next_labels = final_labels

            if self.mode == "engineering":
                if bool(engineering_fall.any()) or bool((terminated | truncated).any()):
                    engineering_stopped = True
                    self._save_stop(
                        record=record,
                        final_inputs=final_inputs,
                        final_labels=final_labels,
                        engineering_fall=engineering_fall,
                    )
            elif self.mode == "play":
                # P7 play lifecycle: ONLY the confirmed formal fall gates
                # (the three termination terms — 10-step persisted low height
                # / 10-step persisted tilt / immediate flip) pause the run;
                # the engineering instantaneous gate does NOT apply, and the
                # 20 s truncation must never fire (the play env cfg removed
                # the timeout term). Collisions never pause/reset anything.
                if play_truncation:
                    raise RuntimeError(
                        "play received a truncation flag: the 20 s "
                        "episode_timeout term must be absent from the play "
                        "env config (a leak here is a config fault)"
                )
                if play_fall:
                    fall_state = get_termination_state(self.env)
                    engineering_stopped = True  # = fall-pause in play mode
                    self._save_stop(
                        record=record,
                        final_inputs=final_inputs,
                        final_labels=final_labels,
                        engineering_fall=None,
                        kind="play_fall_stop",
                        extra={
                            "final_input_semantics": (
                                "natural t_(k+1) final observation/labels after "
                                "ordinary command cadence and only scans already "
                                "due at this boundary; PlaySession separately "
                                "retains consumed pre-action o_k for predictions, "
                                "with no extra prepare/scan/noise/forward after "
                                "this transition returns"
                            ),
                            "reason": (
                                "formal fall gates (10-step low height "
                                "<0.20 m / 10-step tilt >60 deg / immediate "
                                "flip >90 deg); scene saved and the run is "
                                "PAUSED — only the explicit full-episode "
                                "manual reset continues play"
                            ),
                            "fall_low_height": fall_state.low_height_done
                            .detach().cpu().tolist(),
                            "fall_tilt": fall_state.tilt_done.detach().cpu().tolist(),
                            "fall_flipped": fall_state.flip_done.detach().cpu().tolist(),
                            "low_height_counts": fall_state.low_height_count
                            .detach().cpu().tolist(),
                            "tilt_counts": fall_state.tilt_count.detach().cpu().tolist(),
                        },
                    )
            elif self.mode in ("training_lifecycle_fixture", "training"):
                done = terminated | truncated
                if bool(done.any()):
                    reset_env_ids = done.nonzero(as_tuple=False).flatten()
                    self._reset_selected(reset_env_ids)
                    next_inputs = clone_prepared(self.prepare())
                    next_labels = self.labels()

            transition = TrainingTransition(
                control_id=int(record["control_id"]),
                env_ids=input_snapshot.observation.env_ids.clone(),
                episode_ids=episodes_before,
                step_ids=steps_before,
                time_s=time_s,
                input=input_snapshot,
                command=command,
                reward=record["reward"].clone(),
                reward_raw_terms={name: value.clone() for name, value in raws.items()},
                reward_weights=dict(TRAINING_REWARD_WEIGHTS),
                reward_expected=expected.clone(),
                terminated=terminated.clone(),
                truncated=truncated.clone(),
                final_inputs=final_inputs,
                final_labels=final_labels,
                reset_env_ids=None if reset_env_ids is None else reset_env_ids.clone(),
                next_inputs=next_inputs,
                next_labels=next_labels,
                context_record=record,
                mode=self.mode,
                engineering_stopped=engineering_stopped,
                scan_writes=scan_writes,
                diagnostics=diagnostics,
            )
            self.transitions.append(transition)
            if len(self.transitions) > _MAX_RETAINED_TRANSITIONS:
                self.transitions.pop(0)
            if engineering_stopped:
                self._stopped = True
            return transition
        except BaseException as err:
            if not self._faulted:
                self._save_fault(err)
            raise

    # ── evidence / lifecycle ───────────────────────────────────────────────

    def _guard_usable(self, phase: str) -> None:
        if self._closed:
            raise TrainingRuntimeFrozenError("training runtime closed")
        if self._faulted:
            raise TrainingRuntimeFrozenError(
                f"training runtime frozen after fault "
                f"({self.last_fault['error'] if self.last_fault else '?'}); "
                "reset/step cannot recover a fault"
            )
        if self._stopped:
            if self.mode == "play" and phase in ("prepare", "labels", "reset"):
                # Play fall-pause: cached read-only prepares (identity-keyed,
                # no rescan/no draws) and the EXPLICIT full-episode manual
                # reset stay available. Only step refuses until the reset
                # clears the pause — continue can never bypass a fall.
                return
            if self.mode == "play":
                raise TrainingRuntimeFrozenError(
                    "play runtime paused after a saved formal fall; stepping "
                    "refuses until the explicit full-episode manual reset"
                )
            raise TrainingRuntimeFrozenError(
                "training runtime stopped after an engineering fall/done; the "
                "whole run ends there (no auto-reset continuation)"
            )

    def _physics_snapshot(self) -> dict:
        import warp as wp

        data = {}
        for name in (
            "time",
            "qpos",
            "qvel",
            "qacc_warmstart",
            "ctrl",
            "qfrc_applied",
            "xfrc_applied",
        ):
            tensor = wp.to_torch(getattr(self.env.sim.wp_data, name))
            data[name] = tensor.detach().cpu().clone()
        return data

    def _save_evidence_dir(self, kind: str) -> Path:
        directory = self.output_dir / f"{kind}_{self._evidence_count:03d}"
        self._evidence_count += 1
        directory.mkdir()
        (directory / "NOT_A_RESUMABLE_CHECKPOINT.txt").write_text(
            "Runtime saves fault/stop scenes only. This is NOT a checkpoint: there is "
            "no network/optimizer state, no PRNG stream, no manager history. It "
            "must never be used to resume training.\n"
        )
        return directory

    def _save_fault(self, err: BaseException, *, phase="step") -> None:
        self._faulted = True
        self.last_fault = {
            "phase": phase,
            "control_step": self._control_steps,
            "actual_common_step_counter": int(self.env.common_step_counter),
            "context_phase": self._context.phase,
            "control_time_s": self.control_time_s(),
            "error": repr(err),
            "traceback": traceback.format_exc(),
            "mode": self.mode,
            "episode_ids": self.episode_ids.detach().cpu().tolist(),
            "scan_counters": self.scan_counters.detach().cpu().tolist(),
            "sensor_scan_counts": self.sensor_scan_counts.detach().cpu().tolist(),
        }
        try:
            directory = self._save_evidence_dir("fault")
            torch.save(self._physics_snapshot(), directory / "physics.pt")
            torch.save(_cpu_tree({"pd":self._context.pd,"contact_tables":self._context.tables,
                "boundary":self._context.record,"input":self._input_cache.get(),
                "commands":self._commands.state_dict(),
                "perturbations": (
                    None if self._perturbations is None
                    else self._perturbations.state_dict()
                )}),
                directory / "current_scene.pt")
            last = self.transitions[-1] if self.transitions else None
            if last is not None:
                torch.save(
                    {
                        "control_id": last.control_id,
                        "reward": last.reward.cpu(),
                        "terminated": last.terminated.cpu(),
                        "truncated": last.truncated.cpu(),
                    },
                    directory / "last_transition_summary.pt",
                )
            (directory / "fault.json").write_text(json.dumps(self.last_fault, indent=2))
        except Exception as save_error:
            # Saving must never mask the original fault; the in-memory record
            # above still carries the evidence.
            self.last_fault["evidence_save_error"] = repr(save_error)

    def _save_stop(
        self,
        *,
        record,
        final_inputs,
        final_labels,
        engineering_fall,
        kind: str = "engineering_stop",
        extra: dict | None = None,
    ) -> None:
        self.stop_record = {
            "kind": kind,
            "control_step": self._control_steps,
            "control_time_s": self.control_time_s(),
            "mode": self.mode,
            "engineering_fall": (
                None if engineering_fall is None
                else engineering_fall.detach().cpu().tolist()
            ),
            "terminated": record["terminated"].detach().cpu().tolist(),
            "truncated": record["truncated"].detach().cpu().tolist(),
            "episode_ids": self.episode_ids.detach().cpu().tolist(),
            "steps_completed": self._control_steps,
        }
        if extra:
            self.stop_record.update(extra)
        directory = self._save_evidence_dir(kind)
        torch.save(self._physics_snapshot(), directory / "physics.pt")
        torch.save(
            {
                "input": _cpu_prepared(final_inputs),
                "final_labels_envelope": final_labels["envelope"].values.cpu(),
            },
            directory / "terminal_observation.pt",
        )
        (directory / "stop.json").write_text(json.dumps(self.stop_record, indent=2))

    def close(self) -> None:
        """Restore the instance hooks (idempotent)."""
        if self._closed:
            return
        self._context.close()
        self._closed = True


def _cpu_prepared(prepared: PreparedControlInputs) -> dict:
    observation = prepared.observation
    return {
        "observation": {
            name: getattr(observation, name).detach().cpu().clone()
            for name in (
                "env_ids",
                "episode_ids",
                "step_ids",
                "time_s",
                "lin_vel_b",
                "ang_vel_b",
                "gravity_b",
                "joint_pos_rel_ref",
                "joint_vel",
                "prev_action",
                "command",
            )
        },
        "near_points": prepared.near_points.detach().cpu().clone(),
        "near_valid": prepared.near_valid.detach().cpu().clone(),
        "far_points": prepared.far_points.detach().cpu().clone(),
        "far_valid": prepared.far_valid.detach().cpu().clone(),
        "near_scan_ids": prepared.near_scan_ids.detach().cpu().clone(),
    }


def _cpu_tree(value):
    from dataclasses import fields, is_dataclass
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if is_dataclass(value):return {f.name:_cpu_tree(getattr(value,f.name)) for f in fields(value)}
    if isinstance(value,dict):return {k:_cpu_tree(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [_cpu_tree(v) for v in value]
    return value
