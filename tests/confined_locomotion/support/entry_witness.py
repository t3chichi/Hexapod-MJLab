"""Observation of the real official launch path for finite engineering checks."""
from __future__ import annotations

import time
from pathlib import Path

from .evidence import write_json
from .snapshots import cpu_tree, runtime_snapshot


class OfficialEntryWitness:
  """Capture public runner events without constructing a second trainer."""

  def __init__(self, output_dir: str | Path):
    self.output_dir = Path(output_dir)
    self.output_dir.mkdir(parents=True, exist_ok=True)
    self.events: list[dict] = []
    self.runner = None
    self.wrapper = None
    self.live_boundary = None
    self.initial_counters = None

  def __call__(self, event, runner, payload) -> None:
    self.events.append({
      "event": str(event),
      "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
      "runner_type": type(runner).__name__,
      "payload": cpu_tree(payload, omit=frozenset({"trainer", "optimizer", "parameters"})),
      "object_presence": {
        name: getattr(runner, name, None) is not None
        for name in ("runtime", "collector", "trainer", "policy_adapter", "network")
      },
    })
    if event == "constructed":
      self.runner = runner
      self.wrapper = runner.env
      self.initial_counters = {
        "runtime_control_steps": int(runner.runtime.control_steps_taken),
        "env_common_step_counter": int(runner.env.unwrapped.common_step_counter),
        "simulation_step_counter": int(runner.env.unwrapped._sim_step_counter),
        "episode_length_buf": cpu_tree(runner.env.unwrapped.episode_length_buf),
        "initial_reset_recipe": cpu_tree(
          (runner.recipe.get("wrapper") or {}).get("initial_reset")
        ),
      }
    elif event == "learn_finished":
      if self.runner is not runner:
        raise RuntimeError("official learn_finished came from a different runner instance")
      self.live_boundary = {
        "runtime": runtime_snapshot(runner.runtime),
        "runner": {
          "current_learning_iteration": int(runner.current_learning_iteration),
          "total_control_batches": int(runner.total_control_batches),
          "total_env_samples": int(runner.total_env_samples),
          "optimizer_step_count": int(runner.trainer.optimizer_step_count),
          "update_count": int(runner.trainer.update_count),
          "unique_parameter_count": int(runner.trainer.unique_parameter_count),
          "adam_param_groups": len(runner.trainer.optimizer.param_groups),
          "lifecycle_phase": str(getattr(runner, "_lifecycle_phase", "unknown")),
        },
      }
      import torch

      torch.save(self.live_boundary, self.output_dir / "official_live_boundary.pt")

  def summary(self) -> dict:
    return {
      "events": self.events,
      "event_names": [item["event"] for item in self.events],
      "constructed_runner_observed": self.runner is not None,
      "official_wrapper_observed": self.wrapper is not None,
      "pre_close_boundary_observed": self.live_boundary is not None,
      "initial_counters": self.initial_counters,
      "runner_class": None if self.runner is None else type(self.runner).__name__,
      "wrapper_class": None if self.wrapper is None else type(self.wrapper).__name__,
    }

  @property
  def event_names(self) -> tuple[str, ...]:
    return tuple(item["event"] for item in self.events)


class WrapperInitialResetWitness:
  """Count the real reset called by the official vector wrapper constructor.

  The temporary observer delegates directly to the original bound reset and
  wrapper constructor. It records only calls made inside that constructor;
  selected terminal resets during learning are outside the observation window.
  """

  def __init__(self):
    self.counts: list[int] = []
    self.wrapper_types: list[str] = []
    self._wrapper_class = None
    self._original_init = None

  def __enter__(self):
    from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper

    if self._original_init is not None:
      raise RuntimeError("wrapper reset witness is already active")
    self._wrapper_class = RslRlVecEnvWrapper
    self._original_init = RslRlVecEnvWrapper.__init__
    state = self
    original_init = self._original_init

    def observed_init(wrapper, env, *args, **kwargs):
      original_reset = env.reset
      count = 0

      def observed_reset(*reset_args, **reset_kwargs):
        nonlocal count
        count += 1
        return original_reset(*reset_args, **reset_kwargs)

      env.reset = observed_reset
      try:
        return original_init(wrapper, env, *args, **kwargs)
      finally:
        env.reset = original_reset
        state.counts.append(count)
        state.wrapper_types.append(type(wrapper).__name__)

    RslRlVecEnvWrapper.__init__ = observed_init
    return self

  def __exit__(self, exc_type, exc, tb):
    if self._wrapper_class is not None and self._original_init is not None:
      self._wrapper_class.__init__ = self._original_init
    self._wrapper_class = None
    self._original_init = None
    return False

  def summary(self) -> dict:
    return {
      "constructor_count": len(self.counts),
      "initial_reset_counts": list(self.counts),
      "wrapper_types": list(self.wrapper_types),
      "exactly_one_wrapper_reset_each": bool(self.counts)
        and all(count == 1 for count in self.counts),
    }


def install_official_entry_observer(output_dir: str | Path) -> OfficialEntryWitness:
  """Attach to the production runner observer channel used by official launch."""
  from hexapod_mjlab.tasks.confined_locomotion.rl.runner import (
    add_engineering_observer,
    clear_engineering_observers,
  )

  witness = OfficialEntryWitness(output_dir)
  clear_engineering_observers()
  add_engineering_observer(witness)
  return witness


def clear_official_entry_observers() -> None:
  from hexapod_mjlab.tasks.confined_locomotion.rl.runner import clear_engineering_observers

  clear_engineering_observers()


def write_witness(
  output_dir: str | Path,
  witness: OfficialEntryWitness,
  checks: dict,
  *,
  wrapper_reset: WrapperInitialResetWitness | None = None,
) -> Path:
  return write_json(Path(output_dir) / "entry_witness.json", {
    "schema": "confined_locomotion.official_entry_witness/v1",
    "mode": "formal_training",
    "claim": "one official finite GPU iteration; no learning-effect claim",
    "witness": witness.summary(),
    "wrapper_initial_reset": None if wrapper_reset is None else wrapper_reset.summary(),
    "checks": checks,
  })

