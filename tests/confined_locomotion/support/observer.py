"""Observers for actual runtime transitions and optimizer calls."""
from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

from .evidence import write_json
from .snapshots import cpu_tree


def _tensor_fields(value) -> dict:
  import torch

  if not torch.is_tensor(value):
    return {}
  return {"shape": list(value.shape), "dtype": str(value.dtype)}


class RuntimeStepObserver:
  """Wrap the current runtime's step only to clone its live inputs/results."""

  def __init__(self, runtime, output_dir: str | Path, *, label: str = "transition"):
    self.runtime = runtime
    self.output_dir = Path(output_dir)
    self.output_dir.mkdir(parents=True, exist_ok=True)
    self.label = label
    self.records: list[dict] = []
    self._original = None

  def __enter__(self):
    if self._original is not None:
      raise RuntimeError("runtime observer is already active")
    self._original = self.runtime.step

    def recorded_step(raw_action):
      runtime = self.runtime
      prepared = runtime.prepare()
      labels = runtime.labels()
      before_control = int(runtime.control_steps_taken)
      before_scans = cpu_tree(runtime.sensor_scan_counts)
      before_counts = {
        "common_step_counter": int(runtime.env.common_step_counter),
        "sim_step_counter": int(runtime.env._sim_step_counter),
      }
      transition = self._original(raw_action)
      record = {
        "prepared": cpu_tree(prepared),
        "labels": cpu_tree(labels),
        "raw_action_argument": cpu_tree(raw_action),
        "transition": cpu_tree(transition),
        "before_control_steps": before_control,
        "after_control_steps": int(runtime.control_steps_taken),
        "before_scan_counts": before_scans,
        "after_scan_counts": cpu_tree(runtime.sensor_scan_counts),
        "before_env_counts": before_counts,
        "after_env_counts": {
          "common_step_counter": int(runtime.env.common_step_counter),
          "sim_step_counter": int(runtime.env._sim_step_counter),
        },
      }
      index = len(self.records) + 1
      self.records.append(record)
      import torch

      torch.save(record, self.output_dir / f"{self.label}_{index:03d}.pt")
      return transition

    self.runtime.step = recorded_step
    return self

  def __exit__(self, exc_type, exc, tb):
    if self._original is not None:
      self.runtime.step = self._original
      self._original = None
    return False


class TrainerObserver:
  """Clone real minibatch tensors/gradients and observe optimizer call order."""

  def __init__(self, trainer, output_dir: str | Path, *, full_gradients: bool = True):
    self.trainer = trainer
    self.output_dir = Path(output_dir)
    self.output_dir.mkdir(parents=True, exist_ok=True)
    self.full_gradients = bool(full_gradients)
    self.events: list[dict] = []
    self.calls: list[dict] = []
    self.forward_count = 0
    self.grad_pre_count = 0
    self.grad_post_count = 0
    self.adam_done_count = 0
    self._original_observer = None
    self._active = False
    self._patches = []
    self._parameter_names = {
      id(parameter): name
      for name, parameter in trainer.policy_adapter.named_parameters()
    }
    self._parameter_ids = {id(parameter) for parameter in trainer.unique_parameters}

  def _capture(self, stage: str, payload: dict) -> None:
    epoch = int(payload.get("epoch", -1))
    minibatch = int(payload.get("minibatch", -1))
    key = f"epoch{epoch:02d}_minibatch{minibatch:02d}"
    record: dict[str, Any] = {"stage": stage, "epoch": epoch, "minibatch": minibatch}
    import torch

    if stage == "minibatch_forward":
      self.forward_count += 1
      for name in (
        "batch", "network_output", "distribution", "mirrored_batch",
        "mirrored_output", "mirrored_distribution", "old_mirrored",
        "old_mirror_log_prob", "loss_policy", "loss_value", "loss_envelope",
        "loss_ground", "loss_policy_original", "loss_policy_mirrored",
        "loss_value_original", "loss_value_mirrored", "loss_envelope_original",
        "loss_envelope_mirrored", "loss_ground_original", "loss_ground_mirrored",
        "loss_mirror_mean", "loss_total", "kl_event", "kl_event_original",
        "kl_event_mirrored", "indices", "normalized_advantage",
        "ground_predictions", "ground_bucket", "ground_supervision_size",
        "mirrored_ground_predictions", "mirrored_ground_frames",
        "symmetry_identity",
      ):
        if name in payload:
          record[name] = cpu_tree(payload[name])
      distribution = payload.get("distribution")
      batch = payload.get("batch")
      if distribution is not None and batch is not None:
        record["distribution_values"] = {
          "mean": cpu_tree(distribution.mean), "std": cpu_tree(distribution.std),
          "raw_action": cpu_tree(batch.raw_action),
        }
      mirrored = payload.get("mirrored_distribution")
      old_mirrored = payload.get("old_mirrored")
      if mirrored is not None and old_mirrored is not None:
        record["mirrored_distribution_values"] = {
          "mean": cpu_tree(mirrored.mean), "std": cpu_tree(mirrored.std),
          "raw_action": cpu_tree(old_mirrored.raw_action),
        }
    elif stage in ("grad_pre_clip", "grad_post_clip"):
      counter_name = "grad_pre_count" if stage == "grad_pre_clip" else "grad_post_count"
      setattr(self, counter_name, getattr(self, counter_name) + 1)
      if self.full_gradients and (epoch, minibatch) in ((0, 0), (4, 3)):
        gradients = {}
        parameter_ids = []
        for index, parameter in enumerate(payload.get("parameters", ())):
          parameter_ids.append(id(parameter))
          if parameter.grad is not None:
            name = self._parameter_names.get(id(parameter), f"parameter_{index}")
            gradients[name] = cpu_tree(parameter.grad)
        record["parameter_ids"] = parameter_ids
        record["parameter_names"] = [
          self._parameter_names.get(pid, f"unknown:{pid}") for pid in parameter_ids
        ]
        record["gradients"] = gradients
      for name in ("grad_norm_pre_clip", "grad_norm_post_clip"):
        if name in payload:
          record[name] = float(payload[name])
    elif stage == "adam_step_done":
      self.adam_done_count += 1
      if (epoch, minibatch) in ((0, 0), (4, 3)):
        record["learning_rate"] = float(payload["learning_rate"])
        record["optimizer_state"] = cpu_tree(payload["optimizer"].state_dict())
        record["parameters"] = {
          self._parameter_names.get(id(parameter), f"parameter_{index}"): cpu_tree(parameter)
          for index, parameter in enumerate(payload.get("parameters", ()))
        }
    else:
      if stage in ("update_started", "update_finished", "update_faulted"):
        record["payload"] = cpu_tree(payload, omit=frozenset({"trainer", "optimizer", "parameters"}))
    self.events.append(record)
    if stage == "minibatch_forward" or stage in ("grad_pre_clip", "grad_post_clip", "adam_step_done"):
      torch.save(record, self.output_dir / f"{key}_{stage}.pt")

  def _install_call_observers(self) -> None:
    import torch

    state = self
    original_backward = torch.Tensor.backward
    original_clip = torch.nn.utils.clip_grad_norm_
    original_adam_step = torch.optim.Adam.step

    def backward(tensor, *args, **kwargs):
      state.calls.append({"kind": "backward"})
      return original_backward(tensor, *args, **kwargs)

    def clip(parameters, *args, **kwargs):
      actual_parameters = tuple(parameters)
      identities = {id(parameter) for parameter in actual_parameters}
      state.calls.append({
        "kind": "joint_clip",
        "parameter_count": len(actual_parameters),
        "unique_parameter_count": len(identities),
        "expected_parameter_count": len(state._parameter_ids),
        "exact_parameter_set": identities == state._parameter_ids,
        "max_norm": kwargs.get("max_norm", args[0] if args else None),
      })
      return original_clip(actual_parameters, *args, **kwargs)

    def adam_step(optimizer, *args, **kwargs):
      state.calls.append({
        "kind": "adam_step",
        "is_expected_optimizer": optimizer is state.trainer.optimizer,
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
      })
      return original_adam_step(optimizer, *args, **kwargs)

    torch.Tensor.backward = backward
    torch.nn.utils.clip_grad_norm_ = clip
    torch.optim.Adam.step = adam_step
    self._patches = [
      (torch.Tensor, "backward", original_backward),
      (torch.nn.utils, "clip_grad_norm_", original_clip),
      (torch.optim.Adam, "step", original_adam_step),
    ]

  def __enter__(self):
    if self._active:
      raise RuntimeError("trainer observer is already active")
    self._active = True
    self._original_observer = self.trainer._observer

    def observer(stage, payload):
      self._capture(stage, payload)
      if self._original_observer is not None:
        self._original_observer(stage, payload)

    self.trainer._observer = observer
    self._install_call_observers()
    return self

  def __exit__(self, exc_type, exc, tb):
    import torch

    self.trainer._observer = self._original_observer
    self._original_observer = None
    for owner, name, original in reversed(self._patches):
      setattr(owner, name, original)
    self._patches.clear()
    self._active = False
    return False

  def summary(self) -> dict:
    call_kinds = [item["kind"] for item in self.calls]
    return {
      "forward_count": self.forward_count,
      "grad_pre_count": self.grad_pre_count,
      "grad_post_count": self.grad_post_count,
      "adam_step_done_count": self.adam_done_count,
      "call_counts": {kind: call_kinds.count(kind) for kind in ("backward", "joint_clip", "adam_step")},
      "actual_call_order": call_kinds,
      "all_clip_calls_used_exact_parameter_set": all(
        item.get("exact_parameter_set") for item in self.calls if item["kind"] == "joint_clip"
      ),
      "all_steps_used_expected_adam": all(
        item.get("is_expected_optimizer") for item in self.calls if item["kind"] == "adam_step"
      ),
    }


def trainer_observer(trainer, output_dir: str | Path, *, full_gradients: bool = True) -> TrainerObserver:
  return TrainerObserver(trainer, output_dir, full_gradients=full_gradients)

