"""Checkpoint contracts: stdlib CPU supplement (no GPU claim).

These checks cover the CPU-checkable checkpoint contracts (no pytest dependency;
plain ``test_*`` functions + a stdlib ``main()`` driver). What is NOT covered
here and never claimed: real same-device full save/restore, including live
physics, optimizer, and random-stream continuation. Those checks require the
explicit GPU restore suite; CPU checks cover callable validation and
weights-only operations.

Checks (deliberately beyond "key exists" assertions — counterexamples drop
required fields and inject non-finite values):

* envelope: schema/kind/resumable/section completeness — a weights-only
  file, another schema, a NON-RESUMABLE artifact or a missing section is
  rejected as a full-resume payload;
* identity rejections (pure ``_compare_identity``): same M with MOVED grid
  positions, different M, changed model digest, changed source SHA, changed
  dependency version, changed device index and changed env count all refuse;
* stable save boundary rejections: fresh/collecting/updating phase, faulted/
  stopped/closed runtime, faulted trainer, active rollout buffer,
  mid-transition context, pending terminal rows, no completed iteration,
  partially executed optimizer steps and counter desyncs all refuse — the
  fully consistent completed 24/20 boundary alone passes;
* weights mode: roundtrip (bit-equal params + log_std), no-overwrite, and
  malformed payloads (NaN weight, missing key, wrong ground M, wrong grid
  identity, changed source) rejected with NO partial mutation of the target
  network; explicit allow_source_change is the only source-change rule;
* the default guard stays a non-serialized ClassVar, and the registered
  default factory keeps max_iterations=0 / engineering_enabled=False;
* runner surface: ``save``/``load`` signatures match the official
  mjlab 1.6.0 base (path/infos; path/load_cfg/strict/map_location), and
  ``load_weights`` exposes the explicit weights-only contract;
* source identity: current runtime, collector, transition, diagnostics, and
  checkpoint implementation modules contribute their content digests;
* finiteness counterexamples: NaN inside the payload refuses; the ONE
  documented ``-inf`` exception (empty-window ``last_time`` with
  ``EMPTY_SCAN_ID``) passes and an inconsistent ``-inf`` refuses.
"""

from __future__ import annotations

import copy as copylib
import dataclasses
import inspect
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch

from support.fixtures import build_adapter, build_network, build_provider

from hexapod_mjlab.rl.confined_locomotion.network import ConfinedLocomotionNetwork
from hexapod_mjlab.rl.confined_locomotion.policy import ConfinedPolicyAdapter
from hexapod_mjlab.tasks.confined_locomotion.rl import checkpoint as checkpoint_module
from hexapod_mjlab.tasks.confined_locomotion.rl.checkpoint import (
  CHECKPOINT_KIND_FULL,
  CHECKPOINT_KIND_WEIGHTS,
  CHECKPOINT_SCHEMA,
  CheckpointError,
  _compare_identity,
  _require_stable_save_boundary,
  _validate_payload_finiteness,
  load_weights,
  save_weights_only,
  validate_payload_envelope,
)
from hexapod_mjlab.tasks.confined_locomotion.rl.runner import ConfinedLocomotionRunner

_SEED = 20261008


def _make_adapter(seed: int = _SEED) -> ConfinedPolicyAdapter:
  network = build_network(seed=seed, ground_samples=12)
  provider = build_provider(seed=seed, device="cpu")
  return build_adapter(network=network, provider=provider, seed=seed, device="cpu")


def _params_digest(adapter_like) -> str:
  import hashlib

  digest = hashlib.sha256()
  for name, parameter in sorted(adapter_like.named_parameters()):
    digest.update(name.encode())
    digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
  return digest.hexdigest()


def _full_envelope_payload() -> dict:
  return {
    "schema": CHECKPOINT_SCHEMA,
    "kind": CHECKPOINT_KIND_FULL,
    "resumable": True,
    "task": "El-4090-Confined-Locomotion-Flat",
    "created": "2026-10-08T00:00:00",
    "identity": {},
    "physics": {"manifest": {}, "arrays": {}},
    "algorithm": {},
    "runtime": {},
    "env_state": {},
    "action_state": {},
    "foot_air_state": None,
    "termination_state": None,
    "prng": {},
    "runner_state": {},
    "manager_state": {},
  }


# ═══════════════════════ envelope: schema / kind / resumable ═══════════════


def test_envelope_accepts_a_complete_full_payload():
  validate_payload_envelope(_full_envelope_payload())


def test_envelope_rejects_wrong_schema():
  payload = _full_envelope_payload()
  payload["schema"] = "confined_locomotion.checkpoint/v0"
  try:
    validate_payload_envelope(payload)
  except CheckpointError as error:
    assert "schema" in str(error)
  else:
    raise AssertionError("wrong schema accepted")


def test_envelope_rejects_weights_kind_and_non_resumable():
  weights = _full_envelope_payload()
  weights["kind"] = CHECKPOINT_KIND_WEIGHTS
  try:
    validate_payload_envelope(weights)
  except CheckpointError as error:
    assert "weights-only" in str(error) or "load_weights" in str(error)
  else:
    raise AssertionError("weights kind accepted as full resume")

  artifact = _full_envelope_payload()
  artifact["kind"] = "unrecognized_kind"
  try:
    validate_payload_envelope(artifact)
  except CheckpointError:
    pass
  else:
    raise AssertionError("an unsupported artifact kind was accepted")

  marked = _full_envelope_payload()
  marked["resumable"] = False
  try:
    validate_payload_envelope(marked)
  except CheckpointError as error:
    assert "non-resumable" in str(error)
  else:
    raise AssertionError("resumable=False accepted")


def test_envelope_rejects_missing_section_by_name():
  payload = _full_envelope_payload()
  del payload["prng"]
  del payload["physics"]
  try:
    validate_payload_envelope(payload)
  except CheckpointError as error:
    assert "'prng'" in str(error) and "'physics'" in str(error)
  else:
    raise AssertionError("missing sections accepted")


# ═══════════════════════ identity compatibility rejections ═════════════════


def _base_identity() -> dict:
  return {
    "schema": CHECKPOINT_SCHEMA,
    "kind": CHECKPOINT_KIND_FULL,
    "resumable": True,
    "task": "El-4090-Confined-Locomotion-Flat",
    "runner_identity": "El-4090-Confined-Locomotion-Flat/ConfinedLocomotionRunner/mode=engineering",
    "source_sha256": {"runner.py": "aa", "el_4090_lidar.xml": "bb"},
    "dependencies": {"torch": "2.11.0", "mjlab": "1.6.0"},
    "device": {"type": "cuda", "index": 0, "name": "NVIDIA GeForce RTX 4050"},
    "num_envs": 2,
    "grid": {
      "identity": {
        "x_coords": [0.1, 0.3333333333333333, 0.5666666666666667, 0.8],
        "y_coords": [-0.3, 0.0, 0.3],
        "num_points": 12,
      },
      "identity_hash": "gridhash",
    },
    "model": {"digest": "modeldigest", "q_ref": [[0.0] * 18]},
    "ppo_config": {"learning_rate": 0.001},
  }


def test_identity_accepts_exact_match():
  _compare_identity(_base_identity(), copylib.deepcopy(_base_identity()))


def test_identity_rejects_same_moved_grid_positions():
  saved = _base_identity()
  saved["grid"]["identity"]["x_coords"] = [0.2, 0.4, 0.6, 0.8]  # same M=12
  try:
    _compare_identity(_base_identity(), saved)
  except CheckpointError as error:
    assert "grid" in str(error)
  else:
    raise AssertionError("same-M moved grid positions accepted")


def test_identity_rejects_different_m():
  saved = _base_identity()
  saved["grid"]["identity"]["num_points"] = 13
  saved["grid"]["identity"]["x_coords"] = [0.1, 0.3, 0.5, 0.7, 0.8]
  try:
    _compare_identity(_base_identity(), saved)
  except CheckpointError:
    pass
  else:
    raise AssertionError("different M accepted")


def test_identity_rejects_model_source_dependency_device_and_env_changes():
  for mutate, expected in (
    (lambda s: s["model"].__setitem__("digest", "other"), "model"),
    (lambda s: s["source_sha256"].__setitem__("runner.py", "zz"), "source_sha256"),
    (lambda s: s["dependencies"].__setitem__("mjlab", "1.7.0"), "dependencies"),
    (lambda s: s["device"].__setitem__("index", 1), "device"),
    (lambda s: s.__setitem__("num_envs", 4), "num_envs"),
    (lambda s: s["ppo_config"].__setitem__("learning_rate", 0.002), "ppo_config"),
  ):
    saved = _base_identity()
    mutate(saved)
    try:
      _compare_identity(_base_identity(), saved)
    except CheckpointError as error:
      assert expected in str(error), (expected, str(error))
    else:
      raise AssertionError(f"identity change on {expected} accepted")


def test_identity_rejects_extra_and_missing_keys():
  extra = _base_identity()
  extra["surprise"] = 1
  try:
    _compare_identity(_base_identity(), extra)
  except CheckpointError as error:
    assert "surprise" in str(error)
  else:
    raise AssertionError("unknown extra identity key accepted")

  missing = _base_identity()
  missing.pop("model")
  try:
    _compare_identity(_base_identity(), missing)
  except CheckpointError as error:
    assert "missing" in str(error) and "model" in str(error)
  else:
    raise AssertionError("missing identity key accepted")


# ═══════════════════════ stable save boundary rejections ═══════════════════


def _fake_runner(**overrides) -> SimpleNamespace:
  pending = torch.zeros(2, dtype=torch.bool)
  defaults = dict(
    _lifecycle_phase="ready",
    _log_failed=False,
    _active_buffer=None,
    current_learning_iteration=1,
    total_control_batches=24,
    total_env_samples=48,
    trainer=SimpleNamespace(
      faulted=False, fault_stage=None, update_count=1, optimizer_step_count=20
    ),
    runtime=SimpleNamespace(
      _closed=False,
      _faulted=False,
      _stopped=False,
      _control_steps=24, num_envs=2,
      _context=SimpleNamespace(phase="idle"),
      env=SimpleNamespace(
        common_step_counter=24, _manual_reset_pending=pending
      ),
    ),
  )
  defaults.update(overrides)
  return SimpleNamespace(**defaults)


def test_stable_boundary_accepts_the_consistent_completed_iteration():
  _require_stable_save_boundary(_fake_runner())


def test_stable_boundary_rejects_every_unstable_state():
  cases = {
    "fresh phase": dict(_lifecycle_phase="fresh", current_learning_iteration=0,
                        total_control_batches=0),
    "collecting phase": dict(_lifecycle_phase="collecting"),
    "updating phase": dict(_lifecycle_phase="updating"),
    "faulted phase": dict(_lifecycle_phase="faulted"),
    "runtime faulted": dict(runtime=SimpleNamespace(
      _closed=False, _faulted=True, _stopped=False, _control_steps=24, num_envs=2,
      _context=SimpleNamespace(phase="idle"),
      env=SimpleNamespace(common_step_counter=24,
                          _manual_reset_pending=torch.zeros(2, dtype=torch.bool)),
    )),
    "runtime stopped": dict(runtime=SimpleNamespace(
      _closed=False, _faulted=False, _stopped=True, _control_steps=24, num_envs=2,
      _context=SimpleNamespace(phase="idle"),
      env=SimpleNamespace(common_step_counter=24,
                          _manual_reset_pending=torch.zeros(2, dtype=torch.bool)),
    )),
    "trainer faulted": dict(trainer=SimpleNamespace(
      faulted=True, fault_stage="epoch0/minibatch0/backward", update_count=1,
      optimizer_step_count=7)),
    "active buffer": dict(_active_buffer=object()),
    "mid transition": dict(runtime=SimpleNamespace(
      _closed=False, _faulted=False, _stopped=False, _control_steps=24, num_envs=2,
      _context=SimpleNamespace(phase="capturing"),
      env=SimpleNamespace(common_step_counter=24,
                          _manual_reset_pending=torch.zeros(2, dtype=torch.bool)),
    )),
    "pending terminal rows": dict(runtime=SimpleNamespace(
      _closed=False, _faulted=False, _stopped=False, _control_steps=24, num_envs=2,
      _context=SimpleNamespace(phase="idle"),
      env=SimpleNamespace(common_step_counter=24,
                          _manual_reset_pending=torch.ones(2, dtype=torch.bool)),
    )),
    "partially executed optimizer": dict(trainer=SimpleNamespace(
      faulted=False, fault_stage=None, update_count=1, optimizer_step_count=19)),
    "counter desync": dict(runtime=SimpleNamespace(
      _closed=False, _faulted=False, _stopped=False, _control_steps=23, num_envs=2,
      _context=SimpleNamespace(phase="idle"),
      env=SimpleNamespace(common_step_counter=24,
                          _manual_reset_pending=torch.zeros(2, dtype=torch.bool)),
    )),
    "batches mismatch": dict(total_control_batches=25),
    "no update": dict(trainer=SimpleNamespace(
      faulted=False, fault_stage=None, update_count=0, optimizer_step_count=0)),
  }
  for name, overrides in cases.items():
    try:
      _require_stable_save_boundary(_fake_runner(**overrides))
    except CheckpointError:
      pass
    else:
      raise AssertionError(f"unstable boundary accepted: {name}")


# ═══════════════════════ weights mode: roundtrip + no partial mutation ═════


def _write_mutated_weights(tmp: Path, adapter, mutation, *, name="mutated.pt"):
  original = tmp / ("original_" + name)
  save_weights_only(original, adapter, reason="test")
  payload = torch.load(original, weights_only=False)
  mutation(payload)
  target = tmp / name
  torch.save(payload, target)
  return target


def test_weights_roundtrip_bit_exact_and_no_overwrite():
  adapter = _make_adapter()
  with tempfile.TemporaryDirectory() as tmp_name:
    tmp = Path(tmp_name)
    path = save_weights_only(tmp / "weights.pt", adapter, reason="roundtrip")
    assert path.is_file()
    try:
      save_weights_only(path, adapter, reason="again")
    except CheckpointError as error:
      assert "overwrite" in str(error)
    else:
      raise AssertionError("existing weights file overwritten")

    fresh = _make_adapter(seed=_SEED + 999)
    report = load_weights(
      path, network=fresh.network, provider=fresh.distribution_provider
    )
    assert report["resumable"] is False
    assert "NOT a resume" in report["note"]
    for name, saved in adapter.state_dict().items():
      live = fresh.state_dict()[name]
      assert torch.equal(saved, live), name
    assert torch.equal(adapter.distribution_provider.log_std,
                       fresh.distribution_provider.log_std)


def test_weights_malformed_payloads_reject_without_partial_mutation():
  source = _make_adapter()
  fresh = _make_adapter(seed=_SEED + 7)
  digest_before = _params_digest(fresh)
  with tempfile.TemporaryDirectory() as tmp_name:
    tmp = Path(tmp_name)
    cases = {
      "nan weight": lambda payload: payload["state"][
        sorted(payload["state"])[0]
      ].fill_(float("nan")),
      "missing key": lambda payload: payload["state"].pop(
        sorted(k for k in payload["state"] if k.startswith("network.action_head"))[0]
      ),
      "wrong schema": lambda payload: payload.__setitem__("schema", "bogus/v9"),
      "non-resumable-diagnostic kind": lambda payload: payload.__setitem__(
        "kind", "unrecognized_kind"),
    }
    for name, mutation in cases.items():
      target = _write_mutated_weights(tmp, source, mutation, name=f"{name}.pt")
      try:
        load_weights(
          target, network=fresh.network, provider=fresh.distribution_provider
        )
      except CheckpointError:
        pass
      else:
        raise AssertionError(f"malformed weights accepted: {name}")
      assert _params_digest(fresh) == digest_before, f"partial mutation: {name}"


def test_weights_grid_and_source_rules():
  adapter = _make_adapter()
  with tempfile.TemporaryDirectory() as tmp_name:
    tmp = Path(tmp_name)
    path = save_weights_only(
      tmp / "weights.pt", adapter,
      grid_identity={"x_coords": [0.1], "num_points": 12},
      reason="grid test",
    )
    fresh = _make_adapter(seed=1)
    try:
      load_weights(path, network=fresh.network,
                   provider=fresh.distribution_provider,
                   expected_grid_identity={"x_coords": [0.2], "num_points": 12})
    except CheckpointError as error:
      assert "grid" in str(error).lower()
    else:
      raise AssertionError("moved grid positions accepted in weights mode")

    other_m = ConfinedLocomotionNetwork(
      mlp_activation=torch.nn.ELU(), init_fn=None, envelope_mode="raw",
      ground_samples=15,
    )
    try:
      load_weights(path, network=other_m)
    except CheckpointError as error:
      assert "ground head size" in str(error)
    else:
      raise AssertionError("different M accepted in weights mode")

    mutated = _write_mutated_weights(
      tmp, adapter,
      lambda payload: payload["identity"]["source_sha256"].__setitem__(
        "runner.py", "changed"),
      name="source_changed.pt",
    )
    try:
      load_weights(mutated, network=fresh.network,
                   provider=fresh.distribution_provider)
    except CheckpointError as error:
      assert "source" in str(error)
    else:
      raise AssertionError("source change accepted without the explicit rule")
    report = load_weights(mutated, network=fresh.network,
                          provider=fresh.distribution_provider,
                          allow_source_change=True)
    assert report["source_change_allowed"] is True


# ═══════════════════════ default guard unchanged ══════════════════════════


def test_default_classvar_and_runner_guard_remain_closed():
  from hexapod_mjlab.tasks.confined_locomotion.rl_cfg import (
    confined_locomotion_rl_cfg,
  )

  cfg = confined_locomotion_rl_cfg()
  assert isinstance(cfg.IMPLEMENTED, bool)
  assert "IMPLEMENTED" not in dataclasses.asdict(cfg)
  assert cfg.max_iterations == 0
  assert cfg.engineering_enabled is False
  assert cfg.formal_training_enabled is False
  from hexapod_mjlab.tasks.confined_locomotion.ground_presets import s1_187_ground_grid
  assert cfg.formal_num_envs is None
  assert cfg.ground_grid == s1_187_ground_grid()


# ═══════════════════════ PRNG fields are actually consumed ═════════════════


# ═══════════════════════ runner surface ═══════════════════════════════════


def test_runner_save_load_match_official_mjlab_signatures():
  from mjlab.rl.runner import MjlabOnPolicyRunner

  official_save = list(inspect.signature(MjlabOnPolicyRunner.save).parameters)
  official_load = list(inspect.signature(MjlabOnPolicyRunner.load).parameters)
  task_save = list(inspect.signature(ConfinedLocomotionRunner.save).parameters)
  task_load = list(inspect.signature(ConfinedLocomotionRunner.load).parameters)
  assert task_save[: len(official_save)] == official_save
  assert official_load == ["self", "path", "load_cfg", "strict", "map_location"]
  assert task_load[: len(official_load)] == official_load
  weights_params = inspect.signature(ConfinedLocomotionRunner.load_weights).parameters
  assert {"network", "provider", "expected_grid_identity", "allow_source_change"} <= set(
    weights_params
  )


# ═══════════════════════ finiteness counterexamples ═══════════════════════


def test_finiteness_with_window_minus_inf_semantics():
  def payload(last_time, last_scan_id):
    return {
      "runtime": {
        "window": {
          "last_time": torch.tensor(last_time),
          "last_scan_id": torch.tensor(last_scan_id),
        }
      },
      "physics": {"arrays": {"qpos": torch.zeros(2, 19)}},
    }

  _validate_payload_finiteness(payload([0.0, 1.0], [3, 4]))
  _validate_payload_finiteness(payload([0.0, float("-inf")], [3, -1]))
  try:
    _validate_payload_finiteness(payload([float("-inf"), 0.0], [3, 4]))
  except CheckpointError as error:
    assert "last_time" in str(error)
  else:
    raise AssertionError("inconsistent -inf window state accepted")

  nan_payload = {
    "runtime": {
      "window": {
        "last_time": torch.zeros(2), "last_scan_id": torch.zeros(2),
      }
    },
    "physics": {"arrays": {"qpos": torch.tensor([[float("nan")] * 19] * 2)}},
  }
  try:
    _validate_payload_finiteness(nan_payload)
  except CheckpointError as error:
    assert "qpos" in str(error)
  else:
    raise AssertionError("NaN physics accepted")


# ═══════════════════════ check-entry surface pinned ═══════════════════════


def test_current_source_identity_hashes_function_dependencies():
  from hexapod_mjlab.tasks.confined_locomotion.rl.checkpoint import _source_identity

  sources = _source_identity()
  required = {
    "hexapod_mjlab.tasks.confined_locomotion.runtime",
    "hexapod_mjlab.tasks.confined_locomotion.collector",
    "hexapod_mjlab.tasks.confined_locomotion.transition",
    "hexapod_mjlab.tasks.confined_locomotion.diagnostics",
    "hexapod_mjlab.tasks.confined_locomotion.rl.checkpoint",
    "hexapod_mjlab.tasks.confined_locomotion.rl.runner",
    "hexapod_mjlab.rl.confined_locomotion.inference",
  }
  assert required <= set(sources)
  assert all(len(digest) == 64 for digest in sources.values())


def test_runtime_feature_cache_is_absent_from_strict_weight_state_dict():
  from hexapod_mjlab.rl.confined_locomotion.inference import PointCloudFeatureCache

  network = build_network(seed=20261013, ground_samples=12)
  keys_before = set(network.state_dict())
  cache = PointCloudFeatureCache(network)
  cache.reset_stats()
  cache.clear()
  assert set(network.state_dict()) == keys_before
  assert not any("cache" in name for name in network.state_dict())


# ═══════════════════════ driver ═══════════════════════════════════════════


def main() -> int:
  tests = [
    (name, obj)
    for name, obj in sorted(globals().items())
    if name.startswith("test_") and callable(obj)
  ]
  failures = []
  for name, fn in tests:
    try:
      fn()
      print("PASS", name, flush=True)
    except BaseException as err:  # noqa: BLE001 — report and continue
      failures.append(name)
      print("FAIL", name, repr(err), flush=True)
      traceback.print_exc()
  print(f"{len(tests) - len(failures)}/{len(tests)} passed", flush=True)
  return 0 if not failures else 1


if __name__ == "__main__":
  raise SystemExit(main())
