"""Task-play CPU contracts: stdlib supplement (no pytest, no GPU claim).

Covers only what a CPU can honestly check without constructing the CUDA
play stack. This file does not cover the mean-closed-loop GPU run, live
viewer, natural fall/save/pause path, or long play clock. Those belong to the
explicit GPU play checks; the interactive CLI parser is checked here without
constructing the CUDA play stack.

Checks:

* play env cfg: the independent play object removes EXACTLY the
  episode_timeout term (three formal fall terms remain, no time_out flag,
  17 rewards, auto_reset=False, Null commands, nan_policy='error',
  episode_length_s stays 20.0 as recorded metadata); the training factory
  keeps its four-term set unchanged; a training cfg lacking the timeout
  term is refused by the play factory;
* checkpoint header peek: schema/kind/task rejections (non-dict, v1,
  unsupported artifact kind, missing M) and a REAL weights-only file peeked and
  reloaded on a fresh network (env-count independence is a GPU/check-mode
  matter; the loader contract itself is CPU-checkable);
* weights rejections: NaN state, same-M moved grid, tampered source hash
  (rejected without the explicit flag, accepted-and-recorded with it),
  wrong-M network;
* the confirmed manual-zero takeover order on the real scheduler: ALL rows
  set manual-zero BEFORE the first sample_reset (no random record is ever
  created, generators unconsumed, manual flags survive the reset, value
  zeroed by reset); a random-mode scheduler refuses manual overrides;
* the exact provenance mapping on synthetic windows: duplicate-coordinate
  points resolve to DISTINCT unique
  raw ids via the reproduced stable angle-key permutation — never by XYZ
  equality; invalid slots stay −1; a permutation mismatch raises instead
  of guessing; the clean geometric-fallback key path also maps exactly;
* GUI intent mailbox semantics (take clears, exit is sticky);
* CLI contracts: required explicit grid coordinates and the play CLI defaults
  (clean preset, 1 env, START PAUSED, no allow_source_change);
* the training runner's ``play()`` refusal points at the explicit task
  entry (message contract, no runner constructed).
"""
from __future__ import annotations

import importlib.util
import math
import sys
import tempfile
import traceback
import unittest.mock as mock
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
PLAY_CLI_PATH = (
    REPO_ROOT / "scripts" / "experiments" / "confined_locomotion" / "play.py"
)

_TEST_DIR = str(Path(__file__).resolve().parent)
if _TEST_DIR not in sys.path:
  sys.path.insert(0, _TEST_DIR)

from support.fixtures import build_adapter, build_network, build_provider, m12_training_grid

from hexapod_mjlab.rl.confined_locomotion.contracts import (
    EMPTY_SCAN_ID,
    OnlineObservation,
    ScanWindowSnapshot,
)
from hexapod_mjlab.rl.confined_locomotion.history import WindowAngleKeys
from hexapod_mjlab.rl.confined_locomotion.integration import prepare_control_inputs
from hexapod_mjlab.rl.confined_locomotion.network import ConfinedLocomotionNetwork
from hexapod_mjlab.rl.confined_locomotion.policy import ConfinedPolicyAdapter

from hexapod_mjlab.tasks.confined_locomotion.commands import (
    CommandContractError,
    TrainingCommandScheduler,
)
from hexapod_mjlab.tasks.confined_locomotion.play import (
  COMMAND_RANGES,
  PlayConfigError,
  _compare_compiled_model_identity,
  _slice_batch_payload,
  make_play_env_cfg,
  peek_checkpoint_header,
  ground_grid_from_checkpoint_header,
)
from hexapod_mjlab.tasks.confined_locomotion.visualization.play import (
    PendingControls,
    resolve_point_sources,
)
from hexapod_mjlab.tasks.confined_locomotion.rl.checkpoint import (
    CHECKPOINT_SCHEMA,
    CheckpointError,
    load_weights,
    save_weights_only,
)
from hexapod_mjlab.tasks.confined_locomotion.rl.runner import (
    TASK_PLAY_ENTRY_MSG,
)
from hexapod_mjlab.tasks.confined_locomotion.rl.runner import ConfinedLocomotionRunner
from hexapod_mjlab.tasks.confined_locomotion.runtime import _FORMAL_FALL_TERM_NAMES
from hexapod_mjlab.tasks.confined_locomotion.mdp.rewards import (
    TRAINING_REWARD_WEIGHTS,
)


def _make_adapter(seed: int = 7):
    network = build_network(seed=seed, ground_samples=m12_training_grid().num_points)
    provider = build_provider(seed=seed)
    return network, provider, build_adapter(network=network, provider=provider, seed=seed)


# ═════════════════════════ play env cfg ════════════════════════════════════


def test_play_env_cfg_removes_exactly_the_timeout_term():
  cfg = make_play_env_cfg(num_envs=1)
  assert set(cfg.terminations) == set(_FORMAL_FALL_TERM_NAMES)
  assert "episode_timeout" not in cfg.terminations
  for name, term in cfg.terminations.items():
    assert not bool(getattr(term, "time_out", False)), name
  assert cfg.auto_reset is False
  assert cfg.commands == {}
  assert cfg.observations["policy"].nan_policy == "error"
  assert len(cfg.rewards) == len(TRAINING_REWARD_WEIGHTS) == 17
  assert float(cfg.episode_length_s) == 20.0  # recorded metadata, inert in play
  assert int(cfg.scene.num_envs) == 1


def test_training_factory_keeps_four_term_set():
  from hexapod_mjlab.tasks.confined_locomotion.flat_env_cfg import make_training_env_cfg

  cfg = make_training_env_cfg(num_envs=1)
  assert set(cfg.terminations) == set(_FORMAL_FALL_TERM_NAMES) | {"episode_timeout"}


def test_play_factory_refuses_unexpected_cfg_shape():
  from hexapod_mjlab.tasks.confined_locomotion.flat_env_cfg import make_training_env_cfg

  cfg = make_training_env_cfg(num_envs=1)
  cfg.terminations.pop("episode_timeout")
  with mock.patch(
      "hexapod_mjlab.tasks.confined_locomotion.play.make_training_env_cfg",
      return_value=cfg,
  ):
    try:
      make_play_env_cfg(num_envs=1)
    except PlayConfigError:
      pass
    else:
      raise AssertionError("play factory accepted a cfg without the timeout term")


# ═════════════════════════ checkpoint header ═══════════════════════════════


def test_peek_rejects_non_checkpoints():
  with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    non_dict = root / "non_dict.pt"
    torch.save([1, 2, 3], non_dict)
    try:
      peek_checkpoint_header(non_dict)
    except PlayConfigError:
      pass
    else:
      raise AssertionError("non-dict payload accepted")

    v1 = root / "v1.pt"
    torch.save(
        {"schema": "confined_locomotion.checkpoint/v1",
         "kind": "confined_locomotion_weights_only"},
        v1,
    )
    try:
      peek_checkpoint_header(v1)
    except PlayConfigError as error:
      assert "v1" in str(error) or "schema" in str(error)
    else:
      raise AssertionError("v1 payload accepted")

    fault = root / "fault.pt"
    torch.save(
        {"schema": CHECKPOINT_SCHEMA,
         "kind": "unrecognized_nonresumable_kind"},
        fault,
    )
    try:
      peek_checkpoint_header(fault)
    except PlayConfigError:
      pass
    else:
      raise AssertionError("fault-scene kind accepted")

    missing_m = root / "missing_m.pt"
    torch.save(
        {"schema": CHECKPOINT_SCHEMA,
         "kind": "confined_locomotion_weights_only",
         "identity": {"ground_samples": None}},
        missing_m,
    )
    try:
      peek_checkpoint_header(missing_m)
    except PlayConfigError:
      pass
    else:
      raise AssertionError("missing M accepted")

    garbage = root / "garbage.pt"
    garbage.write_bytes(b"not a torch file at all")
    try:
      peek_checkpoint_header(garbage)
    except PlayConfigError:
      pass
    else:
      raise AssertionError("garbage file accepted")

    try:
      peek_checkpoint_header(root / "does_not_exist.pt")
    except PlayConfigError:
      pass
    else:
      raise AssertionError("missing file accepted")


def test_peek_real_weights_file_and_reload():
  network, _provider, adapter = _make_adapter()
  grid = m12_training_grid()
  with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "weights.pt"
    save_weights_only(
        path, adapter, grid_identity=grid.identity(), reason="CPU play contract"
    )
    header = peek_checkpoint_header(path)
    assert header["kind"] == "confined_locomotion_weights_only"
    assert header["ground_samples"] == int(grid.num_points) == 12
    assert header["grid_identity"] == grid.identity()
    assert ground_grid_from_checkpoint_header(header) == grid
    assert len(header["file_sha256"]) == 64
    # Auto-read is strict about the complete identity schema.
    # Weights-only headers carry no independent grid SHA: an entirely
    # forged-but-self-consistent grid cannot be detected at this step.
    bad_header = dict(header, grid_identity={
        **grid.identity(), "unexpected_field": "not part of GroundGridCfg/v1"
    })
    try:
      ground_grid_from_checkpoint_header(bad_header)
    except PlayConfigError:
      pass
    else:
      raise AssertionError("unknown checkpoint ground-grid identity field accepted")
    bad_count = dict(header, ground_samples=13)
    try:
      ground_grid_from_checkpoint_header(bad_count)
    except PlayConfigError:
      pass
    else:
      raise AssertionError("checkpoint ground-head M mismatch accepted")
    fresh, _, _ = _make_adapter(seed=8)
    report = load_weights(
        path, network=fresh, provider=None,
        expected_grid_identity=grid.identity(),
    )
    assert report["resumable"] is False
    assert report["log_std"] == "present_not_loaded(no provider given)"
    for key, value in network.state_dict().items():
      assert torch.equal(fresh.state_dict()[key], value), key


def test_weights_rejection_counterexamples():
  network, _provider, adapter = _make_adapter()
  grid = m12_training_grid()
  with tempfile.TemporaryDirectory() as tmp:
    good = Path(tmp) / "good.pt"
    save_weights_only(
        good, adapter, grid_identity=grid.identity(), reason="CPU play contract"
    )
    header = peek_checkpoint_header(good)
    state = {f"network.{k}": v.clone() for k, v in network.state_dict().items()}
    state["distribution_provider.log_std"] = (
      adapter.distribution_provider.log_std.detach().clone()
    )
    base_identity = {
        "source_sha256": header["source_sha256"],
        "distribution_identity_frozen": "x",
        "network_state_keys": sorted(state),
        "ground_samples": 12,
        "grid_identity": grid.identity(),
    }

    nan_state = dict(state)
    key = next(key for key in sorted(nan_state) if key.startswith("network."))
    nan_state[key] = nan_state[key].clone().fill_(float("nan"))
    nan_path = Path(tmp) / "nan.pt"
    torch.save(
      {"schema": CHECKPOINT_SCHEMA,
       "kind": "confined_locomotion_weights_only",
         "identity": dict(base_identity), "state": nan_state},
        nan_path,
    )
    try:
      load_weights(nan_path, network=network, provider=None)
    except CheckpointError:
      pass
    else:
      raise AssertionError("NaN weights accepted")

    from hexapod_mjlab.tasks.confined_locomotion.ground import (
        VALIDITY_GEOMETRY_ONLY_FLAT_TRAINING,
        GroundGridCfg,
    )

    moved = GroundGridCfg.from_ranges(
        x_range=(0.2, 0.9), y_range=(-0.3, 0.3), nx=4, ny=3,
        validity_mode=VALIDITY_GEOMETRY_ONLY_FLAT_TRAINING,
    )
    moved_identity = dict(base_identity)
    moved_identity["grid_identity"] = moved.identity()
    moved_path = Path(tmp) / "moved.pt"
    torch.save(
      {"schema": CHECKPOINT_SCHEMA,
       "kind": "confined_locomotion_weights_only",
         "identity": moved_identity, "state": state},
        moved_path,
    )
    try:
      load_weights(
          moved_path, network=network, provider=None,
          expected_grid_identity=grid.identity(),
      )
    except CheckpointError:
      pass
    else:
      raise AssertionError("same-M moved grid accepted")

    tampered_identity = dict(base_identity)
    tampered_identity["source_sha256"] = {"some/module.py": "0" * 64}
    tampered_path = Path(tmp) / "tampered.pt"
    torch.save(
      {"schema": CHECKPOINT_SCHEMA,
       "kind": "confined_locomotion_weights_only",
         "identity": tampered_identity, "state": state},
        tampered_path,
    )
    try:
      load_weights(tampered_path, network=network, provider=None)
    except CheckpointError as error:
      assert "allow_source_change" in str(error)
    else:
      raise AssertionError("tampered source accepted without the explicit flag")
    allowed = load_weights(
        tampered_path, network=network, provider=None,
        expected_grid_identity=grid.identity(), allow_source_change=True,
    )
    assert allowed["source_change_allowed"] is True

    small = ConfinedLocomotionNetwork(
        mlp_activation=torch.nn.ELU(), init_fn=None,
        envelope_mode="raw", ground_samples=6,
    )
    try:
      load_weights(good, network=small, provider=None)
    except CheckpointError:
      pass
    else:
      raise AssertionError("wrong-M network accepted")


# ═════════════════════════ manual-zero takeover order ══════════════════════


def test_manual_zero_before_first_sample_reset():
  scheduler = TrainingCommandScheduler(2, seed=11, device="cpu", allow_manual=True)
  generator_states = [g.get_state().clone() for g in scheduler._generators]
  # The EXACT play takeover order: manual-zero BEFORE the first sample_reset.
  scheduler.set_command(torch.tensor([0, 1]), torch.zeros(2, 3))
  records = scheduler.sample_reset(torch.tensor([0, 1]))
  assert all(r["reason"] == "reset_manual_zero" for r in records)
  assert torch.equal(scheduler.current_command(), torch.zeros(2, 3))
  assert bool(scheduler.manual_mask().all())
  reasons = [r["reason"] for r in scheduler.resample_records]
  assert set(reasons) <= {"manual_set", "reset_manual_zero"}, reasons
  current = [g.get_state() for g in scheduler._generators]
  assert all(
      torch.equal(a, b) for a, b in zip(current, generator_states)
  ), "manual takeover consumed a command random stream"
  # the 4 s period never resamples manual rows
  for _ in range(250):
    scheduler.step()
  assert torch.equal(scheduler.current_command(), torch.zeros(2, 3))
  assert not any(r["reason"] == "period" for r in scheduler.resample_records)


def test_random_mode_scheduler_refuses_manual():
  scheduler = TrainingCommandScheduler(1, seed=11, device="cpu")
  try:
    scheduler.set_command(torch.tensor([0]), torch.zeros(1, 3))
  except CommandContractError:
    pass
  else:
    raise AssertionError("random-mode scheduler accepted a manual override")


def test_command_ranges_are_confirmed_constants():
  assert COMMAND_RANGES == ((-1.5, 1.5), (-1.0, 1.0), (-1.0, 1.0))


def test_full_model_identity_allows_env_count_with_same_t1_reference():
  q_ref = torch.arange(18, dtype=torch.float64).reshape(1, 18).numpy()
  saved = {
    "dims": {"nq": 20, "nu": 18},
    "compiled_binary_sha256": "model-hash",
    "q_ref": q_ref.repeat(2, axis=0).tolist(),
    "digest": "batch-two-digest",
  }
  current = {
    "dims": {"nq": 20, "nu": 18},
    "compiled_binary_sha256": "model-hash",
    "q_ref": q_ref.tolist(),
    "digest": "batch-one-digest",
  }
  mismatches, q_ref_valid = _compare_compiled_model_identity(saved, current)
  assert mismatches == []
  assert q_ref_valid is True
  current["compiled_binary_sha256"] = "different-model"
  mismatches, q_ref_valid = _compare_compiled_model_identity(saved, current)
  assert mismatches == ["compiled_binary_sha256"]
  assert q_ref_valid is True


def test_display_labels_are_sliced_to_selected_env_row():
  labels = {
    "ground_heights": np.arange(12).reshape(2, 6),
    "envelope_label": np.arange(12).reshape(2, 6),
    "ground_metadata": [{"scan": 3}, {"scan": 8}],
    "grid_identity_hash": "shared-grid",
  }
  selected = _slice_batch_payload(labels, 1, 2)
  assert selected["ground_heights"].shape == (6,)
  assert selected["envelope_label"].shape == (6,)
  assert selected["ground_metadata"] == {"scan": 8}
  assert selected["grid_identity_hash"] == "shared-grid"


# ═════════════════════════ exact provenance mapping ════════════════════════


def _synthetic_window_and_prepared(*, duplicate_xyz: bool):
  torch.manual_seed(0)
  batch, near_n, slots, per = 2, 256, 10, 64
  near_valid = torch.zeros(batch, near_n, dtype=torch.bool)
  near_valid[0, :200] = True
  near_valid[1, :180] = True
  far_valid = torch.zeros(batch, slots, per, dtype=torch.bool)
  far_valid[0, :3, :60] = True   # scans 1..3 on env0
  far_valid[1, :1, :50] = True   # scan 1 on env1
  if duplicate_xyz:
    near_points = torch.zeros(batch, near_n, 3)
    near_points[near_valid] = torch.tensor([7.5, 0.0, 0.0])
    far_points = torch.zeros(batch, slots, per, 3)
    far_points[0, 0, :, 0] = 1.0
    far_points[0, 1, :, 0] = 2.0
    far_points[0, 2, :, 0] = 3.0
    far_points[1, 0, :, 0] = 4.0
    far_points[~far_valid] = 0.0  # invalid slots stay EXACTLY zero
  else:
    near_points = torch.zeros(batch, near_n, 3)
    ang = torch.arange(near_n, dtype=torch.float32) * (2 * math.pi / near_n)
    near_points[..., 0] = ang.cos().unsqueeze(0) * 5.0
    near_points[..., 1] = ang.sin().unsqueeze(0) * 5.0
    near_points[~near_valid] = 0.0
    far_points = torch.zeros(batch, slots, per, 3)
    base = torch.arange(per, dtype=torch.float32) * (2 * math.pi / per)
    for row in range(batch):
      for slot in range(slots):
        far_points[row, slot, :, 0] = base.cos() * (1.0 + slot)
        far_points[row, slot, :, 1] = base.sin() * (1.0 + slot)
    far_points[~far_valid] = 0.0

  near_keys = (torch.arange(near_n, dtype=torch.float32) * 0.01).repeat(batch, 1)
  far_keys = (
      torch.arange(slots * per, dtype=torch.float32) * 0.001
  ).view(1, slots, per).repeat(batch, 1, 1)

  far_filled = torch.zeros(batch, slots, dtype=torch.bool)
  far_filled[0, :3] = True
  far_filled[1, :1] = True
  far_scan_ids = torch.full((batch, slots), EMPTY_SCAN_ID, dtype=torch.long)
  far_scan_ids[0, :3] = torch.tensor([1, 2, 3])
  far_scan_ids[1, :1] = torch.tensor([1])
  far_times = torch.zeros(batch, slots, dtype=torch.float32)
  far_times[0, :3] = torch.tensor([0.0, 0.1, 0.2])
  far_times[1, :1] = torch.tensor([0.5])
  near_scan_ids = torch.full((batch,), EMPTY_SCAN_ID, dtype=torch.long)
  near_scan_ids[0] = 3
  near_scan_ids[1] = 1
  near_times = torch.zeros(batch, dtype=torch.float32)
  near_times[0] = 0.2
  near_times[1] = 0.5

  snapshot = ScanWindowSnapshot(
      env_ids=torch.arange(batch, dtype=torch.long),
      episode_ids=torch.zeros(batch, dtype=torch.long),
      near_points=near_points,
      near_valid=near_valid,
      near_filled=torch.ones(batch, dtype=torch.bool),
      near_scan_ids=near_scan_ids,
      near_times=near_times,
      far_points=far_points,
      far_valid=far_valid,
      far_filled=far_filled,
      far_scan_ids=far_scan_ids,
      far_times=far_times,
  )
  observation = OnlineObservation(
      env_ids=torch.arange(batch, dtype=torch.long),
      episode_ids=torch.zeros(batch, dtype=torch.long),
      step_ids=torch.ones(batch, dtype=torch.long),
      time_s=torch.tensor([0.25, 0.6], dtype=torch.float32),
      lin_vel_b=torch.zeros(batch, 3),
      ang_vel_b=torch.zeros(batch, 3),
      gravity_b=torch.tensor([[0.0, 0.0, -1.0]]).repeat(batch, 1),
      joint_pos_rel_ref=torch.zeros(batch, 18),
      joint_vel=torch.zeros(batch, 18),
      prev_action=torch.zeros(batch, 18),
      command=torch.zeros(batch, 3),
  )
  prepared = prepare_control_inputs(
      window=snapshot,
      observation=observation,
      near_angle_keys=near_keys,
      far_angle_keys=far_keys.reshape(batch, slots * per),
  )
  keys = WindowAngleKeys(
      near=near_keys * near_valid,  # window stores 0 at invalid slots
      far=far_keys * far_valid,
      present=torch.ones(batch, dtype=torch.bool),
  )
  return snapshot, keys, prepared


def _capture_entries_env0():
  entries = []
  for scan in (1, 2, 3):
    entries.append({
        "env": 0, "episode": 0, "scan": scan,
        "persistent_scan_count": scan, "time_s": 0.1 * (scan - 1),
        "near_indices": (10000 + torch.arange(256)) + scan,
        "near_valid": torch.zeros(256, dtype=torch.bool),
        "far_indices": (50000 + scan * 100 + torch.arange(64)),
        "far_valid": torch.zeros(64, dtype=torch.bool),
        "near_angle_keys": None, "far_angle_keys": None,
        "near_no_hit": (torch.arange(256) % 3 == 0),
        "far_no_hit": (torch.arange(64) % 5 == 0),
    })
  for entry in entries:
    entry["near_valid"][:200] = True
    entry["far_valid"][:60] = True
  return entries


def test_provenance_maps_duplicates_without_xyz_guessing():
  snapshot, keys, prepared = _synthetic_window_and_prepared(duplicate_xyz=True)
  entries = _capture_entries_env0()
  result = resolve_point_sources(
      snapshot=snapshot, angle_keys=keys, geometric_keys_fn=None,
      prepared=prepared, capture_entries=entries, env_row=0, episode_id=0,
  )
  near_valid = result["near_valid"]
  assert int(near_valid.sum()) == 200
  assert result["near_matched"] == 200
  raw_ids = result["near_raw_ids"][near_valid]
  # THE contract: every valid point keeps a DISTINCT unique raw id even
  # though ALL valid near points share the identical (7.5, 0, 0) xyz —
  # XYZ equality can never provide this mapping.
  assert int(torch.unique(raw_ids).numel()) == 200
  if bool((~near_valid).any()):
    assert int(result["near_raw_ids"][~near_valid].max()) == -1
  # exact expected permutation recomputed independently in the test
  near_keys_row = keys.near[0]
  effective = torch.where(
      snapshot.near_valid[0], near_keys_row,
      torch.full_like(near_keys_row, float("inf")),
  )
  order = torch.argsort(effective, stable=True)
  near_valid_sorted = snapshot.near_valid[0].index_select(0, order)
  entry_near = {e["scan"]: e for e in entries}[3]
  expected = torch.where(
      near_valid_sorted,
      entry_near["near_indices"].index_select(0, order),
      torch.full_like(order, -1),
  )
  assert torch.equal(result["near_raw_ids"], expected)
  assert torch.equal(
      result["near_no_hit"][near_valid],
      entry_near["near_no_hit"].index_select(0, order)[near_valid],
  )

  far_valid = result["far_valid"]
  assert int(far_valid.sum()) == 180
  assert result["far_matched"] == 180
  far_raw = result["far_raw_ids"][far_valid]
  assert int(torch.unique(far_raw).numel()) == 180
  source_scans = result["far_source_scan_ids"][far_valid]
  slot_of_valid = result["far_source_slot"][far_valid]
  for scan in (1, 2, 3):
    selected = source_scans == scan
    assert bool(selected.any())
    assert bool((slot_of_valid[selected] == scan - 1).all())
  if bool((~far_valid).any()):
    assert int(result["far_raw_ids"][~far_valid].min()) == -1


def test_provenance_env_row_one():
  snapshot, keys, prepared = _synthetic_window_and_prepared(duplicate_xyz=True)
  entries = [{
      "env": 1, "episode": 0, "scan": 1,
      "persistent_scan_count": 1, "time_s": 0.5,
      "near_indices": 30000 + torch.arange(256),
      "near_valid": torch.zeros(256, dtype=torch.bool),
      "far_indices": 60000 + torch.arange(64),
      "far_valid": torch.zeros(64, dtype=torch.bool),
      "near_angle_keys": None, "far_angle_keys": None,
      "near_no_hit": torch.zeros(256, dtype=torch.bool),
      "far_no_hit": torch.ones(64, dtype=torch.bool),
  }]
  entries[0]["near_valid"][:180] = True
  entries[0]["far_valid"][:50] = True
  result = resolve_point_sources(
      snapshot=snapshot, angle_keys=keys, geometric_keys_fn=None,
      prepared=prepared, capture_entries=entries, env_row=1, episode_id=0,
  )
  assert result["near_valid_count"] == 180
  assert result["near_matched"] == 180
  assert result["far_valid_count"] == 50
  assert result["far_matched"] == 50
  assert bool(result["far_no_hit"][result["far_valid"]].all())


def test_provenance_refuses_permutation_mismatch():
  # Non-duplicate points: a WRONG key set reorders them differently, so the
  # reproduced permutation no longer matches the prepared points — the
  # mapping must refuse instead of guessing. (With all-identical points any
  # permutation is indistinguishable; the refusal needs distinguishable
  # points, which is exactly when guessing would be most dangerous.)
  snapshot, keys, prepared = _synthetic_window_and_prepared(duplicate_xyz=False)
  entries = _capture_entries_env0()
  wrong_keys = WindowAngleKeys(
      near=torch.flip(keys.near, dims=[1]) * snapshot.near_valid,
      far=torch.flip(keys.far, dims=[2]) * snapshot.far_valid,
      present=torch.ones(2, dtype=torch.bool),
  )
  try:
    resolve_point_sources(
        snapshot=snapshot, angle_keys=wrong_keys, geometric_keys_fn=None,
        prepared=prepared, capture_entries=entries, env_row=0, episode_id=0,
    )
  except RuntimeError as error:
    assert "permutation" in str(error)
  else:
    raise AssertionError("a mismatched permutation was guessed through")


def test_provenance_geometric_fallback_path():
  snapshot, _keys, prepared = _synthetic_window_and_prepared(duplicate_xyz=False)
  entries = _capture_entries_env0()

  def geometric(points):
    return torch.atan2(points[..., 1], points[..., 0])

  # The fallback must reproduce EXACTLY the keys prepare_control_inputs
  # used, i.e. the same geometric function on the same points.
  near_prepared_keys = geometric(snapshot.near_points)
  far_prepared_keys = geometric(snapshot.far_points.reshape(2, -1, 3))
  prepared2 = prepare_control_inputs(
      window=snapshot,
      observation=prepared.observation,
      near_angle_keys=near_prepared_keys,
      far_angle_keys=far_prepared_keys.reshape(2, -1),
  )
  result = resolve_point_sources(
      snapshot=snapshot, angle_keys=None, geometric_keys_fn=geometric,
      prepared=prepared2, capture_entries=entries, env_row=0, episode_id=0,
  )
  assert result["near_matched"] == 200
  assert result["far_matched"] == 180


# ═════════════════════════ GUI mailbox / CLI / runner message ══════════════


def test_pending_controls_take_and_sticky_exit():
  box = PendingControls()
  assert box.take() == {
      "command": None, "zero_command": False, "pause_toggle": False, "step": False,
      "reset": False, "exit": False, "rerender": False, "pick": None,
      "env_row": None,
  }
  box.submit_command(0.5, 0.0, 0.0)
  box.submit_step()
  box.submit_pick("near", 3)
  box.submit_pause_toggle()
  box.submit_env_row(1)
  taken = box.take()
  assert taken["command"] == (0.5, 0.0, 0.0)
  assert taken["step"] is True
  assert taken["pick"] == ("near", 3)
  assert taken["pause_toggle"] is True
  assert taken["env_row"] == 1
  assert box.take()["step"] is False  # take clears
  box.submit_exit()
  taken = box.take()
  assert taken["exit"] is True
  assert not box.exit_requested()  # consumed by the owner


def _load_script(path: Path):
  spec = importlib.util.spec_from_file_location(path.stem, path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def test_play_cli_defaults():
  module = _load_script(PLAY_CLI_PATH)
  common = ["--checkpoint", "ck.pt", "--output", "out"]
  args = module._parse_args(common + [
      "--grid-x", "0.1", "0.8", "--grid-y", "-0.3", "0.0", "0.3",
  ])
  assert args.preset == "clean"
  assert args.num_envs == 1
  assert args.grid_x == [0.1, 0.8]
  assert args.grid_y == [-0.3, 0.0, 0.3]
  assert args.parallel_tol is None  # automatic checkpoint-grid case
  assert args.start_running is False  # START PAUSED by default
  assert args.allow_source_change is False
  assert args.max_steps == 0
  assert args.host == "127.0.0.1"
  assert args.vx == args.vy == args.wz == 0.0
  explicit = module._parse_args([
      *common,
      "--grid-x", "0.1", "0.8", "--grid-y", "-0.3", "0.0", "0.3",
      "--parallel-tol", "2e-6", "--allow-source-change",
  ])
  assert explicit.allow_source_change is True
  assert explicit.parallel_tol == 2e-6
  preset = module._parse_args(common + ["--grid-preset", "s1_187"])
  assert preset.grid_preset == "s1_187"
  assert preset.grid_x is None and preset.grid_y is None
  baseline = module._parse_args(common)
  assert baseline.grid_preset is None and baseline.grid_x is None and baseline.grid_y is None
  assert baseline.parallel_tol is None
  from hexapod_mjlab.tasks.confined_locomotion.ground_presets import resolve_task_ground_grid
  assert resolve_task_ground_grid(
      grid_preset=baseline.grid_preset,
      grid_x=baseline.grid_x, grid_y=baseline.grid_y
  ).num_points == 187
  for params in (
      common + ["--grid-preset", "s1_187", "--grid-x", "0.8", "3.2"],
      common + ["--grid-x", "0.8", "3.2"],
      common + ["--parallel-tol", "2e-6"],
  ):
    try:
      module._parse_args(params)
    except SystemExit:
      pass
    else:
      raise AssertionError(f"play CLI accepted incomplete/mixed ground grid: {params}")


def test_runner_play_refusal_points_to_task_entry():
  try:
    ConfinedLocomotionRunner.play(None)
  except RuntimeError as error:
    assert "PlaySession" in str(error)
  else:
    raise AssertionError("runner.play() did not refuse")
  assert "PlaySession" in TASK_PLAY_ENTRY_MSG


# ═════════════════════════ stdlib 驱动（无 pytest） ═════════════════════════


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
