"""Perturbation contracts: stdlib CPU supplement (no GPU claim).

Covers CPU-checkable perturbation contracts (no pytest dependency;
plain ``test_*`` functions + a stdlib ``main()`` driver). Real CUDA collection
and resume behavior requires separately selected GPU checks; these CPU
contracts do not claim that path. A natural fall there is an honest NOT PASSED,
not a contract failure.

Checks (deliberately including counterexamples — malformed recipes are
rejected, malformed states are rejected, wrong edge angles are excluded):

* typed recipes: asdict→from_dict roundtrip (lists→tuples tolerated), exact
  preset identities (gap sector [74°, 106°), σ=0.015 m, proprio
  half-amplitudes/level), and rejections (bad jitter probabilities, wrapping
  sector, unknown keys, wrong clip order, non-tuple amplitudes);
* gap sector mask: the half-open edges (74.0 in, 105.999 in, 106.0 out,
  73.999 out) on synthetic optical unit rays;
* gap schedule (pure, no env): events fire at φ+10k ± 1 with 8–12 scan
  intervals, anchors never drift (nominal == φ+10k for every recorded
  event), the φ=1/ε=−1 frame-0 event is recorded as a creation-boundary skip
  and never fired, jitter-off fixes ε=0 without consuming the jitter stream,
  the phase is drawn once per construction, and a long clock interleaved
  with "resets" (which have NO hook here) keeps the identical stream/schedule;
* stream discipline: gap/range/proprio seeds are mutually distinct and
  distinct from the command-scheduler derivation; disabled perturbations
  consume nothing (absent streams, None draw counts);
* range consumption rule (pure): clamp 0/15, no-hit rays keep the EXACT
  max-range value, gap-excluded rays get no applied noise, the applied mask
  is exactly hit ∧ pool-valid, and d_obs = 0 keeps the ORIGINAL direction
  key/group (direction-based keys, never a noisy-xyz radius);
* range draws: per-env reproducibility and independence across envs;
* proprio: bounded half-amplitudes × level, gravity re-normalization,
  command/prev_action untouched, per-row identity cache (repeated apply at
  one (env, episode, step) reuses the noise bit-exactly with NO redraw, a
  new episode identity redraws exactly that row, the unselected row of a
  selected reset stays bit-exact), and a clean-value-mismatch guard;
* checkpoint-ability: state_dict roundtrip preserves stream positions
  (next draws identical) and gap decisions; a mismatching recipe refuses;
* window angle keys: write/read roundtrip with the same far-slot rotation as
  ``read()``, per-row presence with selected-reset clearing ONLY that row,
  no-key writes clearing presence, and contract rejections (one-sided keys,
  non-finite keys at valid slots);
* schema: the checkpoint schema is v3 and v1/v2 payloads are rejected by
  ``validate_payload_envelope`` (no undeclared migration);
* rl_cfg: the registered default keeps ``observation_perturbations=None``
  (clean) and the engineering factory accepts/records an explicit recipe;
* scanner frame contract: consumed-copy audit fields exist and the
  direction-derived keys equal the point-derived keys at positive distances.
"""

from __future__ import annotations

import dataclasses
import math
import traceback

import torch

from hexapod_mjlab.rl.confined_locomotion.contracts import (
  FAR_POINTS_PER_SCAN,
  NEAR_POINTS,
  OnlineObservation,
)
from hexapod_mjlab.rl.confined_locomotion.history import ScanWindow
from hexapod_mjlab.common.perturbations import (
  ALL_THREE_PRESET,
  GAP_FIXED_JITTER_OFF_PRESET,
  GAP_ONLY_PRESET,
  PROPRIO_ONLY_PRESET,
  RANGE_ONLY_PRESET,
  GapPerturbationCfg,
  ObservationPerturbationCfg,
  ObservationPerturbations,
  PerturbationContractError,
  ProprioPerturbationCfg,
  RangePerturbationCfg,
  apply_range_noise,
  derive_perturbation_generator_seed,
  gap_sector_mask,
  preset_by_name,
)
from hexapod_mjlab.tasks.confined_locomotion.scanner import (
  ScanFrame,
  direction_angle_keys,
  sensor_frame_angle_keys,
)

_SEED = 20261008


def _cfg_gap_jitter_on() -> ObservationPerturbationCfg:
  return ObservationPerturbationCfg(
    gap=GapPerturbationCfg(), range=None, proprio=None
  )


def _dir_at_azimuth(deg: float) -> torch.Tensor:
  rad = math.radians(deg)
  return torch.tensor([[math.cos(rad), math.sin(rad), 0.0]], dtype=torch.float32)


def _make_observation(rows, *, device=torch.device("cpu")) -> OnlineObservation:
  def field(seed: int, shape: tuple, normalize: bool = False) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    base = torch.randn(*((len(rows),) + shape), generator=generator)
    if normalize:
      base = torch.nn.functional.normalize(base, dim=-1)
    return base.to(device)

  return OnlineObservation(
    env_ids=torch.arange(len(rows), dtype=torch.int64, device=device),
    episode_ids=torch.tensor([e for e, _ in rows], dtype=torch.int64, device=device),
    step_ids=torch.tensor([s for _, s in rows], dtype=torch.int64, device=device),
    time_s=torch.zeros(len(rows), dtype=torch.float32, device=device),
    lin_vel_b=field(11, (3,)),
    ang_vel_b=field(12, (3,)),
    gravity_b=field(13, (3,), normalize=True),
    joint_pos_rel_ref=field(14, (18,)),
    joint_vel=field(15, (18,)),
    prev_action=field(16, (18,)),
    command=field(17, (3,)),
  )


# ── typed recipes ───────────────────────────────────────────────────────────


def test_cfg_roundtrip_asdict_from_dict() -> None:
  for preset in (GAP_ONLY_PRESET, RANGE_ONLY_PRESET, PROPRIO_ONLY_PRESET,
                 GAP_FIXED_JITTER_OFF_PRESET, ALL_THREE_PRESET):
    payload = dataclasses.asdict(preset)
    # asdict preserves tuples; a JSON round trip converts them to lists.
    import json
    payload = json.loads(json.dumps(payload))
    restored = ObservationPerturbationCfg.from_dict(payload)
    assert restored.identity() == preset.identity()
    assert restored == preset
  clean = ObservationPerturbationCfg.from_dict({"gap": None, "range": None, "proprio": None})
  assert clean.is_clean


def test_cfg_rejections() -> None:
  def expect_error(build):
    try:
      build()
    except PerturbationContractError:
      return
    raise AssertionError("malformed recipe was accepted")

  expect_error(lambda: GapPerturbationCfg(jitter_probabilities=(0.5, 0.4, 0.1)))
  expect_error(lambda: GapPerturbationCfg(azimuth_half_span_deg=0.0))
  expect_error(lambda: GapPerturbationCfg(azimuth_center_deg=350.0))  # wraps
  expect_error(lambda: GapPerturbationCfg(period_scans=1))
  expect_error(lambda: RangePerturbationCfg(sigma_m=0.0))
  expect_error(lambda: RangePerturbationCfg(clip_min_m=5.0, clip_max_m=1.0))
  expect_error(lambda: ProprioPerturbationCfg(noise_level=-1.0))
  expect_error(lambda: ProprioPerturbationCfg(half_amplitudes=(0.1, 0.2, 0.05)))
  expect_error(lambda: ObservationPerturbationCfg.from_dict({"gap": {"bad": 1}}))
  expect_error(lambda: ObservationPerturbationCfg.from_dict({"unknown": None}))
  expect_error(lambda: preset_by_name("does_not_exist"))


def test_preset_confirmed_values() -> None:
  gap = GAP_ONLY_PRESET.gap
  assert gap.sector == (74.0, 106.0)
  assert gap.period_scans == 10
  assert gap.jitter_probabilities == (0.1, 0.8, 0.1)
  assert GAP_FIXED_JITTER_OFF_PRESET.gap.jitter_enabled is False
  rng = RANGE_ONLY_PRESET.range
  assert rng.sigma_m == 0.015 and rng.clip_min_m == 0.0 and rng.clip_max_m == 15.0
  proprio = PROPRIO_ONLY_PRESET.proprio
  assert proprio.noise_level == 1.0
  assert proprio.identity()["half_amplitudes"] == {
    "lin_vel_b": .1, "ang_vel_b": .2, "gravity_b": .05,
    "joint_pos_rel_ref": .05, "joint_vel": 1.5,
  }
  assert preset_by_name("clean") is None
  assert preset_by_name("all") == ALL_THREE_PRESET


# ── gap sector mask ─────────────────────────────────────────────────────────


def test_gap_sector_half_open_edges() -> None:
  dirs = torch.cat(
    [_dir_at_azimuth(a) for a in (0.0, 73.999, 74.0, 90.0, 105.999, 106.0, 106.001)]
  )
  mask = gap_sector_mask(dirs)
  expected = [False, False, True, True, True, False, False]
  assert mask.tolist() == expected, mask.tolist()


def test_gap_sector_mask_rejects_bad_shapes() -> None:
  try:
    gap_sector_mask(torch.zeros(5))
  except PerturbationContractError:
    return
  raise AssertionError("(5,) dirs accepted")


# ── gap schedule (pure) ─────────────────────────────────────────────────────


def _run_long_clock(cfg: ObservationPerturbationCfg, scans: int, num_envs: int = 2):
  perturbations = ObservationPerturbations(
    num_envs, cfg=cfg, seed=_SEED, device="cpu"
  )
  fired = [[] for _ in range(num_envs)]
  for number in range(1, scans + 1):
    for row in range(num_envs):
      if perturbations.begin_scan(row, number):
        fired[row].append(number)
  return perturbations, fired


def test_gap_schedule_events_within_pm1_and_intervals() -> None:
  perturbations, fired = _run_long_clock(GAP_ONLY_PRESET, scans=120)
  period = 10
  for row in range(2):
    schedule = perturbations.gap_schedule_record()[row]
    phase = schedule["phase"]
    assert 1 <= phase <= 10
    fired_events = [e for e in schedule["events"] if not e["skipped"]]
    assert fired_events == [e for e in schedule["events"] if e["fired_scan"] is not None]
    for event in schedule["events"]:
      assert event["nominal"] == phase + period * event["k"]
      if event["skipped"]:
        assert event["k"] == 0 and phase == 1 and event["epsilon"] == -1
        assert event["reason"] == "already_passed_before_first_scan"
      else:
        assert event["nominal"] - 1 <= event["fired_scan"] <= event["nominal"] + 1
    assert len(fired[row]) >= 10  # ~12 events over 120 scans
    for a, b in zip(fired[row], fired[row][1:]):
      assert period - 2 <= b - a <= period + 2, (a, b)


def test_gap_schedule_jitter_off_fires_exactly_at_nominal() -> None:
  perturbations = ObservationPerturbations(
    2, cfg=GAP_FIXED_JITTER_OFF_PRESET, seed=_SEED, device="cpu"
  )
  fired = [[] for _ in range(2)]
  for number in range(1, 121):
    for row in range(2):
      if perturbations.begin_scan(row, number):
        fired[row].append(number)
  for row in range(2):
    schedule = perturbations.gap_schedule_record()[row]
    phase = schedule["phase"]
    for event in schedule["events"]:
      if event["skipped"]:
        continue
      assert event["epsilon"] == 0
      assert event["fired_scan"] == event["nominal"] == phase + 10 * event["k"]
    # with the φ=1 skip the first fired event is at 11; otherwise at φ
    assert fired[row][0] == phase or (phase == 1 and fired[row][0] == 11)
    assert schedule["jitter_draws"] == 0  # no jitter draws with jitter disabled


def test_gap_phase_drawn_once_and_streams_isolated() -> None:
  a = ObservationPerturbations(2, cfg=ALL_THREE_PRESET, seed=_SEED, device="cpu")
  b = ObservationPerturbations(2, cfg=ALL_THREE_PRESET, seed=_SEED, device="cpu")
  for row in range(2):
    assert (
      a.gap_schedule_record()[row]["phase"]
      == b.gap_schedule_record()[row]["phase"]
    )
  # consuming OTHER streams never advances the gap stream
  observation = _make_observation([(1, 0), (1, 0)])
  a.apply_proprio_noise(observation, control_step=0)
  a.draw_range_epsilon(0, 128)
  a.draw_range_epsilon(1, 128)
  gap_state_a = a.state_dict()["gap_state"]
  gap_state_b = b.state_dict()["gap_state"]
  for row in range(2):
    assert gap_state_a[row]["phase"] == gap_state_b[row]["phase"]
    assert (
      gap_state_a[row]["next_event_index"]
      == gap_state_b[row]["next_event_index"]
    )
  assert torch.equal(
    a.state_dict()["streams"]["gap"][0], b.state_dict()["streams"]["gap"][0]
  )


def test_gap_begin_scan_validation() -> None:
  perturbations = ObservationPerturbations(
    1, cfg=GAP_ONLY_PRESET, seed=_SEED, device="cpu"
  )
  for bad_call in (lambda: perturbations.begin_scan(0, 0),
                   lambda: perturbations.begin_scan(0, -3),
                   lambda: perturbations.begin_scan(5, 1)):
    try:
      bad_call()
    except PerturbationContractError:
      continue
    raise AssertionError("invalid begin_scan accepted")
  # episode "resets" have NO hook: the persistent numbering simply continues
  # and an identically-seeded twin produces the identical decision sequence
  decisions = [perturbations.begin_scan(0, n) for n in range(1, 12)]
  twin = ObservationPerturbations(1, cfg=GAP_ONLY_PRESET, seed=_SEED, device="cpu")
  twin_decisions = [twin.begin_scan(0, n) for n in range(1, 12)]
  assert decisions == twin_decisions


def test_disabled_streams_consume_nothing() -> None:
  proprio_only = ObservationPerturbations(
    2, cfg=PROPRIO_ONLY_PRESET, seed=_SEED, device="cpu"
  )
  state = proprio_only.state_dict()
  assert state["streams"]["gap"] is None and state["streams"]["range"] is None
  assert state["gap_state"] is None and state["range_draw_counts"] is None
  assert proprio_only.draw_counts()["gap_jitter"] is None
  for method, args in (
    (proprio_only.draw_range_epsilon, (0, 16)),
  ):
    try:
      method(*args)
    except PerturbationContractError:
      continue
    raise AssertionError("disabled range stream drew")
  gap_only = ObservationPerturbations(
    2, cfg=GAP_ONLY_PRESET, seed=_SEED, device="cpu"
  )
  try:
    gap_only.apply_proprio_noise(_make_observation([(1, 0), (1, 0)]), control_step=0)
  except PerturbationContractError:
    raise AssertionError("proprio-off apply should be a no-op, not an error")
  assert gap_only.draw_counts()["proprio"] is None


def test_stream_seeds_distinct() -> None:
  from hexapod_mjlab.tasks.confined_locomotion.commands import (
    derive_command_generator_seed,
  )

  for row in range(3):
    seeds = {
      derive_perturbation_generator_seed(_SEED, row, name)
      for name in ("gap", "range", "proprio")
    }
    seeds.add(derive_command_generator_seed(_SEED, row))
    assert len(seeds) == 4


# ── range consumption (pure) ────────────────────────────────────────────────


def test_range_apply_clamps_and_masks() -> None:
  r = 32
  raw = torch.tensor([10.0] * r, dtype=torch.float32)
  no_hit = torch.zeros(r, dtype=torch.bool)
  no_hit[3] = True  # one miss
  valid = torch.ones(r, dtype=torch.bool)
  valid[7] = False  # one gap-excluded ray (a hit)
  epsilon = torch.zeros(r, dtype=torch.float32)
  epsilon[:] = 0.5
  epsilon[0] = -20.0  # clamp to 0
  epsilon[1] = 20.0   # clamp to 15
  d_obs, applied = apply_range_noise(raw, no_hit, valid, epsilon)
  assert applied.tolist() == ((~no_hit) & valid).tolist()
  assert d_obs[0] == 0.0 and d_obs[1] == 15.0
  assert d_obs[3] == 10.0  # miss: raw value kept EXACTLY (no noise applied)
  assert d_obs[7] == 10.0  # gap-excluded: raw value kept
  assert d_obs[2] == 10.5
  # the max-range representation of misses stays exact under noise
  raw_miss = torch.full((r,), 15.0)
  epsilon_miss = torch.full((r,), -0.3)
  d2, applied2 = apply_range_noise(raw_miss, torch.ones(r, dtype=torch.bool), valid, epsilon_miss)
  assert bool((d2 == 15.0).all()) and not bool(applied2.any())
  # epsilon=None (gap-only): d_obs is a plain copy, mask still reported
  d3, applied3 = apply_range_noise(raw, no_hit, valid, None)
  assert torch.equal(d3, raw) and bool(applied3.any())


def test_range_apply_rejects_malformed() -> None:
  raw = torch.zeros(4, dtype=torch.float32)
  masks = torch.zeros(4, dtype=torch.bool)
  for args in (
    (raw, masks, torch.zeros(3, dtype=torch.bool), None),
    (raw.double(), masks, masks, None),
    (raw, masks.to(torch.int64), masks, None),
    (raw, masks, masks, torch.zeros(5, dtype=torch.float32)),
  ):
    try:
      apply_range_noise(*args)
    except PerturbationContractError:
      continue
    raise AssertionError("malformed apply_range_noise accepted")


def test_range_epsilon_reproducible_and_per_env_independent() -> None:
  a = ObservationPerturbations(2, cfg=RANGE_ONLY_PRESET, seed=_SEED, device="cpu")
  b = ObservationPerturbations(2, cfg=RANGE_ONLY_PRESET, seed=_SEED, device="cpu")
  for _ in range(3):
    for row in (0, 1):
      assert torch.equal(
        a.draw_range_epsilon(row, 64), b.draw_range_epsilon(row, 64)
      )
  # env0-only draws never change env1's stream position
  fresh = ObservationPerturbations(2, cfg=RANGE_ONLY_PRESET, seed=_SEED, device="cpu")
  # Compare env1 at the SAME draw index, while advancing only env0 extra.
  for _ in range(3):
    fresh.draw_range_epsilon(1, 64)
  fresh.draw_range_epsilon(0, 64)
  assert torch.equal(
    a.draw_range_epsilon(1, 64), fresh.draw_range_epsilon(1, 64)
  )


def test_zero_distance_keeps_direction_keys() -> None:
  generator = torch.Generator().manual_seed(3)
  dirs = torch.nn.functional.normalize(
    torch.randn(64, 3, generator=generator), dim=-1
  )
  raw = torch.full((64,), 8.0)
  no_hit = torch.zeros(64, dtype=torch.bool)
  valid = torch.ones(64, dtype=torch.bool)
  epsilon = torch.full((64,), -8.0)  # everything saturates to 0
  d_obs, applied = apply_range_noise(raw, no_hit, valid, epsilon)
  assert bool((d_obs == 0.0).all())
  keys = direction_angle_keys(dirs)
  assert bool(torch.isfinite(keys).all())
  # equivalence at POSITIVE distances with the point-based definition
  mount_pos = torch.zeros(1, 3)
  mount_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
  points = (raw.unsqueeze(-1) * dirs).unsqueeze(0) + mount_pos
  keys_points = sensor_frame_angle_keys(points, mount_pos, mount_quat)[0]
  assert torch.allclose(keys, keys_points, atol=1e-4, rtol=1e-5)
  # a zero point must NOT be used to derive a key: its "key" would be the
  # mount-origin direction — the contract consumes `dirs`, not points


# ── proprio ─────────────────────────────────────────────────────────────────


def test_proprio_bounds_and_gravity_normalization() -> None:
  perturbations = ObservationPerturbations(
    2, cfg=PROPRIO_ONLY_PRESET, seed=_SEED, device="cpu"
  )
  halves = torch.tensor([0.1, 0.2, 0.05, 0.05, 1.5]).repeat_interleave(
    torch.tensor([3, 3, 3, 18, 18])
  )
  gravity_slice = slice(6, 9)  # 45-layout: lin(0:3) ang(3:6) gravity(6:9)
  for step in range(30):
    observation = _make_observation([(1, step), (1, step)])
    perturbations.apply_proprio_noise(observation, control_step=step)
    for row, record in enumerate(perturbations.proprio_records()):
      clean = record["clean"]
      noise = record["noise"]  # the PRE-normalization additive draw
      noisy = record["noisy"]
      assert bool((noise.abs() <= halves * 1.0 + 1e-6).all()), noise.abs().max()
      # non-gravity components are the pure additive draw
      torch.testing.assert_close(
        torch.cat([noisy[:6], noisy[9:]]),
        torch.cat([clean[:6], clean[9:]]) + torch.cat([noise[:6], noise[9:]]),
      )
      # gravity: additive draw THEN re-normalization (unit norm kept)
      assert abs(float(noisy[gravity_slice].norm()) - 1.0) <= 1e-5
      expected_g = torch.nn.functional.normalize(
        clean[gravity_slice] + noise[gravity_slice], dim=0
      )
      torch.testing.assert_close(noisy[gravity_slice], expected_g)
    assert torch.allclose(
      observation.gravity_b.norm(dim=-1), torch.ones(2), atol=1e-5
    )


def test_proprio_cache_identity_and_selected_reset_semantics() -> None:
  perturbations = ObservationPerturbations(
    2, cfg=PROPRIO_ONLY_PRESET, seed=_SEED, device="cpu"
  )
  first = _make_observation([(1, 0), (5, 0)])
  perturbations.apply_proprio_noise(first, control_step=0)
  draws = perturbations.draw_counts()["proprio"]
  # repeated apply at the SAME identity: bit-exact reuse, NO redraw
  repeated = _make_observation([(1, 0), (5, 0)])
  perturbations.apply_proprio_noise(repeated, control_step=0)
  assert perturbations.draw_counts()["proprio"] == draws
  for name in ("lin_vel_b", "ang_vel_b", "gravity_b",
               "joint_pos_rel_ref", "joint_vel"):
    assert torch.equal(getattr(first, name), getattr(repeated, name))
  # a clean-value mismatch at the same identity is a contract fault
  tampered = _make_observation([(1, 0), (5, 0)])
  tampered.lin_vel_b = tampered.lin_vel_b + 1.0
  try:
    perturbations.apply_proprio_noise(tampered, control_step=0)
  except PerturbationContractError:
    pass
  else:
    raise AssertionError("clean-mismatch at a cached identity accepted")
  # selected reset (env0 new episode): env0 redraws, env1 bit-exact
  after = _make_observation([(2, 0), (5, 0)])
  # A partial reset does not advance the global control clock.
  perturbations.apply_proprio_noise(after, control_step=0)
  assert perturbations.draw_counts()["proprio"] == [draws[0] + 1, draws[1]]
  assert torch.equal(after.lin_vel_b[1], first.lin_vel_b[1])
  assert not torch.equal(after.lin_vel_b[0], first.lin_vel_b[0])
  # command / prev_action never touched
  pristine = _make_observation([(2, 0), (5, 0)])
  assert torch.equal(after.command, pristine.command)
  assert torch.equal(after.prev_action, pristine.prev_action)


def test_proprio_disabled_is_noop() -> None:
  perturbations = ObservationPerturbations(
    1, cfg=GAP_ONLY_PRESET, seed=_SEED, device="cpu"
  )
  observation = _make_observation([(1, 0)])
  before = observation.lin_vel_b.clone()
  perturbations.apply_proprio_noise(observation, control_step=0)
  assert torch.equal(observation.lin_vel_b, before)


# ── checkpoint state ────────────────────────────────────────────────────────


def test_state_dict_roundtrip_continuation_bitwise() -> None:
  live = ObservationPerturbations(2, cfg=ALL_THREE_PRESET, seed=_SEED, device="cpu")
  for number in range(1, 26):
    for row in range(2):
      live.begin_scan(row, number)
  observation = _make_observation([(1, 3), (1, 3)])
  live.apply_proprio_noise(observation, control_step=3)
  live.draw_range_epsilon(0, 32)

  restored = ObservationPerturbations(
    2, cfg=ALL_THREE_PRESET, seed=_SEED, device="cpu"
  )
  restored.load_state_dict(live.state_dict())
  # stream positions continue identically (advancing scan numbers in order)
  scan_number = 26
  for _ in range(4):
    for row in range(2):
      assert torch.equal(
        live.draw_range_epsilon(row, 32), restored.draw_range_epsilon(row, 32)
      )
      assert live.begin_scan(row, scan_number) == restored.begin_scan(
        row, scan_number
      )
    scan_number += 1
  twin_observation = _make_observation([(1, 3), (1, 3)])
  restored.apply_proprio_noise(twin_observation, control_step=3)
  for name in ("lin_vel_b", "ang_vel_b", "gravity_b",
               "joint_pos_rel_ref", "joint_vel"):
    assert torch.equal(getattr(twin_observation, name), getattr(observation, name))
  # a different recipe refuses
  try:
    restored.load_state_dict(
      ObservationPerturbations(
        2, cfg=GAP_ONLY_PRESET, seed=_SEED, device="cpu"
      ).state_dict()
    )
  except PerturbationContractError:
    pass
  else:
    raise AssertionError("mismatching recipe state accepted")
  # malformed state (phase out of range) refuses
  state = live.state_dict()
  bad_phase = dict(state)
  gap_rows = [dict(row) for row in state["gap_state"]]
  gap_rows[0]["phase"] = 99
  bad_phase["gap_state"] = gap_rows
  try:
    ObservationPerturbations(
      2, cfg=ALL_THREE_PRESET, seed=_SEED, device="cpu"
    ).load_state_dict(bad_phase)
  except PerturbationContractError:
    pass
  else:
    raise AssertionError("corrupt state accepted")


# ── window angle keys ───────────────────────────────────────────────────────


def _window_write(window: ScanWindow, scan_id: int, *, with_keys: bool) -> None:
  batch = window.batch_size
  near_points = torch.zeros(batch, NEAR_POINTS, 3)
  near_valid = torch.zeros(batch, NEAR_POINTS, dtype=torch.bool)
  far_points = torch.zeros(batch, FAR_POINTS_PER_SCAN, 3)
  far_valid = torch.zeros(batch, FAR_POINTS_PER_SCAN, dtype=torch.bool)
  near_keys = None
  far_keys = None
  for row in range(batch):
    near_points[row, 0] = float(scan_id)
    near_valid[row, 0] = True
    far_points[row, 0] = float(scan_id) + 0.5
    far_valid[row, 0] = True
  if with_keys:
    near_keys = torch.full((batch, NEAR_POINTS), float(scan_id) * 100.0)
    far_keys = torch.full((batch, FAR_POINTS_PER_SCAN), float(scan_id) * 100.0 + 1.0)
  window.write(
    env_ids=list(range(batch)),
    episode_ids=[0] * batch,
    scan_ids=[scan_id] * batch,
    times=[float(scan_id)] * batch,
    near_points=near_points,
    near_valid=near_valid,
    far_points=far_points,
    far_valid=far_valid,
    near_angle_keys=near_keys,
    far_angle_keys=far_keys,
  )


def test_window_angle_keys_roundtrip_and_rotation() -> None:
  window = ScanWindow(2, episode_ids=[0, 0], device="cpu")
  assert window.read_angle_keys() is None  # clean start
  for scan_id in (1, 2, 3):
    _window_write(window, scan_id, with_keys=True)
  snapshot = window.read()
  keys = window.read_angle_keys()
  assert keys is not None
  assert bool(keys.present.all())
  # the newest slot carries the LAST written keys/point (far_count=3 < 10:
  # sequential order ⇒ the newest is slot index 2, the same slot rotation
  # read() applies to far_points)
  assert keys.near[0, 0] == 300.0
  assert keys.far[0, 2, 0] == 301.0
  assert snapshot.far_points[0, 2, 0, 0] == 3.5
  assert snapshot.far_scan_ids[0, 2] == 3
  # near keys at invalid (shortage) slots are the stored dummies (0 here)
  assert keys.near[0, 1] == 0.0


def test_window_angle_keys_selected_reset_clears_only_that_row() -> None:
  window = ScanWindow(2, episode_ids=[0, 0], device="cpu")
  _window_write(window, 1, with_keys=True)
  window.reset_envs(env_ids=[0], new_episode_ids=[1])
  keys = window.read_angle_keys()
  assert keys is not None
  assert keys.present.tolist() == [False, True]
  assert bool((keys.near[0] == 0).all())
  assert keys.near[1, 0] == 100.0
  # a no-key write clears presence for the written row (no stale keys); env0
  # is in episode 1 after the reset, so only env1 may be written with 0
  window.write(
    env_ids=[1], episode_ids=[0], scan_ids=[2], times=[2.0],
    near_points=torch.zeros(1, NEAR_POINTS, 3),
    near_valid=torch.zeros(1, NEAR_POINTS, dtype=torch.bool),
    far_points=torch.zeros(1, FAR_POINTS_PER_SCAN, 3),
    far_valid=torch.zeros(1, FAR_POINTS_PER_SCAN, dtype=torch.bool),
  )
  assert window.read_angle_keys() is None


def test_window_angle_keys_contract_rejections() -> None:
  window = ScanWindow(1, episode_ids=[0], device="cpu")
  near_points = torch.zeros(1, NEAR_POINTS, 3)
  near_points[0, 0] = 1.0
  near_valid = torch.zeros(1, NEAR_POINTS, dtype=torch.bool)
  near_valid[0, 0] = True
  far_points = torch.zeros(1, FAR_POINTS_PER_SCAN, 3)
  far_points[0, 0] = 1.0
  far_valid = torch.zeros(1, FAR_POINTS_PER_SCAN, dtype=torch.bool)
  far_valid[0, 0] = True
  base = dict(
    env_ids=[0], episode_ids=[0], scan_ids=[1], times=[1.0],
    near_points=near_points, near_valid=near_valid,
    far_points=far_points, far_valid=far_valid,
  )
  good_near = torch.full((1, NEAR_POINTS), 1.0)
  good_far = torch.full((1, FAR_POINTS_PER_SCAN), 1.0)
  window.write(**base, near_angle_keys=good_near, far_angle_keys=good_far)
  try:  # one-sided keys
    window.write(
      **{**base, "scan_ids": [2], "times": [2.0],
         "near_angle_keys": good_near, "far_angle_keys": None},
    )
  except Exception as error:
    assert "同时" in str(error)
  else:
    raise AssertionError("one-sided keys accepted")
  nan_near = good_near.clone()
  nan_near[0, 0] = float("nan")
  try:
    window.write(
      **{**base, "scan_ids": [3], "times": [3.0],
         "near_angle_keys": nan_near, "far_angle_keys": good_far},
    )
  except Exception:
    pass
  else:
    raise AssertionError("NaN key at a valid slot accepted")


# ── checkpoint schema / rl_cfg / scanner frame ─────────────────────────────


def test_checkpoint_schema_v3_rejects_v1_and_v2() -> None:
  from hexapod_mjlab.tasks.confined_locomotion.rl.checkpoint import (
    CHECKPOINT_SCHEMA,
    CheckpointError,
    validate_payload_envelope,
  )

  assert CHECKPOINT_SCHEMA == "confined_locomotion.checkpoint/v3"
  for old_schema in (
    "confined_locomotion.checkpoint/v1",
    "confined_locomotion.checkpoint/v2",
  ):
    payload = {
      "schema": old_schema,
      "kind": "confined_locomotion_full_resume",
      "resumable": True,
      "task": "El-4090-Confined-Locomotion-Flat",
      "created": "x",
      "identity": {}, "physics": {}, "algorithm": {}, "runtime": {},
      "env_state": {}, "action_state": {}, "foot_air_state": {},
      "termination_state": {}, "prng": {}, "runner_state": {}, "manager_state": {},
    }
    try:
      validate_payload_envelope(payload)
    except CheckpointError as error:
      assert "schema" in str(error)
      continue
    raise AssertionError(f"{old_schema} schema accepted")


def test_window_serialization_includes_angle_keys() -> None:
  from hexapod_mjlab.tasks.confined_locomotion.rl.checkpoint import _serialize_window

  window = ScanWindow(2, episode_ids=[0, 0], device="cpu")
  _window_write(window, 1, with_keys=True)
  payload = _serialize_window(window)
  assert payload["angle_keys"] is not None
  assert payload["angle_keys"]["present"].tolist() == [True, True]
  clean_window = ScanWindow(2, episode_ids=[0, 0], device="cpu")
  _window_write(clean_window, 1, with_keys=False)
  assert _serialize_window(clean_window)["angle_keys"] is None


def test_source_ray_window_presence_survives_all_invalid_scan_and_reset() -> None:
  from hexapod_mjlab.rl.confined_locomotion.contracts import FAR_POINTS_PER_SCAN

  window = ScanWindow(1, episode_ids=[0], device="cpu")
  near = torch.zeros(1, NEAR_POINTS, 3)
  near_valid = torch.zeros(1, NEAR_POINTS, dtype=torch.bool)
  far = torch.zeros(1, FAR_POINTS_PER_SCAN, 3)
  far_valid = torch.zeros(1, FAR_POINTS_PER_SCAN, dtype=torch.bool)
  window.write(
    env_ids=[0], episode_ids=[0], scan_ids=[0], times=[0.0],
    near_points=near, near_valid=near_valid,
    far_points=far, far_valid=far_valid,
    near_source_dirs_sensor=torch.zeros_like(near),
    far_source_dirs_sensor=torch.zeros_like(far),
    near_source_indices=torch.full((1, NEAR_POINTS), -1, dtype=torch.long),
    far_source_indices=torch.full((1, FAR_POINTS_PER_SCAN), -1, dtype=torch.long),
  )
  directions = window.read_source_directions()
  assert directions is not None
  assert not bool(directions.near_present.any() or directions.far_present.any())
  assert window._source_dirs_attached.tolist() == [True]

  from hexapod_mjlab.tasks.confined_locomotion.rl.checkpoint import _serialize_window
  saved = _serialize_window(window)
  assert saved["source_dirs_attached"].tolist() == [True]
  window.reset_envs(env_ids=[0], new_episode_ids=[1])
  assert window._source_dirs_attached.tolist() == [False]
  assert window.read_source_directions() is None


def test_rl_cfg_default_clean_and_engineering_preset() -> None:
  from support.fixtures import m12_training_grid
  from hexapod_mjlab.tasks.confined_locomotion.rl_cfg import (
    ConfinedLocomotionRlCfg,
    confined_locomotion_engineering_rl_cfg,
    confined_locomotion_rl_cfg,
  )

  default_cfg = confined_locomotion_rl_cfg()
  assert default_cfg.observation_perturbations is None
  assert default_cfg.max_iterations == 0 and default_cfg.engineering_enabled is False
  assert isinstance(ConfinedLocomotionRlCfg.IMPLEMENTED, bool)
  payload = dataclasses.asdict(default_cfg)
  assert "IMPLEMENTED" not in payload
  grid = m12_training_grid()
  engineering = confined_locomotion_engineering_rl_cfg(
    seed=_SEED, ground_grid=grid
  )
  assert engineering.observation_perturbations is None  # clean default
  noisy = confined_locomotion_engineering_rl_cfg(
    seed=_SEED, ground_grid=grid, perturbations=ALL_THREE_PRESET
  )
  assert noisy.observation_perturbations == ALL_THREE_PRESET
  restored = ObservationPerturbationCfg.from_dict(
    dataclasses.asdict(noisy).get("observation_perturbations")
  )
  assert restored.identity() == ALL_THREE_PRESET.identity()
  try:
    confined_locomotion_engineering_rl_cfg(
      seed=_SEED, ground_grid=grid, perturbations={"gap": None}
    )
  except ValueError:
    pass
  else:
    raise AssertionError("untyped perturbations accepted")


def test_scan_frame_exposes_consumed_copy_audit_fields() -> None:
  names = {field.name for field in dataclasses.fields(ScanFrame)}
  for name in (
    "gap_fired", "consumed_valid", "consumed_dists", "range_epsilon",
    "range_applied_mask", "near_angle_keys", "far_angle_keys",
  ):
    assert name in names
  # defaults keep the clean construction identical (None/False)
  defaults = {
    field.name: field.default for field in dataclasses.fields(ScanFrame)
    if field.name in ("gap_fired", "consumed_valid", "consumed_dists",
                      "range_epsilon", "range_applied_mask",
                      "near_angle_keys", "far_angle_keys")
  }
  assert defaults["gap_fired"] is False
  assert all(defaults[name] is None for name in defaults if name != "gap_fired")


def test_direction_keys_match_point_keys_at_positive_distances() -> None:
  generator = torch.Generator().manual_seed(5)
  dirs = torch.nn.functional.normalize(
    torch.randn(128, 3, generator=generator), dim=-1
  )
  dists = 0.5 + 14.0 * torch.rand(128, generator=generator)
  mount_pos = torch.tensor([[0.644974, 0.0, -0.164512]])
  mount_quat = torch.tensor([[0.0, -0.0449, 0.0, 0.99899]])
  mount_quat = torch.nn.functional.normalize(mount_quat, dim=-1)
  from hexapod_mjlab.tasks.confined_locomotion.scanner import _quat_apply_xyzw
  points = _quat_apply_xyzw(mount_quat.expand(128, 4), dists.unsqueeze(-1) * dirs).unsqueeze(0) + mount_pos
  keys_dirs = direction_angle_keys(dirs)
  keys_points = sensor_frame_angle_keys(points, mount_pos, mount_quat)[0]
  assert torch.allclose(keys_dirs, keys_points, atol=1e-3, rtol=1e-4)


def test_full_ring_checkpoint_keys_keep_storage_order() -> None:
  from hexapod_mjlab.tasks.confined_locomotion.rl.checkpoint import _serialize_window
  window = ScanWindow(2, episode_ids=[0, 0], device="cpu")
  for scan in range(1, 14):
    _window_write(window, scan, with_keys=True)
  saved = _serialize_window(window)
  assert torch.equal(saved["angle_keys"]["far"], window._far_angle_keys)
  # Wrapped raw storage must differ from the chronological consumer view.
  assert not torch.equal(saved["angle_keys"]["far"], window.read_angle_keys().far)


def test_bad_late_state_preserves_all_streams() -> None:
  import copy
  owner = ObservationPerturbations(2, cfg=ALL_THREE_PRESET, seed=_SEED)
  owner.apply_proprio_noise(_make_observation([(1, 0), (1, 0)]), control_step=0)
  before = owner.state_dict()
  bad = copy.deepcopy(before)
  other = ObservationPerturbations(2, cfg=ALL_THREE_PRESET, seed=_SEED)
  other.draw_range_epsilon(0, 64)
  bad["streams"]["range"] = other.state_dict()["streams"]["range"]
  bad["range_draw_counts"] = torch.tensor([-1, 0])
  try:
    owner.load_state_dict(bad)
  except PerturbationContractError:
    pass
  else:
    raise AssertionError("negative late draw count accepted")
  after = owner.state_dict()
  for stream in before["streams"]:
    for a, b in zip(before["streams"][stream], after["streams"][stream]):
      assert torch.equal(a, b)
  assert before["gap_state"] == after["gap_state"]
  assert torch.equal(before["proprio_draw_counts"], after["proprio_draw_counts"])


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
