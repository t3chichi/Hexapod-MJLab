"""P6 observation perturbations: typed recipes, named per-env streams, persistent clocks.

This module owns the three CONFIRMED first-version observation perturbations of
the flat task (authorities: docs/tasks/confined_locomotion/平地学习与基准测量.md
§2 "雷达测距噪声起步规则"/"本体观测噪声起步规则", lidar_setup/配置与边界.md
"周期性扫描缺口", docs/research/训练架构对齐.md §1, training/首版训练实施计划.md
§4.5). It is a PURE state/stream module: it never touches the env, the scanner
backend buffers, the network, rewards, terminations or PD. Nothing here is
enabled by default — the registered production config stays clean and the whole
clean P0–P5 path is unchanged (a ``None``/clean configuration draws NOTHING).

1. GAP (periodic upper scan gap, position/period/jitter confirmed 2026-10-07):
   optical azimuth sector centred at 90°, half-open [74°, 106°), defined on the
   RAW unit rays in the SENSOR frame (not body yaw). Per env the phase
   φ ∈ {1..10} is drawn ONCE at creation from that env's independent gap
   stream; the PERSISTENT sensor-scan numbering (1, 2, …, including the
   creation/selected-reset first scan, never cleared by an episode reset) is
   OWNED BY THE RUNTIME and only QUERIED here. Nominal event frames are
   n_k = φ + 10·k; each event draws ε_k ∈ {−1, 0, +1} with probabilities
   (0.1, 0.8, 0.1) EXACTLY ONCE, scheduled before the earliest candidate scan
   (n_k − 1), non-cumulative. If φ = 1 and ε_0 = −1 the event frame would be 0
   (before the first possible scan): it is EXPLICITLY recorded as a
   creation-boundary "already passed/skip" event and never fired; subsequent
   anchors do NOT move. Jitter can be explicitly disabled (fixed ε = 0
   comparison control; then the gap stream consumes nothing beyond the phase
   draw). A selected partial reset NEVER rephases, never redraws a scheduled
   offset and never touches another env's stream — the schedule simply has no
   reset hook; the first post-reset scan just advances the persistent count.
   A bounded per-env event diagnostic table is kept (``last_event``), but the
   causal state (phase / next event index / scheduled ε / generators) is never
   dropped. Checkpoint save/restore preserves all of it bit-exactly.

2. RANGE (per-hit ranging noise on the CONSUMED copy only): for each NEW scan,
   independent ε ∼ Normal(0, 0.015²) m is drawn once per env from that env's
   independent range stream (a FULL-FRAME draw keeps stream consumption
   deterministic regardless of scene content; only hit rays OUTSIDE the current
   gap are ever APPLIED). d_obs = clip(d_raw + ε, 0, 15) m is computed by the
   PURE ``apply_range_noise`` on a copy: no-hit rays stay EXACTLY the 15 m
   max-range representation, gap-excluded and masked rays get no applied noise,
   and the raw dists/points/fault evidence remain byte-identical (the scanner
   keeps them verbatim). One draw per new scan — never re-drawn at control
   reuses, window reads, bootstrap or PPO epochs (those replay the stored
   consumed inputs). d_obs = 0 keeps the ORIGINAL unit-ray direction / angle
   key / group identity — grouping keys and cones are direction-based, never
   derived from the noisy xyz radius.

3. PROPRIO (bounded uniform noise on the 63-dim proprio copy): per env, ONE
   draw per NEW 50 Hz control-observation identity (env, episode, step) —
   including the fresh episode of a selected reset — from that env's
   independent proprio stream. Independent uniform half-amplitudes BEFORE any
   observation multiplier: lin_vel 0.1 m/s, ang_vel 0.2 rad/s, q−q_ref 0.05
   rad, joint_vel 1.5 rad/s, gravity 0.05 (vector re-normalized after the
   additive perturbation); total level 1.0 multiplies ONCE, the network's
   confirmed multipliers once more. command and the clipped prev_action are
   never noisy (63/3 separation preserved). The noise is applied to the
   observation COPY built by ``observation.build_control_observation`` (fresh
   clones of the true state — the true state is never mutated). A per-row
   identity cache guarantees: repeated ``prepare``/multi-forward/bootstrap/
   PPO-update reuse the SAME drawn noise bit-exactly, and a partial reset
   keeps the UNSELECTED rows' noise/streams bit-identical (only the selected
   rows' new-episode identities draw; the whole-batch input-cache
   invalidation can never cause a redraw for unselected rows). True geometry,
   labels, PD, rewards and terminations never read the noisy values.

Stream discipline: every perturbation owns ONE named CPU ``torch.Generator``
PER ENV, derived from (seed, env_row, stream) via ``SeedSequence`` — fully
independent of the command/action/shuffle/global streams. A DISABLED
perturbation consumes NOTHING (its generators are not even created). All
state (generators + gap schedule + proprio cache + draw counters) round-trips
through ``state_dict``/``load_state_dict`` for the P6 checkpoint (schema
``confined_locomotion.perturbations/v1``); restore never redraws the phase or
an already-scheduled offset.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from hexapod_mjlab.rl.confined_locomotion.contracts import JOINT_DIM

__all__ = [
  "ALL_THREE_PRESET",
  "GAP_FIXED_JITTER_OFF_PRESET",
  "GAP_ONLY_PRESET",
  "GapPerturbationCfg",
  "ObservationPerturbationCfg",
  "ObservationPerturbations",
  "PerturbationContractError",
  "PROPRIO_ONLY_PRESET",
  "ProprioPerturbationCfg",
  "RANGE_ONLY_PRESET",
  "RangePerturbationCfg",
  "apply_range_noise",
  "derive_perturbation_generator_seed",
  "gap_sector_mask",
  "preset_by_name",
]

#: Confirmed gap sector (authority: lidar_setup/配置与边界.md; the simulated
#: start position is the CURRENT optical azimuth centre 90°, [74°, 106°)).
GAP_AZIMUTH_CENTER_DEG = 90.0
GAP_AZIMUTH_HALF_SPAN_DEG = 16.0
#: Confirmed fixed nominal period: one gap every 10 NEW scans.
GAP_PERIOD_SCANS = 10
#: Confirmed timing-jitter probabilities (early / stay / late).
GAP_JITTER_PROBABILITIES = (0.1, 0.8, 0.1)

#: Confirmed ranging-noise scale (1σ, metres) on hit rays only.
RANGE_SIGMA_M = 0.015
#: Confirmed consumption clip of d_obs = clip(d_raw + ε, 0, 15) m.
RANGE_CLIP_MIN_M = 0.0
RANGE_CLIP_MAX_M = 15.0

#: Confirmed proprio noise: total level × independent uniform half-amplitudes
#: on the PRE-multiplier physical quantities (authority: 平地学习与基准测量.md
#: §2 本体观测噪声起步规则). Field order matches the 45 noisy components.
PROPRIO_NOISE_LEVEL = 1.0
PROPRIO_HALF_AMPLITUDES = (0.1, 0.2, 0.05, 0.05, 1.5)

#: Proprio noise vector layout (45 = 3 + 3 + 3 + 18 + 18; prev_action 18 and
#: command 3 are NEVER noisy — the 63/3 separation is untouched).
_PROPRIO_FIELD_SHAPES = (
  ("lin_vel_b", 3),
  ("ang_vel_b", 3),
  ("gravity_b", 3),
  ("joint_pos_rel_ref", JOINT_DIM),
  ("joint_vel", JOINT_DIM),
)
_PROPRIO_NOISY_FIELDS = tuple(name for name, _ in _PROPRIO_FIELD_SHAPES)
_PROPRIO_NOISE_DIM = sum(size for _, size in _PROPRIO_FIELD_SHAPES)

#: Named-stream identifiers (spawn keys; order is part of the state identity).
_STREAM_GAP = "gap"
_STREAM_RANGE = "range"
_STREAM_PROPRIO = "proprio"
_STREAM_IDS = {_STREAM_GAP: 1, _STREAM_RANGE: 2, _STREAM_PROPRIO: 3}

#: State schema stamp (checkpoint payload contract).
PERTURBATION_STATE_SCHEMA = "confined_locomotion.perturbations/v1"

#: Bounded per-env gap event diagnostic table (the CAUSAL state — phase,
#: next event index, scheduled ε, generators — is always kept, never dropped).
_MAX_GAP_EVENT_RECORDS = 64


class PerturbationContractError(RuntimeError):
  """Invalid perturbation configuration/use/state (never silently sanitized)."""


# ── stream seeding ──────────────────────────────────────────────────────────


def derive_perturbation_generator_seed(
  base_seed: int, env_row: int, stream_name: str
) -> int:
  """Explicit per-(env, stream) generator seed derivation.

  ``numpy.random.SeedSequence(entropy=base_seed, spawn_key=(env_row,
  stream_id))`` — deterministic from the triple alone, independent of
  num_envs/construction order, and DISJOINT from the command scheduler's
  ``spawn_key=(env_row,)`` derivation and from the action generator. Returns a
  uint32-safe int for ``torch.Generator.manual_seed``.
  """
  if isinstance(base_seed, bool) or not isinstance(base_seed, int):
    raise PerturbationContractError(f"base_seed must be an int, got {base_seed!r}")
  if isinstance(env_row, bool) or not isinstance(env_row, int) or env_row < 0:
    raise PerturbationContractError(f"env_row must be a non-negative int, got {env_row!r}")
  if stream_name not in _STREAM_IDS:
    raise PerturbationContractError(
      f"stream_name must be one of {sorted(_STREAM_IDS)}, got {stream_name!r}"
    )
  sequence = np.random.SeedSequence(
    entropy=int(base_seed), spawn_key=(int(env_row), _STREAM_IDS[stream_name])
  )
  return int(sequence.generate_state(1, dtype=np.uint32)[0])


# ── typed serializable recipes ─────────────────────────────────────────────


@dataclass(frozen=True)
class GapPerturbationCfg:
  """Confirmed periodic-gap recipe (serializable: floats/ints/bools/tuples)."""

  azimuth_center_deg: float = GAP_AZIMUTH_CENTER_DEG
  azimuth_half_span_deg: float = GAP_AZIMUTH_HALF_SPAN_DEG
  period_scans: int = GAP_PERIOD_SCANS
  jitter_enabled: bool = True
  jitter_probabilities: tuple[float, float, float] = GAP_JITTER_PROBABILITIES

  def __post_init__(self) -> None:
    for name in ("azimuth_center_deg", "azimuth_half_span_deg"):
      value = getattr(self, name)
      if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PerturbationContractError(f"{name} must be a real number, got {value!r}")
      if not math.isfinite(float(value)):
        raise PerturbationContractError(f"{name} must be finite, got {value!r}")
    span = float(self.azimuth_half_span_deg)
    if not (0.0 < span < 180.0):
      raise PerturbationContractError(
        f"azimuth_half_span_deg must be in (0, 180), got {span!r}"
      )
    lo = float(self.azimuth_center_deg) - span
    hi = float(self.azimuth_center_deg) + span
    if not (0.0 <= lo < hi <= 360.0):
      raise PerturbationContractError(
        f"gap sector [{lo}, {hi}) must satisfy 0 <= lo < hi <= 360 (no wrap "
        "modelling); got centre "
        f"{self.azimuth_center_deg!r}, half span {span!r}"
      )
    if (
      isinstance(self.period_scans, bool)
      or not isinstance(self.period_scans, int)
      or self.period_scans < 2
    ):
      raise PerturbationContractError(
        f"period_scans must be an int >= 2, got {self.period_scans!r}"
      )
    if isinstance(self.jitter_enabled, bool) is False:
      raise PerturbationContractError(
        f"jitter_enabled must be a bool, got {self.jitter_enabled!r}"
      )
    probs = self.jitter_probabilities
    if (
      not isinstance(probs, tuple)
      or len(probs) != 3
      or any(isinstance(p, bool) or not isinstance(p, (int, float)) for p in probs)
      or any(not math.isfinite(float(p)) for p in probs)
      or any(p < 0.0 for p in probs)
      or abs(float(sum(probs)) - 1.0) > 1e-9
    ):
      raise PerturbationContractError(
        f"jitter_probabilities must be three non-negative numbers summing to "
        f"1, got {probs!r}"
      )
    if (self.azimuth_center_deg != GAP_AZIMUTH_CENTER_DEG
        or self.azimuth_half_span_deg != GAP_AZIMUTH_HALF_SPAN_DEG
        or self.period_scans != GAP_PERIOD_SCANS
        or probs != GAP_JITTER_PROBABILITIES):
      raise PerturbationContractError("gap values differ from the confirmed P6 recipe")

  @property
  def sector(self) -> tuple[float, float]:
    """Half-open [lo, hi) sector in degrees."""
    span = float(self.azimuth_half_span_deg)
    return (float(self.azimuth_center_deg) - span, float(self.azimuth_center_deg) + span)

  def identity(self) -> dict:
    return {
      "azimuth_center_deg": float(self.azimuth_center_deg),
      "azimuth_half_span_deg": float(self.azimuth_half_span_deg),
      "sector_deg": [float(self.sector[0]), float(self.sector[1])],
      "period_scans": int(self.period_scans),
      "jitter_enabled": bool(self.jitter_enabled),
      "jitter_probabilities": [float(p) for p in self.jitter_probabilities],
    }


@dataclass(frozen=True)
class RangePerturbationCfg:
  """Confirmed per-hit ranging-noise recipe (consumed copy only)."""

  sigma_m: float = RANGE_SIGMA_M
  clip_min_m: float = RANGE_CLIP_MIN_M
  clip_max_m: float = RANGE_CLIP_MAX_M

  def __post_init__(self) -> None:
    for name in ("sigma_m", "clip_min_m", "clip_max_m"):
      value = getattr(self, name)
      if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PerturbationContractError(f"{name} must be a real number, got {value!r}")
      if not math.isfinite(float(value)):
        raise PerturbationContractError(f"{name} must be finite, got {value!r}")
    if not (0.0 < float(self.sigma_m)):
      raise PerturbationContractError(f"sigma_m must be positive, got {self.sigma_m!r}")
    if not (0.0 <= float(self.clip_min_m) < float(self.clip_max_m)):
      raise PerturbationContractError(
        f"clip range must satisfy 0 <= min < max, got "
        f"({self.clip_min_m!r}, {self.clip_max_m!r})"
      )

    if (self.sigma_m, self.clip_min_m, self.clip_max_m) != (RANGE_SIGMA_M, RANGE_CLIP_MIN_M, RANGE_CLIP_MAX_M):
      raise PerturbationContractError("range values differ from the confirmed P6 recipe")

  def identity(self) -> dict:
    return {
      "sigma_m": float(self.sigma_m),
      "clip_min_m": float(self.clip_min_m),
      "clip_max_m": float(self.clip_max_m),
    }


@dataclass(frozen=True)
class ProprioPerturbationCfg:
  """Confirmed proprio-noise recipe (63-dim copy; level × half-amplitudes)."""

  noise_level: float = PROPRIO_NOISE_LEVEL
  half_amplitudes: tuple[float, float, float, float, float] = PROPRIO_HALF_AMPLITUDES

  def __post_init__(self) -> None:
    if isinstance(self.noise_level, bool) or not isinstance(self.noise_level, (int, float)):
      raise PerturbationContractError(
        f"noise_level must be a real number, got {self.noise_level!r}"
      )
    if not (0.0 <= float(self.noise_level)) or not math.isfinite(float(self.noise_level)):
      raise PerturbationContractError(
        f"noise_level must be finite and non-negative, got {self.noise_level!r}"
      )
    halves = self.half_amplitudes
    if not isinstance(halves, tuple) or len(halves) != len(_PROPRIO_FIELD_SHAPES):
      raise PerturbationContractError(
        f"half_amplitudes must be a 5-tuple matching {_PROPRIO_NOISY_FIELDS}, "
        f"got {halves!r}"
      )
    for value in halves:
      if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PerturbationContractError(f"half-amplitude must be a number, got {value!r}")
      if not math.isfinite(float(value)) or float(value) < 0.0:
        raise PerturbationContractError(
          f"half-amplitude must be finite and non-negative, got {value!r}"
        )

    if self.noise_level != PROPRIO_NOISE_LEVEL or halves != PROPRIO_HALF_AMPLITUDES:
      raise PerturbationContractError("proprio values differ from the confirmed P6 recipe")

  def identity(self) -> dict:
    return {
      "noise_level": float(self.noise_level),
      "half_amplitudes": {
        name: float(value)
        for (name, _), value in zip(_PROPRIO_FIELD_SHAPES, self.half_amplitudes)
      },
    }


@dataclass(frozen=True)
class ObservationPerturbationCfg:
  """Typed aggregate of the three confirmed perturbations (None = OFF).

  Serializable end-to-end: every leaf is a float/int/bool/tuple, so
  ``dataclasses.asdict`` + YAML round-trips and the runner reconstructs the
  typed object strictly via ``from_dict`` (any deviation rejected). ``None``
  everywhere is the CLEAN configuration — the P0–P5 clean path is exactly it.
  """

  gap: GapPerturbationCfg | None = None
  range: RangePerturbationCfg | None = None  # noqa: A003 — recipe field name
  proprio: ProprioPerturbationCfg | None = None

  def __post_init__(self) -> None:
    if self.gap is not None and not isinstance(self.gap, GapPerturbationCfg):
      raise PerturbationContractError("gap must be a GapPerturbationCfg or None")
    if self.range is not None and not isinstance(self.range, RangePerturbationCfg):
      raise PerturbationContractError("range must be a RangePerturbationCfg or None")
    if self.proprio is not None and not isinstance(self.proprio, ProprioPerturbationCfg):
      raise PerturbationContractError("proprio must be a ProprioPerturbationCfg or None")

  @property
  def gap_enabled(self) -> bool:
    return self.gap is not None

  @property
  def range_enabled(self) -> bool:
    return self.range is not None

  @property
  def proprio_enabled(self) -> bool:
    return self.proprio is not None

  @property
  def lidar_active(self) -> bool:
    return self.gap_enabled or self.range_enabled

  @property
  def is_clean(self) -> bool:
    return self.gap is None and self.range is None and self.proprio is None

  def identity(self) -> dict:
    """JSON-safe summary recorded in recipes and checkpoint identities."""
    return {
      "gap": None if self.gap is None else self.gap.identity(),
      "range": None if self.range is None else self.range.identity(),
      "proprio": None if self.proprio is None else self.proprio.identity(),
      "lidar_active": bool(self.lidar_active),
      "clean": bool(self.is_clean),
    }

  @classmethod
  def from_dict(cls, payload: dict) -> "ObservationPerturbationCfg":
    """Strict reconstruction from an asdict-ed dict (lists→tuples tolerated)."""

    def _sub(kind, name, build):
      value = payload.get(name)
      if value is None:
        return None
      if not isinstance(value, dict):
        raise PerturbationContractError(
          f"{name} must be a dict or None, got {type(value).__name__}"
        )
      expected = {
        "gap": {"azimuth_center_deg", "azimuth_half_span_deg", "period_scans",
                "jitter_enabled", "jitter_probabilities"},
        "range": {"sigma_m", "clip_min_m", "clip_max_m"},
        "proprio": {"noise_level", "half_amplitudes"},
      }[name]
      if set(value) != expected:
        raise PerturbationContractError(f"{name} declaration field set mismatch")
      try:
        return build(value)
      except PerturbationContractError:
        raise
      except (TypeError, ValueError, KeyError) as error:
        raise PerturbationContractError(
          f"{kind} declaration is not the confirmed recipe: {error}"
        ) from error

    if not isinstance(payload, dict) or set(payload) != {"gap", "range", "proprio"}:
      raise PerturbationContractError(
        "observation_perturbations field set mismatch; the "
        "confirmed recipe has exactly gap/range/proprio"
      )
    gap = _sub("gap", "gap", lambda d: GapPerturbationCfg(
      azimuth_center_deg=d["azimuth_center_deg"],
      azimuth_half_span_deg=d["azimuth_half_span_deg"],
      period_scans=d["period_scans"],
      jitter_enabled=d["jitter_enabled"],
      jitter_probabilities=tuple(d["jitter_probabilities"]),
    ))
    rng = _sub("range", "range", lambda d: RangePerturbationCfg(
      sigma_m=d["sigma_m"], clip_min_m=d["clip_min_m"], clip_max_m=d["clip_max_m"],
    ))
    proprio = _sub("proprio", "proprio", lambda d: ProprioPerturbationCfg(
      noise_level=d["noise_level"],
      half_amplitudes=tuple(d["half_amplitudes"]),
    ))
    return cls(gap=gap, range=rng, proprio=proprio)


#: Confirmed-recipe presets (explicit engineering selections; the registered
#: production default stays clean — these are never auto-enabled).
GAP_ONLY_PRESET = ObservationPerturbationCfg(
  gap=GapPerturbationCfg(), range=None, proprio=None
)
GAP_FIXED_JITTER_OFF_PRESET = ObservationPerturbationCfg(
  gap=GapPerturbationCfg(jitter_enabled=False), range=None, proprio=None
)
RANGE_ONLY_PRESET = ObservationPerturbationCfg(
  gap=None, range=RangePerturbationCfg(), proprio=None
)
PROPRIO_ONLY_PRESET = ObservationPerturbationCfg(
  gap=None, range=None, proprio=ProprioPerturbationCfg()
)
ALL_THREE_PRESET = ObservationPerturbationCfg(
  gap=GapPerturbationCfg(),
  range=RangePerturbationCfg(),
  proprio=ProprioPerturbationCfg(),
)

_PRESETS = {
  "clean": None,
  "gap": GAP_ONLY_PRESET,
  "gap_fixed0": GAP_FIXED_JITTER_OFF_PRESET,
  "range": RANGE_ONLY_PRESET,
  "proprio": PROPRIO_ONLY_PRESET,
  "all": ALL_THREE_PRESET,
}


def preset_by_name(name: str) -> ObservationPerturbationCfg | None:
  """Explicit preset resolver (``clean`` → None; unknown names rejected)."""
  if name not in _PRESETS:
    raise PerturbationContractError(
      f"unknown perturbation preset {name!r}; choose from {sorted(_PRESETS)}"
    )
  return _PRESETS[name]


# ── pure per-scan functions (unit-testable without an env) ─────────────────


def gap_sector_mask(
  unit_dirs_sensor: torch.Tensor,
  *,
  azimuth_center_deg: float = GAP_AZIMUTH_CENTER_DEG,
  azimuth_half_span_deg: float = GAP_AZIMUTH_HALF_SPAN_DEG,
) -> torch.Tensor:
  """Boolean mask (True = INSIDE the gap sector) from RAW unit rays.

  The azimuth is ``atan2(y, x)`` of the SENSOR-frame unit ray wrapped to
  [0°, 360°); the sector is HALF-OPEN [lo, hi) with lo = centre − span,
  hi = centre + span (validated 0 ≤ lo < hi ≤ 360 at cfg construction, so no
  wrap handling is needed). This is the OPTICAL azimuth of the raw ray — not
  a body-yaw quantity (authority: lidar_setup/配置与边界.md).
  """
  if unit_dirs_sensor.dim() != 2 or unit_dirs_sensor.shape[1] != 3:
    raise PerturbationContractError(
      f"unit_dirs_sensor must be (R, 3), got {tuple(unit_dirs_sensor.shape)}"
    )
  x, y = unit_dirs_sensor[:, 0], unit_dirs_sensor[:, 1]
  azimuth_deg = torch.rad2deg(torch.atan2(y, x)) % 360.0
  lo = float(azimuth_center_deg) - float(azimuth_half_span_deg)
  hi = float(azimuth_center_deg) + float(azimuth_half_span_deg)
  return (azimuth_deg >= lo) & (azimuth_deg < hi)


def apply_range_noise(
  raw_dists: torch.Tensor,
  no_hit_mask: torch.Tensor,
  consumed_valid: torch.Tensor,
  epsilon: torch.Tensor | None,
  *,
  clip_min_m: float = RANGE_CLIP_MIN_M,
  clip_max_m: float = RANGE_CLIP_MAX_M,
) -> tuple[torch.Tensor, torch.Tensor]:
  """PURE consumption rule: (d_obs, applied_mask) on a new copy.

  * ``applied_mask = (~no_hit) & consumed_valid`` — ONLY actual hits OUTSIDE
    the current gap are perturbed; no-hit rays keep the EXACT max-range value
    (== clip_max_m here by the kernel miss convention), gap-excluded and
    masked rays keep the raw value verbatim.
  * ``d_obs = clip(d_raw + ε, clip_min, clip_max)`` exactly at applied rays
    and ``d_raw`` elsewhere; ``epsilon`` is the per-ray drawn Normal (may be
    None when the range perturbation is off — then d_obs is a plain copy of
    the raw dists and applied_mask is still reported for audit).
  * Saturation NEVER back-propagates into the raw hit identity: the caller
    keeps ``raw_dists``/``no_hit_mask`` verbatim.
  """
  if raw_dists.shape != no_hit_mask.shape or raw_dists.shape != consumed_valid.shape:
    raise PerturbationContractError(
      "raw_dists/no_hit_mask/consumed_valid must share one (R,) shape, got "
      f"{tuple(raw_dists.shape)}/{tuple(no_hit_mask.shape)}/"
      f"{tuple(consumed_valid.shape)}"
    )
  if raw_dists.dtype is not torch.float32 or no_hit_mask.dtype is not torch.bool \
      or consumed_valid.dtype is not torch.bool:
    raise PerturbationContractError("raw_dists float32 + bool masks required")
  applied = (~no_hit_mask) & consumed_valid
  if epsilon is None:
    return raw_dists.clone(), applied
  if epsilon.shape != raw_dists.shape or epsilon.dtype is not torch.float32:
    raise PerturbationContractError(
      f"epsilon must be {(tuple(raw_dists.shape), 'float32')}, got "
      f"{(tuple(epsilon.shape), str(epsilon.dtype))}"
    )
  d_obs = raw_dists.clone()
  d_obs[applied] = (raw_dists[applied] + epsilon[applied]).clamp(
    min=float(clip_min_m), max=float(clip_max_m)
  )
  return d_obs, applied


# ── per-env gap schedule state ──────────────────────────────────────────────


@dataclass
class _GapRowState:
  """Causal gap-schedule state of ONE env row (never reset by episode resets)."""

  phase: int                 # φ ∈ {1..period}; drawn ONCE at env creation
  next_event_index: int      # k of the next (not yet fired/skipped) event
  scheduled_epsilon: int | None  # ε_k once drawn (before earliest candidate)
  events: list = field(default_factory=list)  # bounded diagnostics only
  draws: int = 0             # jitter DRAWS consumed (phase draw excluded; 0
                             # with jitter disabled — the ε=0 control draws
                             # nothing from the stream)


class ObservationPerturbations:
  """Runtime owner of the three perturbations (pure state + streams).

  Constructed ONCE per runtime BEFORE the first scan (the takeover first scan
  is scan number 1 of the persistent clock). ``begin_scan`` is called by the
  scanner integration for each genuinely new scan, in per-env persistent
  order; ``apply_proprio_noise`` by the training runtime for each freshly
  built control observation. No method mutates the true env state.
  """

  def __init__(
    self,
    num_envs: int,
    *,
    cfg: ObservationPerturbationCfg,
    seed: int,
    device: torch.device | str | None = None,
  ) -> None:
    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
      raise PerturbationContractError(f"num_envs must be a positive int, got {num_envs!r}")
    if not isinstance(cfg, ObservationPerturbationCfg):
      raise PerturbationContractError("cfg must be an ObservationPerturbationCfg")
    self.num_envs = int(num_envs)
    self.cfg = cfg
    self.seed = int(seed)
    self.device = torch.device(device) if device is not None else torch.device("cpu")

    def _make_generators(name: str) -> list[torch.Generator]:
      generators = []
      for row in range(self.num_envs):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(derive_perturbation_generator_seed(self.seed, row, name))
        generators.append(generator)
      return generators

    # Only ACTIVE streams exist; a disabled perturbation consumes nothing.
    self._gap_generators = _make_generators(_STREAM_GAP) if cfg.gap_enabled else []
    self._range_generators = _make_generators(_STREAM_RANGE) if cfg.range_enabled else []
    self._proprio_generators = _make_generators(_STREAM_PROPRIO) if cfg.proprio_enabled else []
    self._range_draws = torch.zeros(self.num_envs, dtype=torch.long)
    self._proprio_draws = torch.zeros(self.num_envs, dtype=torch.long)

    self._gap_rows: list[_GapRowState | None] = (
      [self._new_gap_row(row) for row in range(self.num_envs)]
      if cfg.gap_enabled
      else [None] * self.num_envs
    )
    # Per-row proprio noise cache: identity (episode, step, control_step) →
    # drawn noise + clean/noisy component snapshots (45 each).
    self._proprio_cache: list[dict | None] = [None] * self.num_envs

  # ── configuration views ──────────────────────────────────────────────────

  @property
  def gap_enabled(self) -> bool:
    return self.cfg.gap_enabled

  @property
  def range_enabled(self) -> bool:
    return self.cfg.range_enabled

  @property
  def proprio_enabled(self) -> bool:
    return self.cfg.proprio_enabled

  @property
  def lidar_active(self) -> bool:
    return self.cfg.lidar_active

  # ── gap schedule ─────────────────────────────────────────────────────────

  def _new_gap_row(self, row: int) -> _GapRowState:
    """Env-creation phase draw: φ ∼ U{1..period}, ONCE, from the gap stream."""
    generator = self._gap_generators[row]
    period = int(self.cfg.gap.period_scans)
    phase = int(torch.randint(1, period + 1, (), generator=generator).item())
    return _GapRowState(phase=phase, next_event_index=0, scheduled_epsilon=None)

  def begin_scan(self, row: int, persistent_scan_number: int) -> bool:
    """Advance ONE env's schedule to its next NEW scan; True = gap fires.

    ``persistent_scan_number`` is the runtime-owned persistent count of the
    scan being taken (1-based, never reset by episode resets). The event
    offset ε_k is drawn (or fixed to 0 with jitter disabled) EXACTLY ONCE,
    at the latest when the scan number reaches the earliest candidate frame
    n_k − 1 — always BEFORE this scan is classified. The φ=1, ε=−1 event-0
    frame 0 lies before the first possible scan: it is recorded as
    "already_passed" and skipped without firing and without moving later
    anchors. Episode resets never call in here; the schedule is continuous.
    """
    if not self.gap_enabled:
      return False
    if isinstance(row, bool) or not isinstance(row, int) or not (0 <= row < self.num_envs):
      raise PerturbationContractError(f"env row {row!r} out of range")
    n = int(persistent_scan_number)
    if n < 1:
      raise PerturbationContractError(
        f"persistent scan numbering starts at 1, got {n}"
      )
    state = self._gap_rows[row]
    if state is None:  # pragma: no cover - guarded by gap_enabled
      raise PerturbationContractError("gap schedule missing (disabled stream)")
    period = int(self.cfg.gap.period_scans)
    while True:
      nominal = int(state.phase) + period * int(state.next_event_index)
      if state.scheduled_epsilon is None and n >= nominal - 1:
        state.scheduled_epsilon = self._draw_epsilon(row)
        if self.cfg.gap.jitter_enabled:
          state.draws += 1  # jitter draws only; ε=0 control consumes nothing
      epsilon = state.scheduled_epsilon
      if epsilon is not None and n == nominal + epsilon:
        self._record_event(
          state,
          {
            "k": int(state.next_event_index),
            "nominal": nominal,
            "epsilon": int(epsilon),
            "fired_scan": n,
            "skipped": False,
            "reason": None,
          },
        )
        state.next_event_index += 1
        state.scheduled_epsilon = None
        return True
      if epsilon is not None and n > nominal + epsilon:
        # Only reachable for the creation boundary: an event frame before the
        # first possible scan (φ=1 with ε=−1 ⇒ frame 0). Recorded, skipped,
        # NEVER fired and later anchors are untouched.
        self._record_event(
          state,
          {
            "k": int(state.next_event_index),
            "nominal": nominal,
            "epsilon": int(epsilon),
            "fired_scan": None,
            "skipped": True,
            "reason": "already_passed_before_first_scan",
          },
        )
        state.next_event_index += 1
        state.scheduled_epsilon = None
        continue
      return False

  def _draw_epsilon(self, row: int) -> int:
    if not self.cfg.gap.jitter_enabled:
      return 0  # explicit fixed-ε=0 comparison control; no stream draw
    p_early, p_stay, _late = (float(p) for p in self.cfg.gap.jitter_probabilities)
    u = float(torch.rand((), generator=self._gap_generators[row]).item())
    if u < p_early:
      return -1
    if u < p_early + p_stay:
      return 0
    return 1

  @staticmethod
  def _record_event(state: _GapRowState, record: dict) -> None:
    state.events.append(record)
    if len(state.events) > _MAX_GAP_EVENT_RECORDS:
      del state.events[:-_MAX_GAP_EVENT_RECORDS]

  def last_event(self, row: int) -> dict | None:
    """Most recent bounded diagnostic record (or None); NOT causal state."""
    if not self.gap_enabled:
      return None
    state = self._gap_rows[row]
    return None if state is None or not state.events else dict(state.events[-1])

  def gap_ray_mask(self, unit_dirs_sensor: torch.Tensor) -> torch.Tensor:
    """True = ray inside the gap sector (uses the confirmed recipe values)."""
    gap_cfg = self.cfg.gap
    if gap_cfg is None:  # pragma: no cover - guarded by callers
      raise PerturbationContractError("gap disabled: no gap ray mask exists")
    return gap_sector_mask(
      unit_dirs_sensor,
      azimuth_center_deg=gap_cfg.azimuth_center_deg,
      azimuth_half_span_deg=gap_cfg.azimuth_half_span_deg,
    )

  def gap_schedule_record(self) -> list[dict]:
    """Read-only causal-state view (for evidence/checks; not the state itself)."""
    rows = []
    for state in self._gap_rows:
      rows.append(
        None
        if state is None
        else {
          "phase": int(state.phase),
          "next_event_index": int(state.next_event_index),
          "scheduled_epsilon": (
            None if state.scheduled_epsilon is None else int(state.scheduled_epsilon)
          ),
          "event_count": len(state.events),
          "events": [dict(e) for e in state.events],
          "jitter_draws": int(state.draws),
        }
      )
    return rows

  # ── range noise ──────────────────────────────────────────────────────────

  def draw_range_epsilon(self, row: int, num_rays: int) -> torch.Tensor:
    """FULL-FRAME ε ∼ N(0, σ²) for ONE new scan of ONE env (on self.device).

    A fixed-size full-frame draw keeps the per-env stream consumption
    deterministic and independent of scene content (which rays happen to be
    hits); the scanner applies it only to hit rays outside the current gap
    via the PURE ``apply_range_noise``. One call per NEW scan — the runtime
    and the window/PPO replay the stored consumed values instead.
    """
    if not self.range_enabled:
      raise PerturbationContractError("range disabled: no epsilon may be drawn")
    if isinstance(row, bool) or not isinstance(row, int) or not (0 <= row < self.num_envs):
      raise PerturbationContractError(f"env row {row!r} out of range")
    if isinstance(num_rays, bool) or not isinstance(num_rays, int) or num_rays < 1:
      raise PerturbationContractError(f"num_rays must be a positive int, got {num_rays!r}")
    sigma = float(self.cfg.range.sigma_m)
    epsilon = torch.randn(num_rays, generator=self._range_generators[row]).mul_(sigma)
    self._range_draws[row] += 1
    return epsilon.to(device=self.device, dtype=torch.float32)

  # ── proprio noise ────────────────────────────────────────────────────────

  def apply_proprio_noise(self, observation, *, control_step: int) -> None:
    """Apply the per-row cached/fresh proprio noise to the observation COPY.

    ``observation`` must be the freshly built ``OnlineObservation`` from
    ``build_control_observation`` — its fields are independent clones of the
    true state, so mutating them here never touches the true state. The row
    identity is (env row, episode, step) plus the global control step; a cache
    hit REUSES the stored noise and noisy values bit-exactly (repeated
    ``prepare``/forward/bootstrap/update never redraw, and unselected rows
    after a whole-batch input-cache invalidation keep their noise bitwise),
    a miss draws exactly that one row's stream. command / prev_action are
    never touched; gravity is re-normalized after the additive draw.
    """
    if not self.proprio_enabled:
      return
    batch = int(observation.env_ids.shape[0])
    if batch != self.num_envs:
      raise PerturbationContractError(
        f"observation batch {batch} != perturbation num_envs {self.num_envs}"
      )
    level = float(self.cfg.proprio.noise_level)
    halves = torch.tensor(
      self.cfg.proprio.half_amplitudes, dtype=torch.float32
    ).repeat_interleave(
      torch.tensor([size for _, size in _PROPRIO_FIELD_SHAPES])
    )
    slices = {}
    offset = 0
    for name, size in _PROPRIO_FIELD_SHAPES:
      slices[name] = slice(offset, offset + size)
      offset += size
    clean = torch.cat(
      [getattr(observation, name).detach().reshape(batch, -1)
       for name, _ in _PROPRIO_FIELD_SHAPES],
      dim=1,
    )
    if tuple(clean.shape) != (batch, _PROPRIO_NOISE_DIM):
      raise PerturbationContractError("clean proprio concatenation shape mismatch")
    # Cache/noise arithmetic stays on CPU with the same per-row generators.
    # Transfer the immutable observation and identities once instead of one
    # CUDA synchronization per row, then copy all noisy rows back together.
    identities = torch.stack((observation.episode_ids, observation.step_ids), dim=1).cpu().tolist()
    clean = clean.cpu()
    noisy_rows = []
    for row, (episode_id, step_id) in enumerate(identities):
      entry = self._proprio_cache[row]
      if (
        entry is not None
        and entry["episode_id"] == episode_id
        and entry["step_id"] == step_id
        and entry["control_step"] == int(control_step)
      ):
        if not torch.equal(entry["clean"], clean[row]):
          raise PerturbationContractError(
            f"env {row}: proprio cache hit at (episode={episode_id}, step="
            f"{step_id}) but the rebuilt clean values differ — the same "
            "observation identity must come from the same clean state"
          )
        noise = entry["noise"]
        noisy = entry["noisy"]
      else:
        u = torch.rand(_PROPRIO_NOISE_DIM, generator=self._proprio_generators[row])
        noise = ((u * 2.0) - 1.0).mul(halves).mul(level)
        noisy = clean[row] + noise
        g = noisy[slices["gravity_b"]]
        g = g / torch.linalg.vector_norm(g)
        noisy[slices["gravity_b"]] = g
        if not bool(torch.isfinite(noisy).all()):
          raise PerturbationContractError(
            f"env {row}: non-finite proprio noise draw (fault; never sanitized)"
          )
        self._proprio_draws[row] += 1
        entry = {
          "episode_id": episode_id,
          "step_id": step_id,
          "control_step": int(control_step),
          "noise": noise.clone(),
          "clean": clean[row].clone(),
          "noisy": noisy.clone(),
          "draw_index": int(self._proprio_draws[row]),
        }
        self._proprio_cache[row] = entry
      noisy_rows.append(noisy)
    noisy_device = torch.stack(noisy_rows).to(device=observation.lin_vel_b.device, dtype=torch.float32)
    with torch.no_grad():
      for name, size in _PROPRIO_FIELD_SHAPES:
        getattr(observation, name).copy_(noisy_device[:, slices[name]].reshape(batch, size))

  def proprio_records(self) -> list[dict | None]:
    """Per-row audit copies of the current proprio noise cache (for evidence)."""
    return [
      None
      if entry is None
      else {key: (value.clone() if torch.is_tensor(value) else value)
            for key, value in entry.items()}
      for entry in self._proprio_cache
    ]

  def draw_counts(self) -> dict:
    """Audit counters (per-stream per-env draw counts)."""
    return {
      "gap_jitter": (
        None
        if not self.gap_enabled
        else [int(state.draws) if state else None for state in self._gap_rows]
      ),
      "range": (
        None if not self.range_enabled
        else self._range_draws.detach().cpu().tolist()
      ),
      "proprio": (
        None if not self.proprio_enabled
        else self._proprio_draws.detach().cpu().tolist()
      ),
    }

  # ── checkpoint state ─────────────────────────────────────────────────────

  def state_dict(self) -> dict:
    """Complete continuation state (generators + schedules + caches)."""
    state = {
      "schema": PERTURBATION_STATE_SCHEMA,
      "config_identity": self.cfg.identity(),
      "num_envs": self.num_envs,
      "seed": self.seed,
      "streams": {
        name: (
          [generator.get_state().clone() for generator in generators]
          if generators
          else None
        )
        for name, generators in (
          (_STREAM_GAP, self._gap_generators),
          (_STREAM_RANGE, self._range_generators),
          (_STREAM_PROPRIO, self._proprio_generators),
        )
      },
      "gap_state": (
        None
        if not self.gap_enabled
        else [
          {
            "phase": int(state.phase),
            "next_event_index": int(state.next_event_index),
            "scheduled_epsilon": state.scheduled_epsilon,
            "events": [dict(event) for event in state.events],
            "draws": int(state.draws),
          }
          for state in self._gap_rows
        ]
      ),
      "proprio_cache": (
        None
        if not self.proprio_enabled
        else [
          None
          if entry is None
          else {
            "episode_id": int(entry["episode_id"]),
            "step_id": int(entry["step_id"]),
            "control_step": int(entry["control_step"]),
            "noise": entry["noise"].clone(),
            "clean": entry["clean"].clone(),
            "noisy": entry["noisy"].clone(),
            "draw_index": int(entry["draw_index"]),
          }
          for entry in self._proprio_cache
        ]
      ),
      "range_draw_counts": (
        None if not self.range_enabled else self._range_draws.detach().clone()
      ),
      "proprio_draw_counts": (
        None if not self.proprio_enabled else self._proprio_draws.detach().clone()
      ),
    }
    return state

  def load_state_dict(self, state: dict) -> None:
    """Strict restore; never redraws the phase or a scheduled offset."""
    if not isinstance(state, dict):
      raise PerturbationContractError("perturbation state must be a dict")
    if set(state) != set(self.state_dict()):
      raise PerturbationContractError("perturbation state field set mismatch")
    if state.get("schema") != PERTURBATION_STATE_SCHEMA:
      raise PerturbationContractError(
        f"perturbation state schema {state.get('schema')!r} != "
        f"{PERTURBATION_STATE_SCHEMA!r}"
      )
    if state.get("config_identity") != self.cfg.identity():
      raise PerturbationContractError(
        "perturbation config identity mismatch: a checkpoint of a different "
        "noise recipe cannot be restored here"
      )
    if int(state["num_envs"]) != self.num_envs or int(state["seed"]) != self.seed:
      raise PerturbationContractError("perturbation num_envs/seed mismatch")

    expected_streams = {
      _STREAM_GAP: self._gap_generators,
      _STREAM_RANGE: self._range_generators,
      _STREAM_PROPRIO: self._proprio_generators,
    }
    streams = state["streams"]
    if not isinstance(streams, dict) or set(streams) != set(expected_streams):
      raise PerturbationContractError("perturbation stream section malformed")
    # Validate on isolated generators first: a bad late cache/count field
    # must not leave a partially restored live random stream.
    restored_streams = {}
    for name, generators in expected_streams.items():
      saved = streams[name]
      if generators:
        if not isinstance(saved, list) or len(saved) != self.num_envs:
          raise PerturbationContractError(f"{name} stream state list malformed")
        restored_streams[name] = []
        for tensor in saved:
          if not torch.is_tensor(tensor) or tensor.dtype is not torch.uint8 or tensor.dim() != 1:
            raise PerturbationContractError(f"{name} generator state malformed")
          trial = torch.Generator(device="cpu")
          try:
            trial.set_state(tensor.cpu())
          except RuntimeError as error:
            raise PerturbationContractError(f"{name} generator state invalid") from error
          restored_streams[name].append(trial)
      elif saved is not None:
        raise PerturbationContractError(
          f"{name} stream state present although the perturbation is disabled"
        )

    if self.gap_enabled:
      rows = state["gap_state"]
      if not isinstance(rows, list) or len(rows) != self.num_envs:
        raise PerturbationContractError("gap_state malformed")
      period = int(self.cfg.gap.period_scans)
      restored: list[_GapRowState | None] = []
      for row, saved in enumerate(rows):
        if not isinstance(saved, dict):
          raise PerturbationContractError(f"gap_state[{row}] malformed")
        phase = int(saved["phase"])
        next_index = int(saved["next_event_index"])
        epsilon = saved["scheduled_epsilon"]
        if not (1 <= phase <= period) or next_index < 0:
          raise PerturbationContractError(
            f"gap_state[{row}] phase/next_event_index out of range"
          )
        if epsilon is not None and int(epsilon) not in (-1, 0, 1):
          raise PerturbationContractError(f"gap_state[{row}] scheduled epsilon invalid")
        events = saved["events"]
        if not isinstance(events, list) or len(events) > _MAX_GAP_EVENT_RECORDS:
          raise PerturbationContractError(f"gap_state[{row}] events malformed")
        if set(saved) != {"phase", "next_event_index", "scheduled_epsilon", "events", "draws"}:
          raise PerturbationContractError(f"gap_state[{row}] field set mismatch")
        expected_draws = next_index + int(epsilon is not None) if self.cfg.gap.jitter_enabled else 0
        if saved["draws"] != expected_draws or (not self.cfg.gap.jitter_enabled and epsilon not in (None, 0)):
          raise PerturbationContractError(f"gap_state[{row}] jitter count/offset inconsistent")
        for index, event in enumerate(events):
          k = next_index - len(events) + index
          if (not isinstance(event, dict) or set(event) != {"k", "nominal", "epsilon", "fired_scan", "skipped", "reason"}
              or event["k"] != k or event["nominal"] != phase + period * k
              or event["epsilon"] not in (-1, 0, 1)):
            raise PerturbationContractError(f"gap_state[{row}] event malformed")
          if event["skipped"]:
            if (k, phase, event["epsilon"], event["fired_scan"], event["reason"]) != (0, 1, -1, None, "already_passed_before_first_scan"):
              raise PerturbationContractError(f"gap_state[{row}] invalid skipped event")
          elif event["fired_scan"] != event["nominal"] + event["epsilon"] or event["reason"] is not None:
            raise PerturbationContractError(f"gap_state[{row}] firing identity mismatch")
        restored.append(
          _GapRowState(
            phase=phase,
            next_event_index=next_index,
            scheduled_epsilon=None if epsilon is None else int(epsilon),
            events=[dict(event) for event in events],
            draws=int(saved["draws"]),
          )
        )
    elif state["gap_state"] is not None:
      raise PerturbationContractError("gap_state present although gap is disabled")

    if self.proprio_enabled:
      cache = state["proprio_cache"]
      if not isinstance(cache, list) or len(cache) != self.num_envs:
        raise PerturbationContractError("proprio_cache malformed")
      restored_cache: list[dict | None] = []
      for row, saved in enumerate(cache):
        if saved is None:
          restored_cache.append(None)
          continue
        if not isinstance(saved, dict) or set(saved) != {"episode_id", "step_id", "control_step", "noise", "clean", "noisy", "draw_index"}:
          raise PerturbationContractError(f"proprio_cache[{row}] field set mismatch")
        for key in ("episode_id", "step_id", "control_step", "draw_index"):
          if isinstance(saved[key], bool) or not isinstance(saved[key], int) or saved[key] < 0:
            raise PerturbationContractError(f"proprio_cache[{row}].{key} invalid")
        for key in ("noise", "clean", "noisy"):
          tensor = saved[key]
          if (
            not torch.is_tensor(tensor)
            or tuple(tensor.shape) != (_PROPRIO_NOISE_DIM,)
            or tensor.dtype is not torch.float32
            or not bool(torch.isfinite(tensor).all())
          ):
            raise PerturbationContractError(f"proprio_cache[{row}].{key} malformed")
        restored_cache.append(
          {
            "episode_id": int(saved["episode_id"]),
            "step_id": int(saved["step_id"]),
            "control_step": int(saved["control_step"]),
            "noise": saved["noise"].clone(),
            "clean": saved["clean"].clone(),
            "noisy": saved["noisy"].clone(),
            "draw_index": int(saved["draw_index"]),
          }
        )
      counts = state["proprio_draw_counts"]
      if (not torch.is_tensor(counts) or tuple(counts.shape) != (self.num_envs,)
          or counts.dtype is not torch.int64 or bool((counts < 0).any())):
        raise PerturbationContractError("proprio_draw_counts malformed")
      for row, saved in enumerate(restored_cache):
        if saved is not None and saved["draw_index"] != int(counts[row]):
          raise PerturbationContractError(f"proprio_cache[{row}] draw index mismatch")
    else:
      if state["proprio_cache"] is not None or state["proprio_draw_counts"] is not None:
        raise PerturbationContractError(
          "proprio state present although proprio is disabled"
        )

    if self.range_enabled:
      counts = state["range_draw_counts"]
      if (not torch.is_tensor(counts) or tuple(counts.shape) != (self.num_envs,)
          or counts.dtype is not torch.int64 or bool((counts < 0).any())):
        raise PerturbationContractError("range_draw_counts malformed")
    elif state["range_draw_counts"] is not None:
      raise PerturbationContractError(
        "range state present although range is disabled"
      )
    # All sections passed. Commit without sampling any stream.
    for name, generators in expected_streams.items():
      for live, trial in zip(generators, restored_streams.get(name, [])):
        live.set_state(trial.get_state())
    if self.gap_enabled:
      self._gap_rows = restored
    if self.proprio_enabled:
      self._proprio_cache = restored_cache
      self._proprio_draws.copy_(state["proprio_draw_counts"].cpu())
    if self.range_enabled:
      self._range_draws.copy_(state["range_draw_counts"].cpu())
