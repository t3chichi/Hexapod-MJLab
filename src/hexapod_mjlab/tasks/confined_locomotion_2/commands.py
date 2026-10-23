"""Per-env body-velocity command streams for the flat training task (P1).

Confirmed recipe (docs/tasks/confined_locomotion/平地学习与基准测量.md §2,
"平地指令采样/指令重采样/运动指令与跟踪坐标", all confirmed 2026-10-07):

* every value is a PHYSICAL body-frame component (before any observation
  multiplier): vx ∈ [−1.5, 1.5] m/s, vy ∈ [−1.0, 1.0] m/s,
  ωz ∈ [−1.0, 1.0] rad/s;
* per resample each env independently picks a category: 80 % mixed motion
  (all three components uniform in their ranges), 10 % pure turning
  (vx = vy = 0, ωz uniform — a near-zero ωz draw stays legal, no deadzone),
  10 % standstill (all three zero);
* resampling happens at every reset of the selected envs and every 4 s of
  SIMULATED time afterwards (= 200 control steps at 50 Hz); partial resets
  only touch the selected envs' streams and period;
* each env consumes its OWN ``torch.Generator`` stream: resetting one env
  never advances another env's stream;
* the category selection and the raw sample draws are recorded per resample
  for independent recomputation;
* manual mode (for the later P7 play entry): ``set_command`` inside the
  legal ranges, no 4 s resampling, reset zeroes the value. The scheduler
  must be constructed with ``allow_manual=True`` for that; a random-mode
  scheduler REFUSES manual overrides so training can never pick them up
  accidentally.

The scheduler owns the command VALUES only. It does not know the policy,
the clock of the environment, or the observation; the training runtime tells
it when a step boundary passed (``step``) and which rows were reset
(``sample_reset``). State (generators, values, counters, manual flags) is
fully serializable via ``state_dict``/``load_state_dict``.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = [
    "COMMAND_PERIOD_CONTROL_STEPS",
    "CommandContractError",
    "TrainingCommandScheduler",
    "derive_command_generator_seed",
]

#: 4 s of simulated time at 50 Hz control (confirmed resample period).
COMMAND_PERIOD_CONTROL_STEPS = 200

#: Bounded audit history of resample records (memory safety only; every
#: record is small and CPU-side).
MAX_RESAMPLE_RECORDS = 4096


class CommandContractError(RuntimeError):
    """Invalid command-scheduler use (manual overrides, ranges, state)."""


def derive_command_generator_seed(base_seed: int, env_row: int) -> int:
    """Explicit, documented per-env generator seed derivation.

    ``numpy.random.SeedSequence(entropy=base_seed, spawn_key=(env_row,))``
    hashes the pair deterministically; the derivation depends on nothing
    else (not num_envs, not construction order), so an env's stream is
    reproducible from (base_seed, env_row) alone and stable when the batch
    size changes. Returns a uint32-safe int for ``torch.Generator.manual_seed``.
    """
    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise CommandContractError(f"base_seed must be an int, got {base_seed!r}")
    if isinstance(env_row, bool) or not isinstance(env_row, int) or env_row < 0:
        raise CommandContractError(f"env_row must be a non-negative int, got {env_row!r}")
    sequence = np.random.SeedSequence(entropy=int(base_seed), spawn_key=(int(env_row),))
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


class TrainingCommandScheduler:
    """Independent per-env body-command streams (values only, no env access)."""

    RANGE_VX = (-1.5, 1.5)
    RANGE_VY = (-1.0, 1.0)
    RANGE_OMEGA = (-1.0, 1.0)
    MIXED_PROBABILITY = 0.8
    PURE_TURN_PROBABILITY = 0.1  # standstill takes the remaining 0.1

    def __init__(
        self,
        num_envs: int,
        *,
        seed: int,
        device: torch.device | str | None = None,
        period: int = COMMAND_PERIOD_CONTROL_STEPS,
        allow_manual: bool = False,
    ) -> None:
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
            raise CommandContractError(f"num_envs must be a positive int, got {num_envs!r}")
        if isinstance(period, bool) or not isinstance(period, int) or period < 1:
            raise CommandContractError(f"period must be a positive int, got {period!r}")
        self.num_envs = int(num_envs)
        self.seed = int(seed)
        self.period = int(period)
        self.allow_manual = bool(allow_manual)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        # One CPU torch.Generator per env; all draws for env e come from
        # generators[e] only.
        self._generators: list[torch.Generator] = []
        for row in range(self.num_envs):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(derive_command_generator_seed(self.seed, row))
            self._generators.append(generator)
        self._command = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self._steps_since_resample = torch.zeros(self.num_envs, dtype=torch.long)
        self._manual = torch.zeros(self.num_envs, dtype=torch.bool)
        self._sampled_once = torch.zeros(self.num_envs, dtype=torch.bool)
        self.resample_records: list[dict] = []
        self.total_resamples = 0

    # ── sampling ────────────────────────────────────────────────────────────

    def _append_record(self, record: dict) -> None:
        self.total_resamples += 1
        record["sample_index"] = self.total_resamples
        self.resample_records.append(record)
        if len(self.resample_records) > MAX_RESAMPLE_RECORDS:
            self.resample_records.pop(0)

    def _sample_row(self, row: int, *, reason: str, episode_step: int | None) -> dict:
        """Consume env ``row``'s OWN stream and write its new command value."""
        generator = self._generators[row]
        u = float(torch.rand((), generator=generator).item())
        if u < self.MIXED_PROBABILITY:
            category = "mixed"
            draws = torch.rand(3, generator=generator).tolist()
            vx = self.RANGE_VX[0] + draws[0] * (self.RANGE_VX[1] - self.RANGE_VX[0])
            vy = self.RANGE_VY[0] + draws[1] * (self.RANGE_VY[1] - self.RANGE_VY[0])
            omega = self.RANGE_OMEGA[0] + draws[2] * (self.RANGE_OMEGA[1] - self.RANGE_OMEGA[0])
        elif u < self.MIXED_PROBABILITY + self.PURE_TURN_PROBABILITY:
            category = "pure_turn"
            draws = torch.rand(1, generator=generator).tolist()
            vx = 0.0
            vy = 0.0
            omega = self.RANGE_OMEGA[0] + draws[0] * (self.RANGE_OMEGA[1] - self.RANGE_OMEGA[0])
        else:
            category = "standstill"
            draws = []
            vx = vy = omega = 0.0
        values = (float(vx), float(vy), float(omega))
        if not all(np.isfinite(values)):
            raise CommandContractError(f"env {row}: non-finite sampled command {values!r}")
        self._command[row] = torch.tensor(values, dtype=torch.float32, device=self.device)
        self._steps_since_resample[row] = 0
        self._sampled_once[row] = True
        self._append_record(
            {
                "env": int(row),
                "reason": reason,
                "episode_step": None if episode_step is None else int(episode_step),
                "category": category,
                "category_draw": u,
                "component_draws": list(draws),
                "command": list(values),
            }
        )
        return self.resample_records[-1]

    def sample_reset(self, env_ids, *, episode_step: int | None = None) -> list[dict]:
        """Reset-time sampling: consume ONLY the selected envs' own streams.

        Manual rows are zeroed (manual survives reset per the confirmed play
        contract) without consuming their random stream. Returns the records
        of the resamples performed.
        """
        rows = self._resolve_rows(env_ids)
        records = []
        for row in rows:
            if bool(self._manual[row]):
                self._command[row] = 0.0
                self._steps_since_resample[row] = 0
                self._append_record(
                    {
                        "env": int(row),
                        "reason": "reset_manual_zero",
                        "episode_step": None if episode_step is None else int(episode_step),
                        "category": "manual_zero",
                        "category_draw": None,
                        "component_draws": [],
                        "command": [0.0, 0.0, 0.0],
                    }
                )
                records.append(self.resample_records[-1])
            else:
                records.append(self._sample_row(row, reason="reset", episode_step=episode_step))
        return records

    def step(self, *, episode_steps=None) -> list[dict]:
        """One completed control step for ALL envs: advance period counters.

        Non-manual rows whose counter reaches the period consume their own
        stream again (period restarts). Returns the resample records.
        """
        self._steps_since_resample += 1
        due = [
            int(row)
            for row in range(self.num_envs)
            if (not bool(self._manual[row]))
            and int(self._steps_since_resample[row]) >= self.period
        ]
        records = []
        for row in due:
            records.append(self._sample_row(
                row,
                reason="period",
                episode_step=None if episode_steps is None else int(episode_steps[row]),
            ))
        return records

    # ── reads ───────────────────────────────────────────────────────────────

    def current_command(self) -> torch.Tensor:
        """(B, 3) clone of the current command values (physical units)."""
        return self._command.detach().clone()

    def manual_mask(self) -> torch.Tensor:
        return self._manual.clone()

    def sampled_once(self) -> torch.Tensor:
        return self._sampled_once.clone()

    # ── manual mode (P7 play entry; refused in random mode) ─────────────────

    def set_command(self, env_ids, values) -> None:
        """Manual override for the selected envs (legal ranges enforced).

        Requires ``allow_manual=True`` at construction: a random-mode
        scheduler refuses so training can never silently accept a manual
        override. Manual rows are exempt from period resampling; resetting
        them zeroes the value (``sample_reset``).
        """
        if not self.allow_manual:
            raise CommandContractError(
                "set_command refused: this scheduler was constructed with "
                "allow_manual=False (random training mode must never accept "
                "manual overrides)"
            )
        rows = self._resolve_rows(env_ids)
        if isinstance(values, torch.Tensor):
            if values.dim() != 2 or int(values.shape[0]) != len(rows) or int(values.shape[1]) != 3:
                raise CommandContractError(
                    f"values must be ({len(rows)}, 3), got {tuple(values.shape)}"
                )
            triplets = [tuple(float(v) for v in row) for row in values.detach().cpu().tolist()]
        else:
            flat = [float(v) for v in values]
            if len(flat) == 3:
                triplets = [tuple(flat) for _ in rows]
            elif len(flat) == 3 * len(rows):
                triplets = [tuple(flat[i : i + 3]) for i in range(0, len(flat), 3)]
            else:
                raise CommandContractError(
                    f"values must be one (vx, vy, ω) triple or {len(rows)} triples, "
                    f"got {len(flat)} components"
                )
        for triplet in triplets:
            if len(triplet) != 3:
                raise CommandContractError(f"each command must have 3 components, got {triplet!r}")
            self._check_legal(triplet)
        for row, triplet in zip(rows, triplets):
            self._command[row] = torch.tensor(triplet, dtype=torch.float32, device=self.device)
            self._manual[row] = True
            self._append_record(
                {
                    "env": int(row),
                    "reason": "manual_set",
                    "episode_step": None,
                    "category": "manual",
                    "category_draw": None,
                    "component_draws": [],
                    "command": list(triplet),
                }
            )

    def clear_manual(self, env_ids) -> None:
        """Return the selected envs to random mode (immediate resample)."""
        rows = self._resolve_rows(env_ids)
        for row in rows:
            self._manual[row] = False
            self._sample_row(row, reason="clear_manual", episode_step=None)

    def _check_legal(self, triplet: tuple[float, float, float]) -> None:
        for name, value, bounds in (
            ("vx", triplet[0], self.RANGE_VX),
            ("vy", triplet[1], self.RANGE_VY),
            ("omega", triplet[2], self.RANGE_OMEGA),
        ):
            if not np.isfinite(value) or not (bounds[0] <= value <= bounds[1]):
                raise CommandContractError(
                    f"manual {name}={value!r} outside the legal range {bounds}"
                )

    # ── state ───────────────────────────────────────────────────────────────

    def state_dict(self) -> dict:
        """Complete continuation state (generator streams included)."""
        return {
            "num_envs": self.num_envs,
            "seed": self.seed,
            "period": self.period,
            "allow_manual": self.allow_manual,
            "generators": [g.get_state().clone() for g in self._generators],
            "command": self._command.detach().clone(),
            "steps_since_resample": self._steps_since_resample.clone(),
            "manual": self._manual.clone(),
            "sampled_once": self._sampled_once.clone(),
            "resample_records_count": len(self.resample_records),
            "total_resamples": self.total_resamples,
        }

    def load_state_dict(self, state: dict) -> None:
        for key in ("num_envs", "seed", "period", "allow_manual"):
            if state.get(key) != getattr(self, key):
                raise CommandContractError(
                    f"state_dict mismatch on {key!r}: {state.get(key)!r} != "
                    f"{getattr(self, key)!r}"
                )
        generators = state["generators"]
        if len(generators) != self.num_envs:
            raise CommandContractError("generator count mismatch")
        for generator, saved in zip(self._generators, generators):
            generator.set_state(saved.cpu().to(torch.uint8))
        self._command = state["command"].detach().to(
            device=self.device, dtype=torch.float32
        ).clone()
        self._steps_since_resample = state["steps_since_resample"].detach().to(
            dtype=torch.long
        ).clone()
        self._manual = state["manual"].detach().to(dtype=torch.bool).clone()
        self._sampled_once = state["sampled_once"].detach().to(dtype=torch.bool).clone()
        self.total_resamples = int(state["total_resamples"])

    # ── helpers ─────────────────────────────────────────────────────────────

    def _resolve_rows(self, env_ids) -> list[int]:
        if isinstance(env_ids, torch.Tensor):
            if env_ids.dim() != 1:
                raise CommandContractError("env_ids must be a 1-D tensor/list")
            rows = [int(r) for r in env_ids.detach().cpu().tolist()]
        else:
            rows = [int(r) for r in env_ids]
        if not rows:
            raise CommandContractError("empty env selection")
        if len(set(rows)) != len(rows):
            raise CommandContractError("duplicate env rows in selection")
        for row in rows:
            if not (0 <= row < self.num_envs):
                raise CommandContractError(f"env row {row} out of range [0, {self.num_envs})")
        return rows
