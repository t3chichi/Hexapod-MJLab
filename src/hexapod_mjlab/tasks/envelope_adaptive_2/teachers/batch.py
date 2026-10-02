"""Stateful CPU teacher adapter and explicit, independently checked validity.

MIN is a finite placeholder on unavailable frames, never a certified target.
The reference-rate certificate concerns this proxy shape, not physical joints.
"""
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

import numpy as np
import torch

from .causal import LookaheadOracle
from .static_seed import MIN, SPAN, NOMINAL_S

TEACHER_VERSION = "whole_polygon_preview_axial_v1"


@dataclass(frozen=True)
class TeacherConfig:
    margin: float = .10
    horizon: float = .12
    weight: float = 4.
    cooldown: float = .10
    dt: float = .02
    headroom: float = .15
    yaw_reserve: float = 0.


def teacher_identity(config=None):
    config = config or TeacherConfig()
    root = Path(__file__).parent
    return {"kind": "exact", "version": TEACHER_VERSION,
            "config": asdict(config),
            "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(root.glob('*.py'))}}


class BatchTeacher:
    def __init__(self, rects, config=None):
        self.config = config or TeacherConfig()
        self.oracle = LookaheadOracle(rects, **asdict(self.config))

    def reset(self, ids):
        self.oracle.reset(ids)

    def evaluate(self, pos, yaw, ego, ids=None):
        """Advance exactly once per requested frame; reads must use cached output."""
        pos_np = pos.detach().cpu().double().numpy()
        yaw_np = yaw.detach().cpu().double().numpy()
        ego_np = ego.detach().cpu().double().numpy()
        if pos_np.shape != (len(yaw_np), 2) or ego_np.shape != (len(yaw_np), 3):
            raise ValueError("Teacher expects pos(N,2), yaw(N), ego(N,3)")
        if not all(np.isfinite(v).all() for v in [pos_np, yaw_np, ego_np]):
            raise ValueError("Nonfinite teacher pose/motion input cannot be certified")
        ids = list(range(len(pos_np))) if ids is None else [int(e) for e in ids]
        if len(ids) != len(pos_np) or len(set(ids)) != len(ids):
            raise ValueError("Teacher IDs must identify each requested environment once")
        if not ids:
            raise ValueError("Teacher evaluate requires at least one frame")
        rows = []
        for i, env_id in enumerate(ids):
            previous = self.oracle.prev.get(env_id)
            previous = None if previous is None else previous.copy()
            result = self.oracle.step(pos_np[i], yaw_np[i], ego_np[i], env_id)
            s = np.zeros(5) if result.s is None else result.s.copy()
            finite = bool(np.isfinite(s).all())
            range_residual = float(max(0., -s.min(), s.max() - NOMINAL_S)) if finite else float('inf')
            solver = self.oracle.solver
            axial_residual = float(max(0., np.max(solver.extra_a @ s - solver.extra_b))) if finite and len(solver.extra_a) else 0. if finite else float('inf')
            clearance = solver.clearance(s, pos_np[i], yaw_np[i]) if finite else 0.
            geometric = (result.status == 'feasible' and finite and range_residual <= 1e-8
                         and axial_residual <= 1e-8 and clearance >= self.config.margin - 1e-7)
            rate_checked = previous is not None
            rate_residual = 0.
            if rate_checked and finite:
                delta = (s - previous) * np.abs(SPAN)
                rate_residual = float(max(0., np.max(delta - .5 * self.config.dt),
                                          np.max(-delta - 2. * self.config.dt)))
            rate_valid = rate_checked and rate_residual <= 1e-7
            emergency = bool(result.emergency or (geometric and rate_checked and not rate_valid))
            # Initial/static frames carry no rate certificate but can supervise shape.
            supervision = geometric and not emergency
            if not geometric:
                self.oracle.reset([env_id])
                s = np.zeros(5)
            rows.append((s, geometric, supervision, emergency, rate_checked, rate_valid,
                         clearance, range_residual, axial_residual, rate_residual,
                         result.status, result.reason, result.elapsed_s))
        def tensor(values, dtype):
            return torch.as_tensor(np.asarray(values), dtype=dtype, device=pos.device)
        output = {"s": tensor([r[0] for r in rows], pos.dtype)}
        output["params"] = output["s"] * tensor(SPAN, pos.dtype) + tensor(MIN, pos.dtype)
        for j, name in enumerate(["geometric_valid", "supervision_valid", "emergency",
                                  "rate_checked", "rate_valid"], 1):
            output[name] = tensor([r[j] for r in rows], torch.bool)
        for j, name in enumerate(["clearance", "range_residual", "axial_residual",
                                  "rate_residual"], 6):
            output[name] = tensor([r[j] for r in rows], pos.dtype)
        output["status"] = [r[10] for r in rows]
        output["reason"] = [r[11] for r in rows]
        output["elapsed_s"] = sum(r[12] for r in rows)
        return output
