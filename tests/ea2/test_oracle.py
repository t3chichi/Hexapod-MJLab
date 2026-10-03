"""Analytic geometry, whole-polygon reference and causal teacher contracts."""

import _bootstrap
import torch
from hexapod_mjlab.tasks.envelope_adaptive_2.teachers.batch import BatchTeacher
from pathlib import Path
import json
import numpy as np
from hexapod_mjlab.tasks.envelope_adaptive_2.contracts import RectPrimitive
from hexapod_mjlab.tasks.envelope_adaptive_2.experiments.static_qp import (
    StaticOracle,
    MIN,
    SPAN,
    NOMINAL_S,
    vertices,
    _convex_nearest_pair,
    rect_polygon,
)
from whole_polygon_reference import WholePolygonChecker, polygon_distance
from hexapod_mjlab.tasks.envelope_adaptive_2.experiments.refined_oracle import RefinedOracle
from hexapod_mjlab.tasks.envelope_adaptive_2.experiments.causal_oracle import CausalOracle
from hexapod_mjlab.tasks.envelope_adaptive_2.experiments.lookahead_oracle import LookaheadOracle


def fixtures(margin):
    nominal = MIN + SPAN * NOMINAL_S
    front = RectPrimitive((1.22, 0), (0.6, 4))
    corridor = [RectPrimitive((0, y), (8, 0.6)) for y in [-0.9, 0.9]]
    n = np.array([1.0, 1.0]) / np.sqrt(2)
    oblique = RectPrimitive(tuple(n * 1.25), (0.6, 6), np.pi / 4)
    turning = [RectPrimitive((0, y), (8, 0.6)) for y in [-1.2, 1.2]]
    blocked = [RectPrimitive((0, y), (8, 0.6)) for y in [-0.65, 0.65]]
    expected_front = nominal.copy()
    expected_front[3] = 0.92 - margin
    expected_corridor = nominal.copy()
    expected_corridor[:3] = 0.6 - margin
    expected_oblique = nominal.copy()
    delta = (nominal[0] + nominal[3] - np.sqrt(2) * (0.95 - margin)) / 2
    expected_oblique[[0, 3]] -= delta
    expected_turn45 = nominal.copy()
    delta = (nominal[0] + nominal[3] - np.sqrt(2) * (0.9 - margin)) / 2
    expected_turn45[[0, 2, 3]] -= delta
    expected_turn45[4] += delta
    expected_turn90 = nominal.copy()
    expected_turn90[3] = 0.9 - margin
    expected_turn90[4] = -(0.9 - margin)
    return [
        ("open", [], 0.0, nominal),
        ("front_wall", [front], 0.0, expected_front),
        ("corridor", corridor, 0.0, expected_corridor),
        ("oblique", [oblique], 0.0, expected_oblique),
        ("turn_0", turning, 0.0, nominal),
        ("turn_45", turning, np.pi / 4, expected_turn45),
        ("turn_90", turning, np.pi / 2, expected_turn90),
        ("minimum_infeasible", blocked, 0.0, None),
    ]


def check_static_geometry():
    evidence = []
    for margin in [0.1, 0.2]:
        for name, rects, yaw, expected in fixtures(margin):
            result = StaticOracle(rects, margin).solve(np.zeros(2), yaw)
            record = {"name": name, "margin": margin, "status": result.status}
            if expected is None:
                assert result.status == "minimum_infeasible", record
            else:
                assert result.status == "feasible", record
                error = float(np.max(np.abs(result.params - expected)))
                assert error <= 0.0001, (record, error, result.params, expected)
                clearance = WholePolygonChecker(rects).clearance(result.params, np.zeros(2), yaw)
                assert clearance >= margin - 1e-06, (record, clearance)
                assert result.constraint_residual_m <= 1e-06
                record.update(
                    error_m=error, clearance_m=clearance if np.isfinite(clearance) else None
                )
            evidence.append(record)
    square = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
    assert polygon_distance(square, square * 0.02) == 0
    assert polygon_distance(square, square + np.array([1.5, 0])) == 0
    assert abs(polygon_distance(square, square + np.array([3, 0])) - 1) < 1e-12
    concave = np.array([0.6, 0.3, 0.6, 0.6, -0.6])
    checker = WholePolygonChecker([RectPrimitive((0, 0.5), (0.04, 0.04))])
    assert checker.clearance(concave, np.zeros(2), 0) > 0
    rect = RectPrimitive((0.88, 0.58), (0.4, 0.4))
    d, pa, pb = _convex_nearest_pair(vertices(MIN)[[0, 2, 3, 5]], rect_polygon(rect))
    assert abs(d - np.sqrt(2) * 0.08) < 1e-12 and np.allclose(pb - pa, [0.08, 0.08])
    for margin in [0.1, 0.2]:
        result = StaticOracle([rect], margin).solve(np.zeros(2), 0)
        assert result.status == ("feasible" if margin == 0.1 else "minimum_infeasible")
        if result.params is not None:
            assert (
                WholePolygonChecker([rect]).clearance(result.params, np.zeros(2), 0)
                >= margin - 1e-06
            )

    print(
        "STATIC GATE PASS: six scenario classes, 16 analytic solutions/status checks; independent concavity/containment/crossing/corner distance probes"
    )


def check_causal_geometry():
    records = []
    for margin in [0.1, 0.2]:
        for name, rects, yaw, expected in fixtures(margin):
            result = RefinedOracle(rects, margin).solve(np.zeros(2), yaw)
            if expected is None:
                assert result.status == "minimum_infeasible"
            else:
                assert result.status == "feasible"
                error = np.max(np.abs(MIN + SPAN * result.s - expected))
                assert error < 0.0001, (name, margin, error)
                assert (
                    WholePolygonChecker(rects).clearance(MIN + SPAN * result.s, [0, 0], yaw)
                    >= margin - 1e-06
                )
            records.append({"case": name, "margin": margin, "status": result.status})
    narrow = [RectPrimitive((0, y), (8, 0.6)) for y in [-0.715, 0.715]]
    capped = RefinedOracle(narrow, 0.1, headroom=0.1)
    r = capped.solve([0, 0], 0)
    expected = np.array([0.05, 0.0375, 0.05, 0.5125, 0.5125])
    assert r.status == "feasible" and np.max(np.abs(r.s - expected)) < 0.0001, (r, expected)
    assert (
        capped.shape_valid(r.s)
        and capped.shape_valid(np.zeros(5))
        and capped.shape_valid(np.full(5, 0.975))
    )
    for scale in np.linspace(0, 1, 11):
        assert capped.shape_valid(scale * r.s)
    assert WholePolygonChecker(narrow).clearance(MIN + SPAN * r.s, [0, 0], 0) >= 0.1 - 1e-06
    oracle = CausalOracle([], horizon=0.24)
    oracle.prev[0] = np.full(5, 0.4)
    oracle.counter[0] = np.zeros(5, int)
    queries = []
    original = oracle.solver.solve

    def capture(pos, yaw, **kwargs):
        queries.append((np.asarray(pos).copy(), float(yaw)))
        return original(pos, yaw, **kwargs)

    oracle.solver.solve = capture
    oracle.step([2, 3], np.pi / 2, [0, 0.4, 0.2])
    assert any(
        (
            np.allclose(p, [2.096, 3], atol=1e-12) and abs(h - (np.pi / 2 + 0.048)) < 1e-12
            for p, h in queries
        )
    ), queries
    for cls in [CausalOracle, LookaheadOracle]:
        walls = [RectPrimitive((1.22, 0), (0.6, 4))]
        multi = cls(walls)
        fresh = cls(walls)
        for e in [0, 1]:
            for t in range(12):
                multi.step([-0.1, 0], 0, [0.2, 0, 0], e)
        other = multi.prev[1].copy()
        multi.reset([0])
        assert np.array_equal(multi.prev[1], other)
        a = multi.step([-0.1, 0], 0, [0, 0, 0], 0)
        b = fresh.step([-0.1, 0], 0, [0, 0, 0], 0)
        assert np.array_equal(a.s, b.s)
        checker = WholePolygonChecker(walls)
        for t in range(60):
            velocity = 0.2 if 20 <= t < 40 else 0.0
            pos = [-0.1 + 0.004 * min(max(t - 20, 0), 20), 0]
            r = multi.step(pos, 0, [velocity, 0, 0], 0)
            assert r.status == "feasible"
            assert checker.clearance(MIN + SPAN * r.s, pos, 0) >= 0.1 - 1e-06
        blocked = [RectPrimitive((0, y), (8, 0.6)) for y in [-0.65, 0.65]]
        invalid = cls(blocked)
        r = invalid.step([0, 0], 0, [0, 0, 0], 7)
        assert r.status == "minimum_infeasible" and 7 not in invalid.prev
    predictive = LookaheadOracle([RectPrimitive((1.22, 0), (0.6, 4))], horizon=0.12, weight=4)
    for _ in range(120):
        r = predictive.step([0, 0], 0, [0.3, 0, 0])
    expected_fwd = 0.92 - 0.1 - 0.3 * 0.12
    assert abs((MIN + SPAN * r.s)[3] - expected_fwd) < 0.0001, ((MIN + SPAN * r.s)[3], expected_fwd)
    from whole_polygon_reference import polygon_distance

    center = np.array([1.7, -0.3])
    pos = np.array([0.2, 0.1])
    heading = 0.7
    original = RectPrimitive(tuple(center), (0.4, 1.1), 0.37)
    params = MIN + SPAN * np.array([0.8, 0.3, 0.7, 0.6, 0.5])

    def outline(p, h):
        fw, mw, bw, f, b = params
        poly = np.array([[f, -fw], [0, -mw], [b, -bw], [b, bw], [0, mw], [f, fw]])
        co, si = (np.cos(h), np.sin(h))
        return poly @ np.array([[co, si], [-si, co]]) + p

    for angle in [-0.24, 0.12]:
        future_pos = pos + np.array([0.07, -0.03])
        co, si = (np.cos(angle), np.sin(angle))
        translated = pos + (center - future_pos) @ np.array([[co, -si], [si, co]])
        changed = RectPrimitive(tuple(translated), original.size, original.yaw - angle)
        first = WholePolygonChecker([original])
        second = WholePolygonChecker([changed])
        d1 = polygon_distance(outline(future_pos, heading + angle), first.polygons[0])
        d2 = polygon_distance(outline(pos, heading), second.polygons[0])
        assert abs(d1 - d2) < 1e-12, (d1, d2)
    result = {
        "analytic_cases": records,
        "legacy_velocity_direction": "pass",
        "partial_reset_isolation": "pass",
        "stop_start_stop_safety": "pass",
        "minimum_infeasible_resets_history": "pass",
        "common_shape_front_wall": "pass",
        "rotated_forecast_distance_identity": "pass",
        "near_floor_rule_analytic_corridor": "pass",
    }

    print("REFINEMENT GATE PASS", json.dumps(result))


def check_teacher_certificates():
    wall = [RectPrimitive((1.03, 0), (0.6, 4))]
    teacher = BatchTeacher(wall)
    pos = torch.zeros(2, 2)
    yaw = torch.zeros(2)
    ego = torch.zeros(2, 3)
    out = teacher.evaluate(pos, yaw, ego)
    expected = np.array([0.975, 0.975, 0.975, 0.1, 0.975])
    assert np.max(abs(out["s"][0].numpy() - expected)) < 0.0001
    assert out["geometric_valid"].all() and out["supervision_valid"].all()
    assert not out["rate_checked"].any()
    checker = WholePolygonChecker(wall)
    assert checker.clearance(out["params"][0].numpy(), [0, 0], 0) >= 0.1 - 1e-06
    other = teacher.oracle.prev[1].copy()
    teacher.reset([0])
    assert np.array_equal(other, teacher.oracle.prev[1]) and 0 not in teacher.oracle.prev
    out = teacher.evaluate(pos, yaw, ego)
    assert out["rate_checked"].tolist() == [False, True] and out["rate_valid"].tolist() == [
        False,
        True,
    ]
    narrow = [RectPrimitive((0, z), (8, 0.6)) for z in [-0.715, 0.715]]
    narrow_teacher = BatchTeacher(narrow)
    narrow_out = narrow_teacher.evaluate(pos[:1], yaw[:1], ego[:1])
    expected = np.array(
        [
            0.05,
            0.0375,
            0.05,
            0.05 + (0.975 - 0.05) / 0.15 * 0.05,
            0.05 + (0.975 - 0.05) / 0.15 * 0.05,
        ]
    )
    assert np.max(abs(narrow_out["s"][0].numpy() - expected)) < 0.0001
    assert (
        WholePolygonChecker(narrow).clearance(narrow_out["params"][0].numpy(), [0, 0], 0)
        >= 0.1 - 1e-06
    )
    blocked = BatchTeacher([RectPrimitive((0, z), (8, 0.6)) for z in [-0.65, 0.65]])
    out = blocked.evaluate(pos[:1], yaw[:1], ego[:1])
    assert not out["geometric_valid"].any() and (not out["supervision_valid"].any())
    assert torch.isfinite(out["s"]).all() and (not blocked.oracle.prev)
    try:
        teacher.evaluate(torch.full_like(pos, float("nan")), yaw, ego)
    except ValueError:
        pass
    else:
        raise AssertionError("Nonfinite pose certified")


def main():
    torch.set_num_threads(1)
    for check in (check_static_geometry, check_causal_geometry, check_teacher_certificates):
        check()
        print("PASS", check.__name__, flush=True)


if __name__ == "__main__":
    main()
