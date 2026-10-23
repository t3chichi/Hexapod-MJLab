#!/usr/bin/env python
"""Validate the original-surface diagnostic against analytic and convex references."""
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts/experiments/joint_passage/lidar_setup'))
import mesh_relations as geometry


def reference_distance(a, b):
    # Two triangle barycentric coordinates: independent bounded convex QP.
    def difference(x):
        return (a[0] + x[0] * (a[1] - a[0]) + x[1] * (a[2] - a[0])
                - b[0] - x[2] * (b[1] - b[0]) - x[3] * (b[2] - b[0]))
    result = minimize(lambda x: np.dot(difference(x), difference(x)), np.ones(4) / 3,
                      method='SLSQP', bounds=[(0, 1)] * 4,
                      constraints=[{'type': 'ineq', 'fun': lambda x: 1 - x[0] - x[1]},
                                   {'type': 'ineq', 'fun': lambda x: 1 - x[2] - x[3]}],
                      options={'ftol': 1e-14, 'maxiter': 300})
    assert result.success, result.message
    return np.linalg.norm(difference(result.x))


class MeshRelations(unittest.TestCase):
    def setUp(self):
        self.a = np.array([[0., 0, 0], [1., 0, 0], [0., 1, 0]])

    def test_crossing_contact_and_separation(self):
        crossing = np.array([[.2, .2, -1.], [.2, .2, 1.], [.8, .2, 1.]])
        h, proper, coplanar, points = geometry.relations(self.a[None], crossing[None])
        self.assertTrue(h[0] and proper[0] and not coplanar[0])
        witness = points[0]
        self.assertAlmostEqual(witness[2], 0., places=12)
        self.assertGreaterEqual(witness[0], 0)
        self.assertGreaterEqual(witness[1], 0)
        self.assertLessEqual(witness[:2].sum(), 1)
        for triangle, contact in [(self.a + [.1, .1, 0], True),
                                  (self.a + [1., 0, 0], True),
                                  (self.a + [2., 0, 0], False),
                                  (self.a + [0., 0, .1], False)]:
            hit, _, _, _ = geometry.relations(self.a[None], triangle[None])
            self.assertEqual(bool(hit[0]), contact)
        self.assertAlmostEqual(geometry.inspect(self.a[None], (self.a + [0, 0, .1])[None])['surface_distance_m'], .1, places=12)

    def test_triangle_distances_against_independent_qp(self):
        rng = np.random.default_rng(730)
        # Includes parallel, clamped-edge and skew pairs at the millimetre scale.
        pairs = [(self.a * .01, (self.a + [0, 0, .03]) * .01),
                 (self.a, self.a + [2, 1, .1])]
        pairs += [(rng.normal(size=(3, 3)) * .01,
                   rng.normal(size=(3, 3)) * .01 + [.04, 0, .01]) for _ in range(30)]
        for a, b in pairs:
            hit = geometry.relations(a[None], b[None])[0][0]
            actual = 0. if hit else geometry.distances(a[None], b[None])[0]
            self.assertAlmostEqual(actual, reference_distance(a, b), delta=2e-7)

    def test_broad_phase_keeps_every_pair_within_distance(self):
        rng = np.random.default_rng(26)
        a = rng.uniform(-.1, .1, (12, 3, 3))
        b = rng.uniform(-.1, .1, (13, 3, 3))
        for padding in [0., .003, .1]:
            retained = set(map(tuple, geometry.candidates(a, b, padding)))
            for i in range(len(a)):
                for j in range(len(b)):
                    # Independent QP also checks broad phase on intersections.
                    distance = reference_distance(a[i], b[j])
                    if distance <= padding + 1e-8:
                        self.assertIn((i, j), retained)

    def test_global_minimum_and_closed_surface_containment(self):
        rng = np.random.default_rng(82)
        a = rng.normal(size=(5, 3, 3)) * .01
        b = rng.normal(size=(6, 3, 3)) * .01 + [.08, 0, 0]
        reference = min(reference_distance(x, y) for x in a for y in b)
        self.assertAlmostEqual(geometry.inspect(a, b)['surface_distance_m'], reference, delta=2e-7)
        import trimesh
        box = trimesh.creation.box(extents=[2, 2, 2])
        winding = geometry.winding(np.array([[0, 0, 0], [3, 0, 0]]), box.triangles)
        self.assertAlmostEqual(abs(winding[0]), 1., places=12)
        self.assertAlmostEqual(abs(winding[1]), 0., places=12)


if __name__ == '__main__':
    unittest.main()
