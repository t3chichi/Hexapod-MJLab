#!/usr/bin/env python
"""Adversarial scientific gates with physically scaled positive controls."""

import sys, unittest, copy, tempfile, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/tools/t0"))
import numpy as np, torch
from hexapod_mjlab.tasks.el_4090_lidar.t0.evaluation import validate_episode, SAFETY_KEYS


def positive(vx=0.25):
    n = 1000
    h = 0.4598590386845707
    t = (np.arange(n) + 1) * 0.02
    r = np.column_stack([vx * t, np.zeros(n), np.full(n, h)])
    qp = np.zeros((4 * n, 25))
    qp[:, 3] = 1
    qp[:, :3] = np.repeat(r, 4, axis=0)
    qv = np.zeros((4 * n, 24))
    qv[:, 0] = vx
    physics = dict(
        qpos=qp, qvel=qv, ctrl=np.full((4 * n, 18), 20.0), active=np.ones(4 * n, dtype=bool)
    )
    for k in SAFETY_KEYS:
        physics[k] = (
            np.arange(1, 4 * n + 1)
            if k == "substep_samples"
            else (
                np.arange(1, 4 * n + 1) * 18
                if k == "tau_samples"
                else (
                    np.full(4 * n, 20.0)
                    if k in ("preclip_peak", "applied_peak")
                    else np.zeros(4 * n)
                )
            )
        )
    samples = np.column_stack([physics[k][3::4] for k in SAFETY_KEYS])
    safety = {k: float(samples[-1, i]) for i, k in enumerate(SAFETY_KEYS)}
    return dict(
        episode_id="positive",
        command_name="stand" if vx == 0 else "forward_025",
        requested_vx=vx,
        seed=1001,
        n_steps=n,
        early_terminated=False,
        termination_step=None,
        witness_height=h,
        initial_yaw_rad=0.0,
        initial_root_pos=[0, 0, h],
        any_violation=False,
        safety=safety,
        root_pos=r,
        root_quat=np.tile([1.0, 0, 0, 0], (n, 1)),
        lin_vel_b=np.tile([vx, 0, 0], (n, 1)),
        ang_vel_b=np.zeros((n, 3)),
        joint_pos=np.zeros((n, 18)),
        joint_vel=np.zeros((n, 18)),
        actuator_force=np.full((n, 18), 20.0),
        actions=np.zeros((n, 18)),
        cmd=np.tile([vx, 0, 0], (n, 1)),
        tilt_deg=np.zeros(n),
        safety_samples=samples,
        physics=physics,
    )


class EpisodeTests(unittest.TestCase):
    def test_physical_positive_stand_forward(self):
        for vx in (0.0, 0.25):
            self.assertTrue(validate_episode(positive(vx))["passed"])

    def test_false_successes(self):
        scenarios = (
            "negative_lateral",
            "positive_lateral",
            "height_cancellation",
            "return_to_origin",
            "wrong_nonzero_command",
            "zero_command",
            "safety_metadata",
            "events",
            "bad_length",
            "root_nan",
            "overflow",
            "angle",
            "torque",
        )
        for name in scenarios:
            with self.subTest(name=name):
                e = positive(0.0 if name == "return_to_origin" else 0.25)
                if name in ("negative_lateral", "positive_lateral"):
                    e["root_pos"][:, 1] = np.linspace(
                        0, -5 if name == "negative_lateral" else 5, 1000
                    )
                elif name == "height_cancellation":
                    e["root_pos"][100:, 2] += np.tile([-0.1, 0.1], 450)
                elif name == "return_to_origin":
                    e["root_pos"][100:, 0] = 2 * np.sin(np.linspace(0, np.pi, 900))
                elif name == "wrong_nonzero_command":
                    e["cmd"][:, 0] = 0.35
                elif name == "zero_command":
                    e["cmd"][:, 0] = 0
                elif name == "safety_metadata":
                    e["any_violation"] = True
                elif name == "events":
                    e["safety"]["self_events"] = 1
                elif name == "bad_length":
                    e["joint_pos"] = e["joint_pos"][:-1]
                elif name == "root_nan":
                    e["physics"]["qvel"][40, 0] = np.nan
                elif name == "overflow":
                    e["physics"]["overflow_events"][40:] = 1
                elif name == "angle":
                    e["physics"]["qpos"][4, 7] = 4
                elif name == "torque":
                    e["physics"]["ctrl"][7, 0] = 81
                # Keep geometric control and physics records consistent.
                e["physics"]["qpos"][:, :3] = np.repeat(e["root_pos"], 4, axis=0)
                self.assertFalse(validate_episode(e)["passed"])

    def test_early_termination_all_lengths(self):
        for n in (0, 1, 50, 100, 101, 400, 999):
            with self.subTest(n=n):
                e = positive()
                e.update(n_steps=n, early_terminated=True, termination_step=n - 1)
                for k, v in list(e.items()):
                    if isinstance(v, np.ndarray) and len(v) == 1000:
                        e[k] = v[:n]
                e["physics"] = {k: v[: 4 * n] for k, v in e["physics"].items()}
                if n:
                    e["safety"] = {
                        k: float(e["safety_samples"][-1, i]) for i, k in enumerate(SAFETY_KEYS)
                    }
                self.assertFalse(validate_episode(e)["passed"])


class ContractTests(unittest.TestCase):
    def test_witness_assets_and_plane(self):
        import mujoco
        from hexapod_mjlab.tasks.el_4090_lidar.t0.flat_env_cfg import (
            _get_el_4090_t0_spec,
            make_t0_flat_terrain_config,
        )
        from hexapod_mjlab.tasks.el_4090_lidar.t0.witness import (
            verify_against_library,
            compute_lifted_root,
            SETTLED_QPOS,
        )
        from hexapod_mjlab.tasks.el_4090_lidar.el_4090_lidar_env_cfg import _EL4090_XML
        from hexapod_mjlab.terrains.builder import TerrainBuilder
        import t0lib

        before = t0lib.sha256_file(_EL4090_XML)
        m = _get_el_4090_t0_spec().compile()
        self.assertEqual(int(m.jnt_limited.sum()), 0)
        self.assertEqual(before, t0lib.sha256_file(_EL4090_XML))
        self.assertTrue(verify_against_library(t0lib.REPO_ROOT)["match"])
        self.assertAlmostEqual(compute_lifted_root(_EL4090_XML, SETTLED_QPOS)[2], 0.002, places=8)
        for seed in (0, 42, 1234):
            self.assertFalse(
                TerrainBuilder(make_t0_flat_terrain_config()).build(seed).heightfield_raw.any()
            )

    def test_mirror_absolute_targets(self):
        from types import SimpleNamespace
        from tensordict import TensorDict
        from hexapod_mjlab.tasks.el_4090_lidar.t0.symmetry import t0_symmetry
        from hexapod_mjlab.tasks.el_4090_lidar.t0.witness import JOINT_TARGETS

        env = SimpleNamespace(
            _sensor_offset_quat=torch.tensor([[0.0, 0, 0, 1]]),
            _sensor_translation=torch.zeros(1, 3),
        )
        d = torch.tensor(JOINT_TARGETS)
        q = torch.linspace(-0.8, 0.7, 18)
        a = torch.linspace(-1.0, 1, 18).view(1, 18)
        o = torch.zeros(1, 2754)
        o[:, 12:30] = q - d
        o[:, 48:66] = a
        obs = TensorDict({"policy": o, "height_grid": torch.zeros(1, 187)}, batch_size=[1])
        m, ma = t0_symmetry(env, obs, a)
        swap = lambda v: torch.cat([v[..., 9:], v[..., :9]], dim=-1)
        self.assertTrue(torch.allclose(m["policy"][1:, 12:30] + d, swap(q), atol=1e-6))
        self.assertTrue(torch.allclose(d + 0.25 * ma[1:], swap(d + 0.25 * a), atol=1e-6))
        self.assertTrue(torch.allclose(m["policy"][1:, 48:66], ma[1:], atol=1e-6))
        mm, maa = t0_symmetry(env, m[1:], ma[1:])
        self.assertTrue(torch.allclose(mm["policy"][1:], o, atol=1e-6))
        self.assertTrue(torch.allclose(maa[1:], a, atol=1e-6))

    def test_signed_hull(self):
        from hold_gate import hull_margin

        square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        self.assertAlmostEqual(hull_margin(square, [0.5, 0.5]), 0.5)
        self.assertLess(hull_margin(square, [2, 0.5]), 0)

    def test_fail_closed_states_and_overwrite(self):
        import t0lib

        for state in (t0lib.FAIL, t0lib.NOT_VERIFIED, t0lib.BLOCKED, t0lib.PARTIAL):
            with tempfile.TemporaryDirectory() as directory:
                p = Path(directory)
                r = t0lib.StageReport("test", p)
                r.gate("gate", state)
                self.assertNotEqual(r.finalize(returncode=0)["returncode"], 0)
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            r = t0lib.StageReport("test", p)
            with self.assertRaises(ValueError):
                r.gate("x", "typo")
            self.assertFalse(r.finalize()["pipeline_passed"])
            with self.assertRaises(FileExistsError):
                t0lib.write_json(p / "test.json", {})
            with self.assertRaises(ValueError):
                t0lib.write_json(p / "nan.json", dict(v=float("nan")))
            self.assertFalse((p / "nan.json").exists())

    def test_required_stage_completeness(self):
        from run_acceptance import REQUIRED_STAGES, stages_passed

        rows = [dict(name=s, returncode=0, state="PASS") for s in REQUIRED_STAGES]
        self.assertTrue(stages_passed(rows))
        self.assertFalse(stages_passed(rows[:-1]))
        self.assertFalse(stages_passed(rows + rows[:1]))
        rows[0]["state"] = "BLOCKED"
        self.assertFalse(stages_passed(rows))

    def test_replay_report_location(self):
        from run_acceptance import run_stage
        import t0lib

        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "replay").mkdir()
            t0lib.write_json(
                run / "replay/replay_verify.json",
                dict(pipeline_passed=True, gates=[dict(name="identity", state="PASS")]),
            )
            record = run_stage("replay_verify", [sys.executable, "-c", "pass"], run, 1)
            self.assertEqual(record["state"], "PASS")

    def test_angular_ties_roundtrip(self):
        from types import SimpleNamespace
        from tensordict import TensorDict
        from hexapod_mjlab.tasks.el_4090_lidar.t0.symmetry import t0_symmetry
        from hexapod_mjlab.tasks.el_4090_lidar.t0.point_order import canonical_points

        env = SimpleNamespace(
            _sensor_offset_quat=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
            _sensor_translation=torch.zeros(1, 3),
        )
        points = torch.tensor([[[1.0, 1.0, 0.0], [2.0, 2.0, 0.0], [3.0, 3.0, 0.0]]]).repeat(
            1, 86, 1
        )[:, :256]
        o = torch.zeros(1, 2754)
        o[:, 66:834] = canonical_points(
            points, env._sensor_offset_quat, env._sensor_translation
        ).reshape(1, -1)
        obs = TensorDict({"policy": o, "height_grid": torch.zeros(1, 187)}, batch_size=[1])
        mirrored, _ = t0_symmetry(env, obs)
        twice, _ = t0_symmetry(env, mirrored[1:])
        self.assertTrue(torch.equal(twice["policy"][1:, 66:834], o[:, 66:834]))

    def test_identity_mutations(self):
        from unittest.mock import patch
        import provenance, t0lib

        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            root = p / "repo"
            run = p / "run"
            root.mkdir()
            run.mkdir()
            f = root / "pyproject.toml"
            f.write_text("original")
            lib = root / "docs/research/morphology/evidence/library.json"
            lib.parent.mkdir(parents=True)
            lib.write_text("{}")
            snap = run / "source_snapshot"
            sh = snap / "docs/research/morphology/evidence/library.json"
            sh.parent.mkdir(parents=True)
            sh.write_text("{}")
            (snap / "pyproject.toml").write_text("original")
            table = provenance.inputs(root)
            t0lib.write_json(run / "input_identity.json", dict(files=table, backend={}))
            with patch.object(provenance, "backend_identity", return_value={}):
                self.assertFalse(provenance.verify_inputs(run, root))
                f.write_text("tampered")
                self.assertIn("pyproject.toml", provenance.verify_inputs(run, root))
                f.unlink()
                self.assertIn("pyproject.toml", provenance.verify_inputs(run, root))
            t0lib.write_json(run / "data.json", {})
            provenance.seal_outputs(run)
            self.assertTrue(
                any(x.startswith("missing_manifest_entry:") for x in provenance.verify_outputs(run))
            )
            (run / "data.json").write_text('{"tampered":true}')
            self.assertIn("data.json", provenance.verify_outputs(run))


if __name__ == "__main__":
    unittest.main(verbosity=2)
