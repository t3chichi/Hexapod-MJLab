"""GPU environment, sensor, clock/reset, teacher parity and collection checks."""

import _bootstrap
import torch
from hexapod_mjlab.tasks.envelope_adaptive_2.env import (
    EA2Env,
    SLAdapter,
    make_platform_cfg,
    PlatformState,
    make_env,
)
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from hexapod_mjlab.tasks.envelope_adaptive_2.config.parameters import EA2Cfg
from hexapod_mjlab.tasks.envelope_adaptive_2.lidar import EnvelopeLidar
from hexapod_mjlab.tasks.envelope_adaptive_2.math_utils import wrap_to_pi, selected_channel_mask
from hexapod_mjlab.tasks.envelope_adaptive_2.quaternion import quat_from_euler_xyz
import json
from hexapod_mjlab.tasks.envelope_adaptive_2.teachers.batch import BatchTeacher
from whole_polygon_reference import WholePolygonChecker
from hexapod_mjlab.tasks.envelope_adaptive_2.sl.dataset import collect_map, save_map, load_map
from hexapod_mjlab.tasks.envelope_adaptive_2.sl.evaluate import closed_loop_rollout
from hexapod_mjlab.tasks.envelope_adaptive_2.sl.config import SLConfig
from hexapod_mjlab.tasks.envelope_adaptive_2.sl.model import EnvelopeNet
from hexapod_mjlab.tasks.envelope_adaptive_2.sl.dataset import build_env
import tempfile


def check_platform():
    env = EA2Env(make_platform_cfg(2, episode_length_s=0.06), state=PlatformState(2, "cuda:0"))
    try:
        assert env.sim.mj_model.nq == 0 and env.sim.mj_model.nv == 0
        assert env.single_action_space.shape == (5,)
        adapter = SLAdapter(env)
        assert adapter.reset().shape == (2, 190)
        assert env.ea2.logical_step == 1
        before = env.ea2.age.clone()
        env.get_observations()
        env.get_observations()
        assert torch.equal(env.ea2.age, before)
        adapter.step(torch.zeros((2, 5), device=env.device))
        assert env.ea2.logical_step == 2
        env.reset(env_ids=torch.tensor([0], device=env.device))
        assert env.ea2.age.tolist() == [0, 2]
        _, _, done, _ = adapter.step(torch.zeros((2, 5), device=env.device))
        assert done.tolist() == [False, True]
        assert env.ea2.age.tolist() == [1, 0]
        assert abs(env.step_dt - 0.02) < 1e-10
        print(
            "PLATFORM PASS: jointless GPU scene, 190/5, 50Hz, cached reads, explicit/partial/auto reset"
        )
    finally:
        env.close()


def check_sensor_motion():
    torch.set_num_threads(1)
    cfg = EA2Cfg()
    cfg.envelope.oracle_kind = "legacy"
    cfg.env.num_envs = 2
    cfg.lidar.enable_sensor_noise = False
    env = make_env(cfg)
    adapter = SLAdapter(env)
    try:
        obs = adapter.reset()
        s = env.ea2
        assert obs.shape == (2, 190) and s.sensor._lidar_timer == 5
        prev = s._oracle_smoother.prev_s.clone()
        counter = s._oracle_smoother.counter.clone()
        env.get_observations()
        env.get_observations()
        assert torch.equal(prev, s._oracle_smoother.prev_s) and torch.equal(
            counter, s._oracle_smoother.counter
        )
        z = torch.zeros((2, 5), device=env.device)
        adapter.step(z)
        ids = torch.tensor([0], device=env.device)
        other = s.base_pos[1].clone()
        env.reset(env_ids=ids)
        assert torch.equal(other, s.base_pos[1]) and (s.range_image[0] == s.range_max).all()
        for _ in range(3):
            adapter.step(z)
            assert (s.range_image[0] == s.range_max).all()
        adapter.step(z)
        assert not s.range_image_stale.any() and (s.range_image[0] < s.range_max).any()
        assert env.ea2.logical_step == 6
        fullcfg = copy.deepcopy(cfg)
        fullcfg.lidar.use_reduced_raycast = False
        full = EnvelopeLidar(fullcfg, s.map_data, 2, env.device, capture_debug=True)
        reduced = EnvelopeLidar(cfg, s.map_data, 2, env.device, capture_debug=True)
        pos = torch.tensor([[0, 0, 0.52], [10, 0, 0.52]], dtype=torch.float32, device=env.device)
        heading = torch.tensor([0.31, -0.42], device=env.device)
        full.scan(pos, heading)
        reduced.scan(pos, heading)
        a = full.lidar_sensor.lidar_dist_tensor.view(2, -1)[:, full.selected_ray_indices]
        b = reduced._reduced_dist_tensor.view(2, -1)
        assert torch.equal(a < 60, b < 60)
        assert torch.allclose(a, b, atol=0.0001, rtol=1e-05), (a - b).abs().max()
        assert torch.allclose(full.range_image, reduced.range_image, atol=0.0001, rtol=1e-05)
        assert not full._debug_is_reduced and reduced._debug_is_reduced
        red, green = selected_channel_mask(
            full._debug_dists[0], full.selected_ray_indices, False, 60
        )
        assert red.any() and green.any() and (not (red & green).any())
        q = quat_from_euler_xyz(torch.zeros_like(s.heading), torch.zeros_like(s.heading), s.heading)
        assert torch.allclose(s.carrier_pose()[:, 3:], q[:, [3, 0, 1, 2]], atol=1e-06)
        env.cfg.episode_length_s = 0.1
        env.reset()
        for i in range(5):
            _, _, done, _ = adapter.step(z)
            assert bool(done.all()) == (i == 4)
        s._replan_reset_buf[0] = True
        _, _, done, _ = adapter.step(z)
        assert done.tolist() == [True, False]
        print(
            "SENSOR GPU PASS: full/reduced rays, miss sentinel, 10Hz/reset stale, cache, timeout and path failure"
        )
    finally:
        env.close()
    cfg.attitude.enable = True
    env = make_env(cfg)
    try:
        adapter = SLAdapter(env)
        adapter.reset()
        for _ in range(6):
            adapter.step(torch.zeros((2, 5), device=env.device))
        for t in [
            env.ea2.attitude_pitch,
            env.ea2.attitude_roll,
            env.ea2.base_pos,
            env.ea2.observation,
        ]:
            assert torch.isfinite(t).all()
        print("T2 ATTITUDE PASS: packaged replay, finite pitch/roll/height/ranges")
    finally:
        env.close()


def check_teacher_timing():
    torch.set_num_threads(1)
    cfg = EA2Cfg()
    cfg.env.num_envs = 2
    cfg.path.path_plan_workers = 0
    env = make_env(cfg)
    times = []
    valid = emergency = total = 0
    try:
        env.reset()
        s = env.ea2
        mirror = BatchTeacher(s.map_data.rects)
        checker = WholePolygonChecker(s.map_data.rects)

        def initial(ids):
            mirror.reset(ids)
            idx = torch.tensor(ids, device=env.device)
            r = mirror.evaluate(s.base_pos[idx, :2], s.heading[idx], s.ego_motion[idx], ids)
            assert torch.allclose(r["s"], s.oracle_s[idx], atol=1e-06, rtol=0)
            assert not s.oracle_rate_checked[idx].any()

        initial([0, 1])
        zero = torch.zeros((2, 5), device=env.device)
        for t in range(100):
            prev = {k: v.copy() for k, v in s._teacher.oracle.prev.items()}
            counter = {k: v.copy() for k, v in s._teacher.oracle.counter.items()}
            env.get_observations()
            env.get_observations()
            assert prev.keys() == s._teacher.oracle.prev.keys()
            for key in prev:
                assert np.array_equal(prev[key], s._teacher.oracle.prev[key])
            for key in counter:
                assert np.array_equal(counter[key], s._teacher.oracle.counter[key])
            _, _, terminated, truncated, info = env.step(zero)
            frame = info["executed_frame"]
            ego = frame["observation"][:, -3:] * torch.tensor([1.5, 1.0, 1.5], device=env.device)
            result = mirror.evaluate(frame["pos"], frame["heading"], ego)
            assert torch.allclose(result["s"], frame["target"], atol=1e-05, rtol=0)
            for name in [
                "geometric_valid",
                "supervision_valid",
                "emergency",
                "rate_checked",
                "rate_valid",
            ]:
                assert torch.equal(result[name], frame[name]), (t, name)
            for e in range(2):
                if frame["geometric_valid"][e]:
                    assert (
                        checker.clearance(
                            result["params"][e].cpu().numpy(),
                            frame["pos"][e].cpu().numpy(),
                            float(frame["heading"][e]),
                        )
                        >= 0.1 - 1e-06
                    )
            valid += int(frame["supervision_valid"].sum())
            emergency += int(frame["emergency"].sum())
            total += 2
            times.append(s.teacher_elapsed_s)
            done = (terminated | truncated).nonzero().flatten().tolist()
            if done:
                initial(done)
            if t == 10:
                other = s._teacher.oracle.prev[1].copy()
                env.reset(env_ids=torch.tensor([0], device=env.device))
                assert np.array_equal(other, s._teacher.oracle.prev[1])
                initial([0])
            if t == 40:
                env.cfg.episode_length_s = 0.1
                env.reset()
                initial([0, 1])
        report = {
            "frames": total,
            "supervision_frames": valid,
            "emergency_frames": emergency,
            "mean_teacher_batch_s": float(np.mean(times)),
            "p95_teacher_batch_s": float(np.quantile(times, 0.95)),
            "num_envs": 2,
            "cached_reads": "pass",
            "partial_reset": "pass",
            "online_offline": "pass",
            "whole_polygon": "pass",
        }
        print("TEACHER GPU PASS", json.dumps(report), flush=True)
    finally:
        env.close()


def check_collection():
    torch.set_num_threads(1)
    data = collect_map(1, num_envs=2, num_steps=37)
    assert data.obs.shape == (37, 2, 190) and data.target.shape == (37, 2, 5)
    assert torch.isfinite(data.obs).all() and torch.isfinite(data.target).all()
    assert ((data.target >= 0) & (data.target <= 1)).all()
    with tempfile.TemporaryDirectory() as directory:
        p = str(Path(directory) / "data.pt")
        save_map(data, p)
        loaded = load_map(p)
        assert torch.equal(loaded.obs, data.obs) and torch.equal(loaded.target, data.target)
    env = build_env(1, 2)
    try:
        net = EnvelopeNet().to(env.device)
        result = closed_loop_rollout(env, net, SLConfig(), num_steps=36)
        assert result.n_steps == 6
        for name in [
            "step_reward",
            "same_frame_mse",
            "executed_frame_mse",
            "pred_target_mse",
            "area",
            "oracle_area",
        ]:
            assert torch.isfinite(torch.tensor(getattr(result, name))), name
    finally:
        env.close()
    print("DATA GPU PASS: real same-frame data collection, save/load, rollout metrics")


def check_legacy_motion():
    from source_reference import load_reference, source_root

    load_reference()  # validates every original-source hash before executing a reference
    cfg = EA2Cfg()
    cfg.envelope.oracle_kind = "legacy"
    cfg.env.num_envs = 2
    env = make_env(cfg)
    try:
        env.reset()
        s = env.ea2
        source = (
            source_root()
            / "legged_gym/legged_gym/envs/el_4090/envelope_adaptive_2/el_4090_ea2_env.py"
        )
        tree = ast.parse(source.read_text())
        cls = next((n for n in tree.body if isinstance(n, ast.ClassDef)))
        fn = next(
            (
                n
                for n in cls.body
                if isinstance(n, ast.FunctionDef) and n.name == "_step_kinematics_batched"
            )
        )
        ns = {"torch": torch, "wrap_to_pi": wrap_to_pi}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), ns)
        attrs = [
            "s",
            "v",
            "heading",
            "tangent",
            "tangent_rate",
            "delta_target",
            "delta_actual",
            "omega",
            "base_pos",
            "_turn_in_place",
            "_turn_target",
            "ego_motion",
        ]
        ref = SimpleNamespace(**{name: getattr(s, name).clone() for name in attrs})
        ref.num_envs = s.num_envs
        ref.device = s.device
        ref.dt = s.dt
        ref.cfg = s.cfg
        ref._path_batch = s._path_batch
        ref._attitude = None
        ref._batch_replan = lambda ids: (_ for _ in ()).throw(
            AssertionError("probe reached path end")
        )
        for _ in range(20):
            s._step_kinematics_batched()
            ns["_step_kinematics_batched"](ref)
            for name in attrs:
                assert torch.allclose(
                    getattr(s, name), getattr(ref, name), atol=1e-06, rtol=0
                ), name
    finally:
        env.close()


def main():
    import argparse

    checks = {
        "platform": check_platform,
        "sensor": check_sensor_motion,
        "teacher": check_teacher_timing,
        "collection": check_collection,
        "legacy-motion": check_legacy_motion,
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", nargs="+", choices=checks)
    args = parser.parse_args()
    torch.set_num_threads(1)
    for name in args.check or ("platform", "sensor", "teacher", "collection"):
        checks[name]()
        print("PASS", name, flush=True)


if __name__ == "__main__":
    main()
