"""Numerical comparison against frozen original pure modules, including gradients."""

import _bootstrap
import numpy as np
import torch
import ast
from pathlib import Path
from types import SimpleNamespace
from source_reference import load_reference, source_root
from hexapod_mjlab.tasks.envelope_adaptive_2 import (
    contracts,
    geometry,
    oracle,
    target_smoother,
    airy_mount,
)
from hexapod_mjlab.tasks.envelope_adaptive_2.config.condition import ENVELOPE_SPEC
from hexapod_mjlab.tasks.envelope_adaptive_2.config.parameters import EA2Cfg
from hexapod_mjlab.tasks.envelope_adaptive_2.map_generator import generate_map
from hexapod_mjlab.tasks.envelope_adaptive_2.path_planner import plan_path


def same(a, b):
    if isinstance(a, torch.Tensor):
        assert a.shape == b.shape and torch.allclose(a, b, atol=1e-6, rtol=0), (a - b).abs().max()
    else:
        assert np.array_equal(a, b)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(123)
    ref = load_reference()
    spec = ref["haa"].load_envelope_condition_spec()
    assert ENVELOPE_SPEC == spec or vars(ENVELOPE_SPEC) == vars(spec)
    low = torch.tensor(ENVELOPE_SPEC.low[:5])
    high = torch.tensor(ENVELOPE_SPEC.high[:5])
    p = low + torch.rand(12, 5) * (high - low)
    heading = torch.linspace(-3.0, 3.0, 12)
    pos = torch.rand(12, 2) - 0.5
    y = -37 + (torch.arange(740) + 0.5) * 0.1
    field = (0.55 - y.abs()).clamp_min(0)[:, None].expand(740, 740).contiguous()
    for func in ["compute_hex_vertices", "hex_body_sample_points"]:
        args = p.unbind(-1) if func == "compute_hex_vertices" else (p,)
        same(getattr(geometry, func)(*args), getattr(ref["envelope_geometry"], func)(*args))
    for fn in ["hex_collision_violation", "hex_collision_edge_sum"]:
        args = (p, heading, pos, field, 0.1, 0.1)
        same(getattr(geometry, fn)(*args), getattr(ref["envelope_geometry"], fn)(*args))
    for mode in ["axis", "coupled"]:
        args = (heading, pos, field, low, high)
        kw = dict(
            margin=0.2,
            step=0.05,
            max_dist=5.0,
            soft_dof_pos_limit=0.95,
            interp_crossing=True,
            group_mode=mode,
        )
        a = oracle.compute_direct_oracle_params_with_stats(*args, **kw)
        b = ref["envelope_oracle"].compute_direct_oracle_params_with_stats(*args, **kw)
        for x, z in zip(a, b):
            same(x, z)
    grads = []
    for module in [geometry, ref["envelope_geometry"]]:
        q = p.clone().requires_grad_()
        loss = module.hex_collision_edge_sum(
            q, heading, pos, field, 0.1, 0.1, sampling="bilinear"
        ).mean()
        loss.backward()
        assert q.grad.abs().sum() > 0
        grads.append(q.grad)
    same(*grads)
    smoothers = [
        m.RateLimitedOracle(12, 0.02, "cpu", low, high)
        for m in [target_smoother, ref["target_smoother"]]
    ]
    for t in range(80):
        q = low + torch.rand(12, 5) * (high - low)
        if t == 35:
            for s in smoothers:
                s.reset_ids([1, 5])
        same(*(s.update(q) for s in smoothers))
    same(
        airy_mount.generate_full_airy_directions(),
        ref["airy_mount"].generate_full_airy_directions(),
    )
    channels = airy_mount.load_selected_channels()
    gold = ref["airy_mount"].load_selected_channels()
    same(channels["ray_indices"], gold["ray_indices"])
    same(channels["ray_directions"], gold["ray_directions"])
    # Same-backend noise comparison to the frozen original formula, no old imports.
    from hexapod_mjlab.utils.lidar import LidarSensor

    original = source_root() / "legged_gym/legged_gym/utils/LidarSensor/lidar_sensor.py"
    cls = next(n for n in ast.parse(original.read_text()).body if isinstance(n, ast.ClassDef))
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "apply_noise")
    fn.returns = None
    for arg in fn.args.args:
        arg.annotation = None
    ns = {"torch": torch}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(original), "exec"), ns)
    ctx = SimpleNamespace(
        sensor_cfg=SimpleNamespace(
            enable_sensor_noise=True, pixel_std_dev_multiplier=0.02, pixel_dropout_prob=0.02
        ),
        far_plane=60.0,
    )
    outputs = []
    for func in [LidarSensor.apply_noise, ns["apply_noise"]]:
        torch.manual_seed(99)
        d = torch.full((2, 1, 1000, 1), 10.0)
        p = torch.zeros((2, 1, 1000, 1, 3))
        p[..., 0] = 10.0
        outputs.append(func(ctx, p, d))
    for a, b in zip(*outputs):
        same(a, b)
    cfg = EA2Cfg()
    mapkw = vars(cfg.map).copy()
    mapkw["grid_shape"] = tuple(mapkw["grid_shape"])
    mapkw["near_obstacle_range"] = tuple(mapkw["near_obstacle_range"])
    pillar_kw = {k.removeprefix("pillar_"): v for k, v in vars(cfg.obstacles).items()}
    for seed in [1, 7]:
        a = generate_map(contracts.MapGenCfg(**mapkw), contracts.PillarFieldCfg(**pillar_kw), seed)
        b = ref["map_generator"].generate_map(
            ref["_contracts"].MapGenCfg(**mapkw),
            ref["_contracts"].PillarFieldCfg(**pillar_kw),
            seed,
        )
        for name in ["occupancy", "inflated", "distance_field", "vertices", "triangles"]:
            same(getattr(a, name), getattr(b, name))
        assert a.acceptance == b.acceptance
        kw = {k: getattr(cfg.path, k) for k in vars(contracts.PathCfg())}
        free = np.argwhere(a.inflated == 0)
        start = -37 + (free[len(free) // 3][::-1] + 0.5) * 0.1
        goal = -37 + (free[len(free) // 2][::-1] + 0.5) * 0.1
        pa = plan_path(
            a.occupancy, a.inflated, start, goal, contracts.PathCfg(**kw), np.random.default_rng(9)
        )
        pb = ref["path_planner"].plan_path(
            b.occupancy,
            b.inflated,
            start,
            goal,
            ref["_contracts"].PathCfg(**kw),
            np.random.default_rng(9),
        )
        for name in ["points", "yaws", "arc", "corner_arcs", "corner_targets"]:
            same(getattr(pa, name), getattr(pb, name))
    print(
        "T2 CPU PASS: source hashes, spec, geometry, both oracle modes, gradients, smoother reset, Airy channels, maps 1/7 and paths"
    )


if __name__ == "__main__":
    main()
