"""Finite sparse collection measurement; no optimizer updates or checkpoints."""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import cProfile
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import pstats
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))


def reference_method(module, cls, method, revision):
    """Load only an accepted method body for a bounded in-process comparison."""
    path = Path(module.__file__).resolve().relative_to(ROOT)
    source = subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=ROOT, text=True)
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == cls.__name__)
    method_node = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == method)
    # Reference code runs against the current common modules; do not restore
    # retired task packages merely to load the old numerical implementation.
    for item in ast.walk(method_node):
        if isinstance(item, ast.ImportFrom) and item.level == 1 and item.module == "perturbations":
            item.module = "hexapod_mjlab.common.perturbations"
            item.level = 0
    namespace = dict(vars(module))
    # Legacy constructors owned one bundle per environment. Keep that fixture
    # in the comparison namespace, never in the optimized runtime implementation.
    if method == "__init__":
        bundles = [n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "_SensorBundle"]
        if bundles:
            exec(compile(ast.Module(body=bundles, type_ignores=[]), f"{revision}:{path}", "exec"), namespace)
    exec(compile(ast.Module(body=[method_node], type_ignores=[]), f"{revision}:{path}", "exec"), namespace)
    return namespace[method]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--task", choices=("flat", "terrain"), default="terrain")
    p.add_argument("--seed", type=int, default=20261027)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--collections", type=int, default=2)
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--variant", choices=("baseline", "scan", "cache", "optimized"), default="optimized")
    p.add_argument("--reference-rev", default="2c08bab09c2d97f966bfc6905b21562e9ef81978")
    p.add_argument("--collector-chunk", type=int)
    p.add_argument("--profile", action="store_true")
    args = p.parse_args()
    if args.num_envs < 1 or args.collections < 1 or args.warmups < 0:
        p.error("positive environment/collection counts and nonnegative warmups required")
    if args.collector_chunk is not None and (args.collector_chunk < 1 or args.variant not in ("cache", "optimized")):
        p.error("collector-chunk must be positive and used with cache/optimized")
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
    from hexapod_mjlab.tasks.confined_locomotion_2 import collector, scanner, sparse_raycast
    from hexapod_mjlab.tasks.confined_locomotion_2 import runtime
    from hexapod_mjlab.rl.confined_locomotion_2 import history
    from hexapod_mjlab.common import perturbations
    from hexapod_mjlab.tasks.confined_locomotion_2.config.terrain_memory import make_training_env_cfg, terrain_memory_formal_rl_cfg
    from hexapod_mjlab.tasks.confined_locomotion_2.rl.runner import ConfinedLocomotionRunner

    if args.variant == "baseline":
        runtime.TrainingRuntime._scan_rows_now = reference_method(runtime, runtime.TrainingRuntime, "_scan_rows_now", args.reference_rev)
        history.ScanWindow.write = reference_method(history, history.ScanWindow, "write", args.reference_rev)
    if args.variant in ("baseline", "scan"):
        collector.TrainingCollector._frame = reference_method(collector, collector.TrainingCollector, "_frame", args.reference_rev)
    if args.variant in ("baseline", "cache"):
        sparse_raycast.ChannelRaycaster.__init__ = reference_method(sparse_raycast, sparse_raycast.ChannelRaycaster, "__init__", args.reference_rev)
        sparse_raycast.ChannelRaycaster.update = reference_method(sparse_raycast, sparse_raycast.ChannelRaycaster, "update", args.reference_rev)
        scanner.ConfinedScanPipeline.__init__ = reference_method(scanner, scanner.ConfinedScanPipeline, "__init__", args.reference_rev)
        scanner.ConfinedScanPipeline.scan_rows = reference_method(scanner, scanner.ConfinedScanPipeline, "scan_rows", args.reference_rev)
    if args.collector_chunk is not None:
        collector._FRAME_ENCODING_CHUNK = args.collector_chunk
    report = {"passed": False, "task": args.task, "variant": args.variant, "num_envs": args.num_envs,
              "seed": args.seed, "warmups": args.warmups, "profiled": args.profile,
              "reference_rev": args.reference_rev, "optimizer_updates": 0,
              "collector_chunk": getattr(collector, "_FRAME_ENCODING_CHUNK", 128), "records": [],
              "source_sha256": {Path(m.__file__).name: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                                for m in (collector, scanner, sparse_raycast, runtime, history, perturbations)}}
    runner = env = None
    try:
        assert torch.cuda.is_available(), "real CUDA required"
        device = f"cuda:{args.gpu}"
        report["gpu"] = torch.cuda.get_device_name(args.gpu)
        report["torch"] = torch.__version__
        if args.task == "terrain":
            cfg = terrain_memory_formal_rl_cfg(seed=args.seed, num_envs=args.num_envs,
                                              max_iterations=1, height_loss_weight=1.0)
        else:
            from hexapod_mjlab.tasks.confined_locomotion_2.config.flat_baseline import confined_locomotion_formal_rl_cfg, make_training_env_cfg
            cfg = confined_locomotion_formal_rl_cfg(seed=args.seed, num_envs=args.num_envs, max_iterations=1)
        cfg.lidar_config = {"raycast_mode": "sparse", "far_sampling": "hit_only_10m",
                            "sparse_selection_path": str(ROOT / "src/hexapod_mjlab/tasks/confined_locomotion_2/assets/fixed_channels/v1_hit10/near_x6_far_x10.npz")}
        env = ManagerBasedRlEnv(make_training_env_cfg(num_envs=args.num_envs, seed=args.seed), device=device)
        runner = ConfinedLocomotionRunner(RslRlVecEnvWrapper(env, clip_actions=None), asdict(cfg), args.output / "runner", device)
        counters = defaultdict(lambda: {"calls": 0, "host_s_inclusive": 0.0})
        def instrument(obj, name, label):
            original = getattr(obj, name)
            def wrapped(*a, **kw):
                start = time.perf_counter()
                try:
                    return original(*a, **kw)
                finally:
                    counters[label]["calls"] += 1
                    counters[label]["host_s_inclusive"] += time.perf_counter() - start
            setattr(obj, name, wrapped)
        for obj, name, label in ((runner.runtime, "prepare", "prepare"),
                                 (runner.runtime, "labels", "labels"),
                                 (runner.runtime, "step", "runtime_step"),
                                 (runner.runtime.wrapper, "step", "physics_reward_step"),
                                 (runner.runtime.pipeline, "scan_rows", "scan_rows"),
                                 (runner.collector, "_frame", "frame_cache_encode")):
            instrument(obj, name, label)
        for index in range(args.warmups + args.collections):
            counters.clear()
            runner.phase = "collecting"
            runner.learning_metrics.begin_iteration()
            profiler = cProfile.Profile() if args.profile and index == args.warmups else None
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            before_rows = runner.collector.frame_encoding_rows
            start = time.perf_counter()
            if profiler:
                profiler.enable()
            buffer = runner.collector.collect()
            if profiler:
                profiler.disable()
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            metrics = runner.learning_metrics.finish_iteration()
            if index >= args.warmups:
                record = {"collection": index - args.warmups, "collect_s": elapsed,
                          "profiled": profiler is not None, "env_steps_per_s": args.num_envs * 24 / elapsed,
                          "torch_peak_MiB": torch.cuda.max_memory_allocated(device) / 2**20,
                          "frame_encoding_rows": runner.collector.frame_encoding_rows - before_rows,
                          "host_sections_inclusive": dict(counters), "mean_step_reward": metrics["Reward/actual"]}
                report["records"].append(record)
                print(json.dumps(record, ensure_ascii=False), flush=True)
            if profiler:
                stream = io.StringIO()
                pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(45)
                (args.output / "collection_profile.txt").write_text(stream.getvalue())
            del buffer
        report["passed"] = True
    finally:
        if runner is not None:
            runner.close()
        elif env is not None:
            env.close()
        (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
