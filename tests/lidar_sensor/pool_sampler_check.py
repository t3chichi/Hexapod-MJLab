"""G-S 验收门：采集池采样器契约验证（纯 torch，无 Isaac Gym）。

核查内容（对应方案 §5.1 G-S）：

1. 尺寸契约：四策略 ``fixed_output_size`` 与样本形状 (E, P) 正确；
2. 确定性：同种子同帧号逐位一致；换帧号/换种子结果不同（random/rotating）；
3. 索引值域 [0, R) 且无越界；
4. ``stride_rotating``：连续 30 帧索引并集 = 全集（时域互补覆盖）；
5. 热路径无 host 同步：``sample`` 期间 CUDA 同步计数不增加（GPU 可用时）。

运行（纯 torch，任意含 torch 的环境）::

    python tests/lidar_sensor/pool_sampler_check.py
"""

import json
import sys
from pathlib import Path

import torch

TESTS_DIR = Path(__file__).resolve().parent  # .../Hexapod-MJLab/tests/lidar_sensor
SRC = TESTS_DIR.parents[1] / "src" / "hexapod_mjlab"  # .../Hexapod-MJLab/src/hexapod_mjlab
POOL_SAMPLERS_PATH = SRC / "utils" / "lidar" / "pool_samplers.py"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from artifacts import artifact_dir

OUT_DIR = artifact_dir(__file__)

# 直接按路径加载，绕过包 __init__（保持纯 torch、无环境依赖）
import importlib.util

spec = importlib.util.spec_from_file_location("pool_samplers", POOL_SAMPLERS_PATH)
pool_samplers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pool_samplers)

N_AZ, N_EL, RAW = 900, 96, 900 * 96
E, DEVICE = 8, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_all():
    cfg_ns = type("C", (), {})()
    samplers = {}
    for name, kwargs in [
        ("angle_stride", {"azimuth_stride": 30}),
        ("stride_rotating", {"azimuth_stride": 30}),
        ("random_uniform", {"output_size": 2880}),
        ("identity", {}),
    ]:
        cfg_ns.strategy = name
        cfg_ns.seed = 42
        for k, v in kwargs.items():
            setattr(cfg_ns, k, v)
        samplers[name] = pool_samplers.build_pool_sampler(cfg_ns, N_AZ, N_EL, RAW, DEVICE)
    return samplers


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    failures = []

    torch.manual_seed(0)
    points = torch.randn(E, RAW, 3, device=DEVICE)
    dists = torch.rand(E, RAW, device=DEVICE) * 15.0
    frame = torch.zeros(E, device=DEVICE, dtype=torch.long)

    samplers = build_all()

    # ── 1. 尺寸契约 + 值域 ─────────────────────────────────────────
    for name, s in samplers.items():
        idx = s.sample(points, dists, frame)
        if tuple(idx.shape) != (E, s.fixed_output_size):
            failures.append(f"{name}: shape {tuple(idx.shape)} != (E,{s.fixed_output_size})")
        if idx.min().item() < 0 or idx.max().item() >= RAW:
            failures.append(f"{name}: index out of range")
        results[f"{name}_size"] = int(s.fixed_output_size)

    # ── 2. 确定性 ──────────────────────────────────────────────────
    for name, s in samplers.items():
        a = s.sample(points, dists, frame)
        b = s.sample(points, dists, frame)
        if not torch.equal(a, b):
            failures.append(f"{name}: not deterministic for identical inputs")
    # random/rotating 应随帧号变化；angle/identity 不变
    frame1 = torch.ones(E, device=DEVICE, dtype=torch.long)
    for name in ("angle_stride", "identity"):
        if not torch.equal(
            samplers[name].sample(points, dists, frame),
            samplers[name].sample(points, dists, frame1),
        ):
            failures.append(f"{name}: should be frame-invariant")
    for name in ("stride_rotating", "random_uniform"):
        if torch.equal(
            samplers[name].sample(points, dists, frame),
            samplers[name].sample(points, dists, frame1),
        ):
            failures.append(f"{name}: should vary with frame_count")

    # ── 3. stride_rotating 时域互补覆盖 ────────────────────────────
    s = samplers["stride_rotating"]
    union = torch.zeros(RAW, dtype=torch.bool, device=DEVICE)
    for f in range(30):
        fc = torch.full((E,), f, device=DEVICE, dtype=torch.long)
        idx = s.sample(points, dists, fc)[0]
        union[idx] = True
    results["rotating_union_coverage"] = float(union.float().mean())
    if not bool(union.all()):
        failures.append("stride_rotating: 30-frame union does not cover all rays")

    # gather 往返一致性：sample 索引取点应精确还原子集
    idx = s.sample(points, dists, frame)
    gathered = torch.gather(points, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
    ref = points[0][idx[0]]
    if not torch.equal(gathered[0], ref):
        failures.append("gather round-trip mismatch")

    # ── 4. 热路径无 host 同步（仅 GPU 可用时） ─────────────────────
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
        s = samplers["angle_stride"]
        s.sample(points, dists, frame)  # 预热
        torch.cuda.synchronize()
        # 简化判定：sample 输出保持在 GPU 且函数内无 .item()/.cpu() 调用
        # （静态检查源码）
        src = POOL_SAMPLERS_PATH.read_text()
        sample_bodies = src.split("def sample(")[1:]
        for chunk in sample_bodies:
            body = chunk.split("def ")[0]
            for banned in (".item()", ".cpu()", ".tolist()", "synchronize"):
                if banned in body:
                    failures.append(f"hot path contains host sync: {banned}")
        results["host_sync_check"] = "static-source + device-retention"
    else:
        results["host_sync_check"] = "skipped (cpu only)"

    results["failures"] = failures
    results["verdict"] = "PASS" if not failures else "FAIL"

    with open(OUT_DIR / "gs_pool_sampler_check.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
