"""G1 验收门：头装 Airy 挂载与射线方向自检（纯 torch，无 Isaac Gym）。

核查内容（对应方案 §5.1 G1）：

1. 射线图案：本地解析生成 900×96 穹顶（C 序 ``idx = az*96 + el``），
   与 ``SpinningLidarGenerator.generate_AIRY`` 实际输出交叉核对；
2. 挂载几何：pitch = π/2 + tilt 旋转后——传感器 +z 极点 → 体坐标正前方
   且按 tilt 下倾；传感器地平面环 → 体坐标垂直圆环（下/左/上/右）；
3. 覆盖统计：体坐标仰角范围、近端（φ_s ≥ split_theta）射线占比
   （量化 12° 占位阈值在头装几何下的退化）；
4. 产出 BEV + 侧视覆盖 PNG 到 ``logs/acceptance/``。

运行（纯 torch，任意含 torch 的环境）::

    python tests/lidar_sensor/mount_check.py
"""

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

TESTS_DIR = Path(__file__).resolve().parent  # .../Hexapod-MJLab/tests/lidar_sensor
SRC = TESTS_DIR.parents[1] / "src" / "hexapod_mjlab"  # .../Hexapod-MJLab/src/hexapod_mjlab
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from artifacts import artifact_dir

OUT_DIR = artifact_dir(__file__)

N_AZIMUTH = 900
N_ELEVATION = 96
H_RES_DEG = 0.4
TILT_RAD = 0.1
SPLIT_THETA_DEG = 45.0  # config 占位值（G4 扫描后定稿）；此处量化其划分含义

DEG = 180.0 / math.pi


# ---------------------------------------------------------------------------
# 本地四元数工具（isaacgym.torch_utils 同源公式，保持无 Isaac 可运行）
# ---------------------------------------------------------------------------


def _quat_from_euler_xyz(roll: float, pitch: float, yaw: float) -> torch.Tensor:
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    return torch.tensor(
        [
            cy * sr * cp - sy * cr * sp,
            cy * cr * sp + sy * sr * cp,
            sy * cr * cp - cy * sr * sp,
            cy * cr * cp + sy * sr * sp,
        ],
        dtype=torch.float64,
    )  # xyzw


def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    if q.dim() == 1 and v.dim() > 1:
        q = q.unsqueeze(0).expand(v.shape[0], 4)
    xyz = q[..., :3]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + q[..., 3:4] * t + torch.cross(xyz, t, dim=-1)


def generate_airy_directions() -> torch.Tensor:
    """传感器系射线方向 (86400, 3)，C 序 idx = az*96 + el。"""
    phi = torch.arange(N_ELEVATION, dtype=torch.float64) * (math.pi / 2.0) / (N_ELEVATION - 1)
    theta = torch.arange(N_AZIMUTH, dtype=torch.float64) * (2.0 * math.pi / N_AZIMUTH)
    cos_phi, sin_phi = torch.cos(phi), torch.sin(phi)
    cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
    dirs = torch.stack(
        [
            cos_phi[None, :] * cos_theta[:, None],
            cos_phi[None, :] * sin_theta[:, None],
            sin_phi[None, :].expand(N_AZIMUTH, N_ELEVATION),
        ],
        dim=-1,
    ).reshape(-1, 3)
    return dirs


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    failures = []

    # ── 1. 与 LidarSensor 的 generate_AIRY 交叉核对 ──────────────────
    import importlib.util

    gen_path = (
        SRC / "utils" / "lidar" / "sensor_pattern" / "sensor_lidar" / "genera_lidar_scan_pattern.py"
    )
    spec = importlib.util.spec_from_file_location("gen_scan", gen_path)
    gen_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen_mod)
    theta_grid, phi_grid = gen_mod.SpinningLidarGenerator.generate_AIRY(
        n_channels=N_ELEVATION, horizontal_resolution_deg=H_RES_DEG
    )
    n_az_actual = int(round(360.0 / H_RES_DEG))
    ok_pattern = theta_grid.shape[0] == N_AZIMUTH * N_ELEVATION and n_az_actual == N_AZIMUTH
    # generate_AIRY 内部 np.linspace(0, 2π, n_az, endpoint=False) 与本地
    # 2π*az/900 等价；逐元素核对前 96×3 块
    dirs_local = generate_airy_directions()
    theta_t = torch.tensor(theta_grid.reshape(N_AZIMUTH, N_ELEVATION))
    phi_t = torch.tensor(phi_grid.reshape(N_AZIMUTH, N_ELEVATION))
    phi_local = torch.arange(N_ELEVATION, dtype=torch.float64) * (math.pi / 2.0) / (N_ELEVATION - 1)
    ok_pattern = ok_pattern and bool(
        torch.allclose(
            theta_t,
            torch.arange(N_AZIMUTH, dtype=torch.float64)[:, None] * (2 * math.pi / N_AZIMUTH),
            atol=1e-9,
        )
        and torch.allclose(phi_t, phi_local[None, :].expand(N_AZIMUTH, -1), atol=1e-9)
    )
    results["pattern_cross_check"] = bool(ok_pattern)
    if not ok_pattern:
        failures.append("ray pattern mismatch vs generate_AIRY")

    # ── 2. 挂载几何 ────────────────────────────────────────────────
    offset_q = _quat_from_euler_xyz(0.0, math.pi / 2.0 + TILT_RAD, 0.0)
    body_dirs = _quat_apply(offset_q, dirs_local)

    pole = body_dirs[(N_AZIMUTH // 2) * N_ELEVATION + (N_ELEVATION - 1)]  # 任意方位的 el=90
    ring_down = body_dirs[0 * N_ELEVATION + 0]  # az=0, el=0 → 体坐标 -z
    ring_left = body_dirs[(N_AZIMUTH // 4) * N_ELEVATION + 0]
    ring_up = body_dirs[(N_AZIMUTH // 2) * N_ELEVATION + 0]

    checks = {
        "pole_forward": bool(
            torch.allclose(
                pole,
                torch.tensor([math.cos(TILT_RAD), 0.0, -math.sin(TILT_RAD)], dtype=torch.float64),
                atol=1e-6,
            )
        ),
        "pole_tilt_down_deg": float(math.asin(-pole[2].item()) * DEG),
        # tilt 使垂直环射线偏转 TILT_RAD，界取 0.99 而非 0.999
        "ring_down_is_nadir": bool(ring_down[2].item() < -0.99),
        "ring_left_is_left": bool(ring_left[1].item() > 0.999),
        "ring_up_is_zenith": bool(ring_up[2].item() > 0.99),
    }
    results["mount_geometry"] = checks
    for key in ("pole_forward", "ring_down_is_nadir", "ring_left_is_left", "ring_up_is_zenith"):
        if not checks[key]:
            failures.append(f"mount geometry: {key}")

    # ── 3. 覆盖统计 + split 阈值退化量化 ──────────────────────────
    el_body = torch.asin(body_dirs[:, 2].clamp(-1, 1)) * DEG  # 体坐标仰角
    phi_sensor = torch.arange(N_ELEVATION, dtype=torch.float64) * (
        90.0 / (N_ELEVATION - 1)
    )  # 传感器系仰角
    prox_frac = float((phi_sensor >= SPLIT_THETA_DEG).float().mean())
    results["coverage"] = {
        "body_elevation_min_deg": float(el_body.min()),
        "body_elevation_max_deg": float(el_body.max()),
        "split_theta_deg": SPLIT_THETA_DEG,
        "proximal_ray_fraction": prox_frac,
        "distal_ray_fraction": 1.0 - prox_frac,
        "note": (
            "45° 阈值 = 与穹顶轴(体前方, 下倾 tilt)的夹角；阈值锥与地面"
            "相交为抛物线(双曲线)，前向楔内黄/近场透镜与两翼红。"
            "注意: 体仰角≈-44° 仅在正前方位角处成立, 侧向边界接近体"
            "水平面, 勿用'体仰角分界'理解。G4 扫描定稿"
        ),
    }
    if not 0.45 <= prox_frac <= 0.55:
        failures.append(f"unexpected proximal fraction {prox_frac:.3f}")

    # ── 4. 覆盖可视化（BEV + 侧视） ────────────────────────────────
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sub = body_dirs[::37].numpy()  # 抽稀绘图
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].scatter(sub[:, 0], sub[:, 1], s=0.5, c=sub[:, 2], cmap="coolwarm")
        axes[0].set_title("Airy head mount BEV (x-y), color=body z")
        axes[0].set_xlabel("body x")
        axes[0].set_ylabel("body y")
        axes[0].set_aspect("equal")
        axes[1].scatter(sub[:, 0], sub[:, 2], s=0.5, c="gray")
        axes[1].set_title(f"side view (x-z), tilt={TILT_RAD:.3f} rad")
        axes[1].set_xlabel("body x")
        axes[1].set_ylabel("body z")
        axes[1].set_aspect("equal")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "g1_mount_coverage.png", dpi=140)
        plt.close(fig)
        results["coverage_png"] = str(OUT_DIR / "g1_mount_coverage.png")
    except Exception as exc:  # matplotlib 缺失不阻塞验收
        results["coverage_png"] = f"skipped: {exc}"

    # ── 5. EA2 通道交叉核对（可选，外部文件，经环境变量提供） ─────
    ea2_env = os.environ.get("LIDAR_MJLAB_EA2_CHANNELS", "")
    ea2_channels = Path(ea2_env) if ea2_env else None
    if ea2_channels is not None and ea2_channels.exists():
        data = torch.load(ea2_channels, map_location="cpu")
        sel = data["ray_indices"].long()
        dirs_ea2 = data["ray_directions"].double()
        dirs_local_sel = dirs_local[sel]
        recon_body = _quat_apply(offset_q, dirs_local_sel)
        expect_body = _quat_apply(offset_q, dirs_ea2)
        ok_cross = bool(torch.allclose(recon_body, expect_body, atol=1e-4))
        results["ea2_channels_cross_check"] = ok_cross
        if not ok_cross:
            failures.append("EA2 selected channels cross-check failed")
    else:
        results["ea2_channels_cross_check"] = "skipped (file not found)"

    results["failures"] = failures
    results["verdict"] = "PASS" if not failures else "FAIL"

    with open(OUT_DIR / "g1_mount_check.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
