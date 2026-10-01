"""G2 验收门：左右对称增强在头装 Airy 挂载下的数值验证（纯 torch）。

复用 pd_gru_lidar 的 ``get_el_4090_lidar_xsym_obs_act``（纯函数、挂载无关），
以新挂载参数（pitch = π/2 + 0.1, 平移 (0.7, 0, -0.05)）核查：

1. 输出形状：obs (2B, 2754)、actions (2B, 18)；
2. proprio 符号翻转（idx 1,3,5,7,10,11）与左右 dof/动作交换逐位正确；
3. 点云镜像等变性：增广半批的近端/远端点多重集合 ≡ 原始点集的 Y 镜像
   （(x, y, z) → (x, -y, z)，重排序不改变集合）；
4. 维度守卫触发 ValueError；auxiliary 路径网格 Y 翻转正确。

运行（纯 torch，任意含 torch 的环境）::

    python tests/pd_gru_lidar_head/symmetry_check.py
"""

import json
import math
import sys
from pathlib import Path

import torch

TESTS_DIR = Path(__file__).resolve().parent        # .../Lidar_mjlab/tests/pd_gru_lidar_head
SRC = TESTS_DIR.parents[1] / "src" / "hexapod_mjlab" # .../Hexapod-MJLab/src/hexapod_mjlab
OUT_DIR = TESTS_DIR / "logs" / "acceptance"     # 验收产物与测试同目录

# ── 按路径加载待测模块 ─────────────────────────────────────────────
# 模块内部的 ``from hexapod_mjlab.utils...`` 依赖经 editable 安装的
# hexapod_mjlab 包直接解析，无需旧版 sys.modules 预注入。
import importlib.util


def _load_module(dotted_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(dotted_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted_name] = mod
    spec.loader.exec_module(mod)
    return mod


sym = _load_module("el_4090_lidar_symmetry",
                   SRC / "rl" / "el_4090_lidar_symmetry.py")

PROPRIO_DIM, PROX, DIST, N_DOF = 66, 256, 640, 18
GX, GY = 17, 11
TOTAL = PROPRIO_DIM + PROX * 3 + DIST * 3
B, DEVICE = 16, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _quat_from_euler_xyz(roll, pitch, yaw):
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    return torch.tensor([cy * sr * cp - sy * cr * sp,
                         cy * cr * sp + sy * sr * cp,
                         sy * cr * cp - cy * sr * sp,
                         cy * cr * cp + sy * sr * sp])


def _row_codes(t: torch.Tensor) -> torch.Tensor:
    """行 → 可排序 int64 编码（量化 1e-5，值域 ±5 m 内安全）。"""
    q = torch.round(t * 1e5).long()
    return (q[:, 0] + 5_000_00) * 10_000_00 ** 2 \
        + (q[:, 1] + 5_000_00) * 10_000_00 \
        + (q[:, 2] + 5_000_00)


def _multiset_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    ca, cb = _row_codes(a), _row_codes(b)
    return bool(torch.equal(torch.sort(ca).values, torch.sort(cb).values))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results, failures = {}, []

    torch.manual_seed(7)
    obs = torch.randn(B, TOTAL, device=DEVICE)
    # 点云段坐标钳到编码值域内：_row_codes 以 1e-5 量化、值域 ±5 m，
    # randn 长尾 >5σ 时编码会跨轴串位（偶发 flaky）——预先钳位。
    obs[:, PROPRIO_DIM:] = obs[:, PROPRIO_DIM:].clamp(-4.9, 4.9)
    # 造结构化 proprio：角度量小一些避免出格（数学性质与量级无关）
    actions = torch.randn(B, N_DOF, device=DEVICE)

    sensor_quat = _quat_from_euler_xyz(0.0, math.pi / 2.0 + 0.1, 0.0) \
        .unsqueeze(0).to(DEVICE)
    sensor_trans = torch.tensor([[0.7, 0.0, -0.05]], device=DEVICE)

    obs_aug, act_aug = sym.get_el_4090_lidar_xsym_obs_act(
        obs=obs, actions=actions, obs_type="policy",
        proprio_dim=PROPRIO_DIM, proximal_points=PROX,
        distal_history_points=DIST, num_dof=N_DOF,
        sensor_quat=sensor_quat, sensor_trans=sensor_trans,
    )

    # ── 1. 形状 ────────────────────────────────────────────────────
    if obs_aug.shape != (2 * B, TOTAL) or act_aug.shape != (2 * B, N_DOF):
        failures.append(f"shape mismatch: {tuple(obs_aug.shape)}, {tuple(act_aug.shape)}")
    results["shapes"] = {"obs": list(obs_aug.shape), "actions": list(act_aug.shape)}

    # ── 2. proprio 符号/交换 ───────────────────────────────────────
    half = obs_aug[B:]
    for i in (1, 3, 5, 7, 10, 11):
        if not torch.allclose(half[:, i], -obs[:, i]):
            failures.append(f"proprio sign flip wrong at idx {i}")
    for start, end in ((12, 30), (30, 48), (48, 66)):
        h = end - start
        if not torch.allclose(half[:, start:start + h // 2], obs[:, start + h // 2:end]):
            failures.append(f"proprio L/R swap wrong at [{start},{end})")
    act_swap = torch.cat([actions[:, N_DOF // 2:], actions[:, :N_DOF // 2]], dim=1)
    if not torch.equal(act_aug[B:], act_swap):
        failures.append("action L/R swap wrong")

    # ── 3. 点云镜像等变（多重集合） ────────────────────────────────
    prox_orig = obs[:, PROPRIO_DIM:PROPRIO_DIM + PROX * 3].reshape(-1, PROX, 3)
    prox_aug = obs_aug[B:, PROPRIO_DIM:PROPRIO_DIM + PROX * 3].reshape(-1, PROX, 3)
    dist_orig = obs[:, PROPRIO_DIM + PROX * 3:].reshape(-1, DIST, 3)
    dist_aug = obs_aug[B:, PROPRIO_DIM + PROX * 3:].reshape(-1, DIST, 3)
    mirrored = prox_orig.clone()
    mirrored[:, :, 1] *= -1
    if not _multiset_equal(mirrored.reshape(-1, 3), prox_aug.reshape(-1, 3)):
        failures.append("proximal mirror multiset mismatch")
    mirrored_d = dist_orig.clone()
    mirrored_d[:, :, 1] *= -1
    if not _multiset_equal(mirrored_d.reshape(-1, 3), dist_aug.reshape(-1, 3)):
        failures.append("distal mirror multiset mismatch")

    # ── 4. 维度守卫 + auxiliary 路径 ───────────────────────────────
    try:
        sym.get_el_4090_lidar_xsym_obs_act(
            obs=torch.randn(B, TOTAL + 1, device=DEVICE), obs_type="policy",
            proprio_dim=PROPRIO_DIM, proximal_points=PROX,
            distal_history_points=DIST)
        failures.append("dim guard did not raise")
    except ValueError:
        pass

    grid = torch.randn(B, GX * GY, device=DEVICE)
    grid_aug, _ = sym.get_el_4090_lidar_xsym_obs_act(
        obs=grid, obs_type="auxiliary",
        height_grid_x_count=GX, height_grid_y_count=GY)
    expect = torch.cat([grid, grid.reshape(-1, GX, GY).flip(dims=[2]).reshape_as(grid)])
    if not torch.equal(grid_aug, expect):
        failures.append("auxiliary grid y-flip wrong")

    results["mirror_multiset"] = "checked exact quantized equality"
    results["failures"] = failures
    results["verdict"] = "PASS" if not failures else "FAIL"

    with open(OUT_DIR / "g2_symmetry_check.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
