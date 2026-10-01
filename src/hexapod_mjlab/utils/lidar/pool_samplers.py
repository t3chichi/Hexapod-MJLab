"""采集池采样器：从 Airy 原始帧中选出策略消费的点池（pd_gru_lidar_head）。

双缓冲架构（见包 README）中，传感器层每 10 Hz 产出全量原始帧
（R = 900 方位 × 96 俯仰），采集池层由本模块按可插拔策略降采样出
每控制步供 LidarWrapper/网络消费的小池（P ≈ 数千点）。

接口契约（所有策略必须满足）：

* ``sample(points_raw, dists_raw, frame_count) -> LongTensor (E, P)``——
  返回**索引**而非点值，调用方据此 gather 点与距离，审计/可视化可反查
  哪些原始射线进了池子；
* ``fixed_output_size == P`` 固定：``lidar_points_base`` 为预分配缓冲；
* 纯 torch、批量、热路径无 host 同步（禁止 ``.item()`` / CPU 循环）；
* 以 ``(seed, frame_count)`` 为确定性来源：同种子同帧号 → 逐位一致，
  支持可复现实验与跨 run 对照；
* ``stats()`` 仅返回静态元数据（策略名/尺寸等），保持热路径零开销；
  池构成统计由消费方基于返回索引自行计算。

新增策略：实现 ``PoolSampler`` 子类并 ``@register_pool_sampler("名字")``，
然后在 config ``pool.strategy`` 中引用即可，环境代码无需改动。

本模块仅依赖 torch，可独立测试
（``tests/pd_gru_lidar_head/pool_sampler_check.py``，
验收门 G-S）。
"""

from __future__ import annotations

import math

import torch

REGISTRY: dict[str, type] = {}


def register_pool_sampler(name: str):
    """策略注册装饰器。"""

    def _deco(cls):
        if name in REGISTRY:
            raise KeyError(f"pool sampler '{name}' already registered")
        REGISTRY[name] = cls
        return cls

    return _deco


class PoolSampler:
    """采样器基类：只定义契约，不实现采样。"""

    #: 采样输出尺寸 P（子类必须设置为正整数）
    fixed_output_size: int = 0

    def __init__(self, n_azimuth: int, n_elevation: int, raw_num_rays: int,
                 seed: int, device: torch.device):
        self.n_azimuth = int(n_azimuth)
        self.n_elevation = int(n_elevation)
        self.raw_num_rays = int(raw_num_rays)
        self.seed = int(seed)
        self.device = torch.device(device)
        if self.fixed_output_size <= 0:
            raise ValueError(f"{type(self).__name__}.fixed_output_size must be positive")
        if self.fixed_output_size > self.raw_num_rays:
            raise ValueError(
                f"{type(self).__name__}: output {self.fixed_output_size} > raw {self.raw_num_rays}")

    def sample(self, points_raw: torch.Tensor, dists_raw: torch.Tensor,
               frame_count: torch.Tensor) -> torch.Tensor:
        """返回 (E, P) 射线索引。子类必须实现。"""
        raise NotImplementedError

    def stats(self) -> dict:
        """静态元数据（热路径外使用；池构成统计由审计脚本基于索引计算）。"""
        return {
            "strategy": REGISTRY_TO_NAME[type(self)],
            "output_size": int(self.fixed_output_size),
            "raw_num_rays": int(self.raw_num_rays),
            "n_azimuth": self.n_azimuth,
            "n_elevation": self.n_elevation,
            "seed": self.seed,
        }


# 反查注册名（stats 用；在类定义完成后填充）
REGISTRY_TO_NAME: dict[type, str] = {}


def _finalize_registry():
    for name, cls in REGISTRY.items():
        REGISTRY_TO_NAME[cls] = name


# ---------------------------------------------------------------------------
# 策略实现
# ---------------------------------------------------------------------------


@register_pool_sampler("angle_stride")
class AngleStrideSampler(PoolSampler):
    """方位维等间隔采样（默认基线）。

    Airy 展平索引约定 ``idx = az * n_elevation + el``（C 序）。保留全部
    96 条俯仰激光通道（物理线），每 ``azimuth_stride`` 个方位取 1：
    P = (n_azimuth / azimuth_stride) × n_elevation。
    空间均匀、逐帧确定、与帧号无关。
    """

    def __init__(self, n_azimuth: int, n_elevation: int, raw_num_rays: int,
                 seed: int, device: torch.device, azimuth_stride: int):
        if n_azimuth % azimuth_stride != 0:
            raise ValueError(
                f"n_azimuth {n_azimuth} not divisible by azimuth_stride {azimuth_stride}")
        self.azimuth_stride = int(azimuth_stride)
        device = torch.device(device)
        az_idx = torch.arange(0, n_azimuth, self.azimuth_stride, device=device)
        el_idx = torch.arange(n_elevation, device=device)
        self._indices = (az_idx[:, None] * n_elevation + el_idx[None, :]).reshape(-1)
        # 固定输出尺寸必须在基类构造校验之前就绪
        self.fixed_output_size = int(self._indices.numel())
        super().__init__(n_azimuth, n_elevation, raw_num_rays, seed, device)

    def sample(self, points_raw, dists_raw, frame_count):
        envs = points_raw.shape[0]
        return self._indices.unsqueeze(0).expand(envs, -1)

    def stats(self):
        info = super().stats()
        info.update({"azimuth_stride": self.azimuth_stride})
        return info


@register_pool_sampler("stride_rotating")
class StrideRotatingSampler(AngleStrideSampler):
    """方位维等间隔 + 按帧轮转偏移。

    方位偏移 = ``frame_count % azimuth_stride``（逐环境独立），连续
    ``azimuth_stride`` 帧的索引并集覆盖全部方位——时域互补覆盖，用于
    研究"跨帧互补采样能否降低对 distal 记忆的依赖"。
    """

    def sample(self, points_raw, dists_raw, frame_count):
        envs = points_raw.shape[0]
        offset = frame_count.to(self._indices.device) % self.azimuth_stride  # (E,)
        base_az = self._az_base.unsqueeze(0) + offset.unsqueeze(1)           # (E, A)
        base_az = base_az % self.n_azimuth
        idx = base_az[:, :, None] * self.n_elevation + self._el[None, None, :]
        return idx.reshape(envs, -1)

    def __init__(self, n_azimuth, n_elevation, raw_num_rays, seed, device,
                 azimuth_stride):
        super().__init__(n_azimuth, n_elevation, raw_num_rays, seed, device,
                         azimuth_stride)
        n_az_sel = self._indices.numel() // self.n_elevation
        self._az_base = (torch.arange(n_az_sel, device=self.device)
                         * self.azimuth_stride)
        self._el = torch.arange(self.n_elevation, device=self.device)


@register_pool_sampler("random_uniform")
class RandomUniformSampler(PoolSampler):
    """逐帧无放回均匀随机（对照：消除等角采样的角度混叠）。

    用计数器型哈希（sin-hash）生成 (E, R) 分数后 topk——纯函数于
    ``(seed, env_id, frame_count)``，GPU 向量化、无生成器状态、可复现。
    """

    def __init__(self, n_azimuth: int, n_elevation: int, raw_num_rays: int,
                 seed: int, device: torch.device, output_size: int):
        if not 0 < output_size <= raw_num_rays:
            raise ValueError(f"output_size {output_size} out of (0, {raw_num_rays}]")
        self.fixed_output_size = int(output_size)
        super().__init__(n_azimuth, n_elevation, raw_num_rays, seed, device)

    def sample(self, points_raw, dists_raw, frame_count):
        envs, raw = points_raw.shape[0], points_raw.shape[1]
        device = points_raw.device
        env_ids = torch.arange(envs, device=device, dtype=torch.float32)
        ray_ids = torch.arange(raw, device=device, dtype=torch.float32)
        x = torch.sin(
            self.seed
            + env_ids[:, None] * 1.6180339887
            + frame_count[:, None].to(device).float() * 2.7182818284
            + ray_ids[None, :] * 1.2020569032
        ) * 43758.5453123
        scores = x - torch.floor(x)
        _, idx = torch.topk(scores, self.fixed_output_size, dim=1)
        return idx

    def stats(self):
        info = super().stats()
        info.update({"hash": "sin-counter"})
        return info


@register_pool_sampler("identity")
class IdentitySampler(PoolSampler):
    """不降采样（P = R）。仅用于 G5 吞吐基准与全量池实验。

    ⚠ 0.4° 全帧（86,400 点）下每控制步的 FPS/argsort 开销不可行，
    本策略仅供基准测试，禁止用于正式训练。
    """

    def __init__(self, n_azimuth: int, n_elevation: int, raw_num_rays: int,
                 seed: int, device: torch.device):
        device = torch.device(device)
        self.fixed_output_size = int(raw_num_rays)
        self._indices = torch.arange(raw_num_rays, device=device)
        super().__init__(n_azimuth, n_elevation, raw_num_rays, seed, device)

    def sample(self, points_raw, dists_raw, frame_count):
        envs = points_raw.shape[0]
        return self._indices.unsqueeze(0).expand(envs, -1)


_finalize_registry()


def build_pool_sampler(pool_cfg, n_azimuth: int, n_elevation: int,
                       raw_num_rays: int, device) -> PoolSampler:
    """config 字符串 → 采样器实例。``pool_cfg`` 需提供 strategy/seed，
    以及策略各自的参数（azimuth_stride / output_size）。"""
    strategy = str(getattr(pool_cfg, "strategy", "angle_stride"))
    seed = int(getattr(pool_cfg, "seed", 42))
    if strategy == "angle_stride":
        return AngleStrideSampler(n_azimuth, n_elevation, raw_num_rays, seed,
                                  device,
                                  azimuth_stride=int(getattr(pool_cfg, "azimuth_stride", 30)))
    if strategy == "stride_rotating":
        return StrideRotatingSampler(n_azimuth, n_elevation, raw_num_rays, seed,
                                     device,
                                     azimuth_stride=int(getattr(pool_cfg, "azimuth_stride", 30)))
    if strategy == "random_uniform":
        return RandomUniformSampler(n_azimuth, n_elevation, raw_num_rays, seed,
                                    device,
                                    output_size=int(getattr(pool_cfg, "output_size",
                                                            raw_num_rays // 30)))
    if strategy == "identity":
        return IdentitySampler(n_azimuth, n_elevation, raw_num_rays, seed, device)
    raise KeyError(f"unknown pool sampler '{strategy}'; known: {sorted(REGISTRY)}")
