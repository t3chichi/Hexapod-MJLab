# 文档导航

当前注册四个任务，包含 Tripod2 Low 历史迁移对照。两个任务包共享 `hexapod_mjlab.common` 和必要的 RL 实现，各任务保留自己的配方与运行语义。

| 任务 | 训练入口 | Play 入口 | 快速开始 |
|---|---|---|---|
| `El-4090-Confined-Locomotion-Flat` | `scripts/experiments/confined_locomotion/train.py` | `scripts/experiments/confined_locomotion/play.py` | [v1 Flat](../src/hexapod_mjlab/tasks/confined_locomotion/快速开始.md) |
| `El-4090-Confined-Locomotion-Flat_2` | `scripts/experiments/confined_locomotion_2/train.py` | `scripts/experiments/confined_locomotion_2/play_flat.py` | [v2 Flat / Terrain](../src/hexapod_mjlab/tasks/confined_locomotion_2/快速开始.md) |
| `El-4090-Confined-Locomotion-Terrain-Memory_2` | `scripts/experiments/confined_locomotion_2/train_terrain_memory.py` | `scripts/experiments/confined_locomotion_2/play_terrain_memory.py` | 同上 |
| `El-4090-Confined-Locomotion-Flat-Tripod2-Low_2` | `scripts/experiments/confined_locomotion_2/train_tripod2_low.py` | `scripts/experiments/confined_locomotion_2/play_tripod2_low.py` | [Tripod2 Low](../src/hexapod_mjlab/tasks/confined_locomotion_2/tripod2_low/README.md) |

Flat_2 / Terrain 的 `play_*`默认使用原生窗口；网页检查改用同目录的 `debug_flat.py` / `debug_terrain_memory.py`。远程只开网页可加 `--no-native`，具体参数见v2快速开始。v1与Tripod的查看入口保持各自原有行为。

## 当前阅读路径

| 需要做什么 | 入口 |
|---|---|
| 新机器安装与部署 | [新机器部署](tasks/confined_locomotion_2/training/部署.md) |
| 训练、续训和 Play | [v1 快速开始](../src/hexapod_mjlab/tasks/confined_locomotion/快速开始.md)、[v2 快速开始](../src/hexapod_mjlab/tasks/confined_locomotion_2/快速开始.md) |
| 理解任务配置、架构和信息边界 | [初版 Flat 任务](tasks/confined_locomotion/README.md)、[Flat_2 / Terrain Memory](tasks/confined_locomotion_2/README.md) |
| 查看曲线、指标和 checkpoint | [监控与保存](tasks/confined_locomotion_2/training/监控与保存.md) |
| 运行 CPU/GPU 检查 | [测试说明](../tests/README.md)、[v2 测试说明](../tests/confined_locomotion_2/README.md) |
| 查看形态测量与反例 | [形态可达性测量](research/morphology/形态可达性测量.md)、[实验记录](research/morphology/实验记录.md)、[冻结库](research/morphology/evidence/library.json) |
| 查看平台迁移历史 | [平台迁移](development/history/平台迁移.md)、[五份历史数据](development/history/evidence/platform_migration/) |

## 当前主题说明

- 初版 Flat 数值配方：[平地学习与基准测量](tasks/confined_locomotion/平地学习与基准测量.md)
- 连接语义与架构：[训练架构对齐](research/训练架构对齐.md)
- 雷达范围、扫描扰动与地面监督：[配置与边界](tasks/confined_locomotion/lidar_setup/配置与边界.md)
- 接口、映射与设计依据：[接口与约定](tasks/confined_locomotion/framework/接口与约定.md)、[输出与模型约定](tasks/confined_locomotion/mapping/输出与模型约定.md)、[设计依据](tasks/confined_locomotion/reference/设计依据.md)
- Flat_2 / Terrain Memory 当前验收与部署：[任务说明](tasks/confined_locomotion_2/README.md)、[部署](tasks/confined_locomotion_2/training/部署.md)、[监控与保存](tasks/confined_locomotion_2/training/监控与保存.md)、[稀疏射线设计](tasks/confined_locomotion_2/sparse_rays/设计与验收.md)
- 2026-10-24 平地历史基线（保留原证据）：[部署与基线](tasks/confined_locomotion_2/training/部署与基线.md)、[冻结证据](tasks/confined_locomotion_2/baseline/evidence/README.md)

## 命名与历史边界

当前目录、脚本、测试和正式入口使用功能名称。历史文档中的 `M0–M5`、`B8`、`S1`、`P0–P9` 等代号只表示当时的平台迁移、实验配置或工程批次，不作为当前入口名称；保留这些代号是为了不丢失实验身份与历史结论。冻结通道版本目录（如 `v1_hit10`）、`t1_witness_v1.py.frozen`、任务注册 ID、checkpoint schema 与历史哈希均按原样保留。

文档维护规则见[文档维护](development/文档维护.md)，记录保留与清理原则见[记录保留与清理](development/记录保留与清理.md)。
