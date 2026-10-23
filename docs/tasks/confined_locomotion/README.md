# 受限空间全身协同运动任务

任务ID：`El-4090-Confined-Locomotion-Flat` · 仓库：`/home/t3chichi/Hexapod-MJLab` · Conda环境：`hp_mjlab`。

本页是**任务概览与资料导航**，不再重复终端操作命令。日常运行请直接查看：

**[快速开始：环境 / Play / 训练 / 断点续训 / 测试](../../../src/hexapod_mjlab/tasks/confined_locomotion/快速开始.md)**

## 当前基线与研究状态

- **近端点云**：B8分组（θ=28°、δ=56°、BASE x≤3.5m），近端FPS256；远端每帧64点、保留10帧历史。参数及已知上仰遮蔽风险见[雷达与地面查询边界](lidar_setup/配置与边界.md)。
- **地面辅助监督**：BASE x∈[0.8,3.2]m、y∈[-0.5,0.5]m，17×11=187点；监督值为扫描时机体z轴上的**有符号相对高度**，不是世界绝对高程。四边界定义位于源码`ground_presets.py`，正式训练默认使用该网格；**Play默认恢复所选Checkpoint保存的真实网格**。界面中的候选矩形仅显示，不影响策略与标签。
- **已经验收**：正式任务配置、CPU回归、近端与地面几何支持，以及真实GPU 1环境/1次PPO更新和Checkpoint Play工程链路。原始结果导航：[验证记录索引](../../../logs/verification/README.md)。
- **尚未验收**：正式长期训练效果、平地稳定行走指标、崎岖地形可观测性、狭窄通道/低顶通行成功率。工程更新成功不等于策略已经学会通行。

## 按需求查阅文档

| 你要了解什么 | 位置 |
|---|---|
| **直接复制终端命令：环境、Play、训练、续训** | **[快速开始](../../../src/hexapod_mjlab/tasks/confined_locomotion/快速开始.md)** |
| 当前支持哪些功能、还有哪些边界 | [功能状态](training/功能状态.md) |
| 正式训练、Play、Checkpoint兼容性与约束 | [训练使用说明](training/使用说明.md) |
| GPU/CPU验证通过了什么、如何复现 | [工程验收与复现](training/工程验收与复现.md) |
| 起步平地环境、奖励、重置与PPO数值 | [平地学习与基准测量](平地学习与基准测量.md) |
| 点云编码、标签时刻、网络接口、梯度与损失 | [计算接口与约定](framework/接口与约定.md) |
| 雷达范围、扫描扰动和地面监督网格 | [雷达与地面查询边界](lidar_setup/配置与边界.md) |
| 策略输出、包络和初始化机制 | [输出与模型约定](mapping/输出与模型约定.md) |
| 运动学、动作范围与左右镜像设计依据 | [设计依据](reference/设计依据.md) |
| 下一阶段工作、通行目标及效果验收 | [第一阶段任务说明](第一阶段任务说明.md) |

## 主要代码与结果位置

| 目录或脚本 | 作用 |
|---|---|
| `src/hexapod_mjlab/tasks/confined_locomotion/` | 注册任务、正式配置、扫描器、真实地面标签、runtime、collector、Play |
| `src/hexapod_mjlab/tasks/confined_locomotion/ground_presets.py` | 当前地面监督矩形四边界和自动17×11分解 |
| `src/hexapod_mjlab/tasks/confined_locomotion/visualization/` | 点云、包络和地面高度的交互显示 |
| `src/hexapod_mjlab/rl/confined_locomotion/` | Torch网络、PPO、动作分布与loss |
| `scripts/experiments/confined_locomotion/train.py` | 正式训练和完整续训入口 |
| `scripts/experiments/confined_locomotion/play.py` | 任务专用Checkpoint Play入口 |
| `logs/confined_locomotion/` | 新训练run、完整Checkpoint及Play输出 |
| `logs/verification/` | 带目录导航的历史与当前GPU/CPU验证证据 |

## 历史资料与使用边界

2026-10-12的源码整理、P0–P9和基础门A/B属于各自的历史工程快照；2026-10-13已完成点云特征缓存复用及数值验收。完整出处分别见[工程验收与复现](training/工程验收与复现.md)和[精简历史证据索引](../../../logs/maintenance/20261012_confined_archive_retirement/README.md)。历史完整归档按此前授权清理，**不要**直接将旧源码或旧Checkpoint当作当前任务的无条件可恢复基线。

命令变化时优先核对实际CLI `--help`，随后同步更新源码目录中的《快速开始》；算法约定、功能状态与工程证据分别在上述专题页维护，不在本README堆积运行参数或实验流水账。
