# Hexapod-MJLab

EL4090六足机器人研究项目，基于 **mjlab / MuJoCo Warp**，研究受限空间运动与感知。当前训练入口为 `confined_locomotion_2`：Flat Baseline平地基线，以及使用高度重建辅助监督的Terrain Memory任务。

两个任务共享双GRU与运动策略实现，分别配置、注册和可视化。当前具有真实地形采集、标签对齐、联合更新、完整保存/恢复及训练执行优化；长期行走、地形记忆效果与RTX5090容量仍需验证。

## 获取代码与安装

当前发布在开发分支 `codex/confined-flat-2-pretrain-baseline`：

```bash
git clone --branch codex/confined-flat-2-pretrain-baseline --single-branch \
  https://github.com/t3chichi/Hexapod-MJLab.git
cd Hexapod-MJLab
conda env create -f environment.yml
conda activate hp_mjlab
python -m pip install -e .
```

需要Linux、NVIDIA GPU、可用驱动和conda。依赖版本由 [environment.yml](environment.yml)维护，包依赖及资源声明见 [pyproject.toml](pyproject.toml)。目标机器的安装与GPU确认步骤见[部署说明](docs/tasks/confined_locomotion_2/training/部署.md)。默认 `main` 和旧平地tag不代表上述开发分支。

机器人资产、雷达选集及默认地形构建代码均在仓库内；新机器不需要开发机的日志、checkpoint、缓存或性能工作树。仓库不附训练好的模型。

## 开始使用

| 用途 | 文档 |
|---|---|
| 训练、继续训练、Play | [快速开始](src/hexapod_mjlab/tasks/confined_locomotion_2/快速开始.md) |
| 任务架构、配置及信息边界 | [confined_locomotion_2](docs/tasks/confined_locomotion_2/README.md) |
| 曲线、指标及checkpoint | [监控与保存](docs/tasks/confined_locomotion_2/training/监控与保存.md) |
| 测试与打包检查 | [任务测试](tests/confined_locomotion_2/README.md)、[全项目测试入口](tests/README.md) |
| 地形网格预览、离线选集、性能实验 | [实验工具](scripts/experiments/confined_locomotion_2/README.md) |
| 其他任务与研究资料 | [文档导航](docs/README.md) |

## 仓库结构

```text
src/hexapod_mjlab/
  assets/                 机器人MJCF、URDF及网格
  utils/                  雷达、点云几何与公共显示组件
  rl/                     网络、PPO、镜像与训练数据接口
  terrains/               地形构建与几何视图
  tasks/
    confined_locomotion_2/ Flat Baseline与Terrain Memory
    confined_locomotion/   原受限空间任务
    envelope_adaptive_2/   五维包络监督任务
    el_4090_lidar/         雷达及T0-A基础任务
scripts/                  任务运行入口、离线实验与工具
tests/                    自动检查与验收工具
docs/                     当前说明及研究资料
logs/                     训练与测试输出（Git忽略）
```

原任务继续独立保留：[confined_locomotion](docs/tasks/confined_locomotion/README.md)、[EA2](docs/tasks/envelope_adaptive_2/使用说明.md)、[T0-A](docs/tasks/el_4090_lidar/T0-A使用与验收.md)。项目来源与平台迁移见[平台迁移历史](docs/development/history/平台迁移.md)。
