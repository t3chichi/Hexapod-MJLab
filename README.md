# Hexapod-MJLab

EL4090 六足机器人研究项目，基于 **mjlab / MuJoCo Warp**，研究受限空间运动、感知与地形记忆。

## 正式任务

当前保留三个并列任务，共享 `hexapod_mjlab.common` 与必要的 RL 实现：

| 任务 | 说明 | 训练 | Play |
|---|---|---|---|
| `El-4090-Confined-Locomotion-Flat` | 初版受限空间平地任务 | [train.py](scripts/experiments/confined_locomotion/train.py) | [play.py](scripts/experiments/confined_locomotion/play.py) |
| `El-4090-Confined-Locomotion-Flat_2` | 独立时序记忆实现的 Flat Baseline | [train.py](scripts/experiments/confined_locomotion_2/train.py) | [play_flat.py](scripts/experiments/confined_locomotion_2/play_flat.py) |
| `El-4090-Confined-Locomotion-Terrain-Memory_2` | Flat_2 的高度重建辅助地形记忆任务 | [train_terrain_memory.py](scripts/experiments/confined_locomotion_2/train_terrain_memory.py) | [play_terrain_memory.py](scripts/experiments/confined_locomotion_2/play_terrain_memory.py) |

两套任务包分别位于 `src/hexapod_mjlab/tasks/confined_locomotion/` 与 `src/hexapod_mjlab/tasks/confined_locomotion_2/`。EA2、T0-A、旧专用 RL、`joint_passage` 兼容层和旧 TerrainBuilder 已退役，不再属于当前入口。

## 获取代码与安装

当前训练版本发布在 `main` 分支：

```bash
git clone --branch main --single-branch \
  https://github.com/t3chichi/Hexapod-MJLab.git
cd Hexapod-MJLab
conda env create -f environment.yml
conda activate hp_mjlab
python -m pip install -e .
```

需要 Linux、NVIDIA GPU、可用驱动和 conda。依赖版本由 [environment.yml](environment.yml) 维护，包依赖及资源声明见 [pyproject.toml](pyproject.toml)。机器人资产、固定雷达选集、冻结 witness 与默认地形构建代码均随仓库提供；仓库不附训练好的模型。

> 当前工作树的最终整理提交尚未 push。新机器部署前请核对远端实际 commit/tag；不要仅凭本地 `main` 名称推断远端已包含本机未推送改动。

## 仓库结构

```text
src/hexapod_mjlab/
  assets/                 机器人 MJCF/URDF/网格与冻结 witness
  common/                 两套任务共享的常量、reset、雷达、MDP 与 checkpoint 辅助
  utils/                  雷达、点云几何与公共显示组件
  rl/
    confined_locomotion/  初版任务的网络、PPO 与数据接口
    confined_locomotion_2/时序记忆任务的网络、PPO 与数据接口
  tasks/
    confined_locomotion/  初版 Flat 任务
    confined_locomotion_2/Flat_2 与 Terrain Memory
scripts/                  六个正式入口、当前分析工具与实验工具
tests/                    CPU/GPU 自动检查与固定输入
docs/                     当前说明、形态测量证据与平台迁移历史
logs/                     训练与测试输出（Git 忽略）
```

## 文档入口

- [三个任务的训练、续训与 Play](docs/README.md)
- [初版 Flat 快速开始](src/hexapod_mjlab/tasks/confined_locomotion/快速开始.md)
- [Flat_2 / Terrain Memory 快速开始](src/hexapod_mjlab/tasks/confined_locomotion_2/快速开始.md)
- [任务测试与固定资源检查](tests/README.md)
- [平台迁移历史与五份历史数据](docs/development/history/平台迁移.md)
