# Hexapod-MJLab

EL4090六足机器人研究项目，运行于 **mjlab / MuJoCo Warp**。当前毕业设计主题为受限空间全身协同通行，EA2五维包络监督与T0-A旧任务作为研究基础。
当前已完成EA2迁移与oracle优化、限定范围形态测量，以及T0-A确定性平地训练管线验收；正式学习行走和六维受限空间联合任务仍待后续验证。
原项目来源与平台迁移记录见[平台迁移历史](docs/development/history/平台迁移.md)，当前成果与边界见[阶段总结](docs/research/阶段总结.md)。

## 安装

需要 NVIDIA GPU 与 conda（本仓库在 RTX 4050 6 GB laptop 上开发验证）。

```bash
git clone https://github.com/t3chichi/Hexapod-MJLab.git
cd Hexapod-MJLab
conda env create -f environment.yml    # 一键复现开发环境（hp_mjlab，锁定版本）
conda run --no-capture-output -n hp_mjlab pip install -e .
```

不使用 environment.yml 时也可手动建环境
（`conda create -n hp_mjlab python=3.12 -y`，宽松版本约束见 pyproject.toml）。
本次预训练基线验证版本：torch 2.11.0+cu130（CUDA runtime 13.0）、warp-lang 1.17.0、mujoco 3.11.0、
mujoco-warp 3.11.0、mjlab 1.6.0、rsl-rl-lib 5.4.2、numpy 2.5.3、
scipy 1.18.1、trimesh 5.1.0。

`confined_locomotion_2` 的当前工程基线、干净检出验证和5090部署顺序见
[部署与基线](docs/tasks/confined_locomotion_2/training/部署与基线.md)。
该基线支持全量/固定稀疏雷达、连续时间记忆、周期保存和学习指标；尚未完成5090容量测试或平地学习效果验收。

## 仓库结构

源码包位于`src/hexapod_mjlab/`，跨任务组件和任务实现分别维护：

```
src/hexapod_mjlab/
  utils/                 复用组件：雷达、点云几何与公共可视化
  rl/                    点云封装、双GRU Actor-Critic、PPO与镜像适配
  terrains/              地形构建与几何视图
  assets/                URDF、网格与MJCF机器人资产
  tasks/
    el_4090_lidar/        机器人雷达训练任务，含T0-A平地管线
    envelope_adaptive_2/  五维包络监督任务
    confined_locomotion/  受限空间研究任务，已注册的平地开发闭环
    confined_locomotion_2/ 合并点云帧编码与连续时间记忆的独立平地任务
scripts/                 工具、基准与离线实验
tests/                  功能检查和验收入口
docs/                   研究方案、任务说明与冻结证据
logs/                   训练、实验和验收产物（Git忽略）
```

测试入口与兼容的本地产物目录见[测试说明](tests/README.md)，文档分类见[docs导航](docs/README.md)。受限空间任务已注册为`El-4090-Confined-Locomotion-Flat`，正典计算包位于`rl/confined_locomotion/`，旧`rl/joint_passage/`保留兼容入口。

验收测试（纯 torch，无需 GPU 仿真器）：

```bash
conda run --no-capture-output -n hp_mjlab python tests/lidar_sensor/mount_check.py
conda run --no-capture-output -n hp_mjlab python tests/lidar_sensor/pool_sampler_check.py
conda run --no-capture-output -n hp_mjlab python tests/lidar_sensor/symmetry_check.py
```

## 五维包络监督学习任务

`tasks/envelope_adaptive_2/` 与 `tasks/el_4090_lidar/` 并列。它使用实际
`ManagerBasedRlEnv` 承载地图、脚本运动和雷达，但由独立监督学习入口训练
190维观测→5维包络的GRU网络。没有机器人关节控制，五维动作不会改变载体路径。

EA2历史验收使用的小规模完整流程（仅适用于EA2，不构成当前训练授权）（两地图各8环境×800步、两配方各5轮）：

```bash
PYTHONPATH=src conda run --no-capture-output -n hp_mjlab python -m \
  hexapod_mjlab.tasks.envelope_adaptive_2.sl.scripts.pipeline --small --play
```

数据、checkpoint和证据分别位于 `logs/ea2/data/`、`logs/ea2/runs/`、
`logs/ea2/acceptance/`。采集、训练、评估和交互显示也可分别调用。
使用命令、对齐范围、验收结果及后续复用边界见
[EA2使用说明](docs/tasks/envelope_adaptive_2/使用说明.md)与[迁移验收](docs/tasks/envelope_adaptive_2/迁移验收.md)。公共世界坐标绘制位于
`utils/visualization/`，独立示例为 `utils/visualization/example.py`。

当前默认教师已升级为完整多边形与因果前视方案；有效性、紧急参考和监督标签分别记录，旧模型自动恢复legacy教师。固定0.10米配方的四图验证、已知尾部代价、任务重构及复现命令见
[oracle选型与重构验收](docs/tasks/envelope_adaptive_2/oracle/选型与验收.md)。
原静态实验与负结果仍见[oracle实验记录](docs/tasks/envelope_adaptive_2/oracle/实验记录.md)。

## 受限空间全身协同任务

`confined_locomotion`已接入任务runtime、collector、训练runner/checkpoint与任务专用play；历史P0–P9和基础门A/B已完成各自GPU工程验收。本轮当前源码拆分GPU最终门仍由根代理补记，不能由历史结果替代。注册默认因缺seed/grid/budget而拒绝执行，正式学习及行走/通行效果尚未开展。当前入口和边界见[任务说明](docs/tasks/confined_locomotion/README.md)、[功能状态](docs/tasks/confined_locomotion/training/功能状态.md)与[阶段总结](docs/research/阶段总结.md)。

日常检查：`python tests/run.py --profile cpu`；有CUDA时用`--profile gpu`。范围和冻结重放见[测试说明](tests/README.md)，分类与按需查证见[文档导航](docs/README.md)。
