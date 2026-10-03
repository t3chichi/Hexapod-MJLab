# Hexapod-MJLab

EL4090 六足机器人与五维包络监督任务，运行于 **mjlab / MuJoCo Warp**。
当前已完成EA2迁移与oracle优化、限定范围形态测量，以及T0-A确定性平地训练管线验收；正式学习行走和六维受限空间联合任务仍待后续验证。
原项目来源与平台迁移记录见[MIGRATION.md](MIGRATION.md)，当前成果与边界见[阶段总结](docs/research/阶段总结.md)。

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
开发验证版本：torch 2.11.0+cu128、warp-lang 1.17.0、mujoco 3.11.0、
mujoco-warp 3.11.0、mjlab 1.6.0、rsl-rl-lib 5.4.2、numpy 2.5.3、
scipy 1.18.1、trimesh 5.1.0。

## 仓库结构

包内布局：`utils/` 为跨任务复用的组件库（含 LiDAR 传感器仿真组件），
`rl/terrains/` 为域模块，`assets/` 为数据资产，`tasks/` 为任务包：

```
src/hexapod_mjlab/
  utils/            复用组件库（跨任务复用）：
                    lidar/（LiDAR 传感器仿真：LidarSensor、sensor_config/、
                    sensor_kernels/、sensor_pattern/、pool_samplers.py
                    采集池采样——传感器帧→策略消费点池）
                    + pointcloud_geometry.py（四元数/球坐标/角键排序）
  rl/               RL 域（≙ mjlab rl/）：LidarWrapper、双 GRU Actor-Critic、
                    LidarPPO5x、symmetry_adapter.py、
                    el_4090_lidar_symmetry.py（对标官方 rl/spatial_softmax.py
                    的域内计算模块）
  terrains/         numpy-only 地形系统（≙ mjlab terrains/）
  assets/           机器人资产包（el_4090 的 urdf/meshes/mjcf，随包分发；
                    ≡ mjlab asset_zoo/ 的包内资产做法）
  tasks/el_4090_lidar/  任务包（≙ mjlab tasks/velocity/：mdp/ + config/ + rl/），
                    经 pyproject [project.entry-points."mjlab.tasks"] 注册
scripts/            工具与实验 harness：tools/（URDF→MJCF 转换、golden
                    生成、m5 对比）、bench/（吞吐/摩擦基准 + 新旧基线
                    启动器）、spike/（mujoco-warp 集成 spike）
tests/              验收门（域名分组 robot/ terrain/ env/ rl_parity/
                    lidar_sensor/，见 tests/README.md）
logs/               (gitignored) 运行产物唯一根：logs/rsl_rl/ = mjlab 训练
                    输出默认位置；logs/m5/ = m5 harness 元数据；测试本地产物
                    在 tests/<组>/logs/
```

验收测试（纯 torch，无需 GPU 仿真器）：

```bash
conda run --no-capture-output -n hp_mjlab python tests/lidar_sensor/mount_check.py
conda run --no-capture-output -n hp_mjlab python tests/lidar_sensor/pool_sampler_check.py
conda run --no-capture-output -n hp_mjlab python tests/lidar_sensor/symmetry_check.py
```

迁移背景、旧→新文件映射与迁移决策见 [MIGRATION.md](MIGRATION.md)。

## 五维包络监督学习任务

`tasks/envelope_adaptive_2/` 与 `tasks/el_4090_lidar/` 并列。它使用实际
`ManagerBasedRlEnv` 承载地图、脚本运动和雷达，但由独立监督学习入口训练
190维观测→5维包络的GRU网络。没有机器人关节控制，五维动作不会改变载体路径。

在仓库根目录运行已批准的小规模完整流程（两地图各8环境×800步、两配方各5轮）：

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
后续六维形态/真实机器人联合任务准备见[联合任务细化方案](docs/research/联合任务方案.md)。

形态测量已通过限定范围验收：35个静态保持构型、7个指定慢速运动构型，见
[形态测量与质量审查](docs/research/morphology/形态可达性测量.md)。T0-A确定性平地管线已重构，当前状态见[T0-A使用与验收](docs/tasks/el_4090_lidar/T0-A使用与验收.md)，
验收协议、操作和修复记录已统一到上述文档；正式学习行走仍待后续验收。

日常回归从`python tests/run.py --profile cpu`开始；有CUDA时运行`--profile gpu`。两组均不启动任务训练，旧源码对照和长测试另行选择。入口、覆盖范围与历史重放见[测试说明](tests/README.md)。

文档分类与维护规则见[docs导航](docs/README.md)。
