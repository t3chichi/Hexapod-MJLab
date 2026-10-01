# Hexapod-MJLab

EL4090 LiDAR locomotion on **mjlab** (MuJoCo Warp)。本仓库为自
`Lidar_legged_gym`（commit `888e493` + dirty working tree）迁移的 M0 里程碑：
包含 LiDAR 传感器仿真（Warp ray-cast kernel）、采集池采样器、点云几何工具、
左右对称数据增强与 rsl-rl 用的 LiDAR Actor-Critic / Wrapper。

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
