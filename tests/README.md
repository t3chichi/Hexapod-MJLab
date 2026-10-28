# 功能回归与验收入口

测试按功能和设备维护。`tests/` 保存自动回归、独立参考实现和固定输入；训练、采集与科学实验工具在 `scripts/` 或任务自身 CLI 中维护。测试不会因为作者不同而保留重复入口。

所有命令从仓库根目录运行，并使用 `hp_mjlab` 环境：

```bash
conda activate hp_mjlab
python tests/run.py --profile cpu
python tests/run.py --profile gpu
python tests/run.py --profile confined_cpu
python tests/run.py --profile confined_gpu --list
```

运行器为每项检查启动独立进程；失败和超时返回非零状态，同时继续执行同组其余检查。输出目录保存逐项日志、命令、耗时和 `summary.json`，已有输出目录拒绝覆盖。默认输出在 `logs/verification/<组>-<时间>/`，不得使用 `python -O` 关闭断言。

## 基础 CPU profile

`python tests/run.py --profile cpu` 当前包含 10 项：

| 功能 | 实现 |
|---|---|
| 运行器 | `tests/test_runner.py` |
| 物理模型签名与拒绝反例 | `tests/test_checkpoint_physics.py` |
| Git代码溯源及不可用场景 | `tests/test_provenance.py` |
| 初版任务 partial reset | `tests/confined_locomotion/test_runtime.py` |
| 雷达挂载参考 | `tests/lidar_sensor/mount_check.py` |
| 雷达光学对齐 | `tests/lidar/alignment/test_alignment.py` |
| 雷达 mesh 几何 | `tests/lidar/alignment/test_mesh_relations.py` |
| 雷达采样 | `tests/lidar_sensor/pool_sampler_check.py` |
| 基础 PD 姿态 | `tests/robot/verify_stance.py` |
| 形态测量 | `scripts/tools/morphology/run_checks.py` |

基础 CPU 组不执行任务级 PPO 训练；形态测量使用真实碰撞几何和独立静态复核。雷达光学与 mesh 诊断工具位于 `scripts/tools/lidar/`，测试通过动态加载调用。

## 初版 Flat 任务

```bash
python tests/run.py --profile confined_cpu
python tests/run.py --profile confined_gpu --list
```

- `confined_cpu`：14 个测试模块，共 187 个测试函数；覆盖 runtime、transition、scanner、ground、visualization、policy、采集/更新、checkpoint、扰动、Play、对称和训练入口。
- `confined_gpu`：26 种真实 GPU 模式，包含 runtime、collection、update、entry、checkpoint、perturbations、symmetry 和 lifecycle。
- 真实 GPU 检查使用 2 个环境、固定 seed，并通过 `--output` 写入新目录；不同模式的超时和前置条件不同，先运行 `--list` 查看。

当前 GPU profile 的完整清单由 [tests/run.py](run.py) 维护；单项训练/checkpoint 检查会实际推进仿真并更新临时模型参数，输出不是正式学习效果验收。

## Flat_2 与 Terrain Memory

```bash
python -B tests/confined_locomotion_2/run_checks.py
```

该入口逐个调用 `test_*.py` 的测试函数，不依赖 pytest 或 GPU。当前包含 **167** 项 CPU 检查（其中一个入口执行7个地形边界子用例），覆盖固定通道、采集、时序重放、PPO 更新、高度辅助、Terrain Geometry、checkpoint、Play、可视化和指标。

真实 GPU、checkpoint、Play 与打包检查位于同一目录：

- `check_gpu.py`：Flat_2 的有限训练、保存、恢复和同轨迹优化器重放。
- `check_physics_identity_gpu.py`：使用新生成的Flat hit10检查点，验证质量翻倍/旧物理身份缺失在任何状态应用前拒绝，以及同模型/源码工厂搬移仍能恢复。
- `check_terrain_geometry_gpu.py`、`check_terrain_sequence_gpu.py`、`check_terrain_auxiliary_gpu.py`：真实 Heightfield、标签、时序重放和辅助更新。
- `check_terrain_checkpoint_gpu.py`、`check_terrain_play_gpu.py`：需要匹配当前源码与配方的 Terrain checkpoint。
- `check_sampling_gpu.py`、`check_fixed_sparse_gpu.py`、`check_far_hit_gpu.py`、`check_far_shorttrain_gpu.py`、`benchmark_fixed_training_gpu.py`：采样与性能专项，部分需要匹配 checkpoint 或 selection。
- `check_viewer_gpu.py`：双 viewer 共存验证，使用独立端口。
- `check_package.py`：隔离 wheel 资源与固定 selection 检查。

详细命令、固定配方和新输出目录要求见 [v2 测试说明](confined_locomotion_2/README.md)。GPU 检查并非全部能在无 checkpoint、无 wheel 的环境中直接运行，不要求全部塞进一个默认 profile。

Tripod 独立 CPU 入口为 `python -B tests/tripod2_low/run_checks.py`（34项）；`python -B tests/tripod2_low/check_gpu.py --run` 只做2环境1轮更新，检查DR模型状态、执行器与下一transition恢复，不代表学习收敛。完整恢复的历史文件政策见[保存规则](../docs/tasks/confined_locomotion_2/training/监控与保存.md)。

## 固定输入与历史数据

- 当前固定输入：`tests/data/lidar_alignment_baseline.json`、`tests/robot/joint_table.json`、`tests/robot/joint_table.md`。
- 平台迁移时期的五份历史数据已移至 `docs/development/history/evidence/platform_migration/`，保留原始内容和运行身份，不作为当前测试输入。
- `logs/` 只作为运行时输出根；历史日志已在 2026-10-27 清理中删除，历史结论以 `docs/` 中的记录为准。当前测试从源码、固定资产和新输出目录生成自己的结果。

## 失败处理

失败和超时必须保留并分析。不得通过放宽数值阈值、跳过断言、修改算法或恢复废弃兼容层来掩盖由路径变动、导入变化或文件删除造成的问题。路径调整后应同时核查 `__file__` 根目录定位、`sys.path`、动态加载和测试 profile 中的实际路径。
