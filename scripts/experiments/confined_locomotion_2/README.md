# confined_locomotion_2 实验工具

本页说明可复用的离线设计与可视化工具。训练、续训和正式 Play 的日常命令见[快速开始](../../../src/hexapod_mjlab/tasks/confined_locomotion_2/快速开始.md)；固定雷达通道的设计语义、历史几何结果和负结果见[稀疏射线设计与验收](../../../docs/tasks/confined_locomotion_2/sparse_rays/设计与验收.md)。从项目根目录、`hp_mjlab` 环境运行下列命令。

## 监督网格静态预览

```bash
python -B scripts/experiments/confined_locomotion_2/preview_supervision_grid.py
```

浏览器打开 `http://127.0.0.1:8095`。预览读取 Terrain Memory 当前的 11×17 网格范围，可调候选 X/Y 半范围并查看间距；退出用页面按钮或 `Ctrl+C`，端口可用 `--port` 更改。当前训练候选是 X±0.75 m、Y±0.65 m。预览只显示静态参考姿态和水平面，不加载 checkpoint、不运行仿真/训练，也不写配置。Terrain Play 从 checkpoint 配方读取实际训练网格，不能用此预览的候选值替代。

## 离线固定通道选集

`sparse_rays/select_channels.py` 在固定 T1 catalog 上生成版本化的 near/far source-ray selection。输出必须是不存在的新目录；工具不会覆盖旧结果。

```bash
python -B scripts/experiments/confined_locomotion_2/sparse_rays/select_channels.py \
  --near-multipliers 4 5 6 8 \
  --far-multipliers 4 5 6 8 \
  --output "logs/confined_locomotion_2/channel_candidates_$(date +%Y%m%d_%H%M%S)"
```

这会生成 16 组组合及每组配套 JSON/NPZ。multipliers 是候选池倍率，网络输入仍固定为 near 256、far 128 个槽；候选池规模为 `256×near_multiplier + 128×far_multiplier`。命令只做离线选集，不训练。固定通道采样与 `hit_only_10m` 的差异见上述设计文档；已发布的 `assets/fixed_channels/` 文件是身份资产，不要用新输出覆盖。

## 训练入口

- Flat Baseline：`scripts/experiments/confined_locomotion_2/train.py`
- Terrain Memory：`scripts/experiments/confined_locomotion_2/train_terrain_memory.py`
- Tripod2 Low迁移对照：`scripts/experiments/confined_locomotion_2/train_tripod2_low.py`；独立Legacy配方及Play见[任务说明](../../../src/hexapod_mjlab/tasks/confined_locomotion_2/tripod2_low/README.md)

Flat/Terrain都要求显式 seed、num-envs 和 iterations；Terrain Memory 还要求 `--height-loss-weight`。Tripod的默认预算和运行语义按其独立说明执行，不套用Flat/Terrain性能测量。正式运行的任务身份、resume 参数和只读 Play 入口以[快速开始](../../../src/hexapod_mjlab/tasks/confined_locomotion_2/快速开始.md)为准。性能实验的临时脚本、工作树副本或个人日志目录不是正式入口，也不作为克隆仓库的依赖。

## 容量测量建议

`benchmark_collection.py` 是有限稀疏采集测量入口，不创建优化器或checkpoint。默认使用near×6/far×10 hit10、全部扰动与镜像，先预热再测量；`--output`必须是新目录。例如：

```bash
python -B scripts/experiments/confined_locomotion_2/benchmark_collection.py \
  --task terrain --num-envs 512 --variant optimized --warmups 1 --collections 2 \
  --output "logs/verification/confined_locomotion_2/collection_$(date +%Y%m%d_%H%M%S)"
```

`--task flat`选择Flat_2。`--variant baseline --reference-rev 2c08bab`仅在该实验进程中使用Git历史中的参考采集方法；需要本地具有对应提交。对照保持相同配置并分别使用新进程。`collect_s`只包含采集，不能作为整轮PPO耗时；`--profile`的包含等待/嵌套时间不能直接相加或当作无插桩吞吐。原型与负结果未接入正式训练。

目标机容量尚未实测。测量前先用匹配配置的小规模真实 GPU 工程检查，再固定 task、seed、near×6/far×10 hit10 selection、24 步 rollout、镜像和扰动，逐级增加环境数。分别记录采集、20 次 optimizer 更新、周期保存、`torch.cuda.max_memory_allocated`、PyTorch reserved，以及进程/整卡显存；每个候选使用新进程和新输出目录。不要从 PyTorch 张量显存推算整卡余量，也不要跨环境数或不同源码/selection 的结果直接比较。

当前 RTX 4050 结果与测量边界整理在[2026-10-26 地形记忆与性能记录](records/20261026_terrain_memory_and_performance.md)。这些有限工程运行不构成 5090 容量、长期稳定性或学习效果结论。

已验证的两批采集优化在目录重构后的迁移范围、回归结果与512环境有限采集记录见[迁移验收](records/20261027_optimization_migration.md)。

上述records保留当时的源码身份、恢复政策和性能数据，不表示当前loader仍接受那些历史文件。当前优化保留情况见[任务说明](../../../docs/tasks/confined_locomotion_2/README.md#执行与显存设置)，完整恢复与物理签名政策见[监控与保存](../../../docs/tasks/confined_locomotion_2/training/监控与保存.md)。
