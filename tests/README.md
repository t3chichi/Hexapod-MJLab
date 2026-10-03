# 功能回归与验收入口

测试按功能维护。`tests/`存放自动回归、独立参考实现和固定输入；数据采样、训练、重放与科学实验工具在`scripts/`或任务自身CLI中维护。测试不会因为作者不同而保留重复入口。

在仓库根目录使用`hp_mjlab`环境：

```bash
conda activate hp_mjlab
python tests/run.py --profile cpu
python tests/run.py --profile gpu
# 列出所选组，不执行
python tests/run.py --profile gpu --list
# 只复查有改动的功能
python tests/run.py --profile cpu --only envelope_geometry envelope_supervision
```

运行器按顺序为每项检查启动独立进程；失败和超时返回非零状态，仍执行同组其余检查。新目录下保存逐项日志、命令、耗时和`summary.json`；已有输出目录拒绝覆盖。默认位置为`logs/verification/<组>-<时间>/`。不得使用`python -O`关闭断言。

## 日常CPU组

| 功能 | 实现 | 主要覆盖 |
|---|---|---|
| 运行器 | `tests/test_runner.py` | 失败/超时传播、无空选择、无重复、输出不可覆盖 |
| 雷达挂载 | `tests/lidar_sensor/mount_check.py` | AIRY通道、挂载旋转、扫描几何 |
| 雷达采样 | `tests/lidar_sensor/pool_sampler_check.py` | 形状、固定种子、索引、轮转覆盖 |
| 雷达镜像 | `tests/lidar_sensor/symmetry_check.py` | 本体状态、动作与点云镜像 |
| 地形 | `tests/terrain/test_terrain_parity.py` | 与冻结高度场逐位一致 |
| 基础PD | `tests/robot/verify_stance.py` | 原stand姿态保持 |
| 网络转换 | `tests/rl_parity/test_parity_forward.py` | 原/新actor、critic、分布与辅助输出数值对齐 |
| 包络几何 | `tests/ea2/test_oracle.py` | 解析墙/通道/转向、完整多边形、因果前视、部分reset、形态有效性 |
| 监督学习 | `tests/ea2/test_supervision.py` | 时序/窗口复位、有效掩码、NaN隔离、五档辅助梯度、多地图损失、指标、数据和checkpoint |
| T0统计与契约 | `tests/t0/test_units.py` | 假成功、早终止、镜像offset、身份篡改、缺门与报告定位反例 |
| 形态测量 | `scripts/tools/morphology/run_checks.py` | 18项完整碰撞体、实际足底/支撑、根平衡、归一化反例 |

CPU组不需要原团队仓库，也不执行任务级PPO训练。监督单元包含极小合成数据上的优化与保存重载，以检查真实梯度和数据生命周期；不能解释为模型性能验收。

## 日常GPU组

| 功能 | 实现 | 固定范围 |
|---|---|---|
| MuJoCo Warp | `tests/robot/warp_check.py` | 8环境×50步，有限状态和CPU对照 |
| 原行走环境 | `tests/env/gate_a.py` | 8环境×40步，注册/观测/奖励/接触/地形 |
| 原雷达环境 | `tests/env/gate_b.py` | 8环境×40步，真实扫描、近远池、时钟/reset/mount |
| 包络环境 | `tests/ea2/test_environment.py` | 平台、传感器/脚本运动、姿态回放、teacher离在线一致、局部reset、采集与回灌 |
| T0配置 | `scripts/tools/t0/contract_gate.py` | 16环境，确定性平地、指令、时间、绝对镜像 |
| T0接触 | `scripts/tools/t0/contact_gate.py` | 8环境，足底/非足/自碰、只读性、NaN与真实overflow |
| T0保持 | `scripts/tools/t0/hold_gate.py` | 32环境×5秒，200Hz状态、支撑、力矩与根受力 |
| T0终止 | `scripts/tools/t0/terminal_gate.py` | 8环境，首步真实终止状态与其余环境20秒连续性 |

GPU组不执行任务训练。EA2也可按功能独立运行，例如`python tests/ea2/test_environment.py --check teacher collection`。显示检查`tests/ea2/visualization_gate.py [--full-lidar]`需要GUI；wheel检查`tests/ea2/packaging_gate.py <wheel路径>`需先构建wheel，两者按相关改动单独运行。

## 可选对照与较长测试

```bash
# 冻结原项目的纯数值/模型/窗口对照
python tests/run.py --profile legacy
# 原脚本运动对照，需要CUDA和原项目
python tests/run.py --profile optional --only envelope_source_motion
# hfield物理诊断，或合成环境PPO学习回归，按相关改动单独选择
python tests/run.py --profile optional --only hfield_contacts
python tests/run.py --profile optional --only ppo_learning
```

原项目默认`~/el4090_legged_gym`，可用`EA2_LEGACY_ROOT`指定另一路径。参考文件先核对`tests/data/ea2/source_manifest.json`；运行时任务和日常回归不依赖该项目。

`hfield_contacts`曾出现GPU HFIELD–BOX抖动，历史现象见[MIGRATION.md](../MIGRATION.md)。失败必须保留并分析，不用重跑PASS覆盖之前FAIL。合成PPO回归属于较长检查，默认CPU/GPU组均不包含它；它也不能替代真实机器人学习行走。

## 科学实验与完整任务验收

- oracle全语料比较、敏感性/heldout分析、报告与冻结四图重放：[`scripts/experiments/ea2/`](../scripts/experiments/ea2/README.md)。这些需要相应语料与独立新输出名称，不属于日常回归。
- EA2采集/训练/评估/导出/play：使用[任务说明](../docs/tasks/envelope_adaptive_2/使用说明.md)中的公共CLI；一次性固定名称发布脚本已删除。
- T0-A完整固定预算验收：`python scripts/tools/t0/run_acceptance.py`。它包含真实5迭代训练和64条评测，范围与结果见[使用与验收](../docs/tasks/el_4090_lidar/T0-A使用与验收.md)。GPU回归通过不等于重新完成这套科学验收。
- T1完整测量、分层库和冻结复验：见[形态测量](../docs/research/morphology/形态可达性测量.md)。

2026-10-04整理前源码/文档保存在`logs/maintenance/stage-cleanup-20261004/source_snapshot/`和Git提交`f9b907a`中。历史证据的源码/模型SHA不改写；源码身份改变后使用该run的冻结实现重放，不绕过身份守卫。

固定输入保留在`tests/data/`及`tests/robot/joint_table.json`。实验结果、权重、编译模型和临时报告继续位于被Git忽略的`logs/`。来源记录属于复现信息；当前测试入口只按功能和设备分类。
