# EA2 迁移与验收记录

用户于2026-10-02批准完整规划，以及I1—I3修复、I4保留执行指标并增加同帧指标。主代理实施，独立子代理只读审查。

## 阶段状态

| 阶段 | 状态 | 证据 |
|---|---|---|
| T0 | 独立审查PASS | 42项源哈希核对，`tests/data/ea2/source_manifest.json`，原仓库只读 |
| T1 | 自检与独立复审PASS | `tests/ea2/platform_gate.py`，`logs/ea2/acceptance/t1_gpu.log` |
| T2 | 自检与独立审查PASS | `numerical_gate.py`、`environment_gate.py`，`logs/ea2/acceptance/t2_*.log` |
| T3 | 自检与独立审查PASS | `supervision_gate.py`、`collection_gate.py`，`logs/ea2/acceptance/t3_*.log` |
| T4 | 自检与独立审查PASS | `visualization_gate.py`、`t4_gui.log`、`t4_full_gui.log`及实际渲染PNG |
| T5 | 自检与独立审查PASS | `ea2-migration-small-20261002/pipeline.json`，两配方指标/重载/PNG |
| T6 | 自检与最终独立审查PASS | 六项CPU回归、原任务Gate A/B、EA2最终GPU复检、wheel隔离导入 |

## 功能对齐矩阵

| 功能 | 实施位置 | 验收内容 |
|---|---|---|
| 五维动作与六边形 | config / geometry / mdp.actions | 参数顺序、半宽、后缘符号、软限幅 |
| 地图与路径 | map_generator / path_* / mdp.state | 固定种子、占据栅格、停止转向、重规划 |
| 雷达与姿态 | lidar / airy_mount / attitude_replay | 187/86400通道、10Hz、xyzw转换、噪声 |
| oracle与平滑 | oracle / target_smoother | 可微采样、逐帧缓存、收缩增长、安全snap |
| 数据生命周期 | env / SL适配器 / dataset | 首次reset附带step、部分reset、done隔离、同帧标签 |
| 训练与评估 | sl | 默认与辅助配方、保存重载、I1—I4 |
| 公共可视化 | utils.visualization / EA2 visualization | 无任务依赖、复用示例、native viewer |

历史checkpoint与语料缺失，验收不宣称重现原README性能。具体预算和门槛见已批准的任务规划。

原共享LiDAR实现与EA2继承配置均已补入源哈希清单。新任务复用当前传感器时，以冻结的原LiDAR为行为参考；差异通过EA2适配处理，不修改旧基类或现有任务。

## 审查与适配记录

- T1独立审查发现超时比较符必须为`>=`，已修正并重跑GPU平台门通过。
- mjlab 1.6的`write_mocap_pose`将`(N,3)`写入目标时增加额外轴，实际GPU报形状错误。任务内`write_carrier_pose`使用公开模拟缓冲区并保留单mocap轴，未修改安装包。
- 用户另批准完整点云颜色修正：187选定通道红色，其余命中通道绿色。GPU验收覆盖此分类。
- T2 CPU核对固定输入下几何、轴向/耦合oracle、可微梯度、平滑复位、AIRY模式和通道、地图1/7及路径、原噪声公式；GPU核对原脚本运动、完整/精简测距与未命中、缓存读取、10Hz/空帧、超时、路径失败和姿态回放。
- 用户批准修正辅助安全损失的多地图索引：时间×环境展平时地图ID必须按时间重复整个环境列表，原`repeat_interleave`错配字段。训练与验证共用`frame_map_ids`，混合安全/违规地图的梯度检查通过。
- T3保留原`pred_target_mse`（预测与step后返回目标比较），增加`same_frame_mse`以及`executed_frame_mse`（terminal帧也有明确目标）；修正面积并保留旧面积对照。零动作没有模型预测的字段使用JSON `null`。
- T3 CPU通过旧权重预测/序列逐步、窗口量化及环境划分、episode记忆隔离、五档回忆头梯度、多地图安全与floor掩码、优化和保存重载；GPU通过短时真实数据采集与回灌评估。完整400步序列学习留给T5预算。
- T3复审建议已落实：常量基线来源/均值/哈希写入eval JSON；确定性CPU探针刻意区分before、terminal、post-reset标签并直接核对三种MSE和面积。
- 用户批准PCA方差占比修正：协方差特征值本身即方差，`pc_var`不再重复平方，旧值保留为`legacy_pc_var`。不改变训练和checkpoint选择。
- T4实际native viewer通过暂停、单步、复位、环境切换与恢复；720×960实际渲染图已人工查看，地图/预测/参考/红色选定点可见。公共绘制用独立方形与三点数据验证。
- mjlab native viewer关闭perturbations同时多环境会断言；本任务使用viewer默认交互配置。纯脚本状态和无关节载体不引入动力学控制。
- T4复审发现offscreen renderer的容量小于完整点云数量。公共`DrawingBudget`以公开绘制API限制每帧4096个overlay图元；预留包络/方向，优先选定红点，再均匀抽样绿色点。只减少显示点，完整扫描和监督输入保持不变。实际完整点云截图与交互复检通过，独立审查PASS。
- 辅助记忆和安全损失集中于`sl/losses.py`，不依赖SL runner，便于后续接入联合训练；拆分后监督门复检通过。
- T5配置追溯修正：旧训练入口保存的`config.data`仍带默认96×2200预算。现在从冻结语料生成`actual_collection`，记录每图完整metadata/哈希与实际train/val窗口索引；同规格语料的全局字段投影为实际值。本次两checkpoint仅更新元数据，保留更新前副本，逐项核对所有权重张量完全相同；旧/新文件哈希见`metadata_correction.json`。没有重新采集、训练或重跑完整评估。

## 固定预算结果（2026-10-02）

运行名：`ea2-migration-small-20261002`。证据目录：
`logs/ea2/acceptance/ea2-migration-small-20261002/`。

- 地图1、7各采集一次，均为800步×8环境；400步序列产生400窗口，训练325、验证75。两配方文件哈希与划分索引相同。
- warmup后五维target标准差为`[0.3383, 0.3430, 0.3260, 0.1222, 0.1058]`，至少一维`s≤0.8`的帧占74.17%，通过数据变化/收缩事件门槛。
- 每配方5轮、batch64、lr0.001、梯度裁剪1、seed1。辅助配方为recall k25/50/100/200/300、β0.5、安全λ1。

| 配方 | 首轮→末轮训练MSE | 相对下降 | 最佳验证MSE | 最佳验证R² | 保存epoch |
|---|---:|---:|---:|---:|---:|
| 默认 | 0.289185→0.075247 | 73.98% | 0.084628 | -0.025587 | 4 |
| 辅助+安全 | 0.300900→0.081217 | 73.01% | 0.085938 | -0.041459 | 3 |

两套模型均在独立进程重载后与保存时探针预测完全相同（最大误差0）。各模型
在两地图、各8环境×700步中完成零动作、冻结数据oracle均值常量、stateful和
window四模式；每模式扣除30步后统计670步。默认模型另外完成1环境10步
native play，实际截图已查看。

辅助验证各头MSE为`[0.09727,0.10109,0.10262,0.08910,0.10864]`；安全损失
触发率47.58%，平均违规量0.05540，均有限。非零/有限梯度和不可行最小包络
掩码由T3专门检查验证。

这次证明完整流程和学习梯度在mjlab下可用，尚未证明训练充分收敛。两个验证
R²仍为负，不能据此断言辅助配方提升预测精度。回灌各模式顺次reset，使用
不同路径样本，指标不是同轨迹配对因果比较；window模式填满400帧前输出
0.5，当前700步评估也包含这段启动阶段。

图像：`learning_curves.png`、`rollout_comparison.png`、`trained_play.png`；
完整雷达诊断图在上一级`viewer_full_smoke.png`。图像由真实日志/渲染生成，
已逐张查看。`sl.scripts.report --run-name ea2-migration-small-20261002`
只读取已保存结果，不增加训练或采集。

最终回归：原站姿400步、雷达安装/采样/对称、RL前向转换、地形整数对齐全部
PASS；原任务Gate A为8环境×20步、Gate B为8环境×60步并覆盖reset，均PASS。
加入可视地图后EA2 GPU运动/射线/时钟/复位/姿态检查再次PASS。wheel构建和
临时目录隔离导入PASS，两个任务资产字节一致，无Isaac/legged_gym运行导入。

最终独立只读审查PASS：审查者另行CPU核对数据/模型旧新哈希、全部10/20个
模型权重张量、共同split与环境互斥、实际预算和元数据；核对最终wheel中的
41个EA2/公共绘制源码文件与工作树逐字节一致。原仓库无改动，现有任务、RL
代码和共享雷达实现无diff。完整证据索引为`final_evidence.json`，运行产物按
仓库规则位于被gitignore的`logs/`，指标摘要与命令保存在本文件。

## 使用入口

以下命令在仓库根目录、`hp_mjlab`环境中执行。开发worktree必须使用
`PYTHONPATH=src`，避免已有editable安装指向另一个checkout。日志使用当前目录
下的`logs/ea2/`，也可用明确的绝对路径指定数据或输出。

```bash
# 固定小规模验收：新run-name会重新采集及训练，请勿覆盖已有验收证据
PYTHONPATH=src python -m hexapod_mjlab.tasks.envelope_adaptive_2.sl.scripts.pipeline \
  --small --play --run-name my-ea2-check

# 分步采集（默认精简雷达；姿态回放可用 --attitude 开启）
PYTHONPATH=src python -m hexapod_mjlab.tasks.envelope_adaptive_2.sl.scripts.collect \
  --seeds 1,7 --num-envs 8 --num-steps 800 --out-dir logs/ea2/data/example

# 独立默认监督训练
PYTHONPATH=src python -m hexapod_mjlab.tasks.envelope_adaptive_2.sl.scripts.train \
  --data-dir logs/ea2/data/example --seeds 1,7 --epochs 5 \
  --out-dir logs/ea2/runs/example

# 辅助配方：同一数据，输出到另一个目录
PYTHONPATH=src python -m hexapod_mjlab.tasks.envelope_adaptive_2.sl.scripts.train \
  --data-dir logs/ea2/data/example --seeds 1,7 --epochs 5 \
  --aux-ks 25,50,100,200,300 --aux-mode recall --aux-beta 0.5 --safe-lambda 1 \
  --out-dir logs/ea2/runs/example-aux

# 新进程重载，两个网络模式和零动作/常量基线
PYTHONPATH=src python -m hexapod_mjlab.tasks.envelope_adaptive_2.sl.scripts.eval \
  logs/ea2/runs/example/model.pt --seeds 1,7 --num-envs 8 --num-steps 700 \
  --data-dir logs/ea2/data/example

# native viewer交互；可增加 --full-lidar 查看红绿完整点云
PYTHONPATH=src python -m hexapod_mjlab.tasks.envelope_adaptive_2.sl.scripts.play \
  logs/ea2/runs/example/model.pt --num-envs 1 --mode stateful
```

`pipeline`不加`--small`时使用原SLConfig的大规模预算，不是本次验收命令。
五维输出是归一化形态`s`，再映射为原动作与物理参数；可视化中的青色为预测、
黄色为oracle、红色为选定雷达通道、绿色为其余完整扫描命中点。

## 后续可以复用什么

几何、oracle、平滑器、地图/路径、雷达适配与时钟、显式缓存帧、辅助损失和
公共绘制已在新运行环境下分开验证。后续联合训练可以调用这些实现，减少重复
迁移和重新排查数据时序的工作。`sl/losses.py`只接收网络输出和几何数据，
`utils/visualization/`只接收世界坐标数据。

后续仍需设计统一网络、近远场输入、机器人运动控制、六维高度项及联合优化。
当前载体轨迹与包络输出没有动力学闭环；独立SL训练的成功不能直接证明这些
研究任务已完成。尚无原历史权重或数据，无法比较历史训练性能。
