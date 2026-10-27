# confined_locomotion_2

`confined_locomotion_2`包含两个独立任务：Flat Baseline平地行走基线，以及通过高度重建辅助监督学习地形记忆的Terrain Memory。二者共享双GRU、Runtime、rollout和PPO实现，通过独立配置、注册身份及Play入口区分。

当前支持真实地形采集、动作前高度标签、时序重放、高度辅助联合更新、完整checkpoint及任务独立可视化。已有小规模运行验证，尚未完成长期学习效果或RTX5090容量验证。

| 用途 | 入口 |
|---|---|
| 安装与新机器配置 | [部署](training/部署.md) |
| 日常训练、续训、Play | [快速开始](../../../src/hexapod_mjlab/tasks/confined_locomotion_2/快速开始.md) |
| 指标与保存规则 | [监控与保存](training/监控与保存.md) |
| CPU/GPU及打包检查 | [测试说明](../../../tests/confined_locomotion_2/README.md) |
| 网格预览、离线选集和性能测量 | [实验工具](../../../scripts/experiments/confined_locomotion_2/README.md) |

## 任务组织与配置

| 任务 | 注册身份 | 配置 | 训练 / Play脚本 |
|---|---|---|---|
| Flat Baseline | `El-4090-Confined-Locomotion-Flat_2` | [`config/flat_baseline.py`](../../../src/hexapod_mjlab/tasks/confined_locomotion_2/config/flat_baseline.py) | `train.py` / `play_flat.py` |
| Terrain Memory | `El-4090-Confined-Locomotion-Terrain-Memory_2` | [`config/terrain_memory.py`](../../../src/hexapod_mjlab/tasks/confined_locomotion_2/config/terrain_memory.py) | `train_terrain_memory.py` / `play_terrain_memory.py` |

脚本统一位于 `scripts/experiments/confined_locomotion_2/`。包内 `__init__.py`集中注册，训练需显式提供seed、环境数和轮数；Terrain额外要求正的高度损失权重。注册默认配置不直接授权训练。

环境、MDP、Runtime、雷达及可视化保留在任务包内；网络与PPO位于 `rl/confined_locomotion_2/`。不复制整套环境或训练实现，不提供旧 `play.py`、`rl_cfg.py`的转发层。原 `confined_locomotion`任务仍独立存在，共享地形查询入口未启用时继续使用原平面。

## 网络与训练

| 模块 | 当前配置与连接 |
|---|---|
| 当前扫描 | 近端256点、远端128点；按光学方向角全局稳定排序，排除无效填充点；不拼接历史扫描 |
| 点GRU | 单层、单向，3→187；每份扫描从零隐藏状态编码 |
| 时间GRU | 单层、单向，196→187；输入为帧特征187维及缩放后的线速度、角速度和重力9维；50 Hz更新 |
| 六维包络 | `[时间状态187, 本体63]`→MLP 250→128→64→6；保留米制映射及初始化 |
| Actor / Critic | 对称输入 `[时间状态187, 本体63, 指令3, detach(包络6)]` 共259维；分别为512→256→128的MLP，输出18 / 1 |
| 高度头 | 仅Terrain Memory：时间状态187维→`Linear(187,187)`→11×17；只用于辅助损失 |

每轮采集24个控制步，5 epochs×4 minibatches，共20次Adam更新。完整rollout优势统一归一化；原始/镜像保留独立GRU历史，episode边界清零，跨rollout保留时间状态。Bootstrap使用隐藏状态副本，不推进在线历史。具体固定PPO数值由 [`TemporalPPOConfig`](../../../src/hexapod_mjlab/rl/confined_locomotion_2/ppo.py)维护。

Flat Baseline无高度头、标签或高度损失，既有网络、观测、动作、奖励和控制语义保留。六维包络暂使用既有固定形态参考；未扩展动态形态Oracle或形态控制。

## 地形与高度辅助监督

当前Terrain Memory使用10×8 m、49×65采样的静态单值Heightfield，包含平坦出生平台和非平地表。地面由 [`make_default_terrain`](../../../src/hexapod_mjlab/tasks/confined_locomotion_2/config/terrain_memory.py)生成，不需要外部地形文件。接触、雷达及标签查询读取同一实际编译地形，没有额外隐藏平面或独立近似标签地形。

网格以当前动作前Root为中心，X方向17点、Y方向11点，物理范围为X±0.75 m、Y±0.65 m；相邻点间距分别为0.09375 m和0.13 m。网格随水平位置及Yaw移动，不随Roll/Pitch倾斜。沿世界竖直方向查询地表，所有187点共用Root的世界Z参考：

```text
H[iy, ix] = terrain_world_z[iy, ix] - root_world_z
```

标签按 `[11,17]` 保存，并绑定当前env/episode/control-step身份，与动作前时间GRU输出对齐。具有有效地形真值的点参与监督，即使当前雷达不可见；地形边界之外mask=False、标签为零。高度标签不涉及足端高度、姿态投影、接触判断或落足可行性。

Masked MSE先在每个控制步样本内对有效格求均值，再对当前minibatch中至少有一个有效格的样本等权平均；原始/镜像分支取平均，镜像标签只翻转Y行。高度损失通过共享编码器联合反向传播，首轮权重候选为1.0，需在启动时显式指定。

**高度预测和地形真值均不输入Actor或Critic。** 二者当前使用相同的在线信息；辅助监督改变训练梯度，不扩充在线策略观测。

运动MDP的地形适配独立于高度监督：已有root高度、足端离地和支撑识别按需查询实际地形，保留原公式、权重和阈值，不新增运动奖励。尚未支持多种地形课程、动态地形、通用复杂几何或动态形态教师。

## 雷达与采样

固定T1 catalog提供96×900 Airy源方向及near/far分组，物理量程15 m、扫描10 Hz。训练可选full或显式固定sparse selection。当前稀疏候选为near×6/far×10，共2816条原始射线，资源随Git及package-data交付：

```text
src/hexapod_mjlab/tasks/confined_locomotion_2/assets/fixed_channels/v1_hit10/near_x6_far_x10.npz
```

`hit_only_10m`仅让P6 gap有效、真实命中且雷达原点原始距离≤10 m的远端射线参与FPS；near规则和物理量程不变。候选不足时保留invalid零槽，不复制点。`all_endpoints`仍是Flat入口默认的旧采样身份，包含无回波量程端点；它与hit10不互用checkpoint。日常命令显式指定hit10，避免依赖默认值。

FPS按米制BASE XYZ进行，首点为最大范数点，并列取pool原顺序，保持源射线身份。更完整的采样设计见[扫描设计](sparse_rays/设计与验收.md)，实验命令见[实验工具](../../../scripts/experiments/confined_locomotion_2/README.md)。

## 执行与显存设置

两个训练入口接受 `--execution-mode low_memory|speed`，默认 `low_memory`。它对点GRU激活做反向重计算，`speed`保留激活；FP32、样本、损失及20次更新顺序相同。当前采集编码块最多128行，点GRU重放块最多1024份独立扫描；1024是面向32 GB RTX5090的首轮候选，实际块大小受本轮扫描数限制，不是PPO minibatch大小。

固定点云打包、观测缩放、目标/镜像标签在rollout内复用；GRU特征及预测仍随每次Adam更新重算。有限性检查在各优化器阶段合批，日志延后读取；逐次KL学习率调度保留。没有引入混合精度或自动OOM降级。性能测量与未采用方案见[实验记录](../../../scripts/experiments/confined_locomotion_2/records/20261026_terrain_memory_and_performance.md)，不将本机容量外推到5090。

## 可视化与模型边界

Flat和Terrain各有独立Play脚本，共享必要显示实现。Terrain显示真实地形、高度预测/真值、当前及候选监督网格、预测/监督六棱柱包络；六棱柱在机器人坐标系内跟随机器人，上下六边形使用相同五维平面参数，上底高度由第六维控制。近端圆锥包络默认关闭。界面使用中文分组控件，候选范围只预览、不写训练配置。

Play只读权重，实际监督范围读取checkpoint。其扫描override仅用于同sampling、同固定catalog内的权重评估；不能作为完整续训或跨任务兼容机制。

完整checkpoint严格比较源码、架构、编译模型、配方、seed、环境数、扫描身份、GPU型号/设备及PyTorch版本。源码改变后应使用对应版本恢复；4050模型不能直接在5090进行严格完整恢复。当前新机器部署从零训练，不改写旧身份。细则见[监控与保存](training/监控与保存.md)。

## 检查与历史记录

当前CPU/GPU检查入口见[测试说明](../../../tests/confined_locomotion_2/README.md)。地形几何、标签对齐、联合更新及性能阶段结论统一放在[实验记录](../../../scripts/experiments/confined_locomotion_2/records/20261026_terrain_memory_and_performance.md)；历史原始日志和 checkpoint 已在 2026-10-27 清理中删除，不能恢复或逐帧复验，复现需重新运行当前检查。

2026-10-24平地tag的交付记录见[旧部署与基线](training/部署与基线.md)，对应[冻结证据](baseline/evidence/README.md)保持原内容。该tag不包含后续Terrain Memory与当前性能优化；新机器使用[当前分支部署](training/部署.md)。
