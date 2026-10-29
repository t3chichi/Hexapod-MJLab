# confined_locomotion_2

`confined_locomotion_2`包含三个注册任务：Flat Baseline平地行走基线、通过高度重建辅助监督学习地形记忆的Terrain Memory，以及作为旧方案迁移对照的Tripod2 Low。三个v2任务共用 `TemporalPPOTrainer` 和 `TemporalPPOConfig`，并保留各自网络/辅助头、Runtime、注册身份和Play入口；Tripod另保留LSTM执行器及独立Runtime/Runner。

当前支持真实地形采集、动作前高度标签、时序重放、高度辅助联合更新、完整checkpoint及任务独立可视化。已有小规模运行验证，尚未完成长期学习效果或RTX5090容量验证。

| 用途 | 入口 |
|---|---|
| 安装与新机器配置 | [部署](training/部署.md) |
| 日常训练、续训、Play | [快速开始](../../../src/hexapod_mjlab/tasks/confined_locomotion_2/快速开始.md) |
| 指标与保存规则 | [监控与保存](training/监控与保存.md) |
| CPU/GPU及打包检查 | [测试说明](../../../tests/confined_locomotion_2/README.md) |
| 网格预览、离线选集和性能测量 | [实验工具](../../../scripts/experiments/confined_locomotion_2/README.md) |

## 任务组织与配置

| 任务 | 注册身份 | 配置 | 训练 / 查看入口 |
|---|---|---|---|
| Flat Baseline | `El-4090-Confined-Locomotion-Flat_2` | [`config/flat_baseline_config.py`](../../../src/hexapod_mjlab/tasks/confined_locomotion_2/config/flat_baseline_config.py) | `train.py` / `play_flat.py`（原生MuJoCo） / `debug_flat.py`（网页调试） |
| Terrain Memory | `El-4090-Confined-Locomotion-Terrain-Memory_2` | [`config/terrain_memory_config.py`](../../../src/hexapod_mjlab/tasks/confined_locomotion_2/config/terrain_memory_config.py) | `train_terrain_memory.py` / `play_terrain_memory.py`（原生MuJoCo） / `debug_terrain_memory.py`（网页调试） |
| Tripod2 Low | `El-4090-Confined-Locomotion-Flat-Tripod2-Low_2` | [`tripod_env_cfg.py`](../../../src/hexapod_mjlab/tasks/confined_locomotion_2/tripod_env_cfg.py) / [`runtime_tripod.py`](../../../src/hexapod_mjlab/tasks/confined_locomotion_2/runtime_tripod.py) / [`config/tripod2_low_config.py`](../../../src/hexapod_mjlab/tasks/confined_locomotion_2/config/tripod2_low_config.py) | `train_tripod2_low.py` / `play_tripod2_low.py`（原生MuJoCo） |

脚本统一位于 `scripts/experiments/confined_locomotion_2/`。三个任务均由包内 `__init__.py`集中注册，训练需显式提供seed、环境数和轮数；Terrain额外要求正的高度损失权重。注册默认配置不直接授权训练。

Flat 环境的唯一实现位于 [`common/flat_env.py`](../../../src/hexapod_mjlab/common/flat_env.py)；任务 `flat_env_cfg.py`只保留Task ID和既有任务接口。Terrain/Tripod复用该工厂并各自覆盖地形、执行器或MDP。`config/`仅放各注册任务的配方配置；`components/`放任务家族共享或按需组合的环境与感知组件，其中地形实现与查询分别位于 [`components/terrain_geometry.py`](../../../src/hexapod_mjlab/tasks/confined_locomotion_2/components/terrain_geometry.py) 和 [`components/terrain_query.py`](../../../src/hexapod_mjlab/tasks/confined_locomotion_2/components/terrain_query.py)，LiDAR扫描、通道、稀疏射线和采样实现位于 [`components/lidar/`](../../../src/hexapod_mjlab/tasks/confined_locomotion_2/components/lidar/)。Tripod的执行器和MDP分别位于 [`components/tripod_actuator.py`](../../../src/hexapod_mjlab/tasks/confined_locomotion_2/components/tripod_actuator.py) 与 [`components/tripod_mdp.py`](../../../src/hexapod_mjlab/tasks/confined_locomotion_2/components/tripod_mdp.py)；其独立Runtime/Runner、checkpoint与入口分别位于任务根目录和 `rl/`。三个v2任务的PPO实现位于 `rl/confined_locomotion_2/`，动作分布位于 [`distribution.py`](../../../src/hexapod_mjlab/rl/confined_locomotion_2/distribution.py)。目录迁移本身不改变网络、奖励、动作、LSTM、DR、课程或Runtime数值。不复制整套环境或训练实现，也不提供旧路径转发层。原 `confined_locomotion`任务仍独立存在，共享地形查询入口未启用时继续使用原平面。

## 网络与训练

| 模块 | 当前配置与连接 |
|---|---|
| 当前扫描 | 近端256点、远端128点；按光学方向角全局稳定排序，排除无效填充点；不拼接历史扫描 |
| 点GRU | 单层、单向，3→187；每份扫描从零隐藏状态编码 |
| 时间GRU | 单层、单向，196→187；输入为帧特征187维及缩放后的线速度、角速度和重力9维；50 Hz更新 |
| 六维包络 | `[时间状态187, 本体63]`→MLP 250→128→64→6；保留米制映射及初始化 |
| Actor / Critic | 对称输入 `[时间状态187, 本体63, 指令3, detach(包络6)]` 共259维；分别为512→256→128的MLP，输出18 / 1 |
| 高度头 | 仅Terrain Memory：时间状态187维→`Linear(187,187)`→11×17；只用于辅助损失 |

每轮采集24个控制步，完整24步BPTT，5 epochs×4 minibatches，共20次Adam更新。三个v2任务共用 [`TemporalPPOTrainer` / `TemporalPPOConfig`](../../../src/hexapod_mjlab/rl/confined_locomotion_2/ppo.py)。直接优化初值为1的策略标准差；KL使用解析计算，不加旧的epsilon稳定项。每个rollout只shuffle一次，并在5个epoch间复用该顺序；KL与熵只在原始样本上计算，镜像样本继续参与策略、价值、包络及均值对称损失。GAE使用每步pre-reset的next_value；termination不bootstrap，truncation bootstrap。镜像损失保留，Terrain高度辅助保留；sparse、`hit_only_10m`、near×6/far×10为新训练默认。原 v1 Flat仍保留自己的 `std=0.5` 配方。新旧优化语义身份由trainer schema和Recipe中必需的`ppo_semantics`标识，缺少身份的旧文件不可完整续训；同sampling条件下仍可weights-only Play，不迁移Adam，也不宣称旧训练与当前配方等价。

Flat v2、Terrain和Tripod使用官方 RSL-RL Logger 负责writer、episode统计与默认终端输出。官方 `Train/mean_reward` 和 `Train/mean_episode_length` 是最近最多100个完成episode的滚动均值；既有 `Episode/mean_return` 与 `Episode/mean_length_steps` 仍统计本轮完成的episode。TensorBoard官方迭代索引从0起，JSONL与checkpoint继续使用累计完成轮数；详见[监控与保存](training/监控与保存.md)。

Flat Baseline无高度头、标签或高度损失，既有网络、观测、动作、奖励和控制语义保留。六维包络暂使用既有固定形态参考；未扩展动态形态Oracle或形态控制。

## 地形与高度辅助监督

当前Terrain Memory使用10×8 m、49×65采样的静态单值Heightfield，包含平坦出生平台和非平地表。地面由 [`make_default_terrain`](../../../src/hexapod_mjlab/tasks/confined_locomotion_2/config/terrain_memory_config.py)生成，不需要外部地形文件。接触、雷达及标签查询读取同一实际编译地形，没有额外隐藏平面或独立近似标签地形。

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

`hit_only_10m`仅让P6 gap有效、真实命中且雷达原点原始距离≤10 m的远端射线参与FPS；near规则和物理量程不变。候选不足时保留invalid零槽，不复制点。Flat v2、Terrain和Tripod新训练默认使用sparse、`hit_only_10m`及上述包内选集。`full`与`all_endpoints`保留为显式选项；PPO身份兼容的 `full/all_endpoints` checkpoint续训时必须传入原扫描配置，不能静默转换为新默认。

FPS按米制BASE XYZ进行，首点为最大范数点，并列取pool原顺序，保持源射线身份。更完整的采样设计见[扫描设计](sparse_rays/设计与验收.md)，实验命令见[实验工具](../../../scripts/experiments/confined_locomotion_2/README.md)。

## 执行与显存设置

三个训练入口接受正整数 `--point-replay-chunk`，默认1024，控制点GRU编码/重计算块大小；RTX 4050可按工程需要设为256或128。该值不改变24步BPTT、PPO minibatches或网络结构，且写入Recipe/Checkpoint执行元数据，不作为完整恢复硬门。`--execution-mode low_memory|speed`默认 `low_memory`，对点GRU激活做反向重计算；`speed`保留激活。FP32、样本、损失及20次更新顺序不变。

固定点云打包、观测缩放、目标/镜像标签在rollout内复用；GRU特征及预测仍随每次Adam更新重算。有限性检查在各优化器阶段合批，日志延后读取；逐次KL学习率调度保留。没有引入混合精度或自动OOM降级。性能测量与未采用方案见[实验记录](../../../scripts/experiments/confined_locomotion_2/records/20261026_terrain_memory_and_performance.md)，不将本机容量外推到5090。

以重构前 `f818178` 为对照，Flat/Terrain 的以下优化仍在正式路径中：

| 优化 | 当前实现与边界 |
|---|---|
| 采集特征复用 | [`collector.py`](../../../src/hexapod_mjlab/tasks/confined_locomotion_2/collector.py) 按扫描/episode身份仅编码变化行，原始与镜像缓存独立，输出有独立存储，更新后清空 |
| 批量扫描与历史写入 | [`components/lidar/scanner.py`](../../../src/hexapod_mjlab/tasks/confined_locomotion_2/components/lidar/scanner.py)、[`components/lidar/sparse_raycast.py`](../../../src/hexapod_mjlab/tasks/confined_locomotion_2/components/lidar/sparse_raycast.py)、`runtime.py`及RL `history.py`保留到期行批量处理、逐行随机流和窗口原子写入；full/sparse仍显式选择 |
| 重放与显存 | [`training_rollout.py`](../../../src/hexapod_mjlab/rl/confined_locomotion_2/training_rollout.py) 保留唯一扫描去重、packed输入、分段时间GRU及参数无关的准备复用；`low_memory`保留点GRU激活重计算 |
| 同步与指标 | [`ppo.py`](../../../src/hexapod_mjlab/rl/confined_locomotion_2/ppo.py) 保留分阶段有限性归约、更新后批量搬运诊断标量；不会跨Adam更新缓存训练特征 |
| 保存与日志 | Flat/Terrain保留累计周期保存、周期/最终文件硬链接复用和已有指标摘要；提取到公共reporting模块未改变这些计算 |

Tripod复用统一PPO trainer、网络重放和Collector相关路径，但保留独立Runtime/Runner；其最终文件仍独立保存，不声称采用Flat/Terrain的硬链接复用。v1继续保留独立的无梯度点云特征缓存。优化分支未采用的RL CUDA Graph、密集分组和fused Adam试验没有因重构进入正式训练实现；MJLab既有的仿真图机制保持原样。以上确认实现保留，不代表重新测得相同吞吐：物理签名增加模型读取与哈希工作，代码溯源在Runner启动时采集，整轮墙钟性能需在固定条件下另测。

## 可视化与模型边界

Flat和Terrain各有原生Play与网页Debug脚本。`play_flat.py` / `play_terrain_memory.py`默认打开暂停状态的MuJoCo passive窗口，Space运行/暂停、F8单步、Enter重置、F12退出；↑/↓调前进、←/→调侧移、Home/End调转向，Delete清零，F9/F10切换前一个/后一个环境。原有 `--command` 和 `--no-viewer --steps` 有限推理入口保留，Play不提供网页端口参数。

`debug_flat.py` / `debug_terrain_memory.py`用于网页调试，默认同时打开原生窗口与事件驱动的Viser网页；`--port`指定网页端口，`--no-native`适用于远程只开网页，`--no-viewer`关闭全部界面。网页保留现有图层、点源、扫描切换、指令和网格预览能力，并增加刷新操作。运行时网页展示最近快照，不持续传输mesh；暂停、单步、刷新、重置和指令预览触发新快照。图层开关、点选和网格预览仅在CPU快照上重绘，不推进物理或GRU。扫描重建时使用独立持久CPU展示模型，身份校验通过后复用展示窗口。Native每个控制步读取Runtime动作后的姿态与速度，不经过完整网页Debug捕获。以上描述功能边界，不代表流畅度已验收或已消除卡顿。

Terrain网页显示真实地形，可开关高度预测/真值、当前及候选监督网格、预测/监督六棱柱包络；六棱柱在机器人坐标系内跟随机器人，上下六边形使用相同五维平面参数，上底高度由第六维控制。近端圆锥包络默认关闭。界面使用中文分组控件，候选范围只预览、不写训练配置。

Play只读权重，实际监督范围读取checkpoint。其扫描override仅用于同sampling、同固定catalog内的权重评估；不能作为完整续训或跨任务兼容机制。

Tripod也使用同一原生查看器和交互循环，保留自己的LegacyRuntime、LSTM执行器、warmup及翻倒暂停规则；`--no-viewer --steps N`保留有限诊断。其指令范围、重置及weights-only边界见[Tripod说明](tripod2_low.md)。

完整checkpoint比较架构、编译模型结构与物理签名、训练配方、seed、环境数、扫描身份和状态张量。缺少本轮新增物理签名的旧文件拒绝完整恢复，符合条件时仍可用于weights-only Play。Recipe与完整文件持久化Git代码来源，不用源码位置、commit或设备元数据门控恢复；跨GPU完整迁移未在本轮验证。细则见[监控与保存](training/监控与保存.md)。

## 检查与历史记录

2026-10-29统一PPO后的有限回归：共享CPU组182项、Tripod专项29项通过；三个任务各以2环境完成1轮/20次Adam更新、保存恢复及下一transition数值比对。旧PPO文件在应用状态前拒绝完整恢复，新旧三个任务的权重各完成2步Play，隔离wheel资源检查通过。恢复边界及同一rollout优化重演逐位一致，后续自然仿真轨迹未逐位一致；未开展新配方长训、学习效果或跨GPU验收。

当前CPU/GPU检查入口见[测试说明](../../../tests/confined_locomotion_2/README.md)。地形几何、标签对齐、联合更新及性能阶段结论统一放在[实验记录](../../../scripts/experiments/confined_locomotion_2/records/20261026_terrain_memory_and_performance.md)；历史原始日志和 checkpoint 已在 2026-10-27 清理中删除，不能恢复或逐帧复验，复现需重新运行当前检查。

2026-10-24平地tag的交付记录见[旧部署与基线](training/部署与基线.md)，对应[冻结证据](baseline/evidence/README.md)保持原内容。该tag不包含后续Terrain Memory与当前性能优化；新机器使用[当前分支部署](training/部署.md)。
