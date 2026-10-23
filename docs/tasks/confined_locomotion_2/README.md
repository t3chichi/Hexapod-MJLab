# confined_locomotion_2

更新：2026-10-24。独立注册名为 `El-4090-Confined-Locomotion-Flat_2`。可部署的预训练版本与当前干净检出证据见[部署与基线](training/部署与基线.md)。
原 `confined_locomotion` 的任务与 RL 源文件保持不变；公共任务发现入口只追加新任务导入。
本页是新任务的架构、配置和验收入口。日常复制执行的训练、续训、Play和检查指令见[快速开始](../../../src/hexapod_mjlab/tasks/confined_locomotion_2/快速开始.md)。机器人、T1、PD、奖励、指令、六维物理输出和观测扰动的共用依据见[原任务配置](../confined_locomotion/平地学习与基准测量.md)。固定稀疏射线的离线设计、历史实验与当前接入边界见[设计与验收](sparse_rays/设计与验收.md)。

## 已确认架构

| 部分 | 配置 |
|---|---|
| 当前扫描 | 近端 XYZ FPS 256 点；远端 XYZ FPS 128 点；不拼接历史扫描 |
| 分组与原始雷达 | 固定 T1 catalog 提供 96×900 Airy 源方向及固定 near/far 分组；raycast 默认 full（86,400条），也可显式指定经过校验的固定 sparse NPZ；15 m、10 Hz，硬件与安装不变 |
| 帧内编码 | 合并近远端并按光学方向角键全局稳定排序；掩码排除填充点；单层单向点 GRU，3→187，每份扫描 h0=0 |
| 时间记忆 | `[帧特征187, 缩放后的v/ω/g共9]` → 单层单向 GRU，196→187；50 Hz；跨 rollout 保留隐藏状态，episode 重置清零 |
| 输入时间信息 | 无扫描年龄、全局时间戳或新扫描标志通道 |
| 六维预测 | `[时间状态187, 当前完整本体63]` → MLP 250→128→64→6；ELU；统一六维头；保留已有米制映射和初始化 |
| 动作、价值 | 分别由 `[时间状态187, 本体63, 指令3, detach(六维包络)]` 共259维输入 MLP 512→256→128，输出18/1 |
| 地面监督 | 无地面头、地面标签生成、网格配置或地面损失 |
| 辅助接口 | `rl/confined_locomotion_2/auxiliary.py`，默认空实现；未实现或启用任何足端预测监督 |

近端固定从P6 gap有效的near候选池中作BASE米制XYZ FPS 256。远端候选可显式选择 `far_sampling=all_endpoints`（默认legacy语义，no-hit 15m端点仍有效）或 `hit_only_10m`（仅P6 gap有效、原始Raycast真实命中且雷达原点原始距离≤10m的射线参与FPS）。该门槛不改变物理15m量程、原始无回波定义、近端、P6扰动或固定catalog分组。FPS从最大范数点起步、距离并列时取pool中较早源索引，选中索引显式排除；不足128点时valid=False、xyz=0、source id=-1，不复制或合成点。最终near/far网络输入仍是256/128个定长槽。

CUDA后端为任务独立的`sampling.py`/`_sampling_warp.py`：先按原顺序对到期环境raycast、分组和应用扰动，再将这些frame的近端、远端分别批量FPS；GPU稳定压紧有效候选、执行选点迭代并反映射原pool槽位，使用当前PyTorch流。原始射线ID、方向角键与采样点直接gather关联；选完后仍由原输入流程排序。CPU保留确定性参考实现。非到期行不扫描/不采样，扫描10 Hz、控制50 Hz与重置边界不变。

采集、推理时仅缓存无状态帧特征，旧扫描之间继续用当前本体信息更新时间状态。
训练时从原始点云重新计算帧特征，每个 fresh replay graph 内可复用同一扫描的特征，保留对帧编码器的梯度；不使用采集期 no-grad 特征训练。

## 时序训练与镜像

rollout=24、5 epochs、4 minibatches、一个 Adam，保留原 PPO 数值起步配方及单一18维可学习 log_std。FP32、关闭 TF32 和 AMP，cuDNN RNN 使用 IEEE FP32。

每个 minibatch 按顺序完整重放24步，只对本批互斥选中的行计算直接损失，因此小环境数下仍能维持4批与连续时间梯度。片段开始的隐藏状态 detach，片段内按真实 episode 边界清零；不因 rollout 边界清零在线记忆。完整重放以计算开销换取清晰的首版正确性，尚未做大规模性能调优。

镜像增强保留：观测、点云、指令、动作和标签执行已有镜像映射，点云按镜像后的光学方向重新排序；镜像历史独立维护隐藏状态，不对187维状态做逐维镜像。增强样本的旧行为分布使用真实行为高斯的镜像 pushforward，不假定网络已等变。均值一致性系数0.6，正式配方保留 ALL_THREE 观测扰动。

bootstrap 从自然 pre-reset final 输入与动作后的隐藏状态计算 value，结果作为纯分支，不提交其新隐藏状态。真终止不 bootstrap，超时使用自然 final value，二者都切断跨集 GAE。

## 运行

在项目根目录、`hp_mjlab` 环境运行，具体指令见[快速开始](../../../src/hexapod_mjlab/tasks/confined_locomotion_2/快速开始.md)。注册默认 max_iterations=0，不能直接启动学习；显式工厂不再需要地面网格参数。工程验证最多2个环境、1次迭代；正式长训练的预算尚未确定或启动。

完整 checkpoint 恢复始终要求源码/配方/模型/设备、扫描身份和远端采样身份严格匹配。扫描身份包含固定T1 catalog、full/sparse模式、（sparse时）具体selection哈希及非legacy远端采样身份。full与sparse、不同selection之间不能完整续训。Play权重必须与 `far_sampling` 完全一致；`--allow-scan-override` 只允许在同一采样身份、同一 `fixed_t1_channels/v1` grouping和完全相同固定T1 catalog下改变full/sparse mode或selection，不能跨 `all_endpoints` 与 `hit_only_10m`。旧动态分组checkpoint因架构身份不同而拒绝。保存包括物理与manager状态、命令和扰动随机流、输入缓存、两个历史状态、Adam与shuffle随机流；稳定的24步采集及20次更新之后才能保存。默认每累计50轮保存完整checkpoint，正常结束另保留final文件；学习指标写入TensorBoard与JSONL。保存语义、指标单位、当前GPU验证与5090启动条件见[监控与保存](training/监控与保存.md)。

扫描配置默认 `raycast_mode=full`、`far_sampling=all_endpoints`。near/far固定分组仍来自同一个T1 catalog；sparse只通过显式、校验过的selection NPZ（或同名校验JSON）启用，不支持运行时生成或按负载改变射线。候选池预算为 `rawN = 256 × near_multiplier + 128 × far_multiplier`，网络仍接收near 256槽、far 128槽。既有 `v1/`、`v1_independent/` 选集保持原样；新增 `assets/fixed_channels/v1_hit10/near_x6_far_x10.npz` 是当前已选工程基线的固定源方向资产（2,816条候选）。Play默认也发现 `v1_hit10/`。用户已选择 `hit_only_10m` + near×6/far×10 作为当前工程验证与短训基线；尚未决定长训规模或宣称学习效果。

Play默认接入独立的本地浏览器界面（8094，旧任务默认8093），初始暂停；`--no-viewer`保留有限推理。`PlaySession`只读权重并从新episode、零时间状态开始，不建立optimizer或动作采样器。所选 `far_sampling` 必须与checkpoint匹配，GUI不提供跨sampling切换。浏览器提供 `Raycast mode`、独立的 `Near candidate multiplier` 与 `Far candidate multiplier` 下拉，以及 `Apply acquisition + reset`；默认资产发现 `v1/`、`v1_independent/`、`v1_hit10/` 中可用组合，包含near×6/far×10。两倍率互不联动；只有对应的离线文件存在时才能提交，缺失组合会提示并禁用按钮。same-sampling模型可显式评估同catalog下不同full/sparse或selection，跨sampling身份始终拒绝。状态栏显示far_sampling和当前far有效点数。Apply后新episode、命令和GRU状态归零并保持暂停，浏览器server继续运行。GUI适配near 256槽、far 128槽，无地面预测/标签；点源查询通过原始射线ID关联。物理平面只是场景参考。

显示层使用独立CPU快照；输入在动作前保留，姿态在动作后读取并分别标记身份。切换显示和查点不调用网络或推进runtime；初始、改指令、重置的显式preview不提交时间隐藏状态。WORLD显示仅是采集时BASE点到当前姿态的投影，没有历史运动补偿或建图。GUI回调只写意图，由主循环统一执行控制、推理与重置；不得用通用play循环绕过扫描/记忆时钟。

采样身份写入checkpoint架构、训练配方、runtime配方及runtime状态。`all_endpoints`保留既有legacy identity；`hit_only_10m`使用独立版本身份并记录原始命中、原始雷达距离门槛、gap有效性和空槽语义。两种采样身份不能互相加载Play权重或续训。此前远端角采样或其他legacy checkpoint不能用作hit10模型示例；hit10和legacy都支持从零训练。原`confined_locomotion`源码与校验逻辑不受影响。

## 验证与边界

新增周期保存与学习指标后的当前CPU检查98项、真实GPU恢复/保存回归通过，见[监控与保存](training/监控与保存.md)。以下far-hit验收为当时冻结源码的结果；其中旧checkpoint仍可用于匹配sampling的Play，但不再匹配当前源码的严格续训。

raw BASE审计修正后的86项CPU检查通过，命令为 `tests/confined_locomotion_2/run_checks.py`。`hit_only_10m` near×6/far×10的真实pose parity通过：full 86,400与sparse 2,816在实际roll/pitch/height下，同通道、同候选池FPS及P6一致，见[actual-pose parity](../../../logs/verification/confined_locomotion_2/20261023_far_hit10_parity_actual_pose/checks.json)。此前[parity报告](../../../logs/verification/confined_locomotion_2/20261023_far_hit10_parity/checks.json)的同通道/P6/同候选池FPS结果仍有效，但其姿态用例未实际改变底层root pose，不能作为姿态变化证据。

最终配对短训通过：相同种子、相同near×6/far×10候选，每组2环境、10迭代，480次环境转移和200次优化更新；损失/梯度有限、无跌倒或超时。hit10最少51个有效远端点。耗时和PyTorch显存峰值接近，未证明学习收益。完整数值、测量边界及先导记录统一见[扫描设计与验收](sparse_rays/设计与验收.md)。

227帧真实Warp扫描、少量远端输入、P6、缓存/reset及GRU检查通过；下楼最少31个有效远端点，顶部与地面持续保留。这是合成人为姿态、固定本体输入与随机权重的点云诊断，不能证明真实楼梯运动或训练后策略稳定性。几何、特征和恢复结果见[扫描设计与验收](sparse_rays/设计与验收.md)。

最终Play CUA验收通过：[browser checks](../../../logs/verification/confined_locomotion_2/20261023_far_hit10_play_browser/browser_checks.json)、[play summary](../../../logs/verification/confined_locomotion_2/20261023_far_hit10_play_browser/play_summary.json)和[截图](../../../logs/verification/confined_locomotion_2/20261023_far_hit10_play_browser/hit10_controls.png)。实际使用final hit10 checkpoint，在sparse near×6/far×10 single step，再Apply full、回到sparse、Exit；weights SHA保持一致，sampling始终为 `hit_only_10m`，cross_configuration依次false/true/false，Apply后均为fresh episode、zero GRU、paused，状态栏显示命中规则和far valid 128/128。

汇总验收记录[acceptance.json](../../../logs/verification/confined_locomotion_2/20261023_far_hit10_acceptance/acceptance.json)为passed，汇总CPU 86项、Warp姿态stage2、actual-pose parity、严格恢复、当时源码的配对短训及Play浏览器。该记录不代表长期训练或学习效果通过；旧任务43个PY文件及 `v1/`、`v1_independent/` 共42个选集文件SHA保持不变。 GPU入口见[快速开始](../../../src/hexapod_mjlab/tasks/confined_locomotion_2/快速开始.md)。

独立倍率扩展已完成GPU往返验证：full→4/4→4/8→8/8→8/4→full→4/4，实际射线数和通道来源检查通过，同模式同步数显存allocated增量为0；见[GPU检查](../../../logs/verification/confined_locomotion_2/20261023_independent_multipliers_gpu/checks.json)。新增4/8组合的全量/稀疏同通道求交一致性见[parity检查](../../../logs/verification/confined_locomotion_2/20261023_independent_multipliers_parity/checks.json)。真实浏览器完成full→4/8→8/8→Exit，确认两倍率独立、同权重且Apply后清空episode/GRU并保持暂停；见[浏览器检查](../../../logs/verification/confined_locomotion_2/20261023_independent_multipliers_browser/browser_checks.json)和[截图](../../../logs/verification/confined_locomotion_2/20261023_independent_multipliers_browser/independent_controls.png)。以上仅为工程验证。

**此前 fixed-channel 工程见证（2026-10-23）**：当时CPU共63项通过。RTX4050双环境 full 与 sparse near×4/far×4 各完成1次24步采集、20次联合更新；两者均启用ALL_THREE扰动与镜像，完整恢复边界逐位一致，同一保留rollout的优化器重放逐位一致。它们是有限工程见证，不代表学习或行走效果：[full](../../../logs/verification/confined_locomotion_2/20261023_fixed_channels_full_training/checks.json)、[sparse×4](../../../logs/verification/confined_locomotion_2/20261023_fixed_channels_sparse_x4_training/checks.json)。当时源码下full与sparse严格训练恢复回归均通过（历史checkpoint不匹配当前源码）：[full regression](../../../logs/verification/confined_locomotion_2/20261023_independent_multipliers_full_regression/checks.json)、[sparse regression](../../../logs/verification/confined_locomotion_2/20261023_independent_multipliers_sparse_regression/checks.json)；历史checkpoint不匹配新增监控源码，当前恢复验证见[监控与保存](training/监控与保存.md)。

[fixed sparse GPU检查](../../../logs/verification/confined_locomotion_2/20261023_fixed_sparse_scan_x4_final/checks.json)中同姿态原始射线、BASE点、固定分组、同候选池FPS、gap/range关联及障碍物Warp对照仍有效。该冻结报告的roll/pitch/height pose fixture曾写入 `root_link_pose_w` 返回的副本，未改变底层姿态；因此其“姿态变化后分组稳定”不能作为证据，已由[actual-pose parity](../../../logs/verification/confined_locomotion_2/20261023_far_hit10_parity_actual_pose/checks.json)的底层pose断言替代。

同一报告中的扫描阶段性能测量使用2环境、2次预热和5次采样，中位数为 full 7.2578ms、near×4/far×4 sparse 2.7264ms。该数字只计扫描链，不代表训练吞吐。[训练阶段小样本benchmark](../../../logs/verification/confined_locomotion_2/20261023_fixed_channels_training_benchmark/benchmark.json)另以每个模式各自的checkpoint恢复后测量，2环境、每次24步采集和20次优化更新，1次预热、3个顺序样本：full中位采集0.9316s、更新6.6851s、总计7.6409s（6.282 envsteps/s）；sparse分别为0.8821s、6.5602s、7.4423s（6.450 envsteps/s）。这是小幅差异；模式使用不同训练checkpoint且样本数少，不能据此推断大环境扩展或收敛表现。显存字段是运行期间总分配量/增量，不是整卡GPU预算。

当前独立 fixed-channel 几何报告显示了明确限制：[随姿态地面覆盖](../../../logs/verification/confined_locomotion_2/20261023_fixed_channels_moving/report.json)在上仰5°、654个有效探针上，near FPS 的P95最近距离为 full 0.497m、near×4/far×4 0.484m；两者仅约22.8%和24.5%的探针在0.20m内。25cm台阶表面0.20m支持率为 full 27.6%、sparse×4 28.2%（312探针）；两种模式的FPS台阶表面P95分别约0.854m和1.112m。[障碍物报告](../../../logs/verification/confined_locomotion_2/20261023_fixed_channels_obstacles/report.json)中，full与sparse×4的上方横梁均有原始射线命中，但最终near/far FPS都保留0个梁点。增加near倍率并不能解决所有FPS/分组限制；没有推荐稀疏默认比例。

GUI的固定角域边界显示目前标为 `Fixed near-channel boundary rays`，显示固定T1分组的角域边界射线，不再表示动态锥体与BASE-x平面裁切。[full viewer](../../../logs/verification/confined_locomotion_2/20261023_fixed_channels_full_viewer_final/checks.json)和[sparse×4 viewer](../../../logs/verification/confined_locomotion_2/20261023_fixed_channels_sparse_x4_viewer/checks.json) GPU检查均通过：真实Viser渲染与HTTP、点源查询、50个控制步/10帧编码及暂停缓存保持纯UI状态均有验证；新旧viewer的HTTP端口均返回200。该轮没有CUA浏览器点击或截图证据，因此不声称手工浏览器交互验收。

此前 XYZ FPS、可视化、恢复和性能报告均属于**旧动态分组源码身份**，不能用其覆盖率（包括旧报告中的约94%数字）、GUI行为或命令作为当前 fixed-channel 证据；查看各自的旧报告时应按当时源码身份解释。旧任务隔离记录仍见[85文件SHA256检查](../../../logs/verification/confined_locomotion_2/20261023_viewer_cli_browser_final/legacy_hash_check.json)。本说明不修改任何冻结日志。

尚未验收：稳定行走、足下遮挡记忆效果、复杂地形控制、5090适配和大规模性能；单次优化通过不证明学习效果。

显存另经独立进程复测：RTX4050下32环境、ALL_THREE与镜像、一次24步采集+20次优化，full/sparse near×4/far×4的更新后GPU进程占用分别为1,566/1,026 MiB，节省540 MiB（34.5%）；PyTorch张量峰值节省38.6%。该测量只覆盖near×4/far×4，不可外推到其他far倍率。2/16/32环境的定义、原始报告与顺序时间benchmark显存污染说明见[显存复测](sparse_rays/设计与验收.md#显存独立进程复测)。

同一模型切换的最终GPU往返检查通过：[GPU检查](../../../logs/verification/confined_locomotion_2/20261023_acquisition_switch_gpu_final/checks.json)记录full→near×4/far×4 sparse→full→near×8/far×4 sparse→near×4/far×4 sparse；同模式同步数的allocated增量为0，旧session弱引用均释放，退出close后清理通过。真实CUA浏览器交互也通过：[浏览器检查](../../../logs/verification/confined_locomotion_2/20261023_acquisition_switch_browser/browser_checks.json)，实际操作full→sparse×4→full→Exit；Apply才触发切换，同端口重建，三段cross_configuration依次为false/true/false，weights一致，切换后paused且command为零。[sparse截图](../../../logs/verification/confined_locomotion_2/20261023_acquisition_switch_browser/sparse_switch.png)与[full截图](../../../logs/verification/confined_locomotion_2/20261023_acquisition_switch_browser/full_switch.png)保留现场。

**历史计划（legacy all_endpoints阶段）**：当时计划按显存预算确定训练环境数和周期checkpoint方案，再做full与sparse配对平地短训；near倍率仍是候选。该计划记录了当时对台阶与悬梁FPS几何保留的关注，不代表当前sampling决策或状态。当前已选hit10工程基线与待验收事项见上文。
