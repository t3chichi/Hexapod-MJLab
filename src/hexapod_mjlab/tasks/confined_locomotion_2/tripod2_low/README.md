# confined_locomotion_flat_tripod2_low

独立注册任务：`El-4090-Confined-Locomotion-Flat-Tripod2-Low_2`。

本任务迁移旧 `el4090_tripod2_low` 的实际训练行为，使用当前 Flat Baseline 的原始 `ConfinedLocomotionNetwork`、点云扫描与采样实现，保留独立 Legacy 配方。注册由父包 `confined_locomotion_2/__init__.py` 集中完成。

## 配方与复现边界

参考配置为 `/home/t3chichi/el4090_legged_gym/legged_gym/legged_gym/envs/el_4090/spider_nomal/el4090_tripod2_low_config.py`，包括 `El4090SpiderCfg`、`ElSpiderAirRoughCfg`、`LeggedRobotCfg` 和对应 PPO 父类；执行链为 `EL_4090 → ElSpider → LeggedRobot` 及该仓库的 `OnPolicyRunner/PPO/RolloutStorage`。该外部路径仅用于审查，安装和运行新任务不依赖它。

- 策略网络冻结：点 GRU 3→187、时序 GRU 196→187、近端256点、当前远端128点、50Hz控制/10Hz扫描、原 Actor/Critic 和6维包络监督。包络损失系数1、detach关系、固定标签与原采样默认值不变；无187维旧高度观测。
- 原 `anydrive_v3_lstm.pt` 已随包分发，SHA256 为 `21e596737788d7328f28513643e88775a62ef60bf31bdbd0b8fc796928bdb135`。每0.005s子步输入 `[0.25*clip(action,±100)-q, qdot]`，网络内部输入倍率 `[2,.25]`，两层 hidden=8 LSTM，输出倍率20。h/c跨子步保留，局部reset清对应env的18关节状态。普通PD参数不参与控制。
- `legacy.py` 为19项原奖励、接触计时、命令与阶段机制的数值实现。全部权重乘0.02，保留负奖励；高度目标0.25m、腾空时间基准0.25s。仅当前阶段权重非零的项影响回报；保留最后子步力矩惩罚、原小指令门槛与同步/反同步公式。
- 最近最多100个完成episode的总回报均值严格大于2，在PPO更新后升至Stage1；窗口与跨阶段未结束episode累计回报不清零。命令x/y初始±1、yaw±1.2，按episode计数每200步独立均匀重采样，重采样发生在该步奖励之前。x curriculum 每侧扩0.5，上限±2.5，保留旧全局时钟与奖励阈值触发条件。
- 只做固定倍率归一化，无running normalization。按原noise_level=1.5迁移本体uniform noise；网络倍率后的半幅依次为lin_vel=.3、ang_vel=.075、gravity=.075、q=.075、qdot=.1125。command/prev_action无噪声，缩放后clip±100。关闭LiDAR缺口和测距扰动。
- 4096 env、seed=1为旧默认；训练预算必须显式指定。24步rollout、5epochs、4minibatches、Adam lr=.001、γ=.99、λ=.95、clip=.2、value=1、entropy=.01、gradnorm=1、desired_KL=.01、镜像增强及mirror loss=.6。直接优化初值1的18维std，保留原KL的1e-5、仅原样本KL、更新前学习率调度和跨epoch同一shuffle。
- 旧timeout为`episode_length > 1000`，允许timeout与翻倒同时出现；使用`γV(s_t)`补偿，再按旧storage计算GAE。没有低高度/持续倾斜终止。原BaseTask.reset的一个零动作控制步作为warmup保留，随后随机化初始episode计数。warmup不计入PPO采样量；计入物理/执行器/接触状态与全局push时钟。
- DR在创建时固定：64桶机器人摩擦[.3,1.25]、base附加质量[-10,10]kg；reset不重抽。root六维速度U(-.5,.5)，关节参考/初始角均0；保留乘零的joint reset随机数调用。每150个全局控制步覆盖世界系xy速度U(-1,1)，不使用MJLab默认的“增加速度”push。
- 原reset/push后的观测会保留之前计算的body velocity/gravity，下一步才更新；本任务显式缓存该时序。原reset后的previous-action尾值与last-contact滤波尾值也保留。

## 日志与恢复

训练使用官方 `rsl_rl.utils.logger.Logger` 的writer、`process_env_step`、`log` 和默认终端排版。官方 `Train/mean_reward` 与 `Train/mean_episode_length` 是最近最多100个完成episode的滚动均值；Tripod既有 `Episode/mean_return` 与 `Episode/mean_length_steps` 仍按本轮完成的episode统计。未结束episode累计张量与Logger共享，checkpoint另外保存可选的Logger滚动窗口；旧checkpoint缺少该窗口时仍可完整恢复，未结束累计照常恢复，滚动均值从空窗口开始。TensorBoard索引从0起，Recipe及checkpoint的完成轮数仍累计计数。共同规则见[训练监控与保存](../../../../../docs/tasks/confined_locomotion_2/training/监控与保存.md)。

## MuJoCo适配及已知限制

原雷达MJCF文件不修改。任务构建时仅调整内存spec：零关节参考、单一任务keyframe、显式地面接触pair，以及允许LSTM原始输出通过的力矩接口。

1. **初始化高度适配**：按实际编译碰撞mesh的最低顶点确定root高度，保留2mm间隙，当前为`0.4830061266m`，替代会穿透约31mm的旧`0.45m`。不改变关节初角、动作倍率或奖励目标。
2. **高度目标可达性**：root直立且z=.25m时，HIP网格穿地约71.5mm；仅足端IK收敛不能证明合法站姿。HIP对应直立几何下界约.3215m。保留旧.25m软惩罚目标，接受各奖励目标无法同时达到；不宣称该高度是合法直立站姿。
3. **质量与惯量**：当前雷达模型编译质量40.238978kg，无雷达模型约39.551374kg；差值含玻璃visual geom自动推断质量.091034kg。base DR保持原附加质量范围，MuJoCo按质量比缩放base惯量并重算常量。旧PhysX调用`recomputeInertia=True`，其形状重算惯量与此方法未证明完全等价。
4. **摩擦**：27个显式robot-floor pairs使用旧代码注明的`(μ_robot+1)/2`，condim=3；self-contact仍使用机器人材质。该接触系数已通过CPU实际contact检查。MuJoCo摩擦锥、接触正则化与PhysX静/动摩擦、contact_offset等没有严格一一对应。
5. **力矩**：原Python LSTM分支不clip±80，MuJoCo任务spec关闭motor与joint的力矩限制；实际100Nm输入可传递到100Nm关节力。旧URDF仍声明80Nm，本机无法实测PhysX直接effort接口是否还会限幅，因此这里只证明对齐了Python显式输出链，不声称已证明引擎侧限幅等价。
6. **接触数据**：复用原项目控制边界forward与净力传感器；PhysX与MuJoCo的力采样/求解不可能逐位相同。旧步态注释约定脚序LB/LF/LM/RB/RF/RM，本任务按名称映射，不照抄MuJoCo整数索引；旧Isaac实际资产枚举顺序仍需目标旧环境输出核对。
7. **恢复**：新checkpoint保存完整因果状态（物理、DR model、LSTM、双GRU、PPO/Adam/std、阶段/100回报窗口、commands/curriculum、scan/cache/noise、PRNG）。旧checkpoint只保存部分训练状态，新任务采用独立schema并在阶段切换后保存，恢复能力有意增强。

## 本地检查与有限GPU验收

在仓库根目录、`hp_mjlab` 环境执行：

```bash
python -B tests/tripod2_low/run_checks.py
python -B tests/tripod2_low/check_gpu.py
```

第二条仅检查CUDA；不可用时退出码2，不启动训练。GPU可用后，以下命令仅运行2env、1次PPO更新并检查严格恢复后的下一物理transition：

```bash
python -B tests/tripod2_low/check_gpu.py --run --gpu 0
```

当前仅完成CPU数值/行为与1秒MuJoCo接触诊断；没有完成GPU runtime、GPU严格续训或学习效果验收。CPU诊断不作为训练效果证据。正式长训练仍需单独授权。

独立有限训练入口（2env/1update示例）：

```bash
python -B scripts/experiments/confined_locomotion_2/train_tripod2_low.py \
  --seed 1 --num-envs 2 --iterations 1 --engineering-validation
```

续训使用同入口加`--resume-checkpoint /absolute/path/to/checkpoint.pt`；保持seed、env数、采样、模型/地形配方一致；源码、依赖、设备、执行模式和保存间隔只作记录。`--iterations`表示本次追加更新预算。默认4096env仅是配方规模，必须先在目标GPU验证双GRU点云路径的实际显存需求，不自动缩放batch或修改PPO。

有限确定性均值策略回放，无学习、无隐式reset；翻倒即停止：

```bash
python -B scripts/experiments/confined_locomotion_2/play_tripod2_low.py \
  --checkpoint /absolute/path/to/checkpoint.pt --command .5 0 0 --steps 500
```

此入口生成回放诊断，不是旧任务的交互式Play界面。稀疏采样checkpoint需要保持其选集文件路径和内容。

## 历史checkpoint兼容

2026-10-28 P1 移除了源码路径/SHA硬门。本轮物理修复进一步要求 `hexapod_mjlab.model_physics/v1` 签名：名义模型必须兼容，运行时 `body_mass/body_inertia/geom_friction/pair_friction` 从保存的 DR 状态恢复，不能将随机化差异误判成静态模型变化。缺少签名的历史文件在应用状态前拒绝完整恢复；任务/schema/网络/sampling符合条件的文件仍可用于weights-only Play。执行器资产、动作/奖励/终止、PPO配方、seed与环境数仍检查。

Recipe与完整Checkpoint保存启动时的 `code_provenance`，用于追溯commit及未提交改动，不作为恢复门控。兼容预检失败不修改训练状态；应用失败才冻结为faulted。设备/依赖元数据、执行模式和保存间隔不单独门控，但跨GPU完整恢复未在本轮验证。规则见[监控与保存](../../../../../docs/tasks/confined_locomotion_2/training/监控与保存.md)。

修改前保存的本地源码快照 `.checkpoint_sources/9a1f296d3b84c63b775f39f47c91b7a3612163c3/` 仅作为历史资产保留，当前 loader 不再需要它；不要删除用户数据。更早 schema、不同模型或不同扫描/配方仍可能不兼容，以实际加载验证为准。
