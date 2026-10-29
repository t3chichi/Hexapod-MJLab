# confined_locomotion_flat_tripod2_low

独立注册任务：`El-4090-Confined-Locomotion-Flat-Tripod2-Low_2`。

本任务迁移旧 `el4090_tripod2_low` 的实际训练行为，使用当前 Flat Baseline 的原始 `ConfinedLocomotionNetwork`、点云扫描与采样实现。三个v2任务共用 `TemporalPPOTrainer` 与 `TemporalPPOConfig`；Tripod仍有独立Runtime、Runner、配方身份和Play入口。注册由父包 `confined_locomotion_2/__init__.py` 集中完成。

## 配方与复现边界

参考配置为 `/home/t3chichi/el4090_legged_gym/legged_gym/legged_gym/envs/el_4090/spider_nomal/el4090_tripod2_low_config.py`，包括 `El4090SpiderCfg`、`ElSpiderAirRoughCfg`、`LeggedRobotCfg` 和对应 PPO 父类；执行链为 `EL_4090 → ElSpider → LeggedRobot` 及该仓库的 `OnPolicyRunner/PPO/RolloutStorage`。该外部路径仅用于审查旧任务来源，当前训练使用v2统一Temporal PPO，不加载旧PPO实现。

- 策略网络冻结：点 GRU 3→187、时序 GRU 196→187、近端256点、当前远端128点、50Hz控制/10Hz扫描、原 Actor/Critic 和6维包络监督。包络损失系数1、detach关系、固定标签与点数、排序和FPS实现不变；远端候选默认规则按下文当前训练设置更新，无187维旧高度观测。
- 原 `anydrive_v3_lstm.pt` 已随包分发，SHA256 为 `21e596737788d7328f28513643e88775a62ef60bf31bdbd0b8fc796928bdb135`。每0.005s子步输入 `[0.25*clip(action,±100)-q, qdot]`，网络内部输入倍率 `[2,.25]`，两层 hidden=8 LSTM，输出倍率20。h/c跨子步保留，局部reset清对应env的18关节状态。普通PD参数不参与控制。
- [`components/tripod_mdp.py`](../../../src/hexapod_mjlab/tasks/confined_locomotion_2/components/tripod_mdp.py) 为19项原奖励、接触计时、命令与阶段机制的数值实现。全部权重乘0.02，保留负奖励；高度目标0.25m、腾空时间基准0.25s。仅当前阶段权重非零的项影响回报；保留最后子步力矩惩罚、原小指令门槛与同步/反同步公式。
- 最近最多100个完成episode的总回报均值严格大于2，在PPO更新后升至Stage1；窗口与跨阶段未结束episode累计回报不清零。命令x/y初始±1、yaw±1.2，按episode计数每200步独立均匀重采样，重采样发生在该步奖励之前。x curriculum 每侧扩0.5，上限±2.5，保留旧全局时钟与奖励阈值触发条件。
- 只做固定倍率归一化，无running normalization。按原noise_level=1.5迁移本体uniform noise；网络倍率后的半幅依次为lin_vel=.3、ang_vel=.075、gravity=.075、q=.075、qdot=.1125。command/prev_action无噪声，缩放后clip±100。关闭LiDAR缺口和测距扰动。
- 4096 env、seed=1为旧默认；训练预算必须显式指定。共享配方保留24步rollout、完整24步BPTT、5 epochs、4 minibatches、Adam lr=.001、γ=.99、λ=.95、clip=.2、value=1、entropy=.01、gradnorm=1、desired_KL=.01、镜像增强及mirror loss=.6。策略直接优化初值1的18维std；KL使用解析计算，不含旧epsilon。每个rollout只shuffle一次并跨5个epoch复用；KL与熵只使用原始样本，镜像损失仍使用镜像样本。学习率按当前KL在每次Adam step前调整。GAE使用每步pre-reset的next_value；termination不bootstrap，truncation bootstrap。Terrain高度辅助语义保留在共享trainer中。
- 旧timeout为`episode_length > 1000`，允许timeout与翻倒同时出现；仅timeout时按truncation bootstrap，若同时翻倒则termination优先、不bootstrap。没有低高度/持续倾斜终止。原BaseTask.reset的一个零动作控制步作为warmup保留，随后随机化初始episode计数。warmup不计入PPO采样量；计入物理/执行器/接触状态与全局push时钟。
- DR在创建时固定：64桶机器人摩擦[.3,1.25]、base附加质量[-10,10]kg；reset不重抽。root六维速度U(-.5,.5)，关节参考/初始角均0；保留乘零的joint reset随机数调用。每150个全局控制步覆盖世界系xy速度U(-1,1)，不使用MJLab默认的“增加速度”push。
- 原reset/push后的观测会保留之前计算的body velocity/gravity，下一步才更新；本任务显式缓存该时序。原reset后的previous-action尾值与last-contact滤波尾值也保留。

## 日志与恢复

训练使用官方 `rsl_rl.utils.logger.Logger` 的writer、`process_env_step`、`log` 和默认终端排版。官方 `Train/mean_reward` 与 `Train/mean_episode_length` 是最近最多100个完成episode的滚动均值；Tripod既有 `Episode/mean_return` 与 `Episode/mean_length_steps` 仍按本轮完成的episode统计。未结束episode累计张量与Logger共享，checkpoint另外保存可选的Logger滚动窗口；其余物理和PPO身份兼容的checkpoint缺少该窗口时仍可完整恢复，未结束累计照常恢复，滚动均值从空窗口开始。TensorBoard索引从0起，Recipe及checkpoint的完成轮数仍累计计数。共同规则见[训练监控与保存](training/监控与保存.md)。

## MuJoCo适配及已知限制

原雷达MJCF文件不修改。任务构建时仅调整内存spec：零关节参考、单一任务keyframe、显式地面接触pair，以及允许LSTM原始输出通过的力矩接口。Tripod环境入口为 [`tripod_env_cfg.py`](../../../src/hexapod_mjlab/tasks/confined_locomotion_2/tripod_env_cfg.py)，LSTM执行器资产位于[`assets/actuators/anydrive_v3_lstm.pt`](../../../src/hexapod_mjlab/tasks/confined_locomotion_2/assets/actuators/anydrive_v3_lstm.pt)。

1. **初始化高度适配**：按实际编译碰撞mesh的最低顶点确定root高度，保留2mm间隙，当前为`0.4830061266m`，替代会穿透约31mm的旧`0.45m`。不改变关节初角、动作倍率或奖励目标。
2. **高度目标可达性**：root直立且z=.25m时，HIP网格穿地约71.5mm；仅足端IK收敛不能证明合法站姿。HIP对应直立几何下界约.3215m。保留旧.25m软惩罚目标，接受各奖励目标无法同时达到；不宣称该高度是合法直立站姿。
3. **质量与惯量**：当前雷达模型编译质量40.238978kg，无雷达模型约39.551374kg；差值含玻璃visual geom自动推断质量.091034kg。base DR保持原附加质量范围，MuJoCo按质量比缩放base惯量并重算常量。旧PhysX调用`recomputeInertia=True`，其形状重算惯量与此方法未证明完全等价。
4. **摩擦**：27个显式robot-floor pairs使用旧代码注明的`(μ_robot+1)/2`，condim=3；self-contact仍使用机器人材质。该接触系数已通过CPU实际contact检查。MuJoCo摩擦锥、接触正则化与PhysX静/动摩擦、contact_offset等没有严格一一对应。
5. **力矩**：原Python LSTM分支不clip±80，MuJoCo任务spec关闭motor与joint的力矩限制；实际100Nm输入可传递到100Nm关节力。旧URDF仍声明80Nm，本机无法实测PhysX直接effort接口是否还会限幅，因此这里只证明对齐了Python显式输出链，不声称已证明引擎侧限幅等价。
6. **接触数据**：复用原项目控制边界forward与净力传感器；PhysX与MuJoCo的力采样/求解不可能逐位相同。旧步态注释约定脚序LB/LF/LM/RB/RF/RM，本任务按名称映射，不照抄MuJoCo整数索引；旧Isaac实际资产枚举顺序仍需目标旧环境输出核对。
7. **恢复**：新checkpoint保存完整因果状态（物理、DR model、LSTM、双GRU、PPO/Adam/std、阶段/100回报窗口、commands/curriculum、scan/cache/noise、PRNG），trainer schema为`temporal_ppo/v2`，Recipe记录必需的`ppo_semantics`身份。旧PPO checkpoint缺少该身份时，在应用任何状态前拒绝完整续训；不会迁移Adam，也不宣称旧训练等价。任务、架构和sampling匹配时，旧文件仍可用于原sampling下的weights-only Play。Tripod仍独立维护Runtime/Runner及checkpoint包装。

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

已完成CPU数值/行为、有限真实GPU训练、完整恢复及下一物理transition检查。统一PPO之前，本机在2026-10-29产生了100环境、1000次旧PPO更新并正常结束的训练产物；这不等同于策略收敛、步态质量或跨设备迁移验收。CPU诊断和短时闭环不作为学习效果证据。

独立有限训练入口（2env/1update示例）：

```bash
python -B scripts/experiments/confined_locomotion_2/train_tripod2_low.py \
  --seed 1 --num-envs 2 --iterations 1 --engineering-validation
```

续训使用同入口加`--resume-checkpoint /absolute/path/to/checkpoint.pt`；保持seed、env数、采样、模型/地形配方一致。Tripod新训练默认使用sparse、`hit_only_10m`及包内固定选集 `assets/fixed_channels/v1_hit10/near_x6_far_x10.npz`；与当前PPO身份兼容的 `full/all_endpoints` checkpoint续训须显式传入对应扫描参数；旧PPO文件只能按原扫描配置回放，不能通过调整扫描参数绕过PPO身份门控。`--point-replay-chunk`接受正整数，默认1024，控制点GRU编码/重计算块大小；4050可按需设为256或128，不改变24步BPTT、PPO minibatches或网络，并记录在Recipe/Checkpoint执行元数据中，不作为完整恢复硬门。源码、依赖、设备、执行模式、此chunk参数和保存间隔只作记录。`--iterations`表示本次追加更新预算。默认4096env仅是配方规模，不自动缩放batch或修改PPO。

原生窗口使用与Flat/Terrain相同的MuJoCo passive渲染器和交互循环，保留独立Runtime与LSTM执行器。默认暂停，Space运行/暂停、F8单步、Enter重置、F12退出；↑/↓调前进、←/→调侧移、Home/End调转向，Delete清零，F9/F10切换前一个/后一个环境。Tripod指令限幅为vx±2.5、vy±1、yaw±1.2。翻倒后暂停，须显式重置；不使用Flat的低高度/持续倾斜终止规则。

```bash
python -B scripts/experiments/confined_locomotion_2/play_tripod2_low.py \
  --checkpoint /absolute/path/to/checkpoint.pt --command .5 0 0
```

用`--start-running`直接开始；`--steps N`限制累计策略步数（交互默认不限）。无桌面环境使用`--no-viewer --steps 500`进行有限推理，无界面默认500步。`--output-dir`（别名`--output`）须为新目录，省略时保留到临时目录；诊断包含`play_metrics.jsonl`与`play_summary.json`。

这是weights-only Play：可以将100环境训练的网络用于一个新建回放环境，不恢复训练时的优化器、100份环境状态或循环状态。初始化保留一次零动作warmup，不计入策略步数；手动重置沿用Tripod Runtime规则，清理策略GRU及执行器LSTM，不额外执行warmup。扫描配置从checkpoint读取，稀疏模型仍需保留选集路径和内容。本轮补齐原生界面，尚未给Tripod增加网页Debug。

## 历史checkpoint兼容

2026-10-28 P1 移除了源码路径/SHA硬门。本轮完整恢复要求 `hexapod_mjlab.model_physics/v1` 签名及新Temporal PPO身份`temporal_ppo/v2` + Recipe的必需`ppo_semantics`：名义模型必须兼容，运行时 `body_mass/body_inertia/geom_friction/pair_friction` 从保存的 DR 状态恢复，不能将随机化差异误判成静态模型变化。缺少任一身份的历史文件在应用状态前拒绝完整恢复；任务/网络/sampling符合条件的文件仍可用于weights-only Play。执行器资产、动作/奖励/终止、PPO配方、seed与环境数仍检查，不迁移Adam状态。

Recipe与完整Checkpoint保存启动时的 `code_provenance`，用于追溯commit及未提交改动，不作为恢复门控。兼容预检失败不修改训练状态；应用失败才冻结为faulted。设备/依赖元数据、执行模式和保存间隔不单独门控，但跨GPU完整恢复未在本轮验证。规则见[监控与保存](training/监控与保存.md)。

修改前保存的本地源码快照 `.checkpoint_sources/9a1f296d3b84c63b775f39f47c91b7a3612163c3/` 仅作为历史资产保留，当前 loader 不再需要它；不要删除用户数据。更早 schema、不同模型或不同扫描/配方仍可能不兼容，以实际加载验证为准。
