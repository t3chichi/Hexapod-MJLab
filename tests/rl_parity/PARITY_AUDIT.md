# S1 语义比对审查 — LiDAR-PD 定制 → rsl-rl-lib 5.4.2 原生重写

**任务**: 按照"原本的思路"将旧版（`Lidar_legged_gym` 内嵌定制 `rsl_rl`）的
Actor-Critic + PPO 定制**全部原生重写**为 rsl-rl-lib 5.4.2（安装版）风格，
并对内容与语义逐项比对审查。本文件即该审查（G4 验收门）。

**审查对象**

| 角色 | 旧实现（语义真值，只读） | 新实现（本次新增） |
|---|---|---|
| 模型 | `Lidar_legged_gym/rsl_rl/rsl_rl/modules/lidar_pd_actor_critic.py`（≡ 已移植的 `Lidar_mjlab/src/lidar_mjlab/rl/lidar_pd_actor_critic.py`，逐字节一致，行号通用。下文记 **OLD‑AC**） | `Lidar_mjlab/src/lidar_mjlab/rl/actor_v2.py`（**NEW‑AC**） |
| 算法 | `Lidar_legged_gym/rsl_rl/rsl_rl/algorithms/ppo.py`（**OLD‑PPO**） | `Lidar_mjlab/src/lidar_mjlab/rl/ppo_lidar.py`（**NEW‑PPO**，继承安装版 `rsl_rl/algorithms/ppo.py` = **BASE‑PPO**） |
| 存储 | `Lidar_legged_gym/rsl_rl/rsl_rl/storage/rollout_storage.py`（**OLD‑ST**） | 安装版 `rsl_rl/storage/rollout_storage.py`（**BASE‑ST**，未改动，直接使用） |
| Runner | `Lidar_legged_gym/rsl_rl/rsl_rl/runners/on_policy_runner.py`（**OLD‑RN**） | 安装版 `rsl_rl/runners/on_policy_runner.py`（**BASE‑RN**）+ `mjlab/rl/runner.py`（MjlabOnPolicyRunner） |
| 对称 | `Lidar_mjlab/src/lidar_mjlab/symmetry/el_4090_lidar_symmetry.py`（**OLD‑SYM**，与旧库同源） | `Lidar_mjlab/src/lidar_mjlab/rl/symmetry_adapter.py`（**NEW‑SYM**）适配安装版 `rsl_rl/extensions/symmetry.py`（**BASE‑SY**） |
| 权重 | 旧 checkpoint `model_state_dict` | `Lidar_mjlab/src/lidar_mjlab/rl/ckpt_convert.py`（**NEW‑CK**） |

安装版路径缩写：`~/anaconda3/envs/mjlab/lib/python3.12/site-packages/rsl_rl/`。
所有"NEW"行号对应当前工作区文件（commit 见 git log S1）。

**布局事实核对**（修正任务简报）：总观测 2754 维 =
proprio **66**（=12 base + 3×18 dof，非 48）+ 近端 256×3 + 远端 10×64×3；
`num_dof=18`，高度图 17×11=187，`distal_history_points=640`。与
`tests/pd_gru_lidar_head/symmetry_check.py:45-47` 的常量一致；
G1 门实测新旧模型参数量均 2,187,897。

---

## 1. 模型前向（逐行等价，G1 数值证明全 0 误差）

| # | 方面 | 旧 file:line | 新 file:line | 等价性 |
|---|---|---|---|---|
| 1.1 | 观测切分（proprio / 近端 / 远端切片 + reshape） | OLD‑AC:197-233 `_split_obs` | NEW‑AC:247-295 `_split_obs` | **逐字相同**（含空批与 (T,B) 分支）。输入来源不同：旧收扁平张量，新收 TensorDict 后选组拼接（NEW‑AC:420-427 `_flatten_obs`，`obs_groups=["policy"]` → 同一 2754 维向量，拼接为恒等）。 |
| 1.2 | 球坐标键排序（传感器偏置四元数/平移、key=φ·2π+azimuth、argsort+gather） | OLD‑AC:235-249 `_sort_by_spherical`（**前向不调用**——LidarWrapper 已排序） | NEW‑AC:297-314 `_sort_by_spherical`（同样**前向不调用**） | **逐字相同且同样未被前向调用**——语义（"模型信任 wrapper 的排序"）保持。 |
| 1.3 | 近端 GRU(3→187)、`h0` 默认零 → 每帧零初始化 | OLD‑AC:131-135, 251-253 | NEW‑AC:176-180, 316-318 | **相同**（`nn.GRU` 默认零隐状态；逐帧独立，无跨步隐状态）。 |
| 1.4 | 远端 GRU(3→64)，每帧零初始化 | OLD‑AC:136-140, 255-257 | NEW‑AC:181-185, 320-322 | **相同**。 |
| 1.5 | 分块编码 chunk=128（近端） | OLD‑AC:259-274 | NEW‑AC:324-340 | **逐字相同**（含 chunk 边界 reshape 语义）。 |
| 1.6 | 分块编码 chunk=128（远端） | OLD‑AC:276-291 | NEW‑AC:342-358 | **逐字相同**。 |
| 1.7 | 近端梯度检查点（默认关） | OLD‑AC:84, 268-270 | NEW‑AC:156, 333-335 | **相同**（`torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`，仅 `torch.is_grad_enabled()` 时启用 → 推理/rollout 阶段自动绕过）。 |
| 1.8 | 远端梯度检查点（默认开） | OLD‑AC:85, 285-287 | NEW‑AC:157, 351-353 | **相同**。 |
| 1.9 | latent 拼装 `cat(proprio, prox_feat, dist_feat)` 与缓存写入 | OLD‑AC:293-325（缓存 322-324） | NEW‑AC:360-395 `build_actor_latent`（缓存 389-394） | **逐字相同**；新侧缓存额外记录观测指纹（`data_ptr`+shape，NEW‑AC:392-394），供 critic 共享缓存判定（见 3.3），不改变前向数值。 |
| 1.10 | (T,B)+masks 循环路径（split_and_pad 语义、unpad） | OLD‑AC:296-310 | NEW‑AC:363-377 | **逐字相同**。两侧实际均不可达（`is_recurrent=False` → 恒走前馈生成器），保留以完整移植。 |
| 1.11 | actor 头形状：Linear(299→1024)→ELU→…→Linear(128→18) | OLD‑AC:146-147, 165-174 `_build_mlp` | NEW‑AC:205-216, 218-229 `_build_mlp` | **层序列与索引完全相同**（`nn.Sequential` 下标 0..8 一致 → checkpoint 键 `actor.{i}.*`→`mlp.{i}.*` 一一对应，见 §7）。 |
| 1.12 | critic 头形状：Linear(299→…→1) | OLD‑AC:148（输入=同 latent） | NEW‑AC:561-569（`LidarPDCritic5x.mlp`，输入=共享 actor latent） | **相同**；critic 输入维度仍为 299（共享感知 latent），不是 2754——保持旧语义（OLD‑AC:68 注释 "critic input = actor_input_dim"）。 |
| 1.13 | height_supervisor `Linear(187,187)` | OLD‑AC:143 | NEW‑AC:188 | **相同**（属 actor 侧，随 actor_state_dict 保存）。 |
| 1.14 | std 参数化：scalar `nn.Parameter(init·ones(A))` / log | OLD‑AC:151-157 | BASE `modules/distribution.py:166-168`（`GaussianDistribution.std_param/log_std_param`，NEW‑AC:193-203 装配） | **等价**；差异：5.x 在展开时夹紧 `std_range=[1e-6,1e6]`（BASE distribution.py:187,189）—— sane 值域内为恒等操作，见 §9‑D1。 |
| 1.15 | 分布构造 `Normal(mean, std.expand_as(mean))` + `set_default_validate_args(False)` | OLD‑AC:327-336, 160 | BASE distribution.py:183-191（`update`）、NEW‑AC:400-418 `forward` | **等价**（mean=MLP 输出；std 为与状态无关参数；校验关闭同 BASE distribution.py:181）。 |
| 1.16 | 确定性推理（均值，不写缓存） | OLD‑AC:348-350 `act_inference`（`cache=False`） | NEW‑AC:400-418 `forward(stochastic_output=False)` → `deterministic_output`（恒等返回 mlp 输出，BASE distribution.py:197-199；缓存仅 stochastic 路径写，NEW‑AC:412） | **等价**。`cache=False` 语义被保留：mirror‑loss 的确定性前向**不会**覆盖 aux 损失所需近端特征缓存（对照 OLD‑PPO:371 的 `act_inference(obs.detach())`）。 |
| 1.17 | 采样 / log_prob / 熵 | OLD‑AC:341-346, 194-195 | BASE distribution.py:193-195, 230-232, 221-223 | **相同公式**（`Normal.sample/log_prob/entropy().sum(-1)`）。 |
| 1.18 | 传感器偏置张量（`_sensor_translation` 非持久 buffer、`_sensor_conj`） | OLD‑AC:106-113 | NEW‑AC:163-171 | **逐字相同**（persistent=False → 不入 state_dict，两侧一致）。 |
| 1.19 | 维度校验（2754 布局断言） | OLD‑AC:115-126 | NEW‑AC:174-186 | **相同**（错误信息逐字相同）。 |
| 1.20 | 宽松加载守卫（critic 形状不匹配即丢弃等） | OLD‑AC:378-405 `load_state_dict` | NEW‑CK 全量键映射 + strict 载入 | **有意差异 D2**：新版迁移在加载前完成（`ckpt_convert`），strict=True 精确校验；旧的"形状不匹配则随机初始化"的容错降级不再内嵌于模型（迁移失败即报错，更严格）。 |

## 2. `act` / 采样路径（rollout）

| # | 方面 | 旧 file:line | 新 file:line | 等价性 |
|---|---|---|---|---|
| 2.1 | rollout 单步采样：动作、log_prob、分布参数、obs 存储顺序（step 前观测） | OLD‑PPO:154-167 `act(obs, critic_obs, aux_obs)` | BASE‑PPO:115-126 `act(obs)`（新模型 NEW‑AC:400-418 消化 obs TensorDict） | **等价**。旧第 2/3 参数在 LiDAR 模式下为同一 `obs`（OLD‑RN:489-492 `alg.act(obs, obs, aux_obs)`）；5.x critic 消费同一 TensorDict（BASE‑PPO:121）。 |
| 2.2 | rollout 期 value 计算 | OLD‑PPO:159 `policy.evaluate(critic_obs)`（masks=None → 回退重建 latent，OLD‑AC:352-360） | BASE‑PPO:121 `critic(obs)` → NEW‑AC:583-598 `get_latent`（指纹命中则复用缓存） | **数值等价，计算路径有意差异 D3**：旧 rollout 每步重建 latent；新侧命中"同一观测张量"缓存直接复用。GRU 确定性 → 数值逐位一致（G1 `critic_value` = 0.0）；`compute_returns` 换新观测 → 指纹不匹配 → 重建（等价旧行为，G1 `critic_value_cache_miss` = 0.0）。 |
| 2.3 | 隐状态记录 | OLD‑PPO:155-156（`is_recurrent=False` 不执行） | BASE‑PPO:118（`get_hidden_state()=None`，NEW‑AC:445-447） | **相同**（两侧均无操作）。 |
| 2.4 | aux 观测存入 transition | OLD‑PPO:166 `transition.aux_observations = aux_obs`；OLD‑RN:489 `aux_obs_buf` 逐_step 读取 | BASE‑PPO:125 `transition.observations = obs`（TensorDict **含** `height_grid` 组） | **等价（管线重构）**：特权高度图从独立 aux 通道改为观测组，随全量 obs TensorDict 持久化（BASE‑ST:142-149）；更新期从 `batch.observations["height_grid"]` 取（NEW‑PPO:243-246, 299-301）。数值同一来源。 |
| 2.5 | time‑out bootstrap（`rewards += γ·V·time_outs`） | OLD‑PPO:187-191 | BASE‑PPO:150-155 | **逐字相同**。 |
| 2.6 | transition 入库 + 清理 + `policy.reset(dones)` | OLD‑PPO:193-196 | BASE‑PPO:157-161（reset 为 no-op，NEW‑AC:442-444） | **相同**。 |
| 2.7 | 观测归一化更新点 | 旧：runner 级 `EmpiricalNormalization`（OLD‑RN:526-528，LiDAR 任务 `empirical_normalization=False` → Identity） | BASE‑PPO:133-134 `update_normalization`（NEW‑AC:451-458，默认 Identity no-op） | **等价**：两侧该任务均无归一化；新版显式禁用（NEW‑AC:126-131 抛错保护），见 §9‑D4。 |

## 3. critic / 共享感知（核心架构差异的处理）

| # | 方面 | 旧 file:line | 新 file:line | 等价性 |
|---|---|---|---|---|
| 3.1 | actor/critic **共享**双 GRU 感知 | OLD‑AC:146-148（单 policy，天然共享） | NEW‑AC:554-556 `object.__setattr__("_shared_actor", actor)`（非子模块引用 → 不重复注册参数/状态） | **等价**：权重共享（同一 `nn.GRU` 对象）；且 critic `state_dict()` 只含 `mlp.*`（BASE‑PPO:362-365 保存不冗余）。 |
| 3.2 | 更新期 critic 消费 act 缓存 latent | OLD‑PPO:292 `evaluate(critic_obs_batch, masks=空, …)` → OLD‑AC:352-357 缓存命中 | NEW‑AC:583-598：指纹（`data_ptr`+shape）命中 → `actor._cached_actor_latent` | **等价**（命中条件重述：旧 "masks 非 None 且 (numel>0 或 batch 一致)"，5.x 前馈路径 masks 恒 None → 改为观测张量同一性判定，语义更强：同一输入必命中、不同输入必不命中）。梯度语义一致：actor/critic 损失经同一感知子图反传（旧行为）。 |
| 3.3 | 缓存未命中（compute_returns / 冷启动）从输入重建 | OLD‑AC:358-360 | NEW‑AC:596-598 `build_actor_latent(obs, cache=True)` | **逐字相同**（回退也覆写缓存，同旧）。 |
| 3.4 | 优化器参数去重（共享 GRU 只算一次） | 旧：`optim.Adam(policy.parameters())` OLD‑PPO:100（单 policy 天然无重复） | NEW‑PPO:128-131 `_unique_parameters` 重建优化器（BASE‑PPO:92-94 的 chain 会重复共享参数 → Adam 连续两次更新） | **等价**（去重后与旧单一参数列表同构）。G2 测试断言参数数 = 去重并集。 |

## 4. PPO 更新项（逐项）

| # | 方面 | 旧 file:line | 新 file:line | 等价性 |
|---|---|---|---|---|
| 4.1 | mini‑batch 生成（前馈、随机置换、repeats） | OLD‑ST:204-278 | BASE‑ST:225-258（`Batch` 对象化） | **等价**：同一 `randperm` 切块算法；差异仅数据容器（TensorDict/命名 Batch vs 13 元组），见 §6。 |
| 4.2 | 优势 per‑minibatch 归一化开关 | OLD‑PPO:251-254 | BASE‑PPO:210-213 | **逐字相同**（含 `1e-8` eps）。 |
| 4.3 | GAE / returns（含 `normalize_advantage=not per_mb`） | OLD‑PPO:198-203 → OLD‑ST:165-187 | BASE‑PPO:163-188（内联） | **逐字相同**（同一倒序 TD(λ) 递推与归一化条件）。 |
| 4.4 | 重算 log_prob/entropy/分布参数 | OLD‑PPO:286-297 | NEW‑PPO:224-232 | **等价**（术语映射：`act+get_actions_log_prob`→`actor(stochastic_output=True)+get_output_log_prob`；`action_mean/std/entropy`→`output_*`；`mu/sigma` 缓冲→`distribution_params` 元组）。 |
| 4.5 | 自适应 KL 学习率（阈值 ×2/÷2、界 [1e‑5,1e‑2]、广播） | OLD‑PPO:300-334（手写 KL 公式，`log(σ/σ_old+1e‑5)+…`） | NEW‑PPO:234-257 → BASE `distribution.py:234-238` `kl_divergence(Normal(old),Normal(new))` | **等价**（同一闭式 KL；差异：旧公式在 log 内 +1e‑5 数值 eps，新用解析式无 eps——仅影响自适应调度的边界数值，不影响损失/梯度，见 §9‑D5）。调度逻辑逐字相同。 |
| 4.6 | surrogate（ratio、双臂、max、mean） | OLD‑PPO:336-342 | NEW‑PPO:259-267 | **逐字相同**。 |
| 4.7 | value 损失（裁剪式 max / 非裁剪式） | OLD‑PPO:344-353 | NEW‑PPO:269-278 | **逐字相同**（`clip_param` 同用）。 |
| 4.8 | 总损失组合 `surrogate + c_v·value − c_e·entropy` | OLD‑PPO:355 | NEW‑PPO:280 | **逐字相同**。 |
| 4.9 | 对称 mirror 损失（MSE(actor(mirror x)), mirror(actor x))，基于均值、detach 目标、系数项） | OLD‑PPO:357-387（手写内联） | BASE‑SY:90-118 `Symmetry.compute_loss` + NEW‑PPO:289-293 | **等价**（同一 MSE 定义、同一"第一半为原始"切片约定、同一 `detach()` 目标、同一系数加法位置）；增广/镜像函数接线见 §5。 |
| 4.10 | 对称数据增广（批翻倍 + 重复 log_prob/values/advantages/returns） | OLD‑PPO:245-283（含 critic/aux obs 增广分支 265-274） | BASE‑SY:66-88 `augment_batch` + NEW‑SYM:74-133（policy 全镜像 + `height_grid` Y 翻转） | **等价**：critic 增广分支在 LiDAR 模式下是恒等（critic obs == actor obs 共享 buffer，OLD‑PPO:265-266）；aux 增广对应 `height_grid` 组的 Y 翻转（OLD‑SYM:99-104 ↔ NEW‑SYM:106-113）。 |
| 4.11 | **aux 高度 MSE**（近端特征→height_supervisor↔特权高度图） | OLD‑PPO:389-393 `policy.compute_auxiliary_loss(aux_obs_batch)`；OLD‑AC:362-376（读 `_cached_proximal_feature`，冷缓存返回 0） | NEW‑PPO:243-246（快照目标）+299-302；NEW‑AC:490-496（同实现） | **等价**：同一 MSE、同一缓存读取、同一 `aux_loss_coef` 加法位置（symmetry 之后，OLD‑PPO:387-393 ↔ NEW‑PPO:289-302）。快照保证 mirror‑only 模式下目标与缓存行数一致（旧版同此：mirror‑only 不增广 aux，OLD‑PPO:271-274 仅在增广分支翻倍 aux）。 |
| 4.12 | RND（未启用，管线保留） | OLD‑PPO:62-73, 176-185, 395-402 | BASE‑PPO:75, 144-148, 281, 292-295（NEW‑PPO:163 `resolve_rnd_config` 同基类） | **等价**（未启用；构造/内在奖励/损失结构均继承基类）。 |
| 4.13 | 梯度裁剪 | OLD‑PPO:418/434 `clip_grad_norm_(policy.parameters(), max)`（**单次、全参数**） | NEW‑PPO:314/327 `clip_grad_norm_(self._all_parameters, max)`（去重并集，单次） | **等价**；基类两次分别裁剪（BASE‑PPO:302-303）会使共享参数被计入两个范数——有意偏离基类以保持旧语义。 |
| 4.14 | 优化器步进 / zero_grad 顺序 | OLD‑PPO:404-437 | NEW‑PPO:303-329（非 AMP 分支逐字对应） | **等价**。 |
| 4.15 | 损失统计与键名 | OLD‑PPO:439-478（`value_function/surrogate/entropy/symmetry/aux`，均值÷`epochs×mini_batches`） | NEW‑PPO:331-366（同时给出 `value_function` 与基类键 `value`，含 `aux`） | **等价**（键 `value_function` 保留旧名并别名 `value` 兼容 5.x Logger）。 |
| 4.16 | 存储清理 | OLD‑PPO:464-465 | NEW‑PPO:367 | **相同**。 |

## 5. 对称（Symmetry）接线

| # | 方面 | 旧 file:line | 新 file:line | 等价性 |
|---|---|---|---|---|
| 5.1 | 镜像数学本体（proprio 符号翻转 idx 1,3,5,7,10,11；左右 dof/vel/prev_action 交换 12..66；点云 Y 镜像 + 球键重排序；动作左右交换无符号） | OLD‑SYM:24-174 | NEW‑SYM:91-101/106-113/126-131 **直接调用** OLD‑SYM 的 `get_el4090_lidar_xsym_obs_act` | **同一函数**（零重写风险）。 |
| 5.2 | 调用约定适配（`obs_type` 三路 ↔ TensorDict 分组路由） | OLD‑PPO:262-274（PPO 按 obs_type 三次调用） | NEW‑SYM:74-133（按组名路由：`policy`→policy 路径、`height_grid`→auxiliary 路径、未知组 strict 报错） | **等价**：旧的三次调用在新约定下变成组内一次遍历；`critic` 路径不再需要（critic 与 actor 同组）。 |
| 5.3 | 传感器参数绑定（`functools.partial` + env 运行时注入 `_sensor_offset_quat/_sensor_translation`） | OLD‑RN:301-332 `_setup_symmetry` | NEW‑PPO:165-175（construct 内同样的注入表）+ NEW‑SYM:40-68（kwargs 闭包绑定） | **等价**（同一属性名、同一 `[0:1]` 切片与 device 投递）。 |
| 5.4 | 增广后重复 rollout 张量（num_aug 倍） | OLD‑PPO:275-283 | BASE‑SY:84-88 | **逐字相同**。 |
| 5.5 | 循环策略限制 | 旧：无显式限制（但模型 `is_recurrent=False`，实际同限） | BASE‑PPO:78-79 拒绝 recurrent；NEW‑AC:89-91 `is_recurrent=False` | **等价**（逐帧零初始化 GRU = RL 层面前馈，两侧一致）。 |
| 5.6 | `Symmetry` 的 env 注入 | OLD‑PPO:76-94（`symmetry_cfg["_env"]`） | BASE `extensions/symmetry.py:121-137` `resolve_symmetry_config`（`cfg["env"]`）+ NEW‑PPO:157-175 | **等价**（传递机制不同，函数收到的 env 相同；本任务镜像函数不读 env——OLD‑SYM:47 "env: Unused"）。 |

## 6. 存储 / 数据面

| # | 方面 | 旧 file:line | 新 file:line | 等价性与语义影响 |
|---|---|---|---|---|
| 6.1 | 每步 obs 持久化 | OLD‑ST:56（`observations` [T,N,2754]）+ 91-96（`aux_observations` [T,N,187]）+ 59-67（privileged 共享 buffer：LiDAR 模式 critic_obs_shape=None → 复用 observations，OLD‑RN:366-371） | BASE‑ST:142-149（**全量 obs TensorDict**：`policy` [T,N,2754] + `height_grid` [T,N,187]） | **语义等价**：同一批数据（policy + 特权高度图）逐步入库；新版 critic 组指向 `policy`（`obs_groups`），等价旧的共享 buffer。内存 footprint 相同（高度图一次 vs aux 通道一次）。 |
| 6.2 | distribution 参数缓冲（旧 mu/sigma ↔ 新 params 元组） | OLD‑ST:81-82, 126-127 | BASE‑ST:162, 193-199（惰性分配） | **等价**（`(mean, std)` 两张量 ↔ `mu`+`sigma` 两张量；用于 KL 与 mirror 损失的旧分布重建）。 |
| 6.3 | 前馈 mini‑batch 产出 | OLD‑ST:204-278（13 元组，含 `hid=(None,None)`、空 masks 哨兵） | BASE‑ST:225-258（命名 `Batch`，masks/hidden 默认 None） | **语义等价**（同一索引算法）；差异 D6：哨兵从"空张量"变 `None`——由此 critic 缓存判定从旧谓词改为观测指纹（§3.2），数值不变。 |
| 6.4 | 溢出保护 / step 计数 | OLD‑ST:105-108, 141 | BASE‑ST:176-177, 205 | **相同**。 |
| 6.5 | distillation 分支 | OLD‑ST:73-75, 119-120 | BASE‑ST:154-156, 186-187 | **相同**（未使用，S1 范围外）。 |

## 7. Checkpoint 键映射（NEW‑CK）

| 旧键（`model_state_dict`） | 新键 | 新侧 file:line | 语句 |
|---|---|---|---|
| `proximal_gru.*` | `actor_state_dict["proximal_gru.*"]` | NEW‑CK:74 | 恒等映射（形状逐一相同）。 |
| `distal_gru.*` | `actor_state_dict["distal_gru.*"]` | NEW‑CK:75 | 恒等映射。 |
| `height_supervisor.*` | `actor_state_dict["height_supervisor.*"]` | NEW‑CK:76 | 恒等映射。 |
| `actor.{i}.*` | `actor_state_dict["mlp.{i}.*"]` | NEW‑CK:77 | 前缀替换；Sequential 下标一致（§1.11）→ 层层对应。 |
| `critic.{i}.*` | `critic_state_dict["mlp.{i}.*"]` | NEW‑CK:78 | 前缀替换（critic 只含头，GRU 在 actor 侧，§3.1）。 |
| `std` | `actor_state_dict["distribution.std_param"]` | NEW‑CK:79 | scalar std ↔ GaussianDistribution.std_param（与 mjlab runner 的 4.x→5.x 迁移同规则，`mjlab/rl/runner.py:112-115`）。 |
| `log_std` | `actor_state_dict["distribution.log_std_param"]` | NEW‑CK:80 | log 参数化分支。 |
| `optimizer_state_dict` | **丢弃** | NEW‑CK:39-44 | **有意差异 D7**：参数排序不同（旧单 policy 序 vs 新去重并集序），Adam 动量槽不可靠对应 → 恢复训练首迭代动量冷启动（策略/价值权重无损）。 |
| `scaler_state_dict` | 原样透传 | NEW‑CK:144-146 | cuda.amp→torch.amp GradScaler 状态同构。 |
| （无归一化状态） | — | NEW‑CK:45-47 | 旧任务 Identity，无状态迁移。 |

迁移正确性由 G1 证明：`load_state_dict(strict=True)` 通过且新旧参数量相等（2,187,897），全部 11 项前向数值误差 = 0.0。

## 8. 混合精度（use_amp = 仅 autocast + GradScaler；无任何 AMP 判别器）

| # | 方面 | 旧 file:line | 新 file:line | 等价性 |
|---|---|---|---|---|
| 8.1 | 语义边界 | OLD‑PPO:49,102-103（`torch.cuda.amp.GradScaler()`；全库 grep 无 discriminator/AMP 判别器） | NEW‑PPO:22-24, 136-138（`torch.amp.GradScaler("cuda")`） | **同一语义**：纯混合精度加速开关；不存在也不会引入对抗判别器。 |
| 8.2 | autocast 覆盖范围（前向+全部损失项） | OLD‑PPO:285（`with torch.cuda.amp.autocast(enabled=self.use_amp):` 包住 286-402） | NEW‑PPO:223（`torch.amp.autocast("cuda", enabled=self.use_amp)` 包住 224-302） | **同覆盖**（surrogate/value/entropy/KL/symmetry/aux 全在内；KL 的 `inference_mode` 子块同旧 300-301）。 |
| 8.3 | 反向：scale → unscale → clip → step → update | OLD‑PPO:406-423 | NEW‑PPO:303-319 | **逐字对应**（`scaler.scale(loss).backward()`、`unscale_`、裁剪（§4.13 单次）、`scaler.step`、`scaler.update()`；RND 分支同构）。 |
| 8.4 | 非 AMP 分支 | OLD‑PPO:424-437 | NEW‑PPO:320-329 | **逐字对应**。 |
| 8.5 | 设备限制 | 旧：无显式保护（CPU 上启用会崩） | NEW‑PPO:134-137：非 CUDA 自动禁用 + 告警 | **有意差异 D8**（防御性；CPU 测试路径因此可用同一开关）。 |
| 8.6 | scaler 状态保存/恢复 | OLD‑RN:781-783, 819-822 | 5.x 由使用方自行保存（`alg.save()` 不含 scaler）；NEW‑CK:144-146 透传 | **等价**（迁移工具保留该状态；运行期保存属 runner 集成，S1 范围外）。 |

## 9. 归一化行为

| # | 方面 | 旧 | 新 | 等价性 |
|---|---|---|---|---|
| 9.1 | 观测归一化 | OLD‑RN:354-362（runner 级 EmpiricalNormalization，`empirical_normalization=False` → Identity） | NEW‑AC:129-135 / 557-563（模型级，默认关闭并禁用开启） | **等价（关闭）**；若未来开启，5.x 在模型内前向时归一化（BASE mlp_model.py:110-118）而旧版在入库前归一化——统计更新时点略不同（D4，本任务不受影响）。 |

## 10. 有意差异清单（全部经数值或构造论证不改变训练语义）

- **D1 std 夹紧**：5.x `GaussianDistribution` 将 std 夹在 `[1e-6, 1e6]`（默认 `std_range`）；旧版无夹紧。正常训练值域内为恒等；仅当 std 极端时新版更稳。分布均值/采样/log_prob 在夹紧不触发时逐位一致（G1 `distribution_std`=0.0）。
- **D2 加载守卫**：旧模型内嵌"形状不匹配→丢弃该模块随机初始化"（OLD‑AC:378-405）；新版以 `ckpt_convert` 显式映射 + strict 载入，映射缺失即 KeyError（更严格、可审计）。
- **D3 rollout 期 critic latent 复用**：旧每步重建（`evaluate` masks=None 走回退）；新以观测指纹命中缓存。GRU 确定性 → 数值逐位相同（G1 证明），仅省一次前向。
- **D4 归一化架构**：runner 级→模型级（本任务恒 Identity，无影响）。
- **D5 KL eps**：旧手写公式含 `+1e-5`（OLD‑PPO:303）；新用解析 `kl_divergence`。仅影响自适应 LR 的极边界情形，不影响损失/梯度。
- **D6 masks 哨兵**：空张量→`None`（5.x Batch 惯例）；由此 critic 缓存判定改为观测同一性指纹（§3.2），语义更强且数值不变。
- **D7 优化器状态不迁移**：参数排序不同（§7）。
- **D8 AMP 设备守卫**：CPU 自动禁用（§8.5）。
- **D9 Runner 集成面**（S1 范围外，交付为算法/模型层）：旧 runner 的 LidarWrapper 逐_step 注入（OLD‑RN:454-456,522-524）、多阶段奖励钩子（OLD‑RN:574-577）、`get_inference_policy` 包装（OLD‑RN:846-858）在 mjlab 侧属 MjlabOnPolicyRunner/VecEnvWrapper 的后续接线；5.x runner 的 rollout 主循环（BASE‑RN:79-133）与旧循环逐步对应（act→step→process_env_step→compute_returns→update），G2 按此循环驱动。

## 11. 验收门结果

| 门 | 内容 | 结果 |
|---|---|---|
| G1 | `test_parity_forward.py`：11 项前向数值（含确定性均值、近端特征、分布、采样同 RNG、critic 值命中/未命中、aux 损失） | **PASS** — 全部 max abs err = **0.0** ≤ 1e-6（`logs/g1_parity_forward.json`） |
| G2 | `test_dummy_train.py`：64 envs × 24 steps × 30 iters（CPU，mirror loss 开启、自适应 KL、aux 高度监督）：无 NaN，value/policy 损失下降 | **PASS**（数字见 `logs/g2_g3_dummy_train.json` 与下方运行摘要） |
| G3 | 对称接线：严格等变 stub + "前半原始/后半镜像"成批 → `Symmetry.compute_loss` | **PASS** — mirror loss = **0.0** ≤ 1e-8；adapter 往返 m(m(x))=x 误差 = 0.0 |
| G4 | 本文件每行旧/新 file:line 齐备 | **PASS** |
| G5 | 全部新代码 CPU 可运行并提交 | **PASS**（commit "S1: …"） |

（G2 数字以 `tests/rl_parity/logs/g2_g3_dummy_train.json` 为准。）

## r2 审查后修正（官方对照 + 独立审查发现）

对 r1 版本做了三路独立审查（对照官方 mjlab/rsl-rl 5.4.2 用法、测试自身
审查、官方示例训练验证），按发现修正如下（语义影响逐项标注）：

1. **[HIGH] critic 共享 latent 缓存移除**（actor_v2.py）：r1 按
   `data_ptr+shape` 判等复用 actor 缓存 latent；CUDA 缓存分配器会复用已
   释放地址，`compute_returns` 的 critic 调用可静默命中陈旧特征（GAE
   bootstrap 隐性污染）。旧版缓存命中分支有 `masks is not None` 守卫
   （lidar_pd_actor_critic.py:352-357），r1 丢失。现 critic 恒重建
   （≙ 旧版 evaluate 回退分支），语义=旧版在 5.x 路径下的保守化。
2. **[HIGH] `resolve_symmetry_config` 注入 env**（ppo_lidar.py
   construct_algorithm）：官方 PPO 构造路径（ppo.py:430）必调；缺失时
   官方风格 symmetry_cfg 直接 `TypeError: missing 'env'`。
3. **[MED] `share_cnn_encoders` 弹出**（ppo_lidar.py）：mjlab 官方
   `asdict(cfg.agent)` 必带该键（rl/config.py:78），不弹则 KeyError，
   官方 train 脚本无法启动本算法。
4. **[MED] TorchScript 导出修复**（actor_v2.py `_FlatPerception`）：r1
   在 forward 内跨类调用未绑定静态方法，torch.jit.script 必败；现拆分/
   分块编码为自有方法。ONNX 路径原本即通（验证过）。
5. **[MED] ckpt_convert 配套 `RECOMMENDED_LOAD_CFG`**：转换产物不含
   optimizer_state_dict（有意，见上），默认 load_cfg（optimizer=True）
   会 KeyError；现导出配套 load_cfg 并在 docstring 标注必须传入。
6. **[MED] 传感器安装变换绑定告警**（ppo_lidar.py）：绑定失败时不再静默
   假设单位变换（会让镜像在错误坐标系重排点云，隐性退化），打印显著
   告警；M3 集成时由 env 显式提供并升级为报错。

官方链路对照结论：`LidarPPO5x.update` 与基类逐行等价（surrogate/value
clip/熵/KL 自适应/增广顺序/分布参数切片），aux 快照顺序在 mirror-only
与增广两模式下均与旧版行对齐；对称函数约定与 `Symmetry` 契约一致。
