# T0-A执行结果：实际基线审计与平地训练、评测管线

2026-10-03。历史状态：**Codex审查验收不通过**，存在评测、物理诊断、静态判据、镜像及身份重放等缺陷。初次审查结论和修复要求见[T0-A审查与验收](T0-A审查与验收.md)。

修复后的当前结论见[使用与验收](../T0-A使用与验收.md)。

以下保留GLM原交付记录及其自报结果，**不代表经审查认可的通过项**；原文副本保存于`logs/t0/review-20261003/original_docs/`。

> 原自报摘要：2026-10-03。状态：**已执行，待Codex审查；交付为PARTIAL**——契约、接触、短训练、评测、重放与旧任务回归全部通过；唯一未通过项是32环境5秒保持门中的尾段关节速度检查（T1协议0.05 rad/s，实测0.094 rad/s），按方案未放宽阈值，差分证据指向hfield地形表示的接触离采样抖动（详见下文）。短训练成功仅证明训练管线，**不能声称已学会稳定行走**；前进指令下0/32通过是如实结果。

范围与验收标准以[T0-A执行方案](../T0-A执行方案.md)为唯一依据，本文件只报告结果，不回写方案要求。

## 冻结交付与复现

权威run目录：`logs/t0/t0a-accept-20261003-203540/`（独占目录，所有阶段报告、逐门evidence、原始episode、checkpoint与源码快照在内；`acceptance.json`汇总，`returncode=1`如实反映保持门失败）。修改前基线冻结于`logs/t0/t0a-20261003-193252/audit_identity.json`（共享文件改动前的全量源码/资产SHA）。

```bash
conda activate hp_mjlab
# 完整验收（约4分钟，GPU暖机后；exit 1=有必需门未过，当前预期即如此）
PYTHONDONTWRITEBYTECODE=1 python scripts/tools/t0/run_acceptance.py
# 单阶段
PYTHONDONTWRITEBYTECODE=1 python scripts/tools/t0/audit_identity.py
PYTHONDONTWRITEBYTECODE=1 python scripts/tools/t0/contract_gate.py --num-envs 16
PYTHONDONTWRITEBYTECODE=1 python scripts/tools/t0/contact_gate.py --num-envs 8
PYTHONDONTWRITEBYTECODE=1 python scripts/tools/t0/hold_gate.py --num-envs 32
PYTHONDONTWRITEBYTECODE=1 python scripts/tools/t0/train_short.py --num-envs 32 --iters 5 --seed 1
PYTHONDONTWRITEBYTECODE=1 python scripts/tools/t0/eval_episodes.py \
  --checkpoint logs/t0/t0a-accept-20261003-203540/t0_short_model.pt
# 重放：身份核验 + checkpoint加载前向 + 从原始episode重算summary（输出到新目录）
PYTHONDONTWRITEBYTECODE=1 python scripts/tools/t0/replay_verify.py \
  --run-dir logs/t0/t0a-accept-20261003-203540
# CPU单元测试与旧任务回归
PYTHONDONTWRITEBYTECODE=1 python tests/t0/test_units.py
PYTHONDONTWRITEBYTECODE=1 python tests/env/gate_a.py --num-envs 8 --steps 40
PYTHONDONTWRITEBYTECODE=1 python tests/env/gate_b.py --num-envs 8 --steps 40
```

每阶段JSON含argv、returncode、墙钟、逐门verdict与evidence；`stage_logs/`保存完整stdout；`source_snapshot/`冻结可执行源码副本与SHA。训练checkpoint SHA256：`90fae00ce120e864c6910f1aacd9e73bc466d99115d4592fd3488016a7197759`（`t0_short_model.pt`，同时有model_0..4.pt逐迭代副本）。

## GLM原自报逐门结果

| 验收门（方案§5） | 结果 | 关键证据 |
|---|---|---|
| 身份/范围 | **PASS** | dirty工作树为基线冻结（20项）；模块`__file__`全部登记（mjlab为site-packages安装版）；资产/源码/依赖SHA；两组既有checkpoint登记为`UNKNOWN_ORIGIN_SHORT_PIPELINE`（仅model_0/1，不作成熟策略声明）。改动限于任务内4个共享文件（新增保持旧默认值的参数）+新`t0/`包+工具/测试/文档；资产字节与冻结基线完全一致 |
| 平地契约 | **PASS**（12/12门） | 物理/LiDAR/高度监督同一平地：measured_heights全0、LiDAR地面命中世界z最大偏差0.0m（44,602条命中）、height_grid公式误差0、6足真实接触；50/200/10Hz时钟与tick间距5/5；18铰链名义限位在内存spec关闭（env模型jnt_limited全0）、地形μ=1.0且priority=1；观测2754/187/18全有限；单环境强制跌倒→仅该环境reset回见证（root/joints复位、distal窗口清零、其余环境连续）；mount三方一致（quat/平移误差0）；镜像对合恒等0误差、FK经验证8种交换/取反约定中"交换不取反"为几何argmin（残差9.4mm=模型自身左右不对称度）、相对观测对任意默认offset不变；指令重采样960次全部∈{0}∪[0.25,0.35]、站立比例0.173（期望0.20±0.05）、heading转换已禁用、强制0.25指令从观测逐帧回读一致；父子/焊接排除边界28+6对登记 |
| 真接触/静态 | **接触门PASS（10/10）；保持门FAIL（15/16检查）** | 真实物理反例：负例站立（足接触允许、0违规）；HIP真实触地（收腿构型HFE=-1.2/KFE=+2.0足离地~10mm、髋入地21-33mm，`RF_HIP` dist=-0.033m被分类器捕获，旧penalised_contact结构上看不见HIP）；真实自碰（RM/RF HAA对折→`BASE-RF_THIGH`等3对，均通过MuJoCo父子/焊接过滤后仍存在，CPU同模型重建一致）；悬空0接触；NaN单环境注入被有限性检测捕获且reset清除；违规严格限定于受害环境。保持门：见下节 |
| 训练 | **PASS**（8/8门） | 真实CUDA：网络参数全在GPU；32env×24步×5迭代=3,840步（=冻结上限，无扩大）；墙钟31.1s（预算1200s）；每迭代PPO/symmetry/aux损失有限；梯度有限（197次优化器步，grad max 0.63）；参数实际改变（max 0.029）；checkpoint保存+SHA；reload入新runner后确定性前向**逐位一致（err=0.0）**、common_step_counter恢复 |
| 评测 | **PASS**（评测器有效性6/6门） | 64个episode（stand/forward各32，seed 1001–1032）；指令请求=实际逐帧回读一致；安全信号（非足/自碰/NaN/力矩/clip）全记录；4类统计反例（静止却要求前进、中途跌倒、截断、指令被置零）全部被拒绝；滑足诊断明示为FOOT body原点速度。步行结果独立报告：**stand 32/32通过**（5迭代未训练策略的均值输出≈见证PD目标，静态站立满足全部T0-B判据含安全项）；**forward 0/32**（位移≈0、vx MAE≈0.25——如实失败，5迭代不足以学会行走，符合"短训练只证明管线"） |
| 交付 | **PASS** | 独占run、双份身份冻结（改动前/后）、源码快照+SHA、每阶段argv/rc/墙钟、原始episode npz（20MB）+summary JSON、重放器三项全过（身份重算、checkpoint前向、summary从npz重算与原判定全一致+反例复拒）、验收脚本非0退出传播、旧任务gate_a/gate_b回归通过 |

单元测试`tests/t0/test_units.py`（见证冻结一致性、抬升精确性、平地构型零柱、限位解锁恰好18关节、hull裕度数学、镜像offset不变性、评测器含"合成通过样本必须通过"防全拒、PD公式）全部通过。

## GLM原自报保持门失败详情

32环境×5秒零动作PD保持（零动作=目标偏移0，PD仍工作，见证目标为默认offset），T1保持协议15/16检查通过：无自碰/非足触地/NaN、200Hz×1000子步全覆盖、关节RMSE 0.053（<0.20）、倾角0.42°（<5°）、漂移5.9mm（<3cm）、末段根部线速度0.006/角速度0.014（<0.01/0.02）、末态高度误差2.8mm（<3.5cm）、6/6承力足、支撑裕度0.327m（≥3cm）、力矩13.6N·m无饱和。**唯尾段关节速度0.0937>0.05 rad/s**（LF_HFE约5-10Hz、约2mm幅值极限环，与诊断器无关）。

差分证据（保持门evidence内嵌，可复算）：

| 同一PD协议（kp130/kd2/clip80/同一初始态） | 尾段max\|关节速度\| |
|---|---|
| 本环境GPU（mujoco-warp，hfield地形） | 0.0937 rad/s |
| mjlab CPU设备（mujoco-warp，hfield地形） | 0.0931 rad/s |
| 经典MuJoCo + 本环境导出模型（hfield地形） | 0.3188 rad/s |
| 经典MuJoCo + plane平面模型（T1协议原设定） | 0.00008 rad/s |
| 柱场基线El-4090-Lidar中心平台（hfield） | 0.3268 rad/s |

结论：这不是GPU/CPU差异（warp在CPU设备同样振铃），而是**hfield地形离散接触采样**与T1平面协议的差异；柱场基线本身同样存在（且更差），非T0回归。按方案不放宽阈值、不改碰撞过滤，保留反例与差分分析。附带发现：深穿透静态写入会触发MuJoCo hfield碰撞缓冲告警并静默丢接触（接触门已规避并在证据中记录）。

## 审计结论（对应方案§1关注项）

固定0.25m体高奖励与T1验证保持高度0.4599m冲突→T0参数化为见证高度（旧默认0.25保留）；HIP及全部非足碰撞件触地、真实自碰在旧penalised_contact中不可见→200Hz原始接触分类器+nonfoot_any传感器补齐；URDF名义±3 rad限位未标定→T0内存spec关闭+dof_pos_limits奖励移除（旧任务行为不变）；play残留质量/摩擦startup随机化与reset随机化→T0确定性配置替代；0.2m/s deadzone对T0指令域惰性（指令为0或≥0.25）且指令逐帧回读记录；pillar terrain view依赖→平地沿用同一builder/view链路（零柱配置），物理/raycast/监督一致性门通过；`run_baseline_mjlab.sh`的`|| true`与根路径问题→新入口从文件位置定根、严格传播非0退出。既有两组checkpoint（仅model_0/1）登记为来源不明短管线证据，不作成熟策略使用。

## 未验证/缺失项与边界

- 保持门尾段关节速度检查未通过（如上）；其余15项通过。按方案，下一次应先修复T0-A再进T0-B。
- GPU侧mujoco-warp无mjWARNING结构：物理警告由有限性探针+CPU重建侧warning读读覆盖（已记录为能力边界）。
- 确定性协议下32环境episode本质相同（实测跨环境散度3-8mm，为GPU浮点非确定性）；种子1001-1032为记录身份，不引入物理多样性——T0-B引入策略随机性/DR后episode才会真正分化。
- raycast仅含地形网格不含机器人本体（沿既有实现）；真实自遮挡能力缺失已登记，本轮未实现（按方案留给T2/T3）。
- `z₀=0.4599m`仅为T0调试参考（T1见证实际保持高度），不是论文定稿值。
- env_model.mjb（197MB，32世界编译模型）保留在run内供CPU差分复算。

## 改动清单

新增：`src/hexapod_mjlab/tasks/el_4090_lidar/t0/`（witness.py见证冻结+抬升、flat_env_cfg.py隔离确定性平地配置与注册ID `El-4090-Lidar-T0-Flat`、diagnostics.py 200Hz子步接触/力矩记录器）；`scripts/tools/t0/`（t0lib、audit_identity、contract_gate、contact_gate、hold_gate、train_short、eval_episodes、replay_verify、run_acceptance）；`tests/t0/test_units.py`；本文档。
修改（均保持旧默认行为，gate_a/gate_b回归通过）：`mdp/rewards.py`（base_height增target参数，默认None=旧0.25）、`mdp/lidar.py`（get_lidar_pipeline/lidar_prox_obs/lidar_distal_obs增cfg参数，默认None=旧随机化配置）、`config/__init__.py`（追加T0注册）、`config/rl_cfg.py`（追加T0 rl配置工厂）、`docs/README.md`与`docs/research/联合任务方案.md`（状态同步）。
未动：mjlab安装包、`src/hexapod_mjlab/rl/`共享actor/PPO/镜像、EA2、T1源码、机器人XML/STL/URDF（SHA与基线一致）、主目录用户既有未提交修改。

## GLM原建议（未经采用；先修复T0-A）

1. 保持门修复路径二选一：(a) 在T0任务内把平地表示改为plane/网格几何并保持LiDAR/监督一致（需重新过全部契约门，equivalent一致性检查已就绪）；(b) 与Codex商定针对hfield地形的尾段速度阈值并冻结为新协议（属协议变更，需显式记录理由，不得静默放宽）。
2. T0-B长训练配方可直接复用`El-4090-Lidar-T0-Flat`+本管线：建议先32env、观察值/策略损失曲线后定迭代数；站立指令占比0.20与vx∈[0.25,0.35]已被证明可训练、可评测。
3. 前进评测当前0/32主要失败于位移与vx MAE；训练预算扩大后按同一evaluator验收（T0-B通过线：两指令各≥29/32）。
4. hfield深穿透丢接触问题建议在T0-B训练前评估是否影响安全门可靠性（当前200Hz分类器在正常站立/行走穿透量级下工作正常）。
5. 既有两组model_0/1 checkpoint建议保留登记状态，不进入任何评测声明。
