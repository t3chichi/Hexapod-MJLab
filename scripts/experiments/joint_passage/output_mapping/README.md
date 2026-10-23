# output_mapping — 形态输出映射离线验证实验

验证 `mappings.py` 中的平滑正值映射（Softplus / ELU+1 / 有理分段，各 r∈{0.10,0.25,0.50,1.00}）
在已有形态数值附近的梯度、扰动响应与隔离误差恢复。只评估映射本身：
不构建网络主干、不启动 mjlab/仿真/PPO、不导入任务注册器；EA2 配置与动作纯函数
通过 AST 从活源码（`parameters.py`/`condition.py`/`math_utils.py`）提取核对，不手填常数。

## 命令（仓库根目录）

```bash
# 完整实验（唯一配方：12 正值候选 + linear 对照，各 2 学习率；
# 每组 180 案例 = 36 主目标 × 5 起点，1000 次更新 + step0 共 1001 状态，float32）
conda run --no-capture-output -n hp_mjlab python \
  scripts/experiments/joint_passage/output_mapping/run.py \
  --out-dir logs/joint_passage/output_mapping/<新run_name>

# 短实现自检（仍需新目录；experiment_complete=false，self_check_only=true）
conda run --no-capture-output -n hp_mjlab python \
  scripts/experiments/joint_passage/output_mapping/run.py \
  --out-dir logs/joint_passage/output_mapping/<新run_name> --self-check

# 独立只读复核（复算损失/映射/准入，不重优化、不重写 run）
conda run --no-capture-output -n hp_mjlab python \
  scripts/experiments/joint_passage/output_mapping/verify.py \
  --run-dir logs/joint_passage/output_mapping/<实际run_name>
# 可选：--report-out <新文件路径> 将复核报告写出到显式新文件
```

依赖：标准库、NumPy、PyTorch（CPU，单线程，seed=0）、Matplotlib（Agg）。
退出码：0=成功（候选可以全部不合格，只要实验完整并如实报告）；1=正确性失败/运行错误；
2=输出目录已存在。失败目录保留（含 `error.json`），不覆盖既有 run。

## 文件说明

- `mappings.py`：隔离实验的纯映射、逆映射及导数；EA2仅作诊断，不修改生产映射。
- `run.py`：唯一实验入口。产物写入新 `--out-dir`：
  `config.json`（完整配方/门限/CLI）、`samples.json`（主36+扩展30+压力8样本与分母）、
  `curves.npz`（[-20,20] 4001 点 + 极端点扫描）、`responses.npz`（逆坐标/G/双向扰动/float32 差分）、
  `recovery.npz`（完整 180×1001 轨迹 u/p/loss/grad_norm，逐状态对齐）、`metrics.json`、
  `checks.json`、`array_metadata.json`（逐数组shape/dtype/轴/单位与非有限计数）、`provenance.json`（代码/输入 SHA256、分支与 dirty 状态、真实时刻）、
  `verification.json`（由 verify 逻辑生成）、`manifest.json`（不含自哈希）、`run.log`、`figures/`。
  NPZ 仅普通数值数组（读取 `allow_pickle=False`），逐键说明以 `array_metadata.json` 为准；扫描包含float32/float64极端输出与梯度，EA2包含真实梯度及三档双向扰动。
- `verify.py`：独立NumPy闭式公式复算，不调用实验映射或重做优化；默认只读。可打印stdout，或写原run之外的新报告文件。
- [实验报告](../../../../docs/tasks/confined_locomotion/mapping/实验报告.md)：9/12候选通过；首版已选Softplus、r=1，新任务尚未实现或训练。

## 边界

正确性与"本配方可用"严格区分：Sigmoid/无约束线性/EA2 仅作诊断对照，不进入合格集合。
压力起点（c∈{0.1,5}）与主恢复（c∈{0.5,1,2}）分母分开。恢复通过要求单一固定学习率
通过全部 108 个主案例（最后 100 步逐坐标 |p−T|/p0 ≤ 1e-3）。允许多候选合格或全部不合格；
配方、门限、样本与预算不可为结果调整。

单次入口设置900秒和512MiB上限。失败记录保留；源码或输入改变后应写新run，不能用当前源码身份核验旧版本run。学习率0.01/0.05只用于隔离变量恢复，不能据此确定正式任务优化器。

冻结配方已归入docs/tasks/confined_locomotion/mapping/evidence/，不作为当前开发指令。20261004_accepted的源码身份恢复与只读复核方式见[实验报告复现节](../../../../docs/tasks/confined_locomotion/mapping/实验报告.md#6-复现)。
