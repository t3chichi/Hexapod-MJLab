# confined_locomotion_2 测试与工程检查

本页列出可复用的测试入口。所有命令从仓库根目录、`hp_mjlab` 环境运行。测试证据输出到新目录；检查会按各自用途创建仿真环境、推进有限步数，部分检查会执行 PPO 更新并改变临时模型参数。它们不是正式学习验收。

Flat v2、Terrain Memory和Tripod共用`TemporalPPOTrainer` / `TemporalPPOConfig`，trainer schema为`temporal_ppo/v2`，Recipe必须带`ppo_semantics`。身份语义包含直接优化std、解析KL（无旧epsilon）、原始样本KL/熵、单次shuffle跨5 epoch复用，以及pre-reset next_value的GAE边界（termination不bootstrap、truncation bootstrap）。完整24步BPTT、镜像损失和Terrain高度辅助保持。Tripod继续使用独立Runtime/Runner与LSTM。历史checkpoint缺少PPO语义身份时拒绝完整续训，不迁移Adam；原sampling下的weights-only Play仍可用。Tripod检查入口见[测试总览](../README.md#flat_2-与-terrain-memory)和[任务说明](../../docs/tasks/confined_locomotion_2/tripod2_low.md)。

2026-10-26独立干净克隆的安装、资源、CPU及有限GPU检查见[部署检查摘要](evidence/20261026_clean_checkout.json)。该检查复用了开发机已有依赖，不代表5090或全新网络求解环境已验收。

## CPU 检查

```bash
python -B tests/confined_locomotion_2/run_checks.py
```

该入口逐个调用目录内 `test_*.py` 的测试函数，不依赖 pytest 或 GPU。当前代码包含182项CPU检查，其中一个入口执行7个地形边界子用例；包括Logger初始化时序与Runner异常资源清理。检查通过仅说明对应接口和数值约束成立，不说明策略学会行走或地形记忆。

`test_scan_batching.py` 和 `test_history_batching.py` 覆盖采集批量化的行顺序、故障与随机流边界、窗口写入原子性及快照所有权。真实同场景参考对照可运行：

```bash
python -B tests/confined_locomotion_2/check_collection_optimization_gpu.py \
  --flat --output "logs/verification/confined_locomotion_2/flat_collection_$(date +%Y%m%d_%H%M%S)"
python -B tests/confined_locomotion_2/check_collection_optimization_gpu.py \
  --output "logs/verification/confined_locomotion_2/terrain_collection_$(date +%Y%m%d_%H%M%S)"
```

该检查使用4个真实环境，不创建优化器；从本地Git历史读取 `2c08bab` 的参考方法，不需要旧任务包或旧checkpoint。浅克隆若没有该提交，需先取得相应历史，不能跳过对照。

## 真实 GPU 工程检查

### Flat Baseline 训练、保存与恢复

`check_gpu.py` 是有限真实 CUDA 集成门：最多 2 个环境、执行训练更新、检查 checkpoint 边界及同一保留 rollout 的优化器重放。下面固定使用 hit10 sparse 身份，并把保存间隔设为 1，以在短运行内检查累计周期保存；生成的权重只是临时工程检查产物。

```bash
python -B tests/confined_locomotion_2/check_gpu.py \
  --output "logs/verification/confined_locomotion_2/check_hit10_$(date +%Y%m%d_%H%M%S)" \
  --seed 20261023 --num-envs 2 --gpu 0 \
  --far-sampling hit_only_10m --raycast-mode sparse \
  --sparse-selection src/hexapod_mjlab/tasks/confined_locomotion_2/assets/fixed_channels/v1_hit10/near_x6_far_x10.npz \
  --save-interval 1
```

该检查每轮采集 24 个控制步并执行 20 次 Adam 更新；保存间隔 1 是测试设置，正式入口默认 50。`all_endpoints` 与 `hit_only_10m` 是不同采样身份，selection 也属于恢复身份，不能互换 checkpoint。

`check_gpu.py` 还核对真实 Recipe 与完整文件中的 `code_provenance` 一致。新完整恢复要求 `hexapod_mjlab.model_physics/v1`；旧文件缺少签名时，不能用同结构或源码/二进制哈希代替物理校验。用上一步新生成的两环境、low_memory hit10模型执行物理身份反例：

```bash
python -B tests/confined_locomotion_2/check_physics_identity_gpu.py \
  --checkpoint "上一步输出目录/training/final_checkpoint_full_000001.pt" \
  --output "logs/verification/confined_locomotion_2/physics_identity_$(date +%Y%m%d_%H%M%S)"
```

该检查不训练：质量10→20 kg必须拒绝，且训练状态/RNG不变；同模型和复制到新路径的环境工厂必须恢复相同边界。它还删除临时文件中的物理签名模拟历史输入，确认完整加载拒绝、weights-only仍可用。生产资产和原checkpoint均不改写。

### 地形几何、标签和辅助损失

几何门只核对真实 Heightfield、射线和地面标签，不训练：

```bash
python -B tests/confined_locomotion_2/check_terrain_geometry_gpu.py \
  --gpu 0 --output "logs/verification/confined_locomotion_2/terrain_geometry_$(date +%Y%m%d_%H%M%S)"
```

序列门核对真实采集、标签配对、重放和重置，不含网络头、损失或优化器。Terrain额外构造真实场地边界截断，检查只重置越界行、双GRU历史清零及重置前Critic bootstrap；加 `--flat` 可核对 Flat 默认采集/重放路径：

```bash
python -B tests/confined_locomotion_2/check_terrain_sequence_gpu.py \
  --gpu 0 --output "logs/verification/confined_locomotion_2/terrain_sequence_$(date +%Y%m%d_%H%M%S)"

python -B tests/confined_locomotion_2/check_terrain_sequence_gpu.py \
  --gpu 0 --flat --output "logs/verification/confined_locomotion_2/flat_sequence_$(date +%Y%m%d_%H%M%S)"
```

辅助损失门在显式传入 `--weight 1.0` 时执行一次 PPO 更新；省略权重只测量损失和梯度，不代表权重已获长训确认：

```bash
python -B tests/confined_locomotion_2/check_terrain_auxiliary_gpu.py \
  --weight 1.0 --output "logs/verification/confined_locomotion_2/terrain_auxiliary_$(date +%Y%m%d_%H%M%S)"
```

### checkpoint、Play、扫描与界面

下面Terrain checkpoint检查要求含新物理签名的1或2环境、low_memory、near×6/far×10 hit10 sparse模型。检查脚本使用默认保存间隔50，但保存间隔不参与兼容门控。其他采样/环境配方不要直接套用。可用[快速开始](../../src/hexapod_mjlab/tasks/confined_locomotion_2/快速开始.md)的训练命令从零生成；本仓库不要求克隆 ignored 日志或模型。

```bash
TERRAIN_CKPT="填写本机生成的 Terrain Memory checkpoint 路径"
SELECTION="src/hexapod_mjlab/tasks/confined_locomotion_2/assets/fixed_channels/v1_hit10/near_x6_far_x10.npz"

python -B tests/confined_locomotion_2/check_terrain_checkpoint_gpu.py \
  --checkpoint "$TERRAIN_CKPT" \
  --output "logs/verification/confined_locomotion_2/terrain_checkpoint_$(date +%Y%m%d_%H%M%S)"

python -B tests/confined_locomotion_2/check_terrain_play_gpu.py \
  --checkpoint "$TERRAIN_CKPT" --port 8097 \
  --output "logs/verification/confined_locomotion_2/terrain_play_$(date +%Y%m%d_%H%M%S)"
```

Terrain Play 检查会绑定本地 Viser/HTTP 端口，并逐项比较延迟事件快照与原动作前捕获（含扫描槽替换、标签与姿态身份）。运行前确认请求端口空闲；不要用检查脚本探测或覆盖已有服务。

原生窗口与扫描重建可用已有有限检查，`--native`需要图形桌面。下面使用同sampling/catalog内显式允许的weights-only扫描override，从hit10 sparse模型检查full/sparse切换：

```bash
FLAT_CKPT="填写本机 Flat hit10 checkpoint 路径"
python -B tests/confined_locomotion_2/check_acquisition_switch_gpu.py \
  --checkpoint "$FLAT_CKPT" --seed 20261025 --native \
  --far-sampling hit_only_10m --allow-scan-override --port 8196 \
  --output "logs/verification/confined_locomotion_2/native_switch_$(date +%Y%m%d_%H%M%S)"
```

该检查核对六次重建中同一CPU展示model/data和窗口仍存活、旧GPU对象被回收；不代表实时流畅度或完整Checkpoint恢复验收。日常 `play_*` 使用原生窗口，`debug_*`使用事件驱动网页，见[快速开始](../../src/hexapod_mjlab/tasks/confined_locomotion_2/快速开始.md)。

扫描器的 GPU/reference 对照与 fixed sparse/full 对照都不训练：

```bash
python -B tests/confined_locomotion_2/check_sampling_gpu.py \
  --output "logs/verification/confined_locomotion_2/sampling_$(date +%Y%m%d_%H%M%S)" \
  --seed 20261023 --num-envs 2 --gpu 0

python -B tests/confined_locomotion_2/check_fixed_sparse_gpu.py \
  --selection "$SELECTION" --far-sampling hit_only_10m \
  --output "logs/verification/confined_locomotion_2/fixed_sparse_$(date +%Y%m%d_%H%M%S)" \
  --seed 20261023 --num-envs 2 --gpu 0
```

性能 benchmark `benchmark_fixed_training_gpu.py` 需要同一源码身份下配对的 full/sparse 完整 checkpoint、selection 和新输出目录；它会各自恢复检查点，测量 24 步采集与 20 次更新，不应拿不同配置或旧源码权重作比较。先用 `--help` 查看必需参数，再按实际路径传入。所有 `--output` 都必须是尚不存在的目录；上例使用带时间戳的目录名。

## wheel 固定资源检查

`check_package.py` 在隔离 wheel 中校验固定通道 JSON/NPZ 随包发布，并从 wheel 自身加载 catalog 与 selection。它不依赖 editable checkout 的导入路径：

```bash
python -m pip wheel --no-deps . --wheel-dir build/wheel
python -B tests/confined_locomotion_2/check_package.py \
  build/wheel/hexapod_mjlab-0.1.0-py3-none-any.whl \
  --output logs/verification/confined_locomotion_2/package_check.json
```

wheel 构建后请核对文件名与项目版本一致。预期资源集合为 45 个 JSON/NPZ 和 21 个 selection NPZ；hit10 near×6/far×10 选集包含 2,816 条候选射线。检查结果是针对所构建 wheel 的记录。

## 旧实验检查

`check_viewer_gpu.py`固定采用Flat legacy `all_endpoints`，不支持当前hit10模型，不能直接套用上方Terrain checkpoint。它会使用8093/8094两个显示端口。`check_far_hit_gpu.py`、`check_far_shorttrain_gpu.py`和固定射线 benchmark 属于各自版本的专项工程配方；部分会更新参数，部分要求与源码严格匹配的 checkpoint。先读脚本头部说明和 `--help`，再创建专属新输出目录。它们的历史结果不能替代当前 task 的训练效果验收。
