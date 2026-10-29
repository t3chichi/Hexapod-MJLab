# 项目运行环境

- 项目已有的 Conda 环境为 `hp_mjlab`。所有项目 Python 脚本、测试、训练与 Play 均使用该环境；执行前核对 `sys.executable`，不要仅凭终端提示符判断环境。
- 可使用 `conda run --no-capture-output -n hp_mjlab python ...`。本机解释器为 `/home/t3chichi/miniconda3/envs/hp_mjlab/bin/python`；其他机器按实际 Conda 安装位置定位同名环境。
- 未经用户明确授权，不创建 Conda、uv、venv 等替代环境，不执行会自动创建项目环境的 `uv run`、`uv sync` 或 `uv venv`，也不升级或重装已有训练依赖。
- `hexapod-mjlab` 是 Python 包名，不是本项目约定的运行环境名。
- 本机沙箱外可使用 GPU。若沙箱内不可见 GPU，应按工具权限流程运行必要的有限 GPU 检查；不要因此重建环境或替换 CUDA/PyTorch。长期训练仍需用户授权。

# 入口与验证

- 当前任务与操作入口见 [文档导航](docs/README.md)；测试命令与前置条件见 [测试说明](tests/README.md)。优先使用项目现有检查入口，不因缺少 pytest 等可选工具而安装依赖或另建环境。
- 按改动范围运行必要的有限验证，如实报告设备、配置和覆盖范围。CPU Mock 不等于真实 GPU 验收，短时训练闭环不等于收敛，恢复边界一致不等于未来仿真轨迹逐位一致。

# 研究资产与训练语义

- 保留已有日志、Checkpoint、`.checkpoint_sources/` 和冻结证据；清理、覆盖或改写这些资产须在用户授权范围内。检查输出使用新的目录，临时验证可放在 `/tmp`，不要覆盖既有实验结果。
- 目录整理和基础设施复用应保持任务的训练语义；PPO、采样或其他训练配方变更须与纯架构移动区分说明，并同步 Checkpoint 兼容规则。不得将不兼容的旧优化器状态静默转换成“等价续训”。
