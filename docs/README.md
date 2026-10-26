# 文档导航

当前训练与5090部署使用[confined_locomotion_2](tasks/confined_locomotion_2/README.md)，包含Flat Baseline与Terrain Memory。其他研究任务继续独立保留；历史结果按其对应版本理解。

## 当前任务的阅读顺序

| 需要做什么 | 入口 |
|---|---|
| 在新机器克隆并安装 | [当前开发分支部署](tasks/confined_locomotion_2/training/部署.md) |
| 训练、续训和Play | [快速开始](../src/hexapod_mjlab/tasks/confined_locomotion_2/快速开始.md) |
| 理解任务配置、架构和信息边界 | [任务说明](tasks/confined_locomotion_2/README.md) |
| 读曲线和管理checkpoint | [监控与保存](tasks/confined_locomotion_2/training/监控与保存.md) |
| 运行检查 | [任务测试](../tests/confined_locomotion_2/README.md) |
| 调整网格、生成选集或测量吞吐 | [实验工具](../scripts/experiments/confined_locomotion_2/README.md) |

原任务的连接约定见[研究架构](research/训练架构对齐.md)，原数值配方见[平地配置](tasks/confined_locomotion/平地学习与基准测量.md)，原任务能力及历史门见[功能状态](tasks/confined_locomotion/training/功能状态.md)与[工程验收](tasks/confined_locomotion/training/工程验收与复现.md)。这些历史材料不替代当前 `_2` 的配置和使用入口。

## 按主题查证

| 主题 | 入口 |
|---|---|
| 开发闭环与接口 | [当前接口与训练工程门](tasks/confined_locomotion/framework/接口与约定.md)、[当前验收与复现](tasks/confined_locomotion/training/工程验收与复现.md)；必要历史出处见[精简证据索引](../logs/maintenance/20261012_confined_archive_retirement/README.md)，完整旧归档已删除 |
| 纯计算框架 | [当前接口](tasks/confined_locomotion/framework/接口与约定.md)；2026-10-05历史分母和身份按[精简证据索引](../logs/maintenance/20261012_confined_archive_retirement/README.md)的必要报告/原身份查证 |
| 雷达 | [配置与边界](tasks/confined_locomotion/lidar_setup/配置与边界.md)；静态实验与失败分母见精简保留报告和对应run身份，不再提供完整旧run恢复 |
| 映射、动作与旧任务参考 | [输出与模型约定](tasks/confined_locomotion/mapping/输出与模型约定.md)、[设计依据](tasks/confined_locomotion/reference/设计依据.md) |
| 机器人形态T1 | [测量与验收](research/morphology/形态可达性测量.md)、[实验记录](research/morphology/实验记录.md) |
| 旧任务与T0-A | [使用与验收](tasks/el_4090_lidar/T0-A使用与验收.md) |
| EA2监督流程与二维oracle | [使用](tasks/envelope_adaptive_2/使用说明.md)、[迁移验收](tasks/envelope_adaptive_2/迁移验收.md)、[oracle选型](tasks/envelope_adaptive_2/oracle/选型与验收.md)、[实验记录](tasks/envelope_adaptive_2/oracle/实验记录.md) |

## 分类与工程规范

- `research/`：跨任务架构、研究路线、阶段成果与形态测量。
- `tasks/<任务>/`：任务要求和配置；功能主题保存对应接口、实验与验收。
- `tasks/confined_locomotion/training/`：当前功能状态、工程验收与复现、使用说明。
- `tasks/confined_locomotion/{framework,lidar_setup,mapping,reference}/`：当前接口/传感/输出/设计依据权威页；独有历史结论已迁移，必要原报告在精简证据索引，未保留的详细历史原件不能恢复。
- `tasks/confined_locomotion/development/实施计划.md`：源码历史出处索引；已结束批次结论在当前主题页，不再提供完整旧归档恢复。
- `development/`：共用[文档维护](development/文档维护.md)、[GLM协作](development/GLM协作说明.md)；其 `history/` 保存[平台迁移](development/history/平台迁移.md)和[维护记录](development/history/维护记录.md)。

[测试说明](../tests/README.md)维护测试入口；实验脚本旁的README维护该工具用法。主题 `evidence/` 和 `figures/` 保存冻结身份、配方与图表，`logs/` 保存原始数组、checkpoint、失败、完整报告和源码快照。冻结路径按当时身份理解，不能回填最新状态。

本次删除/合并/移动清单、整理前原文及保护核对见[2026-10-08整理记录](../logs/maintenance/document-cleanup-20261008/report.md)。此前[2026-10-06整理记录](../logs/maintenance/document-cleanup-20261006/)继续保留。

P0–P7建设散件的后续精简按[记录保留与清理](development/记录保留与清理.md)执行；保留阶段最终说明、权威约定及必要证据，逐文件清单区分原路径保留、压缩归档与直接删除。

本轮文档主题迁移、旧页覆盖与删除候选见[迁移表](../logs/maintenance/20261011_confined_final_cleanup/execution/docs_migration.md)；必要原报告、源码身份及归档删除回执见[精简证据索引](../logs/maintenance/20261012_confined_archive_retirement/README.md)。历史镜像和工程交付边界统一见[工程验收与复现](tasks/confined_locomotion/training/工程验收与复现.md)及[设计依据](tasks/confined_locomotion/reference/设计依据.md)。
