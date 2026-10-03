# EA2 oracle实验工具

这里维护离线比较与报告生成，不放入日常测试入口。当前算法及科学结论见[选型与验收](../../../docs/tasks/envelope_adaptive_2/oracle/选型与验收.md)；解析反例统一位于`tests/ea2/test_oracle.py`，独立完整多边形参考实现仍在`tests/ea2/whole_polygon_reference.py`。

| 工具 | 功能 |
|---|---|
| `static_oracle_compare.py` | 全部冻结位姿的旧/新、同间隙静态对照 |
| `static_oracle_diagnostics.py` | 已保存结果的分母、源码、批量与密集边界审计 |
| `static_oracle_report.py` | 静态结果统计与图表 |
| `oracle_refinement_compare.py` | 各因果/平滑候选逐帧比较，记录配置和源码 |
| `oracle_refinement_audit.py` | 冻结候选结果的独立指标复算 |
| `oracle_refinement_heldout.py` | 固定候选在额外地图上的验证 |
| `oracle_selection_report.py` | 冻结四图选型结论与图表 |
| `teacher_frozen_parity.py --tag <新名称>` | 当前生产teacher对全部选定历史帧重放与几何复核 |

在仓库根目录、`hp_mjlab`环境中运行，例如：

```bash
python scripts/experiments/ea2/static_oracle_compare.py --help
python scripts/experiments/ea2/oracle_refinement_compare.py --help
python scripts/experiments/ea2/teacher_frozen_parity.py --tag my-new-verification
```

工具依赖文档中指定的历史语料和冻结对照文件，普通源码checkout没有这些被Git忽略的产物。数据采集、网络训练/评估/导出使用任务自身`sl.scripts`入口；不再维护第二套固定日期的发布训练入口。

2026-10-03原脚本及其真实命令仍随科学证据保存；2026-10-04只整理入口和导入路径，不重跑算法选型、不改历史指标。需要历史严格复现时使用归档源码；新比较应使用新tag/目录，不能将整理后的源码身份回填到旧报告。

静态比较、诊断、静态报告和选型报告必须显式给出`--out-dir <新目录>`；报告/诊断用`--input-dir`只读历史输入。目录已有即拒绝，选型图也输出到新目录。heldout必须显式使用新tag，不能覆写既有来源记录。
