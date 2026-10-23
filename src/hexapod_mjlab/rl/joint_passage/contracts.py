"""兼容垫片：正典实现已迁移至 ``hexapod_mjlab.rl.confined_locomotion.contracts``（M2）。

本模块整体替换为正典模块对象（sys.modules 别名）：所有公有/私有导入身份
（类、函数、常量、模块属性）与正典实现完全同一，历史测试与脚本的
``from hexapod_mjlab.rl.joint_passage.contracts import X`` 行为不变；
数学语义零改动。历史证据/冻结配方中的旧路径继续有效，实验产物不迁移。
"""

import sys

from hexapod_mjlab.rl.confined_locomotion import contracts as _canonical

sys.modules[__name__] = _canonical
