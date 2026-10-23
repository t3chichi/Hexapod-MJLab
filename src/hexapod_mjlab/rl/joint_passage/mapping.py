"""兼容垫片：正典实现已迁移至 ``hexapod_mjlab.rl.confined_locomotion.mapping``（M2）。

本模块整体替换为正典模块对象（sys.modules 别名）：所有公有/私有导入身份
与正典实现完全同一（EnvelopeSoftplusMapping 及常量不变），历史测试行为
不变；数学语义零改动。
"""

import sys

from hexapod_mjlab.rl.confined_locomotion import mapping as _canonical

sys.modules[__name__] = _canonical
