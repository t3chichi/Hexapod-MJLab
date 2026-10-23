"""兼容垫片：正典实现已迁移至 ``hexapod_mjlab.rl.confined_locomotion.network``（M2）。

正典类名为 ``ConfinedLocomotionNetwork``；``JointPassageNetwork`` 在正典模块
内是同一类对象的兼容别名。本模块整体替换为正典模块对象（sys.modules 别名）：
``joint_passage.network.JointPassageNetwork is
confined_locomotion.network.ConfinedLocomotionNetwork`` 恒真（类身份不变），
历史测试行为不变；数学语义零改动。
"""

import sys

from hexapod_mjlab.rl.confined_locomotion import network as _canonical

sys.modules[__name__] = _canonical
