"""兼容垫片：正典实现已迁移至 ``hexapod_mjlab.rl.confined_locomotion.integration``（M2）。

本模块整体替换为正典模块对象（sys.modules 别名）：所有公有/私有导入身份
与正典实现完全同一（prepare_control_inputs/execute_control_step/
bind_offline_labels 不变；execute_control_step 接受正典类
ConfinedLocomotionNetwork，其兼容别名 JointPassageNetwork 为同一类对象），
历史测试行为不变；数学语义零改动。
"""

import sys

from hexapod_mjlab.rl.confined_locomotion import integration as _canonical

sys.modules[__name__] = _canonical
