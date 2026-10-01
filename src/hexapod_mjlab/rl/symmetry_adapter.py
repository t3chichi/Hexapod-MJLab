"""EL4090 LiDAR 左右对称函数的 rsl-rl 5.x ``Symmetry`` 适配层。

旧版调用约定（``Lidar_legged_gym/rsl_rl`` PPO）::

    obs_aug, actions_aug = func(obs=obs_batch, actions=actions_batch,
                                env=env, obs_type="policy")
    # 平铺张量进出，返回 [batch*2, dim]（原始在前，镜像在后）

5.x 安装版 ``rsl_rl.extensions.Symmetry`` 调用约定::

    obs_aug, actions_aug = func(env=env, obs=TensorDict, actions=Tensor)
    # obs 为 TensorDict（各观测组），返回 (TensorDict, Tensor)，
    #   每组第一半 = 原始，第二半 = 镜像（[2B, ...]）

本模块提供：

* :func:`make_el_4090_symmetry_func` —— 生成符合 5.x 约定的镜像函数：
  - ``policy`` 组走旧版完整镜像（proprio 符号翻转 + 左右腿交换 + 点云
    Y 镜像 + 球坐标键重排序，逐位复用
    ``hexapod_mjlab.rl.el_4090_lidar_symmetry.get_el_4090_lidar_xsym_obs_act``）；
  - 特权高度图组（默认 ``height_grid``，17×11 网格）走旧版
    ``obs_type="auxiliary"`` 的 Y 翻转路径；
  - 动作走旧版 ``_process_actions``（左右腿组交换，无符号翻转）。
* :func:`build_symmetry_cfg` —— 生成可直接放进
  ``cfg["algorithm"]["symmetry_cfg"]`` 的配置字典（供 ``LidarPPO5x``）。

模型侧要求：actor/critic 必须保持 ``is_recurrent = False``
（``LidarPDActor5x`` 满足——逐帧零初始化 GRU 在 RL 层面前馈），
与安装版 ``Symmetry`` 的限制一致。
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from hexapod_mjlab.rl.el_4090_lidar_symmetry import get_el_4090_lidar_xsym_obs_act


def make_el_4090_symmetry_func(
    *,
    proprio_dim: int = 66,
    proximal_points: int = 256,
    distal_history_points: int = 640,
    num_dof: int = 18,
    height_grid_x_count: int = 17,
    height_grid_y_count: int = 11,
    sensor_quat: torch.Tensor | None = None,
    sensor_trans: torch.Tensor | None = None,
    policy_key: str = "policy",
    aux_keys: tuple[str, ...] = ("height_grid",),
    strict: bool = True,
):
    """构造 5.x ``Symmetry`` 约定的镜像函数 ``f(env, obs, actions) -> (obs, actions)``。

    Args:
        proprio_dim: 本体感知前缀维度（EL4090: 12 + 3*18 = 66）。
        proximal_points: 近端点数（256）。
        distal_history_points: 远端历史总点数（10 帧 × 64 = 640）。
        num_dof: 自由度（18）；动作与 proprio 的 dof 段按左右半交换。
        height_grid_x_count, height_grid_y_count: 特权高度图网格（17×11=187）。
        sensor_quat: 传感器偏置四元数 ``[1,4]``；None → 单位。
        sensor_trans: 传感器平移 ``[1,3]``；None → 零。
        policy_key: 主观测组名。
        aux_keys: 走 auxiliary Y 翻转路径的组名集合。
        strict: 未知观测组时抛错（False 则原样复制，仅记录告警语义）。

    Returns:
        可直接作为 ``Symmetry(data_augmentation_func=...)`` 的可调用对象；
        ``obs``/``actions`` 任一为 ``None`` 时对应返回 ``None``
        （``Symmetry.compute_loss`` 会以 ``obs=None, actions=mean`` 调用）。
    """

    def symmetry_func(
        env=None,
        obs: TensorDict | None = None,
        actions: torch.Tensor | None = None,
    ):
        obs_aug: TensorDict | None = None
        if obs is not None:
            if not isinstance(obs, TensorDict):
                raise TypeError(
                    f"el_4090 symmetry adapter expects a TensorDict obs (5.x convention), "
                    f"got {type(obs)}."
                )
            batch_size = obs.batch_size[0]
            aug_entries: dict[str, torch.Tensor] = {}
            for key, value in obs.items():
                if key == policy_key:
                    # 完整镜像：proprio 翻转/交换 + 点云 Y 镜像 + 球键重排序
                    policy_aug, _ = get_el_4090_lidar_xsym_obs_act(
                        obs=value,
                        actions=None,
                        env=env,
                        obs_type="policy",
                        proprio_dim=proprio_dim,
                        proximal_points=proximal_points,
                        distal_history_points=distal_history_points,
                        num_dof=num_dof,
                        sensor_quat=sensor_quat,
                        sensor_trans=sensor_trans,
                    )
                    aug_entries[key] = policy_aug
                elif key in aux_keys:
                    # 特权高度图：网格 Y 翻转（旧版 auxiliary 路径）
                    aux_aug, _ = get_el_4090_lidar_xsym_obs_act(
                        obs=value,
                        actions=None,
                        env=env,
                        obs_type="auxiliary",
                        height_grid_x_count=height_grid_x_count,
                        height_grid_y_count=height_grid_y_count,
                    )
                    aug_entries[key] = aux_aug
                else:
                    if strict:
                        raise ValueError(
                            f"el_4090 symmetry adapter has no mirror rule for obs group "
                            f"'{key}' (policy_key='{policy_key}', aux_keys={list(aux_keys)})."
                        )
                    aug_entries[key] = torch.cat([value, value], dim=0)
            obs_aug = TensorDict(aug_entries, batch_size=[2 * batch_size], device=obs.device)

        actions_aug: torch.Tensor | None = None
        if actions is not None:
            _, actions_aug = get_el_4090_lidar_xsym_obs_act(
                obs=None,
                actions=actions,
                env=env,
                num_dof=num_dof,
            )

        return obs_aug, actions_aug

    symmetry_func.__name__ = "el_4090_lidar_xsym_5x"
    return symmetry_func


def build_symmetry_cfg(
    *,
    use_data_augmentation: bool = True,
    use_mirror_loss: bool = False,
    mirror_loss_coeff: float = 1.0,
    env=None,
    **adapter_kwargs,
) -> dict:
    """组装 ``cfg["algorithm"]["symmetry_cfg"]`` 字典。

    ``LidarPPO5x.construct_algorithm`` / ``rsl_rl`` 的
    ``resolve_symmetry_config`` 会补上 ``env``；此处也允许显式传入。
    其余 kwargs 原样传给 :func:`make_el_4090_symmetry_func`。
    """
    cfg = {
        "data_augmentation_func": make_el_4090_symmetry_func(**adapter_kwargs),
        "use_data_augmentation": use_data_augmentation,
        "use_mirror_loss": use_mirror_loss,
        "mirror_loss_coeff": mirror_loss_coeff,
    }
    if env is not None:
        cfg["env"] = env
    return cfg
