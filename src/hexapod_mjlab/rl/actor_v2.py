"""LidarPD 5.x 原生 Actor/Critic（rsl-rl-lib 5.4.2 ``MLPModel`` 协议）。

本模块把旧版 ``hexapod_mjlab.rl.lidar_pd_actor_critic.LidarPDActorCritic``
（单策略 ActorCritic）的数学 **逐行重写** 为 rsl-rl 5.x 的 actor/critic
分离模型协议，语义完全保持：

* 观测布局（与 LidarWrapper 输出一致；典型任务 66+256×3+640×3 = 2754 维，
  实际维度由构造参数决定——任务侧必须显式传参，默认值非任务布局）::

      [0 : proprio_dim)                                本体感知
      [proprio_dim : proprio_dim + prox*3)             近端点云 (256 点)
      [proprio_dim + prox*3 : end)                     远端点云 (10 帧 x 64 点)

* 近端 GRU(3→187)、远端 GRU(3→64)，**每帧零初始化**（h0=0，seq=点序列），
  训练期按 batch 分块（chunk=128）；远端梯度检查点默认开启。
* actor 头 / critic 头均为 4 层 MLP（默认 1024-512-256-128，ELU），
  输入 latent = cat(proprio, prox_feat, dist_feat)。
* critic 与 actor **共享** 双 GRU 感知（旧版单策略里 critic 消费
  ``_cached_actor_latent``）；5.x 拆分后通过持有同一批 ``nn.GRU`` 模块
  对象实现权重共享（优化器参数需去重，见 ``ppo_lidar.LidarPPO5x``）。
* height_supervisor: ``Linear(187, 187)``，读取 act() 缓存的近端特征做
  高度图 MSE 辅助监督（目标来自特权观测组 ``height_grid``）。
* 标准差参数化：``GaussianDistribution(std_type="scalar")`` →
  ``distribution.std_param``（对应旧版 ``self.std``）。

确定性推理路径：``model(obs)``（``stochastic_output=False``）返回分布均值，
等价旧版 ``act_inference``。

与旧版一致的前向细节（重要）：
* 前向 **不做** 球坐标重排序 —— 点云已由 ``LidarWrapper`` 排序；
  ``_sort_by_spherical`` 仅作为工具方法保留（与旧版相同，未被前向调用）。
* ``cache=True`` 只发生在随机（act）路径；确定性路径 ``cache=False``，
  与旧版 ``act_inference`` 一致，保证 PPO.update 中 mirror-loss 前向
  不会覆盖 aux-loss 所需的 ``_cached_proximal_feature``。
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.modules import GaussianDistribution
from rsl_rl.utils import resolve_nn_activation, unpad_trajectories


# ───────────────────────── 四元数工具（与旧版逐字一致） ─────────────────────────
def _euler_to_quat(roll: float, pitch: float, yaw: float) -> torch.Tensor:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return torch.tensor(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ]
    )


def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return q * torch.tensor([-1, -1, -1, 1], device=q.device)


def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_vec = q[:3]
    q_scalar = q[3]
    q_vec_exp = q_vec.expand(*v.shape[:-1], 3)
    t = 2.0 * torch.cross(q_vec_exp, v, dim=-1)
    return v + q_scalar * t + torch.cross(q_vec_exp, t, dim=-1)


class LidarPDActor5x(nn.Module):
    """rsl-rl 5.x MLPModel 协议的 LiDAR-PD actor。

    协议面（对照 ``rsl_rl/models/mlp_model.py``）：``forward``、``get_latent``、
    ``reset``、``get_hidden_state``、``detach_hidden_state``、
    ``update_normalization``、``output_mean/std/entropy/distribution_params``、
    ``get_output_log_prob``、``get_kl_divergence``、``as_jit``、``as_onnx``。
    """

    is_recurrent: bool = False
    """5.x Symmetry 扩展要求非循环策略；本模型零初始化逐帧 GRU 在 RL 层面
    等价前馈（与旧版 ``is_recurrent = False`` 相同）。"""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        # 5.x 标准参数
        hidden_dims: tuple[int, ...] | list[int] = (1024, 512, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        # 旧版 LiDAR-PD 参数（默认值与旧版一致）
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        proximal_points: int = 256,
        distal_history_length: int = 10,
        distal_points: int = 128,
        proximal_feature_dim: int = 187,
        distal_feature_dim: int = 64,
        proprio_obs_dim: int = 48,
        privileged_height_dim: int = 187,
        sensor_offset_rpy: list[float] | None = None,
        sensor_offset_pos: list[float] | None = None,
        gradient_checkpointing_proximal: bool = False,
        gradient_checkpointing_distal: bool = True,
        **kwargs,
    ):
        if kwargs:
            print(
                "LidarPDActor5x.__init__ got unexpected arguments, which will be ignored: "
                + str(list(kwargs.keys()))
            )
        super().__init__()

        # -- 5.x 观测组解析（等价旧版：policy 组展平成一条 2754 维向量） --
        self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)

        # -- 观测归一化（旧版无归一化 → 默认 Identity，语义一致） --
        self.obs_normalization = obs_normalization
        if obs_normalization:
            raise NotImplementedError(
                "LidarPDActor5x preserves the legacy no-normalization semantics; "
                "obs_normalization must be False."
            )
        self.obs_normalizer = torch.nn.Identity()

        # -- LiDAR 结构参数（与旧版 LidarPDActorCritic 完全一致） --
        self.proximal_points = int(proximal_points)
        self.distal_history_length = int(distal_history_length)
        self.distal_points = int(distal_points)
        self.proximal_feature_dim = int(proximal_feature_dim)
        self.distal_feature_dim = int(distal_feature_dim)
        self.proprio_obs_dim = int(proprio_obs_dim)
        self.privileged_height_dim = int(privileged_height_dim)
        self.num_actions = output_dim
        self.gradient_checkpointing_proximal = gradient_checkpointing_proximal
        self.gradient_checkpointing_distal = gradient_checkpointing_distal

        expected_obs = (
            self.proprio_obs_dim
            + self.proximal_points * 3
            + self.distal_history_length * self.distal_points * 3
        )
        if self.obs_dim != expected_obs:
            raise ValueError(
                f"LidarPDActor5x expects {expected_obs} actor obs dims "
                f"(proprio={self.proprio_obs_dim} + proximal={self.proximal_points}*3 "
                f"+ distal={self.distal_history_length}*{self.distal_points}*3), "
                f"got {self.obs_dim}"
            )

        self._sensor_conj: torch.Tensor | None = None
        if sensor_offset_rpy is not None and any(v != 0.0 for v in sensor_offset_rpy):
            self._sensor_conj = _quat_conjugate(_euler_to_quat(*sensor_offset_rpy))
        if sensor_offset_pos is not None and any(v != 0.0 for v in sensor_offset_pos):
            sensor_t = torch.tensor(sensor_offset_pos, dtype=torch.float32)
        else:
            sensor_t = torch.zeros(3, dtype=torch.float32)
        self.register_buffer("_sensor_translation", sensor_t, persistent=False)

        act_fn = resolve_nn_activation(activation)

        # -- 双 GRU 编码器（无 PointNet，原始 xyz 输入；h0 默认零 → 每帧零初始化） --
        self.proximal_gru = nn.GRU(
            input_size=3,
            hidden_size=self.proximal_feature_dim,
            batch_first=True,
        )
        self.distal_gru = nn.GRU(
            input_size=3,
            hidden_size=self.distal_feature_dim,
            batch_first=True,
        )

        # -- 高度监督头（辅助 MSE，读取 act() 缓存的近端特征） --
        self.height_supervisor = nn.Linear(self.proximal_feature_dim, self.privileged_height_dim)

        # -- 分布（5.x GaussianDistribution；scalar std ↔ 旧版 self.std） --
        if distribution_cfg is None:
            distribution_cfg = {}
        distribution_cfg = dict(distribution_cfg)
        distribution_cfg.setdefault("class_name", GaussianDistribution)
        distribution_cfg.setdefault("init_std", init_noise_std)
        distribution_cfg.setdefault("std_type", noise_std_type)
        dist_class = distribution_cfg.pop("class_name")
        if dist_class is not GaussianDistribution:
            raise ValueError(
                "LidarPDActor5x preserves the legacy Normal(mean, scalar-std) policy; "
                f"got distribution class {dist_class}."
            )
        self.distribution: GaussianDistribution = dist_class(output_dim, **distribution_cfg)

        # -- actor 头：5.x 惯例命名 self.mlp（结构/默认初始化与旧版 self.actor 相同） --
        actor_input_dim = self.proprio_obs_dim + self.proximal_feature_dim + self.distal_feature_dim
        self.mlp = self._build_mlp(actor_input_dim, list(hidden_dims), output_dim, act_fn)

        Normal.set_default_validate_args(False)

        # -- act() 缓存（供共享感知 critic 与 aux 高度监督使用，语义同旧版） --
        self._cached_proximal_feature: torch.Tensor | None = None
        self._cached_actor_latent: torch.Tensor | None = None

    # ───────────────────────── 构建工具 ─────────────────────────
    @staticmethod
    def _build_mlp(
        in_dim: int, hidden_dims: list[int], out_dim: int, activation: nn.Module
    ) -> nn.Sequential:
        """与旧版 ``_build_mlp`` 完全相同的层序列（等价 5.x ``MLP``）。"""
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dims[0]), activation]
        for i in range(len(hidden_dims)):
            if i == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[i], out_dim))
            else:
                layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
                layers.append(activation)
        return nn.Sequential(*layers)

    @staticmethod
    def _get_obs_dim(
        obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str
    ) -> tuple[list[str], int]:
        active_obs_groups = obs_groups[obs_set]
        obs_dim = 0
        for obs_group in active_obs_groups:
            if len(obs[obs_group].shape) != 2:
                raise ValueError(
                    f"The LidarPD model only supports 1D observations, got shape "
                    f"{obs[obs_group].shape} for '{obs_group}'."
                )
            obs_dim += obs[obs_group].shape[-1]
        return active_obs_groups, obs_dim

    # ───────────────────────── 感知前向（与旧版逐行对应） ─────────────────────────
    def _split_obs(self, observations: torch.Tensor):
        prox_len = self.proximal_points * 3
        dist_len = self.distal_history_length * self.distal_points * 3

        if observations.numel() == 0:
            if observations.dim() == 2:
                return (
                    torch.empty(0, self.proprio_obs_dim, device=observations.device),
                    torch.empty(0, self.proximal_points, 3, device=observations.device),
                    torch.empty(
                        0, self.distal_history_length * self.distal_points, 3, device=observations.device
                    ),
                )
            elif observations.dim() == 3:
                t, b, _ = observations.shape
                return (
                    torch.empty(t, b, self.proprio_obs_dim, device=observations.device),
                    torch.empty(t, b, self.proximal_points, 3, device=observations.device),
                    torch.empty(
                        t, b, self.distal_history_length * self.distal_points, 3, device=observations.device
                    ),
                )
            raise ValueError(f"Unexpected observations dim: {observations.dim()}")

        if observations.dim() == 2:
            proprio = observations[:, : self.proprio_obs_dim]
            prox = observations[:, self.proprio_obs_dim : self.proprio_obs_dim + prox_len]
            dist = observations[
                :, self.proprio_obs_dim + prox_len : self.proprio_obs_dim + prox_len + dist_len
            ]
            return (
                proprio,
                prox.reshape(-1, self.proximal_points, 3),
                dist.reshape(-1, self.distal_history_length * self.distal_points, 3),
            )

        # 3D: (T, B, dim)
        t, b, _ = observations.shape
        obs_flat = observations.reshape(t * b, -1)
        proprio = obs_flat[:, : self.proprio_obs_dim].reshape(t, b, self.proprio_obs_dim)
        prox = obs_flat[:, self.proprio_obs_dim : self.proprio_obs_dim + prox_len]
        dist = obs_flat[
            :, self.proprio_obs_dim + prox_len : self.proprio_obs_dim + prox_len + dist_len
        ]
        return (
            proprio,
            prox.reshape(t, b, self.proximal_points, 3),
            dist.reshape(t, b, self.distal_history_length * self.distal_points, 3),
        )

    def _sort_by_spherical(self, points: torch.Tensor) -> torch.Tensor:
        """球坐标键排序（与旧版相同：前向不调用，LidarWrapper 已排序）。"""
        t = self._sensor_translation.to(device=points.device, dtype=points.dtype).view(1, 1, 3)
        pts = points - t
        if self._sensor_conj is not None:
            q = self._sensor_conj.to(device=points.device, dtype=points.dtype)
            pts = _quat_apply(
                q.unsqueeze(0).expand(pts.shape[0] * pts.shape[1], 4), pts.reshape(-1, 3)
            )
            pts = pts.reshape(points.shape)

        x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]
        r = torch.norm(pts, dim=-1)
        azimuth = torch.atan2(y, x)
        phi = torch.asin(z / (r + 1e-9))
        key = phi * (2.0 * math.pi) + azimuth
        order = torch.argsort(key, dim=1)
        return torch.gather(points, 1, order.unsqueeze(-1).expand_as(points))

    def _proximal_gru_hidden(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.proximal_gru(x)
        return h

    def _distal_gru_hidden(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.distal_gru(x)
        return h

    def _encode_proximal_chunked(self, prox_points: torch.Tensor) -> torch.Tensor:
        B, T_prox, P, _ = prox_points.shape
        chunk_size = 128
        outputs = []
        for start in range(0, B, chunk_size):
            end = min(start + chunk_size, B)
            chunk = prox_points[start:end]
            c = end - start
            chunk_seq = chunk.reshape(c * T_prox, P, 3)
            if torch.is_grad_enabled() and self.gradient_checkpointing_proximal:
                h = torch.utils.checkpoint.checkpoint(
                    self._proximal_gru_hidden, chunk_seq, use_reentrant=False
                )
            else:
                _, h = self.proximal_gru(chunk_seq)
            outputs.append(h.squeeze(0).reshape(c, T_prox, -1))
        return torch.cat(outputs, dim=0)

    def _encode_distal_chunked(self, dist_points: torch.Tensor) -> torch.Tensor:
        B, T_dist, D, _ = dist_points.shape
        chunk_size = 128
        outputs = []
        for start in range(0, B, chunk_size):
            end = min(start + chunk_size, B)
            chunk = dist_points[start:end]
            c = end - start
            chunk_seq = chunk.reshape(c * T_dist, D, 3)
            if torch.is_grad_enabled() and self.gradient_checkpointing_distal:
                h = torch.utils.checkpoint.checkpoint(
                    self._distal_gru_hidden, chunk_seq, use_reentrant=False
                )
            else:
                _, h = self.distal_gru(chunk_seq)
            outputs.append(h.squeeze(0).reshape(c, T_dist, -1))
        return torch.cat(outputs, dim=0)

    def build_actor_latent(
        self, observations: torch.Tensor, masks: torch.Tensor | None = None, *, cache: bool = True
    ) -> torch.Tensor:
        """旧版 ``_build_actor_latent`` 的逐行移植。"""
        proprio, proximal, distal = self._split_obs(observations)

        if masks is not None and masks.numel() > 0:
            # Recurrent path: flatten (T, B, ...) for zero-init GRU per frame.
            T_seq, B = proprio.shape[:2]
            prox_flat = proximal.reshape(T_seq * B, self.proximal_points, 3)
            prox_feat_flat = self._encode_proximal_chunked(prox_flat.unsqueeze(1)).squeeze(1)
            prox_feat = prox_feat_flat.reshape(T_seq, B, self.proximal_feature_dim)

            dist_flat = distal.reshape(T_seq * B, self.distal_history_length * self.distal_points, 3)
            dist_feat_flat = self._encode_distal_chunked(dist_flat.unsqueeze(1)).squeeze(1)
            dist_feat = dist_feat_flat.reshape(T_seq, B, self.distal_feature_dim)

            proprio = unpad_trajectories(proprio, masks)
            prox_feat = unpad_trajectories(prox_feat, masks)
            dist_feat = unpad_trajectories(dist_feat, masks)
        else:
            # Inference / feedforward: 近端点云已由 LidarWrapper 按球坐标排序，直接使用
            prox_feat_t = self._encode_proximal_chunked(proximal.unsqueeze(1))
            prox_feat = prox_feat_t.squeeze(1)

            dist_feat_t = self._encode_distal_chunked(distal.unsqueeze(1))
            dist_feat = dist_feat_t.squeeze(1)

        actor_latent = torch.cat((proprio, prox_feat, dist_feat), dim=-1)

        if cache:
            self._cached_proximal_feature = prox_feat
            self._cached_actor_latent = actor_latent
        return actor_latent

    # ───────────────────────── 5.x MLPModel 协议面 ─────────────────────────
    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state=None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """``stochastic_output=True`` ≙ 旧版 ``act``（缓存 latent）；
        ``False`` ≙ 旧版 ``act_inference``（均值输出，不写缓存）。"""
        # masks 已在 _flatten_obs 内完成去填充（仅 recurrent 填充路径可达，
        # 本模型恒为前馈路径），感知编码按逐帧零初始化 GRU 处理扁平观测。
        obs = self._flatten_obs(obs, masks)
        latent = self.get_latent(obs, masks=None, cache=stochastic_output)
        mlp_output = self.mlp(latent)
        if stochastic_output:
            self.distribution.update(mlp_output)
            return self.distribution.sample()
        # 确定性输出 = 分布均值（GaussianDistribution.deterministic_output 恒等）
        return self.distribution.deterministic_output(mlp_output)

    def _flatten_obs(
        self, obs: TensorDict, masks: torch.Tensor | None = None
    ) -> torch.Tensor:
        """选组拼接（+ 去填充），等价旧版直接收到的扁平观测张量。"""
        if masks is not None and masks.numel() > 0 and not self.is_recurrent:
            obs = unpad_trajectories(obs, masks)
        flat = torch.cat([obs[obs_group] for obs_group in self.obs_groups], dim=-1)
        return self.obs_normalizer(flat)

    def get_latent(
        self,
        obs: TensorDict | torch.Tensor,
        masks: torch.Tensor | None = None,
        hidden_state=None,
        *,
        cache: bool = True,
    ) -> torch.Tensor:
        """构建感知 latent（GRU 编码）。``cache`` 语义见 ``forward``。"""
        if isinstance(obs, TensorDict):
            obs = self._flatten_obs(obs, masks)
        return self.build_actor_latent(obs, masks=masks, cache=cache)

    def reset(self, dones: torch.Tensor | None = None, hidden_state=None) -> None:
        pass

    def get_hidden_state(self):
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        pass

    def update_normalization(self, obs: TensorDict) -> None:
        """旧版无观测归一化 → 无操作（保持语义）。"""
        if self.obs_normalization:
            obs_list = [obs[obs_group] for obs_group in self.obs_groups]
            mlp_obs = torch.cat(obs_list, dim=-1)
            self.obs_normalizer.update(mlp_obs)  # type: ignore[attr-defined]

    @property
    def output_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        return self.distribution.std

    @property
    def output_entropy(self) -> torch.Tensor:
        return self.distribution.entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        return self.distribution.params

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(outputs)

    def get_kl_divergence(
        self,
        old_params: tuple[torch.Tensor, ...],
        new_params: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        return self.distribution.kl_divergence(old_params, new_params)

    # ───────────────────────── 辅助高度监督（旧版语义） ─────────────────────────
    @property
    def cached_proximal_feature(self) -> torch.Tensor | None:
        """act() 调用后有效的近端特征缓存（供 critic / aux loss 使用）。"""
        return self._cached_proximal_feature

    def compute_auxiliary_loss(self, aux_targets: torch.Tensor) -> torch.Tensor:
        """高度图监督 MSE：读取同批 act() 缓存的近端特征（须在 act 之后调用）。"""
        prox_feat = self._cached_proximal_feature
        if prox_feat is None:
            return torch.zeros((), device=aux_targets.device)
        pred = self.height_supervisor(prox_feat)
        return torch.nn.functional.mse_loss(pred, aux_targets)

    # ───────────────────────── 导出 ─────────────────────────
    def as_jit(self) -> nn.Module:
        return _TorchLidarPDActor(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        return _OnnxLidarPDActor(self, verbose=verbose)


class LidarPDCritic5x(nn.Module):
    """rsl-rl 5.x critic：MLP 头 + **共享** actor 的双 GRU 感知。

    旧版单策略中 critic 消费 ``self.critic(self._cached_actor_latent)``。
    5.x 拆分后本类持有对 ``LidarPDActor5x`` 实例的引用（非子模块注册，
    避免参数/状态字典重复）。latent **恒从输入重建**（等价旧版
    ``evaluate`` 的回退分支）：旧版缓存命中分支有 ``masks`` 语义守卫，
    5.x 下 actor/critic 各自扁平化产生新张量，按 ``data_ptr`` 判等会在
    CUDA 缓存分配器复用地址时静默命中陈旧特征（架构审查 HIGH-1），故不
    设跨调用缓存；GRU 重编码成本可接受。
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (1024, 512, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = False,
        actor: LidarPDActor5x | None = None,
        **kwargs,
    ):
        if kwargs:
            print(
                "LidarPDCritic5x.__init__ got unexpected arguments, which will be ignored: "
                + str(list(kwargs.keys()))
            )
        super().__init__()

        self.obs_groups, self.obs_dim = LidarPDActor5x._get_obs_dim(obs, obs_groups, obs_set)
        if actor is None:
            raise ValueError("LidarPDCritic5x requires the shared LidarPDActor5x instance.")
        if actor.obs_dim != self.obs_dim:
            raise ValueError(
                f"Critic obs dim {self.obs_dim} must match the shared actor obs dim {actor.obs_dim}."
            )
        # 关键：非子模块引用（object.__setattr__ 绕过 nn.Module.__setattr__ 的注册），
        # 保证 critic.parameters()/state_dict() 只含自己的 MLP 头。
        object.__setattr__(self, "_shared_actor", actor)

        self.obs_normalization = obs_normalization
        if obs_normalization:
            raise NotImplementedError(
                "LidarPDCritic5x preserves the legacy no-normalization semantics; "
                "obs_normalization must be False."
            )
        self.obs_normalizer = torch.nn.Identity()

        act_fn = resolve_nn_activation(activation)
        # critic 输入 = actor latent（与 actor 共享 GRU 特征），仅头参数独立
        critic_input_dim = actor.proprio_obs_dim + actor.proximal_feature_dim + actor.distal_feature_dim
        self.mlp = LidarPDActor5x._build_mlp(critic_input_dim, list(hidden_dims), output_dim, act_fn)

    @property
    def shared_actor(self) -> LidarPDActor5x:
        return self._shared_actor  # type: ignore[no-any-return]

    def _flatten_obs(
        self, obs: TensorDict, masks: torch.Tensor | None = None
    ) -> torch.Tensor:
        if masks is not None and masks.numel() > 0 and not self.is_recurrent:
            obs = unpad_trajectories(obs, masks)
        flat = torch.cat([obs[obs_group] for obs_group in self.obs_groups], dim=-1)
        return self.obs_normalizer(flat)

    def get_latent(
        self,
        obs: TensorDict | torch.Tensor,
        masks: torch.Tensor | None = None,
        hidden_state=None,
    ) -> torch.Tensor:
        if isinstance(obs, TensorDict):
            obs_flat = self._flatten_obs(obs, masks)
        else:
            obs_flat = obs
        # 恒重建（≙ 旧版 evaluate 回退分支；cache=True 同旧版）——见类 docstring
        return self.shared_actor.build_actor_latent(obs_flat, masks=masks, cache=True)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state=None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        latent = self.get_latent(obs, masks, hidden_state)
        return self.mlp(latent)

    def reset(self, dones: torch.Tensor | None = None, hidden_state=None) -> None:
        pass

    def get_hidden_state(self):
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        pass

    def update_normalization(self, obs: TensorDict) -> None:
        pass


class _TorchLidarPDActor(nn.Module):
    """JIT 导出壳：输入扁平观测（任务典型 2754 维），输出确定性动作均值。"""

    def __init__(self, model: LidarPDActor5x) -> None:
        super().__init__()
        self.perception = _FlatPerception(copy.deepcopy(model))
        self.mlp = copy.deepcopy(model.mlp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.perception(x)
        return self.mlp(latent)

    @torch.jit.export
    def reset(self) -> None:
        pass


class _OnnxLidarPDActor(_TorchLidarPDActor):
    def __init__(self, model: LidarPDActor5x, verbose: bool = False) -> None:
        super().__init__(model)
        self.verbose = verbose
        self.input_size = model.obs_dim

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]


class _FlatPerception(nn.Module):
    """把 ``LidarPDActor5x`` 的感知部分（双 GRU）封装为扁平输入 → latent。

    TorchScript 可编译：拆分/分块编码逻辑以自有方法实现（不跨类调用
    未绑定的静态方法——那会使 torch.jit.script 失败，架构审查 MED-4）；
    导出路径无梯度，不使用梯度检查点。
    """

    def __init__(self, model: LidarPDActor5x) -> None:
        super().__init__()
        self.proximal_gru = copy.deepcopy(model.proximal_gru)
        self.distal_gru = copy.deepcopy(model.distal_gru)
        self.proprio_obs_dim = model.proprio_obs_dim
        self.proximal_points = model.proximal_points
        self.distal_history_length = model.distal_history_length
        self.distal_points = model.distal_points

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proprio, prox, dist = self._split_obs(x)
        prox_feat = self._encode_proximal_chunked(prox.unsqueeze(1)).squeeze(1)
        dist_feat = self._encode_distal_chunked(dist.unsqueeze(1)).squeeze(1)
        return torch.cat((proprio, prox_feat, dist_feat), dim=-1)

    def _split_obs(self, observations: torch.Tensor):
        prox_len = self.proximal_points * 3
        dist_len = self.distal_history_length * self.distal_points * 3
        proprio = observations[:, : self.proprio_obs_dim]
        prox = observations[:, self.proprio_obs_dim : self.proprio_obs_dim + prox_len]
        dist = observations[:, self.proprio_obs_dim + prox_len : self.proprio_obs_dim + prox_len + dist_len]
        return (
            proprio,
            prox.reshape(-1, self.proximal_points, 3),
            dist.reshape(-1, self.distal_history_length * self.distal_points, 3),
        )

    def _encode_proximal_chunked(self, prox_points: torch.Tensor) -> torch.Tensor:
        b, t_prox, p, _ = prox_points.shape
        outputs: list[torch.Tensor] = []
        for start in range(0, b, 128):
            end = min(start + 128, b)
            c = end - start
            chunk_seq = prox_points[start:end].reshape(c * t_prox, p, 3)
            _, h = self.proximal_gru(chunk_seq)
            outputs.append(h.squeeze(0).reshape(c, t_prox, -1))
        return torch.cat(outputs, dim=0)

    def _encode_distal_chunked(self, dist_points: torch.Tensor) -> torch.Tensor:
        b, t_dist, d, _ = dist_points.shape
        outputs: list[torch.Tensor] = []
        for start in range(0, b, 128):
            end = min(start + 128, b)
            c = end - start
            chunk_seq = dist_points[start:end].reshape(c * t_dist, d, 3)
            _, h = self.distal_gru(chunk_seq)
            outputs.append(h.squeeze(0).reshape(c, t_dist, -1))
        return torch.cat(outputs, dim=0)
