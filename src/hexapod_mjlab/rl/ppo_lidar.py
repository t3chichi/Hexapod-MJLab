"""rsl-rl 5.4.2 原生 LiDAR-PD PPO（``LidarPPO5x``）。

以 **安装版 rsl_rl 5.4.2 的 ``PPO``** 为基类原生重写，完整保留旧版
（``Lidar_legged_gym/rsl_rl`` 定制版）语义：

术语映射（旧 → 新）::

    policy.act(obs)                    → actor(obs, stochastic_output=True)
    policy.act_inference(obs)          → actor(obs)                     # 均值
    policy.evaluate(critic_obs)        → critic(critic_obs)             # 共享感知
    policy.get_actions_log_prob(a)     → actor.get_output_log_prob(a)
    policy.action_mean / action_std    → actor.output_mean / output_std
    policy.entropy                     → actor.output_entropy
    (mu, sigma) KL 公式                → actor.get_kl_divergence(old, new)
    aux_obs_buf / aux_observations     → 观测组 ``height_grid``（TensorDict 内）
    policy.compute_auxiliary_loss      → actor.compute_auxiliary_loss
    torch.cuda.amp autocast+GradScaler → torch.amp.autocast("cuda")+GradScaler("cuda")
                                         （use_amp 仍仅表示混合精度，无判别器）

与基类的差异（全部为保持旧语义而引入）：

1. **共享感知去重优化器**：actor/critic 共享双 GRU；基类
   ``chain(actor.parameters(), critic.parameters())`` 会让共享参数在
   Adam 中出现两次（等效两次更新）。此处按 id 去重后重建优化器，
   等价旧版单一 ``optim.Adam(policy.parameters())``。
2. **单次梯度裁剪**：基类分别对 actor/critic 各裁一次（共享参数会被裁
   两次）；此处对去重并集裁一次，等价旧版 ``clip_grad_norm_(policy.parameters())``。
3. **辅助高度 MSE**：``update()`` 中读取 mini-batch 观测 TensorDict 的
   ``height_grid`` 组调用 ``actor.compute_auxiliary_loss``（替换旧版
   aux_obs_buf 管线），系数 ``aux_loss_coef``。
4. **混合精度（可选）**：``use_amp=True`` 时前向/损失包在
   ``torch.amp.autocast("cuda")``，反向走 ``torch.amp.GradScaler("cuda")``
   （unscale → clip → step → update），与旧版 PPO 的 AMP 分支一致。
   仅 CUDA 设备可用（CPU 上自动禁用并告警——旧版同样只在 GPU 训练用）。

其余项（surrogate、value 裁剪、熵正则、GAE、自适应 KL 学习率、
mini-batch 生成、time-out bootstrap）全部继承基类实现，
逐项等价性见 ``tests/rl_parity/PARITY_AUDIT.md``。
"""

from __future__ import annotations

from itertools import chain

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO as RslRlPPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups, resolve_optimizer

from hexapod_mjlab.rl.actor_v2 import LidarPDActor5x, LidarPDCritic5x
from hexapod_mjlab.rl.symmetry_adapter import make_el_4090_symmetry_func


def _unique_parameters(param_iter):
    """按 id 去重（保持顺序）——共享感知参数只在优化器中出现一次。"""
    seen: dict[int, nn.Parameter] = {}
    for p in param_iter:
        seen.setdefault(id(p), p)
    return list(seen.values())


class LidarPPO5x(RslRlPPO):
    """LiDAR-PD 原生 PPO（rsl-rl 5.4.2）。"""

    aux_obs_key: str = "height_grid"
    """特权观测组名：高度图监督目标（旧版 aux_obs_buf 的等价物）。"""

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        optimizer: str = "adam",
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        device: str = "cpu",
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        multi_gpu_cfg: dict | None = None,
        # ── 旧版定制参数 ──
        aux_loss_coef: float = 1.0,
        use_amp: bool = False,
    ) -> None:
        self._optimizer_name = optimizer
        self.aux_loss_coef = aux_loss_coef
        super().__init__(
            actor=actor,
            critic=critic,
            storage=storage,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            optimizer=optimizer,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            normalize_advantage_per_mini_batch=normalize_advantage_per_mini_batch,
            device=device,
            rnd_cfg=rnd_cfg,
            symmetry_cfg=symmetry_cfg,
            multi_gpu_cfg=multi_gpu_cfg,
        )

        # ── 优化器参数表（去重，防御性）──
        # critic 经 object.__setattr__ 引用共享 actor，其 parameters() 不含
        # 共享 GRU，基类的 chain 并不会重复；去重为防御注册方式变化，且
        # update() 的单次梯度裁剪需要这份并集。
        self._all_parameters = _unique_parameters(chain(self.actor.parameters(), self.critic.parameters()))
        self.optimizer = resolve_optimizer(self._optimizer_name)(self._all_parameters, lr=learning_rate)

        # ── 混合精度（旧 use_amp = 仅 autocast + GradScaler，无判别器） ──
        self.use_amp = bool(use_amp)
        if self.use_amp and not str(self.device).startswith("cuda"):
            print("[LidarPPO5x] use_amp requested on a non-CUDA device; disabling AMP.")
            self.use_amp = False
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

    # ─────────────────────────────────────────────────────────────
    # 构造：注入 LiDAR-PD actor/critic（共享感知）
    # ─────────────────────────────────────────────────────────────
    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> LidarPPO5x:
        """基类 ``PPO.construct_algorithm`` 的 LiDAR-PD 版。

        差异：
        * actor 固定为 ``LidarPDActor5x``（cfg 中的 actor/critic ``class_name``
          被弹出并忽略——critic 构造需要共享 actor 引用，不走 class_name 路径）；
        * critic 固定为 ``LidarPDCritic5x`` 并注入共享 actor 引用
          （5.x 的 critic 构造签名没有 actor 参数，故不走 class_name 路径）；
        * ``symmetry_cfg["symmetry_kwargs"]``（旧 runner 的传感器绑定习惯）
          在此吸收并绑定到镜像函数上，不透传给 ``Symmetry``。
        """
        alg_class = resolve_callable(cfg["algorithm"].pop("class_name"))
        cfg["actor"].pop("class_name", None)
        cfg["critic"].pop("class_name", None)
        # mjlab 官方 asdict(cfg.agent) 必带该键（基类在官方路径 pop 后用于
        # 共享 CNN 决策）；本算法无 CNN，弹出并忽略（架构审查 MED-3）
        cfg["algorithm"].pop("share_cnn_encoders", None)

        # Resolve observation groups
        default_sets = ["actor", "critic"]
        if cfg["algorithm"].get("rnd_cfg") is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)
        # Resolve RND config if used
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)

        # Resolve symmetry config: bind legacy sensor kwargs into the adapter
        alg_cfg = cfg["algorithm"]
        if alg_cfg.get("symmetry_cfg") is not None:
            sym_cfg = dict(alg_cfg["symmetry_cfg"])
            symmetry_kwargs = sym_cfg.pop("symmetry_kwargs", None)
            if symmetry_kwargs is not None:
                symmetry_kwargs = dict(symmetry_kwargs)
                bound = False
                for src_attr, dst_key in [
                    ("_sensor_offset_quat", "sensor_quat"),
                    ("_sensor_translation", "sensor_trans"),
                ]:
                    if hasattr(env, src_attr):
                        symmetry_kwargs[dst_key] = getattr(env, src_attr)[0:1].to(device)
                        bound = True
                if not bound:
                    # 架构审查 MED-6：绑定失败时静默假设单位安装变换会让镜像
                    # 在错误坐标系里重排点云（隐性训练退化）。M3 集成时必须由
                    # env 显式提供安装变换；届时应将此告警升级为按配置报错。
                    print(
                        "[LidarPPO5x] symmetry sensor transform not found on env "
                        "(looked for _sensor_offset_quat/_sensor_translation); assuming "
                        "identity mount for the mirror function — INCORRECT if the LiDAR "
                        "is mounted off-origin (see MIGRATION.md M3 notes)."
                    )
                sym_cfg["data_augmentation_func"] = make_el_4090_symmetry_func(**symmetry_kwargs)
            alg_cfg["symmetry_cfg"] = sym_cfg
        # 官方路径在此注入 env（Symmetry.__init__ 必需）——先前缺失导致
        # 官方风格 symmetry_cfg 直接 TypeError（架构审查 HIGH-2）
        alg_cfg = resolve_symmetry_config(alg_cfg, env)

        # Initialize the actor (LiDAR-PD perception + head + distribution)
        actor = LidarPDActor5x(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        print(f"Actor Model: {actor}")
        # Initialize the critic: shared-perception head
        critic = LidarPDCritic5x(obs, cfg["obs_groups"], "critic", 1, actor=actor, **cfg["critic"]).to(device)
        print(f"Critic Model: {critic}")

        # Initialize the storage: full obs TensorDict (policy + height_grid) is
        # persisted per step — this is how the privileged height grid reaches update().
        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)

        # Initialize the algorithm
        alg = alg_class(actor, critic, storage, device=device, **cfg["algorithm"], multi_gpu_cfg=cfg["multi_gpu"])
        alg.compile(cfg.get("torch_compile_mode"))
        return alg

    # ─────────────────────────────────────────────────────────────
    # update(): 基类 5.4.2 流程 + 旧版定制（aux 高度 MSE、AMP、单次裁剪）
    # ─────────────────────────────────────────────────────────────
    def update(self) -> dict[str, float]:
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_rnd_loss = 0 if self.rnd else None
        mean_symmetry_loss = 0 if self.symmetry else None
        mean_aux_loss = 0.0
        num_aux_updates = 0

        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)

            if self.symmetry:
                self.symmetry.augment_batch(batch, original_batch_size)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                # Recompute actions log prob and entropy for the current batch
                self.actor(
                    batch.observations,
                    masks=batch.masks,
                    hidden_state=batch.hidden_states[0],
                    stochastic_output=True,
                )
                actions_log_prob = self.actor.get_output_log_prob(batch.actions)
                values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
                distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
                entropy = self.actor.output_entropy[:original_batch_size]
                # Snapshot the height-grid targets NOW: they must stay aligned with the
                # proximal-feature cache written by the stochastic forward above. If only
                # the mirror loss is active, Symmetry.compute_loss below replaces
                # batch.observations with its augmented (2B) copy — the legacy code kept
                # aux_obs_batch un-augmented in that mode (B rows), which this snapshot
                # reproduces. With data augmentation the batch was already augmented
                # before the forward, so the snapshot is the augmented (2B) targets —
                # also matching the legacy behavior.
                aux_targets = (
                    batch.observations[self.aux_obs_key].clone() if self.aux_obs_key in batch.observations else None
                )

                # KL divergence + adaptive learning rate（等价旧版公式）
                if self.desired_kl is not None and self.schedule == "adaptive":
                    with torch.inference_mode():
                        kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)
                        kl_mean = torch.mean(kl)
                        if self.is_multi_gpu:
                            torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                            kl_mean /= self.gpu_world_size
                        if self.gpu_global_rank == 0:
                            if kl_mean > self.desired_kl * 2.0:
                                self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                                self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        if self.is_multi_gpu:
                            lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                            torch.distributed.broadcast(lr_tensor, src=0)
                            self.learning_rate = lr_tensor.item()
                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = self.learning_rate

                # Surrogate loss（逐项同旧版）
                ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
                surrogate = -torch.squeeze(batch.advantages) * ratio
                surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss（裁剪式，同旧版）
                if self.use_clipped_value_loss:
                    value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - batch.returns).pow(2)
                    value_losses_clipped = (value_clipped - batch.returns).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (batch.returns - values).pow(2).mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

                # RND loss（未使用时为 None）
                rnd_loss = self.rnd.compute_loss(batch.observations[:original_batch_size]) if self.rnd else None

                # Symmetry mirror loss（旧版顺序：先 symmetry 后 aux）
                symmetry_loss = None
                if self.symmetry:
                    symmetry_loss = self.symmetry.compute_loss(self.actor, batch, original_batch_size)
                    if self.symmetry.use_mirror_loss:
                        loss = loss + self.symmetry.mirror_loss_coeff * symmetry_loss

                # Auxiliary height-grid MSE loss（旧版 aux_obs 管线的原生替代；
                # 读取上面快照的目标，与 act() 缓存的近端特征保持同批对齐）
                aux_loss = None
                if aux_targets is not None and hasattr(self.actor, "compute_auxiliary_loss"):
                    aux_loss = self.actor.compute_auxiliary_loss(aux_targets)
                    loss = loss + self.aux_loss_coef * aux_loss

            # ── 反向 + 优化（AMP 分支与旧版一致：scale → unscale → clip → step → update） ──
            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.rnd:
                    self.rnd.optimizer.zero_grad()
                    self.scaler.scale(rnd_loss).backward()
                if self.is_multi_gpu:
                    self.reduce_parameters()
                self.scaler.unscale_(self.optimizer)
                # 单次裁剪（去重并集）≙ 旧版 clip_grad_norm_(policy.parameters())
                nn.utils.clip_grad_norm_(self._all_parameters, self.max_grad_norm)
                self.scaler.step(self.optimizer)
                if self.rnd:
                    self.scaler.unscale_(self.rnd.optimizer)
                    self.scaler.step(self.rnd.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.rnd:
                    self.rnd.optimizer.zero_grad()
                    rnd_loss.backward()
                if self.is_multi_gpu:
                    self.reduce_parameters()
                nn.utils.clip_grad_norm_(self._all_parameters, self.max_grad_norm)
                self.optimizer.step()
                if self.rnd:
                    self.rnd.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None and symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()
            if aux_loss is not None:
                mean_aux_loss += aux_loss.item()
                num_aux_updates += 1

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        if num_aux_updates > 0:
            mean_aux_loss /= num_aux_updates

        loss_dict = {
            "value_function": mean_value_loss,  # 旧版键名
            "value": mean_value_loss,  # 5.x 基类键名（同值）
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        if num_aux_updates > 0:
            loss_dict["aux"] = mean_aux_loss

        self.storage.clear()
        return loss_dict

    # ── save/load：基类已按 actor/critic 分别保存状态字典；
    #    共享 GRU 只存在于 actor_state_dict（critic 不注册共享模块），
    #    旧 checkpoint 迁移见 hexapod_mjlab.rl.ckpt_convert。 ──
