"""F2 网络与形态输出：掩码点序列 GRU、共享感知/形态头、独立动作均值/价值头。

M2 起本文件为唯一正典实现（由 ``rl/joint_passage/network.py`` 兼容垫片别名）。
正典类名为 ``ConfinedLocomotionNetwork``；``JointPassageNetwork`` 是同一类的
兼容别名（同一类对象，两个名字的 isinstance/pickle/state_dict 完全一致）。
网络数学与已验收的 joint_passage F2 网络保持一致；为外置推理缓存新增了
``PointCloudFeatures``、分离点云编码和 ``forward_from_features`` 接口。
普通 ``forward`` 仍每次重新编码点云，保持 PPO 的新鲜前向与梯度路径。

仅标准库与 PyTorch 的纯 CPU/GPU 计算模块；不注册任务、不导入 mjlab 或任务
注册器；不含 PPO 采样/logprob/标准差/优化器/真实训练。依据：
docs/tasks/confined_locomotion/framework/实验记录.md、docs/research/训练架构对齐.md、
docs/tasks/confined_locomotion/平地学习与基准测量.md 第 2 节。

结构（已确认尺寸）：
* 近端 256 点 → 单层 GRU 隐藏 187；远端 10×64=640 点 → 单层 GRU 隐藏 64。
  掩码 false 点不递归更新 hidden、填充点不作为真实零点输入 GRU；逐行有效点
  稳定压紧后经 pack_padded_sequence 一次前向（非 640 次 Python 逐步循环）；
  空行返回零特征（等价于未运行、h≡h0=0）；支持不同有效点数与非前缀掩码。
* 每次 forward h0=0（nn.GRU 默认零初始隐藏），不跨控制步保存隐藏状态；网络
  本身无模块级/全局前向缓存，no-grad 推理特征复用由独立普通 Python 包装器持有；
  无运动补偿、无点云缩放、无第 4 通道。
* 输入点序列须已由调用方用 F1 显式角键排序（sort_points_by_angle_keys）；
  本模块不选择/重选角度规则、不隐含坐标外参、不改输入顺序。
* 水平形态头读 [近端特征, 本体]；垂向头读 [近端, 远端, 本体]；均不读指令。
* 动作均值头/价值头相互独立，同读 [近端, 远端, 本体, 指令, detach(Ê)](B,323)；
  动作 18 维、价值 1 维，隐藏 512→256→128；形态隐藏 128→64（ELU 已确认）；
  地面高度头 Linear(187, M)，M 由调用方显式给定或显式 None（不创建，不默设 187）。
* 物理映射唯一在 mapping 模块（Softplus r=1）；对外 Ê 为米制、非 detach
  （辅助损失路径）；actor/critic 消费 detach(Ê) 副本，截断只切断 Ê→动作/价值
  路径，不冻结共享编码器（感知直连路径仍在）。

未确认项显式化（本模块不设任务默认）：
* 动作/价值 MLP 激活：构造参数 mlp_activation 显式给出——无状态可调用直接使用；
  nn.Module 实例（可训练激活，如 nn.PReLU）按头深拷贝为私有副本，动作/价值头
  的激活参数互不共享（传入实例仅作模板，本身不被注册），保证两头独立，
  不以参数去重掩盖跨头共享；
* 其余初始化：init_fn 回调显式给出（None=保留 PyTorch 默认，同样须显式传入）；
  回调可覆盖 GRU/Linear 权重，但其后形态末层仍统一强制零权重+按模式偏置；
* 形态 raw/残差表示：envelope_mode 显式给出（"raw"/"residual"），无默认；
* 地面头输出维 ground_samples 显式给出（int≥1）或显式 None。M2 真实闭环
  使用 None：真实地面辅助网格/覆盖未确认，这是**显式未实现的真实辅助**
  （不是放弃地面目标），契约与绑定 API 原样保留。

缩放（已确认倍率；唯一施加点在本网络 forward；输入契约=未缩放命名字段）：
本体 [lin_vel×2, ang_vel×0.25, gravity×1, q−q_ref×1, joint_vel×0.05,
prev_action×1]，指令 [vx×2, vy×2, ωz×0.25]；不加 running normalization。
调用方必须保存并传入原始（未缩放）观测：预缩放字段会被本网络再次缩放
（双重缩放错误）；rollout 应保存 F1 OnlineObservation 的原始字段。

其他边界：
* 在线前向接口不接受标签/场景真值；env_ids 可非连续/重复，仅作身份，
  不用于查活窗口或旧 rollout（近/远点与掩码由调用方排序后直接传入张量）。
* 同一次 forward 返回命名的近/远特征、Ê、action_mean/value 及可选 ground_pred；
  形态头输入不 detach（辅助损失经形态头更新共享编码器）。
* 共享子模块各只注册一次；unique_parameters() 提供按 id 去重的参数视图，
  普通 model.parameters() 亦无重复（测试验证）。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from .contracts import (
    COMMAND_DIM,
    ENVELOPE_DIM,
    FAR_POINTS_PER_SCAN,
    FAR_WINDOW_SLOTS,
    FLOAT_DTYPE,
    JOINT_DIM,
    NEAR_POINTS,
    PROPRIO_DIM,
    PROPRIO_LAYOUT,
    ContractError,
    OnlineObservation,
    validate_point_block,
)
from .mapping import H_REF_STARTUP, P0_STARTUP, EnvelopeSoftplusMapping

__all__ = [
    "COMMAND_SCALE",
    "CONTROL_HIDDEN_DIMS",
    "CONTROL_INPUT_DIM",
    "FAR_GRU_HIDDEN",
    "FAR_SEQUENCE_POINTS",
    "GROUND_INPUT_DIM",
    "ConfinedLocomotionNetwork",
    "JointPassageNetwork",
    "MORPHOLOGY_HIDDEN_DIMS",
    "MaskedPointSequenceGRU",
    "MLPHead",
    "MorphologyHead",
    "NEAR_GRU_HIDDEN",
    "NetworkOutput",
    "PointCloudFeatures",
    "PROPRIO_FIELD_SCALE",
    "build_proprio_scale",
]

# —— 已确认尺寸（docs/tasks/confined_locomotion/平地学习与基准测量.md 第 2 节）——
NEAR_GRU_HIDDEN = 187                       # 近端 256 点 → 187 维
FAR_GRU_HIDDEN = 64                         # 远端 640 点 → 64 维
FAR_SEQUENCE_POINTS = FAR_WINDOW_SLOTS * FAR_POINTS_PER_SCAN   # 10×64=640
MORPHOLOGY_HIDDEN_DIMS = (128, 64)          # 水平/垂向形态 MLP 隐藏层（ELU）
CONTROL_HIDDEN_DIMS = (512, 256, 128)       # 动作/价值 MLP 隐藏层
GROUND_INPUT_DIM = NEAR_GRU_HIDDEN          # 地面高度头读近端特征

# —— 已确认观测倍率（唯一施加点：本网络 forward）——
PROPRIO_FIELD_SCALE = {
    "lin_vel_b": 2.0,
    "ang_vel_b": 0.25,
    "gravity_b": 1.0,
    "joint_pos_rel_ref": 1.0,
    "joint_vel": 0.05,
    "prev_action": 1.0,
}
COMMAND_SCALE = (2.0, 2.0, 0.25)

_VECTOR3_FIELDS = frozenset({"lin_vel_b", "ang_vel_b", "gravity_b"})

# 动作/价值输入 = 近端特征 + 远端特征 + 本体 + 指令 + detach(Ê) = 187+64+63+3+6
CONTROL_INPUT_DIM = NEAR_GRU_HIDDEN + FAR_GRU_HIDDEN + PROPRIO_DIM + COMMAND_DIM + ENVELOPE_DIM


def build_proprio_scale() -> torch.Tensor:
    """按 PROPRIO_LAYOUT 顺序构造 (63,) float32 本体倍率向量（已确认倍率）。"""
    parts = []
    for field in PROPRIO_LAYOUT:
        dim = 3 if field in _VECTOR3_FIELDS else JOINT_DIM
        parts.append(torch.full((dim,), PROPRIO_FIELD_SCALE[field], dtype=FLOAT_DTYPE))
    scale = torch.cat(parts)
    if int(scale.numel()) != PROPRIO_DIM:
        raise ContractError("本体倍率向量维数与 PROPRIO_DIM 不一致")
    return scale


def _require_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ContractError(f"{name} 须为 >=1 的整数（显式给定），实际 {value!r}")
    return int(value)


class MaskedPointSequenceGRU(nn.Module):
    """单层掩码点序列 GRU 编码器：逐行仅沿真实有效点递归，空行返回零特征。

    * h0=0（nn.GRU 默认零初始隐藏，从不传入/保存隐藏状态），单层，batch_first。
    * 掩码 false 点不进入 GRU：填充点不是真实零点，也不以零输入"混过"递归；
      逐行把有效点按原相对顺序稳定压紧到序列前部（支持非前缀掩码、不同有效
      点数），再以 pack_padded_sequence 走一次 nn.GRU（向量化，无逐点 Python
      循环），取各行长度的末步隐藏 h_n 作为该行特征。
    * 全无效行返回零向量：等价于该行 GRU 未运行（h≡h0=0），不捏造输入。
    * 本模块不改输入顺序、不重排、不排序（排序属上游 F1 角键职责），
      无坐标外参/缩放/运动补偿。
    """

    def __init__(self, *, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = _require_positive_int(hidden_size, name="hidden_size")
        self.gru = nn.GRU(
            input_size=3, hidden_size=self.hidden_size, num_layers=1, batch_first=True
        )

    def forward(self, points: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if not isinstance(points, torch.Tensor) or points.dim() != 3:
            raise ContractError("points 应为 (B, N, 3) float32 张量")
        validate_point_block(
            points, valid, n_points=int(points.shape[1]), name="point_sequence"
        )
        batch = int(points.shape[0])
        lengths = valid.sum(dim=1)                       # (B,) int64，每行有效点数
        nonempty_rows = (lengths > 0).nonzero(as_tuple=False).flatten()
        features = points.new_zeros(batch, self.hidden_size)
        if nonempty_rows.numel() == 0:
            return features                              # 全空批：零特征
        sub_points = points.index_select(0, nonempty_rows)
        sub_valid = valid.index_select(0, nonempty_rows)
        sub_lengths = lengths.index_select(0, nonempty_rows)
        # 稳定压紧：有效点按原相对顺序前移，无效点置尾（不重排有效点）。
        order = torch.argsort((~sub_valid).to(torch.int64), dim=1, stable=True)
        gathered_index = order.unsqueeze(-1).expand(-1, sub_points.shape[1], sub_points.shape[2])
        compacted = torch.gather(sub_points, 1, gathered_index).contiguous()
        max_length = int(sub_lengths.max())
        packed = pack_padded_sequence(
            compacted[:, :max_length].contiguous(), sub_lengths.cpu(),
            batch_first=True, enforce_sorted=False,
        )
        _, h_n = self.gru(packed)                        # h0 默认全 0；h_n=各行有效末步隐藏
        return features.index_copy(0, nonempty_rows, h_n[0])


class MLPHead(nn.Module):
    """通用 MLP：Linear(in,h1)→act→…→Linear(hk,out)，末层线性输出。

    激活由调用方显式注入（未确认项，无任务默认），两种形式：
    * 张量→张量的无状态可调用（如 torch.tanh）：直接调用，多个头可共用同一函数；
    * nn.Module 实例（可含训练参数，如 nn.PReLU）：深拷贝为本头私有副本，
      动作/价值等各头的激活参数互不共享（传入实例仅作模板，自身不被注册），
      保证"动作与价值头独立"——不能以 model.parameters() 的参数去重掩盖跨头共享。
    """

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        activation: Callable[[torch.Tensor], torch.Tensor] | nn.Module,
    ) -> None:
        super().__init__()
        if isinstance(activation, nn.Module):
            self.activation = copy.deepcopy(activation)
        elif callable(activation):
            self.activation = activation
        else:
            raise ContractError("activation 须为张量→张量可调用或 nn.Module 实例（须显式给出）")
        dims = [_require_positive_int(input_dim, name="input_dim")]
        dims += [_require_positive_int(h, name="hidden_dims") for h in hidden_dims]
        dims.append(_require_positive_int(output_dim, name="output_dim"))
        self.hidden = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 2)
        )
        self.output = nn.Linear(dims[-2], dims[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = x
        for layer in self.hidden:
            hidden = self.activation(layer(hidden))
        return self.output(hidden)


class MorphologyHead(nn.Module):
    """形态 MLP：Linear(in,128)→ELU→Linear(128,64)→ELU→Linear(64,k) 末层。

    隐藏层尺寸与 ELU 为已确认值（平地说明第 2 节）。末层（terminal）由所属网络
    在 init_fn 之后统一置零权重+按模式偏置（raw→softplus⁻¹(1)/residual→0），
    使初始物理输出等于启动基准且参数仍可训练；load_state_dict 恢复训练后的值。
    """

    def __init__(self, *, input_dim: int, output_dim: int) -> None:
        super().__init__()
        first, second = MORPHOLOGY_HIDDEN_DIMS
        in_dim = _require_positive_int(input_dim, name="input_dim")
        out_dim = _require_positive_int(output_dim, name="output_dim")
        self.fc1 = nn.Linear(in_dim, first)
        self.fc2 = nn.Linear(first, second)
        self.terminal = nn.Linear(second, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.elu(self.fc1(x))
        hidden = F.elu(self.fc2(hidden))
        return self.terminal(hidden)

    def reset_terminal(self, bias_value: float) -> None:
        """末层零权重+指定偏置；在网络构造（init_fn 之后）时调用一次。"""
        with torch.no_grad():
            self.terminal.weight.zero_()
            self.terminal.bias.fill_(float(bias_value))


@dataclass(frozen=True)
class PointCloudFeatures:
    """Near/far point-cloud embeddings produced by this network's GRU encoders.

    This value object owns no parameters and is not part of ``state_dict``.
    Fresh training forwards retain their autograd graph in these tensors;
    inference-only caches store detached copies instead.
    """

    near_feature: torch.Tensor        # (B,187)
    far_feature: torch.Tensor         # (B,64)


@dataclass(frozen=True)
class NetworkOutput:
    """一次在线前向的命名输出（无标签/真值字段；ground_pred 可选）。

    envelope 为米制 Ê 的非 detach 版本（包络辅助损失路径使用）；动作/价值头
    在本次前向内部已消费 detach(Ê) 副本（截断路径）。
    """

    near_feature: torch.Tensor            # (B,187) 近端 GRU 特征
    far_feature: torch.Tensor             # (B,64) 远端 GRU 特征
    envelope: torch.Tensor                # (B,6) 米制 Ê=[fw,mw,bw,front_x,−rear,Δz]
    action_mean: torch.Tensor             # (B,18) 动作均值头输出
    value: torch.Tensor                   # (B,1) 价值头输出
    ground_pred: torch.Tensor | None = None   # (B,M) 地面高度辅助预测，未建头时 None


class ConfinedLocomotionNetwork(nn.Module):
    """confined_locomotion 在线网络：共享近/远 GRU 与水平/垂向形态头、独立动作/价值头。

    必需配置（均无默认，未确认项须显式给出）：
      mlp_activation  动作/价值 MLP 激活（无状态可调用直接共用；nn.Module 实例
                      按头深拷贝为私有副本，动作/价值激活参数互不共享）；
      init_fn         初始化回调（nn.Module→None，self.apply 覆盖 GRU/Linear），
                      None=保留 PyTorch 默认（也须显式传入）；
      envelope_mode   形态表示 "raw"/"residual"（正式表示未确认）；
      ground_samples  地面头输出维 M（int>=1）或 None（不建头，不默设 187）。
    可选覆盖（已确认启动值作默认）：p0（六启动正量）、h_ref（高度参考）。
    """

    def __init__(
        self,
        *,
        mlp_activation: Callable[[torch.Tensor], torch.Tensor] | nn.Module,
        init_fn: Callable[[nn.Module], None] | None,
        envelope_mode: str,
        ground_samples: int | None,
        p0: Sequence[float] | torch.Tensor = P0_STARTUP,
        h_ref: float = H_REF_STARTUP,
    ) -> None:
        super().__init__()
        if not callable(mlp_activation):
            raise ContractError("mlp_activation 必须可调用（未确认项，须显式给出）")
        if init_fn is not None and not callable(init_fn):
            raise ContractError("init_fn 须为可调用回调或显式 None")
        if ground_samples is not None:
            _require_positive_int(ground_samples, name="ground_samples")
        self.ground_samples = None if ground_samples is None else int(ground_samples)

        self.mapping = EnvelopeSoftplusMapping(mode=envelope_mode, p0=p0, h_ref=h_ref)
        self.near_encoder = MaskedPointSequenceGRU(hidden_size=NEAR_GRU_HIDDEN)
        self.far_encoder = MaskedPointSequenceGRU(hidden_size=FAR_GRU_HIDDEN)
        self.horizontal_head = MorphologyHead(
            input_dim=NEAR_GRU_HIDDEN + PROPRIO_DIM, output_dim=ENVELOPE_DIM - 1
        )
        self.vertical_head = MorphologyHead(
            input_dim=NEAR_GRU_HIDDEN + FAR_GRU_HIDDEN + PROPRIO_DIM, output_dim=1
        )
        self.action_head = MLPHead(
            input_dim=CONTROL_INPUT_DIM,
            hidden_dims=CONTROL_HIDDEN_DIMS,
            output_dim=JOINT_DIM,
            activation=mlp_activation,
        )
        self.value_head = MLPHead(
            input_dim=CONTROL_INPUT_DIM,
            hidden_dims=CONTROL_HIDDEN_DIMS,
            output_dim=1,
            activation=mlp_activation,
        )
        self.ground_head = (
            None if self.ground_samples is None
            else nn.Linear(GROUND_INPUT_DIM, self.ground_samples)
        )
        self.register_buffer("proprio_scale", build_proprio_scale())
        self.register_buffer("command_scale", torch.tensor(COMMAND_SCALE, dtype=FLOAT_DTYPE))

        if init_fn is not None:
            self.apply(init_fn)
        # init_fn 之后强制形态末层：零权重 + 按模式偏置（raw→b / residual→0），
        # 回调不得破坏"初始输出=启动基准、参数可训练"的约定。
        bias = self.mapping.head_terminal_bias
        self.horizontal_head.reset_terminal(bias)
        self.vertical_head.reset_terminal(bias)

    def forward(
        self,
        observation: OnlineObservation,
        near_points: torch.Tensor,
        near_valid: torch.Tensor,
        far_points: torch.Tensor,
        far_valid: torch.Tensor,
    ) -> NetworkOutput:
        """在线前向：输入为未缩放命名字段（OnlineObservation）与已排序点/掩码。

        近/远点序列须已由调用方按 F1 显式角键排序：近端 (B,256,3)/(B,256)、
        远端 (B,640,3)/(B,640)（flatten_far_window+sort 后直接传入）。
        不接受任何标签/场景真值；不查窗口/旧 rollout；无隐藏状态跨调用
        （每次 GRU 前向从零隐藏状态开始）。
        """
        self.validate_inputs(
            observation, near_points, near_valid, far_points, far_valid
        )
        features = self.encode_points(
            near_points, near_valid, far_points, far_valid
        )
        return self._forward_from_validated_features(observation, features)

    def validate_inputs(
        self,
        observation: OnlineObservation,
        near_points: torch.Tensor,
        near_valid: torch.Tensor,
        far_points: torch.Tensor,
        far_valid: torch.Tensor,
    ) -> None:
        """Validate the complete raw online input contract without encoding it.

        Keep this check at inference-cache hits too: a cached embedding never
        makes malformed observations or point blocks acceptable.
        """
        if not isinstance(observation, OnlineObservation):
            raise ContractError("observation 必须为 F1 OnlineObservation（未缩放命名字段）")
        observation.validate()
        batch = int(observation.env_ids.shape[0])
        device = observation.env_ids.device
        validate_point_block(near_points, near_valid, n_points=NEAR_POINTS, name="near_points")
        validate_point_block(
            far_points, far_valid, n_points=FAR_SEQUENCE_POINTS, name="far_points"
        )
        for name, block in (("near_points", near_points), ("far_points", far_points)):
            if int(block.shape[0]) != batch:
                raise ContractError(f"{name} 批量 {int(block.shape[0])} 与观测批量 {batch} 不一致")
            if block.device != device:
                raise ContractError(f"{name} 设备 {block.device} 与观测 {device} 不一致")

    @staticmethod
    def _check_encoder_block_size(
        points: torch.Tensor, *, n_points: int, name: str
    ) -> None:
        if not isinstance(points, torch.Tensor) or points.dim() != 3:
            raise ContractError(f"{name} 应为 (B, {n_points}, 3) float32 张量")
        if int(points.shape[0]) < 1 or tuple(points.shape[1:]) != (n_points, 3):
            raise ContractError(
                f"{name} 形状应为 (B, {n_points}, 3)，实际 {tuple(points.shape)}"
            )

    def encode_near_points(
        self, near_points: torch.Tensor, near_valid: torch.Tensor
    ) -> torch.Tensor:
        """Encode only the near block; used for row-selective inference misses."""
        self._check_encoder_block_size(
            near_points, n_points=NEAR_POINTS, name="near_points"
        )
        return self.near_encoder(near_points, near_valid)

    def encode_far_points(
        self, far_points: torch.Tensor, far_valid: torch.Tensor
    ) -> torch.Tensor:
        """Encode only the far block; used for row-selective inference misses."""
        self._check_encoder_block_size(
            far_points, n_points=FAR_SEQUENCE_POINTS, name="far_points"
        )
        return self.far_encoder(far_points, far_valid)

    def encode_points(
        self,
        near_points: torch.Tensor,
        near_valid: torch.Tensor,
        far_points: torch.Tensor,
        far_valid: torch.Tensor,
    ) -> PointCloudFeatures:
        """Freshly encode both validated point blocks, preserving GRU gradients."""
        self._check_encoder_block_size(
            near_points, n_points=NEAR_POINTS, name="near_points"
        )
        self._check_encoder_block_size(
            far_points, n_points=FAR_SEQUENCE_POINTS, name="far_points"
        )
        if int(near_points.shape[0]) != int(far_points.shape[0]):
            raise ContractError(
                "near/far point batches differ: "
                f"{int(near_points.shape[0])} vs {int(far_points.shape[0])}"
            )
        if near_points.device != far_points.device:
            raise ContractError(
                f"near/far point devices differ: {near_points.device} vs {far_points.device}"
            )
        return PointCloudFeatures(
            near_feature=self.encode_near_points(near_points, near_valid),
            far_feature=self.encode_far_points(far_points, far_valid),
        )

    def _validate_observation(self, observation: OnlineObservation) -> tuple[int, torch.device]:
        if not isinstance(observation, OnlineObservation):
            raise ContractError("observation 必须为 F1 OnlineObservation（未缩放命名字段）")
        observation.validate()
        return int(observation.env_ids.shape[0]), observation.env_ids.device

    def _validate_features(
        self,
        observation: OnlineObservation,
        features: PointCloudFeatures,
    ) -> None:
        batch, device = self._validate_observation(observation)
        if not isinstance(features, PointCloudFeatures):
            raise ContractError("features 必须为 PointCloudFeatures")
        expected = (
            ("near_feature", features.near_feature, NEAR_GRU_HIDDEN),
            ("far_feature", features.far_feature, FAR_GRU_HIDDEN),
        )
        for name, feature, width in expected:
            if not isinstance(feature, torch.Tensor) or tuple(feature.shape) != (batch, width):
                shape = None if not isinstance(feature, torch.Tensor) else tuple(feature.shape)
                raise ContractError(
                    f"{name} 形状应为 ({batch}, {width})，实际 {shape}"
                )
            if feature.dtype is not FLOAT_DTYPE:
                raise ContractError(
                    f"{name} dtype 应为 {FLOAT_DTYPE}，实际 {feature.dtype}"
                )
            if feature.device != device:
                raise ContractError(
                    f"{name} 设备 {feature.device} 与观测 {device} 不一致"
                )
            if not bool(torch.isfinite(feature).all()):
                raise ContractError(f"{name} 含 NaN/Inf")

    def forward_from_features(
        self,
        observation: OnlineObservation,
        features: PointCloudFeatures,
    ) -> NetworkOutput:
        """Run all observation-dependent heads from checked point features.

        This lower-level entry point is intended for the no-grad cache wrapper.
        The ordinary ``forward`` remains the raw-input, fresh-encoder API used
        by PPO and continues to preserve gradients through both encoders.
        """
        self._validate_features(observation, features)
        return self._forward_from_validated_features(observation, features)

    def _forward_from_validated_features(
        self, observation: OnlineObservation, features: PointCloudFeatures
    ) -> NetworkOutput:
        proprio = observation.proprio_vector()      # (B,63) 未缩放；缩放唯一施加点在此
        scaled_proprio = proprio * self.proprio_scale
        scaled_command = observation.command * self.command_scale

        near_feature = features.near_feature
        far_feature = features.far_feature

        horizontal = self.horizontal_head(torch.cat([near_feature, scaled_proprio], dim=1))
        vertical = self.vertical_head(
            torch.cat([near_feature, far_feature, scaled_proprio], dim=1)
        )
        head_output = torch.cat([horizontal, vertical], dim=1)   # (B,6)=[p1..p5,p_height]
        envelope = self.mapping(head_output)                     # 米制 Ê（非 detach）

        control = torch.cat(
            [near_feature, far_feature, scaled_proprio, scaled_command, envelope.detach()],
            dim=1,
        )
        action_mean = self.action_head(control)
        value = self.value_head(control)
        ground_pred = (
            None if self.ground_head is None else self.ground_head(near_feature)
        )
        return NetworkOutput(
            near_feature=near_feature,
            far_feature=far_feature,
            envelope=envelope,
            action_mean=action_mean,
            value=value,
            ground_pred=ground_pred,
        )

    def unique_parameters(self) -> tuple[nn.Parameter, ...]:
        """按 id 去重的参数视图（供优化器构建）。

        本网络各子模块只注册一次（近/远 GRU 为单一实例、多头共享调用），
        普通 self.parameters() 本无重复（测试验证）；本方法提供显式去重保证。
        """
        unique: dict[int, nn.Parameter] = {}
        for parameter in self.parameters():
            unique.setdefault(id(parameter), parameter)
        return tuple(unique.values())


# 兼容别名：M1 及历史测试/脚本沿用旧类名，指向同一类对象（类身份不变）。
JointPassageNetwork = ConfinedLocomotionNetwork
