#!/usr/bin/env python
"""F2 自检：网络与形态输出（合成输入，CPU，unittest）。

覆盖真实风险反例：掩码编码与逐行真实有效序列 GRU 的参考对齐（部分有效/非前缀/
空行/不同有效点数）、填充点不作为真实零点、不重排输入、无跨前向隐藏记忆、
Softplus 启动基准初始化（两种表示）、后缘符号/Δz、raw/残差数值等价且偏置单次
叠加、非零微扰响应与极负端梯度衰减（记录非认证）、命令不进形态头、改远端不动
水平头（均用非零末层夹具避免零权重初值遮蔽依赖）、策略/价值损失截断形态头但
更新感知直连、非零包络辅助更新形态头与对应感知（远端仅经垂向头）、零末层初始
的结构性零梯度不误报冻结、地面辅助只更新近端与地面头、本体/指令倍率恰一次
（含观测字段不被就地修改）、必需配置缺失/非法拒绝、输入契约边界（NaN/无效槽
非零/dtype/形状/批量）、参数无重复、保存重载一致、训练/推理一致、单样本与
batch 一致、任意/重复 env_id 行独立。

合成夹具声明：动作/价值 MLP 激活固定用 torch.tanh、init_fn 用 None 或显式
constant_ 回调、形态末层非零夹具用固定种子 randn——均为本检查的合成选择，
不是任务默认；正式激活/初始化/动作分布/loss 形式与权重均未确认。本检查由 GLM
编写，本机 headless 无审批客户端未运行；由 Codex 在交付后执行，失败以非零
退出传播。合成夹具不代表任务默认配置，不含真实训练结论。

修正轮（2026-10-05）：真实首跑 25 项中 23 通过、2 项失败——①sigmoid(b) 闭式误写为
1−exp(−b)（注释与值不一致，正确为 1/(1+exp(−b))=1−1/e）；②ground_pred 相同行比较
误用 torch.equal 逐位门（浮点矩阵乘跨行可能有末位差异）。记录见
logs/joint_passage/framework/20261005_build/F2_tests.log，两项失败记录保留、未删除
对应检查。并按独立反例 F2_independent_review_initial.json 修正/新增：③模块型激活
（如 nn.PReLU）按头深拷贝，动作/价值头参数不交叉且策略反传不更新价值头任何参数；
④envelope_mode 经 _extra_state 随 checkpoint 保存，跨模式/缺元数据重载拒绝。
修正代码未运行，由 Codex 复跑。
"""

import copy
import inspect
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import torch
import torch.nn as nn

from hexapod_mjlab.rl.joint_passage.contracts import (
    COMMAND_DIM,
    ContractError,
    ENVELOPE_DIM,
    JOINT_DIM,
    NEAR_POINTS,
    PROPRIO_DIM,
    OnlineObservation,
)
from hexapod_mjlab.rl.joint_passage.mapping import (
    P0_STARTUP,
    SOFTPLUS_INVERSE_ONE,
    EnvelopeSoftplusMapping,
)
from hexapod_mjlab.rl.joint_passage.network import (
    FAR_GRU_HIDDEN,
    FAR_SEQUENCE_POINTS,
    MORPHOLOGY_HIDDEN_DIMS,
    NEAR_GRU_HIDDEN,
    JointPassageNetwork,
    MaskedPointSequenceGRU,
    NetworkOutput,
    build_proprio_scale,
)

# 启动基准物理输出（docs/tasks/confined_locomotion/平地学习与基准测量.md 第 2 节）。
BASELINE = torch.tensor([0.59, 0.69, 0.59, 0.89, -0.89, 0.0], dtype=torch.float32)


def make_points(batch, count, valid_counts, *, scattered_rows=(), radius=1.0, seed=7):
    """确定性点块：每行 valid_counts[b] 个不同有效点，其余精确 0 且掩码 False。

    scattered_rows 中的行用散布（非前缀）位置：偶数索引处取前 k 个有效点，
    用于验证压紧不依赖前缀掩码；其余行用前缀。
    """
    assert len(valid_counts) == batch
    generator = torch.Generator().manual_seed(seed)
    points = torch.randn(batch, count, 3, generator=generator).mul_(0.05).add_(radius)
    valid = torch.zeros(batch, count, dtype=torch.bool)
    for row, row_count in enumerate(valid_counts):
        if row_count <= 0:
            continue
        if row in scattered_rows:
            positions = torch.arange(0, count, 2)[:row_count]
        else:
            positions = torch.arange(row_count)
        valid[row, positions] = True
    points = torch.where(valid.unsqueeze(-1), points, torch.zeros((), dtype=torch.float32))
    return points.contiguous(), valid


def make_observation(batch=3, *, env_ids=(0, 1, 2), command=(0.5, -0.25, 0.8),
                     lin=(0.5, 0.25, -1.0), ang=(2.0, -4.0, 0.5), grav=(0.0, 0.0, -1.0),
                     joint_rel=0.3, joint_vel=4.0, prev_action=-7.0, episode=7):
    """确定性在线观测：字段取互不相同的常数便于倍率/接线检查（合成夹具）。"""

    def block(value, dim):
        if isinstance(value, (tuple, list)):
            tensor = torch.tensor(value, dtype=torch.float32)
            assert tensor.numel() == dim
        else:
            tensor = torch.full((dim,), float(value), dtype=torch.float32)
        return tensor.unsqueeze(0).repeat(batch, 1)

    env = torch.tensor(env_ids, dtype=torch.int64)
    assert env.numel() == batch
    return OnlineObservation(
        env_ids=env,
        episode_ids=torch.full((batch,), episode, dtype=torch.int64),
        step_ids=torch.arange(batch, dtype=torch.int64),
        time_s=torch.linspace(1.0, 2.0, batch, dtype=torch.float32),
        lin_vel_b=block(lin, 3),
        ang_vel_b=block(ang, 3),
        gravity_b=block(grav, 3),
        joint_pos_rel_ref=block(joint_rel, JOINT_DIM),
        joint_vel=block(joint_vel, JOINT_DIM),
        prev_action=block(prev_action, JOINT_DIM),
        command=block(command, 3),
    )


def make_network(*, seed=11, envelope_mode="raw", ground_samples=3, init_fn=None,
                 mlp_activation=torch.tanh, p0=P0_STARTUP, h_ref=0.46):
    """构造网络：构造参数全部显式；mlp_activation=torch.tanh 为合成夹具选择。

    envelope_mode/ground_samples/mlp_activation 的取值是本检查夹具的显式选择；
    被测构造函数本身无任何默认（见 test_required_configuration_missing_or_invalid_rejected）。
    """
    torch.manual_seed(seed)
    return JointPassageNetwork(
        mlp_activation=mlp_activation,
        init_fn=init_fn,
        envelope_mode=envelope_mode,
        ground_samples=ground_samples,
        p0=p0,
        h_ref=h_ref,
    )


def make_nonzero_terminals(model, *, std=0.05, seed=99):
    """把两个形态末层权重置为固定非零（合成夹具）：避免零权重初值遮蔽依赖检查。"""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for head in (model.horizontal_head, model.vertical_head):
            weight = torch.randn(head.terminal.weight.shape, generator=generator).mul(std)
            head.terminal.weight.copy_(weight)


def run_model(model, *, batch=3, near_counts=(24, 5, 0), far_counts=(40, 0, 13),
              env_ids=(0, 1, 2), command=(0.5, -0.25, 0.8), seed=7):
    """确定性前向夹具：近端行 0 用非前缀（散布）掩码；返回输出与全部输入。"""
    near_points, near_valid = make_points(
        batch, NEAR_POINTS, near_counts, scattered_rows=(0,), radius=1.0, seed=seed
    )
    far_points, far_valid = make_points(
        batch, FAR_SEQUENCE_POINTS, far_counts, scattered_rows=(), radius=2.0, seed=seed + 1
    )
    observation = make_observation(batch, env_ids=env_ids, command=command)
    output = model(observation, near_points, near_valid, far_points, far_valid)
    return output, (observation, near_points, near_valid, far_points, far_valid)


def grad_abs_sum(module):
    total = 0.0
    for _, parameter in module.named_parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().abs().sum())
    return total


def assert_grads_absent(test, module, label):
    """梯度为 None（无连接/截断）或精确零；零末层夹具下另见结构性零梯度用例。"""
    for name, parameter in module.named_parameters():
        if parameter.grad is not None:
            test.assertTrue(
                bool((parameter.grad == 0).all()),
                f"{label}.{name} 梯度应为 None/零，实际非零",
            )


def assert_grads_present(test, module, label):
    test.assertGreater(grad_abs_sum(module), 0.0, f"{label} 应收到非零梯度")


class TestMaskedPointSequenceGRU(unittest.TestCase):
    def test_reference_alignment_partial_nonprefix_and_empty_rows(self):
        torch.manual_seed(3)
        encoder = MaskedPointSequenceGRU(hidden_size=16)
        points, valid = make_points(4, 32, (17, 0, 5, 32), scattered_rows=(2,), seed=11)
        features = encoder(points, valid)
        self.assertEqual(tuple(features.shape), (4, 16))
        self.assertTrue(torch.equal(features[1], torch.zeros(16)))  # 空行零特征
        for row in range(4):
            if int(valid[row].sum()) == 0:
                self.assertTrue(torch.equal(features[row], torch.zeros(16)))
                continue
            row_points = points[row][valid[row]].unsqueeze(0)  # 单行真实有效序列
            _, h_n = encoder.gru(row_points)
            self.assertTrue(
                torch.allclose(features[row], h_n[0, 0], atol=1e-5),
                f"行 {row} 与逐行参考不一致",
            )

    def test_padding_is_not_real_zero_points_and_order_respected(self):
        torch.manual_seed(5)
        encoder = MaskedPointSequenceGRU(hidden_size=16)
        points, valid = make_points(1, 8, (6,), seed=13)
        base = encoder(points, valid)
        # 填充位不是真实零点：把零填槽标成有效必须真正进入递归
        valid_long = valid.clone()
        valid_long[0, 6:8] = True
        extended = encoder(points, valid_long)
        self.assertFalse(torch.allclose(base, extended))
        _, h_n = encoder.gru(points[0][valid_long[0]].unsqueeze(0))
        self.assertTrue(torch.allclose(extended[0], h_n[0, 0], atol=1e-5))
        # 不重排输入：交换前两个有效点的顺序必须改变编码
        swapped = points.clone()
        swapped[0, 0] = points[0, 1]
        swapped[0, 1] = points[0, 0]
        self.assertFalse(torch.allclose(base, encoder(swapped, valid)))

    def test_all_empty_batch_and_mixed_rows(self):
        torch.manual_seed(7)
        encoder = MaskedPointSequenceGRU(hidden_size=8)
        points, valid = make_points(2, 16, (0, 0), seed=17)
        self.assertTrue(torch.equal(encoder(points, valid), torch.zeros(2, 8)))
        points_mixed, valid_mixed = make_points(3, 16, (0, 4, 0), seed=19)
        features = encoder(points_mixed, valid_mixed)
        self.assertTrue(torch.equal(features[0], torch.zeros(8)))
        self.assertTrue(torch.equal(features[2], torch.zeros(8)))
        _, h_n = encoder.gru(points_mixed[1][valid_mixed[1]].unsqueeze(0))
        self.assertTrue(torch.allclose(features[1], h_n[0, 0], atol=1e-5))
        single = encoder(points_mixed[1:2], valid_mixed[1:2])  # B=1
        self.assertTrue(torch.allclose(single[0], features[1], atol=1e-6))

    def test_no_hidden_state_across_forwards(self):
        torch.manual_seed(9)
        encoder = MaskedPointSequenceGRU(hidden_size=12)
        points_a, valid_a = make_points(2, 16, (5, 3), seed=23)
        points_b, valid_b = make_points(2, 16, (9, 2), seed=29)
        first = encoder(points_a, valid_a)
        encoder(points_b, valid_b)  # 中间前向不得污染后续结果
        again = encoder(points_a, valid_a)
        self.assertTrue(torch.equal(first, again))
        fresh = copy.deepcopy(encoder)
        self.assertTrue(torch.equal(again, fresh(points_a, valid_a)))


class TestEnvelopeMapping(unittest.TestCase):
    def test_startup_baseline_signs_overrides_and_buffers(self):
        b = SOFTPLUS_INVERSE_ONE
        self.assertAlmostEqual(b, 0.5413248546, places=9)
        raw_mapping = EnvelopeSoftplusMapping(mode="raw")
        residual_mapping = EnvelopeSoftplusMapping(mode="residual")
        self.assertAlmostEqual(raw_mapping.head_terminal_bias, b, places=9)
        self.assertEqual(residual_mapping.head_terminal_bias, 0.0)
        batch = 4
        head_at_b = torch.full((batch, ENVELOPE_DIM), b)
        outputs = (
            raw_mapping(head_at_b),                    # raw 末层偏置 b
            residual_mapping(torch.zeros(batch, ENVELOPE_DIM)),  # 残差 u=0 加 b 一次
        )
        for envelope in outputs:
            for row in range(batch):
                self.assertTrue(torch.allclose(envelope[row], BASELINE, atol=1e-6))
            self.assertTrue(bool((envelope[:, 4] < 0).all()))  # 后缘还原负号
            self.assertTrue(bool((envelope[:, 3] > 0).all()))  # 前缘为正
        # 显式覆盖 p0/h_ref（基准切换预留）：Δz = p_height − h_ref
        custom = EnvelopeSoftplusMapping(mode="raw", p0=(1.5, 2.5, 1.5, 3.5, 3.5, 0.9), h_ref=0.9)
        self.assertTrue(
            torch.allclose(custom(head_at_b)[0],
                           torch.tensor([1.5, 2.5, 1.5, 3.5, -3.5, 0.0]), atol=1e-6)
        )
        # p0/h_ref 为 buffer：进入 state_dict 并可重载；mode 经 _extra_state 一并保存
        self.assertEqual(set(raw_mapping.state_dict().keys()), {"p0", "h_ref", "_extra_state"})
        other = EnvelopeSoftplusMapping(mode="raw", p0=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0), h_ref=1.0)
        other.load_state_dict(raw_mapping.state_dict())
        for row in range(batch):
            self.assertTrue(torch.allclose(other(head_at_b)[row], BASELINE, atol=1e-6))

    def test_raw_residual_equivalence_and_single_bias(self):
        raw_mapping = EnvelopeSoftplusMapping(mode="raw")
        residual_mapping = EnvelopeSoftplusMapping(mode="residual")
        generator = torch.Generator().manual_seed(31)
        head_output = torch.randn(5, ENVELOPE_DIM, generator=generator).mul(0.3)
        # 模块级：residual(u) ≡ raw(u + b)，b 只在残差侧加一次
        self.assertTrue(
            torch.equal(residual_mapping(head_output),
                        raw_mapping(head_output + SOFTPLUS_INVERSE_ONE))
        )
        # 网络级：同种子同非零末层权重下两模式数值等价；末层偏置结构检查
        model_raw = make_network(seed=21, envelope_mode="raw")
        model_residual = make_network(seed=21, envelope_mode="residual")
        torch.manual_seed(77)
        w_horizontal = torch.randn(ENVELOPE_DIM - 1, MORPHOLOGY_HIDDEN_DIMS[1]).mul(0.05)
        w_vertical = torch.randn(1, MORPHOLOGY_HIDDEN_DIMS[1]).mul(0.05)
        for model, bias_value in ((model_raw, SOFTPLUS_INVERSE_ONE), (model_residual, 0.0)):
            with torch.no_grad():
                model.horizontal_head.terminal.weight.copy_(w_horizontal)
                model.vertical_head.terminal.weight.copy_(w_vertical)
                model.horizontal_head.terminal.bias.fill_(bias_value)
                model.vertical_head.terminal.bias.fill_(bias_value)
        self.assertTrue(
            torch.equal(model_residual.horizontal_head.terminal.bias, torch.zeros(ENVELOPE_DIM - 1))
        )
        self.assertTrue(
            torch.allclose(model_raw.horizontal_head.terminal.bias,
                           torch.full((ENVELOPE_DIM - 1,), SOFTPLUS_INVERSE_ONE))
        )
        out_raw, _ = run_model(model_raw)
        out_residual, _ = run_model(model_residual)
        self.assertTrue(torch.allclose(out_raw.envelope, out_residual.envelope, atol=1e-6))
        self.assertTrue(torch.allclose(out_raw.action_mean, out_residual.action_mean, atol=1e-5))
        self.assertTrue(torch.allclose(out_raw.value, out_residual.value, atol=1e-5))

    def test_perturbation_response_and_negative_end_decay(self):
        mapping = EnvelopeSoftplusMapping(mode="raw")
        b = SOFTPLUS_INVERSE_ONE
        # 正确闭式：sigmoid(b)=1/(1+exp(−b))；softplus(b)=1 时亦等于 1−exp(−softplus(b))=1−1/e≈0.6321
        sigmoid_b = 1.0 / (1.0 + math.exp(-b))
        eps = 1e-3
        at_b = mapping(torch.full((1, ENVELOPE_DIM), b))
        above = mapping(torch.full((1, ENVELOPE_DIM), b + eps))
        delta = above - at_b
        signs = torch.tensor([1.0, 1.0, 1.0, 1.0, -1.0, 1.0])  # 后缘维反向
        expected = signs * (torch.tensor(P0_STARTUP) * sigmoid_b * eps)
        self.assertTrue(torch.allclose(delta, expected, atol=1e-6))
        self.assertTrue(bool((delta.abs() > 0).all()))  # 初始点非死区
        # 极负端梯度衰减：如实记录的数值性质（非认证安全，不加裁剪/solver）。
        # 语义修正轮（2026-10-05）后 forward 在此正确拒绝：low=b−40≈−39.46 的
        # p_height≈3.4e-18 远小于 ulp(0.46)，Δz=p_height−h_ref 完全消减为 −0.46、
        # 加回参考的重建高度恰为 0（与 raw=−30 反例同一算术形态），属非法物理
        # 输出。梯度性质改经 positive_quantities（与 forward 共享正量检查的专用
        # 接口，极负端正量微正且有限、合法返回）观察：Ê_j=signs_j·p_j（第 6 维
        # 另有 −h_ref 常数，导数 0），故 (p⊙signs).sum() 的梯度与 Ê.sum() 同号
        # 同值。不换激活/尺度/参考高度，不另加范围，断言保留不放宽。
        low = torch.full((1, ENVELOPE_DIM), b - 40.0, requires_grad=True)
        (mapping.positive_quantities(low) * signs).sum().backward()
        mid = torch.full((1, ENVELOPE_DIM), b, requires_grad=True)
        (mapping.positive_quantities(mid) * signs).sum().backward()
        self.assertLess(float(low.grad.abs().max()), 1e-10)
        self.assertLess(float(low.grad.abs().max()), float(mid.grad.abs().max()))

    def test_mode_metadata_saved_and_cross_mode_load_rejected(self):
        generator = torch.Generator().manual_seed(41)
        head_output = torch.randn(3, ENVELOPE_DIM, generator=generator).mul(0.2)
        raw_mapping = EnvelopeSoftplusMapping(mode="raw")
        residual_mapping = EnvelopeSoftplusMapping(mode="residual")
        # mode 随 state_dict 保存
        self.assertIn("_extra_state", raw_mapping.state_dict())
        # 同模式重载：p0/h_ref、输出与模式保持对应（两种模式各验证一次）
        fresh_raw = EnvelopeSoftplusMapping(mode="raw", p0=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0), h_ref=1.0)
        fresh_raw.load_state_dict(raw_mapping.state_dict())
        self.assertEqual(fresh_raw.mode, "raw")
        self.assertTrue(torch.allclose(fresh_raw(head_output), raw_mapping(head_output), atol=1e-7))
        fresh_residual = EnvelopeSoftplusMapping(mode="residual", p0=(2.0, 2.0, 2.0, 2.0, 2.0, 2.0), h_ref=2.0)
        fresh_residual.load_state_dict(residual_mapping.state_dict())
        self.assertEqual(fresh_residual.mode, "residual")
        self.assertTrue(
            torch.allclose(fresh_residual(head_output), residual_mapping(head_output), atol=1e-7)
        )
        # 错误模式重载明确拒绝：同一 head 输出在两种模式下物理量不同，不得静默混载
        with self.assertRaises((ContractError, RuntimeError)):
            residual_mapping.load_state_dict(raw_mapping.state_dict())
        # 缺模式元数据（修正前 checkpoint 形式）同样拒绝，不静默跳过校验
        stripped = {k: v for k, v in raw_mapping.state_dict().items() if k != "_extra_state"}
        with self.assertRaises(RuntimeError):
            EnvelopeSoftplusMapping(mode="raw").load_state_dict(stripped)
        # 网络级同样拒绝跨模式重载
        model_raw = make_network(envelope_mode="raw")
        model_residual = make_network(envelope_mode="residual")
        with self.assertRaises((ContractError, RuntimeError)):
            model_residual.load_state_dict(model_raw.state_dict())


class TestNetworkStructure(unittest.TestCase):
    def test_initial_envelope_baseline_for_any_input(self):
        # 末层零权重 + 反解偏置：初始物理输出与输入无关地等于启动基准（两模式）
        for mode in ("raw", "residual"):
            model = make_network(envelope_mode=mode, ground_samples=None)
            out, _ = run_model(
                model,
                near_counts=(NEAR_POINTS, 1, 0),
                far_counts=(FAR_SEQUENCE_POINTS, 0, 7),
            )
            for row in range(3):
                self.assertTrue(
                    torch.allclose(out.envelope[row], BASELINE, atol=1e-6),
                    f"mode={mode} row={row}: {out.envelope[row].tolist()}",
                )

    def test_callback_init_overrides_but_terminal_forced(self):
        def linear_init(module):
            if isinstance(module, nn.Linear):
                nn.init.constant_(module.weight, 0.013)
                nn.init.constant_(module.bias, 0.017)
            elif isinstance(module, nn.GRU):
                for _, parameter in module.named_parameters():
                    nn.init.constant_(parameter, 0.019)

        model = make_network(init_fn=linear_init)
        # 回调可覆盖 GRU/Linear
        self.assertTrue(
            torch.equal(model.action_head.hidden[0].weight,
                        torch.full_like(model.action_head.hidden[0].weight, 0.013))
        )
        self.assertTrue(
            torch.equal(model.near_encoder.gru.weight_ih_l0,
                        torch.full_like(model.near_encoder.gru.weight_ih_l0, 0.019))
        )
        # 形态末层在回调之后仍被强制：零权重 + raw 模式反解偏置
        for head in (model.horizontal_head, model.vertical_head):
            self.assertTrue(torch.equal(head.terminal.weight, torch.zeros_like(head.terminal.weight)))
            self.assertTrue(
                torch.allclose(head.terminal.bias,
                               torch.full_like(head.terminal.bias, SOFTPLUS_INVERSE_ONE))
            )
        out, _ = run_model(model)
        for row in range(3):
            self.assertTrue(torch.allclose(out.envelope[row], BASELINE, atol=1e-6))

    def test_scaling_applied_once_with_confirmed_factors(self):
        model = make_network(ground_samples=None)
        near_points, near_valid = make_points(
            3, NEAR_POINTS, (24, 5, 0), scattered_rows=(0,), radius=1.0, seed=7
        )
        far_points, far_valid = make_points(
            3, FAR_SEQUENCE_POINTS, (40, 0, 13), radius=2.0, seed=8
        )
        observation = make_observation(3)
        raw_snapshot = {
            name: getattr(observation, name).clone()
            for name in ("lin_vel_b", "ang_vel_b", "gravity_b", "joint_pos_rel_ref",
                         "joint_vel", "prev_action", "command")
        }
        captured_horizontal = []
        model.horizontal_head.fc1.register_forward_hook(
            lambda module, args, output: captured_horizontal.append(args[0].detach())
        )
        captured_control = []
        model.action_head.hidden[0].register_forward_hook(
            lambda module, args, output: captured_control.append(args[0].detach())
        )
        out = model(observation, near_points, near_valid, far_points, far_valid)
        # 倍率缓冲恰为已确认倍率
        self.assertTrue(torch.equal(model.proprio_scale, build_proprio_scale()))
        self.assertTrue(torch.equal(model.command_scale, torch.tensor([2.0, 2.0, 0.25])))
        expected_proprio = torch.cat([
            torch.tensor([1.0, 0.5, -2.0]),        # lin ×2
            torch.tensor([0.5, -1.0, 0.125]),      # ang ×0.25
            torch.tensor([0.0, 0.0, -1.0]),        # gravity ×1
            torch.full((JOINT_DIM,), 0.3),         # q−q_ref ×1
            torch.full((JOINT_DIM,), 0.2),         # qvel ×0.05
            torch.full((JOINT_DIM,), -7.0),        # prev_action ×1
        ])
        expected_command = torch.tensor([1.0, -0.5, 0.2])  # [vx,vy,ωz]×[2,2,0.25]
        # 水平头输入 = [近端特征, 缩放本体]
        horizontal_input = captured_horizontal[0]
        self.assertEqual(tuple(horizontal_input.shape), (3, NEAR_GRU_HIDDEN + PROPRIO_DIM))
        self.assertTrue(torch.allclose(horizontal_input[:, NEAR_GRU_HIDDEN:],
                                       expected_proprio.unsqueeze(0).expand(3, PROPRIO_DIM), atol=1e-6))
        self.assertTrue(torch.allclose(horizontal_input[:, :NEAR_GRU_HIDDEN], out.near_feature, atol=1e-6))
        # 动作头输入 = [近端, 远端, 缩放本体, 缩放指令, detach(Ê)]
        control = captured_control[0]
        self.assertEqual(tuple(control.shape), (3, NEAR_GRU_HIDDEN + FAR_GRU_HIDDEN + PROPRIO_DIM + COMMAND_DIM + ENVELOPE_DIM))
        far_end = NEAR_GRU_HIDDEN + FAR_GRU_HIDDEN
        proprio_end = far_end + PROPRIO_DIM
        command_end = proprio_end + COMMAND_DIM
        self.assertTrue(torch.allclose(control[:, :NEAR_GRU_HIDDEN], out.near_feature, atol=1e-6))
        self.assertTrue(torch.allclose(control[:, NEAR_GRU_HIDDEN:far_end], out.far_feature, atol=1e-6))
        self.assertTrue(torch.allclose(control[:, far_end:proprio_end],
                                       expected_proprio.unsqueeze(0).expand(3, PROPRIO_DIM), atol=1e-6))
        self.assertTrue(torch.allclose(control[:, proprio_end:command_end],
                                       expected_command.unsqueeze(0).expand(3, COMMAND_DIM), atol=1e-6))
        self.assertTrue(torch.equal(control[:, command_end:], out.envelope))
        # 观测字段不被就地缩放（调用方保存的原始观测不被修改）
        for name, tensor in raw_snapshot.items():
            self.assertTrue(torch.equal(getattr(observation, name), tensor), f"{name} 被就地修改")

    def test_output_structure_and_online_interface_has_no_labels(self):
        parameters = list(inspect.signature(JointPassageNetwork.forward).parameters)
        self.assertEqual(
            parameters,
            ["self", "observation", "near_points", "near_valid", "far_points", "far_valid"],
        )
        self.assertEqual(
            list(NetworkOutput.__dataclass_fields__),
            ["near_feature", "far_feature", "envelope", "action_mean", "value", "ground_pred"],
        )
        model = make_network(ground_samples=2)
        out, _ = run_model(model)
        self.assertEqual(tuple(out.near_feature.shape), (3, NEAR_GRU_HIDDEN))
        self.assertEqual(tuple(out.far_feature.shape), (3, FAR_GRU_HIDDEN))
        self.assertEqual(tuple(out.envelope.shape), (3, ENVELOPE_DIM))
        self.assertEqual(tuple(out.action_mean.shape), (3, JOINT_DIM))
        self.assertEqual(tuple(out.value.shape), (3, 1))
        self.assertEqual(tuple(out.ground_pred.shape), (3, 2))
        # Ê 为非 detach 版本（辅助损失路径）；动作/价值内部消费 detach 副本
        self.assertIsNotNone(out.envelope.grad_fn)
        self.assertTrue(out.envelope.requires_grad)

    def test_ground_head_dimension_explicit(self):
        model = make_network(ground_samples=5)
        self.assertIsInstance(model.ground_head, nn.Linear)
        self.assertEqual(model.ground_head.in_features, NEAR_GRU_HIDDEN)
        self.assertEqual(model.ground_head.out_features, 5)
        out, _ = run_model(model)
        self.assertEqual(tuple(out.ground_pred.shape), (3, 5))
        # 显式 None：不建头（不默设 187）
        model_none = make_network(ground_samples=None)
        self.assertIsNone(model_none.ground_head)
        out_none, _ = run_model(model_none)
        self.assertIsNone(out_none.ground_pred)

    def test_required_configuration_missing_or_invalid_rejected(self):
        base = {
            "mlp_activation": torch.tanh,
            "init_fn": None,
            "envelope_mode": "raw",
            "ground_samples": 3,
        }
        with self.assertRaises(TypeError):
            JointPassageNetwork()
        for key in list(base):
            kwargs = dict(base)
            kwargs.pop(key)
            with self.assertRaises(TypeError):
                JointPassageNetwork(**kwargs)
        with self.assertRaises(ContractError):
            JointPassageNetwork(**{**base, "envelope_mode": "linear"})
        for bad in (0, -2, True, 2.0, "4"):
            with self.assertRaises(ContractError):
                JointPassageNetwork(**{**base, "ground_samples": bad})
        with self.assertRaises(ContractError):
            JointPassageNetwork(**{**base, "mlp_activation": 3})
        with self.assertRaises(ContractError):
            JointPassageNetwork(**{**base, "init_fn": "reset"})
        with self.assertRaises(ContractError):
            JointPassageNetwork(**{**base, "p0": (0.59, 0.69, 0.59, 0.89, 0.89, -0.46)})
        with self.assertRaises(ContractError):
            JointPassageNetwork(**{**base, "p0": (1.0, 2.0)})
        with self.assertRaises(ContractError):
            JointPassageNetwork(**{**base, "h_ref": float("nan")})
        with self.assertRaises(TypeError):
            EnvelopeSoftplusMapping()
        with self.assertRaises(ContractError):
            EnvelopeSoftplusMapping(mode="linear")

    def test_input_contract_boundaries_rejected(self):
        model = make_network()
        _, (observation, near_points, near_valid, far_points, far_valid) = run_model(model)
        bad_nan = near_points.clone()
        bad_nan[0, 0, 1] = float("nan")  # 行 0 位置 0 为有效点
        with self.assertRaises(ContractError):
            model(observation, bad_nan, near_valid, far_points, far_valid)
        bad_padding = near_points.clone()
        bad_padding[0, -1, 0] = 0.5  # 无效槽必须精确 0
        with self.assertRaises(ContractError):
            model(observation, bad_padding, near_valid, far_points, far_valid)
        with self.assertRaises(ContractError):
            model(observation, near_points.double(), near_valid, far_points, far_valid)
        with self.assertRaises(ContractError):
            model(observation, near_points[:, :255, :], near_valid[:, :255], far_points, far_valid)
        with self.assertRaises(ContractError):
            model(observation, near_points[:2], near_valid[:2], far_points, far_valid)
        with self.assertRaises(ContractError):
            model(observation, near_points, near_valid, far_points, far_valid[:, :639])
        with self.assertRaises(ContractError):
            model({}, near_points, near_valid, far_points, far_valid)
        mapping = EnvelopeSoftplusMapping(mode="raw")
        with self.assertRaises(ContractError):
            mapping(torch.zeros(3, 5))
        with self.assertRaises(ContractError):
            mapping(torch.full((2, ENVELOPE_DIM), float("nan")))

    def test_parameters_have_no_duplicates(self):
        model = make_network(ground_samples=3)
        parameters = list(model.parameters())
        identifiers = [id(parameter) for parameter in parameters]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        unique = model.unique_parameters()
        self.assertEqual(len(parameters), len(unique))
        self.assertEqual({id(parameter) for parameter in unique}, set(identifiers))


class TestNetworkGradientBoundary(unittest.TestCase):
    def test_policy_and_value_losses_truncate_morphology_and_update_encoders(self):
        model = make_network(ground_samples=3)
        out, _ = run_model(model)
        out.action_mean.sum().backward()
        assert_grads_absent(self, model.horizontal_head, "水平头(策略)")
        assert_grads_absent(self, model.vertical_head, "垂向头(策略)")
        assert_grads_present(self, model.near_encoder, "近端GRU(策略直连)")
        assert_grads_present(self, model.far_encoder, "远端GRU(策略直连)")
        assert_grads_present(self, model.action_head, "动作头")
        assert_grads_absent(self, model.value_head, "价值头(策略)")
        assert_grads_absent(self, model.ground_head, "地面头(策略)")
        model.zero_grad(set_to_none=True)
        out_value, _ = run_model(model)
        out_value.value.sum().backward()
        assert_grads_absent(self, model.horizontal_head, "水平头(价值)")
        assert_grads_absent(self, model.vertical_head, "垂向头(价值)")
        assert_grads_present(self, model.near_encoder, "近端GRU(价值直连)")
        assert_grads_present(self, model.far_encoder, "远端GRU(价值直连)")
        assert_grads_absent(self, model.action_head, "动作头(价值)")
        assert_grads_present(self, model.value_head, "价值头")
        assert_grads_absent(self, model.ground_head, "地面头(价值)")

    def test_envelope_auxiliary_updates_morphology_heads_and_both_encoders(self):
        # 非零末层夹具：非零包络辅助误差须更新两个形态头、近端（双头）与远端（仅垂向）
        model = make_network()
        make_nonzero_terminals(model)
        out, _ = run_model(model)
        target = BASELINE.unsqueeze(0) + torch.tensor([[0.05, -0.03, 0.02, 0.07, -0.04, 0.03]])
        loss = ((out.envelope - target) ** 2).sum()
        self.assertGreater(float(loss), 0.0)
        loss.backward()
        assert_grads_present(self, model.horizontal_head, "水平头(包络辅助)")
        assert_grads_present(self, model.vertical_head, "垂向头(包络辅助)")
        assert_grads_present(self, model.near_encoder, "近端GRU(包络辅助)")
        assert_grads_present(self, model.far_encoder, "远端GRU(包络辅助经垂向)")
        assert_grads_absent(self, model.action_head, "动作头(包络辅助)")
        assert_grads_absent(self, model.value_head, "价值头(包络辅助)")
        assert_grads_absent(self, model.ground_head, "地面头(包络辅助)")

    def test_envelope_auxiliary_horizontal_only_skips_far_and_vertical(self):
        # 仅水平五维有误差：垂向头/远端编码器梯度为零（远端只连垂向头）
        model = make_network()
        make_nonzero_terminals(model)
        out, _ = run_model(model)
        target = out.envelope.detach().clone()
        target[:, :5] = target[:, :5] + 0.1
        loss = ((out.envelope - target) ** 2).sum()
        self.assertGreater(float(loss), 0.0)
        loss.backward()
        assert_grads_present(self, model.horizontal_head, "水平头(仅水平误差)")
        assert_grads_present(self, model.near_encoder, "近端GRU(仅水平误差)")
        assert_grads_absent(self, model.vertical_head, "垂向头(仅水平误差)")
        assert_grads_absent(self, model.far_encoder, "远端GRU(仅水平误差)")
        assert_grads_absent(self, model.action_head, "动作头(仅水平误差)")
        assert_grads_absent(self, model.value_head, "价值头(仅水平误差)")

    def test_zero_terminal_masks_upstream_aux_grads_but_terminal_trainable(self):
        # 默认零末层：非零误差下末层仍可训练（非冻结），上游梯度结构性为零（遮蔽而非冻结）
        model = make_network()
        out, _ = run_model(model)
        target = BASELINE.unsqueeze(0) + torch.tensor([[0.05, -0.03, 0.02, 0.07, -0.04, 0.03]])
        loss = ((out.envelope - target) ** 2).sum()
        self.assertGreater(float(loss), 0.0)
        loss.backward()
        for head, label in ((model.horizontal_head, "水平"), (model.vertical_head, "垂向")):
            self.assertIsNotNone(head.terminal.weight.grad)
            self.assertIsNotNone(head.terminal.bias.grad)
            self.assertGreater(float(head.terminal.weight.grad.abs().sum()), 0.0,
                               f"{label}末层权重梯度非零（可训练，未冻结）")
            self.assertGreater(float(head.terminal.bias.grad.abs().sum()), 0.0)
            for layer_name in ("fc1", "fc2"):
                layer = getattr(head, layer_name)
                for name, parameter in layer.named_parameters():
                    if parameter.grad is not None:
                        self.assertTrue(bool((parameter.grad == 0).all()),
                                        f"{label}.{layer_name}.{name} 应结构性零梯度")
        assert_grads_absent(self, model.near_encoder, "近端GRU(零末层遮蔽)")
        assert_grads_absent(self, model.far_encoder, "远端GRU(零末层遮蔽)")

    def test_ground_auxiliary_updates_only_near_and_ground_head(self):
        model = make_network(ground_samples=4)
        out, _ = run_model(model)
        generator = torch.Generator().manual_seed(5)
        target = torch.randn(3, 4, generator=generator).mul(0.1)
        loss = ((out.ground_pred - target) ** 2).sum()
        self.assertGreater(float(loss), 0.0)
        loss.backward()
        assert_grads_present(self, model.ground_head, "地面头(地面辅助)")
        assert_grads_present(self, model.near_encoder, "近端GRU(地面辅助)")
        assert_grads_absent(self, model.far_encoder, "远端GRU(地面辅助)")
        assert_grads_absent(self, model.horizontal_head, "水平头(地面辅助)")
        assert_grads_absent(self, model.vertical_head, "垂向头(地面辅助)")
        assert_grads_absent(self, model.action_head, "动作头(地面辅助)")
        assert_grads_absent(self, model.value_head, "价值头(地面辅助)")

    def test_module_activation_copies_are_independent_between_heads(self):
        # 独立反例：同一 nn.PReLU 实例曾被两头注册共享（参数交集 1）。
        # 修正后模块型激活按头深拷贝：两头参数集无交集、策略反传不更新价值头任何参数
        #（含激活副本）；不以 unique_parameters 去重代替头独立检查。
        model = make_network(mlp_activation=nn.PReLU())
        self.assertIsNot(model.action_head.activation, model.value_head.activation)
        action_ids = {id(p) for p in model.action_head.parameters()}
        value_ids = {id(p) for p in model.value_head.parameters()}
        self.assertEqual(action_ids & value_ids, set(), "动作/价值头参数存在交叉共享")
        action_prelu = [m for m in model.action_head.modules() if isinstance(m, nn.PReLU)]
        value_prelu = [m for m in model.value_head.modules() if isinstance(m, nn.PReLU)]
        self.assertEqual(len(action_prelu), 1)
        self.assertEqual(len(value_prelu), 1)
        self.assertIsNot(action_prelu[0], value_prelu[0])
        out, _ = run_model(model)
        out.action_mean.sum().backward()
        for name, parameter in model.value_head.named_parameters():
            if parameter.grad is not None:
                self.assertTrue(bool((parameter.grad == 0).all()),
                                f"策略损失更新了价值头参数 {name}")
        self.assertIsNotNone(action_prelu[0].weight.grad)
        self.assertGreater(float(action_prelu[0].weight.grad.abs().sum()), 0.0,
                           "动作头激活副本应收到非零梯度（未被冻结）")
        self.assertIsNone(value_prelu[0].weight.grad)


class TestNetworkConsistency(unittest.TestCase):
    def test_command_does_not_change_envelope(self):
        model = make_network()
        make_nonzero_terminals(model)  # 非零末层：依赖检查不被零权重初值遮蔽
        out_a, inputs = run_model(model, command=(0.5, -0.25, 0.8))
        observation_b = make_observation(3, env_ids=(0, 1, 2), command=(-1.0, 2.0, 3.0))
        out_b = model(observation_b, *inputs[1:])
        self.assertTrue(torch.equal(out_a.envelope, out_b.envelope))  # 指令不进形态头
        self.assertFalse(torch.equal(out_a.action_mean, out_b.action_mean))
        self.assertFalse(torch.equal(out_a.value, out_b.value))

    def test_far_change_keeps_horizontal_but_moves_vertical(self):
        model = make_network()
        make_nonzero_terminals(model)
        out_a, (observation, near_points, near_valid, _, _) = run_model(model)
        far_points_b, far_valid_b = make_points(
            3, FAR_SEQUENCE_POINTS, (30, 9, 41), scattered_rows=(2,), radius=4.0, seed=99
        )
        out_b = model(observation, near_points, near_valid, far_points_b, far_valid_b)
        self.assertTrue(torch.equal(out_a.envelope[:, :5], out_b.envelope[:, :5]))  # 水平不读远端
        self.assertFalse(torch.allclose(out_a.envelope[:, 5:], out_b.envelope[:, 5:], atol=1e-6))
        self.assertTrue(torch.equal(out_a.near_feature, out_b.near_feature))
        self.assertFalse(torch.equal(out_a.far_feature, out_b.far_feature))

    def test_train_eval_repeated_forward_and_no_hidden_memory(self):
        model = make_network()
        model.train()
        out_train, _ = run_model(model)
        model.eval()
        out_eval, _ = run_model(model)
        for field in ("near_feature", "far_feature", "envelope", "action_mean", "value"):
            self.assertTrue(torch.equal(getattr(out_train, field), getattr(out_eval, field)),
                            f"{field} 在 train/eval 间不一致")
        run_model(model, near_counts=(3, 3, 3), far_counts=(1, 1, 1))  # 中间前向
        out_again, _ = run_model(model)
        for field in ("near_feature", "far_feature", "envelope", "action_mean", "value"):
            self.assertTrue(torch.equal(getattr(out_train, field), getattr(out_again, field)),
                            f"{field} 受先前前向影响（隐藏记忆）")
        model_copy = copy.deepcopy(model)
        out_copy, _ = run_model(model_copy)
        for field in ("near_feature", "far_feature", "envelope", "action_mean", "value"):
            self.assertTrue(torch.equal(getattr(out_train, field), getattr(out_copy, field)))

    def test_single_row_matches_batch_and_env_ids_arbitrary(self):
        model = make_network()
        _, (observation, near_points, near_valid, far_points, far_valid) = run_model(
            model, env_ids=(7, 3, 7), near_counts=(12, 0, 12), far_counts=(5, 2, 5)
        )
        # 行 0/2 同环境 id 且同数据 → 输出逐位一致（env_id 不参与数值计算）
        near_points[2] = near_points[0]
        near_valid[2] = near_valid[0]
        far_points[2] = far_points[0]
        far_valid[2] = far_valid[0]
        out = model(observation, near_points, near_valid, far_points, far_valid)
        # 行 0/2 同环境 id 且同数据 → 输出数值一致。用 atol 容差而非逐位相等：
        # 同批浮点矩阵乘在不同行位置可能有末位差异（首跑 ground_pred 逐位门失败）。
        # 这是合成框架数值一致性检查，不是新增的正式科学验收门；ground 检查保留。
        for field in ("near_feature", "far_feature", "envelope", "action_mean", "value", "ground_pred"):
            self.assertTrue(
                torch.allclose(getattr(out, field)[0], getattr(out, field)[2], atol=1e-6),
                f"{field} 行0/行2 应数值一致",
            )
        # 单样本前向与 batch 对应行一致
        for row in (0, 1, 2):
            single_observation = make_observation(1, env_ids=(int(observation.env_ids[row]),))
            single = model(
                single_observation,
                near_points[row:row + 1].contiguous(),
                near_valid[row:row + 1].contiguous(),
                far_points[row:row + 1].contiguous(),
                far_valid[row:row + 1].contiguous(),
            )
            for field in ("near_feature", "far_feature", "envelope", "action_mean", "value", "ground_pred"):
                self.assertTrue(
                    torch.allclose(getattr(single, field)[0], getattr(out, field)[row], atol=1e-6),
                    f"{field} 单样本行 {row} 与 batch 不一致",
                )

    def test_state_dict_roundtrip_and_buffer_reload(self):
        model_a = make_network(seed=11, ground_samples=3)
        model_b = make_network(seed=99, ground_samples=3)
        out_a, _ = run_model(model_a)
        out_b_before, _ = run_model(model_b)
        self.assertFalse(torch.allclose(out_a.action_mean, out_b_before.action_mean))
        keys = set(model_a.state_dict().keys())
        for required in ("mapping.p0", "mapping.h_ref", "proprio_scale", "command_scale"):
            self.assertIn(required, keys)
        model_b.load_state_dict(model_a.state_dict())
        out_b, _ = run_model(model_b)
        for field in ("near_feature", "far_feature", "envelope", "action_mean", "value", "ground_pred"):
            self.assertTrue(torch.allclose(getattr(out_a, field), getattr(out_b, field), atol=1e-7),
                            f"{field} 重载后不一致")
        # 显式覆盖 p0/h_ref 时初始基准随之改变（buffer 参与前向）
        custom = make_network(p0=(1.5, 2.5, 1.5, 3.5, 3.5, 0.9), h_ref=0.9)
        out_custom, _ = run_model(custom)
        self.assertTrue(
            torch.allclose(out_custom.envelope[0],
                           torch.tensor([1.5, 2.5, 1.5, 3.5, -3.5, 0.0]), atol=1e-6)
        )


if __name__ == "__main__":
    unittest.main()
