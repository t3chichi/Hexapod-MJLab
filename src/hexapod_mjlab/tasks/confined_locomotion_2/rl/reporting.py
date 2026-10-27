"""Pure reporting helpers shared by the v2 temporal and legacy runners.

These functions only format already-materialized metrics and decide the
periodic checkpoint boundary. They own no runner state, device access or
training lifecycle.
"""

from __future__ import annotations

import re

__all__ = [
    "checkpoint_due",
    "format_iteration_summary",
    "optimizer_scalar_summary",
]


def checkpoint_due(iteration, interval):
    """Use completed cumulative PPO iterations, including after a resume."""
    return iteration > 0 and iteration % interval == 0


def optimizer_scalar_summary(metrics):
    """Retain full minibatch JSON while plotting concise iteration means."""
    names = {"loss": "Loss/total", "policy": "Loss/policy", "value": "Loss/value",
             "envelope": "Loss/envelope", "kl": "Optimizer/kl", "lr": "Optimizer/learning_rate"}
    values = {name: [] for name in names}
    for key, value in metrics.items():
        match = re.fullmatch(r"epoch_\d+_minibatch_\d+_(loss|policy|value|envelope|kl|lr)", key)
        if match:
            values[match.group(1)].append(float(value))
    result = {name: value for name, value in metrics.items() if "/" in name}
    result.update({names[name]: sum(items) / len(items) for name, items in values.items() if items})
    return result


def format_iteration_summary(iteration, target_iteration, num_envs, scalars):
    """Format already materialized iteration metrics without touching device state."""
    def metric(name, spec=".4f"):
        return format(scalars[name], spec) if name in scalars else "--"

    lines = [
        f"{'=' * 18} 训练轮次 {iteration}/{target_iteration} | 环境 {num_envs} {'=' * 18}",
        f"采集 {metric('Performance/collect_s', '.2f')} s | "
        f"更新 {metric('Performance/update_s', '.2f')} s | "
        f"吞吐 {metric('Performance/env_steps_per_s', '.0f')} 环境步/s",
        f"每步奖励 {metric('Reward/actual')}",
    ]
    if scalars.get("Episode/completed_count", 0) > 0:
        lines.append(
            f"完成 episode {metric('Episode/completed_count', '.0f')} | "
            f"平均回报 {metric('Episode/mean_return')} | "
            f"平均长度 {metric('Episode/mean_length_steps', '.1f')} 步")
    else:
        lines.append("完成 episode 0（本轮无结束，平均回报和长度暂缺）")
    lines.extend([
        f"损失：策略 {metric('Loss/policy')} | 价值 {metric('Loss/value')} | "
        f"包络 {metric('Loss/envelope')} | 总计 {metric('Loss/total')}",
        f"学习率 {metric('Optimizer/learning_rate', '.3e')} | "
        f"Torch 峰值显存 {metric('Memory/torch_peak_allocated_MiB', '.0f')} MiB",
    ])
    if "Height/mse_m2" in scalars:
        lines.append(f"高度：MSE {metric('Height/mse_m2', '.6f')} m² | "
                     f"加权损失 {metric('Height/weighted_loss', '.6f')}")
    if "Performance/checkpoint_s" in scalars:
        lines.append(f"周期保存 {metric('Performance/checkpoint_s', '.2f')} s")
    return "\n".join(lines)
