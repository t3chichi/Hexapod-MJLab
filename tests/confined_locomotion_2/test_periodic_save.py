from dataclasses import asdict

from hexapod_mjlab.tasks.confined_locomotion_2.rl.runner import checkpoint_due, resolve_recipe, optimizer_scalar_summary
from hexapod_mjlab.tasks.confined_locomotion_2.config.flat_baseline import confined_locomotion_formal_rl_cfg


def test_periodic_checkpoint_uses_50_completed_cumulative_iterations():
    assert [i for i in range(151) if checkpoint_due(i, 50)] == [50, 100, 150]
    # Resume at 75 does not shift the next save to 125.
    assert [i for i in range(76, 151) if checkpoint_due(i, 50)] == [100, 150]
    cfg = asdict(confined_locomotion_formal_rl_cfg(seed=4, num_envs=2, max_iterations=1))
    assert resolve_recipe(cfg)["save_interval"] == 50
    for invalid in (0, -1, True, 1.5):
        try:
            resolve_recipe({**cfg, "save_interval": invalid})
        except ValueError:
            pass
        else:
            raise AssertionError("invalid checkpoint interval accepted")


def test_optimizer_chart_means_exclude_other_loss_suffixes():
    metrics = {"epoch_0_minibatch_0_loss": 2.0, "epoch_0_minibatch_1_loss": 4.0,
               "epoch_0_minibatch_0_auxiliary_loss": 200.0,
               "epoch_0_minibatch_0_kl": 0.01, "epoch_0_minibatch_0_lr": 1e-3,
               "Optimizer/preclip_grad_norm_mean": 5.0}
    charts = optimizer_scalar_summary(metrics)
    assert charts["Loss/total"] == 3.0
    assert charts["Optimizer/kl"] == 0.01
    assert charts["Optimizer/learning_rate"] == 1e-3
    assert charts["Optimizer/preclip_grad_norm_mean"] == 5.0
