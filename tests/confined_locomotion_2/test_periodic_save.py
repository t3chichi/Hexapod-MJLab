from dataclasses import asdict

from hexapod_mjlab.tasks.confined_locomotion_2.rl.reporting import checkpoint_due, optimizer_scalar_summary
from hexapod_mjlab.tasks.confined_locomotion_2.rl.runner import resolve_recipe
from hexapod_mjlab.tasks.confined_locomotion_2.config.flat_baseline_config import confined_locomotion_formal_rl_cfg


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


def test_official_logger_writes_once_and_uses_resumed_global_iteration(tmp_path):
    from contextlib import redirect_stdout
    import io
    import torch
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from hexapod_mjlab.common.training_logger import TrainingLogger

    logger = TrainingLogger(log_dir=tmp_path, cfg={"num_steps_per_env": 24},
                            env_cfg={}, num_envs=2, device="cpu", source_path=__file__)
    try:
        logger.begin_learn(completed_iterations=75, environment_transitions=75 * 48)
        scalars = {"Performance/collect_s": .2, "Performance/update_s": .3,
                   "Loss/policy": .4, "Reward/actual": .12, "Height/mse_m2": .02,
                   "Proprio/vx_rmse_mps": .1, "Episode/completed_count": 0.}
        output = io.StringIO()
        with redirect_stdout(output):
            logger.log_iteration(completed_iteration=76, target_iteration=76, scalars=scalars,
                                 learning_rate=.001, action_std=torch.ones(18))
        assert "Learning iteration 75/76" in output.getvalue()
        assert "Mean action std" in output.getvalue() and "ETA:" in output.getvalue()
        assert "Mean reward:" not in output.getvalue()  # No fabricated completed episodes.
        assert logger.tot_timesteps == 76 * 48
        assert logger.tot_time == .5
        events = EventAccumulator(str(tmp_path / "tensorboard")).Reload()
        for tag in ("Loss/policy", "Reward/actual", "Height/mse_m2", "Proprio/vx_rmse_mps"):
            points = events.Scalars(tag)
            assert len(points) == 1 and points[0].step == 75
        assert not logger.ep_extras
    finally:
        logger.close()
