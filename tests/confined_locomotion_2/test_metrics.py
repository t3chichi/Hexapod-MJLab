from types import SimpleNamespace
import unittest

import torch

from hexapod_mjlab.tasks.confined_locomotion_2.rl.metrics import LearningMetrics


def test_terrain_boundary_metrics_exclude_undefined_heights_and_distinguish_timeout():
    from hexapod_mjlab.common.transition import TransitionContext
    transition = make_transition([.12, .12], [False, False], [True, False])
    valid = torch.tensor([[False] * 7, [True] * 7])
    heights = torch.tensor([[0.] * 7, [.46] + [.1] * 6])
    transition.context_record["terrain_clearances"] = (heights, valid)
    transition.diagnostics = {"termination_reasons": {
        "terrain_boundary": torch.tensor([True, False]), "timeout": torch.tensor([False, False])}}
    context = object.__new__(TransitionContext)
    context.terrain_query = object()
    env = SimpleNamespace(_confined_transition_context=context)
    metrics = LearningMetrics(2, "cpu", terrain_query=SimpleNamespace(env=env))
    metrics.begin_iteration()
    envelope = torch.tensor([[.59, .69, .59, .89, -.89, 0.]] * 2)
    metrics.observe(transition, SimpleNamespace(envelope=envelope),
                    SimpleNamespace(values=envelope, valid=torch.ones(2, dtype=torch.bool)))
    result = metrics.finish_iteration()
    assert abs(result["Proprio/root_height_m"] - .46) < 1e-6
    assert result["Proprio/root_height_valid_count"] == 1
    assert abs(result["Feet/swing_link_origin_clearance_mean_m"] - .1) < 1e-6
    assert result["Episode/truncation_reason_terrain_boundary_count"] == 1
    assert result["Episode/truncation_reason_timeout_count"] == 0


def make_transition(reward, terminated, truncated, *, near=(2, 1), far=(1, 2)):
    b = len(reward)
    return SimpleNamespace(
        command=torch.zeros(b, 3),
        reward=torch.tensor(reward, dtype=torch.float32),
        reward_raw_terms={"test_term": torch.full((b,), 2.0)},
        reward_weights={"test_term": 3.0},
        terminated=torch.tensor(terminated, dtype=torch.bool),
        truncated=torch.tensor(truncated, dtype=torch.bool),
        context_record={
            "lin_vel_b": torch.zeros(b, 3), "ang_vel_b": torch.zeros(b, 3),
            "gravity_b": torch.tensor([[0., 0., -1.]] * b),
            "root_pos_w": torch.tensor([[0., 0., .46]] * b),
            "raw_action": torch.tensor([[0., 2.]] * b),
            "clipped_action": torch.tensor([[0., 1.]] * b),
            "support": torch.tensor([[True, False, True, False, False, False]] * b),
            "foot_pos_w": torch.tensor([[[0., 0., .1]] * 6] * b),
            "foot_vel_w": torch.zeros(b, 6, 3),
        },
        input=SimpleNamespace(
            near_valid=torch.tensor([[True] * n + [False] * (4-n) for n in near[:b]]),
            far_valid=torch.tensor([[True] * n + [False] * (4-n) for n in far[:b]]),
        ),
    )


def observe(metrics, reward=(.12, .12), terminated=(False, False), truncated=(False, False)):
    t = make_transition(reward, terminated, truncated)
    output = SimpleNamespace(envelope=torch.tensor([[.59, .69, .59, .89, -.89, 0.]] * len(reward)))
    label = SimpleNamespace(values=output.envelope.clone(), valid=torch.ones(len(reward), dtype=torch.bool))
    metrics.observe(t, output, label)


class LearningMetricsTest(unittest.TestCase):
  def test_speed_errors_and_per_dimension_envelope_mae(self):
    metrics = LearningMetrics(2, "cpu")
    metrics.begin_iteration()
    t = make_transition([.12, .12], [False, False], [False, False])
    t.context_record["lin_vel_b"][:, 0] = 1.0
    t.context_record["lin_vel_b"][:, 1] = 2.0
    t.context_record["ang_vel_b"][:, 2] = .5
    predicted = torch.tensor([[.69, .69, .59, .89, -.89, .0]] * 2)
    output = SimpleNamespace(envelope=predicted)
    label = SimpleNamespace(values=torch.tensor([[.59, .69, .59, .89, -.89, .0]] * 2),
                            valid=torch.ones(2, dtype=torch.bool))
    metrics.observe(t, output, label)
    result = metrics.finish_iteration()
    self.assertAlmostEqual(result["Proprio/vx_rmse_mps"], 1.0, places=6)
    self.assertAlmostEqual(result["Proprio/vx_mae_mps"], 1.0, places=6)
    self.assertAlmostEqual(result["Proprio/vy_rmse_mps"], 2.0, places=6)
    self.assertAlmostEqual(result["Proprio/wz_mae_radps"], .5, places=6)
    self.assertAlmostEqual(result["Envelope/mae_0_m"], .1, places=6)
    self.assertAlmostEqual(result["Proprio/vx_actual_mean_mps"], 1.0, places=6)
    self.assertAlmostEqual(result["Proprio/vx_target_mean_mps"], 0.0, places=6)
    self.assertAlmostEqual(result["Proprio/tilt_deg"], 0.0, places=6)

  def test_async_done_reward_dt_and_scan_counts(self):
    metrics = LearningMetrics(2, "cpu")
    metrics.begin_iteration()
    observe(metrics)
    observe(metrics, terminated=(True, False))
    observe(metrics, truncated=(False, True))
    result = metrics.finish_iteration()
    self.assertAlmostEqual(result["RewardContribution/test_term"], .12, places=6)
    self.assertAlmostEqual(result["Reward/actual"], .12, places=6)
    self.assertEqual(result["Episode/terminated_count"], 1)
    self.assertEqual(result["Episode/truncated_count"], 1)
    self.assertEqual(result["Episode/completed_count"], 2)
    self.assertAlmostEqual(result["Episode/mean_return"], .30, places=6)
    self.assertAlmostEqual(result["Episode/mean_length_steps"], 2.5, places=6)
    self.assertAlmostEqual(result["Feet/support_count_mean"], 2, places=6)
    self.assertAlmostEqual(result["Scan/near_valid_count"], 1.5, places=6)
    self.assertAlmostEqual(result["Scan/far_valid_count"], 1.5, places=6)
    self.assertAlmostEqual(result["Action/saturation_rate"], .5, places=6)


  def test_partial_episode_state_restores_after_24_control_steps(self):
    first = LearningMetrics(2, "cpu")
    first.begin_iteration()
    for _ in range(24):
        observe(first)
    out = first.finish_iteration()
    self.assertEqual(out["Episode/completed_count"], 0)
    self.assertNotIn("Episode/mean_return", out)
    state = first.state_dict()

    resumed = LearningMetrics(2, "cpu")
    resumed.load_state_dict(state)
    resumed.begin_iteration()
    observe(resumed, terminated=(True, False))
    out = resumed.finish_iteration()
    self.assertEqual(out["Episode/completed_count"], 1)
    self.assertAlmostEqual(out["Episode/mean_return"], 25 * .12, places=6)
    self.assertAlmostEqual(out["Episode/mean_length_steps"], 25, places=6)
    self.assertEqual(resumed.episode_length.tolist(), [0, 25])


  def test_envelope_mae_is_reported_in_meters_and_masks_invalid_labels(self):
    metrics = LearningMetrics(2, "cpu")
    metrics.begin_iteration()
    t = make_transition([.12, .12], [False, False], [False, False])
    output = SimpleNamespace(envelope=torch.tensor([[.60, .69, .59, .89, -.89, .0],
                                                     [1.09, .69, .59, .89, -.89, .0]]))
    label = SimpleNamespace(values=torch.tensor([[.59, .69, .59, .89, -.89, .0]] * 2),
                            valid=torch.tensor([True, False]))
    metrics.observe(t, output, label)
    result = metrics.finish_iteration()
    self.assertEqual(result["Envelope/valid_count"], 1)
    self.assertAlmostEqual(result["Envelope/mae_0_m"], .01, places=6)
    self.assertAlmostEqual(result["Envelope/predicted_0_m"], .60, places=6)

  def test_no_valid_envelope_labels_omits_fake_means(self):
    metrics = LearningMetrics(2, "cpu")
    metrics.begin_iteration()
    t = make_transition([.12, .12], [False, False], [False, False])
    output = SimpleNamespace(envelope=torch.tensor([[.59, .69, .59, .89, -.89, .0]] * 2))
    label = SimpleNamespace(values=output.envelope.clone(), valid=torch.zeros(2, dtype=torch.bool))
    metrics.observe(t, output, label)
    result = metrics.finish_iteration()
    self.assertEqual(result["Envelope/valid_count"], 0)
    self.assertNotIn("Envelope/mae_0_m", result)

  def test_real_termination_and_landing_diagnostics(self):
    metrics = LearningMetrics(2, "cpu")
    metrics.begin_iteration()
    t = make_transition([.12, .12], [True, False], [False, False])
    t.diagnostics = {
        "termination_reasons": {"fall_low_height": torch.tensor([True, False]),
                                "fall_flipped": torch.tensor([False, False]),
                                "timeout": torch.tensor([False, False])},
        "foot_landing": torch.tensor([[True, False, False, False, False, False],
                                      [False, False, False, False, False, False]]),
        "foot_landing_air_time_s": torch.tensor([[.42, 0., 0., 0., 0., 0.], [0.] * 6]),
    }
    observe_output = SimpleNamespace(envelope=torch.tensor([[.59, .69, .59, .89, -.89, 0.]] * 2))
    label = SimpleNamespace(values=observe_output.envelope.clone(), valid=torch.tensor([True, True]))
    metrics.observe(t, observe_output, label)
    result = metrics.finish_iteration()
    self.assertEqual(result["Episode/termination_reason_fall_low_height_count"], 1)
    self.assertEqual(result["Episode/termination_reason_fall_flipped_count"], 0)
    self.assertEqual(result["Episode/truncation_reason_timeout_count"], 0)
    self.assertEqual(result["Feet/landing_event_count"], 1)
    self.assertAlmostEqual(result["Feet/landing_air_time_mean_s"], .42, places=6)


  def test_load_state_rejects_wrong_shape_or_nonfinite_values(self):
    metrics = LearningMetrics(2, "cpu")
    state = metrics.state_dict()
    state["episode_return"] = torch.tensor([0., float("nan")])
    with self.assertRaisesRegex(ValueError, "invalid learning metrics"):
        metrics.load_state_dict(state)


if __name__ == "__main__":
    unittest.main()


# tests/confined_locomotion_2/run_checks.py discovers dependency-free module
# functions rather than unittest.TestCase methods.
def test_async_done_reward_dt_and_scan_counts():
    LearningMetricsTest().test_async_done_reward_dt_and_scan_counts()


def test_partial_episode_state_restores_after_24_control_steps():
    LearningMetricsTest().test_partial_episode_state_restores_after_24_control_steps()


def test_speed_errors_and_per_dimension_envelope_mae():
    LearningMetricsTest().test_speed_errors_and_per_dimension_envelope_mae()


def test_envelope_mae_is_reported_in_meters_and_masks_invalid_labels():
    LearningMetricsTest().test_envelope_mae_is_reported_in_meters_and_masks_invalid_labels()


def test_no_valid_envelope_labels_omits_fake_means():
    LearningMetricsTest().test_no_valid_envelope_labels_omits_fake_means()


def test_real_termination_and_landing_diagnostics():
    LearningMetricsTest().test_real_termination_and_landing_diagnostics()


def test_load_state_rejects_wrong_shape_or_nonfinite_values():
    LearningMetricsTest().test_load_state_rejects_wrong_shape_or_nonfinite_values()
