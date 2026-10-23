"""Mirror configuration and explicit optimizer-probe contracts."""
from __future__ import annotations

import math
import traceback

import torch

from hexapod_mjlab.rl.confined_locomotion.contracts import (
  EnvelopeLabel,
  JOINT_DIM,
  OnlineObservation,
)
from hexapod_mjlab.rl.confined_locomotion.distribution import (
  IndependentDiagonalNormalProvider,
  derive_action_generator_seed,
)
from hexapod_mjlab.rl.confined_locomotion.network import ConfinedLocomotionNetwork
from hexapod_mjlab.rl.confined_locomotion.policy import ConfinedPolicyAdapter
from hexapod_mjlab.rl.confined_locomotion.ppo import (
  ConfinedPPOTrainer,
  MirrorTrainingConfig,
  PPOUpdateError,
)
from hexapod_mjlab.rl.confined_locomotion.symmetry import (
  GridPermutation,
  JointPermutation,
)
from hexapod_mjlab.rl.confined_locomotion.training_rollout import (
  RolloutStepSample,
  TrainingRolloutBuffer,
)


_MOUNT_POS = torch.tensor([0.645, 0.0, -0.165], dtype=torch.float32)
_MOUNT_QUAT = torch.tensor([0.0, -0.0449, 0.0, 0.99899], dtype=torch.float32)
_MOUNT_QUAT = _MOUNT_QUAT / torch.linalg.vector_norm(_MOUNT_QUAT)
_GRID = GridPermutation.from_axes((0.1, 0.33333334, 0.56666666, 0.8), (-0.3, 0.0, 0.3))
_JOINTS = tuple(
  f"{leg}_{kind}"
  for leg in ("RF", "LF", "RM", "LM", "RB", "LB")
  for kind in ("HAA", "HFE", "KFE")
)
_PERMUTATION = JointPermutation.from_joint_names(_JOINTS)


def _make_adapter() -> tuple[ConfinedPolicyAdapter, torch.Tensor]:
  torch.manual_seed(2208)
  network = ConfinedLocomotionNetwork(
    mlp_activation=torch.nn.ELU(), init_fn=None, envelope_mode="raw", ground_samples=12,
  )
  provider = IndependentDiagonalNormalProvider(
    device="cpu", generator_seed=derive_action_generator_seed(2208),
  )
  # Current mirror std must remain in provider order; an asymmetric test vector
  # catches accidental permutation of the current learned std.
  with torch.no_grad():
    provider.log_std.copy_(torch.linspace(-0.7, -0.15, JOINT_DIM))
  adapter = ConfinedPolicyAdapter(
    network=network,
    distribution_provider=provider,
    distribution_identity=provider.distribution_identity(),
  )
  return adapter, provider.log_std.detach().exp().clone()


def _sample(index: int, control_id: int, old_std: torch.Tensor) -> RolloutStepSample:
  batch = 2
  time_s = control_id * 0.02
  env_ids = torch.tensor([0, 1], dtype=torch.long)
  episode_ids = torch.zeros(batch, dtype=torch.long)
  step_ids = torch.full((batch,), control_id, dtype=torch.long)
  old_mean = torch.full((batch, JOINT_DIM), 0.01 * index)
  old_std_batch = old_std.expand(batch, -1).clone()
  raw_action = old_mean + 0.02
  old_log_prob = torch.distributions.Normal(old_mean, old_std_batch).log_prob(raw_action).sum(-1)
  observation = OnlineObservation(
    env_ids=env_ids,
    episode_ids=episode_ids,
    step_ids=step_ids,
    time_s=torch.full((batch,), time_s),
    lin_vel_b=torch.tensor([[0.2, 0.1, 0.0], [0.3, -0.2, 0.1]]),
    ang_vel_b=torch.tensor([[0.1, 0.2, 0.0], [-0.2, 0.1, 0.3]]),
    gravity_b=torch.tensor([[0.0, 0.0, -1.0]] * batch),
    joint_pos_rel_ref=torch.zeros(batch, JOINT_DIM),
    joint_vel=torch.zeros(batch, JOINT_DIM),
    prev_action=torch.zeros(batch, JOINT_DIM),
    command=torch.tensor([[0.2, 0.1, 0.3], [0.3, -0.1, -0.2]]),
  )

  near_points = torch.zeros(batch, 256, 3)
  near_points[:, 0] = _MOUNT_POS
  near_valid = torch.zeros(batch, 256, dtype=torch.bool)
  near_valid[:, 0] = True
  near_dirs = torch.zeros_like(near_points)
  near_dirs[:, 0] = torch.tensor([1.0, 0.0, 0.0])
  near_present = near_valid.clone()
  near_indices = torch.full((batch, 256), -1, dtype=torch.long)
  near_indices[:, 0] = 7

  far_points = torch.zeros(batch, 640, 3)
  far_points[:, 0] = _MOUNT_POS  # exact zero-range endpoint requires the saved ray
  far_valid = torch.zeros(batch, 640, dtype=torch.bool)
  far_valid[:, 0] = True
  far_dirs = torch.zeros_like(far_points)
  far_dirs[:, 0] = torch.tensor([0.0, 1.0, 0.0])
  far_present = far_valid.clone()
  far_indices = torch.full((batch, 640), -1, dtype=torch.long)
  far_indices[:, 0] = 19
  far_scan_ids = torch.full((batch, 640), -1, dtype=torch.long)
  far_scan_ids[:, 0] = 4
  far_times = torch.zeros(batch, 640)
  far_times[:, 0] = time_s
  envelope = EnvelopeLabel(
    env_ids=env_ids.clone(), episode_ids=episode_ids.clone(), step_ids=step_ids.clone(),
    values=torch.tensor([[0.59, 0.69, 0.59, 0.89, -0.89, 0.0]] * batch),
    valid=torch.ones(batch, dtype=torch.bool),
  )
  return RolloutStepSample(
    control_id=control_id,
    time_s=time_s,
    env_ids=env_ids,
    episode_ids=episode_ids,
    step_ids=step_ids,
    observation=observation,
    command=observation.command.clone(),
    near_points=near_points,
    near_valid=near_valid,
    far_points=far_points,
    far_valid=far_valid,
    near_filled=torch.ones(batch, dtype=torch.bool),
    near_scan_ids=torch.full((batch,), 4, dtype=torch.long),
    near_times=torch.full((batch,), time_s),
    far_point_scan_ids=far_scan_ids,
    far_point_times=far_times,
    raw_action=raw_action,
    old_log_prob=old_log_prob,
    old_mean=old_mean,
    old_std=old_std_batch,
    old_value=torch.tensor([0.2 + 0.01 * index, 0.4 + 0.01 * index]),
    reward=torch.tensor([0.05 * (index + 1), -0.03 * (index + 1)]),
    terminated=torch.zeros(batch, dtype=torch.bool),
    truncated=torch.zeros(batch, dtype=torch.bool),
    v_next=torch.tensor([0.3 + 0.01 * index, 0.2 + 0.01 * index]),
    final_episode_ids=episode_ids.clone(),
    final_step_ids=step_ids + 1,
    final_control_id=control_id + 1,
    next_episode_ids=episode_ids.clone(),
    next_step_ids=step_ids + 1,
    reset_applied=torch.zeros(batch, dtype=torch.bool),
    envelope_label=envelope,
    near_angle_keys=torch.zeros(batch, 256),
    far_angle_keys=torch.zeros(batch, 640),
    near_source_dirs_sensor=near_dirs,
    far_source_dirs_sensor=far_dirs,
    near_source_dirs_present=near_present,
    far_source_dirs_present=far_present,
    near_source_indices=near_indices,
    far_source_indices=far_indices,
  )


def cpu_mirror_update_optimizer_probe() -> None:
  adapter, current_std = _make_adapter()
  observed: list[dict] = []

  def observer(stage: str, payload: dict) -> None:
    if stage == "minibatch_forward":
      observed.append({
        **payload,
        "current_mirror_std_snapshot": payload["mirrored_distribution"].std.detach().clone(),
        "old_mirror_std_snapshot": payload["old_mirrored"].std.detach().clone(),
      })

  trainer = ConfinedPPOTrainer(
    policy_adapter=adapter,
    grid_identity_hash="symmetry-cpu-grid",
    seed=2208,
    device="cpu",
    allow_cpu=True,
    symmetry_config=MirrorTrainingConfig(enabled=True),
    joint_permutation=_PERMUTATION,
    grid_permutation=_GRID,
    mount_pos_b=_MOUNT_POS,
    mount_quat_xyzw=_MOUNT_QUAT,
    mount_optical_witness={"position_matches": True, "quaternion_xyzw_matches": True},
    observer=observer,
  )
  buffer = TrainingRolloutBuffer(
    num_envs=2, ground_samples=12, grid_identity_hash="symmetry-cpu-grid",
    device=torch.device("cpu"),
  )
  for index in range(24):
    buffer.append(_sample(index, 100 + index, current_std))
  indices = torch.tensor([0, 1], dtype=torch.long)
  advantage = buffer.advantage.reshape(-1)
  from hexapod_mjlab.rl.confined_locomotion.losses import normalize_rollout_advantage
  normalized = normalize_rollout_advantage(advantage)

  metrics = trainer._run_minibatch(
    buffer,
    indices=indices,
    normalized_advantage=normalized,
    frames=(),
    bucket=[],
    supervision_size=0,
    epoch=0,
    minibatch=0,
    stage="symmetry_cpu",
  )

  assert len(observed) == 1
  payload = observed[0]
  assert payload["mirror_enabled"] is True
  assert torch.equal(
    payload["mirrored_batch"].prepared.observation.env_ids,
    payload["original_batch"].prepared.observation.env_ids,
  )
  assert torch.equal(
    payload["mirrored_batch"].prepared.near_source_indices[:, 0],
    payload["original_batch"].prepared.near_source_indices[:, 0],
  )
  assert torch.equal(
    payload["mirrored_batch"].prepared.far_point_scan_ids[:, 0],
    payload["original_batch"].prepared.far_point_scan_ids[:, 0],
  )
  assert torch.equal(
    payload["current_mirror_std_snapshot"],
    current_std.expand(2, -1),
  )
  assert not torch.equal(
    payload["current_mirror_std_snapshot"],
    _PERMUTATION.apply(current_std).expand(2, -1),
  )
  assert torch.equal(
    payload["old_mirror_std_snapshot"],
    _PERMUTATION.apply(payload["original_batch"].old_std),
  )
  assert metrics["old_mirror_logprob_max_abs_diff"] <= 1e-5
  assert torch.allclose(
    payload["loss_policy"],
    0.5 * (payload["loss_policy_original"] + payload["loss_policy_mirrored"]),
  )
  assert torch.allclose(
    payload["loss_value"],
    0.5 * (payload["loss_value_original"] + payload["loss_value_mirrored"]),
  )
  assert torch.allclose(
    payload["loss_envelope"],
    0.5 * (payload["loss_envelope_original"] + payload["loss_envelope_mirrored"]),
  )
  assert payload["loss_ground"].item() == 0.0
  assert payload["loss_ground_mirrored"].item() == 0.0
  assert payload["kl_event"].shape == (4,)  # two matched original/mirror events
  assert trainer.optimizer_step_count == 1  # one Adam step for the paired objective
  assert math.isfinite(metrics["loss_total"])
  assert metrics["symmetry_identity"]["mapping_schema"]
  assert trainer.config_identity()["symmetry"] == MirrorTrainingConfig(enabled=True).summary()

  # The algorithm-only resume gate rejects an on/off mismatch before touching
  # Adam, LR/counters or shuffle PRNG state.
  off_trainer = ConfinedPPOTrainer(
    policy_adapter=adapter,
    grid_identity_hash="symmetry-cpu-grid",
    seed=2208,
    device="cpu",
    allow_cpu=True,
    joint_permutation=_PERMUTATION,
    grid_permutation=_GRID,
    mount_pos_b=_MOUNT_POS,
    mount_quat_xyzw=_MOUNT_QUAT,
    mount_optical_witness={"position_matches": True, "quaternion_xyzw_matches": True},
  )
  state_before = off_trainer.state_dict()
  try:
    off_trainer.load_state_dict(trainer.state_dict())
  except PPOUpdateError as error:
    assert "配置" in str(error)
  else:
    raise AssertionError("mirror-on state resumed into mirror-off trainer")
  state_after = off_trainer.state_dict()
  assert state_after["learning_rate"] == state_before["learning_rate"]
  assert state_after["update_count"] == state_before["update_count"]
  assert state_after["optimizer_step_count"] == state_before["optimizer_step_count"]
  assert torch.equal(
    state_after["shuffle_generator_state"], state_before["shuffle_generator_state"]
  )
  assert state_after["optimizer"] == state_before["optimizer"]


def test_mirror_config_is_typed_fixed_and_defaults_off() -> None:
  off = MirrorTrainingConfig()
  assert off.enabled is False and off.mean_loss_coeff == 0.6
  assert MirrorTrainingConfig.from_dict({
    "enabled": False, "mean_loss_coeff": 0.6,
  }) == off
  try:
    MirrorTrainingConfig(enabled=True, mean_loss_coeff=0.5)
  except PPOUpdateError:
    pass
  else:
    raise AssertionError("a non-authoritative mirror coefficient was accepted")


def main() -> int:
  failures = 0
  for name, test in sorted(globals().items()):
    if not name.startswith("test_") or not callable(test):
      continue
    try:
      test()
      print("PASS", name, flush=True)
    except BaseException as error:  # noqa: BLE001 — report both checks
      failures += 1
      print("FAIL", name, repr(error), flush=True)
      traceback.print_exc()
  print(f"{1 - failures}/1 CPU checks passed", flush=True)
  return 0 if failures == 0 else 1


if __name__ == "__main__":
  import sys

  if len(sys.argv) > 1 and sys.argv[1] == "--cpu-optimizer-smoke":
    cpu_mirror_update_optimizer_probe()
    raise SystemExit(0)
  raise SystemExit(main())
