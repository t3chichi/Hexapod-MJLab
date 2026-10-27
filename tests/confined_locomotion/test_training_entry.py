"""Official registration and task-entry CPU contracts (no GPU claim).

These checks cover CPU-checkable entry contracts without simulator
construction. The official-chain GPU preflight and execution are covered by
the explicit GPU suite; a natural fall there is an honest not-passed result,
not a failure of these CPU contracts.

Checks:

* registry: the SAME id registered once with the independent training env cfg
  (formal rewards/terminations, Null commands, auto_reset=False,
  nan_policy='error', 64 = original registration scale only), the original
  play cfg, the guarded full rl_cfg and the task runner; deep-copied loads
  never alias the registered objects;
* default guard: the default agent cfg refuses construction BEFORE any
  Adam/network exists (Adam and the network class are patched with counting
  stubs; env=None proves the env is never read first); ``IMPLEMENTED`` is a
  ClassVar absent from ``asdict``/``replace`` (no config string can flip it);
* guarded recipe: clip_actions=None / resume=False / save_interval=0 /
  logger='tensorboard' / engineering-only grid; the explicit engineering
  factory sets engineering_enabled=True, max_iterations=1, explicit seed and
  the serializable M12 declaration (asdict round-trip; identity equal to a
  directly built GroundGridCfg.from_ranges);
* illegal configs rejected (0/over-budget iterations, wrong steps, clipped
  actions, resume, wandb logger, missing grid, tampered ConfinedPPOConfig
  values, non-CUDA device, unsupported entry kwargs, learn-arg budget);
* legacy removal: the retired El-4090-Lidar ids are absent while all three
  formal task ids remain registered.
"""
from __future__ import annotations

import dataclasses
import json
import tempfile
import traceback
import unittest
from pathlib import Path

import hexapod_mjlab.tasks  # noqa: F401  (registers every task id)
import torch
from mjlab.tasks.registry import (
  list_tasks,
  load_env_cfg,
  load_rl_cfg,
  load_runner_cls,
)

from hexapod_mjlab.rl.confined_locomotion.ppo import ConfinedPPOConfig
from hexapod_mjlab.tasks.confined_locomotion import (
  REGISTERED_TRAINING_NUM_ENVS,
  TASK_ID,
  ConfinedLocomotionRlCfg,
  ConfinedLocomotionRunner,
  confined_locomotion_engineering_rl_cfg,
  confined_locomotion_formal_rl_cfg,
  confined_locomotion_rl_cfg,
)
from support.fixtures import m12_training_grid
from hexapod_mjlab.tasks.confined_locomotion.rl import runner as runner_module
from hexapod_mjlab.common.mdp.rewards import (
  TRAINING_REWARD_WEIGHTS,
  build_training_reward_cfgs,
)
from hexapod_mjlab.common.mdp.terminations import (
  build_training_termination_cfgs,
)
_DUMMY_LOG_DIR = "/tmp/confined_entry_guard_probe_never_created"

LEGACY_TASK_IDS = ("El-4090-Lidar", "El-4090-Lidar-T0-Flat")
FORMAL_TASK_IDS = (
  TASK_ID,
  "El-4090-Confined-Locomotion-Flat_2",
  "El-4090-Confined-Locomotion-Terrain-Memory_2",
)


# ═══════════════════════════ 注册与环境绑定 ═══════════════════════════


def test_registry_same_id_registered_once_with_runner():
  tasks = list_tasks()
  assert TASK_ID in tasks, tasks
  assert tasks.count(TASK_ID) == 1
  assert load_runner_cls(TASK_ID) is ConfinedLocomotionRunner


def test_registered_env_cfg_is_training_cfg():
  cfg = load_env_cfg(TASK_ID)
  assert cfg.auto_reset is False
  assert cfg.commands == {}  # Null commands: the runtime owns the streams
  assert int(cfg.scene.num_envs) == REGISTERED_TRAINING_NUM_ENVS == 64
  assert cfg.observations["policy"].nan_policy == "error"
  assert set(cfg.rewards) == set(build_training_reward_cfgs())
  assert len(cfg.rewards) == len(TRAINING_REWARD_WEIGHTS) == 17
  assert set(cfg.terminations) == set(build_training_termination_cfgs())
  assert len(cfg.terminations) == 4


def test_registered_play_cfg_keeps_original_flat_factory():
  from hexapod_mjlab.tasks.confined_locomotion.flat_env_cfg import make_flat_env_cfg

  registered = load_env_cfg(TASK_ID, play=True)
  fresh = make_flat_env_cfg(play=True)
  assert int(registered.scene.num_envs) == int(fresh.scene.num_envs)
  assert set(registered.observations) == set(fresh.observations)
  assert set(registered.rewards) == set(fresh.rewards)
  assert set(registered.terminations) == set(fresh.terminations)


def test_registry_loads_are_deep_copies():
  env_cfg = load_env_cfg(TASK_ID)
  env_cfg.scene.num_envs = 2
  env_cfg.commands["injected"] = object()
  assert int(load_env_cfg(TASK_ID).scene.num_envs) == 64
  assert load_env_cfg(TASK_ID).commands == {}
  rl_cfg = load_rl_cfg(TASK_ID)
  rl_cfg.max_iterations = 999
  rl_cfg.engineering_enabled = True
  rl_cfg.ground_grid = m12_training_grid()
  assert load_rl_cfg(TASK_ID).max_iterations == 0
  assert load_rl_cfg(TASK_ID).engineering_enabled is False
  assert load_rl_cfg(TASK_ID).ground_grid.num_points == 187


# ═══════════════════════════ 守卫完整配方 ═══════════════════════════


def test_registered_rl_cfg_is_guarded_full_recipe():
  cfg = load_rl_cfg(TASK_ID)
  assert isinstance(cfg, ConfinedLocomotionRlCfg)
  assert cfg.class_name == "ConfinedLocomotionRunner"
  assert cfg.max_iterations == 0
  assert cfg.engineering_enabled is False
  assert cfg.formal_training_enabled is False
  assert cfg.formal_num_envs is None
  from hexapod_mjlab.tasks.confined_locomotion.ground_presets import s1_187_ground_grid
  assert cfg.ground_grid == s1_187_ground_grid()  # current baseline, NOT the old M12 fixture
  assert cfg.ground_grid.num_points == 187
  assert cfg.num_steps_per_env == 24
  assert cfg.clip_actions is None
  assert cfg.resume is False
  assert cfg.save_interval == 0
  assert cfg.upload_model is False
  assert tuple(cfg.wandb_tags) == ()
  assert cfg.logger == "tensorboard"  # declared value; local JSON logging only
  assert cfg.ppo_config == ConfinedPPOConfig()


def test_implemented_is_classvar_and_not_configurable():
  cfg = load_rl_cfg(TASK_ID)
  original_implemented = ConfinedLocomotionRlCfg.IMPLEMENTED
  assert isinstance(original_implemented, bool)
  assert "IMPLEMENTED" not in [f.name for f in dataclasses.fields(cfg)]
  as_dict = dataclasses.asdict(cfg)
  assert "IMPLEMENTED" not in as_dict
  with unittest.TestCase().assertRaises(TypeError):
    dataclasses.replace(cfg, IMPLEMENTED=True)  # not a field: no string/dict flip
  try:
    # Simulate the pre-acceptance gate even after the real source is unlocked.
    ConfinedLocomotionRlCfg.IMPLEMENTED = False
    forged = dict(as_dict)
    forged["IMPLEMENTED"] = True
    try:
      ConfinedLocomotionRunner(None, forged, _DUMMY_LOG_DIR, "cpu")
    except RuntimeError as err:
      assert "incomplete and guarded" in str(err)
    else:
      raise AssertionError("forged IMPLEMENTED must not enable construction")
  finally:
    ConfinedLocomotionRlCfg.IMPLEMENTED = original_implemented


def test_default_guard_rejects_before_adam_or_network():
  calls = {"adam": 0, "network": 0}
  real_adam = torch.optim.Adam
  real_network = runner_module.ConfinedLocomotionNetwork

  def counting_adam(*args, **kwargs):
    calls["adam"] += 1
    return real_adam(*args, **kwargs)

  def counting_network(*args, **kwargs):
    calls["network"] += 1
    return real_network(*args, **kwargs)

  torch.optim.Adam = counting_adam
  runner_module.ConfinedLocomotionNetwork = counting_network
  try:
    try:
      # env=None proves the env is never read/reset before the refusal
      ConfinedLocomotionRunner(
        None, dataclasses.asdict(confined_locomotion_rl_cfg()),
        _DUMMY_LOG_DIR, "cpu",
      )
    except RuntimeError as err:
      assert "incomplete and guarded" in str(err)
      assert "engineering" in str(err)
    else:
      raise AssertionError("default construction must refuse")
    assert calls == {"adam": 0, "network": 0}
  finally:
    torch.optim.Adam = real_adam
    runner_module.ConfinedLocomotionNetwork = real_network


def test_non_dict_train_cfg_refused_by_first_guard():
  with unittest.TestCase().assertRaises(RuntimeError):
    ConfinedLocomotionRunner(None, "engineering_enabled=True", _DUMMY_LOG_DIR, "cpu")


def test_unsupported_entry_kwargs_refused():
  cfg = dataclasses.asdict(confined_locomotion_engineering_rl_cfg(
    seed=20261008, ground_grid=m12_training_grid()
  ))
  try:
    ConfinedLocomotionRunner(
      None, cfg, _DUMMY_LOG_DIR, "cuda:0", registry_name="something"
    )
  except RuntimeError as err:
    assert "unsupported runner entry options" in str(err)
  else:
    raise AssertionError("unsupported kwargs must be refused explicitly")


# ═══════════════════════════ 工程配置非法拒绝 ═══════════════════════════


def _engineering_dict(**overrides):
  cfg = dataclasses.asdict(confined_locomotion_engineering_rl_cfg(
    seed=20261008, ground_grid=m12_training_grid()
  ))
  cfg.update(overrides)
  return cfg


def _expect_refusal(cfg_dict, fragment, device="cuda:0"):
  try:
    ConfinedLocomotionRunner(None, cfg_dict, _DUMMY_LOG_DIR, device)
  except RuntimeError as err:
    if fragment is not None:
      assert fragment in str(err), f"{fragment!r} not in {str(err)!r}"
  except (TypeError, ValueError) as err:
    if fragment is not None:
      assert fragment in str(err), f"{fragment!r} not in {str(err)!r}"
  else:
    raise AssertionError("illegal engineering config must be rejected")


def test_engineering_budget_zero_rejected():
  _expect_refusal(_engineering_dict(max_iterations=0), "positive")


def test_engineering_budget_negative_rejected():
  _expect_refusal(_engineering_dict(max_iterations=-1), "positive")


def test_engineering_budget_exceeded_rejected():
  _expect_refusal(_engineering_dict(max_iterations=5), "exceeds")
  _expect_refusal(_engineering_dict(max_iterations=2), "exceeds")


def test_engineering_wrong_rollout_steps_rejected():
  _expect_refusal(_engineering_dict(num_steps_per_env=23), "24")


def test_engineering_clip_actions_rejected():
  _expect_refusal(_engineering_dict(clip_actions=5.0), "clip_actions")


def test_engineering_resume_is_explicit_boolean():
  # Default is still False; only an explicit engineering cfg may request the
  # strict full loader. Refuse truthy strings before env access.
  from hexapod_mjlab.tasks.confined_locomotion.rl.runner import _resolve_engineering_cfg
  _resolve_engineering_cfg(_engineering_dict(resume=True))
  _expect_refusal(_engineering_dict(resume="yes"), "bool")


def test_engineering_wandb_logger_rejected():
  _expect_refusal(_engineering_dict(logger="wandb"), "tensorboard")


def test_engineering_missing_grid_rejected():
  _expect_refusal(_engineering_dict(ground_grid=None), "ground_grid")


def test_engineering_tampered_ppo_config_rejected():
  tampered = _engineering_dict()
  tampered["ppo_config"] = dict(tampered["ppo_config"], learning_rate=0.5)
  _expect_refusal(tampered, "ConfinedPPOConfig")


def test_engineering_cpu_device_rejected():
  _expect_refusal(_engineering_dict(), "actual CUDA", device="cpu")


def test_learn_argument_budget_contract():
  validate = runner_module.validate_engineering_learn_arguments
  validate(1, True)
  validate(1, False)
  for bad in (0, 2, 5):
    try:
      validate(bad, False)
    except RuntimeError as err:
      assert "budget" in str(err)
    else:
      raise AssertionError(f"learn budget {bad} must be rejected")
  with unittest.TestCase().assertRaises(TypeError):
    validate(True, False)  # bool is not an int here
  with unittest.TestCase().assertRaises(TypeError):
    validate(1, "yes")


# ═══════════════════════════ 工程工厂与网格声明 ═══════════════════════════


def test_engineering_factory_explicit_fields():
  grid = m12_training_grid()
  cfg = confined_locomotion_engineering_rl_cfg(seed=20261008, ground_grid=grid)
  assert cfg.engineering_enabled is True
  assert cfg.max_iterations == 1
  assert cfg.seed == 20261008
  assert cfg.ground_grid == grid
  assert cfg.ppo_config == ConfinedPPOConfig()
  with unittest.TestCase().assertRaises(ValueError):
    confined_locomotion_engineering_rl_cfg(seed=True, ground_grid=grid)  # bool is not a seed
  with unittest.TestCase().assertRaises(ValueError):
    confined_locomotion_engineering_rl_cfg(seed="20261008", ground_grid=grid)


def test_m12_fixture_grid_serializable_and_identity_stable():
  grid = m12_training_grid()
  assert grid.num_points == 12
  assert grid.nx == 4 and grid.ny == 3
  assert grid.validity_mode == "geometry_only_flat_training"
  payload = dataclasses.asdict(grid)
  assert set(payload) == {
    "x_coords", "y_coords", "validity_mode", "parallel_tol", "body_xyz_semantics",
  }
  json.dumps(payload)  # must be JSON-serializable as-is
  from hexapod_mjlab.tasks.confined_locomotion.ground import (
    VALIDITY_GEOMETRY_ONLY_FLAT_TRAINING,
    GroundGridCfg,
  )
  rebuilt = GroundGridCfg(**payload)
  assert rebuilt == grid
  direct = GroundGridCfg.from_ranges(
    (0.1, 0.8), (-0.3, 0.3), 4, 3,
    validity_mode=VALIDITY_GEOMETRY_ONLY_FLAT_TRAINING,
  )
  assert grid.identity() == direct.identity()
  assert grid.identity_hash() == direct.identity_hash()


def test_formal_factory_reconstructs_full_grid_and_confirmed_recipe():
  from hexapod_mjlab.tasks.confined_locomotion.ground import (
    GroundGridCfg,
    VALIDITY_GEOMETRY_ONLY_FLAT_TRAINING,
  )
  from hexapod_mjlab.tasks.confined_locomotion.rl.runner import _resolve_formal_cfg

  grid = GroundGridCfg(
    x_coords=(0.1, 0.5, 0.9),
    y_coords=(-0.2, 0.0, 0.2),
    validity_mode=VALIDITY_GEOMETRY_ONLY_FLAT_TRAINING,
    parallel_tol=2e-6,
  )
  cfg = confined_locomotion_formal_rl_cfg(
    seed=20261010,
    ground_grid=grid,
    num_envs=64,
    max_iterations=25,
  )
  assert cfg.formal_training_enabled is True
  assert cfg.engineering_enabled is False
  assert cfg.formal_num_envs == 64
  assert cfg.max_iterations == 25
  assert cfg.ground_grid == grid
  assert cfg.observation_perturbations is not None
  assert cfg.symmetry.enabled is True and cfg.symmetry.mean_loss_coeff == 0.6
  reconstructed = _resolve_formal_cfg(dataclasses.asdict(cfg))
  assert reconstructed["grid"].identity() == grid.identity()
  assert reconstructed["grid"].identity_hash() == grid.identity_hash()
  assert reconstructed["symmetry"].enabled is True
  assert reconstructed["observation_perturbations"] == cfg.observation_perturbations

  bounded = dataclasses.replace(cfg, engineering_enabled=True, formal_num_envs=2, max_iterations=1)
  bounded_resolved = _resolve_formal_cfg(dataclasses.asdict(bounded))
  assert bounded_resolved["engineering_validation"] is True
  with unittest.TestCase().assertRaises(ValueError):
    confined_locomotion_formal_rl_cfg(
      seed=-1, ground_grid=grid, num_envs=2, max_iterations=1
    )
  with unittest.TestCase().assertRaises(RuntimeError):
    _resolve_formal_cfg(dataclasses.asdict(
      dataclasses.replace(bounded, formal_num_envs=3)
    ))


def test_formal_official_config_preflights_without_environment_construction():
  from hexapod_mjlab.tasks.confined_locomotion.rl.entry import (
    build_formal_train_config,
    preflight_formal_train_config,
  )
  from hexapod_mjlab.tasks.confined_locomotion.ground import (
    GroundGridCfg,
    VALIDITY_GEOMETRY_ONLY_FLAT_TRAINING,
  )

  grid = GroundGridCfg(
    x_coords=(0.1, 0.8),
    y_coords=(-0.2, 0.0, 0.2),
    validity_mode=VALIDITY_GEOMETRY_ONLY_FLAT_TRAINING,
  )
  cfg = build_formal_train_config(
    seed=7, ground_grid=grid, num_envs=64, max_iterations=3,
    log_root="/tmp/formal-preflight-does-not-launch",
  )
  assert cfg.env.seed == cfg.agent.seed == 7
  assert cfg.env.scene.num_envs == cfg.agent.formal_num_envs == 64
  assert cfg.agent.max_iterations == 3
  report = preflight_formal_train_config(cfg)
  assert report["pre_env_rejection"] is True
  assert report["mode"] == "formal_training"
  assert report["ordinary_formal_launch_allowed"] is ConfinedLocomotionRlCfg.IMPLEMENTED
  assert report["seed"] == cfg.env.seed == cfg.agent.seed == 7
  assert report["num_envs"] == cfg.env.scene.num_envs == 64
  assert report["requested_new_iterations"] == 3
  assert report["learning_effect_ready"] is False

  validation = build_formal_train_config(
    seed=8, ground_grid=grid, num_envs=2, max_iterations=1,
    engineering_validation=True,
  )
  assert preflight_formal_train_config(validation)["mode"] == "formal_training"
  with unittest.TestCase().assertRaises(ValueError):
    build_formal_train_config(
      seed=8, ground_grid=grid, num_envs=3, max_iterations=1,
      engineering_validation=True,
    )
  with unittest.TestCase().assertRaises(ValueError):
    build_formal_train_config(
      seed=8, ground_grid=grid, num_envs=2, max_iterations=1,
      resume_checkpoint="/tmp/checkpoint-that-does-not-exist.pt",
      engineering_validation=True,
    )


def test_ordinary_formal_acceptance_gate_and_bounded_validation_are_distinct():
  from hexapod_mjlab.tasks.confined_locomotion.rl.entry import (
    build_formal_train_config,
    preflight_formal_train_config,
  )
  from hexapod_mjlab.tasks.confined_locomotion.ground import (
    GroundGridCfg,
    VALIDITY_GEOMETRY_ONLY_FLAT_TRAINING,
  )

  grid = GroundGridCfg(
    x_coords=(0.1, 0.8), y_coords=(-0.2, 0.0, 0.2),
    validity_mode=VALIDITY_GEOMETRY_ONLY_FLAT_TRAINING,
  )
  full = build_formal_train_config(
    seed=19, ground_grid=grid, num_envs=64, max_iterations=10,
  )
  validation = build_formal_train_config(
    seed=19, ground_grid=grid, num_envs=2, max_iterations=1,
    engineering_validation=True,
  )
  original = ConfinedLocomotionRlCfg.IMPLEMENTED
  try:
    ConfinedLocomotionRlCfg.IMPLEMENTED = False
    full_report = preflight_formal_train_config(full)
    assert full_report["ordinary_formal_launch_allowed"] is False
    assert full_report["learning_effect_ready"] is False
    validation_report = preflight_formal_train_config(validation)
    assert validation_report["mode"] == "formal_training"
    assert validation_report["engineering_validation"] is True
    assert validation_report["ordinary_formal_launch_allowed"] is False
    assert validation_report["learning_effect_ready"] is False

    ConfinedLocomotionRlCfg.IMPLEMENTED = True
    full_report = preflight_formal_train_config(full)
    assert full_report["mode"] == "formal_training"
    assert full_report["ordinary_formal_launch_allowed"] is True
    assert full_report["learning_effect_ready"] is False
    assert "deferred" in full_report["learning_effect_acceptance"]
  finally:
    ConfinedLocomotionRlCfg.IMPLEMENTED = original


def test_formal_preflight_rejects_bool_environment_seed_and_count():
  from hexapod_mjlab.tasks.confined_locomotion.rl.entry import (
    FormalTrainingConfigError,
    build_formal_train_config,
    preflight_formal_train_config,
  )
  from hexapod_mjlab.tasks.confined_locomotion.ground import (
    GroundGridCfg,
    VALIDITY_GEOMETRY_ONLY_FLAT_TRAINING,
  )

  cfg = build_formal_train_config(
    seed=1,
    ground_grid=GroundGridCfg(
      x_coords=(0.1, 0.8), y_coords=(-0.2, 0.0, 0.2),
      validity_mode=VALIDITY_GEOMETRY_ONLY_FLAT_TRAINING,
    ),
    num_envs=1,
    max_iterations=1,
    engineering_validation=True,
  )
  cfg.env.seed = True  # True compares equal to agent seed 1 without type validation.
  with unittest.TestCase().assertRaisesRegex(FormalTrainingConfigError, "env.seed"):
    preflight_formal_train_config(cfg)
  cfg.env.seed = 1
  cfg.env.scene.num_envs = True  # True compares equal to formal_num_envs 1.
  with unittest.TestCase().assertRaisesRegex(FormalTrainingConfigError, "num_envs"):
    preflight_formal_train_config(cfg)


def test_post_commit_summary_refresh_is_best_effort():
  from hexapod_mjlab.tasks.confined_locomotion.rl import entry
  from hexapod_mjlab.tasks.confined_locomotion.ground import (
    GroundGridCfg,
    VALIDITY_GEOMETRY_ONLY_FLAT_TRAINING,
  )

  grid = GroundGridCfg(
    x_coords=(0.1, 0.8), y_coords=(-0.2, 0.0, 0.2),
    validity_mode=VALIDITY_GEOMETRY_ONLY_FLAT_TRAINING,
  )
  with tempfile.TemporaryDirectory(prefix="formal-entry-summary-") as temp:
    log_root = Path(temp) / "logs"

    def official_launch(task_id, args):
      assert task_id == TASK_ID
      experiment_dir = Path(args.log_root).expanduser().resolve() / args.agent.experiment_name
      run_dir = experiment_dir / f"2026-10-10_{args.agent.run_name}"
      run_dir.mkdir(parents=True)
      checkpoint = run_dir / "final_checkpoint_full_000001.pt"
      checkpoint.write_bytes(b"committed-target")
      (run_dir / "training_summary.json").write_text(json.dumps({
        "final_checkpoint_path": None,
        "final_checkpoint_commit": {"state": "pending"},
        "environment_close": {"completed": False},
        "runtime_state": {"mode": "training"},
      }))

    original_launch = entry.launch_training
    entry.launch_training = official_launch
    try:
      result = entry.launch_formal_training(
        seed=23, ground_grid=grid, num_envs=2, max_iterations=1,
        log_root=log_root, gpu_ids=(0,), engineering_validation=True,
      )
      assert result.checkpoint_path.is_file()
      assert result.close_completed is True
      assert result.summary["environment_close"]["completed"] is False
      assert result.summary["formal_entry_verification"] == {
        "official_launch_returned": True,
        "full_checkpoint_exists": True,
        "official_close_completed": True,
        "close_record_persisted": False,
        "summary_path": str(result.run_dir / "training_summary.json"),
        "summary_read_error": None,
      }
    finally:
      entry.launch_training = original_launch


def test_close_error_freezes_all_continuation_surfaces():
  from types import SimpleNamespace

  from hexapod_mjlab.tasks.confined_locomotion.rl.checkpoint import CheckpointError
  from hexapod_mjlab.tasks.confined_locomotion.runtime import (
    TrainingRuntime,
    TrainingRuntimeFrozenError,
  )

  close_calls = []
  summary_snapshots = []

  def runtime_close_fails():
    raise RuntimeError("injected runtime close failure")

  def original_env_close():
    close_calls.append("env.close")

  runtime = SimpleNamespace(
    _closed=False,
    _faulted=False,
    _stopped=False,
    _control_steps=0,
    num_envs=1,
    _context=SimpleNamespace(phase="idle"),
    env=SimpleNamespace(
      common_step_counter=0,
      _manual_reset_pending=torch.zeros(1, dtype=torch.bool),
    ),
    close=runtime_close_fails,
  )
  runner = ConfinedLocomotionRunner.__new__(ConfinedLocomotionRunner)
  runner.runtime = runtime
  runner.env = SimpleNamespace(unwrapped=SimpleNamespace(close=lambda: None))
  runner._official_env_close = original_env_close
  runner._official_environment_closed = False
  runner._environment_close_error = None
  runner._lifecycle_phase = "ready"
  runner._write_summary_best_effort = lambda: summary_snapshots.append({
    "phase": runner._lifecycle_phase,
    "faulted": runtime._faulted,
    "closed": runtime._closed,
    "error": runner._environment_close_error,
  })

  with unittest.TestCase().assertRaisesRegex(
    RuntimeError, "injected runtime close failure"
  ):
    runner._close_official_environment()

  assert close_calls == ["env.close"]
  assert runner._official_environment_closed is True
  assert runtime._faulted is True
  assert runtime._closed is False  # runtime.close did not complete
  assert runner._lifecycle_phase == "faulted"
  assert "injected runtime close failure" in runner._environment_close_error
  assert summary_snapshots[-1] == {
    "phase": "faulted",
    "faulted": True,
    "closed": False,
    "error": runner._environment_close_error,
  }

  # The real public learn method refuses after close fault and retains its
  # normal error path without reaching an update or a second env close.
  runner.formal_training = False
  runner._log_failed = False
  runner._learn_calls = []
  runner._records = []
  runner._observer_errors = []
  runner._active_buffer = None
  runner.current_learning_iteration = 0
  runner.total_control_batches = 0
  runner.total_env_samples = 0
  runner.max_iterations = 1
  runner.trainer = SimpleNamespace(
    faulted=False,
    update_count=0,
    optimizer_step_count=0,
  )
  runner._append_jsonl_best_effort = lambda record: None
  runner._notify_observers = lambda event, payload: None
  with unittest.TestCase().assertRaisesRegex(RuntimeError, "lifecycle phase"):
    runner.learn(1)
  assert runner._lifecycle_phase == "faulted"
  assert close_calls == ["env.close"]

  # Save and load both reject at the runner's faulted phase, before touching
  # a checkpoint or inspecting simulator state.
  runner.device = "cpu"
  runner.total_env_samples = 0
  runner.total_control_batches = 0
  with unittest.TestCase().assertRaisesRegex(CheckpointError, "phase is 'faulted'"):
    runner.save(Path("/tmp/close-fault-must-not-save.pt"))
  with unittest.TestCase().assertRaisesRegex(CheckpointError, "phase is 'faulted'"):
    runner.load(Path("/tmp/close-fault-no-checkpoint.pt"))

  # The actual runtime public step gate sees the close-fault marker even
  # though _closed remains False because close() never completed.
  frozen_runtime = TrainingRuntime.__new__(TrainingRuntime)
  frozen_runtime._closed = False
  frozen_runtime._faulted = True
  frozen_runtime._stopped = False
  frozen_runtime.last_fault = None
  frozen_runtime.mode = "training"
  with unittest.TestCase().assertRaises(TrainingRuntimeFrozenError):
    frozen_runtime.step(torch.zeros((1, 18)))


def test_formal_resume_preflight_checks_v3_kind_mode_grid_seed_and_source():
  import copy
  import torch

  from hexapod_mjlab.tasks.confined_locomotion.rl.checkpoint import (
    CHECKPOINT_KIND_FULL,
    CHECKPOINT_SCHEMA,
    _source_identity,
  )
  from hexapod_mjlab.tasks.confined_locomotion.rl.entry import (
    FormalTrainingConfigError,
    build_formal_train_config,
    preflight_formal_train_config,
  )
  from hexapod_mjlab.tasks.confined_locomotion.ground import (
    GroundGridCfg,
    VALIDITY_GEOMETRY_ONLY_FLAT_TRAINING,
  )

  grid = GroundGridCfg(
    x_coords=(0.1, 0.8),
    y_coords=(-0.2, 0.0, 0.2),
    validity_mode=VALIDITY_GEOMETRY_ONLY_FLAT_TRAINING,
  )
  identity = {
    "task": TASK_ID,
    "runner_identity": "El-4090-Confined-Locomotion-Flat/ConfinedLocomotionRunner/mode=formal_training",
    "source_sha256": _source_identity(),
    "num_envs": 2,
    "env_seed": 17,
    "runtime": {"mode": "training", "seed": 17},
    "grid": {"identity": grid.identity(), "identity_hash": grid.identity_hash()},
  }
  payload = {
    "schema": CHECKPOINT_SCHEMA,
    "kind": CHECKPOINT_KIND_FULL,
    "resumable": True,
    "task": TASK_ID,
    "created": "2026-10-10T00:00:00",
    "identity": identity,
    "physics": {}, "algorithm": {}, "runtime": {}, "env_state": {},
    "action_state": {}, "foot_air_state": None, "termination_state": None,
    "prng": {}, "runner_state": {}, "manager_state": {},
  }
  with tempfile.TemporaryDirectory(prefix="formal-preflight-") as temp:
    root = Path(temp) / "runs"
    experiment = root / "confined_locomotion_flat"
    run = experiment / "2026-10-10_formal-test"
    run.mkdir(parents=True)
    target = run / "final_checkpoint_full_000001.pt"
    torch.save(payload, target)

    cfg = build_formal_train_config(
      seed=17, ground_grid=grid, num_envs=2, max_iterations=1,
      resume_checkpoint=target, engineering_validation=True,
    )
    report = preflight_formal_train_config(cfg)
    assert report["resume"]["kind"] == CHECKPOINT_KIND_FULL
    assert report["resume"]["runtime"] == {"mode": "training", "seed": 17}

    for field, mutate in (
      ("mode", lambda item: item["runtime"].update(mode="engineering")),
      ("grid", lambda item: item["grid"].update(identity_hash="other-grid")),
      ("source", lambda item: item["source_sha256"].update({"runner.py": "changed"})),
    ):
      changed = copy.deepcopy(payload)
      mutate(changed["identity"])
      torch.save(changed, target)
      invalid = build_formal_train_config(
        seed=17, ground_grid=grid, num_envs=2, max_iterations=1,
        resume_checkpoint=target, engineering_validation=True,
      )
      with unittest.TestCase().assertRaises(FormalTrainingConfigError):
        preflight_formal_train_config(invalid)
      torch.save(payload, target)

    bad_seed = build_formal_train_config(
      seed=18, ground_grid=grid, num_envs=2, max_iterations=1,
      resume_checkpoint=target, engineering_validation=True,
    )
    with unittest.TestCase().assertRaises(FormalTrainingConfigError):
      preflight_formal_train_config(bad_seed)


# ═══════════════════════════ 历史任务退出（实际枚举） ═══════════════════════════


def test_legacy_tasks_removed_and_formal_tasks_registered():
  tasks = set(list_tasks())
  assert not (set(LEGACY_TASK_IDS) & tasks), tasks
  assert set(FORMAL_TASK_IDS) <= tasks, tasks


# ═══════════════════════════ stdlib 驱动（无 pytest） ═══════════════════════════


def main() -> int:
  tests = [
    (name, obj)
    for name, obj in sorted(globals().items())
    if name.startswith("test_") and callable(obj)
  ]
  failures = []
  for name, fn in tests:
    try:
      fn()
      print("PASS", name, flush=True)
    except BaseException as err:  # noqa: BLE001 — report and continue
      failures.append(name)
      print("FAIL", name, repr(err), flush=True)
      traceback.print_exc()
  print(f"{len(tests) - len(failures)}/{len(tests)} passed", flush=True)
  return 0 if not failures else 1


if __name__ == "__main__":
  raise SystemExit(main())
