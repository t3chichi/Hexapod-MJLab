#!/usr/bin/env python
"""Actual CUDA checks for runtime transitions, reset isolation and physical gates.

Each mode is a fresh explicit real-backend fixture. The command-period mode
is intentionally long and requires 205 natural engineering controls.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from collections.abc import Mapping

import torch

from _common import add_common_arguments, close_runtime, execute_gpu_check


MODES = (
  "transition", "partial_reset", "persistent_low", "persistent_tilt",
  "input_fault", "action_clip", "command_period",
)
CONTROL_DT_S = 0.02
PHYSICS_DT_S = 0.005
REWARD_WEIGHTS = {
  "track_lin_vel": 5.0,
  "track_ang_vel": 2.5,
  "pen_lin_vel_z": -2.0,
  "pen_ang_vel_xy": -0.5,
  "pen_gravity_xy": -5.0,
  "pen_torque": -1e-6,
  "pen_joint_vel": -1e-6,
  "pen_joint_acc": -1e-8,
  "pen_action_rate": -0.005,
  "pen_soft_joint_limit": -1.0,
  "pen_nonfoot_env_contact": -1.0,
  "pen_self_contact": -1.0,
  "pen_foot_stumble": -0.2,
  "pen_root_height": -50.0,
  "pen_foot_slip": -0.05,
  "reward_foot_air": 3.0,
  "pen_foot_clearance": -5.0,
}


def _row_tree(value, row: int):
  """Select an env row recursively while preserving non-batched metadata."""
  import dataclasses
  import numpy as np

  if torch.is_tensor(value):
    return value[row] if value.ndim and value.shape[0] == 2 else value
  if isinstance(value, np.ndarray):
    return value[row] if value.ndim and value.shape[0] == 2 else value
  if dataclasses.is_dataclass(value) and not isinstance(value, type):
    return {
      field.name: _row_tree(getattr(value, field.name), row)
      for field in dataclasses.fields(value)
    }
  if isinstance(value, Mapping):
    return {key: _row_tree(item, row) for key, item in value.items()}
  if isinstance(value, tuple):
    return tuple(_row_tree(item, row) for item in value)
  if isinstance(value, list):
    return [_row_tree(item, row) for item in value]
  return value


def _independent_contact_counts(tables: list[dict], num_envs: int, foot_names: set[str]):
  """Recount distinct qualifying contact bodies/pairs from raw solver tables."""
  import numpy as np

  nonfoot_by_step = []
  self_pairs_by_step = []
  for table in tables:
    bodies_nonfoot = [set() for _ in range(num_envs)]
    pairs_self = [set() for _ in range(num_envs)]
    for index in range(int(table["n_contacts"])):
      if float(table["dist"][index]) > 0.0:
        continue
      force_norm = float(np.linalg.norm(table["force6_world"][index, :3]))
      if not force_norm > 0.1:
        continue
      env_id = int(table["worldid"][index])
      name1 = str(table["body1_name"][index])
      name2 = str(table["body2_name"][index])
      robot1, robot2 = name1.startswith("robot/"), name2.startswith("robot/")
      foot1 = name1.rsplit("/", 1)[-1] in foot_names
      foot2 = name2.rsplit("/", 1)[-1] in foot_names
      if (robot1 and not foot1 and not robot2):
        bodies_nonfoot[env_id].add(name1)
      elif (robot2 and not foot2 and not robot1):
        bodies_nonfoot[env_id].add(name2)
      if robot1 and robot2 and int(table["body1"][index]) != int(table["body2"][index]):
        lo, hi = sorted((int(table["body1"][index]), int(table["body2"][index])))
        pairs_self[env_id].add((lo, hi))
    nonfoot_by_step.append([len(items) for items in bodies_nonfoot])
    self_pairs_by_step.append([len(items) for items in pairs_self])
  if len(nonfoot_by_step) != 4 or len(self_pairs_by_step) != 4:
    raise AssertionError("independent contact recount requires four raw substep tables")
  return (
    torch.tensor(nonfoot_by_step, dtype=torch.float32).mean(0),
    torch.tensor(self_pairs_by_step, dtype=torch.float32).mean(0),
  )


def _independent_reward_terms(record: dict, transition, plane_z: float,
                              nonfoot_counts: torch.Tensor,
                              self_pair_counts: torch.Tensor,
                              support: torch.Tensor,
                              foot_air: tuple[torch.Tensor, torch.Tensor]) -> dict[str, torch.Tensor]:
  """Re-derive every authority reward directly from sealed physical values."""
  lin, ang, gravity = record["lin_vel_b"], record["ang_vel_b"], record["gravity_b"]
  command = transition.command
  foot_force = record["foot_force_w"]
  foot_velocity = record["foot_vel_w"]
  foot_height = record["foot_pos_w"][:, :, 2] - plane_z
  motion = (torch.linalg.vector_norm(command[:, :2], dim=1) > 0.1) | (command[:, 2].abs() > 0.1)
  expected_terms = {
    "track_lin_vel": torch.exp(-((lin[:, :2] - command[:, :2]).square().sum(1)) / 0.25),
    "track_ang_vel": torch.exp(-((ang[:, 2] - command[:, 2]).square()) / 0.25),
    "pen_lin_vel_z": lin[:, 2].square(),
    "pen_ang_vel_xy": ang[:, :2].square().sum(1),
    "pen_gravity_xy": gravity[:, :2].square().sum(1),
    "pen_torque": torch.stack([step["effort"].square().sum(1) for step in record["pd"]]).mean(0),
    "pen_joint_vel": record["joint_vel"].square().sum(1),
    "pen_joint_acc": ((record["joint_vel"] - record["previous_qdot"]) / CONTROL_DT_S).square().sum(1),
    "pen_action_rate": (record["clipped_action"] - record["previous_action"]).square().sum(1),
    "pen_soft_joint_limit": (record["joint_pos"].abs() - 2.7).clamp(min=0.0).sum(1),
    "pen_nonfoot_env_contact": nonfoot_counts.to(record["joint_pos"].device),
    "pen_self_contact": self_pair_counts.to(record["joint_pos"].device),
    "pen_foot_stumble": (
      torch.linalg.vector_norm(foot_force[:, :, :2], dim=2)
      > 5.0 * foot_force[:, :, 2].abs()
    ).any(1).to(torch.float32),
    "pen_root_height": (record["root_pos_w"][:, 2] - plane_z - 0.46).square(),
    "pen_foot_slip": (foot_velocity[:, :, :2].square().sum(2) * support).sum(1),
    "pen_foot_clearance": torch.where(
      motion,
      ((0.12 - foot_height).clamp(min=0.0).square() * (~support)).sum(1),
      torch.zeros_like(command[:, 0]),
    ),
  }
  landing, consumed = foot_air
  air_per_foot = torch.where(landing, consumed - 0.25, torch.zeros_like(consumed))
  expected_terms["reward_foot_air"] = torch.where(
    motion, air_per_foot.sum(1), torch.zeros_like(command[:, 0])
  )
  return expected_terms


def _check_actual_physics_and_rewards(env, runtime, transition) -> dict:
  """External independent audit of torque, gear, contact netforce, counts and reward."""
  import mujoco
  import numpy as np

  record = transition.context_record
  mm = env.sim.mj_model
  errors: dict[str, float] = {}
  action = env.action_manager.get_term("residual_pd")
  expected_target = action.q_ref + 0.25 * record["clipped_action"].clamp(-100.0, 100.0)
  target_error = 0.0
  torque_error = 0.0
  gear_error = 0.0
  ctrl_error = 0.0
  resolved_gears = []
  for name in action._target_names:
    joint_name = "robot/" + name.split("/")[-1]
    joint_id = next(
      (index for index in range(mm.njnt) if mm.joint(index).name == joint_name), None
    )
    if joint_id is None:
      raise AssertionError(f"actual model joint missing for residual PD target {joint_name!r}")
    actuator_ids = np.flatnonzero(np.asarray(mm.actuator_trnid)[:, 0] == joint_id)
    if len(actuator_ids) != 1:
      raise AssertionError(f"expected one compiled torque actuator for {joint_name!r}")
    resolved_gears.append(float(mm.actuator_gear[int(actuator_ids[0]), 0]))
  for step in record["pd"]:
    target_error = max(target_error, float((step["target"] - expected_target).abs().max()))
    tau = 130.0 * (step["target"] - step["q"]) - 2.0 * step["qdot"]
    expected_effort = tau.clamp(-80.0, 80.0)
    torque_error = max(torque_error, float((expected_effort - step["effort"]).abs().max()))
    observed_gears = torch.as_tensor(resolved_gears, dtype=torch.float32, device=step["effort"].device)
    saved_gears = torch.as_tensor(step["gear"], dtype=torch.float32, device=step["effort"].device)
    gear_error = max(gear_error, float((saved_gears - observed_gears).abs().max()))
    ctrl_error = max(
      ctrl_error,
      float((step["ctrl"] * observed_gears - step["effort"]).abs().max()),
    )
  errors.update(target=target_error, pd_torque=torque_error, actuator_gear=gear_error, ctrl_gear=ctrl_error)

  # Rebuild the six world net-force vectors from the raw contact table using
  # the same primary-body sign convention documented by the original witness.
  context = runtime._context
  from hexapod_mjlab.common.witness import FOOT_BODIES

  foot_names = {name.split("/")[-1] for name in FOOT_BODIES}
  expected_force = np.zeros((env.num_envs, 6, 3), dtype=np.float32)
  for index in range(int(record["boundary_raw_contacts"]["n_contacts"])):
    row = int(record["boundary_raw_contacts"]["worldid"][index])
    for foot_index, body_id in enumerate(context.foot_global_ids):
      sign = int(int(record["boundary_raw_contacts"]["body1"][index]) == body_id) - int(
        int(record["boundary_raw_contacts"]["body2"][index]) == body_id
      )
      if sign:
        expected_force[row, foot_index] += (
          sign * record["boundary_raw_contacts"]["force6_world"][index, :3]
        )
  errors["boundary_netforce"] = float(
    np.max(np.abs(expected_force - record["foot_force_w"].detach().cpu().numpy()))
  )
  root_error = foot_error = foot_velocity_error = 0.0
  root_body_id = int(env.scene["robot"].indexing.root_body_id)
  for row in range(env.num_envs):
    witness_data = mujoco.MjData(mm)
    witness_data.qpos[:] = record["qpos"][row].detach().cpu().numpy()
    witness_data.qvel[:] = record["qvel"][row].detach().cpu().numpy()
    mujoco.mj_forward(mm, witness_data)
    root_error = max(
      root_error,
      float(np.max(np.abs(witness_data.xpos[root_body_id] - record["root_pos_w"][row].detach().cpu().numpy()))),
    )
    for foot_index, body_id in enumerate(context.foot_global_ids):
      foot_error = max(
        foot_error,
        float(np.max(np.abs(witness_data.xpos[body_id] - record["foot_pos_w"][row, foot_index].detach().cpu().numpy()))),
      )
      spatial_velocity = np.zeros(6, dtype=np.float64)
      mujoco.mj_objectVelocity(
        mm, witness_data, mujoco.mjtObj.mjOBJ_XBODY, int(body_id), spatial_velocity, 0
      )
      foot_velocity_error = max(
        foot_velocity_error,
        float(np.max(np.abs(spatial_velocity[3:] - record["foot_vel_w"][row, foot_index].detach().cpu().numpy()))),
      )
  errors.update(root_kinematics=root_error, foot_kinematics=foot_error,
                foot_velocity=foot_velocity_error)
  if root_error > 5e-5 or foot_error > 5e-5 or foot_velocity_error > 1e-4:
    raise AssertionError(f"independent MuJoCo boundary reconstruction failed: {errors}")
  sensor_data = record["boundary_sensor_data"]
  for foot_index, foot_name in enumerate(FOOT_BODIES):
    slot = next(
      (slot for slot in context.sensor._slots
       if slot.field_name == "force" and slot.primary_name.split("/")[-1] == foot_name.split("/")[-1]),
      None,
    )
    if slot is None:
      raise AssertionError(f"ANY/netforce primary foot sensor missing for {foot_name}")
    sensor = mm.sensor(slot.sensor_name)
    address, dimension = int(sensor.adr[0]), int(sensor.dim[0])
    if dimension != 3 or not torch.equal(
      sensor_data[:, address : address + dimension], record["foot_force_w"][:, foot_index]
    ):
      raise AssertionError(f"compiled boundary sensordata differs for {foot_name}")
  if not np.allclose(expected_force, record["foot_force_w"].detach().cpu().numpy(), atol=2e-4, rtol=1e-5):
    raise AssertionError(f"raw contact netforce independent reconstruction failed: {errors}")

  plane_ids = np.flatnonzero(np.asarray(mm.geom_type) == int(mujoco.mjtGeom.mjGEOM_PLANE))
  if len(plane_ids) != 1:
    raise AssertionError(f"expected one compiled plane for reward geometry, got {len(plane_ids)}")
  plane_z = float(mm.geom_pos[int(plane_ids[0]), 2])
  nonfoot, self_pairs = _independent_contact_counts(
    record["substep_tables"], env.num_envs, foot_names
  )
  if not torch.allclose(record["c_nonfoot_env"].detach().cpu(), nonfoot, atol=0.0, rtol=0.0):
    raise AssertionError("sealed nonfoot/environment counts differ from raw contact-table recount")
  if not torch.allclose(record["c_self"].detach().cpu(), self_pairs, atol=0.0, rtol=0.0):
    raise AssertionError("sealed self-pair counts differ from raw contact-table recount")
  support = torch.zeros((env.num_envs, 6), dtype=torch.bool, device=record["foot_force_w"].device)
  table = record["substep_tables"][-1]
  plane_geom_id = int(plane_ids[0])
  foot_global_ids = context.foot_global_ids
  for foot_index, body_id in enumerate(foot_global_ids):
    for index in range(int(table["n_contacts"])):
      matches = (
        (int(table["body1"][index]) == body_id and int(table["geom"][index, 1]) == plane_geom_id)
        or (int(table["body2"][index]) == body_id and int(table["geom"][index, 0]) == plane_geom_id)
      )
      qualifies = float(table["dist"][index]) <= 0.0 and float(
        np.linalg.norm(table["force6_world"][index, :3])
      ) > 0.1
      if matches and qualifies:
        support[int(table["worldid"][index]), foot_index] = True
  if not torch.equal(support, record["support"]):
    raise AssertionError("sealed support mask differs from independent last-substep plane-contact query")

  foot_state = env._confined_foot_air_state
  foot_air = (
    foot_state.landing.detach().clone(), foot_state.consumed_air_s.detach().clone()
  )
  expected_terms = _independent_reward_terms(
    record, transition, plane_z, nonfoot, self_pairs, support, foot_air
  )
  if set(expected_terms) != set(REWARD_WEIGHTS):
    raise AssertionError("independent authority reward term set differs from frozen authority table")
  term_errors = {}
  for name, expected in expected_terms.items():
    actual = transition.reward_raw_terms[name]
    delta = float((actual - expected).abs().max())
    term_errors[name] = delta
    if not torch.allclose(actual, expected, atol=1e-6, rtol=1e-5):
      raise AssertionError(f"independent raw reward formula failed for {name}: max_abs={delta}")
  total = torch.zeros_like(transition.reward)
  for name, weight in REWARD_WEIGHTS.items():
    total = total + expected_terms[name] * weight * CONTROL_DT_S
  errors["reward_total"] = float((total - transition.reward).abs().max())
  errors["reward_term_max_abs"] = max(term_errors.values())
  if not torch.allclose(total, transition.reward, atol=1e-6, rtol=1e-5):
    raise AssertionError(f"independent weighted-dt reward total failed: {errors}")
  if transition.reward_weights != REWARD_WEIGHTS:
    raise AssertionError("transition reward-weight identity differs from frozen authority weights")
  return errors


def _compare_unselected_row(before: dict, after: dict, key: str, row: int = 1) -> list[str]:
  from confined_locomotion.support.snapshots import tree_equal

  if key == "window":
    before_row = {
      "fields": _row_tree(before[key]["fields"], row),
      "seen_episodes": before[key]["seen_episodes"][row],
    }
    after_row = {
      "fields": _row_tree(after[key]["fields"], row),
      "seen_episodes": after[key]["seen_episodes"][row],
    }
  elif key == "ground_cache":
    before_row, after_row = before[key][row], after[key][row]
  elif key == "commands":
    before_state, after_state = before[key], after[key]
    before_row = {
      "generator": before_state["generators"][row],
      "command": before_state["command"][row],
      "steps_since_resample": before_state["steps_since_resample"][row],
      "manual": before_state["manual"][row],
      "sampled_once": before_state["sampled_once"][row],
    }
    after_row = {
      "generator": after_state["generators"][row],
      "command": after_state["command"][row],
      "steps_since_resample": after_state["steps_since_resample"][row],
      "manual": after_state["manual"][row],
      "sampled_once": after_state["sampled_once"][row],
    }
  else:
    before_row, after_row = _row_tree(before[key], row), _row_tree(after[key], row)
  return tree_equal(before_row, after_row, path=key)


def _derived_row_close(before: dict, after: dict, row: int = 1) -> tuple[bool, dict[str, float]]:
  deltas = {}
  for name, left in before["derived_robot_state"].items():
    right = after["derived_robot_state"][name]
    a, b = _row_tree(left, row), _row_tree(right, row)
    if torch.is_tensor(a) and torch.is_tensor(b) and a.is_floating_point():
      delta = float((a - b).abs().max()) if a.numel() else 0.0
      deltas[name] = delta
      if not torch.allclose(a, b, atol=1e-7, rtol=0.0):
        return False, deltas
    elif torch.is_tensor(a) and torch.is_tensor(b):
      if not torch.equal(a, b):
        return False, deltas
    elif a != b:
      return False, deltas
  return True, deltas


def _physics_row_equal(before: dict, after: dict, row: int) -> list[str]:
  import torch

  errors = []
  if before["manifest"] != after["manifest"]:
    errors.append("manifest")
  if set(before["arrays"]) != set(after["arrays"]):
    return errors + ["array_keys"]
  for name in sorted(before["arrays"]):
    left, right = before["arrays"][name], after["arrays"][name]
    if not torch.equal(left[row], right[row]):
      errors.append(name)
  return errors


def _runtime_runner(mode: str, *, grid, args):
  from confined_locomotion.support.fixtures import build_runtime

  return build_runtime(
    args.output,
    mode=mode,
    device=f"cuda:{args.gpu}",
    seed=args.seed,
    num_envs=args.num_envs,
    ground_grid=grid,
  )


def _run_transition(out: Path, device: str, recipe: dict, *, args, grid) -> dict:
  from confined_locomotion.support.observer import RuntimeStepObserver
  from confined_locomotion.support.snapshots import cpu_tree
  import warp as wp

  env, _wrapper, runtime = _runtime_runner("training", grid=grid, args=args)
  checks = {}
  try:
    prepared = runtime.prepare()
    scans_before = runtime.sensor_scan_counts.clone()
    labels = runtime.labels()
    checks["cached_prepare_identity"] = runtime.prepare() is prepared
    checks["cached_prepare_no_scan"] = torch.equal(scans_before, runtime.sensor_scan_counts)
    before_sim_steps = int(env._sim_step_counter)
    before_sim_time = wp.to_torch(env.sim.wp_data.time).detach().clone()
    with RuntimeStepObserver(runtime, out / "raw", label="transition") as observer:
      transition = runtime.step(torch.zeros((args.num_envs, 18), device=device))
    after_sim_time = wp.to_torch(env.sim.wp_data.time).detach().clone()
    record = transition.context_record
    checks["one_control_identity_advanced"] = (
      transition.control_id == 1
      and runtime.control_steps_taken == 1
      and int(env.common_step_counter) == 1
    )
    checks["exactly_four_real_solver_substeps"] = (
      len(record["pd"]) == 4 and len(record["substep_tables"]) == 4
      and int(env._sim_step_counter) - before_sim_steps == 4
      and abs(float(env.physics_dt) - PHYSICS_DT_S) <= 1e-12
      and torch.allclose(
        after_sim_time - before_sim_time,
        torch.full_like(after_sim_time, CONTROL_DT_S),
        atol=1e-7, rtol=0.0,
      )
    )
    checks["real_forward_refresh"] = (
      runtime._context.forward_count == 1
      and "cached_force_before_forward" in record
      and "boundary_sensor_data" in record
      and "boundary_raw_contacts" in record
    )
    checks["raw_and_clipped_action_recorded"] = (
      torch.equal(record["raw_action"], torch.zeros_like(record["raw_action"]))
      and torch.equal(record["clipped_action"], record["raw_action"].clamp(-100, 100))
    )
    checks["prepared_and_labels_bound_to_consumed_step"] = (
      torch.equal(transition.input.observation.episode_ids, labels["envelope"].episode_ids)
      and torch.equal(transition.input.observation.step_ids, labels["envelope"].step_ids)
      and transition.final_inputs.observation.step_ids.tolist() == [1] * args.num_envs
    )
    checks["observer_saved_full_transition"] = len(observer.records) == 1
    independent_audit = _check_actual_physics_and_rewards(env, runtime, transition)
    checks["independent_numeric_torque_gear_netforce_contact_reward_audit"] = (
      independent_audit["target"] <= 2e-4
      and independent_audit["pd_torque"] <= 2e-4
      and independent_audit["actuator_gear"] == 0.0
      and independent_audit["ctrl_gear"] <= 2e-4
      and independent_audit["boundary_netforce"] <= 2e-4
      and independent_audit["reward_total"] <= 1e-6
      and independent_audit["reward_term_max_abs"] <= 1e-6
    )
    torch.save(independent_audit, out / "independent_transition_audit.pt")
    torch.save(cpu_tree({
      "transition": transition,
      "labels_before": labels,
      "runtime_boundary": {
        "control_steps": runtime.control_steps_taken,
        "scan_counts": runtime.sensor_scan_counts,
        "next_due_scan_step": runtime.next_due_scan_step,
      },
    }), out / "transition_evidence.pt")
  finally:
    close_runtime(env, runtime)
  checks["passed"] = all(checks.values())
  return checks


def _run_partial_reset(out: Path, device: str, recipe: dict, *, args, grid) -> dict:
  from confined_locomotion.support.snapshots import (
    global_prng_snapshot,
    persistent_physics_snapshot,
    runtime_snapshot,
    tree_equal,
  )

  env, _wrapper, runtime = _runtime_runner("training_lifecycle_fixture", grid=grid, args=args)
  checks = {}
  try:
    if args.num_envs != 2:
      raise ValueError("partial_reset requires exactly two env rows to witness selected/unselected isolation")
    baseline_transition = runtime.step(torch.zeros((2, 18), device=device))
    checks["real_control_baseline_completed"] = (
      baseline_transition.control_id == 1
      and runtime.control_steps_taken == 1
      and int(env.common_step_counter) == 1
      and int(env._sim_step_counter) == 4
    )
    before = runtime_snapshot(runtime)
    before_physics = persistent_physics_snapshot(env)
    before_prng = global_prng_snapshot(device=device, runtime=runtime)
    before_episode = runtime.episode_ids.clone()
    before_steps = runtime.step_in_episode.clone()
    before_counts = runtime.sensor_scan_counts.clone()
    before_scans = runtime.scan_counters.clone()
    before_due = runtime.next_due_scan_step.clone()
    global_step_before_reset = runtime.control_steps_taken
    runtime.reset(torch.tensor([0], device=device, dtype=torch.long))
    after = runtime_snapshot(runtime)
    after_physics = persistent_physics_snapshot(env)
    after_prng = global_prng_snapshot(device=device, runtime=runtime)
    errors = _physics_row_equal(before_physics, after_physics, 1)
    checks["all_unselected_nonempty_physics_fields_bit_exact"] = not errors
    checks["physics_array_storage_addresses_unchanged"] = all(
      before_physics["manifest"][name]["ptr"] == after_physics["manifest"][name]["ptr"]
      for name in before_physics["manifest"]
    )
    checks["selected_episode_advanced_only"] = (
      int(runtime.episode_ids[0]) == int(before_episode[0]) + 1
      and int(runtime.episode_ids[1]) == int(before_episode[1])
      and int(runtime.step_in_episode[0]) == 0
      and int(runtime.step_in_episode[1]) == int(before_steps[1])
    )
    checks["persistent_scan_count_selected_plus_one"] = (
      int(runtime.sensor_scan_counts[0]) == int(before_counts[0]) + 1
      and int(runtime.sensor_scan_counts[1]) == int(before_counts[1])
    )
    checks["episode_scan_phase_selected_only"] = (
      int(runtime.scan_counters[0]) == 1
      and int(runtime.scan_counters[1]) == int(before_scans[1])
      and int(runtime.next_due_scan_step[0]) == global_step_before_reset + 5
      and int(runtime.next_due_scan_step[1]) == int(before_due[1])
    )
    isolated_exact = {}
    for key in (
      "window", "window_read", "window_source_directions", "ground_cache",
      "action_state", "foot_air_state", "termination_state", "commands",
    ):
      isolated_exact[key] = _compare_unselected_row(before, after, key, row=1)
    checks["unselected_window_ground_action_foot_and_termination_history_exact"] = (
      all(not differences for differences in isolated_exact.values())
    )
    derived_close, derived_deltas = _derived_row_close(before, after, row=1)
    checks["unselected_derived_position_and_velocity_within_1e-7"] = derived_close
    checks["global_cpu_cuda_prng_untouched"] = all(
      not tree_equal(before_prng[name], after_prng[name])
      for name in ("python", "numpy", "torch_cpu", "torch_cuda")
    )
    checks["unselected_command_stream_unchanged"] = (
      torch.equal(before["commands"]["generators"][1], after["commands"]["generators"][1])
      and torch.equal(before["commands"]["command"][1], after["commands"]["command"][1])
    )
    checks["global_control_clocks_unchanged"] = (
      runtime.control_steps_taken == before["control_steps"]
      and int(env.common_step_counter) == before["common_step_counter"]
      and int(env._sim_step_counter) == before["sim_step_counter"]
    )
    torch.save({
      "before": before,
      "after": after,
      "before_prng": before_prng,
      "after_prng": after_prng,
      "unselected_physics_differences": errors,
      "unselected_state_differences": isolated_exact,
      "derived_state_max_abs_differences": derived_deltas,
      "real_control_baseline": baseline_transition,
    }, out / "partial_reset_evidence.pt")
  finally:
    close_runtime(env, runtime)
  checks["passed"] = all(checks.values())
  return checks


def _run_persistent(out: Path, device: str, recipe: dict, *, args, grid, kind: str) -> dict:
  from confined_locomotion.support.snapshots import cpu_tree, persistent_physics_snapshot

  env, _wrapper, runtime = _runtime_runner("training_lifecycle_fixture", grid=grid, args=args)
  state = env._confined_termination_state
  robot = env.scene["robot"]
  checks = {}
  try:
    if args.num_envs != 2:
      raise ValueError(f"persistent_{kind} requires exactly two env rows")
    pose = torch.cat((robot.data.root_link_pos_w, robot.data.root_link_quat_w), dim=1).clone()
    if kind == "low":
      pose[0, 2] = 0.10
      state.preset_low_height_counts(torch.tensor([9, 0], device=device))
    else:
      angle_half = math.radians(75.0) / 2.0
      pose[0, 3:] = torch.tensor(
        [math.cos(angle_half), math.sin(angle_half), 0.0, 0.0], device=device
      )
      state.preset_tilt_counts(torch.tensor([9, 0], device=device))
    robot.write_root_link_pose_to_sim(pose[:1], env_ids=torch.tensor([0], device=device))
    env.scene.write_data_to_sim()
    env.sim.forward()
    runtime._input_cache.invalidate()
    runtime_snapshot_before_reset = None
    reset_evidence = {}
    original = runtime._reset_selected

    def observe_selected_reset(env_ids):
      nonlocal runtime_snapshot_before_reset
      runtime_snapshot_before_reset = persistent_physics_snapshot(env)
      result = original(env_ids)
      after_reset = persistent_physics_snapshot(env)
      reset_evidence["unselected_physics_differences"] = _physics_row_equal(
        runtime_snapshot_before_reset, after_reset, 1
      )
      reset_evidence["before"] = runtime_snapshot_before_reset
      reset_evidence["after"] = after_reset
      return result

    runtime._reset_selected = observe_selected_reset
    transition = runtime.step(torch.zeros((args.num_envs, 18), device=device))
    runtime._reset_selected = original
    checks["declared_counter_preset_recorded"] = (
      bool(state.presets)
      and state.presets[-1]["field"] == ("low_height_count" if kind == "low" else "tilt_count")
      and state.presets[-1]["values"] == [9, 0]
    )
    checks["selected_row_formally_terminated"] = (
      transition.terminated.tolist() == [True, False]
      and transition.truncated.tolist() == [False, False]
      and transition.reset_env_ids.tolist() == [0]
    )
    actual_gate_condition = (
      transition.context_record["root_pos_w"][0, 2]
      - float(env.sim.mj_model.geom_pos[env._confined_transition_context.plane_geom_id, 2]) < 0.20
      if kind == "low"
      else -transition.context_record["gravity_b"][0, 2] < 0.5
    )
    checks["ninth_count_fixture_plus_one_real_tenth_gate_step"] = (
      checks["declared_counter_preset_recorded"] and bool(actual_gate_condition)
    )
    checks["unselected_physics_preserved_across_selected_reset"] = (
      reset_evidence.get("unselected_physics_differences") == []
    )
    checks["selected_episode_reset_and_first_scan"] = (
      transition.final_inputs.observation.episode_ids.tolist() == [0, 0]
      and transition.next_inputs.observation.episode_ids.tolist() == [1, 0]
      and int(runtime.sensor_scan_counts[0]) == int(runtime.sensor_scan_counts[1]) + 1
      and int(runtime.scan_counters[0]) == 1
    )
    checks["runtime_reports_actual_gate_counters"] = (
      int(state.low_height_count[0]) == 0 if kind == "low"
      else int(state.tilt_count[0]) == 0
    )
    torch.save(cpu_tree({
      "transition": transition,
      "preset_history": state.presets,
      "reset_evidence": reset_evidence,
    }), out / "persistent_gate_evidence.pt")
  finally:
    runtime._reset_selected = original if "original" in locals() else runtime._reset_selected
    close_runtime(env, runtime)
  checks["passed"] = all(checks.values())
  return checks


def _run_input_fault(out: Path, device: str, recipe: dict, *, args, grid) -> dict:
  from confined_locomotion.support.snapshots import persistent_physics_snapshot, tree_equal
  from hexapod_mjlab.tasks.confined_locomotion.runtime import TrainingRuntimeFrozenError

  env, _wrapper, runtime = _runtime_runner("engineering", grid=grid, args=args)
  checks = {}
  try:
    before = persistent_physics_snapshot(env)
    action = torch.zeros((args.num_envs, 18), device=device)
    action[0, 0] = float("inf")
    try:
      runtime.step(action)
    except RuntimeError as error:
      checks["nonfinite_action_rejected"] = "NaN/Inf" in str(error)
    else:
      checks["nonfinite_action_rejected"] = False
    after = persistent_physics_snapshot(env)
    checks["physics_unchanged_before_integration"] = not tree_equal(before, after)
    checks["runtime_frozen_fault_recorded"] = bool(runtime._faulted and runtime.last_fault)
    try:
      runtime.reset(torch.tensor([0], device=device))
    except TrainingRuntimeFrozenError:
      checks["reset_cannot_recover_fault"] = True
    else:
      checks["reset_cannot_recover_fault"] = False
    fault_dirs = sorted((out / "runtime").glob("fault_*"))
    checks["nonresumable_scene_evidence_saved"] = bool(fault_dirs) and any(
      path.is_file() for path in fault_dirs[0].glob("*")
    )
    torch.save({"before": before, "after": after, "fault": runtime.last_fault}, out / "input_fault.pt")
  finally:
    close_runtime(env, runtime)
  checks["passed"] = all(checks.values())
  return checks


def _run_action_clip(out: Path, device: str, recipe: dict, *, args, grid) -> dict:
  from confined_locomotion.support.snapshots import cpu_tree

  env, _wrapper, runtime = _runtime_runner("training", grid=grid, args=args)
  checks = {}
  try:
    action = torch.zeros((args.num_envs, 18), device=device)
    action[:, 0] = torch.tensor([101.0, -101.0][:args.num_envs], device=device)
    action[:, 1] = torch.tensor([-101.0, 0.5][:args.num_envs], device=device)
    transition = runtime.step(action)
    record = transition.context_record
    checks["raw_actions_preserved"] = torch.equal(record["raw_action"], action)
    checks["execution_actions_clipped_at_actual_boundary"] = torch.equal(
      record["clipped_action"], action.clamp(-100.0, 100.0)
    )
    action_term = env.action_manager.get_term("residual_pd")
    checks["action_term_raw_and_clipped_match_sealed_transition"] = (
      torch.equal(action_term.last_raw_action, record["raw_action"])
      and torch.equal(action_term.last_clipped_action, record["clipped_action"])
    )
    torch.save(cpu_tree(transition), out / "action_clip_transition.pt")
  finally:
    close_runtime(env, runtime)
  checks["passed"] = all(checks.values())
  return checks


def _run_command_period(out: Path, device: str, recipe: dict, *, args, grid) -> dict:
  from confined_locomotion.support.snapshots import cpu_tree

  env, _wrapper, runtime = _runtime_runner("engineering", grid=grid, args=args)
  checks = {}
  completed = 0
  period_records = []
  natural_stop = False
  try:
    start_record_count = len(runtime._commands.resample_records)
    for step in range(205):
      transition = runtime.step(torch.zeros((args.num_envs, 18), device=device))
      completed = step + 1
      if step + 1 == 200:
        period_records = [
          item for item in runtime._commands.resample_records[start_record_count:]
          if item["reason"] == "period"
        ]
        checks["step_200_final_observation_has_new_command"] = torch.equal(
          transition.final_inputs.observation.command,
          runtime._commands.current_command(),
        )
      if transition.engineering_stopped:
        natural_stop = True
        break
    checks["no_natural_engineering_stop"] = not natural_stop
    checks["exactly_205_natural_controls"] = completed == 205 and not natural_stop
    checks["two_envs_resampled_once_at_200"] = (
      len(period_records) == args.num_envs
      and sorted(item["env"] for item in period_records) == list(range(args.num_envs))
    )
    checks["resample_records_carry_period_boundary_identity"] = (
      [item["episode_step"] for item in period_records] == [200] * args.num_envs
      and [item["sample_index"] for item in period_records]
      == list(range(start_record_count + 1, start_record_count + 1 + args.num_envs))
    )
    checks["raw_persistent_runtime_state_recorded"] = True
    torch.save(cpu_tree({
      "completed_controls": completed,
      "resample_records": runtime._commands.resample_records,
      "commands": runtime._commands.state_dict(),
      "sensor_scan_counts": runtime.sensor_scan_counts,
      "scan_counters": runtime.scan_counters,
      "engineering_stop": runtime.last_stop if hasattr(runtime, "last_stop") else runtime._stopped,
    }), out / "command_period_evidence.pt")
  finally:
    close_runtime(env, runtime)
  checks["passed"] = all(checks.values())
  return checks


def _parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--mode", required=True, choices=MODES)
  add_common_arguments(parser)
  return parser.parse_args(argv)


def main(argv=None) -> int:
  args = _parse_args(argv)
  from _common import configure_imports

  configure_imports()
  from confined_locomotion.support.fixtures import m12_training_grid

  grid = m12_training_grid()
  runners = {
    "transition": _run_transition,
    "partial_reset": _run_partial_reset,
    "persistent_low": lambda out, device, recipe, **kw: _run_persistent(out, device, recipe, kind="low", **kw),
    "persistent_tilt": lambda out, device, recipe, **kw: _run_persistent(out, device, recipe, kind="tilt", **kw),
    "input_fault": _run_input_fault,
    "action_clip": _run_action_clip,
    "command_period": _run_command_period,
  }
  return execute_gpu_check(
    args,
    feature="runtime",
    mode=args.mode,
    runner=lambda out, device, recipe: runners[args.mode](
      out, device, recipe, args=args, grid=grid
    ),
    grid=grid,
    extra_recipe={"declared_fixtures": {
      "mode": args.mode,
      "optimizer": False,
      "command_period_controls": 205 if args.mode == "command_period" else None,
    }},
  )


if __name__ == "__main__":
  raise SystemExit(main())
