"""CPU MuJoCo geometry/interface diagnostics, never training-effect evidence.

Run directly with hp_mjlab Python or through a test runner. The .25 m
height check intentionally distinguishes a foot-only IK solution from a
valid stance: the unchanged HIP collision meshes intersect the floor at
that root height even when all feet have been lifted onto the plane.
"""

from functools import lru_cache
import hashlib
import json
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.optimize import least_squares
import torch

from mjlab.scene import Scene
from hexapod_mjlab.common.robot import JOINT_NAMES, ROBOT_XML
from hexapod_mjlab.tasks.confined_locomotion_2.components.tripod_actuator import ACTUATOR_PATH
from hexapod_mjlab.tasks.confined_locomotion_2.tripod_env_cfg import (
    collision_min_z, make_env_cfg, randomize_model, reset_height,
)

RADAR_XML_SHA256 = "4f1725aa814d279b3c907dc98889c55b59db35871d74f2baf9857487f3089dd0"


@lru_cache(maxsize=1)
def _compiled_model():
    cfg = make_env_cfg(num_envs=1)
    model = Scene(cfg.scene, device="cpu").compile()
    # Scene attachment intentionally does not inherit robot XML <option>.
    # Apply the same official MujocoCfg used by Simulation construction.
    cfg.sim.mujoco.apply(model)
    return model


def _fresh_data(model, height=None):
    data = mujoco.MjData(model)
    data.qpos[:] = 0.
    data.qpos[2] = reset_height() if height is None else height
    data.qpos[3] = 1.
    mujoco.mj_forward(model, data)
    return data


def _robot_collision_geoms(model, suffix=None):
    return [i for i in range(model.ngeom)
            if (model.geom_contype[i] or model.geom_conaffinity[i])
            and model.body(model.geom_bodyid[i]).name.startswith("robot/")
            and (suffix is None or model.body(model.geom_bodyid[i]).name.endswith(suffix))]


def test_compiled_radar_mass_reset_geometry_and_asset_unchanged():
    before = hashlib.sha256(ROBOT_XML.read_bytes()).hexdigest()
    assert before == RADAR_XML_SHA256
    model = _compiled_model()
    data = _fresh_data(model)
    assert abs(model.body_mass.sum() - 40.23897807283746) < 1e-8
    assert abs(reset_height() - .48300612662523257) < 1e-8
    assert abs(collision_min_z(model, data) - .002) < 1e-8
    assert all(contact.dist >= -1e-8 for contact in data.contact)
    assert model.opt.timestep == .005
    assert hashlib.sha256(ROBOT_XML.read_bytes()).hexdigest() == before


def test_actual_motor_and_joint_forces_are_not_clipped_at_80():
    model = _compiled_model()
    assert model.nu == 18
    assert not model.actuator_ctrllimited.any()
    assert not model.actuator_forcelimited.any()
    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot/" + name)
              for name in JOINT_NAMES]
    assert not model.jnt_actfrclimited[joints].any()
    data = _fresh_data(model)
    data.ctrl[:] = 100.
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.actuator_force, 100., atol=1e-10, rtol=0.)
    dof_ids = model.jnt_dofadr[joints]
    np.testing.assert_allclose(data.qfrc_actuator[dof_ids], 100., atol=1e-10, rtol=0.)


def test_explicit_floor_pairs_and_actual_contact_use_legacy_average_friction():
    model = _compiled_model()
    robot_geom_ids = _robot_collision_geoms(model)
    pairs = [i for i in range(model.npair) if model.pair(i).name.startswith("tripod_floor_")]
    assert len(pairs) == len(robot_geom_ids) == 27
    assert np.all(model.pair_dim[pairs] == 3)
    np.testing.assert_array_equal(model.pair_friction[pairs, 2:], 0.)
    # Exercise the REAL startup adapter using normal torch buffers, without
    # initializing Warp/CUDA. This checks per-env realized DR assignments.
    buffers = SimpleNamespace(**{
        name: torch.tensor(getattr(model, name).copy(), dtype=torch.float32).unsqueeze(0).repeat(2, *([1] * getattr(model, name).ndim))
        for name in ("body_mass", "body_inertia", "geom_friction", "pair_friction")
    })
    robot = SimpleNamespace(indexing=SimpleNamespace(geom_ids=robot_geom_ids))
    env = SimpleNamespace(
        device="cpu", _tripod_commands=object(),
        _tripod_dr=SimpleNamespace(friction=torch.tensor([.3, 1.25]), added_mass=torch.tensor([-5., 5.])),
        scene={"robot": robot}, sim=SimpleNamespace(mj_model=model, model=buffers),
    )
    randomize_model(env)
    np.testing.assert_allclose(buffers.pair_friction[0, pairs, :2], .65, atol=1e-7, rtol=0.)
    np.testing.assert_allclose(buffers.pair_friction[1, pairs, :2], 1.125, atol=1e-7, rtol=0.)
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot/BASE")
    np.testing.assert_allclose(buffers.body_mass[:, base], [5., 15.], atol=1e-8, rtol=0.)
    # Verify MuJoCo uses the explicit pair's coefficient even though the
    # floor's geom priority would otherwise override robot geom material.
    retained = model.pair_friction.copy()
    try:
        model.pair_friction[pairs, :2] = .65
        data = _fresh_data(model, reset_height() - .003)
        floor_contacts = [contact for contact in data.contact
                          if model.geom_type[contact.geom1] == mujoco.mjtGeom.mjGEOM_PLANE
                          or model.geom_type[contact.geom2] == mujoco.mjtGeom.mjGEOM_PLANE]
        assert floor_contacts
        for contact in floor_contacts:
            assert contact.dim == 3
            np.testing.assert_allclose(contact.friction[:2], .65, atol=1e-12, rtol=0.)
    finally:
        model.pair_friction[:] = retained


def test_target_height_foot_only_ik_is_not_a_legal_upright_stance():
    """Record a physical incompatibility without changing the reward target."""
    model = _compiled_model()
    data = _fresh_data(model, .25)
    foot_geoms = _robot_collision_geoms(model, "FOOT")
    hip_geoms = _robot_collision_geoms(model, "HIP")
    assert len(foot_geoms) == len(hip_geoms) == 6
    # HAA rotates about approximately world vertical in the upright robot;
    # HFE/KFE do not move the HIP collision mesh. Thus feet-only IK cannot
    # remove the HIP-ground intersection. Sample the complete HAA range as
    # an additional numerical check (not a global pose feasibility proof).
    sampled_hip_min = []
    for haa in np.linspace(-3., 3., 25):
        data.qpos[7::3] = haa
        mujoco.mj_forward(model, data)
        sampled_hip_min.append(collision_min_z(model, data, hip_geoms))
    assert max(sampled_hip_min) < -.0714
    data.qpos[7:] = 0.
    ids = np.asarray([i for i in range(18) if i % 3])

    def residual(x):
        data.qpos[7 + ids] = x
        mujoco.mj_forward(model, data)
        height_error = np.asarray([collision_min_z(model, data, [i]) - .002 for i in foot_geoms])
        return np.r_[height_error, .001 * x]

    result = least_squares(residual, np.tile([-.6, -.45], 6), bounds=(-2.7, 2.7), max_nfev=100)
    foot_errors = residual(result.x)[:6]
    assert abs(foot_errors).max() < 1e-5
    hip_min_z = collision_min_z(model, data, hip_geoms)
    assert hip_min_z < -.0714
    nonfoot_ground = [contact for contact in data.contact
                     if contact.dist < -.01 and
                     (model.geom_type[contact.geom1] == mujoco.mjtGeom.mjGEOM_PLANE
                      or model.geom_type[contact.geom2] == mujoco.mjtGeom.mjGEOM_PLANE)
                     and not any(model.body(model.geom_bodyid[g]).name.endswith("FOOT")
                                 for g in (contact.geom1, contact.geom2))]
    assert nonfoot_ground
    print(json.dumps({"root_height_target": .25, "foot_only_ik_max_error_m": float(abs(foot_errors).max()),
                      "hip_min_z_m": hip_min_z,
                      "upright_hip_clearance_root_height_m": .25 - hip_min_z,
                      "valid_upright_stance": False,
                      "hfe_kfe_radians": result.x.reshape(6, 2).tolist()}))


def test_zero_input_lstm_one_second_cpu_contact_diagnostic():
    """Only finite execution/contact evidence; no learning or GPU claim."""
    model = _compiled_model()
    data = _fresh_data(model)
    network = torch.jit.load(str(ACTUATOR_PATH), map_location="cpu").eval()
    hidden = torch.zeros(2, 18, 8)
    cell = torch.zeros_like(hidden)
    height, effort, up, distances = [], [], [], []
    nonfoot_ground = set()
    for _ in range(200):
        inputs = torch.as_tensor(np.stack([-data.qpos[7:], data.qvel[6:]], -1), dtype=torch.float32).unsqueeze(1)
        with torch.inference_mode():
            torque, (hidden, cell) = network(inputs, (hidden, cell))
        data.ctrl[:] = torque.numpy().reshape(18)
        mujoco.mj_step(model, data)
        assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
        assert torch.isfinite(torque).all()
        height.append(float(data.qpos[2]))
        effort.append(float(abs(data.ctrl).max()))
        base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot/BASE")
        up.append(float(data.xmat[base_id].reshape(3, 3)[2, 2]))
        for contact in data.contact:
            distances.append(float(contact.dist))
            if model.geom_type[contact.geom1] == mujoco.mjtGeom.mjGEOM_PLANE:
                other = contact.geom2
            elif model.geom_type[contact.geom2] == mujoco.mjtGeom.mjGEOM_PLANE:
                other = contact.geom1
            else:
                continue
            name = model.body(model.geom_bodyid[other]).name
            if not name.endswith("FOOT"):
                nonfoot_ground.add(name)
    assert not data.warning.number.any()
    assert distances  # real floor contact occurred
    print(json.dumps({"cpu_diagnostic_seconds": float(data.time), "root_height_range_m": [min(height), max(height)],
                      "final_root_height_m": height[-1], "maximum_abs_effort_Nm": max(effort),
                      "minimum_contact_distance_m": min(distances), "minimum_up_dot_z": min(up),
                      "nonfoot_ground_contacts": sorted(nonfoot_ground), "training_validation": False}))


if __name__ == "__main__":
    for _name, _function in list(globals().items()):
        if _name.startswith("test_"):
            _function()
            print(_name, "PASS")
