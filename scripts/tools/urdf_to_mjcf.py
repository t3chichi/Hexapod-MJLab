#!/usr/bin/env python3
"""M1 — Convert the el_4090 spider robot URDF to MJCF for MuJoCo / mujoco-warp.

Reproducible, in-repo pipeline (no references outside this repository):

    src/hexapod_mjlab/assets/robots/el_4090/urdf/el_4090.urdf   (source, copied from the old repo)
    src/hexapod_mjlab/assets/robots/el_4090/meshes/*.STL        (25 meshes, copied from the old repo)
        |
        v  mujoco.MjSpec
    src/hexapod_mjlab/assets/robots/el_4090/mjcf/el_4090.xml    (deliverable)
    tests/robot/joint_table.json / .md      (URDF vs MJCF cross-check + old-config gains)

Programmatic edits applied to the imported spec:

  a) compiler meshdir="../meshes" (XML lives in mjcf/, meshes in ../meshes) and
     per-mesh maxhullvert=64 (convex-hull collision simplification for mujoco-warp;
     the compiler-level attribute is not exposed on MjsCompiler in mujoco 3.11,
     the mesh-level attribute is authoritative and serializes to XML).
     Additionally compiler fusestatic=False: mujoco 3.11's to_xml() serializes
     geoms of statically-fused bodies (the 6 FOOT links) into the parent SHANK
     body WITHOUT the fixed-joint transform, silently displacing the foot
     collision geoms by exactly the URDF fixed-origin (~30 cm). Keeping the
     FOOT bodies unfused round-trips exactly and also matches the old env's
     `collapse_fixed_joints = False` (Isaac Gym asset option).
  b) freejoint on the BASE body (URDF root has no joint; without it MjSpec's
     fusestatic folds BASE into the worldbody).
  c) 18 <motor> actuators, one per URDF revolute joint, added in URDF order:
     gear=1 (direct torque), ctrlrange = [-effort, +effort] with effort taken
     from the URDF <limit effort="..."> (80 N*m, symmetric).
  d) "stand" keyframe via spec.add_key(): qpos = freejoint pose (base elevated so
     the lowest foot point sits FOOT_CLEARANCE above ground, quat identity) +
     18 default joint angles (= old legged_gym default_dof_pos, all 0.0).
  e) option timestep=0.005 s (matches old sim dt; MuJoCo default gravity
     (0,0,-9.81) already matches the old sim params).

Old-config values embedded below (source: legged_gym/legged_gym/envs/el_4090/
spider_nomal/el_4090_spider_config.py and its parents in elspider_air):
  stiffness HAA/HFE/KFE = 130 N*m/rad, damping = 2 N*m*s/rad, action_scale 0.25,
  decimation 4, sim dt 0.005. The PD law stays in torch at migration time —
  MuJoCo only gets motors.
"""

from __future__ import annotations

import json
import math
import pathlib
import re
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
URDF_PATH = REPO / "src/hexapod_mjlab/assets/robots/el_4090/urdf/el_4090.urdf"
MESHES_DIR = REPO / "src/hexapod_mjlab/assets/robots/el_4090/meshes"
XML_OUT = REPO / "src/hexapod_mjlab/assets/robots/el_4090/mjcf/el_4090.xml"
JOINT_TABLE_JSON = REPO / "tests/robot/joint_table.json"
JOINT_TABLE_MD = REPO / "tests/robot/joint_table.md"

# --------------------------------------------------------------------------- #
# Old legged_gym config values (see module docstring for provenance).
# --------------------------------------------------------------------------- #
OLD_CONFIG = {
    "source_config": "legged_gym/legged_gym/envs/el_4090/spider_nomal/el_4090_spider_config.py",
    "stiffness": {"HAA": 130.0, "HFE": 130.0, "KFE": 130.0},   # [N*m/rad]
    "damping": {"HAA": 2.0, "HFE": 2.0, "KFE": 2.0},           # [N*m*s/rad]
    "action_scale": 0.25,                                       # target = scale*action + default
    "decimation": 4,
    "sim_dt": 0.005,                                            # legged_robot_config.py sim.dt
    "gravity": [0.0, 0.0, -9.81],
    "isaac_init_pos": [0.0, 0.0, 0.45],                         # informational (see below)
    "torque_clip": 80.0,                                        # URDF effort limit, old env clips
    # default_dof_pos from init_state.default_joint_angles: every joint 0.0 rad.
    "default_dof_pos": {f"{leg}_{jnt}": 0.0
                        for leg in ("RF", "RM", "RB", "LF", "LM", "LB")
                        for jnt in ("HAA", "HFE", "KFE")},
}

# Stand pose: base height chosen so the lowest foot collision-hull point floats
# FOOT_CLEARANCE above the ground at default joint angles (quat identity).
# Measured exact-touch height at q=0 is 0.48101 m; note the old Isaac init
# z=0.45 would put feet ~3.1 cm *below* the plane — an Isaac spawn quirk, not
# reproduced here. The measurement is re-derived at runtime below and asserted.
FOOT_CLEARANCE = 0.005
STAND_HEIGHT = 0.48101 + FOOT_CLEARANCE  # 0.48601 -> runtime-verified

TOL = 1e-9  # exactness tolerance for the cross-check


# --------------------------------------------------------------------------- #
# URDF parsing
# --------------------------------------------------------------------------- #
def rpy_to_quat(rpy) -> np.ndarray:
    """URDF roll-pitch-yaw (fixed-axis XYZ) -> quaternion (w, x, y, z)."""
    r, p, y = rpy
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    # R = Rz(y) @ Ry(p) @ Rx(r)
    return np.array([
        cy * cp * cr + sy * sp * sr,
        cy * cp * sr - sy * sp * cr,
        cy * sp * cr + sy * cp * sr,
        sy * cp * cr - cy * sp * sr,
    ])


def parse_urdf(path: pathlib.Path) -> dict:
    root = ET.parse(path).getroot()
    joints = []
    for j in root.findall("joint"):
        origin = j.find("origin")
        xyz = np.array([float(v) for v in (origin.get("xyz", "0 0 0")).split()]) if origin is not None else np.zeros(3)
        rpy = np.array([float(v) for v in (origin.get("rpy", "0 0 0")).split()]) if origin is not None else np.zeros(3)
        axis_el = j.find("axis")
        axis = np.array([float(v) for v in (axis_el.get("xyz", "0 0 0")).split()]) if axis_el is not None else np.zeros(3)
        limit = j.find("limit")
        joints.append({
            "name": j.get("name"),
            "type": j.get("type"),
            "parent": j.find("parent").get("link"),
            "child": j.find("child").get("link"),
            "xyz": xyz,
            "rpy": rpy,
            "quat": rpy_to_quat(rpy),
            "axis": axis,
            "lower": float(limit.get("lower")) if limit is not None else None,
            "upper": float(limit.get("upper")) if limit is not None else None,
            "effort": float(limit.get("effort")) if limit is not None else None,
            "velocity": float(limit.get("velocity")) if limit is not None else None,
        })
    links = {}
    for link in root.findall("link"):
        entry = {"visual_meshes": [], "collision_meshes": []}
        for tag, key in (("visual", "visual_meshes"), ("collision", "collision_meshes")):
            for el in link.findall(tag):
                mesh = el.find("geometry/mesh")
                if mesh is not None:
                    entry[key].append(pathlib.PurePosixPath(mesh.get("filename")).name)
        links[link.get("name")] = entry
    return {"joints": joints, "links": links}


# --------------------------------------------------------------------------- #
# Stand-height measurement (separate throwaway spec with a floor plane)
# --------------------------------------------------------------------------- #
def measure_stand_height() -> tuple[float, float]:
    """Return (exact_touch_height, min_foot_clearance_at_z1) at default joint angles."""
    spec = mujoco.MjSpec.from_file(str(URDF_PATH))
    spec.body("BASE").add_joint(type=mujoco.mjtJoint.mjJNT_FREE)
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.1],
                            name="measurement_floor")
    model = spec.compile()
    data = mujoco.MjData(model)
    floor_gid = int(np.where(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)[0][0])
    foot_gids = [g for g in range(model.ngeom)
                 if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                 and "FOOT" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH,
                                                  model.geom_dataid[g]) or "")]
    assert len(foot_gids) == 6, f"expected 6 foot geoms, got {len(foot_gids)}"

    probe_z = 1.0  # comfortably above the floor
    data.qpos[:] = 0.0
    data.qpos[2] = probe_z
    data.qpos[3] = 1.0  # identity quat
    mujoco.mj_forward(model, data)
    # self-contact sanity at the default pose
    ncon = int(data.ncon)
    assert ncon == 0, f"unexpected self-contacts at default pose: {ncon}"
    dists = [mujoco.mj_geomDistance(model, data, g, floor_gid, 5.0, np.zeros(6))
             for g in foot_gids]
    min_dist = float(min(dists))
    assert min_dist > 0.4, f"feet suspiciously close to floor at z={probe_z}: {min_dist}"
    return probe_z - min_dist, min_dist


# --------------------------------------------------------------------------- #
# Main conversion
# --------------------------------------------------------------------------- #
def main() -> None:
    urdf = parse_urdf(URDF_PATH)
    rev_joints = [j for j in urdf["joints"] if j["type"] == "revolute"]
    fixed_joints = [j for j in urdf["joints"] if j["type"] == "fixed"]
    assert len(rev_joints) == 18 and len(fixed_joints) == 6

    # --- stand height (measured, must match the constant above) -------------
    exact_touch, min_dist = measure_stand_height()
    stand_height = round(exact_touch + FOOT_CLEARANCE, 5)
    assert abs(stand_height - STAND_HEIGHT) < 1e-4, (
        f"measured stand height {stand_height} drifted from constant {STAND_HEIGHT}")
    print(f"[stand] exact-touch height at q=0: {exact_touch:.5f} m "
          f"(min foot clearance {min_dist:.4f} @ z=1.0); "
          f"keyframe z = {stand_height:.5f} m ({FOOT_CLEARANCE*1000:.0f} mm clearance)")

    # --- import URDF ---------------------------------------------------------
    spec = mujoco.MjSpec.from_file(str(URDF_PATH))
    spec.option.timestep = OLD_CONFIG["sim_dt"]  # match old sim dt

    # (a) meshdir + maxhullvert + fusestatic ------------------------------------
    spec.compiler.meshdir = "../meshes"  # relative to mjcf/el_4090.xml
    spec.compiler.fusestatic = False  # keep FOOT bodies (see module docstring)
    for mesh in spec.meshes:
        mesh.file = pathlib.PurePosixPath(mesh.file).name  # basename only
        mesh.maxhullvert = 64  # convex-hull simplification for mujoco-warp

    # (b) freejoint on BASE ----------------------------------------------------
    base = spec.body("BASE")
    base.add_joint(type=mujoco.mjtJoint.mjJNT_FREE, name="root")

    # (c) motors, one per hinge joint, in URDF order ---------------------------
    for j in rev_joints:
        spec.add_actuator(
            name=j["name"],                      # actuator name == joint name (1:1)
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            target=j["name"],
            gear=[1.0, 0, 0, 0, 0, 0],           # direct torque, ctrl [N*m]
            gaintype=mujoco.mjtGain.mjGAIN_FIXED,
            gainprm=[1.0] + [0.0] * 9,
            biastype=mujoco.mjtBias.mjBIAS_NONE,
            ctrlrange=[-j["effort"], j["effort"]],  # symmetric ± URDF effort
        )

    # (d) stand keyframe — qpos ordered by the *compiled* joint order ----------
    model = spec.compile()  # probe compile to learn joint ordering
    jnames = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
              for i in range(model.njnt)]
    assert jnames[0] == "root" and model.nq == 25, f"unexpected joint layout: {jnames}"
    qpos = np.zeros(model.nq)
    qpos[0:3] = [0.0, 0.0, stand_height]  # URDF base pose elevated, feet clear ground
    qpos[3] = 1.0                          # identity quat
    for jid, name in enumerate(jnames[1:], start=1):
        qpos[7 + jid - 1] = OLD_CONFIG["default_dof_pos"][name]
    spec.add_key(name="stand", qpos=qpos.tolist(), qvel=np.zeros(model.nv).tolist())

    # --- final compile + write ------------------------------------------------
    model = spec.compile()  # 26 bodies: world + BASE + 18 jointed + 6 fixed FOOT
    assert model.nkey == 1 and model.nu == 18 and model.nbody == 26 and model.nq == 25
    xml_text = spec.to_xml()  # mesh paths relative to XML location via meshdir

    # Post-processing (both verified by assertions on reload below):
    #  - MjSpec.to_xml does not serialize mesh.maxhullvert (mujoco 3.11); write it
    #    explicitly so the standalone XML reproduces the simplified collision hulls.
    #  - to_xml emits <general> for API-added actuators; spell them as <motor>
    #    (identical semantics: fixed unit gain, no bias, joint transmission).
    xml_text, n_mesh = re.subn(
        r'(<mesh name="[^"]+"[^>]*?file="[^"]+)"\s*/>', r'\1" maxhullvert="64"/>',
        xml_text)
    assert n_mesh == 25, f"mesh maxhullvert rewrite hit {n_mesh} elements, expected 25"
    xml_text, n_act = re.subn(
        r'<general (name="(\w+)" joint="\2") (ctrlrange="-80 80")/>',
        r'<motor \1 \3/>', xml_text)
    assert n_act == 18, f"motor rewrite hit {n_act} elements, expected 18"

    XML_OUT.parent.mkdir(parents=True, exist_ok=True)
    XML_OUT.write_text(xml_text)
    print(f"[write] {XML_OUT}")

    # --- G1 gate: reload from disk -------------------------------------------
    m = mujoco.MjModel.from_xml_path(str(XML_OUT))
    print(f"[G1] reload OK: nbody={m.nbody} njnt={m.njnt} ngeom={m.ngeom} "
          f"nu={m.nu} nq={m.nq} nkey={m.nkey} timestep={m.opt.timestep}")

    # maxhullvert round-trip: hull graphs from the standalone XML must equal the
    # spec-compiled (maxhullvert=64) reference exactly.
    assert np.array_equal(np.asarray(m.mesh_graph), np.asarray(model.mesh_graph)), (
        "convex-hull mismatch: XML maxhullvert rewrite did not reproduce the "
        "spec-compiled hulls")

    # FOOT geom pose round-trip: with fusestatic=False the 6 fixed FOOT bodies
    # (and their geoms) must reload at the exact in-memory poses. This guards
    # against the mujoco 3.11 to_xml() fused-geom transform loss documented in
    # the module docstring.
    for leg in ("RF", "RM", "RB", "LF", "LM", "LB"):
        mesh_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MESH, f"{leg}_FOOT")
        for mdl, tag in ((model, "spec"), (m, "xml")):
            g = next(gg for gg in range(mdl.ngeom)
                     if mdl.geom_type[gg] == mujoco.mjtGeom.mjGEOM_MESH
                     and mdl.geom_dataid[gg] == mesh_id)
            assert mesh_id >= 0 and mdl.geom_bodyid[g] > 0
            if tag == "spec":
                ref = (np.asarray(mdl.geom_pos[g]).copy(),
                       np.asarray(mdl.geom_quat[g]).copy(),
                       int(mdl.geom_bodyid[g]))
            else:
                assert np.allclose(mdl.geom_pos[g], ref[0], atol=1e-12), (
                    f"{leg}_FOOT geom pos lost in XML round-trip "
                    f"({mdl.geom_pos[g]} vs {ref[0]}) — fusestatic regression")
                q1, q2 = np.asarray(mdl.geom_quat[g]), ref[1]
                if q1 @ q2 < 0:
                    q2 = -q2
                assert np.allclose(q1, q2, atol=1e-12)
                body_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY,
                                               mdl.geom_bodyid[g])
                assert body_name == f"{leg}_FOOT", (
                    f"{leg}_FOOT geom attached to {body_name}")

    # --- joint table cross-check (G2) ----------------------------------------
    table = build_joint_table(m, urdf, rev_joints)

    # keyframe round-trip check
    key_qpos = m.key_qpos[0].copy()
    assert np.allclose(key_qpos, qpos, atol=1e-8), "keyframe qpos changed across save/load"
    assert abs(key_qpos[2] - stand_height) < 1e-6 and key_qpos[3] == 1.0

    # actuator checks
    for j in rev_joints:
        aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, j["name"])
        trnid = m.actuator_trnid[aid]
        assert trnid[0] == mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j["name"])
        assert m.actuator_trntype[aid] == mujoco.mjtTrn.mjTRN_JOINT
        assert m.actuator_gaintype[aid] == mujoco.mjtGain.mjGAIN_FIXED
        assert m.actuator_biastype[aid] == mujoco.mjtBias.mjBIAS_NONE
        assert np.allclose(m.actuator_ctrlrange[aid], [-j["effort"], j["effort"]])
        assert abs(m.actuator_gear[aid][0] - 1.0) < TOL
    print(f"[G2] joint table: {len(table['joints'])} joints, all exact matches")

    # --- collision audit ------------------------------------------------------
    audit = collision_audit(m, urdf)

    # --- deliverables ---------------------------------------------------------
    JOINT_TABLE_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_by": "scripts/urdf_to_mjcf.py",
        "urdf": "src/hexapod_mjlab/assets/robots/el_4090/urdf/el_4090.urdf",
        "mjcf": "src/hexapod_mjlab/assets/robots/el_4090/mjcf/el_4090.xml",
        "stand_keyframe": {
            "name": "stand",
            "base_pos": [0.0, 0.0, stand_height],
            "base_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
            "joint_angles": OLD_CONFIG["default_dof_pos"],
            "exact_touch_height": exact_touch,
            "foot_clearance_m": FOOT_CLEARANCE,
        },
        "old_config": OLD_CONFIG,
        "joint_order": [j["name"] for j in rev_joints],
        "joints": table["joints"],
        "collision_audit": audit,
        "notes": table["notes"],
    }
    JOINT_TABLE_JSON.write_text(json.dumps(payload, indent=2))
    write_markdown_table(payload)
    print(f"[write] {JOINT_TABLE_JSON}")
    print(f"[write] {JOINT_TABLE_MD}")


def build_joint_table(m: mujoco.MjModel, urdf: dict, rev_joints: list[dict]) -> dict:
    """Exact-match cross-check URDF revolute joints vs compiled MJCF joints."""
    rows, notes = [], ["MJCF has no native per-joint velocity limit; URDF velocity "
                       "limits are preserved here (consumed by the env/training stack, "
                       "as in the old legged_gym setup).",
                       "URDF joint frame == MJCF child body frame at q=0 (MjSpec keeps "
                       "joint pos=[0,0,0]); position/rotation are compared there."]
    for k, j in enumerate(rev_joints):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j["name"])
        assert jid == k + 1, (f"joint order mismatch: {j['name']} at id {jid}, "
                              f"expected {k + 1} (id 0 is the freejoint)")
        body_id = m.jnt_bodyid[jid]
        body_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body_id)
        # frame equivalence
        assert body_name == j["child"], f"{j['name']}: body {body_name} != URDF child {j['child']}"
        assert np.allclose(m.body_pos[body_id], j["xyz"], atol=TOL), f"{j['name']}: pos mismatch"
        q_mj = m.body_quat[body_id]
        q_u = j["quat"]
        if q_mj @ q_u < 0:
            q_u = -q_u  # quaternion double cover
        assert np.allclose(q_mj, q_u, atol=1e-6), f"{j['name']}: quat mismatch {q_mj} vs {q_u}"
        # joint itself
        assert m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE
        assert np.allclose(m.jnt_axis[jid], j["axis"], atol=TOL), f"{j['name']}: axis mismatch"
        assert np.allclose(m.jnt_pos[jid], [0, 0, 0], atol=TOL)
        assert np.allclose(m.jnt_range[jid], [j["lower"], j["upper"]], atol=1e-9)
        # effort: URDF limit vs MJCF joint actuator-force range and motor ctrlrange
        assert np.allclose(m.jnt_actfrcrange[jid], [-j["effort"], j["effort"]], atol=1e-9)
        aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, j["name"])
        ctrlrange = m.actuator_ctrlrange[aid].tolist()
        assert ctrlrange == [-j["effort"], j["effort"]]
        rows.append({
            "index": k,
            "name": j["name"],
            "urdf_type": j["type"],
            "mjcf_type": "hinge",
            "parent_body": j["parent"],
            "child_body": j["child"],
            "axis_xyz": j["axis"].tolist(),
            "origin_xyz": j["xyz"].tolist(),
            "origin_rpy_rad": j["rpy"].tolist(),
            "range_rad": [j["lower"], j["upper"]],
            "effort_limit_Nm": j["effort"],
            "mjcf_ctrlrange_Nm": ctrlrange,
            "velocity_limit_rad_s": j["velocity"],
            "mjcf_velocity_limit": None,
            "motor_name": j["name"],
            "match": True,
        })
    return {"joints": rows, "notes": notes}


def collision_audit(m: mujoco.MjModel, urdf: dict) -> dict:
    """Geoms per body before (spec/URDF links) and after (compiled) fixed-body merge."""
    # before: straight from the URDF (one link = one geom here, since URDF
    # visual and collision reference the same STL per link)
    before = {}
    visual_only = {}
    for link, entry in urdf["links"].items():
        before[link] = entry["collision_meshes"]
        visual_only[link] = [v for v in entry["visual_meshes"]
                             if v not in entry["collision_meshes"]]
    # after: compiled model geoms per body
    after = {}
    urdf_collision_meshes = {pathlib.PurePosixPath(mesh).stem
                             for e in urdf["links"].values()
                             for mesh in e["collision_meshes"]}
    survived = []
    for b in range(m.nbody):
        bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
        after[bname] = []
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != b or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mesh_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[g])
            after[bname].append(mesh_name)
            survived.append(mesh_name)
    assert sorted(survived) == sorted(urdf_collision_meshes), (
        "URDF collision mesh set did not survive conversion")
    return {
        "urdf_geoms_per_link": before,
        "mjcf_geoms_per_body": after,
        "fixed_bodies": {
            "note": "fusestatic=False keeps all 25 URDF links as bodies "
                    "(matches old Isaac collapse_fixed_joints=False and works "
                    "around mujoco 3.11 to_xml() dropping fused-geom transforms)",
            "BASE": "dynamic body (freejoint 'root')",
            "FOOT bodies": "retained as joint-less fixed bodies, one FOOT mesh "
                           "geom each (verified pose-exact across XML round-trip)",
        },
        "visual_only_meshes": {k: v for k, v in visual_only.items() if v},
        "all_geoms_are_collision_geoms": bool(np.all(m.geom_contype == 1)
                                              and np.all(m.geom_conaffinity == 1)),
        "urdf_collision_mesh_count": len(urdf_collision_meshes),
        "mjcf_mesh_geom_count": len(survived),
    }


def write_markdown_table(payload: dict) -> None:
    lines = [
        "# M1 joint table — el_4090 URDF vs MJCF",
        "",
        f"Generated by `scripts/urdf_to_mjcf.py`. URDF: `{payload['urdf']}`, MJCF: `{payload['mjcf']}`.",
        "",
        "All 18 revolute joints match exactly: name, order, type, axis, joint-frame "
        "position/rotation, range, and effort limit (MJCF motor `ctrlrange` = ± URDF "
        "`effort`, gear 1). MuJoCo has no native per-joint velocity limit; URDF "
        "`velocity` is preserved in `joint_table.json` for the env stack.",
        "",
        "| # | joint | type | parent | child | axis | origin xyz [m] | origin rpy [rad] | range [rad] | effort [N*m] | ctrlrange [N*m] | velocity [rad/s] |",
        "|---|-------|------|--------|-------|------|----------------|------------------|-------------|--------------|-----------------|------------------|",
    ]
    for j in payload["joints"]:
        lines.append(
            f"| {j['index']} | {j['name']} | {j['urdf_type']}→{j['mjcf_type']} "
            f"| {j['parent_body']} | {j['child_body']} | {j['axis_xyz']} "
            f"| {j['origin_xyz']} | {j['origin_rpy_rad']} | {j['range_rad']} "
            f"| {j['effort_limit_Nm']} | {j['mjcf_ctrlrange_Nm']} | {j['velocity_limit_rad_s']} |")
    kf = payload["stand_keyframe"]
    oc = payload["old_config"]
    lines += [
        "",
        "## Stand keyframe `stand`",
        "",
        f"- base pos `{kf['base_pos']}` (quat identity), joint angles all 0.0 rad "
        f"(old `default_dof_pos`).",
        f"- exact-touch height at q=0: {kf['exact_touch_height']:.5f} m; keyframe "
        f"adds {kf['foot_clearance_m']*1000:.0f} mm foot clearance. The old Isaac "
        f"init z=0.45 would place feet ~3.1 cm below the plane and is NOT reused.",
        "",
        "## Old-config PD gains (applied in torch, not in MJCF)",
        "",
        "| group | kp [N*m/rad] | kd [N*m*s/rad] |",
        "|-------|--------------|----------------|",
    ]
    for g in ("HAA", "HFE", "KFE"):
        lines.append(f"| {g} | {oc['stiffness'][g]} | {oc['damping'][g]} |")
    lines += [
        "",
        f"action_scale = {oc['action_scale']}, decimation = {oc['decimation']}, "
        f"sim dt = {oc['sim_dt']} s, torque clip ±{oc['torque_clip']} N*m.",
        "",
        "## Collision audit (URDF links → MJCF bodies)",
        "",
        "- URDF: 25 links, one collision geom each (visual and collision reference "
        "the same STL, so the importer keeps a single mesh geom per link; all "
        "contype=conaffinity=1, group 0 — there are no visual-only meshes).",
        "- MJCF compiled: world + BASE (freejoint) + 18 joint bodies + 6 fixed FOOT "
        "bodies (fusestatic=False, mirroring the old env's collapse_fixed_joints="
        "False). Every URDF collision geom survives on its own link body; foot "
        "geom poses verified exact across the XML round-trip.",
    ]
    JOINT_TABLE_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
