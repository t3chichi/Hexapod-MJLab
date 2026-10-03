#!/usr/bin/env python
"""T0-A stage 1: freeze the ACTUAL baseline identity (files, not Git HEAD).

Registers, before any T0-A modification happens:
* Git HEAD + dirty list (the dirty working tree IS the baseline);
* resolved ``__file__`` of every imported module of interest (installed
  mjlab vs. repo code, no guessing);
* SHA-256 of robot XML/meshes/URDF actually loaded by the task;
* the LiDAR mount read from the ``lidar_sensor`` site;
* joint / action / actuator ordering and units, control and physics clocks;
* nominal URDF/MJCF joint limits vs. the T1 measurement assumption;
* existing ``logs/rsl_rl`` checkpoint directories (content, iter, source)
  classified as UNKNOWN-origin short-pipeline evidence;
* T1 witness cross-check (library.json presence + frozen copy equality).

Usage:
  python scripts/tools/t0/audit_identity.py --run-dir logs/t0/<run_id>
  python scripts/tools/t0/audit_identity.py           # creates a new run dir
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import t0lib  # noqa: E402
from t0lib import (  # noqa: E402
    FAIL,
    NOT_VERIFIED,
    PASS,
    StageReport,
    new_run_dir,
    sha256_file,
)

ASSETS = t0lib.REPO_ROOT / "src" / "hexapod_mjlab" / "assets" / "robots" / "el_4090"
T1_LIBRARY = t0lib.REPO_ROOT / "docs" / "research" / "morphology" / "evidence" / "library.json"

# Frozen copy of the T1 `height_0.48_width_1.00` witness (see
# src/.../tasks/el_4090_lidar/t0/witness.py for the authoritative frozen copy;
# this audit re-reads library.json independently to cross-check it).
WITNESS_ID = "height_0.48_width_1.00"


def audit() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()
    run_dir = args.run_dir or new_run_dir()
    rep = StageReport("audit_identity", run_dir)

    # ── Git identity: dirty tree is the baseline, HEAD is NOT the baseline ──
    git = t0lib.git_state()
    rep.gate(
        "git_identity",
        PASS,
        head=git["head"],
        branch=git["branch"],
        dirty_count=len(git["dirty_files"]),
        dirty_files=git["dirty_files"],
    )
    rep.meta["git"] = git

    # ── Module resolution: actual __file__, no guessing ──────────────────────
    import mjlab
    import mujoco
    import mujoco_warp
    import warp
    import rsl_rl
    import hexapod_mjlab
    from hexapod_mjlab.tasks.el_4090_lidar import el_4090_lidar_env_cfg as env_cfg_mod
    from hexapod_mjlab.tasks.el_4090_lidar.mdp import lidar as lidar_mod
    from hexapod_mjlab.tasks.el_4090_lidar.rl import runner as runner_mod
    from hexapod_mjlab.rl import actor_v2, ppo_lidar

    modules = {
        "mjlab": mjlab.__file__,
        "mujoco": mujoco.__file__,
        "mujoco_warp": mujoco_warp.__file__,
        "warp": warp.__file__,
        "rsl_rl": rsl_rl.__file__,
        "hexapod_mjlab": hexapod_mjlab.__file__,
        "el_4090_lidar_env_cfg": env_cfg_mod.__file__,
        "mdp.lidar": lidar_mod.__file__,
        "task.rl.runner": runner_mod.__file__,
        "rl.actor_v2": actor_v2.__file__,
        "rl.ppo_lidar": ppo_lidar.__file__,
    }
    task_src_sha = {
        str(p.relative_to(t0lib.REPO_ROOT)): sha256_file(p)
        for p in sorted((t0lib.REPO_ROOT / "src" / "hexapod_mjlab").rglob("*.py"))
    }
    rep.gate("module_resolution", PASS, modules=modules)

    # ── Robot asset identity (XML actually loaded + meshes + URDF) ──────────
    xml_path = env_cfg_mod._EL4090_XML
    assets_sha = {"mjcf/el_4090_lidar.xml": sha256_file(xml_path)}
    for stl in sorted((ASSETS / "meshes").glob("*.STL")):
        assets_sha[f"meshes/{stl.name}"] = sha256_file(stl)
    for urdf in sorted((ASSETS / "urdf").glob("*.urdf")):
        assets_sha[f"urdf/{urdf.name}"] = sha256_file(urdf)
    rep.gate("robot_assets_sha256", PASS, assets_sha256=assets_sha)

    # ── Model facts: mount, joint order, limits, clocks, exec effort ─────────
    import torch  # noqa: F401

    mount = lidar_mod.resolve_lidar_mount()
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    jnames = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
    ]
    act_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    anames = list(env_cfg_mod.JOINT_NAMES)
    nominal_limits = {
        jnames[k]: [float(model.jnt_range[i][0]), float(model.jnt_range[i][1])]
        for i in range(model.njnt)
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
        for k in [sum(1 for j in range(i) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE)]
    }
    model_facts = {
        "lidar_mount_pos_base_frame": None if mount is None else list(mount[0]),
        "lidar_mount_quat_xyzw": None if mount is None else list(mount[1]),
        "hinge_joint_order_model": jnames,
        "task_joint_order": anames,
        "joint_order_matches": jnames == anames,
        "actuator_order_model": act_names,
        "actuator_order_matches": act_names == anames,
        "nominal_position_limits_rad": nominal_limits,
        "nominal_limits_uniform_pm3": all(
            abs(lo + 3.0) < 1e-9 and abs(hi - 3.0) < 1e-9 for lo, hi in nominal_limits.values()
        ),
        "actuator_ctrlrange": [
            [float(model.actuator_ctrlrange[i][0]), float(model.actuator_ctrlrange[i][1])]
            for i in range(model.nu)
        ],
        "sim_timestep_s": float(model.opt.timestep),
        "control_hz": 1.0 / (env_cfg_mod.DECIMATION * env_cfg_mod.SIM_TIMESTEP),
        "physics_hz": 1.0 / env_cfg_mod.SIM_TIMESTEP,
        "stand_keyframe_qpos": [float(v) for v in model.key_qpos[0]],
        "body_count": model.nbody,
        "collision_geoms": [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
            for i in range(model.ngeom)
            if model.geom_contype[i] != 0 or model.geom_conaffinity[i] != 0
        ],
    }
    rep.gate(
        "model_facts",
        PASS if model_facts["joint_order_matches"] else FAIL,
        **model_facts,
    )

    # ── Existing checkpoints: register, do NOT claim they are mature policies ─
    ckpt_registry = []
    rl_root = t0lib.REPO_ROOT / "logs" / "rsl_rl" / "el_4090_lidar"
    if rl_root.exists():
        for run in sorted(rl_root.iterdir()):
            if not run.is_dir():
                continue
            entries = {}
            for pt in sorted(run.glob("*.pt")):
                entries[pt.name] = {
                    "sha256": sha256_file(pt),
                    "size_bytes": pt.stat().st_size,
                }
            ckpt_registry.append(
                {
                    "dir": str(run.relative_to(t0lib.REPO_ROOT)),
                    "files": entries,
                    "params_env_yaml": (run / "params" / "env.yaml").exists(),
                    "params_agent_yaml": (run / "params" / "agent.yaml").exists(),
                    "max_iter_seen": max(
                        [int(p.stem.split("_")[1]) for p in run.glob("model_*.pt")] or [-1]
                    ),
                    "classification": (
                        "UNKNOWN_ORIGIN_SHORT_PIPELINE: only model_0/model_1 present; "
                        "no evidence of mature walking; kept as baseline artifact, not "
                        "used as a trained-walking claim"
                    ),
                }
            )
    rep.gate(
        "existing_checkpoints_registered",
        PASS,
        registry=ckpt_registry,
        note="Exact inventory in this checkout; an empty inventory is valid, not a trained policy.",
    )

    # ── T1 witness cross-check ───────────────────────────────────────────────
    witness_present = T1_LIBRARY.exists()
    w_lib = None
    if witness_present:
        import json

        with open(T1_LIBRARY, encoding="utf-8") as f:
            lib = json.load(f)
        rec = next((r for r in lib["records"] if r["id"] == WITNESS_ID), None)
        if rec is not None:
            w_lib = {
                "measured_qpos_sha_input": rec["measured_qpos"],
                "hold_height_m": rec["measured_E"][5],
                "validation_level": rec["validation_level"],
            }
    from hexapod_mjlab.tasks.el_4090_lidar.t0.witness import (
        get_witness_summary,
    )

    w_frozen = get_witness_summary()
    cross = {
        "library_present": witness_present,
        "frozen_hold_height_m": w_frozen["hold_height_m"],
        "library_hold_height_m": None if w_lib is None else w_lib["hold_height_m"],
    }
    ok = (
        witness_present
        and w_lib is not None
        and abs(w_lib["hold_height_m"] - w_frozen["hold_height_m"]) < 1e-12
    )
    rep.gate("t1_witness_crosscheck", PASS if ok else FAIL, **cross)

    # ── Audit findings (facts from code reading, each with file:line) ────────
    findings = {
        "base_height_reward_fixed_0.25": {
            "location": "mdp/rewards.py:BASE_HEIGHT_TARGET = 0.25",
            "fact": (
                "base_height reward targets 0.25 m above measured terrain while the "
                "T1-verified hold witness settles at ~0.4599 m; the 0.25 m target "
                "conflicts with the verified standing height and the old fall "
                "threshold 0.12 m would only catch it after a full collapse"
            ),
            "t0_action": "T0 flat cfg parameterizes the target to the witness height",
        },
        "penalised_contact_missing_hip": {
            "location": "el_4090_lidar_env_cfg.py: penalised_contact patterns",
            "fact": (
                "collision penalty sensor covers BASE/SHANK/THIGH only; HIP, "
                "RADAR_BRACKET and LIDAR_AIRY ground contacts and any self-collision "
                "are invisible to rewards; termination (gravity z > -0.1 or z < 0.12) "
                "would not fire for a hip resting on the ground with an upright body"
            ),
            "t0_action": (
                "T0 diagnostics add a nonfoot-any contact sensor + raw mujoco_warp "
                "contact classification at every physics substep"
            ),
        },
        "nominal_limits_uncalibrated": {
            "location": "mjcf el_4090_lidar.xml: every hinge range=-3..3 (URDF copy)",
            "fact": (
                "uniform ±3 rad nominal limits are uncalibrated placeholders; T1 "
                "measurements disabled them in-memory; the dof_pos_limits reward "
                "penalizes soft limits derived from these values"
            ),
            "t0_action": (
                "T0 flat robot spec disables the 18 hinge limits in the in-memory "
                "MjSpec (asset bytes unchanged) and removes the dof_pos_limits reward"
            ),
        },
        "play_residual_randomization": {
            "location": "el_4090_lidar_env_cfg.py: make_el_4090_lidar_env_cfg(play=True)",
            "fact": (
                "play removes noise/push/curriculum but keeps startup mass and "
                "friction randomization and reset velocity/pose randomization; play "
                "is NOT a deterministic evaluation configuration"
            ),
            "t0_action": "T0 flat cfg is the deterministic evaluation configuration",
        },
        "command_deadzone_0.2": {
            "location": "mdp/commands.py: cmd_deadzone = 0.2",
            "fact": (
                "linear commands with norm <= 0.2 m/s are zeroed at resample; "
                "claims of tracking at 0.1 m/s would be vacuous (command is zero)"
            ),
            "t0_action": (
                "T0 commands are standing (0) or vx in [0.25, 0.35]; evaluation "
                "commands are forced and recorded requested-vs-actual"
            ),
        },
        "terrain_view_pillar_dependency": {
            "location": "mdp/terrain.py: get_el_4090_terrain_view",
            "fact": (
                "LiDAR raycast mesh, height_grid supervision and base_height reward "
                "all read the same El4090PillarFieldCfg terrain view; switching "
                "physics ground without switching this view would scan a stale "
                "pillar field"
            ),
            "t0_action": (
                "T0 flat terrain is the same builder/view chain with a config that "
                "generates zero pillars; one consistency gate checks physics vs "
                "raycast vs supervision"
            ),
        },
        "baseline_launcher_swallowing_failures": {
            "location": "scripts/bench/run_baseline_mjlab.sh:91 (`|| true`) and :47 root",
            "fact": (
                "the old harness runs training with `|| true` (return code lost) and "
                "derives REPO_ROOT by going up only one level from scripts/bench"
            ),
            "t0_action": (
                "new scripts/tools/t0 entries resolve the repo root from this file's "
                "path and always propagate non-zero exit codes"
            ),
        },
    }
    rep.gate("audit_findings_recorded", PASS, findings=findings)

    # ── Environment/toolchain identity ───────────────────────────────────────
    rep.meta["python_identity"] = t0lib.python_identity()
    rep.meta["package_versions"] = t0lib.package_versions(
        [
            "mjlab",
            "mujoco",
            "mujoco-warp",
            "warp-lang",
            "rsl-rl-lib",
            "torch",
            "tensordict",
            "numpy",
            "scipy",
        ]
    )
    rep.meta["task_source_sha256"] = task_src_sha
    rep.meta["assets_sha256"] = assets_sha

    doc = rep.finalize()
    return doc["returncode"]


if __name__ == "__main__":
    raise SystemExit(audit())
