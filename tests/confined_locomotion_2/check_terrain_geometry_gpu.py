"""Finite real-robot Heightfield geometry gate; no network, PPO or training.

Run from the repository root with the hp_mjlab interpreter and --output DIR.
All tolerances are engineering geometry tolerances, not locomotion acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import mujoco
import numpy as np
import torch
import warp as wp
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


from hexapod_mjlab.tasks.confined_locomotion_2.config.terrain_memory import make_default_terrain as fixture_terrain


def check(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def quat(roll, pitch, yaw, device):
    q = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_quat()
    return torch.tensor(q[[3, 0, 1, 2]], device=device, dtype=torch.float32)


def write_pose(env, row, position, orientation):
    ids = torch.tensor([row], device=env.device)
    robot = env.scene["robot"]
    pose = torch.cat((torch.tensor(position, device=env.device, dtype=torch.float32), orientation))[None]
    robot.write_root_link_pose_to_sim(pose, env_ids=ids)
    robot.write_root_link_velocity_to_sim(torch.zeros(1, 6, device=env.device), env_ids=ids)
    env.sim.forward()
    check(not bool(wp.to_torch(env.sim.wp_data.overflow).any()), "physics overflow after pose refresh")


def foot_vertices(env, row):
    """Compiled collision mesh vertices at actual simulated qpos, not link origins."""
    from hexapod_mjlab.tasks.confined_locomotion_2.flat_env_cfg import FOOT_BODIES
    model, data = env.sim.mj_model, mujoco.MjData(env.sim.mj_model)
    data.qpos[:] = wp.to_torch(env.sim.wp_data.qpos)[row].cpu().numpy()
    mujoco.mj_kinematics(model, data)
    result = []
    for gid in range(model.ngeom):
        name = model.body(int(model.geom_bodyid[gid])).name.split("/")[-1]
        if name not in FOOT_BODIES or not (model.geom_contype[gid] or model.geom_conaffinity[gid]):
            continue
        check(model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH, "foot clearance requires actual mesh vertices")
        mid = int(model.geom_dataid[gid])
        adr, num = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
        result.append(np.asarray(model.mesh_vert[adr:adr+num]) @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid])
    check(len(result) == 6, "expected six actual foot collision meshes")
    return np.concatenate(result)


def model_hash(model):
    binary = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, binary)
    return hashlib.sha256(binary.tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"stage": "geometry_only", "training": False}
    env = None
    started = time.monotonic()
    try:
        check(torch.cuda.is_available(), "real CUDA required")
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.tasks.registry import list_tasks
        from hexapod_mjlab.tasks.confined_locomotion_2.channels import load_sparse_selection
        from hexapod_mjlab.tasks.confined_locomotion_2.config.terrain_geometry import make_geometry_env_cfg
        from hexapod_mjlab.tasks.confined_locomotion_2.config.terrain_memory import TASK_ID
        from hexapod_mjlab.common.reset import reset_env_preserving_unselected_physics
        from hexapod_mjlab.tasks.confined_locomotion_2.terrain_geometry import HeightGridCfg, TerrainGeometryQuery
        from hexapod_mjlab.common.diagnostics import ContactTableReader

        device = f"cuda:{args.gpu}"
        registered_before = list_tasks()
        cfg = make_geometry_env_cfg(terrain=fixture_terrain(), num_envs=2, seed=20261025)
        check(not cfg.rewards and not cfg.terminations and not cfg.commands, "geometry gate must not install training terms")
        env = ManagerBasedRlEnv(cfg, device=device)
        check(not bool(wp.to_torch(env.sim.wp_data.overflow).any()), "physics overflow during construction")
        original_sim_step = env.sim.step
        def checked_sim_step():
            original_sim_step()
            check(not bool(wp.to_torch(env.sim.wp_data.overflow).any()), "physics overflow within a 200 Hz substep")
        env.sim.step = checked_sim_step
        # A nonzero, distinct origin proves world queries do not add it twice.
        env.scene.env_origins[1, 1] += 0.3
        env.reset(seed=20261025)
        check(not bool(wp.to_torch(env.sim.wp_data.overflow).any()), "physics overflow during initial reset")
        query = TerrainGeometryQuery(env)
        check(not np.any(env.sim.mj_model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE), "no hidden plane under the Heightfield")
        check(list_tasks() == registered_before, "geometry/collection changed the task registry")
        grid = HeightGridCfg((-1.2, 1.2), (-0.9, 0.9))
        rows = torch.arange(2, device=device)
        episodes = torch.zeros(2, dtype=torch.int64, device=device)
        report["candidate_grid"] = grid.identity()
        report["surface"] = query.surface.identity
        report["compiled_model_sha256"] = model_hash(env.sim.mj_model)
        initial_foot = foot_vertices(env, 0)
        ground, valid = query.surface.reference_heights(initial_foot[:, :2])
        clearance = initial_foot[:, 2] - ground
        check(valid.all() and clearance.min() > 0.001, "spawn collision geometry does not clear the actual Heightfield")
        root = env.scene["robot"].data.root_link_pos_w[0].cpu().numpy()
        report["spawn_foot_mesh_clearance_min_m"] = float(clearance.min())
        report["nominal_foot_collision_extent_relative_root_xy_m"] = {
            "min": (initial_foot[:, :2] - root[:2]).min(axis=0).tolist(),
            "max": (initial_foot[:, :2] - root[:2]).max(axis=0).tolist(),
            "scope": "one settled witness pose, not the full moving workspace"}

        errors = []
        for row, pose in enumerate(((1.3, 0.2, 0.65), (0., 1.8, 0.65))):
            write_pose(env, row, pose, quat(0.19, -0.13, 0.7 if row == 0 else 0., device))
        labels = query.height_labels(rows, episodes, env.episode_length_buf.clone(), grid)
        expected, valid = query.surface.reference_heights(labels.query_xy_w.cpu().numpy())
        check(np.array_equal(labels.valid.cpu().numpy(), valid), "label geometry mask differs from compiled surface")
        target = expected - labels.root_pos_w[:, None, None, 2].cpu().numpy()
        errors.append(float(np.abs(labels.heights.cpu().numpy()[valid] - target[valid]).max()))
        difference = abs(labels.heights.cpu().numpy() - target)
        worst = np.unravel_index(difference.argmax(), difference.shape)
        report["height_reference_comparison"] = {
            "max_error_m": errors[-1], "worst_grid_index": list(map(int, worst)),
            "worst_world_xy": labels.query_xy_w.cpu().numpy()[worst].tolist(),
            "native_height": float(labels.heights.cpu().numpy()[worst]),
            "reference_height": float(target[worst]),
        }
        check(errors[-1] < 2e-5, "native height labels differ from exact compiled triangles")
        check(bool((labels.heights[0] < 0).any()) and bool((labels.heights[1] > 0).any()), "signed root-relative label heights lost")
        # Same XY/yaw, different roll/pitch: targets and XY remain the same.
        retained = labels.heights.clone()
        write_pose(env, 0, (1.3, 0.2, 0.65), quat(-0.23, 0.27, 0.7, device))
        tilted = query.height_labels(rows[:1], episodes[:1], env.episode_length_buf[:1].clone(), grid)
        check(torch.allclose(tilted.query_xy_w, labels.query_xy_w[:1], atol=1e-6, rtol=0), "grid follows roll/pitch")
        check(torch.allclose(tilted.heights, labels.heights[:1], atol=2e-6, rtol=0), "height label changes with roll/pitch alone")
        check(torch.equal(labels.heights, retained), "retained labels aliased live state")
        report["coordinates_and_signed_labels"] = True

        # Native selected rows may be reordered; neither calls nor reads mutate Data.
        poses_before = wp.to_torch(env.sim.wp_data.geom_xpos).clone()
        reverse = rows.flip(0)
        selected = query.height_labels(reverse, episodes.flip(0), env.episode_length_buf.flip(0).clone(), grid)
        full_labels = query.height_labels(rows, episodes, env.episode_length_buf.clone(), grid)
        check(torch.equal(selected.heights, full_labels.heights.flip(0)), "compact world rows mapped to wrong geometry")
        check(torch.equal(poses_before, wp.to_torch(env.sim.wp_data.geom_xpos)), "queries mutate live geometry")
        with torch.cuda.stream(torch.cuda.Stream(device=device)):
            other_stream = query.height_labels(rows, episodes, env.episode_length_buf.clone(), grid)
        torch.cuda.synchronize(device)
        check(torch.equal(other_stream.heights, full_labels.heights), "query differs on another PyTorch stream")
        report["selected_worlds_and_read_purity"] = True

        # Boundary is geometrically invalid even if another ray would see a skirt.
        write_pose(env, 0, (4.9, 0., 0.65), quat(0, 0, 0, device))
        edge = query.height_labels(rows[:1], episodes[:1], env.episode_length_buf[:1].clone(), grid)
        check(bool(edge.valid.any()) and bool((~edge.valid).any()), "finite terrain boundary mask missing")
        check(bool((edge.heights[~edge.valid] == 0).all()), "invalid height slots not zero")
        report["finite_boundary_mask"] = True

        # Full and sparse query the actual tilted-pose scene, not a proxy.
        write_pose(env, 0, (1.3, 0.2, 0.65), quat(0.19, -0.13, 0.7, device))
        selection = ROOT / "src/hexapod_mjlab/tasks/confined_locomotion_2/assets/fixed_channels/v1_hit10/near_x6_far_x10.npz"
        sparse_ids, _ = load_sparse_selection(selection, query.catalog)
        full = query.scan(rows[:1])
        sparse = query.scan(rows[:1], source_ids=torch.tensor(sparse_ids, device=device))
        check(torch.equal(full.hit[:, sparse.source_ids], sparse.hit), "full/sparse source hit identity differs")
        ray_error = float((full.distances[:, sparse.source_ids] - sparse.distances).abs().max())
        check(ray_error < 2e-5, "full/sparse same-channel distances differ")
        # Independent classic MuJoCo rays on source hits and misses.
        model, host = env.sim.mj_model, mujoco.MjData(env.sim.mj_model)
        host.qpos[:] = wp.to_torch(env.sim.wp_data.qpos)[0].cpu().numpy()
        mujoco.mj_kinematics(model, host)
        # Obtain physical sensor origin from the compiled site pose.
        site_id = next(i for i in range(model.nsite) if model.site(i).name.endswith("lidar_sensor"))
        origin = host.site_xpos[site_id]
        selected_ids = np.linspace(0, 86399, 160, dtype=int)
        cpu_ray_errors = []
        for index in selected_ids:
            endpoint = full.points_world[0, index].cpu().numpy()
            direction = endpoint - origin
            direction /= np.linalg.norm(direction)
            geomid = np.zeros(1, dtype=np.int32)
            cpu_dist = mujoco.mj_ray(model, host, origin, direction, np.array([0,0,0,1,0,0], dtype=np.uint8), 1, -1, geomid)
            cpu_hit = cpu_dist >= 0 and cpu_dist < 15
            check(cpu_hit == bool(full.hit[0, index]), "native ray differs from classic MuJoCo hit")
            cpu_ray_errors.append(abs((cpu_dist if cpu_hit else 15.) - float(full.distances[0, index])))
        check(max(cpu_ray_errors) < 3e-4, "native ray distances disagree with classic MuJoCo")
        hit_points = full.points_world[full.hit].cpu().numpy()
        top_z, inside = query.surface.reference_heights(hit_points[:, :2])
        # Exclude the solid field's vertical outer skirt from top-surface checks.
        interior = inside & (abs(hit_points[:, 0]) < 4.999) & (abs(hit_points[:, 1]) < 3.999)
        top_error = float(abs(hit_points[interior, 2] - top_z[interior]).max())
        check(top_error < 2e-5, "LiDAR top hits do not lie on the height-label surface")
        check(sparse.near_points.shape == (1,256,3) and sparse.far_points.shape == (1,128,3), "scan consumption shapes changed")
        check(bool((sparse.far_points[~sparse.far_valid] == 0).all()), "FPS shortage fabricated points")
        report["raycast"] = {"full_rays": 86400, "sparse_rays": len(sparse_ids),
                             "same_channel_max_error_m": ray_error,
                             "classic_cpu_rays": len(selected_ids), "classic_cpu_max_error_m": max(cpu_ray_errors),
                             "top_surface_hits_checked": int(interior.sum()), "top_surface_max_error_m": top_error,
                             "backend": "mujoco_warp.rays / compiled HField BVH",
                             "near_valid": int(sparse.near_valid.sum()), "far_valid": int(sparse.far_valid.sum())}

        env.reset()
        episodes += 1
        # Actual contact on the NON-FLAT region. Place the foot mesh 2 mm above
        # the compiled triangles, then let the existing PD controller settle.
        write_pose(env, 1, (1.8, 0.2, 0.8), quat(0, 0, 0.3, device))
        feet = foot_vertices(env, 1)
        gz, valid = query.surface.reference_heights(feet[:, :2])
        check(valid.all(), "foot collision mesh outside actual terrain")
        z = .8 - float((feet[:, 2]-gz).min()) + .002
        write_pose(env, 1, (1.8, 0.2, z), quat(0, 0, .3, device))
        reader = ContactTableReader(env)
        contact_errors, contact_distances, contacting_feet = [], [], set()
        scan_counts = [0, 0]
        next_due = [int(env.common_step_counter)] * 2
        retained_label = None
        first_scan_label = None
        motion_between_scans = False
        lifecycle_start = int(env.common_step_counter)
        for tick in range(80):
            now = int(env.common_step_counter)
            current = query.height_labels(rows, episodes, env.episode_length_buf.clone(), grid)
            if tick == 0:
                first_scan_label = current.heights.clone()
            elif tick < 5 and not torch.equal(first_scan_label, current.heights):
                motion_between_scans = True
            if retained_label is not None:
                check(torch.equal(retained_label.heights, retained_copy), "retained pre-action label changed after stepping")
            retained_label, retained_copy = current, current.heights.clone()
            due = [row for row in range(2) if now >= next_due[row]]
            if due:
                scan = query.scan(torch.tensor(due, device=device), source_ids=torch.tensor(sparse_ids, device=device))
                check(scan.env_ids.tolist() == due, "scan row ownership lost")
                for row in due:
                    scan_counts[row] += 1
                    next_due[row] = now+5
            env.step(torch.zeros(2,18,device=device))
            check(not bool(wp.to_torch(env.sim.wp_data.overflow).any()), "physics/contact overflow during geometry lifecycle")
            table = reader.read_substep_table()
            for i, pair in enumerate(table["geom"]):
                if table["worldid"][i] != 1 or query.surface.geom_id not in pair:
                    continue
                other = int(pair[1] if pair[0] == query.surface.geom_id else pair[0])
                name = model.body(int(model.geom_bodyid[other])).name.split("/")[-1]
                if not name.endswith("_FOOT") or np.linalg.norm(table["force6_world"][i,:3]) <= .1:
                    continue
                point = table["pos"][i]
                expected_z, valid = query.surface.reference_heights(point[None,:2])
                check(valid[0], "support contact outside labelled Heightfield")
                error = abs(point[2] - expected_z[0])
                # HField collision uses convex triangular prisms. Its contact
                # midpoint/witness may belong to a prism face within the solid;
                # it is not an exact top-surface ray intersection. Check actual
                # terrain ownership and bounded soft-contact proximity instead.
                check(error < abs(float(table["dist"][i])) + .004, "physical contact surface and height label disagree")
                check(error < .01, "physical contact midpoint exceeds 1 cm top-surface proximity gate")
                contact_errors.append(float(error))
                contact_distances.append(float(table["dist"][i]))
                contacting_feet.add(name)
            if tick == 26:
                # Asynchronous selected reset: protect all unselected physics.
                before = {name: wp.to_torch(getattr(env.sim.wp_data,name))[1].clone()
                          for name in ("qpos","qvel","qacc_warmstart","time")}
                untouched = query.height_labels(rows[1:], episodes[1:], env.episode_length_buf[1:].clone(), grid)
                reset_env_preserving_unselected_physics(env, rows[:1])
                episodes[0] += 1
                for name, values in before.items():
                    check(torch.equal(values, wp.to_torch(getattr(env.sim.wp_data,name))[1]), "partial reset changed unselected physics")
                after_reset = query.height_labels(rows, episodes, env.episode_length_buf.clone(), grid)
                check(torch.equal(after_reset.heights[1:], untouched.heights), "partial reset changed unselected height label")
                check(int(after_reset.step_ids[0]) == 0 and int(after_reset.episode_ids[0]) == 2, "fresh episode label identity incorrect")
                check(int(after_reset.step_ids[1]) == 27, "unselected control identity reset")
                query.scan(rows[:1], source_ids=torch.tensor(sparse_ids,device=device))
                scan_counts[0] += 1
                next_due[0] = int(env.common_step_counter)+5
                report["partial_reset_isolation"] = True
        check(len(contact_errors) > 0 and len(contacting_feet) >= 3, "real robot did not support on the non-flat Heightfield")
        check(motion_between_scans, "height labels did not update at control rate between scans")
        check(scan_counts == [17,16], f"unexpected 10 Hz scan counts after asynchronous reset: {scan_counts}")
        try:
            query.height_labels(rows, episodes, env.episode_length_buf.clone()+1, grid)
        except ValueError:
            report["future_control_identity_rejected"] = True
        else:
            raise AssertionError("future label accepted")
        report["contact"] = {"qualified_contacts": len(contact_errors), "foot_bodies": sorted(contacting_feet),
                             "max_contact_midpoint_surface_error_m": max(contact_errors),
                             "min_contact_distance_m": min(contact_distances),
                             "scope": "actual foot forces on compiled Heightfield; soft contact proximity, not exact ray intersections"}
        report["lifecycle"] = {"control_steps": int(env.common_step_counter)-lifecycle_start,
                               "scan_counts": scan_counts, "labels_update_between_scans": motion_between_scans}
        report["label_max_error_m"] = max(errors)
        report["physics_overflow_free"] = True
        report["passed"] = True
    except BaseException as error:
        report.update(passed=False, error=repr(error), traceback=traceback.format_exc())
    finally:
        if env is not None:
            if "original_sim_step" in locals():
                env.sim.step = original_sim_step
            env.close()
        report["elapsed_s"] = time.monotonic()-started
        (args.output/"checks.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
        print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
