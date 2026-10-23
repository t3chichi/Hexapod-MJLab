#!/usr/bin/env python
"""Approved cone replay: down-tilt or aperture, same saved scans/pool.

Uses the production wrapper on exact saved float32 inputs. No raycasting,
physics, history-window evaluation, sampling change or candidate adoption.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from compare_coverage import ROOT, MODEL, LIBRARY, load_model, frame, analytic, stats, nearest_distances, LidarWrapper

RECIPES = [(f'D{delta}', 32, delta, 30, 96) for delta in [56, 48, 40, 32]]
APERTURES = [(f'T{theta}', theta, 40, 30, 96) for theta in [24, 28, 32, 36, 40]]


def spatial_stats(ids, points, label, blocked, translation):
    result = stats(ids, points, label, blocked, translation)
    p, lab = points[ids], label[ids]
    ground = p[lab == 0]
    walls = p[np.isin(lab, [1, 2])]
    result['ground_body_xyz_bounds_m'] = [ground.min(0).tolist(), ground.max(0).tolist()] if len(ground) else None
    result['wall_at_or_above_root_body_z'] = int((walls[:, 2] >= 0).sum())
    result['wall_at_or_above_root_body_xyz_bounds_m'] = [walls[walls[:, 2] >= 0].min(0).tolist(), walls[walls[:, 2] >= 0].max(0).tolist()] if result['wall_at_or_above_root_body_z'] else None
    result['wall_closest_root_height_point_BODY_m'] = walls[np.argmin(np.abs(walls[:, 2]))].tolist() if len(walls) else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--sweep', choices=['tilt', 'aperture'], default='tilt')
    parser.add_argument('--reference', type=Path, help='accepted down-tilt run; required for aperture')
    args = parser.parse_args()
    source, out = args.source.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(1)
    recipes = RECIPES if args.sweep == 'tilt' else APERTURES
    reference = None
    reference_report = None
    if args.sweep == 'aperture':
        if args.reference is None:
            parser.error('--reference is required for aperture')
        args.reference = args.reference.resolve()
        reference_report = json.loads((args.reference / 'report.json').read_text())
        assert reference_report['recipes'] == [list(x) for x in RECIPES]
        reference = np.load(args.reference / 'raw.npz')
    prior = json.loads((source / 'report.json').read_text())
    for path, digest in prior['sha256'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    data = np.load(source / 'raw.npz')
    model = load_model(MODEL)
    records = json.loads(LIBRARY.read_text())['records']
    assert len(records) == 35 and prior['raw_scans'] == 175
    assert data['case_ids'].tolist() == [r['id'] for r in records]
    np.testing.assert_array_equal(data['qpos'], [r['measured_qpos'] for r in records])
    rays = data['pattern_dirs']
    pool = data['C0/pool_indices']
    np.testing.assert_array_equal(pool, (np.arange(0, 900, 30)[:, None] * 96 + np.arange(96)).ravel())
    arrays = {key: data[key] for key in ['qpos', 'case_ids', 'pattern_dirs', 'robot_distance', 'robot_geomid']}
    if reference_report is not None:
        assert reference_report['input_source'] == str(source.relative_to(ROOT))
        for path, digest in reference_report['input_sha256'].items():
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    for name, *_ in recipes:
        arrays[name + '/pool_indices'] = pool
    gx, gy = np.meshgrid(np.linspace(1.2, 3.8, 17), np.linspace(-.9, .9, 11))
    grid = np.c_[gx.ravel(), gy.ravel(), np.zeros(187)]
    rows = []
    unchanged_replays = 0
    for scene, width, roof in prior['scenes']:
        arrays[scene + '/raw_ranges'] = data[scene + '/raw_ranges']
        arrays[scene + '/raw_labels'] = data[scene + '/raw_labels']
        packs = {name: {key: [] for key in ['pool_body', 'pool_labels', 'pool_occluded', 'near_mask', 'full_near_mask',
                    'final_near_body', 'final_far_body', 'final_near_raw_index', 'final_far_raw_index']} for name, *_ in recipes}
        for k, rec in enumerate(records):
            _, rb, pb, rs, ps, rotation, translation = frame(model, rec['measured_qpos'])
            lengths = data[scene + '/raw_ranges'][k]
            local_rays = rays.astype(np.float32)
            local_rays /= np.linalg.norm(local_rays, axis=1)[:, None]
            points = (local_rays * lengths[:, None]).astype(float) @ rotation.T + translation
            label = data[scene + '/raw_labels'][k]
            expected, independent_label = analytic(ps, rays @ rs.T, width, roof)
            np.testing.assert_array_equal(label, independent_label)
            blocked = (data['robot_distance'][k] >= 0) & (data['robot_distance'][k] < expected - 1e-6)
            # Keep exact saved pool pixels, not reconstructed coordinates, for
            # production classification/sorting: ULP ties affect far selection.
            p = data[scene + '/C0/pool_body'][k]
            points[pool] = p
            yaw = math.atan2(rb[1, 0], rb[0, 0])
            world_grid = grid @ Rotation.from_euler('z', yaw).as_matrix().T + pb
            world_grid[:, 2] = 0
            body_grid = (world_grid - pb) @ rb
            rel_grid = world_grid - ps
            grid_range = np.linalg.norm(rel_grid, axis=1)
            gd, gl = analytic(ps, rel_grid / grid_range[:, None], width, roof)
            scene_visible = (gl == 0) & (np.abs(gd - grid_range) < 1e-6)
            dome = rel_grid @ rs[:, 2] >= 0
            for name, theta, delta, *_ in recipes:
                axis = Rotation.from_euler('y', delta, degrees=True).apply(rotation[:, 2])
                wrapper = LidarWrapper(1, 2880, 1, 256, 64, theta, 0, torch.device('cpu'),
                                      torch.tensor(Rotation.from_matrix(rotation).as_quat()[None], dtype=torch.float32),
                                      torch.tensor(translation[None], dtype=torch.float32), torch.tensor(axis, dtype=torch.float32))
                inputs = torch.tensor(p[None], dtype=torch.float32)
                norms = torch.norm(wrapper._to_sensor_frame(inputs), dim=-1)
                near = (((inputs - wrapper._sensor_t) * wrapper._cone_axis).sum(-1) / (norms + 1e-9) >= wrapper._cone_sin_thr)[0].numpy()
                assert near.sum() >= 256 and (~near).sum() >= 64, (scene, rec['id'], name)
                with torch.no_grad():
                    selected = wrapper.wrap_obs(torch.empty(1, 0), inputs, torch.zeros(1, dtype=torch.bool)).reshape(320, 3).numpy()
                raw_ids = []
                for chunk, mask in [(selected[:256], near), (selected[256:], ~near)]:
                    subset = np.flatnonzero(mask)
                    distance, nearest = cKDTree(p[mask]).query(chunk)
                    assert distance.max() < 5e-5
                    raw_ids.append(pool[subset[nearest]])
                ni, fi = raw_ids
                # Exact equality against two existing columns certifies that
                # replay has not silently altered sampling or input precision.
                if theta == 32 and delta in [56, 48]:
                    old = 'C0' if delta == 56 else 'C2'
                    np.testing.assert_array_equal(near, data[scene + '/' + old + '/near_mask'][k])
                    for branch, values in [('near', selected[:256]), ('far', selected[256:])]:
                        np.testing.assert_array_equal(values, data[scene + '/' + old + '/final_' + branch + '_body'][k])
                    np.testing.assert_array_equal(ni, data[scene + '/' + old + '/final_near_raw_index'][k])
                    np.testing.assert_array_equal(fi, data[scene + '/' + old + '/final_far_raw_index'][k])
                    unchanged_replays += 1
                if reference is not None and theta == 32:
                    ref_prefix = scene + '/D40/'
                    for key, values in [('near_mask', near), ('final_near_body', selected[:256]),
                                        ('final_far_body', selected[256:]), ('final_near_raw_index', ni),
                                        ('final_far_raw_index', fi)]:
                        np.testing.assert_array_equal(values, reference[ref_prefix + key][k])
                    unchanged_replays += 1
                fullnear = (rays @ rotation.T) @ axis >= math.sin(math.radians(theta))
                gridnear = (body_grid - translation) @ axis / grid_range >= math.sin(math.radians(theta))
                angular_visible = scene_visible & dome & gridnear
                fixed_visible = scene_visible & dome
                final_ground = selected[:256][label[ni] == 0]
                row = {'scene': scene, 'case': rec['id'], 'recipe': name, 'theta_deg': theta, 'delta_deg': delta,
                       'cone_upper_BODY_elevation_deg': float(np.rad2deg(np.arcsin(axis[2])) + 90 - theta),
                       'full': {'near': spatial_stats(np.flatnonzero(fullnear), points, label, blocked, translation),
                                'far': spatial_stats(np.flatnonzero(~fullnear), points, label, blocked, translation)},
                       'pool': {'near': spatial_stats(pool[near], points, label, blocked, translation),
                                'far': spatial_stats(pool[~near], points, label, blocked, translation)},
                       'final': {'near': spatial_stats(ni, points, label, blocked, translation),
                                 'far': spatial_stats(fi, points, label, blocked, translation)},
                       'legacy_grid': {'queries': 187, 'scene_unoccluded_ground_queries': int(scene_visible.sum()),
                                       'near_angular_scene_visible_queries': int(angular_visible.sum()),
                                       'fixed_scene_visible_queries': int(fixed_visible.sum()),
                                       'pool_ground_distance_all_queries': nearest_distances(body_grid, p[near & (label[pool] == 0)]),
                                       'final_ground_distance_all_queries': nearest_distances(body_grid, final_ground),
                                       'final_ground_distance_visible_queries': nearest_distances(body_grid[angular_visible], final_ground),
                                       'final_ground_distance_fixed_scene_visible_queries': nearest_distances(body_grid[fixed_visible], final_ground)}}
                rows.append(row)
                for key, val in [('pool_body', p), ('pool_labels', label[pool]), ('pool_occluded', blocked[pool]),
                                 ('near_mask', near), ('full_near_mask', fullnear), ('final_near_body', selected[:256]), ('final_far_body', selected[256:]),
                                 ('final_near_raw_index', ni), ('final_far_raw_index', fi)]:
                    packs[name][key].append(val)
        for name, pack in packs.items():
            for key, values in pack.items():
                arrays[scene + '/' + name + '/' + key] = np.array(values)
        print(scene, 'replayed', flush=True)
    np.savez_compressed(out / 'raw.npz', **arrays)
    hashes = dict(prior['sha256'])
    hashes[str(Path(__file__).relative_to(ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    source_files = {str((source / n).relative_to(ROOT)): hashlib.sha256((source / n).read_bytes()).hexdigest() for n in ['report.json', 'raw.npz']}
    if reference is not None:
        source_files.update({str((args.reference / n).relative_to(ROOT)): hashlib.sha256((args.reference / n).read_bytes()).hexdigest() for n in ['report.json', 'raw.npz']})
    report = {'status': 'static_replay_completed_no_adoption', 'scope': __doc__, 'scenes': prior['scenes'], 'recipes': recipes,
              'sweep': args.sweep, 'reference': str(args.reference.relative_to(ROOT)) if reference is not None else None,
              'raw_scans': 175, 'new_raw_scans': 0, 'selection_cases': 175 * len(recipes), 'unchanged_selection_replays': unchanged_replays,
              'input_source': str(source.relative_to(ROOT)), 'input_sha256': source_files,
              'sha256': hashes, 'rows': rows, 'elapsed_s': time.monotonic() - started}
    assert unchanged_replays == (350 if args.sweep == 'tilt' else 175)
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (out / Path(__file__).name).write_text(Path(__file__).read_text())
    print('completed', report['elapsed_s'], 's')


if __name__ == '__main__':
    main()
