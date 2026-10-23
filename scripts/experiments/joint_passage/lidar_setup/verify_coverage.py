#!/usr/bin/env python
"""Independent replay of saved coverage identities, intersections and selections.

Does not run the producer's analytic/stats functions or reselect points. Checks
all 175 raw frames and all 700 outputs, including normal no-hit measurements.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from compare_flat import ROOT, MODEL, LIBRARY, load_model, frame

SURFACES = ['ground', 'left_wall', 'right_wall', 'ceiling', 'normal_nohit']
BOUNDARIES = []


def intersect(origin, rays, width, roof):
    # Construct explicit planes; choose the nearest admissible intersection.
    planes = [(np.array([0., 0., 1.]), 0., 0)]
    if width is not None:
        planes += [(np.array([0., 1., 0.]), width, 1),
                   (np.array([0., 1., 0.]), -width, 2)]
    if roof is not None:
        planes += [(np.array([0., 0., 1.]), roof, 3)]
    distance = np.full(len(rays), 15.)
    label = np.full(len(rays), 4)
    for normal, offset, surface in planes:
        denominator = rays @ normal
        length = np.divide(offset - origin @ normal, denominator,
                           out=np.full(len(rays), np.inf), where=np.abs(denominator) > 1e-12)
        with np.errstate(invalid='ignore'):
            hit = origin + length[:, None] * rays
        valid = (length >= 0) & (length < distance)
        if surface:
            valid &= (hit[:, 0] >= -2) & (hit[:, 0] <= 6)
            if surface in (1, 2):
                valid &= hit[:, 2] >= 0
        distance[valid] = length[valid]
        label[valid] = surface
    return distance, label


def population(label):
    return {name: int(np.count_nonzero(label == i)) for i, name in enumerate(SURFACES)}


def check_stats(saved, ids, xyz, labels, blocked, sensor):
    p, lab, oc = xyz[ids], labels[ids], blocked[ids]
    walls = np.isin(lab, [1, 2])
    assert saved['points'] == len(ids)
    assert saved['surfaces'] == population(lab)
    assert saved['occluded_surfaces'] == population(lab[oc])
    assert saved['occluded'] == int(oc.sum())
    for key, margin in [
        ('wall_ahead_of_sensor', p[:, 0] - sensor[0]),
        ('wall_behind_body_root', -p[:, 0]),
        ('wall_in_startup_reference_x', np.minimum(p[:, 0] + .89, .89 - p[:, 0])),
        ('wall_above_sensor_body_z', p[:, 2] - sensor[2]),
    ]:
        value = int(np.count_nonzero(walls & (margin > 0)))
        if saved[key] != value:
            # Reconstructed float32 ray normalization can differ by one ULP
            # from Warp. Only positional categories at the plane may vary;
            # exact surface identities and primary population counts may not.
            lower = int(np.count_nonzero(walls & (margin > 5e-6)))
            upper = int(np.count_nonzero(walls & (margin >= -5e-6)))
            assert lower <= saved[key] <= upper, (key, saved[key], value, lower, upper)
            BOUNDARIES.append({'category': key, 'saved': saved[key], 'reconstructed': value,
                               'boundary_raw_indices': ids[walls & (np.abs(margin) <= 5e-6)].tolist()})
    if walls.any():
        np.testing.assert_allclose(saved['wall_body_xyz_bounds_m'],
                                   [p[walls].min(0), p[walls].max(0)], atol=5e-6, rtol=0)
    else:
        assert saved['wall_body_xyz_bounds_m'] is None


def check_gaps(saved, query, sample):
    if not len(query) or not len(sample):
        assert saved is None
        return
    # Brute force 187 x 256, independent of the producer's KD tree.
    nearest = np.linalg.norm(query[:, None, :] - sample[None, :, :], axis=2).min(1)
    expected = np.r_[np.percentile(nearest, [5, 50, 95]), nearest.max()]
    np.testing.assert_allclose(saved['p05_p50_p95_max_m'], expected, atol=5e-6, rtol=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.run = args.run.resolve()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    report = json.loads((args.run / 'report.json').read_text())
    data = np.load(args.run / 'raw.npz')
    for path, digest in report['sha256'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    records = json.loads(LIBRARY.read_text())['records']
    assert data['case_ids'].tolist() == [r['id'] for r in records]
    np.testing.assert_array_equal(data['qpos'], [r['measured_qpos'] for r in records])
    cases = 175 * len(report['recipes'])
    assert report['raw_scans'] == 175 and report['selection_cases'] == cases
    assert len(report['rows']) == cases
    model = load_model(MODEL)
    rows = {(r['scene'], r['recipe'], r['case']): r for r in report['rows']}
    assert len(rows) == cases
    rays = data['pattern_dirs']
    assert rays.shape == (86400, 3)
    np.testing.assert_allclose(np.linalg.norm(rays, axis=1), 1, atol=1e-12)
    gx, gy = np.meshgrid(np.linspace(1.2, 3.8, 17), np.linspace(-.9, .9, 11))
    grid = np.c_[gx.ravel(), gy.ravel(), np.zeros(187)]
    max_error = 0.
    checked = 0
    angular_boundaries = []
    for k, rec in enumerate(records):
        _, rb, pb, rs, ps, rotation, translation = frame(model, data['qpos'][k])
        for scene, width, roof in report['scenes']:
            length = data[scene + '/raw_ranges'][k]
            expected, label = intersect(ps, rays @ rs.T, width, roof)
            np.testing.assert_array_equal(data[scene + '/raw_labels'][k], label)
            np.testing.assert_array_equal(length < 15, expected < 15)
            assert np.isfinite(length).all() and (length >= 0).all() and (length <= 15).all()
            error = float(np.abs(length - expected).max())
            assert error <= .001
            max_error = max(max_error, error)
            # Production pixels are normalized float32 directions times ranges.
            local_rays = rays.astype(np.float32)
            local_rays /= np.linalg.norm(local_rays, axis=1)[:, None]
            xyz = (local_rays * length[:, None]).astype(float) @ rotation.T + translation
            blocked = (data['robot_distance'][k] >= 0) & (data['robot_distance'][k] < expected - 1e-6)
            yaw = math.atan2(rb[1, 0], rb[0, 0])
            world_grid = grid @ Rotation.from_euler('z', yaw).as_matrix().T + pb
            world_grid[:, 2] = 0
            body_grid = (world_grid - pb) @ rb
            rel_grid = world_grid - ps
            grid_range = np.linalg.norm(rel_grid, axis=1)
            gd, gl = intersect(ps, rel_grid / grid_range[:, None], width, roof)
            visible_ground = (gl == 0) & (np.abs(gd - grid_range) < 1e-6)
            dome = rel_grid @ rs[:, 2] >= 0
            for name, theta, delta, azimuths, elevations in report['recipes']:
                prefix = scene + '/' + name + '/'
                pool = data[name + '/pool_indices']
                az = np.arange(0, 900, 900 // azimuths)
                el = np.round(np.linspace(0, 95, elevations)).astype(int)
                np.testing.assert_array_equal(pool, (az[:, None] * 96 + el).ravel())
                assert len(np.unique(pool)) == 2880 and el[0] == 0 and el[-1] == 95
                pool_xyz = data[prefix + 'pool_body'][k]
                np.testing.assert_allclose(pool_xyz, xyz[pool], atol=5e-6, rtol=0)
                np.testing.assert_array_equal(data[prefix + 'pool_labels'][k], label[pool])
                np.testing.assert_array_equal(data[prefix + 'pool_occluded'][k], blocked[pool])
                angle = math.radians(delta)
                ry = np.array([[math.cos(angle), 0, math.sin(angle)], [0, 1, 0],
                               [-math.sin(angle), 0, math.cos(angle)]])
                axis = ry @ rotation[:, 2]
                threshold = math.sin(math.radians(theta))
                margin = (pool_xyz - translation) @ axis / np.linalg.norm(pool_xyz - translation, axis=1) - threshold
                near = data[prefix + 'near_mask'][k]
                assert np.all(near[margin > 1e-6]) and not np.any(near[margin < -1e-6])
                fullnear = (rays @ rotation.T) @ axis >= threshold
                fullkey = prefix + 'full_near_mask'
                if fullkey in data:
                    saved_full = data[fullkey][k]
                    assert saved_full.shape == (86400,) and saved_full.dtype == np.bool_
                    different = np.flatnonzero(saved_full != fullnear)
                    margin_full = (rays @ rotation.T) @ axis - threshold
                    assert np.all(np.abs(margin_full[different]) <= 1e-12)
                    if len(different):
                        angular_boundaries.append({'scene': scene, 'case': rec['id'], 'recipe': name,
                                                   'raw_indices': different.tolist(),
                                                   'independent_margin': margin_full[different].tolist()})
                    fullnear = saved_full
                row = rows[scene, name, rec['id']]
                for branch, mask in [('near', near), ('far', ~near)]:
                    check_stats(row['full'][branch], np.flatnonzero(fullnear if branch == 'near' else ~fullnear),
                                xyz, label, blocked, translation)
                    check_stats(row['pool'][branch], pool[mask], xyz, label, blocked, translation)
                    ids = data[prefix + 'final_' + branch + '_raw_index'][k]
                    points = data[prefix + 'final_' + branch + '_body'][k]
                    assert ids.shape == ((256,) if branch == 'near' else (64,))
                    assert len(np.unique(ids)) == len(ids)
                    assert np.isin(ids, pool[mask]).all()
                    np.testing.assert_allclose(points, xyz[ids], atol=5e-6, rtol=0)
                    check_stats(row['final'][branch], ids, xyz, label, blocked, translation)
                old = row['legacy_grid']
                visible = visible_ground & dome & ((body_grid - translation) @ axis / grid_range >= threshold)
                assert old['scene_unoccluded_ground_queries'] == int(visible_ground.sum())
                assert old['near_angular_scene_visible_queries'] == int(visible.sum())
                ground_pool = pool_xyz[near & (label[pool] == 0)]
                ids = data[prefix + 'final_near_raw_index'][k]
                ground_final = data[prefix + 'final_near_body'][k][label[ids] == 0]
                check_gaps(old['pool_ground_distance_all_queries'], body_grid, ground_pool)
                check_gaps(old['final_ground_distance_all_queries'], body_grid, ground_final)
                check_gaps(old['final_ground_distance_visible_queries'], body_grid[visible], ground_final)
                checked += 1
    result = {'status': 'PASS', 'raw_frames': 175, 'selection_cases': checked,
              'max_independent_range_error_m': max_error,
              'point_tolerance_m': 5e-6, 'range_tolerance_m': .001,
              'near_boundary_tolerance': 1e-6,
              'positional_boundary_differences': BOUNDARIES,
              'full_angular_boundary_differences': angular_boundaries,
              'scope': 'identities, bounded plane intersections, masks, point sources, surface/occlusion statistics and all query gaps; no reselection or dynamics',
              'sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in [Path(__file__), args.run / 'raw.npz', args.run / 'report.json']}}
    (args.out / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    (args.out / Path(__file__).name).write_text(Path(__file__).read_text())
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
