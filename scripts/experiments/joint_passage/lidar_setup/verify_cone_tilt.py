#!/usr/bin/env python
"""Independent geometric replay plus source equality and new spatial metrics."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import verify_coverage as core
from compare_flat import ROOT, MODEL, load_model, frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run, out = args.run.resolve(), args.out.resolve()
    report = json.loads((run / 'report.json').read_text())
    aperture = report.get('sweep') == 'aperture'
    expected = [[f'T{x}', x, 40, 30, 96] for x in [24, 28, 32, 36, 40]] if aperture else [[f'D{x}', 32, x, 30, 96] for x in [56, 48, 40, 32]]
    assert report['recipes'] == expected
    replays = 175 if aperture else 350
    assert report['new_raw_scans'] == 0 and report['unchanged_selection_replays'] == replays
    for path, digest in report['input_sha256'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    source = np.load(ROOT / report['input_source'] / 'raw.npz')
    reference = np.load(ROOT / report['reference'] / 'raw.npz') if aperture else None
    data = np.load(run / 'raw.npz')
    # Full independent range/labels/indices/masks/statistics and legacy queries.
    core.main()
    model = load_model(MODEL)
    gx, gy = np.meshgrid(np.linspace(1.2, 3.8, 17), np.linspace(-.9, .9, 11))
    grid = np.c_[gx.ravel(), gy.ravel(), np.zeros(187)]
    rows = {(r['scene'], r['recipe'], r['case']): r for r in report['rows']}
    checked = 0
    for scene, width, roof in report['scenes']:
        for key in ['raw_ranges', 'raw_labels']:
            np.testing.assert_array_equal(data[scene + '/' + key], source[scene + '/' + key])
        for k, case in enumerate(data['case_ids']):
            _, rb, pb, rs, ps, r, t = frame(model, data['qpos'][k])
            rays = data['pattern_dirs'].astype(np.float32)
            rays /= np.linalg.norm(rays, axis=1)[:, None]
            xyz = (rays * data[scene + '/raw_ranges'][k][:, None]).astype(float) @ r.T + t
            pool = source['C0/pool_indices']
            xyz[pool] = source[scene + '/C0/pool_body'][k]
            label = data[scene + '/raw_labels'][k]
            yaw = math.atan2(rb[1, 0], rb[0, 0])
            world_grid = grid @ Rotation.from_euler('z', yaw).as_matrix().T + pb
            world_grid[:, 2] = 0
            body_grid = (world_grid - pb) @ rb
            relative = world_grid - ps
            grid_range = np.linalg.norm(relative, axis=1)
            gd, gl = core.intersect(ps, relative / grid_range[:, None], width, roof)
            fixed_visible = (gl == 0) & (np.abs(gd - grid_range) < 1e-6) & (relative @ rs[:, 2] >= 0)
            for name, theta, delta, *_ in report['recipes']:
                prefix = scene + '/' + name + '/'
                row = rows[scene, name, case]
                np.testing.assert_array_equal(data[prefix + 'pool_body'][k], source[scene + '/C0/pool_body'][k])
                if theta == 32 and delta in [56, 48]:
                    old = 'C0' if delta == 56 else 'C2'
                    for key in ['near_mask', 'final_near_body', 'final_far_body', 'final_near_raw_index', 'final_far_raw_index']:
                        np.testing.assert_array_equal(data[prefix + key][k], source[scene + '/' + old + '/' + key][k])
                if reference is not None and theta == 32:
                    for key in ['near_mask', 'final_near_body', 'final_far_body', 'final_near_raw_index', 'final_far_raw_index']:
                        np.testing.assert_array_equal(data[prefix + key][k], reference[scene + '/D40/' + key][k])
                angle = math.radians(delta)
                ry = np.array([[math.cos(angle), 0, math.sin(angle)], [0, 1, 0], [-math.sin(angle), 0, math.cos(angle)]])
                axis = ry @ r[:, 2]
                self_upper = math.degrees(math.asin(axis[2])) + 90 - theta
                assert abs(row['cone_upper_BODY_elevation_deg'] - self_upper) < 1e-10
                # core independently checks saved masks, recording all scalar
                # dot-product ties within 1e-12 before using them for statistics.
                fullnear = data[prefix + 'full_near_mask'][k]
                near = data[prefix + 'near_mask'][k]
                for level in ['full', 'pool', 'final']:
                    for branch in ['near', 'far']:
                        if level == 'full':
                            ids = np.flatnonzero(fullnear if branch == 'near' else ~fullnear)
                        elif level == 'pool':
                            ids = pool[near if branch == 'near' else ~near]
                        else:
                            ids = data[prefix + 'final_' + branch + '_raw_index'][k]
                        p, lab = xyz[ids], label[ids]
                        walls = p[np.isin(lab, [1, 2])]
                        ground = p[lab == 0]
                        saved = row[level][branch]
                        count = int((walls[:, 2] >= 0).sum())
                        assert saved['wall_at_or_above_root_body_z'] == count
                        for key, subset in [('ground_body_xyz_bounds_m', ground),
                                            ('wall_at_or_above_root_body_xyz_bounds_m', walls[walls[:, 2] >= 0])]:
                            if len(subset):
                                np.testing.assert_allclose(saved[key], [subset.min(0), subset.max(0)], atol=1e-12, rtol=0)
                            else:
                                assert saved[key] is None
                        point = saved['wall_closest_root_height_point_BODY_m']
                        if len(walls):
                            closest = walls[np.argmin(np.abs(walls[:, 2]))]
                            np.testing.assert_array_equal(point, closest)
                        else:
                            assert point is None
                ni = data[prefix + 'final_near_raw_index'][k]
                final_ground = data[prefix + 'final_near_body'][k][label[ni] == 0]
                assert row['legacy_grid']['fixed_scene_visible_queries'] == int(fixed_visible.sum())
                core.check_gaps(row['legacy_grid']['final_ground_distance_fixed_scene_visible_queries'], body_grid[fixed_visible], final_ground)
                checked += 1
    assert checked == (875 if aperture else 700)
    result = {'status': 'PASS', 'selection_cases': checked, 'exact_old_selection_replays': replays,
              'new_scans': 0, 'scope': 'same inputs; root-height wall coverage, ground bounds, fixed-query gaps and cone elevation verified independently',
              'sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), run / 'raw.npz', run / 'report.json']}}
    (out / 'spatial_metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    (out / Path(__file__).name).write_text(Path(__file__).read_text())
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
