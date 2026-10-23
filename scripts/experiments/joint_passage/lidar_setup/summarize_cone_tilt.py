#!/usr/bin/env python
"""All-pose spatial summaries of the approved down-tilt comparison."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def span(values):
    return {'min': float(min(values)), 'median': float(np.median(values)), 'max': float(max(values))} if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    report = json.loads((args.run / 'report.json').read_text())
    data = np.load(args.run / 'raw.npz')
    summaries = []
    paired = []
    aperture = report.get('sweep') == 'aperture'
    baseline_name = 'T32' if aperture else 'D56'
    for scene, *_ in report['scenes']:
        baseline = {r['case']: r for r in report['rows'] if r['scene'] == scene and r['recipe'] == baseline_name}
        for name, *_ in report['recipes']:
            rows = [r for r in report['rows'] if r['scene'] == scene and r['recipe'] == name]
            assert len(rows) == 35
            entry = {'scene': scene, 'recipe': name, 'states': 35}
            for level in ['full', 'pool', 'final']:
                for branch in ['near', 'far']:
                    for surface in ['ground', 'left_wall', 'right_wall', 'ceiling', 'normal_nohit']:
                        entry[f'{level}_{branch}_{surface}'] = span([r[level][branch]['surfaces'][surface] for r in rows])
                    for key in ['points', 'occluded', 'wall_at_or_above_root_body_z', 'wall_in_startup_reference_x']:
                        entry[f'{level}_{branch}_{key}'] = span([r[level][branch][key] for r in rows])
            entry['ground_angular_queries'] = span([r['legacy_grid']['near_angular_scene_visible_queries'] for r in rows])
            entry['fixed_ground_query_p95_m'] = span([r['legacy_grid']['final_ground_distance_fixed_scene_visible_queries']['p05_p50_p95_max_m'][2] for r in rows if r['legacy_grid']['final_ground_distance_fixed_scene_visible_queries'] is not None])
            entry['fixed_ground_query_max_m'] = span([r['legacy_grid']['final_ground_distance_fixed_scene_visible_queries']['p05_p50_p95_max_m'][3] for r in rows if r['legacy_grid']['final_ground_distance_fixed_scene_visible_queries'] is not None])
            bounds = [r['final']['near']['wall_at_or_above_root_body_xyz_bounds_m'] for r in rows]
            entry['root_height_wall_x_min_m'] = span([b[0][0] for b in bounds if b is not None])
            entry['root_height_wall_present_cases'] = sum(b is not None for b in bounds)
            summaries.append(entry)
            differences = []
            for r in rows:
                current = r['legacy_grid']['final_ground_distance_fixed_scene_visible_queries']
                old = baseline[r['case']]['legacy_grid']['final_ground_distance_fixed_scene_visible_queries']
                assert r['legacy_grid']['fixed_scene_visible_queries'] == baseline[r['case']]['legacy_grid']['fixed_scene_visible_queries']
                if current is not None and old is not None:
                    differences.append(np.array(current['p05_p50_p95_max_m'])[2:] - np.array(old['p05_p50_p95_max_m'])[2:])
            differences = np.array(differences).reshape(-1, 2)
            paired.append({'scene': scene, 'recipe': name, 'same_fixed_queries_cases': len(differences),
                           'p95_difference_m': span(differences[:, 0].tolist()),
                           'max_difference_m': span(differences[:, 1].tolist()),
                           'p95_better': int((differences[:, 0] < -1e-6).sum()),
                           'p95_worse': int((differences[:, 0] > 1e-6).sum())})
    nominal = 'height_0.48_width_1.00'
    nominal_rows = [r for r in report['rows'] if r['case'] == nominal]
    result = {'scope': 'all 35 poses; no score weights, training sufficiency threshold or configuration adoption',
              'ground_query_definition': 'same old diagnostic grid, ground scene-visible and raw hemisphere-visible; independent of candidate cone, not a proposed auxiliary grid',
              'aggregates': summaries, 'paired_fixed_query_gaps': paired, 'nominal_rows': nominal_rows}
    (args.out / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    k = data['case_ids'].tolist().index(nominal)
    fig, axes = plt.subplots(2, len(report['recipes']), figsize=(3.8 * len(report['recipes']), 8), constrained_layout=True)
    for col, (name, *_) in enumerate(report['recipes']):
        for row, scene in enumerate(['floor', 'channel_1.6']):
            prefix = scene + '/' + name + '/'
            pts = data[prefix + 'final_near_body'][k]
            label = data[scene + '/raw_labels'][k][data[prefix + 'final_near_raw_index'][k]]
            current = next(r for r in nominal_rows if r['scene'] == scene and r['recipe'] == name)
            ax = axes[row, col]
            dims = [0, 1] if row == 0 else [0, 2]
            for surface, color in [(0, '#2563eb'), (1, '#d97706'), (2, '#d97706'), (4, '#9ca3af')]:
                p = pts[label == surface]
                ax.scatter(p[:, dims[0]], p[:, dims[1]], s=13, c=color)
            ax.set_xlim(0, 6.1)
            if row == 0:
                ax.set_ylim(-1.3, 1.3)
                ax.set_ylabel('body y (m)')
                extra = ''
            else:
                ax.set_ylim(-.6, 1.85)
                ax.set_ylabel('body z (m)')
                ax.axhline(0, color='black', lw=1, label='root height')
                extra = f"\nwall points at/above root={current['final']['near']['wall_at_or_above_root_body_z']}"
            ax.set_xlabel('body x (m)')
            ax.set_aspect('equal')
            ax.grid(alpha=.2)
            n = current['final']['near']['surfaces']
            ax.set_title(f"delta={current['delta_deg']}deg, theta={current['theta_deg']}deg\nground={n['ground']}, wall={n['left_wall']+n['right_wall']}, nohit={n['normal_nohit']}" + extra, fontsize=10)
    fig.suptitle('Near cone comparison: exact same scans, pool and final 256-point FPS\nTop: floor plan view. Bottom: 1.6m channel side view, root height z=0. Blue ground, orange walls, grey no-hit.\nOnly points within axes shown; counts include all points, including distant ground/no-hit.', fontsize=12)
    fig.savefig(args.out / 'cone_tilt_comparison.png', dpi=160)
    plt.close(fig)
    (args.out / Path(__file__).name).write_text(Path(__file__).read_text())


if __name__ == '__main__':
    main()
