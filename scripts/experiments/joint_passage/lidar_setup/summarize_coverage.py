#!/usr/bin/env python
"""Summarize every saved pose; no scoring weights or automatic adoption."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def span(values):
    return {'min': float(min(values)), 'median': float(np.median(values)), 'max': float(max(values))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    report = json.loads((args.run / 'report.json').read_text())
    data = np.load(args.run / 'raw.npz')
    aggregates = []
    comparisons = []
    for scene, *_ in report['scenes']:
        selected = [r for r in report['rows'] if r['scene'] == scene]
        for name, *_ in report['recipes']:
            rows = [r for r in selected if r['recipe'] == name]
            assert len(rows) == 35
            record = {'scene': scene, 'recipe': name, 'states': 35}
            for branch in ['near', 'far']:
                for surface in ['ground', 'left_wall', 'right_wall', 'ceiling', 'normal_nohit']:
                    record[branch + '_' + surface] = span([r['final'][branch]['surfaces'][surface] for r in rows])
                record[branch + '_occluded'] = span([r['final'][branch]['occluded'] for r in rows])
            for key in ['wall_above_sensor_body_z', 'wall_in_startup_reference_x']:
                record['near_' + key] = span([r['final']['near'][key] for r in rows])
            gap = [r['legacy_grid']['final_ground_distance_visible_queries'] for r in rows]
            record['visible_query_gap_p95_m'] = span([g['p05_p50_p95_max_m'][2] for g in gap if g is not None])
            record['visible_query_gap_max_m'] = span([g['p05_p50_p95_max_m'][3] for g in gap if g is not None])
            aggregates.append(record)
        baseline = {r['case']: r for r in selected if r['recipe'] == 'C0'}
        for name in ['C1', 'C2', 'C3']:
            rows = [r for r in selected if r['recipe'] == name]
            # C1/C2 change the query support; these are not like-for-like gap
            # scores. Explicitly report changed-query cases instead of hiding it.
            same = rows if name == 'C3' else []
            differences = []
            for r in same:
                a = r['legacy_grid']['final_ground_distance_visible_queries']
                b = baseline[r['case']]['legacy_grid']['final_ground_distance_visible_queries']
                if a is not None and b is not None:
                    differences.append(np.array(a['p05_p50_p95_max_m'])[2:] - np.array(b['p05_p50_p95_max_m'])[2:])
            differences = np.array(differences).reshape(-1, 2)
            comparisons.append({'scene': scene, 'recipe': name, 'same_angular_query_support_cases': len(same),
                                'compared_cases': len(differences),
                                'p95_better_cases': int((differences[:, 0] < -1e-6).sum()),
                                'max_better_cases': int((differences[:, 1] < -1e-6).sum()),
                                'p95_difference_m': span(differences[:, 0]) if len(differences) else None,
                                'max_difference_m': span(differences[:, 1]) if len(differences) else None})
    nominal = 'height_0.48_width_1.00'
    k = data['case_ids'].tolist().index(nominal)
    nominal_rows = [r for r in report['rows'] if r['case'] == nominal]
    result = {'scope': 'all 35 poses, min/median/max; diagnostic grid only, no training acceptance',
              'aggregates': aggregates, 'paired_gap_comparisons': comparisons, 'nominal_rows': nominal_rows}
    (args.out / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
    colors = {0: '#2563eb', 1: '#d97706', 2: '#d97706', 3: '#16a34a', 4: '#9ca3af'}
    for col, name in enumerate(['C0', 'C1', 'C2', 'C3']):
        for row, scene in enumerate(['floor', 'channel_1.6']):
            prefix = scene + '/' + name + '/'
            pts = data[prefix + 'final_near_body'][k]
            labels = data[scene + '/raw_labels'][k][data[prefix + 'final_near_raw_index'][k]]
            ax = axes[row, col]
            for surface in range(5):
                mask = labels == surface
                ax.scatter(pts[mask, 0], pts[mask, 1], s=9, c=colors[surface], alpha=.8)
            ax.set(xlim=(-.1, 5.5), ylim=(-1.15, 1.15), xlabel='body x (m)', ylabel='body y (m)')
            ax.set_aspect('equal')
            if row == 1:
                ax.axhline(.8, color='#d97706', alpha=.3)
                ax.axhline(-.8, color='#d97706', alpha=.3)
            current = next(r for r in nominal_rows if r['scene'] == scene and r['recipe'] == name)
            gap = current['legacy_grid']['final_ground_distance_visible_queries']['p05_p50_p95_max_m'][2]
            ax.set_title(f'{name} {scene}\nground={int((labels==0).sum())}, wall={int(np.isin(labels,[1,2]).sum())}, nohit={int((labels==4).sum())}\nvisible grid p95 gap={gap:.3f}m', fontsize=10)
            ax.grid(alpha=.2)
    fig.suptitle('Nominal pose: final 256 near points; blue ground, orange walls, grey no-hit\nOnly points inside axes shown; no-hit endpoints remain valid inputs. Grid gaps are not height errors.', fontsize=12)
    fig.savefig(args.out / 'near_comparison.png', dpi=160)
    plt.close(fig)
    (args.out / Path(__file__).name).write_text(Path(__file__).read_text())


if __name__ == '__main__':
    main()
