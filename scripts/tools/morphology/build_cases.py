#!/usr/bin/env python3
"""Declared posture families; record failed branch searches as well as witnesses."""
import argparse
import itertools
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from morphology.meter import MorphologyMeter, DEFAULT_XML, JOINT_NAMES, LEGS
from morphology.provenance import identity, write_new


def specifications():
    rows = []
    for h in (.32, .34, .36, .38, .40, .44, .48, .52, .54, .56):
        for sy in (1., 1.1):
            rows.append(dict(id=f'height_{h:.2f}_width_{sy:.2f}', category='height', root_height=h, lateral_scale=sy, front_shift=0., rear_shift=0.))
    for sy in (.65, .75, .85, .95):
        for h in (.40, .48):
            rows.append(dict(id=f'contract_{sy:.2f}_h{h:.2f}', category='lateral_contract', root_height=h, lateral_scale=sy, front_shift=0., rear_shift=0.))
    for front, rear in ((.15, .05), (.25, .05), (-.05, -.15), (-.05, -.25)):
        for h in (.40, .48):
            rows.append(dict(id=f'asym_{front:+.2f}_{rear:+.2f}_h{h:.2f}', category='front_rear_asymmetry', root_height=h, lateral_scale=1., front_shift=front, rear_shift=rear))
    return rows


def build(meter, specification, branch_budget=256):
    h = specification['root_height']
    meter.set_state((0, 0, h), (1, 0, 0, 0), np.zeros(18))
    targets = {}
    for leg, body in meter.foot_body_ids.items():
        xy = meter.to_base_frame(meter.data.xpos[body])[:2].copy()
        xy[1] *= specification['lateral_scale']
        xy[0] += specification['front_shift'] if leg[1] == 'F' else specification['rear_shift'] if leg[1] == 'B' else 0.
        targets[leg] = [*xy, -h]
    branches = [meter.leg_ik(leg, targets[leg], sole_on_ground=True) for leg in LEGS]
    if any(not x for x in branches):
        return {**specification, 'error': 'sole_IK_budget_not_found', 'branch_counts': list(map(len, branches)), 'source': 'declared posture family'}
    count = 0
    for selected in itertools.islice(itertools.product(*branches), branch_budget):
        count += 1
        q = {n: v for part in selected for n, v in part.items()}
        meter.set_state((0, 0, h), (1, 0, 0, 0), q)
        if meter.self_collision() or meter.clearance(True) < .005:
            continue
        return {**specification, 'joint_angles': q, 'source': 'declared posture family; first collision-clear branch, dynamic checks separate', 'branches_tried': count, 'initial_hex': meter.measure().hex_params}
    return {**specification, 'error': 'collision_clear_branch_budget_not_found', 'branches_tried': count, 'source': 'declared posture family'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--xml', type=Path, default=DEFAULT_XML)
    ap.add_argument('--branch-budget', type=int, default=256)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    if a.branch_budget < 1:
        ap.error('--branch-budget must be positive')
    m = MorphologyMeter(a.xml)
    cases = [build(m, s, a.branch_budget) for s in specifications()]
    write_new(a.out, dict(cases=cases, provenance=identity(a.xml, dict(branch_budget=a.branch_budget, specifications=specifications()))))
    print('cases', len(cases), 'constructed', sum('error' not in c for c in cases))


if __name__ == '__main__':
    main()
