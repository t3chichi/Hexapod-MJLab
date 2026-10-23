#!/usr/bin/env python
"""Replay URDF assembly independently and check nominal raw-triangle minima.

The 70-pair global search is supported by primitive/QP regressions; this replay
does not independently repeat every global triangle search. Open HIP solids and
dynamic joint paths remain uncertified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
import trimesh
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from compare_flat import ROOT, MODEL, LIBRARY, URDF, load_model, frame
import mesh_relations as geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    report = json.loads((args.run / 'report.json').read_text())
    for path, digest in report['sha256'].items():
        source = args.run / 'mesh_relations.py' if path.endswith('/mesh_relations.py') else ROOT / path
        assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    records = json.loads(LIBRARY.read_text())['records']
    names = ['RADAR_BRACKET', 'RF_HIP', 'LF_HIP']
    tree = ET.parse(URDF)
    model = load_model(MODEL)
    meshes = {n: trimesh.load(MODEL.parent.parent / 'meshes' / (n + '.STL'), process=True) for n in names}
    saved = {(r['case'], r['pair'][1]): r for r in report['rows']}
    assert len(saved) == 70 and report['pairs'] == 70
    output = []
    arrays = {}
    max_transform_error = 0.
    max_vertex_error = 0.
    for rec in records:
        data, rb, pb, *_ = frame(model, rec['measured_qpos'])
        triangles = {}
        vertex_errors = {}
        for name in names:
            link = tree.find(f".//link[@name='{name}']")
            for visual in ['visual', 'collision']:
                origin = link.find(visual + '/origin')
                assert np.all(np.fromstring(origin.attrib['xyz'], sep=' ') == 0)
                assert np.all(np.fromstring(origin.attrib['rpy'], sep=' ') == 0)
                mesh = link.find(visual + '/geometry/mesh')
                assert mesh.attrib['filename'].endswith(name + '.STL')
                assert np.all(np.fromstring(mesh.attrib.get('scale', '1 1 1'), sep=' ') == 1)
            joint = next(j for j in tree.findall('joint') if j.find('child').attrib['link'] == name)
            assert joint.find('parent').attrib['link'] == 'BASE'
            origin = joint.find('origin')
            rotation = Rotation.from_euler('xyz', np.fromstring(origin.attrib['rpy'], sep=' ')).as_matrix()
            position = np.fromstring(origin.attrib['xyz'], sep=' ')
            if joint.attrib['type'] == 'revolute':
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint.attrib['name'])
                q = rec['measured_qpos'][model.jnt_qposadr[jid]]
                axis = np.fromstring(joint.find('axis').attrib['xyz'], sep=' ')
                rotation = rotation @ Rotation.from_rotvec(axis * q).as_matrix()
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            compiled_r = rb.T @ data.xmat[bid].reshape(3, 3)
            compiled_t = rb.T @ (data.xpos[bid] - pb)
            error = float(np.abs(rotation - compiled_r).max())
            # Exported MJCF quaternions are rounded; previous alignment checks
            # use the same 1e-6 rotation-component tolerance. Translation is exact.
            assert error < 1e-6
            assert np.abs(position - compiled_t).max() < 1e-12
            max_transform_error = max(max_transform_error, error)
            vertex_errors[name] = float(np.linalg.norm(meshes[name].vertices @ (rotation - compiled_r).T
                                                       + position - compiled_t, axis=1).max())
            max_vertex_error = max(max_vertex_error, vertex_errors[name])
            triangles[name] = (meshes[name].vertices @ rotation.T + position)[meshes[name].faces]
        for name in names[1:]:
            row = saved[rec['id'], name]
            assert row['contact_triangle_pairs'] == 0 and not row['hip_component_representatives_inside_bracket']
            assert row['surface_distance_m'] > 0 and not row['solid_relation_certified']
            if rec['id'] != 'height_0.48_width_1.00':
                continue
            a, b = triangles[names[0]], triangles[name]
            pairs = geometry.candidates(a, b, row['surface_distance_m'] + 1e-7)
            best = None
            for offset in range(0, len(pairs), 8192):
                batch = pairs[offset:offset + 8192]
                distance = geometry.distances(a[batch[:, 0]], b[batch[:, 1]])
                i = int(np.argmin(distance))
                if best is None or distance[i] < best[0]:
                    best = (distance[i], batch[i])
            value, pair = best
            ta, tb = a[pair[0]], b[pair[1]]
            def points(x):
                return ta[0] + x[0] * (ta[1] - ta[0]) + x[1] * (ta[2] - ta[0]), tb[0] + x[2] * (tb[1] - tb[0]) + x[3] * (tb[2] - tb[0])
            def objective(x):
                pa, pb = points(x)
                return np.dot(pa - pb, pa - pb)
            solution = minimize(objective, np.ones(4) / 3, method='SLSQP', bounds=[(0, 1)] * 4,
                                constraints=[{'type': 'ineq', 'fun': lambda x: 1 - x[0] - x[1]},
                                             {'type': 'ineq', 'fun': lambda x: 1 - x[2] - x[3]}],
                                options={'ftol': 1e-15, 'maxiter': 300})
            assert solution.success, solution.message
            qp_distance = float(np.sqrt(solution.fun))
            assembly_bound = vertex_errors['RADAR_BRACKET'] + vertex_errors[name]
            assert abs(value - row['surface_distance_m']) < assembly_bound + 1e-8
            assert abs(qp_distance - value) < 2e-7
            arrays[name + '/triangle_bracket'] = ta
            arrays[name + '/triangle_hip'] = tb
            arrays[name + '/closest_points'] = np.array(points(solution.x))
            output.append({'case': rec['id'], 'pair': ['RADAR_BRACKET', name], 'raw_face_indices': pair.tolist(),
                           'saved_minimum_m': row['surface_distance_m'], 'independent_qp_pair_distance_m': qp_distance,
                           'URDF_to_MJCF_vertex_rounding_distance_bound_m': assembly_bound,
                           'witness_points_BASE_m': np.array(points(solution.x)).tolist()})
    np.savez_compressed(args.out / 'nearest_witnesses.npz', **arrays)
    result = {'status': 'PASS', 'scope': __doc__, 'urdf_pose_checks': 105, 'max_rotation_component_error': max_transform_error,
              'max_URDF_MJCF_vertex_distance_m': max_vertex_error, 'translation_tolerance_m': 1e-12,
              'rotation_component_tolerance': 1e-6, 'triangle_contact_epsilon_m': geometry.EPS,
              'nominal_minimum_witnesses': output, 'open_HIP_solid_and_dynamic_safety_certified': False}
    (args.out / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    (args.out / Path(__file__).name).write_text(Path(__file__).read_text())
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
