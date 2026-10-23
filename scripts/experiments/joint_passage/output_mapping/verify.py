"""Read-only acceptance from raw arrays, with independent NumPy formulae.

No optimization, simulation, task imports or writes to the input run. Stored
success flags are compared with recomputed decisions, never used as truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
P0 = np.array([.59, .69, .59, .89, .89, .46], dtype=np.float64)
RS = [.1, .25, .5, 1.]
LRS = [.01, .05]
FAMILIES = ['softplus', 'elu', 'rational']
GROUPS = [(f'{fam}_r{r:.2f}', fam, r) for fam in FAMILIES for r in RS]
REC_GROUPS = GROUPS + [(f'linear_r{r:.2f}', 'linear', r) for r in RS]


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite_json(value):
    if isinstance(value, dict): return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [finite_json(v) for v in value]
    if isinstance(value, np.ndarray): return finite_json(value.tolist())
    if isinstance(value, (np.integer, np.bool_)): return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def formula(u, family, r):
    """Independent p and dp/du, preserving the supplied floating dtype."""
    dtype = u.dtype
    p0 = P0.astype(dtype)
    scale = np.asarray(r, dtype=dtype) * p0
    ratio = p0 / scale
    if family == 'linear': return p0 + scale * u, np.broadcast_to(scale, u.shape)
    if family == 'sigmoid':
        sig = np.exp(-np.logaddexp(np.asarray(0., dtype=dtype), -u))
        return 2 * p0 * sig, 2 * p0 * sig * (1 - sig)
    if family == 'softplus':
        b = ratio + np.log(-np.expm1(-ratio))
    else:
        b = np.where(ratio < 1, np.log(ratio) if family == 'elu' else 1 - 1 / ratio, ratio - 1)
    x = b + u
    if family == 'softplus':
        out = np.logaddexp(np.asarray(0., dtype=dtype), x)
        deriv = np.exp(-np.logaddexp(np.asarray(0., dtype=dtype), -x))
    elif family == 'elu':
        negative = np.exp(np.minimum(x, 0))
        out = np.where(x < 0, negative, 1 + x)
        deriv = np.where(x < 0, negative, 1)
    else:
        denom = 1 - np.minimum(x, 0)
        out = np.where(x < 0, 1 / denom, 1 + x)
        deriv = np.where(x < 0, 1 / denom**2, 1)
    return scale * out, scale * deriv


def run_verification(run_dir: Path, report_out: Path | None = None) -> dict:
    run_dir = run_dir.resolve()
    results = []
    def check(name, passed, detail=None):
        results.append({'id': name, 'status': 'PASS' if bool(passed) else 'FAIL', 'detail': finite_json(detail or {})})
    config, metrics, samples = [load(run_dir / f'{n}.json') for n in ('config', 'metrics', 'samples')]
    checks = load(run_dir / 'checks.json')
    protocol = (config['p0_m'] == P0.tolist() and config['families'] == FAMILIES
                and config['rs'] == RS and config['learning_rates'] == LRS
                and config['start_cs'] == [.1, .5, 1., 2., 5.] and config['main_cs'] == [.5, 1., 2.]
                and config['n_steps'] == 1000 and config['n_states_saved'] == 1001
                and config['adam'] == {'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': 0.}
                and config['seed'] == 0 and config['cpu_threads'] == 1
                and config['gates']['gain_min'] == .1 and config['gates']['fd_rel_tol'] == .05
                and config['gates']['recovery_last100_rel'] == 1e-3)
    check('fixed_protocol', protocol)
    complete = metrics.get('experiment_complete') and not metrics.get('self_check_only')
    check('experiment_complete', complete)
    if not complete:
        return {'schema_version': 1, 'all_pass': False, 'experiment_complete': False, 'results': results}
    failures = [c['id'] for c in checks['checks'] if c['status'] != 'PASS']
    check('correctness_checks', not failures and checks['n_fail'] == len(failures), {'failed': failures})

    provenance = load(run_dir / 'provenance.json')
    bad_identity = []
    for rel, expected in provenance['identity_before_sha256'].items():
        path = ROOT / rel
        if (not path.is_file() or sha(path) != expected
                or provenance['identity_after_sha256'].get(rel) != expected): bad_identity.append(rel)
    check('code_input_identity', not bad_identity and provenance['inputs_unchanged'], {'mismatched': bad_identity})
    library_path = ROOT / samples['library']['path']
    library = load(library_path)
    source = library['records']
    main = samples['library']['main']
    check('source_library', library['schema_version'] == 2 and len(source) == 35
          and sha(library_path) == samples['library']['sha256'])
    expected = np.array([r['measured_E'] for r in source]); expected[:, 4] *= -1
    expected = np.vstack([expected, P0])
    targets64 = np.array([r['p_positive'] for r in main])
    check('main_samples_identity', targets64.shape == (36, 6) and np.array_equal(expected, targets64)
          and [r['id'] for r in main[:35]] == [r['id'] for r in source]
          and [r['validation_level'] for r in main[:35]] == [r['validation_level'] for r in source]
          and sum(r['validation_level'] == 'static_hold' for r in main) == 28
          and sum(r['validation_level'] == 'scripted_slow_crawl' for r in main) == 7)
    ext = np.array([r['p_positive'] for r in samples['library']['extended']])
    ext_expected = np.array([P0 * np.where(np.arange(6) == dim, ratio, 1.)
                             for dim in range(6) for ratio in [.5, .75, 1., 1.25, 1.5]])
    stress = np.array([r['p_positive'] for r in samples['library']['stress']])
    check('extended_stress_identity', np.array_equal(ext, ext_expected)
          and np.array_equal(stress, np.array([P0 * r for r in [.001, .01, .05, .1, .25, 2., 3., 10.]])))
    denom = {'main_samples': 36, 'extended_samples': 30, 'stress_samples': 8,
             'recovery_main_cases': 108, 'recovery_pressure_cases': 72}
    check('denominators', metrics['denominators'] == denom and samples['library']['denominators'] == denom)

    curves = np.load(run_dir / 'curves.npz', allow_pickle=False)
    response = np.load(run_dir / 'responses.npz', allow_pickle=False)
    recovery = np.load(run_dir / 'recovery.npz', allow_pickle=False)
    check('scan_grid', np.array_equal(curves['scan/u_f64'], np.linspace(-20, 20, 4001)))
    check('extreme_grid', np.array_equal(curves['extremes/u_f64'], [-1000., -100., -50., 50., 100., 1000.]))
    ea2 = config['ea2_live']
    action64 = curves['scan/ea2/a']
    for dtype, tag in [(np.float64, 'f64'), (np.float32, 'f32')]:
        action = action64.astype(dtype)[:, None]
        lo = np.array(ea2['envelope_low5'], dtype=dtype)
        hi = np.array(ea2['envelope_high5'], dtype=dtype)
        scale = (hi-lo)*ea2['soft_dof_pos_limit']/(2*ea2['action_max'])
        raw = (lo+hi)/2 + action*scale
        expected_p = np.clip(raw, lo, hi)
        expected_d = ((raw>=lo)&(raw<=hi))*scale
        check('ea2_gradient/' + tag, np.allclose(curves[f'scan/ea2/params_{tag}'],expected_p,atol=1e-7,rtol=1e-6)
              and np.allclose(curves[f'scan/ea2/deriv_autograd_{tag}'],expected_d,atol=1e-8,rtol=1e-6))
        for du in [.001, .01, .1]:
            plus = np.clip((lo+hi)/2+(action+dtype(du))*scale,lo,hi)-expected_p
            minus = np.clip((lo+hi)/2+(action-dtype(du))*scale,lo,hi)-expected_p
            check(f'ea2_perturbation/{tag}/{du}',np.allclose(plus,curves[f'scan/ea2/dp_plus_du{du}_{tag}'],atol=1e-7,rtol=1e-5)
                  and np.allclose(minus,curves[f'scan/ea2/dp_minus_du{du}_{tag}'],atol=1e-7,rtol=1e-5))
        far = np.abs(action64)>=50
        check('ea2_hard_plateau/' + tag,(curves[f'scan/ea2/deriv_autograd_{tag}'][far]==0).all()
              and (curves[f'scan/ea2/dp_plus_du0.001_{tag}'][far]==0).all()
              and (curves[f'scan/ea2/dp_minus_du0.001_{tag}'][far]==0).all())
    response_pass = {}
    main_ext = np.vstack([targets64, ext])
    all_target = np.vstack([main_ext, stress])
    for gid, fam, r in GROUPS + [(f'linear_r{r:.2f}', 'linear', r) for r in RS] + [('sigmoid', 'sigmoid', 1.)]:
        scan64 = np.repeat(curves['scan/u_f64'][:, None], 6, axis=1)
        p, d = formula(scan64, fam, r)
        check('curve_formula/' + gid, np.allclose(p, curves[f'scan/{gid}/p_f64'], atol=1e-10, rtol=1e-9)
              and np.allclose(d, curves[f'scan/{gid}/deriv_analytic_f64'], atol=1e-10, rtol=1e-8)
              and np.allclose(d, curves[f'scan/{gid}/deriv_autograd_f64'], atol=1e-10, rtol=1e-8))
        if fam in FAMILIES:
            check('positive_monotone/' + gid, np.isfinite(p).all() and (p > 0).all() and (np.diff(p, axis=0) >= 0).all())
        u = response[f'inv/{gid}']
        reachable = response[f'reachable/{gid}']
        expected_reach = (all_target < 2 * P0).all(axis=1) if fam == 'sigmoid' else np.ones(74, dtype=bool)
        check('response_reachability/' + gid, np.array_equal(reachable, expected_reach))
        actual, d = formula(u[reachable], fam, r)
        check('response_roundtrip/' + gid, np.allclose(actual, all_target[reachable], atol=1e-10, rtol=1e-9))
        base_deriv = formula(np.zeros(6), fam, r)[1]
        gain = d / base_deriv
        check('response_gain_formula/' + gid, np.allclose(gain, response[f'gain/{gid}'][reachable], atol=1e-12, rtol=1e-9))
        u32 = u[reachable].astype(np.float32)
        base, derivative32 = formula(u32, fam, r)
        plus, _ = formula(u32 + np.float32(.001), fam, r)
        minus, _ = formula(u32 - np.float32(.001), fam, r)
        dp_plus = response[f'dp_plus/{gid}/du0.001'][reachable]
        dp_minus = response[f'dp_minus/{gid}/du0.001'][reachable]
        # The gate uses saved actual Torch responses, checked against independent formulae.
        check('perturbation_formula/' + gid, np.allclose(dp_plus, plus - base, atol=5e-7, rtol=.005)
              and np.allclose(dp_minus, minus - base, atol=5e-7, rtol=.005))
        if fam in FAMILIES:
            local_d = formula(u[:66], fam, r)[1]
            # Float32 rounding tolerance is tested from the actual saved deltas.
            d32 = formula(u[:66].astype(np.float32), fam, r)[1].astype(np.float64)
            slopes = np.stack([response[f'dp_plus/{gid}/du0.001'][:66] / np.float32(.001),
                               -response[f'dp_minus/{gid}/du0.001'][:66] / np.float32(.001)])
            min_gain = float((local_d / base_deriv).min())
            fd_rel = float((np.abs(slopes - d32) / np.abs(d32)).max())
            changed = ((response[f'dp_plus/{gid}/du0.001'][:66] != 0)
                       & (response[f'dp_minus/{gid}/du0.001'][:66] != 0)).all()
            response_pass[gid] = (min_gain >= .1, bool(changed and fd_rel <= .05))
            stored = metrics['candidate_configs'][gid]
            check('response_decision/' + gid, response_pass[gid] == (stored['gain_gate_pass'], stored['fd_gate_pass'])
                  and np.isclose(min_gain, stored['gain_min_main_ext'], atol=1e-12), {'min_gain': min_gain, 'fd_rel_max': fd_rel})

    case_c = recovery['case_c']; case_idx = recovery['case_target_idx']
    main_rows = recovery['main_rows']; pressure_rows = recovery['pressure_rows']
    check('case_order', np.array_equal(case_idx, np.tile(np.arange(36), 5))
          and np.array_equal(case_c, np.repeat([.1, .5, 1., 2., 5.], 36).astype(np.float32))
          and np.array_equal(main_rows, np.arange(36, 144))
          and np.array_equal(pressure_rows, np.r_[0:36, 144:180])
          and np.array_equal(recovery['targets'], targets64.astype(np.float32)))
    target = recovery['targets'][case_idx, None, :]
    usable = []
    for gid, fam, r in REC_GROUPS:
        lr_pass = []
        for lr in LRS:
            key = f'{gid}_lr{lr}'
            u = recovery[f'u/{key}']; p = recovery[f'p/{key}']
            loss = recovery[f'loss/{key}']; grad_norm = recovery[f'grad_norm/{key}']
            check('trajectory_shape/' + key, u.shape == p.shape == (180, 1001, 6)
                  and loss.shape == grad_norm.shape == (180, 1001) and u.dtype == p.dtype == np.float32)
            pred, deriv = formula(u, fam, r)
            reference_loss = .5 * (((pred - target) / P0.astype(np.float32))**2).mean(axis=-1)
            grad = (pred - target) * deriv / (np.float32(6) * P0.astype(np.float32)**2)
            reference_norm = np.linalg.norm(grad, axis=-1)
            finite = np.isfinite(u).all(axis=-1) & np.isfinite(p).all(axis=-1) & np.isfinite(loss) & np.isfinite(grad_norm)
            check('trajectory_formula/' + key, np.allclose(pred, p, atol=2e-6, rtol=2e-5, equal_nan=True)
                  and np.allclose(reference_loss, loss, atol=1e-6, rtol=2e-5, equal_nan=True)
                  and np.allclose(reference_norm, grad_norm, atol=2e-6, rtol=1e-4, equal_nan=True))
            start = case_c[:, None] * P0
            check('equal_physical_start/' + key, np.allclose(p[:, 0], start, atol=1e-6, rtol=1e-5))
            rel = np.abs(p - target) / P0
            passed = (finite.all(axis=1) & (p > 0).all(axis=(1, 2))
                      & np.isfinite(rel[:, -100:]).all(axis=(1, 2)) & (rel[:, -100:] <= 1e-3).all(axis=(1, 2)))
            main_pass = bool(passed[main_rows].all())
            stored = (metrics['candidate_configs'][gid] if fam in FAMILIES else metrics['linear_control'][gid])['recovery'][f'lr{lr}']
            check('recovery_decision/' + key, main_pass == stored['main_all_pass']
                  and passed.tolist() == stored['case_gate_pass']
                  and np.where(passed[pressure_rows])[0].size == stored['pressure_pass_count'])
            if main_pass: lr_pass.append(lr)
        if fam in FAMILIES and all(response_pass[gid]) and lr_pass:
            usable.append({'group': gid, 'passing_lr': lr_pass})
    stored_usable = {x['group']: x['passing_lr'] for x in metrics['usable_candidates']}
    check('usable_set', stored_usable == {x['group']: x['passing_lr'] for x in usable}, {'recomputed': usable})

    manifest = load(run_dir / 'manifest.json')
    files = {str(p.relative_to(run_dir)): p for p in run_dir.rglob('*') if p.is_file() and p.name != 'manifest.json'}
    # During run bootstrap, verification.json is not yet present. External checks include it.
    check('manifest_coverage', set(files) == set(manifest['files_sha256']) and 'manifest.json' not in manifest['files_sha256'])
    check('manifest_hashes', all(sha(files[k]) == v for k, v in manifest['files_sha256'].items() if k in files))
    check('artifact_budget', sum(p.stat().st_size for p in files.values()) < 512 * 1024**2)
    for npz in (curves, response, recovery): npz.close()
    report = {'schema_version': 1, 'run_dir': str(run_dir), 'experiment_complete': True,
              'all_pass': all(x['status'] == 'PASS' for x in results), 'n_checks': len(results),
              'n_fail': sum(x['status'] == 'FAIL' for x in results), 'results': results,
              'usable_candidates_recomputed': usable,
              'note': 'verification.json is created before final manifest; standalone verify also checks its final hash'}
    if report_out is not None:
        report_out = report_out.resolve()
        if report_out.exists() or report_out == run_dir or run_dir in report_out.parents:
            raise ValueError('report-out must be a new path outside the immutable run directory')
        report_out.write_text(json.dumps(finite_json(report), ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description='Read-only independent verification from raw arrays')
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--report-out', type=Path)
    args = parser.parse_args()
    try:
        report = run_verification(args.run_dir, args.report_out)
        print(json.dumps(finite_json(report), ensure_ascii=False, indent=2, allow_nan=False))
        return 0 if report['all_pass'] else 1
    except (OSError, ValueError, KeyError) as exc:
        print(json.dumps({'all_pass': False, 'error': str(exc)}, ensure_ascii=False))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
