"""Cross-run reproduction is a comparison, not a declaration.

These checks cover the dimension-level diagnosis (code / data / config /
environment / initialization / RNG / recorded curve), the per-cell aggregate
report, the tri-state handling of missing provenance, and the CLI/audit
wiring.  They use synthetic summaries only: no training, no GPU and no
test-split read.
"""
import contextlib
import copy
import hashlib
import inspect
import io
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

if not _HAVE_TORCH:
    import pytest as _pytest
    _pytest.skip('CPU torch dependencies are unavailable', allow_module_level=True)


def _objective():
    return {
        'comp_reg_applies': True,
        'comp_lambda': 0.05,
        'comp_lambda_effective': 0.05,
        'delta_lr_effective': 10.0,
        'boundary_policy': ['threshold'],
        'loss_terms': ['ce', 'comp_reg'],
    }


def _record(*, tree='a' * 64, code='CS', data='data', val='val',
            data_version='dv', corpus='corpus', cfg='cfg_csCS', env='env',
            init='init', rng='rng', ppl=10.0, tokens=1000, synth=False,
            losses=(3.0, 2.5, 2.3)):
    prov = {
        'code': {
            'code_semantics': 'v11.128',
            'code_stamp': code,
            'source_tree': {'sha256': tree, 'n_files': 3, 'bytes': 100,
                            'scope': 'synthetic'},
        },
        'environment': {'python': '3.12.15', 'torch': '2.14.1+cpu'},
        'data': {
            'dataset': 'wikitext', 'data_fp': data, 'val_fp': val,
            'data_version': data_version, 'corpus_fp': corpus,
        },
    }
    return {
        'variant': 'csa_dynamic', 'seed': 0, 'run_cfg': cfg, '_code': code,
        'exec_fp': env, 'init_core_fp': init, 'rng_fp': rng,
        'data_fp': data, 'val_fp': val, 'data_version': data_version,
        'corpus_fp': corpus, 'ppl': float(ppl), 'n_tokens': int(tokens),
        'train_knobs': L.training_knobs(amp_dtype='fp32', eager_batch=False),
        'train_objective': _objective(),
        'provenance': prov,
        'losses': list(losses) if losses is not None else None,
        'synthesized': bool(synth),
    }


def _summary(*keys):
    out = {}
    for key in keys:
        variant, seed = key.split('::seed')
        rec = _record()
        rec['variant'] = variant
        rec['seed'] = int(seed)
        out[key] = rec
    return out


def test_identical_runs_are_identical_not_merely_close():
    a, b = _record(), _record()
    d = L.reproduction_diagnosis(a, b)
    assert d['verdict'] == 'identical' and d['reproduced'] is True, d
    assert d['bit_identical'] is True and d['identity_ok'] is True, d
    for dim in ('code', 'data', 'config', 'environment', 'initialization'):
        assert d['dimensions'][dim]['status'] == 'match', (dim, d['dimensions'][dim])
    assert d['dimensions']['rng_state']['status'] == 'match', d['dimensions']['rng_state']
    assert d['metric']['abs_diff'] == 0.0 and d['metric']['status'] == 'exact', d['metric']
    # The report is JSON-safe and stable: no NaN/Inf token leaks into the body.
    blob = json.dumps(d, sort_keys=True, allow_nan=False)
    assert 'reproduction_diagnosis_v2' in blob and 'NaN' not in blob


def test_tolerance_is_declared_and_drift_is_not_reproduction():
    exact = L.reproduction_diagnosis(_record(), _record(ppl=10.0 + 1e-12),
                                     rtol=0.0, atol=0.0)
    assert exact['verdict'] == 'numerical_drift' and exact['reproduced'] is False, exact
    close = L.reproduction_diagnosis(_record(), _record(ppl=10.0 + 1e-7),
                                     rtol=1e-6, atol=0.0)
    assert close['verdict'] == 'within_tolerance' and close['reproduced'] is True, close
    assert close['bit_identical'] is False and close['metric']['rel_diff'] > 0, close
    far = L.reproduction_diagnosis(_record(), _record(ppl=10.001),
                                   rtol=1e-6, atol=0.0)
    assert far['verdict'] == 'numerical_drift' and far['reproduced'] is False, far
    # An absolute tolerance is allowed and is echoed back unchanged.
    absolute = L.reproduction_diagnosis(_record(), _record(ppl=10.0005),
                                        rtol=0.0, atol=1e-3)
    assert absolute['verdict'] == 'within_tolerance', absolute
    assert absolute['atol'] == 1e-3 and absolute['rtol'] == 0.0, absolute


def test_each_identity_dimension_gets_its_own_failure_mode():
    cases = [
        ('code', _record(tree='b' * 64), 'code_changed'),
        ('code', _record(code='OTHER'), 'code_changed'),
        ('data', _record(data='other'), 'data_changed'),
        ('data', _record(val='other'), 'data_changed'),
        ('data', _record(data_version='other'), 'data_changed'),
        ('config', _record(cfg='other_cfg'), 'config_changed'),
        ('environment', _record(env='other_env'), 'environment_changed'),
        ('initialization', _record(init='other_init'), 'initialization_changed'),
    ]
    for dim, changed, verdict in cases:
        d = L.reproduction_diagnosis(_record(), changed)
        assert d['verdict'] == verdict, (dim, verdict, d)
        assert d['reproduced'] is False, d
        assert d['dimensions'][dim]['status'] == 'mismatch', (dim, d['dimensions'][dim])
    # Same seed with a different initial RNG stream is a warning that survives
    # the metric: it is reported, but it is not the arm-pairing gate.
    rng = L.reproduction_diagnosis(_record(), _record(rng='other'))
    assert rng['verdict'] == 'identical', rng
    assert rng['dimensions']['rng_state']['status'] == 'mismatch', rng['dimensions']
    assert any('rng_fp' in p for p in rng['warnings']), rng['warnings']
    assert not rng['problems'], rng['problems']


def test_missing_provenance_is_not_a_reproduction():
    missing = []
    for key in ('exec_fp', 'init_core_fp'):
        rec = _record()
        rec.pop(key)
        missing.append(rec)
    rec = _record()
    rec['provenance']['code'].pop('source_tree')
    missing.append(rec)
    rec = _record()
    for key in ('data_fp', 'val_fp', 'eval_fp', 'data_version', 'corpus_fp'):
        rec.pop(key, None)
        rec['provenance']['data'].pop(key, None)
    missing.append(rec)
    for rec in missing:
        d = L.reproduction_diagnosis(_record(), rec)
        assert d['verdict'] == 'provenance_incomplete', d
        assert d['reproduced'] is False and d['identity_ok'] is False, d
        assert d['problems'], d
    # A valid PPL over a different token count is not the same measurement.
    denom = L.reproduction_diagnosis(_record(), _record(tokens=999))
    assert denom['verdict'] == 'metric_mismatch' and denom['reproduced'] is False, denom
    # A missing/invalid PPL is never silently read as "reproduced".
    bad = L.reproduction_diagnosis(_record(), _record(ppl=0.5))
    assert bad['verdict'] == 'metric_missing' and bad['reproduced'] is False, bad
    # Synthesized records (reconstructed aggregates) have no run to reproduce.
    synth = L.reproduction_diagnosis(_record(), _record(synth=True))
    assert synth['verdict'] == 'not_comparable' and synth['reproduced'] is False, synth


def test_malformed_inputs_and_tolerances_fail_closed():
    for bad in (None, [], 'x', 3):
        try:
            L.reproduction_diagnosis(bad, _record())
        except ValueError:
            pass
        else:
            raise AssertionError(f'non-mapping record {bad!r} was accepted')
    for bad in (-1.0, float('nan'), float('inf'), True, 'fast'):
        try:
            L.reproduction_diagnosis(_record(), _record(), rtol=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'bad rtol {bad!r} was accepted')
    for bad in (-1e-6, float('nan'), float('inf'), False):
        try:
            L.reproduction_diagnosis(_record(), _record(), atol=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'bad atol {bad!r} was accepted')


def test_compare_report_is_cellwise_and_tristate():
    a = _summary('csa_dynamic::seed0', 'csa_dynamic::seed1', 'full::seed0')
    b = copy.deepcopy(a)
    before = copy.deepcopy(a)
    rep = L.compare_reproductions(a, b)
    assert a == before, 'the comparator must be read-only on its inputs'
    assert rep['verdict'] == 'reproduced' and rep['reproduced'] is True, rep
    assert rep['bit_identical'] is True and rep['n_compared'] == 3, rep
    assert set(rep['cells']) == set(a), rep['cells']
    assert rep['counts'] == {'identical': 3}, rep['counts']

    b2 = copy.deepcopy(a)
    b2['csa_dynamic::seed0']['ppl'] = 10.5
    rep2 = L.compare_reproductions(a, b2)
    assert rep2['verdict'] == 'drift' and rep2['reproduced'] is False, rep2
    assert rep2['counts'].get('numerical_drift') == 1, rep2['counts']
    assert rep2['bit_identical'] is False, rep2

    b3 = copy.deepcopy(a)
    b3.pop('full::seed0')
    rep3 = L.compare_reproductions(a, b3)
    assert rep3['verdict'] == 'incomplete' and rep3['reproduced'] is False, rep3
    assert rep3['only_original'] == ['full::seed0'], rep3
    assert rep3['n_compared'] == 2 and rep3['n_only_original'] == 1, rep3

    b4 = copy.deepcopy(a)
    b4['full::seed0']['exec_fp'] = 'another-box'
    rep4 = L.compare_reproductions(a, b4)
    assert rep4['verdict'] == 'not_reproduced', rep4
    assert any('environment' in p.lower() for p in rep4['problems']), rep4['problems']

    # A key that exists only in the rerun is also incomplete, not an implicit
    # union of two different panels.
    b6 = copy.deepcopy(a)
    b6['extra_arm::seed0'] = _record()
    rep6 = L.compare_reproductions(a, b6)
    assert rep6['verdict'] == 'incomplete', rep6
    assert rep6['only_rerun'] == ['extra_arm::seed0'], rep6

    # No key in common: not a comparison at all, and certainly not a pass.
    rep5 = L.compare_reproductions(a, {'other_arm::seed0': _record()})
    assert rep5['verdict'] == 'not_comparable' and rep5['reproduced'] is False, rep5
    assert rep5['n_compared'] == 0, rep5

    for bad in (None, 123, object()):
        try:
            L.compare_reproductions(a, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'a non-path/non-mapping source {bad!r} was accepted')
    try:
        L.compare_reproductions(a, os.path.join(tempfile.gettempdir(), 'definitely-absent-summary.json'))
    except FileNotFoundError:
        pass
    else:
        raise AssertionError('a missing summary path was accepted')


def test_cli_and_audit_gate_are_wired():
    with tempfile.TemporaryDirectory() as td:
        a, b = os.path.join(td, 'a'), os.path.join(td, 'b')
        os.makedirs(a)
        os.makedirs(b)
        original = _summary('csa_dynamic::seed0', 'csa_dynamic::seed1')
        with open(os.path.join(a, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump(original, f)
        with open(os.path.join(b, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump(original, f)
        out_json = os.path.join(td, 'out', 'reproduction.json')
        proc = subprocess.run(
            [sys.executable, '-m', 'explib.reproduce', 'compare', a, b,
             '--json', out_json],
            cwd=REPO, capture_output=True, text=True)
        assert proc.returncode == 0, (proc.stdout[-2000:], proc.stderr[-2000:])
        assert os.path.isfile(out_json), proc.stdout[-2000:]
        with open(out_json, encoding='utf-8') as f:
            written = json.load(f)
        assert written['verdict'] == 'reproduced', written

        # Within tolerance under --require exact: the identity is the same but
        # the numbers are not bit-identical, so the exact demand must fail.
        changed = copy.deepcopy(original)
        changed['csa_dynamic::seed0']['ppl'] = 10.0 + 1e-7
        with open(os.path.join(b, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump(changed, f)
        proc = subprocess.run(
            [sys.executable, '-m', 'explib.reproduce', 'compare', a, b],
            cwd=REPO, capture_output=True, text=True)
        assert proc.returncode == 0, (proc.stdout[-2000:], proc.stderr[-2000:])
        proc = subprocess.run(
            [sys.executable, '-m', 'explib.reproduce', 'compare', a, b,
             '--require', 'exact'],
            cwd=REPO, capture_output=True, text=True)
        assert proc.returncode == 1, (proc.stdout[-2000:], proc.stderr[-2000:])

        # A genuine drift fails both the CLI and the audit gate.
        changed['csa_dynamic::seed0']['ppl'] = 10.5
        with open(os.path.join(b, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump(changed, f)
        proc = subprocess.run(
            [sys.executable, '-m', 'explib.reproduce', 'compare', a, b],
            cwd=REPO, capture_output=True, text=True)
        assert proc.returncode == 1, (proc.stdout[-2000:], proc.stderr[-2000:])
        with contextlib.redirect_stdout(io.StringIO()):
            audit = A.audit_reproduction(a, b)
        assert audit['status'] == 'fail' and audit['verdict'] == 'drift', audit
        with contextlib.redirect_stdout(io.StringIO()):
            missing = A.audit_reproduction()
        assert missing['status'] == 'not_verifiable_locally', missing
        with contextlib.redirect_stdout(io.StringIO()):
            broken = A.audit_reproduction(a, os.path.join(td, 'absent'))
        assert broken['status'] == 'fail', broken

        # The audit CLI accepts the two explicit paths; the other gates keep
        # their historical no-argument invocation.
        old_argv = sys.argv
        try:
            sys.argv = ['audit_gates.py', 'reproduction', a, b]
            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    A._run_cli()
                except SystemExit as exc:
                    assert exc.code == 1, exc.code
                else:
                    raise AssertionError('a drifted rerun exited successfully')
        finally:
            sys.argv = old_argv
    src = inspect.getsource(A.run_report)
    assert "'reproduction': audit_reproduction()" in src, src[-2000:]


def _write_curve_panel(root, cell, losses, *, sha=None):
    """Write the summary + curve sidecar shape the real panel writer emits."""
    os.makedirs(os.path.join(root, 'curves'), exist_ok=True)
    payload = {'losses': list(losses)}
    curve_path = os.path.join(root, 'curves', 'cell.json')
    with open(curve_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, separators=(',', ':'))
    cell = copy.deepcopy(cell)
    cell.pop('losses', None)
    cell['curve_ref'] = 'curves/cell.json'
    if sha is None:
        with open(curve_path, 'rb') as f:
            sha = hashlib.sha256(f.read()).hexdigest()
    cell['curve_sha256'] = sha
    cell['curve_n_points'] = len(losses)
    with open(os.path.join(root, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump({'csa_dynamic::seed0': cell}, f)
    return cell


def test_equal_endpoint_with_different_curve_is_not_a_reproduction():
    a = _record()
    b = _record(losses=(3.0, 2.5, 2.0))
    d = L.reproduction_diagnosis(a, b)
    assert d['verdict'] == 'curve_changed' and d['reproduced'] is False, d
    assert d['bit_identical'] is False and d['curve_identical'] is False, d
    assert d['dimensions']['curve']['status'] == 'mismatch', d['dimensions']['curve']
    assert d['dimensions']['curve']['first_difference_index'] == 2, d['dimensions']['curve']
    assert any('curve' in p for p in d['problems']), d['problems']

    rep = L.compare_reproductions({'csa_dynamic::seed0': a},
                                  {'csa_dynamic::seed0': b})
    assert rep['verdict'] == 'not_reproduced' and rep['reproduced'] is False, rep
    assert rep['counts'] == {'curve_changed': 1}, rep['counts']
    assert rep['n_curve_mismatch'] == 1 and rep['bit_identical'] is False, rep


def test_curve_tolerance_missing_curve_and_coarse_summary_are_distinct():
    close = _record(losses=(3.0, 2.5, 2.3 + 1e-9))
    d = L.reproduction_diagnosis(_record(), close)
    assert d['verdict'] == 'within_tolerance' and d['reproduced'] is True, d
    assert d['bit_identical'] is False, d
    assert d['dimensions']['curve']['status'] == 'within_tolerance', d['dimensions']['curve']
    assert any('tolerance' in w for w in d['warnings']), d['warnings']

    exact = L.reproduction_diagnosis(_record(), close, rtol=0.0, atol=0.0)
    assert exact['verdict'] == 'curve_changed' and exact['reproduced'] is False, exact

    # A shape change is a definite curve failure, not a tolerance question.
    shortened = L.reproduction_diagnosis(_record(), _record(losses=(3.0, 2.5)))
    assert shortened['verdict'] == 'curve_changed', shortened
    assert shortened['dimensions']['curve']['status'] == 'mismatch', shortened

    # A boolean/string is not a loss value; it is unverified, never coerced
    # into 1.0 or 3.0 and never silently matched.
    for bad_value in (True, '3.0'):
        bad = _record(losses=(3.0, bad_value, 2.3))
        d = L.reproduction_diagnosis(_record(), bad)
        assert d['dimensions']['curve']['status'] == 'unverified', (bad_value, d)
        assert d['verdict'] == 'endpoint_only' and d['bit_identical'] is False, d

    missing = _record()
    missing.pop('losses')
    d = L.reproduction_diagnosis(_record(), missing)
    # The endpoint reproduced, but without a recorded trajectory this is not
    # a bit-identical run and the report says so explicitly.
    assert d['verdict'] == 'endpoint_only' and d['reproduced'] is True, d
    assert d['bit_identical'] is False and d['curve_compared'] is False, d
    assert any('no comparable' in w for w in d['warnings']), d['warnings']

    coarse_a = _record()
    coarse_b = _record()
    for rec, best in ((coarse_a, 2.3), (coarse_b, 2.3)):
        rec.pop('losses')
        rec.update({'curve_n_points': 3, 'curve_first': 3.0,
                    'curve_last': 2.3, 'curve_best': best,
                    'curve_n_nonfinite': 0})
    d = L.reproduction_diagnosis(coarse_a, coarse_b)
    assert d['verdict'] == 'endpoint_only' and d['reproduced'] is True, d
    assert d['dimensions']['curve']['coarse'] is True, d['dimensions']['curve']
    assert any('scalar curve summary' in w for w in d['warnings']), d['warnings']

    coarse_b['curve_best'] = 2.1
    d = L.reproduction_diagnosis(coarse_a, coarse_b)
    assert d['verdict'] == 'curve_changed' and d['reproduced'] is False, d
    assert d['dimensions']['curve']['method'] == 'summary_stats', d['dimensions']['curve']


def test_curve_sidecar_digest_binds_more_than_the_values():
    with tempfile.TemporaryDirectory() as td:
        a_dir = os.path.join(td, 'a')
        b_dir = os.path.join(td, 'b')
        c_dir = os.path.join(td, 'c')
        os.makedirs(a_dir)
        os.makedirs(b_dir)
        os.makedirs(c_dir)
        base = _record()
        _write_curve_panel(a_dir, base, (3.0, 2.5, 2.3))
        _write_curve_panel(b_dir, base, (3.0, 2.5, 2.3))
        rep = L.compare_reproductions(a_dir, b_dir)
        assert rep['verdict'] == 'reproduced' and rep['bit_identical'] is True, rep
        assert rep['n_curve_compared'] == 1 and rep['n_curve_identical'] == 1, rep

        # Same numeric values, different sidecar digest: the values reproduce,
        # but the serialized run artifact (delta/ppl/opt diagnostics) does not.
        _write_curve_panel(c_dir, base, (3.0, 2.5, 2.3), sha='0' * 64)
        rep = L.compare_reproductions(a_dir, c_dir)
        assert rep['verdict'] == 'reproduced' and rep['reproduced'] is True, rep
        assert rep['counts'] == {'within_tolerance': 1}, rep['counts']
        assert rep['bit_identical'] is False and rep['n_curve_identical'] == 0, rep
        assert rep['n_curve_compared'] == 1, rep

        # A differing digest with no loadable payload is a definite mismatch,
        # not an unverifiable one: the recorded bytes are the evidence.
        with open(os.path.join(c_dir, 'summary.json'), encoding='utf-8') as f:
            c_cell = json.load(f)
        c_cell['csa_dynamic::seed0'].pop('curve_ref')
        with open(os.path.join(c_dir, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump(c_cell, f)
        rep = L.compare_reproductions(a_dir, c_dir)
        assert rep['verdict'] == 'not_reproduced', rep
        assert rep['counts'] == {'curve_changed': 1}, rep['counts']


def test_cli_rejects_same_endpoint_different_curve():
    with tempfile.TemporaryDirectory() as td:
        a, b = os.path.join(td, 'a'), os.path.join(td, 'b')
        os.makedirs(a)
        os.makedirs(b)
        original = _summary('csa_dynamic::seed0')
        with open(os.path.join(a, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump(original, f)
        changed = copy.deepcopy(original)
        changed['csa_dynamic::seed0']['losses'] = [3.0, 2.5, 1.0]
        with open(os.path.join(b, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump(changed, f)
        for extra in ([], ['--require', 'exact']):
            proc = subprocess.run(
                [sys.executable, '-m', 'explib.reproduce', 'compare', a, b,
                 *extra],
                cwd=REPO, capture_output=True, text=True)
            assert proc.returncode == 1, (proc.stdout[-2000:], proc.stderr[-2000:])
            assert 'curve_changed' in proc.stdout, proc.stdout[-2000:]
        with contextlib.redirect_stdout(io.StringIO()):
            audit = A.audit_reproduction(a, b)
        assert audit['status'] == 'fail' and audit['verdict'] == 'not_reproduced', audit
        assert audit['counts'] == {'curve_changed': 1}, audit['counts']
        assert audit['n_curve_mismatch'] == 1, audit


def test_curve_ref_cannot_escape_the_summary_directory():
    with tempfile.TemporaryDirectory() as td:
        panel = os.path.join(td, 'panel')
        os.makedirs(panel)
        with open(os.path.join(td, 'outside.json'), 'w', encoding='utf-8') as f:
            json.dump({'losses': [3.0, 2.5, 2.3]}, f)
        cell = _record()
        for key in ('losses', 'curve_sha256', 'ppl_history_sha256',
                    'curve_n_points', 'curve_first', 'curve_last',
                    'curve_best', 'curve_n_nonfinite',
                    'curve_first_nonfinite_index'):
            cell.pop(key, None)
        cell['curve_ref'] = '../outside.json'
        with open(os.path.join(panel, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump({'csa_dynamic::seed0': cell}, f)
        rep = L.compare_reproductions(panel, panel)
        diag = rep['cells']['csa_dynamic::seed0']
        assert diag['dimensions']['curve']['status'] == 'unverified', diag
        assert 'resolves outside' in (diag['dimensions']['curve']['detail'] or ''), diag
        assert rep['bit_identical'] is False and rep['n_curve_unverified'] == 1, rep
