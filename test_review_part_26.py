"""CPU review checks (split group 26): the confirmation p-value replay gate.

The frozen round stores a p-value and a Holm-adjusted p-value for every
pre-registered comparison.  ``audit_confirmation_verdicts`` replays the
decision *from those stored numbers*; this group covers the independent gate
that recomputes the numbers themselves from the recorded per-seed paired
deltas, plus the by-length curve that must agree with the primary vector.
"""
import contextlib
import copy
import io
import json
import os
import sys as _sys

_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

if not _HAVE_TORCH:
    import pytest as _pytest
    _pytest.skip('CPU torch dependencies are unavailable', allow_module_level=True)


_DELTAS = {
    'main': [-0.30, -0.28, -0.31, -0.29, -0.27, -0.30, -0.32, -0.28,
             -0.29, -0.31, -0.30, 0.01],
    'control': [0.10, 0.11, 0.09, 0.10, 0.12, 0.08, 0.11, 0.10,
                0.09, 0.10, 0.11, 0.10],
    'dense': [-0.02, -0.01, -0.03, -0.02, 0.00, -0.02, -0.01, -0.03,
              -0.02, -0.01, -0.02, -0.02],
}
_FAMILY_NAMES = sorted(list(_DELTAS) + ['omitted'])


def _make_artifact(n=12):
    """A minimal but internally consistent final-test stats artifact."""
    comps = {}
    for name, base in _DELTAS.items():
        deltas = list(base[:n])
        rep = L.exact_signflip_p(deltas)
        rec = {
            'n': len(deltas), 'n_effective': len(deltas),
            'seeds': list(range(len(deltas))),
            'per_seed': deltas,
            'mean': float(np.mean(deltas)),
            'n_nonzero': int(rep['n_nonzero']),
            'p_floor': rep['p_floor'],
            'p_exact_signflip': float(rep['p']),
            'by_length_curve': {
                '512': {
                    'n': len(deltas), 'seeds': list(range(len(deltas))),
                    'mean': float(np.mean(deltas)), 'per_seed': list(deltas),
                    'n_nonzero': int(rep['n_nonzero']),
                    'p_floor': rep['p_floor'],
                    'p_exact_signflip': float(rep['p']),
                    'role': 'primary',
                },
                '1024': {
                    'n': len(deltas), 'seeds': list(range(len(deltas))),
                    'mean': float(np.mean(deltas)) * 1.5,
                    'per_seed': [x * 1.5 for x in deltas],
                    'n_nonzero': int(rep['n_nonzero']),
                    'p_floor': rep['p_floor'],
                    'p_exact_signflip': float(rep['p']),
                    'role': 'descriptive',
                },
            },
        }
        comps[name] = rec
    comps['omitted'] = {'omitted': 'no complete seed pairs', 'n': 0}
    family = [(name, (float(comps[name]['p_exact_signflip'])
                      if 'omitted' not in comps[name] else 1.0))
              for name in comps]
    return {
        'alpha': 0.05,
        'comparisons': comps,
        'holm': L.holm_adjust(family, alpha=0.05),
        'manifest': {
            'seeds': list(range(n)),
            'primary_comparison_names': list(_FAMILY_NAMES),
            'family_size': len(_FAMILY_NAMES),
        },
        'min_attainable_p_holm': L.holm_min_attainable_p(n, len(_FAMILY_NAMES), 0.05),
    }


def _run(ft, path):
    """Write the artifact and a matching tracked-manifest stand-in."""
    import hashlib
    td = os.path.dirname(os.path.abspath(path))
    manifest = {
        'seeds': list(ft['manifest']['seeds']),
        'primary_comparisons': [{'name': n} for n in _FAMILY_NAMES],
    }
    with open(os.path.join(td, 'confirmation_frozen.json'), 'w',
              encoding='utf-8') as f:
        json.dump(manifest, f)
    with open(os.path.join(td, 'confirmation_frozen.json'), 'rb') as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    ft['manifest']['manifest_sha256'] = sha
    old_repo = A.REPO
    A.REPO = td
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'finaltest': ft}, f)
        with contextlib.redirect_stdout(io.StringIO()):
            return A.audit_confirmation_pvalues(path)
    finally:
        A.REPO = old_repo


def test_pvalue_replay_accepts_consistent_artifact():
    ft = _make_artifact()
    with tempfile.TemporaryDirectory() as td:
        got = _run(ft, os.path.join(td, 'test_stats.json'))
    assert got['status'] == 'ok', got
    assert got['n_comparisons'] == 4 and got['n_replayed'] == 3, got
    assert got['n_omitted'] == 1 and got['n_not_replayable'] == 0, got
    assert got['n_curve_replayed'] == 6, got
    assert got['holm']['m'] == 4, got['holm']
    for name in ('main', 'control', 'dense'):
        assert got['comparisons'][name]['status'] == 'replayed', got['comparisons']
        assert got['comparisons'][name]['n'] == 12
    # The omitted comparison holds a p=1 slot in the family.
    assert got['comparisons']['omitted']['family_p'] == 1.0


def test_pvalue_replay_detects_p_and_holm_tampering():
    ft = _make_artifact()
    with tempfile.TemporaryDirectory() as td:
        fake = ft['comparisons']['main']['p_exact_signflip'] + 0.01
        got = _run(copy.deepcopy(ft), os.path.join(td, 'p.json'))
        assert got['status'] == 'ok'
        bad = copy.deepcopy(ft)
        bad['comparisons']['main']['p_exact_signflip'] = fake
        got = _run(bad, os.path.join(td, 'p.json'))
        assert got['status'] == 'fail', got
        assert any('does not match the recomputed' in p for p in got['problems']), got
        # The Holm check must also fire when only the adjusted p was edited.
        bad = copy.deepcopy(ft)
        bad['holm']['main']['p_holm'] = 0.9
        got = _run(bad, os.path.join(td, 'p.json'))
        assert got['status'] == 'fail', got
        assert any("holm['main']" in p for p in got['problems']), got
        # A silently edited denominator is a fail, not a warning.
        bad = copy.deepcopy(ft)
        bad['comparisons']['main']['n_effective'] = 11
        got = _run(bad, os.path.join(td, 'p.json'))
        assert got['status'] == 'fail', got
        assert any('does not equal the number of paired deltas' in p
                   for p in got['problems']), got


def test_pvalue_replay_detects_a_deleted_or_extra_family_slot():
    ft = _make_artifact()
    with tempfile.TemporaryDirectory() as td:
        bad = copy.deepcopy(ft)
        bad['comparisons'].pop('control')
        bad['holm'].pop('control')
        got = _run(bad, os.path.join(td, 'p.json'))
        assert got['status'] == 'fail', got
        assert any('declared frozen family' in p for p in got['problems']), got

        bad = copy.deepcopy(ft)
        bad['comparisons']['extra'] = {'omitted': 'undeclared', 'n': 0}
        bad['holm']['extra'] = {'p_holm': 1.0}
        got = _run(bad, os.path.join(td, 'p.json'))
        assert got['status'] == 'fail', got
        assert any('declared frozen family' in p for p in got['problems']), got

        # A legacy artifact without the declared-name list is only a warning;
        # the deletion risk is stated instead of being treated as a pass.
        bad = copy.deepcopy(ft)
        bad['manifest'].pop('primary_comparison_names')
        got = _run(bad, os.path.join(td, 'p.json'))
        assert got['status'] == 'warnings', got
        assert any('deleted comparison' in w for w in got['warnings']), got


def test_pvalue_replay_requires_deltas_and_replays_the_curve():
    ft = _make_artifact()
    with tempfile.TemporaryDirectory() as td:
        bad = copy.deepcopy(ft)
        for rec in bad['comparisons'].values():
            rec.pop('per_seed', None)
            for cell in (rec.get('by_length_curve') or {}).values():
                cell.pop('per_seed', None)
        got = _run(bad, os.path.join(td, 'p.json'))
        assert got['status'] == 'not_verifiable_locally', got
        assert got['n_replayed'] == 0 and got['n_not_replayable'] == 3, got

        bad = copy.deepcopy(ft)
        bad['comparisons']['main'].pop('per_seed')
        got = _run(bad, os.path.join(td, 'p.json'))
        assert got['status'] == 'warnings', got
        assert got['n_not_replayable'] == 1, got
        assert any("no per_seed" in w for w in got['warnings']), got

        # Every stored curve p is replayable too.
        bad = copy.deepcopy(ft)
        bad['comparisons']['main']['by_length_curve']['1024']['p_exact_signflip'] = 0.9
        got = _run(bad, os.path.join(td, 'p.json'))
        assert got['status'] == 'fail', got
        assert any('by_length L1024' in p for p in got['problems']), got

        # The primary endpoint's vector must equal the primary curve's.
        bad = copy.deepcopy(ft)
        bad['comparisons']['main']['by_length_curve']['512']['per_seed'] = \
            [x + 0.005 for x in bad['comparisons']['main']['per_seed']]
        got = _run(bad, os.path.join(td, 'p.json'))
        assert got['status'] == 'fail', got
        assert any('primary by-length curve' in p for p in got['problems']), got


def test_pvalue_replay_rejects_malformed_vectors():
    ft = _make_artifact()
    with tempfile.TemporaryDirectory() as td:
        cases = {}
        cases['dup seeds'] = lambda r: r.update(seeds=[0] * 12)
        cases['length mismatch'] = lambda r: r.update(
            seeds=list(range(11)))
        cases['boolean delta'] = lambda r: r['per_seed'].__setitem__(0, True)
        cases['non-finite delta'] = lambda r: r['per_seed'].__setitem__(
            0, float('nan'))
        cases['p floor'] = lambda r: r.update(p_floor=0.5)
        cases['mean'] = lambda r: r.update(mean=999.0)
        cases['too short'] = lambda r: r.update(
            per_seed=[0.1], seeds=[0], n_effective=1, n=1)
        for label, mutate in cases.items():
            bad = copy.deepcopy(ft)
            mutate(bad['comparisons']['main'])
            got = _run(bad, os.path.join(td, 'p.json'))
            assert got['status'] == 'fail', (label, got)
            assert got['problems'], label
        # The trial statistics themselves remain independently checkable.
        assert L.exact_signflip_p([0.1] * 12)['p'] == float(2 / 2 ** 12)


def test_pvalue_replay_is_wired_into_report_and_cli():
    import inspect
    src = inspect.getsource(A.run_report)
    assert "'pvalue_replay': audit_confirmation_pvalues()" in src, src
    cli = inspect.getsource(A._run_cli)
    assert "'pvalues': audit_confirmation_pvalues" in cli, cli
    assert "'pvalue_replay', 'recompute_p'" in cli, cli

    ft = _make_artifact()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'test_stats.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'finaltest': ft}, f)
        old = list(_sys.argv)
        try:
            _sys.argv = ['audit_gates.py', 'pvalues', path]
            with contextlib.redirect_stdout(io.StringIO()):
                A._run_cli()
        finally:
            _sys.argv = old
