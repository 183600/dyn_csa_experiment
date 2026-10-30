"""CPU checks for the dependence-aware confirmation family correction.

The frozen confirmation family answers its six pairwise questions with Holm
step-down.  Holm is valid but conservative when the per-seed contrasts are
correlated.  ``confirmation_family_correction`` adds a descriptive exact
Westfall-Young max-T on the sign-invariant score statistic, a same-statistic
Holm comparison, the studentised t twin (simultaneous interval/MDE) and the
family dependence diagnostics.  It never changes ``p_exact_signflip``,
``p_holm``, ``verified`` or ``confirmed``.

No training or GPU is read; the end-to-end case writes synthetic by-length
records to a temporary directory.
"""
import contextlib
import copy
import inspect
import io
import json
import math
import os
import sys as _sys
import tempfile

_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

if not _HAVE_TORCH:
    import pytest as _pytest
    _pytest.skip('CPU torch dependencies are unavailable', allow_module_level=True)

import explib as L
import explib.confirmation as V12
import audit_gates as A


def _deltas(raw, seeds=None):
    seeds = list(range(len(raw))) if seeds is None else list(seeds)
    return {'per_seed': [float(x) for x in raw], 'seeds': seeds,
            'p_exact_signflip': L.exact_signflip_p(raw)['p']}


def _family(rows, names=None):
    names = list(rows) if names is None else list(names)
    comps = {name: _deltas(rows[name]) for name in names}
    return comps, names


def _synthetic_manifest(td, arms, seeds, lengths):
    m = {'manifest_version': 3, 'frozen_at': '2026-01-01',
         'arms': list(arms), 'seeds': list(seeds),
         'primary_comparisons': [
             {'name': f'{arms[0]} vs {a}', 'a': arms[0], 'b': a,
              'endpoint': 'by_len.ppl@2048', 'expected_sign': 'negative'}
             for a in arms[1:]],
         'statistics': {'test': 'exact two-sided sign-flip',
                        'correction': 'holm step-down',
                        'descriptive_only': True,
                        'require_complete_coverage': True,
                        'min_seeds_for_confirmation': 2},
         'eval': {'by_length_ppl': list(lengths), 'n_seqs_clean': 4}}
    p = os.path.join(td, 'm.json')
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(m, f)
    return V12._load_manifest(p)


def _record(arm, seed, msha, lengths, tail_delta):
    by = {}
    nll = 0.0
    prev = 1
    for Ln in lengths:
        n_tok = 4 * (Ln - 1)
        band_tok = n_tok if Ln == min(lengths) else 4 * (Ln - prev)
        nll += 4.0 * band_tok
        if Ln == max(lengths):
            nll += float(tail_delta.get(arm, 0.0))
        by[str(Ln)] = {'ppl': math.exp(nll / n_tok), 'n_tok': n_tok,
                       'nll_sum': nll, 'truncated': False, 'n_seq': 4}
        prev = Ln
    return {'variant': arm, 'seed': seed, 'split': 'test', 'data_fp': 'FP',
            'by_len': by, '_code': L.code_stamp(),
            'manifest_sha256': msha, 'n_seq': 4}


def _synthetic_stats(td, arms=('a', 'b', 'c'), seeds=range(8),
                     tail_delta=None):
    tail_delta = tail_delta or {'a': -20.0, 'b': 0.0, 'c': -5.0}
    m = _synthetic_manifest(td, arms, seeds, (512, 1024, 2048))
    outdir = os.path.join(td, 'res')
    os.makedirs(outdir)
    V12._check_or_write_freeze_lock(outdir, m, 'FP', stage='accessed')
    raw = {}
    for s in seeds:
        for arm in arms:
            raw[f'{arm}::s{s}'] = _record(
                arm, s, m['manifest_sha256'], (512, 1024, 2048), tail_delta)
    bp = os.path.join(outdir, 'bylen.json')
    with open(bp, 'w', encoding='utf-8') as f:
        json.dump(raw, f)
    p = os.path.join(td, 'stats.json')
    with contextlib.redirect_stdout(io.StringIO()):
        st = V12.finaltest_stats(
            manifest=m, bylen_path=bp,
            probe_path=os.path.join(td, 'no_probe.json'), out=p)
    return st, p


def test_policy_and_structural_fail_closed():
    assert L.FAMILY_CORRECTION_POLICY['version'].startswith(
        'confirmation-family-correction-')
    assert set(L.FAMILY_CORRECTION_STATUSES) == {'ok', 'degenerate',
                                                 'not_estimable'}
    base, names = _family({'a': [0.1, -0.1, 0.2, -0.2]})
    for bad in (None, [], 'a', ['a', 'a'], ['a', '']):
        out = L.confirmation_family_correction(base, family_names=bad)
        assert out['status'] == 'not_estimable', (bad, out)
    out = L.confirmation_family_correction(base, family_names=['a', 'b'])
    assert out['status'] == 'not_estimable' and 'missing' in out['reason']
    out = L.confirmation_family_correction(
        {'a': {'omitted': 'no data'}}, family_names=['a'])
    assert out['status'] == 'not_estimable' and 'omitted' in out['reason']
    # Ragged seed vectors are refused, not silently intersected.
    ragg = {'a': _deltas([0.1, -0.1, 0.2], seeds=[0, 1, 2]),
            'b': _deltas([0.1, -0.1, 0.2], seeds=[0, 1, 3])}
    out = L.confirmation_family_correction(ragg, family_names=['a', 'b'])
    assert out['status'] == 'not_estimable'
    assert out['n_common'] == 2 and out['common_seeds'] == [0, 1], out
    # Duplicate/non-integer/non-finite values are refused by name.
    for bad in ({'per_seed': [0.1, True], 'seeds': [0, 1]},
                {'per_seed': [0.1, float('nan')], 'seeds': [0, 1]},
                {'per_seed': [0.1, -0.2], 'seeds': [0, 0]}):
        out = L.confirmation_family_correction({'a': bad}, family_names=['a'])
        assert out['status'] == 'not_estimable', (bad, out)
    one = _deltas([0.1])
    out = L.confirmation_family_correction({'a': one}, family_names=['a'])
    assert out['status'] == 'not_estimable' and 'at least 2' in out['reason']
    many = _deltas([0.1] * 25)
    out = L.confirmation_family_correction({'a': many}, family_names=['a'])
    assert out['status'] == 'not_estimable' and 'cap' in out['reason']


def test_single_contrast_and_score_identity():
    raw = [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3]
    comps, names = _family({'only': raw})
    out = L.confirmation_family_correction(comps, family_names=names)
    assert out['status'] == 'ok', out
    r = out['comparisons']['only']
    assert math.isclose(r['p_raw_score'], r['p_maxt_score'], abs_tol=1e-15), r
    assert r['p_holm_score'] == r['p_raw_score']
    assert r['reject_maxt_score'] is True
    assert out['sets']['all_three_agree'] is True
    # The output is strict JSON and carries no frozen decision field.
    json.dumps(out, allow_nan=False)
    for key in ('confirmed', 'p_holm', 'verified', 'verdict', 'decision'):
        assert key not in out, key


def test_perfect_dependence_gains_power_and_effective_tests():
    # Four proportional contrasts have identical score statistics and are
    # perfectly dependent.  A marginal p just below alpha/m is rejected by
    # max-T but not by Holm, and the effective number of tests is about one.
    base = [-1.0] * 7 + [0.5]
    rows = {f'c{i}': [x * (i + 1) for x in base] for i in range(4)}
    comps, names = _family(rows)
    out = L.confirmation_family_correction(comps, family_names=names)
    assert out['status'] == 'ok', out
    dep = out['dependence']
    assert dep['available'] and dep['effective_tests'] is not None
    assert dep['effective_tests'] < 1.01, dep
    assert dep['max_abs_offdiag'] > 0.999999, dep
    for name in names:
        r = out['comparisons'][name]
        assert math.isclose(r['p_maxt_score'], r['p_raw_score'], abs_tol=1e-15)
        assert r['p_holm_score'] >= r['p_raw_score'] - 1e-15
    # The marginal score p is 2/128 = 0.015625; Holm charges m*p = 0.0625
    # and does not reject, while the perfectly dependent max-T stays at the
    # marginal p and does.  This is the concrete power gain the block exists to
    # expose.
    assert out['sets']['holm_score'] == [], out
    assert set(out['sets']['maxt_score']) == set(names), out
    assert out['sets']['maxt_score_added_vs_holm_score'] == sorted(names), out
    assert out['summary']['correction_effect'] == 'maxt_rejects_more', out


def test_independent_family_is_not_falsely_more_powerful():
    rng = np.random.default_rng(7)
    rows = {f'c{i}': (rng.normal(size=12) * 0.2 + 0.15).tolist()
            for i in range(5)}
    comps, names = _family(rows)
    out = L.confirmation_family_correction(comps, family_names=names)
    assert out['status'] == 'ok', out
    assert out['dependence']['effective_tests'] > 1.0
    for name in names:
        r = out['comparisons'][name]
        assert r['p_maxt_score'] >= r['p_raw_score'] - 1e-15, r
        assert r['p_holm_score'] >= r['p_raw_score'] - 1e-15, r
    assert out['summary']['correction_effect'] in (
        'same', 'maxt_rejects_more', 'maxt_rejects_fewer', 'crossed')


def test_constant_effect_is_estimable_without_studentised_twin():
    rows = {f'c{i}': [0.1 * (i + 1)] * 8 for i in range(3)}
    comps, names = _family(rows)
    out = L.confirmation_family_correction(comps, family_names=names)
    assert out['status'] == 'ok', out
    assert out['studentized']['available'] is False, out
    assert out['studentized']['reason'], out
    for name in names:
        r = out['comparisons'][name]
        assert r['ci_simultaneous'] is None
        assert r['mde_maxt'] is None
        assert 0.0 < r['p_maxt_score'] < 0.05
    json.dumps(out, allow_nan=False)


def test_all_zero_family_is_degenerate_not_a_pass():
    rows = {'a': [0.0] * 8, 'b': [0.0] * 8}
    comps, names = _family(rows)
    out = L.confirmation_family_correction(comps, family_names=names)
    assert out['status'] == 'degenerate', out
    assert out['reason'] and out['p_floor'] == 1.0, out
    assert out['sets']['maxt_score'] == [] and out['sets']['holm_score'] == []


def test_studentised_twin_cross_check_and_mde():
    rng = np.random.default_rng(11)
    rows = {f'c{i}': (rng.normal(size=12) * 0.2 + 0.25).tolist()
            for i in range(4)}
    comps, names = _family(rows)
    out = L.confirmation_family_correction(comps, family_names=names)
    assert out['status'] == 'ok', out
    st = out['studentized']
    assert st['available'] is True, st
    assert st['raw_score_matches'] is True and st['maxt_score_matches'] is True, st
    assert st['critical_value'] is not None
    for name in names:
        r = out['comparisons'][name]
        ci = r['ci_simultaneous']
        assert isinstance(ci, list) and len(ci) == 2 and ci[0] <= ci[1], r
        assert r['mde_holm'] is not None and r['mde_maxt'] is not None, r
        assert r['mde_ratio'] is not None and r['mde_ratio'] > 0.0, r
    assert out['summary']['mean_mde_ratio'] is not None


def test_extreme_finite_values_stay_strict_json():
    rows = {'a': [1e308, -1e308, 1e308, -1e308] * 3,
            'b': [1e308, -1e308, 1e308, -1e308] * 3}
    # The two rows are identical, so the score family is perfectly dependent.
    comps, names = _family(rows)
    out = L.confirmation_family_correction(comps, family_names=names)
    json.dumps(out, allow_nan=False)
    for name in names:
        r = out['comparisons'][name]
        assert math.isfinite(r['p_raw_score'])
        assert math.isfinite(r['p_maxt_score'])
        assert math.isfinite(r['p_holm_score'])


def test_finaltest_emits_block_and_audit_replays_it():
    with tempfile.TemporaryDirectory() as td:
        st, p = _synthetic_stats(td)
        fc = st['family_correction']
        assert fc['kind'] == 'confirmation_family_correction', fc
        assert fc['status'] == 'ok', fc
        assert fc['m_declared'] == 2 and fc['n_seeds'] == 8, fc
        assert fc['n_patterns'] == 256, fc
        assert set(fc['family_names']) == set(st['comparisons']), fc
        json.dumps(fc, allow_nan=False)
        with contextlib.redirect_stdout(io.StringIO()):
            out = A.audit_family_correction(path=p)
        assert out['status'] == 'ok', out
        assert out['n_replayed'] == 2 and not out['problems'], out


def test_audit_detects_tampering_and_legacy_artifacts():
    with tempfile.TemporaryDirectory() as td:
        _st, p = _synthetic_stats(td)
        good = json.load(open(p, encoding='utf-8'))
        name = sorted(good['family_correction']['family_names'])[0]

        def write(mut, name='bad.json'):
            q = os.path.join(td, name)
            with open(q, 'w', encoding='utf-8') as f:
                json.dump(mut, f)
            with contextlib.redirect_stdout(io.StringIO()):
                return A.audit_family_correction(path=q)

        with contextlib.redirect_stdout(io.StringIO()):
            assert A.audit_family_correction(path=p)['status'] == 'ok'
        cases = []
        d = copy.deepcopy(good)
        d['family_correction']['comparisons'][name]['p_maxt_score'] = 0.99
        cases.append(d)
        d = copy.deepcopy(good)
        d['family_correction']['comparisons'][name]['ci_simultaneous'] = [1.0, 0.0]
        cases.append(d)
        d = copy.deepcopy(good)
        d['family_correction']['sets']['maxt_score'] = []
        cases.append(d)
        d = copy.deepcopy(good)
        d['family_correction']['summary']['correction_effect'] = 'maxt_rejects_more'
        cases.append(d)
        d = copy.deepcopy(good)
        d['family_correction']['dependence']['effective_tests'] = 5.0
        cases.append(d)
        d = copy.deepcopy(good)
        d['family_correction']['confirmed'] = [name]
        cases.append(d)
        d = copy.deepcopy(good)
        d['family_correction']['family_names'] = [name]
        cases.append(d)
        for bad in cases:
            assert write(bad)['status'] == 'fail', bad
        legacy = copy.deepcopy(good)
        legacy.pop('family_correction')
        assert write(legacy, 'legacy.json')['status'] == 'not_verifiable_locally'
        bare = copy.deepcopy(good)
        bare['family_correction'] = {
            'kind': 'confirmation_family_correction',
            'status': 'not_estimable', 'family_names': [],
            'comparisons': {}}
        assert write(bare, 'bare.json')['status'] == 'not_applicable'


def test_report_renders_section():
    rng = np.random.default_rng(13)
    rows = {f'c{i}': (rng.normal(size=8) * 0.2 + 0.2).tolist()
            for i in range(3)}
    comps, names = _family(rows)
    block = L.confirmation_family_correction(comps, family_names=names)
    md = '\n'.join(L._family_correction_markdown(block))
    assert '校正方法' in md and 'max-T' in md and 'score' in md, md
    assert '不改变' in md and '`confirmed`' in md, md
    assert 'Holm(score)' in md or 'Holm(mean)' in md, md
    assert L._family_correction_markdown({}) == []
    bad = L.confirmation_family_correction(
        {'a': {'omitted': 'x'}}, family_names=['a'])
    md2 = '\n'.join(L._family_correction_markdown(bad))
    assert '不可估计' in md2 and '不改变' in md2, md2


def test_cli_aliases_and_audit_documentation():
    assert hasattr(A, 'audit_family_correction')
    src = inspect.getsource(A)
    assert 'family_correction' in src, 'audit CLI is not registered'
    assert "'family_maxt': audit_family_correction" in src, src[-2000:]
    assert 'family_correction' in (A.__doc__ or ''), 'audit gate is not documented'
