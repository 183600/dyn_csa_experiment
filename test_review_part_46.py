"""CPU checks for effect-scale and robust-location sensitivity.

The frozen confirmation test is an exact sign-flip test on the raw PPL
difference.  ``effect_sensitivity`` adds two descriptive checks on the same
paired seeds: the log-PPL ratio test (a different estimand) and an exact
Hodges-Lehmann interval for the paired pseudo-median.  ``audit_effect_sensitivity``
rebuilds both from the stored per-seed PPL pairs so an edited sensitivity
block cannot survive.

No training, checkpoint or GPU is read; the end-to-end case uses synthetic
records on a temporary filesystem.
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

import explib.confirmation as V12


def _concordant_pairs():
    a = [50.00, 49.98, 50.01, 49.97, 50.02, 49.99, 50.00, 49.98]
    b = [50.10, 50.09, 50.11, 50.08, 50.10, 50.09, 50.10, 50.08]
    return a, b


def _scale_direction_pairs():
    # Raw PPL increases (four huge-baseline pairs each +1 point) while the
    # geometric ratio decreases (eight near-baseline pairs each ~1% lower):
    # the mean raw delta is positive, the mean log ratio is negative.
    a = [1e6 + 1.0] * 4 + [0.99] * 8
    b = [1e6] * 4 + [1.0] * 8
    return a, b


def _scale_significance_pairs():
    # Same direction, but raw PPL reaches alpha while the log ratio does not.
    a = [6.630208570587, 87.124169514619, 0.904299067193, 166.309764544646,
         2.23677569716, 8.937112844701, 3.045139399511, 2.904543714381,
         133.914771276933, 9.150077804655, 1.123920401681, 16.027697904848]
    b = [5.832867623517, 104.553981635573, 1.122204201475, 184.824834403213,
         2.130423195205, 10.796142103382, 3.012149853518, 3.194734992363,
         136.385996609333, 9.247094569628, 1.177344970542, 18.312854195585]
    return a, b


def _location_discordant_pairs():
    # Nine seeds move slightly down and three move massively up: the raw mean
    # test rejects, but the signed-rank location interval still includes zero.
    deltas = [-1.117705474614405, -0.09193101555606623, -0.01200071598249243,
              -0.02635707558901296, -1.2104461019472013, 0.6537788830389132,
              0.07076181568316273, -8.236148762576823, 0.5039597038146193,
              -2.1101106243855288, -33.5542994435064, -48.135481193265235]
    b = [100.0] * len(deltas)
    a = [y + d for y, d in zip(b, deltas)]
    return a, b


def test_policy_and_statuses_are_declared():
    pol = L.EFFECT_SENSITIVITY_POLICY
    assert pol['version'] == 'effect-sensitivity-1', pol
    assert pol['conf'] == 0.95 and pol['min_pairs'] >= 2, pol
    assert pol['max_location_pairs'] >= 20, pol
    assert set(L.EFFECT_SENSITIVITY_STATUSES) == {
        'concordant', 'scale_direction_discordant',
        'scale_significance_discordant',
        'scale_concordant_location_unavailable', 'location_discordant',
        'direction_unspecified', 'zero_effect', 'degenerate',
        'not_estimable'}


def test_clear_paired_shift_is_concordant():
    a, b = _concordant_pairs()
    r = L.effect_sensitivity(a, b)
    assert r['kind'] == 'effect_scale_location_sensitivity', r
    assert r['status'] == 'concordant', r
    assert r['n'] == 8, r
    assert r['raw']['direction'] == 'negative', r['raw']
    assert r['log_ratio']['direction'] == 'negative', r['log_ratio']
    assert r['raw']['p_exact_signflip'] < 0.05, r['raw']
    assert r['log_ratio']['p_exact_signflip'] < 0.05, r['log_ratio']
    assert r['log_ratio']['geometric_ratio'] < 1.0, r['log_ratio']
    assert r['log_ratio']['relative_pct'] < 0.0, r['log_ratio']
    loc = r['location']
    assert loc['status'] == 'ok', loc
    assert loc['ci'][0] <= loc['estimate'] <= loc['ci'][1], loc
    assert loc['ci'][1] < 0.0, loc
    assert loc['excludes_zero'] is True, loc
    assert loc['attained_conf'] >= 0.95, loc
    assert r['concordance']['all_agree'] is True, r['concordance']


def test_hl_interval_matches_small_sample_order_statistics():
    r6 = L.hodges_lehmann_interval([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert r6['status'] == 'ok', r6
    assert r6['ci'] == [1.0, 6.0], r6
    assert math.isclose(r6['attained_conf'], 0.96875), r6
    assert math.isclose(r6['estimate'], 3.5), r6
    r8 = L.hodges_lehmann_interval([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    assert r8['status'] == 'ok', r8
    assert r8['ci'] == [2.0, 7.0], r8
    assert math.isclose(r8['attained_conf'], 0.9609375), r8
    assert math.isclose(r8['estimate'], 4.5), r8


def test_hl_refuses_too_small_and_too_large_panels():
    small = L.hodges_lehmann_interval([1.0, 2.0, 3.0])
    assert small['status'] == 'unbounded', small
    assert small['ci'] is None and small['excludes_zero'] is None, small
    too_big = L.hodges_lehmann_interval(list(range(1, 202)))
    assert too_big['status'] == 'not_estimable', too_big
    assert 'Walsh' in (too_big['reason'] or ''), too_big


def test_hl_zero_and_nonfinite_handling_and_degenerate_ties():
    r = L.hodges_lehmann_interval([float('nan'), 0.0, 1.0, 2.0, 3.0, 4.0,
                                   5.0, 6.0, float('inf')])
    assert r['n_dropped'] == 2, r
    assert r['n_zero'] == 1, r
    assert r['n'] == 6, r
    assert r['status'] == 'ok', r
    deg = L.hodges_lehmann_interval([1.0] * 6)
    assert deg['status'] == 'degenerate', deg
    assert deg['ci'] is None, deg


def test_low_n_scale_agreement_without_location_is_labelled():
    # Five seeds cannot reach a 95% signed-rank interval, so the scale check
    # may pass without the location check being claimed.
    a = [50.0, 49.9, 50.1, 49.8, 50.0]
    b = [50.2, 50.1, 50.3, 50.0, 50.2]
    r = L.effect_sensitivity(a, b)
    assert r['status'] == 'scale_concordant_location_unavailable', r
    assert r['concordance']['direction_agree'] is True, r['concordance']
    assert r['concordance']['significance_agree'] is True, r['concordance']
    assert r['location']['status'] == 'unbounded', r['location']
    assert r['concordance']['all_agree'] is None, r['concordance']


def test_scale_direction_discordance():
    a, b = _scale_direction_pairs()
    r = L.effect_sensitivity(a, b)
    assert r['status'] == 'scale_direction_discordant', r
    assert r['raw']['direction'] == 'positive', r['raw']
    assert r['log_ratio']['direction'] == 'negative', r['log_ratio']
    assert r['concordance']['direction_agree'] is False, r['concordance']
    assert r['log_ratio']['geometric_ratio'] < 1.0, r['log_ratio']


def test_scale_significance_discordance():
    a, b = _scale_significance_pairs()
    r = L.effect_sensitivity(a, b)
    assert r['status'] == 'scale_significance_discordant', r
    assert r['concordance']['direction_agree'] is True, r['concordance']
    assert r['raw']['significant'] is True, r['raw']
    assert r['log_ratio']['significant'] is False, r['log_ratio']
    assert r['concordance']['significance_agree'] is False, r['concordance']


def test_location_discordance_is_reported_not_hidden():
    a, b = _location_discordant_pairs()
    r = L.effect_sensitivity(a, b)
    assert r['status'] == 'location_discordant', r
    assert r['raw']['significant'] is True, r['raw']
    assert r['location']['status'] == 'ok', r['location']
    assert r['location']['excludes_zero'] is False, r['location']
    assert r['concordance']['location_excludes_zero'] is False, r
    assert r['concordance']['all_agree'] is False, r['concordance']


def test_cancelling_pairs_have_no_direction():
    # Equal-and-opposite pairs cancel exactly in both scales; a mean of zero
    # with non-zero observations is not a directional result.
    r = L.effect_sensitivity([50.0, 51.0, 50.0, 51.0],
                             [51.0, 50.0, 51.0, 50.0])
    assert r['status'] == 'direction_unspecified', r
    assert r['raw']['direction'] is None, r['raw']
    assert r['log_ratio']['direction'] is None, r['log_ratio']
    assert r['concordance']['direction_agree'] is None, r['concordance']


def test_zero_effect_and_invalid_pairs_are_named():
    r = L.effect_sensitivity([50.0] * 6, [50.0] * 6)
    assert r['status'] == 'zero_effect', r
    assert r['raw']['p_exact_signflip'] == 1.0, r['raw']
    assert r['location']['status'] == 'not_estimable', r['location']
    mixed = L.effect_sensitivity(
        [50.0, 0.0, float('nan'), 50.0, 50.0, 50.0],
        [50.1, 50.1, 50.1, 50.1, 50.1, 50.1],
        seeds=[0, 1, 2, 3, 4, 5])
    assert mixed['n'] == 4, mixed
    assert {d['seed'] for d in mixed['dropped']} == {1, 2}, mixed['dropped']
    assert mixed['status'] in L.EFFECT_SENSITIVITY_STATUSES, mixed


def test_bad_arguments_fail_closed():
    a, b = _concordant_pairs()
    for bad in (
        lambda: L.effect_sensitivity('abc', b),
        lambda: L.effect_sensitivity(a, b[:-1]),
        lambda: L.effect_sensitivity(a, b, seeds=[0] * len(a)),
        lambda: L.effect_sensitivity(a, b, seeds=[f's{i}' for i in range(len(a))]),
        lambda: L.effect_sensitivity(a, b, alpha=1.5),
        lambda: L.hodges_lehmann_interval([1.0, 2.0], conf=0.0),
        lambda: L.hodges_lehmann_interval('abc'),
    ):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError('malformed input was accepted')


def test_json_is_strict_and_replay_is_deterministic():
    a, b = _concordant_pairs()
    r = L.effect_sensitivity(a, b)
    stored = json.loads(json.dumps(r, allow_nan=False))
    again = L.effect_sensitivity(
        [p['with'] for p in stored['per_seed']],
        [p['without'] for p in stored['per_seed']],
        seeds=[p['seed'] for p in stored['per_seed']],
        alpha=stored['alpha'], conf=stored['conf'])
    assert again == stored, (again, stored)


def test_extreme_but_finite_values_do_not_emit_nan_or_inf():
    big = 1e308
    r = L.effect_sensitivity(
        [big, big, big * 0.999, big * 0.998],
        [big * 0.999, big * 0.998, big, big * 0.997])
    json.dumps(r, allow_nan=False)
    assert r['status'] in L.EFFECT_SENSITIVITY_STATUSES, r


def _synthetic_manifest(td, arms, seeds):
    m = {'manifest_version': 3, 'frozen_at': '2026-01-01',
         'arms': list(arms), 'seeds': list(seeds),
         'primary_comparisons': [
             {'name': f'{arms[0]} vs {a}', 'a': arms[0], 'b': a,
              'endpoint': 'by_len.ppl@4096', 'expected_sign': 'negative'}
             for a in arms[1:]],
         'statistics': {'test': 'exact two-sided sign-flip',
                        'correction': 'holm step-down',
                        'descriptive_only': True,
                        'require_complete_coverage': True,
                        'min_seeds_for_confirmation': 2}}
    p = os.path.join(td, 'm.json')
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(m, f)
    return V12._load_manifest(p)


def _synthetic_record(arm, seed, ppl, msha):
    return {'variant': arm, 'seed': seed, 'split': 'test', 'data_fp': 'FP',
            'by_len': {'4096': {'ppl': ppl, 'n_tok': 1000,
                                'truncated': False}},
            '_code': L.code_stamp(), 'manifest_sha256': msha}


def _synthetic_stats(td, arms, seeds):
    m = _synthetic_manifest(td, arms, seeds)
    outdir = os.path.join(td, 'res')
    os.makedirs(outdir)
    V12._check_or_write_freeze_lock(outdir, m, 'FP', stage='accessed')
    raw = {}
    for s in seeds:
        for k, arm in enumerate(arms):
            raw[f'{arm}::s{s}'] = _synthetic_record(
                arm, s, 50.0 + 1.0 * k + 0.01 * s, m['manifest_sha256'])
    bp = os.path.join(outdir, 'bylen.json')
    with open(bp, 'w', encoding='utf-8') as f:
        json.dump(raw, f)
    p = os.path.join(td, 'stats.json')
    with contextlib.redirect_stdout(io.StringIO()):
        st = V12.finaltest_stats(
            manifest=m, bylen_path=bp,
            probe_path=os.path.join(td, 'no_probe.json'), out=p)
    return st, p, m


def test_finaltest_stats_emits_a_replayable_effect_sensitivity_block():
    arms = ('a_arm', 'b_arm', 'c_arm')
    with tempfile.TemporaryDirectory() as td:
        st, p, _m = _synthetic_stats(td, arms, range(6))
        for name, comp in st['comparisons'].items():
            es = comp.get('effect_sensitivity')
            assert isinstance(es, dict), (name, comp)
            assert es['kind'] == 'effect_scale_location_sensitivity', es
            assert es['status'] in L.EFFECT_SENSITIVITY_STATUSES, es
            assert es['n'] == 6, es
            assert len(es['per_seed']) == 6, es
            assert es['raw']['p_exact_signflip'] < 0.05, es['raw']
            assert es['log_ratio']['p_exact_signflip'] < 0.05, es['log_ratio']
        with contextlib.redirect_stdout(io.StringIO()):
            out = A.audit_effect_sensitivity(path=p)
        assert out['status'] == 'ok', out
        assert out['n_replayed'] == 2, out
        assert not out['problems'], out['problems']


def _write_stats(td, st):
    p = os.path.join(td, 'stats.json')
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(st, f)
    return p


def _audit_block(st, name):
    return st['finaltest']['comparisons'][name]['effect_sensitivity']


def test_audit_rejects_tampered_effect_sensitivity_blocks():
    arms = ('a_arm', 'b_arm', 'c_arm')
    with tempfile.TemporaryDirectory() as td:
        st, _p, _m = _synthetic_stats(td, arms, range(6))
        wrapped = {'finaltest': copy.deepcopy(st)}
        name = next(iter(wrapped['finaltest']['comparisons']))
        good = _write_stats(td, wrapped)
        with contextlib.redirect_stdout(io.StringIO()):
            assert A.audit_effect_sensitivity(path=good)['status'] == 'ok'
        cases = []
        d = copy.deepcopy(wrapped)
        _audit_block(d, name)['raw']['p_exact_signflip'] = 0.99
        cases.append(d)
        d = copy.deepcopy(wrapped)
        _audit_block(d, name)['log_ratio']['p_exact_signflip'] = 0.99
        cases.append(d)
        d = copy.deepcopy(wrapped)
        _audit_block(d, name)['location']['ci'] = [0.0, 1.0]
        cases.append(d)
        d = copy.deepcopy(wrapped)
        _audit_block(d, name)['status'] = 'scale_direction_discordant'
        cases.append(d)
        d = copy.deepcopy(wrapped)
        _audit_block(d, name)['per_seed'][0]['with'] *= 1.01
        cases.append(d)
        d = copy.deepcopy(wrapped)
        _audit_block(d, name)['confirmed'] = [name]
        cases.append(d)
        d = copy.deepcopy(wrapped)
        _audit_block(d, name)['concordance']['all_agree'] = False
        cases.append(d)
        for bad in cases:
            p = _write_stats(td, bad)
            with contextlib.redirect_stdout(io.StringIO()):
                out = A.audit_effect_sensitivity(path=p)
            assert out['status'] == 'fail', (bad, out)


def test_audit_distinguishes_legacy_missing_and_bare_not_estimable():
    with tempfile.TemporaryDirectory() as td:
        legacy = {'finaltest': {'alpha': 0.05, 'manifest': {'seeds': [0, 1]},
                                'comparisons': {'c': {'n': 2, 'seeds': [0, 1],
                                                      'per_seed': [1.0, 2.0]}}}}
        p = _write_stats(td, legacy)
        with contextlib.redirect_stdout(io.StringIO()):
            out = A.audit_effect_sensitivity(path=p)
        assert out['status'] == 'not_verifiable_locally', out
        bare = {'finaltest': {'alpha': 0.05, 'manifest': {'seeds': [0, 1]},
                              'comparisons': {'c': {
                                  'seeds': [0, 1], 'per_seed': [1.0, 2.0],
                                  'effect_sensitivity': {
                                      'kind': 'effect_scale_location_sensitivity',
                                      'status': 'not_estimable', 'n': 1,
                                      'status_reason': 'one pair'}}}}}
        p2 = _write_stats(td, bare)
        with contextlib.redirect_stdout(io.StringIO()):
            out2 = A.audit_effect_sensitivity(path=p2)
        assert out2['status'] == 'not_verifiable_locally', out2
        assert out2['comparisons']['c']['status'] == 'not_applicable', out2
        with contextlib.redirect_stdout(io.StringIO()):
            missing = A.audit_effect_sensitivity(path=os.path.join(td, 'none.json'))
        assert missing['status'] == 'not_verifiable_locally', missing


def test_report_renders_effect_scale_and_location_section():
    a, b = _concordant_pairs()
    es = L.effect_sensitivity(a, b)
    comp = {'n': 8, 'mean': -0.05, 'std': 0.01,
            'p_exact_signflip': es['raw']['p_exact_signflip'],
            'p_floor': es['raw']['p_floor'], 'coverage_ok': True,
            'param_matched': True, 'same_init_core_fp': True,
            'direction_ok': True, 'dropped_pairs': [], 'doc_sampling': {},
            'effect_sensitivity': es, 'seeds': list(range(8)),
            'per_seed': [p['raw_delta'] for p in es['per_seed']]}
    st = {'finaltest': {
        'alpha': 0.05,
        'manifest': {'arms': ['a', 'b'], 'seeds': list(range(8))},
        'by_length': {}, 'comparisons': {'a vs b': comp},
        'holm': {'a vs b': {'p_holm': 0.03}}, 'confirmed': ['a vs b'],
        'verdicts': {},
    }}
    with tempfile.TemporaryDirectory() as td:
        sp = _write_stats(td, st)
        out = os.path.join(td, 'REPORT.md')
        original = V12.v12_analysis
        V12.v12_analysis = lambda out=None: st  # noqa: ARG005
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                V12.build_report(out=out, stats_p=sp)
        finally:
            V12.v12_analysis = original
        with open(out, encoding='utf-8') as f:
            text = f.read()
    assert '效应尺度/位置敏感性' in text, text[:1000]
    assert 'HL 伪中位数' in text, text
    assert '几何比' in text, text
    assert 'concordant' in text, text
    assert '不进入' in text and 'Holm' in text, text
    assert '不是算术均值的置信区间' in text, text


def test_wiring_is_in_audit_cli_and_confirmation():
    assert 'effect_sensitivity' in (A.__doc__ or ''), 'audit command not documented'
    assert "'effect_sensitivity': audit_effect_sensitivity()" in inspect.getsource(A.run_report)
    assert "'effect_sensitivity': audit_effect_sensitivity" in inspect.getsource(A)
    cli = inspect.getsource(A._run_cli)
    for alias in ('effect_sensitivity', 'effect_scale', 'scale_location'):
        assert alias in cli, alias
    assert 'effect_sensitivity' in inspect.getsource(V12.finaltest_stats)
    assert '效应尺度/位置敏感性' in inspect.getsource(V12.build_report)
