"""CPU checks for the winner's-curse / selection-bias readout.

The frozen study reports the observed best arm and its top-two gap, but that
mean is a maximum over K noisy arms.  ``winner_curse_correction`` adds a paired
seed-bootstrap optimism estimate, an optimism-corrected endpoint, and a
split-sample honest value / selection regret.  ``audit_gates.audit_winner_curse``
replays the stored block so a hand-edited corrected value cannot survive.

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


def _clear_arms():
    return {
        'a': [50.0, 50.02, 49.98, 50.01, 49.99, 50.03, 50.00, 49.97],
        'b': [51.0, 51.02, 50.98, 51.01, 50.99, 51.03, 51.00, 50.97],
        'c': [52.0, 52.02, 51.98, 52.01, 51.99, 52.03, 52.00, 51.97],
    }


def _noise_dominated_arms():
    # Arm ``a`` has a large seed SD; ``b`` is slightly worse on average but
    # very stable.  The observed winner is b, yet the observed gap is smaller
    # than what selecting on this panel is expected to cost.
    return {
        'a': [49.835, 49.512, 50.608, 50.038, 51.221, 49.739],
        'b': [50.063, 49.998, 50.033, 50.062, 49.98, 50.015],
        'c': [50.048, 50.046, 50.07, 50.091, 50.082, 50.053],
    }


def _write_block(td, block):
    p = os.path.join(td, 'stats.json')
    with open(p, 'w', encoding='utf-8') as f:
        json.dump({'finaltest': {'winner_curse': block}}, f)
    return p


def test_policy_and_statuses_are_declared():
    pol = L.WINNER_CURSE_POLICY
    assert pol['version'] == 'winner-curse-1', pol
    assert pol['better'] == 'min', pol
    assert pol['n_boot'] >= 200 and pol['min_seeds'] >= 4, pol
    assert 0.0 < pol['lead_level'] < 1.0, pol
    assert set(L.WINNER_CURSE_STATUSES) == {
        'selection_robust', 'selection_noise_dominated', 'winner_unstable',
        'tied_best', 'not_estimable'}


def test_clear_separation_has_negligible_optimism():
    r = L.winner_curse_correction(_clear_arms(), n_boot=2000)
    assert r['status'] == 'selection_robust', r
    assert r['observed_winner'] == 'a', r
    assert r['observed_winner_p_best'] > 0.99, r['observed_winner_p_best']
    assert abs(r['bootstrap']['optimism_mean']) < 0.05, r['bootstrap']
    assert abs(r['bootstrap']['corrected_winner_value']
               - r['observed_winner_value']) < 0.1, r['bootstrap']
    assert r['bootstrap']['selection_regret_mean'] == 0.0, r['bootstrap']
    assert r['split_sample']['selection_frequency']['a'] == 1.0, r['split_sample']
    assert r['split_sample']['observed_gap_exceeds_regret'] is True, r['split_sample']
    assert r['observed_tied_for_best'] == ['a'], r
    assert r['complete_coverage'] is True, r
    # Strict JSON, then the stored block must replay bit-for-bit.
    stored = json.loads(json.dumps(r, allow_nan=False))
    again = L.winner_curse_correction(
        stored['per_arm_seed_values'],
        better=stored['policy']['better'], n_boot=stored['policy']['n_boot'],
        seed=stored['policy']['bootstrap_seed'],
        split_seed=stored['policy']['split_seed'],
        max_partitions=stored['policy']['max_partitions'],
        lead_level=stored['policy']['lead_level'],
        min_seeds=stored['policy']['min_seeds'])
    assert again['status'] == stored['status'], again
    assert again['bootstrap'] == stored['bootstrap'], again['bootstrap']
    assert again['split_sample'] == stored['split_sample'], again['split_sample']


def test_noisy_panel_is_selection_noise_dominated():
    r = L.winner_curse_correction(
        _noise_dominated_arms(), n_boot=1000, min_seeds=4)
    assert r['status'] == 'selection_noise_dominated', r
    assert r['observed_winner'] == 'b', r
    assert r['observed_winner_p_best'] >= 0.60, r['observed_winner_p_best']
    # The selected extreme is better than the full-sample value of the
    # procedure: corrected/honest are worse (higher for PPL) than observed.
    assert r['bootstrap']['optimism_mean'] < 0.0, r['bootstrap']
    assert r['bootstrap']['corrected_winner_value'] > r['observed_winner_value'], r
    assert r['split_sample']['honest_value_mean'] > r['observed_winner_value'], r
    assert (r['bootstrap']['selection_regret_mean']
            > r['observed_gap']), r['bootstrap']
    assert r['bootstrap']['observed_gap_exceeds_regret'] is False, r['bootstrap']
    assert r['split_sample']['observed_gap_exceeds_regret'] is False, r['split_sample']
    assert r['split_sample']['regret_share_of_gap'] > 1.0, r['split_sample']


def test_better_max_direction_and_correction_orientation():
    r = L.winner_curse_correction(_clear_arms(), better='max', n_boot=1000)
    assert r['status'] == 'selection_robust', r
    assert r['observed_winner'] == 'c', r
    assert r['observed_means']['c'] == 52.0, r['observed_means']
    assert abs(r['observed_gap'] - 1.0) < 1e-12, r['observed_gap']
    # Higher is better, so a positive optimism makes the corrected value lower.
    assert r['bootstrap']['optimism_mean'] >= 0.0, r['bootstrap']
    assert r['bootstrap']['corrected_winner_value'] <= r['observed_winner_value'], r
    assert r['split_sample']['selection_frequency']['c'] == 1.0, r


def test_split_sample_is_deterministic_and_respects_the_cap():
    arms = _clear_arms()
    a = L.winner_curse_correction(arms, n_boot=200, max_partitions=5)
    b = L.winner_curse_correction(arms, n_boot=200, max_partitions=5)
    assert a['split_sample']['n_partitions'] == 5, a['split_sample']
    assert a['split_sample']['n_folds'] == 10, a['split_sample']
    assert a['split_sample'] == b['split_sample'], (a, b)
    c = L.winner_curse_correction(
        arms, n_boot=200, max_partitions=5, split_seed=12345)
    assert c['split_sample']['n_partitions'] == 5, c['split_sample']
    # A different split seed can change the sampled partitions, but the
    # observed selection is untouched.
    assert c['observed_winner'] == a['observed_winner'], (c, a)


def test_seed_intersection_and_dropped_seeds_are_reported():
    a = {s: 50.0 + 0.01 * s for s in range(8)}
    b = {s: 51.0 + 0.01 * s for s in range(1, 9)}
    r = L.winner_curse_correction({'a': a, 'b': b}, n_boot=500)
    assert r['n_seeds'] == 7, r
    assert r['seeds'] == list(range(1, 8)), r
    assert r['complete_coverage'] is False, r
    assert r['dropped_seeds'] == {'a': [0], 'b': [8]}, r
    assert set(r['per_arm_seed_values']['a']) == {str(s) for s in range(1, 8)}, r


def test_tied_best_has_no_unique_winner_to_correct():
    r = L.winner_curse_correction(
        {'a': [1.0] * 6, 'b': [1.0] * 6, 'c': [1.2] * 6}, n_boot=500)
    assert r['status'] == 'tied_best', r
    assert r['observed_tied_for_best'] == ['a', 'b'], r
    assert abs(sum(r['p_best'].values()) - 1.0) < 1e-12, r['p_best']


def test_declared_practical_tie_blocks_a_unique_winner():
    r = L.winner_curse_correction(
        {'a': [50.0] * 6, 'b': [50.05] * 6, 'c': [51.0] * 6},
        n_boot=300, tie_tol=0.1)
    assert r['status'] == 'tied_best', r
    assert r['observed_tied_for_best'] == ['a', 'b'], r
    assert r['observed_gap'] <= 0.1, r
    # The declared margin must replay through the audit too.
    with tempfile.TemporaryDirectory() as td:
        with contextlib.redirect_stdout(io.StringIO()):
            out = A.audit_winner_curse(path=_write_block(td, r))
    assert out['status'] == 'ok', out


def test_insufficient_arms_and_seeds_are_not_estimable():
    one = L.winner_curse_correction({'a': [1.0, 2.0, 3.0, 4.0, 5.0]})
    assert one['status'] == 'not_estimable', one
    assert one['kind'] == 'winner_curse_correction', one
    short = L.winner_curse_correction(
        {'a': [1.0, 2.0, 3.0], 'b': [2.0, 3.0, 4.0]}, min_seeds=5)
    assert short['status'] == 'not_estimable', short
    assert 'min_seeds' in short['reason'], short
    five = L.winner_curse_correction(
        {'a': [1.0, 1.1, 0.9, 1.0, 1.05],
         'b': [2.0, 2.1, 1.9, 2.0, 2.05]}, n_boot=200, min_seeds=5)
    assert five['status'] in L.WINNER_CURSE_STATUSES, five
    assert five['status'] != 'not_estimable', five


def test_invalid_inputs_fail_closed():
    good = {'a': [1.0, 2.0, 3.0, 4.0, 5.0],
            'b': [2.0, 3.0, 4.0, 5.0, 6.0]}
    for bad in (
        {'a': [True, 2.0, 3.0, 4.0, 5.0], 'b': [2.0] * 5},
        {'a': [float('nan')] + [1.0] * 4, 'b': [2.0] * 5},
        {'a': [float('inf')] + [1.0] * 4, 'b': [2.0] * 5},
        {'a': ['1.0', 2.0, 3.0, 4.0, 5.0], 'b': [2.0] * 5},
        {'a': [1.0] * 4, 'b': [2.0] * 5},
    ):
        with _pytest.raises(ValueError):
            L.winner_curse_correction(bad, n_boot=200)
    two_seeds = L.winner_curse_correction(
        {'a': [1.0, 2.0], 'b': [2.0, 3.0]}, n_boot=200)
    assert two_seeds['status'] == 'not_estimable', two_seeds
    with _pytest.raises(ValueError):
        L.winner_curse_correction(
            [('a', [1.0] * 5), ('a', [2.0] * 5)], n_boot=200)
    with _pytest.raises(ValueError):
        L.winner_curse_correction(
            {'a': [1.0] * 5, 'b': {i: 2.0 for i in range(5)}})
    for kwargs in ({'n_boot': 10}, {'n_boot': True}, {'lead_level': 0.0},
                   {'lead_level': 1.0}, {'split_seed': -1}, {'min_seeds': 3},
                   {'max_partitions': 0}, {'tie_tol': -1.0},
                   {'tie_tol': float('inf')}):
        with _pytest.raises(ValueError):
            L.winner_curse_correction(good, **kwargs)


def test_extreme_finite_values_stay_json_safe():
    arms = {'a': [1e308] * 6, 'b': [1.01e308] * 6}
    r = L.winner_curse_correction(arms, n_boot=200, min_seeds=4)
    json.dumps(r, allow_nan=False)
    assert r['status'] != 'not_estimable', r
    assert r['observed_winner'] == 'a', r
    assert math.isfinite(r['observed_gap']), r['observed_gap']
    # The per-arm means may be representable; the corrected value must not be
    # inf/NaN either way.
    assert (r['bootstrap']['corrected_winner_value'] is None
            or math.isfinite(r['bootstrap']['corrected_winner_value'])), r
    # A mixed-sign extreme panel can overflow a naive sum even though every
    # value and the mean are finite; JSON must stay strict.
    mixed = L.winner_curse_correction(
        {'a': [-1e308, 1e308, -1e308, 1e308, -1e308],
         'b': [0.0] * 5}, n_boot=200, min_seeds=5)
    json.dumps(mixed, allow_nan=False)
    assert mixed['status'] != 'not_estimable', mixed


def test_audit_replays_and_rejects_tampering():
    r = L.winner_curse_correction(_clear_arms(), n_boot=500)
    with tempfile.TemporaryDirectory() as td:
        p = _write_block(td, r)
        with contextlib.redirect_stdout(io.StringIO()):
            out = A.audit_winner_curse(path=p)
        assert out['status'] == 'ok', out
        for name, mutate in (
            ('corrected', lambda x: x['bootstrap'].__setitem__(
                'corrected_winner_value', 1.0)),
            ('optimism', lambda x: x['bootstrap'].__setitem__(
                'optimism_mean', 99.0)),
            ('p_best', lambda x: x['p_best'].__setitem__('a', 0.5)),
            ('confirmed', lambda x: x.__setitem__('confirmed', True)),
            ('status', lambda x: x.__setitem__('status', 'winner_unstable')),
        ):
            d = copy.deepcopy(r)
            mutate(d)
            with contextlib.redirect_stdout(io.StringIO()):
                bad = A.audit_winner_curse(path=_write_block(td, d))
            assert bad['status'] == 'fail', (name, bad)
    with tempfile.TemporaryDirectory() as td:
        with contextlib.redirect_stdout(io.StringIO()):
            missing = A.audit_winner_curse(path=os.path.join(td, 'none.json'))
        assert missing['status'] == 'not_verifiable_locally', missing
        bare = {'kind': 'winner_curse_correction',
                'status': 'not_estimable', 'reason': 'no data'}
        with contextlib.redirect_stdout(io.StringIO()):
            out = A.audit_winner_curse(path=_write_block(td, bare))
        assert out['status'] == 'not_applicable', out


def test_audit_rejects_structural_inconsistency():
    r = L.winner_curse_correction(_clear_arms(), n_boot=500)
    cases = []
    d = copy.deepcopy(r)
    d['status'] = 'selection_robust'
    d['bootstrap']['observed_gap_exceeds_regret'] = False
    cases.append(d)
    d = copy.deepcopy(r)
    d['status'] = 'tied_best'
    d['observed_tied_for_best'] = ['a']
    cases.append(d)
    d = copy.deepcopy(r)
    d['p_best']['a'] = 0.7
    cases.append(d)
    d = copy.deepcopy(r)
    d['split_sample']['selection_frequency']['a'] = 0.7
    cases.append(d)
    with tempfile.TemporaryDirectory() as td:
        for d in cases:
            with contextlib.redirect_stdout(io.StringIO()):
                out = A.audit_winner_curse(path=_write_block(td, d))
            assert out['status'] == 'fail', (d['status'], out)


def test_wiring_is_in_finaltest_report_and_cli():
    assert 'winner_curse' in (A.__doc__ or ''), 'the audit command is not documented'
    assert "'winner_curse': audit_winner_curse()" in inspect.getsource(A.run_report)
    assert "'winner_curse': audit_winner_curse" in inspect.getsource(A)
    cli = inspect.getsource(A._run_cli)
    for alias in ('winner_curse', 'selection_bias', 'selection_adjusted'):
        assert alias in cli, alias
    assert 'winner_curse_correction' in inspect.getsource(V12.finaltest_stats)
    assert '最优臂的选择偏差' in inspect.getsource(V12.build_report)


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


def test_finaltest_stats_emits_a_replayable_winner_curse_block():
    arms = ('a_arm', 'b_arm', 'c_arm')
    with tempfile.TemporaryDirectory() as td:
        m = _synthetic_manifest(td, arms, range(6))
        outdir = os.path.join(td, 'res')
        os.makedirs(outdir)
        V12._check_or_write_freeze_lock(outdir, m, 'FP', stage='accessed')
        raw = {}
        for s in range(6):
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
        wc = st['winner_curse']
        assert wc['kind'] == 'winner_curse_correction', wc
        assert wc['endpoint'] == 'by_len.ppl@4096', wc
        assert wc['n_arms'] == 3 and wc['n_seeds'] == 6, wc
        assert wc['observed_winner'] == 'a_arm', wc
        with contextlib.redirect_stdout(io.StringIO()):
            out = A.audit_winner_curse(path=p)
        assert out['status'] == 'ok', out


def test_build_report_renders_the_selection_bias_section():
    arms = _clear_arms()
    wc = L.winner_curse_correction(arms, n_boot=1000)
    wc['endpoint'] = 'by_len.ppl@4096'
    rk = L.arm_rank_inference(arms, n_boot=1000)
    rk['endpoint'] = 'by_len.ppl@4096'
    st = {'finaltest': {
        'alpha': 0.05,
        'manifest': {'arms': list(arms), 'seeds': list(range(8))},
        'by_length': {
            a: {'4096': {'mean': rk['observed_means'][a], 'n': 8,
                         'ppl_by_seed': rk['per_arm_seed_values'][a]}}
            for a in arms},
        'comparisons': {}, 'holm': {}, 'arm_ranking': rk,
        'winner_curse': wc, 'confirmed': [], 'verdicts': {},
    }}
    with tempfile.TemporaryDirectory() as td:
        sp = os.path.join(td, 'stats.json')
        out = os.path.join(td, 'REPORT.md')
        with open(sp, 'w', encoding='utf-8') as f:
            json.dump(st, f)
        original = V12.v12_analysis
        V12.v12_analysis = lambda out=None: st  # noqa: ARG005
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                V12.build_report(out=out, stats_p=sp)
        finally:
            V12.v12_analysis = original
        with open(out, encoding='utf-8') as f:
            text = f.read()
    assert '最优臂的选择偏差与赢家诅咒校正' in text, text[:800]
    assert 'bootstrap 选择乐观度' in text, text
    assert 'selection_robust' in text, text
    assert '不是新的显著性检验' in text, text
    assert '无偏均值' in text, text
