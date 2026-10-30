"""CPU checks for the arm-ranking resolution readout and its replay gate.

The frozen study compares six arms with pre-registered pairwise sign-flip
tests, but it never says how much of the observed *ordering* is resolved:
the observed minimum is a selected extreme, and with a dozen seeds its
probability of really being best can be far below one.  ``arm_rank_inference``
adds a paired seed-bootstrap ranking readout (P(best), P(top2), rank
intervals, top-two gap, pairwise order probabilities, leave-one-seed-out
winner stability).  ``audit_gates.audit_arm_ranking`` replays the stored block
so a hand-edited winner/p-value cannot survive.

No test split, checkpoint or GPU is read; the end-to-end case uses synthetic
records on a temporary filesystem.
"""
import contextlib
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


def _overlap_arms():
    return {
        'a': [50.0, 50.6, 49.4, 50.5, 49.5, 50.4, 49.6, 50.3],
        'b': [50.1, 49.9, 50.7, 49.3, 50.6, 49.4, 50.5, 49.5],
        'c': [50.2, 50.0, 49.8, 50.1, 49.9, 50.3, 49.7, 50.4],
    }


def test_clear_separation_is_decisive_and_replays():
    r = L.arm_rank_inference(_clear_arms(), n_boot=2000)
    assert r['status'] == 'decisive', r
    assert r['observed_winner'] == 'a', r
    assert r['observed_rank'] == {'a': 1, 'b': 2, 'c': 3}, r
    assert r['p_best']['a'] > 0.99, r['p_best']
    assert r['observed_winner_p_best'] > 0.99, r
    assert r['top2_gap_resolved'] is True, r
    assert r['p_best']['b'] == 0.0 and r['p_best']['c'] == 0.0, r['p_best']
    assert r['rank_ci95']['a'] == [1, 1], r['rank_ci95']
    assert r['winner_stable_loo'] is True, r['loo']
    # The block is JSON-serialisable and its own values replay it exactly.
    stored = json.loads(json.dumps(r))
    again = L.arm_rank_inference(
        stored['per_arm_seed_values'], better=stored['better'],
        n_boot=stored['n_boot'], seed=stored['bootstrap_seed'],
        rank_level=stored['rank_level'], lead_level=stored['lead_level'],
        tie_tol=stored['tie_tol'], min_seeds=stored['min_seeds'])
    assert again['p_best'] == stored['p_best'], (again['p_best'], stored['p_best'])
    assert again['rank_ci95'] == stored['rank_ci95'], again['rank_ci95']
    assert again['n_distinct_bootstrap_winners'] == 1, again


def test_overlapping_panel_is_not_overclaimed():
    r = L.arm_rank_inference(_overlap_arms(), n_boot=2000)
    assert r['status'] in ('leading', 'undifferentiated'), r['status']
    assert abs(sum(r['p_best'].values()) - 1.0) < 1e-9, r['p_best']
    assert max(r['p_best'].values()) < 0.95, r['p_best']
    assert set(r['observed_rank']) == {'a', 'b', 'c'}, r['observed_rank']


def test_max_direction_and_pairwise_order_probabilities():
    r = L.arm_rank_inference(_clear_arms(), better='max', n_boot=1000)
    assert r['observed_winner'] == 'c', r
    assert r['observed_rank'] == {'c': 1, 'b': 2, 'a': 3}, r
    assert r['p_best']['c'] > 0.99, r['p_best']
    pp = r['pairwise_p_better']
    assert pp['a']['a'] is None and pp['c']['c'] is None, pp
    for x, y in (('a', 'b'), ('a', 'c'), ('b', 'c')):
        assert abs(pp[x][y] + pp[y][x] - 1.0) < 1e-12, (x, y, pp)
    assert pp['c']['a'] > 0.99, pp


def test_top_two_gap_and_order_probability():
    r = L.arm_rank_inference(_clear_arms(), n_boot=1000, tie_tol=0.5)
    lo, hi = r['top2_gap_ci95']
    assert 0.0 < lo <= hi, r['top2_gap_ci95']
    assert r['top2_order_prob'] > 0.99, r['top2_order_prob']
    assert r['practical_tie'] is False, r
    # A declared practical margin larger than the gap is a tie even though the
    # statistical ordering is resolved.
    tie = L.arm_rank_inference(_clear_arms(), n_boot=1000, tie_tol=1.5)
    assert tie['practical_tie'] is True, tie['practical_tie']


def test_identical_arms_are_undifferentiated_not_a_fake_winner():
    v = {a: [1.0] * 6 for a in ('a', 'b', 'c')}
    r = L.arm_rank_inference(v, n_boot=500)
    assert r['status'] == 'undifferentiated', r
    assert r['observed_rank'] == {'a': 1, 'b': 1, 'c': 1}, r
    assert abs(r['p_best']['a'] - 1.0 / 3.0) < 1e-12, r['p_best']
    assert r['top2_gap_ci95'] == [0.0, 0.0], r
    assert r['practical_tie'] is True, r


def test_tied_means_are_undifferentiated_even_with_unequal_variance():
    # Equal observed means but different per-seed spread: the bootstrap's
    # p_best is not uniform, so the status must not pretend there is a winner.
    r = L.arm_rank_inference(
        {'a': [-1.0, 1.0] * 4, 'b': [0.0] * 8, 'c': [0.1, -0.1] * 4},
        n_boot=500)
    assert r['status'] == 'undifferentiated', r
    assert r['observed_tied_for_best'] == ['a', 'b', 'c'], r
    assert r['observed_rank'] == {'a': 1, 'b': 1, 'c': 1}, r


def test_seed_keyed_intersection_and_missing_coverage():
    v = {
        'a': {s: 50.0 + s for s in range(7)},
        'b': {s: 51.0 + s for s in range(6)},
        'c': {s: 52.0 + s for s in range(7)},
    }
    r = L.arm_rank_inference(v, n_boot=500, min_seeds=5)
    assert r['seeds'] == [0, 1, 2, 3, 4, 5], r['seeds']
    assert r['complete_coverage'] is False, r
    assert r['dropped_seeds']['a'] == [6], r['dropped_seeds']
    assert r['dropped_seeds']['b'] == [], r['dropped_seeds']
    assert r['dropped_seeds']['c'] == [6], r['dropped_seeds']
    assert r['status'] == 'decisive', r
    assert set(r['per_arm_seed_values']['b']) == {str(s) for s in range(6)}, r


def test_pairwise_only_and_not_estimable():
    two = L.arm_rank_inference({'a': [1, 2, 3, 4, 5], 'b': [2, 3, 4, 5, 6]},
                               n_boot=500)
    assert two['status'] == 'pairwise_only', two
    assert abs(sum(two['p_best'].values()) - 1.0) < 1e-12, two['p_best']
    assert two['p_top2'] == {'a': 1.0, 'b': 1.0}, two['p_top2']
    one = L.arm_rank_inference({'a': [1, 2, 3, 4, 5]}, n_boot=500)
    assert one['status'] == 'not_estimable' and 'at least' in one['reason'], one
    few = L.arm_rank_inference(
        {'a': [1, 2, 3], 'b': [2, 3, 4], 'c': [3, 4, 5]}, n_boot=500)
    assert few['status'] == 'not_estimable' and 'min_seeds' in few['reason'], few


def test_determinism_and_bootstrap_seed():
    a = L.arm_rank_inference(_overlap_arms(), n_boot=1000, seed=7)
    b = L.arm_rank_inference(_overlap_arms(), n_boot=1000, seed=7)
    for k in ('p_best', 'p_top2', 'mean_rank', 'rank_ci95',
              'top2_gap_ci95', 'pairwise_p_better'):
        assert a[k] == b[k], k
    c = L.arm_rank_inference(_overlap_arms(), n_boot=1000, seed=8)
    # A different bootstrap seed may move the last-digit frequencies; the
    # observed ranking and the structural contract must not move.
    assert a['observed_rank'] == c['observed_rank']
    assert a['status'] == c['status']


def test_extreme_finite_values_do_not_overflow():
    v = {'a': [1e308] * 6,
         'b': [1.0000000000000002e308] * 6,
         'c': [1.0000000000000004e308] * 6}
    r = L.arm_rank_inference(v, n_boot=500)
    assert r['status'] == 'decisive', r
    assert r['observed_winner'] == 'a', r
    assert all(math.isfinite(x) for x in r['observed_means'].values() if x is not None), r


def test_invalid_inputs_fail_closed():
    bad_calls = [
        lambda: L.arm_rank_inference({'a': [1, 2, 3], 'b': [1, 2, float('nan')],
                                      'c': [1, 2, 3]}),
        lambda: L.arm_rank_inference({'a': [1, 2, 3], 'b': [1, 2, float('inf')],
                                      'c': [1, 2, 3]}),
        lambda: L.arm_rank_inference({'a': [True, 2, 3], 'b': [1, 2, 3],
                                      'c': [1, 2, 3]}),
        lambda: L.arm_rank_inference({'a': [1, '2', 3], 'b': [1, 2, 3],
                                      'c': [1, 2, 3]}),
        lambda: L.arm_rank_inference({'a': [], 'b': [1], 'c': [1]}),
        lambda: L.arm_rank_inference({'a': [1, 2, 3], 'b': [1, 2],
                                      'c': [1, 2, 3]}),
        lambda: L.arm_rank_inference([('a', [1, 2, 3]), ('a', [1, 2, 3]),
                                      ('c', [1, 2, 3])]),
        lambda: L.arm_rank_inference({'a': {0: 1, 1: 2}, 'b': [1, 2],
                                      'c': [1, 2]}),
        lambda: L.arm_rank_inference(_clear_arms(), better='median'),
        lambda: L.arm_rank_inference(_clear_arms(), n_boot=100),
        lambda: L.arm_rank_inference(_clear_arms(), n_boot=True),
        lambda: L.arm_rank_inference(_clear_arms(), seed=-1),
        lambda: L.arm_rank_inference(_clear_arms(), rank_level=0.5,
                                     lead_level=0.7),
        lambda: L.arm_rank_inference(_clear_arms(), tie_tol=-1.0),
        lambda: L.arm_rank_inference(_clear_arms(), min_seeds=1),
    ]
    for i, call in enumerate(bad_calls):
        try:
            call()
        except ValueError:
            pass
        else:
            raise AssertionError(f'invalid arm-ranking input #{i} was accepted')


def _write_stats(td, rk):
    p = os.path.join(td, 'test_stats.json')
    with open(p, 'w', encoding='utf-8') as f:
        json.dump({'finaltest': {'arm_ranking': rk}}, f)
    return p


def test_audit_replays_and_detects_tampering():
    r = L.arm_rank_inference(_clear_arms(), n_boot=1000)
    with tempfile.TemporaryDirectory() as td:
        p = _write_stats(td, r)
        with contextlib.redirect_stdout(io.StringIO()):
            out = A.audit_arm_ranking(path=p)
        assert out['status'] == 'ok', out
        assert out['replayed'] is True and out['n_arms'] == 3, out

        bad = json.loads(json.dumps(r))
        bad['p_best']['a'] = 0.5
        bad['p_best']['b'] = 0.5
        with open(p, 'w', encoding='utf-8') as f:
            json.dump({'finaltest': {'arm_ranking': bad}}, f)
        with contextlib.redirect_stdout(io.StringIO()):
            out = A.audit_arm_ranking(path=p)
        assert out['status'] == 'fail', out
        assert any('p_best' in x for x in out['problems']), out['problems']

        bad2 = json.loads(json.dumps(r))
        bad2['rank_ci95']['a'] = [2, 2]
        with open(p, 'w', encoding='utf-8') as f:
            json.dump({'finaltest': {'arm_ranking': bad2}}, f)
        with contextlib.redirect_stdout(io.StringIO()):
            out = A.audit_arm_ranking(path=p)
        assert out['status'] == 'fail', out

        bad_del = json.loads(json.dumps(r))
        del bad_del['observed_means']
        with open(p, 'w', encoding='utf-8') as f:
            json.dump({'finaltest': {'arm_ranking': bad_del}}, f)
        with contextlib.redirect_stdout(io.StringIO()):
            out = A.audit_arm_ranking(path=p)
        assert out['status'] == 'fail', out
        assert any('missing required field' in x for x in out['problems']), out

        bad3 = json.loads(json.dumps(r))
        bad3['confirmed'] = ['a']
        with open(p, 'w', encoding='utf-8') as f:
            json.dump({'finaltest': {'arm_ranking': bad3}}, f)
        with contextlib.redirect_stdout(io.StringIO()):
            out = A.audit_arm_ranking(path=p)
        assert out['status'] == 'fail', out
        assert any('confirmatory field' in x for x in out['problems']), out


def test_audit_absent_malformed_and_not_estimable():
    with tempfile.TemporaryDirectory() as td:
        gone = os.path.join(td, 'gone.json')
        with contextlib.redirect_stdout(io.StringIO()):
            out = A.audit_arm_ranking(path=gone)
        assert out['status'] == 'not_verifiable_locally', out
        p = os.path.join(td, 'test_stats.json')
        with open(p, 'w', encoding='utf-8') as f:
            json.dump({'finaltest': {}}, f)
        with contextlib.redirect_stdout(io.StringIO()):
            out = A.audit_arm_ranking(path=p)
        assert out['status'] == 'not_verifiable_locally', out
        with open(p, 'w', encoding='utf-8') as f:
            json.dump({'finaltest': {'arm_ranking': ['not', 'an', 'object']}}, f)
        with contextlib.redirect_stdout(io.StringIO()):
            out = A.audit_arm_ranking(path=p)
        assert out['status'] == 'fail', out
        ne = L.arm_rank_inference({'a': [1, 2, 3, 4], 'b': [2, 3, 4, 5],
                                   'c': [3, 4, 5, 6]}, n_boot=500)
        with open(p, 'w', encoding='utf-8') as f:
            json.dump({'finaltest': {'arm_ranking': ne}}, f)
        with contextlib.redirect_stdout(io.StringIO()):
            out = A.audit_arm_ranking(path=p)
        # A not_estimable block that still stores its raw cells is replayable,
        # so the audit can verify the refusal instead of downgrading it.
        assert out['status'] == 'ok' and out['replayed'] is True, out
        bare = {'kind': 'arm_rank_inference', 'status': 'not_estimable',
                'reason': 'fewer than two arms'}
        with open(p, 'w', encoding='utf-8') as f:
            json.dump({'finaltest': {'arm_ranking': bare}}, f)
        with contextlib.redirect_stdout(io.StringIO()):
            out = A.audit_arm_ranking(path=p)
        assert out['status'] == 'not_applicable', out
        # A forbidden confirmatory field must not be washed out by the
        # no-raw-values downgrade.
        tainted = dict(bare, confirmed=['a'])
        with open(p, 'w', encoding='utf-8') as f:
            json.dump({'finaltest': {'arm_ranking': tainted}}, f)
        with contextlib.redirect_stdout(io.StringIO()):
            out = A.audit_arm_ranking(path=p)
        assert out['status'] == 'fail', out


def test_audit_and_report_are_wired():
    assert 'arm_ranking' in (A.__doc__ or ''), 'the audit command is not documented'
    assert "'arm_ranking': audit_arm_ranking()" in inspect.getsource(A.run_report)
    assert "'arm_ranking': audit_arm_ranking" in inspect.getsource(A)
    assert 'arm_rank_inference' in inspect.getsource(V12.finaltest_stats)
    assert '冻结臂的排序分辨率' in inspect.getsource(V12.build_report)


def _synthetic_manifest(td, arms, seeds):
    m = {'frozen_at': '2026-10-30',
         'code_semantics_expected': L.CODE_SEMANTICS,
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


def test_finaltest_stats_emits_replayable_ranking():
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
        rk = st['arm_ranking']
        assert rk['kind'] == 'arm_rank_inference', rk
        assert rk['endpoint'] == 'by_len.ppl@4096', rk
        assert rk['n_arms'] == 3 and rk['n_seeds'] == 6, rk
        assert rk['observed_winner'] == 'a_arm', rk
        with contextlib.redirect_stdout(io.StringIO()):
            out = A.audit_arm_ranking(path=p)
        assert out['status'] == 'ok', out


def test_build_report_renders_ranking_table():
    rk = L.arm_rank_inference(_clear_arms(), n_boot=1000)
    rk['endpoint'] = 'by_len.ppl@4096'
    st = {'finaltest': {
        'alpha': 0.05,
        'manifest': {'arms': list(_clear_arms()), 'seeds': list(range(8))},
        'by_length': {
            a: {'4096': {'mean': rk['observed_means'][a], 'n': 8,
                         'ppl_by_seed': rk['per_arm_seed_values'][a]}}
            for a in _clear_arms()},
        'comparisons': {}, 'holm': {}, 'arm_ranking': rk,
        'confirmed': [], 'verdicts': {},
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
    assert '冻结臂的排序分辨率' in text, text[:500]
    assert '| `a` | 8 |' in text, text
    assert '不是「各臂等效」' in text, 'the ranking table lacks the equivalence caveat'
