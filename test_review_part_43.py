"""CPU checks for the evaluation-unit sensitivity of the arm ranking.

The section-102 ranking resamples only the observed seeds.  ``arm_rank_unit_sensitivity``
adds the other sampling axis of the same fixed-arm ordering: the frozen
evaluation units (articles or line-level documents).  It reports P(best) under
seed-only, unit-only and two-way resampling, plus a seed-vs-unit variance split
of the observed top-two gap.  ``audit_gates.audit_arm_rank_units`` replays the
stored block from its own per-unit NLL/token matrices, so a hand-edited
P(best)/rank interval/status cannot survive.

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


def _payload(logps, *, tokens=None, seeds=range(6), ids=True):
    """Build a nested arm/seed per-unit payload from per-arm log-PPL vectors."""
    if tokens is None:
        tokens = [100.0] * 10
    out = {}
    for arm, per_unit in logps.items():
        out[arm] = {}
        for s in seeds:
            cell = {'nll': [v * t for v, t in zip(per_unit, tokens)],
                    'tokens': list(tokens)}
            if ids:
                cell['unit_ids'] = [f'u{i}' for i in range(len(tokens))]
            out[arm][s] = cell
    return out


def _clear_payload():
    return _payload({
        'a': [math.log(50.0)] * 10,
        'b': [math.log(51.0)] * 10,
        'c': [math.log(52.0)] * 10,
    })


def _unit_dominated_payload():
    # One large article carries 90% of the tokens.  Arm b wins the token-pooled
    # endpoint only because of that article; on the nine small articles arm a
    # is better.  A unit bootstrap must show the winner is not unit-stable.
    tokens = [10.0] * 9 + [810.0]
    a = [math.log(49.0)] * 9 + [math.log(60.0)]
    b = [math.log(50.0)] * 9 + [math.log(40.0)]
    return _payload({'a': a, 'b': b}, tokens=tokens)


def test_policy_and_statuses_are_declared():
    assert L.ARM_RANK_UNIT_POLICY['version'] == 'arm-rank-units-1'
    assert set(L.ARM_RANK_UNIT_STATUSES) == {
        'stable', 'seed_sensitive', 'unit_sensitive', 'both_sensitive',
        'undifferentiated', 'not_estimable'}
    assert L.ARM_RANK_UNIT_POLICY['role'].startswith('exploratory')


def test_clear_separation_is_stable_and_replays():
    r = L.arm_rank_unit_sensitivity(_clear_payload(), n_boot=300)
    assert r['status'] == 'stable', r
    assert r['observed_winner'] == 'a', r
    assert r['observed_rank'] == {'a': 1, 'b': 2, 'c': 3}, r
    for axis in ('seed_only', 'unit_only', 'two_way'):
        assert r[axis]['p_best']['a'] > 0.99, (axis, r[axis]['p_best'])
        assert r[axis]['rank_ci95']['a'] == [1, 1], (axis, r[axis]['rank_ci95'])
    assert r['unit_only']['p_best']['b'] == 0.0, r['unit_only']['p_best']
    assert r['per_unit_tokens'] == [100.0] * 10
    # The block is JSON-serialisable with strict NaN/inf checks and the same
    # bootstrap seed replays it exactly.
    stored = json.loads(json.dumps(r, allow_nan=False))
    again = L.arm_rank_unit_sensitivity(
        _clear_payload(), better=stored['better'], n_boot=stored['n_boot'],
        seed=stored['bootstrap_seed'], min_seeds=stored['min_seeds'],
        min_units=stored['min_units'], unit_level=stored['unit_level'],
        seed_level=stored['seed_level'], lead_level=stored['lead_level'],
        tie_tol=stored['tie_tol'])
    assert again['seed_only']['p_best'] == stored['seed_only']['p_best']
    assert again['unit_only']['rank_ci95'] == stored['unit_only']['rank_ci95']


def test_ranking_uses_the_arithmetic_mean_of_per_seed_ppl():
    # Arm a has a lower geometric mean (sqrt(10*100)=31.6) but a higher
    # arithmetic mean (55) than arm b (40).  Section 102 ranks the arithmetic
    # mean of per-seed PPLs; a geometric-mean implementation would silently
    # pick the wrong winner.
    logs = {'a': [math.log(10.0), math.log(100.0)] * 3,
            'b': [math.log(40.0)] * 6}
    p = {}
    for arm, vals in logs.items():
        p[arm] = {}
        for s, v in enumerate(vals):
            p[arm][s] = {'nll': [v * 100.0] * 10,
                         'tokens': [100.0] * 10,
                         'unit_ids': [f'u{i}' for i in range(10)]}
    seed_values = {'a': {str(s): round(math.exp(v), 6)
                         for s, v in enumerate(logs['a'])},
                   'b': {str(s): 40.0 for s in range(6)}}
    r = L.arm_rank_unit_sensitivity(p, n_boot=50,
                                    per_arm_seed_values=seed_values)
    assert r['status'] != 'not_estimable', r
    assert abs(r['observed_means']['a'] - 55.0) < 1e-6, r['observed_means']
    assert abs(r['observed_means']['b'] - 40.0) < 1e-6, r['observed_means']
    assert r['observed_winner'] == 'b', r
    assert r['observed_rank'] == {'a': 2, 'b': 1}, r


def test_unit_dominance_makes_the_ranking_unit_sensitive():
    r = L.arm_rank_unit_sensitivity(_unit_dominated_payload(), n_boot=400)
    assert r['observed_winner'] == 'b', r
    assert r['seed_only']['p_best']['b'] > 0.99, r['seed_only']['p_best']
    # The same seeds always prefer b because the big article dominates the
    # token-pooled endpoint; dropping/resampling units can hand the win to a.
    assert r['unit_only']['p_best']['b'] < 0.90, r['unit_only']['p_best']
    assert r['unit_only']['p_best']['a'] > 0.10, r['unit_only']['p_best']
    assert r['status'] == 'unit_sensitive', r
    assert 'unit' in r['status_reason'], r['status_reason']
    assert r['gap_decomposition']['dominant_axis'] == 'unit', r['gap_decomposition']
    assert r['gap_decomposition']['seed_variance_share'] == 0.0, r['gap_decomposition']
    assert r['gap_decomposition']['unit_variance_share'] == 1.0, r['gap_decomposition']


def test_endpoint_cross_check_refuses_a_different_estimand():
    p = _clear_payload()
    wrong = {'a': {str(s): 50.5 for s in range(6)},
             'b': {str(s): 51.0 for s in range(6)},
             'c': {str(s): 52.0 for s in range(6)}}
    r = L.arm_rank_unit_sensitivity(p, n_boot=100, per_arm_seed_values=wrong)
    assert r['status'] == 'not_estimable', r
    assert 'do not reproduce' in r['reason'], r['reason']
    ok = {'a': {str(s): 50.0 for s in range(6)},
          'b': {str(s): 51.0 for s in range(6)},
          'c': {str(s): 52.0 for s in range(6)}}
    r2 = L.arm_rank_unit_sensitivity(p, n_boot=100, per_arm_seed_values=ok)
    assert r2['status'] == 'stable', r2
    assert r2['endpoint_values_match'] == 'checked', r2
    assert abs(r2['per_arm_seed_values']['a']['0'] - 50.0) < 1e-9, r2['per_arm_seed_values']


def test_malformed_inputs_fail_closed():
    p = _clear_payload()
    bad = []
    bad.append(None)
    bad.append([])
    bad.append({})
    bad.append({'a': {}})
    bad.append({'a': {0: {'nll': [1.0], 'tokens': [1.0]}, 'x': {}}})
    bad.append({'a': {0: {'nll': [1.0], 'tokens': [1.0]}, 1: {'nll': [1.0], 'tokens': [1.0]}},
                'b': {0: 'not-a-cell'}})
    bad.append({'a': {0: {'nll': 'x', 'tokens': [1.0]}},
                'b': {0: {'nll': [1.0], 'tokens': [1.0]}}})
    bad.append({'a': {0: {'nll': ['1.0'], 'tokens': [1.0]}},
                'b': {0: {'nll': [1.0], 'tokens': [1.0]}}})
    bad.append({'a': {0: {'nll': [1.0], 'tokens': [True]}},
                'b': {0: {'nll': [1.0], 'tokens': [1.0]}}})
    bad.append({'a': {0: {'nll': [1.0], 'tokens': [0.0]}},
                'b': {0: {'nll': [1.0], 'tokens': [1.0]}}})
    bad.append({'a': {0: {'nll': [1.0, 2.0], 'tokens': [1.0]}},
                'b': {0: {'nll': [1.0], 'tokens': [1.0]}}})
    bad.append({'a': {0: {'nll': [1.0], 'tokens': [1.0], 'unit_ids': ['x', 'y']}},
                'b': {0: {'nll': [1.0], 'tokens': [1.0]}}})
    bad.append({'a': {0: {'ppl': [-1.0], 'tokens': [1.0]}},
                'b': {0: {'ppl': [1.0], 'tokens': [1.0]}}})
    for item in bad:
        try:
            L.arm_rank_unit_sensitivity(item, n_boot=10)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError(f'malformed payload was accepted: {item!r}')
    for kwargs in ({'better': 'median'}, {'n_boot': 0}, {'min_units': 1},
                   {'unit_level': 1.5}, {'seed_level': -0.1},
                   {'tie_tol': -1.0}, {'n_boot': 10.5}):
        try:
            L.arm_rank_unit_sensitivity(p, n_boot=10, **kwargs)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError(f'malformed parameter was accepted: {kwargs!r}')


def test_alignment_and_coverage_refusals():
    base = _clear_payload()
    # Ragged unit counts.
    bad = json.loads(json.dumps(base))
    bad['b']['0']['nll'] = bad['b']['0']['nll'][:-1]
    bad['b']['0']['tokens'] = bad['b']['0']['tokens'][:-1]
    bad['b']['0'].pop('unit_ids')
    r = L.arm_rank_unit_sensitivity(bad, n_boot=10)
    assert r['status'] == 'not_estimable', r
    # Different per-unit token counts.
    bad = json.loads(json.dumps(base))
    bad['b']['0']['tokens'] = [101.0] * 10
    r = L.arm_rank_unit_sensitivity(bad, n_boot=10)
    assert r['status'] == 'not_estimable', r
    assert 'token counts' in r['reason'], r['reason']
    # Different unit identity order.
    bad = json.loads(json.dumps(base))
    bad['b']['0']['unit_ids'] = list(reversed(bad['b']['0']['unit_ids']))
    r = L.arm_rank_unit_sensitivity(bad, n_boot=10)
    assert r['status'] == 'not_estimable', r
    # Unit ids present for only part of the panel.
    bad = json.loads(json.dumps(base))
    bad['b']['0'].pop('unit_ids')
    r = L.arm_rank_unit_sensitivity(bad, n_boot=10)
    assert r['status'] == 'not_estimable', r
    # Insufficient seeds/units.
    r = L.arm_rank_unit_sensitivity(_payload({'a': [1.0] * 10, 'b': [2.0] * 10},
                                             seeds=range(4)), n_boot=10)
    assert r['status'] == 'not_estimable' and 'seed' in r['reason'], r
    r = L.arm_rank_unit_sensitivity(
        _payload({'a': [1.0] * 4, 'b': [2.0] * 4}, tokens=[100.0] * 4),
        n_boot=10)
    assert r['status'] == 'not_estimable' and 'unit' in r['reason'], r


def test_seed_intersection_is_reported_not_silently_dropped():
    def cell(v):
        return {'nll': [v * 100.0] * 10, 'tokens': [100.0] * 10}
    p = {'a': {s: cell(math.log(50.0)) for s in range(6)},
         'b': {s: cell(math.log(51.0)) for s in range(1, 7)}}
    r = L.arm_rank_unit_sensitivity(p, n_boot=100)
    assert r['status'] == 'stable', r
    assert r['seeds'] == [1, 2, 3, 4, 5], r['seeds']
    assert r['dropped_seeds'] == {'a': [0], 'b': [6]}, r['dropped_seeds']
    assert r['complete_seed_coverage'] is False, r


def test_max_direction_ranks_the_largest_arm_first():
    p = _payload({'a': [math.log(50.0)] * 10,
                  'b': [math.log(51.0)] * 10,
                  'c': [math.log(52.0)] * 10})
    r = L.arm_rank_unit_sensitivity(p, better='max', n_boot=100)
    assert r['status'] == 'stable', r
    assert r['observed_winner'] == 'c', r
    assert r['observed_rank'] == {'a': 3, 'b': 2, 'c': 1}, r
    assert r['unit_only']['p_best']['c'] > 0.99, r['unit_only']['p_best']
    assert r['observed_gap'] == r['observed_gap'] and r['observed_gap'] > 0.0, r


def test_extreme_finite_values_do_not_overflow_or_emit_nan():
    tokens = [1e300] * 8
    p = _payload({'a': [300.0] * 8, 'b': [300.0] * 8}, tokens=tokens, ids=False)
    r = L.arm_rank_unit_sensitivity(p, n_boot=50)
    assert r['status'] in ('undifferentiated', 'not_estimable'), r
    if r['status'] != 'not_estimable':
        assert all(v is None or math.isfinite(float(v))
                   for v in r['observed_means'].values()), r['observed_means']
        json.dumps(r, allow_nan=False)


def _write_block(td, block):
    p = os.path.join(td, 'stats.json')
    with open(p, 'w', encoding='utf-8') as f:
        json.dump({'finaltest': {'arm_rank_units': block}}, f)
    return p


def test_audit_replays_a_clean_block_and_detects_tampering():
    r = L.arm_rank_unit_sensitivity(_unit_dominated_payload(), n_boot=200)
    with tempfile.TemporaryDirectory() as td:
        p = _write_block(td, r)
        with contextlib.redirect_stdout(io.StringIO()):
            out = A.audit_arm_rank_units(path=p)
        assert out['status'] == 'ok', out
        assert out['replayed'] is True and out['n_units'] == 10, out
        bad = json.loads(json.dumps(r))
        bad['unit_only']['p_best']['a'] = 0.99
        with contextlib.redirect_stdout(io.StringIO()):
            out = A.audit_arm_rank_units(path=_write_block(td, bad))
        assert out['status'] == 'fail', out
        assert any('unit_only' in x for x in out['problems']), out['problems']
        tainted = json.loads(json.dumps(r))
        tainted['confirmed'] = ['a']
        with contextlib.redirect_stdout(io.StringIO()):
            out = A.audit_arm_rank_units(path=_write_block(td, tainted))
        assert out['status'] == 'fail', out
        assert any('confirmatory' in x for x in out['problems']), out['problems']
        with contextlib.redirect_stdout(io.StringIO()):
            out = A.audit_arm_rank_units(path=os.path.join(td, 'gone.json'))
        assert out['status'] == 'not_verifiable_locally', out
        bare = {'kind': 'arm_rank_unit_sensitivity', 'status': 'not_estimable',
                'reason': 'not enough shared units'}
        with contextlib.redirect_stdout(io.StringIO()):
            out = A.audit_arm_rank_units(path=_write_block(td, bare))
        assert out['status'] == 'not_applicable', out


def test_audit_is_wired_into_report_and_cli():
    assert 'arm_rank_units' in (A.__doc__ or '')
    assert "'arm_rank_units': audit_arm_rank_units()" in inspect.getsource(A.run_report)
    assert "'arm_rank_units': audit_arm_rank_units" in inspect.getsource(A)
    cli = inspect.getsource(A._run_cli)
    for alias in ('arm_rank_units', 'ranking_units', 'unit_ranking'):
        assert alias in cli, alias


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


def _synthetic_record(arm, seed, ppl, msha, *, n_units=10):
    tokens = [100.0] * n_units
    per_unit_ppl = [ppl + 0.01 * i for i in range(n_units)]
    per_unit_nll = [math.log(v) * t for v, t in zip(per_unit_ppl, tokens)]
    pooled = math.exp(sum(per_unit_nll) / sum(tokens))
    return {'variant': arm, 'seed': seed, 'split': 'test', 'data_fp': 'FP',
            'by_len': {'4096': {
                'ppl': pooled, 'n_tok': int(sum(tokens)),
                'nll_sum': sum(per_unit_nll), 'truncated': False,
                'n_seq': n_units,
                'ppl_per_article': per_unit_ppl,
                'n_tokens_per_article': tokens,
                'nll_per_article': per_unit_nll,
                'article_ids': [f'a{i}' for i in range(n_units)]}},
            '_code': L.code_stamp(), 'manifest_sha256': msha}


def test_finaltest_stats_emits_a_replayable_unit_block():
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
        ru = st['arm_rank_units']
        assert ru['kind'] == 'arm_rank_unit_sensitivity', ru
        assert ru['status'] == 'stable', ru
        assert ru['unit_granularity'] == 'article', ru
        assert ru['n_units'] == 10 and ru['n_seeds'] == 6, ru
        assert ru['endpoint'] == 'by_len.ppl@4096', ru
        assert ru['observed_winner'] == 'a_arm', ru
        with contextlib.redirect_stdout(io.StringIO()):
            out = A.audit_arm_rank_units(path=p)
        assert out['status'] == 'ok', out


def test_build_report_renders_the_unit_sensitivity_section():
    ru = L.arm_rank_unit_sensitivity(_unit_dominated_payload(), n_boot=200)
    ru['endpoint'] = 'by_len.ppl@4096'
    ru['unit_granularity'] = 'article'
    rk = L.arm_rank_inference(
        {'a': [58.8] * 6, 'b': [40.9] * 6}, n_boot=200)
    rk['endpoint'] = 'by_len.ppl@4096'
    st = {'finaltest': {
        'alpha': 0.05,
        'manifest': {'arms': ['a', 'b'], 'seeds': list(range(6))},
        'by_length': {
            a: {'4096': {'mean': rk['observed_means'][a], 'n': 6,
                         'ppl_by_seed': rk['per_arm_seed_values'][a]}}
            for a in ('a', 'b')},
        'comparisons': {}, 'holm': {}, 'arm_ranking': rk,
        'arm_rank_units': ru, 'confirmed': [], 'verdicts': {},
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
    assert '排序的评测单元敏感性' in text, text[:800]
    assert '`unit_sensitive`' in text, text
    assert '评测单元：10 个' in text, text
    assert '不是新的显著性检验' in text, text
