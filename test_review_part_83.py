"""Tests for the distribution-free exact sign-flip TOST companion.

The new readout is deliberately descriptive: it must reproduce the exact
one-sided sign-flip p-values, expose the discrete resolution floor, and never
move the historical paired-t TOST decision or any frozen verdict.
"""

import inspect
import json
import math
import os

import numpy as np
import pytest

import audit_gates as A
import exp_lib as L


def _brute_force_tost(deltas, margin, conf=0.90):
    """Independent 2**n enumeration of both one-sided shifted tests."""
    d = np.asarray(deltas, dtype=float)
    n = int(d.size)
    alpha = (1.0 - float(conf)) / 2.0
    low_shift = d + float(margin)
    high_shift = float(margin) - d
    low_obs = float(np.sum(low_shift))
    high_obs = float(np.sum(high_shift))
    low_sums, high_sums = [], []
    for mask in range(1 << n):
        signs = np.asarray([1.0 if (mask >> i) & 1 else -1.0
                            for i in range(n)])
        low_sums.append(float(np.sum(signs * low_shift)))
        high_sums.append(float(np.sum(signs * high_shift)))
    low_sums = np.asarray(low_sums)
    high_sums = np.asarray(high_sums)
    tol = 1e-12 * n
    p_low = float(np.mean(low_sums >= low_obs - tol))
    p_high = float(np.mean(high_sums >= high_obs - tol))
    return {
        'p_low': p_low,
        'p_high': p_high,
        'p_tost': max(p_low, p_high),
        'equivalent': bool(max(p_low, p_high) < alpha),
    }


def test_exact_tost_matches_independent_enumeration():
    rng = np.random.default_rng(20261101)
    for n, margin in ((5, 0.4), (7, 0.8), (8, 1.5)):
        d = rng.normal(0.0, 0.5, size=n)
        got = L.exact_signflip_tost(d.tolist(), margin, conf=0.90)
        want = _brute_force_tost(d.tolist(), margin, conf=0.90)
        assert got['status'] == 'ok'
        assert got['p_low'] == pytest.approx(want['p_low'], abs=1e-15)
        assert got['p_high'] == pytest.approx(want['p_high'], abs=1e-15)
        assert got['p_tost'] == pytest.approx(want['p_tost'], abs=1e-15)
        assert got['equivalent'] is want['equivalent']
        assert got['method'] == 'exact'


def test_exact_tost_clear_equivalence_and_non_equivalence():
    inside = [0.001, -0.002, 0.0005, -0.0009, 0.0012, -0.0003]
    r = L.exact_signflip_tost(inside, margin=0.01)
    assert r['status'] == 'ok'
    assert r['p_tost'] == pytest.approx(2.0 ** -6)
    assert r['equivalent'] is True
    assert r['resolution_limited'] is False

    above = [0.05, 0.052, 0.049, 0.051, 0.05, 0.048]
    r2 = L.exact_signflip_tost(above, margin=0.01)
    assert r2['p_tost'] == pytest.approx(1.0)
    assert r2['equivalent'] is False

    far = [0.9, 0.95, 0.85, 1.0, 0.92, 0.88]
    r3 = L.exact_signflip_tost(far, margin=0.1)
    assert r3['p_tost'] == pytest.approx(1.0)
    assert r3['equivalent'] is False


def test_exact_tost_boundary_is_never_equivalent():
    # A delta exactly on the margin makes one shifted arm fully degenerate
    # (all entries zero): every sign flip ties, so that arm's one-sided p is
    # identically 1.0 and its attainable floor is 1.0 (the zero-aware floor
    # of test_review_part_84, taken to its n_nonzero=0 limit).  The panel is
    # therefore resolution-limited, and equivalence is never declared at the
    # exact boundary.
    for d in ([0.1] * 6, [-0.1] * 6):
        r = L.exact_signflip_tost(d, margin=0.1)
        assert r['p_tost'] == pytest.approx(1.0), r
        assert r['equivalent'] is False, r
        assert r['resolution_limited'] is True, r
        assert r['p_floor'] == pytest.approx(1.0), r
        assert sorted((r['n_nonzero_low'], r['n_nonzero_high'])) == [0, 6], r


def test_exact_tost_discrete_resolution_floor():
    # A zero effect is the best case for equivalence; with n=4 the one-sided
    # sign-flip floor is 1/16 = 0.0625 > 0.05, so equivalence is impossible.
    n4 = L.exact_signflip_tost([0.0] * 4, margin=0.1)
    assert n4['p_tost'] == pytest.approx(1.0 / 16.0)
    assert n4['p_floor_by_n'] == pytest.approx(1.0 / 16.0)
    assert n4['resolution_limited'] is True
    assert n4['equivalent'] is False
    assert n4['min_n_for_alpha'] == 5
    # n=5 is the first n whose 1/2**n floor is below 0.05.
    n5 = L.exact_signflip_tost([0.0] * 5, margin=0.1)
    assert n5['p_tost'] == pytest.approx(1.0 / 32.0)
    assert n5['resolution_limited'] is False
    assert n5['equivalent'] is True
    # A stricter 95% two-sided TOST needs 1/2**n < 0.025, i.e. n >= 6.
    n5_95 = L.exact_signflip_tost([0.0] * 5, margin=0.1, conf=0.95)
    assert n5_95['alpha_one_sided'] == pytest.approx(0.025)
    assert n5_95['p_floor_by_n'] == pytest.approx(1.0 / 32.0)
    assert n5_95['resolution_limited'] is True
    assert n5_95['equivalent'] is False
    assert n5_95['min_n_for_alpha'] == 6
    # At an alpha that is exactly 1/2**k, strict p < alpha needs the next n.
    pow2 = L.exact_signflip_tost([0.0] * 2, margin=0.1, conf=0.5)
    assert pow2['alpha_one_sided'] == pytest.approx(0.25)
    assert pow2['min_n_for_alpha'] == 3
    assert pow2['resolution_limited'] is True
    assert pow2['equivalent'] is False


def test_exact_tost_reveals_degenerate_t_equivalence():
    # With two identical, tiny deltas the t limit returns p=0 and calls the
    # panel equivalent; the exact test can only reach 1/4 and must not.
    d = [0.001, 0.001]
    t = L.tost_paired(d, margin=0.1)
    assert t['equivalent'] is True
    ex = t['distribution_free']
    assert ex['equivalent'] is False
    assert ex['p_tost'] == pytest.approx(0.25)
    assert ex['resolution_limited'] is True
    assert ex['t_equivalent'] is True
    assert ex['agreement'] is False


def test_exact_tost_agrees_with_t_on_resolved_panel():
    d = [0.001] * 5
    t = L.tost_paired(d, margin=0.1)
    assert t['equivalent'] is True
    assert t['distribution_free']['equivalent'] is True
    assert t['distribution_free']['p_tost'] == pytest.approx(1.0 / 32.0)
    assert t['distribution_free']['agreement'] is True


def test_exact_tost_scale_invariance_and_extremes():
    d = [0.001, -0.002, 0.0005, -0.0009, 0.0012, -0.0003]
    base = L.exact_signflip_tost(d, margin=0.01)
    for factor in (1e-120, 1e120, 1e-200, 1e200):
        scaled = L.exact_signflip_tost([x * factor for x in d],
                                       margin=0.01 * factor)
        assert scaled['status'] == 'ok'
        assert scaled['p_tost'] == pytest.approx(base['p_tost'], abs=1e-15)
        assert scaled['equivalent'] is base['equivalent']
    # A finite panel near the float limit must not overflow when shifted.
    extreme = L.exact_signflip_tost([1e308, 1.5e308], margin=1e308)
    assert extreme['status'] == 'ok'
    assert math.isfinite(extreme['p_tost'])
    assert extreme['equivalent'] is False


def test_exact_tost_nonfinite_drop_and_invalid_inputs():
    r = L.exact_signflip_tost(
        [0.001, float('nan'), -0.002, float('inf'), 0.0005, -0.0009],
        margin=0.01)
    assert r['status'] == 'ok'
    assert r['n'] == 4
    assert r['n_dropped'] == 2
    assert math.isfinite(r['p_tost'])
    for margin, conf in ((0.0, 0.90), (float('nan'), 0.90),
                         (float('inf'), 0.90), (0.1, 0.0), (0.1, 1.0),
                         (0.1, float('nan'))):
        r = L.exact_signflip_tost([0.001, -0.002, 0.0005],
                                  margin=margin, conf=conf)
        assert r['status'] == 'not_estimable', ((margin, conf), r)
        assert r['equivalent'] is None
        assert r['reason']
    short = L.exact_signflip_tost([0.001], margin=0.1)
    assert short['status'] == 'not_estimable'
    with pytest.raises(ValueError):
        L.exact_signflip_tost('abc', margin=0.1)


def test_exact_tost_artifact_roundtrip_and_tamper_rejection():
    d = [0.001, -0.002, 0.0005, -0.0009, 0.0012, -0.0003]
    art = L.build_exact_tost(d, margin=0.01, created_at='2026-11-01T00:00:00')
    assert art['kind'] == 'exact_tost_artifact'
    assert len(art['content_sha256']) == 64
    v = L.validate_exact_tost(art)
    assert v['status'] == 'ok' and not v['problems'], v
    # Same inputs and timestamp => identical content identity.
    art2 = L.build_exact_tost(d, margin=0.01,
                              created_at='2026-11-01T00:00:00')
    assert art2['content_sha256'] == art['content_sha256']
    tampered = json.loads(json.dumps(art))
    tampered['result']['p_tost'] = 0.5
    assert L.validate_exact_tost(tampered)['status'] == 'fail'
    no_hash = json.loads(json.dumps(art))
    del no_hash['content_sha256']
    assert L.validate_exact_tost(no_hash)['status'] == 'fail'
    stolen = json.loads(json.dumps(art))
    stolen['result']['p_holm'] = 0.01
    assert L.validate_exact_tost(stolen)['status'] == 'fail'
    extra = json.loads(json.dumps(art))
    extra['inputs']['alpha'] = 0.05
    assert L.validate_exact_tost(extra)['status'] == 'fail'


def test_exact_tost_builder_rejects_non_replayable_inputs():
    for args in (([0.001, float('nan')], 0.1),
                 ([0.001], 0.1),
                 ([0.001, 0.002], 0.0),
                 ([0.001, 0.002], float('inf'))):
        with pytest.raises(ValueError):
            L.build_exact_tost(*args)
    with pytest.raises(ValueError):
        L.build_exact_tost([0.001, 0.002], 0.1, conf=1.0)
    with pytest.raises(ValueError):
        L.build_exact_tost([0.001, 0.002], 0.1, conf=float('nan'))


def test_exact_tost_artifact_gate_and_stats_scan(tmp_path):
    d = [0.001, -0.002, 0.0005, -0.0009, 0.0012, -0.0003]
    art = L.build_exact_tost(d, margin=0.01)
    p = tmp_path / 'exact_tost.json'
    L.write_exact_tost(str(p), art)
    ok = A.audit_exact_tost(str(p), verbose=False)
    assert ok['status'] == 'ok', ok
    assert ok['mode'] == 'artifact'
    bad = json.loads(json.dumps(art))
    bad['result']['equivalent'] = False
    q = tmp_path / 'bad.json'
    q.write_text(json.dumps(bad), encoding='utf-8')
    assert A.audit_exact_tost(str(q), verbose=False)['status'] == 'fail'
    missing = A.audit_exact_tost(str(tmp_path / 'none.json'), verbose=False)
    assert missing['status'] == 'not_verifiable_locally'

    # Nested stats scan: build the same public shape the final-test artifact
    # uses and check both a clean companion and a tampered one.
    row = L.exact_signflip_tost(d, margin=0.01)
    row['t_equivalent'] = True
    row['agreement'] = True
    stats = {
        'comparisons': {
            'c0': {
                'equivalent': True,
                'practical_significance': {
                    'status': 'equivalent_within_sesoi',
                    'margin': 0.01,
                    'tost_conf': 0.90,
                    'per_seed_log_ratio': d,
                    'distribution_free': row,
                },
            },
        },
        'practical_significance': {
            'tost_conf': 0.90,
            'comparisons': {
                'c0': {
                    'status': 'equivalent_within_sesoi',
                    'margin': 0.01,
                    'tost_conf': 0.90,
                    'per_seed_log_ratio': d,
                    'distribution_free': row,
                },
            },
        },
    }
    sp = tmp_path / 'test_stats.json'
    sp.write_text(json.dumps(stats), encoding='utf-8')
    scanned = A.audit_exact_tost(str(sp), verbose=False)
    assert scanned['status'] == 'ok', scanned
    assert scanned['n_replayed'] == 1
    bad_stats = json.loads(json.dumps(stats))
    bad_stats['comparisons']['c0']['practical_significance'][
        'distribution_free']['p_tost'] = 0.5
    bp = tmp_path / 'bad_stats.json'
    bp.write_text(json.dumps(bad_stats), encoding='utf-8')
    assert A.audit_exact_tost(str(bp), verbose=False)['status'] == 'fail'
    legacy = json.loads(json.dumps(stats))
    del legacy['comparisons']['c0']['practical_significance'][
        'distribution_free']
    del legacy['practical_significance']['comparisons']['c0'][
        'distribution_free']
    lp = tmp_path / 'legacy.json'
    lp.write_text(json.dumps(legacy), encoding='utf-8')
    old = A.audit_exact_tost(str(lp), verbose=False)
    assert old['status'] in ('warnings', 'not_verifiable_locally'), old


def test_exact_tost_gate_warns_on_t_only_equivalence(tmp_path):
    d = [0.001, 0.001]
    t = L.tost_paired(d, margin=0.1)
    assert t['equivalent'] is True
    df = dict(t['distribution_free'])
    stats = {
        'comparisons': {
            'c0': {
                'equivalent': True,
                'practical_significance': {
                    'status': 'equivalent_within_sesoi',
                    'margin': 0.1,
                    'tost_conf': 0.90,
                    'per_seed_log_ratio': d,
                    'equivalent': True,
                    'distribution_free': df,
                },
            },
        },
    }
    p = tmp_path / 'degenerate_stats.json'
    p.write_text(json.dumps(stats), encoding='utf-8')
    out = A.audit_exact_tost(str(p), verbose=False)
    assert out['status'] == 'warnings', out
    assert any('t-TOST' in w for w in out['warnings']), out
    assert any('resolution floor' in w for w in out['warnings']), out


def test_exact_tost_gate_scans_zero_effect_rows(tmp_path):
    p = L.confirmation_practical_significance([0.0] * 4, margin=0.1,
                                              alpha=0.05)
    stats = {
        'comparisons': {
            'zero': {
                'equivalent': None,
                'practical_significance': p,
            },
        },
    }
    path = tmp_path / 'zero_stats.json'
    path.write_text(json.dumps(stats), encoding='utf-8')
    out = A.audit_exact_tost(str(path), verbose=False)
    assert out['n_replayed'] == 1, out
    assert out['status'] == 'warnings', out
    assert out['comparisons']['zero']['equivalent'] is False
    assert any('resolution floor' in w for w in out['warnings']), out


def test_exact_tost_practical_significance_companion():
    d = [0.001, -0.002, 0.0005, -0.0009, 0.0012, -0.0003]
    r = L.confirmation_practical_significance(
        d, margin=0.01, alpha=0.05, expected_sign='negative')
    assert r['status'] == 'equivalent_within_sesoi'
    assert r['equivalent'] is True
    assert r['distribution_free']['equivalent'] is True
    assert r['distribution_free']['agreement'] is True
    # A zero-effect n=4 panel is not evidence of exact equivalence even
    # though the t limit is degenerate; the main zero_effect status stays.
    z = L.confirmation_practical_significance([0.0] * 4, margin=0.1,
                                              alpha=0.05)
    assert z['status'] == 'zero_effect'
    assert z['equivalent'] is None
    assert z['distribution_free']['equivalent'] is False
    assert z['distribution_free']['resolution_limited'] is True


def test_exact_tost_markdown_is_reported():
    import explib.confirmation as C
    d = [0.001, -0.002, 0.0005, -0.0009, 0.0012, -0.0003]
    p = L.confirmation_practical_significance(
        d, margin=0.01, alpha=0.05, expected_sign='negative')
    block = {
        'status': 'ok', 'alpha': 0.05, 'tost_conf': 0.95,
        'family_names': ['c0'], 'policy': {'source': 'manifest'},
        'comparisons': {'c0': dict(p, p_tost_holm=0.01,
                                   p_beyond_holm=0.2)},
        'summary': {},
    }
    md = '\n'.join(C._practical_significance_markdown(block))
    assert '精确符号翻转 TOST' in md
    assert '分辨率' in md
    assert '不替换预注册的 t 判定' in md


def test_exact_tost_registration_and_entry_points():
    import explib
    import _audit_gates
    assert 'part_77_exact_tost.py' in explib._PARTS
    assert 'part_43_audit_exact_tost.py' in _audit_gates._PARTS
    assert hasattr(L, 'exact_signflip_tost')
    assert hasattr(L, 'build_exact_tost')
    assert hasattr(L, 'validate_exact_tost')
    assert hasattr(A, 'audit_exact_tost')
    assert hasattr(A, 'audit_exact_tost_artifact')
    text = open('Makefile', encoding='utf-8').read()
    assert 'exact-tost:' in text and 'audit_gates.py exact_tost' in text
    assert 'exact_tost' in inspect.getsource(A.run_report)
    assert os.path.isdir('.github')
    assert os.path.isfile('README.md')
