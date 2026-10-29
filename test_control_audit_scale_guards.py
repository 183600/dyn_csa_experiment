"""Regression checks for control-ladder audits and extreme-scale PPL spreads.

These are zero-GPU design checks.  They cover two failure modes that ordinary
design fixtures do not exercise:

* a design audit must not mutate the process RNG stream (a later training cell
  would otherwise inherit a different data/dropout order), and must fail closed
  on malformed nested declarations instead of raising ``TypeError``;
* a finite PPL panel can be extreme enough that naive ``np.std``, ``np.mean``
  or an even-count median overflows even though the reported statistic is
  representable.  The analysis helpers therefore use the same guarded
  mean/SD/quantile paths, and a bootstrap runs on a common unit scale before
  mapping its endpoints back.
"""
import copy
import math

import numpy as np
import pytest
import torch

import exp_lib as L


def test_sink_audit_preserves_rng_state():
    torch.manual_seed(20261029)
    before = torch.get_rng_state().clone()
    report = L.sink_control_audit()
    assert report['status'] == 'ok', report['violations']
    assert torch.equal(before, torch.get_rng_state())


def test_control_audits_validate_declared_modes():
    sink = copy.deepcopy(L.SINK_CONTROL_DESIGN)
    sink['levels']['learned']['sink_mode'] = 'random'
    assert L.sink_control_audit(ladder=sink)['status'] == 'fail'
    sink = copy.deepcopy(L.SINK_CONTROL_DESIGN)
    sink['levels']['absent']['sink_mode'] = 'random'
    assert L.sink_control_audit(ladder=sink)['status'] == 'fail'
    content = copy.deepcopy(L.CONTENT_CONTROL_DESIGN)
    content['levels']['aligned']['content_mode'] = 'zero'
    assert L.content_control_audit(ladder=content)['status'] == 'fail'


def test_control_audits_fail_closed_on_nested_malformations():
    sink = copy.deepcopy(L.SINK_CONTROL_DESIGN)
    sink['absence_keys']['fixed'] = 1.5
    assert L.sink_control_audit(ladder=sink)['status'] == 'fail'
    sink = copy.deepcopy(L.SINK_CONTROL_DESIGN)
    sink['levels']['randomized']['arms']['fixed'] = {'not': 'an arm id'}
    assert L.sink_control_audit(ladder=sink)['status'] == 'fail'
    content = copy.deepcopy(L.CONTENT_CONTROL_DESIGN)
    content['levels']['misaligned']['arms']['fixed'] = ['not', 'an arm id']
    assert L.content_control_audit(ladder=content)['status'] == 'fail'


def test_extreme_finite_ppl_spread_stays_finite():
    ppls = [1.0e200, 1.1e200, 0.9e200]
    ci = L._seed_t_ci(ppls)
    assert ci is not None and all(np.isfinite(ci))
    assert np.isfinite(L._scaled_std_or_raw(ppls))
    # Ordinary magnitudes keep the historical numpy value exactly.
    ordinary = [1.0, 2.0, 3.0]
    assert L._scaled_std_or_raw(ordinary) == float(np.std(ordinary, ddof=1))


def test_extreme_finite_ppl_mean_and_median_stay_finite():
    # Three seeds at 1e308: the true mean is representable, but np.mean sums
    # them first and overflows to inf.  A two-seed even median has the same
    # failure mode through (a + b) / 2.
    assert L._scaled_mean_or_raw([1e308, 1e308, 1e308]) == 1e308
    assert math.isclose(L._scaled_mean_or_raw([1e308, 1.5e308]), 1.25e308,
                        rel_tol=1e-15)
    assert math.isclose(
        L._scaled_median_or_raw(np.sort(np.array([1e308, 1.5e308]))),
        1.25e308, rel_tol=1e-15)
    q1, q3 = L._scaled_percentile_or_raw(
        np.sort(np.array([1e308, 1.5e308])), (25, 75))
    assert np.isfinite(q1) and np.isfinite(q3)

    ordinary = [1.0, 2.0, 3.0]
    assert L._scaled_mean_or_raw(ordinary) == float(np.mean(ordinary))
    assert L._scaled_median_or_raw(np.sort(np.array(ordinary))) == float(
        np.median(np.sort(np.array(ordinary))))


def test_aggregate_extreme_ppl_panel_keeps_finite_mean_and_median():
    recs = {
        f'a::seed{s}': {'variant': 'a', 'seed': s, 'ppl': ppl,
                        'run_cfg': 'cfg'}
        for s, ppl in enumerate([1e308, 1.5e308])
    }
    agg = L.aggregate(recs)
    entry = next(v for v in agg.values()
                 if isinstance(v, dict) and 'ppl_mean' in v)
    for key in ('ppl_mean', 'ppl_mean_arithmetic', 'ppl_median', 'ppl_primary'):
        assert np.isfinite(entry[key]), (key, entry[key])
    assert tuple(entry['ppl_iqr']) == tuple(
        L._scaled_percentile_or_raw(np.sort(np.array([1e308, 1.5e308])),
                                    (25, 75)))


def test_unpaired_bootstrap_rescales_extreme_representable_panels():
    # Every arm value is finite and the true difference (1e308) is
    # representable, but np.mean's per-draw sum overflows to inf.  A common
    # unit-scale bootstrap must recover the finite interval.
    ci = L.unpaired_bootstrap_ci([1e308] * 3, [0.0] * 3, n_boot=64)
    assert ci is not None and all(np.isfinite(ci)), ci
    assert ci == [1e308, 1e308]
    # Ordinary magnitudes keep the historical path (and stay finite).
    assert L.unpaired_bootstrap_ci([1, 2, 3], [0, 1, 2], n_boot=64) is not None


def test_extreme_interval_is_unavailable_not_infinite():
    # The width is mathematically beyond float64 here; emit "unavailable"
    # instead of an infinite bound that a strict JSON writer turns into null
    # only after the in-memory consumer has already accepted it.
    assert L._seed_t_ci([1e308, 1.5e308]) is None
    stats = L.paired_effect_stats([1e308, 1.5e308])
    assert not np.isinf(stats['ci95_low'])
    assert not np.isinf(stats['ci95_high'])
    assert np.isnan(stats['ci95_low']) and np.isnan(stats['ci95_high'])


def test_tost_extreme_finite_panel_does_not_emit_infinite_bounds():
    # The t statistic and its interval are scale invariant.  A finite panel
    # near the float limit can have a representable mean/p-value but an
    # endpoint beyond float64; that interval must be withheld, never reported
    # as an infinite bound.  A tiny but finite panel keeps a real interval.
    r = L.tost_paired([1e308, 1.5e308], margin=1e308)
    assert math.isfinite(r['mean']) and math.isfinite(r['p_tost']), r
    assert r['ci'] is None or all(math.isfinite(v) for v in r['ci']), r
    t = L.tost_paired([1e-200, -2e-200, 3e-200], margin=2e-200)
    assert t['ci'] is not None and all(math.isfinite(v) for v in t['ci']), t
    assert t['ci'][0] < t['ci'][1] and t['p_tost'] > 0.0, t


def test_extreme_finite_scale_analysis_withholds_unrepresentable_variance():
    base = [{'label': f'S{i}', 'scale': 10.0 ** i, 'seeds': [0, 1, 2],
             'effects': [0.1 * i + 0.01 * j for j in range(3)]}
            for i in range(3)]
    factor = 1e200
    scaled = [dict(p, effects=[e * factor for e in p['effects']])
              for p in base]
    meta = L.meta_analyze_effects(scaled, min_seeds=2)
    assert meta['status'] == 'ok', meta
    assert meta['heterogeneity']['tau2'] is None, meta['heterogeneity']
    assert meta['heterogeneity']['tau2_available'] is False, meta['heterogeneity']
    for block in ('fixed_effect', 'random_effects'):
        assert math.isfinite(meta[block]['mean']), (block, meta[block])
        if meta[block]['ci'] is not None:
            assert all(math.isfinite(v) for v in meta[block]['ci']), meta[block]
    trend = L.scaling_trend(scaled, min_seeds=3, min_points=3)
    het = trend.get('heterogeneity') or {}
    assert het.get('Q') is None or math.isfinite(het['Q']), het
    assert het.get('tau2') is None or math.isfinite(het['tau2']), het
