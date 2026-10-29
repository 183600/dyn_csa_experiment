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
import json
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


def test_methodology_audit_extreme_ppl_arm_summary_stays_finite():
    # arm_summary used raw np.mean/np.std, which overflow for a finite PPL
    # panel near the float limit even though both summaries are representable.
    # The report is written to JSON, so an inf here is not a harmless display
    # artefact: strict writers turn it into null or a non-standard token.
    cfg = ('steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_'
           'cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_'
           'dfabc_csX')
    summary = {}
    for i, ppl in enumerate([8e307, 1.0e308, 1.2e308]):
        for arm in ('a', 'b'):
            summary[f'{arm}::seed{i}'] = {
                'variant': arm, 'seed': i, 'run_cfg': cfg,
                'steps': 1, 'tokens_seen': 1, 'params': 100,
                'ppl': float(ppl),
            }
    rep = L.methodology_audit(
        summary, design={'compA': [('a', 'b', 'x')]}, baselines={},
        verbose=False, write=False)
    assert set(rep['arm_summary']) == {'a', 'b'}, rep['arm_summary']
    for name, entry in rep['arm_summary'].items():
        for key in ('ppl_mean', 'ppl_mean_arithmetic', 'ppl_primary',
                    'ppl_std'):
            assert math.isfinite(float(entry[key])), (name, key, entry)
        assert entry['ppl_mean'] == entry['ppl_mean_arithmetic']
    # Ordinary panels keep the historical NumPy values (no formula change).
    assert L._scaled_mean_or_raw([1.0, 2.0, 3.0]) == float(
        np.mean([1.0, 2.0, 3.0]))
    assert L._scaled_std_or_raw([1.0, 2.0, 3.0]) == float(
        np.std([1.0, 2.0, 3.0], ddof=1))


def test_error_cluster_stats_extreme_finite_deltas_stay_json_safe():
    # The bucket/confirmation error path used raw mean/std and could emit
    # mean=inf, sd=inf and a CI of [nan, inf] for finite deltas.  A finite
    # summary must stay finite, and an interval whose endpoint is genuinely
    # beyond float64 must be withheld as None rather than reported as inf.
    r = L._error_cluster_stats([8e307, 1.0e308, 1.2e308])
    assert math.isfinite(r['mean']) and math.isfinite(r['sd'])
    if r['ci95_low'] is not None:
        assert math.isfinite(r['ci95_low']) and math.isfinite(r['ci95_high'])
    anti = L._error_cluster_stats([1.0e308, -1.0e308, 1.0e308])
    assert math.isfinite(anti['mean']) and math.isfinite(anti['sd'])
    assert anti['ci95_low'] is None and anti['ci95_high'] is None
    # An unrepresentable sample SD is `None`, not an infinite measured spread.
    tiny = L._error_cluster_stats([1.7e308, -1.7e308])
    assert tiny['sd'] is None and tiny['se'] is None
    assert tiny['ci95_low'] is None and tiny['ci95_high'] is None
    # The ordinary path is unchanged.
    ordinary = L._error_cluster_stats([1.0, 2.0, 3.0])
    assert ordinary['mean'] == float(np.mean([1.0, 2.0, 3.0]))
    assert ordinary['sd'] == float(np.std([1.0, 2.0, 3.0], ddof=1))


def test_extreme_scale_auxiliary_intervals_are_withheld_not_infinite():
    # Paired bootstrap endpoints can exceed float64 even when every input is
    # finite; the interval must be withheld instead of returning inf.
    assert L.paired_bootstrap_ci([1e308, 1.2e308, 1.7e308], n_boot=200) is None
    assert L.bootstrap_mean_ci([1e308, 1.2e308, 1.7e308], n_boot=200) is None

    # The combined seed x document CI exponentiates a finite log interval; a
    # PPL endpoint can still be beyond float64.  `aggregate` must stay strict-
    # JSON serializable.
    summary = {}
    for i, off in enumerate([0.0, .02, .04]):
        nll = [7089 + off * 10 + j for j in range(16)]
        tok = [10] * 16
        summary[f'csa::seed{i}'] = {
            'variant': 'csa', 'seed': i,
            'ppl': math.exp(sum(nll) / sum(tok)),
            'nll_per_row': nll, 'n_tokens_per_row': tok,
            'params': 10, 'steps': 1, 'tokens_seen': 100,
        }
    agg = L.aggregate(summary)
    combined = agg['csa']['ppl_ci_seed_doc_combined']
    assert combined['status'] == 'ci_unavailable', combined
    assert combined['ci95_ppl'] is None, combined
    json.dumps(agg, allow_nan=False)

    # Seed-averaged PPL curves overflow a naive column sum for two finite
    # 1.5e308 seed points; the guarded mean keeps the curve finite.
    hist = {f'csa::seed{i}': {
        'variant': 'csa', 'seed': i, 'ppl': 1.5e308, 'nll_sum': 1e308,
        'n_tokens': 1, 'params': 10, 'steps': 1, 'tokens_seen': 100,
        'ppl_history': [[100, 1.5e308], [200, 1.5e308]],
    } for i in range(2)}
    curve = L.aggregate(hist)['csa']['ppl_curve']
    assert all(math.isfinite(float(v)) for _s, v in curve), curve
    json.dumps(curve, allow_nan=False)


def test_confirmation_replay_uses_the_guarded_mean():
    import audit_gates as A
    deltas = [1e308, 1e308, -1e308]
    st = L.paired_effect_stats(deltas)
    rec = {'per_seed': deltas, 'seeds': [0, 1, 2],
           'p_exact_signflip': st['p_exact_signflip'],
           'n_effective': 3, 'n_nonzero': 3, 'p_floor': st['p_floor'],
           'mean': st['mean']}
    problems, warnings = [], []
    A._pvr_replay_one(L, 'x', rec, where='x', problems=problems,
                      warnings=warnings)
    assert problems == [], problems


def test_max_t_unrepresentable_sd_keeps_the_representable_sem():
    deltas = [1.79e308] * 10 + [-1.79e308] * 10
    r = L.maxt_signflip_adjust({'c': deltas})
    c = r['contrasts']['c']
    assert c['sd'] is None and c['sd_available'] is False, c
    assert math.isfinite(c['sem']), c
    assert (c['ci_simultaneous'] is None
            or all(math.isfinite(v) for v in c['ci_simultaneous'])), c
    json.dumps(r, allow_nan=False)


def test_paired_unit_deltas_does_not_reject_a_large_finite_panel():
    ra = [{'ppl_per_article': [1.5e308, 1.5e308],
           'article_ids': ['a', 'b']} for _ in range(2)]
    rb = [{'ppl_per_article': [1.0, 1.0],
           'article_ids': ['a', 'b']} for _ in range(2)]
    got = L.paired_unit_deltas(ra, rb)
    assert got['status'] == 'ok', got
    assert all(math.isfinite(v) for v in got['deltas']), got
    assert math.isfinite(got['mean']) and math.isfinite(got['sd']), got


def test_zero_variance_fit_reports_a_json_safe_r2():
    r = L.loglog_regression([(1.0, 2.0), (2.0, 2.0), (3.0, 2.0)])
    assert r['status'] == 'ok', r
    assert r['r2'] == 1.0, r
    json.dumps(r, allow_nan=False)
