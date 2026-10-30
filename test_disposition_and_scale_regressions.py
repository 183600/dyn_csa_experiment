"""Regression checks for disposition fail-closed rules and scale-safe summaries.

These are zero-GPU checks.  They cover two classes of latent defect:

* a disposition failure text that matches both a replaceable and a
  non-replaceable class was recorded as ``ambiguous`` but then accepted as
  infrastructure noise, so a retry could pass the audit;
* the retained attempt was only checked against the declared ``first_success``
  rule, while ``last_success``/``single_attempt`` policies were not actually
  enforced;
* finite per-seed PPL panels near the float limit overflowed the raw NumPy
  mean/SD in the tuning audits and left an infinite ``se`` / non-JSON number.

The ordinary paths must stay byte-for-byte on their historical values.
"""
import json
import math

import exp_lib as L


def _plan(cells, **policy):
    return L.build_run_disposition_plan(
        cells, policy=policy or None, created_at='2020-01-01T00:00:00')


def _event(key, status, run_id, *, error=None,
           started_at='2026-01-01T00:00:00'):
    arm, _sep, seed_text = key.rpartition('::seed')
    return {'kind': 'cell', 'key': key, 'variant': arm, 'seed': int(seed_text),
            'status': status, 'run_id': run_id, 'error': error,
            'started_at': started_at, 'finished_at': started_at}


def _retained(**cells):
    return {key: {'run_id': run_id} for key, run_id in cells.items()}


def test_ambiguous_failure_class_cannot_be_replaced():
    plan = _plan(['a::seed0'])
    rep = L.run_disposition_audit(
        [_event('a::seed0', 'error', 'r1',
                error='connection lost while the loss is NaN'),
         _event('a::seed0', 'ok', 'r2')],
        plan=plan, retained=_retained(**{'a::seed0': 'r2'}))
    assert rep['status'] == 'fail', rep
    assert rep['n_ambiguous_failures'] == 1, rep
    assert rep['n_ambiguous_retried'] == 1, rep
    assert any('ambiguous' in p for p in rep['problems']), rep['problems']
    assert rep['n_non_replaceable_retried'] == 0, rep


def test_last_success_policy_rejects_an_earlier_retained_success():
    plan = _plan(['a::seed0'], retention='last_success')
    rep = L.run_disposition_audit(
        [_event('a::seed0', 'ok', 'r1'), _event('a::seed0', 'ok', 'r2')],
        plan=plan, retained=_retained(**{'a::seed0': 'r1'}))
    assert rep['status'] == 'fail', rep
    assert rep['n_retained_not_declared_success'] == 1, rep
    assert any('last_success' in p for p in rep['problems']), rep['problems']


def test_last_success_policy_accepts_the_last_retained_success():
    plan = _plan(['a::seed0'], retention='last_success')
    rep = L.run_disposition_audit(
        [_event('a::seed0', 'ok', 'r1'), _event('a::seed0', 'ok', 'r2')],
        plan=plan, retained=_retained(**{'a::seed0': 'r2'}))
    assert rep['status'] in ('warn', 'ok'), rep
    assert rep['n_retained_not_declared_success'] == 0, rep


def test_single_attempt_policy_rejects_a_retry_after_failure():
    plan = _plan(['a::seed0'], retention='single_attempt')
    rep = L.run_disposition_audit(
        [_event('a::seed0', 'error', 'r1', error='timeout'),
         _event('a::seed0', 'ok', 'r2')],
        plan=plan, retained=_retained(**{'a::seed0': 'r2'}))
    assert rep['status'] == 'fail', rep
    assert rep['n_single_attempt_violations'] == 1, rep


def test_single_attempt_policy_counts_multiple_successes_once():
    plan = _plan(['a::seed0'], retention='single_attempt')
    rep = L.run_disposition_audit(
        [_event('a::seed0', 'ok', 'r1'), _event('a::seed0', 'ok', 'r2')],
        plan=plan, retained=_retained(**{'a::seed0': 'r2'}))
    assert rep['status'] == 'fail', rep
    assert rep['n_single_attempt_violations'] == 1, rep
    assert sum('requires a single attempt' in p
               for p in rep['problems']) == 1, rep['problems']
    assert not any('explicitly retains the last success' in w
                   for w in rep['warnings']), rep['warnings']


def test_iso_z_and_fractional_timestamps_are_comparable():
    plan = L.build_run_disposition_plan(
        ['a::seed0'], created_at='2026-01-01T00:00:00Z')
    rep = L.run_disposition_audit(
        [_event('a::seed0', 'ok', 'r1',
                started_at='2025-12-31T23:59:59.500Z')],
        plan=plan, retained=_retained(**{'a::seed0': 'r1'}))
    assert rep['plan_before_first_attempt'] is False, rep
    assert rep['status'] == 'fail', rep


def test_tuning_margin_extreme_sd_is_unavailable_not_infinite():
    best = {'seed_ppls': {'0': 1e-300, '1': 1.7e308}, 'ppl': 1.0}
    runner = {'seed_ppls': {'0': 1.7e308, '1': 1e-300}, 'ppl': 2.0}
    rep = L.tuning_selection_margin(best, runner)
    assert rep['se'] is None and rep['z'] is None, rep
    assert rep['decisive'] is None, rep
    assert rep['margin'] is not None and math.isfinite(rep['margin']), rep
    json.dumps(rep, allow_nan=False)


def test_tuning_stability_extreme_ppl_mean_stays_finite():
    trials = [
        {'config': {'x': 1}, 'n_seeds': 3,
         'seed_ppls': {'0': 1e308, '1': 1e308, '2': 1e308}},
        {'config': {'x': 2}, 'n_seeds': 3,
         'seed_ppls': {'0': 1.1e308, '1': 1.1e308, '2': 1.1e308}},
    ]
    rep = L.tuning_selection_stability(trials)
    assert rep['status'] == 'ok', rep
    for choice in rep['choices']:
        assert math.isfinite(choice['metric']), choice
    json.dumps(rep, allow_nan=False)


def test_unit_precision_multiple_of_sesoi_is_withheld_when_unrepresentable():
    rep = L.unit_precision_design(10000, pilot_unit_sd=1.7e308,
                                  sesoi=1e-300)
    assert rep['mde']['multiple_of_sesoi'] is None, rep['mde']
    json.dumps(rep, allow_nan=False)


def test_document_sampling_extreme_finite_sizes_is_json_safe():
    # Every size is finite, but both the total and one stratum sum overflow.
    # Coverage is a ratio and must stay a finite fraction; an unrepresentable
    # absolute size sum is reported as None rather than inf.
    plan = L.document_stratified_plan(
        [1e308, 1.7e308, 1.5e308, 1e-300], 3, n_strata=2)
    cov = plan['size_coverage_fraction']
    assert math.isfinite(cov) and 0.0 <= cov <= 1.0, plan
    assert all(s['size_sum'] is None or math.isfinite(s['size_sum'])
               for s in plan['strata']), plan['strata']
    json.dumps(plan, allow_nan=False)


def test_size_weighted_sampling_handles_a_total_weight_overflow():
    # Each stratum sum is finite (1e308), but their total is not.  The
    # allocation must normalise the representable weights instead of calling
    # math.isfinite on the whole NumPy array (which raises TypeError) or
    # silently falling back to unit-count weighting.
    plan = L.document_stratified_plan(
        [1e308, 1e308], 2, n_strata=2, allocation='size_weighted')
    assert plan['selected'] == [0, 1], plan
    assert plan['size_coverage_fraction'] == 1.0, plan
    assert all(s['n_selected'] == 1 for s in plan['strata']), plan['strata']
    json.dumps(plan, allow_nan=False)

    # Same guard at the allocation helper: the relative size weights are
    # retained (stratum 1 has the larger population cap here), rather than
    # being replaced by the unit-count fallback [1, 1, 1].
    alloc = L._sampling_allocation(
        6, 3, [1, 2, 3], 'size_weighted', 0,
        stratum_size_sums=[1.7e308, 1e308, 1e-300])
    assert alloc.tolist() == [1, 2, 0], alloc


def test_scaling_trend_normalises_mixed_extreme_and_ordinary_effects():
    # The largest effect is above the safe-squaring limit but one seed is
    # ordinary.  Scaling on the maximum (not only when EVERY non-zero effect
    # is huge) is required: otherwise np.mean overflows and the whole trend
    # turns into NaN.  The sign-flip inference is scale invariant.
    points = [
        {'label': 'a', 'scale': 1.0, 'seeds': [0, 1, 2],
         'effects': [1.7e308, 1.7e308, 1.0]},
        {'label': 'b', 'scale': 10.0, 'seeds': [0, 1, 2],
         'effects': [1.6e308, 1.6e308, 1.0]},
        {'label': 'c', 'scale': 100.0, 'seeds': [0, 1, 2],
         'effects': [1.5e308, 1.5e308, 1.0]},
    ]
    trend = L.scaling_trend(points, min_seeds=2, min_points=2)
    assert math.isfinite(trend['wls']['slope']), trend['wls']
    assert math.isfinite(trend['direction_slope']), trend
    assert math.isfinite(trend['paired']['slope_mean']), trend['paired']
    json.dumps(trend, allow_nan=False)
    meta = L.meta_analyze_effects(points, min_seeds=2)
    json.dumps(meta, allow_nan=False)


def _tuning_artifact(per_arm):
    """Minimal tuning artifact; each trial row carries an explicit seed panel."""
    trials = L.BASELINE_TUNING_PROTOCOL['trials'][:2]

    def _arm(rows):
        out = []
        for cfg, row in zip(trials, rows):
            rec = {'config': dict(cfg), 'ppl': row['ppl'], 'ppl_std': 0.0,
                   'seed_ppls': {str(k): v for k, v in row['seeds'].items()}}
            rec.update({k: v for k, v in row.items()
                        if k not in ('ppl', 'seeds', 'ppl_std')})
            out.append(rec)
        best = min(out, key=lambda r: r['ppl'])
        return {'trials': out, 'selected': dict(best['config']),
                'selected_metric_value': best['ppl']}

    return {'selection_metric': 'ppl', 'selection_split': 'validation',
            'direction': 'lower_is_better',
            'arms': {'full_matched': _arm(per_arm[0]),
                     'csa_dynamic': _arm(per_arm[1])}}


def test_tuning_audit_recomputes_an_overflowing_per_seed_mean():
    # The true mean of [1.5e308, 1.5e308] is representable, but np.mean's sum
    # overflows to inf.  The old comparison `inf > inf` was False, so a
    # forged finite `ppl` passed the replay gate.  The guarded mean must catch
    # the mismatch, and the correctly reported extreme mean must still pass.
    protocol = dict(L.BASELINE_TUNING_PROTOCOL,
                    arms=['full_matched', 'csa_dynamic'])
    forged = _tuning_artifact([
        [{'ppl': 1.4e308, 'seeds': {0: 1.5e308, 1: 1.5e308}},
         {'ppl': 1.6e308, 'seeds': {0: 1.55e308, 1: 1.55e308}}],
        [{'ppl': 1.4e308, 'seeds': {0: 1.5e308, 1: 1.5e308}},
         {'ppl': 1.5e308, 'seeds': {0: 1.45e308, 1: 1.45e308}}],
    ])
    rep = L.baseline_tuning_audit(forged, protocol=protocol)
    assert rep['tuning_verified'] is False, rep
    assert any('mean of its per-seed' in p for p in rep['problems']), rep['problems']
    assert not any('not representable' in p for p in rep['problems']), rep['problems']

    good = _tuning_artifact([
        [{'ppl': 1.5e308, 'seeds': {0: 1.5e308, 1: 1.5e308}},
         {'ppl': 1.55e308, 'seeds': {0: 1.55e308, 1: 1.55e308}}],
        [{'ppl': 1.5e308, 'seeds': {0: 1.5e308, 1: 1.5e308}},
         {'ppl': 1.45e308, 'seeds': {0: 1.45e308, 1: 1.45e308}}],
    ])
    rep_good = L.baseline_tuning_audit(good, protocol=protocol)
    assert not any('mean of its per-seed' in p for p in rep_good['problems']), \
        rep_good['problems']
    assert not any('not representable' in p for p in rep_good['problems']), \
        rep_good['problems']


def test_tuning_audit_rejects_an_unrepresentable_pooled_ppl():
    # exp(nll_sum/n_tokens) is beyond float64.  A finite reported PPL cannot
    # be that value; the old `inf > inf` comparison accepted it silently.
    row = {'ppl': 1.0, 'seeds': {0: 1.0, 1: 1.0}, 'nll_sum': 1e308,
           'n_tokens': 1, 'ppl_aggregation': 'pooled_token_nll'}
    protocol = dict(L.BASELINE_TUNING_PROTOCOL,
                    arms=['full_matched', 'csa_dynamic'])
    rep = L.baseline_tuning_audit(
        _tuning_artifact([[row, row], [row, row]]), protocol=protocol)
    assert rep['tuning_verified'] is False, rep
    assert any('not representable' in p for p in rep['problems']), rep['problems']
    assert any('exp(nll_sum/n_tokens)' in p for p in rep['problems']), rep['problems']


def test_lesion_design_rejects_an_unhashable_source_without_crashing():
    # A hand-written/third-party declaration can carry a list/dict where a
    # registry key belongs.  The design gate must report a violation, not
    # raise TypeError from `source in ABLATION_DESIGN` / `.get(source)`.
    for bad_source in ([], {'name': 'attention_sink'}):
        design = dict(L.LESION_DESIGN)
        design['attention_sink'] = dict(design['attention_sink'],
                                        source_component=bad_source)
        rep = L.lesion_design_audit(design)
        assert rep['status'] == 'fail', rep
        assert any('source_component' in v for v in rep['violations']), \
            rep['violations']
    # A hashable but unregistered source reports the same failure.
    design = dict(L.LESION_DESIGN)
    design['attention_sink'] = dict(design['attention_sink'],
                                    source_component='not_a_component')
    rep = L.lesion_design_audit(design)
    assert rep['status'] == 'fail', rep
    assert any('source_component' in v for v in rep['violations']), rep['violations']


def test_paired_fragility_extreme_ci_is_unavailable_not_nan():
    # Every delta is finite, but the true sample SD (and therefore the t
    # interval) is beyond float64.  The old report returned NaN bounds, which
    # a strict-JSON writer rejects; "unavailable" must be None.
    rep = L.paired_fragility([1.7e308, -1.7e308, 1.7e308])
    assert rep['ci95_low'] is None and rep['ci95_high'] is None, rep
    assert rep['ci_excludes_zero'] is None, rep
    assert rep['std'] is None and rep['sem'] is None, rep
    json.dumps(rep, allow_nan=False)
    # The composite robustness block inherits the same fields.
    rob = L.confirmation_robustness({'a': [1.7e308, -1.7e308, 1.7e308]})
    assert rob['per_comparison']['a']['ci95_low'] is None, rob
    json.dumps(rob, allow_nan=False)
    # Ordinary panels keep a real interval.
    ordinary = L.paired_fragility([0.5] * 12)
    assert ordinary['ci95_low'] is not None and ordinary['ci95_high'] is not None
    assert ordinary['ci_excludes_zero'] is True
