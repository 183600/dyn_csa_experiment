"""Regressions for exact-integer/extreme-value boundaries in shared helpers.

These are CPU-only checks for helpers that are also used by future scale-up
panels: a token count may be an exact integer beyond float64, and a legacy
weight may remain fractional.  A helper must return the documented unavailable
value (or the representable result) instead of escaping with ``OverflowError``
or silently discarding part of the input.
"""
import json
import math

import pytest

import exp_lib as L


HUGE = 10 ** 400


def test_ls_div_rejects_invalid_denominators_and_withholds_underflow():
    # A zero denominator used to raise ZeroDivisionError.
    assert L._ls_div(1.0, 0) is None
    assert L._uq_div(1.0, 0) is None
    # A fractional denominator was truncated by int(den), dividing by a
    # different count; reject it instead.
    assert L._ls_div(1.0, 2.5) is None
    assert L._ls_div(1.0, -2) is None
    # A non-zero quotient below the smallest float is not a measured zero.
    assert L._ls_div(1.0, HUGE) is None
    # Huge denominators with a representable quotient keep the exact result.
    assert L._ls_div(10 ** 308, HUGE) == pytest.approx(1e-92, rel=1e-15)
    # Ordinary division is unchanged.
    assert L._ls_div(1.0, 2) == 0.5
    assert L._ls_div(0.0, 3) == 0.0


def test_unit_split_assignment_survives_mixed_huge_int_and_float_weights():
    # The greedy balance accumulated an exact huge int and a fractional float
    # in one running total; whichever came second raised OverflowError even
    # though the token shares are representable after normalisation.
    units = [f'u{i}' for i in range(6)]
    tokens = [HUGE, 1.5, 2 ** 53 + 1, 1, 7, 3]
    assignment = L.unit_split_assignment(units, tokens)
    indices = [i for half in assignment['halves'] for i in half['indices']]
    assert sorted(indices) == list(range(len(units)))
    assert all(half['n_units'] >= 2 for half in assignment['halves'])
    # Exact token counts remain exact in the replayable body.
    assert assignment['n_units'] == len(units)
    json.dumps(assignment, allow_nan=False)

    # The same panel through the public estimator must not crash and must keep
    # its strict-JSON contract (the huge/tiny weights can make it
    # non-estimable, but never an escaping conversion error).
    rows = [[-0.10, -0.20, -0.30, -0.40, -0.50, -0.60] for _ in range(6)]
    result = L.unit_split_replication(
        rows, tokens, unit_ids=units, seeds=list(range(6)),
        direction='negative', min_n=6, min_units=4,
        min_units_per_half=2, n_salts=0)
    assert result['status'] in ('ok', 'warnings', 'not_estimable')
    json.dumps(result, allow_nan=False)


def test_ablation_lattice_helpers_handle_exact_integers_beyond_float64():
    # The raw mean used to raise from ``sum(values) / len(values)``.
    assert L._raw_mean_or_scaled([HUGE, -HUGE]) == 0.0
    assert L._raw_mean_or_scaled([1.0, 2.0]) == 1.5
    assert L._raw_mean_or_scaled([HUGE, 0]) is None

    # A span is exact before the final float conversion.
    assert L._al_span(HUGE, HUGE + 1) == 1.0
    assert L._al_span(0, HUGE) is None
    assert L._al_span(1.0, 2.0) == 1.0

    # The ranking tolerance multiplied a huge int by a float.  The tie rule is
    # still well defined in the exact domain.
    assert L._al_rank([HUGE, 0, HUGE + 1]) == [2.5, 1.0, 2.5]
    assert L._al_spearman([HUGE, 0], [0, HUGE]) == -1.0


def test_multiverse_descriptive_helpers_handle_exact_integers():
    assert L._am_span([HUGE, HUGE + 1]) == 1.0
    assert L._am_span([0, HUGE]) is None
    assert L._am_span([1e308, -1e308]) is None

    assert L._am_median([HUGE]) is None
    assert L._am_median([HUGE, HUGE + 1]) is None
    assert L._am_median([1, 2]) == 1.5

    assert L._am_mean([HUGE, -HUGE]) == 0.0
    assert L._am_mean([HUGE, HUGE + 1]) is None
    assert L._am_mean([1, 2, 3]) == 2.0

    # All-identical huge counts have true SD 0, not a conversion crash.
    assert L._am_sd([HUGE, HUGE]) == 0.0
    assert L._am_sd([HUGE, HUGE + 1]) == pytest.approx(2 ** -0.5)
    # A spread that really does not fit float64 is reported as infinite, as
    # the existing contract for an unrepresentable SD.
    assert L._am_sd([HUGE, -HUGE]) == float('inf')


def test_sampling_finite_sum_keeps_an_exact_cancelling_pair():
    # np.asarray(..., dtype=float) used to raise before the sum could cancel.
    assert L._sampling_finite_sum([HUGE, -HUGE]) == 0.0
    assert L._sampling_finite_sum([HUGE, 1]) is None
    assert L._sampling_finite_sum([1.0, 2.0]) == 3.0


def test_budget_ablation_mean_handles_exact_integer_panels():
    assert L._ba_mean_or_raw([HUGE, -HUGE]) == 0.0
    assert L._ba_mean_or_raw([HUGE, 0]) is None
    assert L._ba_mean_or_raw([1, 2, 3]) == 2.0


def test_normal_tails_with_an_exact_integer_beyond_float64():
    # A huge positive z has upper tail 0 and CDF 1 (and the mirror for a
    # negative z); float(x) used to raise before the limiting value.
    assert L._seq_norm_sf(HUGE) == 0.0
    assert L._seq_norm_sf(-HUGE) == 1.0
    assert L._normal_cdf(HUGE) == 1.0
    assert L._normal_cdf(-HUGE) == 0.0
    assert L._cv_norm_sf(HUGE) == 0.0
    assert L._cv_norm_cdf(-HUGE) == 0.0
    # Ordinary arguments keep the historical erfc arithmetic.
    assert L._seq_norm_sf(0.0) == 0.5
    assert L._normal_cdf(0.0) == 0.5


def test_ordered_unit_helpers_handle_huge_counts_without_conversion_crash():
    assert L._od_mean_sd([HUGE]) == (None, None)
    assert L._od_primary_batches(HUGE) == 10
    assert L._po_primary_batches(HUGE) == 10
    # The capped batch count is unchanged for ordinary panel sizes.
    assert L._od_primary_batches(16) == 4
    assert L._po_primary_batches(1000) == 10


def test_unit_split_stats_withhold_an_unrepresentable_extreme():
    stats = L._usr_stats([HUGE])
    assert stats['n'] == 1
    assert stats['mean'] is None
    assert stats['min'] is None and stats['max'] is None


def test_unit_quantile_sort_and_quantiles_survive_exact_integers():
    values = L._uq_sort_values([HUGE, 0, HUGE + 1])
    assert values == [0, HUGE, HUGE + 1]
    # An order statistic beyond float64 is unavailable, not a conversion
    # exception.
    assert L._uq_quantile(values, 0.5) is None
    assert L._uq_lerp(HUGE, HUGE, 0.5) is None
    # Ordinary values keep the exact historical result.
    assert L._uq_sort_values([3, 1.0, 2]) == [1.0, 2.0, 3.0]
    assert L._uq_quantile([1.0, 2.0, 3.0], 0.5) == 2.0


def test_global_fdr_harmonic_does_not_iterate_to_a_huge_count():
    # The exact harmonic sum is kept for real family sizes; a pathological
    # count must return the analytic limit instead of looping forever.
    assert L._gfdr_harmonic(10) == pytest.approx(
        sum(1.0 / i for i in range(1, 11)))
    huge_h = L._gfdr_harmonic(HUGE)
    assert math.isfinite(huge_h)
    assert huge_h == pytest.approx(math.log(HUGE) + 0.5772156649015329,
                                   rel=1e-12)


def test_endpoint_comparison_helpers_handle_exact_integers():
    assert L._es_close(HUGE, HUGE + 1) is True
    assert L._es_close(HUGE, 0) is False
    assert L._es_ge(0, HUGE) is False
    assert L._es_le(0, HUGE) is True
    assert L._es_lt(HUGE, 0) is False
    # Ordinary values keep the historical relative comparison.
    assert L._es_close(1.0, 1.0 + 1e-13) is True


def test_ordered_unit_batch_and_interval_guards_fail_closed():
    assert L._od_batch_design([1, 2, 3, 4], 0) is None
    assert L._od_batch_design([1, 2, 3, 4], 1) is None
    assert L._od_percentile_ci([1.0, 2.0, 3.0], HUGE) is None
    assert L._od_percentile_ci([1.0, 2.0, 3.0], 0.95) == pytest.approx(
        [1.05, 2.95])


def test_unit_quantile_rejects_an_out_of_range_probability():
    assert L._uq_quantile([1.0, 2.0, 3.0], HUGE) is None
    assert L._uq_quantile([1.0, 2.0, 3.0], 1.5) is None
    assert L._uq_quantile([1.0, 2.0, 3.0], -0.1) is None
    assert L._uq_quantile([1.0, 2.0, 3.0], 0.25) == 1.5


def test_length_strata_summary_withholds_ci_for_an_invalid_conf():
    summary = L._ls_summary([1.0, 2.0, 3.0], HUGE)
    assert summary['conf'] is None
    assert summary['ci95'] is None
    # A valid conf keeps the ordinary interval.
    assert L._ls_summary([1.0, 2.0, 3.0], 0.95)['ci95'] is not None


def test_sequential_step_convolution_fails_closed_beyond_its_budget():
    with pytest.raises(ValueError, match='exact-convolution budget'):
        L._seq_step_counts({0: 1}, HUGE)
    # Ordinary look increments keep the exact integer counts.
    assert L._seq_step_counts({0: 1}, 2) == {-2: 1, 0: 2, 2: 1}





def test_arm_rank_sample_sd_handles_exact_integers():
    assert L._aru_sample_sd([HUGE, HUGE]) == 0.0
    assert L._aru_sample_sd([HUGE, HUGE + 1]) == pytest.approx(2 ** -0.5)
    assert L._aru_sample_sd(HUGE) is None
    assert L._aru_sample_sd([1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_exact_tost_primitive_drops_unrepresentable_measurements():
    scaled, maxabs = L._etost_scaled([HUGE])
    assert scaled is None and maxabs is None
    record = L._etost_upper_tail([HUGE])
    assert record['n'] == 0
    assert math.isnan(record['p'])
    assert record['method'] == 'none'
    # An ordinary one-sided tail keeps its exact probability.
    ordinary = L._etost_upper_tail([-0.1, -0.2, -0.3])
    assert 0.0 <= ordinary['p'] <= 1.0
    assert ordinary['method'] in ('exact', 'exact-mitm', 'saddlepoint')


def test_length_strata_grouping_does_not_iterate_from_a_huge_stratum_count():
    # Every g above len(tokens)//min_units is skipped by the size check, so a
    # huge declared count must not turn the search into an unbounded loop.
    assert L._ls_groups([1, 2, 3, 4], HUGE, 1) == [[0], [1], [2], [3]]
    assert L._ls_groups([1, 2, 3, 4], 2, 1) == [[0, 1], [2, 3]]
