"""Exact-integer boundaries in secondary readout helpers.

These helpers summarize per-cluster effects, rank bucket means and normalise
ordered-unit panels.  A declared count or effect can be an exact Python integer
beyond float64; the readout must return its documented unavailable value (or
the exact rank/ratio) instead of leaking ``OverflowError`` into an analysis
artifact that is otherwise strict-JSON.
"""
import json
import math

import numpy as np
import pytest

import exp_lib as L


HUGE = 10 ** 400


def test_error_cluster_stats_with_exact_integers_stays_json_safe():
    # The float conversion used to raise before the cluster count could be
    # reported.  The panel is unrepresentable, so its moments/p are withheld.
    stats = L._error_cluster_stats([HUGE, -HUGE])
    assert stats['n_clusters'] == 2
    assert stats['mean'] is None and stats['sd'] is None
    assert stats['p_exact_signflip'] is None
    json.dumps(stats, allow_nan=False)

    # The exact token total must not be rounded through a float.
    with_tokens = L._error_cluster_stats([HUGE], token_counts=[HUGE])
    assert with_tokens['n_clusters'] == 1
    assert with_tokens['n_tokens'] == HUGE
    json.dumps(with_tokens, allow_nan=False)

    # A one-shot generator is consumed once, not silently treated as empty.
    gen_stats = L._error_cluster_stats(iter([1.0, 2.0, 3.0]))
    assert gen_stats['n_clusters'] == 3
    assert gen_stats['mean'] == 2.0

    ordinary = L._error_cluster_stats([1.0, 2.0, 3.0], token_counts=[2, 4, 6])
    assert ordinary['n_tokens'] == 12
    json.dumps(ordinary, allow_nan=False)


def test_error_stat_rank_handles_exact_integers_beyond_float64():
    # Ranking is exact: H and H+1 are distinct, while the ordinary np path
    # used to raise on the float conversion.
    assert L._error_stat_rank([HUGE, 0, HUGE + 1]).tolist() == [2.0, 1.0, 3.0]
    assert L._error_stat_rank([HUGE, HUGE, 0]).tolist() == [2.5, 2.5, 1.0]
    # Ordinary ties and values keep the historical average-rank result.
    assert L._error_stat_rank([3.0, 1.0, 2.0]).tolist() == [3.0, 1.0, 2.0]
    assert L._error_stat_rank([1.0, 1.0]).tolist() == [1.5, 1.5]


def test_ordered_unit_centre_and_lag1_survive_exact_integers():
    ratios, scale = L._od_centre([HUGE, -HUGE])
    assert ratios == [1.0, -1.0] and scale == HUGE
    ratios, scale = L._od_centre([HUGE, 1.0])
    assert scale == HUGE and ratios[0] == 1.0 and math.isfinite(ratios[1])

    # The lag-1 correlation is scale invariant; the raw float sum used to
    # raise on an exact integer panel.
    assert L._od_lag1([HUGE, HUGE + 1, HUGE + 2]) == pytest.approx(0.0)
    assert L._od_lag1([HUGE, HUGE, HUGE]) is None
    # Ordinary values keep the historical result.
    assert L._od_lag1([1.0, 2.0, 3.0, 4.0]) == pytest.approx(0.25)


def test_family_dependence_and_ci_formatting_do_not_raise_on_huge_ints():
    dep = L._fc_dependence([[HUGE, -HUGE], [1.0, 2.0]])
    assert dep['available'] is False
    assert 'representable' in dep['reason']
    json.dumps(dep, allow_nan=False)

    text = L._fc_ci([HUGE, HUGE + 1])
    assert text.startswith('[') and text.endswith(']')
    assert L._fc_ci([1.0, 2.0]) == '[+1.0000, +2.0000]'


def test_exact_signflip_ci_integer_scaling_accepts_exact_integers():
    xi, denom = L._esci_scale_to_integers([HUGE, -HUGE])
    assert xi == [HUGE, -HUGE] and denom == 1
    # An all-zero observed sum admits every sign pattern at mu=0.
    assert L._esci_p_at_zero_for([HUGE, -HUGE]) == 1
    # Ordinary fractional deltas still share a common denominator.
    xi, denom = L._esci_scale_to_integers([0.5, 0.25])
    assert xi == [2, 1] and denom == 4


def test_type_sm_normal_tails_accept_exact_integers():
    # The limiting tail is well defined even though float(x) raises.
    assert L._tsm_sf(HUGE) == 0.0
    assert L._tsm_sf(-HUGE) == 1.0
    assert L._tsm_cdf(HUGE) == 1.0
    assert L._tsm_cdf(-HUGE) == 0.0
    assert L._tsm_pdf(HUGE) == 0.0 and L._tsm_pdf(-HUGE) == 0.0
    # Ordinary arguments keep the historical erfc arithmetic.
    assert L._tsm_sf(0.0) == 0.5
    assert L._tsm_cdf(0.0) == 0.5


def test_binary_ranking_metrics_rank_exact_integer_scores():
    # AUC/AP depend only on the score order, so huge scores are still usable.
    y = [0, 1, 0, 1]
    huge_scores = [HUGE, -HUGE, HUGE + 1, 0]
    assert L._binary_roc_auc(y, huge_scores) == 0.0
    assert 0.0 <= L._binary_average_precision(y, huge_scores) <= 1.0
    # Ordinary scores keep the historical values exactly.
    assert L._binary_roc_auc(y, [1, 2, 3, 4]) == 0.75
    assert L._binary_average_precision(y, [1, 2, 3, 4]) == pytest.approx(
        5.0 / 6.0)


def test_winner_curse_percentile_sorts_exact_integers():
    # The float sort used to raise before the percentile could be withheld.
    assert L._wc_percentile([HUGE, 0, HUGE + 1], 0.0) == 0.0
    assert L._wc_percentile([HUGE, 0, HUGE + 1], 0.5) is None
    assert L._wc_percentile([HUGE, 0, HUGE + 1], 1.0) is None
    # Ordinary panels keep the historical interpolation.
    assert L._wc_percentile([1.0, 2.0, 3.0], 0.5) == 2.0
    assert L._wc_percentile([1.7e308, -1.7e308], 0.5) == 0.0


def test_saddlepoint_primitive_drops_unrepresentable_measurements():
    # The public exact test drops a measurement it cannot represent; the
    # saddlepoint primitive must not reintroduce the conversion exception.
    p, method = L._signflip_saddlepoint_p([HUGE, -HUGE, 1.0])
    assert 0.0 <= p <= 1.0 and method == 'saddlepoint'
    # An ordinary one-sided panel keeps its finite deterministic tail.
    p, method = L._signflip_saddlepoint_p([-0.1, -0.2, -0.3, -0.4])
    assert 0.0 <= p <= 1.0
    assert method in ('saddlepoint', 'saddlepoint-normal-fallback')
