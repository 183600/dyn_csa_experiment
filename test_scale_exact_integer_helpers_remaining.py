"""Exact-integer boundaries in rank, ratio and calibration helpers.

These helpers summarize a panel or compute a ratio/score readout.  A valid
measurement can be an exact Python integer beyond float64; the helper must
keep the exactly defined result (rank order, exact ratio, or the documented
unavailable verdict) instead of leaking ``OverflowError`` from its float
conversion.
"""
import json
import math

import numpy as np
import pytest

import exp_lib as L


HUGE = 10 ** 400


def test_arm_rank_ranks_exact_integers_beyond_float64():
    # Ranking depends only on the value order and exact ties, so huge ints are
    # still rankable even though they have no float spelling.
    assert L._aru_ranks([[HUGE, 0], [HUGE + 1, 1]], 'min').tolist() == [
        [1, 1], [2, 2]]
    assert L._aru_ranks([[HUGE, HUGE], [HUGE, HUGE + 1]], 'min').tolist() == [
        [1, 1], [1, 2]]
    assert L._aru_ranks([[HUGE, 0], [HUGE + 1, 1]], 'max').tolist() == [
        [2, 2], [1, 1]]
    # Ordinary values keep the historical competition ranks.
    assert L._aru_ranks([[3.0, 1.0], [1.0, 1.0], [2.0, 2.0]],
                        'min').tolist() == [[3, 1], [1, 1], [2, 3]]
    with pytest.raises(ValueError):
        L._aru_ranks([HUGE, HUGE + 1], 'min')


def test_unit_pooled_row_survives_exact_integer_terms():
    # A huge numerator and a huge exact denominator still give a small ratio.
    assert L._uq_pooled_row([HUGE, -HUGE], HUGE) == 0.0
    assert L._uq_pooled_row([HUGE, 0], HUGE) == 1.0
    # An underflowing non-zero ratio is withheld, not reported as zero.
    assert L._uq_pooled_row([1.0, 2.0], HUGE) is None
    # A one-shot iterator is materialised once, and ordinary rows stay exact.
    assert L._uq_pooled_row(iter([HUGE, -HUGE]), HUGE) == 0.0
    assert L._uq_pooled_row([1.0, 2.0], 3) == 1.0


def test_nll_row_cross_entropy_accepts_exact_integer_entries():
    assert L._nll_row_ce([[HUGE, -HUGE]], HUGE).tolist() == [0.0]
    assert L._nll_row_ce([[HUGE, 0]], HUGE).tolist() == [1.0]
    assert L._nll_row_ce_exact([[HUGE, -HUGE]], [HUGE]).tolist() == [0.0]
    # A ratio below the smallest float is not a measured zero.
    assert L._nll_row_ce_exact([[1.0, 2.0]], [HUGE]) is None
    # Ordinary panels keep the historical result.
    assert L._nll_row_ce([[1.0, 2.0]], 3).tolist() == [1.0]


def test_calibration_withholds_unrepresentable_scores():
    got = L._binary_calibration(np.array([0, 1]), np.array([HUGE, 0.5]))
    assert got['status'] == 'unavailable'
    json.dumps(got, allow_nan=False)
    got = L._multiclass_calibration(
        np.array([0, 1]), [[HUGE, 0.0], [0.0, 1.0]], np.array([0, 1]))
    assert got['status'] == 'unavailable'
    json.dumps(got, allow_nan=False)
    # Representable probabilities keep the ordinary Brier/ECE path.
    ordinary = L._binary_calibration(np.array([0, 1]),
                                     np.array([0.2, 0.8]))
    assert ordinary['status'] == 'ok'
    assert math.isfinite(ordinary['brier_score'])
