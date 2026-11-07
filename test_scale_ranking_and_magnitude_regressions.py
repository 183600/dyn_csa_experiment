"""Exact-integer boundaries in ranking, threshold and magnitude helpers.

A score/effect can be an exact Python integer beyond float64.  Ranking only
needs its order, a threshold only needs an exact cut, and a common magnitude can
still normalise a panel whose individual values have no float spelling.  These
helpers must return the exactly defined readout (or their documented
unavailable value) instead of leaking ``OverflowError`` from a float
conversion.
"""
import json

import numpy as np
import pytest

import exp_lib as L


HUGE = 10 ** 400


def test_binary_threshold_curve_ranks_exact_integer_scores():
    y = [0, 1, 0, 1]
    score = [HUGE, -HUGE, HUGE + 1, 0]
    curve = L.binary_threshold_curve(y, score)
    # -HUGE is the only cut that predicts every row positive; f1 is 2/3.
    assert curve['best_threshold'] == -HUGE, curve['best_threshold']
    assert curve['best_value'] == pytest.approx(2.0 / 3.0)
    assert curve['n_positive'] == 2 and curve['n'] == 4
    assert curve['at_default_0.5']['f1'] == 0.0
    # Every row must match the brute-force confusion matrix at its exact cut.
    obj = np.asarray(score, dtype=object)
    y_arr = np.asarray(y, dtype=np.int64)
    for row in curve['thresholds']:
        brute = L._binary_threshold_metrics(y_arr, obj, row['threshold'])
        assert (row['tp'], row['fp'], row['tn'], row['fn']) == (
            brute['tp'], brute['fp'], brute['tn'], brute['fn'])
    json.dumps(curve, allow_nan=False)

    # A caller-declared exact cut is accepted and preserved exactly.
    explicit = L.binary_threshold_curve(y, score, thresholds=[HUGE, 0.5])
    assert [row['threshold'] for row in explicit['thresholds']] == [0.5, HUGE]

    # The selected operating point reduces the same curve without overflow.
    sel = L.select_binary_threshold(y, score)
    assert sel['threshold'] == -HUGE
    json.dumps(sel, allow_nan=False)

    # Ordinary scores keep the historical result.
    ordinary = L.binary_threshold_curve(
        [0] * 90 + [1] * 10,
        np.array([0.1] * 90 + [0.4] * 5 + [0.9] * 5), positive_label=1)
    assert ordinary['best_threshold'] == 0.4
    assert ordinary['best_value'] == 1.0


def test_classification_metrics_accepts_exact_integer_scores():
    y = [0, 1, 0, 1]
    score = [HUGE, -HUGE, HUGE + 1, 0]
    m = L.classification_metrics(y, y_score=score, labels=[0, 1])
    # Ranking is exact; unrepresentable endpoints are withheld, not raised.
    assert m['roc_auc'] == 0.0
    assert m['pr_auc'] is not None and 0.0 <= m['pr_auc'] <= 1.0
    assert m['score_min'] is None and m['score_max'] is None
    assert m['calibration']['status'] == 'unavailable'
    json.dumps(m, allow_nan=False)

    # A one-hot multiclass score matrix beyond float64 still ranks perfectly.
    y3 = [0, 1, 2]
    score3 = [[HUGE, 0, 1], [1, HUGE, 0], [0, 1, HUGE]]
    mc = L.classification_metrics(y3, y_score=score3, labels=[0, 1, 2])
    assert mc['roc_auc_macro'] == 1.0
    assert mc['pr_auc_macro'] == 1.0
    assert mc['calibration']['status'] == 'unavailable'
    json.dumps(mc, allow_nan=False)


def test_majority_floor_keeps_exact_supports():
    floor = L._majority_class_floor([0, 1], [HUGE, 1], [0, 1])
    assert floor['label'] == '0' and floor['support'] == HUGE
    assert floor['accuracy'] == 1.0
    # Ordinary ties keep the first declared label.
    ordinary = L._majority_class_floor([0, 1, 2], [4, 4, 1], [0, 1, 2])
    assert ordinary['label'] == '0' and ordinary['accuracy'] == pytest.approx(4 / 9)


def test_winner_curse_scale_normalises_exact_integers():
    work, scale, shift, orientation = L._wc_scale([HUGE, -HUGE], 'min')
    assert scale == HUGE and shift == -1.0 and orientation == 1.0
    assert np.all(np.isfinite(work))
    assert work.tolist() == [2.0, 0.0]
    work_max, scale_max, _shift, orientation_max = L._wc_scale(
        [HUGE, HUGE + 1], 'max')
    assert scale_max == HUGE + 1 and orientation_max == -1.0
    assert np.all(np.isfinite(work_max))
    # Ordinary values keep the historical unit scale.
    work0, scale0, shift0, orientation0 = L._wc_scale([1.0, 2.0], 'min')
    assert (scale0, shift0, orientation0) == (1.0, 0.0, 1.0)
    assert list(work0) == [1.0, 2.0]


def test_look_signature_keeps_exact_scaled_ratios():
    assert L._look_values_signature([HUGE, -HUGE], HUGE) == (1.0, -1.0)
    assert L._look_values_signature([1.0, HUGE], HUGE) == (0.0, 1.0)
    assert L._look_values_signature([1.0, 2.0], 2.0) == (0.5, 1.0)
    # No finite scale remains the historical all-zero signature.
    assert L._look_values_signature([HUGE], 0) == (0.0,)


def test_signed_sum_table_keeps_an_exact_integer_cancellation():
    sums = L._signflip_signed_sums([HUGE, -HUGE])
    assert [int(value) for value in sums] == [0, -2 * HUGE, 2 * HUGE, 0]
    assert [int(value) for value in L._etost_signed_sums([HUGE])] == [
        HUGE, -HUGE]
    # Ordinary tables keep their historical float order and values.
    assert L._signflip_signed_sums([1.0, 2.0]).tolist() == [3.0, 1.0, -1.0, -3.0]


def test_family_score_is_invariant_to_an_exact_integer_row_scale():
    got = L._fc_score_maxt([[HUGE, HUGE + 1]], 0.05)
    # |1 + 1| / sqrt(1 + 1) with each row divided by its exact maximum.
    assert got['score'][0] == pytest.approx(2.0 / np.sqrt(2.0))
    assert got['p_raw'][0] == 0.5
    ordinary = L._fc_score_maxt([[1.0, 2.0]], 0.05)
    assert ordinary['score'][0] == pytest.approx(3.0 / np.sqrt(5.0))
