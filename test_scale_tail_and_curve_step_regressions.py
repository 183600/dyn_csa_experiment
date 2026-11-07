"""Regression checks for far-tail and unbounded-step scale failures.

CPU-only.  Both cases are finite but outside the range where an intermediate
float (the incomplete-beta argument, or a squared step offset) is
representable; the public helpers must keep the representable result or
withhold an undefined one instead of returning a fake zero / raising.
"""
import json
import math

import pytest

import exp_lib as L


def test_student_t_far_tail_keeps_a_representable_power_law_tail():
    # x = df/(df+t^2) underflows to 0 at this scale, but the t tail for small
    # df is a power law and remains representable: ~3.207e-101 for df=0.5.
    assert L._student_t_sf(1e200, 0.5) == pytest.approx(
        3.207009754142229e-101, rel=1e-12)
    assert L._student_t_sf(1e200, 1.0) == pytest.approx(
        3.1830988618379068e-201, rel=1e-12)
    assert L._student_t_sf(-1e200, 0.5) == pytest.approx(
        1.0 - 3.207009754142229e-101, rel=1e-15)
    # The two-sided p is twice the upper tail, not an underflowed zero.
    assert L.student_t_two_sided_p(1e200, 0.5) == pytest.approx(
        6.414019508284458e-101, rel=1e-12)
    # A tail that really is below float64 still rounds to zero.
    assert L._student_t_sf(1e200, 2.0) == 0.0


def test_student_t_far_tail_tiny_df_limit_is_symmetric_half_mass():
    # df/2 underflows for the smallest positive df.  The tiny-shape limit puts
    # half the mass on each side at any finite t, so this is 0.5, not 0/1.
    for df in (5e-324, 1e-323, 1e-100):
        assert L._student_t_sf(1e200, df) == 0.5, df
        assert L._student_t_sf(-1e200, df) == 0.5, df


def test_t_critical_alpha_uses_the_far_tail_for_small_df():
    # alpha=1e-100 has an ordinary two-sided critical value (~4.11e199) for
    # df=0.5.  Before the far-tail branch, _student_t_sf returned 0 above
    # t~1e154 and the bisection stopped at a ~1e155 boundary.
    alpha = 1e-100
    crit = L._t_critical_alpha(alpha, 0.5)
    assert math.isfinite(crit) and crit > 1e199, crit
    assert L._student_t_sf(crit, 0.5) == pytest.approx(alpha / 2.0,
                                                       rel=1e-12)


def test_curve_slope_with_unrepresentable_step_span_withholds_slope():
    # The step labels are exact integers.  Their span can exceed float64 even
    # though every label is valid; converting the squared offsets used to
    # escape as OverflowError from the health classifier.
    huge = 10 ** 400
    points = [
        {'step': 0, 'train_ce': 1.0},
        {'step': huge, 'train_ce': 2.0},
        {'step': huge + 1, 'train_ce': 3.0},
    ]
    assert L._curve_slope_stats(points, 'train_ce') == (None, None)
    # An inf ``den`` (representable labels, unrepresentable squares) also
    # withholds the slope instead of emitting a NaN/overflowing SE.
    for span in (10 ** 200, 10 ** 160):
        points = [
            {'step': i * span, 'train_ce': float(i)}
            for i in range(6)
        ]
        assert L._curve_slope_stats(points, 'train_ce') == (None, None)
        history = [
            {'step': p['step'], 'train_ce': p['train_ce'],
             'val_ce': p['train_ce']}
            for p in points
        ]
        out = L.classify_training_health(history)
        json.dumps(out, allow_nan=False)
        assert out['train_ce_slope_per_step'] is None, out
        assert out['val_ce_slope_per_step'] is None, out
        assert out['status'] == 'insufficient_data', out


def test_curve_slope_near_labels_at_huge_step_origin_still_works():
    # Translation invariance is preserved: labels above float64 are fine when
    # their SPAN is representable.
    huge = 10 ** 400
    points = [
        {'step': huge, 'train_ce': 1.0},
        {'step': huge + 1, 'train_ce': 2.0},
        {'step': huge + 2, 'train_ce': 3.0},
    ]
    slope, se = L._curve_slope_stats(points, 'train_ce')
    assert slope == pytest.approx(1.0, rel=1e-15)
    assert se == pytest.approx(0.0, abs=1e-15)


def test_binomial_far_tail_at_huge_n_is_a_probability_not_nan():
    # n=1e200, p=0.1, k~5e5 is astronomically far below the mean.  The beta
    # prefactor underflows to 0 while the complementary continued fraction
    # overflows; ``0 * nan`` used to turn both tails into NaN.  The lower
    # tail is 0 and the upper tail is 1 at float precision.
    n, p = 10 ** 200, 0.1
    for k in (532840, 10 ** 6):
        assert L._binomial_cdf_stable(k, n, p) == 0.0, k
        assert L._binomial_sf_stable(k, n, p) == 1.0, k
    # The same shape regime through the incomplete-beta entry point.
    assert L._betai(532841.0, float(n - 532841), p) == 1.0
    # n beyond float64 takes the exact-tail branch and must stay finite too.
    n = 10 ** 400
    assert L._binomial_cdf_stable(2, n, p) == 0.0
    assert L._binomial_sf_stable(2, n, p) == 1.0


def test_beta_with_one_shape_beyond_float64_does_not_overflow():
    # ``a + b`` converts the huge integer b to float even though a float plus
    # an int is a valid Python expression for ordinary values.  Both shapes
    # are exact integers here, so the binomial identity can use their exact
    # sum; the result is an ordinary 0/1 tail, not OverflowError.
    huge = 10 ** 400
    assert L._betai(1e300, huge, 1e-10) == 1.0
    assert L._betai(1e50, huge, 1.0 - 1e-16) == 1.0
    assert L._betai(huge, 1e300, 1e-10) == 0.0
    assert L._betai(huge, huge, 0.5) == pytest.approx(0.5, abs=1e-12)


def test_regularized_beta_is_clamped_to_a_probability():
    # The continued fraction can overshoot a probability by a few ulps at
    # extreme shape ratios; callers should never receive p<0 or p>1.
    for a, b, x in ((1e-300, 10.0, 1e-300),
                    (2.5, 1e-300, 1.0 - 1e-16),
                    (0.5, 1e-300, 1.0 - 1e-16)):
        value = L._betai(a, b, x)
        assert 0.0 <= value <= 1.0, (a, b, x, value)
