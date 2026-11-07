"""Scale regressions for the chi-square tail and scaled moment helpers.

Both failures are CPU-only, finite-input cases:

* ``_gammaincc``/``chi2_sf`` used to raise ``ValueError: math domain error``
  when the integer-shape branch formed ``r = x/a - 1`` at a tiny x.  The
  subtraction rounds to exactly -1, so ``math.log1p(r)`` has no value even
  though the upper tail is simply one.
* the scaled standard-error helper formed ``ss * multiplier`` before taking
  the square root; that intermediate can overflow while the final,
  scale-normalised standard error is representable.  A multiplier above
  float64 (an exact Python integer) was also passed straight to
  ``math.isfinite``/``float`` and raised instead of being combined in log
  space.
"""
import math

import pytest

import exp_lib as L


def test_chi2_upper_tail_when_the_statistic_is_tiny_and_df_is_large():
    # a = df/2 is in the 1000..1e7 series branch; x = statistic/2 is so small
    # that (x - a)/a rounds to -1.0.  The upper tail is one at float
    # precision, not a domain error.
    for df, x in ((2000.0, 1e-300), (2000.0, 1e-100),
                  (2_000_000.0, 1e-300), (2000.0, 5e-324)):
        assert L.chi2_sf(x, df) == 1.0, (df, x)
        assert L._gammaincc(df / 2.0, x / 2.0) == 1.0, (df, x)
    assert L._gammaincc(1000.0, 0.0) == 1.0
    assert L._gammaincc(999.0, 1e-300) == 1.0
    # The ordinary calibration point is unchanged.
    assert L.chi2_sf(3.841458820694124, 1.0) == pytest.approx(
        0.05, abs=1e-12)


def test_chi2_tail_when_log_gamma_itself_overflows():
    # df=1e308 makes a=df/2 ~5e307, beyond the log-gamma range.  The torch
    # path returns NaN there and the old fallback raised OverflowError.
    # Gamma(a,1) is then concentrated at a, so one float64 ulp is many
    # standard deviations: the tail is 1 below the mode, 0.5 at it and 0
    # above it.
    for df, x, expected in (
            (1e308, 1e307, 1.0),
            (1e308, 2e307, 1.0),
            (1e308, 5e307, 1.0),
            (1e308, 1e308, 0.5),
            (1e308, 1.1e308, 0.0)):
        assert L.chi2_sf(x, df) == expected, (df, x)
    assert L._gammaincc(5e307, 1e307) == 1.0
    assert L._gammaincc(1e308, 1e308) == 0.5


def test_chi2_tail_accepts_exact_integer_arguments_beyond_float64():
    # ``math.isnan``/``math.isfinite`` convert to float and raise for an
    # exact integer above 1e308.  The shape argument is still a finite
    # positive number and the tail saturates exactly.
    huge = 10 ** 400
    assert L._gammaincc(huge, 1.0) == 1.0
    # Both arguments beyond float64: the gap to the mode is compared in log
    # space against sd=sqrt(a), so a difference of one integer is still at the
    # mode, while a difference beyond sqrt(a) saturates.
    assert L._gammaincc(huge, huge - 1) == 0.5
    assert L._gammaincc(huge, huge) == 0.5
    assert L._gammaincc(huge, huge + 1) == 0.5
    assert L._gammaincc(huge, huge - 10 ** 300) == 1.0
    assert L._gammaincc(huge, huge + 10 ** 300) == 0.0
    assert L._gammaincc(0.5, huge) == 0.0
    assert L._gammaincc(1000.0, huge) == 0.0


def test_scaled_root_recovers_an_overflowing_ss_times_multiplier():
    values = [1e-300] * 100
    multiplier = 1e308
    # The naive intermediate ss * multiplier is ~1e310 (inf), while the true
    # standard error is ~1e-145.
    expected = math.exp(math.log(1e-300) + 0.5 * math.log(100.0)
                        + 0.5 * math.log(multiplier))
    got = L._scaled_root_sum_of_squares(values, multiplier)
    assert got is not None and math.isfinite(got), got
    assert got == pytest.approx(expected, rel=1e-12)
    # The variance helper already handled this case; the two must agree.
    var = L._scaled_sum_of_squares(values, multiplier)
    assert var is not None
    assert got * got == pytest.approx(var, rel=1e-12)


def test_scaled_moments_accept_an_exact_multiplier_beyond_float64():
    values = [1e-300] * 100
    multiplier = 10 ** 400
    expected = math.exp(math.log(1e-300) + 0.5 * math.log(100.0)
                        + 0.5 * math.log(multiplier))
    got = L._scaled_root_sum_of_squares(values, multiplier)
    assert got is not None and math.isfinite(got), got
    assert got == pytest.approx(expected, rel=1e-12)
    var = L._scaled_sum_of_squares(values, multiplier)
    assert var is not None and math.isfinite(var)
    assert got * got == pytest.approx(var, rel=1e-12)


def test_scaled_root_still_withholds_unrepresentable_or_invalid_results():
    # A genuinely unrepresentable SE is withheld, not returned as inf.
    assert L._scaled_root_sum_of_squares(
        [1.7e308, -1.7e308], 1.0) is None
    # The SE is below the smallest representable float.
    assert L._scaled_root_sum_of_squares([5e-324], 5e-324) is None
    assert L._scaled_root_sum_of_squares([1.0, 2.0], -1.0) is None
    assert L._scaled_root_sum_of_squares([1.0, 2.0], True) is None
    assert L._scaled_sum_of_squares([1.0, 2.0], -1.0) is None
    # A constant panel keeps the exact zero, and the all-zero panel is zero.
    assert L._scaled_root_sum_of_squares([0.0, 0.0], 1e308) == 0.0
