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
import sys

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


def test_chi2_tail_keeps_the_standardised_gap_beyond_float64():
    # Both arguments can be arbitrary-precision integers.  Their difference is
    # then exact even though it cannot be converted to float; the tail at one
    # standard deviation from the mode is ~0.159, not the saturated 0/1.
    huge = 10 ** 400
    sd = 10 ** 200                      # sqrt(huge)
    for gap in (10 ** 200, 2 * sd, 10 ** 250):
        z = gap / sd
        upper = 0.5 * math.erfc(z / math.sqrt(2.0))
        assert L._gammaincc(huge, huge + gap) == pytest.approx(
            upper, rel=1e-12, abs=0.0), gap
        assert L._gammaincc(huge, huge - gap) == pytest.approx(
            1.0 - upper, rel=1e-12, abs=0.0), gap
        assert L._gammaincc(huge + gap, huge) == pytest.approx(
            1.0 - upper, rel=1e-12, abs=0.0), gap
    # A float shape at the top of the float range and an integer statistic just
    # beyond it: the gap is one, ~1e-154 sd, so the tail is still at the mode.
    top = int(sys.float_info.max)
    assert L._gammaincc(float(sys.float_info.max), top + 1) == 0.5
    # The same near-mode precision for a huge int shape with a float statistic.
    assert L._gammaincc(huge, 1e300) == 1.0


def test_chi2_tail_without_the_special_function_does_not_loop_forever(monkeypatch):
    # The fallback series needs O(sqrt(a)) terms.  If the special function is
    # unavailable, a=1e15 would need ~1.6e9 iterations; use the normal limit
    # that is already accurate well below float64 resolution at this shape.
    def _nan_gammaincc(*_args, **_kwargs):
        return L.torch.tensor(float('nan'), dtype=L.torch.float64)

    monkeypatch.setattr(L.torch.special, 'gammaincc', _nan_gammaincc)
    assert L._gammaincc(1e15, 1e15) == pytest.approx(0.5, abs=1e-8)
    sd = math.sqrt(1e15)
    assert L._gammaincc(1e15, 1e15 - sd) == pytest.approx(
        0.8413447460685429, abs=1e-8)
    assert L._gammaincc(1e15, 1e15 + sd) == pytest.approx(
        0.15865525393145707, abs=1e-8)


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


def test_scaled_moments_withhold_exact_values_beyond_float64():
    # float(10**400) raises OverflowError before the value can be scaled.
    # The result (variance, SE or mean) is unrepresentable, so the documented
    # fail-closed contract is None -- never an escaping int-to-float error.
    huge = 10 ** 400
    assert L._scaled_sum_of_squares([huge], 1.0) is None
    assert L._scaled_sum_of_squares([1.0, huge], 1.0) is None
    assert L._scaled_sum_of_squares([huge], huge) is None
    assert L._scaled_root_sum_of_squares([huge], 1.0) is None
    assert L._scaled_root_sum_of_squares([1.0, huge], 1.0) is None
    assert L._scaled_mean_std_if_extreme([huge, huge]) is None
    # Ordinary values still take the representable path.
    assert L._scaled_sum_of_squares([1.0, 2.0], 1.0) == pytest.approx(5.0)
    assert L._scaled_root_sum_of_squares([3.0, 4.0], 1.0) == pytest.approx(5.0)


def test_scaled_means_withhold_exact_values_beyond_float64():
    # The single-value scaled-moment helpers shared the same conversion
    # hazard the variance/SE helpers were already hardened against: the
    # historical ``np.asarray(values, dtype=float)`` raised OverflowError for
    # an exact integer beyond float64.  A mean/median/percentile/SD that
    # cannot be represented is unavailable, not an escaping conversion error.
    huge = 10 ** 400
    assert L._scaled_mean_or_raw([huge]) is None
    assert L._scaled_mean_or_raw([1.0, huge]) is None
    assert L._scaled_std_or_raw([huge, huge]) is None
    assert L._scaled_median_or_raw([huge]) is None
    assert L._scaled_percentile_or_raw([huge], (25, 75)) == (None, None)
    # A column-wise reduction cannot form any column value, so every column
    # is unavailable (NaN) instead of crashing in the initial conversion.
    out = L._scaled_mean_axis0([[huge, 1.0], [huge, 2.0]])
    assert out.shape == (2,)
    assert all(math.isnan(x) for x in out)
    # The panel-specific mean/SD wrappers share the same fail-closed contract.
    assert L._uq_mean([huge]) is None
    assert L._ls_mean_std([huge]) == (None, None)
    assert L._es_mean_sd([huge]) == (None, None)
    # Ordinary panels keep the historical bit-for-bit values.
    assert L._scaled_mean_or_raw([1.0, 2.0, 3.0]) == 2.0
    assert L._scaled_std_or_raw([1.0, 2.0, 3.0]) == 1.0
    assert L._scaled_median_or_raw([1.0, 2.0, 3.0]) == 2.0
    assert L._scaled_percentile_or_raw([1.0, 2.0, 3.0], (25, 75)) == (1.5, 2.5)


def test_row_ce_accepts_an_exact_token_total_beyond_float64():
    # float(total_tokens) used to raise before the exact-integer sibling
    # could be used.  A representable per-row CE is still recovered, while a
    # quotient that underflows is withheld rather than reported as a fake 0.
    from fractions import Fraction
    huge = 10 ** 400
    got = L._nll_row_ce([[1e308, 1e308]], huge)
    assert got is not None and got.shape == (1,)
    exact = float(sum(Fraction(1e308) for _ in range(2)) / huge)
    assert got[0] == exact
    # A per-row quotient that underflows to zero is withheld as NaN (the
    # caller treats NaN rows as an unavailable interval), never a fake 0.
    under = L._nll_row_ce([[1.0, 2.0]], huge)
    assert under is not None and math.isnan(under[0])


def test_safe_exp_ppl_keeps_the_limit_of_an_exact_ce_beyond_float64():
    # A finite but unrepresentably large cross-entropy has exp=+inf (the
    # divergent-run sentinel), and an unrepresentably negative one underflows
    # to 0.  Neither is a missing measurement, so None would be wrong.
    huge = 10 ** 400
    assert L._safe_exp_ppl(huge) == float('inf')
    assert L._safe_exp_ppl(-huge) == 0.0
    assert L._safe_exp_ppl(1000.0) == float('inf')
    assert L._safe_exp_ppl(-1000.0) == 0.0
