"""Mixed exact-integer/float panels at scale.

The exact-integer fallbacks must not assume that every value is a Python
``int``.  A real panel can mix an exact count beyond float64 with an ordinary
floating measurement; converting that huge int to float raises, and even
Python's object arithmetic raises on ``huge_int + 1.0``.  These helpers must
keep the exact value (or report the documented unavailable statistic), not
crash or silently drop the observation.
"""
import json
import math
from fractions import Fraction

import numpy as np
import pytest

import exp_lib as L


HUGE = 10 ** 400


def test_signed_sum_table_handles_a_mixed_exact_and_float_panel():
    values = [HUGE, 1.0]
    sums = L._signflip_signed_sums(values)
    assert [Fraction(v) for v in sums] == [
        Fraction(HUGE + 1), Fraction(-HUGE + 1),
        Fraction(HUGE - 1), Fraction(-HUGE - 1)]
    # The exact TOST table reuses the same primitive.
    assert [Fraction(v) for v in L._etost_signed_sums(values)] == [
        Fraction(v) for v in sums]
    # A fractional term is still exact, not rounded through float.
    half = L._signflip_signed_sums([HUGE, 0.5])
    assert [Fraction(v) for v in half] == [
        Fraction(2 * HUGE + 1, 2), Fraction(-2 * HUGE + 1, 2),
        Fraction(2 * HUGE - 1, 2), Fraction(-2 * HUGE - 1, 2)]
    # Ordinary float panels keep the historical float table bit-for-bit.
    assert L._signflip_signed_sums([1.0, 2.0]).tolist() == [3.0, 1.0, -1.0, -3.0]


def test_length_strata_summary_counts_an_unrepresentable_observation():
    summary = L._ls_summary([HUGE, 1.0], 0.95)
    # The huge value is an observation; dropping it would describe a one-seed
    # panel and make its mean look representable.
    assert summary['n'] == 2
    assert summary['mean'] is None and summary['std'] is None
    assert summary['ci95'] is None
    json.dumps(summary, allow_nan=False)
    # A genuinely non-finite value is still dropped, not counted.
    assert L._ls_summary([float('nan'), 1.0], 0.95)['n'] == 1
    # Ordinary panels keep the historical result.
    ordinary = L._ls_summary([1.0, 2.0, 3.0], 0.95)
    assert ordinary['n'] == 3 and ordinary['mean'] == 2.0
    assert ordinary['ci95'] is not None


def test_validation_transfer_derived_ratios_accept_exact_integers():
    ratio, shrinkage = L._vt_ratio_and_shrinkage(HUGE, HUGE + 1)
    assert ratio == 1.0 and shrinkage == 0.0
    ratio, shrinkage = L._vt_ratio_and_shrinkage(HUGE, 2 * HUGE)
    assert ratio == 2.0 and shrinkage == pytest.approx(-100.0)
    assert L._vt_direction_agreement(HUGE, HUGE + 1) is True
    assert L._vt_direction_agreement(HUGE, -HUGE) is False
    assert L._vt_ratio_and_shrinkage(0.0, 1.0) == (None, None)
    assert L._vt_ratio_and_shrinkage(1.0, HUGE) == (None, None)
    # Ordinary values keep the historical ratio.
    assert L._vt_ratio_and_shrinkage(2.0, 1.0) == (0.5, 50.0)


def test_family_display_helpers_do_not_abort_on_exact_integers():
    assert L._fc_num(HUGE).lstrip('+') == str(HUGE)
    assert L._fc_num(HUGE, '{:.2f}').lstrip('+') == str(HUGE)
    assert L._fc_ci([HUGE, HUGE + 1]) == f'[{HUGE:+d}, {HUGE + 1:+d}]'
    # Ordinary values keep the historical formatting.
    assert L._fc_num(1.25) == '1.2500'
