"""CPU review checks (split group 19): unit-scale invariance of design and
rank-based planning outputs.

The effect/PL difference unit is arbitrary.  A statistic that compares values
with an absolute tolerance, or that squares deviations with ``numpy.std``,
must not change its answer when the whole panel is multiplied by a positive
constant.  These checks cover the two remaining planning paths where a tiny
(or huge) panel used to be misclassified as degenerate: the exact sign-flip
conditional-power table and the seed/evaluation-unit precision designs.
"""
import math
import os
import sys as _sys

import numpy as np

_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

_SCALES = (1e-6, 1e-12, 1e-13, 1e-50, 1e-100, 1e-150, 1e-200, 1e-300,
           1e100, 1e150, 1e300)


def _close(a, b, rel=1e-11):
    a, b = float(a), float(b)
    return abs(a - b) <= rel * max(abs(a), abs(b), 1e-300)


def test_signflip_power_table_is_scale_invariant():
    base = [0.2 + 0.11 * i for i in range(8)]
    qs = [0.5, 0.8, 0.95, 1.0]
    ref = V12._signflip_power_table(base, 6, 0.05, qs, 'positive')
    for factor in _SCALES:
        got = V12._signflip_power_table([x * factor for x in base], 6, 0.05,
                                        qs, 'positive')
        assert got['n'] == ref['n'] and got['enumerated'] == ref['enumerated']
        assert got['min_same_sign_for_alpha'] == ref['min_same_sign_for_alpha'], (
            factor, got, ref)
        assert _close(got['min_attainable_p_holm'],
                      ref['min_attainable_p_holm']), (factor, got, ref)
        for key in ref['power']:
            assert _close(got['power'][key], ref['power'][key]), (
                factor, key, got['power'], ref['power'])


def test_seed_pilot_design_survives_extreme_delta_units():
    base = [-1.0, -0.8, -1.2, -0.9, -1.1, -0.95]
    ref = L.paired_design_sensitivity(8, 6, pilot_deltas=base)
    assert ref['status'] == 'ok' and ref['pilot']['sd'] > 0.0, ref
    for factor in (1e-200, 1e-150, 1e-100, 1e-12, 1e100, 1e150, 1e300):
        got = L.paired_design_sensitivity(
            8, 6, pilot_deltas=[x * factor for x in base])
        assert got['status'] == 'ok', (factor, got)
        assert _close(got['pilot']['mean'] / factor, ref['pilot']['mean']), (
            factor, got['pilot'], ref['pilot'])
        assert _close(got['pilot']['sd'] / factor, ref['pilot']['sd']), (
            factor, got['pilot'], ref['pilot'])
        assert _close(got['precision']['ci95_halfwidth'] / factor,
                      ref['precision']['ci95_halfwidth']), (factor, got)
        assert _close(got['mde']['absolute'] / factor,
                      ref['mde']['absolute']), (factor, got)
        assert _close(got['min_q_for_target_power'],
                      ref['min_q_for_target_power'], rel=1e-10), (factor, got)


def test_unit_pilot_design_survives_extreme_delta_units():
    base = [-1.0, -0.8, -1.2, -0.9, -1.1, -0.95]
    ref = L.unit_precision_design(100, pilot_unit_deltas=base)
    assert ref['status'] == 'ok' and ref['pilot']['sd'] > 0.0, ref
    for factor in (1e-300, 1e-200, 1e-150, 1e-100, 1e-12, 1e100, 1e150,
                   1e300):
        got = L.unit_precision_design(
            100, pilot_unit_deltas=[x * factor for x in base])
        assert got['status'] == 'ok', (factor, got)
        assert _close(got['pilot']['mean'] / factor, ref['pilot']['mean']), (
            factor, got['pilot'], ref['pilot'])
        assert _close(got['pilot']['sd'] / factor, ref['pilot']['sd']), (
            factor, got['pilot'], ref['pilot'])
        for key in ('ci_halfwidth', 'ci95_halfwidth'):
            assert _close(got['precision'][key] / factor,
                          ref['precision'][key]), (factor, key, got)
        assert _close(got['mde']['absolute'] / factor,
                      ref['mde']['absolute']), (factor, got)


def test_explicit_pilot_sd_consistency_is_relative_to_the_unit():
    base = [-1.0, -0.8, -1.2, -0.9, -1.1, -0.95]
    for factor in (1.0, 1e-200, 1e150):
        d = [x * factor for x in base]
        ref = L.unit_precision_design(100, pilot_unit_deltas=d)
        sd = ref['pilot']['sd']
        assert sd > 0.0, (factor, ref)
        same = L.unit_precision_design(100, pilot_unit_deltas=d,
                                       pilot_unit_sd=sd)
        assert not any('overrides the SD' in w for w in same['warnings']), (
            factor, same['warnings'])
        # A genuinely different SD (twice the recomputed value) is still
        # reported at every unit scale, including tiny ones.
        diff = L.unit_precision_design(100, pilot_unit_deltas=d,
                                       pilot_unit_sd=sd * 2.0)
        assert any('overrides the SD' in w for w in diff['warnings']), (
            factor, diff['warnings'])


def test_paired_unit_deltas_large_ppl_panel_keeps_a_finite_sd():
    # PPL values around 1e300 make the naive squared deviations overflow to
    # inf even though the paired mean/SD are representable.  The unit is
    # arbitrary, so this must still produce a usable pilot, not an
    # "unusable variance" refusal.
    big = 1e300
    rows_a = [{'ppl_per_seq': [1.0 + big * 1.1, 1.0 + big * 1.2,
                               1.0 + big * 0.9]}]
    rows_b = [{'ppl_per_seq': [1.0 + big * 1.0, 1.0 + big * 1.1,
                               1.0 + big * 0.8]}]
    got = L.paired_unit_deltas(rows_a, rows_b, unit='row')
    assert got['status'] == 'ok', got
    assert math.isfinite(got['mean']) and got['mean'] > 0.0, got
    assert math.isfinite(got['sd']) and got['sd'] > 0.0, got
