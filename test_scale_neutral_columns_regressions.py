"""Scale guards for sign-neutral zero columns and NumPy scalar flags.

A future panel can carry one column/row per planned seed, including seeds whose
measured effect is exactly zero (a tie or an unused slot).  For the sign-flip
statistics a zero column cannot change any signed sum: it only duplicates every
pattern.  The exact-enumeration branch and its cap must therefore use the
effective (non-zero) width, while still reporting the declared width.  This is
the same convention the exact sign-flip p-value already uses.

The scalar checks pin the adjacent Python/NumPy type boundary: ``np.bool_`` is
not a ``bool`` subclass, but it is a flag, not a p-value, a seed count or an
integer index.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

import numpy as np
import pytest

ACTIVE = [-0.20, 0.15, -0.10, 0.25, -0.05, 0.12,
          -0.18, 0.22, -0.08, 0.30, -0.14, 0.18]
PAD = 20
PADDED = ACTIVE + [0.0] * PAD


def _band_rows(values):
    return [
        {'label': 'early',
         'per_seed': [{'seed': i, 'log_ratio': v}
                      for i, v in enumerate(values)]},
        {'label': 'late',
         'per_seed': [{'seed': i, 'log_ratio': -0.7 * v}
                      for i, v in enumerate(values)]},
    ]


def test_location_p_zero_padding_uses_effective_width():
    base = L.exact_signflip_location_p(ACTIVE, 0.0)
    padded = L.exact_signflip_location_p(PADDED, 0.0)
    assert padded['p'] == pytest.approx(base['p'])
    assert padded['n'] == len(PADDED)
    assert padded['n_nonzero'] == len(ACTIVE)
    assert padded['n_effective'] == len(ACTIVE)

    # At a non-zero null a zero delta is no longer sign-neutral after
    # centring: the full declared width must be enumerated, and it is refused
    # instead of hanging on 2**32 patterns.
    with pytest.raises(ValueError, match='location-p inversion hard cap'):
        L.exact_signflip_location_p(PADDED, 0.1)
    with pytest.raises(ValueError, match='location-p inversion hard cap'):
        L.exact_signflip_location_p(list(range(1, 20)), 0.0)


def test_position_band_fallback_ignores_zero_columns():
    base = L._lpd_max_band_signflip(_band_rows(ACTIVE), alpha=0.05)
    padded = L._lpd_max_band_signflip(_band_rows(PADDED), alpha=0.05)
    assert base['status'] == padded['status'] == 'ok'
    assert padded['n'] == len(PADDED)
    assert padded['n_effective'] == len(ACTIVE)
    assert padded['p_family'] == pytest.approx(base['p_family'])
    for label in ('early', 'late'):
        assert padded['contrasts'][label]['p_maxt'] == pytest.approx(
            base['contrasts'][label]['p_maxt'])


def test_conditional_power_table_ignores_zero_deltas():
    qs = [0.5, 0.8]
    base = V12._signflip_power_table(ACTIVE, 6, 0.05, qs, 'positive')
    padded = V12._signflip_power_table(PADDED, 6, 0.05, qs, 'positive')
    assert padded['n'] == len(PADDED)
    assert padded['n_nonzero'] == len(ACTIVE)
    assert padded['n_effective'] == len(ACTIVE)
    assert padded['enumerated'] == 1 << len(ACTIVE)
    for key in base['power']:
        assert padded['power'][key] == pytest.approx(base['power'][key])
    assert (padded['min_same_sign_for_alpha']
            == base['min_same_sign_for_alpha'])
    assert padded['min_attainable_p_holm'] == pytest.approx(
        base['min_attainable_p_holm'])


def test_unit_split_zero_rows_do_not_consume_the_exact_cap():
    active = [
        [-0.20, -0.10, -0.30, -0.15],
        [0.10, -0.20, -0.05, -0.10],
        [-0.30, 0.10, -0.20, -0.25],
        [-0.10, -0.15, -0.20, -0.05],
        [-0.25, -0.05, -0.30, -0.10],
        [-0.15, -0.20, -0.10, -0.30],
    ]
    units = [10, 20, 30, 40]
    base = L.unit_split_replication(active, units, direction='negative')
    padded = L.unit_split_replication(
        active + [[0.0] * len(units)] * PAD, units, direction='negative')
    assert base['status'] == padded['status'] == 'warnings'
    assert padded['n_seeds'] == len(active) + PAD
    assert padded['iut']['p_iut'] == pytest.approx(base['iut']['p_iut'])
    # Informative rows still consume the cap: 21 non-zero rows (above the
    # default max_n=20) are refused rather than enumerated.
    rows = [list(row) for row in active]
    for i in range(15):
        rows.append([-0.01 * (i + 1), 0.02, -0.03, 0.01])
    out = L.unit_split_replication(rows, units, direction='negative')
    assert out['status'] == 'not_estimable'
    assert 'exact sign-flip cap' in (out.get('reason') or '')


def test_numpy_flags_are_not_probabilities_counts_or_indices():
    plain = L.global_fdr([{'name': 'a', 'p': 0.01}])['claims']['a']
    for flag in (np.bool_(True), np.bool_(False)):
        claim = L.global_fdr([{'name': 'a', 'p': flag}])['claims']['a']
        assert claim['p'] is None, claim
        assert claim['reject'] is False, claim
        assert claim['rank'] is None, claim
    assert plain['p'] == 0.01
    for flag in (np.bool_(True), np.bool_(False)):
        with pytest.raises(ValueError):
            L._gfdr_floor(flag)
        with pytest.raises(ValueError, match='expected an integer'):
            L._arb_int(flag, where='index')
