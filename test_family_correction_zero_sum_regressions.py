"""Exact family-correction floor for non-zero contrasts with zero sign sum.

``p_floor`` is a property of the observed *magnitudes* (the 2**n sign patterns
of the exact test), not of the observed signed sum.  A contrast such as
``[1, -1, 1, -1, ...]`` has a zero observed score, so its observed p is 1, but
its non-zero support still gives the test a floor of ``2/2**n``.  Reading the
zero score as "no informative support" made the floor jump to 1.0 and marked a
perfectly resolvable family as resolution-limited.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403


def _family(values, seeds=None):
    seeds = list(range(len(values))) if seeds is None else seeds
    return {'a': {'per_seed': list(values), 'seeds': seeds,
                  'p_exact_signflip': L.exact_signflip_p(values)['p']}}


def test_zero_sum_contrast_keeps_its_nonzero_floor():
    values = [1.0 if i % 2 == 0 else -1.0 for i in range(12)]
    assert sum(values) == 0.0
    exact = L.exact_signflip_p(values)
    out = L.confirmation_family_correction(
        _family(values), family_names=['a'])
    assert out['status'] == 'ok', out
    assert out['comparisons']['a']['p_raw_score'] == 1.0
    # 12 non-zero entries: 2/2**12, not 1.0 from the zero observed score.
    assert out['p_floor'] == pytest.approx(exact['p_floor'])
    assert out['p_floor'] == pytest.approx(2.0 ** -11)
    assert out['holm_gate_attainable'] is True
    json.dumps(out, allow_nan=False)


def test_zero_sum_padding_is_still_sign_neutral():
    active = [1.0 if i % 2 == 0 else -1.0 for i in range(12)]
    padded = active + [0.0] * 20
    base = L.confirmation_family_correction(
        _family(active), family_names=['a'])
    pad = L.confirmation_family_correction(
        _family(padded), family_names=['a'])
    assert pad['n_seeds'] == len(padded)
    assert pad['n_seeds_effective'] == len(active)
    assert pad['p_floor'] == pytest.approx(base['p_floor'])
    assert pad['holm_gate_attainable'] == base['holm_gate_attainable']
    assert pad['comparisons']['a']['p_maxt_score'] == pytest.approx(
        base['comparisons']['a']['p_maxt_score'])


def test_all_zero_family_still_reports_no_resolution():
    out = L.confirmation_family_correction(
        {'a': {'per_seed': [0.0] * 8, 'seeds': list(range(8)),
               'p_exact_signflip': 1.0}},
        family_names=['a'])
    assert out['status'] == 'degenerate', out
    assert out['p_floor'] == 1.0
    assert out['holm_gate_attainable'] is False
