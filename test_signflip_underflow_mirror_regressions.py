"""The report's exact sign-flip mirror must survive a scaling underflow.

Both implementations normalise the panel by its largest magnitude before the
exact/MITM comparison.  With a genuinely extreme panel, a small but non-zero
delta can underflow to +/-0.0 during that division.  The library retains the
column (masking the *original* values), which is correct: a zero column doubles
both the exact count and the denominator.  The report mirror filtered the
scaled list and then divided by the original non-zero count, so its p was too
small by 2**(underflowed columns); at n<=20 the exact branch also broadcast a
shorter value array against the full sign mask and raised.
"""

import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403


def _underflow_panel(n_nonzero):
    # Three huge values ensure the scaling factor is ~1e300, so 1e-300
    # underflows to zero; the moderate values keep p away from 0/1.
    return ([1e300] * 3 + [1.0] * 9
            + [-1.0] * (n_nonzero - 3 - 9 - 1) + [1e-300])


@pytest.mark.parametrize('n_nonzero', [19, 20, 21, 23])
def test_report_mirror_keeps_underflowed_columns(n_nonzero):
    deltas = _underflow_panel(n_nonzero)
    assert len(deltas) == n_nonzero
    assert sum(1 for x in deltas if x != 0.0) == n_nonzero
    # The scaling step really does underflow the tiny delta.
    assert 1e-300 / max(abs(x) for x in deltas) == 0.0

    exact = L.exact_signflip_p(deltas)
    mirror = BR.exact_signflip(deltas)
    assert mirror is not None
    assert mirror['p_exact_signflip'] == pytest.approx(exact['p'])
    assert mirror['p_floor'] == pytest.approx(exact['p_floor'])
    assert mirror['n_nonzero'] == exact['n_nonzero']
    assert mirror['method'] == exact['method']
    assert 0.0 < mirror['p_exact_signflip'] < 1.0
    json.dumps(mirror, allow_nan=False)


def test_report_mirror_does_not_crash_on_exact_branch_underflow():
    # n=20 uses the full 2**20 enumeration, whose sign mask is sized from
    # n_nonzero; an underflow-shortened value array used to raise here.
    deltas = _underflow_panel(20)
    mirror = BR.exact_signflip(deltas)
    assert mirror['method'] == 'exact'
    assert mirror['p_exact_signflip'] == pytest.approx(0.25)
