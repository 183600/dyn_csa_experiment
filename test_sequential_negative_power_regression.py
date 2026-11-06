"""A one-sided ``less`` group-sequential design must report its MDE.

``q`` in the sequential sign-test planner is ``P(delta > 0)``.  A negative
alternative therefore has ``q < 0.5`` and power increases as q falls.  The
power-boundary search used to walk only [0.5, 1), so a design with real power
against a negative effect returned ``min_q_for_target_power=None`` and
withheld the MDE.  CPU-only, zero-experiment check.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403


def test_less_direction_reports_the_negative_power_boundary_and_mde():
    art = L.build_sequential_design(
        n_max=20, alpha=0.05, sides='less', pilot_sign_rate=0.20,
        pilot_sd=0.1, pilot_ppl=30.0, target_power=0.80)
    result = art['result']
    assert result['power'] > 0.80, result['power']
    q = result['min_q_for_target_power']
    assert q is not None and 0.0 < q < 0.5, q
    assert L._seq_power(result['boundaries'], q, 'less')['power'] >= 0.80
    assert L._seq_power(result['boundaries'], min(0.5, q + 1e-3),
                        'less')['power'] < 0.80
    # The boundary is the effect in the tested (negative) direction: the
    # sign-rate quantile is negative and the PPL effect is a reduction.
    assert result['mde']['normal_z'] < 0.0
    assert result['mde']['log_ppl'] < 0.0
    assert result['mde']['relative_pct'] < 0.0
    assert math.isfinite(result['mde']['log_ppl'])
    assert L.validate_sequential_design(art)['status'] == 'ok'
