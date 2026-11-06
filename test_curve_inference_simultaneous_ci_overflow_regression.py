"""An unrepresentable simultaneous CI endpoint is withheld, not ``[null, null]``.

The normal fallback multiplies the unitless interval by each condition's
max-abs scale.  A finite panel near float64's limit can have a representable
mean/SEM but an endpoint beyond the float range; emitting ``[None, None]``
reads like a reported interval (and differs from the documented "unavailable"
value ``None``).  CPU-only, zero-experiment check.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403


def test_overflowing_simultaneous_ci_is_none():
    n = 21
    values = [1.7e308] * 20 + [-1.7e308]
    effects = {'a': values, 'b': values}
    out = L.curve_simultaneous_inference(
        list(effects), effects, seeds=list(range(n)), min_n=3)
    assert out['method'] == 'normal_approximation_fallback'
    for row in out['per_condition'].values():
        assert row['mean'] is not None
        assert row['sem'] is not None
        # The pointwise t interval is also unrepresentable here; the important
        # contract is that neither field is a half-null placeholder.
        assert row['ci95_simultaneous'] is None
        assert row['ci95_pointwise'] is None
