"""Scale/tamper regressions for curve-level simultaneous inference.

The exact path normalises each condition by its max-abs scale before the
max-T/range enumeration.  The normal fallback used by larger panels must do the
same: squaring an unscaled ``1e300`` effect overflows even though the mean,
standard error, p-value and interval are representable.  These are CPU-only,
zero-experiment checks.
"""
import copy
import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

REPO = os.path.dirname(os.path.abspath(__file__))
SEEDS = list(range(21))


def _scaled_curve(scale):
    effects = {
        label: [scale * (-0.20 + 0.01 * math.sin(i + k))
                for i in range(len(SEEDS))]
        for k, label in enumerate(('a', 'b', 'c'))
    }
    return effects


def test_huge_effects_have_finite_fallback_summaries_and_replay():
    base = _scaled_curve(1.0)
    baseline = L.curve_simultaneous_inference(
        list(base), base, seeds=SEEDS, alpha=0.05, min_n=6,
        margin=0.05, flatness_margin=0.05)
    for scale in (1e100, 1e200, 1e300, 1.7e308):
        effects = _scaled_curve(scale)
        artifact = L.build_curve_inference(
            list(effects), effects, seeds=SEEDS, alpha=0.05, min_n=6,
            margin=0.05 * scale, flatness_margin=0.05 * scale)
        result = artifact['result']
        assert result['method'] == 'normal_approximation_fallback'
        row = result['per_condition']['a']
        for key in ('mean', 'sd', 'sem'):
            assert row[key] is not None, (scale, key)
            assert math.isfinite(row[key]), (scale, key, row[key])
        assert row['ci95_pointwise'] is not None
        assert all(math.isfinite(x) for x in row['ci95_pointwise'])
        assert row['p_pointwise_lower'] is not None
        assert row['p_pointwise_lower'] == pytest.approx(
            baseline['per_condition']['a']['p_pointwise_lower'])
        assert result['uniform']['negative']['supported'] is True
        verdict = L.validate_curve_inference(artifact)
        assert verdict['status'] in ('ok', 'warnings'), verdict
        assert verdict['problems'] == [], verdict
        json.dumps(artifact, allow_nan=False)
        json.dumps(verdict, allow_nan=False)


def test_replay_false_still_rejects_knob_and_shape_tamper():
    effects = {
        'a': [-0.20, -0.30, -0.10, -0.25, -0.15, -0.22],
        'b': [-0.10, -0.20, -0.15, -0.18, -0.12, -0.16],
    }
    artifact = L.build_curve_inference(
        list(effects), effects, seeds=list(range(6)), min_n=3)
    cases = [
        (('inputs', 'alpha'), 0.3),
        (('inputs', 'min_n'), 2),
        (('result', 'alpha'), 0.3),
        (('result', 'min_n'), 2),
        (('result', 'kind'), 'not_curve_inference'),
        (('result', 'schema'), 2),
        (('result', 'per_condition', 'a', 'p_adj_single'), 1.5),
    ]
    for path, value in cases:
        bad = copy.deepcopy(artifact)
        node = bad
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
        bad['content_sha256'] = L._cv_content_sha(bad)
        verdict = L.validate_curve_inference(bad, replay=False)
        assert verdict['status'] == 'fail', (path, verdict)
        assert verdict['problems'], (path, verdict)
        json.dumps(verdict, allow_nan=False)


def test_normal_fallback_upper_tail_is_not_a_fake_zero():
    # As a panel scales past the exact cap, a small SE against the mean gives
    # z ~ 9.  ``1 - NormalDist().cdf(9)`` rounds to exactly 0.0, so the old
    # fallback reported p=0 and a TOST p=0 (certainty).  The upper tail must
    # stay the positive erfc value instead.
    n = len(SEEDS)
    sd = math.sqrt(n) / 9.0          # sem = sd/sqrt(n) = 1/9, mean = 1
    effects = {'a': [1.0 - sd] * 10 + [1.0] + [1.0 + sd] * 10}
    out = L.curve_simultaneous_inference(
        ['a'], effects, seeds=SEEDS, alpha=0.05, min_n=6, margin=2.0)
    assert out['method'] == 'normal_approximation_fallback'
    row = out['per_condition']['a']
    assert row['p_pointwise_upper'] > 0.0, row
    assert row['p_pointwise_upper'] == pytest.approx(
        0.5 * math.erfc(9.0 / math.sqrt(2.0)), rel=1e-9)
    assert row['equivalence']['p_tost'] > 0.0, row
    assert out['uniform']['equivalence']['p_tost'] > 0.0, out['uniform']
    json.dumps(out, allow_nan=False)


def test_normal_fallback_critical_value_survives_alpha_below_one_ulp():
    # A large Bonferroni family can push alpha below the spacing below 1, so
    # ``1 - alpha/2`` rounds to exactly 1.0.  Clamping to 1-1e-15 silently
    # narrowed the interval; the lower-tail solve keeps the exact z.
    from statistics import NormalDist
    rng = np.random.default_rng(3)
    D = rng.normal(loc=1.0, scale=0.1, size=(2, 21))
    alpha = 1e-18
    out = L._cv_normal_enumeration(
        D, alpha=alpha, margin=None, flatness_margin=None)
    expected = -NormalDist().inv_cdf(alpha / 2.0)
    assert out['zcrit'] == pytest.approx(expected, rel=1e-15)
    assert out['zcrit'] > 8.3
    assert out['ci'] is not None
    assert np.all(np.isfinite(out['ci'])), out['ci']


def test_exact_path_critical_value_survives_alpha_below_one_ulp():
    # The exact sign-flip path still builds its pointwise t interval through
    # ``1 - alpha``; a level below the spacing at 1.0 used to make the
    # critical value NaN and silently drop the interval.  The alpha-form
    # helper must keep it finite (and very wide) instead.
    rng = np.random.default_rng(4)
    D = rng.normal(loc=1.0, scale=0.1, size=(2, 6))
    for alpha in (1e-12, 1e-18, 1e-300):
        out = L.curve_simultaneous_inference(
            ['a', 'b'], {'a': D[0].tolist(), 'b': D[1].tolist()},
            alpha=alpha, min_n=3)
        assert out['method'] == 'exact_signflip_maxt_iut', out['method']
        ci = out['per_condition']['a']['ci95_pointwise']
        assert ci is not None, (alpha, out['per_condition']['a'])
        assert all(math.isfinite(x) for x in ci), (alpha, ci)
        assert ci[0] < 1.0 < ci[1], (alpha, ci)
        json.dumps(out, allow_nan=False)


def test_seed_cap_and_unrepresentable_seed_label_fail_closed():
    with pytest.raises(ValueError, match='maximum|4096'):
        L.curve_simultaneous_inference(
            ['a'], {'a': [0.1] * 4097})
    with pytest.raises(ValueError, match='maximum|4096'):
        L.curve_simultaneous_inference(
            ['a'], {'a': [0.1] * 4097}, seeds=list(range(4097)))
    with pytest.raises(ValueError, match='strict-JSON|representable'):
        L.curve_simultaneous_inference(
            ['a'], {'a': [0.1, 0.2]}, seeds=[10 ** 10000, 1])
