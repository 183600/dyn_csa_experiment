"""Scale-invariance regressions for the effect statistics.

The sign-flip test, the sequential look correction and the meta-analysis all
consume paired effects whose unit is arbitrary (a log PPL ratio, a delta, a
rescaled panel).  Multiplying every effect by one positive constant must not
change a p-value, a resolution verdict or the zero-variance classification.
These tests exercise moderate magnitudes as well as the extreme ones already
covered by the historical suite: an absolute 1e-12/1e-9 tolerance is larger
than a 1e-13 effect, which used to make every sign assignment look tied.
"""
import math

import numpy as np

import exp_lib as L
import build_report as BR
import explib.rope as V7


_SCALES = (1e-6, 1e-12, 1e-13, 1e-50, 1e-90, 1e-100, 1e-200, 1e150)


def _assert_p_close(got, ref, where):
    assert abs(float(got) - float(ref)) <= max(1e-18, abs(float(ref)) * 1e-12), (
        f'{where}: got p={got!r}, expected {ref!r}')


def test_exact_signflip_is_scale_invariant_at_moderate_magnitudes():
    for n in (6, 21, 41):
        base = [1.0] * n
        ref = L.exact_signflip_p(base)
        ref_br = BR.exact_signflip(base)
        for scale in _SCALES:
            d = [x * scale for x in base]
            got = L.exact_signflip_p(d)
            got_br = BR.exact_signflip(d)
            _assert_p_close(got['p'], ref['p'], f'exp_lib n={n} scale={scale}')
            _assert_p_close(got_br['p_exact_signflip'], ref['p'],
                            f'build_report n={n} scale={scale}')
            assert got['method'] == ref['method'], (n, scale, got, ref)
            # The report and the audit must agree at every scale, not only at
            # the extremes already tested elsewhere.
            _assert_p_close(got_br['p_exact_signflip'], got['p'],
                            f'report vs audit n={n} scale={scale}')


def test_meta_analysis_zero_variance_is_scale_relative():
    base = [
        {'label': 'S0', 'scale': 1.0, 'seeds': [0, 1, 2, 3],
         'effects': [0.05, 0.12, 0.08, 0.10]},
        {'label': 'S1', 'scale': 10.0, 'seeds': [0, 1, 2, 3],
         'effects': [0.20, 0.18, 0.25, 0.22]},
        {'label': 'S2', 'scale': 100.0, 'seeds': [0, 1, 2, 3],
         'effects': [0.35, 0.31, 0.40, 0.36]},
    ]
    ref = L.meta_analyze_effects(base)
    assert ref['status'] == 'ok' and ref['zero_variance_points'] == [], ref
    assert ref['n_usable'] == 3, ref
    for factor in (1e-6, 1e-12, 1e-50, 1e-90, 1e-100, 1e-200, 1e150):
        scaled = [dict(p, effects=[x * factor for x in p['effects']])
                  for p in base]
        got = L.meta_analyze_effects(scaled)
        assert got['status'] == ref['status'], (factor, got)
        assert got['zero_variance_points'] == [], (factor, got)
        assert got['n_usable'] == ref['n_usable'], (factor, got)
        assert got['poolable'] == ref['poolable'], (factor, got)
        for key in ('fixed_effect', 'random_effects'):
            assert math.isclose(got[key]['mean'] / factor,
                                ref[key]['mean'], rel_tol=1e-9), (
                factor, key, got[key], ref[key])
    # A genuinely flat panel is still zero variance, at any unit.
    flat = [dict(p, effects=[0.9, 1.0, 1.1, 1.0]) for p in base]
    flat[2] = dict(flat[2], effects=[1.0, 1.0, 1.0, 1.0])
    assert L.meta_analyze_effects(flat)['zero_variance_points'] == ['S2']
    tiny_flat = [dict(p, effects=[x * 1e-12 for x in p['effects']])
                 for p in flat]
    assert L.meta_analyze_effects(tiny_flat)['zero_variance_points'] == ['S2']


def test_rope_sign_permutation_is_scale_invariant_and_keeps_dz_finite():
    base = [1.0, 0.8, 1.2, 0.9, 1.1, -0.5]
    ref = V7.exact_sign_permutation(base)
    assert ref['p_exact_signflip'] == L.exact_signflip_p(base)['p'], ref
    for scale in _SCALES:
        got = V7.exact_sign_permutation([x * scale for x in base])
        assert got.get('method', 'exact') == 'exact', (scale, got)
        _assert_p_close(got['p_exact_signflip'], ref['p_exact_signflip'],
                        f'rope permutation scale={scale}')
        # A genuinely variable tiny panel has a finite effect size; the old
        # absolute 1e-12 std floor reported an infinite d_z.
        assert math.isfinite(got['cohens_dz']), (scale, got)
    # A constant non-zero panel is a perfect (infinite) effect, not zero.
    const = V7.exact_sign_permutation([1e-13] * 4)
    assert math.isinf(const['cohens_dz']) and const['cohens_dz'] > 0, const


def test_retrieval_metrics_undeclared_prediction_lookup_uses_one_set():
    # The undeclared-prediction tally used to rebuild `set(_lab_keys)` once
    # per predicted class (O(K^2) for a large NIAH answer vocabulary).  The
    # result must not change when the lookup is hoisted out of the loop.
    labels = list(range(200))
    y_true = np.asarray(labels)
    y_pred = np.asarray(labels)
    y_pred[0] = 999  # an answer id outside the declared class set
    got = L.retrieval_metrics(y_true, y_pred, labels=labels)
    assert got['n'] == 200 and got['n_correct'] == 199, got
    assert got['n_undeclared_predictions'] == 1, got
    assert math.isclose(got['chance_accuracy'], 1.0 / 200.0), got


def test_sequential_signflip_is_scale_invariant_and_detects_tiny_drift():
    d = [-1.34, -1.18, -1.46, -1.21, -0.7, -1.1, 0.42, -0.48, 0.18, -0.04]
    looks = [(list(range(6)), d[:6]), (list(range(10)), d)]
    ref = L.sequential_signflip_p(looks)
    assert ref['method'] == 'exact-max-over-looks-final-direction', ref
    for factor in _SCALES:
        scaled = [(seeds, [x * factor for x in values])
                  for seeds, values in looks]
        got = L.sequential_signflip_p(scaled)
        assert got['method'] == ref['method'], (factor, got)
        _assert_p_close(got['p_sequential'], ref['p_sequential'],
                        f'sequential scale={factor}')
        _assert_p_close(got['p_final_look'], ref['p_final_look'],
                        f'sequential final scale={factor}')
    # A changed seed delta is inconsistent even when the deltas are tiny: the
    # tolerance is relative to the panel, not an absolute 1e-9.
    tiny_bad = [([0, 1, 2], [-1e-12, -1e-12, -1e-12]),
                ([0, 1, 2, 3], [-1e-12, -1e-12, -2e-12, -1e-12])]
    got = L.sequential_signflip_p(tiny_bad)
    assert 'inconsistent' in got['method'], got
    assert got['inconsistent_seeds'] == [2], got
