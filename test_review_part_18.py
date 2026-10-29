"""CPU review checks (split group 18): prospective new-seed replication.

A confidence interval for the mean paired effect is routinely read as if it
were a prediction for the next seed.  These checks cover the separate
Student-t predictive readout: its interval must be wider than the mean CI, its
sign probability must be scale-invariant and match an independent t sampler,
and degenerate/one-seed panels must be refused instead of turned into
zero-width certainty.
"""
import math
import os
import sys as _sys

import numpy as np

_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403


def _close(a, b, tol=1e-12):
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)), abs(float(b)))


def test_prediction_interval_is_wider_than_the_mean_ci_and_matches_the_t_formula():
    d = [-1.00, -1.10, -0.90, -1.20, -1.00, -0.95]
    pred = L.new_seed_replication(d)
    assert pred['available'] is True, pred
    assert pred['n'] == 6 and pred['df'] == 5, pred
    mean = float(np.mean(d))
    sd = float(np.std(d, ddof=1))
    scale = sd * math.sqrt(1.0 + 1.0 / len(d))
    tc = L._t_critical(0.95, len(d) - 1)
    assert _close(pred['mean'], mean, 1e-15), pred
    assert _close(pred['std'], sd, 1e-15), pred
    assert _close(pred['prediction_scale'], scale, 1e-15), pred
    lo, hi = pred['prediction_interval']
    assert _close(lo, mean - tc * scale, 1e-12), pred
    assert _close(hi, mean + tc * scale, 1e-12), pred

    st = L.paired_effect_stats(d)
    mean_lo, mean_hi = st['ci95_low'], st['ci95_high']
    assert lo < mean_lo < mean_hi < hi, (pred, st)
    assert (hi - lo) > (mean_hi - mean_lo), (pred, st)

    # Sign probabilities come from the same predictive t law and partition 1.
    assert 0.5 < pred['prob_same_sign'] < 1.0, pred
    assert _close(pred['prob_new_positive'] + pred['prob_new_negative'], 1.0, 1e-15)
    assert _close(pred['prob_same_sign'], pred['prob_new_negative'], 1e-15)


def test_predictive_sign_probability_matches_an_independent_t_sampler():
    d = [-1.0, -0.8, -1.2, -0.9, -1.1]
    pred = L.new_seed_replication(d)
    rng = np.random.default_rng(20261030)
    draws = rng.standard_t(pred['df'], size=200000)
    x = pred['mean'] + draws * pred['prediction_scale']
    empirical_positive = float(np.mean(x > 0.0))
    empirical_same = float(np.mean(np.sign(x) == np.sign(pred['mean'])))
    assert abs(empirical_positive - pred['prob_new_positive']) < 0.005, (
        empirical_positive, pred)
    assert abs(empirical_same - pred['prob_same_sign']) < 0.005, (empirical_same, pred)


def test_prediction_is_scale_invariant_and_reversal_is_symmetric():
    base = [-1.0, -0.8, -1.2, -0.9, -1.1, -0.95]
    ref = L.new_seed_replication(base)
    for factor in (1e-200, 1e-90, 1e-12, 1e6, 1e150):
        got = L.new_seed_replication([x * factor for x in base])
        assert got['available'] is True, (factor, got)
        assert _close(got['prob_new_positive'], ref['prob_new_positive'], 1e-12), (
            factor, got, ref)
        assert _close(got['prob_same_sign'], ref['prob_same_sign'], 1e-12), (
            factor, got, ref)
        lo, hi = got['prediction_interval']
        assert _close(lo / factor, ref['prediction_interval'][0], 1e-10), (
            factor, got, ref)
        assert _close(hi / factor, ref['prediction_interval'][1], 1e-10), (
            factor, got, ref)
    flipped = L.new_seed_replication([-x for x in base])
    assert _close(flipped['prob_same_sign'], ref['prob_same_sign'], 1e-12), (
        flipped, ref)
    assert _close(flipped['prob_new_positive'], ref['prob_new_negative'], 1e-12)
    assert _close(flipped['prob_new_negative'], ref['prob_new_positive'], 1e-12)


def test_prediction_direction_probability_is_not_inverted():
    d = [-1.0, -1.2, -0.8, -1.1, -0.9, -1.0]
    lower_is_better = L.new_seed_replication(d, direction=-1)
    higher_is_better = L.new_seed_replication(d, direction=+1)
    assert lower_is_better['prob_improvement'] > 0.5, lower_is_better
    assert higher_is_better['prob_improvement'] < 0.5, higher_is_better
    assert _close(lower_is_better['prob_improvement'],
                  lower_is_better['prob_new_negative'], 1e-15)
    assert _close(higher_is_better['prob_improvement'],
                  higher_is_better['prob_new_positive'], 1e-15)
    assert L.new_seed_replication(d)['prob_improvement'] is None


def test_degenerate_panels_are_refused_instead_of_zero_width_certainty():
    for d, why in (([], 'empty'), ([1.0], 'one seed'), ([2.0, 2.0, 2.0], 'constant')):
        pred = L.new_seed_replication(d)
        assert pred['available'] is False, (why, pred)
        assert pred['prediction_interval'] is None, (why, pred)
        assert pred['prob_new_positive'] is None, (why, pred)
        assert pred['reason'], (why, pred)
    # A symmetric panel has a zero mean: the "same sign" question is undefined
    # even though the positive/negative sign probabilities are both 0.5.
    sym = L.new_seed_replication([-1.0, 1.0, -0.5, 0.5])
    assert sym['available'] is True and sym['mean'] == 0.0, sym
    assert sym['prob_same_sign'] is None, sym
    assert _close(sym['prob_new_positive'], 0.5, 1e-15), sym


def test_malformed_options_and_payloads_fail_closed():
    for bad in (True, False, 0.0, 1.0, 1.5, float('nan')):
        try:
            L.new_seed_replication([1.0, 2.0], conf=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'conf={bad!r} accepted')
    for bad in (True, 2, -2, 0.5, float('nan')):
        try:
            L.new_seed_replication([1.0, 2.0], direction=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'direction={bad!r} accepted')
    try:
        L.new_seed_replication('12')
    except ValueError:
        pass
    else:
        raise AssertionError('a string payload was accepted')
    # Non-finite draws are dropped and the accounting stays explicit; a
    # remaining pair is still usable.
    pred = L.new_seed_replication([-1.0, float('nan'), -1.2, float('inf')])
    assert pred['n'] == 2 and pred['n_input'] == 4 and pred['n_dropped'] == 2, pred
    assert pred['available'] is True, pred


def test_paired_effect_stats_exposes_prediction_and_claim_echoes_it():
    d = [-1.0, -1.2, -0.8, -1.1, -0.9, -1.0]
    st = L.paired_effect_stats(d, min_n=6)
    pred = st['new_seed_replication']
    assert pred['available'] is True, pred
    assert st['prediction_interval_95'] == pred['prediction_interval'], st
    assert st['prob_new_seed_same_sign'] == pred['prob_same_sign'], st
    # The predictive fields are additive: the exact-test verdict is unchanged.
    assert st['n'] == 6 and st['p_exact_signflip'] == 2.0 / 2 ** 6, st

    claim = L.reportable_claim(dict(st, p_holm=st['p_exact_signflip'],
                                    min_n=6), require_holm=False, min_n=6)
    assert claim['allowed'] is True, claim
    assert claim['prediction_interval_95'] == pred['prediction_interval'], claim
    assert claim['prob_new_seed_same_sign'] == pred['prob_same_sign'], claim
    # A malformed advisory field is dropped, never promoted into a gate.
    bad = L.reportable_claim(dict(st, p_holm=st['p_exact_signflip'],
                                  prediction_interval_95=[1.0],
                                  prob_new_seed_same_sign=2.0),
                             require_holm=False, min_n=6)
    assert bad['allowed'] is True, bad
    assert bad['prediction_interval_95'] is None, bad
    assert bad['prob_new_seed_same_sign'] is None, bad
