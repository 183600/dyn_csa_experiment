"""CPU review checks (split group 81): distribution-free prediction for a new seed.

A Student-t prediction interval is a model-based extrapolation: it assumes iid
normal seed effects with a common variance.  At n=12 the assumption-free
order-statistic (conformal) interval is the honest complement, but its
attainable coverage is only 11/13, not 95%.  These checks cover the exact
order-statistic construction, its finite-sample coverage, the minimum number of
seeds needed for a nominal interval, degenerate/malformed inputs, and the
nested fields exposed by ``new_seed_replication``/``paired_effect_stats``.
"""
import math
import os

import numpy as np

from test_support import *  # noqa: F401,F403


def _close(a, b, tol=1e-12):
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)), abs(float(b)))


def test_attainable_coverage_and_minimum_seed_count():
    d = [-1.00, -1.10, -0.90, -1.20, -1.00, -0.95,
         -1.05, -0.85, -1.15, -0.98, -1.02, -1.08]
    out = L.distribution_free_prediction_interval(d)
    assert out['available'] is True, out
    assert out['n'] == 12 and out['order_statistic_j'] == 1, out
    assert out['interval'] == [min(d), max(d)], out
    assert _close(out['attained_conf'], 11.0 / 13.0), out
    assert out['conf_achievable'] is False, out
    assert _close(out['coverage_shortfall'], 0.95 - 11.0 / 13.0), out
    assert out['n_needed_for_conf'] == 39, out
    assert out['n_shortfall'] == 27, out
    assert out['interval_side'] == 'below_zero', out
    assert out['n_negative'] == 12 and out['n_positive'] == 0, out
    assert out['n_zero'] == 0, out

    # n=39 reaches exactly 95% with [min, max]; n=79 trims one observation
    # per side and also reaches 95%.
    d39 = list(range(39))
    o39 = L.distribution_free_prediction_interval(d39)
    assert o39['order_statistic_j'] == 1, o39
    assert o39['interval'] == [0.0, 38.0], o39
    assert _close(o39['attained_conf'], 0.95), o39
    assert o39['conf_achievable'] is True and o39['n_shortfall'] == 0, o39
    d79 = list(range(79))
    o79 = L.distribution_free_prediction_interval(d79)
    assert o79['order_statistic_j'] == 2, o79
    assert o79['interval'] == [1.0, 77.0], o79
    assert _close(o79['attained_conf'], 0.95), o79
    assert o79['conf_achievable'] is True, o79


def test_nominal_99_percent_needs_199_exchangeable_seeds():
    out = L.distribution_free_prediction_interval(list(range(20)), conf=0.99)
    assert out['n_needed_for_conf'] == 199, out
    assert out['conf_achievable'] is False, out
    assert out['n_shortfall'] == 179, out
    # A loose nominal level is attainable with few values and still uses a
    # non-empty observed order-statistic interval.
    loose = L.distribution_free_prediction_interval([-1.0, 0.0, 2.0], conf=0.5)
    assert loose['available'] is True, loose
    assert loose['conf_achievable'] is True, loose
    assert loose['interval'] == [-1.0, 2.0], loose


def test_interval_is_scale_invariant_and_direction_aware():
    base = [-1.0, -0.8, -1.2, -0.9, -1.1, -0.95,
            -1.05, -0.85, -1.15, -0.98, -1.02, -1.08]
    ref = L.distribution_free_prediction_interval(base, direction=-1)
    assert ref['favourable_side'] is True, ref
    for factor in (1e-200, 1e-90, 1e-12, 1e6, 1e150):
        got = L.distribution_free_prediction_interval(
            [x * factor for x in base], direction=-1)
        assert got['available'] is True, (factor, got)
        lo, hi = got['interval']
        assert _close(lo / factor, ref['interval'][0], 1e-10), (factor, got)
        assert _close(hi / factor, ref['interval'][1], 1e-10), (factor, got)
        assert got['favourable_side'] is True, (factor, got)
    flipped = L.distribution_free_prediction_interval(
        [-x for x in base], direction=-1)
    assert flipped['favourable_side'] is False, flipped
    # A direction-less interval stays undefined on the benefit side.
    assert L.distribution_free_prediction_interval(base)['favourable_side'] is None
    spanning = L.distribution_free_prediction_interval(
        base + [1.0], direction=-1)
    assert spanning['favourable_side'] is None, spanning


def test_empirical_coverage_matches_the_order_statistic_bound():
    rng = np.random.default_rng(20261101)
    n, reps = 12, 30000
    x = rng.standard_t(3.0, size=(reps, n + 1))
    cal = np.sort(x[:, :n], axis=1)
    new = x[:, n]
    covered = np.mean((new >= cal[:, 0]) & (new <= cal[:, -1]))
    assert abs(float(covered) - 11.0 / 13.0) < 0.012, (covered, 11.0 / 13.0)


def test_non_finite_values_are_dropped_and_counted():
    out = L.distribution_free_prediction_interval(
        [-1.0, float('nan'), -1.2, float('inf'), -0.8, -1.1])
    assert out['available'] is True, out
    assert out['n'] == 4 and out['n_input'] == 6 and out['n_dropped'] == 2, out
    assert out['dropped_positions'] == [1, 3], out
    assert out['interval'] == [-1.2, -0.8], out


def test_small_and_constant_panels_are_refused_not_zero_width():
    one = L.distribution_free_prediction_interval([-1.0])
    assert one['available'] is False and one['interval'] is None, one
    assert 'at least 2' in one['reason'], one
    empty = L.distribution_free_prediction_interval([])
    assert empty['available'] is False and empty['n'] == 0, empty
    const = L.distribution_free_prediction_interval([-1.0] * 12)
    assert const['available'] is False and const['interval'] is None, const
    assert 'zero width' in const['reason'] or 'equal' in const['reason'], const
    # A trailing NaN does not turn a one-value panel into a usable interval.
    mixed = L.distribution_free_prediction_interval([-1.0, float('nan')])
    assert mixed['available'] is False and mixed['n'] == 1, mixed


def test_malformed_inputs_fail_closed():
    for bad in ('12', b'12', 12, None):
        if bad is None:
            # None is an empty panel, not a TypeError.
            continue
        try:
            L.distribution_free_prediction_interval(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'accepted malformed deltas {bad!r}')
    for conf in (0.0, 1.0, -0.1, float('nan'), float('inf'), True, '0.95'):
        try:
            L.distribution_free_prediction_interval([1.0, 2.0], conf=conf)
        except ValueError:
            pass
        else:
            raise AssertionError(f'accepted malformed conf {conf!r}')
    for direction in (2, -2, 0.5, True, 'up'):
        try:
            L.distribution_free_prediction_interval(
                [1.0, 2.0], direction=direction)
        except ValueError:
            pass
        else:
            raise AssertionError(f'accepted malformed direction {direction!r}')
    # direction=0 is the documented "unspecified" spelling.
    assert L.distribution_free_prediction_interval(
        [1.0, 2.0], direction=0)['direction'] is None


def test_new_seed_replication_exposes_both_methods_without_changing_the_t_law():
    d = [-1.0, -0.8, -1.2, -0.9, -1.1, -0.95,
         -1.05, -0.85, -1.15, -0.98, -1.02, -1.08]
    pred = L.new_seed_replication(d)
    assert pred['available'] is True, pred
    df = pred['distribution_free']
    assert df['available'] is True, df
    assert pred['distribution_free_interval'] == df['interval'], pred
    assert pred['distribution_free_attained_conf'] == df['attained_conf'], pred
    assert pred['distribution_free_conf_achievable'] is False, pred
    assert pred['distribution_free_n_needed'] == 39, pred
    t_lo, t_hi = pred['prediction_interval']
    df_lo, df_hi = df['interval']
    assert df_lo == min(d) and df_hi == max(d), (pred, df)
    assert t_lo <= t_hi and df_lo <= df_hi, (pred, df)
    # The two intervals answer the same question under different assumptions;
    # neither is required to contain the other.
    assert (t_lo, t_hi) != (df_lo, df_hi), (pred, df)
    # The Student-t fields are untouched by the new nested block.
    mean = float(np.mean(d))
    sd = float(np.std(d, ddof=1))
    scale = sd * math.sqrt(1.0 + 1.0 / len(d))
    assert _close(pred['prediction_scale'], scale, 1e-15), pred


def test_paired_effect_stats_and_reportable_claim_echo_the_new_fields():
    d = [-1.0, -0.8, -1.2, -0.9, -1.1, -0.95,
         -1.05, -0.85, -1.15, -0.98, -1.02, -1.08]
    st = L.paired_effect_stats(d)
    assert st['distribution_free_prediction_95'] == [min(d), max(d)], st
    assert _close(st['distribution_free_attained_conf'], 11.0 / 13.0), st
    assert st['distribution_free_conf_achievable'] is False, st
    assert st['distribution_free_n_needed'] == 39, st
    claim = L.reportable_claim(st)
    assert claim['distribution_free_prediction_95'] == [min(d), max(d)], claim
    assert _close(claim['distribution_free_attained_conf'], 11.0 / 13.0), claim
    # A malformed echoed interval is dropped, never promoted to a gate.
    bad = dict(st)
    bad['distribution_free_prediction_95'] = [1.0]
    out = L.reportable_claim(bad)
    assert out.get('distribution_free_prediction_95') is None, out


def test_strict_json_and_replay_are_deterministic():
    import json
    d = [-1.0, -0.8, -1.2, -0.9, -1.1, -0.95]
    a = L.distribution_free_prediction_interval(d, conf=0.8, direction=1)
    b = L.distribution_free_prediction_interval(d, conf=0.8, direction=1)
    assert json.dumps(a, allow_nan=False, sort_keys=True) == json.dumps(
        b, allow_nan=False, sort_keys=True)
    pred = L.new_seed_replication(d)
    json.dumps(pred, allow_nan=False, sort_keys=True)


def test_audit_gate_replays_the_stored_artifact_and_refuses_tampering():
    import json
    import tempfile
    import audit_gates as A

    d = [-1.0, -0.8, -1.2, -0.9, -1.1, -0.95,
         -1.05, -0.85, -1.15, -0.98, -1.02, -1.08]
    artifact = L.build_distribution_free_prediction(d)
    assert artifact['kind'] == 'distribution_free_prediction_artifact', artifact
    assert artifact['inputs']['deltas'] == d, artifact
    result = artifact['result']
    with tempfile.TemporaryDirectory() as td:
        good = os.path.join(td, 'good.json')
        with open(good, 'w', encoding='utf-8') as fh:
            json.dump(artifact, fh)
        audit = A.audit_distribution_free_prediction(good)
        assert audit['status'] == 'ok', audit
        assert audit['interval'] == [min(d), max(d)], audit
        assert audit['attained_conf'] is not None, audit

        bad = json.loads(json.dumps(artifact))
        bad['result']['interval'] = [0.0, 1.0]
        bad_path = os.path.join(td, 'bad.json')
        with open(bad_path, 'w', encoding='utf-8') as fh:
            json.dump(bad, fh)
        audit = A.audit_distribution_free_prediction(bad_path)
        assert audit['status'] == 'fail', audit

        frozen = json.loads(json.dumps(artifact))
        frozen['result']['confirmed'] = True
        frozen_path = os.path.join(td, 'frozen.json')
        with open(frozen_path, 'w', encoding='utf-8') as fh:
            json.dump(frozen, fh)
        audit = A.audit_distribution_free_prediction(frozen_path)
        assert audit['status'] == 'fail', audit
        assert any('frozen-decision' in p for p in audit['problems']), audit

        missing = A.audit_distribution_free_prediction(
            os.path.join(td, 'absent.json'))
        assert missing['status'] == 'not_verifiable_locally', missing

        unavailable = {
            'kind': 'distribution_free_prediction_artifact',
            'schema': 1,
            'inputs': {'deltas': [1.0, 1.0], 'conf': 0.95, 'direction': None},
            'result': L.distribution_free_prediction_interval([1.0, 1.0]),
        }
        unavailable_path = os.path.join(td, 'unavailable.json')
        with open(unavailable_path, 'w', encoding='utf-8') as fh:
            json.dump(unavailable, fh)
        audit = A.audit_distribution_free_prediction(unavailable_path)
        assert audit['status'] == 'not_applicable', audit


def test_artifact_builder_requires_finite_replayable_inputs():
    artifact = L.build_distribution_free_prediction(
        [-1.0, -0.8, -1.2, -0.9], conf=0.9, direction=-1)
    assert artifact['inputs'] == {
        'deltas': [-1.0, -0.8, -1.2, -0.9], 'conf': 0.9, 'direction': -1}, artifact
    assert artifact['result']['available'] is True, artifact
    for bad in ([-1.0, float('nan')], [-1.0, float('inf')]):
        try:
            L.build_distribution_free_prediction(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'accepted non-finite artifact input {bad!r}')
