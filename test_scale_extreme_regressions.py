"""Regression checks for exact tails and fallbacks at extreme scale.

These are CPU-only checks for failure modes that only appear once an integer
count is larger than float64 (or larger than CPython's decimal-conversion
limit), and for a degenerate zero-variance normal fallback.
"""
import copy
import json
import math

import pytest

import exp_lib as L


def test_boolean_schema_is_not_coerced_to_version_one():
    # True == 1, so a schema comparison alone would accept a boolean flag as
    # the current plan/artifact format.
    plan = L.build_run_disposition_plan(
        ['a::seed0'], created_at='2020-01-01T00:00:00')
    with pytest.raises(ValueError, match='schema'):
        L.run_disposition_audit([], plan=dict(plan, schema=True))

    effects = {
        'a': [0.10, 0.20, 0.30, -0.10, 0.05, 0.15],
        'b': [0.00, 0.10, 0.20, -0.20, 0.00, 0.10],
    }
    artifact = L.build_curve_inference(
        list(effects), effects, seeds=list(range(6)), min_n=3)
    bad = copy.deepcopy(artifact)
    bad['result']['schema'] = True
    bad['content_sha256'] = L._cv_content_sha(bad)
    verdict = L.validate_curve_inference(bad, replay=False)
    assert verdict['status'] == 'fail', verdict
    assert any('schema' in problem for problem in verdict['problems']), verdict
    json.dumps(verdict, allow_nan=False)


def test_binomial_central_cdf_survives_sqrt_overflow():
    # For these even n, sqrt(n) is not representable as a float.  The exact
    # normal-limit helper must still return the central z ~ 1/sqrt(n), not
    # None: returning None used to turn P(X <= n/2) into 1.0 instead of 0.5.
    for bits in (2049, 4097, 5000):
        n = (1 << (bits - 1)) + 2
        n -= n % 2
        z = L._binom_normal_z_half(n, n // 2)
        assert z is not None and math.isfinite(z), (bits, z)
        cdf = L._binom_cdf_half(n, n // 2)
        log_cdf = L._log_binom_cdf_half(n, n // 2)
        assert abs(cdf - 0.5) < 1e-12, (bits, cdf)
        assert abs(log_cdf + math.log(2.0)) < 1e-12, (bits, log_cdf)
        lower = L._binom_cdf_half(n, n // 2 - 1)
        # At this scale the lower side rounds to the same float as the
        # centre; the important property is that it is a probability, not
        # the old symmetry artefact 0.0/1.0.
        assert 0.0 < lower <= 0.5, (bits, lower)


def test_signflip_planning_is_not_feasible_at_half_the_seeds():
    # The old sqrt-overflow failure made the central CDF collapse to 1.0 and
    # made signflip_min_consistency accept k = n/2 + 1.  The real gate needs
    # about n/2 + z*sqrt(n)/2 same-sign seeds.
    n = 1 << 4096  # even, bit_length 4097
    k = L.signflip_min_consistency(n, 10, 0.05)
    assert k is not None
    assert k > n // 2
    assert (2 * k - n).bit_length() > 100
    assert 20.0 * L._binom_cdf_half(n, n - k) <= 0.05 + 1e-15


def test_signflip_power_at_sqrt_overflow_scale_is_finite_and_consistent():
    for bits in (1025, 2049, 4097):
        n = 1 << (bits - 1)
        k = L.signflip_min_consistency(n, 10, 0.05)
        power = L.signflip_power(n, 10, 0.5, 0.05)
        tail = L._binom_cdf_half(n, n - k)
        assert math.isfinite(power), (bits, power)
        assert abs(power - tail) < 1e-12, (bits, power, tail)
        for q in (0.51, 0.55, 0.9, 0.99):
            value = L.signflip_power(n, 10, q, 0.05)
            assert 0.0 <= value <= 1.0 and math.isfinite(value), (
                bits, q, value)


def test_zero_variance_normal_fallback_withholds_tost_readouts():
    n = 25  # above the exact-enumeration cap, so the normal fallback runs
    identical = {
        'a': [float(i) for i in range(n)],
        'b': [float(i) for i in range(n)],
    }
    out = L.curve_simultaneous_inference(
        ['a', 'b'], identical, flatness_margin=0.1)
    assert out['method'] == 'normal_approximation_fallback'
    eq = out['flatness']['equivalence']
    assert eq['p_tost'] is None, eq
    assert eq['supported'] is False, eq
    json.dumps(out, allow_nan=False)

    # A constant condition has a zero SEM; its per-condition TOST must be
    # withheld rather than reported as a zero-variance equivalence.
    constant = {'a': [1.0] * n, 'b': [float(i) for i in range(n)]}
    out = L.curve_simultaneous_inference(
        ['a', 'b'], constant, margin=0.1)
    assert out['per_condition']['a']['equivalence']['p_tost'] is None
    assert out['per_condition']['a']['equivalence']['equivalent'] is False
    assert out['uniform']['equivalence']['p_tost'] is None
    assert out['uniform']['equivalence']['supported'] is False
    json.dumps(out, allow_nan=False)


def test_public_binomial_wrappers_survive_huge_n():
    n = 1 << 4096
    central = L._binomial_cdf(n // 2, n, 0.5)
    upper = L._binomial_sf(n // 2 + 1, n, 0.5)
    assert abs(central - 0.5) < 1e-12, central
    assert abs(upper - 0.5) < 1e-12, upper
    assert abs(central + upper - 1.0) < 1e-12


def test_training_exposure_derived_products_are_named_not_digit_limit():
    big = 10 ** 4000
    # Each factor is at the accepted input boundary; the product is not
    # spellable by json.dumps.  It must fail with a named ValueError, not
    # CPython's raw integer-to-decimal conversion error.
    with pytest.raises(ValueError, match='strict-JSON'):
        L.training_window_exposure(
            big, 1, big, big, max_exact_draws=10)
    plan = L.training_coverage_budget(
        big, 1, big, target_coverage=0.9,
        max_steps=big, world_size=big)
    assert plan['status'] == 'not_estimable', plan
    assert 'strict-JSON' in (plan.get('reason') or ''), plan
    json.dumps(plan, allow_nan=False)
