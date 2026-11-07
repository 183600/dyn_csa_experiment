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
import build_report as BR


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


def test_numerical_canary_size_is_validated_before_the_digit_limit():
    # The out-of-range branch used to format the raw arbitrary-precision
    # integer, so its own error message raised CPython's decimal-conversion
    # ValueError.  The named range diagnostic must survive any int size.
    for size in (10 ** 10000, 10 ** 4000, -10 ** 10000, 65, 0, 1):
        with pytest.raises(ValueError, match=r'size must be in \[2, 64\]'):
            L.numerical_canary(size=size)
    for size in (True, 2.5, '8'):
        with pytest.raises(ValueError, match='size must be an integer'):
            L.numerical_canary(size=size)


def test_global_fdr_does_not_treat_numpy_flags_as_seed_counts():
    # ``np.bool_`` is not a Python bool subclass.  Treating it as the integer
    # 1 used to invent a one-seed resolution floor, turning a malformed claim
    # into a restriction on the reported p-value.
    import numpy as np

    plain = L.global_fdr([
        {'name': 'a', 'p': 0.01, 'n': 3, 'n_nonzero': 3}])['claims']['a']
    flagged = L.global_fdr([
        {'name': 'a', 'p': 0.01, 'n': np.bool_(True),
         'n_nonzero': np.bool_(True)}])['claims']['a']
    assert flagged['n'] is None and flagged['n_nonzero'] is None
    assert flagged['p_floor'] is None
    assert flagged['at_floor'] is None
    assert plain['p_floor'] == 0.01  # ordinary integer counts still work


def test_global_fdr_two_sided_floor_is_not_doubled():
    # The exact two-sided floor for n non-zero seeds is 2 / 2**n.  An extra
    # factor of two made a resolvable family look resolution-limited.
    for n_nonzero in (2, 3, 6, 12):
        expected = 2.0 ** (1 - n_nonzero)
        assert L._gfdr_floor(None, n_nonzero=n_nonzero) == expected
        assert L.signflip_p_floor(n_nonzero) == expected
    out = L.global_fdr({'a': {'p': 0.5, 'n_nonzero': 6}}, alpha=0.05)
    assert out['claims']['a']['p_floor'] == 2.0 ** -5
    assert out['family_resolution_limited'] is False


def test_subnormal_weighted_mean_is_not_a_fake_zero():
    # ldexp() used to underflow to 0 before dividing by the weight mantissa;
    # an all-positive subnormal panel then looked like an exact zero panel.
    sub = 5e-324
    assert L._usr_weighted_mean([sub], [sub]) == sub
    out = L.unit_split_replication(
        [[sub] * 4 for _ in range(6)], [sub] * 4,
        unit_ids=['a', 'b', 'c', 'd'], seeds=list(range(6)),
        direction='positive', alpha=0.05, min_n=2,
        min_units=4, min_units_per_half=2, salt='x', n_salts=0)
    assert out['halves'][0]['stats']['n_positive'] == 6
    assert out['halves'][0]['test']['p_one_sided'] == 0.015625
    assert out['sign_agreement'] == 'both_positive'
    assert out['iut']['claim'] == 'both_halves_support_direction'


def test_curve_small_scale_condition_is_not_flattened_by_a_large_one():
    # The per-condition sign-flip p must not depend on adding an independent
    # much-larger condition.  A single global scale/tolerance used to count
    # every pattern for the small row.
    n = 6
    huge = list(range(1, n + 1))
    tiny = [1e-13 * x for x in range(1, n + 1)]
    together = L.curve_simultaneous_inference(
        ['huge', 'tiny'], {'huge': huge, 'tiny': tiny},
        seeds=list(range(n)), alpha=0.05, min_n=3)
    alone = L.curve_simultaneous_inference(
        ['tiny'], {'tiny': tiny}, seeds=list(range(n)),
        alpha=0.05, min_n=3)
    expected = 0.015625
    assert together['per_condition']['tiny']['p_pointwise_upper'] == expected
    assert (alone['per_condition']['tiny']['p_pointwise_upper']
            == expected)
    assert together['uniform']['positive']['p_iut'] == expected
    assert together['uniform']['positive']['supported'] is True


def test_sequential_obf_with_unrepresentable_alpha_fails_closed():
    # 1 - alpha/2 rounds to 1.0 below ~2.2e-16 and NormalDist.inv_cdf rejects
    # p=1.0.  The plan must return not_estimable, not crash.
    out = L.build_sequential_design(n_max=20, look_sizes=[10, 20],
                                    alpha=1e-16)
    assert out['result']['status'] == 'not_estimable'
    out = L.build_sequential_design(
        n_max=20, look_sizes=[10, 20], alpha=1e-11, family_size=100000)
    assert out['result']['status'] == 'not_estimable'


def test_canary_builder_is_strict_json_for_untrusted_provenance():
    import numpy as np

    result = L.numerical_canary(seed=1, size=4)
    for provenance in ({'code_semantics': 10 ** 10000, 'source_tree': {}},
                       {'code_semantics': float('nan'), 'source_tree': {}}):
        with pytest.raises(ValueError, match='strict-JSON'):
            L.build_numerical_canary_artifact(
                result=result, provenance=provenance)
    art = L.build_numerical_canary_artifact(
        result=result, created_at=10 ** 10000)
    json.dumps(art, allow_nan=False)
    assert isinstance(art['created_at'], str)


def test_tiny_alpha_planning_and_bh_use_relative_tolerances():
    # An absolute +1e-15 slack swamps alpha=1e-20 and both approved an
    # infeasible panel and called q=1e-19 an FDR rejection.
    assert L.signflip_min_n_for_holm(1, 1e-20, n_max=64) is None
    pvals = [('target', 1e-21)] + [(f'n{i}', 0.9) for i in range(99)]
    row = L.global_multiplicity_audit(pvals, alpha=1e-20)['entries'][0]
    assert row['bh_rejected'] is False
    assert row['global_status'] == 'family_only'
    # The report builder keeps a local twin of this audit; it used the same
    # absolute slack and called q=1e-19 an FDR rejection at alpha=1e-20.
    twin = BR.global_multiplicity_adjust(pvals, alpha=1e-20)['rows'][0]
    assert twin['bh_rejected'] is False, twin
    assert twin['global_status'] == 'family_only', twin
    # The design-sensitivity gate had the same absolute slack: at n=60 the
    # exact Holm floor is 1.73e-18, only ~1.7e-19 above a slightly smaller
    # alpha, which the old ``> a + 1e-15`` comparison called attainable.
    floor = L.holm_min_attainable_p(60, 1, 0.05)
    assert floor > 0.0
    tiny = floor * 0.9
    sensitivity = L.paired_design_sensitivity(60, 1, alpha=tiny)
    assert sensitivity['min_attainable_p_holm'] > tiny
    assert any('resolution-limited' in v for v in sensitivity['violations']), (
        sensitivity['violations'])


def test_binomial_huge_n_tiny_p_keeps_the_poisson_limit():
    # n*p == 1 is representable even though n itself is not a float.  The
    # int-to-float overflow must not silently select the normal limit.
    n, p = 1 << 1024, 2.0 ** -1024
    assert L._binomial_cdf_stable(0, n, p) == pytest.approx(
        math.exp(-1.0), abs=1e-15)
    assert L._binomial_sf_stable(1, n, p) == pytest.approx(
        1.0 - math.exp(-1.0), abs=1e-15)
    assert L._binomial_cdf(0, n, p) == pytest.approx(
        math.exp(-1.0), abs=1e-15)


def test_binomial_tail_at_huge_n_does_not_convert_the_count_distance():
    # ``n`` itself is beyond float64 and ``p`` is tiny, so n*p is an ordinary
    # large float (1e200..1e300), while the central count k=n//2 minus n*p is
    # a ~1e300 integer.  Converting that distance to float before dividing by
    # sd used to raise OverflowError inside the Edgeworth tail even though the
    # standardised z is finite (or +inf) and the tail is 0/1.  The exact-int
    # normal-z fallback must return the limiting tail instead.
    n = 10 ** 400
    for p in (1e-300, 1e-200, 1e-100, 2.0 ** -1074):
        # Central count: the integer distance k - n*p is ~n, far beyond
        # float64, while z is finite-huge or +inf.
        for k in (n // 2, n - 1):
            near_max = (k == n - 1)
            sf = L._binomial_sf_stable(k, n, p)
            cdf = L._binomial_cdf_stable(k, n, p)
            assert sf == 0.0, (p, near_max, sf)
            assert cdf == 1.0, (p, near_max, cdf)
            assert L._binomial_sf(k, n, p) == 0.0, (p, near_max)
        # A tiny index is far below the mean: the same log-pmf path must
        # report probability one instead of converting k to float.
        sf = L._binomial_sf_stable(2, n, p)
        cdf = L._binomial_cdf_stable(2, n, p)
        assert sf == 1.0, (p, sf)
        assert cdf == 0.0, (p, cdf)
        # signflip_power builds k_min ~ n/2 and passes the same tail through
        # the same dispatcher; it must report zero power, not crash.
        power = L.signflip_power(n, 4, p, 0.05)
        assert power == 0.0, (p, power)


def test_dominant_delta_uses_an_exact_mitm_count():
    # A single dominant delta defeats the saddlepoint approximation: every
    # sign assignment has |sum| >= the observed sum, so the exact p is 1.
    deltas = [1.0] + [-0.01] * 40
    out = L.exact_signflip_p(deltas)
    assert out['method'] == 'exact-mitm-dominant'
    assert out['p'] == 1.0


def test_exposure_fallback_does_not_underflow_unique_counts_to_zero():
    # 10**2000 / 10**4000 is far below float64, but the unique-token count is
    # still representable; it must not become a fake zero/100% collision.
    out = L.training_window_exposure(
        10 ** 4000, 10 ** 2000, 1, 1, max_exact_draws=4_000_000)
    assert out['exact'] is False
    assert out['unique_starts'] == 1
    assert out['unique_tokens'] == 10 ** 2000
    assert out['start_collision_fraction'] == 0.0
    json.dumps(out, allow_nan=False)
