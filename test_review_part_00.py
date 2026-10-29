"""CPU review checks (split group 00; t_holm_min_attainable .. t_scale_missing_seed_identity_is_not_paired)."""
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

if not _HAVE_TORCH:
    import pytest as _pytest
    _pytest.skip('CPU torch dependencies are unavailable', allow_module_level=True)

def test_holm_min_attainable():
    assert abs(L.holm_min_attainable_p(6, 4, 0.05) - 0.125) < 1e-12, '6 seeds x Holm/4 floor must be 0.125'
    assert abs(L.holm_min_attainable_p(12, 4, 0.05) - 0.001953125) < 1e-12
    assert abs(L.holm_min_attainable_p(8, 1, 0.05) - 0.0078125) < 1e-12
    assert L.holm_min_attainable_p(6, 4, 0.05) > 0.05 and L.holm_min_attainable_p(12, 4, 0.05) < 0.05
    # Duplicate names cannot share one dict slot: rejecting is the only
    # representation that does not shrink/corrupt the family.
    try:
        L.holm_adjust([('a', 0.01), ('a', 0.02)])
    except ValueError:
        pass
    else:
        raise AssertionError('duplicate comparison names must be rejected')


def test_min_consistency():
    assert L.signflip_min_consistency(12, 4, 0.05) == 11, 'n=12, m=4: 11/12 same-sign clears the first Holm gate (raw p = 2*(1+12)/4096 = 0.00635 < 0.0125)'
    assert L.signflip_min_consistency(6, 4, 0.05) is None, 'n=6, m=4: even 6/6 cannot clear the gate'
    assert L.signflip_min_consistency(8, 1, 0.05) == 8
    assert L.signflip_min_consistency(6, 1, 0.05) == 6
    # Holm rejects at p_holm <= alpha.  The planning helper must use the same
    # boundary: at n=3, m=1, alpha=0.25 the attainable floor IS 0.25.
    assert L.signflip_min_consistency(3, 1, 0.25) == 3
    assert L.signflip_power(3, 1, 1.0, 0.25) == 1.0
    assert L.signflip_min_consistency(5, 1, 0.0625) == 5
    assert L.signflip_min_consistency(3, 1, 0.25 - 1e-12) is None
    # Scale-up: the planning math must not overflow int->float at large n.
    big = L.signflip_min_consistency(2000, 10, 0.05)
    assert big is not None and 0 < big < 2000, big
    # The same applies to a huge comparison family: a 401-digit m must not
    # turn into an OverflowError or an underflowed "p=0" pass.
    huge_m = 10 ** 400
    assert L.holm_min_attainable_p(12, huge_m, 0.05) == 1.0
    assert L.signflip_min_consistency(12, huge_m, 0.05) is None
    assert L.signflip_power(12, huge_m, 0.95, 0.05) == 0.0
    assert L.signflip_min_n_for_holm(huge_m, 0.05, n_max=64) is None
    assert L.signflip_min_n_for_holm(huge_m, 0.05, n_max=2000) == 1335
    assert 0.0 < L.holm_min_attainable_p(1400, huge_m, 0.05) < 0.05
    assert L.signflip_min_consistency(1400, huge_m, 0.05) == 1393
    # The planning alpha is a probability threshold, not a coerced float:
    # True->1.0 and a string could otherwise change the required seed count.
    for _bad_alpha in (True, 0, 1.0, 1.5, float('nan'), 'not-a-number'):
        try:
            L.signflip_min_consistency(12, 4, _bad_alpha)
        except ValueError:
            pass
        else:
            raise AssertionError(f'signflip_min_consistency accepted alpha={_bad_alpha!r}')
        try:
            L.signflip_power(12, 4, 0.9, _bad_alpha)
        except ValueError:
            pass
        else:
            raise AssertionError(f'signflip_power accepted alpha={_bad_alpha!r}')
        try:
            L.signflip_min_n_for_holm(4, _bad_alpha)
        except ValueError:
            pass
        else:
            raise AssertionError(f'signflip_min_n_for_holm accepted alpha={_bad_alpha!r}')


def test_binomial_tail_at_scale():
    n = 10 ** 8
    mean = n // 2
    logpmf = (math.lgamma(n + 1.0) - 2.0 * math.lgamma(mean + 1.0)
              - n * math.log(2.0))
    exact = 0.5 + 0.5 * math.exp(logpmf)
    got = L._binom_cdf_half(n, mean)
    assert abs(got - exact) < 1e-7, f'n=1e8 central binomial CDF: {got!r} vs {exact!r}'
    k = L.signflip_min_consistency(n, 10, 0.05)
    assert k is not None and 0 < k < n, k
    tail = L._binom_cdf_half(n, n - k)
    prev = L._binom_cdf_half(n, n - k + 1)
    assert 20.0 * tail <= 0.05 + 1e-15, f'k_min={k} does not clear the gate'
    assert 20.0 * prev > 0.05 + 1e-15, f'k_min={k} is not minimal'
    power = L.signflip_power(n, 10, 0.5, 0.05)
    assert abs(power - tail) < 1e-12, f'q=0.5 power {power!r} != tail {tail!r}'


def test_binomial_tail_large_scale():
    # The incomplete-beta continued fraction hits its 1e6-step cap around
    # n~1e14; the central CDF then drifted by ~3e-10 (n=1e14) and ~1e-7
    # (n=1e16).  At these scales the continuity correction is itself only
    # O(1/sqrt(n)), so a path that drops it is visible even at the mean.
    for n in (10 ** 10, 10 ** 12, 10 ** 14, 10 ** 16, 10 ** 17, 10 ** 18):
        got = L._binom_cdf_half(n, n // 2)
        ref = 0.5 + 1.0 / math.sqrt(2.0 * math.pi * n)
        assert abs(got - ref) < 1e-13, (n, got, ref)
    # Above 2**53 a float count loses the +1 that distinguishes the mean from
    # mean+1; both the public wrappers and the beta identity must keep it.
    n = 10 ** 17
    cdf_mean = L._binom_cdf_half(n, n // 2)
    assert abs(L._binomial_cdf(n // 2, n, 0.5) - cdf_mean) < 1e-15
    assert abs(L._binomial_sf(n // 2 + 1, n, 0.5) - (1.0 - cdf_mean)) < 1e-15
    # A tiny p with enormous n is the Poisson corner: 1-p rounds to 1 in
    # float, so lgamma/log(q) cancellation used to return e.g. 6e-17 instead
    # of 0.632 at n=1e16.  The exact small-tail path avoids log C(n,k).
    for n, p, tol in ((10 ** 12, 1e-12, 5e-12),
                      (10 ** 16, 1e-16, 1e-12)):
        lam = n * p
        ref1 = 1.0 - math.exp(-lam)
        ref2 = 1.0 - math.exp(-lam) * (1.0 + lam)
        got1 = L._binomial_sf(1, n, p)
        got2 = L._binomial_sf(2, n, p)
        assert abs(got1 - ref1) < tol, (n, p, got1, ref1)
        assert abs(got2 - ref2) < tol, (n, p, got2, ref2)
    # The integer-parameter beta identity I_x(a,b)=P(Bin(a+b-1,x)>=a)
    # must preserve both symmetry and the large-n central value.
    for a in (10 ** 15, 10 ** 18):
        assert abs(L._betai(a, a, 0.5) - 0.5) < 1e-15
        assert abs(L._betai(a, a + 1, 0.5)
                   + L._betai(a + 1, a, 0.5) - 1.0) < 1e-15
        assert 0.5 < L._betai(a, a + 1, 0.5) < 0.5000001
    # The planning helpers must stay consistent with the same tail primitive.
    for n in (10 ** 14, 10 ** 18):
        k = L.signflip_min_consistency(n, 10, 0.05)
        assert k is not None and 0 < k < n, k
        tail = L._binom_cdf_half(n, n - k)
        assert 20.0 * tail <= 0.05 + 1e-12
        power = L.signflip_power(n, 10, 0.5, 0.05)
        assert abs(power - tail) < 1e-12, (power, tail)
    # Public wrappers must take p directly.  Passing 1-p into the beta helper
    # and folding the complement back in (1-(1-p)) loses ~eps/p relative
    # accuracy: at n=1e16, p=1e-12 the old lower tail was 0.5035421 instead of
    # the high-precision 0.5026596 (mpmath, 80 digits).  Keep both tails exact
    # enough that they still partition the sample space at this scale.
    _n, _p, _k = 10 ** 16, 1e-12, 10000
    assert abs(L._binomial_cdf(0, 1000, 0.5) - 0.5 ** 1000) < 1e-300
    assert abs(L._binomial_cdf(_k, _n, _p) - 0.5026595812190084) < 1e-11
    assert abs(L._binomial_cdf(_k, _n, _p)
               + L._binomial_sf(_k + 1, _n, _p) - 1.0) < 1e-12
    # Just below the mean the complement route folded x through 1-x and
    # returned 0.4792308 instead of 0.4827166 at n=1e16, p=1e-12, k=9995.
    # A second scale keeps the exact lower-sum fallback covered.
    assert abs(L._binomial_cdf(9995, 10 ** 16, 1e-12)
               - 0.4827166222951466) < 1e-9
    assert abs(L._binomial_cdf(999000, 10 ** 15, 1e-9)
               - 0.15877629969068482) < 1e-9
    # The planning power uses the same direct upper tail; it must not
    # subtract 1-(1-q) anywhere on the way back out.
    _kmin = L.signflip_min_consistency(2000, 10, 0.05)
    for _q in (0.5, 0.55, 0.9, 0.999999):
        assert abs(L.signflip_power(2000, 10, _q, 0.05)
                   - L._binomial_sf_stable(_kmin, 2000, _q)) < 1e-15
    # The log helper must agree with the float CDF at the centre.  Its
    # large-m geometric tail approximation was only valid far below the
    # mean; at m=n//2 it returned a positive "probability" (or raised a
    # domain error through log1p(-exp(positive))).
    for n in (50000, 100000, 1000000):
        cdf = L._binom_cdf_half(n, n // 2)
        log_cdf = L._log_binom_cdf_half(n, n // 2)
        assert 0.0 < cdf < 1.0 and math.isfinite(log_cdf), (n, cdf, log_cdf)
        assert abs(log_cdf - math.log(cdf)) < 1e-12, (n, log_cdf, cdf)
    # n near the largest float and beyond it must still be evaluable: the
    # beta path's int->float conversion overflows there, so the normal
    # limit keeps the planning tail defined instead of raising.
    for n in (2 ** 1024 - 1, 10 ** 309):
        assert abs(L._binom_cdf_half(n, n // 2) - 0.5) < 1e-12, n
        assert abs(L._log_binom_cdf_half(n, n // 2) + math.log(2.0)) < 1e-12, n
        k = L.signflip_min_consistency(n, 1, 0.05)
        assert k is not None and 0 < k < n, (n, k)
        assert 2.0 * L._binom_cdf_half(n, n - k) <= 0.05 + 1e-15, (n, k)
        assert 2.0 * L._binom_cdf_half(n, n - k + 1) > 0.05 + 1e-15, (n, k)
    assert L.holm_min_attainable_p(2 ** 1024 - 1, 1, 0.05) == 0.0
    assert math.isfinite(L._log_binom_cdf_half(2 ** 1024 - 1, 20001))


def test_power():
    assert L.signflip_power(6, 4, 0.99) == 0.0, 'an impossible plan has zero power at any q'
    pw = L.signflip_power(12, 4, 0.9, 0.05)
    assert abs(pw - 0.659) < 0.01, f'q=0.9 power at first Holm gate should be ~0.659, got {pw:.4f}'
    assert L.signflip_power(12, 4, 0.8, 0.05) < 0.30
    pw_big = L.signflip_power(2000, 10, 0.55, 0.05)
    assert 0.0 < pw_big <= 1.0 and math.isfinite(pw_big), pw_big


def test_tost():
    inside = [0.001, -0.002, 0.0005, -0.0009, 0.0012, -0.0003]
    r = L.tost_paired(inside, margin=0.01)
    assert r['equivalent'] is True and r['p_tost'] < 0.05, f'tiny deltas vs margin must be equivalent: {r}'
    above = [0.05, 0.052, 0.049, 0.051, 0.05, 0.048]
    r2 = L.tost_paired(above, margin=0.01)
    assert r2['equivalent'] is False and r2['p_tost'] > 0.05, f'mean far above margin must NOT be equivalent: {r2}'
    # direction check:
    far = [0.9, 0.95, 0.85, 1.0, 0.92, 0.88]
    r3 = L.tost_paired(far, margin=0.1)
    assert r3['equivalent'] is False, 'mean ~0.9 vs margin 0.1 must not be equivalent even with noise'
    small_n = L.tost_paired([0.0, 0.0], margin=0.1)
    assert small_n['equivalent'] is not None


def test_tost_student_t_and_boundary():
    inside = [0.001, -0.002, 0.0005, -0.0009, 0.0012, -0.0003]
    r = L.tost_paired(inside, margin=0.01)
    assert r['equivalent'] is True, r
    assert r['df'] == 5 and 'Student t' in r['method'], r
    se = float(np.std(inside, ddof=1) / math.sqrt(len(inside)))
    tc = L._t_critical(0.90, 5)
    assert abs((r['ci'][1] - r['mean']) - tc * se) < 1e-12, 'TOST CI must use the t critical value'
    assert (r['ci'][1] - r['mean']) > 1.6448536269514722 * se, 'at n=6 the t interval must be wider than the normal-z interval'
    # Zero-variance sample exactly at the margin is the t limit P(T>=0)=0.5,
    # not certainty: equality at the boundary must never be called equivalent.
    for boundary in ([0.1] * 6, [-0.1] * 6):
        b = L.tost_paired(boundary, margin=0.1)
        assert b['equivalent'] is False and abs(b['p_tost'] - 0.5) < 1e-12, b
    # The t critical value must invert the t tail for high confidence and
    # small/fractional df too; a fixed 1000 cap used to return a too-narrow CI.
    for _df, _conf in ((0.5, 0.99), (1.0, 0.9999), (10.0, 0.999)):
        _tc = L._t_critical(_conf, _df)
        assert math.isfinite(_tc), (_df, _conf, _tc)
        assert abs(L._student_t_sf(_tc, _df) - (1.0 - _conf) / 2.0) <= 1e-10, (
            _df, _conf, _tc)


def test_chi2_large_df_tail():
    assert abs(L._gammaincc(500000.0, 500000.0) - 0.4998119368033945) < 1e-11
    assert abs(L.chi2_sf(1_000_000.0, 1_000_000.0) - 0.4998119368033945) < 1e-11
    assert abs(L.chi2_sf(10_000_000.0, 10_000_000.0) - 0.4999405291960622) < 1e-11
    assert abs(L.chi2_sf(3.841458820694124, 1.0) - 0.05) < 1e-10
    # Scale-up: the recurrence's `a + 1` used to round back to `a` (b == 0,
    # ZeroDivisionError), and the direct series needed billions of terms and
    # lost accuracy before that.  Both huge-shape cases must stay fast and
    # accurate (reference values from SciPy's gammaincc).
    assert abs(L._gammaincc(1e15, 1e15) - 0.4999999957947791) < 1e-12
    assert abs(L._gammaincc(1e16, 1e16) - 0.4999999986701924) < 1e-12
    assert abs(L.chi2_sf(1e16, 1e16) - 0.49999999811936807) < 1e-12


def test_safe_exp_and_doc_ci():
    # A finite exponent must stay a finite PPL; only an actual IEEE overflow
    # yields inf.
    assert math.isfinite(L._safe_exp_ppl(705.0))
    assert L._safe_exp_ppl(1000.0) == float('inf')
    assert L._safe_exp_ppl(float('nan')) != L._safe_exp_ppl(float('nan'))  # NaN stays NaN
    # Token-pooled CE and its interval must target the same estimand even when
    # document lengths differ.
    nll = [10.0, 1.0, 4.0]
    ntok = [100, 10, 20]
    ci = L._nll_doc_ci(nll, ntok)
    ce = sum(nll) / sum(ntok)
    assert abs(math.log(ci['ppl']) - ce) < 1e-12, ci
    assert ci['ppl_ci95_low'] <= ci['ppl'] <= ci['ppl_ci95_high'], ci
    assert ci['ci_method'].startswith('cluster-robust'), ci
    assert ci['n_docs'] == 3
    # NumPy arrays are the natural per-document inputs, and a one-shot
    # iterator must not be exhausted before the residual pass.  Neither input
    # may raise or report a zero document count.
    ci_np = L._nll_doc_ci(np.array(nll), np.array(ntok))
    assert ci_np['n_docs'] == 3 and abs(ci_np['ppl'] - ci['ppl']) < 1e-12, ci_np
    ci_gen = L._nll_doc_ci(iter(nll), iter(ntok))
    assert ci_gen['n_docs'] == 3 and abs(ci_gen['ppl'] - ci['ppl']) < 1e-12, ci_gen
    # A single scalar pair is one document, not an unhandled TypeError.
    ci_scalar = L._nll_doc_ci(10.0, 100)
    assert ci_scalar['n_docs'] == 1, ci_scalar
    assert abs(math.log(ci_scalar['ppl']) - 0.1) < 1e-12, ci_scalar
    # The same iterable contract holds for the curve diagnostics.
    assert L._finite_curve(np.array([1.0, np.inf, 2.0]))['n_finite'] == 2
    assert L._finite_curve(iter([1.0, 2.0]))['n_finite'] == 2
    # PPL is >= 1 by definition: a symmetric CE interval that crosses zero
    # must not be exponentiated into an impossible sub-1 lower bound.
    _wide = L._nll_doc_ci([6.369616873214543, 2.697867137638703], [1, 1])
    assert _wide['ppl_ci95_low'] >= 1.0, _wide
    assert _wide['ppl_ci95_low'] <= _wide['ppl'] <= _wide['ppl_ci95_high'], _wide
    # Mismatched per-document accounting is malformed evidence, not an
    # invitation to silently zip to the shorter list.
    try:
        L._nll_doc_ci([1.0, 2.0], [10])
    except ValueError as e:
        assert 'paired' in str(e), e
    else:
        raise AssertionError('mismatched NLL/token lists must be refused')


def test_bootstrap_mean_ci():
    vals = [0.5, 1.0, 1.5, 2.0, 2.5]
    a = L.bootstrap_mean_ci(vals, n_boot=500, seed=7)
    b = L.bootstrap_mean_ci(vals, n_boot=500, seed=7)
    assert a == b and a['n'] == 5 and a['ci_low'] <= a['mean'] <= a['ci_high'], a
    assert L.bootstrap_mean_ci([1.0]) is None
    assert L.bootstrap_mean_ci([1.0, float('nan'), float('inf')]) is None
    # Paired/unpaired helpers are chunked for seed panels that can reach
    # hundreds of seeds; the result stays deterministic and well-formed.
    rng = np.random.default_rng(3)
    d = rng.normal(0.5, 1.0, size=400)
    p1 = L.paired_bootstrap_ci(d, n_boot=600, seed=11)
    p2 = L.paired_bootstrap_ci(d, n_boot=600, seed=11)
    assert p1 == p2 and p1[0] < float(d.mean()) < p1[1], (p1, d.mean())
    u1 = L.unpaired_bootstrap_ci(d + 1.0, d, n_boot=600, seed=11)
    u2 = L.unpaired_bootstrap_ci(d + 1.0, d, n_boot=600, seed=11)
    assert u1 == u2 and u1[0] < 1.0 < u1[1], u1


def test_training_health_classifier():
    # Overfitting: train CE falls while val CE rises in the tail.
    hist = [{'step': i * 100, 'train_ce': 6.0 - 0.30 * i, 'val_ce': 4.0 + 0.05 * i,
             'val_ppl': 55.0 + 0.5 * i} for i in range(6)]
    h = L.classify_training_health(hist)
    assert h['status'] == 'overfitting', h
    assert h['best_step'] is not None and h['final_minus_best'] is not None, h
    # Underfitting/stalled: neither curve decreases.
    hist = [{'step': i * 100, 'train_ce': 4.0 + 0.001 * i, 'val_ce': 4.0 + 0.001 * i,
             'val_ppl': 60.0 + 0.01 * i} for i in range(6)]
    h = L.classify_training_health(hist)
    assert h['status'] == 'underfitting_or_stalled', h
    # Healthy: train falls, val does not rise.
    hist = [{'step': i * 100, 'train_ce': 6.0 - 0.30 * i, 'val_ce': 4.0 - 0.02 * i,
             'val_ppl': 55.0 - 0.5 * i} for i in range(6)]
    h = L.classify_training_health(hist)
    assert h['status'] == 'healthy', h
    # A MEASURED non-finite point must be flagged...
    hist = [{'step': i * 100, 'train_ce': 6.0 - 0.30 * i, 'val_ce': 4.0 - 0.02 * i,
             'val_ppl': 55.0 - 0.5 * i} for i in range(6)]
    hist[3]['val_ppl'] = float('nan')
    hist[3]['val_ce'] = float('nan')
    h = L.classify_training_health(hist)
    assert h['status'] == 'nonfinite' and h['n_nonfinite'] == 2, h
    # ...but an UNMONITORED train CE (None) must not be miscounted as
    # non-finite: the val-only classification speaks, and the reason says
    # the train curve is missing.
    hist = [{'step': i * 100, 'train_ce': None, 'val_ce': 4.0 - 0.05 * i,
             'val_ppl': 55.0 - 0.5 * i} for i in range(6)]
    h = L.classify_training_health(hist)
    assert h['status'] == 'healthy' and h['n_nonfinite'] == 0, h
    assert 'train CE unmonitored' in h['reason'] and h['n_train_ce_measured'] == 0, h
    hist = [{'step': i * 100, 'train_ce': None, 'val_ce': 4.0 + 0.05 * i,
             'val_ppl': 55.0 + 0.5 * i} for i in range(6)]
    h = L.classify_training_health(hist)
    assert h['status'] == 'validation_degrading' and h['n_nonfinite'] == 0, h
    hist = [{'step': i * 100, 'train_ce': None, 'val_ce': 4.0, 'val_ppl': 55.0} for i in range(6)]
    h = L.classify_training_health(hist)
    assert h['status'] == 'underfitting_or_stalled', h
    # A train CE that was measured and came back NaN is still nonfinite.
    hist = [{'step': i * 100, 'train_ce': (float('nan') if i == 2 else 6.0 - 0.3 * i),
             'val_ce': 4.0 - 0.02 * i, 'val_ppl': 55.0 - 0.5 * i} for i in range(6)]
    h = L.classify_training_health(hist)
    assert h['status'] == 'nonfinite' and h['n_nonfinite'] == 1, h
    # Validation improving while training CE is flat must NOT be labelled
    # "validation_degrading" (the old else branch inverted the direction).
    hist = [{'step': i * 100, 'train_ce': 4.0, 'val_ce': 4.0 - 0.05 * i,
             'val_ppl': 55.0 - 0.5 * i} for i in range(6)]
    h = L.classify_training_health(hist)
    assert h['status'] == 'validation_improving_train_flat', h
    assert 'validation CE is falling' in h['reason'], h
    assert any('not read this as overfitting' in a for a in h['actions']), h
    # Too few points stays insufficient regardless of which side is missing.
    h = L.classify_training_health([{'step': 100, 'train_ce': 5.0, 'val_ce': 4.0, 'val_ppl': 55.0}])
    assert h['status'] == 'insufficient_data', h
    h = L.classify_training_health([])
    assert h['status'] == 'insufficient_data', h


def test_training_health_noise_aware_threshold():
    # Four equally spaced eval points with realistic CE jitter: the tail slope
    # is +3.2e-5, five times the fixed 1e-5 tolerance, yet it is entirely
    # within the slope's own noise (2*SE ~ 3.4e-5).  The adaptive threshold
    # must call it healthy.
    hist = [{'step': 0, 'train_ce': 5.0, 'val_ce': 4.00, 'val_ppl': 54.6},
            {'step': 250, 'train_ce': 4.5, 'val_ce': 4.02, 'val_ppl': 55.7},
            {'step': 500, 'train_ce': 4.0, 'val_ce': 4.01, 'val_ppl': 55.1},
            {'step': 750, 'train_ce': 3.5, 'val_ce': 4.03, 'val_ppl': 56.3}]
    h = L.classify_training_health(hist)
    assert h['status'] == 'healthy', h
    assert h['slope_tol_mode'] == 'adaptive_2se', h
    assert h['val_ce_slope_se'] is not None and h['val_slope_tol'] > 1e-5, h
    assert h['val_ce_slope_per_step'] > 1e-5, h
    # The historical fixed threshold is still available explicitly and still
    # calls the same noise "overfitting" (proving the test is discriminating).
    h_fixed = L.classify_training_health(hist, slope_tol=1e-5)
    assert h_fixed['status'] == 'overfitting', h_fixed
    assert h_fixed['slope_tol_mode'] == 'fixed', h_fixed
    # A genuine rising trend must still be caught even under the adaptive rule.
    hist_bad = [dict(hh, val_ce=4.0 + 0.01 * i, val_ppl=55.0 + 0.5 * i)
                for i, hh in enumerate(hist)]
    assert L.classify_training_health(hist_bad)['status'] == 'overfitting'


def test_training_health_actions_and_gap():
    # The classifier must not stop at a label: overfitting needs concrete
    # next steps and the widening train-val gap must be measured, not implied.
    hist = [{'step': i * 100, 'train_ce': 6.0 - 0.30 * i,
             'val_ce': 4.0 + 0.05 * i, 'val_ppl': 55.0 + 0.5 * i,
             'gap': (4.0 + 0.05 * i) - (6.0 - 0.30 * i)} for i in range(6)]
    h = L.classify_training_health(hist)
    assert h['status'] == 'overfitting', h
    assert h.get('actions'), h
    assert any('regularizer' in a for a in h['actions']), h
    assert h['gap_slope_per_step'] is not None and h['gap_slope_per_step'] > 0.0, h
    assert h['steps_since_best'] is not None and h['steps_since_best'] > 0, h
    assert any('best-validation' in a for a in h['actions']), h
    stalled = [{'step': i * 100, 'train_ce': 4.0, 'val_ce': 4.0,
                'val_ppl': 55.0, 'gap': 0.0} for i in range(6)]
    hs = L.classify_training_health(stalled)
    assert hs['status'] == 'underfitting_or_stalled', hs
    assert any('data pipeline' in a for a in hs['actions']), hs
    bad = [{'step': i * 100, 'train_ce': 4.0, 'val_ce': 4.0,
            'val_ppl': float('nan')} for i in range(6)]
    hn = L.classify_training_health(bad)
    assert hn['status'] == 'nonfinite', hn
    assert any('NaN triage' in a for a in hn['actions']), hn
    rising = [{'step': i * 100, 'train_ce': 4.0 + 0.02 * i,
               'val_ce': 4.0 + 0.02 * i, 'val_ppl': 55.0 + 0.5 * i} for i in range(6)]
    hr = L.classify_training_health(rising)
    assert hr['train_ce_rising'] is True, hr
    assert any('training CE itself is rising' in a for a in hr['actions']), hr


def test_health_accepts_val_ce_only_and_orders_steps():
    # The API documents ``val_ce`` OR ``val_ppl``.  A CE-only curve must still
    # produce the best/final checkpoint evidence the report uses, and the gap
    # must be derived from the same two measurements.
    hist = [{'step': i * 100, 'train_ce': 6.0 - 0.30 * i,
             'val_ce': 4.0 + 0.05 * i} for i in range(5)]
    h = L.classify_training_health(hist)
    assert h['status'] == 'overfitting', h
    assert h['val_ppl_source'] == 'val_ce_derived', h
    assert h['best_step'] == 0 and h['best_val_ppl'] is not None, h
    assert abs(h['best_val_ppl'] - math.exp(4.0)) < 1e-9, h
    assert h['final_val_ppl'] is not None and h['final_minus_best'] > 0, h
    assert h['gap_source'] == 'derived' and h['gap_last'] is not None, h
    assert h['gap_slope_source'] == 'derived', h
    assert h['gap_slope_per_step'] is not None and h['gap_slope_per_step'] > 0, h

    # Out-of-order history and a repeated step are one evaluation grid, not a
    # different best step or a two-point noise fit.  The last write to a step
    # wins (resume semantics), and the tail is chronological.
    shuffled = [hist[2], hist[0], hist[4], hist[1], hist[3],
                {'step': 100, 'train_ce': 6.0, 'val_ce': 3.5}]
    hs = L.classify_training_health(shuffled)
    assert hs['n_points'] == 6 and hs['n_eval_points'] == 5, hs
    assert hs['best_step'] == 100, hs
    assert abs(hs['best_val_ppl'] - math.exp(3.5)) < 1e-9, hs
    assert hs['steps_since_best'] == 300, hs

    # An explicitly recorded non-finite PPL is a measured failure: the finite
    # CE twin must not paper over it by becoming the best PPL.
    bad = [{'step': 0, 'train_ce': 5.0, 'val_ce': 4.0, 'val_ppl': 55.0},
           {'step': 100, 'train_ce': 4.5, 'val_ce': 4.1, 'val_ppl': float('nan')},
           {'step': 200, 'train_ce': 4.0, 'val_ce': 4.2, 'val_ppl': 60.0}]
    hb = L.classify_training_health(bad)
    assert hb['status'] == 'nonfinite', hb
    assert hb['val_ppl_source'] == 'val_ppl', hb

    # Early stopping accepts the same CE-only representation.
    ce_only = [{'step': 0, 'val_ce': 4.0}, {'step': 100, 'val_ce': 4.1},
               {'step': 200, 'val_ce': 4.2}]
    d = L.early_stop_decision(ce_only, patience=2)
    assert d['should_stop'] is True and d['best_step'] == 0, d
    assert d['val_ppl_source'] == 'val_ce_derived', d
    assert abs(d['best_val_ppl'] - math.exp(4.0)) < 1e-9, d
    # Two evaluation points cannot establish a trend; the label is kept but
    # explicitly marked as interpolated/provisional.
    short = L.classify_training_health([
        {'step': 0, 'train_ce': 6.0, 'val_ce': 4.0},
        {'step': 100, 'train_ce': 5.0, 'val_ce': 4.1}])
    assert short['trend_reliable'] is False and short['n_eval_points'] == 2, short
    assert any('only 2 validation evaluation point' in a for a in short['actions']), short
    assert h['trend_reliable'] is True, h

    # Fractional/Boolean steps are not evaluation identities: they must not
    # enter the slope fit while best-step selection rejects them.
    weird = [{'step': 0, 'train_ce': 5.0, 'val_ce': 4.0},
             {'step': True, 'train_ce': 1.0, 'val_ce': 1.0},
             {'step': 100.5, 'train_ce': 1.0, 'val_ce': 1.0},
             {'step': 200, 'train_ce': 4.0, 'val_ce': 4.2}]
    hw = L.classify_training_health(weird)
    assert hw['n_eval_points'] == 2 and hw['best_step'] == 0, hw
    assert hw['train_ce_slope_per_step'] == -0.005, hw


def test_early_stop_decision():
    hist = [{'step': 100, 'val_ppl': 50.0}, {'step': 200, 'val_ppl': 48.0}]
    assert L.early_stop_decision(hist, patience=2)['should_stop'] is False
    hist += [{'step': 300, 'val_ppl': 48.0}, {'step': 400, 'val_ppl': 48.1}]
    d = L.early_stop_decision(hist, patience=2)
    assert d['should_stop'] is True and d['stale_evals'] == 2, d
    assert d['best_step'] == 200 and d['best_val_ppl'] == 48.0, d
    # A real improvement resets the patience counter; min_delta makes a tiny
    # change explicitly insufficient.
    hist2 = [{'step': 100, 'val_ppl': 50.0}, {'step': 200, 'val_ppl': 49.0},
             {'step': 300, 'val_ppl': 48.9}, {'step': 400, 'val_ppl': 47.0}]
    assert L.early_stop_decision(hist2, patience=2)['should_stop'] is False
    tight = L.early_stop_decision(hist2[:3], patience=2, min_delta=2.0)
    assert tight['should_stop'] is True, tight
    # "at least min_delta below" is inclusive: exactly 0.5 below the best is
    # an improvement, not a stale point.
    edge = L.early_stop_decision(
        [{'step': 0, 'val_ppl': 10.0}, {'step': 1, 'val_ppl': 9.5}],
        patience=1, min_delta=0.5)
    assert edge['should_stop'] is False and edge['best_step'] == 1, edge
    assert L.early_stop_decision([], patience=1)['should_stop'] is False
    for bad_p in (0, -1, True, 1.5, '2'):
        try:
            L.early_stop_decision(hist, patience=bad_p)
        except ValueError:
            pass
        else:
            raise AssertionError(f'patience={bad_p!r} must be refused')
    for bad_d in (-1.0, float('nan'), float('inf'), True):
        try:
            L.early_stop_decision(hist, patience=1, min_delta=bad_d)
        except ValueError:
            pass
        else:
            raise AssertionError(f'min_delta={bad_d!r} must be refused')
    # A fractional/boolean step is not an evaluation identity: truncating it
    # would merge two different eval points into one step.
    fractional = L.early_stop_decision(
        [{'step': 100, 'val_ppl': 50.0}, {'step': 100.5, 'val_ppl': 40.0}],
        patience=1)
    assert fractional['n_points'] == 1 and fractional['best_step'] == 100, fractional


def test_nan_diagnosis_path():
    clean = L.diagnose_training_instability(
        {'opt_diag_summary': {'nonfinite_loss_steps': 0, 'nonfinite_grad_steps': 0,
                              'amp_scaler_overflows': 0, 'events': []}},
        data_checked='startup')
    assert clean['status'] == 'clean' and clean['primary_suspect'] is None, clean
    # fp16 scaler overflow is triaged before a generic gradient claim.
    fp16 = L.diagnose_training_instability({
        'amp_dtype': 'fp16',
        'opt_diag_summary': {'nonfinite_loss_steps': 0, 'nonfinite_grad_steps': 1,
                             'amp_scaler_overflows': 3, 'amp_scaler_scale_min': 0.25,
                             'grad_norm_max': 1000.0,
                             'events': [{'kind': 'grad', 'grad_norm': float('inf'), 'lr': 3e-4}]}})
    assert fp16['status'] == 'instability_observed', fp16
    assert fp16['primary_suspect'] == 'fp16_overflow', fp16
    # A non-finite loss whose CE is finite points at the auxiliary term.
    loss = L.diagnose_training_instability({
        'amp_dtype': 'bf16',
        'opt_diag_summary': {'nonfinite_loss_steps': 2, 'nonfinite_grad_steps': 0,
                             'events': [{'kind': 'loss', 'loss': float('nan'),
                                         'ce': 4.2, 'comp_reg': float('nan'),
                                         'boundary_pg': 0.0, 'lr': 3e-4}]}})
    assert loss['primary_suspect'] == 'forward_loss_numerics', loss
    _fl = [c for c in loss['checks'] if c['check'] == 'forward_loss_numerics'][0]
    assert 'compression regularizer' in _fl['evidence'], loss
    # A guarded attention value is evidence even when the loss never went bad.
    guard = L.diagnose_training_instability({
        'amp_dtype': None,
        'opt_diag_summary': {'nonfinite_loss_steps': 0, 'nonfinite_grad_steps': 0,
                             'nan_guard': {'elements_cleaned': 12,
                                           'first': {'where': 'indexer_softmax'}},
                             'events': []}})
    assert guard['primary_suspect'] == 'attention_pool_nan_to_num', guard
    assert guard['checks'][0]['status'] == 'unknown', guard
    # Plain gradient explosion with finite loss in bf16.
    grad = L.diagnose_training_instability({
        'amp_dtype': 'bf16',
        'opt_diag_summary': {'nonfinite_loss_steps': 0, 'nonfinite_grad_steps': 1,
                             'amp_scaler_overflows': 0, 'grad_norm_max': float('inf'),
                             'events': [{'kind': 'grad', 'grad_norm': float('inf'), 'lr': 1e-2}]}})
    assert grad['primary_suspect'] == 'gradient_explosion_lr', grad
    # A FINITE but far-above-clip gradient is still an observed instability.
    # It must not be reported as clean just because no NaN/Inf step occurred
    # (the gradient_explosion_lr check itself is `observed`).
    huge = L.diagnose_training_instability({
        'amp_dtype': 'bf16',
        'opt_diag_summary': {'nonfinite_loss_steps': 0, 'nonfinite_grad_steps': 0,
                             'amp_scaler_overflows': 0, 'grad_norm_max': 100.0,
                             'events': []}})
    assert huge['status'] == 'instability_observed', huge
    assert huge['primary_suspect'] == 'gradient_explosion_lr', huge
    # An explicitly recorded inf/NaN maximum is evidence too, even if the
    # integer event counters are zero (e.g. a partially written record).
    nonfinite_max = L.diagnose_training_instability({
        'amp_dtype': 'bf16',
        'opt_diag_summary': {'nonfinite_loss_steps': 0, 'nonfinite_grad_steps': 0,
                             'amp_scaler_overflows': 0,
                             'grad_norm_max': float('nan'), 'events': []}})
    assert nonfinite_max['status'] == 'instability_observed', nonfinite_max
    assert nonfinite_max['primary_suspect'] == 'gradient_explosion_lr', nonfinite_max
    # A non-finite VALIDATION point can coexist with a finite training loss;
    # it must not be reported as clean.
    hbad = L.diagnose_training_instability({
        'curve_health': {'status': 'nonfinite', 'reason': 'val_ppl NaN at step 200'},
        'opt_diag_summary': {'nonfinite_loss_steps': 0, 'nonfinite_grad_steps': 0,
                             'events': []}})
    assert hbad['status'] == 'instability_observed', hbad
    assert hbad['primary_suspect'] == 'validation_curve_numerics', hbad
    # A legacy/edited record may carry the curve but not the cached summary;
    # an unsummarised non-finite point is still evidence, never `clean`.
    hhist = L.diagnose_training_instability({
        'health_history': [
            {'step': 0, 'train_ce': 5.0, 'val_ce': 4.0},
            {'step': 100, 'train_ce': 4.8, 'val_ce': float('nan')}],
        'opt_diag_summary': {'nonfinite_loss_steps': 0, 'nonfinite_grad_steps': 0,
                             'events': []}})
    assert hhist['status'] == 'instability_observed', hhist
    assert hhist['primary_suspect'] == 'validation_curve_numerics', hhist
    assert hhist['curve_health_source'] == 'derived_from_history', hhist
    # No curve evidence at all means the validation-numerics check is unknown,
    # not silently "not observed".
    noc = L.diagnose_training_instability({
        'opt_diag_summary': {'nonfinite_loss_steps': 0, 'nonfinite_grad_steps': 0,
                             'events': []}})
    vc = [c for c in noc['checks'] if c['check'] == 'validation_curve_numerics'][0]
    assert vc['status'] == 'unknown', noc
    assert noc['curve_health_source'] == 'missing', noc
    # A malformed summary is record-integrity evidence, but a usable history
    # still supplies the validation verdict.
    both = L.diagnose_training_instability({
        'curve_health': ['healthy'],
        'health_history': [{'step': 0, 'train_ce': 5.0, 'val_ce': 4.0},
                           {'step': 100, 'train_ce': 4.8, 'val_ce': 4.2}],
        'opt_diag_summary': {'nonfinite_loss_steps': 0, 'nonfinite_grad_steps': 0,
                             'events': []}})
    assert both['status'] == 'unverifiable', both
    assert both['curve_health_source'] == 'derived_from_history', both
    # A cached healthy status cannot mask a non-finite point in its own
    # history: the curve wins and the contradiction is evidence.
    conflict = L.diagnose_training_instability({
        'curve_health': {'status': 'healthy', 'reason': 'ok'},
        'health_history': [{'step': 0, 'train_ce': 5.0, 'val_ce': 4.0},
                           {'step': 100, 'train_ce': 4.8, 'val_ce': float('nan')}],
        'opt_diag_summary': {'nonfinite_loss_steps': 0, 'nonfinite_grad_steps': 0,
                             'events': []}})
    assert conflict['status'] == 'instability_observed', conflict
    assert conflict['primary_suspect'] == 'validation_curve_numerics', conflict
    assert any(c['check'] == 'record_integrity' for c in conflict['checks']), conflict

    # A malformed/corrupt count field must neither crash the triage nor be
    # silently read as a zero-event (clean) record.
    malformed = L.diagnose_training_instability({
        'opt_diag_summary': {'nonfinite_loss_steps': 1.5,
                             'nonfinite_grad_steps': {'bad': 1},
                             'amp_scaler_overflows': float('nan'),
                             'nan_guard': {'elements_cleaned': 'many'}}})
    assert malformed['status'] == 'unverifiable', malformed
    assert malformed['primary_suspect'] == 'record_integrity', malformed
    assert any(c['check'] == 'record_integrity' for c in malformed['checks']), malformed
    bad_grad_max = L.diagnose_training_instability({
        'opt_diag_summary': {'nonfinite_loss_steps': 0, 'nonfinite_grad_steps': 0,
                             'grad_norm_max': 'not-a-number', 'events': []}})
    assert bad_grad_max['status'] == 'unverifiable', bad_grad_max
    assert bad_grad_max['primary_suspect'] == 'record_integrity', bad_grad_max


def test_data_audit_contract():
    # The NaN triage's first step is the data.  The trainer must record an
    # actual check, not a free-text "data validated" claim.
    tr = (np.arange(10000, dtype=np.int64) % 97)
    va = (np.arange(40, dtype=np.int64) % 97).reshape(4, 10)
    a = L.audit_token_data(tr, va, 97)
    assert a['status'] == 'ok', a
    assert a['sections']['train']['n_checked'] == tr.size, a
    assert a['sections']['validation']['n_checked'] == va.size, a
    assert a['sections']['train']['n_nonfinite'] == 0, a
    assert a['sections']['train']['n_out_of_range'] == 0, a
    assert a['checks']['integer_ids'] and a['checks']['finite_ok'], a
    big = (np.arange(5_000_000, dtype=np.int64) % 97)
    b = L.audit_token_data(big, va, 97, max_sample_tokens=10000)
    assert b['sections']['train']['sampled'] is True, b
    assert b['sections']['train']['n_checked'] <= 10000, b
    for bad in (np.array([1.0, float('nan')]), np.array([1.0, 2.0]),
                np.array([1, -1]), np.array([1, 97]), np.array([True, False])):
        try:
            L.audit_token_data(bad, va, 97)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(f'audit_token_data accepted {bad!r}')
    # A valid audit feeds the triage as NOT_OBSERVED evidence; a failed audit
    # is an observed data cause, not a clean run.
    clean = L.diagnose_training_instability({
        'data_audit': a,
        'opt_diag_summary': {'nonfinite_loss_steps': 0, 'nonfinite_grad_steps': 0,
                             'events': []}})
    checks = {c['check']: c for c in clean['checks']}
    assert checks['data_nan_inf']['status'] == 'not_observed', clean
    assert checks['data_label_shift']['status'] == 'unknown', clean
    bad_audit = {**a, 'status': 'failed',
                 'sections': {**a['sections'],
                              'train': {**a['sections']['train'], 'n_nonfinite': 1}}}
    bad = L.diagnose_training_instability({
        'data_audit': bad_audit,
        'opt_diag_summary': {'nonfinite_loss_steps': 0, 'nonfinite_grad_steps': 0,
                             'events': []}})
    assert bad['status'] == 'instability_observed', bad
    assert bad['primary_suspect'] == 'data_nan_inf', bad
    shift = {**a, 'label_shift': {'checked': True, 'ok': False, 'n_rows': 2,
                                 'n_tokens': 20}}
    sh = L.diagnose_training_instability({
        'data_audit': shift,
        'opt_diag_summary': {'nonfinite_loss_steps': 0, 'nonfinite_grad_steps': 0,
                             'events': []}})
    assert sh['status'] == 'instability_observed', sh
    assert sh['primary_suspect'] == 'data_label_shift', sh


def test_paired_errors_use_common_finite_support():
    # Two arms can each have non-finite CE at DIFFERENT tokens.  Comparing
    # their independent finite means would use different supports; the paired
    # delta must be computed only where both are finite.
    class _Fake:
        def __init__(self, vocab, bad_input):
            self.vocab = int(vocab)
            self.bad_input = int(bad_input)
            self.training = True

        def eval(self):
            self.training = False
            return self

        def train(self, mode=True):
            self.training = bool(mode)
            return self

        def trunk(self, ids):
            return ids.unsqueeze(-1)

        def head(self, h):
            import torch.nn.functional as _F
            ids = h[..., 0]
            lg = _F.one_hot(((ids + 1) % self.vocab).long(), self.vocab).float() * 8.0
            lg = lg.clone()
            lg[ids == self.bad_input] = float('inf')
            return lg

    vb = np.array([[1, 2, 3, 4, 5], [5, 4, 2, 3, 1]], dtype=np.int64)
    a = _Fake(16, bad_input=2)
    b = _Fake(16, bad_input=3)
    r = L.compare_token_errors(a, b, vb, 16, torch.device('cpu'), top_n=20)
    assert r['n_tokens'] == 8, r
    assert r['n_nonfinite_a'] > 0 and r['n_nonfinite_b'] > 0, r
    assert r['n_paired_finite'] < r['n_tokens'], r
    assert r['n_paired_finite'] == 4, r
    assert abs(r['mean_delta_b_minus_a']
               - (r['mean_ce_b'] - r['mean_ce_a'])) < 1e-12, r
    assert r['mean_ce_a'] != r['mean_ce_a_all_finite'], r
    assert r['mean_ce_b'] != r['mean_ce_b_all_finite'], r
    for t in r['worst_tokens_for_b']:
        assert math.isfinite(t['ce_a']) and math.isfinite(t['ce_b']), t
    assert r['b_worse_at_n'] + r['a_worse_at_n'] <= r['n_paired_finite'], r


def test_error_analysis_rejects_bad_token_tables():
    # The summary says the next-token ids were range-checked; that claim must
    # be true even when analyze_errors is called directly (not via the CLI).
    L.set_seed(0)
    model = L.SmallGPT(32, 16, 1, 2, 8, 8, L.make_layer_cfgs(1, 'full')).eval()
    bad = np.array([[1, 2, 32, 4]], dtype=np.int64)
    try:
        L.analyze_errors(model, bad, 32, torch.device('cpu'), top_tokens=4)
    except ValueError as e:
        assert 'out of range' in str(e), e
    else:
        raise AssertionError('out-of-vocabulary ids must be refused')
    frac = np.array([[1.5, 2, 3, 4]], dtype=np.float64)
    try:
        L.analyze_errors(model, frac, 32, torch.device('cpu'), top_tokens=4)
    except TypeError as e:
        assert 'integer token ids' in str(e), e
    else:
        raise AssertionError('float token tables must be refused')
    # The paired comparison has the same label contract.
    try:
        L.compare_token_errors(None, None, bad, 32, torch.device('cpu'))
    except ValueError as e:
        assert 'out of range' in str(e), e
    else:
        raise AssertionError('paired comparison must refuse out-of-vocabulary ids')
    try:
        L.compare_token_errors(None, None, frac, 32, torch.device('cpu'))
    except TypeError as e:
        assert 'integer token ids' in str(e), e
    else:
        raise AssertionError('paired comparison must refuse float token tables')


def test_error_analysis_paired_vocab_guard():
    EA._paired_vocab_guard({'vocab': 8192}, {'vocab': 8192})
    EA._paired_vocab_guard({'vocab': 32.0}, {'vocab': 32})
    EA._paired_vocab_guard({}, {})
    for bad in ({'vocab': 8192}, {'vocab': 8193}), ({'vocab': True}, {'vocab': 1}), \
               ({'vocab': 32}, {'vocab': -32}), ({'vocab': 32}, {'vocab': 32.5}), \
               ({'vocab': 32}, {'vocab': 'x'}):
        try:
            EA._paired_vocab_guard(*bad)
        except SystemExit as e:
            assert 'REFUSE' in str(e), e
        else:
            raise AssertionError(f'paired vocab guard accepted {bad!r}')


def test_error_analysis_paired_none_is_renderable():
    class _M:
        def to(self, device):
            return self

    class _Cfg:
        pass

    import tempfile as _tf
    td = _tf.mkdtemp(prefix='err_', dir=REPO)
    saved = (EA._load_model, EA._sha256, L.load_run_corpus,
             L.analyze_errors, L.compare_token_errors)
    try:
        EA._load_model = lambda path: (_M(), {'vocab': 32, 'd': 16, 'n_layers': 1,
                                              'n_heads': 2, 'd_head': 8, 'train_len': 8,
                                              'cfg': [_Cfg()]})
        EA._sha256 = lambda path: 'deadbeef'
        vb = np.arange(32, dtype=np.int64).reshape(4, 8) % 32
        L.load_run_corpus = lambda cfg: (np.arange(64, dtype=np.int64), vb, 32, None, None)
        L.analyze_errors = lambda *a, **k: {
            'mean_ce': 1.0, 'pooled_ppl': 2.7, 'buckets': {},
            'worst_tokens': [], 'worst_sequences': [],
            'cause_counts': {}, 'cause_counts_all': {}, 'cause_guide': {}}
        L.compare_token_errors = lambda *a, **k: {
            'checkpoint_a': 'a', 'checkpoint_b': 'b',
            'mean_delta_b_minus_a': None, 'n_paired_finite': 0, 'n_tokens': 8,
            'b_worse_at_n': 0, 'a_worse_at_n': 0, 'worst_tokens_for_b': []}
        EA.main(['--ckpt', 'a.pt', '--compare-ckpt', 'b.pt', '--outdir', td,
                 '--max-rows', '4'])
        txt = open(os.path.join(td, 'bad_cases.md'), encoding='utf-8').read()
        assert 'n/a' in txt and 'paired finite coverage' in txt, txt
    finally:
        (EA._load_model, EA._sha256, L.load_run_corpus,
         L.analyze_errors, L.compare_token_errors) = saved
        shutil.rmtree(td, ignore_errors=True)


def test_token_error_cause_taxonomy():
    # The bad-case file must say why a token is hard, not only how large its CE
    # is.  These are deliberately heuristic labels with a manual-review caveat.
    rare = L.classify_token_error(ce=8.0, entropy=1.0, margin=4.0, rare_target=True,
                                  target_id=7, topk_ids=[7, 8], vocab=100)
    assert rare['primary'] == 'annotation_or_rare_target', rare
    amb = L.classify_token_error(ce=4.0, entropy=math.log(100) * 0.9, margin=0.1,
                                 rare_target=False, target_id=7, topk_ids=[7, 8], vocab=100)
    assert amb['primary'] == 'ambiguous_context' and 'ambiguous_context' in amb['causes'], amb
    blind = L.classify_token_error(ce=8.0, entropy=1.0, margin=5.0, rare_target=False,
                                   target_id=7, topk_ids=[3, 4], vocab=100)
    assert blind['primary'] == 'confident_model_blind_spot', blind
    assert blind['target_in_topk'] is False, blind
    outside = L.classify_token_error(ce=4.0, entropy=2.0, margin=1.0, rare_target=False,
                                     target_id=7, topk_ids=[3, 4], vocab=100)
    assert outside['primary'] == 'target_outside_topk' and outside['target_in_topk'] is False, outside
    unmeasured = L.classify_token_error(ce=float('nan'), entropy=1.0, margin=1.0,
                                        rare_target=False, target_id=7,
                                        topk_ids=[3, 4], vocab=100)
    assert unmeasured['primary'] == 'unmeasured', unmeasured


def test_error_analysis_rare_basis_is_measured():
    # An empty/malformed training sample is not a frequency table of zeros.
    # The old path labelled every target "rare", manufacturing an
    # annotation/rare-target cause with no evidence.
    vocab = 8

    class _Stub:
        training = True

        def eval(self):
            self.training = False
            return self

        def train(self, mode=True):
            self.training = bool(mode)
            return self

        def trunk(self, ids):
            return torch.zeros((ids.shape[0], ids.shape[1], 2))

        def head(self, h):
            return torch.zeros((h.shape[0], h.shape[1], vocab))

    vb = (np.arange(8, dtype=np.int64) % vocab).reshape(2, 4)
    empty = L.analyze_errors(_Stub(), vb, vocab, torch.device('cpu'),
                             top_n=2, top_tokens=8, train_ids=np.array([], dtype=np.int64))
    assert empty['rare_target_basis'].startswith('unavailable'), empty
    assert 'empty train_ids' in (empty['rare_target_problem'] or ''), empty
    assert empty['cause_counts_all']['annotation_or_rare_target']['n'] == 0, empty
    assert empty['cause_counts_all']['ambiguous_context']['n'] == empty['n_targets'], empty

    # A valid table marks only tokens at/under the rare threshold.
    tr = np.array([1] * 5 + [2] * 6 + [3] * 100, dtype=np.int64)
    ok = L.analyze_errors(_Stub(), vb, vocab, torch.device('cpu'),
                          top_n=2, top_tokens=8, train_ids=tr)
    assert ok['rare_target_basis'] == 'train_frequency', ok
    assert ok['rare_target_problem'] is None, ok
    assert ok['cause_counts_all']['annotation_or_rare_target']['n'] > 0, ok
    if ok['worst_tokens']:
        assert all('target_train_count' in t for t in ok['worst_tokens']), ok['worst_tokens']

    # Out-of-range / non-integer samples disable the flag instead of being
    # silently truncated into a different vocabulary's frequencies.
    for bad in (np.array([-1, 2]), np.array([vocab, 2]), np.array([1.5, 2.0])):
        r = L.analyze_errors(_Stub(), vb, vocab, torch.device('cpu'),
                             top_n=2, top_tokens=8, train_ids=bad)
        assert r['rare_target_basis'].startswith('unavailable'), (bad, r)
        assert r['rare_target_problem'], (bad, r)
        assert r['cause_counts_all']['annotation_or_rare_target']['n'] == 0, (bad, r)


def test_error_analysis_zero_and_invalid_budgets():
    # A zero inventory is not one row, and a malformed budget must be refused
    # (negative top_n/-chunk used to be silently clamped to 1).
    vocab = 8

    class _Stub:
        training = True

        def eval(self):
            self.training = False
            return self

        def train(self, mode=True):
            self.training = bool(mode)
            return self

        def trunk(self, ids):
            return torch.zeros((ids.shape[0], ids.shape[1], 2))

        def head(self, h):
            return torch.zeros((h.shape[0], h.shape[1], vocab))

    vb = (np.arange(12, dtype=np.int64) % vocab).reshape(3, 4)
    r = L.analyze_errors(_Stub(), vb, vocab, torch.device('cpu'),
                         top_n=0, top_tokens=0)
    assert r['worst_sequences'] == [] and r['worst_tokens'] == [], r
    for kw in ({'top_n': -1}, {'top_tokens': -1}, {'top_n': 1.5},
               {'chunk': 0}, {'context_tokens': -1},
               {'top_n': True}):
        try:
            L.analyze_errors(_Stub(), vb, vocab, torch.device('cpu'), **kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f'analyze_errors must refuse {kw!r}')
    p = L.compare_token_errors(_Stub(), _Stub(), vb, vocab, torch.device('cpu'),
                               top_n=0)
    assert p['worst_tokens_for_b'] == [], p
    assert p['n_paired_finite'] == 9, p
    for bad in (-1, 1.5, True):
        try:
            L.compare_token_errors(_Stub(), _Stub(), vb, vocab,
                                   torch.device('cpu'), top_n=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'compare_token_errors must refuse top_n={bad!r}')
    try:
        L.compare_token_errors(_Stub(), _Stub(), vb, vocab,
                               torch.device('cpu'), chunk=0)
    except ValueError:
        pass
    else:
        raise AssertionError('compare_token_errors must refuse chunk=0')


def test_error_analysis_flags_prefix_ambiguity():
    # A high-CE token whose exact prefix occurs with a different next token
    # elsewhere is not uniquely labelled: the review must see that, rather
    # than being told only "ambiguous distribution".
    vocab = 8

    class _Stub:
        training = True

        def eval(self):
            self.training = False
            return self

        def train(self, mode=True):
            self.training = bool(mode)
            return self

        def trunk(self, ids):
            return torch.zeros((ids.shape[0], ids.shape[1], 2))

        def head(self, h):
            return torch.zeros((h.shape[0], h.shape[1], vocab))

    vb = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 6]], dtype=np.int64)
    r = L.analyze_errors(_Stub(), vb, vocab, torch.device('cpu'),
                         top_n=2, top_tokens=8)
    ca = r['context_ambiguity']
    assert ca['n_contexts_with_multiple_targets'] == 1, ca
    assert ca['n_positions_with_ambiguous_context'] == 2, ca
    assert ca['n_ambiguous_worst'] >= 2, ca
    hit = [t for t in r['worst_tokens'] if t['same_context_ambiguous']]
    assert hit, r['worst_tokens']
    for t in hit:
        assert t['same_context_target_variants'] == 2, t
        assert t['same_context_support'] == 2, t
    r0 = L.analyze_errors(_Stub(), vb, vocab, torch.device('cpu'),
                          top_n=0, top_tokens=0)
    assert r0['context_ambiguity']['n_worst_reviewed'] == 0, r0
    assert r0['context_ambiguity']['context_len'] is None, r0


def test_reference_count_model_is_exact_and_bounded():
    # The reference used to separate model misses from hard labels must be the
    # actual argmax of the interpolated count model, not a backoff guess, and
    # a large corpus fit must be a disclosed deterministic prefix.
    rng = np.random.default_rng(7)
    V = 7
    train = rng.integers(0, V, size=250, dtype=np.int64)
    ev = np.array([[0, 1, 2, 3, 4, 5], [6, 0, 0, 1, 1, 2]], dtype=np.int64)
    counts = L._ngram_train_counts(train, V)
    uni, pair_keys, ctx, pair_counts = counts
    alpha = 0.1
    uni_prob = (uni + alpha) / (uni.sum() + alpha * V)
    for order in (1, 2):
        lams = [0.0] if order == 1 else [0.0, 0.25, 0.5, 0.75, 1.0]
        for lam in lams:
            r = L.ngram_reference_score(train, ev, V, order=order,
                                        alpha=alpha, interpolation=lam)
            assert r['nll'].shape == (2, 5) and r['top1_ids'].shape == (2, 5), r
            for row in range(ev.shape[0]):
                for j in range(1, ev.shape[1]):
                    prev, tgt = int(ev[row, j - 1]), int(ev[row, j])
                    if order == 1:
                        probs = uni_prob
                        p = uni_prob[tgt]
                    else:
                        key = prev * V + tgt
                        idx = int(np.searchsorted(pair_keys, key))
                        cp = (float(pair_counts[idx])
                              if idx < pair_keys.size and int(pair_keys[idx]) == key else 0.0)
                        c = float(ctx[prev])
                        p = lam * ((cp + alpha) / (c + alpha * V)) + (1 - lam) * uni_prob[tgt]
                        cp_all = np.array([
                            (float(pair_counts[int(np.searchsorted(pair_keys, prev * V + v))])
                             if (int(np.searchsorted(pair_keys, prev * V + v)) < pair_keys.size
                                 and int(pair_keys[int(np.searchsorted(pair_keys, prev * V + v))]) == prev * V + v)
                             else 0.0)
                            for v in range(V)], dtype=float)
                        probs = lam * ((cp_all + alpha) / (c + alpha * V)) + (1 - lam) * uni_prob
                    assert abs(float(r['nll'][row, j - 1]) + math.log(p)) < 1e-9, (order, lam, row, j)
                    assert int(r['top1_ids'][row, j - 1]) == int(np.argmax(probs)), (order, lam, row, j)
    # A capped fit reports the truncation instead of pretending it saw the
    # whole stream.
    cap = L.ngram_reference_score(train, ev, V, order=2, max_fit_tokens=40)
    assert cap['fit']['n_fit_tokens'] == 40 and cap['fit']['fit_truncated'] is True, cap['fit']
    assert cap['fit']['n_source_tokens'] == int(train.size), cap['fit']
    # Sparse/tied contexts: the best unseen token is not always the global
    # unigram argmax (it may already be a successor), and an exact argmax
    # breaks a score tie by the lowest token id.  A backoff guess used to
    # disagree with np.argmax here.
    rng2 = np.random.default_rng(11)
    for _ in range(40):
        _V = int(rng2.integers(2, 13))
        _train = rng2.integers(0, _V, size=int(rng2.integers(0, 40)), dtype=np.int64)
        _ev = rng2.integers(0, _V, size=(2, 6), dtype=np.int64)
        _uni, _pk, _ctx, _pc = L._ngram_train_counts(_train, _V)
        _up = (_uni + alpha) / (_uni.sum() + alpha * _V)
        for _lam in (0.0, 0.25, 0.5, 0.75, 1.0):
            _r = L.ngram_reference_score(_train, _ev, _V, order=2,
                                          alpha=alpha, interpolation=_lam)
            for _row in range(_ev.shape[0]):
                for _j in range(1, _ev.shape[1]):
                    _p = int(_ev[_row, _j - 1])
                    _probs = np.array([
                        (float(_pc[int(np.searchsorted(_pk, _p * _V + _v))])
                         if (int(np.searchsorted(_pk, _p * _V + _v)) < _pk.size
                             and int(_pk[int(np.searchsorted(_pk, _p * _V + _v))]) == _p * _V + _v)
                         else 0.0)
                        for _v in range(_V)], dtype=float)
                    _probs = _lam * ((_probs + alpha) / (_ctx[_p] + alpha * _V)) + (1 - _lam) * _up
                    assert int(_r['top1_ids'][_row, _j - 1]) == int(np.argmax(_probs)), (
                        _V, _lam, _row, _j, _r['top1_ids'][_row, _j - 1], int(np.argmax(_probs)))
    # Invalid smoothing/order is refused rather than silently replaced.
    for bad in ({'order': 3}, {'order': True}, {'alpha': 0.0}, {'alpha': float('nan')},
                {'interpolation': -0.1}, {'interpolation': 1.1}):
        try:
            L.ngram_reference_score(train, ev, V, **bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'ngram_reference_score accepted {bad!r}')
    try:
        L.ngram_reference_score(train, ev.astype(float), V)
    except TypeError:
        pass
    else:
        raise AssertionError('reference scoring must refuse float token ids')


def test_error_analysis_reference_separates_model_miss():
    # With a deterministic 5->7 transition in the training stream and a
    # uniform model, the reference must catch positions the model misses.
    # Those positions are model blind-spot evidence; both-miss positions are
    # the population that needs a manual label check.
    V = 8

    class _Stub:
        training = True

        def eval(self):
            self.training = False
            return self

        def train(self, mode=True):
            self.training = bool(mode)
            return self

        def trunk(self, ids):
            return torch.zeros((ids.shape[0], ids.shape[1], 2))

        def head(self, h):
            return torch.zeros((h.shape[0], h.shape[1], V))

    train = np.array([5, 7] * 100 + [1] * 100, dtype=np.int64)
    vb = np.array([[5, 7, 5, 7], [5, 7, 1, 7]], dtype=np.int64)
    r = L.analyze_errors(_Stub(), vb, V, torch.device('cpu'), top_n=2,
                         top_tokens=8, train_ids=train, reference='bigram')
    rc = r['reference_comparison']
    assert rc['available'] is True and rc['complete'] is True, rc
    assert rc['n_paired_targets'] == 6, rc
    assert rc['n_reference_only_correct'] > 0, rc
    assert rc['n_both_correct'] == 0 and rc['n_model_correct'] == 0, rc
    assert rc['reference_ce'] < rc['model_ce'], rc
    assert rc['excess_ce_model_minus_reference'] > 0, rc
    assert rc['row_excess_ce_ci95'] is not None, rc
    assert r['reference_basis'] == 'bigram', r
    assert 'reference_catches_model_miss' in r['cause_guide'], r['cause_guide']
    assert 'reference_also_miss' in r['cause_guide'], r['cause_guide']
    hit = [t for t in r['worst_tokens'] if t['reference_verdict'] == 'reference_catches_model_miss']
    assert hit, r['worst_tokens']
    for t in hit:
        assert t['reference_correct'] is True and t['model_correct_vs_reference'] is False, t
        assert t['reference_pred_id'] == t['target_id'], t
        assert 'reference_catches_model_miss' in t['causes'], t
    # Disabling the reference is explicit, and a malformed/empty training
    # sample disables it with a reason instead of fabricating counts.
    off = L.analyze_errors(_Stub(), vb, V, torch.device('cpu'), top_n=1,
                           top_tokens=4, train_ids=train, reference='none')
    assert off['reference_comparison'] is None and off['reference_basis'] is None, off
    for bad in (np.array([], dtype=np.int64), np.array([-1, 2]), np.array([1.5, 2.0])):
        rr = L.analyze_errors(_Stub(), vb, V, torch.device('cpu'), top_n=1,
                              top_tokens=4, train_ids=bad, reference='bigram')
        assert rr['reference_comparison']['available'] is False, (bad, rr['reference_comparison'])
        assert rr['reference_problem'], (bad, rr)
    try:
        L.analyze_errors(_Stub(), vb, V, torch.device('cpu'), reference='bogus')
    except ValueError:
        pass
    else:
        raise AssertionError('analyze_errors must refuse an unknown reference mode')


def test_error_analysis_cli_surfaces_reference_floor():
    # The console and Markdown artifact (not only the JSON) must show the
    # reference comparison and its row-level uncertainty.
    class _M:
        def to(self, device):
            return self

    class _Cfg:
        pass

    td = tempfile.mkdtemp(prefix='err_ref_', dir=REPO)
    saved = (EA._load_model, EA._sha256, L.load_run_corpus, L.analyze_errors)
    fake = {
        'mean_ce': 3.0, 'pooled_ppl': 20.1, 'buckets': {},
        'worst_tokens': [], 'worst_sequences': [],
        'cause_counts': {}, 'cause_counts_all': {}, 'cause_guide': {},
        'reference_comparison': {
            'available': True, 'complete': True, 'reason': None,
            'fit': {'mode': 'bigram', 'alpha': 0.1, 'interpolation': 0.5,
                    'n_fit_tokens': 100, 'n_source_tokens': 100,
                    'fit_truncated': False},
            'n_paired_targets': 10, 'model_ce': 3.0, 'reference_ce': 2.5,
            'excess_ce_model_minus_reference': 0.5,
            'n_model_only_correct': 1, 'n_reference_only_correct': 3,
            'n_both_wrong': 2, 'n_both_wrong_unexplained': 1,
            'share_model_only_correct': 0.1, 'share_reference_only_correct': 0.3,
            'share_both_wrong': 0.2, 'share_both_wrong_unexplained': 0.1,
            'row_mean_excess_ce': 0.5, 'row_excess_ce_se': 0.1,
            'row_excess_ce_ci95': [0.304, 0.696], 'n_rows_with_paired_targets': 2},
        'reference_basis': 'bigram', 'reference_problem': None,
    }
    try:
        EA._load_model = lambda path: (_M(), {'vocab': 32, 'd': 16, 'n_layers': 1,
                                              'n_heads': 2, 'd_head': 8, 'train_len': 8,
                                              'cfg': [_Cfg()]})
        EA._sha256 = lambda path: 'deadbeef'
        vb = np.arange(32, dtype=np.int64).reshape(4, 8) % 32
        L.load_run_corpus = lambda cfg: (np.arange(64, dtype=np.int64), vb, 32, None, None)
        L.analyze_errors = lambda *a, **k: dict(fake)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            EA.main(['--ckpt', 'a.pt', '--outdir', td, '--max-rows', '4'])
        out = buf.getvalue()
        assert 'reference floor' in out and 'row-level excess-CE 95% CI' in out, out
        md = open(os.path.join(td, 'bad_cases.md'), encoding='utf-8').read()
        assert 'Reference floor' in md and 'reference only correct' in md.lower(), md
        summ = json.load(open(os.path.join(td, 'summary.json'), encoding='utf-8'))
        assert summ['reference_comparison']['available'] is True, summ
        assert summ['analysis_cost']['n_scored_tokens'] == 28, summ['analysis_cost']
    finally:
        (EA._load_model, EA._sha256, L.load_run_corpus, L.analyze_errors) = saved
        shutil.rmtree(td, ignore_errors=True)


def test_train_only_curve_health():
    desc = [10.0 - 0.05 * i for i in range(200)]
    h = L.classify_train_loss_curve(desc)
    assert h['status'] == 'descending', h
    assert h['mode'] == 'train_only' and h['n_points'] == 200, h
    stall = L.classify_train_loss_curve([5.0] * 200)
    assert stall['status'] == 'stalled', stall
    diverge = [5.0 - 0.01 * i for i in range(100)] + [4.0 + 0.05 * i for i in range(100)]
    dh = L.classify_train_loss_curve(diverge)
    assert dh['status'] == 'diverging', dh
    spiky = [5.0 - 0.01 * i for i in range(200)]
    spiky[185] = 30.0
    sh = L.classify_train_loss_curve(spiky)
    assert sh['status'] == 'unstable_spikes', sh
    assert sh['first_spike_index'] == 185, sh
    nh = L.classify_train_loss_curve([1.0, float('nan'), 0.9] + [0.8] * 20)
    assert nh['status'] == 'nonfinite' and nh['first_nonfinite_index'] == 1, nh
    ih = L.classify_train_loss_curve([1.0, 0.9])
    assert ih['status'] == 'insufficient_data', ih
    for _bad in (True, 1.5, 0, -1):
        try:
            L.classify_train_loss_curve(desc, min_points=_bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'min_points={_bad!r} must be refused')


def test_train_variant_records_train_curve_without_validation():
    ids = np.arange(2048, dtype=np.int64) % 997
    val = (np.arange(64, dtype=np.int64) % 997).reshape(4, 16)
    r = L.train_variant('full', ids, val, 997, seed=0, d=32, n_layers=1,
                        n_heads=2, d_head=16, seq_len=16, batch_size=2, steps=60,
                        lr=1e-3, weight_decay=0.0, warmup=1, comp_lambda=0.0,
                        eval_every=0, mlp_ratio=4.0, device=torch.device('cpu'),
                        log_every=0)
    th = r.get('train_curve_health') or {}
    assert th.get('mode') == 'train_only' and th.get('status') not in (None, 'insufficient_data'), th
    assert th.get('n_points') == 60, th
    assert (r.get('curve_health') or {}).get('status') in ('unmonitored', 'nonfinite'), r.get('curve_health')
    checks = {c['check']: c['status'] for c in
              (r.get('instability_diagnosis') or {}).get('checks', [])}
    assert checks.get('train_loss_curve') == 'not_observed', checks


def test_error_analysis_context_scan_is_bounded():
    vocab = 8

    class _Stub:
        training = True

        def eval(self):
            self.training = False
            return self

        def train(self, mode=True):
            self.training = bool(mode)
            return self

        def trunk(self, ids):
            return torch.zeros((ids.shape[0], ids.shape[1], 2))

        def head(self, h):
            return torch.zeros((h.shape[0], h.shape[1], vocab))

    vb = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 6]], dtype=np.int64)
    full = L.analyze_errors(_Stub(), vb, vocab, torch.device('cpu'),
                            top_n=2, top_tokens=8)
    assert full['context_ambiguity']['scan_complete'] is True, full['context_ambiguity']
    bounded = L.analyze_errors(_Stub(), vb, vocab, torch.device('cpu'),
                               top_n=2, top_tokens=8,
                               context_scan_max_positions=4)
    ca = bounded['context_ambiguity']
    assert ca['scan_complete'] is False, ca
    assert ca['scan']['n_rows_scanned'] == 1 and ca['scan']['n_rows_total'] == 2, ca
    assert ca['scan']['n_positions_scanned'] == 4, ca
    assert ca.get('scan_note'), ca
    for _bad in (True, 0, -1, 1.5):
        try:
            L.analyze_errors(_Stub(), vb, vocab, torch.device('cpu'),
                             top_n=1, top_tokens=1,
                             context_scan_max_positions=_bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'context_scan_max_positions={_bad!r} must be refused')


def test_context_scan_budget_holds_below_one_row():
    vb = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=np.int64)
    for _max in (1, 2, 3):
        _, _, meta = L._context_target_counts(vb, 3, max_positions=_max)
        assert meta['complete'] is False, meta
        assert meta['n_positions_scanned'] <= _max, meta
        assert meta['n_positions_scanned'] == _max, meta
    _, _, meta = L._context_target_counts(vb, 3, max_positions=100)
    assert meta['complete'] is True, meta
    assert meta['n_positions_scanned'] == 8, meta


def test_first_occurrence_of_valid_drops_invalid_lanes():
    idx = torch.tensor([[3, 1, 3, 0, 1]], dtype=torch.int32)
    valid = torch.tensor([[False, False, True, True, True]])
    got = L._first_occurrence_of_valid(idx, valid)
    assert not bool(got[0, 0]) and not bool(got[0, 1]), got
    assert got[0, 2:].tolist() == [True, True, True], got


def test_cost_increase_flags_sub_noise_gain():
    rows = [{'variant': 'best', 'ppl': p, 'prefill_ms_p50': 10.0,
             'params': 100, 'peak_mem_allocated_mb': 10.0, 'tokens_seen': 100}
            for p in (48.0, 48.2, 47.8)]
    rows += [{'variant': 'pricey', 'ppl': p, 'prefill_ms_p50': 25.0,
              'params': 100, 'peak_mem_allocated_mb': 10.0, 'tokens_seen': 100}
             for p in (48.05, 48.15, 47.95)]
    s = BR.efficiency_accuracy_summary(rows)
    pricey = [r for r in s if r['variant'] == 'pricey'][0]
    assert 'cost_increase_without_clear_gain' in pricey['flags'], pricey
    assert 'latency_2x_or_more' in pricey['flags'], pricey


def test_loss_curve_downsample_keeps_spikes_and_nan():
    # Long runs are downsampled before the curve is written.  A one-step
    # explosion/NaN must survive that reduction, or a diverged run looks clean.
    vals = [float(i) for i in range(5000)]
    vals[1234] = float('nan')
    vals[2345] = 999.0
    out = L._reduce_curve_values(vals, 100)
    assert len(out) <= 100, len(out)
    assert out[0] == 0.0 and out[-1] == 4999.0, (out[0], out[-1])
    assert any(isinstance(v, float) and math.isnan(v) for v in out), 'NaN spike was dropped'
    assert 999.0 in out, 'loss spike was dropped'
    assert L._reduce_curve_values([1.0, 2.0, 3.0], 100) == [1.0, 2.0, 3.0]
    # A tiny point budget is still a bound: the five-landmarks-per-bin rule
    # must not return six samples when the caller asked for at most one.
    for _mp in range(1, 8):
        _v = [float(i) for i in range(50)]
        _v[20] = float('nan')
        _got = L._reduce_curve_values(_v, _mp)
        assert len(_got) <= _mp, (_mp, len(_got))
        assert _got[-1] == 49.0, (_mp, _got)
        _p = [[i, float(i)] for i in range(50)]
        _p[20][1] = float('nan')
        _gp = L._reduce_curve_points(_p, _mp)
        assert len(_gp) <= _mp, (_mp, len(_gp))
        assert _gp[-1][0] == 49, (_mp, _gp)
    assert L._reduce_curve_values([1.0, 2.0], 0) == []
    assert L._reduce_curve_points([[0, 1.0], [1, 2.0]], 0) == []


def test_final_loss_smoothed_uses_last_steps():
    # `final_loss_smoothed` must describe the final 50 steps.  Reading a
    # scalar only on sparse diagnostic steps made the old value an average of
    # points that could span a large part of the run.
    ids = np.arange(4096, dtype=np.int64) % 997
    val = (np.arange(64, dtype=np.int64) % 997).reshape(4, 16)
    kw = dict(d=32, n_layers=1, n_heads=2, d_head=16, seq_len=16, batch_size=2,
              steps=70, lr=1e-3, weight_decay=0.0, warmup=1, comp_lambda=0.0,
              eval_every=0, mlp_ratio=4.0, device=torch.device('cpu'), log_every=0)
    r = L.train_variant('full', ids, val, 997, seed=0, **kw)
    assert len(r['losses']) == 70, len(r['losses'])
    tail = [v for v in r['losses'][-50:] if math.isfinite(v)]
    assert r['final_loss_smoothed_n'] == len(tail) == 50, r.get('final_loss_smoothed_n')
    assert abs(r['final_loss_smoothed'] - sum(tail) / len(tail)) < 1e-12, r['final_loss_smoothed']
    assert r['final_loss_smoothed_window'] == 50, r
    assert r['data_audit']['status'] == 'ok', r.get('data_audit')
    assert r['data_audit']['label_shift']['ok'] is True, r.get('data_audit')


def test_train_variant_early_stop_record():
    ids = np.arange(2048, dtype=np.int64) % 997
    val = (np.arange(64, dtype=np.int64) % 997).reshape(4, 16)
    kw = dict(d=32, n_layers=1, n_heads=2, d_head=16, seq_len=16, batch_size=2,
              steps=4, lr=1e-3, weight_decay=0.0, warmup=1, comp_lambda=0.0,
              eval_every=1, eval_subset=2, mlp_ratio=4.0,
              device=torch.device('cpu'), log_every=0)
    r = L.train_variant('full', ids, val, 997, seed=0,
                        early_stop_patience=1, early_stop_min_delta=1e9, **kw)
    es = r.get('early_stop') or {}
    assert es.get('stopped') is True and es.get('stopped_step') == 2, es
    assert r['steps'] == 4 and r['steps_completed'] == 2, r
    assert r['tokens_seen'] == 2 * 2 * 16, r['tokens_seen']
    assert r['opt_diag_summary']['n_steps_completed'] == 2, r['opt_diag_summary']
    assert r['final_full_val_ppl'] == r['ppl'], r
    assert r['final_subset_val_ppl'] is not None and r['best_val_ppl'] is not None, r
    assert abs(r['final_minus_best'] - (r['final_subset_val_ppl'] - r['best_val_ppl'])) < 1e-12, r
    r2 = L.train_variant('full', ids, val, 997, seed=0, **dict(kw, steps=2))
    assert r2['steps_completed'] == 2 and r2.get('early_stop') is None, r2
    assert r2['tokens_seen'] == 2 * 2 * 16, r2['tokens_seen']
    try:
        L.train_variant('full', ids, val, 997, seed=0,
                        early_stop_patience=1, **dict(kw, eval_every=0))
    except ValueError:
        pass
    else:
        raise AssertionError('early stopping without a validation grid must be refused')


def test_v12_health_gate_refuses_unstable_training():
    cfg = {'smoke': False}
    healthy = {'curve_health': {'status': 'healthy', 'reason': 'ok'}}
    assert V12._training_health_gate(cfg, 'a', 0, healthy, healthy) is None
    bad = {'curve_health': {'status': 'overfitting', 'reason': 'val rising'}}
    msg = V12._training_health_gate(cfg, 'a', 0, bad, bad)
    assert msg and 'overfitting' in msg, msg
    triage = {'curve_health': {'status': 'healthy', 'reason': 'ok'},
              'instability_diagnosis': {'status': 'instability_observed',
                                        'primary_suspect': 'fp16_overflow',
                                        'summary': 'scaler overflow'}}
    msg2 = V12._training_health_gate(cfg, 'a', 0, triage, {})
    assert msg2 and 'fp16_overflow' in msg2, msg2
    _old = os.environ.get('V12_ALLOW_UNHEALTHY')
    os.environ['V12_ALLOW_UNHEALTHY'] = '1'
    try:
        assert V12._training_health_gate(cfg, 'a', 0, triage, {}) is None
    finally:
        if _old is None:
            os.environ.pop('V12_ALLOW_UNHEALTHY', None)
        else:
            os.environ['V12_ALLOW_UNHEALTHY'] = _old


def test_rope_arms_are_not_batched():
    # The implementation binding must report eager_batch=False for the rope csa
    # arms because `HybridAttentionRoPE.forward` unconditionally takes the
    # per-row `_single_rope` loop.  A CUDA-capable device still yields False for
    # all four frozen arms; a non-rope csa config remains eligible (the check is
    # not a blanket False).
    class _Cuda:
        type = 'cuda'
    for arm in ('csa_fixed_rope', 'full_rope', 'csa_fixed_rope_randidx',
                'csa_fixed_rope_allblocks'):
        cfgs = L.make_layer_cfgs(6, arm)
        assert L._uses_batched_eager(cfgs, device=_Cuda(), batch_size=12) is False, arm
    assert L._uses_batched_eager(L.make_layer_cfgs(6, 'csa_fixed'), device=_Cuda(), batch_size=12) is True
    assert L._uses_batched_eager(L.make_layer_cfgs(6, 'csa_fixed_rope'), device=_Cuda(), batch_size=1) is False


def test_by_len_requires_one_ppl_per_row():
    cell = {'n_tok': 16 * 4095, 'truncated': False,
            'ppl_per_seq': [1.0] * 64}
    reason = V12._by_len_shape_reason({'by_len': {'4096': cell}}, 16)
    assert reason is not None and 'ppl_per_seq' in reason, reason
    cell['ppl_per_seq'] = [1.0] * 16
    assert V12._by_len_shape_reason({'by_len': {'4096': cell}}, 16) is None


def test_plan_check_rejects_bad_endpoint_and_finaltest_is_gated():
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(REPO, 'confirmation_frozen.json'), encoding='utf-8') as f:
            man = json.load(f)
        man['primary_comparisons'][0] = dict(man['primary_comparisons'][0],
                                             endpoint='by_len.ppl@L4096')
        bad = os.path.join(td, 'bad_endpoint.json')
        with open(bad, 'w', encoding='utf-8') as f:
            json.dump(man, f)
        _buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(_buf):
                V12.plan_check(bad)
        except SystemExit:
            assert 'endpoint' in _buf.getvalue(), _buf.getvalue()[:500]
        else:
            raise AssertionError('an unparseable endpoint must fail plan_check')
        # run_finaltest must not skip the plan preflight for a direct P5T call.
        _old = V12.plan_check
        _seen = []
        def _boom(path=None):
            _seen.append(path)
            raise SystemExit('plan refused in test')
        try:
            V12.plan_check = _boom
            try:
                _cfg = dict(V12.FINALTEST_CFG)
                _cfg.update(manifest=os.path.join(REPO, 'confirmation_frozen.json'),
                            outdir=os.path.join(td, 'testout'), smoke=False)
                V12.run_finaltest(_cfg)
            except SystemExit as e:
                assert 'plan refused in test' in str(e), str(e)[:200]
            else:
                raise AssertionError('run_finaltest bypassed plan_check')
        finally:
            V12.plan_check = _old
        assert _seen, 'plan_check was not called'


def test_manifest_v3_admission_can_succeed():
    # The zero-GPU plan check is the end-to-end guard against a stamp-only
    # alignment that would admit no records.  Exercise it on the shipped
    # manifest.
    rep = V12.plan_check(os.path.join(REPO, 'confirmation_frozen.json'))
    assert rep['ok'] and rep['admitted'] == rep['n_arms'] * rep['n_seeds'], rep
    assert all(c['pairs_planned'] == rep['n_seeds'] for c in rep['comparisons'].values()), rep
    assert rep['predicted']['full_rope']['eager_batch'] is False, rep
    assert rep['predicted']['csa_fixed_rope']['amp_dtype'] == 'bf16', rep


def test_v7_rope_gather_rotation_order():
    # The per-chunk RoPE refactor rotates the gathered window rows.  The
    # gathered tensor is [queries, lanes, heads, head_dim], so the position
    # table needs a singleton HEAD axis; inserting a singleton LANE axis only
    # broadcasts when lanes == heads.
    torch.manual_seed(0)
    T, nh, hd, q, m = 13, 3, 8, 7, 5
    k = torch.randn(T, nh, hd)
    idx = torch.randint(0, T, (q, m))
    dev = torch.device('cpu')
    cos, sin = V7.rope_cos_sin(hd, hd, L._arange_cache(T, dev), dev)
    old = L._take_2d(V7.apply_rope(k, cos[:, None, :], sin[:, None, :], hd), idx)
    new = V7.apply_rope(L._take_2d(k, idx),
                        cos[idx][:, :, None, :], sin[idx][:, :, None, :], hd)
    assert torch.equal(old, new), (old - new).abs().max()
    # End-to-end: the frozen csa_fixed_rope arm must run with lanes != heads.
    cfg = V7.make_layer_cfgs_v7(1, 'csa_fixed_rope')[0]
    model = V7.SmallGPTRoPE(64, 32, 1, 2, 16, 16, [cfg])
    x = torch.randint(0, 64, (1, 16))
    y = torch.randint(0, 64, (1, 16))
    loss = model.ce_loss(x, y, 64)
    assert math.isfinite(float(loss.detach())), loss
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(bool(torch.isfinite(g).all()) for g in grads), 'RoPE backward produced non-finite gradients'


def test_scale_input_guards():
    try:
        L.meta_analyze_effects([{'label': 'A', 'scale': 1.0, 'effects': [1.0, 2.0], 'seeds': [0, 0]},
                                {'label': 'B', 'scale': 2.0, 'effects': [2.0, 3.0], 'seeds': [0, 1]}])
        raise AssertionError('duplicate seeds inside one scale point must be rejected')
    except ValueError:
        pass
    try:
        L.meta_analyze_effects([{'label': 'A', 'scale': 1.0, 'effects': [1.0, 2.0], 'seeds': [0, 1]},
                                {'label': 'A', 'scale': 2.0, 'effects': [2.0, 3.0], 'seeds': [0, 1]}])
        raise AssertionError('two rows for one scale label must not be pooled as independent studies')
    except ValueError:
        pass
    for fn, args in (
        (L.paired_log_ratio, ([1.0, 2.0], [1.1, 2.1])),
        (L.scale_interaction, ([1.0, 2.0], [1.1, 2.1])),
    ):
        try:
            fn(*args, seeds=[0, 0])
            raise AssertionError(f'{fn.__name__} must reject duplicate seeds instead of double-counting them')
        except ValueError:
            pass
    # A non-finite pair is dropped from the test but must stay visible as a
    # dropped observation; otherwise the surviving subset is misreported as a
    # complete paired panel.
    si = L.scale_interaction([1.0, float('nan'), 2.0, 3.0],
                             [1.1, float('nan'), 2.1, 3.1],
                             seeds=[10, 11, 12, 13])
    assert si['n'] == 3 and si['n_input'] == 4 and si['n_dropped'] == 1, si
    assert si['complete_pairs'] is False and si['dropped_seeds'] == [11], si
    assert si['seeds'] == [10, 12, 13], si
    # A caller-declared minimum-n floor is part of the claim, not just a label
    # on the text.  n=6 reaches p=0.03125 but an explicit min_n=8 must keep
    # the interaction-level `significant` flag false (the raw test flag stays
    # visible for transparency).
    und = L.scale_interaction([0.0] * 6, [1.0] * 6, seeds=list(range(6)),
                              min_n=8)
    assert und['n'] == 6 and abs(und['p_exact_signflip'] - 0.03125) < 1e-12, und
    assert und['underpowered'] is True, und
    assert und['significant_raw'] is True and und['significant'] is False, und
    powered = L.scale_interaction([0.0] * 6, [1.0] * 6, seeds=list(range(6)),
                                  min_n=6)
    assert powered['significant'] is True and powered['significant_raw'] is True, powered
    # One dropped seed makes this a complete-case subset; even though six
    # non-zero pairs can reach p=0.031, the interaction-level claim is refused.
    partial = L.scale_interaction([0.0] * 7, [1.0, float('nan'), 1.0, 1.0,
                                              1.0, 1.0, 1.0],
                                  seeds=list(range(7)), min_n=6)
    assert partial['n'] == 6 and partial['complete_pairs'] is False, partial
    assert partial['significant_raw'] is True, partial
    assert partial['significant'] is False, partial


def test_exact_signflip_baseline():
    p = L.exact_signflip_p([1.0] * 6)['p']
    assert abs(p - 0.03125) < 1e-12, 'all-same-sign n=6 two-sided p must stay 2/64'
    p12 = L.exact_signflip_p([1.0] * 12)['p']
    assert abs(p12 - 2.0 / 4096) < 1e-12


def test_report_gate_refuses_attrition_and_seed_driven():
    # The report builder must expose the same two reliability diagnostics as
    # exp_lib.paired_effect_stats, not just n/raw p/Holm/|mean|>std.
    robust = BR.exact_signflip([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], n_input=6)
    assert robust['n'] == 6 and robust['n_dropped'] == 0, robust
    assert robust['seed_driven'] is False and robust['seed_robust'] is True, robust
    driven = BR.exact_signflip([10.0, 10.0, -10.0], n_input=4)
    assert driven['n'] == 3 and driven['n_dropped'] == 1, driven
    assert driven['seed_driven'] is True and driven['seed_robust'] is False, driven
    assert len(driven['loo_means']) == 3, driven
    # The gate must refuse both a complete-case-only panel and a sign that is
    # owned by a single seed, even when every numeric threshold passes.
    ok = {'n': 6, 'p_exact_signflip': 0.01, 'p_holm': 0.02,
          'mean': 1.0, 'std': 0.5}
    assert BR._sig_gate(ok) is True
    assert BR._sig_gate(dict(ok, n_dropped=1)) is False
    assert BR._sig_gate(dict(ok, seed_driven=True)) is False
    # Seeds skipped by the controlled-comparison gate are an incomplete paired
    # panel too; the report gate must mirror exp_lib.reportable_claim.
    for _key in ('n_skipped_config_mismatch', 'n_skipped'):
        assert BR._sig_gate(dict(ok, **{_key: 1})) is False, _key
        assert BR._sig_gate(dict(ok, **{_key: -1})) is False, _key
        assert BR._sig_gate(dict(ok, **{_key: True})) is False, _key
    # An explicit complete panel / robust sign stays admissible.
    assert BR._sig_gate(dict(ok, n_dropped=0, seed_driven=False)) is True
    assert BR._sig_gate(dict(ok, n_skipped_config_mismatch=0)) is True
    assert BR._sig_gate(dict(ok, n_skipped=0)) is True
    # Corrupt panel accounting must fail closed exactly as in
    # exp_lib.reportable_claim (negative drops, drops exceeding the input,
    # an incomplete-panel flag, or tie-reduced resolution).
    for _bad in (dict(ok, n_dropped=-1),
                 dict(ok, n_input=-1, n_dropped=0),
                 dict(ok, n_input=3, n_dropped=6),
                 dict(ok, n_input=3, n_dropped=0),
                 dict(ok, n_input=7, n_dropped=0),
                 dict(ok, n_input=7),
                 dict(ok, n_input=6.5),
                 dict(ok, n_dropped=0.5),
                 dict(ok, complete_pairs=False),
                 dict(ok, n_nonzero=7),
                 dict(ok, n_nonzero=-1),
                 dict(ok, n_nonzero=1.5),
                 dict(ok, n_nonzero=2)):
        assert BR._sig_gate(_bad) is False, _bad
    assert BR._sig_gate(dict(ok, n_nonzero=6)) is True
    assert BR._sig_gate(dict(ok, n_input=6, n_dropped=0)) is True


def test_exact_signflip_at_scale():
    for n in (21, 22, 25, 40):
        r = L.exact_signflip_p([0.7] * n)
        assert r['method'] == 'exact-mitm', f'n={n} must stay exact via meet-in-the-middle, got {r}'
        assert abs(r['p'] - 2.0 / (1 << n)) < 1e-18, f'all-equal d n={n}: only +/- assignments reach the observed mean, p={r["p"]}'
    assert BR.exact_signflip([0.7] * 22)['method'] == 'exact-mitm', 'the zero-GPU report builder must use the same exact path'
    assert abs(BR.exact_signflip([0.7] * 22)['p_exact_signflip'] - 2.0 / (1 << 22)) < 1e-15
    z = L.exact_signflip_p([0.0] * 24)
    assert z['p'] == 1.0 and z['method'] == 'exact-mitm', 'every sign assignment matches a zero observed mean'
    # The driver used by the confirmation analyzer must not silently substitute
    # a random Monte-Carlo p-value for the pre-registered exact test at n>20.
    v = V7.exact_sign_permutation([0.7] * 21)
    assert abs(v['p_exact_signflip'] - 2.0 / (1 << 21)) < 1e-18 and v.get('method') == 'exact-mitm', v
    for n in (6, 21):
        base = [1.0] * (n - 1) + [-1.0]
        ref = L.exact_signflip_p(base)['p']
        for scale in (1e-300, 1e150, 1e308):
            d = [x * scale for x in base]
            got = L.exact_signflip_p(d)['p']
            br = BR.exact_signflip(d)['p_exact_signflip']
            assert abs(got - ref) <= max(1e-18, ref * 1e-12), (n, scale, got, ref)
            assert abs(br - ref) <= max(1e-18, ref * 1e-12), (n, scale, br, ref)
    rng = np.random.default_rng(11)
    d = rng.normal(size=21)
    got = L.exact_signflip_p(d)['p']
    ref = _brute_signflip_p(d)
    assert abs(got - ref) < 1e-15, f'MITM count must match full enumeration at n=21 ({got} vs {ref})'
    # The zero-GPU report builder must use the same n*1e-12 tie tolerance as
    # exp_lib; a 1e-12 sum threshold can disagree near the decision boundary.
    assert abs(BR.exact_signflip(d)['p_exact_signflip'] - got) < 1e-15, (
        BR.exact_signflip(d)['p_exact_signflip'], got)


def test_signflip_saddlepoint_large_n():
    # n>40 must not use the centered-SD normal approximation (whose variance
    # shrinks with the effect) and must not be a random Monte-Carlo draw.
    rng = np.random.default_rng(17)
    for n, shift in ((45, 0.1), (45, 0.3), (60, 0.2), (100, 0.15)):
        d = rng.normal(shift, 1.0, size=n)
        d = d - d.mean() + shift
        a = L.exact_signflip_p(d)
        b = L.exact_signflip_p(d)
        assert a['method'].startswith('saddlepoint'), a
        assert a['p'] == b['p'], 'saddlepoint fallback must be deterministic'
        # Fixed-seed Monte-Carlo reference (200k draws) — the saddlepoint tail
        # must agree within tolerance for the larger shifts.
        draws = 200000
        obs = abs(d.mean())
        signs = rng.choice([-1.0, 1.0], size=(draws, n))
        mc = float((np.abs((signs * d).mean(1)) >= obs - 1e-12).mean())
        tol = max(0.003, 5.0 * math.sqrt(max(mc * (1.0 - mc), 1e-9) / draws))
        assert abs(a['p'] - mc) <= tol, f'n={n} shift={shift}: saddlepoint {a["p"]} vs MC {mc}'
        # Never below the exact two-sided resolution floor 2/2^n.
        assert a['p'] >= L.signflip_p_floor(n) - 1e-18, (a, n)
        # The report builder must return the same deterministic number.
        br = BR.exact_signflip(list(d))
        assert br['method'].startswith('saddlepoint'), br
        assert abs(br['p_exact_signflip'] - a['p']) < 1e-12, (br, a)
    # All deltas equal: only the two all-same-sign assignments reach the
    # observed sum, so the p is exactly the floor (checked at n=41, > 40).
    eq = L.exact_signflip_p([0.7] * 41)
    assert eq['method'] == 'saddlepoint' and abs(eq['p'] - 2.0 / (1 << 41)) < 1e-30, eq
    assert abs(BR.exact_signflip([0.7] * 41)['p_exact_signflip'] - 2.0 / (1 << 41)) < 1e-30
    # Near-maximum observed sum: the saddlepoint root sits at a large t, so
    # the implementation must stay finite (no cosh overflow) and above floor.
    near = [1.0] * 50
    near[0] = -0.998
    nr = L.exact_signflip_p(near)
    assert nr['method'].startswith('saddlepoint') and 0.0 < nr['p'] < 1.0, nr
    brn = BR.exact_signflip(near)
    assert abs(brn['p_exact_signflip'] - nr['p']) < 1e-12, (brn, nr)
    # The confirmation driver delegates to exp_lib, so it inherits the fix.
    v = V7.exact_sign_permutation(list(rng.normal(0.2, 1.0, size=50)))
    assert v['method'].startswith('saddlepoint'), v
    assert 'saddlepoint' in v['test'], v
    # A positive rescaling of every delta cannot change the sign-flip p.
    # Very large/small magnitudes used to overflow/underflow the squared
    # null variance and either raise or silently collapse to the floor.
    base = [1.0] * 40 + [-1.0]
    ref = L.exact_signflip_p(base)['p']
    for scale in (1e-300, 1e-150, 1e150, 1e154, 1e308):
        d = [x * scale for x in base]
        r = L.exact_signflip_p(d)
        br = BR.exact_signflip(d)
        tol = max(1e-18, ref * 1e-12)
        assert r['method'].startswith('saddlepoint'), (scale, r)
        assert abs(r['p'] - ref) <= tol, (scale, r['p'], ref)
        assert abs(br['p_exact_signflip'] - ref) <= tol, (scale, br, ref)


def test_wilcoxon_exact_dp():
    def brute_wilcoxon(d):
        d = np.asarray([float(x) for x in d if math.isfinite(float(x))], dtype=float)
        d = d[np.abs(d) > 1e-15]
        n = len(d)
        if n == 0:
            return float('nan')
        order = np.argsort(np.abs(d))
        ranks = np.empty(n, dtype=float)
        a = np.abs(d)[order]
        i = 0
        while i < n:
            j = i + 1
            while j < n and abs(a[j] - a[i]) <= 1e-15:
                j += 1
            ranks[order[i:j]] = (i + j + 1) / 2.0
            i = j
        wp = float(ranks[d > 0].sum())
        total = float(ranks.sum())
        w = min(wp, total - wp)
        cnt = 0
        for mask in range(1 << n):
            sm = sum(ranks[k] for k in range(n) if mask >> k & 1)
            if sm <= w + 1e-12 or sm >= total - w - 1e-12:
                cnt += 1
        return cnt / float(1 << n)
    rng = np.random.default_rng(5)
    for n in range(1, 13):
        for variant in range(3):
            d = rng.normal(size=n)
            if variant == 1:
                d = np.round(d, 0)          # ties + zeros
            elif variant == 2:
                d = np.round(np.abs(d), 1)  # many ties, all positive
            got = L.wilcoxon_signed_rank_p(d)
            if not np.any(np.abs(d) > 1e-15):
                continue
            assert got['method'] == 'exact-dp', got
            assert abs(got['p'] - brute_wilcoxon(d)) < 1e-12, (n, variant, got, brute_wilcoxon(d))
    # The DP stays exact well beyond the seed counts the scale ladder may top
    # up to.
    d = rng.normal(0.2, 1.0, size=100)
    a = L.wilcoxon_signed_rank_p(d)
    b = L.wilcoxon_signed_rank_p(d)
    assert a['method'] == 'exact-dp' and a['p'] == b['p'], a
    assert 0.0 <= a['p'] <= 1.0
    # A perfectly balanced zero-effect panel has p=1 (both tails meet).
    assert L.wilcoxon_signed_rank_p([1.0, -1.0, 1.0, -1.0])['p'] == 1.0


def test_timeguard_variant_calibration():
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, 'state.json')
        g = L.TimeGuard(dict(total_hours=10, margin=0.9, state_path=path))
        key_a = g._class_key(64, 2, 256, 4, 'csa_fixed')
        key_b = g._class_key(64, 2, 256, 4, 'full_matched')
        assert key_a != key_b and key_a.endswith('|csa_fixed'), (key_a, key_b)
        # A 10 s / 10 step calibration for variant a must price variant a from
        # its exact class, not from the shared geometry rate.
        g.record_run(10.0, 10, 64, 2, 256, 4, calib_seconds=10.0, variant='csa_fixed')
        a = g.estimate_seconds(100, d=64, n_layers=2, seq_len=256, batch_size=4, variant='csa_fixed')
        assert abs(a - 100 * 1.0 * 1.15) < 1e-9, a
        # Variant b must not borrow a's exact rate; it uses the normalised
        # fallback (which may be the same by coincidence only if the factors
        # match, so check the code path via a deliberately different rate).
        g.record_run(20.0, 10, 64, 2, 256, 4, calib_seconds=20.0, variant='full_matched')
        b = g.estimate_seconds(100, d=64, n_layers=2, seq_len=256, batch_size=4, variant='full_matched')
        assert abs(b - 100 * 2.0 * 1.15) < 1e-9, b
        # A geometry-only calibration remains the fallback for a variant with
        # no per-variant entry (backwards compatible with v7-v12 state files).
        g2 = L.TimeGuard(dict(total_hours=10, margin=0.9, state_path=os.path.join(tmp, 's2.json')))
        g2.record_run(5.0, 10, 128, 2, 256, 4, calib_seconds=5.0)
        c = g2.estimate_seconds(100, d=128, n_layers=2, seq_len=256, batch_size=4, variant='whatever')
        assert abs(c - 100 * 0.5 * 1.15) < 1e-9, c
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_paired_effect_resolution():
    r = L.paired_effect_stats([1.0, 1.1, 0.9, 1.2], alpha=0.05)
    assert r['resolution_limited'] is True, 'n=4 can never reach 0.05 (floor 0.125) — descriptive tier'
    # Missing/non-finite pairs must stay visible: the p-value is computed on
    # the survivors, so a complete-case result cannot be presented as if the
    # whole paired panel were measured.
    r = L.paired_effect_stats([1.0, float('nan'), -0.5], seeds=[0, 1, 2])
    assert r['n'] == 2 and r['n_input'] == 3 and r['n_dropped'] == 1, r
    assert r['dropped_seeds'] == [1] and r['seeds'] == [0, 2], r
    assert r['complete_pairs'] is False, r
    try:
        L.paired_effect_stats([1.0, 2.0], seeds=[0, 0])
    except ValueError:
        pass
    else:
        raise AssertionError('duplicate paired seeds must be rejected')
    try:
        L.paired_effect_stats([1.0, 2.0], seeds=[0])
    except ValueError:
        pass
    else:
        raise AssertionError('a mismatched seed vector must be rejected')


def test_paired_effect_extreme_scale_is_not_zero_variance():
    base = [1.0] * 5 + [2.0]
    ref = L.paired_effect_stats(base)
    tiny = [x * 1e-200 for x in base]
    r = L.paired_effect_stats(tiny)
    # Squaring 1e-200 underflows: the old path reported std=sem=0, an infinite
    # d_z and a zero-width CI while the panel was genuinely variable.
    assert r['std'] > 0.0 and r['sem'] > 0.0, r
    assert math.isfinite(r['cohens_dz']), r
    assert r['ci95_low'] < r['ci95_high'], r
    assert math.isclose(r['std'] / r['mean'], ref['std'] / ref['mean'],
                        rel_tol=1e-12), (r, ref)
    assert r['p_exact_signflip'] == ref['p_exact_signflip'], (r, ref)
    assert r['p_wilcoxon'] == ref['p_wilcoxon'], (r, ref)
    assert r['n_nonzero'] == ref['n_nonzero'] and r['significant'] == ref['significant'], (r, ref)
    assert math.isclose(r['loo_means'][-1], 1e-200, rel_tol=1e-12), r['loo_means']
    # The report-side std and the TOST/bootstrap helpers share the scale
    # convention; a tiny but non-degenerate panel must not look exactly equal.
    assert math.isclose(BR.sample_std(tiny) / 1e-200, math.sqrt(1.0 / 6.0),
                        rel_tol=1e-12), BR.sample_std(tiny)
    t = L.tost_paired([1e-200, -2e-200, 3e-200], margin=2e-200)
    assert t['ci'][0] < t['ci'][1] and t['p_tost'] > 0.0, t


    b = L.bootstrap_mean_ci(tiny)
    assert math.isfinite(b['mean']) and b['mean'] > 0.0, b
    assert b['ci_low'] < b['ci_high'], b
    # Huge finite values overflow the same squares; the mean/std must remain
    # finite and the panel must stay within seed noise.
    h = L.paired_effect_stats([1e308, 1e308, -1e308])
    assert math.isfinite(h['mean']) and math.isfinite(h['std']) and h['std'] > 0.0, h
    assert math.isclose(h['mean'], 1e308 / 3.0, rel_tol=1e-12), h
    assert h['within_seed_noise'] is True and h['significant'] is False, h
    assert math.isclose(BR.sample_std([1e308, 1e308, -1e308]),
                        1e308 * math.sqrt(4.0 / 3.0), rel_tol=1e-12)
    _driven = [1e-310, 1e-310, -1e-310]
    assert L.paired_effect_stats(_driven)['seed_driven'] is True
    assert BR.exact_signflip(_driven)['seed_driven'] is True


def test_meta_analysis_extreme_scale_is_not_zero_variance():
    base = [
        {'label': 'S0', 'scale': 1.0, 'seeds': [0, 1, 2, 3],
         'effects': [0.05, 0.12, 0.08, 0.10]},
        {'label': 'S1', 'scale': 10.0, 'seeds': [0, 1, 2, 3],
         'effects': [0.20, 0.18, 0.25, 0.22]},
        {'label': 'S2', 'scale': 100.0, 'seeds': [0, 1, 2, 3],
         'effects': [0.35, 0.31, 0.40, 0.36]},
    ]
    ref = L.meta_analyze_effects(base)
    assert ref['status'] == 'ok' and ref['n_usable'] == 3, ref
    assert ref['zero_variance_points'] == [], ref
    for factor in (1e-200, 1e200):
        scaled = [dict(p, effects=[x * factor for x in p['effects']])
                  for p in base]
        got = L.meta_analyze_effects(scaled)
        assert got['status'] == ref['status'], (factor, got)
        assert got['n_usable'] == ref['n_usable'], (factor, got)
        assert got['zero_variance_points'] == [], (factor, got)
        assert got['non_estimable_points'] == [], (factor, got)
        assert got['poolable'] == ref['poolable'], (factor, got)
        for key in ('mean', 'se'):
            assert math.isclose(got['fixed_effect'][key] / factor,
                                ref['fixed_effect'][key], rel_tol=1e-12), (factor, key, got)
            assert math.isclose(got['random_effects'][key] / factor,
                                ref['random_effects'][key], rel_tol=1e-12), (factor, key, got)
        for _got, _ref in zip(got['per_point'], ref['per_point']):
            assert math.isclose(_got['mean'] / factor, _ref['mean'],
                                rel_tol=1e-12), (factor, _got)
            assert math.isclose(_got['se'] / factor, _ref['se'],
                                rel_tol=1e-12), (factor, _got)


def test_scaling_trend_extreme_scale_is_invariant():
    base = []
    for i in range(3):
        effects = [0.1 * i + 0.01 * j + (0.003 if j == i else 0.0)
                   for j in range(6)]
        base.append({'label': 'S%d' % i, 'scale': 10.0 ** i,
                     'seeds': list(range(6)), 'effects': effects})
    ref = L.scaling_trend(base, min_seeds=3, min_points=3)
    ref_unpaired = L.scaling_trend(
        [{'label': p['label'], 'scale': p['scale'], 'effects': p['effects']}
         for p in base], min_seeds=3, min_points=3)
    for factor in (1e-200, 1e200):
        scaled = [dict(p, effects=[e * factor for e in p['effects']])
                  for p in base]
        got = L.scaling_trend(scaled, min_seeds=3, min_points=3)
        assert got['verdict'] == ref['verdict'], (factor, got)
        assert math.isclose(got['wls']['slope'] / factor,
                            ref['wls']['slope'], rel_tol=1e-12), (factor, got)
        assert math.isclose(got['wls']['se'] / factor,
                            ref['wls']['se'], rel_tol=1e-12), (factor, got)
        assert math.isclose(got['wls']['p'], ref['wls']['p'],
                            rel_tol=1e-12), (factor, got)
        assert got['paired']['p_exact_signflip'] == ref['paired']['p_exact_signflip']
        assert math.isclose(got['paired']['slope_mean'] / factor,
                            ref['paired']['slope_mean'], rel_tol=1e-12), (factor, got)
        assert math.isclose(got['paired']['se'] / factor,
                            ref['paired']['se'], rel_tol=1e-12), (factor, got)
        unpaired = [{'label': p['label'], 'scale': p['scale'],
                     'effects': [e * factor for e in p['effects']]}
                    for p in base]
        got_unpaired = L.scaling_trend(unpaired, min_seeds=3, min_points=3)
        assert got_unpaired['verdict'] == ref_unpaired['verdict'], (factor, got_unpaired)
        assert math.isclose(got_unpaired['wls']['p'], ref_unpaired['wls']['p'],
                            rel_tol=1e-12), (factor, got_unpaired)


def test_scaling_trend_ragged_pairs():
    pts = [{'label': 'A', 'scale': 0.1, 'effects': [0.0, 1.0, 2.0, 3.0], 'seeds': [0, 1, 2, 3]},
           {'label': 'B', 'scale': 1.0, 'effects': [1.0, 0.0], 'seeds': [0, 1]}]
    r = L.scaling_trend(pts, min_seeds=2, min_points=2)
    assert r['paired']['per_seed_slopes'] == {'0': 1.0, '1': -1.0}, r['paired']
    assert r['paired']['slope_mean'] == 0.0, r['paired']
    for bad in (
        [{'label': 'A', 'scale': 1, 'effects': [1, 2], 'seeds': [0, 1]},
         {'label': 'A', 'scale': 2, 'effects': [2, 3], 'seeds': [0, 1]}],
        [{'label': 'A', 'scale': 1, 'effects': [1, 2], 'seeds': [0, 0]},
         {'label': 'B', 'scale': 2, 'effects': [2, 3], 'seeds': [0, 1]}],
    ):
        try:
            L.scaling_trend(bad, min_points=2)
        except ValueError:
            pass
        else:
            raise AssertionError('duplicate labels/seeds must be rejected instead of mis-paired')
    # Seed identity is an exact integer across every scale helper.  A
    # fractional value would quietly become a different paired draw, and
    # True would collide with seed 1 in the complete-case dict.
    # Non-iterable seed/plan descriptors are malformed input, not a crash.
    for _bad in ({'seeds': 3}, {'planned_seeds': 3}):
        try:
            L._scale_points([dict({'label': 'A', 'scale': 1,
                                   'effects': [1.0]}, **_bad)])
        except ValueError:
            pass
        else:
            raise AssertionError(f'non-iterable {_bad!r} was accepted')
    for _bad_seed in (0.5, True, 'not-an-int'):
        for _call in (
            lambda: L._scale_points([{'label': 'A', 'scale': 1,
                                      'effects': [1.0], 'seeds': [_bad_seed]}]),
            lambda: L.paired_log_ratio([1.0], [2.0], seeds=[_bad_seed]),
            lambda: L.scale_interaction([1.0], [2.0], seeds=[_bad_seed]),
        ):
            try:
                _call()
            except ValueError:
                pass
            else:
                raise AssertionError(f'non-integer seed {_bad_seed!r} was accepted')


def test_scale_missing_seed_identity_is_not_paired():
    # Positional effect indices are not training seeds.  Without declared seed
    # identities the cross-scale paired slope/interaction must be withheld,
    # even though the effect vectors happen to have equal length.
    pts = [{'label': 'P1', 'scale': 1.0, 'effects': [0.0, 1.0, 2.0]},
           {'label': 'P2', 'scale': 10.0, 'effects': [1.0, 2.0, 3.0]}]
    r = L.scaling_trend(pts, min_seeds=3, min_points=2)
    assert r['seed_pairing'] == 'unattested', r
    assert r['paired']['n_complete_seeds'] == 0, r['paired']
    assert r['paired']['seed_identity_attested'] is False, r['paired']
    assert r['primary_inference'].startswith('WLS'), r['primary_inference']
    assert r['interaction_extremes']['status'] == 'seed_identity_unattested', r['interaction_extremes']
    m = L.meta_analyze_effects(pts)
    assert m['seed_pairing'] == 'positional_only' and m['n_points_with_declared_seeds'] == 0, m
    # A partly declared family is still not attested: one undeclared point
    # could have run a different seed set.
    mixed = [pts[0], dict(pts[1], seeds=[0, 1, 2])]
    rm = L.scaling_trend(mixed, min_seeds=3, min_points=2)
    assert rm['seed_pairing'] == 'unattested' and rm['paired']['n_complete_seeds'] == 0, rm
    assert L.meta_analyze_effects(mixed)['seed_pairing'] == 'partial'
    # Both declared: the paired path is available again.
    both = [dict(pts[0], seeds=[0, 1, 2]), dict(pts[1], seeds=[0, 1, 2])]
    rb = L.scaling_trend(both, min_seeds=3, min_points=2)
    assert rb['seed_pairing'] == 'declared' and rb['paired']['n_complete_seeds'] == 3, rb
    assert rb['interaction_extremes'].get('seed_pairing') == 'declared', rb['interaction_extremes']
    # The unpaired WLS fallback must also honour min_seeds: a perfect line
    # through two-draw points is not a confirmable trend.
    low = [{'label': 'W%d' % i, 'scale': 10.0 ** i,
            'effects': [float(i)] * 2} for i in range(3)]
    rw = L.scaling_trend(low, min_seeds=3, min_points=3)
    assert rw['primary_inference'].startswith('WLS'), rw['primary_inference']
    assert rw['underpowered'] is True and rw['significant'] is False, rw
    assert rw['verdict'] == 'underpowered', rw['verdict']
    assert rw['underpowered_points'] == ['W0', 'W1', 'W2'], rw



def test_report_sample_std_accepts_exact_huge_integers():
    H = 10 ** 400
    # Converting H to float raised before any sample spread was computed, so a
    # perfectly well-defined SD (or a zero-variance panel) crashed the report.
    assert BR.sample_std([H, H]) == 0.0
    assert math.isclose(BR.sample_std([H, H + 1]), math.sqrt(0.5), rel_tol=1e-12)
    assert math.isclose(BR.sample_std([H, H + 2]), math.sqrt(2.0), rel_tol=1e-12)
    assert math.isclose(BR.sample_std([H, H + 1, H + 2]), 1.0, rel_tol=1e-12)
    # A spread genuinely beyond float64 is reported as inf, not an exception.
    assert math.isinf(BR.sample_std([H, 0]))
    # The normal and tiny/huge-float paths keep their previous values.
    assert math.isclose(BR.sample_std([1.0, 2.0, 3.0]), 1.0, rel_tol=1e-12)
    assert math.isclose(BR.sample_std([1e-300, 2e-300]),
                        math.sqrt(0.5) * 1e-300, rel_tol=1e-12)
