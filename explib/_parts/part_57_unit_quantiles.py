"""Distribution of a paired PPL effect across evaluation units.

The frozen endpoint is a **token-pooled mean** log PPL ratio, and the seed-level
exact sign-flip test asks whether that mean is distinguishable from zero.  A
mean (or even its cluster interval) does not say whether the method improves
the *typical* evaluation unit, whether it buys a small average gain by harming
a tail of units, or whether the gain is carried by a few very long articles.
Those are different scientific claims, and a report that shows only the mean
lets the reader silently substitute one for another.

This fragment adds one descriptive readout: the distribution of the paired
per-unit effect

    d[s, u] = (nll_a[s, u] - nll_b[s, u]) / tokens[u]

where ``d`` is the log PPL ratio on unit ``u`` in training seed ``s`` (negative
means arm ``a`` has lower PPL).  It reports:

* quantiles of the seed-averaged unit effect with **distribution-free
  order-statistic intervals** (binomial ranks), so the interval does not rely
  on a bootstrap RNG and remains honest at the 8-64 unit panels this repository
  evaluates;
* the fraction of units improved, with an exact Clopper-Pearson interval
  (ties count as "not better"), and the same fraction weighted by tokens;
* a gross gain/harm decomposition: how much log improvement and how much log
  harm coexist in the same panel, and their net;
* best/worst decile means, a tail-asymmetry readout and the share of gross
  harm/gain carried by the worst/best decile;
* per-seed quantiles and their spread, plus per-unit seed sign consistency,
  so unit heterogeneity is not confused with training-seed noise;
* the sufficient statistics (the per-seed NLL-difference matrix, token counts,
  unit ids, seeds and requested quantiles) needed by
  ``audit_gates.py unit_quantiles`` to replay every derived number.

Nothing here is a new confirmatory test.  It adds no p-value to the Holm
family, does not change ``p_exact_signflip``/``p_holm``/``verified``/any frozen
``confirmed`` verdict, and conditions on the trained seeds.  The unit
quantiles describe the observed evaluation panel; they are not a population
claim, and the equal-unit quantiles are deliberately a different estimand from
the token-pooled mean.  The companion length-stratification readout asks
whether the pooled mean is composition-sensitive; this fragment asks what the
effect looks like across units.
"""

import math as _uq_math

import numpy as _uq_np


UNIT_QUANTILE_POLICY = {
    'version': 'unit-quantile-1',
    'quantiles': (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95),
    'conf': 0.95,
    'min_units': 8,
    'min_seeds': 1,
    'tail_fraction': 0.10,
    'quantile_method': ('linear interpolation between order statistics '
                        '(Hyndman-Fan type 7)'),
    'interval_method': ('distribution-free order-statistic interval from '
                        'exact binomial ranks; a one-sided side is reported '
                        'as null rather than a fabricated finite bound'),
    'improvement_interval_method': ('exact Clopper-Pearson binomial interval; '
                                    'a tied unit counts as not better'),
    'role': ('descriptive distributional readout of one paired PPL contrast; '
             'it adds no p-value to the frozen family and never changes a '
             'frozen verdict'),
    'boundary': (
        'post-hoc descriptive distribution of the paired per-unit log PPL '
        'ratio. The quantiles are equal-unit estimands, not the token-pooled '
        'mean the frozen test uses; the two can disagree in sign when long '
        'units behave differently, and that disagreement is the point of the '
        'readout. Intervals are pointwise, not simultaneous over the quantile '
        'profile, and condition on the trained seeds: they cover evaluation-'
        'unit sampling only. The order-statistic quantile interval assumes '
        'independent units and is conservative at very small n; an unbounded '
        'side is reported as null. The unit effect is seed-averaged, so its '
        'spread still contains residual training-seed noise (visible in '
        'per_seed_quantiles/seed_range and the variance components); it is '
        'not an estimate of latent between-unit heterogeneity. The '
        'Clopper-Pearson fraction treats a tie as not better. Tail means and '
        'the gross gain/harm split are point estimates without intervals. No '
        'p-value is added to the Holm family, and no confirmed/verified field '
        'is produced or changed.'),
}

UNIT_QUANTILE_STATUSES = ('ok', 'not_estimable')


def _uq_conf(value):
    if isinstance(value, (bool, _uq_np.bool_)):
        raise ValueError(f'conf must be a probability in (0, 1), got {value!r}')
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'conf must be a probability in (0, 1), got {value!r}') from None
    if not (_uq_math.isfinite(out) and 0.0 < out < 1.0):
        raise ValueError(f'conf must be a probability in (0, 1), got {value!r}')
    return out


def _uq_quantiles(value):
    if value is None:
        return [float(q) for q in UNIT_QUANTILE_POLICY['quantiles']]
    if isinstance(value, (str, bytes, bytearray, dict)):
        raise ValueError('quantiles must be an iterable of probabilities in (0, 1)')
    try:
        raw = list(value)
    except TypeError:
        raise ValueError('quantiles must be an iterable of probabilities in (0, 1)') from None
    if not raw or len(raw) > 20:
        raise ValueError(f'quantiles must contain between 1 and 20 values, got {len(raw)}')
    out = []
    for q in raw:
        if isinstance(q, (bool, _uq_np.bool_)):
            raise ValueError(f'quantiles must be probabilities, got {q!r}')
        try:
            f = float(q)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f'quantiles must be probabilities, got {q!r}') from None
        if not (_uq_math.isfinite(f) and 0.0 < f < 1.0):
            raise ValueError(f'quantiles must be probabilities in (0, 1), got {q!r}')
        out.append(f)
    if any(out[i] >= out[i + 1] for i in range(len(out) - 1)):
        raise ValueError(f'quantiles must be strictly increasing, got {out!r}')
    return out


def _uq_tail_fraction(value):
    if isinstance(value, (bool, _uq_np.bool_)):
        raise ValueError(f'tail_fraction must be in (0, 0.5), got {value!r}')
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'tail_fraction must be in (0, 0.5), got {value!r}') from None
    if not (_uq_math.isfinite(out) and 0.0 < out < 0.5):
        raise ValueError(f'tail_fraction must be in (0, 0.5), got {value!r}')
    return out


def _uq_div(num, den):
    """``num / den`` with an exact positive-integer denominator."""
    try:
        return float(num) / int(den)
    except OverflowError:
        from fractions import Fraction as _Fraction
        try:
            return float(_Fraction(num) / int(den))
        except (OverflowError, ValueError, ZeroDivisionError):
            return None


def _uq_mean(values):
    """Overflow-safe mean of a finite scalar vector."""
    vals = [float(v) for v in values]
    if not vals:
        return None
    scale = max(abs(v) for v in vals)
    if not _uq_math.isfinite(scale):
        return None
    if scale == 0.0:
        return 0.0
    try:
        z = [v / scale for v in vals]
        mz = _uq_math.fsum(z) / len(z)
        out = scale * mz
    except (OverflowError, ValueError):
        return None
    return out if _uq_math.isfinite(out) else None


def _uq_lerp(lo, hi, frac):
    if frac <= 0.0:
        return float(lo)
    if frac >= 1.0:
        return float(hi)
    try:
        out = _uq_math.fsum([(1.0 - frac) * lo, frac * hi])
    except (OverflowError, ValueError):
        return None
    return out if _uq_math.isfinite(out) else None


def _uq_quantile(sorted_values, q):
    """Linear-interpolation quantile of an already sorted finite vector."""
    n = len(sorted_values)
    if n == 0:
        return None
    h = (n - 1) * float(q)
    if not _uq_math.isfinite(h) or h < 0.0:
        return None
    i = int(_uq_math.floor(h))
    if i >= n - 1:
        return float(sorted_values[-1])
    return _uq_lerp(sorted_values[i], sorted_values[i + 1], h - i)


def _uq_sign(value):
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not _uq_math.isfinite(f):
        return None
    if f > 0.0:
        return 1
    if f < 0.0:
        return -1
    return 0


def _uq_sort_values(values):
    return sorted(float(v) for v in values)


def _uq_binom_cdf(k, n, p):
    return float(_binomial_cdf_stable(int(k), int(n), float(p)))


def _uq_binom_sf(k, n, p):
    return float(_binomial_sf_stable(int(k), int(n), float(p)))


def _uq_order_stat_interval(sorted_values, q, conf):
    """Pointwise distribution-free interval for one quantile.

    The exact coverage of ``[X_(k), X_(l)]`` is
    ``P(k <= B <= l-1)`` for ``B ~ Bin(n, q)``.  The ranks are chosen so each
    missing tail is at most ``(1-conf)/2``; if a side cannot be bounded with
    that tail probability it is returned as ``None`` instead of replacing it
    with the sample minimum/maximum (which would under-cover).
    """
    n = len(sorted_values)
    alpha = 1.0 - float(conf)
    target = alpha / 2.0
    # Largest k in [1, n] with P(B <= k-1) <= alpha/2.
    lo, hi, k_rank = 1, n, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        c = _uq_binom_cdf(mid - 1, n, q)
        if c <= target:
            k_rank = mid
            lo = mid + 1
        else:
            hi = mid - 1
    # Smallest l in [1, n] with P(B <= l-1) >= 1-alpha/2.
    lo, hi, l_rank = 1, n, n + 1
    while lo <= hi:
        mid = (lo + hi) // 2
        c = _uq_binom_cdf(mid - 1, n, q)
        if c >= 1.0 - target:
            l_rank = mid
            hi = mid - 1
        else:
            lo = mid + 1
    low = float(sorted_values[k_rank - 1]) if k_rank >= 1 else None
    high = float(sorted_values[l_rank - 1]) if l_rank <= n else None
    if k_rank == 0:
        coverage = _uq_binom_cdf(l_rank - 1, n, q) if l_rank <= n else 1.0
    elif l_rank == n + 1:
        coverage = 1.0 - _uq_binom_cdf(k_rank - 1, n, q)
    else:
        coverage = (_uq_binom_cdf(l_rank - 1, n, q)
                    - _uq_binom_cdf(k_rank - 1, n, q))
    if not (_uq_math.isfinite(coverage) and 0.0 <= coverage <= 1.0):
        coverage = None
    method = ('distribution-free order statistic; rank interval from exact '
              'binomial tails')
    if k_rank == 0 or l_rank == n + 1:
        method += ' (one side unbounded at this n)'
    return {
        'ci95': [low, high],
        'ci_rank': [int(k_rank) if k_rank >= 1 else None,
                    int(l_rank) if l_rank <= n else None],
        'ci_coverage': (float(coverage) if coverage is not None else None),
        'ci_method': method,
    }


def _uq_clopper_pearson(k, n, conf):
    """Exact Clopper-Pearson interval for ``k`` successes out of ``n``."""
    if n <= 0:
        return None
    alpha = 1.0 - float(conf)
    target = alpha / 2.0
    if k <= 0:
        low = 0.0
    else:
        lo, hi = 0.0, 1.0
        for _ in range(90):
            mid = (lo + hi) / 2.0
            if _uq_binom_sf(k, n, mid) < target:
                lo = mid
            else:
                hi = mid
        low = (lo + hi) / 2.0
    if k >= n:
        high = 1.0
    else:
        lo, hi = 0.0, 1.0
        for _ in range(90):
            mid = (lo + hi) / 2.0
            if _uq_binom_cdf(k, n, mid) > target:
                lo = mid
            else:
                hi = mid
        high = (lo + hi) / 2.0
    if not (_uq_math.isfinite(low) and _uq_math.isfinite(high)) or low > high:
        return None
    return [float(low), float(high)]


def _uq_pooled_row(row, total_tokens):
    """Token-pooled log ratio for one seed row, without a float64 detour.

    ``sum_i x_i / T`` is the same per-token ratio as
    ``sum_i (T_i/T) * (x_i/T_i)``, but forming the normalised weights first
    underflows when a unit is tiny relative to a token total beyond float64
    (e.g. ``1 / 10**400 == 0.0``).  The fast path keeps the historical
    term-by-term sum; only an unrepresentable term falls back to an exact
    rational accumulation, so a representable pooled ratio survives.
    """
    vals = [float(v) for v in row]
    terms = []
    need_exact = False
    for v in vals:
        term = _uq_div(v, total_tokens)
        if term is None or (term == 0.0 and v != 0.0):
            need_exact = True
            break
        terms.append(term)
    if not need_exact:
        try:
            out = float(_uq_math.fsum(terms))
        except (OverflowError, ValueError):
            return None
        return out if _uq_math.isfinite(out) else None
    from fractions import Fraction as _Fraction
    try:
        exact = _Fraction(0, 1)
        den = int(total_tokens)
        for v in vals:
            exact += _Fraction(v) / den
        out = float(exact)
    except (OverflowError, ValueError, ZeroDivisionError):
        return None
    return out if _uq_math.isfinite(out) else None


def _uq_exp_ratio(value):
    try:
        out = _uq_math.exp(float(value))
    except (OverflowError, ValueError):
        return None
    if not (_uq_math.isfinite(out) and out > 0.0):
        return None
    return float(out)


def _uq_relative_pct(value):
    try:
        out = 100.0 * _uq_math.expm1(float(value))
    except (OverflowError, ValueError):
        return None
    return float(out) if _uq_math.isfinite(out) else None


def _uq_not_estimable(reason, *, n_units=None, n_seeds=None, n_tokens=None,
                      quantiles=None, conf=None):
    return {
        'kind': 'unit_quantile_profile',
        'version': UNIT_QUANTILE_POLICY['version'],
        'status': 'not_estimable',
        'reason': str(reason),
        'n_units': n_units,
        'n_seeds': n_seeds,
        'n_tokens': n_tokens,
        'policy': {
            'version': UNIT_QUANTILE_POLICY['version'],
            'quantiles': [float(q) for q in (
                UNIT_QUANTILE_POLICY['quantiles'] if quantiles is None else quantiles)],
            'conf': (float(UNIT_QUANTILE_POLICY['conf']) if conf is None else float(conf)),
            'min_units': int(UNIT_QUANTILE_POLICY['min_units']),
            'min_seeds': int(UNIT_QUANTILE_POLICY['min_seeds']),
            'tail_fraction': float(UNIT_QUANTILE_POLICY['tail_fraction']),
            'role': UNIT_QUANTILE_POLICY['role'],
            'boundary': UNIT_QUANTILE_POLICY['boundary'],
        },
        'boundary': UNIT_QUANTILE_POLICY['boundary'],
    }


def _uq_matrix(values):
    try:
        raw = _uq_np.asarray(values)
    except (TypeError, ValueError, OverflowError):
        raise ValueError('unit_quantile_profile: per_seed_nll_diff must be numeric') from None
    if _contains_bool_value(values) or raw.dtype.kind == 'b':
        raise ValueError('unit_quantile_profile: booleans are not NLL differences')
    if raw.dtype.kind in ('U', 'S'):
        raise ValueError('unit_quantile_profile: numeric strings are not NLL differences')
    try:
        arr = _uq_np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError):
        raise ValueError('unit_quantile_profile: per_seed_nll_diff must be numeric') from None
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError(
            f'unit_quantile_profile: expected [n_seeds, n_units], got shape {arr.shape}')
    if not _uq_np.all(_uq_np.isfinite(arr)):
        raise ValueError('unit_quantile_profile: every NLL difference must be finite')
    return arr


def unit_quantile_profile(per_seed_nll_diff, n_tokens, *, unit_ids=None,
                          seeds=None, quantiles=None, conf=None,
                          tail_fraction=None):
    """Distribution of the paired per-unit log PPL effect.

    ``per_seed_nll_diff`` is ``[n_seeds, n_units]`` summed NLL differences
    (arm a minus arm b, evaluated on the same tokens), and ``n_tokens`` is the
    shared positive integer token count of each unit.  A unit effect is
    ``nll_diff / n_tokens``, i.e. the log PPL ratio; it is negative when arm a
    has the lower PPL.

    Returns a strict-JSON block.  A valid panel too small to describe returns
    ``status='not_estimable'`` with a reason; malformed inputs raise
    ``ValueError``.  No p-value, Holm member or frozen field is produced.
    """
    qs = _uq_quantiles(quantiles)
    conf = _uq_conf(UNIT_QUANTILE_POLICY['conf'] if conf is None else conf)
    tf = _uq_tail_fraction(
        UNIT_QUANTILE_POLICY['tail_fraction'] if tail_fraction is None else tail_fraction)
    A = _uq_matrix(per_seed_nll_diff)
    S, D = A.shape
    try:
        raw_t = _uq_np.asarray(n_tokens).reshape(-1)
    except (TypeError, ValueError, OverflowError):
        raise ValueError('unit_quantile_profile: n_tokens must be numeric') from None
    if _contains_bool_value(n_tokens) or raw_t.dtype.kind == 'b':
        raise ValueError('unit_quantile_profile: token counts must be integers, not booleans')
    if raw_t.dtype.kind in ('U', 'S'):
        raise ValueError('unit_quantile_profile: token counts must be integers, not strings')
    counts = [_exact_count(v, minimum=1) for v in raw_t.tolist()]
    if any(c is None for c in counts):
        raise ValueError('unit_quantile_profile: every unit needs a finite positive integer token count')
    T = [int(c) for c in counts]
    if len(T) != D:
        raise ValueError(
            f'unit_quantile_profile: {len(T)} token counts for {D} units')
    total_tokens = int(sum(T))
    if total_tokens <= 0:
        raise ValueError('unit_quantile_profile: the total token count must be positive')
    if seeds is None:
        seed_labels = [int(s) for s in range(S)]
    else:
        if isinstance(seeds, (str, bytes, bytearray, dict)):
            raise ValueError('unit_quantile_profile: seeds must be an iterable of integers')
        try:
            raw_seeds = list(seeds)
        except TypeError:
            raise ValueError('unit_quantile_profile: seeds must be an iterable') from None
        if len(raw_seeds) != S:
            raise ValueError(
                f'unit_quantile_profile: {len(raw_seeds)} seed labels for {S} rows')
        seed_labels = [_as_seed_int(s) for s in raw_seeds]
        if any(s is None for s in seed_labels):
            raise ValueError(
                f'unit_quantile_profile: seeds are not exact integers ({raw_seeds!r})')
        if len(set(seed_labels)) != len(seed_labels):
            raise ValueError('unit_quantile_profile: duplicate seed labels')
    if unit_ids is None:
        labels = [f'unit_{i}' for i in range(D)]
    else:
        if isinstance(unit_ids, (str, bytes, bytearray, dict)):
            raise ValueError('unit_quantile_profile: unit_ids must be an iterable')
        try:
            labels = [str(x) for x in unit_ids]
        except TypeError:
            raise ValueError('unit_quantile_profile: unit_ids must be an iterable') from None
        if len(labels) != D:
            raise ValueError(
                f'unit_quantile_profile: {len(labels)} unit ids for {D} units')
        if len(set(labels)) != D:
            raise ValueError('unit_quantile_profile: unit ids must be unique')
    if D < int(UNIT_QUANTILE_POLICY['min_units']):
        return _uq_not_estimable(
            f'unit distribution needs at least {UNIT_QUANTILE_POLICY["min_units"]} '
            f'evaluation units, got {D}',
            n_units=D, n_seeds=S, n_tokens=total_tokens,
            quantiles=qs, conf=conf)
    if S < int(UNIT_QUANTILE_POLICY['min_seeds']):
        return _uq_not_estimable(
            f'unit distribution needs at least {UNIT_QUANTILE_POLICY["min_seeds"]} '
            f'training seed(s), got {S}',
            n_units=D, n_seeds=S, n_tokens=total_tokens,
            quantiles=qs, conf=conf)
    # A token denominator beyond float64 must not hide a representable
    # per-token value (the same failure mode `_ls_div` fixes for the length
    # strata).  Use the vectorised path while every count is representable and
    # all divisions are exact enough; otherwise evaluate each cell through the
    # exact-ratio helper and keep the result only when it is representable.
    try:
        Tf = _uq_np.asarray(T, dtype=float)
    except (OverflowError, ValueError):
        Tf = None
    if Tf is not None and Tf.shape == (D,) and _uq_np.all(_uq_np.isfinite(Tf)):
        with _uq_np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            d = A / Tf[None, :]
        if _uq_np.any((d == 0.0) & (A != 0.0)):
            return _uq_not_estimable(
                'at least one per-token NLL difference underflows the '
                'representable float64 range',
                n_units=D, n_seeds=S, n_tokens=total_tokens,
                quantiles=qs, conf=conf)
    else:
        d = _uq_np.empty_like(A)
        for si in range(S):
            for ui in range(D):
                value = _uq_div(A[si, ui], T[ui])
                if value is None or (value == 0.0 and A[si, ui] != 0.0):
                    return _uq_not_estimable(
                        'at least one per-unit NLL difference is not '
                        'representable as a finite per-token value',
                        n_units=D, n_seeds=S, n_tokens=total_tokens,
                        quantiles=qs, conf=conf)
                d[si, ui] = value
    if not _uq_np.all(_uq_np.isfinite(d)):
        return _uq_not_estimable(
            'at least one per-unit NLL difference is not representable as a '
            'finite per-token value',
            n_units=D, n_seeds=S, n_tokens=total_tokens,
            quantiles=qs, conf=conf)
    pooled_per_seed = []
    for si in range(S):
        value = _uq_pooled_row(A[si, :], total_tokens)
        if value is None:
            return _uq_not_estimable(
                'a token-pooled per-seed log ratio is not representable',
                n_units=D, n_seeds=S, n_tokens=total_tokens,
                quantiles=qs, conf=conf)
        pooled_per_seed.append(float(value))
    if any(not _uq_math.isfinite(v) for v in pooled_per_seed):
        return _uq_not_estimable(
            'a token-pooled per-seed log ratio is not finite',
            n_units=D, n_seeds=S, n_tokens=total_tokens,
            quantiles=qs, conf=conf)
    token_pooled = _uq_mean(pooled_per_seed)
    delta = []
    for ui in range(D):
        m = _uq_mean(d[:, ui])
        if m is None:
            return _uq_not_estimable(
                f'unit {ui} seed-averaged log ratio is not representable',
                n_units=D, n_seeds=S, n_tokens=total_tokens,
                quantiles=qs, conf=conf)
        delta.append(float(m))
    sorted_delta = _uq_sort_values(delta)
    equal_mean = _uq_mean(delta)
    median = _uq_quantile(sorted_delta, 0.5)
    if equal_mean is None or median is None:
        return _uq_not_estimable(
            'the equal-unit mean or median is not representable',
            n_units=D, n_seeds=S, n_tokens=total_tokens,
            quantiles=qs, conf=conf)

    rows = []
    for q in qs:
        value = _uq_quantile(sorted_delta, q)
        if value is None:
            return _uq_not_estimable(
                f'the q={q:g} unit quantile is not representable',
                n_units=D, n_seeds=S, n_tokens=total_tokens,
                quantiles=qs, conf=conf)
        ci = _uq_order_stat_interval(sorted_delta, q, conf)
        rows.append({
            'q': float(q),
            'log_ratio': float(value),
            'ppl_ratio': _uq_exp_ratio(value),
            'relative_pct': _uq_relative_pct(value),
            'ci95': list(ci['ci95']),
            'ci_rank': list(ci['ci_rank']),
            'ci_coverage': ci['ci_coverage'],
            'ci_method': ci['ci_method'],
            'n_units': int(D),
        })

    n_better = int(sum(1 for v in delta if v < 0.0))
    n_worse = int(sum(1 for v in delta if v > 0.0))
    n_tied = int(D - n_better - n_worse)
    frac_better = float(n_better / D)
    frac_ci = _uq_clopper_pearson(n_better, D, conf)
    token_better = int(sum(T[i] for i in range(D) if delta[i] < 0.0))
    token_worse = int(sum(T[i] for i in range(D) if delta[i] > 0.0))
    token_tied = int(total_tokens - token_better - token_worse)
    frac_token_better = _uq_div(token_better, total_tokens)
    frac_token_worse = _uq_div(token_worse, total_tokens)
    frac_token_tied = _uq_div(token_tied, total_tokens)

    gain_terms = [-v for v in delta if v < 0.0]
    harm_terms = [v for v in delta if v > 0.0]
    try:
        gross_gain = float(_uq_math.fsum(gain_terms))
        gross_harm = float(_uq_math.fsum(harm_terms))
        net = gross_gain - gross_harm
    except (OverflowError, ValueError):
        gross_gain = gross_harm = net = None
    if any(v is not None and not _uq_math.isfinite(v)
           for v in (gross_gain, gross_harm, net)):
        gross_gain = gross_harm = net = None
    denom = (None if gross_gain is None or gross_harm is None
             else gross_gain + gross_harm)
    if denom is not None and not _uq_math.isfinite(denom):
        denom = None
    gain_harm_ratio = (gross_gain / gross_harm
                       if (gross_gain is not None and gross_harm is not None
                           and gross_harm > 0.0) else None)
    harm_share = (gross_harm / denom
                  if (gross_harm is not None and denom is not None
                      and denom > 0.0) else None)
    net_share = (net / denom
                 if (net is not None and denom is not None
                     and denom > 0.0) else None)
    for key, value in (('gain_harm_ratio', gain_harm_ratio),
                       ('harm_share', harm_share), ('net_share', net_share)):
        if value is not None and not _uq_math.isfinite(value):
            if key == 'gain_harm_ratio':
                gain_harm_ratio = None
            elif key == 'harm_share':
                harm_share = None
            else:
                net_share = None

    tail_k = max(1, int(round(tf * D)))
    best_vals = sorted_delta[:tail_k]
    worst_vals = sorted_delta[-tail_k:]
    best_tail_mean = _uq_mean(best_vals)
    worst_tail_mean = _uq_mean(worst_vals)
    upper_spread = None
    lower_spread = None
    tail_asymmetry = None
    if worst_tail_mean is not None:
        upper_spread = worst_tail_mean - median
        if not _uq_math.isfinite(upper_spread):
            upper_spread = None
    if best_tail_mean is not None:
        lower_spread = median - best_tail_mean
        if not _uq_math.isfinite(lower_spread):
            lower_spread = None
    if upper_spread is not None and lower_spread is not None:
        tail_asymmetry = upper_spread - lower_spread
        if not _uq_math.isfinite(tail_asymmetry):
            tail_asymmetry = None
    worst_i = max(range(D), key=lambda i: (delta[i], -i))
    best_i = min(range(D), key=lambda i: (delta[i], i))
    worst_tail_harm = None
    best_tail_gain = None
    n_worst_tail_worse = int(sum(1 for v in worst_vals if v > 0.0))
    n_best_tail_better = int(sum(1 for v in best_vals if v < 0.0))
    if gross_harm is not None and gross_harm > 0.0:
        try:
            worst_tail_harm = float(_uq_math.fsum(
                v for v in worst_vals if v > 0.0)) / gross_harm
        except (OverflowError, ValueError):
            worst_tail_harm = None
        if worst_tail_harm is not None and not _uq_math.isfinite(worst_tail_harm):
            worst_tail_harm = None
    if gross_gain is not None and gross_gain > 0.0:
        try:
            best_tail_gain = float(_uq_math.fsum(
                -v for v in best_vals if v < 0.0)) / gross_gain
        except (OverflowError, ValueError):
            best_tail_gain = None
        if best_tail_gain is not None and not _uq_math.isfinite(best_tail_gain):
            best_tail_gain = None

    per_seed = []
    for si in range(S):
        svals = _uq_sort_values(d[si, :])
        qvals = []
        for q in qs:
            v = _uq_quantile(svals, q)
            qvals.append(None if v is None else float(v))
        per_seed.append({
            'seed': int(seed_labels[si]),
            'log_ratio_quantiles': qvals,
        })
    seed_range = []
    for qi, q in enumerate(qs):
        vals = [row['log_ratio_quantiles'][qi] for row in per_seed]
        finite = [v for v in vals if v is not None]
        lo = min(finite) if finite else None
        hi = max(finite) if finite else None
        span = (hi - lo if lo is not None and hi is not None else None)
        if span is not None and not _uq_math.isfinite(span):
            span = None
        seed_range.append({
            'q': float(q),
            'min': (float(lo) if lo is not None else None),
            'max': (float(hi) if hi is not None else None),
            'span': (float(span) if span is not None else None),
            'n_seeds_negative': int(sum(1 for v in finite if v < 0.0)),
            'n_seeds_positive': int(sum(1 for v in finite if v > 0.0)),
            'n_seeds_zero': int(sum(1 for v in finite if v == 0.0)),
        })

    all_same = 0
    all_better = 0
    all_worse = 0
    mixed = 0
    for ui in range(D):
        signs = [_uq_sign(d[si, ui]) for si in range(S)]
        if all(s == signs[0] for s in signs):
            all_same += 1
            if signs[0] < 0:
                all_better += 1
            elif signs[0] > 0:
                all_worse += 1
        else:
            mixed += 1

    q10 = _uq_quantile(sorted_delta, 0.10)
    q90 = _uq_quantile(sorted_delta, 0.90)
    med_sign = _uq_sign(median)
    eq_sign = _uq_sign(equal_mean)
    tok_sign = _uq_sign(token_pooled) if token_pooled is not None else None
    nonzero_signs = [s for s in (med_sign, eq_sign, tok_sign) if s not in (None, 0)]
    direction_agree = (None if token_pooled is None
                       else bool(len(set(nonzero_signs)) <= 1))
    gap = None
    if token_pooled is not None:
        gap = token_pooled - equal_mean
        if not _uq_math.isfinite(gap):
            gap = None

    block = {
        'kind': 'unit_quantile_profile',
        'version': UNIT_QUANTILE_POLICY['version'],
        'status': 'ok',
        'estimand': ('quantiles across evaluation units of the seed-averaged '
                     'per-unit log PPL ratio log(PPL_a/PPL_b) = (nll_a-nll_b)/tokens; '
                     'negative = arm a better'),
        'n_units': int(D),
        'n_seeds': int(S),
        'n_tokens': int(total_tokens),
        'seeds': [int(s) for s in seed_labels],
        'unit_ids': [str(x) for x in labels],
        'quantiles': rows,
        'mean': {
            'equal_unit_log_ratio': float(equal_mean),
            'equal_unit_ppl_ratio': _uq_exp_ratio(equal_mean),
            'equal_unit_relative_pct': _uq_relative_pct(equal_mean),
            'token_pooled_log_ratio': (None if token_pooled is None else float(token_pooled)),
            'token_pooled_ppl_ratio': (_uq_exp_ratio(token_pooled)
                                       if token_pooled is not None else None),
            'token_pooled_relative_pct': (_uq_relative_pct(token_pooled)
                                          if token_pooled is not None else None),
            'token_weighted_gap': (None if gap is None else float(gap)),
            'weighting_flips_sign': bool(
                token_pooled is not None
                and _uq_sign(token_pooled) not in (None, 0)
                and _uq_sign(equal_mean) not in (None, 0)
                and _uq_sign(token_pooled) != _uq_sign(equal_mean)),
            'weighting_note': ('equal-unit and token-pooled means answer '
                               'different questions; the frozen endpoint is '
                               'the token-pooled mean'),
        },
        'median': {
            'log_ratio': float(median),
            'ppl_ratio': _uq_exp_ratio(median),
            'relative_pct': _uq_relative_pct(median),
            'sign': med_sign,
        },
        'improvement': {
            'n_better': n_better,
            'n_worse': n_worse,
            'n_tied': n_tied,
            'fraction_better': frac_better,
            'fraction_worse': float(n_worse / D),
            'fraction_tied': float(n_tied / D),
            'fraction_better_ci95': frac_ci,
            'fraction_better_ci_method': ('exact Clopper-Pearson; a tie counts '
                                          'as not better'),
            'fraction_better_token_weighted': (None if frac_token_better is None
                                               else float(frac_token_better)),
            'fraction_worse_token_weighted': (None if frac_token_worse is None
                                              else float(frac_token_worse)),
            'fraction_tied_token_weighted': (None if frac_token_tied is None
                                             else float(frac_token_tied)),
            'n_tokens_better': int(token_better),
            'n_tokens_worse': int(token_worse),
            'n_tokens_tied': int(token_tied),
        },
        'gross': {
            'gross_gain_log': gross_gain,
            'gross_harm_log': gross_harm,
            'net_log': net,
            'gain_harm_ratio': gain_harm_ratio,
            'harm_share': harm_share,
            'net_share': net_share,
            'note': ('equal-unit sums of the seed-averaged log ratios; '
                     'gross_gain = sum of |negative| units, gross_harm = sum '
                     'of positive units, net = gross_gain - gross_harm.  They '
                     'are not token-weighted'),
        },
        'tails': {
            'tail_fraction': float(tf),
            'tail_k': int(tail_k),
            'best_tail_mean_log': best_tail_mean,
            'worst_tail_mean_log': worst_tail_mean,
            'upper_spread_from_median': upper_spread,
            'lower_spread_from_median': lower_spread,
            'tail_asymmetry': tail_asymmetry,
            'tail_asymmetry_note': ('upper_spread - lower_spread; positive '
                                    'means the harmed tail is further from the '
                                    'median than the improved tail'),
            'n_worst_tail_worse': n_worst_tail_worse,
            'n_best_tail_better': n_best_tail_better,
            'worst_tail_harm_share': worst_tail_harm,
            'best_tail_gain_share': best_tail_gain,
            'worst_unit': {'id': str(labels[worst_i]),
                           'log_ratio': float(delta[worst_i])},
            'best_unit': {'id': str(labels[best_i]),
                          'log_ratio': float(delta[best_i])},
        },
        'sign': {
            'median_sign': med_sign,
            'equal_mean_sign': eq_sign,
            'token_pooled_sign': tok_sign,
            'direction_agree': direction_agree,
            'q10_log_ratio': q10,
            'q90_log_ratio': q90,
            'q10_positive': (None if q10 is None else bool(q10 > 0.0)),
            'q90_negative': (None if q90 is None else bool(q90 < 0.0)),
            'worst_unit_negative': bool(delta[worst_i] < 0.0),
            'best_unit_positive': bool(delta[best_i] > 0.0),
        },
        'per_seed_quantiles': per_seed,
        'seed_range': seed_range,
        'seed_consistency': {
            'n_units_all_seed_same_sign': int(all_same),
            'n_units_all_seed_better': int(all_better),
            'n_units_all_seed_worse': int(all_worse),
            'n_units_seed_sign_mixed': int(mixed),
            'fraction_all_seed_same_sign': float(all_same / D),
            'note': ('descriptive only: with a handful of seeds, chance sign '
                     'agreement is high even under a null; this is not a test'),
        },
        'sufficient_statistics': {
            'per_seed_nll_diff': [[float(v) for v in row] for row in A],
            'n_tokens': [int(t) for t in T],
            'unit_ids': [str(x) for x in labels],
            'seeds': [int(s) for s in seed_labels],
            'quantiles': [float(q) for q in qs],
        },
        'policy': {
            'version': UNIT_QUANTILE_POLICY['version'],
            'quantiles': [float(q) for q in qs],
            'conf': float(conf),
            'min_units': int(UNIT_QUANTILE_POLICY['min_units']),
            'min_seeds': int(UNIT_QUANTILE_POLICY['min_seeds']),
            'tail_fraction': float(tf),
            'quantile_method': UNIT_QUANTILE_POLICY['quantile_method'],
            'interval_method': UNIT_QUANTILE_POLICY['interval_method'],
            'improvement_interval_method': UNIT_QUANTILE_POLICY['improvement_interval_method'],
            'role': UNIT_QUANTILE_POLICY['role'],
            'boundary': UNIT_QUANTILE_POLICY['boundary'],
        },
        'note': ('seed-averaged unit effects; the token-pooled mean is shown '
                 'next to the equal-unit distribution so the weighting choice '
                 'is visible.  Unit quantiles are descriptive and are not a '
                 'second confirmatory test.'),
        'boundary': UNIT_QUANTILE_POLICY['boundary'],
    }
    return block


def paired_nll_quantile_profile(nll_a, nll_b, n_tokens, *, unit_ids=None,
                                seeds=None, quantiles=None, conf=None,
                                tail_fraction=None):
    """Unit-effect distribution from two paired ``[n_seeds, n_units]`` NLLs.

    A thin wrapper around :func:`unit_quantile_profile`: it validates the two
    arm matrices, forms the paired summed-NLL difference and delegates.  The
    matrix contract is the same one used by the cluster interval and the
    leave-one-unit-out readout.
    """
    A = _as_nll_matrix(nll_a, 'paired_nll_quantile_profile(nll_a)',
                       axis_label='n_units', value_label='per-unit')
    B = _as_nll_matrix(nll_b, 'paired_nll_quantile_profile(nll_b)',
                       axis_label='n_units', value_label='per-unit')
    if A.shape != B.shape:
        raise ValueError(
            f'paired_nll_quantile_profile: arm shapes differ ({A.shape} vs '
            f'{B.shape}); a paired readout needs the same seeds and units')
    with _uq_np.errstate(over='ignore', invalid='ignore'):
        diff = A - B
    if not _uq_np.all(_uq_np.isfinite(diff)):
        return _uq_not_estimable(
            'the paired NLL difference overflows the representable float64 range',
            n_units=A.shape[1], n_seeds=A.shape[0],
            quantiles=_uq_quantiles(quantiles),
            conf=_uq_conf(UNIT_QUANTILE_POLICY['conf'] if conf is None else conf))
    return unit_quantile_profile(
        diff, n_tokens, unit_ids=unit_ids, seeds=seeds, quantiles=quantiles,
        conf=conf, tail_fraction=tail_fraction)
