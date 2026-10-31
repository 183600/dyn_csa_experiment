"""Post-stratification of a paired PPL contrast by evaluation-unit length.

A token-pooled PPL difference is a **token-weighted** estimand: a handful of
long documents can dominate it, while the same relative effect on short
documents is invisible.  That is not a bug in the estimator, but it is a
result-analysis choice that must be shown: a corpus (or a future evaluation
slice) with a different length composition can give a different answer, and a
mean difference can be broad-based on the log scale while being concentrated
in the longest units on the raw-PPL scale.

This fragment adds one descriptive readout:

* split the frozen evaluation units into length strata (equal-count quantile
  bins by token count, deterministic and independent of the arm outcomes);
* report the paired token-pooled ``log(PPL_a / PPL_b)`` and raw ``ΔPPL`` in
  every stratum, with a seed-level t interval;
* **standardise** the per-seed log effect to a declared reference length
  distribution (equal strata by default), which answers "what would the
  relative effect be if each length stratum counted equally?";
* test for aggregation/composition sensitivity (does standardisation reverse
  the sign?), sign consistency across strata, contribution concentration and
  a Cochran-Q/I² heterogeneity readout;
* report the largest pairwise stratum contrast with its paired-seed interval.

Nothing here is a new confirmatory test.  The strata are defined by the
observed token counts, the intervals condition on the trained seeds, and the
readout never changes ``p_exact_signflip``, Holm, ``verified`` or a frozen
``confirmed`` verdict.  It is the stratification companion of the
leave-one-unit-out influence readout: influence asks "does one unit own the
effect?", stratification asks "does the effect depend on which kind of unit we
weight?".

The block stores its sufficient statistics (per-seed per-stratum summed NLLs
and the shared unit token counts) so ``audit_gates.py length_strata`` can
replay every derived number without re-reading the test split.
"""

import math as _ls_math

import numpy as _ls_np


LENGTH_STRATA_POLICY = {
    'version': 'length-strata-1',
    'n_strata': 4,
    'min_units_per_stratum': 2,
    'conf': 0.95,
    'reference': 'equal_strata',
    'q_method': ('Cochran Q on seed-level stratum log ratios with the '
                 'chi-square reference; approximate because the same seeds '
                 'serve every stratum (the strata are correlated, not '
                 'independent studies)'),
    'role': ('descriptive post-stratification of a paired PPL contrast; it '
             'adds no p-value to the frozen family and never changes a '
             'frozen verdict'),
    'boundary': (
        'post-hoc descriptive length post-stratification. The strata are '
        'defined from the observed evaluation-unit token counts, not from a '
        'pre-registered population distribution; the interval is a seed-level '
        't interval conditioned on the trained seeds and does not resample '
        'documents or tokens. Equal-strata standardisation is one declared '
        'reference, not the "correct" corpus. Cochran Q assumes independent '
        'strata and is therefore approximate (the same seeds appear in every '
        'stratum); it is a heterogeneity readout, not a confirmatory test. '
        'No p-value is added to the Holm family, and no confirmed/verified '
        'field is produced or changed. The standardised ΔPPL is a weighted '
        'average of stratum ΔPPLs (each stratum still token-pooled), not a '
        'single PPL recomputed after re-weighting tokens; because exp is '
        'nonlinear these differ, so the log ratio is the composition-safe '
        'readout.'),
}

LENGTH_STRATA_STATUSES = ('ok', 'not_estimable')


def _ls_int(value, name, *, minimum=None, maximum=None):
    """Exact integer parameter (booleans and fractional values are refused)."""
    if isinstance(value, (bool, _ls_np.bool_)):
        raise ValueError(f'{name} must be an integer, not a boolean')
    if isinstance(value, (_ls_np.integer,)):
        out = int(value)
    elif isinstance(value, int):
        out = int(value)
    else:
        # 4.0 is not "the integer 4" here: a float parameter is a sign that
        # the caller is forwarding a computed quantity that may be fractional.
        raise ValueError(f'{name} must be an integer, got {value!r}')
    if minimum is not None and out < minimum:
        raise ValueError(f'{name} must be >= {minimum}, got {out}')
    if maximum is not None and out > maximum:
        raise ValueError(f'{name} must be <= {maximum}, got {out}')
    return out


def _ls_conf(value):
    if isinstance(value, (bool, _ls_np.bool_)):
        raise ValueError(f'conf must be a probability in (0, 1), got {value!r}')
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'conf must be a probability in (0, 1), got {value!r}') from None
    if not (_ls_math.isfinite(out) and 0.0 < out < 1.0):
        raise ValueError(f'conf must be a probability in (0, 1), got {value!r}')
    return out


def _ls_sign(value):
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not _ls_math.isfinite(out):
        return None
    if out > 0.0:
        return 1
    if out < 0.0:
        return -1
    return 0


def _ls_mean_std(values):
    """Mean and sample SD, with the repository's overflow-safe path."""
    vals = [float(v) for v in values]
    if not vals:
        return None, None
    scaled = _scaled_mean_std_if_extreme(vals, ddof=1)
    if scaled is not None:
        mean, std = scaled
        return (float(mean) if _ls_math.isfinite(mean) else None,
                float(std) if _ls_math.isfinite(std) else None)
    mean = float(_ls_np.mean(vals))
    std = float(_ls_np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return (mean if _ls_math.isfinite(mean) else None,
            std if _ls_math.isfinite(std) else None)


def _ls_div(num, den):
    """``num / den`` with an exact positive-integer denominator.

    Python converts the integer divisor to float before dividing, so a token
    denominator beyond ``float64``'s exponent range raises ``OverflowError``
    even when the quotient is representable.  The fallback does the division
    in the exact rational domain and rounds once.  Ordinary panels keep the
    historical float division bit-for-bit.
    """
    try:
        d = int(den)
    except (OverflowError, ValueError, ZeroDivisionError, TypeError):
        return None
    if abs(d) <= (1 << 53):
        try:
            return float(num) / d
        except OverflowError:
            pass
    try:
        from fractions import Fraction as _Fraction
        try:
            quotient = _Fraction(num) / d
        except TypeError:
            quotient = _Fraction(float(num)) / d
        return float(quotient)
    except (OverflowError, ValueError, ZeroDivisionError, TypeError):
        return None


def _ls_summary(values, conf):
    """Mean and two-sided t interval for a seed-level vector."""
    finite = [float(v) for v in values if _finite_scalar(v) is not None]
    n = len(finite)
    out = {'n': int(n), 'mean': None, 'std': None, 'sem': None,
           'ci95': None, 'conf': float(conf)}
    if not finite:
        return out
    mean, std = _ls_mean_std(finite)
    out['mean'] = mean
    out['std'] = std
    if mean is None:
        return out
    if n > 1 and std is not None:
        sem = std / _ls_math.sqrt(n)
        if _ls_math.isfinite(sem):
            out['sem'] = float(sem)
            tc = _t_critical(conf, n - 1)
            if _ls_math.isfinite(tc):
                lo = mean - tc * sem
                hi = mean + tc * sem
                if _ls_math.isfinite(lo) and _ls_math.isfinite(hi):
                    out['ci95'] = [float(lo), float(hi)]
    return out


def _ls_groups(tokens, n_strata, min_units, *, preserve_order=False):
    """Deterministic equal-count length strata; returns index arrays or None.

    Equal-count bins make the comparison of short and long units use a
    similar number of units in each bin.  Boundaries come only from the token
    counts (never from the arm outcomes), so seeing the data cannot choose
    which strata to report.

    ``preserve_order`` is an internal replay hook used by the audit gate after
    it has aggregated the stored strata into one unit per stratum.  In that
    representation ``n_strata == len(tokens)`` and the caller has already
    fixed the stratum order: re-sorting by the *summed* token count can move a
    two-unit stratum with 6 tokens before a one-unit stratum with 4 tokens,
    even though every individual token count in the first stratum is smaller.
    The ordinary path (the default) still sorts by unit length.
    """
    d = len(tokens)
    if preserve_order and int(n_strata) == d and int(min_units) <= 1 and d >= 2:
        return [[i] for i in range(d)]
    # Sort by the exact integer token count (not float64, which would round
    # counts above 2**53 and could merge distinct lengths into a fake tie).
    order = sorted(range(d), key=lambda i: (int(tokens[i]), i))
    for g in range(int(n_strata), 1, -1):
        if d < g * int(min_units):
            continue
        cuts = [(i * d) // g for i in range(1, g)]
        if len(set(cuts)) != len(cuts):
            continue
        groups = [order[start:stop] for start, stop in zip([0] + cuts, cuts + [d])]
        if all(len(part) >= int(min_units) for part in groups):
            return [[int(i) for i in part] for part in groups]
    return None


def _ls_reference_weights(reference, strata, total_tokens):
    """Normalised reference weights for standardisation."""
    g = len(strata)
    if reference is None:
        return 'equal_strata', [1.0 / g] * g
    if isinstance(reference, str):
        if reference == 'equal_strata':
            return 'equal_strata', [1.0 / g] * g
        if reference == 'token':
            # Divide the exact Python integer counts instead of converting
            # each count through float64 first.  Besides avoiding a rounding
            # step, this keeps a token total beyond the float64 exponent range
            # from raising OverflowError even though every normalised weight
            # is representable.  A genuinely unrepresentable share must not be
            # rounded to 0.0: that would silently drop the stratum from the
            # standardised estimate, so report the reference as unavailable.
            weights = []
            for s in strata:
                w = _ls_div(s['n_tokens'], total_tokens)
                if (w is None or not _ls_math.isfinite(w) or w <= 0.0):
                    return 'token', None
                weights.append(float(w))
            return 'token', weights
        raise ValueError(
            "reference must be 'equal_strata', 'token', or a numeric sequence "
            f"of {g} weights, got {reference!r}")
    if isinstance(reference, (bytes, bytearray)):
        raise ValueError(
            "reference must be 'equal_strata', 'token', or a numeric sequence "
            f"of {g} weights, got {reference!r}")
    if isinstance(reference, dict):
        raise ValueError('reference must not be a mapping; pass a numeric sequence')
    try:
        raw = list(reference)
    except TypeError:
        raise ValueError('reference must be a numeric sequence') from None
    if len(raw) != g:
        raise ValueError(f'reference has {len(raw)} weights for {g} strata')
    weights = []
    _need_exact = False
    for w in raw:
        if isinstance(w, (bool, _ls_np.bool_)):
            raise ValueError('reference weights must be finite positive numbers, not booleans')
        try:
            f = float(w)
        except (TypeError, ValueError, OverflowError):
            _need_exact = True
            continue
        if not (_ls_math.isfinite(f) and f > 0.0):
            # The declared value may still be a finite positive integer or
            # Decimal beyond float64 (or a tiny positive value that rounded to
            # zero); defer the verdict to the exact path below.
            _need_exact = True
            continue
        weights.append(f)
    if not _need_exact:
        try:
            total = _ls_math.fsum(weights)
        except (OverflowError, ValueError):
            total = None
        if total is None or not (_ls_math.isfinite(total) and total > 0.0):
            _need_exact = True
        else:
            return 'custom', [w / total for w in weights]
    # Exact normalisation for huge/tiny rational weights and for a float sum
    # that overflows even though each normalised weight is representable.
    from fractions import Fraction as _Fraction
    exact = []
    for w in raw:
        if isinstance(w, (bool, _ls_np.bool_)):
            raise ValueError('reference weights must be finite positive numbers, not booleans')
        try:
            fr = _Fraction(w)
        except (TypeError, ValueError, OverflowError):
            raise ValueError('reference weights must be finite positive numbers') from None
        if fr <= 0:
            raise ValueError('reference weights must be finite and positive')
        exact.append(fr)
    total = sum(exact, _Fraction(0, 1))
    if total <= 0:
        raise ValueError('reference weights must have a finite positive sum')
    try:
        normalised = [float(fr / total) for fr in exact]
    except (OverflowError, ValueError):
        raise ValueError('reference weights are not representable') from None
    if not all(_ls_math.isfinite(x) and x > 0.0 for x in normalised):
        # A positive but extremely small share may round to 0.0; the weights
        # would then no longer form a positive reference distribution.
        raise ValueError('reference weights are not representable as positive floats')
    return 'custom', normalised


def _ls_not_estimable(reason, *, n_units=None, n_seeds=None, n_strata=None):
    return {'kind': 'length_stratification', 'status': 'not_estimable',
            'policy': dict(LENGTH_STRATA_POLICY), 'reason': str(reason),
            'n_units': n_units, 'n_seeds': n_seeds, 'n_strata': n_strata,
            'boundary': LENGTH_STRATA_POLICY['boundary']}


def length_stratified_effect(nll_a, nll_b, n_tokens, *, n_strata=None,
                             min_units_per_stratum=None, conf=None,
                             reference=None, seeds=None,
                             preserve_stratum_order=False):
    """Post-stratify a paired PPL contrast by evaluation-unit length.

    ``nll_a`` / ``nll_b`` are ``[n_seeds, n_units]`` summed per-unit NLLs for
    the two arms, and ``n_tokens`` is the shared ``[n_units]`` token count
    (the same units and positions must underpin both arms).  The point
    estimate uses the same token-pooled construction as the frozen
    comparison; the strata only re-weight it.

    Returns a strict-JSON block with ``status='ok'`` and the sufficient
    statistics necessary to replay every derived field, or
    ``status='not_estimable'`` with a reason when the unit panel is too small
    or a value is not representable.  Malformed inputs raise ``ValueError``;
    an unavailable but valid readout is reported, never faked.
    """
    conf = _ls_conf(LENGTH_STRATA_POLICY['conf'] if conf is None else conf)
    n_strata = _ls_int(LENGTH_STRATA_POLICY['n_strata'] if n_strata is None else n_strata,
                       'n_strata', minimum=2, maximum=8)
    min_units = _ls_int(
        LENGTH_STRATA_POLICY['min_units_per_stratum']
        if min_units_per_stratum is None else min_units_per_stratum,
        'min_units_per_stratum', minimum=1, maximum=64)

    A = _as_nll_matrix(nll_a, 'length_stratified_effect(nll_a)',
                       axis_label='n_units', value_label='per-unit')
    B = _as_nll_matrix(nll_b, 'length_stratified_effect(nll_b)',
                       axis_label='n_units', value_label='per-unit')
    if A.shape != B.shape:
        raise ValueError(
            f'length_stratified_effect: arm shapes differ ({A.shape} vs {B.shape}); '
            'a paired readout needs the same seeds and units')
    try:
        raw_t = _ls_np.asarray(n_tokens).reshape(-1)
    except (TypeError, ValueError, OverflowError):
        raise ValueError('length_stratified_effect: n_tokens must be numeric') from None
    if _contains_bool_value(n_tokens) or raw_t.dtype.kind == 'b':
        raise ValueError('length_stratified_effect: token counts must be integers, not booleans')
    counts = [_exact_count(v, minimum=1) for v in raw_t.tolist()]
    if any(c is None for c in counts):
        raise ValueError('length_stratified_effect: every unit needs a finite positive integer token count')
    T = [int(c) for c in counts]
    if len(T) != A.shape[1]:
        raise ValueError(
            f'length_stratified_effect: {len(T)} token counts for {A.shape[1]} units')
    s_count, d_count = A.shape
    if seeds is None:
        seed_labels = [int(s) for s in range(s_count)]
    else:
        if isinstance(seeds, (str, bytes, bytearray, dict)):
            raise ValueError('length_stratified_effect: seeds must be an iterable of integers')
        try:
            raw_seeds = list(seeds)
        except TypeError:
            raise ValueError('length_stratified_effect: seeds must be an iterable') from None
        if len(raw_seeds) != s_count:
            raise ValueError(
                f'length_stratified_effect: {len(raw_seeds)} seed labels for '
                f'{s_count} rows')
        seed_labels = [_as_seed_int(s) for s in raw_seeds]
        if any(s is None for s in seed_labels):
            raise ValueError(
                f'length_stratified_effect: seeds are not exact integers ({raw_seeds!r})')
        if len(set(seed_labels)) != len(seed_labels):
            raise ValueError('length_stratified_effect: duplicate seed labels')
    if d_count < 2:
        return _ls_not_estimable(
            'length post-stratification needs at least two evaluation units',
            n_units=d_count, n_seeds=s_count, n_strata=0)

    groups = _ls_groups(T, n_strata, min_units,
                        preserve_order=bool(preserve_stratum_order))
    if groups is None:
        return _ls_not_estimable(
            f'cannot form two length strata with at least {min_units} units '
            f'each from {d_count} units and {len(set(T))} distinct token counts',
            n_units=d_count, n_seeds=s_count, n_strata=0)

    total_tokens = int(sum(T))
    if total_tokens <= 0:
        raise ValueError('length_stratified_effect: the total token count must be positive')

    # Per-stratum sufficient statistics.  Products are accumulated with fsum
    # so a finite panel near the float limit does not silently overflow in a
    # matrix multiply.
    strata = []
    lr_vals = []          # [n_strata][n_seeds]
    dp_vals = []          # [n_strata][n_seeds]
    for gi, idx in enumerate(groups):
        tg = [T[i] for i in idx]
        n_tok_g = int(sum(tg))
        if n_tok_g <= 0:
            return _ls_not_estimable(
                f'length stratum {gi} has no scored tokens',
                n_units=d_count, n_seeds=s_count, n_strata=len(groups))
        a_sum = []
        b_sum = []
        for si in range(s_count):
            try:
                a_s = _ls_math.fsum(float(A[si, i]) for i in idx)
                b_s = _ls_math.fsum(float(B[si, i]) for i in idx)
            except (OverflowError, ValueError):
                return _ls_not_estimable(
                    f'length stratum {gi} summed NLL is not representable',
                    n_units=d_count, n_seeds=s_count, n_strata=len(groups))
            a_sum.append(a_s)
            b_sum.append(b_s)
        lr = []
        dp = []
        for a_s, b_s in zip(a_sum, b_sum):
            nll_a_g = _ls_div(a_s, n_tok_g)
            nll_b_g = _ls_div(b_s, n_tok_g)
            if (nll_a_g is None or nll_b_g is None
                    or (nll_a_g == 0.0 and a_s != 0.0)
                    or (nll_b_g == 0.0 and b_s != 0.0)):
                return _ls_not_estimable(
                    f'length stratum {gi} pooled NLL is not representable',
                    n_units=d_count, n_seeds=s_count, n_strata=len(groups))
            one_lr = nll_a_g - nll_b_g
            if not _ls_math.isfinite(one_lr):
                return _ls_not_estimable(
                    f'length stratum {gi} log ratio is not representable',
                    n_units=d_count, n_seeds=s_count, n_strata=len(groups))
            lr.append(float(one_lr))
            # exp(a)-exp(b) loses a tiny log difference to rounding; use the
            # same expm1 fallback as the pooled document CI so a representable
            # small ΔPPL is not reported as a fake zero.
            dp.append(_ppl_difference(nll_a_g, nll_b_g))
        lr_vals.append(lr)
        dp_vals.append(dp)
        lo_tok = int(min(tg))
        hi_tok = int(max(tg))
        lr_summary = _ls_summary(lr, conf)
        ratio = _safe_exp_ppl(lr_summary['mean']) if lr_summary['mean'] is not None else None
        if ratio is not None and not _ls_math.isfinite(ratio):
            ratio = None
        dp_finite = [v for v in dp if v is not None]
        dp_summary = _ls_summary(dp_finite, conf) if len(dp_finite) == s_count else (
            {'n': len(dp_finite), 'mean': None, 'std': None, 'sem': None,
             'ci95': None, 'conf': float(conf)})
        strata.append({
            'index': int(gi),
            'unit_indices': [int(i) for i in idx],
            'n_units': int(len(idx)),
            'n_tokens': n_tok_g,
            'token_share': float(n_tok_g / total_tokens),
            'lo_tokens': lo_tok,
            'hi_tokens': hi_tok,
            'log_ratio': lr_summary['mean'],
            'ratio': ratio,
            'log_ratio_ci95': lr_summary['ci95'],
            'delta_ppl': dp_summary['mean'],
            'delta_ppl_ci95': dp_summary['ci95'],
            'n_seeds_with_delta_ppl': int(len(dp_finite)),
            # Sufficient statistics for the audit replay.
            'nll_a_sum': [float(x) for x in a_sum],
            'nll_b_sum': [float(x) for x in b_sum],
            'per_seed_log_ratio': [float(x) for x in lr],
            'per_seed_delta_ppl': [None if x is None else float(x) for x in dp],
        })

    per_seed = []
    overall_lr = []
    overall_dp = []
    for si in range(s_count):
        # A finite panel can still have an unrepresentable total: `fsum`
        # raises once the exact sum exceeds float64, and that is not a raw
        # exception the caller should see.  Report the readout as unavailable
        # and keep the report strict-JSON safe.
        try:
            a_tot = _ls_math.fsum(float(A[si, i]) for i in range(d_count))
            b_tot = _ls_math.fsum(float(B[si, i]) for i in range(d_count))
        except (OverflowError, ValueError):
            return _ls_not_estimable(
                'overall summed NLL is not representable',
                n_units=d_count, n_seeds=s_count, n_strata=len(groups))
        if not (_ls_math.isfinite(a_tot) and _ls_math.isfinite(b_tot)):
            return _ls_not_estimable(
                'overall summed NLL is not representable',
                n_units=d_count, n_seeds=s_count, n_strata=len(groups))
        _a_pooled = _ls_div(a_tot, total_tokens)
        _b_pooled = _ls_div(b_tot, total_tokens)
        if (_a_pooled is None or _b_pooled is None
                or (_a_pooled == 0.0 and a_tot != 0.0)
                or (_b_pooled == 0.0 and b_tot != 0.0)):
            return _ls_not_estimable(
                'overall pooled NLL is not representable',
                n_units=d_count, n_seeds=s_count, n_strata=len(groups))
        one_lr = _a_pooled - _b_pooled
        if not _ls_math.isfinite(one_lr):
            return _ls_not_estimable(
                'overall log ratio is not representable',
                n_units=d_count, n_seeds=s_count, n_strata=len(groups))
        one_dp = _ppl_difference(_a_pooled, _b_pooled)
        overall_lr.append(float(one_lr))
        overall_dp.append(one_dp)
        per_seed.append({
            'seed': int(seed_labels[si]),
            'nll_a_total': float(a_tot),
            'nll_b_total': float(b_tot),
            'log_ratio': float(one_lr),
            'delta_ppl': one_dp,
        })
    overall_lr_summary = _ls_summary(overall_lr, conf)
    overall_dp_finite = [v for v in overall_dp if v is not None]
    overall_dp_summary = _ls_summary(overall_dp_finite, conf) if len(overall_dp_finite) == s_count else (
        {'n': len(overall_dp_finite), 'mean': None, 'std': None, 'sem': None,
         'ci95': None, 'conf': float(conf)})
    overall_ratio = (_safe_exp_ppl(overall_lr_summary['mean'])
                     if overall_lr_summary['mean'] is not None else None)
    if overall_ratio is not None and not _ls_math.isfinite(overall_ratio):
        overall_ratio = None

    ref_name, ref_weights = _ls_reference_weights(reference, strata, total_tokens)
    if ref_weights is None:
        return _ls_not_estimable(
            'the token reference weights are not representable as positive '
            'float64 values',
            n_units=d_count, n_seeds=s_count, n_strata=len(groups))
    std_lr = []
    std_dp = []
    for si in range(s_count):
        try:
            std_lr.append(float(_ls_math.fsum(
                ref_weights[g] * lr_vals[g][si] for g in range(len(groups)))))
        except (OverflowError, ValueError):
            return _ls_not_estimable(
                'a standardised per-seed log ratio is not representable',
                n_units=d_count, n_seeds=s_count, n_strata=len(groups))
        vals = [dp_vals[g][si] for g in range(len(groups))]
        if any(v is None for v in vals):
            std_dp.append(None)
        else:
            try:
                std_dp.append(float(_ls_math.fsum(
                    ref_weights[g] * vals[g] for g in range(len(groups)))))
            except (OverflowError, ValueError):
                return _ls_not_estimable(
                    'a standardised per-seed PPL difference is not representable',
                    n_units=d_count, n_seeds=s_count, n_strata=len(groups))
    std_lr_summary = _ls_summary(std_lr, conf)
    std_dp_finite = [v for v in std_dp if v is not None]
    std_dp_summary = _ls_summary(std_dp_finite, conf) if len(std_dp_finite) == s_count else (
        {'n': len(std_dp_finite), 'mean': None, 'std': None, 'sem': None,
         'ci95': None, 'conf': float(conf)})
    std_ratio = (_safe_exp_ppl(std_lr_summary['mean'])
                 if std_lr_summary['mean'] is not None else None)
    if std_ratio is not None and not _ls_math.isfinite(std_ratio):
        std_ratio = None

    # Heterogeneity: Cochran Q on the seed-level stratum log ratios.  The
    # inverse-variance pooled mean is the common-effect anchor; a zero SE with
    # a non-zero deviation makes Q unbounded (reported as unavailable rather
    # than infinite).
    q = None
    q_p = None
    i2 = None
    tau2 = None
    q_reason = None
    means = [s['log_ratio'] for s in strata]
    ses = []
    for s in strata:
        vals = [float(v) for v in s['per_seed_log_ratio']]
        if len(vals) < 2:
            ses.append(None)
            continue
        _mean, _std = _ls_mean_std(vals)
        if _std is None:
            ses.append(None)
        else:
            ses.append(float(_std / _ls_math.sqrt(len(vals))))
    if all(m is not None for m in means) and all(
            se is not None and _ls_math.isfinite(se) for se in ses):
        # Inverse-variance weights are scale-invariant for Q.  Dividing every
        # SE by their maximum keeps the ordinary path bit-identical (scale 1)
        # but avoids `1 / (se * se)` overflowing or, for a subnormal SE,
        # underflowing to zero and raising ZeroDivisionError.  The returned
        # tau^2 is converted back to the original SE units.
        _naive_se_ok = all(
            se > 0.0 and math.isfinite(se * se) and se * se > 0.0
            for se in ses)
        _se_scale = (1.0 if _naive_se_ok
                     else max(float(se) for se in ses))
        _scaled_ses = [float(se) / _se_scale for se in ses] if _se_scale > 0.0 else []
        w = []
        for se in _scaled_ses:
            if se > 0.0:
                var = se * se
                w.append(1.0 / var if var > 0.0 and _ls_math.isfinite(var) else None)
            else:
                w.append(None)
        _wsum = None
        if all(x is not None and _ls_math.isfinite(x) for x in w):
            try:
                _wsum = _ls_math.fsum(w)
            except (OverflowError, ValueError):
                _wsum = None
        if _wsum is not None and _wsum > 0.0:
            mu = None
            try:
                mu = _ls_math.fsum(
                    w[i] * means[i] for i in range(len(groups))) / _wsum
            except (OverflowError, ValueError):
                mu = None
            good = mu is not None and _ls_math.isfinite(mu)
            terms = []
            if good:
                for i in range(len(groups)):
                    dev = means[i] - mu
                    if not _ls_math.isfinite(dev):
                        good = False
                        break
                    if _se_scale == 1.0:
                        # Keep the historical expression for ordinary SEs.
                        term = (dev * dev) / (_scaled_ses[i] * _scaled_ses[i])
                    else:
                        # ``dev`` is in original units; put it on the same
                        # scale as the normalised SE before squaring.
                        ratio = (dev / _se_scale) / _scaled_ses[i]
                        term = ratio * ratio
                    if not _ls_math.isfinite(term):
                        good = False
                        break
                    terms.append(term)
            if good:
                try:
                    q = float(_ls_math.fsum(terms))
                except (OverflowError, ValueError):
                    q = None
                if q is not None and _ls_math.isfinite(q):
                    q_p = float(chi2_sf(q, len(groups) - 1))
                    if q > 0.0:
                        i2 = float(max(0.0, (q - (len(groups) - 1)) / q))
                        denom = None
                        try:
                            w2 = _ls_math.fsum(x * x for x in w)
                            denom = _wsum - (w2 / _wsum)
                        except (OverflowError, ValueError):
                            denom = None
                        if denom is not None and denom > 0.0 and _ls_math.isfinite(denom):
                            _tau_scaled = float(max(
                                0.0, (q - (len(groups) - 1)) / denom))
                            if _tau_scaled == 0.0:
                                # q <= df: the between-stratum variance is
                                # exactly zero, independent of the SE scale.
                                tau2 = 0.0
                            else:
                                # tau2 is in squared scaled-SE units.
                                try:
                                    tau2 = _tau_scaled * (_se_scale * _se_scale)
                                except OverflowError:
                                    tau2 = None
                                if tau2 is not None and not _ls_math.isfinite(tau2):
                                    tau2 = None
                                # A positive (q - df) cannot have tau2 == 0;
                                # that value is an underflow, not evidence of
                                # no between-stratum variance.  Withhold it.
                                if tau2 == 0.0:
                                    tau2 = None
                else:
                    q = None
                    q_reason = 'Cochran Q is not representable in float64'
            else:
                q_reason = 'Cochran Q deviation/SE is not representable in float64'
        else:
            q_reason = ('a stratum has zero seed-level standard error and a '
                        'non-zero deviation, or the inverse-variance weights '
                        'are not finite')
    else:
        q_reason = 'at least one stratum has fewer than two seeds or no finite SE'

    # Pairwise stratum contrasts: descriptive, no multiplicity claim.
    pairwise = []
    for a_i in range(len(groups)):
        for b_i in range(a_i + 1, len(groups)):
            diffs = [lr_vals[a_i][si] - lr_vals[b_i][si] for si in range(s_count)]
            summary = _ls_summary(diffs, conf)
            ci = summary['ci95']
            ratio_pair = (_safe_exp_ppl(summary['mean'])
                          if summary['mean'] is not None else None)
            if ratio_pair is not None and not _ls_math.isfinite(ratio_pair):
                ratio_pair = None
            pairwise.append({
                'a': int(a_i), 'b': int(b_i),
                'log_ratio_diff': summary['mean'],
                'ratio': ratio_pair,
                'ci95': ci,
                'excludes_zero': (bool(ci is not None
                                       and (ci[0] > 0.0 or ci[1] < 0.0))),
            })
    extreme = None
    if all(m is not None for m in means):
        hi = int(max(range(len(groups)), key=lambda i: means[i]))
        lo = int(min(range(len(groups)), key=lambda i: means[i]))
        diffs = [lr_vals[hi][si] - lr_vals[lo][si] for si in range(s_count)]
        summary = _ls_summary(diffs, conf)
        extreme = {
            'high_stratum': hi, 'low_stratum': lo,
            'log_ratio_diff': summary['mean'],
            'ratio': (_safe_exp_ppl(summary['mean'])
                      if summary['mean'] is not None else None),
            'ci95': summary['ci95'],
            'excludes_zero': (bool(summary['ci95'] is not None
                                   and (summary['ci95'][0] > 0.0
                                        or summary['ci95'][1] < 0.0))),
        }
        if extreme['ratio'] is not None and not _ls_math.isfinite(extreme['ratio']):
            extreme['ratio'] = None

    overall_sign = _ls_sign(overall_lr_summary['mean'])
    stratum_signs = [_ls_sign(s['log_ratio']) for s in strata]
    std_sign = _ls_sign(std_lr_summary['mean'])
    nonzero_signs = [x for x in stratum_signs if x not in (None, 0)]
    opposite = [int(i) for i, x in enumerate(stratum_signs)
                if overall_sign not in (None, 0) and x not in (None, 0)
                and x != overall_sign]
    raw_signs = [_ls_sign(s['delta_ppl']) for s in strata]
    raw_overall_sign = _ls_sign(overall_dp_summary['mean'])
    raw_opposite = [int(i) for i, x in enumerate(raw_signs)
                    if raw_overall_sign not in (None, 0) and x not in (None, 0)
                    and x != raw_overall_sign]
    sign_block = {
        'overall_log_ratio_sign': overall_sign,
        'stratum_log_ratio_signs': stratum_signs,
        'sign_consistent': (bool(len(set(nonzero_signs)) <= 1)
                            if nonzero_signs else None),
        'opposite_strata': opposite,
        'standardized_sign': std_sign,
        'standardization_flips_sign': (bool(std_sign not in (None, 0)
                                            and overall_sign not in (None, 0)
                                            and std_sign != overall_sign)),
        'raw_delta_overall_sign': raw_overall_sign,
        'raw_delta_stratum_signs': raw_signs,
        'raw_delta_opposite_strata': raw_opposite,
        # A strict "every stratum ΔPPL opposes the overall ΔPPL" reversal is
        # impossible for the same units: the pooled CE is a positive weighted
        # average of the stratum CEs and exp is monotone.  The meaningful
        # composition readout is therefore the standardised *log* effect and
        # the list of strata whose sign opposes the overall one, not a
        # fabricated raw-scale Simpson flag.
    }

    # Which stratum carries the token-weighted log effect?  Shares can exceed
    # 1 or be negative when strata point in opposite directions; that is the
    # reason the block reports signed contributions as well as their absolute
    # concentration.
    contributions = []
    for s in strata:
        if s['log_ratio'] is None or s['token_share'] is None:
            contributions.append(None)
        else:
            try:
                c = float(s['token_share'] * s['log_ratio'])
            except (OverflowError, ValueError):
                c = None
            contributions.append(c if (c is not None
                                       and _ls_math.isfinite(c)) else None)
    if any(c is None for c in contributions):
        concentration = {'available': False,
                         'reason': 'a stratum contribution is not representable'}
    else:
        # Contributions are finite individually but their absolute sum can
        # exceed float64 on a very large panel; that must not abort the whole
        # block with a raw `fsum` OverflowError.
        abs_sum = total_contrib = None
        try:
            abs_sum = _ls_math.fsum(abs(c) for c in contributions)
            total_contrib = _ls_math.fsum(contributions)
        except (OverflowError, ValueError):
            pass
        if (abs_sum is not None and total_contrib is not None
                and _ls_math.isfinite(abs_sum)
                and _ls_math.isfinite(total_contrib) and abs_sum > 0.0):
            shares = [abs(c) / abs_sum for c in contributions]
            dom = int(max(range(len(groups)), key=lambda i: shares[i]))
            concentration = {
                'available': True,
                'contributions': contributions,
                'total_log_effect': float(total_contrib),
                'absolute_share': [float(x) for x in shares],
                'max_abs_share': float(shares[dom]),
                'dominant_stratum': dom,
                'cancellation': bool(total_contrib != 0.0 and abs_sum > 0.0
                                     and abs(total_contrib) / abs_sum < 0.5),
            }
        elif abs_sum == 0.0 and total_contrib == 0.0:
            concentration = {'available': False,
                             'reason': 'every stratum contribution is zero'}
        else:
            concentration = {
                'available': False,
                'reason': ('the stratum contributions or their absolute sum '
                           'are not representable in float64')}

    out = {
        'kind': 'length_stratification',
        'status': 'ok',
        'policy': {
            'version': LENGTH_STRATA_POLICY['version'],
            'n_strata': int(len(groups)),
            'requested_n_strata': int(n_strata),
            'min_units_per_stratum': int(min_units),
            'conf': float(conf),
            'reference': ref_name,
            'reference_weights': [float(x) for x in ref_weights],
            'stratification': 'equal-count quantile bins of unit token counts',
            'q_method': LENGTH_STRATA_POLICY['q_method'],
        },
        'n_units': int(d_count),
        'n_seeds': int(s_count),
        'n_strata': int(len(groups)),
        'seeds': [int(x) for x in seed_labels],
        'units': {'n_units': int(d_count), 'n_tokens': int(total_tokens),
                  'token_counts': [int(x) for x in T]},
        'strata': strata,
        'per_seed': per_seed,
        'overall': {
            'log_ratio': overall_lr_summary['mean'],
            'ratio': overall_ratio,
            'log_ratio_ci95': overall_lr_summary['ci95'],
            'delta_ppl': overall_dp_summary['mean'],
            'delta_ppl_ci95': overall_dp_summary['ci95'],
            'n_seeds_with_delta_ppl': int(len(overall_dp_finite)),
        },
        'standardized': {
            'reference': ref_name,
            'weights': [float(x) for x in ref_weights],
            'log_ratio': std_lr_summary['mean'],
            'ratio': std_ratio,
            'log_ratio_ci95': std_lr_summary['ci95'],
            'delta_ppl': std_dp_summary['mean'],
            'delta_ppl_ci95': std_dp_summary['ci95'],
            'log_ratio_shift_from_overall': (
                float(std_lr_summary['mean'] - overall_lr_summary['mean'])
                if std_lr_summary['mean'] is not None
                and overall_lr_summary['mean'] is not None else None),
        },
        'heterogeneity': {
            'q': q, 'df': int(len(groups) - 1), 'p_chi2': q_p,
            'i2': i2, 'tau2': tau2,
            'extreme_contrast': extreme,
            'pairwise': pairwise,
            'reason': q_reason,
        },
        'sign_consistency': sign_block,
        'concentration': concentration,
        'role': LENGTH_STRATA_POLICY['role'],
        'boundary': LENGTH_STRATA_POLICY['boundary'],
        'upstream': {
            'estimand': 'token-pooled paired PPL difference',
            'unit': 'evaluation unit (article/document/row, as supplied)',
            'ci': 'seed-level two-sided t interval',
        },
    }
    return out
