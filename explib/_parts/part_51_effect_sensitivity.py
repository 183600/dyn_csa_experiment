"""Effect-scale and robust-location sensitivity for a paired PPL contrast.

The frozen confirmation test is an exact sign-flip test on the **raw PPL
difference** ``a - b`` for each training seed.  That is a legitimate test of a
well-defined statistic, but it is not the only defensible scale for the same
paired measurement, and it is not robust to a few large-magnitude seeds:

* PPL is a nonlinear transform of mean token NLL.  ``mean(a - b)`` and
  ``mean(log(a) - log(b))`` are different estimands; with heterogeneous seed
  magnitudes they can even point in opposite directions.  A claim that only
  quotes the raw-PPL p-value is therefore conditional on an analysis choice
  that the reader cannot see.
* The mean sign-flip test is driven by magnitudes.  If a handful of seeds
  carry most of the absolute effect, a p-value can reject while a rank-based
  location interval (which uses only the ordering and the signed ranks)
  includes zero.  That is not a contradiction, but it changes what may be
  claimed: "the raw mean moved" is weaker than "the typical seed moved".

This fragment adds two descriptive readouts and one machine-readable
classification, without adding a p-value to the frozen family:

1. :func:`effect_sensitivity` compares the raw-PPL and log-PPL sign-flip tests
   on the same paired seeds, reports the geometric PPL ratio and the relative
   effect, and labels direction / significance / location disagreements.
2. :func:`hodges_lehmann_interval` inverts the exact Wilcoxon signed-rank test
   to give a distribution-free interval for the paired pseudo-median.  It is
   exact for ``n <= EFFECT_SENSITIVITY_POLICY['max_location_pairs']`` and uses
   the same zero-dropping convention as ``wilcoxon_signed_rank_p``.

Neither readout changes ``p_exact_signflip``, ``p_holm``, ``verified`` or a
frozen ``confirmed`` verdict.  They answer a narrower question: *does the
confirmed direction survive the obvious scale and location checks, or is it a
raw-mean-only reading?*
"""

import math as _es_math

import numpy as _es_np


EFFECT_SENSITIVITY_POLICY = {
    'version': 'effect-sensitivity-1',
    'alpha': 0.05,
    'conf': 0.95,
    'min_pairs': 2,
    'max_location_pairs': 200,
    'location_estimator': 'Hodges-Lehmann pseudo-median',
    'location_method': 'exact-signed-rank-inversion',
    'location_zero_policy': 'zero paired differences are dropped before ranking',
    'location_tie_policy': ('Walsh averages are kept with ties; the critical '
                            'value uses the untied rank-sum law (a '
                            'deterministic tie-breaking convention). A tied '
                            'panel whose selected endpoints coincide is '
                            'reported as degenerate rather than used.'),
    'role': ('descriptive effect-scale/location sensitivity; it adds no '
             'p-value to the Holm family and never changes a frozen verdict'),
    'note': ('raw PPL difference and log PPL ratio are different estimands; '
             'the Hodges-Lehmann interval targets the paired pseudo-median, '
             'not the arithmetic mean'),
}

EFFECT_SENSITIVITY_STATUSES = (
    'concordant',
    'scale_direction_discordant',
    'scale_significance_discordant',
    'scale_concordant_location_unavailable',
    'location_discordant',
    'direction_unspecified',
    'zero_effect',
    'degenerate',
    'not_estimable',
)

_EFFECT_SENSITIVITY_BOUNDARY = (
    'descriptive effect-scale and robust-location sensitivity. The raw-PPL '
    'and log-PPL sign-flip tests answer different questions and are not two '
    'independent tests; neither is added to the frozen Holm family. The '
    'Hodges-Lehmann interval is an exact signed-rank interval for the paired '
    'pseudo-median under symmetry, not a confidence interval for the '
    'arithmetic mean, and it is not a substitute for the seed-level exact '
    'sign-flip p or for an independent replication. Zero paired differences '
    'are dropped before ranking; tied Walsh averages use the untied rank '
    'convention, and a zero-width tied interval is reported as degenerate. '
    '`concordant` is not evidence that '
    'the effect is practically important, causal, or transportable.')


def _es_number(value):
    """Finite float or ``None``; booleans and strings are not measurements."""
    if isinstance(value, (bool, _es_np.bool_)):
        return None
    # JSON round-trips can leave a numeric value as ``"1.25"``.  Treating that
    # as a measurement silently invents a PPL instead of dropping the invalid
    # pair (the same contract as the scale/location regression suite).
    if isinstance(value, (str, bytes, bytearray, dict)):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return out if _es_math.isfinite(out) else None


def _es_positive(value):
    out = _es_number(value)
    return out if (out is not None and out > 0.0) else None


def _es_direction(value):
    if value is None:
        return None
    if value > 0.0:
        return 'positive'
    if value < 0.0:
        return 'negative'
    return None


def _es_float_or_none(value):
    out = _es_number(value)
    return out


def _es_mean(values):
    if not values:
        return None
    try:
        out = float(_scaled_mean_or_raw(values))
    except (TypeError, ValueError, OverflowError):
        return None
    return out if _es_math.isfinite(out) else None


def _es_median(values):
    if not values:
        return None
    try:
        out = float(_scaled_median_or_raw(values))
    except (TypeError, ValueError, OverflowError):
        return None
    return out if _es_math.isfinite(out) else None


def _es_sequence(values, where):
    if isinstance(values, (str, bytes, bytearray, dict)):
        raise ValueError(f'{where}: values must be an iterable of numbers, '
                         f'not a {type(values).__name__}')
    try:
        return list(values if values is not None else [])
    except TypeError:
        raise ValueError(f'{where}: values must be an iterable') from None


def _es_conf(conf):
    if isinstance(conf, (bool, _es_np.bool_)):
        raise ValueError(f'conf must be a number in (0, 1), got {conf!r}')
    try:
        out = float(conf)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'conf must be a number in (0, 1), got {conf!r}') from None
    if not (_es_math.isfinite(out) and 0.0 < out < 1.0):
        raise ValueError(f'conf must be a finite number in (0, 1), got {conf!r}')
    return out


def hodges_lehmann_interval(deltas, *, conf=None):
    """Exact distribution-free interval for the paired pseudo-median.

    The interval is obtained by inverting the two-sided Wilcoxon signed-rank
    test: with Walsh averages ``W_(1) <= ... <= W_(N)`` and ``k`` the largest
    rank sum whose lower-tail probability is at most ``(1-conf)/2``, the
    interval is ``[W_(k+1), W_(N-k)]``.  The attained level is reported
    because the signed-rank law is discrete and usually gives more than the
    nominal coverage.

    Zero differences are dropped (the standard zero-difference convention for
    the signed-rank test); the count is reported.  A panel that is too small
    to attain the requested level at all is reported as ``unbounded`` rather
    than given a fake zero-width interval.  Panels above
    ``max_location_pairs`` are refused instead of silently allocating an
    O(n**2) Walsh table.
    """
    conf = _es_conf(EFFECT_SENSITIVITY_POLICY['conf'] if conf is None else conf)
    raw = _es_sequence(deltas, 'hodges_lehmann_interval')
    vals = []
    n_input = len(raw)
    for x in raw:
        f = _es_number(x)
        if f is not None:
            vals.append(f)
    n_dropped = n_input - len(vals)
    nonzero = [v for v in vals if v != 0.0]
    n_zero = len(vals) - len(nonzero)
    n = len(nonzero)
    out = {
        'kind': 'hodges_lehmann_interval',
        'status': 'not_estimable',
        'conf': float(conf),
        'n_input': int(n_input),
        'n_dropped': int(n_dropped),
        'n_zero': int(n_zero),
        'n': int(n),
        'n_walsh': 0,
        'critical_k': None,
        'attained_conf': None,
        'estimate': None,
        'ci': None,
        'excludes_zero': None,
        'direction': None,
        'ties_present': None,
        'method': EFFECT_SENSITIVITY_POLICY['location_method'],
        'estimator': EFFECT_SENSITIVITY_POLICY['location_estimator'],
        'tie_policy': EFFECT_SENSITIVITY_POLICY['location_tie_policy'],
        'reason': None,
    }
    if n < int(EFFECT_SENSITIVITY_POLICY['min_pairs']):
        out['reason'] = (f'{n} nonzero paired difference(s); at least '
                         f'{EFFECT_SENSITIVITY_POLICY["min_pairs"]} are needed '
                         'for a signed-rank interval')
        return out
    if n > int(EFFECT_SENSITIVITY_POLICY['max_location_pairs']):
        out['reason'] = (f'{n} nonzero paired differences exceed the exact '
                         f'Walsh-table cap {EFFECT_SENSITIVITY_POLICY["max_location_pairs"]}; '
                         'refuse rather than allocate an O(n**2) table')
        return out
    scale = max(abs(v) for v in nonzero)
    if not (_es_math.isfinite(scale) and scale > 0.0):
        out['reason'] = 'the nonzero paired differences have no finite scale'
        return out
    scaled = _es_np.asarray([v / scale for v in nonzero], dtype=float)
    walsh = (scaled[:, None] + scaled[None, :]) / 2.0
    walsh = _es_np.sort(walsh[_es_np.triu_indices(n)])
    total_ranks = int(walsh.size)
    out['n_walsh'] = total_ranks
    out['ties_present'] = bool(_es_np.unique(walsh).size < walsh.size)
    estimate = _es_median([v * scale for v in walsh.tolist()])
    out['estimate'] = estimate
    out['direction'] = _es_direction(estimate)
    # Exact null law of W+ for ranks 1..n.  Ties use a deterministic
    # tie-breaking convention; a zero-width tied endpoint is reported as
    # degenerate rather than presented as a coverage statement.
    counts = [0] * (total_ranks + 1)
    counts[0] = 1
    for rank in range(1, n + 1):
        for s in range(total_ranks, rank - 1, -1):
            c = counts[s - rank]
            if c:
                counts[s] += c
    total = 1 << n
    alpha = 1.0 - conf
    cumulative = 0
    critical_k = None
    cumulative_at = None
    for k in range(0, total_ranks // 2):
        cumulative += counts[k]
        if 2 * cumulative <= alpha * total + 1e-12:
            critical_k = k
            cumulative_at = cumulative
        else:
            break
    if critical_k is None:
        out['status'] = 'unbounded'
        out['reason'] = (f'no signed-rank critical value attains {conf:g} '
                         f'coverage at n={n}; the exact acceptance set is the '
                         'whole real line')
        return out
    lo = float(walsh[critical_k] * scale)
    hi = float(walsh[total_ranks - 1 - critical_k] * scale)
    if not (_es_math.isfinite(lo) and _es_math.isfinite(hi) and lo <= hi):
        out['reason'] = ('the Walsh-average endpoints are not representable '
                         'as finite floats at this magnitude')
        return out
    if lo == hi:
        out['status'] = 'degenerate'
        out['reason'] = ('all Walsh averages at the selected ranks coincide; a '
                         'tied panel does not identify a positive-width interval, '
                         'so no confidence statement is made')
        return out
    out['status'] = 'ok'
    out['critical_k'] = int(critical_k)
    out['attained_conf'] = float(1.0 - 2.0 * cumulative_at / total)
    out['ci'] = [lo, hi]
    out['excludes_zero'] = bool(lo > 0.0 or hi < 0.0)
    return out


def effect_sensitivity(with_values, without_values, *, seeds=None, alpha=None,
                       conf=None):
    """Compare raw-PPL and log-PPL tests and a robust location interval.

    ``with_values`` / ``without_values`` are the two arms' per-seed PPLs on
    the **same** paired seeds.  The raw contrast is ``with - without`` (a
    negative value means the ``with`` arm has lower PPL); the log contrast is
    ``log(with) - log(without)``.  Both are tested with the same exact
    sign-flip routine used by the confirmation panel.

    Returns a strictly JSON-safe block with ``raw``, ``log_ratio``,
    ``location`` and ``concordance`` sub-blocks and a top-level ``status``:

    * ``concordant``: direction, alpha-level significance and the
      signed-rank interval agree;
    * ``scale_direction_discordant``: the raw mean and the log-ratio mean point
      in opposite directions (the estimand, not just the p-value, is
      scale-dependent);
    * ``scale_significance_discordant``: same direction, but exactly one of
      the two scales is significant at ``alpha``;
    * ``scale_concordant_location_unavailable``: raw and log agree, but the
      signed-rank interval is unavailable or degenerate;
    * ``location_discordant``: the raw test and the exact signed-rank interval
      disagree about excluding zero;
    * ``direction_unspecified``: both means are exactly zero although some
      paired differences are non-zero (the signs cancel);
    * ``zero_effect``: every valid paired difference is exactly zero;
    * ``not_estimable``: fewer than ``min_pairs`` valid positive pairs or the
      exact tests are unavailable.

    Invalid pairs (non-finite, non-positive PPL, booleans) are dropped by
    index and named in ``dropped``; duplicate or non-integer ``seeds`` are a
    hard error because they would double-count a training run.
    """
    where = 'effect_sensitivity'
    a_in = _es_sequence(with_values, f'{where}.with_values')
    b_in = _es_sequence(without_values, f'{where}.without_values')
    if len(a_in) != len(b_in):
        raise ValueError(f'{where}: with/without lengths differ '
                         f'({len(a_in)} vs {len(b_in)})')
    n_input = len(a_in)
    if seeds is None:
        seed_list = list(range(n_input))
    else:
        if isinstance(seeds, (str, bytes, bytearray, dict)):
            raise ValueError(f'{where}: seeds must be an iterable of integers')
        try:
            seed_list = list(seeds)
        except TypeError:
            raise ValueError(f'{where}: seeds must be an iterable') from None
        if len(seed_list) != n_input:
            raise ValueError(f'{where}: seeds length {len(seed_list)} does not '
                             f'match values length {n_input}')
        norm = [_as_seed_int(s) for s in seed_list]
        if any(s is None for s in norm):
            raise ValueError(f'{where}: seeds must be integers, got {seed_list!r}')
        if len(set(norm)) != len(norm):
            raise ValueError(f'{where}: duplicate seeds {norm!r}')
        seed_list = norm
    alpha = _validated_alpha(
        EFFECT_SENSITIVITY_POLICY['alpha'] if alpha is None else alpha,
        context=f'{where} alpha')
    conf = _es_conf(EFFECT_SENSITIVITY_POLICY['conf'] if conf is None else conf)
    pairs = []
    dropped = []
    for i, (a_raw, b_raw) in enumerate(zip(a_in, b_in)):
        a = _es_positive(a_raw)
        b = _es_positive(b_raw)
        reason = None
        if a is None or b is None:
            reason = 'non-positive or non-finite PPL pair'
        if reason is not None:
            dropped.append({'index': int(i), 'seed': int(seed_list[i]),
                            'reason': reason})
            continue
        pairs.append((int(seed_list[i]), float(a), float(b)))
    n = len(pairs)
    raw_vals = [a - b for (_s, a, b) in pairs]
    log_vals = [_es_math.log(a) - _es_math.log(b) for (_s, a, b) in pairs]
    out = {
        'kind': 'effect_scale_location_sensitivity',
        'version': EFFECT_SENSITIVITY_POLICY['version'],
        'status': 'not_estimable',
        'status_reason': None,
        'alpha': float(alpha),
        'conf': float(conf),
        'n': int(n),
        'n_input': int(n_input),
        'n_dropped': int(len(dropped)),
        'dropped': dropped,
        'seeds': [int(s) for (s, _a, _b) in pairs],
        'per_seed': [
            {'seed': int(s), 'with': float(a), 'without': float(b),
             'raw_delta': float(a - b),
             'log_ratio': float(_es_math.log(a) - _es_math.log(b))}
            for (s, a, b) in pairs
        ],
        'raw': {
            'estimand': 'with - without (PPL points; negative = with is lower)',
            'mean': _es_mean(raw_vals),
            'median': _es_median(raw_vals),
            'direction': None,
            'p_exact_signflip': None,
            'n_nonzero': 0,
            'p_floor': None,
            'method': 'none',
            'significant': None,
        },
        'log_ratio': {
            'estimand': 'log(with) - log(without); exp(mean) is the geometric PPL ratio',
            'mean': _es_mean(log_vals),
            'geometric_ratio': None,
            'relative_pct': None,
            'direction': None,
            'p_exact_signflip': None,
            'n_nonzero': 0,
            'p_floor': None,
            'method': 'none',
            'significant': None,
        },
        'location': hodges_lehmann_interval(raw_vals, conf=conf),
        'concordance': {
            'direction_agree': None,
            'significance_agree': None,
            'raw_significant': None,
            'log_significant': None,
            'location_excludes_zero': None,
            'all_agree': None,
        },
        'policy': {
            'version': EFFECT_SENSITIVITY_POLICY['version'],
            'alpha': float(alpha),
            'conf': float(conf),
            'min_pairs': int(EFFECT_SENSITIVITY_POLICY['min_pairs']),
            'max_location_pairs': int(EFFECT_SENSITIVITY_POLICY['max_location_pairs']),
            'location_estimator': EFFECT_SENSITIVITY_POLICY['location_estimator'],
            'location_method': EFFECT_SENSITIVITY_POLICY['location_method'],
            'location_zero_policy': EFFECT_SENSITIVITY_POLICY['location_zero_policy'],
            'location_tie_policy': EFFECT_SENSITIVITY_POLICY['location_tie_policy'],
            'role': EFFECT_SENSITIVITY_POLICY['role'],
        },
        'boundary': _EFFECT_SENSITIVITY_BOUNDARY,
    }
    if n < int(EFFECT_SENSITIVITY_POLICY['min_pairs']):
        out['status_reason'] = (f'{n} valid paired seed(s); at least '
                                f'{EFFECT_SENSITIVITY_POLICY["min_pairs"]} are '
                                'needed for a scale/location comparison')
        return out
    raw_p = exact_signflip_p(raw_vals)
    log_p = exact_signflip_p(log_vals)
    raw_mean = out['raw']['mean']
    log_mean = out['log_ratio']['mean']
    out['raw'].update({
        'direction': _es_direction(raw_mean),
        'p_exact_signflip': _es_float_or_none(raw_p.get('p')),
        'n_nonzero': int(raw_p.get('n_nonzero') or 0),
        'p_floor': _es_float_or_none(raw_p.get('p_floor')),
        'method': raw_p.get('method'),
    })
    geo = None
    rel = None
    if log_mean is not None:
        try:
            geo = _es_math.exp(log_mean)
        except OverflowError:
            geo = None
        if geo is not None and (_es_math.isfinite(geo) and geo > 0.0):
            rel = 100.0 * (geo - 1.0)
            if not _es_math.isfinite(rel):
                rel = None
        else:
            geo = None
    out['log_ratio'].update({
        'geometric_ratio': (float(geo) if geo is not None else None),
        'relative_pct': (float(rel) if rel is not None else None),
        'direction': _es_direction(log_mean),
        'p_exact_signflip': _es_float_or_none(log_p.get('p')),
        'n_nonzero': int(log_p.get('n_nonzero') or 0),
        'p_floor': _es_float_or_none(log_p.get('p_floor')),
        'method': log_p.get('method'),
    })
    raw_sig = (out['raw']['p_exact_signflip'] is not None
               and out['raw']['p_exact_signflip'] < alpha)
    log_sig = (out['log_ratio']['p_exact_signflip'] is not None
               and out['log_ratio']['p_exact_signflip'] < alpha)
    out['raw']['significant'] = bool(raw_sig)
    out['log_ratio']['significant'] = bool(log_sig)
    raw_dir = out['raw']['direction']
    log_dir = out['log_ratio']['direction']
    if raw_dir is None and log_dir is None:
        direction_agree = None
    else:
        direction_agree = bool(raw_dir is not None and raw_dir == log_dir)
    significance_agree = bool(raw_sig == log_sig)
    loc = out['location']
    loc_excl = loc.get('excludes_zero') if isinstance(loc, dict) else None
    loc_agree = (None if loc_excl is None else bool(loc_excl == raw_sig))
    out['concordance'].update({
        'direction_agree': direction_agree,
        'significance_agree': significance_agree,
        'raw_significant': bool(raw_sig),
        'log_significant': bool(log_sig),
        'location_excludes_zero': loc_excl,
        'all_agree': (None if (direction_agree is None or loc_agree is None)
                      else bool(direction_agree and significance_agree and loc_agree)),
    })
    if all(v == 0.0 for v in raw_vals):
        out['status'] = 'zero_effect'
        out['status_reason'] = ('every valid paired difference is exactly zero; '
                                'there is no direction or location to compare')
    elif direction_agree is None:
        out['status'] = 'direction_unspecified'
        out['status_reason'] = ('the raw and log means are both exactly zero '
                                'even though some paired differences are '
                                'non-zero; there is no direction to compare')
    elif direction_agree is False:
        out['status'] = 'scale_direction_discordant'
        out['status_reason'] = ('the raw-PPL mean and the log-PPL mean point in '
                                'opposite directions; the reported direction '
                                'depends on the chosen effect scale')
    elif significance_agree is False:
        out['status'] = 'scale_significance_discordant'
        out['status_reason'] = ('raw PPL and log PPL agree in direction but '
                                'exactly one reaches alpha; the significance '
                                'claim is scale-dependent')
    elif loc_agree is False:
        out['status'] = 'location_discordant'
        out['status_reason'] = ('the raw-PPL sign-flip test and the exact '
                                'signed-rank location interval disagree about '
                                'excluding zero; the raw mean may be carried by '
                                'large-magnitude seeds rather than a shift of '
                                'the typical seed')
    elif loc_agree is None:
        out['status'] = 'scale_concordant_location_unavailable'
        out['status_reason'] = ('raw and log PPL agree on direction and '
                                'zero-exclusion, but the exact signed-rank '
                                'location interval is unavailable or degenerate; '
                                'the scale check passes and the location check '
                                'is not claimed')
    else:
        out['status'] = 'concordant'
        out['status_reason'] = ('raw PPL, log PPL and the exact signed-rank '
                                'location interval agree on direction and '
                                'zero-exclusion at the declared level')
    return out
