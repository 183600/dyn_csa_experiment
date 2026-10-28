

def paired_bootstrap_ci(deltas, *, n_boot=4000, seed=20261018):
    """Paired studentized bootstrap CI for the mean paired difference.

    The paired bootstrap uses the same seeded local RNG and chunked draws as
    :func:`bootstrap_mean_ci`.  It is a sensitivity/uncertainty readout; the
    exact sign-flip p-value remains the primary test and is not affected by
    the interval construction.
    """
    d = np.asarray([f for f in (_finite_scalar(x) for x in (deltas if deltas is not None else []))
                    if f is not None], dtype=float)
    if d.size < 2:
        return None
    # Draws are generated in chunks, so the panel size does not require a
    # correspondingly large index table: the seed-completion path can grow a
    # panel to hundreds/thousands of seeds and paired_effect_stats calls this
    # for every displayed contrast.
    nb = _as_seed_int(n_boot)
    sd = _as_seed_int(seed)
    if nb is None or sd is None or nb < 2:
        return None
    _ci = _bootstrap_t_interval(d, n_boot=nb, seed=sd, conf=0.95)
    return [_ci['ci_low'], _ci['ci_high']]

def unpaired_bootstrap_ci(with_values, without_values, *, n_boot=4000, seed=20261018):
    a = np.asarray([f for f in (_finite_scalar(x) for x in (with_values if with_values is not None else []))
                    if f is not None], dtype=float)
    b = np.asarray([f for f in (_finite_scalar(x) for x in (without_values if without_values is not None else []))
                    if f is not None], dtype=float)
    if a.size < 2 or b.size < 2:
        return None
    nb = _as_seed_int(n_boot)
    sd = _as_seed_int(seed)
    if nb is None or sd is None or nb < 2:
        return None
    rng = np.random.default_rng(sd)
    diff = np.empty(nb, dtype=float)
    # Each draw materialises one index table per arm, so size the chunk from
    # their combined unit count.
    chunk = _bootstrap_draw_chunk(a.size + b.size)
    for off in range(0, nb, chunk):
        m = min(chunk, nb - off)
        ai = rng.integers(0, a.size, size=(m, a.size))
        bi = rng.integers(0, b.size, size=(m, b.size))
        diff[off:off + m] = a[ai].mean(axis=1) - b[bi].mean(axis=1)
    return _percentile_ci(diff.tolist())

_WILCOXON_EXACT_MAX_N = 200


def wilcoxon_signed_rank_p(deltas):
    """Two-sided Wilcoxon signed-rank p, exact over the integer rank-sum
    distribution for n<=200 and normal above that.  Reported as a SENSITIVITY
    check, not as a replacement for the sign-flip test: it assumes symmetric
    paired differences, which is not guaranteed by sharing a seed number.

    Ranks are doubled so average ranks from ties stay integral; a 0/1 subset-
    sum DP then counts every sign assignment exactly in O(n * sum_ranks),
    which keeps the sensitivity check exact when seed completion grows the
    panels."""
    d = np.asarray([f for f in (_finite_scalar(x) for x in (deltas if deltas is not None else []))
                    if f is not None], dtype=float)
    # Rescaling every delta must not change ranks or drop the panel: an
    # absolute 1e-15 cutoff treats a 1e-200-scaled panel as all-zero.
    _tol = 1e-15 * float(np.max(np.abs(d))) if d.size else 0.0
    d = d[np.abs(d) > _tol]
    n = int(d.size)
    if n == 0:
        return {'p': float('nan'), 'method': 'none', 'n_nonzero': 0}
    order = np.argsort(np.abs(d))
    ranks = np.empty(n, dtype=float)
    _abs = np.abs(d)[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and abs(_abs[j] - _abs[i]) <= _tol:
            j += 1
        ranks[order[i:j]] = (i + j + 1) / 2.0  # average rank (1-based)
        i = j
    w_plus = float(ranks[d > 0].sum())
    total = float(ranks.sum())
    w = min(w_plus, total - w_plus)
    if n <= _WILCOXON_EXACT_MAX_N:
        r2 = [int(round(float(r) * 2.0)) for r in ranks]
        total2 = int(sum(r2))
        w2 = int(round(w * 2.0))
        if 2 * w2 >= total2:
            # w is the smaller tail; equality means both tails meet, i.e. every
            # sign assignment qualifies (p = 1).
            return {'p': 1.0, 'method': 'exact-dp', 'n_nonzero': n, 'w_plus': w_plus, 'w_min': w}
        counts = [0] * (total2 + 1)
        counts[0] = 1
        for r in r2:
            for sm in range(total2, r - 1, -1):
                c = counts[sm - r]
                if c:
                    counts[sm] += c
        cnt = sum(counts[:w2 + 1]) + sum(counts[total2 - w2:])
        p = min(1.0, cnt / float(1 << n))
        return {'p': float(p), 'method': 'exact-dp', 'n_nonzero': n, 'w_plus': w_plus, 'w_min': w}
    mu = total / 2.0
    var = float((ranks ** 2).sum()) / 4.0
    if var <= 0:
        return {'p': float('nan'), 'method': 'normal', 'n_nonzero': n, 'w_plus': w_plus, 'w_min': w}
    z = (abs(w_plus - mu) - 0.5) / math.sqrt(var)
    p = math.erfc(max(0.0, z) / math.sqrt(2.0))
    return {'p': float(p), 'method': 'normal', 'n_nonzero': n, 'w_plus': w_plus, 'w_min': w,
            'note': f'normal approximation; exact DP is used for n<={_WILCOXON_EXACT_MAX_N}'}

def paired_effect_stats(deltas, alpha=0.05, min_n=3, with_values=None, without_values=None,
                        seeds=None):
    """Paired contrast summary with an explicit effect-vs-seed-noise verdict.

    Reports n, mean, sample std (ddof=1), sem, Cohen's d_z, the exact
    sign-flip p, the p-resolution floor at this n, and four booleans:

      underpowered       — n < `min_n` (the 3-seed floor the reporting rules
                           require for a key experiment);
      resolution_limited — the smallest p the exact sign-flip test can reach
                           at this n (2/2**n) is still > alpha, so NO effect,
                           however large, can ever clear the significance bar
                           at this seed count.  This is a resolution ceiling,
                           not evidence against the effect, and must not be
                           read as "no effect";
      within_seed_noise  — |mean| <= std, i.e. the seed-to-seed spread alone is
                           at least as large as the reported effect (a +0.2%
                           mean with 0.5% std is NOT a usable gain);
      significant        — p < alpha AND NOT within_seed_noise AND a powered,
                           resolvable panel (n >= min_n and p_floor <= alpha).
                           A loose alpha must not turn an underpowered n into
                           a positive verdict.

    It additionally reports a two-sided 95% CI for the mean effect and a
    leave-one-seed-out (LOO) stability read: with the 3-5 seeds the reporting
    rules require, a significant mean can still be driven by a single seed.
    `seed_driven` is True when dropping any one seed flips the sign of the
    mean (or makes it exactly zero), i.e. the direction is not stable across
    seeds even though the aggregate test may look significant.  The LOO pass
    is a single O(n) sweep (a running sum minus each seed); `ci95` uses the t
    critical value rather than a hard-coded z.

    Requiring both the test AND the effect-vs-spread check means a large-n
    p-value can never dress up a sub-noise effect as a real improvement, while
    `resolution_limited` stops a small-n panel from being read as a negative
    result when only the test's granularity, not the data, bound the p."""
    alpha = _validated_alpha(alpha)
    # A minimum-n floor is a gate parameter: int(True)=1 / int(5.9)=5 would
    # silently change which panels are labelled underpowered.
    _min_n = _as_seed_int(min_n)
    if _min_n is None or _min_n < 0:
        raise ValueError(f'min_n must be a non-negative integer, got {min_n!r}')
    min_n = _min_n
    raw = list(deltas if deltas is not None else [])
    n_input = len(raw)
    d_vals, kept_pos, dropped_pos = [], [], []
    for _i, _x in enumerate(raw):
        _f = _finite_scalar(_x)
        if _f is None:
            dropped_pos.append(_i)
        else:
            d_vals.append(_f)
            kept_pos.append(_i)
    d = np.asarray(d_vals, dtype=float)
    n = int(d.size)
    n_dropped = n_input - n
    seed_labels = list(kept_pos)
    dropped_seeds = list(dropped_pos)
    if seeds is not None:
        try:
            _seed_in = list(seeds)
        except TypeError:
            raise ValueError('paired_effect_stats: seeds must be an iterable') from None
        if len(_seed_in) != n_input:
            raise ValueError(
                'paired_effect_stats: seeds length %d does not match deltas length %d'
                % (len(_seed_in), n_input))
        _norm_seeds = [_as_seed_int(s) for s in _seed_in]
        if any(s is None for s in _norm_seeds):
            raise ValueError(f'paired_effect_stats: seeds are not integers ({_seed_in!r})')
        if len(set(_norm_seeds)) != len(_norm_seeds):
            raise ValueError(f'paired_effect_stats: duplicate seeds {_norm_seeds!r}')
        seed_labels = [_norm_seeds[i] for i in kept_pos]
        dropped_seeds = [_norm_seeds[i] for i in dropped_pos]
    _pair_fields = {'n_input': n_input, 'n_dropped': n_dropped,
                    'dropped_positions': dropped_pos,
                    'dropped_seeds': dropped_seeds,
                    'complete_pairs': bool(n_dropped == 0),
                    'seeds': seed_labels}
    if n == 0:
        return {'n': 0, 'mean': float('nan'), 'std': float('nan'), 'std_defined': False, 'sem': float('nan'), 'cohens_dz': float('nan'), 'ci95_low': float('nan'), 'ci95_high': float('nan'), 'loo_means': [], 'seed_driven': None, 'seed_robust': None, 'p_exact_signflip': float('nan'), 'p_floor': float('nan'), 'min_reachable_p': float('nan'), 'underpowered': True, 'resolution_limited': True, 'within_seed_noise': True, 'significant': False, 'effect_vs_noise': 'no data', 'unit': 'seed', 'assumption': 'none (no data)', 'sign_counts': {'positive': 0, 'negative': 0, 'zero': 0}, 'p_wilcoxon': None, 'p_wilcoxon_method': None, 'ci95_paired_bootstrap': None, 'ci95_unpaired_bootstrap': None, **_pair_fields}
    # A single seed has no sample spread.  Keep the historical numeric 0.0 for
    # backward-compatible consumers, but mark it explicitly so a report cannot
    # quote it as a measured zero-variance result.
    std_defined = bool(n > 1)
    _ms = _scaled_mean_std_if_extreme(d, ddof=1) if std_defined else None
    if _ms is not None:
        mean, std = _ms
    else:
        mean = float(d.mean())
        std = float(d.std(ddof=1)) if std_defined else 0.0
    sem = float(std / math.sqrt(n)) if std_defined else float('nan')
    _sf = exact_signflip_p(d)
    p = _sf['p']
    n_nonzero = _sf.get('n_nonzero')
    # Cohen's d_z is a ratio, so the only degenerate case is a measured zero
    # spread.  An absolute std floor (the old 1e-12) makes a tiny but genuinely
    # variable panel look infinitely strong.
    if std > 0.0:
        dz = float(mean / std)
    elif mean == 0.0:
        dz = 0.0
    else:
        dz = float('inf') if mean > 0 else float('-inf')
    # Use the attainable floor of THIS vector (zeros reduce resolution), not
    # the theoretical all-nonzero floor 2/2**n.
    p_floor = _sf.get('p_floor', signflip_p_floor(n))
    underpowered = n < min_n
    resolution_limited = bool(math.isfinite(p_floor) and p_floor > alpha)
    within_seed_noise = abs(mean) <= std
    # The n floor and the exact-test resolution are preconditions, not
    # side-notes: a loose alpha (or an unusually permissive caller) must not
    # promote an underpowered/resolution-limited panel to "significant".
    significant = bool(math.isfinite(p) and p < alpha
                       and (not within_seed_noise)
                       and (not underpowered)
                       and (not resolution_limited))
    # Two-sided 95% CI on the paired mean (t interval; the normal approximation
    # is intentionally avoided at the 3-5 seed counts this repo reports).
    ci_lo = ci_hi = float('nan')
    if n > 1 and math.isfinite(sem):
        tc = _t_critical(0.95, n - 1)
        if math.isfinite(tc):
            ci_lo, ci_hi = mean - tc * sem, mean + tc * sem
    # Leave-one-seed-out stability: does any single seed own the direction?
    loo_means = []
    seed_driven = None
    if n >= 3:
        if _ms is not None:
            _scale = max(abs(float(x)) for x in d)
            _z = [float(x) / _scale for x in d]
            _total_z = math.fsum(_z)
            loo_means = [_scale * ((_total_z - _z[i]) / (n - 1))
                         for i in range(n)]
        else:
            _total = float(d.sum())
            loo_means = [(_total - float(d[i])) / (n - 1) for i in range(n)]
        seed_driven = False
        _mean_scale = max((abs(float(x)) for x in d), default=0.0)
        if abs(mean) > 1e-300 * _mean_scale:
            _full_sign = 1.0 if mean > 0 else -1.0
            seed_driven = any((_m == 0.0) or ((_m > 0) != (_full_sign > 0)) for _m in loo_means)
    seed_robust = None if seed_driven is None else (not seed_driven)
    if underpowered:
        verdict = 'underpowered (n<%d)' % min_n
    elif within_seed_noise:
        verdict = 'effect within seed noise (|mean|<=std)'
    elif significant:
        verdict = 'effect exceeds seed noise (p<%.3g)' % alpha
    elif resolution_limited:
        verdict = 'not decidable at this n: exact-test p floor %.4g > alpha (add seeds to resolve; this is a resolution ceiling, not a null result)' % p_floor
    else:
        verdict = 'not significant at this n (p>=%.3g)' % alpha
    if seed_driven:
        verdict += ' | UNSTABLE: dropping one seed flips the mean sign'
    # Robustness readouts: the exact sign-flip test is a
    # seed-level sign test, so report the sign split, a paired bootstrap CI, a
    # Wilcoxon sensitivity p, and — when the caller supplies both arms — an
    # UNPAIRED bootstrap that does not assume exchangeability.  None of these
    # turns the seed into a token-level unit; they make the assumption visible.
    _sign = {'positive': int((d > 0).sum()), 'negative': int((d < 0).sum()), 'zero': int((d == 0).sum())}
    _wil = wilcoxon_signed_rank_p(d)
    _paired_ci = paired_bootstrap_ci(d)
    _unpaired_ci = None
    if with_values is not None and without_values is not None:
        _unpaired_ci = unpaired_bootstrap_ci(with_values, without_values)
    _assumption = ('exact sign-flip is valid as a sign test under symmetric paired differences; '
                   'same seed does NOT randomise the arm, and architecture-induced RNG/optimisation '
                   'differences are part of the arm. p is SEED-level, never token-level.')
    return {'n': n, 'mean': mean, 'std': std, 'std_defined': std_defined, 'sem': sem, 'cohens_dz': dz, 'ci95_low': ci_lo, 'ci95_high': ci_hi, 'loo_means': loo_means, 'seed_driven': seed_driven, 'seed_robust': seed_robust, 'p_exact_signflip': float(p), 'p_floor': p_floor, 'min_reachable_p': p_floor, 'underpowered': underpowered, 'resolution_limited': resolution_limited, 'within_seed_noise': within_seed_noise, 'significant': significant, 'effect_vs_noise': verdict,
            'unit': 'seed', 'assumption': _assumption, 'sign_counts': _sign,
            'n_nonzero': n_nonzero,
            'p_wilcoxon': _wil.get('p'), 'p_wilcoxon_method': _wil.get('method'),
            'ci95_paired_bootstrap': _paired_ci, 'ci95_unpaired_bootstrap': _unpaired_ci,
            **_pair_fields}

ANALYSIS_CHOICE_POLICY = {
    'version': 'analysis-choice-1',
    'min_specs': 2,
    'statuses': ('stable', 'precision_sensitive', 'direction_unstable',
                 'zero_effect', 'insufficient_specs'),
    'note': ('post-hoc specification concordance only: it re-describes one '
             'panel under several aggregation/weighting choices and never adds '
             'a test, a p-value or a Holm member, and never changes a frozen '
             'decision'),
}


def _choice_number(value):
    """Finite real scalar for an analysis-choice row.

    A numeric string is not a measurement: accepting ``"1.0"`` here would let
    a hand-edited/stale artifact replay as if it were a number (and would make
    the audit's stored/replayed comparison type-blind).  Booleans and complex
    values are rejected for the same reason as in :func:`_finite_scalar`.
    """
    if (isinstance(value, (bool, np.bool_))
            or isinstance(value, (complex, np.complexfloating))
            or not isinstance(value, numbers.Number)):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return out if math.isfinite(out) else None


def _choice_interval(value):
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f'ci95 must be a two-element sequence, got {value!r}')
    lo = _choice_number(value[0])
    hi = _choice_number(value[1])
    if lo is None or hi is None or lo > hi:
        raise ValueError(f'ci95 must be finite with lo<=hi, got {value!r}')
    return [lo, hi]


def _choice_row(raw, *, primary=False, zero_tol=0.0):
    if not isinstance(raw, dict):
        raise ValueError(f'specification must be a dict, got {type(raw).__name__}')
    spec = raw.get('spec')
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError('specification needs a non-empty string `spec`')
    spec = spec.strip()
    for _key in ('estimand', 'unit', 'scale'):
        _val = raw.get(_key)
        if _val is not None and not isinstance(_val, str):
            raise ValueError(f'{spec}: {_key} must be a string or None')
    available = raw.get('available', True)
    if not isinstance(available, (bool, np.bool_)):
        raise ValueError(f'{spec}: available must be a boolean')
    available = bool(available)
    point = None
    ci = None
    reason = raw.get('reason')
    if available:
        point = _choice_number(raw.get('point'))
        if point is None:
            raise ValueError(f'{spec}: available specification needs a finite point')
        ci = _choice_interval(raw.get('ci95'))
        if reason is not None and not isinstance(reason, str):
            raise ValueError(f'{spec}: reason must be a string or None')
    else:
        if not isinstance(reason, str) or not reason.strip():
            reason = 'specification explicitly marked unavailable'
    sign = None
    if point is not None:
        sign = 0 if abs(point) <= zero_tol else (1 if point > 0 else -1)
    return {
        'spec': spec,
        'estimand': raw.get('estimand'),
        'unit': raw.get('unit'),
        'scale': raw.get('scale'),
        'point': point,
        'ci95': ci,
        'available': available,
        'reason': reason,
        'is_primary': bool(primary),
        'sign': sign,
        'excludes_zero': (None if ci is None else bool(ci[0] > 0.0 or ci[1] < 0.0)),
    }


def analysis_choice_concordance(primary, alternatives, *, zero_tol=0.0):
    """Summarise whether one panel's conclusion survives analysis choices.

    The same panel can be summarised by the per-seed mean, median or trimmed
    mean, by the token-pooled difference, or by equal-weighted evaluation
    units.  These are not independent tests and they are not exchangeable
    estimands; reporting only the most favourable one is a specification
    choice that the reader cannot see.  This helper makes the set explicit:

    * ``direction_stable``: the primary point is non-zero and every
      available specification with a non-zero point has the primary sign;
    * ``precision_concordant``: every available specification that carries a
      two-sided interval agrees on whether that interval excludes zero;
    * ``status`` is ``stable``, ``precision_sensitive`` (same sign, different
      zero-exclusion), ``direction_unstable``, ``zero_effect`` or
      ``insufficient_specs``;
    * ``effect_range_ppl`` gives the min/max/span across PPL-delta
      specifications only (log-ratio/win-rate points are not pooled into it);
      ``relative_span`` is emitted only when the primary point is itself a
      PPL-delta, so the ratio never mixes two scales.

    The helper deliberately computes **no p-value** and performs no
    multiplicity correction: adding specifications to a significance family
    would create a garden of forking paths rather than control it.  A
    ``precision_sensitive`` or ``direction_unstable`` readout is a reporting
    caveat, not a new confirmatory test and not a change to any frozen
    decision.
    """
    _zt = _choice_number(zero_tol)
    if _zt is None or _zt < 0.0:
        raise ValueError(f'zero_tol must be a finite non-negative number, got {zero_tol!r}')
    p = _choice_row(primary, primary=True, zero_tol=_zt)
    if not p['available']:
        raise ValueError('primary specification must be available with a finite point')
    try:
        alts = list(alternatives if alternatives is not None else [])
    except TypeError:
        raise ValueError('alternatives must be an iterable of dicts') from None
    rows = [p]
    seen = {p['spec']}
    for raw in alts:
        row = _choice_row(raw, zero_tol=_zt)
        if row['spec'] in seen:
            raise ValueError(f'duplicate specification name {row["spec"]!r}')
        seen.add(row['spec'])
        rows.append(row)
    avail = [r for r in rows if r['available']]
    nonzero = [r for r in avail if r['sign'] != 0]
    pos = int(sum(1 for r in nonzero if r['sign'] > 0))
    neg = int(sum(1 for r in nonzero if r['sign'] < 0))
    zero = int(sum(1 for r in avail if r['sign'] == 0))
    primary_sign = p['sign']
    direction_stable = bool(primary_sign not in (None, 0) and nonzero
                            and all(r['sign'] == primary_sign for r in nonzero))
    same_as_primary = int(sum(1 for r in nonzero if r['sign'] == primary_sign))
    opposite_to_primary = int(sum(1 for r in nonzero if r['sign'] != primary_sign))
    ci_rows = [r for r in avail if r['ci95'] is not None]
    excluding = [r for r in ci_rows if r['excludes_zero']]
    crossing = [r for r in ci_rows if not r['excludes_zero']]
    precision_concordant = None
    if len(ci_rows) >= 2:
        precision_concordant = bool(len(excluding) == 0 or len(crossing) == 0)
    if len(avail) < int(ANALYSIS_CHOICE_POLICY['min_specs']):
        status = 'insufficient_specs'
    elif not nonzero:
        status = 'zero_effect'
    elif not direction_stable:
        status = 'direction_unstable'
    elif precision_concordant is False:
        status = 'precision_sensitive'
    else:
        status = 'stable'
    ppl_points = [r['point'] for r in avail
                  if r['scale'] == 'ppl_delta' and r['point'] is not None]
    effect_range = None
    if ppl_points:
        _lo, _hi = min(ppl_points), max(ppl_points)
        _span = float(_hi - _lo)
        effect_range = {
            'min': float(_lo), 'max': float(_hi), 'span': _span,
            'relative_span': (float(_span / abs(p['point']))
                              if (p['scale'] == 'ppl_delta'
                                  and abs(p['point']) > 0.0) else None),
            'n': len(ppl_points),
        }
    claim = {
        'stable': ('all non-zero specifications agree on the primary direction '
                   'and, where intervals exist, agree on whether zero is '
                   'excluded'),
        'precision_sensitive': ('directions agree but the zero-exclusion '
                                'conclusion depends on the analysis choice'),
        'direction_unstable': 'available specifications disagree on the direction',
        'zero_effect': 'every available specification has a point estimate of zero',
        'insufficient_specs': ('fewer than two analysis specifications are '
                               'available'),
    }[status]
    return {
        'kind': 'analysis_choice_concordance',
        'version': ANALYSIS_CHOICE_POLICY['version'],
        'status': status,
        'claim': claim,
        'primary_spec': p['spec'],
        'primary_sign': primary_sign,
        'primary_excludes_zero': p['excludes_zero'],
        'specifications': rows,
        'n_specs': len(rows),
        'n_available': len(avail),
        'n_unavailable': len(rows) - len(avail),
        'n_nonzero': len(nonzero),
        'n_positive': pos,
        'n_negative': neg,
        'n_zero': zero,
        'n_same_as_primary': same_as_primary,
        'n_opposite_to_primary': opposite_to_primary,
        'direction_stable': direction_stable,
        'n_with_ci': len(ci_rows),
        'n_excluding_zero': len(excluding),
        'n_crossing_zero': len(crossing),
        'precision_concordant': precision_concordant,
        'excluding_zero_specs': [r['spec'] for r in excluding],
        'crossing_zero_specs': [r['spec'] for r in crossing],
        'effect_range_ppl': effect_range,
        'fragile': status in ('precision_sensitive', 'direction_unstable'),
        'note': ANALYSIS_CHOICE_POLICY['note'],
    }


# ---- 结果稳健性：脆弱性指数与「丢掉一个种子」的族级判定重算 ------------------
# 统计显著不等于结论稳健：n 个种子里如果有一个特别大或方向相反，删掉它
# 可能就翻转方向或让 Holm 门失守。下面的函数**只做 post-hoc 诊断**，不
# 重算、也不改写冻结判定；它们回答的是「这个判定有多依赖个别种子」。
# 与 `paired_effect_stats.seed_driven`（只看均值符号）相比，这里重算的是
# 检验本身与整个 Holm 族，因此能发现「均值不翻、但校正后不再显著」的情形。

def _paired_payload(deltas, seeds=None, *, where='paired analysis'):
    """Normalise a paired-delta payload to ``(d, seed_labels, kept_positions,
    n_input)``.  Non-finite deltas are dropped, and their seed labels are
    dropped with them, so a later leave-one-seed-out pass cannot silently
    pair a value with the wrong seed."""
    if isinstance(deltas, (str, bytes)):
        raise ValueError(f'{where}: deltas must be an iterable of numbers, not a string')
    try:
        raw = list(deltas if deltas is not None else [])
    except TypeError:
        raise ValueError(f'{where}: deltas must be an iterable') from None
    if seeds is not None:
        if isinstance(seeds, (str, bytes)):
            raise ValueError(f'{where}: seeds must be an iterable of integers, not a string')
        try:
            sl = list(seeds)
        except TypeError:
            raise ValueError(f'{where}: seeds must be an iterable') from None
        if len(sl) != len(raw):
            raise ValueError(f'{where}: seeds length {len(sl)} does not match deltas length {len(raw)}')
        norm = [_as_seed_int(s) for s in sl]
        if any(s is None for s in norm):
            raise ValueError(f'{where}: seeds must be integers ({sl!r})')
        if len(set(norm)) != len(norm):
            raise ValueError(f'{where}: duplicate seeds {norm!r}')
    else:
        norm = list(range(len(raw)))
    d, labels, keep = [], [], []
    for i, x in enumerate(raw):
        f = _finite_scalar(x)
        if f is None:
            continue
        d.append(f)
        labels.append(norm[i])
        keep.append(i)
    return np.asarray(d, dtype=float), labels, keep, len(raw)


def paired_fragility(deltas, *, alpha=0.05, seeds=None):
    """Post-hoc fragility readout for one paired sign-flip contrast.

    Reports how much of the significance rests on individual seeds:

    ``drop_fragility``
        The largest k such that repeatedly removing the *least influential*
        remaining seed (smallest |delta|) still leaves the exact sign-flip
        p < alpha.  ``0`` means removing even the smallest seed already
        overturns the raw test; a large k means the verdict does not need any
        single seed.
    ``flip_fragility``
        The number of same-direction seeds (smallest |delta| first) whose sign
        must be flipped before the test stops rejecting OR the mean changes
        direction.  This is a greedy influence bound, not a claim that no
        smaller set of flips could do it; the definition is reported with the
        number so the readout cannot be mistaken for an exact minimum.
    ``loo``
        Per-seed leave-one-out mean/p.  ``removes_significance`` marks the
        seeds whose removal alone lifts p to/above alpha; ``flips_direction``
        marks those whose removal reverses the mean sign.

    None of these changes the p-value, the Holm family or a frozen decision.
    The exact sign-flip test is reused on every subset, so the readout is
    deterministic and carries the same ``method``/``p_floor`` contract.
    """
    alpha = _validated_alpha(alpha, context='paired_fragility alpha')
    d, labels, _keep, n_input = _paired_payload(deltas, seeds, where='paired_fragility')
    n = int(d.size)
    n_dropped = n_input - n
    out = {'kind': 'paired_fragility', 'unit': 'seed', 'alpha': alpha,
           'n': n, 'n_input': n_input, 'n_dropped': n_dropped,
           'seeds': labels, 'per_seed': [float(x) for x in d.tolist()],
           'mean': float('nan'), 'std': float('nan'), 'sem': float('nan'),
           'ci95_low': float('nan'), 'ci95_high': float('nan'),
           'ci_excludes_zero': None,
           'p': float('nan'), 'p_floor': float('nan'), 'p_signal': False,
           'n_nonzero': 0, 'sign_counts': {'positive': 0, 'negative': 0, 'zero': 0},
           'direction': None, 'same_sign_count': 0, 'same_sign_fraction': None,
           'drop_fragility': None, 'flip_fragility': None,
           'flip_stop_reason': None,
           'loo': [], 'most_influential_seed': None,
           'most_influential_p_increase': None, 'most_influential_p_shift': None,
           'most_influential_tied_count': None,
           'loo_p_min': None, 'loo_p_max': None,
           'method': 'none',
           'definition': ('drop_fragility = least-influential seeds removable while the exact '
                          'sign-flip p stays < alpha; flip_fragility = smallest-|delta| '
                          'same-direction signs that must flip before p >= alpha or the mean '
                          'direction reverses (greedy, not an exact minimum).'),
           'note': ('post-hoc influence diagnostic only; it does not change the reported p, '
                    'the Holm family, or any frozen decision')}
    if n == 0:
        return out
    _ms = _scaled_mean_std_if_extreme(d, ddof=1) if n > 1 else None
    if _ms is not None:
        mean, std = _ms
    else:
        mean = float(d.mean())
        std = float(d.std(ddof=1)) if n > 1 else float('nan')
    sem = float(std / math.sqrt(n)) if (n > 1 and math.isfinite(std)) else float('nan')
    ci_lo = ci_hi = float('nan')
    if n > 1 and math.isfinite(sem):
        tc = _t_critical(0.95, n - 1)
        if math.isfinite(tc):
            ci_lo, ci_hi = mean - tc * sem, mean + tc * sem
    sf = exact_signflip_p(d)
    p = float(sf['p'])
    n_nonzero = int(sf.get('n_nonzero') or 0)
    pos = int((d > 0).sum())
    neg = int((d < 0).sum())
    zero = int((d == 0).sum())
    if mean > 0:
        direction = 'positive'
    elif mean < 0:
        direction = 'negative'
    else:
        direction = None
    same = pos if direction == 'positive' else (neg if direction == 'negative' else max(pos, neg))
    out.update({'mean': mean, 'std': std, 'sem': sem,
                'ci95_low': float(ci_lo), 'ci95_high': float(ci_hi),
                'ci_excludes_zero': (None if not (math.isfinite(ci_lo) and math.isfinite(ci_hi))
                                     else bool(ci_lo > 0.0 or ci_hi < 0.0)),
                'p': p, 'p_floor': sf.get('p_floor'), 'p_signal': bool(p < alpha),
                'n_nonzero': n_nonzero,
                'sign_counts': {'positive': pos, 'negative': neg, 'zero': zero},
                'direction': direction, 'same_sign_count': int(same),
                'same_sign_fraction': (float(same / n) if n else None),
                'method': sf.get('method')})
    # ---- least-influential-seed deletion -----------------------------------
    if n >= 3:
        if not (p < alpha):
            out['drop_fragility'] = 0
        else:
            kept = np.ones(n, dtype=bool)
            k = 0
            while int(kept.sum()) > 2:
                _abs = np.where(kept, np.abs(d), np.inf)
                idx = int(np.argmin(_abs))
                trial = kept.copy()
                trial[idx] = False
                if float(exact_signflip_p(d[trial])['p']) < alpha:
                    kept = trial
                    k += 1
                else:
                    break
            out['drop_fragility'] = int(k)
    # ---- greedy sign-flip influence ----------------------------------------
    if n >= 1:
        if (not (p < alpha)) or direction is None:
            out['flip_fragility'] = 0
            out['flip_stop_reason'] = ('not significant at the full panel'
                                       if not (p < alpha) else 'no mean direction')
        else:
            _sign = 1.0 if direction == 'positive' else -1.0
            same_idx = [i for i in range(n) if d[i] * _sign > 0]
            same_idx.sort(key=lambda i: (abs(float(d[i])), i))
            cur = d.copy()
            k = 0
            reason = 'direction reversed'
            while k < len(same_idx):
                i = same_idx[k]
                cur[i] = -cur[i]
                k += 1
                p_new = float(exact_signflip_p(cur)['p'])
                _ms_new = _scaled_mean_std_if_extreme(cur, ddof=1)
                m_new = _ms_new[0] if _ms_new is not None else float(cur.mean())
                if p_new >= alpha:
                    reason = 'p>=alpha'
                    break
                if (m_new > 0) != (direction == 'positive') or m_new == 0.0:
                    reason = 'direction reversed'
                    break
            out['flip_fragility'] = int(k)
            out['flip_stop_reason'] = reason
    # ---- per-seed leave-one-out -------------------------------------------
    if n >= 2:
        loo = []
        for i in range(n):
            sub = np.delete(d, i)
            sf_i = exact_signflip_p(sub)
            p_i = float(sf_i['p'])
            _ms_i = _scaled_mean_std_if_extreme(sub, ddof=1)
            m_i = _ms_i[0] if _ms_i is not None else float(sub.mean())
            loo.append({'seed': labels[i], 'n': int(sub.size), 'mean': m_i,
                        'p': p_i,
                        'mean_shift': float(m_i - mean),
                        'p_increase': float(p_i - p) if math.isfinite(p_i) else None,
                        'p_shift': (abs(float(p_i) - p) if math.isfinite(p_i) else None),
                        'removes_significance': bool(math.isfinite(p_i) and p < alpha and p_i >= alpha),
                        'flips_direction': bool(mean != 0.0 and (m_i == 0.0 or (m_i > 0) != (mean > 0)))})
        out['loo'] = loo
        # Influence is the ABSOLUTE change in the test, not only an increase:
        # removing a contrary seed can turn a null panel significant, which is
        # just as much a single-seed dependence as dropping a supportive one.
        _rank = [(v.get('p_shift') if isinstance(v.get('p_shift'), float) else float('-inf'),
                  abs(v.get('mean_shift') or 0.0), -i, v) for i, v in enumerate(loo)]
        _best = max(_rank, key=lambda t: (t[0], t[1], t[2]))[3]
        out['most_influential_seed'] = _best['seed']
        out['most_influential_p_increase'] = _best['p_increase']
        out['most_influential_p_shift'] = _best['p_shift']
        # With equal-magnitude deltas every seed ties; naming one of them as
        # "the" driver would be an artefact of the tie-break, so count the tie.
        _top_key = (out['most_influential_p_shift'], abs(_best['mean_shift']))
        out['most_influential_tied_count'] = int(sum(
            1 for v in loo
            if v.get('p_shift') == _top_key[0]
            and abs(float(v.get('mean_shift') or 0.0)) == _top_key[1]))
        _ps = [v['p'] for v in loo if math.isfinite(v['p'])]
        out['loo_p_min'] = float(min(_ps)) if _ps else None
        out['loo_p_max'] = float(max(_ps)) if _ps else None
    return out


def holm_decision_jackknife(named_paired, *, alpha=0.05, family_names=None):
    """Recompute the whole Holm family after dropping one seed at a time.

    ``named_paired`` maps a comparison name to either a delta iterable or a
    ``{'deltas'/'per_seed': [...], 'seeds': [...]}`` mapping.  For every seed
    in the union, the affected comparisons are re-tested on the remaining
    pairs (comparisons that do not contain that seed keep their full-panel p),
    and the family is re-corrected.  This answers the question a single
    comparison's p cannot: does one seed's presence/absence change *which*
    comparisons survive Holm?

    The returned ``rejected`` sets use the same strict ``p_holm < alpha``
    predicate as the frozen decision rule (``holm_adjust.reject`` uses
    ``<=``; the caller's verdict is what is replayed here).  This is a
    post-hoc stability readout, not a second primary analysis: it never
    replaces the full-panel p or the frozen decision.
    """
    alpha = _validated_alpha(alpha, context='holm_decision_jackknife alpha')
    if not isinstance(named_paired, dict):
        raise ValueError('holm_decision_jackknife: named_paired must be a mapping of name -> deltas')
    payload = {}
    for name, val in named_paired.items():
        nm = str(name)
        if not nm:
            raise ValueError('holm_decision_jackknife: comparison names must be non-empty strings')
        if isinstance(val, dict):
            deltas = val.get('deltas', val.get('per_seed'))
            seeds = val.get('seeds')
        else:
            deltas = val
            seeds = None
        d, labels, _keep, _n_in = _paired_payload(deltas, seeds, where=f'holm_decision_jackknife[{nm}]')
        payload[nm] = (d, labels)
    # A declared comparison with no computable delta still occupies a family
    # slot (p=1).  Without this, an omitted arm would silently shrink m and
    # make the remaining Holm corrections anti-conservative in the jackknife.
    if family_names is not None:
        if isinstance(family_names, (str, bytes)):
            raise ValueError('holm_decision_jackknife: family_names must be an iterable of names, not a string')
        try:
            fam_in = list(family_names)
        except TypeError:
            raise ValueError('holm_decision_jackknife: family_names must be an iterable') from None
        for nm in fam_in:
            snm = str(nm)
            if not snm:
                raise ValueError('holm_decision_jackknife: family_names must be non-empty')
            payload.setdefault(snm, (np.asarray([], dtype=float), []))
    if not payload:
        return {'kind': 'holm_decision_jackknife', 'alpha': alpha,
                'n_comparisons': 0, 'n_folds': 0, 'rejected_full': [],
                'folds': [], 'always_rejected': [], 'sometimes_rejected': [],
                'never_rejected': [], 'all_rejections_stable': None,
                'unstable_folds': [], 'per_comparison': {},
                'note': 'no comparisons supplied'}
    names = sorted(payload)
    full_p = {nm: (1.0 if payload[nm][0].size == 0
                   else float(exact_signflip_p(payload[nm][0])['p'])) for nm in names}
    fam_full = holm_adjust([(nm, full_p[nm]) for nm in names], alpha=alpha)
    rejected_full = sorted(nm for nm in names
                           if fam_full.get(nm, {}).get('p_holm') is not None
                           and float(fam_full[nm]['p_holm']) < alpha)
    all_seeds = sorted({s for _d, labs in payload.values() for s in labs})
    folds = []
    for s in all_seeds:
        raw = {}
        n_pairs = {}
        for nm in names:
            d, labs = payload[nm]
            if d.size == 0:
                n_pairs[nm] = 0
                raw[nm] = 1.0
            elif s in labs:
                mask = np.asarray([lab != s for lab in labs], dtype=bool)
                sub = d[mask]
                n_pairs[nm] = int(sub.size)
                raw[nm] = (float(exact_signflip_p(sub)['p']) if sub.size else 1.0)
            else:
                n_pairs[nm] = int(d.size)
                raw[nm] = full_p[nm]
        fam = holm_adjust([(nm, raw[nm]) for nm in names], alpha=alpha)
        p_holm = {nm: (None if fam.get(nm, {}).get('p_holm') is None
                       else float(fam[nm]['p_holm'])) for nm in names}
        rejected = sorted(nm for nm in names
                          if p_holm.get(nm) is not None and p_holm[nm] < alpha)
        folds.append({'seed': int(s), 'n_pairs': n_pairs,
                      'raw_p': {nm: float(raw[nm]) for nm in names},
                      'p_holm': p_holm, 'rejected': rejected,
                      'lost': sorted(set(rejected_full) - set(rejected)),
                      'gained': sorted(set(rejected) - set(rejected_full))})
    _sets = [set(f['rejected']) for f in folds]
    always = sorted(set.intersection(*_sets)) if _sets else []
    sometimes = sorted(set.union(*_sets) - set(always)) if _sets else []
    never = sorted(set(names) - set.union(*_sets)) if _sets else list(names)
    per_comparison = {}
    for nm in names:
        vals = [f['p_holm'][nm] for f in folds
                if f['p_holm'].get(nm) is not None]
        per_comparison[nm] = {
            'n_folds': len(folds),
            'n_folds_present': (sum(1 for f in folds
                                    if f['n_pairs'].get(nm) == len(payload[nm][1]) - 1)
                                if len(payload[nm][1]) > 0 else 0),
            'reject_folds': [f['seed'] for f in folds if nm in f['rejected']],
            'lost_folds': [f['seed'] for f in folds if nm in f['lost']],
            'gained_folds': [f['seed'] for f in folds if nm in f['gained']],
            'p_holm_min': (float(min(vals)) if vals else None),
            'p_holm_max': (float(max(vals)) if vals else None)}
    unstable = [f['seed'] for f in folds if f['rejected'] != rejected_full]
    return {'kind': 'holm_decision_jackknife', 'alpha': alpha,
            'n_comparisons': len(names), 'n_folds': len(folds),
            'rejected_full': rejected_full,
            'folds': folds,
            'always_rejected': always, 'sometimes_rejected': sometimes,
            'never_rejected': never,
            'all_rejections_stable': (None if not folds else (not unstable)),
            'unstable_folds': unstable,
            'per_comparison': per_comparison,
            'definition': ('each fold drops one seed from every comparison that contains it, '
                           're-runs the exact sign-flip test and the full Holm correction; '
                           '`rejected` replays the frozen strict p_holm < alpha rule'),
            'note': ('post-hoc sensitivity readout, not a second primary analysis; the '
                     'full-panel p/Holm/decision remain the reported result')}


def confirmation_robustness(named_paired, *, alpha=0.05, family_size=None, confirmed=None,
                            family_names=None):
    """Compose per-contrast fragility and a family-level seed-drop jackknife.

    This is the results-analysis companion of a frozen confirmation: it takes
    the SAME per-seed deltas that produced the primary p-values and reports
    how many seeds the verdict can lose before it changes.  It deliberately
    does not re-derive, re-select or re-label the decision.

    ``family_size`` must be the DECLARED family size when the caller knows it
    (e.g. the frozen manifest's primary-comparison count), so a comparison
    that became uncomputable cannot shrink the multiplicity correction.
    """
    alpha = _validated_alpha(alpha, context='confirmation_robustness alpha')
    if not isinstance(named_paired, dict):
        raise ValueError('confirmation_robustness: named_paired must be a mapping name -> deltas')
    per = {}
    for name, val in named_paired.items():
        if isinstance(val, dict):
            deltas = val.get('deltas', val.get('per_seed'))
            seeds = val.get('seeds')
        else:
            deltas = val
            seeds = None
        per[str(name)] = paired_fragility(deltas, alpha=alpha, seeds=seeds)
    fam = holm_decision_jackknife(named_paired, alpha=alpha,
                                  family_names=family_names)
    _m = _as_seed_int(family_size)
    _declared = set(per)
    if family_names is not None:
        if isinstance(family_names, (str, bytes)):
            raise ValueError('confirmation_robustness: family_names must be an iterable of names, not a string')
        try:
            _declared |= {str(x) for x in family_names}
        except TypeError:
            raise ValueError('confirmation_robustness: family_names must be an iterable') from None
    m = _m if (_m is not None and _m > 0) else len(_declared)
    if m < len(_declared):
        raise ValueError(
            'confirmation_robustness: family_size=%r is smaller than the %d '
            'supplied/declared comparisons; shrinking the correction family '
            'would make the Holm margin anti-conservative'
            % (family_size, len(_declared)))
    for nm, rec in per.items():
        _nn = rec.get('n_nonzero')
        _n_use = _nn if (isinstance(_nn, int) and _nn > 0) else (0 if _nn == 0 else rec.get('n'))
        req = signflip_min_consistency(_n_use, m, alpha)
        rec['family_size'] = m
        rec['min_same_sign_for_family_gate'] = req
        rec['same_sign_margin'] = (None if req is None or rec.get('same_sign_count') is None
                                   else int(rec['same_sign_count'] - req))
        rec['raw_significant'] = bool(isinstance(rec.get('p'), float)
                                      and math.isfinite(rec['p'])
                                      and rec['p'] < alpha)
        rec['holm_rejected_full'] = nm in set(fam.get('rejected_full') or [])
    _sig = [nm for nm, rec in per.items() if rec.get('raw_significant')]
    _flips = [rec['flip_fragility'] for rec in per.values()
              if isinstance(rec.get('flip_fragility'), int)]
    _drops = [rec['drop_fragility'] for rec in per.values()
              if isinstance(rec.get('drop_fragility'), int)]
    out = {'kind': 'confirmation_robustness', 'alpha': alpha,
           'family_size': m,
           'per_comparison': per,
           'family_jackknife': fam,
           'confirmed_full': (sorted(str(x) for x in confirmed)
                              if confirmed is not None else None),
           'summary': {
               'n_comparisons': len(per),
               'n_declared_family_names': len(_declared),
               'n_raw_significant': len(_sig),
               'raw_significant': sorted(_sig),
               'n_holm_rejected_full': len(fam.get('rejected_full') or []),
               'n_always_rejected_across_folds': len(fam.get('always_rejected') or []),
               'all_holm_rejections_stable_under_one_seed_drop': fam.get('all_rejections_stable'),
               'unstable_seeds': list(fam.get('unstable_folds') or []),
               'min_drop_fragility': (int(min(_drops)) if _drops else None),
               'min_flip_fragility': (int(min(_flips)) if _flips else None)},
           'role': ('post-hoc robustness diagnostic for the frozen result; it never changes '
                    'the full-panel p, the Holm family or the confirmation verdict'),
           'note': ('a frozen decision can still be reported when it is fragile, but the '
                    'fragility must be reported with it: the diagnosis names the seeds whose '
                    'single removal would change the verdict instead of hiding them behind '
                    'an aggregate p-value')}
    return out

# ---- 预注册方向性主张的三向判定 ---------------------------------------------
# 一个双侧检验 + 预注册方向不是二值「确认/不确认」：Holm 显著且方向相反的
# 结果是对该方向性主张的反证，不是「没有证据」。下面的纯函数把冻结判定拆成
# supported / refuted / inconclusive / significant_nondirectional /
# not_testable，且反证必须通过与确认相同的有效性门（覆盖、参数、初始化、
# n 下限），避免拿一个不完整面板去宣布反证。

def directional_claim_verdicts(comparisons, *, names=None, alpha=0.05,
                               require_complete_coverage=False,
                               require_param_match=False,
                               default_min_seeds=None):
    """Three-way verdicts for pre-registered directional comparisons.

    ``comparisons`` maps a comparison name to its statistics record.  The
    relevant keys are ``p_holm``, ``expected_sign`` (``'positive'`` /
    ``'negative'`` / ``None``), ``direction_ok`` (bool or ``None``),
    ``n_effective``, ``min_seeds_for_confirmation``, ``coverage_ok``,
    ``param_matched``, ``param_match_complete``, ``same_init_core_fp`` and
    optionally ``mechanism_claim_allowed``.  A record with a non-empty
    ``omitted`` key is reported as ``not_testable``.

    A comparison is ``supported`` only when the Holm p is below alpha, the
    observed direction matches the pre-registered sign, and the same validity
    gates as the frozen confirmation hold.  It is ``refuted`` when the Holm p
    is below alpha, the direction is opposite to the pre-registered sign, and
    those validity gates hold.  A significant reverse result with a missing
    gate is *not* a refutation: it is ``inconclusive`` with an explicit
    ``reverse_significant_unverified`` diagnostic.  Missing/omitted results are
    ``not_testable``.  A significant comparison with ``expected_sign=None`` is
    ``significant_nondirectional`` and can never enter ``supported`` or
    ``refuted``.

    This function adds no test and no multiplicity correction: it partitions
    the SAME Holm rejections into the pre-registered directional outcomes, so
    refutation inherits the family-wise error control of the two-sided test.
    It is not an equivalence test: ``refuted`` means the directional
    superiority claim was contradicted, not that the opposite mechanism was
    proven or that the effect exceeds a SESOI.
    """
    alpha = _validated_alpha(alpha, context='directional_claim_verdicts alpha')
    if not isinstance(comparisons, dict):
        raise ValueError('directional_claim_verdicts: comparisons must be a mapping name -> record')
    if names is None:
        names = list(comparisons)
    else:
        if isinstance(names, (str, bytes)):
            raise ValueError('directional_claim_verdicts: names must be an iterable of names, not a string')
        try:
            names = [str(x) for x in names]
        except TypeError:
            raise ValueError('directional_claim_verdicts: names must be iterable') from None
    if len(set(names)) != len(names):
        raise ValueError('directional_claim_verdicts: names contain duplicates')
    _dmin = _as_seed_int(default_min_seeds)
    if default_min_seeds is not None and (_dmin is None or _dmin < 0):
        raise ValueError(
            'directional_claim_verdicts: default_min_seeds must be a '
            f'non-negative integer or None, got {default_min_seeds!r}')
    _req_cov = bool(require_complete_coverage)
    _req_param = bool(require_param_match)
    verdicts = {}
    buckets = {'supported': [], 'refuted': [], 'inconclusive': [],
               'significant_nondirectional': [], 'not_testable': [],
               'opposite_nonsignificant': [],
               'reverse_significant_unverified': []}
    for name in names:
        rec = comparisons.get(name)
        out = {'name': name, 'verdict': 'not_testable', 'reason': '',
               'expected_sign': None, 'direction_ok': None,
               'p_holm': None, 'holm_significant': False,
               'validity_ok': False, 'gate_failures': [],
               'n_effective': None, 'min_seeds': None,
               'mechanism_claim_allowed': None, 'claim_scope': 'unspecified'}
        if not isinstance(rec, dict):
            out['reason'] = 'not_reported'
            buckets['not_testable'].append(name)
            verdicts[name] = out
            continue
        out['mechanism_claim_allowed'] = rec.get('mechanism_claim_allowed')
        if rec.get('omitted'):
            out['reason'] = 'omitted: ' + str(rec.get('omitted'))
            buckets['not_testable'].append(name)
            verdicts[name] = out
            continue
        sign_raw = rec.get('expected_sign')
        if sign_raw is None:
            sign = None
        elif isinstance(sign_raw, str) and sign_raw.strip().lower() in ('positive', 'negative'):
            sign = sign_raw.strip().lower()
        else:
            raise ValueError(
                f'directional_claim_verdicts[{name!r}]: expected_sign must be '
                f"'positive', 'negative' or None, got {sign_raw!r}")
        out['expected_sign'] = sign
        dok_raw = rec.get('direction_ok')
        if dok_raw is None:
            dok = None
        elif isinstance(dok_raw, (bool, np.bool_)):
            dok = bool(dok_raw)
        else:
            raise ValueError(
                f'directional_claim_verdicts[{name!r}]: direction_ok must be '
                f'a bool or None, got {dok_raw!r}')
        out['direction_ok'] = dok
        ph_raw = rec.get('p_holm')
        if ph_raw is None:
            out['reason'] = 'no_holm_p'
            buckets['not_testable'].append(name)
            verdicts[name] = out
            continue
        if isinstance(ph_raw, (bool, np.bool_)):
            raise ValueError(
                f'directional_claim_verdicts[{name!r}]: p_holm is a boolean, not a p-value')
        try:
            ph = float(ph_raw)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(
                f'directional_claim_verdicts[{name!r}]: p_holm is not numeric: {ph_raw!r}') from None
        if not math.isfinite(ph) or not (0.0 <= ph <= 1.0):
            raise ValueError(
                f'directional_claim_verdicts[{name!r}]: p_holm must be a finite '
                f'probability in [0,1], got {ph_raw!r}')
        out['p_holm'] = ph
        out['holm_significant'] = bool(ph < alpha)
        # Present-but-malformed floor fields are corrupt, not absent.  Falling
        # back to the default (or accepting a negative floor) would let a
        # record that cannot attest its own n floor support a claim.
        _raw_n_eff = rec.get('n_effective')
        n_eff = _as_seed_int(_raw_n_eff)
        if _raw_n_eff is not None and (n_eff is None or n_eff < 0):
            n_eff = None
        _raw_min_n = rec.get('min_seeds_for_confirmation')
        min_n = _as_seed_int(_raw_min_n)
        _min_n_invalid = False
        if _raw_min_n is None:
            min_n = _dmin
        elif min_n is None or min_n < 0:
            _min_n_invalid = True
            min_n = None
        out['n_effective'] = (n_eff if n_eff is not None else 0)
        out['min_seeds'] = min_n
        failures = []
        if n_eff is None:
            failures.append('n_effective_unverifiable')
        if _min_n_invalid:
            failures.append(
                f'min_seeds_for_confirmation={_raw_min_n!r} is not a '
                'non-negative integer; the n floor is unverifiable')
        if min_n is not None and (n_eff is None or n_eff < min_n):
            failures.append(
                f'n_effective={out["n_effective"]} < min_seeds_for_confirmation={min_n}')
        if _req_cov and rec.get('coverage_ok') is not True:
            failures.append('coverage_incomplete')
        # A definite parameter mismatch always invalidates a paired capacity
        # claim.  A disclosed parameter-matching contract additionally requires
        # the COMPLETE per-seed record; `None` is still unverified, never a pass.
        if rec.get('param_matched') is False:
            failures.append('param_mismatch')
        elif _req_param and rec.get('param_match_complete') is not True:
            failures.append('param_match_incomplete')
        if rec.get('same_init_core_fp') is False:
            failures.append('init_core_fp_mismatch')
        out['gate_failures'] = failures
        out['validity_ok'] = not failures
        if sign is None:
            out['claim_scope'] = 'nondirectional'
        elif rec.get('mechanism_claim_allowed') is True:
            out['claim_scope'] = 'mechanism'
        elif rec.get('mechanism_claim_allowed') is False:
            out['claim_scope'] = 'system_level'
        if not out['holm_significant']:
            out['verdict'] = 'inconclusive'
            out['reason'] = 'not_holm_significant'
            if dok is False and sign is not None:
                buckets['opposite_nonsignificant'].append(name)
        elif sign is None:
            # Preserve the frozen field's semantics: this is a Holm-significant
            # non-directional readout.  It can never be confirmatory.
            buckets['significant_nondirectional'].append(name)
            if out['validity_ok']:
                out['verdict'] = 'significant_nondirectional'
                out['reason'] = 'holm_significant_without_a_directional_hypothesis'
            else:
                out['verdict'] = 'inconclusive'
                out['reason'] = 'holm_significant_but_validity_gates_failed'
        elif failures:
            out['verdict'] = 'inconclusive'
            out['reason'] = 'holm_significant_but_validity_gates_failed'
            if dok is False:
                out['direction_conflict'] = True
                buckets['reverse_significant_unverified'].append(name)
        elif dok is True:
            out['verdict'] = 'supported'
            out['reason'] = 'holm_significant_and_direction_matches'
            buckets['supported'].append(name)
        elif dok is False:
            out['verdict'] = 'refuted'
            out['reason'] = 'holm_significant_and_direction_opposite'
            buckets['refuted'].append(name)
        else:
            out['verdict'] = 'inconclusive'
            out['reason'] = 'direction_unverifiable'
        if out['verdict'] == 'inconclusive' and name not in buckets['inconclusive']:
            buckets['inconclusive'].append(name)
        verdicts[name] = out
    out = {'alpha': alpha, 'names': names, 'verdicts': verdicts,
           'supported': buckets['supported'], 'refuted': buckets['refuted'],
           'inconclusive': buckets['inconclusive'],
           'significant_nondirectional': buckets['significant_nondirectional'],
           'not_testable': buckets['not_testable'],
           'opposite_nonsignificant': buckets['opposite_nonsignificant'],
           'reverse_significant_unverified': buckets['reverse_significant_unverified'],
           'counts': {k: len(v) for k, v in buckets.items()},
           'rule': ('Holm-significant + pre-registered direction + validity gates = '
                    'supported; Holm-significant + opposite direction + validity gates = '
                    'refuted; everything else is inconclusive or not_testable. '
                    'Refutation is a directional claim conflict, not equivalence and '
                    'not proof of the opposite mechanism.'),
           'role': ('reports the pre-registered three-way decision on the SAME '
                    'Holm family; it adds no test, no multiplicity correction and '
                    'does not change any p-value')}
    return out


# ---- 预注册功效 / 分辨率工具 --------------------------------------------------
# 动机：一个冻结方案如果「无论数据多强都不可能达到 p_holm < α」，它落在
# 分辨率上限而不是「保守」。下面的函数把这件事变成加载清单时就能机器判定
# 的硬门槛，而不是统计阶段的一句「保守」警告。

def holm_min_attainable_p(n_seeds, m_comparisons, alpha=0.05):
    """双侧精确符号翻转 + Holm 校正族的最小可达 Holm p。

    最小原始 p = 2/2^n（全部种子同向）；Holm 第一道门乘以族大小 m，故
    最小可达 p_holm = min(1, m * 2/2^n)。若该值 > alpha，则该方案在构造上
    无法确认任何比较——这是分辨率上限，不是「保守的选择」，应当在读到
    任何测试数据之前修订（加种子 / 缩族 / 改检验），而不是事后再解释。"""
    # The threshold is part of the public planning API even though the floor
    # itself is alpha-independent; validate it so an invalid gate cannot be
    # echoed next to a "sufficient plan" verdict.
    _validated_alpha(alpha, context='holm_min_attainable_p alpha')
    n = _as_seed_int(n_seeds)
    m = _as_seed_int(m_comparisons)
    if n is None or m is None or n <= 0 or m <= 0:
        return float('nan')
    floor = signflip_p_floor(n)
    if floor > 0.0:
        try:
            return min(1.0, m * floor)
        except OverflowError:
            return 1.0
    # 2**-n underflowed.  A large enough family can still have
    # m * 2**-n above alpha (or even above 1), so do not report 0.
    log_p = math.log(m) + math.log(2.0) - _int_times_log2(n)
    if log_p >= 0.0:
        return 1.0
    if log_p < -745.0:
        return 0.0
    return math.exp(log_p)


def holm_min_attainable_p_floors(p_floors, alpha=0.05):
    """Minimum attainable Holm p from per-comparison exact-test floors.

    ``holm_min_attainable_p`` assumes every comparison has the same all-nonzero
    floor ``2/2**n``.  In practice ties reduce a comparison's effective n, so
    its exact p cannot go below ``2/2**(#nonzero)``.  Given each comparison's
    own floor, the smallest attainable family-adjusted p is ``m * min(floor)``
    (Holm's first step).  Missing/invalid floors are treated as 1.0 so a
    malformed comparison cannot make the family look more resolvable.
    """
    _validated_alpha(alpha, context='holm_min_attainable_p_floors alpha')
    try:
        vals = list(p_floors)
    except TypeError:
        return float('nan')
    if not vals:
        return float('nan')
    floors = []
    for v in vals:
        f = _finite_scalar(v)
        if f is None or not (0.0 <= f <= 1.0):
            f = 1.0
        floors.append(float(f))
    return min(1.0, len(floors) * min(floors))


def signflip_min_n_for_holm(m_comparisons, alpha=0.05, n_max=64):
    """Smallest seed count n for which a Holm-corrected exact sign-flip test
    can reach p_holm <= alpha EVEN IF every seed moves in the predicted
    direction.  Returns None when no n <= n_max reaches it.

    This is the planning counterpart of `holm_min_attainable_p`: it turns
    "3-5 seeds" into an explicit statement of what those seeds can and cannot
    establish.  At alpha=0.05 a single comparison needs n>=6; a family of m
    comparisons needs roughly 2**n >= 2*m/alpha.  A panel scheduled below the
    returned n is descriptive only (the exact-test floor, not the data, bounds
    the p value), so the audit labels it resolution-limited instead of reading
    it as a null result."""
    m = _as_seed_int(m_comparisons)
    n_max = _as_seed_int(n_max)
    a = _validated_alpha(alpha, context='signflip_min_n_for_holm alpha')
    if m is None or n_max is None or m <= 0 or n_max < 1:
        return None
    if holm_min_attainable_p(n_max, m, a) > a + 1e-15:
        return None
    lo, hi = 1, n_max
    while lo < hi:
        mid = (lo + hi) // 2
        if holm_min_attainable_p(mid, m, a) <= a + 1e-15:
            hi = mid
        else:
            lo = mid + 1
    return lo


def global_multiplicity_audit(named_pvals, *, alpha=0.05,
                              families=None, family_holm=None):
    """Cross-PANEL multiplicity view over every comparison a report displays.

    `attach_holm` / `holm_adjust` control the error rate INSIDE one panel's
    family, but a report or a paper quotes many families, and "it was
    significant in v7, v10 and v11" is itself a selection over every contrast
    the project has displayed.  This audit is the machine-computed global
    view over that larger family:

      * Benjamini-Hochberg step-up q-values (FDR-controlled across panels);
      * Bonferroni p*m (family-wise, the conservative bound);
      * a per-comparison `global_status` that report prose must respect:
          - 'globally_controlled'  : raw p < alpha AND BH-rejected AND
                                     Bonferroni < alpha
          - 'fdr_only'             : BH-rejected but Bonferroni >= alpha
                                     (quote as FDR-controlled, never as
                                     family-wise certain)
          - 'family_only'          : raw p < alpha but not BH-rejected —
                                     prose may not say "significant"
                                     without the "not globally corrected"
                                     qualifier
          - 'not_significant'      : raw p >= alpha
          - 'untestable'           : no valid p (counted in m with a p=1
                                     slot, so a report cannot shrink the
                                     global family by dropping a failed
                                     comparison — the same fail-closed
                                     convention as `holm_adjust`)

    `named_pvals` is [(name, p), ...] or [(name, p, family), ...]; `families`
    may carry one family label per entry (same order) and `family_holm` the
    within-family Holm p per entry (used only for the display column, never
    for the global decision).  Duplicate names are a hard error: two rows for
    one comparison would double-count it in m.

    The audit never forbids REPORTING a number; it only separates the wordings
    a cross-panel quote is allowed to use, exactly like `reportable_claim`."""
    if isinstance(alpha, (bool, np.bool_)):
        raise ValueError(f'alpha must be a real number in (0,1), got {alpha!r}')
    try:
        a = float(alpha)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'alpha must be a real number in (0,1), got {alpha!r}')
    if not (0.0 < a < 1.0):
        raise ValueError(f'alpha must be in (0,1), got {alpha!r}')
    entries = []
    seen = set()
    for i, item in enumerate(named_pvals if named_pvals is not None else []):
        if not isinstance(item, (tuple, list)) or not 2 <= len(item) <= 3:
            raise ValueError(f'global_multiplicity_audit: entry #{i} must be (name, p) or (name, p, family), got {item!r}')
        name = str(item[0])
        if name in seen:
            raise ValueError(f'global_multiplicity_audit: duplicate comparison name {name!r} — two rows for one contrast would double-count it in m')
        seen.add(name)
        fam = item[2] if len(item) == 3 else None
        entries.append({'name': name, 'p': item[1], 'family': (str(fam) if fam is not None else None)})
    if families is not None:
        fams = list(families)
        if len(fams) != len(entries):
            raise ValueError(f'global_multiplicity_audit: families length {len(fams)} does not match {len(entries)} comparisons')
        for e, f in zip(entries, fams):
            if f is not None:
                e['family'] = str(f)
    if family_holm is not None:
        fh = list(family_holm)
        if len(fh) != len(entries):
            raise ValueError(f'global_multiplicity_audit: family_holm length {len(fh)} does not match {len(entries)} comparisons')
        for e, h in zip(entries, fh):
            e['p_holm_in_family'] = h
    m = len(entries)
    out = {'m': m, 'alpha': a, 'entries': [], 'untestable': [],
           'wording_rule': ('a comparison with global_status="family_only" may be '
                            'quoted as within-family significant but must carry the '
                            '"not globally corrected" qualifier; only '
                            '"globally_controlled" supports cross-panel significance '
                            'wording; "fdr_only" supports FDR-controlled wording, '
                            'not family-wise certainty')}
    if m == 0:
        out['summary'] = {'n_raw_significant': 0, 'n_bh_rejected': 0,
                          'n_bonferroni_significant': 0, 'n_family_only': 0,
                          'n_globally_controlled': 0, 'n_untestable': 0}
        return out
    pvals = []
    for e in entries:
        p = e['p']
        valid = False
        if p is not None and not isinstance(p, (bool, np.bool_)):
            try:
                pf = float(p)
                valid = math.isfinite(pf) and 0.0 <= pf <= 1.0
            except (TypeError, ValueError, OverflowError):
                valid = False
        if valid:
            e['p_raw'] = pf
        else:
            e['p_raw'] = None
            out['untestable'].append(e['name'])
        pvals.append(e['p_raw'] if e['p_raw'] is not None else 1.0)
    parr = np.asarray(pvals, dtype=float)
    # Benjamini-Hochberg step-up q-values with the monotone min from the
    # largest rank down, so q is non-decreasing in p and q_i <= q_j for
    # p_i <= p_j (the standard BH q definition).
    order = np.argsort(parr, kind='stable')
    ranked = parr[order]
    q_ranked = ranked * m / np.arange(1, m + 1, dtype=float)
    for i in range(m - 2, -1, -1):
        if q_ranked[i] > q_ranked[i + 1]:
            q_ranked[i] = q_ranked[i + 1]
    np.minimum(q_ranked, 1.0, out=q_ranked)
    q = np.empty(m, dtype=float)
    q[order] = q_ranked
    # BH rejection set == {i : q_i <= alpha} (the defining property of BH
    # q-values; q_i >= p_i always, so tied p values can never receive
    # different verdicts and a p=1 slot is never rejected).
    bh_rejected = q <= a + 1e-15
    bonf = np.minimum(parr * m, 1.0)
    n_raw = n_bh = n_bonf = n_famonly = n_global = 0
    for i, e in enumerate(entries):
        p_raw = e['p_raw']
        rej_bh = bool(bh_rejected[i])
        qv = float(q[i])
        bv = float(bonf[i])
        if p_raw is None:
            status = 'untestable'
        elif p_raw >= a:
            status = 'not_significant'
        elif rej_bh and bv < a:
            status = 'globally_controlled'
        elif rej_bh:
            status = 'fdr_only'
        else:
            status = 'family_only'
        e.update({'q_bh': qv, 'p_bonferroni': bv, 'bh_rejected': rej_bh,
                  'global_status': status})
        if p_raw is not None:
            n_raw += int(p_raw < a)
        n_bh += int(rej_bh and p_raw is not None)
        n_bonf += int(p_raw is not None and bv < a)
        n_famonly += int(status == 'family_only')
        n_global += int(status == 'globally_controlled')
        out['entries'].append(e)
    out['summary'] = {'n_raw_significant': n_raw, 'n_bh_rejected': n_bh,
                      'n_bonferroni_significant': n_bonf,
                      'n_family_only': n_famonly,
                      'n_globally_controlled': n_global,
                      'n_untestable': len(out['untestable'])}
    return out


def _int_times_log2(n):
    if n.bit_length() <= 1023:
        return n * math.log(2.0)
    shift = max(0, n.bit_length() - 1000)
    hi, lo = divmod(n, 1 << shift)
    try:
        head = math.ldexp(float(hi) * math.log(2.0), shift)
    except OverflowError:
        return float('inf')
    return math.fsum((head, float(lo) * math.log(2.0)))


def _binom_normal_z_half(n, m):
    diff = 2 * m + 1 - n
    if diff == 0:
        return 0.0
    try:
        sd = math.sqrt(n)
    except OverflowError:
        root = math.isqrt(n)
        try:
            sd = float(root)
        except OverflowError:
            return None
    try:
        return diff / sd
    except OverflowError:
        return None


def _log_binom_normal_cdf_half(n, m):
    z = _binom_normal_z_half(n, m)
    if z is None:
        return float('-inf')
    if z >= 0.0:
        tail = 0.5 * math.erfc(z / math.sqrt(2.0))
        return float(math.log1p(-tail)) if tail < 1.0 else float('-inf')
    if z >= -37.0:
        cdf = 0.5 * math.erfc(-z / math.sqrt(2.0))
        return float(math.log(cdf)) if cdf > 0.0 else float('-inf')
    z2 = z * z
    if not math.isfinite(z2):
        return float('-inf')
    inv2 = 1.0 / z2
    series = 1.0 + inv2 * (-1.0 + inv2 * (3.0 + inv2 * (-15.0 + inv2 * (105.0 + inv2 * (-945.0)))))
    if series <= 0.0:
        return float('-inf')
    return float(-0.5 * z2 - math.log(-z) - 0.5 * math.log(2.0 * math.pi)
                 + math.log(series))


def _binom_cdf_half(n, m):
    """P(Bin(n, 1/2) <= m), evaluated in log space.

    The planning tools are called with growing seed counts at scale.  The
    direct form `2.0 * sum(comb(n, i)) / 2**n` overflows `float` once n is
    large (int-to-float conversion), even though the probability is perfectly
    representable.  This implementation uses lgamma log-sum-exp for the small
    tail and symmetry for the upper half, so it stays finite for any n."""
    n = int(n)
    m = int(m)
    if n <= 0:
        return 1.0 if m >= 0 else 0.0
    if m < 0:
        return 0.0
    if m >= n:
        return 1.0
    if m * 2 >= n:
        # P(X <= m) = 1 - P(X <= n-m-1) with n-m-1 < n/2.
        return float(1.0 - _binom_cdf_half(n, n - m - 1))
    # Regularized incomplete beta: P(Bin(n,1/2) <= m) = I_{1/2}(n-m, m+1).
    try:
        v = _betai(n - m, m + 1, 0.5)
    except (OverflowError, ValueError):
        v = float('nan')
    if math.isfinite(v):
        return float(min(1.0, max(0.0, v)))
    logp = _log_binom_cdf_half(n, m)
    if not math.isfinite(logp) or logp < -745.0:
        return 0.0
    return float(min(1.0, math.exp(logp)))
