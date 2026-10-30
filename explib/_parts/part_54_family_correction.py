"""Dependence-aware family correction readout for the frozen confirmation family.

The frozen decision uses Holm step-down over the declared primary comparisons.
Holm controls the family-wise error rate without assuming independence, but it
is deliberately conservative: it charges every comparison the full declared
family size even when the per-seed contrasts are strongly correlated (the
confirmatory comparisons are computed on the SAME seeds and often share the
same control arm).  A correction-method choice, not only the data, can then
determine which claims are confirmed.

This fragment adds one *descriptive* readout, never a second decision:

1. It rebuilds the family from the stored per-seed deltas and runs an exact
   Westfall-Young free step-down max-T on the sign-invariant score statistic
   ``|sum_i d_i| / sqrt(sum_i d_i^2)``.  The joint null flips one sign per seed
   and applies it to every comparison, so cross-comparison dependence is
   preserved.  The score has a fixed null scale, so contrasts with different
   seed SDs remain comparable (a raw-mean max-T would be dominated by the
   largest-SD contrast).  Unlike the sample-t version it is also defined for a
   constant non-zero effect, where the sample SD is zero.
2. It reports Holm on the SAME score statistic next to max-T, so the effect of
   the correction method is separated from the effect of changing the marginal
   statistic (frozen mean vs score).
3. It runs the section-92 studentised t twin when its sample-SD denominator is
   available.  That twin supplies the simultaneous interval and a family-level
   MDE; it is not used for the correction-method comparison.
4. It reports the dependence structure (per-seed correlation matrix,
   eigenvalues, effective number of tests) and the correction thresholds.

The frozen ``p_exact_signflip``, ``p_holm``, ``verified`` and ``confirmed``
values are read, never written.  A future manifest may pre-register max-T as
its decision rule; this block only shows what that rule would have concluded on
the same panel, and it is not that rule.
"""

import math as _fc_math

import numpy as _fc_np


# Upper bound on the number of (pattern row x contrast) cells materialised per
# enumeration chunk.  Bounding only the pattern count would let a larger
# declared family multiply the working set by m.
FAMILY_SIGNFLIP_CHUNK_TARGET = 1 << 18


FAMILY_CORRECTION_POLICY = {
    'version': 'confirmation-family-correction-3',
    'alpha': 0.05,
    'max_n': 20,
    'min_seeds': 2,
    'dependence_max_contrasts': 512,
    'method': ('exact sign-flip Westfall-Young free step-down max-T on the '
               'sign-invariant score statistic'),
    'statistic': '|sum_i d_i| / sqrt(sum_i d_i^2) (sign-invariant)',
    'holm_mde_level': 'two-sided Holm first-gate level alpha/m',
    'role': ('descriptive dependence-aware sensitivity for the declared '
             'confirmation family; it adds no p-value to the frozen Holm '
             'family and never changes p_exact_signflip, p_holm, verified or '
             'confirmed'),
    'boundary': ('the score statistic differs from the frozen mean statistic; '
                 'Holm-on-score is the same-statistic comparison that isolates '
                 'the correction method, and the frozen mean test remains the '
                 'confirmatory rule.  The joint null assumes seed-level sign '
                 'exchangeability.  A comparison that cannot supply the same '
                 'complete seed vector is not silently dropped: the block '
                 'becomes not_estimable rather than shrinking the family.  '
                 'Effective tests and the correlation matrix are descriptive '
                 'at n=12 and are not a factor model.  MDE is a t-style '
                 'half-width at the observed seed SD, not power.  High '
                 'dependence can make max-T reject more than Holm, but it '
                 'cannot create a new claim: the frozen Holm/confirmed set '
                 'remains authoritative.'),
}
FAMILY_CORRECTION_STATUSES = ('ok', 'degenerate', 'not_estimable')


def _fc_number(value, name):
    if isinstance(value, (bool, _fc_np.bool_, str, bytes, bytearray)):
        raise ValueError(f'{name} must be a real number, got {value!r}')
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'{name} must be a real number, got {value!r}') from None
    if not _fc_math.isfinite(out):
        raise ValueError(f'{name} must be finite, got {value!r}')
    return out


def _fc_int(value, name, *, minimum=None):
    if isinstance(value, (bool, _fc_np.bool_)) or not isinstance(
            value, (int, _fc_np.integer)):
        raise ValueError(f'{name} must be an integer, got {value!r}')
    out = int(value)
    if minimum is not None and out < minimum:
        raise ValueError(f'{name} must be >= {minimum}, got {out}')
    return out


def _fc_alpha(value):
    out = _fc_number(value, 'alpha')
    if not (0.0 < out < 1.0):
        raise ValueError(f'alpha must be in (0,1), got {out!r}')
    return out


def _fc_not_estimable(reason, *, extra=None):
    out = {
        'kind': 'confirmation_family_correction',
        'version': FAMILY_CORRECTION_POLICY['version'],
        'status': 'not_estimable',
        'reason': str(reason),
        'method': FAMILY_CORRECTION_POLICY['method'],
        'role': FAMILY_CORRECTION_POLICY['role'],
        'boundary': FAMILY_CORRECTION_POLICY['boundary'],
    }
    if isinstance(extra, dict):
        out.update(extra)
    return out


def _fc_score_maxt(matrix, alpha):
    """Exact free step-down max-T on the sign-invariant score statistic.

    ``/ sqrt(sum d_i**2)`` is invariant to a sign flip, so the studentised null
    is exact.  The sample SD is deliberately not used: it is not sign-invariant
    and it is zero for a constant non-zero effect.  A row with all-zero deltas
    has score 0 and can never be rejected; it still occupies its declared slot.
    """
    A = _fc_np.asarray(matrix, dtype=float)
    m, n = A.shape
    total = 1 << n
    row_scale = _fc_np.max(_fc_np.abs(A), axis=1)
    nz_row = row_scale > 0.0
    D = _fc_np.zeros_like(A)
    if _fc_np.any(nz_row):
        D[nz_row] = A[nz_row] / row_scale[nz_row, None]
    scale = _fc_np.sqrt((D * D).sum(axis=1))
    nz = scale > 0.0
    t = _fc_np.zeros(m, dtype=float)
    if _fc_np.any(nz):
        t[nz] = _fc_np.abs(D[nz].sum(axis=1)) / scale[nz]
    tol = 1e-12 * max(1.0, float(t.max()) if t.size else 1.0)
    order = _fc_np.asarray(sorted(range(m), key=lambda i: (-float(t[i]), i)),
                           dtype=_fc_np.int64)
    t_sorted = t[order]
    bitpos = _fc_np.arange(n, dtype=_fc_np.int64)[None, :]
    # Each chunk materialises a (patterns x m) matrix.  Sizing the chunk from
    # n alone is fine for the small frozen family, but a scale-up panel can
    # grow the number of declared contrasts: cap the cell count as well, so
    # memory stays O(target) instead of O(target * m).
    chunk = max(1, min(total,
                       max(1, FAMILY_SIGNFLIP_CHUNK_TARGET // max(n, 1)),
                       max(1, FAMILY_SIGNFLIP_CHUNK_TARGET // max(m, 1))))
    raw_counts = _fc_np.zeros(m, dtype=_fc_np.int64)
    single_counts = _fc_np.zeros(m, dtype=_fc_np.int64)
    step_counts = _fc_np.zeros(m, dtype=_fc_np.int64)
    max_all = _fc_np.empty(total, dtype=float)
    for off in range(0, total, chunk):
        bs = min(chunk, total - off)
        masks = _fc_np.arange(off, off + bs, dtype=_fc_np.int64)[:, None]
        signs = _fc_np.where((masks & (1 << bitpos)) != 0, 1.0, -1.0)
        sums = signs @ D.T
        t_star = _fc_np.zeros_like(sums)
        if _fc_np.any(nz):
            t_star[:, nz] = _fc_np.abs(sums[:, nz]) / scale[None, nz]
        fam_max = t_star.max(axis=1)
        max_all[off:off + bs] = fam_max
        raw_counts += (t_star >= t[None, :] - tol).sum(axis=0)
        single_counts += (fam_max[:, None] >= t[None, :] - tol).sum(axis=0)
        cum = _fc_np.maximum.accumulate(t_star[:, order][:, ::-1], axis=1)[:, ::-1]
        step_counts += (cum >= t_sorted[None, :] - tol).sum(axis=0)
    raw_p = raw_counts / float(total)
    single_p = single_counts / float(total)
    step_p = step_counts / float(total)
    p_sorted, prev = [], 0.0
    for r in range(m):
        value = max(prev, float(step_p[r]), float(raw_p[order[r]]))
        value = min(1.0, value)
        p_sorted.append(value)
        prev = value
    p_maxt = _fc_np.empty(m, dtype=float)
    for r, idx in enumerate(order):
        p_maxt[idx] = p_sorted[r]
    # Exact ties are one hypothesis in a step-down procedure; give tied
    # observed statistics the same maximal adjusted p.
    i = 0
    while i < m:
        j = i + 1
        while (j < m and abs(float(t[order[j]]) - float(t[order[i]])) <= tol):
            j += 1
        if j - i > 1:
            tied = max(float(p_maxt[order[k]]) for k in range(i, j))
            for k in range(i, j):
                p_maxt[order[k]] = tied
        i = j
    critical, achieved = _joint_tail_threshold(_fc_np.sort(max_all), alpha)
    rank_by_name = _fc_np.empty(m, dtype=_fc_np.int64)
    rank_by_name[order] = _fc_np.arange(1, m + 1, dtype=_fc_np.int64)
    p_floor = (1.0 if not _fc_np.any(t > 0.0)
               else min(1.0, 2.0 / float(total)))
    return {
        'p_raw': [float(x) for x in raw_p],
        'p_maxt_single': [float(x) for x in single_p],
        'p_maxt': [float(x) for x in p_maxt],
        'score': [float(x) for x in t],
        'rank': [int(x) for x in rank_by_name],
        'critical_value': (float(critical) if critical is not None else None),
        'ci_tail_level': float(achieved),
        'n_patterns': int(total),
        'p_floor': p_floor,
    }


def _fc_dependence(matrix):
    """Per-seed correlation, eigenvalues and effective number of tests."""
    A = _fc_np.asarray(matrix, dtype=float)
    m, n = A.shape
    if m < 2 or n < 2:
        return {'available': False,
                'reason': 'at least two contrasts and two seeds are needed'}
    # A dense correlation matrix and its eigendecomposition are O(m**2) and
    # O(m**3).  The score max-T above already scales with the family, so a very
    # large declared family should still get its p-values without attempting
    # an infeasible dependence summary.  Fail this DESCRIPTIVE sub-block closed
    # (with a reason) instead of allocating an unbounded matrix.
    _cap = FAMILY_CORRECTION_POLICY['dependence_max_contrasts']
    if m > _cap:
        return {'available': False,
                'n_contrasts': int(m),
                'n_seeds': int(n),
                'reason': (f'{m} contrasts exceed the dense dependence-summary '
                           f'cap {int(_cap)}; the score max-T p-values remain '
                           'valid but the correlation/eigenvalue summary is '
                           'not computed')}
    # Row-scale first: a finite 1e308 panel would overflow a naive variance.
    # Correlation is invariant to a positive per-row scale.
    row_scale = _fc_np.max(_fc_np.abs(A), axis=1)
    nz = row_scale > 0.0
    An = _fc_np.zeros_like(A)
    if _fc_np.any(nz):
        An[nz] = A[nz] / row_scale[nz, None]
    sd = An.std(axis=1, ddof=1)
    if _fc_np.any(~_fc_np.isfinite(sd)) or _fc_np.any(sd <= 0.0):
        return {'available': False,
                'reason': ('at least one contrast has zero seed variance, so '
                           'its correlation is undefined; the score max-T '
                           'p-values remain valid but the dependence summary '
                           'is not')}
    C = _fc_np.corrcoef(An)
    if not _fc_np.all(_fc_np.isfinite(C)):
        return {'available': False,
                'reason': 'the per-seed correlation matrix is not finite'}
    off = C - _fc_np.eye(m)
    n_off = m * (m - 1)
    abs_off = _fc_np.abs(off)
    eigenvalues = _fc_np.linalg.eigvalsh(C)
    n_negative = int((eigenvalues < -1e-8).sum())
    eig = _fc_np.clip(eigenvalues, 0.0, None)
    denom = float((eig * eig).sum())
    m_eff = (float(eig.sum()) ** 2 / denom) if denom > 0.0 else None
    var = float(eig.var(ddof=0)) if eig.size else None
    m_eff_nyholt = (1.0 + (m - 1) * (1.0 - var / m)) if var is not None else None
    if m_eff is not None:
        m_eff = float(min(float(m), max(1.0, m_eff)))
    if m_eff_nyholt is not None:
        m_eff_nyholt = float(min(float(m), max(1.0, m_eff_nyholt)))
    return {
        'available': True,
        'n_contrasts': int(m),
        'n_seeds': int(n),
        'correlation': [[float(x) for x in row] for row in C],
        'eigenvalues': [float(x) for x in eigenvalues],
        'n_materially_negative_eigenvalues': n_negative,
        'effective_tests': (m_eff if n_negative == 0 else None),
        'effective_tests_nyholt': (m_eff_nyholt if n_negative == 0 else None),
        'max_abs_offdiag': (float(abs_off.max()) if n_off else None),
        'mean_abs_offdiag': (float(abs_off.sum() / n_off) if n_off else None),
        'note': ('effective_tests = (sum lambda)^2 / sum(lambda^2) on the '
                 'correlation eigenvalues (Li & Ji); Nyholt is the variance '
                 'form.  Both are descriptive summaries of this seed matrix, '
                 'not a claim that the effective family is that size.'),
    }


def _fc_mde(critical, sem):
    if critical is None or sem is None:
        return None
    try:
        out = float(critical) * float(sem)
    except (TypeError, ValueError, OverflowError):
        return None
    return out if _fc_math.isfinite(out) else None


def _fc_frozen_holm(frozen_holm, names):
    """Normalise a frozen Holm mapping to ``{name: p}``."""
    out = {}
    if isinstance(frozen_holm, dict):
        for name in names:
            raw = frozen_holm.get(name)
            if isinstance(raw, dict):
                raw = raw.get('p_holm')
            if raw is None:
                out[name] = None
                continue
            out[name] = _fc_number(raw, f'frozen_holm[{name!r}]')
    return out or None


def confirmation_family_correction(comparisons, *, family_names, frozen_holm=None,
                                   alpha=None, max_n=None):
    """Replay the declared confirmation family under exact max-T corrections.

    Parameters
    ----------
    comparisons : mapping
        ``name -> {'per_seed': [...], 'seeds': [...], 'p_exact_signflip': ...}``.
        A record may be marked ``{'omitted': reason}``; the family is then not
        estimable because a missing slot may not shrink ``m``.
    family_names : sequence of str
        The declared family, in pre-registered order.  Every name must be
        present in ``comparisons``; extras are ignored.
    frozen_holm : mapping, optional
        The frozen Holm result (``{name: {'p_holm': ...}}`` or ``{name: p}``).
        When absent, Holm is recomputed from the frozen mean p-values.
    alpha, max_n : optional
        Override the policy defaults.

    Returns
    -------
    dict
        A strict-JSON descriptive block.  ``status`` is ``ok``, ``degenerate``
        (the family has no non-zero score) or ``not_estimable`` (missing/ragged
        family, too few seeds, or an invalid input).  The frozen decision is
        never an output field.
    """
    a = _fc_alpha(FAMILY_CORRECTION_POLICY['alpha'] if alpha is None else alpha)
    cap = _fc_int(FAMILY_CORRECTION_POLICY['max_n'] if max_n is None else max_n,
                  'max_n', minimum=2)
    if isinstance(family_names, (str, bytes, bytearray)) or not family_names:
        return _fc_not_estimable('family_names must be a non-empty sequence')
    try:
        names = [str(x) for x in family_names]
    except TypeError:
        return _fc_not_estimable('family_names must be an iterable')
    if any(not n.strip() for n in names):
        return _fc_not_estimable('family_names contains an empty name')
    if len(set(names)) != len(names):
        return _fc_not_estimable('family_names contains a duplicate')
    if not isinstance(comparisons, dict):
        return _fc_not_estimable('comparisons must be a mapping')
    missing = [n for n in names if n not in comparisons]
    if missing:
        return _fc_not_estimable(
            f'declared family member(s) missing from comparisons: {missing}',
            extra={'family_names': names, 'missing': missing})
    omitted = {}
    for n in names:
        rec = comparisons.get(n)
        if not isinstance(rec, dict):
            return _fc_not_estimable(f'{n}: comparison record is not an object')
        if 'omitted' in rec or rec.get('omitted'):
            omitted[n] = str(rec.get('omitted'))
    if omitted:
        return _fc_not_estimable(
            'a declared comparison is omitted; max-T is not computed over a '
            'shrunk family', extra={'family_names': names, 'omitted': omitted})

    rows = {}
    seed_sets = {}
    for n in names:
        rec = comparisons[n]
        raw = rec.get('per_seed')
        seeds = rec.get('seeds')
        if not isinstance(raw, list) or not raw:
            return _fc_not_estimable(f'{n}: no per_seed deltas for replay')
        if not isinstance(seeds, list) or len(seeds) != len(raw):
            return _fc_not_estimable(
                f'{n}: per_seed and seeds are missing or have different lengths')
        try:
            norm_seeds = [_fc_int(s, f'{n}.seeds[{i}]', minimum=0)
                          for i, s in enumerate(seeds)]
        except ValueError as e:
            return _fc_not_estimable(str(e))
        if len(set(norm_seeds)) != len(norm_seeds):
            return _fc_not_estimable(f'{n}: duplicate seed labels')
        try:
            vals = [_fc_number(x, f'{n}.per_seed[{i}]')
                    for i, x in enumerate(raw)]
        except ValueError as e:
            return _fc_not_estimable(str(e))
        rows[n] = vals
        seed_sets[n] = norm_seeds
    # Exact joint sign flips require the SAME seed vector in the SAME order.
    ref_seeds = seed_sets[names[0]]
    ragged = {n: seed_sets[n] for n in names if seed_sets[n] != ref_seeds}
    if ragged:
        common = set(ref_seeds)
        for n in names:
            common &= set(seed_sets[n])
        return _fc_not_estimable(
            'the declared comparisons do not share one complete seed vector; '
            'a joint sign-flip null cannot be formed', extra={
                'family_names': names,
                'seed_sets': {n: seed_sets[n] for n in names},
                'common_seeds': sorted(common),
                'n_common': len(common),
            })
    n = len(ref_seeds)
    if n < FAMILY_CORRECTION_POLICY['min_seeds']:
        return _fc_not_estimable(
            f'only {n} paired seeds; max-T needs at least '
            f'{FAMILY_CORRECTION_POLICY["min_seeds"]}')
    if n > cap:
        return _fc_not_estimable(
            f'n={n} exceeds the exact-enumeration cap max_n={cap}; the '
            'block refuses an approximation rather than mislabel it exact')
    m = len(names)
    D = [rows[name] for name in names]

    score_mx = _fc_score_maxt(D, a)
    raw_score = {name: float(score_mx['p_raw'][i])
                 for i, name in enumerate(names)}
    maxt_score = {name: float(score_mx['p_maxt'][i])
                  for i, name in enumerate(names)}
    try:
        holm_score = holm_adjust([(name, raw_score[name]) for name in names],
                                  alpha=a)
    except ValueError as e:
        return _fc_not_estimable(f'score-statistic Holm failed: {e}')

    # ``frozen_holm`` (when supplied) is ALREADY the adjusted Holm mapping.
    # Keep it separate from the raw mean p: using the adjusted value as
    # ``p_raw_mean`` and then applying Holm again would report the wrong raw
    # p and a double-corrected Holm p.  The raw mean statistic always comes
    # from each comparison's ``p_exact_signflip``; only fall back to
    # recomputing Holm when the caller did not supply the frozen mapping.
    raw_mean_p = {}
    for name in names:
        p = comparisons[name].get('p_exact_signflip')
        raw_mean_p[name] = (None if p is None
                            else _fc_number(p, f'{name}.p_exact_signflip'))
    frozen_holm_p = _fc_frozen_holm(frozen_holm, names)
    holm_mean = None
    if frozen_holm_p is not None:
        # Preserve the frozen adjusted values verbatim.  A missing slot stays
        # None rather than being silently re-estimated from the raw p.
        holm_mean = {name: {'p_holm': frozen_holm_p.get(name)}
                     for name in names}
    elif all(v is not None for v in raw_mean_p.values()):
        try:
            holm_mean = holm_adjust(
                [(name, raw_mean_p[name]) for name in names], alpha=a)
        except ValueError as e:
            return _fc_not_estimable(f'mean-statistic Holm failed: {e}')

    # Studentised t twin: supplies intervals/MDE, and is cross-checked against
    # the score p-values.  A constant non-zero effect makes the sample-SD
    # denominator undefined, so failure here must not hide the valid score
    # max-T p-values.
    t_block = None
    t_reason = None
    try:
        t_block = maxt_signflip_adjust({name: rows[name] for name in names},
                                       alpha=a, max_n=cap)
    except ValueError as e:
        t_reason = str(e)
    if t_block is not None:
        raw_t = {name: float(t_block['contrasts'][name]['p_raw_signflip'])
                 for name in names}
        maxt_t = {name: float(t_block['contrasts'][name]['p_maxt'])
                  for name in names}
        try:
            holm_t = holm_adjust([(name, raw_t[name]) for name in names],
                                 alpha=a)
        except ValueError as e:
            t_block = None
            t_reason = f'same-statistic Holm failed: {e}'
            raw_t = maxt_t = holm_t = None
    else:
        raw_t = maxt_t = holm_t = None

    critical = (t_block.get('critical_value')
                if isinstance(t_block, dict) else None)
    try:
        holm_tc = _t_critical(1.0 - a / float(m), n - 1)
    except Exception:
        holm_tc = None

    rows_out = {}
    for i, name in enumerate(names):
        sem = (float(t_block['contrasts'][name]['sem'])
               if (t_block is not None
                   and t_block['contrasts'][name].get('sem') is not None)
               else None)
        h = _fc_mde(holm_tc, sem)
        x = _fc_mde(critical, sem)
        ratio = (x / h if (h is not None and x is not None and h != 0.0)
                 else None)
        rows_out[name] = {
            'n': int(n),
            'mean': float(_fc_math.fsum(x / float(n) for x in rows[name])),
            'rank_score': int(score_mx['rank'][i]),
            'score_stat': float(score_mx['score'][i]),
            'p_raw_mean': raw_mean_p.get(name),
            'p_holm_mean': (float(holm_mean[name]['p_holm'])
                            if (holm_mean is not None
                                and holm_mean.get(name, {}).get('p_holm')
                                is not None)
                            else None),
            'p_raw_score': raw_score[name],
            'p_holm_score': float(holm_score[name]['p_holm']),
            'p_maxt_score': maxt_score[name],
            'reject_frozen_holm': (None if holm_mean is None
                                   or holm_mean.get(name, {}).get('p_holm')
                                   is None
                                   else bool(float(holm_mean[name]['p_holm'])
                                             < a)),
            'reject_holm_mean': (None if holm_mean is None
                                 or holm_mean.get(name, {}).get('p_holm')
                                 is None
                                 else bool(float(holm_mean[name]['p_holm'])
                                           < a)),
            'reject_holm_score': bool(float(holm_score[name]['p_holm']) < a),
            'reject_maxt_score': bool(maxt_score[name] < a),
            't_stat': (float(t_block['contrasts'][name]['t_stat'])
                       if t_block is not None else None),
            'rank_t': (int(t_block['contrasts'][name]['rank'])
                       if t_block is not None else None),
            'sd': (float(t_block['contrasts'][name]['sd'])
                   if (t_block is not None
                       and t_block['contrasts'][name].get('sd') is not None)
                   else None),
            'sem': sem,
            'p_raw_t': (float(raw_t[name]) if raw_t is not None else None),
            'p_holm_t': (float(holm_t[name]['p_holm'])
                         if holm_t is not None else None),
            'p_maxt_t': (float(maxt_t[name]) if maxt_t is not None else None),
            'reject_maxt_t': (bool(float(maxt_t[name]) < a)
                              if maxt_t is not None else None),
            'ci_simultaneous': (
                list(t_block['contrasts'][name]['ci_simultaneous'])
                if (t_block is not None
                    and isinstance(t_block['contrasts'][name].get('ci_simultaneous'),
                                   (list, tuple))) else None),
            'mde_holm': h,
            'mde_maxt': x,
            'mde_ratio': (float(ratio) if ratio is not None
                          and _fc_math.isfinite(ratio) else None),
        }

    frozen_set = sorted(n for n in names
                        if rows_out[n]['reject_frozen_holm'] is True)
    holm_mean_set = sorted(n for n in names
                           if rows_out[n]['reject_holm_mean'] is True)
    holm_score_set = sorted(n for n in names
                            if rows_out[n]['reject_holm_score'])
    maxt_score_set = sorted(n for n in names
                            if rows_out[n]['reject_maxt_score'])
    maxt_t_set = (sorted(n for n in names if rows_out[n]['reject_maxt_t'])
                  if maxt_t is not None else [])
    added = sorted(set(maxt_score_set) - set(holm_score_set))
    removed = sorted(set(holm_score_set) - set(maxt_score_set))
    if set(maxt_score_set) == set(holm_score_set):
        correction_effect = 'same'
    elif set(maxt_score_set) > set(holm_score_set):
        correction_effect = 'maxt_rejects_more'
    elif set(maxt_score_set) < set(holm_score_set):
        correction_effect = 'maxt_rejects_fewer'
    else:
        correction_effect = 'crossed'
    score_vs_mean = sorted(n for n in names
                           if rows_out[n]['reject_frozen_holm']
                           != rows_out[n]['reject_holm_score'])
    score_vs_t = (sorted(n for n in names
                         if rows_out[n]['reject_maxt_score']
                         != rows_out[n]['reject_maxt_t'])
                  if maxt_t is not None else [])
    score_raw_match = (all(abs(raw_score[n] - raw_t[n]) <= 1e-12 for n in names)
                       if raw_t is not None else None)
    score_maxt_match = (all(abs(maxt_score[n] - maxt_t[n]) <= 1e-12 for n in names)
                        if maxt_t is not None else None)
    ratios = [r['mde_ratio'] for r in rows_out.values()
              if isinstance(r.get('mde_ratio'), (int, float))]
    p_floor = float(score_mx['p_floor'])
    holm_threshold = a / float(m)
    dep = _fc_dependence(D)
    all_zero = not any(abs(x) > 0.0 for row in D for x in row)
    status = 'degenerate' if all_zero else 'ok'
    reason = ('all per-seed deltas are exactly zero; the family has no '
              'non-zero score to test') if all_zero else None
    if t_block is None:
        ci_note = f'studentised t twin unavailable: {t_reason}'
    elif t_block.get('critical_value') is None:
        ci_note = ('studentised t twin: the exact family null cannot reach the '
                   'requested simultaneous level at this n')
    elif (t_block.get('ci_tail_level') is not None
          and float(t_block['ci_tail_level']) > a):
        ci_note = ('studentised t twin: the discrete exact null overshoots '
                   'alpha, so the simultaneous interval has tail level '
                   'ci_tail_level slightly above the nominal alpha')
    else:
        ci_note = None

    return {
        'kind': 'confirmation_family_correction',
        'version': FAMILY_CORRECTION_POLICY['version'],
        'status': status,
        'reason': reason,
        'method': FAMILY_CORRECTION_POLICY['method'],
        'statistic': FAMILY_CORRECTION_POLICY['statistic'],
        'role': FAMILY_CORRECTION_POLICY['role'],
        'boundary': FAMILY_CORRECTION_POLICY['boundary'],
        'note': ci_note,
        'alpha': a,
        'family_names': names,
        'm_declared': int(m),
        'm_used': int(m),
        'n_seeds': int(n),
        'seeds': [int(s) for s in ref_seeds],
        'n_patterns': int(score_mx['n_patterns']),
        'p_floor': p_floor,
        'holm_threshold': float(holm_threshold),
        'holm_gate_attainable': bool(p_floor <= holm_threshold),
        'max_n': int(cap),
        'score_maxt': {
            'critical_value': score_mx['critical_value'],
            'critical_value_available': bool(score_mx['critical_value'] is not None),
            'ci_tail_level': score_mx['ci_tail_level'],
            'ci_available': False,
            'ci_note': ('a common score critical value does not translate to a '
                        'raw-mean interval; intervals come from the '
                        'studentised t twin when available'),
        },
        'studentized': {
            'available': bool(t_block is not None),
            'reason': t_reason,
            'critical_value': (float(critical) if critical is not None else None),
            'ci_level_requested': (float(t_block.get('ci_level_requested'))
                                   if t_block is not None else None),
            'ci_tail_level': (float(t_block['ci_tail_level'])
                              if (t_block is not None
                                  and t_block.get('ci_tail_level') is not None)
                              else None),
            'ci_method': (t_block.get('ci_method')
                          if t_block is not None else None),
            'raw_score_matches': score_raw_match,
            'maxt_score_matches': score_maxt_match,
        },
        'holm_critical_t': (float(holm_tc) if holm_tc is not None else None),
        'dependence': dep,
        'comparisons': rows_out,
        'sets': {
            'frozen_holm': frozen_set,
            'holm_mean': holm_mean_set,
            'holm_score': holm_score_set,
            'maxt_score': maxt_score_set,
            'maxt_t': maxt_t_set,
            'maxt_score_added_vs_holm_score': added,
            'maxt_score_removed_vs_holm_score': removed,
            'score_vs_mean_changed': score_vs_mean,
            'score_vs_t_changed': score_vs_t,
            'all_three_agree': bool(frozen_set == holm_score_set == maxt_score_set),
        },
        'summary': {
            'n_frozen_holm': len(frozen_set),
            'n_holm_score': len(holm_score_set),
            'n_maxt_score': len(maxt_score_set),
            'n_maxt_t': (len(maxt_t_set) if maxt_t is not None else None),
            'correction_effect': correction_effect,
            'statistic_effect': ('same' if not score_vs_mean else 'differ'),
            'n_mde_ratio': len(ratios),
            'mean_mde_ratio': (float(sum(ratios) / len(ratios))
                               if ratios else None),
            'max_mde_ratio': (float(max(ratios)) if ratios else None),
            'note': ('max-T and Holm are both applied to the score statistic, '
                     'so their difference isolates the correction method.  '
                     'The frozen mean test is a different statistic; '
                     '`score_vs_mean_changed` names where that choice matters.  '
                     'The studentised t twin is used only for intervals/MDE.  '
                     'These are descriptive readings, not independent tests.'),
        },
    }


def _fc_num(value, fmt='{:.4f}'):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return fmt.format(float(value))
    return '—'


def _fc_ci(pair):
    if (isinstance(pair, (list, tuple)) and len(pair) >= 2
            and isinstance(pair[0], (int, float))
            and isinstance(pair[1], (int, float))):
        return f'[{float(pair[0]):+.4f}, {float(pair[1]):+.4f}]'
    return '—'


def _family_correction_markdown(block):
    """Render the descriptive family-correction block for REPORT_v12."""
    if not isinstance(block, dict) or not block:
        return []
    lines = []
    A = lines.append
    A('### 校正方法与族依赖性的敏感性（探索性）')
    A('')
    status = block.get('status')
    if status == 'not_estimable':
        A(f'- 不可估计：{block.get("reason", "insufficient data")}')
        A('')
        A('> 边界：本节不改变冻结 Holm 族、`p_exact_signflip`、`p_holm`、'
          '`verified` 或 `confirmed`；它只说明为什么在这个面板上无法给出'
          '依赖感知的联合校正读数（缺失族成员、种子不齐或超出精确枚举上限）。')
        A('')
        return lines
    names = block.get('family_names') or []
    m = block.get('m_declared')
    n = block.get('n_seeds')
    A(f'- 声明族 m={m}；完整配对种子 n={n}；精确符号翻转模式 '
      f'{block.get("n_patterns")}；单比较 p 分辨率 {_fc_num(block.get("p_floor"), "{:.5f}")}；'
      f'Holm 首门阈值 α/m={_fc_num(block.get("holm_threshold"), "{:.5f}")}'
      f'（可达={block.get("holm_gate_attainable")}）。')
    dep = block.get('dependence') or {}
    if dep.get('available'):
        A(f'- 族依赖：有效检验数 {_fc_num(dep.get("effective_tests"), "{:.2f}")}'
          f'（Nyholt {_fc_num(dep.get("effective_tests_nyholt"), "{:.2f}")}）；'
          f'|r| 最大 {_fc_num(dep.get("max_abs_offdiag"), "{:.3f}")}、'
          f'平均 {_fc_num(dep.get("mean_abs_offdiag"), "{:.3f}")}。')
    else:
        A(f'- 族依赖：不可用（{dep.get("reason", "—")}）。')
    A(f'- 同统计量 Holm 临界 t {_fc_num(block.get("holm_critical_t"), "{:.3f}")}；'
      f'studentised 联合临界 t '
      f'{_fc_num((block.get("studentized") or {}).get("critical_value"), "{:.3f}")}'
      f'（精确尾部 {_fc_num((block.get("studentized") or {}).get("ci_tail_level"), "{:.4f}")}）。')
    if status == 'degenerate':
        A(f'- **退化**：{block.get("reason", "—")}')
    if block.get('note'):
        A(f'- 注：{block.get("note")}')
    A('')
    A('| 比较 | n | raw p (mean) | Holm(mean) p | raw p (score) | Holm(score) p | max-T(score) p | max-T 拒绝 | 同时 CI (t) | MDE Holm | MDE max-T |')
    A('|---|---|---|---|---|---|---|---|---|---|---|')
    rows = block.get('comparisons') or {}
    for name in names:
        r = rows.get(name)
        if not isinstance(r, dict):
            continue
        A(f'| {name} | {r.get("n", "—")} | {_fc_num(r.get("p_raw_mean"))} | '
          f'{_fc_num(r.get("p_holm_mean"))} | {_fc_num(r.get("p_raw_score"))} | '
          f'{_fc_num(r.get("p_holm_score"))} | {_fc_num(r.get("p_maxt_score"))} | '
          f'{r.get("reject_maxt_score")} | {_fc_ci(r.get("ci_simultaneous"))} | '
          f'{_fc_num(r.get("mde_holm"))} | {_fc_num(r.get("mde_maxt"))} |')
    A('')
    sets = block.get('sets') or {}
    A(f'- 冻结 Holm 拒绝：{sets.get("frozen_holm") or "—"}；'
      f'同统计量 Holm(score)：{sets.get("holm_score") or "—"}；'
      f'max-T(score)：{sets.get("maxt_score") or "—"}。')
    summ = block.get('summary') or {}
    A(f'- 校正方法效应 `{summ.get("correction_effect", "—")}`'
      f'（max-T 相对同统计量 Holm 新增 {sets.get("maxt_score_added_vs_holm_score") or []}，'
      f'移除 {sets.get("maxt_score_removed_vs_holm_score") or []}）；'
      f'统计量效应（冻结 mean vs score）`{summ.get("statistic_effect", "—")}`'
      f'（差异 {sets.get("score_vs_mean_changed") or []}）。')
    if summ.get('mean_mde_ratio') is not None:
        A(f'- MDE 比值（studentised max-T / Holm）平均 '
          f'{_fc_num(summ.get("mean_mde_ratio"), "{:.3f}")}、'
          f'最大 {_fc_num(summ.get("max_mde_ratio"), "{:.3f}")}；小于 1 表示'
          '同一观测 SD 下联合区间更紧。')
    A('')
    A('> 边界：上表是**同一份冻结逐种子配对差上的校正方法/族依赖敏感性**，'
      '不是新的显著性检验，不进入冻结 Holm 族、不做 FWER 控制、不改变 '
      '`p_exact_signflip`、`p_holm`、`verified` 或冻结 `confirmed`。'
      '`max-T(score)` 与 `Holm(score)` 使用同一个 sign-invariant 统计量 '
      '`|Σd|/sqrt(Σd²)`，因此两者的差异只反映校正方法；冻结的 mean 检验是'
      '另一个统计量，表内并列 `raw p (mean)` 以显示统计量选择的影响。'
      '`同时 CI` / `MDE` 来自 studentised t 孪生读数，仅在样本 SD 可用时给出；'
      '常数非零效应下样本 SD 为 0，t 孪生不可用，但 score max-T 仍有效。'
      '`effective_tests` 只是种子相关矩阵的描述量，不是「真正独立的检验数」'
      '的因果断言；n=12 时相关矩阵本身很粗。max-T 在强正相关时可能比 Holm '
      '多拒绝，但这不构成新主张：冻结 Holm/`confirmed` 仍是唯一权威判定；'
      '未来清单若要把 max-T 写成决策规则，必须在首次 test 读取前预注册，'
      '并同步扩展当前只接受 Holm 的分析器。'
      'MDE 是观测 SD 下的 t 型半宽，不是功效；`degenerate` 只表示这个 n 下'
      '精确联合零分布达不到名义水平，不是「无效应」证据。')
    A('')
    return lines
