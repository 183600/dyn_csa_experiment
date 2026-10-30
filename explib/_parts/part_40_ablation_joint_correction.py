"""Joint (max-T) correction for a family of paired ablation contrasts.

Holm-Bonferroni is valid but treats every declared contrast as if it carried
independent evidence.  The ablation table does not look like that: most
contrasts share a reference arm, are measured on the same seeds and the same
evaluation rows, so their per-seed deltas are strongly correlated.  Holm still
pays the full ``m`` penalty on the smallest p-value, which makes the family
look less informative than the data actually are.

This fragment adds the exact **Westfall-Young free step-down max-T** procedure
for the same sign-flip null the panel already uses.  It is not a new test of a
new hypothesis: it is a second, correlation-aware multiplicity reading of the
existing per-seed deltas.  The declared/frozen correction stays Holm; the max-T
numbers are a sensitivity readout unless a future plan explicitly pre-registers
``max-t`` as the family correction.

Why the sign-flip null makes this exact
---------------------------------------
Under H0 each paired difference is symmetric about zero, so each seed's sign
is exchangeable independently of the others.  Enumerating all ``2**n`` sign
assignments gives the exact joint null of the whole contrast vector, not just
of one contrast.  Studentising by each contrast's own observed SD is harmless:
a sign flip changes signs but not ``|d_i|``, so the SD (and therefore the
scaling constant) is invariant across the enumeration.  The max-T statistic is
thus an exact permutation statistic under the same assumption the per-contrast
exact sign-flip test already relies on.

Boundaries (all fail closed)
----------------------------
* Every contrast in the family must be measured on the **same seed set** and
  be a complete, controlled pair.  Missing arms/contrasts make the family
  ``not_ready``; a max-T p computed on the survivors of a data-dependent
  subset would silently shrink the multiplicity family, so it is reported but
  never promoted to a family-corrected claim.
* ``n > JOINT_SIGNFLIP_MAX_N`` is refused rather than approximated.  Above the
  cap the caller keeps Holm and no max-T p is written.
* A contrast with zero seed variance but a non-zero mean is degenerate for a
  studentised max statistic (its t is infinite); the family is
  ``not_estimable`` instead of manufacturing a p.
* The simultaneous **interval** is studentised (``mean ± c*sd/sqrt(n)``) with
  the exact family-max critical value.  The p-values are exact under the sign
  symmetry null; the interval is a t-style approximation, not an exact
  inversion of a multivariate sign-flip test, and ``ci_tail_level`` is the
  null tail mass at the chosen critical value rather than a coverage proof.
* This is still a seed-level sign test.  It does not make seeds a random
  sample, does not replace the evaluation-unit uncertainty, does not fix
  selection or optional stopping, and does not change any frozen p/Holm
  decision or ``CODE_SEMANTICS``.
"""
import hashlib
import math

import numpy as np

JOINT_SIGNFLIP_MAX_N = 20
JOINT_SIGNFLIP_CHUNK_TARGET = 1 << 18
JOINT_SIGNFLIP_METHOD = 'westfall-young free step-down max-T (exact sign flips)'


def _joint_finite(value, *, where, name):
    """A finite real measurement; booleans and numeric strings are not deltas."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{where}: {name} is a boolean, not a measurement')
    if isinstance(value, (str, bytes, bytearray, dict, complex,
                          np.complexfloating)):
        raise ValueError(f'{where}: {name} is not a real number ({value!r})')
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'{where}: {name} is not a real number ({value!r})') from None
    if not math.isfinite(out):
        raise ValueError(f'{where}: {name} is not finite ({value!r})')
    return out


def _joint_named_matrix(named_deltas, *, where='maxt_signflip_adjust'):
    """Validate ``{name: per-seed deltas}`` into names, matrix and seed count."""
    if not isinstance(named_deltas, dict):
        raise ValueError(f'{where}: named_deltas must be a mapping, '
                         f'got {type(named_deltas).__name__}')
    if not named_deltas:
        raise ValueError(f'{where}: the contrast family is empty')
    names, rows = [], []
    n = None
    for raw_name, raw_vals in named_deltas.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError(f'{where}: contrast names must be non-empty '
                             f'strings, got {raw_name!r}')
        if isinstance(raw_vals, (str, bytes, bytearray)):
            raise ValueError(f'{where}: deltas for {raw_name!r} must be an '
                             'iterable of numbers, not a string')
        try:
            vals = list(raw_vals)
        except TypeError:
            raise ValueError(f'{where}: deltas for {raw_name!r} are not an '
                             'iterable') from None
        vals = [_joint_finite(v, where=where, name=f'{raw_name}[{i}]')
                for i, v in enumerate(vals)]
        if n is None:
            n = len(vals)
        elif len(vals) != n:
            raise ValueError(f'{where}: contrast {raw_name!r} has {len(vals)} '
                             f'deltas but {names[0]!r} has {n}; a joint '
                             'correction needs one row per seed')
        names.append(raw_name)
        rows.append(vals)
    if n is None or n < 2:
        raise ValueError(f'{where}: a joint sign-flip family needs at least 2 '
                         f'paired seeds, got {n}')
    return names, np.asarray(rows, dtype=float), int(n)


def _joint_alpha(alpha, *, where):
    if isinstance(alpha, (bool, np.bool_)):
        raise ValueError(f'{where}: alpha must be a probability in (0,1), '
                         f'got {alpha!r}')
    try:
        out = float(alpha)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'{where}: alpha must be a probability in (0,1), '
                         f'got {alpha!r}') from None
    if not (0.0 < out < 1.0):
        raise ValueError(f'{where}: alpha must be in (0,1), got {alpha!r}')
    return out


def _joint_max_n(value, *, where):
    if isinstance(value, (bool, np.bool_, str, bytes, bytearray)):
        raise ValueError(f'{where}: max_n must be a positive integer, got {value!r}')
    try:
        raw = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'{where}: max_n must be a positive integer, got {value!r}') from None
    if not math.isfinite(raw) or raw != int(raw):
        raise ValueError(f'{where}: max_n must be an exact positive integer, '
                         f'got {value!r}')
    out = int(raw)
    if out < 2:
        raise ValueError(f'{where}: max_n must be >= 2, got {value!r}')
    return out


def _joint_deltas_fingerprint(names, matrix, seeds):
    """Stable identity of the exact matrix a max-T p was computed from."""
    h = hashlib.sha256()
    h.update(b'maxt-signflip-v1\0')
    h.update((','.join(str(int(s)) for s in seeds) + '\0').encode('ascii'))
    # A dict lookup keeps this linear in m.  `names.index(name)` made the
    # fingerprint quadratic in the family size, which is exactly the axis a
    # larger ablation panel grows along.
    name_index = {name: i for i, name in enumerate(names)}
    for name in sorted(names):
        idx = name_index[name]
        h.update(name.encode('utf-8'))
        h.update(b'\0')
        for x in matrix[idx]:
            _x = float(x)
            # Normalise -0.0 so two numerically identical matrices cannot get
            # different identities from the sign of zero.
            _x = 0.0 if _x == 0.0 else _x
            h.update(format(_x, '.17g').encode('ascii'))
            h.update(b'\0')
        h.update(b'\n')
    return h.hexdigest()


def _joint_tail_threshold(sorted_max, alpha):
    """Smallest observed max-T value whose exact tail mass is <= alpha.

    ``sorted_max`` is the sorted (ascending) exact null distribution of the
    family maximum.  Ties are counted with ``>=`` so the achieved level is
    conservative.  ``None`` means the null's minimum attainable two-sided tail
    (``2/2**n``) is already above ``alpha``: no finite interval exists at the
    requested level, and the caller must say so instead of quoting one.
    """
    total = int(sorted_max.size)
    uniq, counts = np.unique(sorted_max, return_counts=True)
    # tail_i = number of null maxima >= uniq_i; with the array sorted this is
    # total minus the number strictly below, computed in O(U) instead of
    # rescanning the (up to 2**20) null distribution per candidate.
    below = np.cumsum(counts) - counts
    tails = total - below
    hit = np.nonzero(tails / float(total) <= alpha)[0]
    if hit.size:
        i = int(hit[0])
        return float(uniq[i]), 1.0 - float(tails[i]) / float(total)
    top_tail = int(tails[-1])
    return None, 1.0 - float(top_tail) / float(total)


def maxt_min_n_for_alpha(alpha=0.05):
    """Best-case seed count at which a max-T family can reach ``alpha``.

    Perfectly dependent contrasts attain the same two-sided tail as one
    contrast, whose exact floor is ``2/2**n``.  This is therefore the
    *optimistic* design floor for a correlated family; independent contrasts
    approach the Holm floor ``m*2/2**n`` (the existing
    ``signflip_min_n_for_holm``).  Report both: quoting only the best case
    would make a panel look resolvable when its contrasts are nearly
    independent.
    """
    where = 'maxt_min_n_for_alpha'
    alpha = _joint_alpha(alpha, where=where)
    # Integer doubling avoids log2 round-off at exact boundaries (e.g.
    # alpha=0.25 must give n=3, not ceil(2.9999999999999996)=3 vs 4).
    n = 2
    while 2.0 / (1 << n) > alpha:
        n += 1
    return {
        'alpha': float(alpha),
        'min_n': int(n),
        'p_floor': float(2.0 / (1 << n)),
        'beyond_enumeration_cap': bool(n > JOINT_SIGNFLIP_MAX_N),
        'enumeration_cap': int(JOINT_SIGNFLIP_MAX_N),
        'scope': ('best case (perfectly dependent contrasts); independent '
                  'contrasts approach the Holm floor m*2/2**n; n above the '
                  'enumeration cap is a design target, not a computable panel'),
    }


def maxt_signflip_adjust(named_deltas, *, alpha=0.05,
                         max_n=JOINT_SIGNFLIP_MAX_N):
    """Exact Westfall-Young step-down max-T over one complete family.

    Parameters
    ----------
    named_deltas : mapping
        ``{contrast_name: per-seed paired deltas}``.  Every contrast must have
        the same number of finite deltas, aligned to the same seeds (the
        caller owns the seed alignment; see :func:`maxt_joint_correction`).
    alpha : float
        Family-wise level in (0,1).
    max_n : int
        Enumeration cap (``2**n``).  Defaults to
        :data:`JOINT_SIGNFLIP_MAX_N`; above it the function raises rather than
        approximating.

    Returns
    -------
    dict
        ``contrasts`` maps each name to its raw exact sign-flip p, single-step
        max-T p, step-down max-T p, rank, t statistic, studentised
        simultaneous interval and reject flag.  ``critical_value`` is the
        family-max quantile used for those intervals; ``ci_tail_level`` is the
        exact null tail mass at that critical value (a diagnostic, not a
        coverage proof for the t-style interval).

    Raises
    ------
    ValueError
        Structural input errors (empty/ragged family, non-finite delta,
        boolean, n<2, n>max_n, bad alpha) and the degenerate zero-variance
        case.  The caller that wants a status object should use
        :func:`maxt_joint_correction`, which turns these into
        ``not_estimable``.
    """
    where = 'maxt_signflip_adjust'
    alpha = _joint_alpha(alpha, where=where)
    max_n = _joint_max_n(max_n, where=where)
    names, D, n = _joint_named_matrix(named_deltas, where=where)
    if n > max_n:
        raise ValueError(f'{where}: n={n} exceeds the exact-enumeration cap '
                         f'max_n={max_n}; use Holm rather than an '
                         'approximation')
    m = len(names)
    # Row-scale before the mean/SD/sum: deltas near 1e308 overflow a naive
    # sum/var even though every delta is finite.  The sign-flip statistic is
    # scale invariant, so work on each row's max-abs scale and restore the
    # mean/SD afterwards.  An all-zero row stays exactly zero.
    row_scale = np.max(np.abs(D), axis=1)
    nz_scale = row_scale > 0.0
    Dn = np.zeros_like(D)
    if np.any(nz_scale):
        Dn[nz_scale] = D[nz_scale] / row_scale[nz_scale, None]
    mean_n = Dn.mean(axis=1)
    sd_n = Dn.std(axis=1, ddof=1)
    mean = mean_n * row_scale
    with np.errstate(over='ignore', invalid='ignore'):
        sd = sd_n * row_scale
    scale = sd_n * math.sqrt(n)          # == sd*sqrt(n)/row_scale
    if np.any((sd == 0.0) & (mean != 0.0)):
        raise ValueError(
            f'{where}: a contrast has zero seed variance but a non-zero mean; '
            'the studentised max-T statistic is degenerate (infinite t) and '
            'no exact family p exists')
    # |sum| / (sd*sqrt(n)) == |mean| / (sd/sqrt(n)); the sign-flip loop divides
    # each column by its own (sign-invariant) constant.
    t = np.zeros(m, dtype=float)
    nz = scale > 0.0
    t[nz] = np.abs(Dn[nz].sum(axis=1)) / scale[nz]
    # Deterministic tie break by contrast name: a JSON round-trip or a
    # different insertion order must not change ranks or adjusted p-values.
    order = np.asarray(sorted(range(m), key=lambda i: (-float(t[i]), names[i])),
                       dtype=np.int64)
    t_sorted = t[order]
    tol = 1e-12 * max(1.0, float(np.max(t)) if m else 1.0)
    total = 1 << n
    bitpos = np.arange(n, dtype=np.int64)[None, :]
    chunk = max(1, min(total,
                       max(1, JOINT_SIGNFLIP_CHUNK_TARGET // max(n, 1))))
    raw_counts = np.zeros(m, dtype=np.int64)
    single_counts = np.zeros(m, dtype=np.int64)
    step_counts = np.zeros(m, dtype=np.int64)
    max_all = np.empty(total, dtype=float)
    for off in range(0, total, chunk):
        bs = min(chunk, total - off)
        masks = np.arange(off, off + bs, dtype=np.int64)[:, None]
        signs = np.where((masks & (1 << bitpos)) != 0, 1.0, -1.0)
        sums = signs @ Dn.T
        t_star = np.zeros_like(sums)
        if np.any(nz):
            t_star[:, nz] = np.abs(sums[:, nz]) / scale[None, nz]
        fam_max = t_star.max(axis=1)
        max_all[off:off + bs] = fam_max
        raw_counts += (t_star >= t[None, :] - tol).sum(axis=0)
        single_counts += (fam_max[:, None] >= t[None, :] - tol).sum(axis=0)
        # cum[:, r] = max over the still-unrejected ranks r..m-1.
        cum = np.maximum.accumulate(t_star[:, order][:, ::-1], axis=1)[:, ::-1]
        step_counts += (cum >= t_sorted[None, :] - tol).sum(axis=0)
    raw_p = raw_counts / float(total)
    single_p = single_counts / float(total)
    step_p = step_counts / float(total)
    # Free step-down, with two extra fail-closed constraints: the adjusted p
    # can neither fall below its own raw p nor below the previously rejected
    # rank's adjusted p.  The first keeps "adjusted >= unadjusted" a literal
    # invariant; the second keeps the reported vector monotone.
    p_sorted, prev = [], 0.0
    for r in range(m):
        value = max(prev, float(step_p[r]), float(raw_p[order[r]]))
        value = min(1.0, value)
        p_sorted.append(value)
        prev = value
    p_maxt = np.empty(m, dtype=float)
    for r, idx in enumerate(order):
        p_maxt[idx] = p_sorted[r]
    # Exact ties are one hypothesis in a step-down procedure, not an arbitrary
    # ordering: give every member of a tied t-group the same (maximal) adjusted
    # p so the result is deterministic and does not depend on names beyond the
    # tie-break above.
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
    critical, achieved = _joint_tail_threshold(np.sort(max_all), alpha)
    # Rank lookup is O(1) per contrast; recomputing `np.nonzero(order == k)`
    # inside the loop made it O(m**2) for a large ablation family.
    rank_by_name = np.empty(m, dtype=np.int64)
    rank_by_name[order] = np.arange(1, m + 1, dtype=np.int64)
    contrasts = {}
    for k, name in enumerate(names):
        # `sd` can exceed float64 even though the standard error (and the
        # interval half-width) is representable: e.g. 20 balanced deltas at
        # +/-1.79e308.  Keep the SD unavailable as JSON-safe None, but build
        # the interval from the representable normalized standard error.
        _sd = float(sd[k])
        _sd_out = _sd if math.isfinite(_sd) else None
        if _sd_out is not None:
            sem = float(_sd / math.sqrt(n))
            half = (critical * _sd / math.sqrt(n)) if critical is not None else None
            if half is not None and not math.isfinite(half) and sem is not None:
                # `critical * sd` can overflow even when the half-width
                # `critical * sem` is representable (mean and half can even
                # cancel to finite endpoints).  Reorder only on that extreme
                # path so ordinary intervals keep their historical value.
                half = critical * sem
        else:
            _sem = float((sd_n[k] / math.sqrt(n)) * row_scale[k])
            sem = _sem if math.isfinite(_sem) else None
            half = (critical * sem
                    if critical is not None and sem is not None else None)
        ci = None
        if (half is not None and math.isfinite(half)
                and math.isfinite(float(mean[k]))):
            _lo = float(mean[k] - half)
            _hi = float(mean[k] + half)
            if math.isfinite(_lo) and math.isfinite(_hi):
                ci = [_lo, _hi]
        contrasts[name] = {
            'p_raw_signflip': float(raw_p[k]),
            'p_maxt_single': float(single_p[k]),
            'p_maxt': float(p_maxt[k]),
            't_stat': float(t[k]),
            'rank': int(rank_by_name[k]),
            'reject_maxt': bool(p_maxt[k] <= alpha),
            'mean': float(mean[k]),
            'sd': _sd_out,
            'sd_available': _sd_out is not None,
            'sem': sem,
            'sem_available': sem is not None,
            'ci_simultaneous': ci,
        }
    return {
        'kind': 'maxt_signflip',
        'status': 'ok',
        'method': JOINT_SIGNFLIP_METHOD,
        'alpha': float(alpha),
        'n_seeds': int(n),
        'm': int(m),
        'n_patterns': int(total),
        'p_floor_single': (2.0 / total if total > 1 else 1.0),
        'critical_value': critical,
        'ci_level_requested': float(1.0 - alpha),
        'ci_tail_level': float(achieved),
        'ci_method': ('studentised max-T critical-value interval '
                      '(t-style; the exact tail level is ci_tail_level)'),
        'ci_note': (None if critical is not None else
                    'the exact family null cannot reach the requested level '
                    'at this n; no finite simultaneous interval is reported'),
        'contrasts': contrasts,
        'reject': sorted(n for n, rec in contrasts.items() if rec['reject_maxt']),
        'deltas_fp': _joint_deltas_fingerprint(names, D,
                                               list(range(n))),
        'note': ('correlation-aware exact max-T reading; the declared '
                 'correction remains Holm unless a plan pre-registers max-t'),
    }


def maxt_joint_correction(records, *, alpha=0.05,
                          family_size=None,
                          max_n=JOINT_SIGNFLIP_MAX_N):
    """Group recorded contrasts and compute one exact max-T per seed set.

    ``records`` is a mapping ``key -> {'deltas': ..., 'seeds': ...,
    'eligible': bool, 'reason': str}``.  A record is eligible only when the
    caller marks it as a complete, controlled pair; ineligible records are
    named in ``excluded`` and never enter a family.  Eligible records are
    grouped by their exact seed tuple, because a joint null across different
    seed sets is not a permutation of one matrix.

    ``family_size`` is the declared/effective family size (the same ``m`` the
    panel's Holm correction used).  A max-T p is reported whenever its own
    seed group is computable, but ``family_ready`` is true only when every one
    of the ``family_size`` declared contrasts is eligible in **one** group.
    That prevents a partial panel from silently replacing Holm's ``m`` with
    the smaller number of contrasts that happened to finish.
    """
    where = 'maxt_joint_correction'
    alpha = _joint_alpha(alpha, where=where)
    max_n = _joint_max_n(max_n, where=where)
    if not isinstance(records, dict):
        raise ValueError(f'{where}: records must be a mapping, got '
                         f'{type(records).__name__}')
    fs = None
    if family_size is not None:
        if isinstance(family_size, (bool, np.bool_, str, bytes, bytearray)):
            raise ValueError(f'{where}: family_size must be a positive '
                             f'integer, got {family_size!r}')
        try:
            _fs_raw = float(family_size)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f'{where}: family_size must be a positive '
                             f'integer, got {family_size!r}') from None
        if not math.isfinite(_fs_raw) or _fs_raw != int(_fs_raw):
            raise ValueError(f'{where}: family_size must be an exact positive '
                             f'integer, got {family_size!r}')
        fs = int(_fs_raw)
        if fs < 1:
            raise ValueError(f'{where}: family_size must be >= 1, got {family_size!r}')
    groups = {}
    excluded = {}
    for key, rec in records.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f'{where}: record keys must be non-empty strings, '
                             f'got {key!r}')
        rec = rec if isinstance(rec, dict) else {}
        if not rec.get('eligible', True):
            excluded[key] = str(rec.get('reason') or 'ineligible')
            continue
        raw_seeds = rec.get('seeds')
        try:
            seeds = []
            for s in (raw_seeds if raw_seeds is not None else []):
                if isinstance(s, (bool, np.bool_, str, bytes, bytearray)):
                    raise ValueError
                _sf = float(s)
                if not math.isfinite(_sf) or _sf != int(_sf) or _sf < 0:
                    raise ValueError
                seeds.append(int(_sf))
        except (TypeError, ValueError, OverflowError):
            excluded[key] = 'seeds are not exact non-negative integers'
            continue
        if len(set(seeds)) != len(seeds):
            excluded[key] = 'duplicate seeds'
            continue
        try:
            names, D, n = _joint_named_matrix({key: rec.get('deltas')},
                                              where=where)
        except ValueError as exc:
            excluded[key] = str(exc)
            continue
        if seeds and len(seeds) != n:
            excluded[key] = (f'{len(seeds)} seeds but {n} deltas')
            continue
        if seeds:
            # Canonical seed order so [0,2,1] and [0,1,2] are one family, not
            # two; reorder the deltas with the seeds to keep them paired.
            order = sorted(range(n), key=lambda i: seeds[i])
            seeds = [seeds[i] for i in order]
            row = D[0][order]
        else:
            seeds = list(range(n))
            row = D[0]
        groups.setdefault(tuple(seeds), []).append((key, row))
    by_key, group_out, warnings = {}, {}, []
    for seeds, members in sorted(groups.items()):
        names = [k for k, _ in members]
        matrix = np.asarray([row for _, row in members], dtype=float)
        try:
            res = maxt_signflip_adjust(
                {name: matrix[i] for i, name in enumerate(names)},
                alpha=alpha, max_n=max_n)
        except ValueError as exc:
            group_out[str(list(seeds))] = {
                'status': 'not_estimable', 'seeds': list(seeds),
                'm': len(names), 'reason': str(exc)}
            for name in names:
                by_key[name] = {'status': 'not_estimable',
                                'reason': str(exc),
                                'group': str(list(seeds))}
            warnings.append(f'seeds={list(seeds)}: {exc}')
            continue
        group_out[str(list(seeds))] = {
            'status': 'ok', 'seeds': list(seeds), 'm': int(res['m']),
            'contrasts': sorted(names),
            'n_patterns': int(res['n_patterns']),
            'critical_value': res['critical_value'],
            'ci_tail_level': res['ci_tail_level'],
            'ci_method': res['ci_method'],
            'p_floor_single': res['p_floor_single'],
            'deltas_fp': _joint_deltas_fingerprint(names, matrix,
                                                   list(seeds)),
        }
        for name, rec in res['contrasts'].items():
            rec = dict(rec)
            rec['status'] = 'ok'
            rec['group'] = str(list(seeds))
            by_key[name] = rec
    m_eligible = sum(len(v) for v in groups.values())
    n_groups = len(groups)
    ready = bool(
        n_groups == 1
        and all(g.get('status') == 'ok' for g in group_out.values())
        and (fs is None or m_eligible == fs))
    if fs is not None and m_eligible != fs:
        warnings.append(
            f'the declared family has m={fs} but only {m_eligible} complete, '
            'controlled contrast(s) share one seed set; a max-T p on this '
            'subset is NOT a family-corrected replacement for Holm')
    if n_groups > 1:
        warnings.append(
            f'the complete contrasts fall into {n_groups} different seed '
            'sets; each max-T is corrected only within its own seed set')
    status = ('ok' if ready else
              ('not_estimable' if group_out and
               all(g.get('status') != 'ok' for g in group_out.values())
               else 'subset_only'))
    return {
        'kind': 'maxt_joint_correction',
        'status': status,
        'method': JOINT_SIGNFLIP_METHOD,
        'alpha': float(alpha),
        'family_size': fs,
        'm_eligible': int(m_eligible),
        'n_groups': int(n_groups),
        'family_ready': ready,
        'role': ('secondary correlation-aware sensitivity readout; it does '
                 'not change p_holm, the declared correction, or any frozen '
                 'decision'),
        'groups': group_out,
        'by_key': by_key,
        'excluded': excluded,
        'warnings': warnings,
    }
