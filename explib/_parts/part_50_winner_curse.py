"""Selection-aware magnitude readout for the winning arm.

The ranking readout (design_notes section 102) answers "how often does each arm
win?" and the evaluation-unit readout (section 103) asks whether the ordering
survives resampling documents.  Neither answers the *magnitude* question that a
reader inevitably asks next: **the reported best arm is the maximum of K noisy
arm means, so how much of its apparent advantage is selection noise?**

A p-value or a rank probability cannot answer that: the observed winner's mean
is a selected extreme even when the ordering is perfectly resolved.  With six
arms and a dozen seeds, the winner's endpoint is biased in the favourable
direction, and the observed top-two gap is a maximum of dependent contrasts.
Quoting either number as an unbiased estimate of "how much better the method
is" is the winner's curse.

This fragment adds two descriptive, resampling-based readouts of that bias:

* **bootstrap optimism** (Efron's optimism for a selection rule).  For each
  paired seed bootstrap replicate, all arms share the resampled seed indices;
  the bootstrap winner is selected, and its bootstrap mean is compared with its
  full-sample mean.  The average difference (apparent minus full-sample) is the
  optimism of this selection procedure.  The optimism-corrected endpoint is
  ``observed winner - optimism``.  A negative optimism (the usual case for
  PPL) means the selected arm looks better on the resample than it does on the
  full panel.
* **split-sample honest evaluation.**  The observed seeds are partitioned into
  two disjoint halves (all partitions when they fit the declared cap, otherwise
  a deterministic sample of them); the winner is selected on one half and
  scored on the other.  This gives an honest held-out value for the *selection
  procedure*, the frequency with which each arm is selected, and the
  **selection regret** (held-out value of the selected arm minus the best
  held-out arm).  Regret is directly comparable with the observed top-two gap:
  if the gap is no larger than the regret, the observed lead is within the
  noise of choosing a winner.  Because a half-panel selects on only ``S/2``
  seeds, its regret is a deliberately conservative check; the status band
  compares the gap with the full-panel *bootstrap* regret, and the
  split-sample regret is reported separately.

Both readouts are descriptive.  A declared ``tie_tol`` (endpoint units) treats
arm means within that margin as a practical tie, so a near-tie is not reported
as a unique winner to correct.  The readouts add no p-value, no FWER control, no
Holm member and never change an exact sign-flip p, a frozen verdict or a
confidence interval.  The split-sample estimate is for the *procedure* (select on this
many seeds, score on that many), not for the identity of the observed winner;
the bootstrap correction is an approximation with a dozen seeds.  The observed
arm set is fixed and non-randomised, so neither quantity is a causal
architecture effect.  A high ``P(best)`` does not turn the corrected value into
an independent replication.
"""

WINNER_CURSE_POLICY = {
    'version': 'winner-curse-1',
    'better': 'min',
    # The bootstrap loop is cheap (S x K per replicate).  Use the same seed as
    # the ranking readout so the selection distribution is comparable.
    'n_boot': 4000,
    'bootstrap_seed': 20261018,
    'split_seed': 20261018,
    # A partition is a size-floor(S/2) subset of the seeds; the complement is
    # the other half and both selection directions are evaluated.  Enumerate
    # every partition when it fits; otherwise draw this many deterministically.
    'max_partitions': 2000,
    'lead_level': 0.60,
    'min_seeds': 5,
    'min_arms': 2,
    # Endpoint-unit practical-tie margin for "there is no unique winner".
    # 0 means only exact equality is a tie; a declared margin lets a panel
    # report a practical tie without pretending the means are identical.
    'tie_tol': 0.0,
    'gap_rule': ('selection_robust requires the observed top-two gap to exceed '
                 'the bootstrap selection regret (the full S-seed rule); the '
                 'split-sample regret selects on only half the seeds and is '
                 'reported as a more conservative independent check'),
    'estimand': ('observed winner endpoint, bootstrap optimism of the '
                 'max-selection rule, optimism-corrected endpoint, and the '
                 'split-sample held-out value/regret of selecting a winner'),
    'role': ('exploratory selection-bias readout; it is not a confirmatory '
             'test, is not FWER-controlled, and never changes an exact '
             'sign-flip p, the Holm family or a frozen verdict'),
}

WINNER_CURSE_STATUSES = (
    'selection_robust', 'selection_noise_dominated', 'winner_unstable',
    'tied_best', 'not_estimable')

_WINNER_CURSE_BOUNDARY = (
    'exploratory selection-bias readout: no new p-value, no FWER control and '
    'no change to p_exact_signflip/p_holm/verified/confirmed.  The bootstrap '
    'optimism is an approximation for a selection rule, not a posterior mean; '
    'the split-sample value is for the procedure, not for the observed '
    'winner.  The arm set is fixed and non-randomised, so the corrected value '
    'is not a causal architecture effect.  A high P(best) is not independent '
    'replication.')


def _wc_percentile(values, q):
    """Linear-interpolation percentile of a finite list; ``None`` if empty."""
    if not values:
        return None
    arr = np.sort(np.asarray(values, dtype=float))
    n = int(arr.size)
    if n == 0:
        return None
    if q <= 0.0:
        return float(arr[0])
    if q >= 1.0:
        return float(arr[-1])
    pos = float(q) * float(n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(arr[lo])
    frac = pos - float(lo)
    value = float(arr[lo]) * (1.0 - frac) + float(arr[hi]) * frac
    return value if math.isfinite(value) else None


def _wc_mean(values):
    """Guarded mean of finite values; ``None`` when nothing is representable."""
    finite = [float(v) for v in values if isinstance(v, (int, float))
              and math.isfinite(float(v))]
    if not finite:
        return None
    n = float(len(finite))
    try:
        total = math.fsum(finite)
    except (OverflowError, ValueError):
        total = None
    if total is not None and math.isfinite(total):
        return total / n
    # A finite panel can still overflow the SUM even when its mean is
    # representable (e.g. 2000 values near 1e308).  Divide first; if the
    # scaled sum also overflows, the mean is not representable.
    try:
        total = math.fsum(v / n for v in finite)
    except (OverflowError, ValueError):
        return None
    return total if math.isfinite(total) else None


def _wc_scale(raw, better):
    """Return ``(work, scale, shift, orientation)`` for stable arithmetic.

    ``work`` is oriented so that smaller is better and, for extreme panels,
    shifted into a small range.  ``scale``/``shift`` invert the transform;
    ``orientation`` is ``+1`` for ``better='min'`` and ``-1`` for
    ``better='max'``.
    """
    oriented = raw if better == 'min' else -raw
    max_abs = float(np.max(np.abs(oriented)))
    orientation = 1.0 if better == 'min' else -1.0
    if max_abs == 0.0:
        return oriented, 1.0, 0.0, orientation
    if max_abs > 1e100 or max_abs < 1e-100:
        scale = max_abs
        scaled = oriented / scale
        shift = float(np.min(scaled))
        return scaled - shift, scale, shift, orientation
    return oriented, 1.0, 0.0, orientation


def _wc_to_raw(value, scale, shift, better):
    """Invert :func:`_wc_scale`; ``None`` if the result is not representable."""
    if value is None:
        return None
    try:
        oriented = (float(value) + float(shift)) * float(scale)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(oriented):
        return None
    raw = oriented if better == 'min' else -oriented
    return raw if math.isfinite(raw) else None


def _wc_diff_to_raw(value, scale):
    """Scale a work-unit difference back to endpoint units."""
    if value is None:
        return None
    try:
        out = float(value) * float(scale)
    except (TypeError, ValueError, OverflowError):
        return None
    return out if math.isfinite(out) else None


def _wc_not_estimable(reason, **extra):
    out = {
        'kind': 'winner_curse_correction',
        'version': WINNER_CURSE_POLICY['version'],
        'status': 'not_estimable',
        'reason': reason,
        'boundary': _WINNER_CURSE_BOUNDARY,
        'policy': dict(WINNER_CURSE_POLICY),
    }
    out.update(extra)
    return out


def winner_curse_correction(arm_values, *, better=None, seeds=None,
                            n_boot=None, seed=None, split_seed=None,
                            max_partitions=None, lead_level=None,
                            min_seeds=None, tie_tol=None):
    """Estimate how much of the observed winner's advantage is selection noise.

    ``arm_values`` is a mapping ``arm -> sequence`` or ``arm -> {seed: value}``
    (a sequence of ``(arm, values)`` pairs is also accepted), with the same
    validation and seed-intersection rules as :func:`arm_rank_inference`.
    ``better='min'`` is the PPL convention; ``better='max'`` orients the
    endpoint the other way.  The result is a JSON-serialisable dict.

    The returned ``bootstrap`` block reports the observed winner's mean, the
    bootstrap optimism (apparent minus full-sample, in endpoint units; negative
    means the selection looked better than the full panel), its percentile
    interval, the optimism-corrected endpoint, and the bootstrap selection
    regret.  The ``split_sample`` block reports the honest held-out value of
    selecting a winner on one half of the seeds, the selection-frequency
    vector, and the selection regret together with its interval.  ``status`` is
    a reporting band, not a test: it combines the winner's ``p_best`` with the
    observed-gap-versus-regret comparison declared in ``WINNER_CURSE_POLICY``.
    """
    policy = WINNER_CURSE_POLICY
    _better = policy['better'] if better is None else better
    if not isinstance(_better, str) or _better not in ('min', 'max'):
        raise ValueError(
            f"winner_curse_correction: better must be 'min' or 'max', got {better!r}")
    _nb = policy['n_boot'] if n_boot is None else _as_seed_int(n_boot)
    if _nb is None or _nb < 200 or _nb > 200000:
        raise ValueError(
            f'winner_curse_correction: n_boot must be an integer in [200, 200000], '
            f'got {n_boot!r}')
    _sd = policy['bootstrap_seed'] if seed is None else _as_seed_int(seed)
    if _sd is None or _sd < 0:
        raise ValueError(
            f'winner_curse_correction: seed must be a non-negative integer, got {seed!r}')
    _ss = policy['split_seed'] if split_seed is None else _as_seed_int(split_seed)
    if _ss is None or _ss < 0:
        raise ValueError(
            f'winner_curse_correction: split_seed must be a non-negative integer, '
            f'got {split_seed!r}')
    _mp = (policy['max_partitions'] if max_partitions is None
           else _as_seed_int(max_partitions))
    if _mp is None or _mp < 1 or _mp > 100000:
        raise ValueError(
            f'winner_curse_correction: max_partitions must be an integer in '
            f'[1, 100000], got {max_partitions!r}')
    _lead = policy['lead_level'] if lead_level is None else lead_level
    if isinstance(_lead, (bool, np.bool_)):
        raise ValueError('winner_curse_correction: lead_level must be a probability')
    try:
        _lead = float(_lead)
    except (TypeError, ValueError, OverflowError):
        raise ValueError('winner_curse_correction: lead_level must be a probability') from None
    if not (0.0 < _lead < 1.0):
        raise ValueError(
            f'winner_curse_correction: lead_level must be in (0, 1), got {lead_level!r}')
    _min_seeds = policy['min_seeds'] if min_seeds is None else _as_seed_int(min_seeds)
    if _min_seeds is None or _min_seeds < 4:
        raise ValueError(
            f'winner_curse_correction: min_seeds must be an integer >= 4 '
            f'(two non-empty halves), got {min_seeds!r}')
    _tie_tol = policy['tie_tol'] if tie_tol is None else tie_tol
    if isinstance(_tie_tol, (bool, np.bool_)):
        raise ValueError('winner_curse_correction: tie_tol must be a non-negative number')
    try:
        _tie_tol = float(_tie_tol)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            'winner_curse_correction: tie_tol must be a non-negative number') from None
    if not math.isfinite(_tie_tol) or _tie_tol < 0.0:
        raise ValueError(
            'winner_curse_correction: tie_tol must be a finite non-negative number')

    if isinstance(arm_values, dict):
        raw_items = list(arm_values.items())
    else:
        try:
            raw_items = list(arm_values)
        except TypeError:
            raise ValueError(
                'winner_curse_correction: arm_values must be a mapping or a '
                'sequence of (arm, values) pairs') from None
    _base_policy = {
        'version': policy['version'], 'better': _better,
        'n_boot': int(_nb), 'bootstrap_seed': int(_sd),
        'split_seed': int(_ss), 'max_partitions': int(_mp),
        'lead_level': _lead, 'min_seeds': int(_min_seeds),
        'tie_tol': _tie_tol, 'gap_rule': policy['gap_rule'],
    }
    if len(raw_items) < policy['min_arms']:
        return _wc_not_estimable(
            f'at least {policy["min_arms"]} arms are required for a selection '
            f'readout, got {len(raw_items)}', policy=_base_policy,
            n_arms=len(raw_items), n_seeds=0)

    arms = []
    normalised = {}
    modes = set()
    for item in raw_items:
        try:
            arm, values = item
        except (TypeError, ValueError):
            raise ValueError(
                'winner_curse_correction: each entry must be an (arm, values) pair') from None
        if not isinstance(arm, str) or not arm.strip():
            raise ValueError('winner_curse_correction: arm names must be non-empty strings')
        arm = arm.strip()
        if arm in normalised:
            raise ValueError(f'winner_curse_correction: duplicate arm {arm!r}')
        arms.append(arm)
        if isinstance(values, dict):
            modes.add('mapping')
            normalised[arm] = _rank_mapping(values, arm)
        else:
            modes.add('sequence')
            normalised[arm] = {'__positions__': values}
    if len(modes) > 1:
        raise ValueError(
            'winner_curse_correction: do not mix seed-keyed mappings and '
            'positional sequences in one call')
    arms.sort()
    mode = modes.pop()

    if mode == 'sequence':
        if seeds is not None:
            if isinstance(seeds, (str, bytes, bytearray, dict)):
                raise ValueError(
                    'winner_curse_correction: seeds must be a sequence of integer labels')
            try:
                seed_list = [_rank_seed_label(s) for s in seeds]
            except TypeError:
                raise ValueError(
                    'winner_curse_correction: seeds must be a sequence of integer labels') from None
            if len(set(seed_list)) != len(seed_list):
                raise ValueError('winner_curse_correction: seeds contains a duplicate label')
        else:
            seed_list = None
        by_seed = {}
        for arm in arms:
            by_seed[arm] = _rank_positional(
                normalised[arm].pop('__positions__'), arm, seed_list)
        if seed_list is None:
            lengths = {len(v) for v in by_seed.values()}
            if len(lengths) != 1:
                raise ValueError(
                    'winner_curse_correction: positional arms must have the same '
                    f'length; got {sorted(lengths)}')
            seed_list = list(range(lengths.pop()))
        dropped = {arm: [] for arm in arms}
        complete_coverage = True
    else:
        if seeds is not None:
            raise ValueError(
                'winner_curse_correction: pass either seed-keyed mappings or an '
                'explicit seeds list, not both')
        by_seed = normalised
        common = set(by_seed[arms[0]])
        for arm in arms[1:]:
            common &= set(by_seed[arm])
        seed_list = sorted(common)
        dropped = {arm: sorted(set(by_seed[arm]) - set(seed_list)) for arm in arms}
        complete_coverage = not any(dropped[arm] for arm in arms)

    n_seeds = len(seed_list)
    n_arms = len(arms)
    _identity = {
        'n_arms': int(n_arms), 'n_seeds': int(n_seeds),
        'arms': list(arms), 'seeds': [int(s) for s in seed_list],
        'complete_coverage': bool(complete_coverage),
        'dropped_seeds': {a: [int(s) for s in dropped[a]] for a in arms},
        'per_arm_seed_values': {
            a: {str(int(s)): float(by_seed[a][s]) for s in seed_list}
            for a in arms},
    }
    if n_seeds < 2 or n_arms < 2:
        return _wc_not_estimable(
            f'a selection readout needs >=2 common seeds and >=2 arms, got '
            f'n_seeds={n_seeds}, n_arms={n_arms}', policy=_base_policy,
            **_identity)
    if n_seeds < _min_seeds:
        return _wc_not_estimable(
            f'{n_seeds} common seed(s) is below the declared min_seeds='
            f'{int(_min_seeds)}; the split-sample halves would be too small',
            policy=_base_policy, **_identity)

    raw = np.asarray([[by_seed[a][s] for a in arms] for s in seed_list], dtype=float)
    work, scale, shift, orientation = _wc_scale(raw, _better)
    observed_work = work.mean(axis=0)
    best_work = float(np.min(observed_work))
    # Practical ties are declared in endpoint units; the comparison is done on
    # the work scale so an extreme panel does not need a huge threshold.
    _tie_work = (_tie_tol / scale) if scale > 0.0 else _tie_tol
    tied_idx = [i for i in range(n_arms)
                if abs(float(observed_work[i]) - best_work) <= _tie_work]
    order = np.argsort(observed_work, kind='stable')
    winner_idx = int(order[0])
    runner_idx = int(order[1])
    winner = arms[winner_idx]
    runner = arms[runner_idx]
    observed_winner_value = _wc_to_raw(observed_work[winner_idx], scale, shift, _better)
    observed_runner_value = _wc_to_raw(observed_work[runner_idx], scale, shift, _better)
    observed_gap = _wc_diff_to_raw(
        float(observed_work[runner_idx] - observed_work[winner_idx]), scale)
    observed_means = {
        a: _wc_to_raw(observed_work[k], scale, shift, _better)
        for k, a in enumerate(arms)}

    # Paired seed bootstrap of the same selection rule.  One seed-index set is
    # shared by every arm, preserving the cross-arm pairing.
    rng = np.random.default_rng(int(_sd))
    p_counts = {a: 0 for a in arms}
    optimism = []
    conditional_optimism = []
    bootstrap_regret = []
    for _ in range(int(_nb)):
        idx = rng.integers(0, n_seeds, size=n_seeds)
        vstar = work[idx].mean(axis=0)
        wstar = int(np.argmin(vstar))
        p_counts[arms[wstar]] += 1
        truth = float(observed_work[wstar])
        opt = _wc_diff_to_raw(
            orientation * (float(vstar[wstar]) - truth), scale)
        if opt is not None:
            optimism.append(opt)
            if wstar == winner_idx:
                conditional_optimism.append(opt)
        regret = _wc_diff_to_raw(float(truth - best_work), scale)
        if regret is not None:
            bootstrap_regret.append(regret)
    p_best = {a: float(p_counts[a]) / float(_nb) for a in arms}
    opt_mean = _wc_mean(optimism)
    opt_ci = [_wc_percentile(optimism, 0.025), _wc_percentile(optimism, 0.975)]
    cond_opt_mean = _wc_mean(conditional_optimism)
    boot_regret_mean = _wc_mean(bootstrap_regret)
    boot_regret_ci = [_wc_percentile(bootstrap_regret, 0.025),
                      _wc_percentile(bootstrap_regret, 0.975)]
    corrected = (observed_winner_value - opt_mean
                 if (observed_winner_value is not None and opt_mean is not None)
                 else None)
    if corrected is not None and not math.isfinite(corrected):
        corrected = None

    # Split-sample honest evaluation: select on one disjoint seed half, score
    # the selected arm on the other.  All partitions are enumerated when they
    # fit; otherwise a deterministic sample is drawn with ``split_seed``.
    k = n_seeds // 2
    parts = []
    n_comb = int(comb(n_seeds, k))
    if n_comb <= int(_mp):
        parts = list(itertools.combinations(range(n_seeds), k))
    else:
        rng2 = random.Random(int(_ss))
        seen = set()
        while len(parts) < int(_mp):
            perm = list(range(n_seeds))
            rng2.shuffle(perm)
            part = tuple(sorted(perm[:k]))
            if part in seen:
                continue
            seen.add(part)
            parts.append(part)
    folds = []
    sel_counts = {a: 0 for a in arms}
    for part in parts:
        a_set = set(part)
        a_idx = list(part)
        b_idx = [i for i in range(n_seeds) if i not in a_set]
        if not a_idx or not b_idx:
            continue
        for sel, ev in ((a_idx, b_idx), (b_idx, a_idx)):
            sel_mean = work[sel].mean(axis=0)
            wsel = int(np.argmin(sel_mean))
            ev_mean = work[ev].mean(axis=0)
            honest = float(ev_mean[wsel])
            regret = float(honest - np.min(ev_mean))
            folds.append((honest, regret))
            sel_counts[arms[wsel]] += 1
    n_folds = len(folds)
    if n_folds == 0:
        return _wc_not_estimable(
            'no non-empty split-sample fold could be formed',
            policy=_base_policy, **_identity)
    honest_mean = _wc_to_raw(_wc_mean([h for h, _r in folds]), scale, shift, _better)
    honest_ci_work = [_wc_percentile([h for h, _r in folds], 0.025),
                      _wc_percentile([h for h, _r in folds], 0.975)]
    honest_ci = [_wc_to_raw(honest_ci_work[0], scale, shift, _better),
                 _wc_to_raw(honest_ci_work[1], scale, shift, _better)]
    regret_mean = _wc_diff_to_raw(_wc_mean([r for _h, r in folds]), scale)
    regret_ci = [_wc_diff_to_raw(_wc_percentile([r for _h, r in folds], 0.025), scale),
                 _wc_diff_to_raw(_wc_percentile([r for _h, r in folds], 0.975), scale)]
    selection_frequency = {a: float(sel_counts[a]) / float(n_folds) for a in arms}
    regret_share = None
    if (observed_gap is not None and regret_mean is not None
            and observed_gap > 0.0 and math.isfinite(observed_gap)):
        regret_share = float(regret_mean / observed_gap)
        if not math.isfinite(regret_share):
            regret_share = None
    gap_exceeds_split = (bool(observed_gap > regret_mean)
                         if (observed_gap is not None and regret_mean is not None)
                         else None)
    gap_within_ci = (bool(observed_gap <= regret_ci[1])
                     if (observed_gap is not None and regret_ci[1] is not None)
                     else None)
    # The status compares the observed gap with the BOOTSTRAP regret, because
    # that resamples the full S-seed panel and therefore matches the selection
    # rule being reported.  The split-sample rule selects on only S/2 seeds and
    # is a deliberately more conservative, independent check.
    boot_regret_share = None
    if (observed_gap is not None and boot_regret_mean is not None
            and observed_gap > 0.0 and math.isfinite(observed_gap)):
        boot_regret_share = float(boot_regret_mean / observed_gap)
        if not math.isfinite(boot_regret_share):
            boot_regret_share = None
    gap_exceeds = (bool(observed_gap > boot_regret_mean)
                   if (observed_gap is not None and boot_regret_mean is not None)
                   else None)
    corrected_minus_honest = None
    if corrected is not None and honest_mean is not None:
        corrected_minus_honest = orientation * (corrected - honest_mean)
        if not math.isfinite(corrected_minus_honest):
            corrected_minus_honest = None
    boot_vs_split = None
    if boot_regret_mean is not None and regret_mean is not None:
        boot_vs_split = boot_regret_mean - regret_mean
        if not math.isfinite(boot_vs_split):
            boot_vs_split = None

    observed_winner_p_best = p_best.get(winner)
    if len(tied_idx) > 1:
        status = 'tied_best'
        reason = (f'{len(tied_idx)} arms share the best observed endpoint '
                  f'({", ".join(arms[i] for i in tied_idx)}); no unique winner '
                  'to correct')
    elif corrected is None or honest_mean is None:
        status = 'not_estimable'
        reason = ('the optimism-corrected or split-sample endpoint is not '
                  'representable for this panel')
    elif observed_winner_p_best is None or observed_winner_p_best < _lead:
        status = 'winner_unstable'
        reason = (f'P(best) for the observed winner is '
                  f'{observed_winner_p_best!r}, below the declared lead_level='
                  f'{_lead:g}; the winner identity is not resolved')
    elif gap_exceeds is None:
        status = 'not_estimable'
        reason = ('the observed gap or the bootstrap selection regret is not '
                  'representable, so the gap-versus-regret comparison is not '
                  'defined')
    elif gap_exceeds is True:
        status = 'selection_robust'
        reason = ('the observed top-two gap exceeds the bootstrap selection '
                  'regret and the observed winner reaches lead_level; '
                  'selection noise does not explain the point gap')
    else:
        status = 'selection_noise_dominated'
        reason = ('the observed top-two gap does not exceed the bootstrap '
                  'selection regret; the point lead is within the noise of '
                  'choosing a winner')

    return {
        'kind': 'winner_curse_correction',
        'version': policy['version'],
        'status': status,
        'status_reason': reason,
        'better': _better,
        'n_arms': int(n_arms),
        'n_seeds': int(n_seeds),
        'arms': list(arms),
        'seeds': [int(s) for s in seed_list],
        'complete_coverage': bool(complete_coverage),
        'dropped_seeds': {a: [int(s) for s in dropped[a]] for a in arms},
        'policy': _base_policy,
        'observed_means': observed_means,
        'observed_winner': winner,
        'observed_runner_up': runner,
        'observed_winner_value': observed_winner_value,
        'observed_runner_up_value': observed_runner_value,
        'observed_gap': observed_gap,
        'observed_tied_for_best': [arms[i] for i in tied_idx],
        'p_best': p_best,
        'observed_winner_p_best': observed_winner_p_best,
        'bootstrap': {
            'n_boot': int(_nb),
            'n_valid': int(len(optimism)),
            'optimism_mean': opt_mean,
            'optimism_ci95': opt_ci,
            'conditional_optimism_mean': cond_opt_mean,
            'n_conditional': int(len(conditional_optimism)),
            'corrected_winner_value': corrected,
            'selection_regret_mean': boot_regret_mean,
            'selection_regret_ci95': boot_regret_ci,
            'regret_share_of_gap': boot_regret_share,
            'observed_gap_exceeds_regret': gap_exceeds,
        },
        'split_sample': {
            'n_partitions': int(len(parts)),
            'n_folds': int(n_folds),
            'honest_value_mean': honest_mean,
            'honest_value_ci95': honest_ci,
            'selection_regret_mean': regret_mean,
            'selection_regret_ci95': regret_ci,
            'regret_share_of_gap': regret_share,
            'observed_gap_exceeds_regret': gap_exceeds_split,
            'gap_within_regret_ci95': gap_within_ci,
            'selection_frequency': selection_frequency,
            'observed_winner_selection_frequency': selection_frequency.get(winner),
        },
        'consistency': {
            'corrected_minus_honest_oriented': corrected_minus_honest,
            'bootstrap_regret_minus_split_regret': boot_vs_split,
            'note': ('positive corrected_minus_honest_oriented means the '
                     'bootstrap-corrected value is worse (higher in the '
                     'better-oriented scale) than the split-sample honest '
                     'value; the two readouts answer related but not identical '
                     'questions and need not agree exactly'),
        },
        'per_arm_seed_values': {
            a: {str(int(s)): float(by_seed[a][s]) for s in seed_list}
            for a in arms},
        'boundary': _WINNER_CURSE_BOUNDARY,
    }
