"""Ranking resolution across a fixed set of measured arms.

The confirmatory machinery answers pre-registered *pairwise* questions: "is
arm A worse than arm B on this endpoint?"  A report reader immediately asks a
different, unregistered question: **which of the K arms is best, and how much
of that ordering is actually resolved by the seed panel?**  Ranking the
observed seed-mean endpoints and quoting the winner answers that question
without saying that the winner could be a noise artefact.  With K=6, 12 seeds
and differences of a few PPL points, the observed minimum is a selected
extreme: its probability of really being the best can be far below 1 even
when every pairwise comparison is individually "significant".

This fragment adds the missing readout.  It is a **paired seed bootstrap of
the ranking**, not a new test:

* resample the observed seeds with replacement, using the **same resampled
  seed indices for every arm** so the cross-arm pairing (same seed, same data
  order, same evaluation rows) is preserved;
* recompute the seed-mean endpoint for every arm and its rank in every
  replicate;
* report ``p_best`` (probability that the arm has rank 1), ``p_top2``,
  the mean/interval of its rank, the observed top-two gap and its interval,
  the full pairwise order-probability matrix, and the leave-one-seed-out
  winner stability;
* classify the panel as ``decisive`` / ``leading`` / ``undifferentiated``
  using declared reporting bands, and cap ``decisive`` when dropping one
  seed changes the winner.

Why the bootstrap is paired
---------------------------
The arms are measured on the *same* seeds.  Resampling each arm's seeds
independently would destroy that design and overstate the spread of the
*difference* (the quantity the pairwise test uses).  A paired resample keeps
seed ``s`` together across arms, exactly like the paired sign-flip test, and
therefore describes the sampling variability of the ranking under the same
design.  It still conditions on the observed finite seed panel: it is not a
claim that the 12 seeds were drawn from a population of seeds, and it does
not resample evaluation documents (that is the separate unit-level/ two-level
readout).

Boundaries
----------
* Not a confirmatory test, not FWER-controlled, no new p-value.  ``p_best``
  is a bootstrap frequency, not a posterior probability and not an exact
  sampling probability.  It never changes an exact sign-flip p, a Holm
  family, ``verified`` or a frozen ``confirmed`` verdict.
* The arm set is **fixed and non-randomised**: arms differ in capacity,
  recipe and compute (the manifest discloses this).  A ranking is a ranking
  of *these trained configurations on this endpoint*, not a causal ordering
  of architectures.
* The winner is still selected on the same data that produced the ranking.
  High ``p_best`` does not replace independent replication; a post-hoc
  selected winner needs a fresh panel before it can be written as a claim.
* ``K < 3`` is ``pairwise_only`` (a two-arm "ranking" is just the existing
  paired comparison); fewer than ``min_seeds`` common seeds is
  ``not_estimable``.  All-identical arms are ``undifferentiated`` rather
  than a fake rank-1 winner.
"""

ARM_RANK_POLICY = {
    'version': 'arm-ranking-1',
    'better': 'min',
    'n_boot': 4000,
    'bootstrap_seed': 20261018,
    'rank_level': 0.95,
    'lead_level': 0.60,
    'min_seeds': 5,
    'min_arms': 2,
    'tie_tol': 0.0,
    'estimand': ('rank of each arm under the seed-mean endpoint; paired seed '
                 'bootstrap keeps the cross-arm pairing of every resample'),
    'role': ('exploratory ranking-resolution readout; it is not a confirmatory '
             'test, is not FWER-controlled, and never changes an exact '
             'sign-flip p, the Holm family or a frozen verdict'),
}

ARM_RANK_STATUSES = (
    'decisive', 'leading', 'undifferentiated', 'pairwise_only', 'not_estimable')


def _rank_seed_label(value):
    """An exact integer seed label from an int, integral float or int string."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError('arm_rank_inference: seed labels must be integers, not booleans')
    if isinstance(value, (bytes, bytearray)):
        raise ValueError('arm_rank_inference: seed labels must be integers, not bytes')
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lstrip('+-').isdigit() is False:
            raise ValueError(
                f'arm_rank_inference: seed label {value!r} is not an integer')
        try:
            return int(text)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(
                f'arm_rank_inference: seed label {value!r} is not an integer') from None
    try:
        as_int = int(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            f'arm_rank_inference: seed label {value!r} is not an integer') from None
    try:
        if float(value) != float(as_int):
            raise ValueError(
                f'arm_rank_inference: seed label {value!r} is not an integer')
    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            f'arm_rank_inference: seed label {value!r} is not an integer') from None
    return as_int


def _rank_value(value, where):
    """A finite real measurement; booleans, strings and non-finites are refused."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'arm_rank_inference: {where} is a boolean; a flag is not a measurement')
    if isinstance(value, (complex, np.complexfloating)):
        raise ValueError(f'arm_rank_inference: {where} is complex, not a measurement')
    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f'arm_rank_inference: {where} is a string, not a measurement')
    try:
        as_float = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'arm_rank_inference: {where} is not numeric') from None
    if not math.isfinite(as_float):
        raise ValueError(f'arm_rank_inference: {where} is not finite')
    return as_float


def _rank_sequence(value):
    """Materialise an arm's positional values, rejecting scalars/strings."""
    if isinstance(value, (str, bytes, bytearray, dict)):
        raise ValueError(
            'arm_rank_inference: an arm value must be a sequence of per-seed '
            'numbers or a seed->number mapping, not a string/dict-as-scalar')
    try:
        return list(value)
    except TypeError:
        raise ValueError(
            'arm_rank_inference: an arm value must be a sequence of per-seed '
            'numbers or a seed->number mapping') from None


def _rank_positional(values, arm, seeds):
    seq = _rank_sequence(values)
    if seeds is None:
        labels = list(range(len(seq)))
    else:
        if len(seq) != len(seeds):
            raise ValueError(
                f'arm_rank_inference: arm {arm!r} has {len(seq)} values for '
                f'{len(seeds)} declared seeds')
        labels = list(seeds)
    if not seq:
        raise ValueError(f'arm_rank_inference: arm {arm!r} has no values')
    out = {}
    for label, raw in zip(labels, seq):
        out[label] = _rank_value(raw, f'arm {arm!r} seed {label}')
    if len(out) != len(labels):
        raise ValueError(f'arm_rank_inference: arm {arm!r} repeats a seed label')
    return out


def _rank_mapping(values, arm):
    if not values:
        raise ValueError(f'arm_rank_inference: arm {arm!r} has no seed values')
    out = {}
    for raw_label, raw in values.items():
        label = _rank_seed_label(raw_label)
        if label in out:
            raise ValueError(
                f'arm_rank_inference: arm {arm!r} repeats seed label {label}')
        out[label] = _rank_value(raw, f'arm {arm!r} seed {label}')
    return out


def _rank_quantile(sorted_values, q):
    """Order-statistic quantile with no interpolation beyond the data."""
    n = len(sorted_values)
    if n <= 0:
        return None
    if q <= 0.0:
        return sorted_values[0]
    if q >= 1.0:
        return sorted_values[-1]
    idx = int(math.floor(q * n))
    if idx >= n:
        idx = n - 1
    return sorted_values[idx]


def arm_rank_inference(arm_values, *, better=None, seeds=None,
                       n_boot=None, seed=None, rank_level=None,
                       lead_level=None, tie_tol=None, min_seeds=None):
    """Paired seed-bootstrap ranking resolution for a fixed set of arms.

    ``arm_values`` is a mapping ``arm -> sequence`` or ``arm -> {seed: value}``
    (a sequence of ``(arm, values)`` pairs is also accepted).  Sequences are
    positional and must have the same length; seed-keyed mappings are
    intersected across arms and the dropped labels are reported, so a missing
    cell cannot silently shrink the panel without a reason.

    Lower endpoint values are better by default (``better='min'``, the PPL
    convention); ``better='max'`` orients the ranking the other way.  The
    return value is a JSON-serialisable dict; see ``ARM_RANK_POLICY`` and the
    module docstring for the estimand and the boundaries.  This helper adds no
    test: ``p_best``/``p_top2``/rank intervals are bootstrap descriptions of
    the observed seed panel, and the status bands are reporting thresholds,
    not a decision rule.
    """
    policy = ARM_RANK_POLICY
    _better = policy['better'] if better is None else better
    if not isinstance(_better, str) or _better not in ('min', 'max'):
        raise ValueError(
            f"arm_rank_inference: better must be 'min' or 'max', got {better!r}")
    _nb = policy['n_boot'] if n_boot is None else _as_seed_int(n_boot)
    if _nb is None or _nb < 200 or _nb > 200000:
        raise ValueError(
            f'arm_rank_inference: n_boot must be an integer in [200, 200000], '
            f'got {n_boot!r}')
    _sd = policy['bootstrap_seed'] if seed is None else _as_seed_int(seed)
    if _sd is None or _sd < 0:
        raise ValueError(
            f'arm_rank_inference: seed must be a non-negative integer, got {seed!r}')
    _rank_level = policy['rank_level'] if rank_level is None else rank_level
    _lead_level = policy['lead_level'] if lead_level is None else lead_level
    if (isinstance(_rank_level, (bool, np.bool_))
            or isinstance(_lead_level, (bool, np.bool_))):
        raise ValueError('arm_rank_inference: rank/lead levels must be probabilities')
    try:
        _rank_level = float(_rank_level)
        _lead_level = float(_lead_level)
    except (TypeError, ValueError, OverflowError):
        raise ValueError('arm_rank_inference: rank/lead levels must be probabilities') from None
    if not (0.0 < _lead_level <= _rank_level < 1.0):
        raise ValueError(
            'arm_rank_inference: require 0 < lead_level <= rank_level < 1, '
            f'got lead={_lead_level!r}, rank={_rank_level!r}')
    _tie_tol = policy['tie_tol'] if tie_tol is None else tie_tol
    if isinstance(_tie_tol, (bool, np.bool_)):
        raise ValueError('arm_rank_inference: tie_tol must be a non-negative number')
    try:
        _tie_tol = float(_tie_tol)
    except (TypeError, ValueError, OverflowError):
        raise ValueError('arm_rank_inference: tie_tol must be a non-negative number') from None
    if not math.isfinite(_tie_tol) or _tie_tol < 0.0:
        raise ValueError('arm_rank_inference: tie_tol must be a finite non-negative number')
    _min_seeds = policy['min_seeds'] if min_seeds is None else _as_seed_int(min_seeds)
    if _min_seeds is None or _min_seeds < 2:
        raise ValueError(
            f'arm_rank_inference: min_seeds must be an integer >= 2, got {min_seeds!r}')

    if isinstance(arm_values, dict):
        raw_items = list(arm_values.items())
    else:
        try:
            raw_items = list(arm_values)
        except TypeError:
            raise ValueError(
                'arm_rank_inference: arm_values must be a mapping or a sequence '
                'of (arm, values) pairs') from None
    if len(raw_items) < policy['min_arms']:
        return {'kind': 'arm_rank_inference', 'version': policy['version'],
                'status': 'not_estimable', 'better': _better,
                'n_arms': len(raw_items), 'n_seeds': 0,
                'reason': (f'at least {policy["min_arms"]} arms are required for a '
                           f'ranking, got {len(raw_items)}'),
                'policy': {'rank_level': _rank_level, 'lead_level': _lead_level,
                           'min_seeds': int(_min_seeds), 'n_boot': int(_nb),
                           'bootstrap_seed': int(_sd), 'tie_tol': _tie_tol},
                'boundary': ('ranking resolution only; no confirmatory test, no '
                             'FWER control, no change to frozen decisions')}

    arms = []
    normalised = {}
    modes = set()
    for item in raw_items:
        try:
            arm, values = item
        except (TypeError, ValueError):
            raise ValueError(
                'arm_rank_inference: each entry must be an (arm, values) pair') from None
        if not isinstance(arm, str) or not arm.strip():
            raise ValueError('arm_rank_inference: arm names must be non-empty strings')
        arm = arm.strip()
        if arm in normalised:
            raise ValueError(f'arm_rank_inference: duplicate arm {arm!r}')
        arms.append(arm)
        if isinstance(values, dict):
            modes.add('mapping')
            normalised[arm] = _rank_mapping(values, arm)
        else:
            modes.add('sequence')
            normalised[arm] = {'__positions__': values}
    if len(modes) > 1:
        raise ValueError(
            'arm_rank_inference: do not mix seed-keyed mappings and positional '
            'sequences in one call')
    arms.sort()
    mode = modes.pop()

    if mode == 'sequence':
        if seeds is not None:
            if isinstance(seeds, (str, bytes, bytearray, dict)):
                raise ValueError(
                    'arm_rank_inference: seeds must be a sequence of integer labels')
            try:
                seed_list = [_rank_seed_label(s) for s in seeds]
            except TypeError:
                raise ValueError(
                    'arm_rank_inference: seeds must be a sequence of integer labels') from None
            if len(set(seed_list)) != len(seed_list):
                raise ValueError('arm_rank_inference: seeds contains a duplicate label')
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
                    'arm_rank_inference: positional arms must have the same length; '
                    f'got {sorted(lengths)}')
            seed_list = list(range(lengths.pop()))
        dropped = {arm: [] for arm in arms}
        complete_coverage = True
    else:
        if seeds is not None:
            raise ValueError(
                'arm_rank_inference: pass either seed-keyed mappings or an '
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
    base = {
        'kind': 'arm_rank_inference',
        'version': policy['version'],
        'better': _better,
        'n_arms': int(n_arms),
        'n_seeds': int(n_seeds),
        'arms': list(arms),
        'seeds': [int(s) for s in seed_list],
        'complete_coverage': bool(complete_coverage),
        'dropped_seeds': {arm: [int(s) for s in dropped[arm]] for arm in arms},
        'rank_level': _rank_level,
        'lead_level': _lead_level,
        'tie_tol': _tie_tol,
        'min_seeds': int(_min_seeds),
        'n_boot': int(_nb),
        'bootstrap_seed': int(_sd),
        'p_best_resolution': float(1.0 / int(_nb)),
        'inference_scope': ('observed seed panel; paired seed bootstrap preserves '
                            'cross-arm pairing; does not resample evaluation units'),
        'estimand': policy['estimand'],
        'role': policy['role'],
        'boundary': ('exploratory ranking resolution only: no new p-value, no '
                     'FWER control, no change to exact sign-flip p, Holm, '
                     'verified or a frozen confirmation verdict'),
    }
    per_arm_values = {
        arm: {str(int(s)): float(by_seed[arm][s]) for s in seed_list}
        for arm in arms}
    base['per_arm_seed_values'] = per_arm_values
    if n_seeds < 2 or n_arms < 2:
        base['status'] = 'not_estimable'
        base['reason'] = (
            f'a ranking needs >=2 common seeds and >=2 arms, got '
            f'n_seeds={n_seeds}, n_arms={n_arms}')
        return base
    if n_seeds < _min_seeds:
        base['status'] = 'not_estimable'
        base['reason'] = (
            f'{n_seeds} common seed(s) is below the declared min_seeds='
            f'{int(_min_seeds)}; the bootstrap ranking is not reported')
        return base

    # Work on a common affine scale so a finite but extreme panel cannot
    # overflow in the bootstrap means.  Ranking is invariant under a common
    # positive scaling and shift; original-unit readouts are mapped back.
    raw = np.asarray([[by_seed[arm][s] for s in seed_list] for arm in arms],
                     dtype=float)
    absmax = float(np.max(np.abs(raw)))
    if absmax == 0.0:
        absmax = 1.0
    work = raw / absmax
    shift = float(np.min(work))
    work = work - shift
    span = float(np.max(work))
    all_identical = bool(span <= 0.0)
    oriented = work if _better == 'min' else -work
    observed_oriented = oriented.mean(axis=1)
    order = np.argsort(observed_oriented, kind='stable')
    observed_rank = np.empty(n_arms, dtype=int)
    # Competition ranking: exactly tied means share a rank (1, 1, 3, ...),
    # instead of an arbitrary arm-order tie-break pretending to order them.
    _rank = 1
    _sorted_means = observed_oriented[order]
    for _pos, _idx in enumerate(order):
        if _pos > 0 and _sorted_means[_pos] > _sorted_means[_pos - 1]:
            _rank = _pos + 1
        observed_rank[_idx] = _rank
    tied_for_best = [arms[i] for i in range(n_arms)
                     if observed_oriented[i] == _sorted_means[0]]
    mean_tie = len(tied_for_best) > 1
    # The displayed mean is computed from the raw per-seed values with a
    # divide-first fsum, so a finite panel near the float maximum cannot
    # overflow here; the ranking itself uses the scaled work matrix.
    observed_means = {arm: _rank_original_mean(raw[i])
                      for i, arm in enumerate(arms)}
    winner = arms[int(np.argmin(observed_oriented))]
    runner = arms[int(order[1])] if n_arms > 1 else None
    winner_idx = arms.index(winner)
    runner_idx = arms.index(runner) if runner is not None else None
    oriented_gap = (float(observed_oriented[runner_idx] - observed_oriented[winner_idx])
                    if runner is not None else None)
    observed_gap = (oriented_gap * absmax
                    if oriented_gap is not None and math.isfinite(oriented_gap * absmax)
                    else None)

    if all_identical:
        p_best = [1.0 / n_arms] * n_arms
        p_top2 = [min(2.0, float(n_arms)) / n_arms] * n_arms
        mean_rank = [(n_arms + 1) / 2.0] * n_arms
        rank_ci = {arm: [1, n_arms] for arm in arms}
        pairwise = {a: {b: (None if a == b else 0.5) for b in arms} for a in arms}
        loo = {'identical': True, 'winner_stable': False,
               'n_distinct_winners': int(n_arms),
               'dropped_seed_winners': {}}
        base.update({
            'status': 'undifferentiated',
            'status_reason': 'all arms have identical endpoint values; no ranking is defined',
            'observed_means': observed_means,
            'observed_rank': {arm: int(observed_rank[i]) for i, arm in enumerate(arms)},
            'observed_winner': winner,
            'observed_runner_up': runner,
            'observed_tied_for_best': list(arms),
            'observed_gap': 0.0,
            'top2_gap_ci95': [0.0, 0.0],
            'top2_gap_resolved': False,
            'top2_order_prob': 0.5,
            'p_best': {arm: float(p_best[i]) for i, arm in enumerate(arms)},
            'p_top2': {arm: float(p_top2[i]) for i, arm in enumerate(arms)},
            'mean_rank': {arm: float(mean_rank[i]) for i, arm in enumerate(arms)},
            'rank_ci95': rank_ci,
            'p_best_max': 1.0 / n_arms,
            'p_best_argmax': winner,
            'observed_winner_p_best': 1.0 / n_arms,
            'p_best_entropy_normalized': 1.0,
            'effective_best_candidates': float(n_arms),
            'p_best_note': ('all arms are identical; the uniform p_best is a '
                            'reporting convention, not a bootstrap frequency'),
            'pairwise_p_better': pairwise,
            'winner_stable_loo': False,
            'loo': loo,
            'practical_tie': True,
            'n_distinct_bootstrap_winners': int(n_arms),
        })
        return base

    # Paired seed bootstrap: one index vector per replicate, shared by all arms.
    rng = np.random.default_rng(int(_sd))
    boot_means = np.empty((int(_nb), n_arms), dtype=float)
    # Each replicate needs n_seeds indices; size the chunk from both dimensions.
    chunk = max(1, min(int(_nb), (1 << 20) // max(1, n_seeds * n_arms)))
    for off in range(0, int(_nb), chunk):
        m = min(chunk, int(_nb) - off)
        idx = rng.integers(0, n_seeds, size=(m, n_seeds))
        boot_means[off:off + m] = oriented[:, idx].mean(axis=2).T
    boot_ranks = np.argsort(np.argsort(boot_means, axis=1, kind='stable'), axis=1) + 1
    p_best = (boot_ranks == 1).mean(axis=0)
    p_top2 = (boot_ranks <= 2).mean(axis=0)
    mean_rank = boot_ranks.mean(axis=0)
    alpha_lo = 0.025
    rank_ci = {}
    for i, arm in enumerate(arms):
        sorted_r = np.sort(boot_ranks[:, i])
        lo = _rank_quantile(sorted_r.tolist(), alpha_lo)
        hi = _rank_quantile(sorted_r.tolist(), 1.0 - alpha_lo)
        rank_ci[arm] = [int(lo), int(hi)]
    pairwise = {}
    for i, a in enumerate(arms):
        pairwise[a] = {}
        for j, b in enumerate(arms):
            if i == j:
                pairwise[a][b] = None
            else:
                pairwise[a][b] = float((boot_means[:, i] < boot_means[:, j]).mean())

    gap_boot = (boot_means[:, runner_idx] - boot_means[:, winner_idx])
    sorted_gap = np.sort(gap_boot)
    gap_lo = _rank_quantile(sorted_gap.tolist(), alpha_lo)
    gap_hi = _rank_quantile(sorted_gap.tolist(), 1.0 - alpha_lo)
    top2_gap_ci = [
        float(gap_lo * absmax) if math.isfinite(float(gap_lo * absmax)) else None,
        float(gap_hi * absmax) if math.isfinite(float(gap_hi * absmax)) else None]
    top2_order_prob = float((gap_boot > 0.0).mean())
    top2_gap_resolved = bool(top2_gap_ci[0] is not None
                             and float(top2_gap_ci[0]) > 0.0)

    # Leave-one-seed-out winner: a decisive label must survive dropping any
    # single seed, otherwise the observed "winner" is one cell, not a panel.
    loo_winners = {}
    for s in seed_list:
        keep = [i for i, x in enumerate(seed_list) if x != s]
        means_drop = oriented[:, keep].mean(axis=1)
        loo_winners[int(s)] = arms[int(np.argmin(means_drop))]
    distinct_winners = sorted(set(loo_winners.values()))
    winner_stable_loo = distinct_winners == [winner]

    p_best_list = p_best.tolist()
    p_best_max = float(max(p_best_list))
    p_best_argmax = arms[int(np.argmax(p_best_list))]
    winner_p_best = float(p_best[winner_idx])
    probs = np.asarray(p_best_list, dtype=float)
    probs = probs[probs > 0.0]
    entropy = 0.0
    if n_arms > 1 and probs.size:
        entropy = float(-(probs * np.log(probs)).sum() / math.log(n_arms))
        entropy = min(1.0, max(0.0, entropy))
    effective = float(1.0 / float((p_best ** 2).sum())) if (p_best ** 2).sum() > 0 else 0.0
    practical_tie = (True if mean_tie else
                     (None if _tie_tol <= 0.0
                      else (observed_gap is not None and observed_gap <= _tie_tol)))

    if mean_tie:
        status = 'undifferentiated'
        status_reason = ('the top observed means are exactly tied; a winner '
                         'cannot be read from the observed ranking')
    elif n_arms < 3:
        status = 'pairwise_only'
        status_reason = ('two arms: the "ranking" is the existing paired '
                         'comparison with a bootstrap order probability')
    elif (p_best_max >= _rank_level and winner_stable_loo
          and p_best_argmax == winner and top2_gap_resolved):
        status = 'decisive'
        status_reason = (f'observed winner has bootstrap p_best={winner_p_best:.3f} '
                         f'>= {_rank_level:g}, the top-two gap interval excludes '
                         'zero, and every leave-one-seed-out fold keeps the winner')
    elif p_best_max >= _lead_level:
        status = 'leading'
        if p_best_argmax != winner:
            status_reason = (f'p_best_max={p_best_max:.3f} belongs to '
                             f'{p_best_argmax!r}, not the observed winner '
                             f'{winner!r}; the ordering is not decisive')
        elif p_best_max >= _rank_level and not winner_stable_loo:
            status_reason = (f'p_best={p_best_max:.3f} >= {_rank_level:g} but '
                             'dropping one seed changes the observed winner; '
                             'the label is capped at leading')
        elif p_best_max >= _rank_level and not top2_gap_resolved:
            status_reason = (f'p_best={p_best_max:.3f} >= {_rank_level:g} but the '
                             'bootstrap interval for the top-two gap includes '
                             'zero; the label is capped at leading')
        else:
            status_reason = (f'observed winner has bootstrap p_best={winner_p_best:.3f} '
                             f'>= {_lead_level:g} but below the decisive band')
    else:
        status = 'undifferentiated'
        status_reason = (f'no arm has bootstrap p_best >= {_lead_level:g}; the '
                         'seed panel does not resolve a leading arm')

    base.update({
        'status': status,
        'status_reason': status_reason,
        'observed_means': observed_means,
        'observed_rank': {arm: int(observed_rank[i]) for i, arm in enumerate(arms)},
        'observed_winner': winner,
        'observed_runner_up': runner,
        'observed_tied_for_best': tied_for_best,
        'observed_gap': observed_gap,
        'top2_gap_ci95': top2_gap_ci,
        'top2_gap_resolved': top2_gap_resolved,
        'top2_order_prob': top2_order_prob,
        'p_best': {arm: float(p_best[i]) for i, arm in enumerate(arms)},
        'p_top2': {arm: float(p_top2[i]) for i, arm in enumerate(arms)},
        'mean_rank': {arm: float(mean_rank[i]) for i, arm in enumerate(arms)},
        'rank_ci95': {arm: rank_ci[arm] for arm in arms},
        'p_best_max': p_best_max,
        'p_best_argmax': p_best_argmax,
        'observed_winner_p_best': winner_p_best,
        'p_best_entropy_normalized': entropy,
        'effective_best_candidates': effective,
        'pairwise_p_better': pairwise,
        'winner_stable_loo': bool(winner_stable_loo),
        'loo': {'identical': False,
                'winner_stable': bool(winner_stable_loo),
                'n_distinct_winners': int(len(distinct_winners)),
                'dropped_seed_winners': {str(k): v for k, v in loo_winners.items()}},
        'practical_tie': practical_tie,
        'n_distinct_bootstrap_winners': int(len(set(np.argmin(boot_means, axis=1).tolist()))),
    })
    if mean_tie:
        base['p_best_note'] = (
            'the top observed means are exactly tied; p_best/rank readouts '
            'reflect bootstrap tie-breaking, not a resolved ordering')
    return base


def _rank_original_mean(values):
    """Mean of ``values`` in original units, or ``None`` if not representable.

    A naive ``np.mean`` over values near the float maximum can overflow even
    when the mathematical mean is finite.  Dividing each term first keeps the
    intermediate sums bounded; the ranking itself is computed on the scaled
    work matrix and never depends on this display value.
    """
    count = len(values)
    if count <= 0:
        return None
    try:
        total = math.fsum(float(v) / count for v in values)
    except (TypeError, ValueError, OverflowError):
        return None
    return float(total) if math.isfinite(total) else None
