"""Evaluation-unit sensitivity of a cross-arm ranking.

``arm_rank_inference`` (design_notes section 102) answers "how much of the
observed arm ordering is resolved by the *seed* panel?"  It deliberately
resamples only seeds.  That leaves the other sampling axis of every reported
ranking unexamined: the frozen evaluation slice itself.  A six-arm ordering
that is stable across seeds can still reverse when a handful of documents
carry most of the corpus tokens.  The report currently states that boundary
("it does not resample evaluation documents") but offers no readout for it;
the existing document bootstrap / two-level bootstrap quantify a *mean*
difference, not a *ranking*.

This fragment adds the missing readout.  It is a **paired two-axis bootstrap
of the ranking**, not a new test:

* the input is per arm, per seed, per evaluation unit (article/document) NLL
  and token counts.  The observed per-seed endpoint is the token-pooled PPL,
  and the arm value is the arithmetic mean of those per-seed PPLs, exactly the
  estimand of ``arm_rank_inference``;
* **seed-only** resampling reuses the section-102 design (one seed index set
  shared by every arm) and is reported as a cross-check, not as the whole
  answer;
* **unit-only** resampling draws evaluation units with replacement, using the
  same unit index set for every arm and every seed, so the cross-arm and
  cross-seed pairing is preserved;
* **two-way** resampling draws both axes and is the combined readout;
* each axis reports ``p_best`` / ``p_top2`` / mean rank / rank interval /
  observed-winner frequency / unique-winner frequency / top-two gap interval,
  so a reader can see whether the ranking is limited by training noise or by
  the particular held-out documents;
* the observed top-two gap variance is split into seed and unit components,
  and a declared status says which axis (if either) fails to reproduce the
  observed winner.

Boundaries
----------
* This is **not** a confirmatory test.  There is no new p-value, no FWER
  control and no Holm member; it never changes an exact sign-flip p,
  ``p_holm``, ``verified`` or a frozen ``confirmed`` verdict.
* ``p_best`` is a bootstrap frequency, not a posterior or exact sampling
  probability.  The percentile intervals can be too narrow at 12 seeds and
  48 units; they are reporting aids, not guarantees.
* The units are the repository's available corpus unit: articles when every
  record carries article cells, otherwise line-level documents.  A line-level
  "document" is a raw WikiText line, so articles spanning several lines are
  pseudoreplicated.  The unit granularity is always reported; a smaller unit
  is never silently substituted.
* Resampling the *observed* units does not create a probability sample of the
  language population.  Below the repository's 64-unit population floor the
  readout is explicitly conditional on the frozen slice, exactly like the
  document bootstrap in design_notes sections 54/81.
* The arm set is fixed and non-randomised, and arms differ in capacity,
  recipe and compute.  This is a ranking of these trained configurations on
  this endpoint and this evaluation slice, not a causal architecture ordering.
* High ``p_best`` under either axis is not independent replication.  The
  winner is still selected on the same data; a rank claim needs a fresh panel
  or a pre-registered replication.
"""

ARM_RANK_UNIT_POLICY = {
    'version': 'arm-rank-units-1',
    'better': 'min',
    # 2000 is enough to resolve the declared 0.90/0.60 bands while keeping the
    # two-axis loop cheap on a CPU-only analysis host.  The bootstrap seed is
    # fixed so the stored block replays bit-for-bit.
    'n_boot': 2000,
    'bootstrap_seed': 20261018,
    'unit_level': 0.90,
    'seed_level': 0.60,
    'lead_level': 0.60,
    'min_seeds': 5,
    'min_units': 8,
    'tie_tol': 0.0,
    'estimand': ('per-seed token-pooled PPL, arithmetic mean across seeds; '
                 'unit resampling is shared across arms and seeds'),
    'role': ('exploratory ranking-sensitivity readout: it separates seed '
             'uncertainty from evaluation-unit uncertainty, is not a '
             'confirmatory test, is not FWER-controlled, and never changes a '
             'frozen verdict'),
}

ARM_RANK_UNIT_STATUSES = (
    'stable', 'seed_sensitive', 'unit_sensitive', 'both_sensitive',
    'undifferentiated', 'not_estimable')

_ARM_RANK_UNIT_BOUNDARY = (
    'exploratory only: no new p-value, no FWER control, no Holm member and no '
    'change to p_exact_signflip/p_holm/verified/confirmed.  The observed '
    'units are resampled, not drawn from a population; n_units below 64 does '
    'not support a population-level claim.  p_best is a bootstrap frequency, '
    'not a posterior or exact sampling probability.  The arm set is fixed and '
    'non-randomised, so the ordering is not a causal architecture ordering.'
)


def _aru_seed_int(value):
    """Normalise a seed key to a Python int; return None if it is not one."""
    if isinstance(value, (bool,)) or value is None:
        return None
    try:
        import numbers
        import numpy as _np
    except Exception:  # pragma: no cover - numpy is a hard dependency
        numbers = None
        _np = None
    if numbers is not None and isinstance(value, numbers.Integral):
        return int(value)
    if _np is not None and isinstance(value, _np.integer):
        return int(value)
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        _s = value.strip()
        if not _s:
            return None
        try:
            _f = float(_s)
        except (TypeError, ValueError, OverflowError):
            return None
        return int(_f) if _f.is_integer() else None
    return None


def _aru_float_list(value, *, name):
    """A finite numeric vector from a JSON list/tuple/ndarray."""
    if isinstance(value, (str, bytes, bytearray, dict)) or value is None:
        raise ValueError(f'{name} must be a numeric sequence, got {type(value).__name__}')
    try:
        raw = list(value)
    except TypeError:
        raise ValueError(f'{name} must be a numeric sequence') from None
    if not raw:
        raise ValueError(f'{name} must not be empty')
    out = []
    for i, x in enumerate(raw):
        if (isinstance(x, (bool, str, bytes, bytearray, complex))
                or x is None):
            raise ValueError(f'{name}[{i}] must be a finite number, got {x!r}')
        try:
            f = float(x)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f'{name}[{i}] is not numeric: {x!r}') from None
        if not _aru_finite(f):
            raise ValueError(f'{name}[{i}] is not finite: {x!r}')
        out.append(f)
    return out


def _aru_finite(value):
    import math
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _aru_not_estimable(reason, *, kind='arm_rank_unit_sensitivity'):
    return {'kind': kind, 'status': 'not_estimable', 'reason': str(reason),
            'role': ARM_RANK_UNIT_POLICY['role'],
            'boundary': _ARM_RANK_UNIT_BOUNDARY}


def _aru_normalise(arm_units):
    """Validate the nested payload and return arms, seeds, nll, tokens, ids."""
    if not isinstance(arm_units, dict):
        raise ValueError('arm_units must be a mapping arm -> seed -> cell')
    if not arm_units:
        raise ValueError('arm_units must not be empty')
    per_arm = {}
    order = []
    for arm, rows in arm_units.items():
        if not isinstance(arm, str) or not arm.strip():
            raise ValueError(f'arm names must be non-empty strings, got {arm!r}')
        if not isinstance(rows, dict) or not rows:
            raise ValueError(f'arm_units[{arm!r}] must be a non-empty seed mapping')
        seed_map = {}
        for raw_seed, rec in rows.items():
            seed = _aru_seed_int(raw_seed)
            if seed is None:
                raise ValueError(f'arm_units[{arm!r}] has a non-integer seed {raw_seed!r}')
            if seed in seed_map:
                raise ValueError(f'arm_units[{arm!r}] has duplicate seed {seed}')
            if not isinstance(rec, dict):
                raise ValueError(f'arm_units[{arm!r}][{seed}] must be a cell object')
            tokens = _aru_float_list(rec.get('tokens'),
                                     name=f'arm_units[{arm!r}][{seed}].tokens')
            if any(t <= 0.0 for t in tokens):
                raise ValueError(f'arm_units[{arm!r}][{seed}].tokens must be positive')
            nll_raw = rec.get('nll')
            if nll_raw is None and rec.get('ppl') is not None:
                import math
                ppl = _aru_float_list(rec.get('ppl'),
                                      name=f'arm_units[{arm!r}][{seed}].ppl')
                if len(ppl) != len(tokens):
                    raise ValueError(f'arm_units[{arm!r}][{seed}] ppl/tokens lengths differ')
                nll = []
                for p, t in zip(ppl, tokens):
                    if p <= 0.0:
                        raise ValueError(f'arm_units[{arm!r}][{seed}].ppl must be positive')
                    nll.append(math.log(p) * t)
            else:
                nll = _aru_float_list(nll_raw,
                                      name=f'arm_units[{arm!r}][{seed}].nll')
            if len(nll) != len(tokens):
                raise ValueError(f'arm_units[{arm!r}][{seed}] nll/tokens lengths differ')
            ids = rec.get('unit_ids')
            if ids is not None:
                if isinstance(ids, (str, bytes, bytearray, dict)):
                    raise ValueError(f'arm_units[{arm!r}][{seed}].unit_ids must be a sequence')
                try:
                    ids = [str(x) for x in ids]
                except TypeError:
                    raise ValueError(f'arm_units[{arm!r}][{seed}].unit_ids must be a sequence') from None
                if len(ids) != len(tokens):
                    raise ValueError(f'arm_units[{arm!r}][{seed}].unit_ids length differs')
                if len(set(ids)) != len(ids):
                    raise ValueError(f'arm_units[{arm!r}][{seed}].unit_ids contains duplicates')
            seed_map[seed] = {'nll': nll, 'tokens': tokens, 'unit_ids': ids}
        per_arm[arm] = seed_map
        order.append(arm)
    return order, per_arm


def _aru_common_seeds(per_arm):
    sets = [set(v) for v in per_arm.values()]
    return sorted(set.intersection(*sets)) if sets else []


def _aru_check_alignment(per_arm, seeds):
    """Return (n_units, tokens, unit_ids, reason)."""
    ref = None
    ids_ref = None
    ids_present = 0
    for arm, seed_map in per_arm.items():
        for seed in seeds:
            cell = seed_map[seed]
            n = len(cell['tokens'])
            if ref is None:
                ref = n
                tokens = list(cell['tokens'])
            elif n != ref:
                return (0, None, None,
                        f'unit counts differ across arms/seeds ({ref} vs {n} for {arm!r}/seed {seed})')
            elif list(cell['tokens']) != list(tokens):
                return (0, None, None,
                        'per-unit token counts differ across arms/seeds; '
                        'the units are not the same denominator')
            if cell['unit_ids'] is not None:
                ids_present += 1
                if ids_ref is None:
                    ids_ref = list(cell['unit_ids'])
                elif list(cell['unit_ids']) != ids_ref:
                    return (0, None, None,
                            'unit id order differs across arms/seeds; paired '
                            'units cannot be identified by position')
    if ids_present and ids_present != len(per_arm) * len(seeds):
        return (0, None, None,
                'unit ids are present for only some arms/seeds; the unit '
                'identity cannot be verified across the whole panel')
    return (ref or 0, tokens if ref is not None else None,
            ids_ref if ids_present else None, None)


def _aru_ranks(values, better):
    """Competition ranks (1,1,3) for a (K, B) array; ties share a rank."""
    import numpy as np
    v = np.asarray(values, dtype=float)
    if v.ndim != 2:
        raise ValueError('rank values must be 2-D')
    if better == 'min':
        better_count = (v[None, :, :] < v[:, None, :]).sum(axis=1)
    else:
        better_count = (v[None, :, :] > v[:, None, :]).sum(axis=1)
    return better_count + 1


def _aru_axis(values, *, arms, better, scale, level):
    """Summarise one bootstrap axis from its (K, B) scaled mean-PPL values."""
    import math
    import numpy as np
    v = np.asarray(values, dtype=float)
    k, b = v.shape
    ranks = _aru_ranks(v, better)
    p_best = (ranks == 1).mean(axis=1)
    p_top2 = (ranks <= 2).mean(axis=1)
    mean_rank = ranks.mean(axis=1)
    lo = np.percentile(ranks, 2.5, axis=1)
    hi = np.percentile(ranks, 97.5, axis=1)
    rank_ci = []
    for i in range(k):
        lo_i = max(1, int(math.floor(float(lo[i]) + 1e-9)))
        hi_i = min(k, int(math.ceil(float(hi[i]) - 1e-9)))
        if hi_i < lo_i:
            hi_i = lo_i
        rank_ci.append([lo_i, hi_i])
    # Entropy of the winner distribution: 0 = one arm wins every replicate,
    # log(K) = all arms equally likely.
    safe = np.clip(p_best, 0.0, 1.0)
    nz = safe > 0.0
    entropy = float(-(safe[nz] * np.log(safe[nz])).sum())
    ent_norm = (entropy / math.log(k)) if k > 1 else 0.0
    eff = float(math.exp(entropy)) if math.isfinite(entropy) else None
    winner = ranks == 1
    p_unique = float((winner.sum(axis=0) == 1).mean())
    winner_sets = {tuple(np.flatnonzero(winner[:, j])) for j in range(b)}
    # The observed winner is supplied out of band; p_obs is filled by the
    # caller because it depends on the observed ranking, not this axis.
    sv = np.sort(v, axis=0)
    gap = None
    gap_units = None
    if k >= 2:
        if better == 'min':
            gap_scaled = sv[1] - sv[0]
        else:
            gap_scaled = sv[-1] - sv[-2]
        # The axis stores scaled mean PPL (ranking is invariant to the common
        # exp(max_log) factor).  Report the gap in original PPL units when
        # that factor is representable, otherwise in the common scaled units.
        if scale is not None and math.isfinite(scale):
            gap = np.asarray(gap_scaled * scale, dtype=float)
            gap_units = 'ppl'
        else:
            gap = np.asarray(gap_scaled, dtype=float)
            gap_units = 'scaled_ppl'

        gap_mean = float(gap.mean()) if np.all(np.isfinite(gap)) else None
        gap_sd = ((float(gap.std(ddof=1)) if gap.size > 1 else
                   (0.0 if gap.size == 1 else None))
                  if np.all(np.isfinite(gap)) else None)
        g_lo, g_hi = (np.percentile(gap, [2.5, 97.5]).tolist()
                      if gap.size and np.all(np.isfinite(gap)) else [None, None])
        gap_ci = ([float(g_lo), float(g_hi)]
                  if g_lo is not None and math.isfinite(float(g_lo))
                  and math.isfinite(float(g_hi)) else None)
        gap_pos = (float((gap > 0.0).mean())
                   if gap.size and np.all(np.isfinite(gap)) else None)
    else:
        gap_mean = gap_sd = gap_pos = None
        gap_ci = None
        gap_units = None
    return {
        'n_boot': int(b),
        'p_best': {arms[i]: float(p_best[i]) for i in range(k)},
        'p_top2': {arms[i]: float(p_top2[i]) for i in range(k)},
        'mean_rank': {arms[i]: float(mean_rank[i]) for i in range(k)},
        'rank_ci95': {arms[i]: rank_ci[i] for i in range(k)},
        'rank_sd': {arms[i]: float(ranks[i].std(ddof=1)) if b > 1 else 0.0
                    for i in range(k)},
        'p_best_max': float(p_best.max()) if k else None,
        'p_best_argmax': arms[int(np.argmax(p_best))] if k else None,
        'p_best_entropy_normalized': float(ent_norm),
        'effective_best_candidates': eff,
        'p_unique_winner': p_unique,
        'n_distinct_winner_sets': int(len(winner_sets)),
        'top2_gap_mean': gap_mean,
        'top2_gap_sd': gap_sd,
        'top2_gap_ci95': gap_ci,
        'top2_gap_positive_prob': gap_pos,
        'top2_gap_units': gap_units,
        'level': float(level),
    }


def _aru_observed(values, *, arms, better, scale, log_offset, tie_tol):
    """Observed arm values/ranks/gap from scaled arithmetic-mean PPLs."""
    import math
    import numpy as np
    v = np.asarray(values, dtype=float)
    k = len(arms)
    ranks = [1 + int(sum((v[j] < v[i]) if better == 'min'
                         else (v[j] > v[i]) for j in range(k)))
             for i in range(k)]
    best = min(v) if better == 'min' else max(v)
    best_idx = [i for i in range(k) if v[i] == best]
    order = sorted(range(k), key=lambda i: (v[i] if better == 'min' else -v[i], arms[i]))
    winner = arms[order[0]]
    runner = None
    if k >= 2:
        for i in order[1:]:
            if v[i] != best:
                runner = arms[i]
                break
    if scale is not None and math.isfinite(scale):
        means = {}
        log_means = {}
        for i in range(k):
            m = float(scale * v[i])
            if math.isfinite(m):
                means[arms[i]] = m
                log_means[arms[i]] = (float(math.log(v[i]) + log_offset)
                                      if v[i] > 0.0 and math.isfinite(log_offset)
                                      else None)
            else:
                means[arms[i]] = None
                log_means[arms[i]] = None
    else:
        means = {arms[i]: None for i in range(k)}
        log_means = {arms[i]: None for i in range(k)}
    gap = None
    gap_units = None
    if runner is not None:
        ri = arms.index(runner)
        if scale is not None and math.isfinite(scale):
            gap = abs(float(scale * (v[order[0]] - v[ri])))
            gap_units = 'ppl'
        else:
            gap = abs(float(v[order[0]] - v[ri]))
            gap_units = 'scaled_ppl'
    practical_tie = bool(gap is not None and tie_tol > 0.0 and gap <= tie_tol)
    return {
        'observed_means': means,
        'observed_log_means': log_means,
        'observed_scaled_means': {arms[i]: float(v[i]) for i in range(k)},
        'observed_rank': {arms[i]: int(ranks[i]) for i in range(k)},
        'observed_winner': winner,
        'observed_runner_up': runner,
        'observed_tied_for_best': [arms[i] for i in best_idx],
        'observed_gap': gap,
        'observed_gap_units': gap_units,
        'practical_tie': practical_tie,
    }


def _aru_status(observed, seed_axis, unit_axis, two_axis, *,
                seed_level, unit_level, lead_level):
    """Declared band describing which axis reproduces the observed winner."""
    if len(observed['observed_tied_for_best']) != 1:
        return 'undifferentiated', (
            'the observed top value is shared by '
            f'{observed["observed_tied_for_best"]}; there is no unique winner')
    w = observed['observed_winner']
    if max(unit_axis['p_best'].values()) < lead_level:
        return 'undifferentiated', (
            f'no arm is the most frequent unit-bootstrap winner at the '
            f'declared lead level {lead_level}')
    seed_ok = seed_axis['p_best'].get(w, 0.0) >= seed_level
    unit_ok = (unit_axis['p_best'].get(w, 0.0) >= unit_level
               and unit_axis['p_best_argmax'] == w)
    two_ok = (two_axis['p_best'].get(w, 0.0) >= unit_level
              and two_axis['p_best_argmax'] == w)
    failed = []
    if not seed_ok:
        failed.append('seed')
    if not unit_ok:
        failed.append('unit')
    if not two_ok:
        failed.append('two_way')
    if not failed:
        return 'stable', (
            f'observed winner {w!r} leads under seed, unit and two-way '
            f'resampling at the declared bands')
    if not seed_ok and (not unit_ok or not two_ok):
        status = 'both_sensitive'
    elif not seed_ok:
        status = 'seed_sensitive'
    else:
        status = 'unit_sensitive'
    return status, (
        f'observed winner {w!r} fails the declared band on {failed}; '
        f'p_best(seed)={seed_axis["p_best"].get(w)}, '
        f'p_best(unit)={unit_axis["p_best"].get(w)}, '
        f'p_best(two_way)={two_axis["p_best"].get(w)}')


def arm_rank_unit_sensitivity(arm_units, *, better=None, n_boot=None, seed=None,
                              min_seeds=None, min_units=None, unit_level=None,
                              seed_level=None, lead_level=None, tie_tol=None,
                              unit_ids=None, per_arm_seed_values=None,
                              unit_granularity=None):
    """Two-axis bootstrap of a fixed-arm ranking.

    Parameters
    ----------
    arm_units : mapping
        ``{arm: {seed: {'nll': [...], 'tokens': [...]}}}`` (or ``ppl`` in
        place of ``nll``).  Units must align across every arm and seed; a
        per-cell ``unit_ids`` vector is optional but verified when present.
    per_arm_seed_values : mapping, optional
        ``{arm: {seed: ppl}}`` to cross-check that the per-unit matrices
        reproduce the endpoint used by ``arm_rank_inference``.  A mismatch is
        a refusal, not a silent substitution.

    Returns a JSON-serialisable dict; malformed structural input raises
    ``ValueError`` and a legitimate-but-insufficient/unaligned panel returns
    ``status='not_estimable'`` with a reason.
    """
    import math
    import numpy as np
    policy = ARM_RANK_UNIT_POLICY
    better = policy['better'] if better is None else better
    if better not in ('min', 'max'):
        raise ValueError(f'better must be min/max, got {better!r}')
    n_boot = policy['n_boot'] if n_boot is None else n_boot
    seed = policy['bootstrap_seed'] if seed is None else seed
    min_seeds = policy['min_seeds'] if min_seeds is None else min_seeds
    min_units = policy['min_units'] if min_units is None else min_units
    unit_level = policy['unit_level'] if unit_level is None else unit_level
    seed_level = policy['seed_level'] if seed_level is None else seed_level
    lead_level = policy['lead_level'] if lead_level is None else lead_level
    tie_tol = policy['tie_tol'] if tie_tol is None else tie_tol
    for name, value, lo, hi in (
            ('n_boot', n_boot, 1, None),
            ('bootstrap_seed', seed, 0, None),
            ('min_seeds', min_seeds, 1, None),
            ('min_units', min_units, 2, None)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f'{name} must be an integer, got {value!r}')
        if int(value) < lo or (hi is not None and int(value) > hi):
            raise ValueError(f'{name} is out of range: {value!r}')
    for name, value in (('unit_level', unit_level), ('seed_level', seed_level),
                        ('lead_level', lead_level)):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
            raise ValueError(f'{name} must be a probability, got {value!r}')
        if not (0.0 <= float(value) <= 1.0):
            raise ValueError(f'{name} must be in [0,1], got {value!r}')
    if isinstance(tie_tol, bool) or not isinstance(tie_tol, (int, float, np.floating)):
        raise ValueError(f'tie_tol must be a non-negative number, got {tie_tol!r}')
    if float(tie_tol) < 0.0 or not math.isfinite(float(tie_tol)):
        raise ValueError(f'tie_tol must be finite and >=0, got {tie_tol!r}')
    if unit_granularity is not None:
        if unit_granularity not in ('article', 'document', 'row', 'unknown'):
            raise ValueError(
                'unit_granularity must be article/document/row/unknown, got '
                f'{unit_granularity!r}')

    arms, per_arm = _aru_normalise(arm_units)
    if len(arms) < 2:
        return _aru_not_estimable('fewer than two arms; a ranking needs K>=2')
    seeds = _aru_common_seeds(per_arm)
    dropped = {arm: sorted(set(per_arm[arm]) - set(seeds)) for arm in arms}
    if len(seeds) < int(min_seeds):
        return _aru_not_estimable(
            f'only {len(seeds)} seed(s) shared by every arm; min_seeds={min_seeds}',
        )
    n_units, tokens, ids, why = _aru_check_alignment(per_arm, seeds)
    if why:
        return _aru_not_estimable(why)
    if unit_ids is not None:
        try:
            _u = [str(x) for x in unit_ids]
        except TypeError:
            return _aru_not_estimable('unit_ids must be a sequence')
        if len(_u) != n_units:
            return _aru_not_estimable('unit_ids length does not match n_units')
        if len(set(_u)) != len(_u):
            return _aru_not_estimable('unit_ids contains duplicates')
        if ids is not None and _u != ids:
            return _aru_not_estimable('unit_ids argument disagrees with the per-cell ids')
        ids = _u
    if n_units < int(min_units):
        return _aru_not_estimable(
            f'only {n_units} evaluation unit(s); min_units={min_units}')

    # Build the K x S log-PPL matrix from the per-unit NLL/token vectors.
    k, s = len(arms), len(seeds)
    logp = np.empty((k, s), dtype=float)
    for i, arm in enumerate(arms):
        for j, sd in enumerate(seeds):
            cell = per_arm[arm][sd]
            tot = float(math.fsum(cell['tokens']))
            if not (math.isfinite(tot) and tot > 0.0):
                return _aru_not_estimable(
                    f'{arm!r}/seed {sd}: total tokens are not positive/finite')
            _nll = float(math.fsum(cell['nll']))
            if not math.isfinite(_nll):
                return _aru_not_estimable(
                    f'{arm!r}/seed {sd}: NLL sum is not finite')
            logp[i, j] = _nll / tot
    if not np.all(np.isfinite(logp)):
        return _aru_not_estimable('the per-unit NLL/token matrix produced a non-finite per-seed log-PPL')
    max_log = float(np.max(logp))
    z = logp - max_log
    scale = (math.exp(max_log) if max_log <= 709.0 else None)
    observed_seed_values = {}
    for i, arm in enumerate(arms):
        observed_seed_values[arm] = {}
        for j, sd in enumerate(seeds):
            value = (scale * math.exp(float(z[i, j]))
                     if scale is not None else None)
            observed_seed_values[arm][str(sd)] = (
                float(value) if (value is not None and math.isfinite(value)) else None)
    # Cross-check the matrices against the endpoint values used by the
    # section-102 ranking.  This is what makes the unit bootstrap a readout of
    # the SAME estimand rather than a new, possibly different, endpoint.
    if per_arm_seed_values is not None:
        if not isinstance(per_arm_seed_values, dict):
            raise ValueError('per_arm_seed_values must be a mapping arm -> seed -> ppl')
        mismatches = []
        for arm in arms:
            row = per_arm_seed_values.get(arm)
            if not isinstance(row, dict):
                mismatches.append(f'{arm!r}: no seed mapping')
                continue
            norm = {}
            for raw_s, raw_v in row.items():
                sd = _aru_seed_int(raw_s)
                if sd is None:
                    mismatches.append(f'{arm!r}: non-integer seed {raw_s!r}')
                    continue
                norm[sd] = raw_v
            for sd in seeds:
                if sd not in norm:
                    mismatches.append(f'{arm!r}: seed {sd} missing')
                    continue
                try:
                    want = float(norm[sd])
                except (TypeError, ValueError, OverflowError):
                    mismatches.append(f'{arm!r}: seed {sd} is not numeric')
                    continue
                got = observed_seed_values[arm][str(sd)]
                if got is None or not math.isfinite(want) or want <= 0.0:
                    mismatches.append(f'{arm!r}: seed {sd} is not a finite positive PPL')
                    continue
                if not math.isclose(got, want, rel_tol=1e-6, abs_tol=1e-9):
                    mismatches.append(
                        f'{arm!r}: seed {sd} endpoint {want!r} != per-unit '
                        f'reconstruction {got!r}')
        if mismatches:
            return _aru_not_estimable(
                'per-unit cells do not reproduce the stored endpoint values: '
                + '; '.join(mismatches[:6])
                + ('' if len(mismatches) <= 6 else f' (+{len(mismatches)-6} more)'))
        values_match = 'checked'
    else:
        values_match = 'not_checked'

    # Ranking estimand: arithmetic mean of the per-seed PPLs (section 102),
    # not the geometric mean of the per-seed log-PPLs.  exp(z) is the per-seed
    # PPL divided by the common exp(max_log) factor, so ranking it is exact.
    observed_values = np.exp(z).mean(axis=1)
    observed = _aru_observed(observed_values, arms=arms, better=better,
                             scale=scale, log_offset=max_log,
                             tie_tol=float(tie_tol))

    rng = np.random.default_rng(int(seed))
    doc_idx = rng.integers(0, n_units, size=(int(n_boot), n_units))
    seed_idx = rng.integers(0, s, size=(int(n_boot), s))
    tok = np.asarray(tokens, dtype=float)
    nll_mats = {arm: np.asarray([per_arm[arm][sd]['nll'] for sd in seeds],
                                dtype=float) for arm in arms}

    # Seed-only: same design as arm_rank_inference, on the observed units.
    seed_values = np.empty((k, int(n_boot)), dtype=float)
    for i, arm in enumerate(arms):
        seed_values[i] = np.exp(z[i][seed_idx]).mean(axis=1)
    seed_axis = _aru_axis(seed_values, arms=arms, better=better, scale=scale,
                          level=seed_level)
    seed_axis['p_best_observed_winner'] = float(seed_axis['p_best'][observed['observed_winner']])

    # Unit-only: one shared unit index set for every arm and seed.
    unit_values = np.empty((k, int(n_boot)), dtype=float)
    for i, arm in enumerate(arms):
        _nll = nll_mats[arm]
        # (S, B): sum the selected unit NLLs for every seed.
        num = _nll[:, doc_idx].sum(axis=-1)
        den = tok[doc_idx].sum(axis=-1)
        unit_values[i] = np.exp(num / den - max_log).mean(axis=0)
    unit_axis = _aru_axis(unit_values, arms=arms, better=better, scale=scale,
                          level=unit_level)
    unit_axis['p_best_observed_winner'] = float(unit_axis['p_best'][observed['observed_winner']])

    # Two-way: draw a fresh seed index and a fresh unit index per replicate.
    # The loop is over B (not over arms) and uses vectorised sums per seed, so
    # it stays cheap even at the default 2000 replicates.
    two_values = np.empty((k, int(n_boot)), dtype=float)
    for b in range(int(n_boot)):
        _di = doc_idx[b]
        den = float(tok[_di].sum())
        for i, arm in enumerate(arms):
            _nll = nll_mats[arm][:, _di]
            two_values[i, b] = float(
                np.exp(_nll.sum(axis=-1) / den - max_log).mean())
    two_axis = _aru_axis(two_values, arms=arms, better=better, scale=scale,
                         level=unit_level)
    two_axis['p_best_observed_winner'] = float(two_axis['p_best'][observed['observed_winner']])

    status, status_reason = _aru_status(
        observed, seed_axis, unit_axis, two_axis,
        seed_level=seed_level, unit_level=unit_level, lead_level=lead_level)

    # Split the observed top-two gap variance into the two axes.  The gap is
    # in original PPL units when the common scale is representable; otherwise
    # the axis block reports common scaled-PPL units and the shares are still
    # computed from comparable squared standard deviations.
    unit_name = (unit_axis.get('top2_gap_units')
                 or seed_axis.get('top2_gap_units') or 'scaled_ppl')
    var_seed = (seed_axis.get('top2_gap_sd') or 0.0) ** 2
    var_unit = (unit_axis.get('top2_gap_sd') or 0.0) ** 2
    total = var_seed + var_unit
    if total > 0.0 and math.isfinite(total):
        seed_share = float(var_seed / total)
        unit_share = float(var_unit / total)
        if seed_share >= 0.65:
            dominant = 'seed'
        elif unit_share >= 0.65:
            dominant = 'unit'
        else:
            dominant = 'balanced'
    else:
        seed_share = unit_share = None
        dominant = 'not_estimable'
    gap_decomposition = {
        'units': unit_name,
        'gap_sd_seed': seed_axis.get('top2_gap_sd'),
        'gap_sd_unit': unit_axis.get('top2_gap_sd'),
        'gap_sd_two_way': two_axis.get('top2_gap_sd'),
        'seed_variance_share': seed_share,
        'unit_variance_share': unit_share,
        'dominant_axis': dominant,
        'note': ('descriptive split of the observed top-two gap variance; '
                 'the axes are not independent in the two-way design, so the '
                 'shares are a readout, not an ANOVA decomposition'),
    }
    out = {
        'kind': 'arm_rank_unit_sensitivity',
        'status': status,
        'status_reason': status_reason,
        'role': policy['role'],
        'estimand': policy['estimand'],
        'boundary': _ARM_RANK_UNIT_BOUNDARY,
        'better': better,
        'n_boot': int(n_boot),
        'bootstrap_seed': int(seed),
        'p_best_resolution': float(1.0 / int(n_boot)),
        'unit_level': float(unit_level),
        'seed_level': float(seed_level),
        'lead_level': float(lead_level),
        'tie_tol': float(tie_tol),
        'min_seeds': int(min_seeds),
        'min_units': int(min_units),
        'n_arms': int(k),
        'n_seeds': int(s),
        'n_units': int(n_units),
        'arms': list(arms),
        'seeds': [int(x) for x in seeds],
        'dropped_seeds': {arm: [int(x) for x in v] for arm, v in dropped.items() if v},
        'complete_seed_coverage': all(not v for v in dropped.values()),
        'unit_ids': list(ids) if ids is not None else None,
        'population_claim_allowed': bool(n_units >= 64),
        'per_unit_tokens': [float(x) for x in tokens],
        'per_arm_seed_nll': {arm: {str(sd): [float(x) for x in per_arm[arm][sd]['nll']]
                                   for sd in seeds} for arm in arms},
        'per_arm_seed_values': observed_seed_values,
        'endpoint_values_match': values_match,
        **observed,
        'seed_only': seed_axis,
        'unit_only': unit_axis,
        'two_way': two_axis,
        'gap_decomposition': gap_decomposition,
        'unit_estimand': policy['estimand'],
        'unit_granularity': unit_granularity,
    }
    if not out['population_claim_allowed']:
        out['population_note'] = (
            f'{n_units} evaluation units is below the repository population '
            'floor of 64: the readout is conditional on the frozen slice, not '
            'a claim about the language population')
    return out
