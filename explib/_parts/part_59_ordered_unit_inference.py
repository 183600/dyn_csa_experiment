"""Order dependence and effective sample size for evaluation-unit readouts.

The confirmatory test is a paired seed-level exact sign-flip.  The companion
document/article interval, however, is usually computed by resampling
evaluation units *i.i.d.*  That is a modelling assumption, not a property of
the corpus: the frozen evaluation slice is a contiguous source-order prefix,
so adjacent units can share an article, a topic or a local style.  When they
do, the i.i.d. interval is anti-conservative and ``n_units`` overstates the
independent information in the slice even though the point estimate is
unchanged.

This fragment makes that assumption checkable.  It takes one already ordered
sequence of per-unit paired effects (the same equal-unit vector that the
document readout reports) and returns:

* the i.i.d. mean/SE/t interval, for comparison with the interval that was
  previously reported;
* a non-overlapping **batch-means** design effect and effective sample size;
* a **moving-block bootstrap** interval, SE and design effect for each
  declared block length (a circular block bootstrap, so every position is
  usable and no block choice is tuned after looking at the result);
* an **order-exchangeability permutation test** of the primary batch-means
  design effect: permuting the values destroys the order but keeps the
  marginal distribution, so a small one-sided p means the observed order
  carries serial structure beyond the values themselves;
* an order-robust interval = the widest i.i.d./block interval at the declared
  block lengths, plus an explicit ``iid_understates`` flag when the block
  evidence materially widens the interval or changes whether zero is
  excluded;
* the sufficient statistics needed by ``audit_gates.py order_dependence`` to
  replay every number.

Boundaries are deliberate and part of the output:

* this is a **descriptive design-effect diagnostic**, not a new confirmatory
  test.  Its permutation p-value never enters the frozen Holm family and never
  changes ``p_exact_signflip``/``p_holm``/``verified``/``confirmed``;
* block resampling addresses serial dependence *within the observed slice*.
  It cannot turn a source-order prefix into a probability sample of the
  corpus, so it does not license a population claim;
* the block length is declared in advance and the reported interval is the
  widest across the declared lengths, not the one that looks best.  The
  batch-means and block estimates are noisy at n of a few dozen and are
  reported with their n and block size;
* the input order must be the corpus/evaluation order.  Pass the evaluation
  cell's first-occurrence ``unit_order`` attestation to verify it; sorting or
  shuffling is then refused.  Without an attestation the values alone cannot
  reveal a reorder, and the readout is explicitly marked ``unverified``.
"""

import math as _od_math

import numpy as _od_np


ORDER_DEPENDENCE_POLICY = {
    'version': 'order-dependence-1',
    'estimand': ('equal-unit mean of the paired per-unit effect over the '
                 'ordered evaluation-unit sequence'),
    'block_lengths': (2, 4, 8),
    'n_batches': (4, 6, 8),
    'primary_batch_rule': ('round(sqrt(n)) clamped to [4, 10] and at most '
                           'n//2; declared before seeing the statistic, used '
                           'for the permutation test'),
    'n_boot': 2000,
    'n_perm': 2000,
    'conf': 0.95,
    'alpha': 0.05,
    'min_units': 8,
    'min_batch_size': 2,
    'seed': 20261031,
    'design_effect_tolerance': 1.10,
    'material_ci_inflation': 1.10,
    'interval_method': ('i.i.d. Student-t interval; circular moving-block '
                        'percentile bootstrap at each declared block length; '
                        'order-robust interval = widest of the declared '
                        'intervals'),
    'design_effect_method': ('non-overlapping batch means (b * var(batch '
                             'means) / var(values)) and moving-block '
                             'bootstrap SE squared / i.i.d. variance of the '
                             'mean'),
    'permutation_method': ('values are permuted without replacement; the '
                           'one-sided p is (1 + #{permuted design effect >= '
                           'observed}) / (n_perm + 1), a randomisation test '
                           'of order exchangeability, not of independence'),
    'role': ('descriptive order-dependence / effective-n readout for one '
             'paired unit panel; it adds no p-value to the frozen family and '
             'never changes a frozen verdict'),
    'boundary': (
        'The diagnostic conditions on the trained seeds and describes the '
        'observed ordered evaluation slice. It cannot repair a non-random '
        'source-order prefix, and a non-significant permutation test is not '
        'proof that units are independent or exchangeable. Block and batch '
        'estimates are noisy at n of a few dozen; the widest declared '
        'interval is reported so the block length cannot be chosen after '
        'looking. The input order is caller-asserted unless a first-occurrence '
        'unit_order attestation is supplied and verified; a sorted/shuffled '
        'id vector without one is not corpus order. The permutation p-value '
        'is a diagnostic and never enters the frozen Holm family, '
        'p_exact_signflip, p_holm, verified or any confirmed verdict.'),
}

ORDER_DEPENDENCE_STATUSES = ('ok', 'not_estimable')


def _od_number(value, name):
    """Finite float, rejecting bools/strings/containers."""
    if isinstance(value, (bool, _od_np.bool_, str, bytes, bytearray)):
        raise ValueError(f'{name} must be a finite number, got {value!r}')
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'{name} must be a finite number, got {value!r}') from None
    if not _od_math.isfinite(out):
        raise ValueError(f'{name} must be a finite number, got {value!r}')
    return out


def _od_int(value, name, *, minimum=None, maximum=None):
    """Exact integer policy knob (integral floats are accepted)."""
    if isinstance(value, (bool, _od_np.bool_)):
        raise ValueError(f'{name} must be an integer, got {value!r}')
    if isinstance(value, (int, _od_np.integer)):
        out = int(value)
    else:
        number = _od_number(value, name)
        if not number.is_integer():
            raise ValueError(f'{name} must be an integer, got {value!r}')
        out = int(number)
    if minimum is not None and out < minimum:
        raise ValueError(f'{name} must be >= {minimum}, got {value!r}')
    if maximum is not None and out > maximum:
        raise ValueError(f'{name} must be <= {maximum}, got {value!r}')
    return out


def _od_values(values):
    """One finite ordered vector; a matrix is not a single ordered sequence."""
    if isinstance(values, (str, bytes, bytearray, dict)):
        raise ValueError('values must be an ordered sequence of finite numbers')
    if not isinstance(values, _od_np.ndarray):
        # A mixed list such as [1.0, True, 3.0] coerces to a float array,
        # losing the flag before the dtype check.  Inspect the raw elements
        # first; a bool is a label, not a measurement.
        try:
            seq = list(values)
        except TypeError:
            raise ValueError('values must be an ordered sequence of finite numbers') from None
        for value in seq:
            if isinstance(value, (bool, _od_np.bool_)):
                raise ValueError('values contains a boolean; a flag is not a measurement')
        values = seq
    try:
        raw = _od_np.asarray(values)
    except (TypeError, ValueError, OverflowError):
        raise ValueError('values must be an ordered sequence of finite numbers') from None
    if raw.ndim != 1:
        raise ValueError(f'values must be 1-D, got {raw.ndim}-D')
    if raw.dtype.kind == 'b':
        raise ValueError('values contains a boolean; a flag is not a measurement')
    if raw.dtype.kind not in ('i', 'u', 'f', 'O'):
        raise ValueError('values contains non-numeric entries')
    out = []
    for value in raw.tolist():
        if isinstance(value, (bool, _od_np.bool_)):
            raise ValueError('values contains a boolean; a flag is not a measurement')
        out.append(_od_number(value, 'values'))
    return out


def _od_scale(values):
    """Largest magnitude (0.0 for an all-zero panel)."""
    return max((abs(v) for v in values), default=0.0)


def _od_mean_sd(values):
    """Overflow-safe mean and sample SD (ddof=1) of finite floats.

    The ordinary numpy expression is used when it is finite.  If a naive sum
    or square overflows, the values are divided by their largest magnitude
    before the reduction and the statistics are mapped back.  A statistic
    genuinely outside float64 is returned as ``None`` rather than ``inf``.
    """
    n = len(values)
    if n == 0:
        return None, None
    arr = _od_np.asarray(values, dtype=float)
    with _od_np.errstate(over='ignore', invalid='ignore'):
        try:
            mean = float(_od_np.mean(arr))
            sd = float(_od_np.std(arr, ddof=1)) if n > 1 else 0.0
        except (OverflowError, ValueError, TypeError):
            mean, sd = None, None
    mean_ok = mean is not None and _od_math.isfinite(mean)
    sd_ok = sd is not None and _od_math.isfinite(sd)
    # Squaring subnormal values underflows to zero, so the naive SD can report
    # ``0.0`` for a genuinely varying panel.  Only trust a zero SD when every
    # value is identical; otherwise use the scaled path, which retains the
    # representable spread.
    naive_sd_trusted = (n < 2 or sd != 0.0
                        or all(v == values[0] for v in values))
    if mean_ok and (n < 2 or sd_ok) and naive_sd_trusted:
        return mean, sd
    scale = _od_scale(values)
    if scale == 0.0:
        return 0.0, 0.0
    z = [v / scale for v in values]
    try:
        mz = _od_math.fsum(z) / n
        scaled_mean = scale * mz
    except (OverflowError, ValueError):
        scaled_mean = None
    if scaled_mean is not None and not _od_math.isfinite(scaled_mean):
        scaled_mean = None
    if scaled_mean == 0.0 and mz != 0.0:
        # The true mean is below the smallest representable float; a fake zero
        # would make a tiny effect indistinguishable from an exact no-effect.
        scaled_mean = None
    scaled_sd = 0.0 if n < 2 else None
    if n >= 2 and scaled_mean is not None:
        try:
            ss = _od_math.fsum((x - mz) ** 2 for x in z)
            scaled_sd = scale * _od_math.sqrt(ss / (n - 1))
        except (OverflowError, ValueError):
            scaled_sd = None
        if scaled_sd is not None and not _od_math.isfinite(scaled_sd):
            scaled_sd = None
        if scaled_sd == 0.0 and any(x != z[0] for x in z):
            # Real spread below the smallest representable float: withhold it
            # rather than reporting a fake zero-variance panel.
            scaled_sd = None
    out_mean = mean if (mean_ok and scaled_mean is not None) else scaled_mean
    out_sd = sd if (sd_ok and naive_sd_trusted) else scaled_sd
    return out_mean, out_sd


def _od_centre(values):
    """Values divided by their largest magnitude, plus the scale.

    Working on this scale makes the bootstrap/batch arithmetic overflow-safe
    for panels near the float64 limit; every design effect and correlation is
    scale invariant, and the mean/interval is mapped back once.
    """
    scale = _od_scale(values)
    if scale == 0.0:
        return [0.0 for _ in values], 0.0
    return [v / scale for v in values], scale


def _od_percentile_ci(reps, conf):
    """Percentile interval from a finite replicate vector."""
    if not reps:
        return None
    lo_q = 100.0 * (1.0 - conf) / 2.0
    hi_q = 100.0 * (1.0 + conf) / 2.0
    try:
        lo, hi = _od_np.percentile(_od_np.asarray(reps, dtype=float),
                                   [lo_q, hi_q])
    except (TypeError, ValueError, OverflowError):
        return None
    lo, hi = float(lo), float(hi)
    if not (_od_math.isfinite(lo) and _od_math.isfinite(hi)):
        return None
    return [lo, hi]


def _od_lag1(values):
    """Lag-1 autocorrelation of the ordered vector, or ``None``."""
    n = len(values)
    if n < 3:
        return None
    mean = _od_math.fsum(values) / n
    den = _od_math.fsum((v - mean) ** 2 for v in values)
    if not (_od_math.isfinite(den) and den > 0.0):
        return None
    num = _od_math.fsum((values[i] - mean) * (values[i + 1] - mean)
                        for i in range(n - 1))
    out = num / den
    return out if _od_math.isfinite(out) else None


def _od_batch_design(values, n_batches):
    """Non-overlapping batch-means SE, design effect and effective n."""
    n = len(values)
    g = int(n_batches)
    b = n // g
    if g < 2 or b < 2:
        return None
    used = g * b
    means = []
    for j in range(g):
        chunk = values[j * b:(j + 1) * b]
        means.append(_od_math.fsum(chunk) / b)
    grand = _od_math.fsum(means) / g
    ss = _od_math.fsum((m - grand) ** 2 for m in means)
    var_batch_means = ss / (g - 1)
    if not _od_math.isfinite(var_batch_means):
        return None
    # Var(mean_n) ~ b * Var(batch mean) / n for a stationary series.
    var_mean = b * var_batch_means / n
    if not _od_math.isfinite(var_mean) or var_mean < 0.0:
        return None
    _, sd = _od_mean_sd(values)
    if sd is None or not (_od_math.isfinite(sd) and sd > 0.0):
        return None
    var_iid = (sd * sd) / n
    if not (_od_math.isfinite(var_iid) and var_iid > 0.0):
        return None
    de = var_mean / var_iid
    if not (_od_math.isfinite(de) and de >= 0.0):
        return None
    n_eff = None
    if de > 0.0:
        n_eff = n / de
        if not _od_math.isfinite(n_eff):
            n_eff = None
    return {'n_batches': g, 'batch_size': b, 'n_used': used,
            'se': _od_math.sqrt(var_mean), 'design_effect': de,
            'n_effective': n_eff,
            'method': 'non-overlapping batch means'}


def _od_block_bootstrap(values, block_length, *, n_boot, seed, conf):
    """Circular moving-block bootstrap of the mean."""
    n = len(values)
    b = int(block_length)
    if b < 1 or b > n:
        return None
    k = int(_od_math.ceil(n / b))
    rng = _od_np.random.default_rng(int(seed))
    arr = _od_np.asarray(values, dtype=float)
    reps = _od_np.empty(int(n_boot), dtype=float)
    for i in range(int(n_boot)):
        starts = rng.integers(0, n, size=k)
        # Build the replicate as concatenated circular blocks, then keep n.
        out = _od_np.empty(k * b, dtype=float)
        for j, start in enumerate(starts.tolist()):
            idx = (start + _od_np.arange(b)) % n
            out[j * b:(j + 1) * b] = arr[idx]
        reps[i] = float(out[:n].mean())
    valid = int(_od_np.isfinite(reps).sum())
    if valid < 2:
        return None
    reps = reps[_od_np.isfinite(reps)]
    se = float(_od_np.std(reps, ddof=1))
    ci = _od_percentile_ci(reps.tolist(), conf)
    _, sd = _od_mean_sd(values)
    de = None
    n_eff = None
    if sd is not None and sd > 0.0:
        var_iid = (sd * sd) / n
        if _od_math.isfinite(var_iid) and var_iid > 0.0:
            de = (se * se) / var_iid
            if _od_math.isfinite(de) and de > 0.0:
                n_eff = n / de
    return {'block_length': b, 'n_blocks': k, 'n_boot': int(n_boot),
            'n_boot_valid': valid, 'se': se, 'ci': ci,
            'design_effect': (de if de is not None
                              and _od_math.isfinite(de) else None),
            'n_effective': (n_eff if n_eff is not None
                            and _od_math.isfinite(n_eff) else None),
            'method': 'circular moving-block percentile bootstrap'}


def _od_primary_batches(n):
    if n < 4:
        return None
    g = int(round(_od_math.sqrt(n)))
    g = max(4, min(10, g))
    g = min(g, n // 2)
    return g if g >= 2 else None


def _od_permutation(values, *, primary_batches, n_perm, seed, alpha):
    """Randomisation p-values for the batch design effect and lag-1 r."""
    n = len(values)
    observed = _od_batch_design(values, primary_batches)
    if observed is None:
        return None
    obs_de = observed['design_effect']
    obs_r = _od_lag1(values)
    rng = _od_np.random.default_rng(int(seed))
    arr = _od_np.asarray(values, dtype=float)
    count_de = 0
    count_r = 0
    obs_abs_r = abs(obs_r) if obs_r is not None else None
    for _ in range(int(n_perm)):
        perm = rng.permutation(arr).tolist()
        d = _od_batch_design(perm, primary_batches)
        if d is not None and d['design_effect'] >= obs_de - 1e-12:
            count_de += 1
        if obs_abs_r is not None:
            r = _od_lag1(perm)
            if r is not None and abs(r) >= obs_abs_r - 1e-12:
                count_r += 1
    p_de = (1.0 + count_de) / (int(n_perm) + 1.0)
    p_r = ((1.0 + count_r) / (int(n_perm) + 1.0)
           if obs_abs_r is not None else None)
    return {'n_perm': int(n_perm), 'seed': int(seed), 'alpha': alpha,
            'primary_n_batches': int(primary_batches),
            'observed_design_effect': obs_de,
            'design_effect_p': p_de,
            'lag1_autocorr': obs_r,
            'lag1_p': p_r,
            'method': ('one-sided permutation of the batch-means design effect; '
                       'two-sided permutation of |lag-1 autocorrelation|; '
                       'values are permuted, never resampled with replacement')}


def _od_order_block(ids, attestation):
    """Verified/unverified order provenance for one ordered-unit readout."""
    if attestation is None:
        return {
            'status': 'unverified',
            'basis': None, 'sha256': None,
            'n_units': (len(ids) if ids is not None else None),
            'reason': ('no unit-order attestation was supplied; the input '
                       'order is caller-asserted, not verified'),
        }
    if ids is None:
        raise ValueError(
            'ordered_unit_inference: order_attestation requires unit_ids so '
            'the stored order fingerprint can be recomputed')
    problems = unit_order_problems(attestation, ids)
    if problems:
        raise ValueError('ordered_unit_inference: ' + '; '.join(problems))
    return {
        'status': 'verified',
        'basis': attestation.get('basis'),
        'sha256': attestation.get('sha256'),
        'unit_kind': attestation.get('unit_kind'),
        'n_units': len(ids),
    }


def ordered_unit_inference(values, *, unit_ids=None, block_lengths=None,
                           n_batches=None, n_boot=None, n_perm=None,
                           conf=None, alpha=None, seed=None,
                           design_effect_tolerance=None,
                           material_ci_inflation=None,
                           order_attestation=None):
    """Order dependence and effective-n readout for one paired unit panel.

    ``values`` is the **already ordered** equal-unit per-unit effect vector
    (for this repository, the seed-mean PPL difference of each evaluation
    unit in corpus order).  The function never sorts it: the order is the
    design information being tested.

    Returns a strict-JSON block with ``kind='ordered_unit_inference'`` and
    ``status='ok'``, or ``status='not_estimable'`` with a reason when the
    panel is too small or a statistic is not representable.  Malformed inputs
    raise ``ValueError``; a valid but unavailable readout is not faked.

    ``order_attestation`` is the cell's first-occurrence attestation.  When
    supplied, the function recomputes the ordered-id digest and refuses a
    sorted/shuffled vector; the returned ``unit_order`` block then says
    ``verified``.  Without it the order is caller-asserted (``unverified``)
    and a report must not call the result corpus-ordered.
    """
    vals = _od_values(values)
    n = len(vals)
    conf = _od_number(ORDER_DEPENDENCE_POLICY['conf'] if conf is None else conf,
                      'conf')
    if not (0.0 < conf < 1.0):
        raise ValueError(f'conf must be in (0, 1), got {conf!r}')
    alpha = _od_number(ORDER_DEPENDENCE_POLICY['alpha'] if alpha is None
                       else alpha, 'alpha')
    if not (0.0 < alpha < 1.0):
        raise ValueError(f'alpha must be in (0, 1), got {alpha!r}')
    seed = _od_int(ORDER_DEPENDENCE_POLICY['seed'] if seed is None else seed,
                   'seed', minimum=0)
    n_boot = _od_int(ORDER_DEPENDENCE_POLICY['n_boot'] if n_boot is None
                     else n_boot, 'n_boot', minimum=2, maximum=200000)
    n_perm = _od_int(ORDER_DEPENDENCE_POLICY['n_perm'] if n_perm is None
                     else n_perm, 'n_perm', minimum=0, maximum=200000)
    tol = _od_number(ORDER_DEPENDENCE_POLICY['design_effect_tolerance']
                     if design_effect_tolerance is None
                     else design_effect_tolerance,
                     'design_effect_tolerance')
    if not (_od_math.isfinite(tol) and tol > 0.0):
        raise ValueError('design_effect_tolerance must be a positive finite '
                         f'number, got {tol!r}')
    infl = _od_number(ORDER_DEPENDENCE_POLICY['material_ci_inflation']
                      if material_ci_inflation is None
                      else material_ci_inflation, 'material_ci_inflation')
    if not (_od_math.isfinite(infl) and infl >= 1.0):
        raise ValueError('material_ci_inflation must be a finite number >= 1, '
                         f'got {infl!r}')
    if isinstance(unit_ids, (str, bytes, bytearray)):
        raise ValueError('unit_ids must be an iterable, not a string')
    if unit_ids is not None:
        try:
            ids = [str(x) for x in unit_ids]
        except TypeError:
            raise ValueError('unit_ids must be an iterable') from None
        if len(ids) != n:
            raise ValueError(f'{len(ids)} unit ids for {n} values')
        if len(set(ids)) != n:
            raise ValueError('unit_ids must be unique')
    else:
        ids = None

    base = {'kind': 'ordered_unit_inference',
            'policy_version': ORDER_DEPENDENCE_POLICY['version'],
            'estimand': ORDER_DEPENDENCE_POLICY['estimand'],
            'n_units': n, 'conf': conf, 'alpha': alpha, 'seed': seed,
            'role': ORDER_DEPENDENCE_POLICY['role'],
            'boundary': ORDER_DEPENDENCE_POLICY['boundary'],
            'unit_order': _od_order_block(ids, order_attestation),
            'sufficient_statistics': {
                'values': [float(v) for v in vals],
                'unit_ids': ids,
                'unit_order': _od_order_block(ids, order_attestation),
                'unit_order_attestation': order_attestation,
                'block_lengths': None,
                'n_batches': None,
                'raw_n_batches': None,
                'n_batches_defaulted': bool(n_batches is None),
                'primary_n_batches': None,
                'n_boot': n_boot, 'n_perm': n_perm, 'conf': conf,
                'alpha': alpha, 'seed': seed,
                'design_effect_tolerance': tol,
                'material_ci_inflation': infl,
            }}
    if n < int(ORDER_DEPENDENCE_POLICY['min_units']):
        base.update({'status': 'not_estimable',
                     'reason': (f'{n} ordered units; at least '
                                f"{ORDER_DEPENDENCE_POLICY['min_units']} are "
                                'needed for a block/batch readout')})
        return base

    mean, sd = _od_mean_sd(vals)
    if mean is None or sd is None:
        base.update({'status': 'not_estimable',
                     'reason': 'the mean/sample SD is not representable'})
        return base
    se = sd / _od_math.sqrt(n)
    if sd > 0.0 and (not _od_math.isfinite(se) or se == 0.0):
        # sb -> 0 would make the i.i.d. interval a fake [mean, mean].  The
        # sample spread is representable but its standard error is not; refuse
        # the readout instead of reporting certainty.
        base.update({'status': 'not_estimable',
                     'reason': ('the sample SD is representable but the '
                                'standard error under/overflows float64')})
        return base
    tcrit = _t_critical(conf, float(n - 1))
    if not _od_math.isfinite(tcrit):
        base.update({'status': 'not_estimable',
                     'reason': 'the Student-t critical value is not representable'})
        return base
    iid_ci = [mean - tcrit * se, mean + tcrit * se]
    if not all(_od_math.isfinite(x) for x in iid_ci):
        base.update({'status': 'not_estimable',
                     'reason': 'the i.i.d. interval is not representable'})
        return base
    if sd > 0.0 and iid_ci[0] == iid_ci[1]:
        base.update({'status': 'not_estimable',
                     'reason': ('the sample SD is positive but the i.i.d. '
                                'interval width underflows to zero')})
        return base

    work, scale = _od_centre(vals)

    # Declared batch counts.  Explicit lists are validated strictly; the
    # default list silently drops infeasible counts and records why, so a small
    # panel does not fail merely because the policy lists a larger count.
    raw_batches = (ORDER_DEPENDENCE_POLICY['n_batches'] if n_batches is None
                   else n_batches)
    if isinstance(raw_batches, (str, bytes, bytearray, dict)):
        raise ValueError('n_batches must be an iterable of integers')
    try:
        batch_list = list(raw_batches)
    except TypeError:
        raise ValueError('n_batches must be an iterable of integers') from None
    if len(batch_list) > 20:
        raise ValueError('n_batches must contain at most 20 layouts')
    batch_counts = []
    raw_batch_list = []
    skipped = []
    for g in batch_list:
        gi = _od_int(g, 'n_batches entry', minimum=2)
        raw_batch_list.append(gi)
        b = n // gi
        if b < int(ORDER_DEPENDENCE_POLICY['min_batch_size']):
            if n_batches is not None:
                raise ValueError(
                    f'n_batches={gi} gives batch size {b} < '
                    f"{ORDER_DEPENDENCE_POLICY['min_batch_size']}")
            skipped.append({'n_batches': gi, 'reason': f'batch size {b} < 2'})
            continue
        if gi not in batch_counts:
            batch_counts.append(gi)
    if not batch_counts:
        if n_batches is not None:
            raise ValueError('n_batches must contain at least one feasible '
                             'batch layout')
        gi = _od_primary_batches(n)
        if gi is None:
            base.update({'status': 'not_estimable',
                         'reason': 'no feasible non-overlapping batch layout'})
            return base
        batch_counts = [gi]
    primary = _od_primary_batches(n)
    if primary is not None and primary not in batch_counts:
        # Keep the declared primary layout in the descriptive table: the
        # permutation test and the table must describe the same statistic.
        batch_counts.append(primary)
        batch_counts.sort()
    if primary is None:
        primary = batch_counts[0]

    batch_rows = []
    for gi in batch_counts:
        row = _od_batch_design(work, gi)
        if row is not None:
            # ``work`` is the unit-scaled copy used so the design effect and
            # n_effective stay overflow-safe.  The batch SE, like the block
            # SE below, is a measured quantity and must be mapped back to the
            # input scale; leaving it in work units made the same panel report
            # a batch SE that was smaller by exactly the (arbitrary) common
            # scale factor.
            row['se'] = (row['se'] * scale
                         if _od_math.isfinite(row['se']) else None)
            batch_rows.append(row)
    if not batch_rows:
        base.update({'status': 'not_estimable',
                     'reason': 'every batch layout produced a non-finite '
                               'design effect'})
        return base

    raw_blocks = (ORDER_DEPENDENCE_POLICY['block_lengths']
                  if block_lengths is None else block_lengths)
    if isinstance(raw_blocks, (str, bytes, bytearray, dict)):
        raise ValueError('block_lengths must be an iterable of integers')
    try:
        block_list = list(raw_blocks)
    except TypeError:
        raise ValueError('block_lengths must be an iterable of integers') from None
    if len(block_list) > 20:
        raise ValueError('block_lengths must contain at most 20 lengths')
    block_lengths_out = []
    for b in block_list:
        bi = _od_int(b, 'block_lengths entry', minimum=1)
        if bi < 2:
            if block_lengths is not None:
                raise ValueError('block lengths must be >= 2; block length 1 '
                                 'is the i.i.d. bootstrap, which is already '
                                 'reported as the iid_ci reference')
            continue
        if bi >= n:
            if block_lengths is not None:
                raise ValueError(f'block length {bi} must be < n={n}')
            continue
        if bi not in block_lengths_out:
            block_lengths_out.append(bi)
    if not block_lengths_out:
        if block_lengths is not None:
            raise ValueError('block_lengths must contain at least one '
                             'feasible length')
        block_lengths_out = [max(2, min(8, n // 2))]

    block_rows = []
    for b in block_lengths_out:
        zseed = (seed * 1000003 + b) % (2 ** 63 - 1)
        row = _od_block_bootstrap(work, b, n_boot=n_boot, seed=zseed,
                                  conf=conf)
        if row is not None:
            # Map the bootstrap SE and interval back to the input scale.  The
            # design effect and n_effective are scale-free.
            row['se'] = (row['se'] * scale
                         if _od_math.isfinite(row['se']) else None)
            if row.get('ci') is not None:
                mapped = [x * scale for x in row['ci']]
                row['ci'] = (mapped if all(_od_math.isfinite(x) for x in mapped)
                             else None)
            block_rows.append(row)
    if not block_rows:
        base.update({'status': 'not_estimable',
                     'reason': 'every moving-block bootstrap replicate was '
                               'non-finite'})
        return base

    perm = _od_permutation(work, primary_batches=primary, n_perm=n_perm,
                           seed=seed + 17, alpha=alpha) if n_perm else None
    p_de = perm.get('design_effect_p') if isinstance(perm, dict) else None
    order_detected = bool(p_de is not None and p_de <= alpha)
    de_values = [r['design_effect'] for r in batch_rows
                 if r.get('design_effect') is not None]
    de_values += [r['design_effect'] for r in block_rows
                  if r.get('design_effect') is not None]
    de_max = max(de_values) if de_values else None
    de_min = min(de_values) if de_values else None
    neff_values = [r['n_effective'] for r in batch_rows
                   if r.get('n_effective') is not None]
    neff_values += [r['n_effective'] for r in block_rows
                    if r.get('n_effective') is not None]
    neff_min = min(neff_values) if neff_values else None
    neff_max = max(neff_values) if neff_values else None

    candidates = [('iid_t', iid_ci)]
    for r in block_rows:
        if r.get('ci') is not None:
            candidates.append((f'block_length_{r["block_length"]}', r['ci']))
    robust_source, robust_ci = max(
        candidates, key=lambda item: abs(item[1][1] - item[1][0]))
    iid_width = abs(iid_ci[1] - iid_ci[0])
    robust_width = abs(robust_ci[1] - robust_ci[0])
    if not _od_math.isfinite(iid_width):
        iid_width = None
    if not _od_math.isfinite(robust_width):
        robust_width = None
    ci_inflation = None
    if (iid_width is not None and robust_width is not None
            and iid_width > 0.0):
        ci_inflation = robust_width / iid_width
        if not _od_math.isfinite(ci_inflation):
            ci_inflation = None
    iid_excludes_zero = bool(iid_ci[0] > 0.0 or iid_ci[1] < 0.0)
    block_zero = {}
    zero_changed = False
    zero_lost = False
    for r in block_rows:
        ci = r.get('ci')
        if ci is None:
            continue
        exc = bool(ci[0] > 0.0 or ci[1] < 0.0)
        block_zero[str(r['block_length'])] = exc
        if exc != iid_excludes_zero:
            zero_changed = True
        if iid_excludes_zero and not exc:
            # Only a lost exclusion means the i.i.d. interval UNDERSTATED the
            # uncertainty.  The opposite direction (i.i.d. included zero but
            # a narrower block interval excludes it) is negative dependence;
            # it is still reported through zero_exclusion_changed, but it is
            # not evidence for the ``iid_understates`` wording.
            zero_lost = True
    inflation = bool(
        order_detected
        and ((de_max is not None and de_max > tol)
             or (ci_inflation is not None and ci_inflation > infl)
             or zero_lost))
    iid_understates = inflation

    if perm is None:
        wording = (
            'The order-exchangeability test was not run (n_perm=0), so no '
            'order-dependence conclusion is available; the i.i.d. interval '
            'still assumes exchangeable units.')
    elif iid_understates:
        wording = (
            'The ordered unit sequence shows design-effect inflation; the '
            'i.i.d. unit interval understates uncertainty. Quote the '
            'order-robust interval (and the effective n), or resample blocks, '
            'and do not describe n_units as independent information.')
    elif order_detected:
        wording = (
            'Order structure is detectable but does not materially inflate the '
            'declared intervals; the i.i.d. interval is not contradicted, yet '
            'independence is not established.')
    else:
        wording = (
            'No order-dependent inflation was detected at this n; this is not '
            'proof that units are independent, exchangeable or a random '
            'sample of the corpus.')

    out = dict(base)
    out.update({
        'status': 'ok',
        'mean': mean,
        'sd': sd,
        'se': se,
        'iid_ci': iid_ci,
        'iid_ci_width': iid_width,
        'iid_excludes_zero': iid_excludes_zero,
        'lag1_autocorr': (_od_lag1(work) if sd > 0.0 else None),
        'batch_means': batch_rows,
        'block_bootstrap': block_rows,
        'skipped_batch_counts': skipped,
        'permutation': perm,
        'design_effect_min': de_min,
        'design_effect_max': de_max,
        'n_effective_min': neff_min,
        'n_effective_max': neff_max,
        'order_dependence_detected': order_detected,
        'order_dependence_p': p_de,
        'inflation_detected': inflation,
        'iid_understates': iid_understates,
        'robust_ci': robust_ci,
        'robust_ci_source': robust_source,
        'robust_ci_width': robust_width,
        'ci_inflation': ci_inflation,
        'zero_exclusion_changed': zero_changed,
        'zero_exclusion_lost': zero_lost,
        'block_zero_exclusion': block_zero,
        'wording': wording,
        'notes': [
            ('The input order is treated as the corpus/evaluation order; the '
             'function does not sort or shuffle it.  The returned unit_order '
             'block records whether that order was verified against a stored '
             'first-occurrence attestation (verified) or merely asserted '
             '(unverified).'),
            'A permutation p is a diagnostic of order exchangeability, not a '
            'confirmatory test; it does not enter the frozen Holm family.',
            'The widest declared interval is reported as the order-robust '
            'interval so the block length cannot be chosen after looking.',
            'A design effect below 1 (negative serial dependence) can make '
            'n_eff exceed n; that is an estimate, not extra independent data.',
        ],
    })
    out['sufficient_statistics'].update({
        'block_lengths': list(block_lengths_out),
        'n_batches': list(batch_counts),
        'raw_n_batches': list(raw_batch_list),
        'primary_n_batches': int(primary),
    })
    return out
