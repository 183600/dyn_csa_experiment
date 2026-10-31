"""Order dependence for the TOKEN-POOLED PPL contrast.

The companion interval next to the frozen comparison is a cluster-robust
interval for the token-pooled PPL difference.  It treats the evaluation units
(articles, or lines when an article map is unavailable) as independent
clusters.  The frozen evaluation slice is a contiguous source-order prefix,
so adjacent units can share an article, a topic or local style; under serial
dependence that i.i.d. cluster interval is anti-conservative even though the
point estimate is unchanged.  ``ordered_unit_inference`` (design_notes 117)
made this checkable for the **equal-unit** readout, but explicitly stopped
there: a weighted/token-pooled estimator has a different influence function,
so its design effect cannot be copied from the equal-unit vector.

This fragment closes that gap using the estimator's own cluster influence:

* ``psi_i`` is the seed-averaged delta-method influence of unit ``i`` on the
  token-pooled PPL difference ``mean_s[exp(mean_a_s) - exp(mean_b_s)]``;
  ``sum_i psi_i`` is that difference (up to floating-point round-off).
* ``q_i`` is the analogous influence on the token-pooled **log** PPL ratio
  ``mean_s[mean_a_s - mean_b_s]``; ``sum_i q_i`` is that ratio.
* both influence series are kept in corpus order.  Batch means and a circular
  moving-block bootstrap estimate a design effect and ``n_eff`` relative to
  the i.i.d. cluster variance; an order-exchangeability permutation test asks
  whether the observed order carries serial structure.
* the order-robust interval is the widest of the i.i.d. cluster interval and
  all declared block-length intervals, so the block length cannot be chosen
  after looking at the result.

The function is descriptive and zero-GPU.  It adds no p-value to the frozen
Holm family and never changes ``p_exact_signflip``/``p_holm``/``verified`` or
a ``confirmed`` verdict.  Block resampling describes the observed slice; it
cannot turn a source-order prefix into a probability sample of the corpus.
"""

import math as _po_math

import numpy as _po_np


POOLED_ORDER_POLICY = {
    'version': 'pooled-order-dependence-1',
    'estimand': (
        'token-pooled paired PPL difference (and its log-ratio analogue), '
        'with evaluation units treated as clusters in corpus order'),
    'endpoints': ('ppl', 'log_ratio'),
    'block_lengths': (2, 4, 8),
    'n_batches': (4, 6, 8),
    'primary_batch_rule': (
        'round(sqrt(n)) clamped to [4, 10] and at most n//2; declared before '
        'seeing the statistic, used for the permutation test'),
    'n_boot': 2000,
    'n_perm': 2000,
    'conf': 0.95,
    'alpha': 0.05,
    'min_units': 8,
    'min_seeds': 2,
    'min_batch_size': 2,
    'seed': 20261031,
    'design_effect_tolerance': 1.10,
    'material_ci_inflation': 1.10,
    'interval_method': (
        'i.i.d. cluster-robust delta-method t interval on the stored '
        'influence series; circular moving-block percentile bootstrap at each '
        'declared block length; order-robust interval = widest of the '
        'declared intervals'),
    'design_effect_method': (
        'non-overlapping batch means (b * var(batch means) / var(influence)) '
        'and moving-block bootstrap SE squared / i.i.d. variance of the '
        'influence mean'),
    'permutation_method': (
        'influence values are permuted without replacement; the one-sided p '
        'is (1 + #{permuted design effect >= observed}) / (n_perm + 1), a '
        'randomisation test of order exchangeability, not of independence'),
    'role': (
        'descriptive order-dependence / effective-n readout for the '
        'token-pooled estimator; it adds no p-value to the frozen family and '
        'never changes a frozen verdict'),
    'boundary': (
        'The diagnostic conditions on the trained seeds and describes the '
        'observed ordered evaluation slice. It cannot repair a non-random '
        'source-order prefix, and a non-significant permutation test is not '
        'proof that units are independent or exchangeable. The influence '
        'series is a delta-method linearisation of the nonlinear PPL '
        'contrast, not the exact finite-sample distribution. Block and batch '
        'estimates are noisy at n of a few dozen; the widest declared '
        'interval is reported so the block length cannot be chosen after '
        'looking. The permutation p-value is a diagnostic and never enters '
        'the frozen Holm family, p_exact_signflip, p_holm, verified or any '
        'confirmed verdict.'),
}

POOLED_ORDER_STATUSES = ('ok', 'not_estimable')


def _po_number(value, name):
    """Finite float, rejecting bools/strings/containers."""
    if isinstance(value, (bool, _po_np.bool_, str, bytes, bytearray)):
        raise ValueError(f'{name} must be a finite number, got {value!r}')
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'{name} must be a finite number, got {value!r}') from None
    if not _po_math.isfinite(out):
        raise ValueError(f'{name} must be a finite number, got {value!r}')
    return out


def _po_int(value, name, *, minimum=None, maximum=None):
    """Exact integer policy knob (integral floats are accepted)."""
    if isinstance(value, (bool, _po_np.bool_)):
        raise ValueError(f'{name} must be an integer, got {value!r}')
    if isinstance(value, (int, _po_np.integer)):
        out = int(value)
    else:
        number = _po_number(value, name)
        if not number.is_integer():
            raise ValueError(f'{name} must be an integer, got {value!r}')
        out = int(number)
    if minimum is not None and out < minimum:
        raise ValueError(f'{name} must be >= {minimum}, got {value!r}')
    if maximum is not None and out > maximum:
        raise ValueError(f'{name} must be <= {maximum}, got {value!r}')
    return out


def _po_point(value, name):
    """Finite endpoint point estimate, rejecting flags and strings."""
    if isinstance(value, (bool, _po_np.bool_, str, bytes, bytearray)):
        raise ValueError(f'{name} must be a finite number, got {value!r}')
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'{name} must be a finite number, got {value!r}') from None
    if not _po_math.isfinite(out):
        raise ValueError(f'{name} must be a finite number, got {value!r}')
    return out


def _po_policy(block_lengths, n_batches, n_boot, n_perm, conf, alpha, seed,
               design_effect_tolerance, material_ci_inflation, n_units):
    """Validate policy knobs and return the declared layouts.

    The default layout silently drops an infeasible batch count (recorded in
    ``skipped``), so a small panel does not fail merely because the policy
    lists a larger count.  An explicit list is validated strictly.
    """
    conf = _po_number(POOLED_ORDER_POLICY['conf'] if conf is None else conf,
                      'conf')
    if not (0.0 < conf < 1.0):
        raise ValueError(f'conf must be in (0, 1), got {conf!r}')
    alpha = _po_number(POOLED_ORDER_POLICY['alpha'] if alpha is None else alpha,
                       'alpha')
    if not (0.0 < alpha < 1.0):
        raise ValueError(f'alpha must be in (0, 1), got {alpha!r}')
    seed = _po_int(POOLED_ORDER_POLICY['seed'] if seed is None else seed,
                   'seed', minimum=0)
    n_boot = _po_int(POOLED_ORDER_POLICY['n_boot'] if n_boot is None
                     else n_boot, 'n_boot', minimum=2, maximum=200000)
    n_perm = _po_int(POOLED_ORDER_POLICY['n_perm'] if n_perm is None
                     else n_perm, 'n_perm', minimum=0, maximum=200000)
    tol = _po_number(POOLED_ORDER_POLICY['design_effect_tolerance']
                     if design_effect_tolerance is None
                     else design_effect_tolerance,
                     'design_effect_tolerance')
    if not (_po_math.isfinite(tol) and tol > 0.0):
        raise ValueError('design_effect_tolerance must be a positive finite '
                         f'number, got {tol!r}')
    infl = _po_number(POOLED_ORDER_POLICY['material_ci_inflation']
                      if material_ci_inflation is None
                      else material_ci_inflation, 'material_ci_inflation')
    if not (_po_math.isfinite(infl) and infl >= 1.0):
        raise ValueError('material_ci_inflation must be a finite number >= 1, '
                         f'got {infl!r}')

    raw_batches = (POOLED_ORDER_POLICY['n_batches'] if n_batches is None
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
        gi = _po_int(g, 'n_batches entry', minimum=2)
        raw_batch_list.append(gi)
        b = n_units // gi
        if b < int(POOLED_ORDER_POLICY['min_batch_size']):
            if n_batches is not None:
                raise ValueError(
                    f'n_batches={gi} gives batch size {b} < '
                    f"{POOLED_ORDER_POLICY['min_batch_size']}")
            skipped.append({'n_batches': gi, 'reason': f'batch size {b} < 2'})
            continue
        if gi not in batch_counts:
            batch_counts.append(gi)
    if not batch_counts:
        gi = _po_primary_batches(n_units)
        if gi is None:
            raise ValueError('no feasible non-overlapping batch layout')
        batch_counts = [gi]
    primary = _po_primary_batches(n_units)
    if primary is not None and primary not in batch_counts:
        batch_counts.append(primary)
        batch_counts.sort()
    if primary is None:
        primary = batch_counts[0]

    raw_blocks = (POOLED_ORDER_POLICY['block_lengths']
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
        bi = _po_int(b, 'block_lengths entry', minimum=1)
        if bi < 2:
            if block_lengths is not None:
                raise ValueError('block lengths must be >= 2; block length 1 '
                                 'is the i.i.d. bootstrap, which is already '
                                 'reported as the iid reference')
            continue
        if bi >= n_units:
            if block_lengths is not None:
                raise ValueError(f'block length {bi} must be < n={n_units}')
            continue
        if bi not in block_lengths_out:
            block_lengths_out.append(bi)
    if not block_lengths_out:
        block_lengths_out = [max(2, min(8, n_units // 2))]
    return {
        'conf': conf, 'alpha': alpha, 'seed': seed,
        'n_boot': n_boot, 'n_perm': n_perm,
        'design_effect_tolerance': tol,
        'material_ci_inflation': infl,
        'batch_counts': batch_counts,
        'raw_batch_counts': raw_batch_list,
        'skipped_batch_counts': skipped,
        'primary_n_batches': primary,
        'block_lengths': block_lengths_out,
        'n_batches_defaulted': bool(n_batches is None),
    }


def _po_primary_batches(n):
    if n < 4:
        return None
    g = int(round(_po_math.sqrt(n)))
    g = max(4, min(10, g))
    g = min(g, n // 2)
    return g if g >= 2 else None


def _po_values(values, name):
    """One finite 1-D influence vector; matrices are not a series."""
    if isinstance(values, (str, bytes, bytearray, dict)):
        raise ValueError(f'{name} must be a sequence of finite numbers')
    if not isinstance(values, _po_np.ndarray):
        try:
            seq = list(values)
        except TypeError:
            raise ValueError(f'{name} must be a sequence of finite numbers') from None
        for value in seq:
            if isinstance(value, (bool, _po_np.bool_)):
                raise ValueError(f'{name} contains a boolean; a flag is not a measurement')
        values = seq
    try:
        raw = _po_np.asarray(values)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'{name} must be a sequence of finite numbers') from None
    if raw.ndim != 1:
        raise ValueError(f'{name} must be 1-D, got {raw.ndim}-D')
    if raw.dtype.kind == 'b':
        raise ValueError(f'{name} contains a boolean; a flag is not a measurement')
    if raw.dtype.kind not in ('i', 'u', 'f', 'O'):
        raise ValueError(f'{name} contains non-numeric entries')
    out = []
    for value in raw.tolist():
        if isinstance(value, (bool, _po_np.bool_)):
            raise ValueError(f'{name} contains a boolean; a flag is not a measurement')
        out.append(_po_number(value, name))
    return out


def _po_cluster_iid(series, point, conf, df):
    """Cluster-robust delta-method t interval on an influence series.

    ``sum(series)`` is the endpoint and the usual cluster variance is
    ``D/(D-1) * sum(series**2)``.  The point estimate is passed separately:
    the PPL contrast is nonlinear, so its exact point is not identical to the
    floating-point sum of the linearised influence.
    """
    D = len(series)
    if D < 2:
        return None
    try:
        ss = _po_math.fsum(float(v) * float(v) for v in series)
    except (OverflowError, ValueError):
        return None
    # The endpoint is a SUM over units (the token-pooled contrast), not a
    # mean over units.  Its delta-method error is Sum(psi), so the cluster
    # variance is D/(D-1) * Sum(psi**2).  The influence vector itself is
    # centred (Sum(psi)=0 by construction at the sample estimate); the
    # bootstrap below therefore uses the resampled sum as the error
    # distribution around the reported point.
    var = ss * (D / (D - 1.0))
    if not _po_math.isfinite(var) or var < 0.0:
        return None
    se = _po_math.sqrt(var)
    tcrit = _t_critical(conf, float(df))
    if not (_po_math.isfinite(se) and _po_math.isfinite(tcrit)):
        return None
    lo, hi = point - tcrit * se, point + tcrit * se
    if not (_po_math.isfinite(lo) and _po_math.isfinite(hi)):
        return None
    return {'se': se, 'ci': [lo, hi]}


def _po_endpoint(series, point, *, name, policy, n_units):
    """Replayable design-effect block for one influence series."""
    D = n_units
    conf = policy['conf']
    alpha = policy['alpha']
    seed = policy['seed']
    tol = policy['design_effect_tolerance']
    infl = policy['material_ci_inflation']
    iid = _po_cluster_iid(series, point, conf, D - 1)
    if iid is None:
        return {'kind': 'pooled_order_endpoint', 'estimand': name,
                'status': 'not_estimable',
                'reason': 'the cluster-robust i.i.d. variance is not representable'}
    iid_ci = iid['ci']
    iid_width = abs(iid_ci[1] - iid_ci[0])
    if not _po_math.isfinite(iid_width) or iid_width <= 0.0:
        return {'kind': 'pooled_order_endpoint', 'estimand': name,
                'status': 'not_estimable',
                'reason': ('the cluster-robust interval has zero or '
                           'non-representable width')}

    work, scale = _od_centre(series)

    batch_rows = []
    for gi in policy['batch_counts']:
        row = _od_batch_design(work, gi)
        if row is not None:
            # ``_od_batch_design`` reports the SE of mean(work); the
            # delta-method error of the endpoint is D * mean(work), so map
            # the SE back to the endpoint scale.  The design effect itself is
            # scale-free and is left unchanged.
            mapped = row['se'] * scale * D
            if _po_math.isfinite(mapped):
                row['se'] = mapped
                batch_rows.append(row)
    block_rows = []
    for b in policy['block_lengths']:
        zseed = (seed * 1000003 + b) % (2 ** 63 - 1)
        row = _od_block_bootstrap(work, b, n_boot=policy['n_boot'],
                                  seed=zseed, conf=conf)
        if row is None:
            continue
        mapped_se = row['se'] * scale * D
        mapped_ci = None
        if row.get('ci') is not None:
            # The bootstrap distribution is for the sampling error D*mean(psi),
            # which is centred on zero; the interval for the endpoint is
            # point + that error.
            cand = [point + x * scale * D for x in row['ci']]
            if all(_po_math.isfinite(x) for x in cand):
                mapped_ci = cand
        if not _po_math.isfinite(mapped_se):
            continue
        row['se'] = mapped_se
        row['ci'] = mapped_ci
        block_rows.append(row)
    if not batch_rows and not block_rows:
        return {'kind': 'pooled_order_endpoint', 'estimand': name,
                'status': 'not_estimable',
                'reason': ('the influence series is constant (or every '
                           'batch/block layout was non-finite); no design '
                           'effect is estimable')}

    perm = (_od_permutation(work,
                            primary_batches=policy['primary_n_batches'],
                            n_perm=policy['n_perm'],
                            seed=seed + 17, alpha=alpha)
            if policy['n_perm'] else None)
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

    candidates = [('iid_cluster_t', iid_ci)]
    for r in block_rows:
        if r.get('ci') is not None:
            candidates.append((f'block_length_{r["block_length"]}', r['ci']))
    robust_source, robust_ci = max(
        candidates, key=lambda item: abs(item[1][1] - item[1][0]))
    robust_width = abs(robust_ci[1] - robust_ci[0])
    if not _po_math.isfinite(robust_width):
        robust_width = None
    ci_inflation = None
    if (iid_width is not None and robust_width is not None
            and iid_width > 0.0):
        ci_inflation = robust_width / iid_width
        if not _po_math.isfinite(ci_inflation):
            ci_inflation = None
    iid_excludes_zero = bool(iid_ci[0] > 0.0 or iid_ci[1] < 0.0)
    zero_changed = False
    zero_lost = False
    block_zero = {}
    for r in block_rows:
        ci = r.get('ci')
        if ci is None:
            continue
        exc = bool(ci[0] > 0.0 or ci[1] < 0.0)
        block_zero[str(r['block_length'])] = exc
        if exc != iid_excludes_zero:
            zero_changed = True
        if iid_excludes_zero and not exc:
            # The i.i.d. cluster interval excluded zero but a declared
            # order-aware interval does not.  This is the conservative
            # direction: serial dependence weakened the claim.  The opposite
            # discrepancy can occur with negative dependence, but it does not
            # mean the i.i.d. interval understated uncertainty.
            zero_lost = True
    inflation = bool(
        order_detected
        and ((de_max is not None and de_max > tol)
             or (ci_inflation is not None and ci_inflation > infl)
             or zero_lost))
    if perm is None:
        wording = ('The order-exchangeability test was not run (n_perm=0); '
                   'the i.i.d. cluster interval still assumes exchangeable '
                   'units.')
    elif inflation:
        wording = ('The token-pooled influence series shows design-effect '
                   'inflation; the i.i.d. cluster interval understates '
                   'uncertainty. Quote the order-robust interval and n_eff, '
                   'or resample blocks, and do not describe n_units as '
                   'independent information.')
    elif order_detected:
        wording = ('Order structure is detectable but does not materially '
                   'inflate the declared intervals; the i.i.d. cluster '
                   'interval is not contradicted, yet independence is not '
                   'established.')
    else:
        wording = ('No order-dependent inflation was detected at this n; '
                   'this is not proof that units are independent, '
                   'exchangeable or a random sample of the corpus.')
    return {
        'kind': 'pooled_order_endpoint',
        'estimand': name,
        'status': 'ok',
        'point': point,
        'df': int(D - 1),
        'iid_ci': iid_ci,
        'iid_se': iid['se'],
        'iid_ci_width': iid_width,
        'iid_excludes_zero': iid_excludes_zero,
        'batch_means': batch_rows,
        'block_bootstrap': block_rows,
        'design_effect_min': de_min,
        'design_effect_max': de_max,
        'n_effective_min': neff_min,
        'n_effective_max': neff_max,
        'order_dependence_p': p_de,
        'order_dependence_detected': order_detected,
        'inflation_detected': inflation,
        'iid_understates': inflation,
        'robust_ci': robust_ci,
        'robust_ci_source': robust_source,
        'robust_ci_width': robust_width,
        'ci_inflation': ci_inflation,
        'zero_exclusion_changed': zero_changed,
        'zero_exclusion_lost': zero_lost,
        'block_zero_exclusion': block_zero,
        'permutation': perm,
        'wording': wording,
    }


def pooled_order_inference_from_influence(
        ppl_influence, log_ratio_influence, *, delta_ppl, delta_log_ratio,
        unit_ids=None, seeds=None, n_seeds=None, block_lengths=None,
        n_batches=None, n_boot=None, n_perm=None, conf=None, alpha=None,
        seed=None, design_effect_tolerance=None,
        material_ci_inflation=None):
    """Replay the pooled design-effect readout from stored influence series.

    This is the audit entry point as well as the core of
    :func:`pooled_order_inference`.  ``ppl_influence`` and
    ``log_ratio_influence`` are the already ordered per-unit cluster
    influences; their sums must equal the supplied points (up to
    floating-point round-off).  The function never sorts or shuffles them.
    """
    psi = _po_values(ppl_influence, 'ppl_influence')
    q = _po_values(log_ratio_influence, 'log_ratio_influence')
    D = len(psi)
    if len(q) != D:
        raise ValueError(f'{D} PPL influences but {len(q)} log-ratio influences')
    delta_ppl = _po_point(delta_ppl, 'delta_ppl')
    delta_log = _po_point(delta_log_ratio, 'delta_log_ratio')
    for series, name in ((psi, 'ppl_influence'), (q, 'log_ratio_influence')):
        # A delta-method influence vector is centred: Sum(psi)=0.  The point
        # estimate is a separate number; requiring Sum(psi)==point would
        # reject every valid influence series.
        _sum = _po_math.fsum(series) if series else 0.0
        _abs = _po_math.fsum(abs(v) for v in series) if series else 0.0
        if abs(_sum) > 1e-8 * max(1.0, _abs):
            raise ValueError(
                f'{name} has non-zero sum {_sum!r}; a delta-method influence '
                'series must be centred (Sum(psi)=0)')
    if unit_ids is not None:
        if isinstance(unit_ids, (str, bytes, bytearray)):
            raise ValueError('unit_ids must be an iterable, not a string')
        try:
            ids = [str(x) for x in unit_ids]
        except TypeError:
            raise ValueError('unit_ids must be an iterable') from None
        if len(ids) != D:
            raise ValueError(f'{len(ids)} unit ids for {D} influences')
        if len(set(ids)) != D:
            raise ValueError('unit_ids must be unique')
    else:
        ids = None
    if seeds is not None:
        if isinstance(seeds, (str, bytes, bytearray, dict)):
            raise ValueError('seeds must be an iterable of integers')
        try:
            seed_list = [_po_int(s, 'seed') for s in seeds]
        except TypeError:
            raise ValueError('seeds must be an iterable of integers') from None
        if len(set(seed_list)) != len(seed_list):
            raise ValueError('seeds must be unique')
    else:
        seed_list = None
    if n_seeds is None:
        n_seeds = len(seed_list) if seed_list is not None else None
    if n_seeds is not None:
        n_seeds = _po_int(n_seeds, 'n_seeds', minimum=1)

    base = {
        'kind': 'pooled_order_inference',
        'policy_version': POOLED_ORDER_POLICY['version'],
        'estimand': POOLED_ORDER_POLICY['estimand'],
        'role': POOLED_ORDER_POLICY['role'],
        'boundary': POOLED_ORDER_POLICY['boundary'],
        'n_units': D,
        'n_seeds': n_seeds,
        'conf': None, 'alpha': None, 'seed': None,
        'sufficient_statistics': {
            'ppl_influence': [float(v) for v in psi],
            'log_ratio_influence': [float(v) for v in q],
            'delta_ppl': delta_ppl,
            'delta_log_ratio': delta_log,
            'unit_ids': ids,
            'seeds': seed_list,
            'n_seeds': n_seeds,
        },
    }
    if D < int(POOLED_ORDER_POLICY['min_units']):
        base.update({
            'status': 'not_estimable',
            'reason': (f'{D} evaluation units; at least '
                       f"{POOLED_ORDER_POLICY['min_units']} are needed for a "
                       'block/batch design-effect readout'),
        })
        return base
    if n_seeds is not None and n_seeds < int(POOLED_ORDER_POLICY['min_seeds']):
        base.update({
            'status': 'not_estimable',
            'reason': (f'{n_seeds} training seeds; at least '
                       f"{POOLED_ORDER_POLICY['min_seeds']} are needed"),
        })
        return base
    policy = _po_policy(block_lengths, n_batches, n_boot, n_perm, conf,
                        alpha, seed, design_effect_tolerance,
                        material_ci_inflation, D)
    base['conf'] = policy['conf']
    base['alpha'] = policy['alpha']
    base['seed'] = policy['seed']
    base['sufficient_statistics'].update({
        'block_lengths': list(policy['block_lengths']),
        'n_batches': list(policy['batch_counts']),
        'raw_n_batches': list(policy['raw_batch_counts']),
        'n_batches_defaulted': policy['n_batches_defaulted'],
        'primary_n_batches': int(policy['primary_n_batches']),
        'n_boot': policy['n_boot'],
        'n_perm': policy['n_perm'],
        'conf': policy['conf'],
        'alpha': policy['alpha'],
        'seed': policy['seed'],
        'design_effect_tolerance': policy['design_effect_tolerance'],
        'material_ci_inflation': policy['material_ci_inflation'],
    })
    ppl_block = _po_endpoint(psi, delta_ppl, name='delta_ppl',
                             policy=policy, n_units=D)
    log_block = _po_endpoint(q, delta_log, name='delta_log_ratio',
                             policy=policy, n_units=D)
    if ppl_block.get('status') != 'ok' and log_block.get('status') != 'ok':
        base.update({
            'status': 'not_estimable',
            'reason': (ppl_block.get('reason') or 'no endpoint estimable'),
            'ppl': ppl_block,
            'log_ratio': log_block,
        })
        return base

    de_vals = [b.get('design_effect_max')
               for b in (ppl_block, log_block)
               if b.get('design_effect_max') is not None]
    neff_vals = [b.get('n_effective_min')
                 for b in (ppl_block, log_block)
                 if b.get('n_effective_min') is not None]
    base.update({
        'status': 'ok',
        'point_ppl': delta_ppl,
        'point_log_ratio': delta_log,
        'ppl': ppl_block,
        'log_ratio': log_block,
        'design_effect_max': (max(de_vals) if de_vals else None),
        'n_effective_min': (min(neff_vals) if neff_vals else None),
        'order_dependence_detected': bool(
            ppl_block.get('order_dependence_detected') is True
            or log_block.get('order_dependence_detected') is True),
        'inflation_detected': bool(
            ppl_block.get('inflation_detected') is True
            or log_block.get('inflation_detected') is True),
        'iid_understates': bool(
            ppl_block.get('iid_understates') is True
            or log_block.get('iid_understates') is True),
        'zero_exclusion_changed': bool(
            ppl_block.get('zero_exclusion_changed') is True
            or log_block.get('zero_exclusion_changed') is True),
        'zero_exclusion_lost': bool(
            ppl_block.get('zero_exclusion_lost') is True
            or log_block.get('zero_exclusion_lost') is True),
        'notes': [
            ('The input order is treated as the corpus/evaluation order; the '
             'function does not sort or shuffle it.'),
            ('The influence series is the delta-method linearisation of the '
             'token-pooled contrast; the PPL point is the exact pooled '
             'contrast, not the floating-point sum of the influence.'),
            ('A permutation p is a diagnostic of order exchangeability, not a '
             'confirmatory test; it does not enter the frozen Holm family.'),
            ('The widest declared interval is reported as the order-robust '
             'interval so the block length cannot be chosen after looking.'),
            ('A design effect below 1 (negative serial dependence) can make '
             'n_eff exceed n; that is an estimate, not extra independent '
             'data.'),
        ],
    })
    return base


def pooled_order_inference(nll_a, nll_b, n_tokens, *, unit_ids=None,
                           seeds=None, block_lengths=None, n_batches=None,
                           n_boot=None, n_perm=None, conf=None, alpha=None,
                           seed=None, design_effect_tolerance=None,
                           material_ci_inflation=None):
    """Order-dependence readout for the token-pooled PPL contrast.

    ``nll_a`` / ``nll_b`` are ``[n_seeds, n_units]`` summed-NLL matrices on
    the same units and positions; ``n_tokens`` is the shared positive integer
    token count of each unit, in corpus order.  The point estimate is the
    same seed-averaged token-pooled PPL difference used by the primary
    comparison, plus its log-ratio analogue.

    A valid but too-small or degenerate panel returns
    ``status='not_estimable'`` with a reason; malformed inputs raise
    ``ValueError``.  No p-value, Holm member or frozen field is produced.
    """
    A = _as_nll_matrix(nll_a, 'pooled_order_inference(nll_a)',
                       axis_label='n_units', value_label='per-unit')
    B = _as_nll_matrix(nll_b, 'pooled_order_inference(nll_b)',
                       axis_label='n_units', value_label='per-unit')
    if A.shape != B.shape:
        raise ValueError(
            f'pooled_order_inference: arm shapes differ ({A.shape} vs '
            f'{B.shape}); the paired influence needs the same seeds and units')
    S, D = A.shape
    try:
        raw_t = _po_np.asarray(n_tokens).reshape(-1)
    except (TypeError, ValueError, OverflowError):
        raise ValueError('pooled_order_inference: n_tokens must be numeric') from None
    if _contains_bool_value(n_tokens) or raw_t.dtype.kind == 'b':
        raise ValueError('pooled_order_inference: token counts must be integers, not booleans')
    if raw_t.dtype.kind in ('U', 'S'):
        raise ValueError('pooled_order_inference: token counts must be integers, not strings')
    counts = [_exact_count(v, minimum=1) for v in raw_t.tolist()]
    if any(c is None for c in counts):
        raise ValueError('pooled_order_inference: every unit needs a finite positive integer token count')
    if len(counts) != D:
        raise ValueError(f'pooled_order_inference: {len(counts)} token counts for {D} units')
    total = int(sum(counts))
    if total <= 0:
        raise ValueError('pooled_order_inference: the total token count must be positive')
    try:
        T = _po_np.asarray(counts, dtype=float)
        total_f = float(total)
        if not _po_math.isfinite(total_f) or total_f <= 0.0:
            raise OverflowError
    except (OverflowError, ValueError):
        return {'kind': 'pooled_order_inference',
                'policy_version': POOLED_ORDER_POLICY['version'],
                'status': 'not_estimable',
                'n_units': D, 'n_seeds': S,
                'reason': ('the token total exceeds the representable float64 '
                           'range; the pooled influence is not formed')}
    with _po_np.errstate(over='ignore', invalid='ignore'):
        ce_a = A.sum(axis=1) / total_f
        ce_b = B.sum(axis=1) / total_f
    if not (_po_np.all(_po_np.isfinite(ce_a))
            and _po_np.all(_po_np.isfinite(ce_b))):
        return {'kind': 'pooled_order_inference',
                'policy_version': POOLED_ORDER_POLICY['version'],
                'status': 'not_estimable',
                'n_units': D, 'n_seeds': S,
                'reason': ('a per-seed token-pooled mean NLL is not '
                           'representable; the confidence readout is withheld')}
    with _po_np.errstate(over='ignore', invalid='ignore'):
        ea = _po_np.asarray([_safe_exp_ppl(x) for x in ce_a], dtype=float)
        eb = _po_np.asarray([_safe_exp_ppl(x) for x in ce_b], dtype=float)
    if not (_po_np.all(_po_np.isfinite(ea)) and _po_np.all(_po_np.isfinite(eb))):
        return {'kind': 'pooled_order_inference',
                'policy_version': POOLED_ORDER_POLICY['version'],
                'status': 'not_estimable',
                'n_units': D, 'n_seeds': S,
                'reason': ('a per-seed token-pooled PPL is not representable; '
                           'the confidence readout is withheld')}
    delta_ppl = float(_po_np.mean(ea - eb))
    delta_log = float(_po_np.mean(ce_a - ce_b))
    if not (_po_math.isfinite(delta_ppl) and _po_math.isfinite(delta_log)):
        return {'kind': 'pooled_order_inference',
                'policy_version': POOLED_ORDER_POLICY['version'],
                'status': 'not_estimable',
                'n_units': D, 'n_seeds': S,
                'reason': ('the token-pooled point estimate is not '
                           'representable')}
    # Per-seed delta-method influence of the pooled CE difference on unit i,
    # then the seed average.  The PPL-scale influence is exactly the one used
    # by `_paired_nll_doc_ci`; the log-scale influence is the weighted NLL
    # residual.  Sums recover the two point estimates (up to round-off).
    with _po_np.errstate(over='ignore', invalid='ignore'):
        psi = (ea[:, None] * (A - ce_a[:, None] * T[None, :]) / total_f
               - eb[:, None] * (B - ce_b[:, None] * T[None, :]) / total_f)
        q = ((A - B) - (ce_a - ce_b)[:, None] * T[None, :]) / total_f
    if not (_po_np.all(_po_np.isfinite(psi)) and _po_np.all(_po_np.isfinite(q))):
        return {'kind': 'pooled_order_inference',
                'policy_version': POOLED_ORDER_POLICY['version'],
                'status': 'not_estimable',
                'n_units': D, 'n_seeds': S,
                'reason': ('the cluster influence is not representable; the '
                           'design-effect readout is withheld')}
    out = pooled_order_inference_from_influence(
        psi.mean(axis=0), q.mean(axis=0),
        delta_ppl=delta_ppl, delta_log_ratio=delta_log,
        unit_ids=unit_ids, seeds=seeds, n_seeds=S,
        block_lengths=block_lengths, n_batches=n_batches, n_boot=n_boot,
        n_perm=n_perm, conf=conf, alpha=alpha, seed=seed,
        design_effect_tolerance=design_effect_tolerance,
        material_ci_inflation=material_ci_inflation)
    out['upstream'] = ('pooled_order_inference: per-seed token-pooled NLL '
                       'matrices; influence is the delta-method cluster '
                       'influence of the same estimand as the primary '
                       'comparison')
    return out
