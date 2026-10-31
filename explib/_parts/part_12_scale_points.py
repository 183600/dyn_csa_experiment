
# Inverse-variance pooling squares a standard error again when it forms
# ``w ** 2``.  For an effect scale below ~1e-77 the raw square overflows even
# though every input is finite, so normalize the whole family first.  The
# historical 1e-100 cutoff left 1e-100..1e-78 panels overflowing their
# weights; 1e-75 keeps a safe margin while leaving ordinary inputs on the
# established path.
_EFFECT_SCALE_FLOOR = 1e-75
# When a family mixes sub-floor effects with ordinary ones, normalising by the
# maximum leaves the small point's SE below the weight domain (1/se**2 becomes
# inf and the point is silently excluded from the pool).  Normalise that mixed
# family so the largest WORK-scale effect is 1e150: then the smallest/largest
# inverse-variance weight and their squares still fit float64 for the dynamic
# range this helper is designed to admit (up to ~1e200), instead of dropping a
# legitimate point for a units-only reason.
_EFFECT_WORK_MAX = 1e150


def _scale_points(points):
    """Normalise [{label, scale, effects, seeds, corpus_fp?, vocab?}] /
    [(scale, effects)] inputs.

    `corpus_fp` (the composite data identity, e.g. the record's
    `data_version`: dataset + tokenizer fit + train/val fingerprints) and
    `vocab` are optional per-point provenance for the scale family: a
    cross-scale effect (paired log PPL ratio) is only defined when every
    scale point measured the SAME corpus and tokenizer — a mixed family must
    be refused by the consumers (meta_analyze_effects / scaling_trend), not
    pooled into a trend."""
    # A bare string/bytes (or a mapping used as a point list) is an iterable
    # of characters/keys, not a scale family.  Iterating it used to invent
    # empty or mislabelled points, so fail closed before the loop.
    if isinstance(points, (str, bytes, bytearray, dict)):
        raise ValueError('scale points must be an iterable of point records, '
                         f'got {type(points).__name__}')
    out = []
    for i, p in enumerate(points if points is not None else []):
        if isinstance(p, dict):
            eff = p.get('effects')
            scale = p.get('scale')
            label = p.get('label', i)
            seeds = p.get('seeds')
            corpus_fp = p.get('corpus_fp')
            vocab = p.get('vocab')
            n_planned = p.get('n_planned')
            planned_seeds = p.get('planned_seeds')
        elif isinstance(p, (tuple, list)) and len(p) >= 2:
            scale, eff = p[0], p[1]
            label = i
            seeds = p[2] if len(p) > 2 else None
            corpus_fp = p[3] if len(p) > 3 else None
            vocab = None
            n_planned = None
            planned_seeds = None
        else:
            continue
        if eff is None:
            # A declared tier with no effects vector is an EMPTY measurement,
            # not a reason to delete the tier from the family: dropping it
            # would let the surviving points look like a complete ladder.
            vals = []
            seeds = None
        else:
            if isinstance(eff, (str, bytes, bytearray)):
                raise ValueError('scale point %r: effects must be an iterable of '
                                 'measurements, got a string %r'
                                 % (label, eff))
            if isinstance(eff, dict):
                # A mapping iterates its keys, which are labels, not
                # measurements; accepting it would silently analyse the keys.
                raise ValueError('scale point %r: effects must be an iterable '
                                 'of measurements, not a mapping %r'
                                 % (label, eff))
            try:
                vals = list(eff)
            except TypeError:
                # Silently `continue`-ing used to delete the declared tier from
                # the family, so the surviving points looked like a complete
                # ladder.  A malformed effects field is a plan/data error, not
                # an empty measurement.
                raise ValueError('scale point %r: effects must be an iterable '
                                 'of measurements (or None for a declared empty '
                                 'tier), got %r' % (label, eff)) from None
        _seeds_declared = seeds is not None
        if seeds is None:
            keep = [(j, f) for j, f in ((j, _finite_scalar(v)) for j, v in enumerate(vals))
                    if f is not None]
            seeds_kept = [j for j, _v in keep]
            effects = [f for _j, f in keep]
        else:
            if isinstance(seeds, (str, bytes, bytearray, dict)):
                # `list("012")` yields ['0', '1', '2'] and the legacy integer
                # reader turns those characters into seeds 0, 1, 2.  A string
                # is one malformed seed descriptor, never a seed panel.
                raise ValueError('scale point %r: seeds must be an iterable of '
                                 'integers, not %s'
                                 % (label, type(seeds).__name__))
            try:
                seeds = list(seeds)
            except TypeError:
                raise ValueError('scale point %r: seeds must be an iterable, got %r'
                                 % (label, seeds)) from None
            if len(seeds) != len(vals):
                raise ValueError('scale point %r: seeds length %d does not match effects length %d' % (label, len(seeds), len(vals)))
            # Seed identity is an exact integer.  A float such as 1.5 or a
            # boolean would otherwise be treated as a distinct draw (and True
            # collides with 1 in the complete-case dictionary), silently
            # mis-pairing scale points.
            _norm_seeds = [_as_seed_int(s) for s in seeds]
            if any(s is None for s in _norm_seeds):
                raise ValueError('scale point %r: seeds must be integers, got %r'
                                 % (label, seeds))
            seeds = _norm_seeds
            keep = [(s, f) for s, f in ((s, _finite_scalar(v)) for s, v in zip(seeds, vals))
                    if f is not None]
            seeds_kept = [s for s, _v in keep]
            effects = [f for _s, f in keep]
        if len(set(seeds_kept)) != len(seeds_kept):
            # A repeated seed would be counted as two independent observations
            # and shrink every within-scale SE in meta_analyze_effects; this
            # must be a hard error, not silently "n looks larger".
            raise ValueError('scale point %r: duplicate seeds %r' % (label, seeds_kept))
        # Attrition is part of the estimate's identity.  A point built from a
        # complete-case subset must say so; otherwise the random-effects
        # weights and the trend treat a selectively missing seed as if it
        # never existed.  Callers may declare the plan as `n_planned` or as
        # the exact `planned_seeds` list.
        _planned_set = None
        _seed_plan_attested = None
        if planned_seeds is not None:
            if isinstance(planned_seeds, (str, bytes, bytearray, dict)):
                raise ValueError('scale point %r: planned_seeds must be an '
                                 'iterable of integers, not %s'
                                 % (label, type(planned_seeds).__name__))
            try:
                _ps = [_as_seed_int(v) for v in planned_seeds]
            except TypeError:
                raise ValueError('scale point %r: planned_seeds must be an '
                                 'iterable, got %r' % (label, planned_seeds)) from None
            if any(v is None for v in _ps):
                raise ValueError('scale point %r: planned_seeds must be integers, got %r'
                                 % (label, planned_seeds))
            _planned_set = set(_ps)
            if len(_planned_set) != len(_ps):
                raise ValueError('scale point %r: duplicate planned_seeds %r'
                                 % (label, planned_seeds))
            if n_planned is not None and _as_seed_int(n_planned) != len(_planned_set):
                raise ValueError('scale point %r: n_planned=%r disagrees with '
                                 'planned_seeds length %d'
                                 % (label, n_planned, len(_planned_set)))
            n_planned = len(_planned_set)
            # An exact planned seed list can only be checked against actual
            # seed identities.  Without them, equal COUNTS do not prove the
            # planned draws were measured; leave the plan unattested instead
            # of certifying a different seed set.
            _seed_plan_attested = bool(_seeds_declared)
        _planned_n = None
        if n_planned is not None:
            _planned_n = _as_seed_int(n_planned)
            if _planned_n is None or _planned_n < 0:
                raise ValueError('scale point %r: n_planned must be a non-negative '
                                 'integer, got %r' % (label, n_planned))
        _n_input = len(vals)
        _n_dropped = _n_input - len(seeds_kept)
        if _planned_n is not None:
            _n_declared = int(_planned_n)
        else:
            _n_declared = _n_input
        if _planned_set is not None and _seeds_declared:
            _n_missing = len(_planned_set - set(seeds_kept))
            _n_unexpected = len(set(seeds_kept) - _planned_set)
        else:
            _n_missing = max(0, _n_declared - _n_input)
            _n_unexpected = max(0, _n_input - _n_declared)
        _complete = bool(_n_dropped == 0 and _n_missing == 0
                         and _n_unexpected == 0
                         and _seed_plan_attested is not False)
        # A scale is a positive real number on the x axis.  Use the same
        # measurement contract as the effects: booleans, numeric strings,
        # complex values and containers are not scales and stay invalid
        # (NaN), so the consumers can report the tier as invalid instead of
        # placing it on the axis with a coerced value.
        sc = _finite_scalar(scale)
        if sc is None:
            sc = float('nan')
        # Vocab size is an exact count: reject non-integral numeric objects
        # instead of truncating them into a different tokenisation.
        vb = None if vocab is None else _as_seed_int(vocab)
        if vb is not None and vb <= 0:
            vb = None
        out.append({'label': str(label), 'scale': sc, 'effects': effects, 'seeds': seeds_kept,
                    'seeds_declared': bool(_seeds_declared),
                    'n_input': int(_n_input), 'n_declared': int(_n_declared),
                    'n_dropped': int(_n_dropped), 'n_missing': int(_n_missing),
                    'n_unexpected': int(_n_unexpected),
                    'complete_pairs': _complete,
                    'seed_plan_attested': _seed_plan_attested,
                    'corpus_fp': (str(corpus_fp) if corpus_fp not in (None, '') else None),
                    'vocab': vb})
    return out


def _scale_corpus_binding(pts):
    """Shared-corpus/tokenizer attestation across scale points.

    Returns a dict with `corpus_binding` in
      * 'shared'    — every point carries the same corpus fingerprint
      * 'mixed'     — >=2 fingerprints differ: cross-scale effects are not
                      defined (different corpora/tokenizers), pooling must be
                      refused
      * 'unverifiable' — fewer than two points carry a fingerprint: the family
                      MIGHT be mixed; absence of evidence is not attestation
    plus `corpus_fps`, `vocabs` and a refusal reason when mixed."""
    fps = [p.get('corpus_fp') for p in pts if p.get('corpus_fp')]
    _vocab_pts = [p for p in pts if p.get('vocab') is not None]
    vbs = sorted({p.get('vocab') for p in _vocab_pts})
    out = {'corpus_fps': sorted(set(fps)), 'vocabs': vbs,
           'n_points': len(pts), 'n_vocab_declared': len(_vocab_pts)}
    if len(fps) >= 2 and len(set(fps)) > 1:
        out['corpus_binding'] = 'mixed'
        out['refusal_reason'] = (
            'the scale points do not share one corpus/tokenizer fingerprint '
            f'({len(set(fps))} distinct); a paired log PPL ratio is only '
            'defined within one tokenisation, so cross-scale pooling is '
            'refused (report the scales separately)')
    elif len(fps) == len(pts) and len(pts) > 0 and len(set(fps)) == 1:
        out['corpus_binding'] = 'shared'
    else:
        out['corpus_binding'] = 'unverifiable'
        out['refusal_reason'] = (
            'the scale points carry no corpus/tokenizer fingerprint; '
            'cross-scale pooling cannot be bound to one tokenisation and '
            'stays unattested (add corpus_fp per point, e.g. the record '
            'data_version)')
    if len(vbs) > 1:
        out['corpus_binding'] = 'mixed'
        out['refusal_reason'] = (
            f'the scale points declare {len(vbs)} different vocab sizes {vbs}; '
            'PPL/log-ratios across different tokenisations are not comparable '
            '(see design_notes §22)')
    elif vbs and len(_vocab_pts) != len(pts):
        # A partly declared tokenizer width is not attestation: the points
        # without a vocab could still be on a different tokenisation.  This is
        # the same fail-closed rule as a missing corpus fingerprint, and it
        # must not be upgraded to 'shared' just because the declared subset
        # happens to agree.
        if out.get('corpus_binding') != 'mixed':
            out['corpus_binding'] = 'unverifiable'
        _why = (f'only {len(_vocab_pts)}/{len(pts)} scale points declare a vocab '
                'size; the shared-tokenisation attestation is incomplete')
        out['refusal_reason'] = ((out.get('refusal_reason') + '; ' + _why)
                                 if out.get('refusal_reason') else _why)
    return out


def meta_analyze_effects(points, *, alpha=0.05, min_seeds=3,
                         _seed_pair_detail_limit=None,
                         _prevalidated_points=False,
                         _point_row_cache=None):
    """跨规模的随机效应元分析（DerSimonian-Laird）+ 异质性诊断。

    输入是**已经无量纲化**的逐种子效应（通常来自
    `paired_log_ratio`），每个规模点一份。输出每个点的 mean/se/CI，
    Cochran Q、I²、tau²、固定/随机效应合并值与预测区间。

    关键口径：不同规模估计的不是同一个总体参数。只有异质性不显著
    （I² 低、Q 的 p>=alpha）且点数足够时才把 pooled 当作「共同效应」；
    否则 `poolable=False`，pooled 只作描述性摘要，结论必须按规模分开报。

    ``_seed_pair_detail_limit`` is an internal scale guard for callers that
    replay this analysis once per omitted panel: the per-pair common-seed map
    is quadratic in the number of points, while the pooled estimate and the
    dependence classification only need the aggregate overlap.  ``0`` skips
    the pair map, ``None`` keeps the historical complete map, and a positive
    integer reports the first N pairs (with explicit truncation metadata).

    ``_prevalidated_points``/``_point_row_cache`` are an internal fast path for
    leave-one-out replay: the caller validates the full family once and reuses
    the per-point mean/SD/SE rows for each subset.  The cache is only used
    when the effects are ordinary magnitudes (no work-scale normalisation), so
    an omitted extreme point cannot change the scale of the cached rows.
    """
    alpha = _validated_alpha(alpha, context='meta_analyze_effects alpha')
    _min_seeds = _as_seed_int(min_seeds)
    if _min_seeds is None or _min_seeds < 0:
        raise ValueError(f'min_seeds must be a non-negative integer, got {min_seeds!r}')
    min_seeds = _min_seeds
    _pair_limit = (None if _seed_pair_detail_limit is None
                   else _as_seed_int(_seed_pair_detail_limit))
    if _seed_pair_detail_limit is not None and (
            _pair_limit is None or _pair_limit < 0):
        raise ValueError(
            'seed_pair_detail_limit must be None or a non-negative integer, '
            f'got {_seed_pair_detail_limit!r}')
    pts = list(points) if _prevalidated_points else _scale_points(points)
    _row_cache = (_point_row_cache if isinstance(_point_row_cache, dict)
                  else {})
    labels = [p['label'] for p in pts]
    if len(set(labels)) != len(labels):
        # Two rows with the same label are two estimates of the SAME scale
        # point, but the random-effects weights below treat them as independent
        # studies — that would double-count one scale and understate tau^2.
        raise ValueError('meta_analyze_effects needs unique scale labels; got %r' % (labels,))
    _effect_abs = [abs(float(e)) for p in pts for e in p['effects']]
    _effect_scale = max(_effect_abs, default=0.0)
    _effect_min_abs = min((a for a in _effect_abs if a > 0.0), default=0.0)
    # Three regimes need a work scale: every effect below the floor, every
    # effect above 1e100, and -- the case the first two miss -- a family that
    # mixes sub-floor effects with ordinary/huge ones.  In the mixed regime a
    # max-only scale still leaves the small SE below 1e-154, so 1/se**2
    # overflows and the point is reported non-estimable even though both
    # effects are legitimate.  Scale by max/1e150 instead, which centres the
    # work range without changing any ordinary family's historical path.
    _mixed_underflow = bool(
        math.isfinite(_effect_scale) and _effect_scale > 0.0
        and _effect_scale >= _EFFECT_SCALE_FLOOR
        and 0.0 < _effect_min_abs < _EFFECT_SCALE_FLOOR)
    _scale_effects = bool(math.isfinite(_effect_scale) and _effect_scale > 0.0
                          and (_effect_scale < _EFFECT_SCALE_FLOOR
                               or _effect_min_abs > 1e100
                               or _mixed_underflow))
    if _mixed_underflow:
        _effect_scale = float(_effect_scale / _EFFECT_WORK_MAX)
        if not (math.isfinite(_effect_scale) and _effect_scale > 0.0):
            # Only an extreme max could make the quotient unrepresentable;
            # fall back to the historical max scale rather than divide by 0.
            _effect_scale = max(_effect_abs, default=0.0)

    def _finite_or_none(value):
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return number if math.isfinite(number) else None

    per = []
    for p in pts:
        _cached = _row_cache.get(p['label'])
        if isinstance(_cached, dict) and not _scale_effects:
            # Per-point mean/SD/SE/CI depend only on this point and on
            # (alpha, min_seeds), not on which sibling points are in the
            # current leave-one-out subset -- but ONLY while the family still
            # uses the unscaled branch.  A cache built before an omission
            # changed the family's work scale stores raw mean/SE values; this
            # subset needs work-scale values, so the cached row would be
            # rescaled a second time.  Recompute in that case.
            per.append(dict(_cached))
            continue
        d = np.asarray(p['effects'], dtype=float)
        n = int(d.size)
        row = {'label': p['label'], 'scale': p['scale'], 'n': n,
               'n_input': p.get('n_input'), 'n_declared': p.get('n_declared'),
               'n_dropped': p.get('n_dropped'), 'n_missing': p.get('n_missing'),
               'n_unexpected': p.get('n_unexpected'),
               'complete_pairs': p.get('complete_pairs'),
               'seed_plan_attested': p.get('seed_plan_attested')}
        if p.get('corpus_fp'):
            row['corpus_fp'] = p['corpus_fp']
        if p.get('vocab') is not None:
            row['vocab'] = p['vocab']
        if n:
            d_work = d / _effect_scale if _scale_effects else d
            _ms = _scaled_mean_std_if_extreme(d_work, ddof=1) if n > 1 else None
            if _ms is not None:
                m_work, sd_work = _ms
            else:
                m_work = float(d_work.mean())
                sd_work = float(d_work.std(ddof=1)) if n > 1 else float('nan')
            se_work = (float(sd_work / math.sqrt(n))
                       if (n > 1 and math.isfinite(sd_work)) else float('nan'))
            if _scale_effects:
                m = float(m_work * _effect_scale)
                sd = float(sd_work * _effect_scale)
                se = float(se_work * _effect_scale)
            else:
                m, sd, se = m_work, sd_work, se_work
            # Rescaling a finite within-scale SD/SE back to input units can
            # exceed float64 (opposite-signed extreme effects).  Keep the
            # internal work-scale values for the pooling arithmetic, but the
            # report fields are unavailable rather than infinite.  n=1 has no
            # within-scale SE at all; None keeps the returned record
            # strict-JSON safe instead of leaking NaN.
            row.update({'mean': _finite_or_none(m),
                        'std': _finite_or_none(sd),
                        'se': _finite_or_none(se),
                        '_mean_work': _finite_or_none(m_work),
                        '_se_work': _finite_or_none(se_work),
                        '_work_scale': (max(abs(float(x)) for x in d_work)
                                        if n else 0.0),
                        'underpowered': n < int(min_seeds)})
            if n > 1:
                tc = _t_critical(1.0 - float(alpha), n - 1)
                _lo = m - tc * se
                _hi = m + tc * se
                row['ci'] = ([_lo, _hi]
                             if math.isfinite(_lo) and math.isfinite(_hi)
                             else None)
            else:
                row['ci'] = None
        else:
            row.update({'mean': None, 'std': None, 'se': None,
                        # These are internal work-scale slots, but they are
                        # part of the returned per-point record.  NaN here
                        # would make `json.dumps(report, allow_nan=False)`
                        # fail for a declared-but-empty tier.
                        '_mean_work': None, '_se_work': None,
                        'underpowered': True, 'ci': None})
        per.append(row)
    # A scale point with identical effects across seeds has zero within-scale
    # variance.  It is NOT the same as a missing/underpowered point, but the
    # DerSimonian-Laird weights cannot be formed (1/0), and silently dropping
    # it would make a perfectly consistent scale disappear from the family.
    # Keep the row, name it, and block the pooled claim until a variance model
    # (or more seeds) is available.
    # A within-scale SE that is zero to numerical precision is just as
    # unusable as an exact zero: 1/se^2 becomes an unbounded weight and one
    # scale would dominate the pooled mean.  Treat it by scale-relative
    # tolerance, not just ``se == 0.0``.
    def _variance_is_zero(r):
        se = r.get('_se_work')
        if r['n'] <= 1 or se is None or not math.isfinite(se):
            return False
        # Compare against the point's own effect magnitude, not an absolute
        # floor of 1.0.  With the old ``max(1.0, |mean|)`` rule a point whose
        # effects are genuinely variable but scaled to 1e-13 looked like
        # exact zero variance (every point was then excluded from the pool),
        # and the verdict changed under a units-only rescaling.
        _scale = r.get('_work_scale')
        if _scale is None or not math.isfinite(float(_scale)):
            return False
        _scale = float(_scale)
        if _scale <= 0.0:
            return se == 0.0
        return se <= 1e-12 * _scale
    _zero_variance = [r['label'] for r in per if _variance_is_zero(r)]

    def _weight_is_representable(r):
        # A point whose within-scale SE is non-finite (e.g. the sample SD
        # overflowed), or whose 1/se^2 is not a finite positive weight, cannot
        # enter an inverse-variance pool.  It must stay visible and block the
        # pooled claim instead of silently disappearing from `usable`.
        if r['n'] <= 1:
            return False
        se = r.get('_se_work')
        if se is None or not math.isfinite(se) or se <= 0.0:
            return False
        se2 = se * se
        return math.isfinite(se2) and se2 > 0.0 and math.isfinite(1.0 / se2)

    _non_estimable = [r['label'] for r in per
                      if r['n'] > 1
                      and (_finite_or_none(r.get('mean')) is None
                           or not _weight_is_representable(r))
                      and not _variance_is_zero(r)]
    usable = [r for r in per
              if _weight_is_representable(r) and not _variance_is_zero(r)]
    _n_seed_declared = sum(1 for p in pts if p.get('seeds_declared'))
    if pts and _n_seed_declared == len(pts):
        _seed_pairing = 'declared'
    elif _n_seed_declared:
        _seed_pairing = 'partial'
    else:
        _seed_pairing = 'positional_only'
    _incomplete = [r['label'] for r in per if r.get('complete_pairs') is False]
    _empty_points = [r['label'] for r in per if r['n'] == 0]
    # A tier with no valid declared scale cannot be placed on the cross-scale
    # axis.  It stays in the family (and blocks a pooled common effect) instead
    # of silently disappearing from the meta-analysis.
    _invalid_scale_points = [r['label'] for r in per
                             if not (math.isfinite(r.get('scale', float('nan')))
                                     and r.get('scale') > 0.0)]
    _unattested_plan_points = [r['label'] for r in per
                               if r.get('seed_plan_attested') is False]
    _attrition = {r['label']: {'n_dropped': r.get('n_dropped'),
                               'n_missing': r.get('n_missing'),
                               'n_unexpected': r.get('n_unexpected')}
                  for r in per if r.get('complete_pairs') is False}
    out = {'alpha': float(alpha), 'min_seeds': int(min_seeds), 'per_point': per,
           'n_points': len(per), 'n_usable': len(usable),
           'incomplete_points': _incomplete,
           'empty_points': _empty_points,
           'invalid_scale_points': _invalid_scale_points,
           'unattested_plan_points': _unattested_plan_points,
           'attrition': _attrition,
           'zero_variance_points': _zero_variance,
           'non_estimable_points': _non_estimable,
           'corpus_binding': _scale_corpus_binding(pts),
           'seed_pairing': _seed_pairing,
           'n_points_with_declared_seeds': int(_n_seed_declared),
           'convention': 'input effects are scale-free; positive = with arm worse (higher PPL)'}
    if _seed_pairing != 'declared':
        out['seed_pairing_note'] = (
            'at least one scale point did not declare seed identities; within-scale '
            'replication is still reportable, but cross-scale seed-level pairing is '
            'unattested (positional indices must not be treated as the same seed)')
    if len(pts) >= 2 and out['corpus_binding']['corpus_binding'] == 'mixed':
        # Log PPL ratios are only defined within ONE tokenisation; pooling
        # across corpora/tokenizers is a units error, not a heterogeneity
        # finding.  Per-scale rows stay in the output; the pooled claims go.
        out.update({'status': 'refused_corpus_mismatch', 'poolable': False,
                    'note': out['corpus_binding']['refusal_reason']
                            + ' — per-scale effects are reported; pooled estimates are withheld'})
        return out
    if not usable:
        out.update({'status': 'insufficient_within_scale_replication', 'poolable': False,
                    'underpowered_points': [r['label'] for r in per if r.get('underpowered')],
                    'note': 'every scale point has n<2 or zero within-scale variance; no random-effects weights can be formed'})
        return out
    w = np.asarray([1.0 / (r['_se_work'] ** 2) for r in usable], dtype=float)
    y = np.asarray([r['_mean_work'] for r in usable], dtype=float)
    sw = float(w.sum())
    mu_fe = float((w * y).sum() / sw)
    q = float((w * (y - mu_fe) ** 2).sum())
    df = len(usable) - 1
    # A single usable scale has no residual heterogeneity df.  `chi2_sf`
    # legitimately returns NaN there, but NaN is not a JSON value and "no
    # heterogeneity test exists" is better represented as unavailable.
    p_q = chi2_sf(q, df) if df > 0 else None
    i2 = float(max(0.0, (q - df) / q) * 100.0) if (df > 0 and q > 0) else 0.0
    c = float(sw - (w ** 2).sum() / sw)
    tau2_work = float(max(0.0, (q - df) / c)) if (df > 0 and c > 0) else 0.0
    ws = 1.0 / (np.asarray([r['_se_work'] ** 2 for r in usable], dtype=float)
                + tau2_work)
    sws = float(ws.sum())
    mu_re = float((ws * y).sum() / sws)
    se_re = float(math.sqrt(1.0 / sws))
    tc = _t_critical(1.0 - float(alpha), max(1, df))
    if _scale_effects:
        mu_fe_out = float(mu_fe * _effect_scale)
        se_fe_out = float(math.sqrt(1.0 / sw) * _effect_scale)
        mu_re_out = float(mu_re * _effect_scale)
        se_re_out = float(se_re * _effect_scale)
        tau2 = float(tau2_work * _effect_scale * _effect_scale)
    else:
        mu_fe_out, se_fe_out = mu_fe, float(math.sqrt(1.0 / sw))
        mu_re_out, se_re_out = mu_re, se_re
        tau2 = tau2_work
    # A prediction interval for a NEW scale point needs k-2 degrees of
    # freedom (the two estimated variance components); reusing the k-1
    # random-effects CI critical value makes it too narrow.  With fewer than
    # three usable scales it is not defined.
    _pi = None
    if len(usable) >= 3:
        _tc_pi = _t_critical(1.0 - float(alpha), len(usable) - 2)
        if math.isfinite(_tc_pi):
            _half = _tc_pi * math.sqrt(tau2_work + se_re ** 2)
            if _scale_effects:
                _half *= _effect_scale
                _pi = [mu_re_out - _half, mu_re_out + _half]
            else:
                _pi = [mu_re - _half, mu_re + _half]
    # The DL pooled CI treats each scale point as an independent study.  When
    # the tiers were run on the SAME training seeds, their effects are paired
    # draws, not independent ones; the pooled interval can then be too narrow.
    # Report a dependence-respecting per-seed common-effect estimate/test as a
    # sensitivity readout, and disclose which assumption the pooled CI uses.
    _measured_pts = [p for p in pts
                     if p['effects'] and math.isfinite(p.get('scale', float('nan')))
                     and p.get('scale') > 0.0]
    _all_seed_declared = bool(_measured_pts) and all(
        p.get('seeds_declared') for p in _measured_pts)
    _paired_common = None
    if len(_measured_pts) >= 2:
        if not _all_seed_declared:
            _paired_common = {
                'status': 'unattested_seed_identity',
                'note': ('at least one measured scale point omitted its seed '
                         'identities; a per-seed common effect across scales is '
                         'not defined')}
        else:
            _by_seed_all = {}
            for _p in _measured_pts:
                for _s, _e in zip(_p['seeds'], _p['effects']):
                    _by_seed_all.setdefault(_s, []).append(float(_e))
            _complete_seeds = sorted(
                _s for _s, _vals in _by_seed_all.items()
                if len(_vals) == len(_measured_pts))
            if len(_complete_seeds) < 2:
                _paired_common = {
                    'status': 'insufficient_complete_seeds',
                    'n_complete_seeds': len(_complete_seeds),
                    'n_scales': len(_measured_pts),
                    'note': ('fewer than two seeds were measured at EVERY scale '
                             'point; the paired common effect is unavailable')}
            else:
                _per_seed = [_scaled_mean_or_raw(_by_seed_all[_s])
                             for _s in _complete_seeds]
                _pst = paired_effect_stats(_per_seed, alpha=alpha,
                                           min_n=min_seeds,
                                           seeds=_complete_seeds)
                _pst.update({
                    'status': 'ok',
                    'n_scales': len(_measured_pts),
                    'scale_labels': [p['label'] for p in _measured_pts],
                    'estimand': ('per-seed equal-weight mean across the '
                                 'measured scales'),
                    'weights': 'equal per scale (sensitivity readout; the DL '
                               'pooled estimate uses inverse-variance weights)',
                    'note': ('computed on the seeds observed at EVERY measured '
                             'scale; this is the dependence-respecting companion '
                             'to the inverse-variance pooled CI, not a '
                             'replacement for the heterogeneity test')})
                _paired_common = _pst
    out['paired_common_effect'] = _paired_common
    # The random-effects model treats each scale point as an INDEPENDENT study.
    # Declaring seed identities is not enough: disjoint seed sets are genuinely
    # independent draws, a complete common set permits a paired sensitivity
    # analysis, but a partially overlapping ladder is neither.  Classify the
    # actual overlap so the pooled claim cannot silently treat shared draws as
    # independent replications.
    _seed_overlap = {'n_points': len(_measured_pts),
                     'n_points_with_declared_seeds': int(sum(
                         1 for p in _measured_pts if p.get('seeds_declared'))),
                     'per_pair_common_seeds': {}, 'common_seeds_all_points': [],
                     'n_common_seeds_all_points': 0}
    if _all_seed_declared and len(_measured_pts) >= 2:
        _seed_sets = [(p['label'], set(p['seeds'])) for p in _measured_pts]
        _common_all = sorted(set.intersection(*(s for _l, s in _seed_sets)), key=str)
        # A seed appearing in >=2 points proves that at least one pair shares
        # a training draw; counting is linear in the total seed list and does
        # not need the quadratic pair map.  `per_pair_common_seeds` is kept
        # only as display detail (and capped for large families).
        _seed_use = {}
        for _l, _s in _seed_sets:
            for _x in _s:
                _seed_use[_x] = int(_seed_use.get(_x, 0)) + 1
        _any_pair_overlap = any(_c > 1 for _c in _seed_use.values())
        _pairs = {}
        _n_pairs = len(_seed_sets) * (len(_seed_sets) - 1) // 2
        if _pair_limit != 0:
            for _i in range(len(_seed_sets)):
                if _pair_limit is not None and len(_pairs) >= _pair_limit:
                    break
                for _j in range(_i + 1, len(_seed_sets)):
                    _li, _si = _seed_sets[_i]
                    _lj, _sj = _seed_sets[_j]
                    _pairs[f'{_li}|{_lj}'] = len(_si & _sj)
                    if _pair_limit is not None and len(_pairs) >= _pair_limit:
                        break
        _seed_overlap.update({
            'per_pair_common_seeds': _pairs,
            'common_seeds_all_points': _common_all,
            'n_common_seeds_all_points': len(_common_all),
            'any_pair_overlap': bool(_any_pair_overlap)})
        if _pair_limit is not None and _n_pairs > len(_pairs):
            _seed_overlap.update({
                'per_pair_common_seeds_truncated': True,
                'n_pairs': int(_n_pairs),
                'pair_detail_limit': int(_pair_limit),
                'pair_detail_note': ('per-pair common-seed detail is capped; '
                                     'the pooled estimate and dependence '
                                     'classification use the aggregate overlap')})
        if _paired_common and _paired_common.get('status') == 'ok':
            out['seed_dependence'] = 'shared_seeds_paired'
            _dependence_note = ('the inverse-variance pooled CI assumes the scale '
                                'points are independent studies, but they share '
                                'training seeds; use paired_common_effect for the '
                                'dependence-respecting readout')
        elif _any_pair_overlap:
            out['seed_dependence'] = 'shared_seeds_unpaired'
            _dependence_note = ('some scale points share training seeds but no '
                                'two complete common seeds exist across all of '
                                'them; the pooled CI treats those paired draws '
                                'as independent studies, so its independence '
                                'assumption is unmet and it is not a valid '
                                'common-effect estimate')
        else:
            out['seed_dependence'] = 'disjoint_seed_sets'
            _dependence_note = ('the scale points used disjoint seed sets, so '
                                'the independence assumption behind the pooled '
                                'inverse-variance CI is consistent with the '
                                'declared draws')
    else:
        out['seed_dependence'] = 'unattested'
        _dependence_note = ('seed identities are not declared for every measured '
                            'scale point, so neither independence nor paired '
                            'dependence can be attested')
    out['seed_overlap'] = _seed_overlap
    # Every declared scale point below the seed floor blocks the pooled claim,
    # INCLUDING points that could not enter the random-effects weights at all
    # (n=1 has no se).  Reading this list only from `usable` silently dropped
    # exactly those points.
    _underpowered_points = [r['label'] for r in per if r.get('underpowered')]
    # A fully shared ladder can have a paired common-effect estimate that is
    # itself underpowered even when every individual scale has n>=min_seeds
    # (only two seeds happen to be measured at EVERY scale).  The pairwise
    # dependence then rests on a smaller paired sample than the per-scale n
    # suggests, so the pooled claim must not be certified by the larger n.
    _paired_common_underpowered = bool(
        out.get('seed_dependence') == 'shared_seeds_paired'
        and isinstance(_paired_common, dict)
        and _paired_common.get('underpowered') is True)
    def _finite_out(value):
        # Scaling back to the input units can overflow even though every
        # input effect is finite.  An infinite SE/variance is not a measured
        # bound; withhold it (the pooled claim is already blocked by the
        # resolution/heterogeneity gates) instead of emitting inf to a strict
        # JSON writer.
        if value is None or isinstance(value, (bool, np.bool_)):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return number if math.isfinite(number) else None

    def _finite_ci(pair):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            return None
        lo, hi = _finite_out(pair[0]), _finite_out(pair[1])
        return [lo, hi] if lo is not None and hi is not None else None

    def _finite_tree(value):
        """Recursively replace non-finite floats with None (strict JSON)."""
        if isinstance(value, dict):
            return {k: _finite_tree(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_finite_tree(v) for v in value]
        if isinstance(value, tuple):
            return [_finite_tree(v) for v in value]
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (float, np.floating)):
            return _finite_out(value)
        if isinstance(value, np.integer):
            return int(value)
        return value

    # The paired common-effect block is a paired_effect_stats report.  Its
    # mean/CI can be non-finite for opposite-signed extreme per-seed means;
    # an unavailable estimate is None, not an infinite JSON value.
    if isinstance(_paired_common, dict) and _paired_common.get('status') == 'ok':
        for _key in ('mean', 'std', 'sem', 'cohens_dz'):
            if _key in _paired_common:
                _paired_common[_key] = _finite_out(_paired_common.get(_key))
        _pc_ci = _finite_ci([_paired_common.get('ci95_low'),
                             _paired_common.get('ci95_high')])
        if _pc_ci is None:
            _paired_common['ci95_low'] = None
            _paired_common['ci95_high'] = None
        else:
            _paired_common['ci95_low'], _paired_common['ci95_high'] = _pc_ci
        _pc_pi = _finite_ci(_paired_common.get('prediction_interval_95'))
        _paired_common['prediction_interval_95'] = _pc_pi
    _paired_common = _finite_tree(_paired_common)

    _tc_fe = _t_critical(1.0 - float(alpha), max(1, df))
    _fe_ci = _finite_ci([mu_fe_out - _tc_fe * se_fe_out,
                         mu_fe_out + _tc_fe * se_fe_out])
    _re_ci = _finite_ci([mu_re_out - tc * se_re_out,
                         mu_re_out + tc * se_re_out])
    _fr = _finite_out(mu_fe_out)
    _fs = _finite_out(se_fe_out)
    _rr = _finite_out(mu_re_out)
    _rs = _finite_out(se_re_out)
    _tau2_out = _finite_out(tau2)
    out.update({'fixed_effect': {'mean': _fr, 'se': _fs, 'ci': _fe_ci},
                'random_effects': {'mean': _rr, 'se': _rs, 'ci': _re_ci},
                'heterogeneity': {'Q': _finite_out(q), 'df': int(df), 'p': p_q,
                                  'I2_percent': i2, 'tau2': _tau2_out,
                                  'tau2_available': _tau2_out is not None,
                                  'method': 'Cochran Q + DerSimonian-Laird tau^2'},
                'prediction_interval': _finite_ci(_pi),
                'underpowered_points': _underpowered_points,
                'poolable': bool(len(usable) >= 3 and df > 0 and i2 <= 50.0
                                 and (not math.isfinite(p_q) or p_q >= alpha)
                                 and not _underpowered_points
                                 and not _incomplete and not _zero_variance
                                 and not _non_estimable
                                 and not _empty_points and not _invalid_scale_points
                                 and not _unattested_plan_points
                                 # Independence of the DL "studies" must be
                                 # attested: a partially shared ladder has
                                 # paired draws, and undeclared seed identities
                                 # cannot be assumed independent.
                                 and out.get('seed_dependence') in
                                     ('shared_seeds_paired', 'disjoint_seed_sets')
                                 and not _paired_common_underpowered),
                'note': ('poolable requires >=3 scales, every included scale at n>=min_seeds '
                         'with a complete paired panel (no dropped/absent seeds) and non-zero '
                         'within-scale variance, '
                         'I2<=50% and a non-significant Q; otherwise report per-scale effects only'
                         + (('; underpowered scale point(s): ' + ', '.join(map(str, _underpowered_points)))
                            if _underpowered_points else '')
                         + (('; the paired common-effect estimate has only '
                             f'{_paired_common.get("n")} complete seed(s) < min_seeds={int(min_seeds)}')
                            if _paired_common_underpowered else '')
                         + (('; incomplete paired panel at: ' + ', '.join(map(str, _incomplete))
                             + ' (a complete-case subset can be biased by which seeds failed)')
                            if _incomplete else '')
                         + (('; zero within-scale variance at: ' + ', '.join(map(str, _zero_variance))
                             + ' (1/se^2 is undefined; the point is reported but not silently '
                               'dropped from the pooled claim)')
                            if _zero_variance else '')
                         + (('; non-estimable within-scale variance at: ' + ', '.join(map(str, _non_estimable))
                             + ' (the mean/SE or its 1/se^2 weight is not finite; the point is '
                               'reported but cannot enter an inverse-variance pool)')
                            if _non_estimable else '')
                         + (('; scale point(s) with no measured effect: ' + ', '.join(map(str, _empty_points))
                             + ' (the tier is declared but contributes no estimable effect)')
                            if _empty_points else '')
                         + (('; scale point(s) with an invalid/non-positive scale: '
                             + ', '.join(map(str, _invalid_scale_points))
                             + ' (the tier cannot be placed on the scale axis)')
                            if _invalid_scale_points else '')
                         + (('; exact planned seed list not matched to declared seed '
                             'identities at: ' + ', '.join(map(str, _unattested_plan_points))
                             + ' (equal counts do not prove the planned draws were measured)')
                            if _unattested_plan_points else ''))})
    out['note'] = ((out.get('note') + '; ') if out.get('note') else '') + _dependence_note
    out['status'] = 'ok'
    return out


def scale_interaction(low_effects, high_effects, *, low_label='low', high_label='high', seeds=None, alpha=0.05, min_n=3):
    """同一批种子在两个规模点上的**效应变化**（交互）的精确检验。

    Δ = effect_high − effect_low（effect 已无量纲化）。正值表示该效应在
    更大规模上更大（若 effect 是 log PPL 比，则正值 = 大模型上相对劣势
    更大）。逐种子配对后做精确符号翻转检验，避免把「大规模 PPL 基线更高」
    误读成「效应随规模变大」。
    """
    def _effect_vector(value, name):
        # A string is an iterable of characters, not a per-seed effect panel;
        # accepting it silently produces a misleading "no data" report.
        if (isinstance(value, (str, bytes, bytearray, dict))
                or not hasattr(value, '__iter__')):
            raise ValueError(
                f'scale_interaction {name} must be an iterable of per-seed '
                f'effects, got {type(value).__name__}')
        try:
            return list(value)
        except TypeError:
            raise ValueError(
                f'scale_interaction {name} must be an iterable of per-seed '
                f'effects') from None

    a = _effect_vector(low_effects, 'low_effects')
    b = _effect_vector(high_effects, 'high_effects')
    _seeds_declared = seeds is not None
    if len(a) != len(b):
        raise ValueError('scale_interaction needs equal-length effects (%d vs %d)' % (len(a), len(b)))
    if seeds is not None and isinstance(seeds, (str, bytes, bytearray, dict)):
        raise ValueError('scale_interaction seeds must be an iterable of '
                         f'integers, not {type(seeds).__name__}')
    ss = list(range(len(a))) if seeds is None else list(seeds)
    if len(ss) != len(a):
        raise ValueError('scale_interaction seeds length does not match effects')
    _norm_ss = [_as_seed_int(s) for s in ss]
    if any(s is None for s in _norm_ss):
        raise ValueError('scale_interaction seeds must be integers, got %r' % (ss,))
    ss = _norm_ss
    if len(set(ss)) != len(ss):
        # A repeated seed would enter paired_effect_stats as two independent
        # observations and shrink the interaction SE.
        raise ValueError('scale_interaction got duplicate seeds %r' % (ss,))
    diff, kept = [], []
    for s, x, y in zip(ss, a, b):
        # Use the repository-wide measurement contract: booleans, complex
        # values and numeric *strings* are not measurements.  The old
        # ``float(x)`` silently accepted "'1.0'" as a real effect, so a
        # hand-edited/replayed artifact could pass as numeric data.
        fx = _finite_scalar(x)
        fy = _finite_scalar(y)
        if fx is None or fy is None:
            # Keep the position so paired_effect_stats can report the dropped
            # seed instead of calling the surviving subset a complete panel.
            diff.append(float('nan'))
            continue
        delta = fy - fx
        if not math.isfinite(delta):
            # Both endpoints are finite, but their difference is beyond
            # float64.  That pair is unmeasured, not a +inf effect: keep its
            # position (so the panel stays incomplete) and do not report its
            # seed as kept.  Otherwise `n` and the reported `seeds` disagree.
            diff.append(float('nan'))
            continue
        diff.append(delta)
        kept.append(s)
    st = paired_effect_stats(diff, alpha=alpha, min_n=min_n, seeds=ss)
    # `significant` on the underlying paired test only speaks about alpha and
    # seed noise; it does not know this caller's declared minimum-n floor.  A
    # n=6 interaction with min_n=8 can have p<0.05 and still be underpowered
    # for the claim.  Keep the raw test flag for transparency, but make the
    # interaction-level `significant` respect the same floor as claim_level.
    # `paired_effect_stats.significant` now already enforces this caller's
    # `min_n` floor, so derive the RAW test flag directly instead of reading a
    # masked flag.  The raw flag stays visible for transparency while the
    # interaction-level verdict below applies the floor and complete-panel
    # requirements.
    _raw_sig = bool(isinstance(st.get('p_exact_signflip'), (int, float))
                    and math.isfinite(float(st['p_exact_signflip']))
                    and float(st['p_exact_signflip']) < alpha
                    and not st.get('within_seed_noise'))
    st['significant_raw'] = _raw_sig
    st['significant'] = bool(_raw_sig and not st.get('underpowered')
                             and not st.get('resolution_limited')
                             # A complete-case interaction can be biased by
                             # which seeds were dropped; significance over the
                             # surviving subset is not a complete-panel result.
                             and st.get('complete_pairs') is not False)
    st.update({'low_label': str(low_label), 'high_label': str(high_label), 'seeds': kept,
               'seed_pairing': ('declared' if _seeds_declared else 'positional'),
               'convention': 'delta = effect_high - effect_low; positive = effect is larger at the higher scale',
               'claim_level': 'confirmatory only if n>=%d and p_floor<alpha' % int(min_n)})
    if not _seeds_declared:
        st['seed_pairing_note'] = ('no seed identities were supplied; positions are '
                                   'paired mechanically and are not attested to be the '
                                   'same training draw')
    # This report is serialised (often through a strict allow_nan=False writer).
    # A zero interaction has no non-zero rank to test, a constant non-zero
    # interaction has an unbounded Cohen's d_z, and an interval whose endpoint
    # is beyond float64 is unavailable; none of them is a measured number.
    # Withhold them as None instead of NaN/Infinity.  Ordinary panels keep the
    # historical floats unchanged.
    for _key in ('mean', 'std', 'sem', 'cohens_dz', 'ci95_low', 'ci95_high',
                 'p_exact_signflip', 'p_floor', 'p_wilcoxon'):
        _value = st.get(_key)
        if (isinstance(_value, (float, np.floating))
                and not math.isfinite(float(_value))):
            st[_key] = None
    # The nested robustness readouts (bootstrap/predictive intervals) can be
    # NaN as well when the panel has no data.  Run the same strict-JSON clean
    # used by the artifact writer so this public report never carries a
    # NaN/Infinity token, including for n=0.
    return _json_clean(st)


def effect_modification(effects_by_condition, *, reference=None, condition_values=None,
                        seeds=None, alpha=0.05, min_n=6):
    """Paired **effect-modification** (interaction) analysis across conditions.

    The scientific question is not "is the effect significant at condition A
    and also at condition B?" but "does the paired effect *change* between A
    and B?".  Comparing two separate significance verdicts is not a test of
    that difference (the classic "difference between significant and
    non-significant is not itself significant" error).  This helper takes one
    already-paired effect per seed at each condition (for PPL the caller should
    pass ``log(PPL_a) - log(PPL_b)`` so conditions with different baseline
    PPLs are comparable), aligns the seeds, and tests:

      * each simple effect (descriptive; same exact sign-flip machinery);
      * ``effect(condition) - effect(reference)`` per seed, with an exact
        sign-flip p, t interval, paired bootstrap and a within-family Holm
        adjustment over the declared non-reference conditions;
      * when a numeric, strictly increasing ``condition_values`` axis is
        supplied, a **per-seed** OLS slope of the effect on that axis, tested
        with the same exact sign-flip test (never a regression on condition
        means, which would discard the seed pairing).

    A seed missing at ANY condition is dropped from the whole analysis and is
    named in ``dropped_seeds``; silently using a different complete-case set
    per interaction would make the simple effects and the interactions answer
    different questions.  Non-finite/boolean values are not measurements.

    ``significant`` on an interaction still requires n >= ``min_n`` and
    ``p_floor <= alpha`` (via :func:`paired_effect_stats`), but this function
    is a **post-hoc diagnostic**: its p-values are not part of a frozen
    confirmatory family and it never rewrites one.  A non-significant
    interaction is NOT evidence that the effect is constant or equivalent
    across conditions; the report must say so next to the number.
    """
    alpha = _validated_alpha(alpha, context='effect_modification alpha')
    _min_n = _as_seed_int(min_n)
    if _min_n is None or _min_n < 0:
        raise ValueError(f'effect_modification: min_n must be a non-negative integer, got {min_n!r}')
    min_n = _min_n
    try:
        from collections.abc import Mapping
    except Exception:  # pragma: no cover - Python 3.12 always has it
        Mapping = dict
    if not isinstance(effects_by_condition, Mapping) or not effects_by_condition:
        raise ValueError('effect_modification: effects_by_condition must be a non-empty mapping')
    keys = list(effects_by_condition.keys())
    labels = [str(k) for k in keys]
    if len(set(labels)) != len(labels):
        raise ValueError(f'effect_modification: condition labels collide after str(): {labels!r}')
    vectors = []
    n_input = None
    for k in keys:
        raw = effects_by_condition[k]
        if isinstance(raw, (str, bytes, bytearray, dict)):
            raise ValueError(
                f'effect_modification: condition {k!r} must map to a sequence, '
                f'not a {type(raw).__name__}')
        try:
            seq = list(raw)
        except TypeError:
            raise ValueError(f'effect_modification: condition {k!r} must map to a sequence') from None
        if n_input is None:
            n_input = len(seq)
        elif len(seq) != n_input:
            raise ValueError(
                'effect_modification: condition vectors differ in length (%d vs %d)'
                % (n_input, len(seq)))
        vectors.append(seq)
    # Seed identity is part of the pairing.  Positional labels are only used
    # when the caller did not declare identities; they are never invented to
    # make a dropped seed look complete.
    if seeds is None:
        seed_labels = list(range(int(n_input)))
    else:
        if isinstance(seeds, (str, bytes, bytearray, dict)):
            raise ValueError('effect_modification: seeds must be an iterable of '
                             f'integers, not a {type(seeds).__name__}')
        try:
            raw_seeds = list(seeds)
        except TypeError:
            raise ValueError('effect_modification: seeds must be an iterable') from None
        if len(raw_seeds) != n_input:
            raise ValueError(
                'effect_modification: seeds length %d does not match condition length %d'
                % (len(raw_seeds), n_input))
        _norm = [_as_seed_int(s) for s in raw_seeds]
        if any(s is None for s in _norm):
            raise ValueError(f'effect_modification: seeds must be integers ({raw_seeds!r})')
        if len(set(_norm)) != len(_norm):
            raise ValueError(f'effect_modification: duplicate seeds {_norm!r}')
        seed_labels = _norm
    ref_label = labels[0] if reference is None else str(reference)
    if ref_label not in labels:
        raise ValueError(
            'effect_modification: reference %r is not one of the conditions %r'
            % (ref_label, labels))
    # Complete-case pass over ALL conditions: one common seed set for every
    # simple effect, interaction and slope in this panel.
    data = {lab: [] for lab in labels}
    kept, dropped = [], []
    for i in range(int(n_input)):
        row, ok = [], True
        for seq in vectors:
            f = _finite_scalar(seq[i])
            if f is None:
                ok = False
                break
            row.append(f)
        if not ok:
            dropped.append(seed_labels[i])
            continue
        for lab, v in zip(labels, row):
            data[lab].append(v)
        kept.append(seed_labels[i])
    n = len(kept)

    def _cell_num(st, key):
        # `paired_effect_stats` withholds an unrepresentable SD/SE and uses
        # NaN for an undefined p/CI.  `float(None)` crashed the whole
        # interaction report on an extreme (but finite) panel; report an
        # unavailable number as None so the artifact stays strict-JSON safe.
        value = st.get(key)
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return number if math.isfinite(number) else None

    def _cell(st):
        return {
            'n': int(st.get('n') or 0),
            'mean': _cell_num(st, 'mean'),
            'std': _cell_num(st, 'std'),
            'std_defined': st.get('std_defined'),
            'ci95_low': _cell_num(st, 'ci95_low'),
            'ci95_high': _cell_num(st, 'ci95_high'),
            'cohens_dz': _cell_num(st, 'cohens_dz'),
            'p_exact_signflip': _cell_num(st, 'p_exact_signflip'),
            'p_floor': _cell_num(st, 'p_floor'),
            'underpowered': bool(st.get('underpowered')),
            'resolution_limited': bool(st.get('resolution_limited')),
            'within_seed_noise': bool(st.get('within_seed_noise')),
            'significant': bool(st.get('significant')),
            'effect_vs_noise': st.get('effect_vs_noise'),
            'ci95_paired_bootstrap': st.get('ci95_paired_bootstrap'),
        }

    simple = {}
    for lab in labels:
        st = paired_effect_stats(data[lab], alpha=alpha, min_n=min_n, seeds=kept)
        cell = _cell(st)
        cell['condition'] = lab
        cell['role'] = 'descriptive simple effect (not an interaction test)'
        simple[lab] = cell
    interactions = {}
    for lab in labels:
        if lab == ref_label:
            continue
        diff = [data[lab][j] - data[ref_label][j] for j in range(n)]
        st = paired_effect_stats(diff, alpha=alpha, min_n=min_n, seeds=kept)
        cell = _cell(st)
        m = cell['mean']
        if m is not None and math.isfinite(m) and -700.0 < m < 700.0:
            cell['ratio'] = float(math.exp(m))
            cell['percent_change'] = float(100.0 * (math.exp(m) - 1.0))
        else:
            # exp() would under/overflow and turn a real effect into 0/inf.
            cell['ratio'] = None
            cell['percent_change'] = None
        cell.update({
            'condition': lab,
            'reference': ref_label,
            'delta_mean': m,
            'definition': ('effect(condition) - effect(reference); positive = the '
                           'paired effect is larger at this condition'),
            'p_holm': None,
            'role': 'post-hoc effect modification (not in the frozen family)',
        })
        interactions[lab] = cell
    # Holm over the declared non-reference interaction family.  As everywhere
    # else in this repo an omitted/invalid member stays at p=1.0 so the family
    # cannot shrink and relax the correction.
    if interactions:
        _fam = [(lab,
                 (float(c['p_exact_signflip'])
                  if c.get('p_exact_signflip') is not None else 1.0))
                for lab, c in interactions.items()]
        h = holm_adjust(_fam, alpha=alpha)
        for lab, c in interactions.items():
            c['p_holm'] = (h.get(lab) or {}).get('p_holm')
        family = {
            'm': len(_fam),
            'method': 'holm step-down over the non-reference interaction conditions',
            'alpha': alpha,
            'min_attainable_p_holm': holm_min_attainable_p_floors(
                [c.get('p_floor') for c in interactions.values()], alpha=alpha),
        }
    else:
        family = {'m': 0, 'method': 'no non-reference condition', 'alpha': alpha,
                  'min_attainable_p_holm': None}
    # Optional numeric axis -> one within-seed slope per seed.  The axis must
    # be supplied (condition labels need not be numeric) and strictly
    # increasing in the declared order.
    axis = None
    trend = None
    trend_reason = None
    if condition_values is None:
        trend_reason = 'no numeric condition axis supplied; a slope is not defined'
    elif not isinstance(condition_values, Mapping):
        trend_reason = 'condition_values must be a mapping from condition label to a numeric value'
    else:
        _cv = {str(k): v for k, v in condition_values.items()}
        _axis, _why = [], None
        for lab in labels:
            if lab not in _cv:
                _why = f'condition {lab!r} has no condition_values entry'
                break
            f = _finite_scalar(_cv[lab])
            if f is None:
                _why = f'condition {lab!r} has a non-numeric/non-finite axis value'
                break
            _axis.append(f)
        if _why is None:
            if any(_axis[i] >= _axis[i + 1] for i in range(len(_axis) - 1)):
                _why = 'condition_values must be strictly increasing in the declared condition order'
            elif len(set(_axis)) < 3:
                _why = 'a slope needs at least three distinct condition values'
        if _why is not None:
            trend_reason = _why
        else:
            axis = _axis
    if axis is not None and n >= 2:
        x = np.asarray(axis, dtype=float)
        xbar = float(x.mean())
        denom = float(((x - xbar) ** 2).sum())
        if denom <= 0.0:
            trend_reason = 'the condition axis has zero variance'
        else:
            slopes = []
            for j in range(n):
                y = np.asarray([data[lab][j] for lab in labels], dtype=float)
                slopes.append(float(((x - xbar) * (y - y.mean())).sum() / denom))
            st = paired_effect_stats(slopes, alpha=alpha, min_n=min_n, seeds=kept)
            trend = _cell(st)
            trend.update({
                'axis': 'condition_values',
                'axis_values': [float(v) for v in axis],
                'n_conditions': len(labels),
                'definition': ('per-seed OLS slope of the paired effect on the supplied '
                               'numeric condition axis; positive = the effect increases '
                               'with the axis'),
                'role': 'post-hoc trend diagnostic (not in the frozen family)',
            })
    elif axis is not None:
        trend_reason = 'fewer than two complete seeds; no within-seed slope'
    out = {
        'kind': 'effect_modification',
        'unit': 'seed',
        'alpha': alpha,
        'min_n': min_n,
        'reference': ref_label,
        'conditions': labels,
        'condition_values': ([float(v) for v in axis] if axis is not None else None),
        'n_input': int(n_input),
        'n_complete': n,
        'n_dropped': int(len(dropped)),
        'seeds': kept,
        'dropped_seeds': dropped,
        'simple_effects': simple,
        'interactions': interactions,
        'interaction_family': family,
        'trend': trend,
        'trend_reason': trend_reason,
        'claim_level': ('post-hoc effect-modification diagnostic; not part of a frozen '
                        'confirmatory family and never rewrites its verdict'),
        'note': ('interactions ask whether the paired effect CHANGES across conditions. '
                 'A non-significant interaction is not evidence that the effect is constant '
                 'or equivalent, and a difference in the significance of two separate simple '
                 'effects is not an interaction test. Simple effects are descriptive.'),
    }
    if n == 0:
        out['status'] = 'no_data'
        out['note'] = ('no seed had a finite effect at every condition; no interaction or '
                       'trend was estimated. ' + out['note'])
    else:
        out['status'] = 'ok'
    return out


def scaling_trend(points, *, alpha=0.05, min_seeds=3, min_points=3, log_scale=True):
    """尺度阶梯上的效应趋势：相对效应 ~ log(scale) 的加权回归 + 精确配对检验。

    `points` 每项为 {label, scale, effects, seeds}（或 (scale, effects, seeds)）。
    effects 必须是**无量纲**的逐种子效应（`paired_log_ratio` 的输出），
    不能直接塞入绝对 PPL 差——否则斜率只反映基线 PPL 随规模的变化。

    推断分两层：
      1. 优先用「完整种子」的逐种子斜率：同一种子在每个规模点都有观测时，
         对每个种子拟合 log(scale) 斜率，再对这些斜率做精确符号翻转检验。
         p 下限仍是 2/2^n，因此 n=4 时 0.125，n<6 的描述性读数不能称为
         「趋势已确认」。
      2. 没有完整种子时，退化为规模点均值的加权回归，用 t 区间与 t 检验，
         并明确标注 `paired=False`（点间独立、不能利用配对降噪）。
    同时报告两点交互（最小 vs 最大规模）和异质性；异质性高时不得用单一
    斜率概括整个阶梯。"""
    alpha = _validated_alpha(alpha, context='scaling_trend alpha')
    _min_seeds = _as_seed_int(min_seeds)
    _min_points = _as_seed_int(min_points)
    if _min_seeds is None or _min_seeds < 0:
        raise ValueError(f'min_seeds must be a non-negative integer, got {min_seeds!r}')
    if _min_points is None or _min_points < 1:
        raise ValueError(f'min_points must be a positive integer, got {min_points!r}')
    min_seeds, min_points = _min_seeds, _min_points
    _parsed_pts = _scale_points(points)
    # A declared tier with an invalid/non-positive scale cannot enter the
    # regression, but it must remain VISIBLE in the declared family: silently
    # dropping it would let the surviving tiers look like a complete ladder.
    def _valid_scale(p):
        return math.isfinite(p['scale']) and p['scale'] > 0.0
    _invalid_scale_points = [p['label'] for p in _parsed_pts if not _valid_scale(p)]
    pts = [p for p in _parsed_pts if _valid_scale(p)]
    pts.sort(key=lambda p: p['scale'])
    _all_labels = [p['label'] for p in _parsed_pts]
    if len(set(_all_labels)) != len(_all_labels):
        raise ValueError('scaling_trend needs unique scale labels; got %r' % (_all_labels,))
    for p in pts:
        if len(set(p['seeds'])) != len(p['seeds']):
            raise ValueError('scale point %r has duplicate seeds %r' % (p['label'], p['seeds']))
    _empty_points = [p['label'] for p in pts if not p['effects']]
    _incomplete_points = [p['label'] for p in pts if p.get('complete_pairs') is False]
    _attrition = {p['label']: {'n_dropped': p.get('n_dropped'),
                               'n_missing': p.get('n_missing'),
                               'n_unexpected': p.get('n_unexpected')}
                  for p in pts if p.get('complete_pairs') is False}
    # Declared-but-empty and invalid-scale tiers stay in the binding/attrition
    # view (corpus identity covers the whole declared family).
    _pts_binding = pts + [p for p in _parsed_pts if not _valid_scale(p)]
    pts = [p for p in pts if p['effects']]
    # The paired path can only use scales that carry a measured effect: a
    # declared-but-empty tier must not make every seed look "incomplete" at a
    # label that has no observations, which would silently disable the preferred
    # exact test on the remaining, fully measured ladder.
    labels = [p['label'] for p in pts]
    out = {'alpha': float(alpha), 'min_seeds': int(min_seeds), 'min_points': int(min_points),
           'n_points': len(pts), 'labels': [p['label'] for p in pts],
           'scales': [p['scale'] for p in pts],
           'declared_labels': [p['label'] for p in _pts_binding],
           'declared_scales': [p['scale'] for p in _pts_binding],
           'incomplete_points': _incomplete_points,
           'empty_points': _empty_points,
           'invalid_scale_points': _invalid_scale_points,
           'attrition': _attrition,
           'corpus_binding': _scale_corpus_binding(_pts_binding),
           'min_seeds_for_significance': signflip_min_n_for_holm(1, alpha),
           'convention': 'effects must be scale-free (e.g. paired log PPL ratio); positive = with arm worse'}
    if len(pts) < 2:
        out.update({'status': 'insufficient_points', 'verdict': 'insufficient_points'})
        return out
    if out['corpus_binding']['corpus_binding'] == 'mixed':
        # A trend across scales that did not share one corpus/tokenizer is a
        # units artefact, not a scaling law: the per-scale numbers may be
        # reported, the slope may not.
        out.update({'status': 'refused_corpus_mismatch', 'verdict': 'refused_corpus_mismatch',
                    'note': out['corpus_binding']['refusal_reason']})
        return out
    _effect_abs = [abs(float(e)) for p in pts for e in p['effects']]
    _effect_scale = max(_effect_abs, default=0.0)
    _scale_effects = bool(math.isfinite(_effect_scale) and _effect_scale > 0.0
                          and (_effect_scale < _EFFECT_SCALE_FLOOR
                               or _effect_scale > 1e100))

    def _effects_work(p):
        if not _scale_effects:
            return [float(e) for e in p['effects']]
        return [float(e) / _effect_scale for e in p['effects']]

    def _finite_out(value):
        # Mapping a work-scale slope back to input units can overflow even
        # though every effect is finite.  An infinite slope is not a measured
        # trend, so expose None and keep the direction from the finite
        # work-scale statistic for the verdict.
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return number if math.isfinite(number) else None

    x = np.asarray([math.log10(p['scale']) if log_scale else p['scale'] for p in pts], dtype=float)
    y = np.asarray([float(np.mean(_effects_work(p))) for p in pts], dtype=float)
    n = np.asarray([float(len(p['effects'])) for p in pts], dtype=float)
    sw = float(n.sum())
    xbar = float((n * x).sum() / sw)
    ybar = float((n * y).sum() / sw)
    sxx = float((n * (x - xbar) ** 2).sum())
    if sxx <= 0.0:
        out.update({'status': 'degenerate_scale_axis', 'verdict': 'degenerate_scale_axis'})
        return out
    slope_work = float((n * (x - xbar) * (y - ybar)).sum() / sxx)
    intercept_work = float(ybar - slope_work * xbar)
    k = len(pts)
    resid = y - (slope_work * x + intercept_work)
    slope = float(slope_work * _effect_scale) if _scale_effects else slope_work
    intercept = float(intercept_work * _effect_scale) if _scale_effects else intercept_work
    slope_out, intercept_out = _finite_out(slope), _finite_out(intercept)
    wls = {'slope': slope_out, 'intercept': intercept_out,
           'x_transform': 'log10' if log_scale else 'linear',
           'weights': 'n_seeds per scale', 'r2': (1.0 - float((n * resid ** 2).sum() / (n * (y - ybar) ** 2).sum())) if float((n * (y - ybar) ** 2).sum()) > 0 else 1.0}
    if k > 2:
        sigma2 = float((n * resid ** 2).sum() / (k - 2))
        se_work = float(math.sqrt(sigma2 / sxx))
        tc = _t_critical(1.0 - float(alpha), k - 2)
        # A zero-residual fit has no residual variance.  Its t statistic is
        # infinite only when the slope is non-zero; an exactly flat line is a
        # perfect fit with an undefined (not zero) p-value, so report p=1
        # instead of turning "no trend" into a maximally significant one.
        if se_work > 0.0:
            _p = student_t_two_sided_p(slope_work / se_work, k - 2)
        else:
            _p = 1.0 if abs(slope_work) <= 1e-15 else 0.0
        se = float(se_work * _effect_scale) if _scale_effects else se_work
        se_out = _finite_out(se)
        _wls_ci = ([slope - tc * se, slope + tc * se]
                   if se_out is not None and slope_out is not None else None)
        if (_wls_ci is not None
                and not all(math.isfinite(float(_x)) for _x in _wls_ci)):
            _wls_ci = None
        wls.update({'se': se_out, 'ci': _wls_ci, 'p': _p, 'df': k - 2})
    if _scale_effects and slope_out is None:
        wls['scaled_slope_unavailable_reason'] = (
            'the slope in input units exceeds the representable float range; '
            'the sign is still available from the unit-scaled fit')
    out['wls'] = wls
    # paired per-seed slopes: complete cases only (same seed at every scale).
    # When any point omitted its seed identities, the positional indices that
    # `_scale_points` assigns are NOT evidence that the same seed was run at
    # every scale.  In that case the paired path is withheld rather than
    # fabricating a cross-scale pairing.
    _seed_identity_attested = all(p.get('seeds_declared') for p in pts)
    by_seed = {}
    for p in pts:
        for s, e in zip(p['seeds'], _effects_work(p)):
            by_seed.setdefault(s, {})[p['label']] = float(e)
    complete = (sorted([s for s, m in by_seed.items() if all(lb in m for lb in labels)], key=str)
                if _seed_identity_attested else [])
    paired = {'n_complete_seeds': len(complete), 'seeds': complete, 'labels': labels,
              'seed_identity_attested': bool(_seed_identity_attested)}
    paired_slope_work = None
    if not _seed_identity_attested:
        paired['note'] = ('at least one scale point omitted its seed identities; '
                          'positional effect indices are not the same seed, so the '
                          'paired exact test and per-seed slopes are withheld')
    elif len(complete) >= 2:
        # A per-seed slope has exactly one observation per scale, so every
        # scale point carries equal weight here.  Using the point-mean weights
        # `n` (the number of seeds completed at each scale) would let a
        # ragged seed-completion panel bias the paired slope toward the
        # over-represented scale, which is exactly the confound this paired
        # path exists to remove.
        xbar_s = float(x.mean())
        sxx_s = float(((x - xbar_s) ** 2).sum())
        slopes = []
        for s in complete:
            yy = np.asarray([by_seed[s][lb] for lb in labels], dtype=float)
            yybar = float(yy.mean())
            slopes.append(float(((x - xbar_s) * (yy - yybar)).sum() / sxx_s))
        slopes = np.asarray(slopes, dtype=float)
        paired_slope_work = float(slopes.mean())
        st = paired_effect_stats(slopes, alpha=alpha, min_n=min_seeds)
        tc = _t_critical(1.0 - float(alpha), len(slopes) - 1)
        se_s = float(slopes.std(ddof=1) / math.sqrt(len(slopes))) if len(slopes) > 1 else float('nan')
        with np.errstate(over='ignore', invalid='ignore'):
            slopes_out = (slopes * _effect_scale if _scale_effects else slopes)
        mean_out = float(slopes_out.mean())
        std_work = (float(slopes.std(ddof=1)) if len(slopes) > 1 else float('nan'))
        std_out = (std_work * _effect_scale if _scale_effects else std_work)
        se_out = (se_s * _effect_scale if _scale_effects else se_s)
        mean_out_f = _finite_out(mean_out)
        std_out_f = _finite_out(std_out)
        se_out_f = _finite_out(se_out)
        _paired_ci = (([mean_out - tc * se_out, mean_out + tc * se_out]
                       if se_out_f is not None and mean_out_f is not None
                       else None))
        if (_paired_ci is not None
                and not all(math.isfinite(float(_x)) for _x in _paired_ci)):
            _paired_ci = None
        paired.update({'slope_mean': mean_out_f, 'slope_std': std_out_f,
                       'se': se_out_f, 'ci': _paired_ci,
                       'p_exact_signflip': st['p_exact_signflip'], 'p_floor': st['p_floor'],
                       'resolution_limited': st['resolution_limited'], 'underpowered': st['underpowered'],
                       'within_seed_noise': st['within_seed_noise'], 'significant': st['significant'],
                       'per_seed_slopes': {str(s): _finite_out(v)
                                           for s, v in zip(complete, slopes_out)}})
        if _scale_effects and mean_out_f is None:
            paired['scaled_slope_unavailable_reason'] = (
                'the mean per-seed slope in input units exceeds the '
                'representable float range; its sign is still available from '
                'the unit-scaled slopes')
    else:
        paired.update({'note': 'no complete seed across every scale point; the paired exact test is unavailable'})
    out['paired'] = paired
    out['seed_pairing'] = ('declared' if _seed_identity_attested else 'unattested')
    paired_primary = len(complete) >= 2
    # The unpaired WLS fallback treats every scale point as an independent
    # sample.  That is valid for disjoint seed sets, but a partially shared
    # ladder has paired draws that the WLS standard error ignores.  Classify
    # the actual overlap and block the trend claim when the scales are neither
    # fully paired nor independent.
    _seed_overlap = {'n_points': len(pts),
                     'per_pair_common_seeds': {},
                     'common_seeds_all_points': [],
                     'n_common_seeds_all_points': 0}
    _seed_dependence = 'unattested'
    _any_pair_overlap = False
    if _seed_identity_attested and len(pts) >= 2:
        _seed_sets = [(p['label'], set(p['seeds'])) for p in pts]
        _common_all = sorted(set.intersection(*(s for _l, s in _seed_sets)), key=str)
        _pairs = {}
        for _i in range(len(_seed_sets)):
            for _j in range(_i + 1, len(_seed_sets)):
                _li, _si = _seed_sets[_i]
                _lj, _sj = _seed_sets[_j]
                _n_ov = len(_si & _sj)
                _pairs[f'{_li}|{_lj}'] = _n_ov
                _any_pair_overlap = _any_pair_overlap or bool(_n_ov)
        _seed_overlap.update({
            'per_pair_common_seeds': _pairs,
            'common_seeds_all_points': _common_all,
            'n_common_seeds_all_points': len(_common_all),
            'any_pair_overlap': bool(_any_pair_overlap)})
        if paired_primary:
            _seed_dependence = 'shared_seeds_paired'
        elif _any_pair_overlap:
            _seed_dependence = 'shared_seeds_unpaired'
        else:
            _seed_dependence = 'disjoint_seed_sets'
    out['seed_dependence'] = _seed_dependence
    out['seed_overlap'] = _seed_overlap
    # A partially shared ladder cannot use the paired test (too few complete
    # seeds) and also cannot use the independent WLS standard error: the
    # dependence is unresolved, so the trend remains descriptive.
    _seed_dependence_blocks_claim = (_seed_dependence == 'shared_seeds_unpaired')
    # extreme-scale interaction (smallest vs largest) on shared seeds
    lo, hi = pts[0], pts[-1]
    lo_map = dict(zip(lo['seeds'], lo['effects']))
    hi_map = dict(zip(hi['seeds'], hi['effects']))
    shared = sorted(set(lo_map) & set(hi_map), key=str)
    if not _seed_identity_attested:
        inter = {'status': 'seed_identity_unattested',
                 'n': len(shared),
                 'note': ('the extreme scale points did not both declare seed identities; '
                          'a paired interaction over positional indices is not defined')}
    elif len(shared) >= 2:
        inter = scale_interaction([lo_map[s] for s in shared], [hi_map[s] for s in shared],
                                  low_label=lo['label'], high_label=hi['label'], seeds=shared,
                                  alpha=alpha, min_n=min_seeds)
    else:
        inter = {'n': len(shared), 'note': 'fewer than 2 shared seeds'}
    out['interaction_extremes'] = inter
    # Heterogeneity is the LACK OF FIT around the fitted slope, not the spread
    # of the point means around their grand mean: a clean monotone ladder has
    # a large grand-mean Q by construction.  Residual Q (df=k-2) asks the
    # question the verdict needs: is a single slope an adequate summary?
    if k >= 2:
        q = float((n * resid ** 2).sum())
        dfh = k - 2
        p_q = chi2_sf(q, dfh) if dfh > 0 else None
        i2 = float(max(0.0, (q - dfh) / q) * 100.0) if (dfh > 0 and q > 0) else 0.0
        c = float(n.sum() - (n ** 2).sum() / n.sum())
        tau2 = float(max(0.0, (q - dfh) / c)) if (dfh > 0 and c > 0) else 0.0
        q_out = float(q * _effect_scale * _effect_scale) if _scale_effects else q
        tau2_out = float(tau2 * _effect_scale * _effect_scale) if _scale_effects else tau2
        # Scaling back to input units can exceed float64 even though every
        # effect is finite.  Withhold an unrepresentable absolute Q/tau2
        # rather than emitting inf; the I2/p verdict remains scale-free.
        if not math.isfinite(q_out):
            q_out = None
        if not math.isfinite(tau2_out):
            tau2_out = None
        out['heterogeneity'] = {'Q': q_out, 'df': int(dfh), 'p': p_q, 'I2_percent': i2, 'tau2': tau2_out,
                                'method': 'lack-of-fit Q around the WLS slope (weights = n)',
                                'note': 'high I2 means the scale points deviate from a single slope; it is not the spread of a monotone ladder around its mean'}
    # The paired path only uses seeds completed at EVERY scale.  Extra seeds at
    # some scales cannot contribute a paired slope, so they must not decide
    # whether the paired trend is "heterogeneous" either: report a separate
    # lack-of-fit over the complete-seed scale means and gate the verdict on
    # that when the paired slope is the primary inference.
    het_for_verdict = out.get('heterogeneity') or {}
    if paired_primary:
        _yc = np.asarray([float(np.mean([by_seed[s][lb] for s in complete])) for lb in labels], dtype=float)
        # Use the work-scale paired slope, not the rescaled report field: the
        # latter may be unrepresentable (None) or inf, and dividing it back
        # would turn the residual lack-of-fit into NaN.
        _sp = (float(paired_slope_work) if _scale_effects
               else float(paired['slope_mean']))
        _ip = float(_yc.mean() - _sp * x.mean())
        _rp = _yc - (_sp * x + _ip)
        _cp = float(len(complete))
        _q_p = float((_cp * _rp ** 2).sum())
        _df_p = k - 2
        _p_p = (chi2_sf(_q_p, _df_p)
                if (_df_p > 0 and math.isfinite(_q_p)) else None)
        _i2_p = (0.0 if (_df_p > 0 and _q_p == 0.0)
                 else (float(max(0.0, (_q_p - _df_p) / _q_p) * 100.0)
                       if (_df_p > 0 and _q_p > 0 and math.isfinite(_q_p))
                       else None))
        _q_p_out = (float(_q_p * _effect_scale * _effect_scale)
                    if _scale_effects else _q_p)
        if not math.isfinite(_q_p_out):
            _q_p_out = None
        out['heterogeneity_paired'] = {'Q': _q_p_out, 'df': int(_df_p), 'p': _p_p,
                                       'I2_percent': _i2_p, 'n_complete_seeds': len(complete),
                                       'method': 'lack-of-fit over complete-seed scale means (weights = n_complete)',
                                       'note': 'gates the paired trend verdict; the all-seed heterogeneity above stays descriptive'}
        het_for_verdict = out['heterogeneity_paired']
    p_primary = paired.get('p_exact_signflip') if paired_primary else wls.get('p')
    resolution_limited = bool(paired.get('resolution_limited')) if paired_primary else (k <= 2)
    # A trend below the pre-registered seed/scale floor is descriptive, not a
    # confirmed effect: the paired test can reject at n<min_seeds (its p floor
    # is 2/2**n), and a two-point "trend" is a line even when the per-seed
    # exact test rejects.  Gate the README-level claim on the same thresholds
    # the rest of this function reports.
    if paired_primary:
        underpowered = bool(paired.get('underpowered'))
        _underpowered_points = ([] if not underpowered
                                else list(paired.get('seeds') or []))
        _within_seed_noise = bool(paired.get('within_seed_noise'))
    else:
        # Without a complete-seed paired slope the trend is a regression over
        # point means.  A point whose mean rests on fewer than `min_seeds`
        # draws is still underpowered, and pooling it into the trend must not
        # bypass the pre-registered seed floor.
        _underpowered_points = [p['label'] for p in pts
                                if len(p['effects']) < int(min_seeds)]
        underpowered = bool(_underpowered_points)
        _within_seed_noise = False
    too_few_scales = k < int(min_points)
    _incomplete = bool(_incomplete_points or _empty_points or _invalid_scale_points)
    # Same claim contract as paired_effect_stats/reportable_claim: an exact
    # test can reject while the per-seed slopes are still within their own
    # spread.  Such a trend is not a usable scaling signal.
    sig = bool(p_primary is not None and math.isfinite(float(p_primary)) and float(p_primary) < alpha
               and not resolution_limited and not underpowered and not too_few_scales
               and not _incomplete and not _within_seed_noise
               and not _seed_dependence_blocks_claim)
    het = het_for_verdict
    heterogeneous = bool(float(het.get('I2_percent') or 0.0) > 75.0 or (isinstance(het.get('p'), float) and math.isfinite(het['p']) and het['p'] < alpha))
    # Direction must come from the same slope that carries the inference.  The
    # WLS fit weights each scale by its seed count, so on a ragged panel its
    # sign can disagree with the paired per-seed slope the exact test just
    # evaluated.  Reporting a verdict from the unpaired slope would then say
    # "decreases" while the primary evidence says "increases".  The sign is
    # taken from the finite WORK-scale slope; the rescaled report field may be
    # unavailable when it exceeds float64.
    direction_work = (float(paired_slope_work) if paired_primary
                      else float(slope_work))
    direction_slope = (paired['slope_mean'] if paired_primary else slope_out)
    out['primary_inference'] = 'paired per-seed slope (exact sign-flip)' if paired_primary else 'WLS point-mean t-test (unpaired scales)'
    out['p_primary'] = float(p_primary) if isinstance(p_primary, (int, float)) and math.isfinite(float(p_primary)) else None
    out['significant'] = sig
    out['resolution_limited'] = resolution_limited
    out['underpowered'] = underpowered
    out['underpowered_points'] = _underpowered_points
    out['within_seed_noise'] = _within_seed_noise
    out['effect_vs_noise'] = ('within_seed_noise' if _within_seed_noise
                              else 'exceeds_seed_noise')
    out['direction_slope'] = direction_slope
    out['direction_slope_available'] = direction_slope is not None
    out['direction'] = ('increases' if direction_work > 0.0
                        else 'decreases' if direction_work < 0.0 else 'zero')
    if heterogeneous:
        verdict = 'heterogeneous_trend'
    elif sig:
        verdict = ('effect_decreases_with_scale' if direction_work < 0
                   else 'effect_increases_with_scale')
    elif _incomplete:
        verdict = 'incomplete_scale_point'
    elif underpowered:
        verdict = 'underpowered'
    elif too_few_scales:
        verdict = 'insufficient_points'
    elif _seed_dependence_blocks_claim:
        verdict = 'seed_dependence_unresolved'
    elif resolution_limited:
        verdict = 'resolution_limited'
    elif _within_seed_noise:
        verdict = 'noise_level'
    else:
        verdict = 'no_detectable_trend'
    out['verdict'] = verdict
    out['benefit_grows_with_scale'] = (bool(sig and direction_work < 0) if not heterogeneous else None)
    if too_few_scales:
        out['note'] = 'fewer than min_points=%d scales: slope is descriptive even if the exact test rejects' % int(min_points)
    if _incomplete_points:
        out['note'] = ((out.get('note') + '; ') if out.get('note') else '') + (
            'incomplete paired panel at %s — the slope/verdict is descriptive even if a '
            'complete-case test rejects (which seeds failed can bias the surviving subset)'
            % ', '.join(map(str, _incomplete_points)))
    if _empty_points:
        out['note'] = ((out.get('note') + '; ') if out.get('note') else '') + (
            'declared scale point(s) with no measured effect at %s — the tier is in the '
            'family but contributes no slope; the trend is descriptive'
            % ', '.join(map(str, _empty_points)))
    if _invalid_scale_points:
        out['note'] = ((out.get('note') + '; ') if out.get('note') else '') + (
            'declared scale point(s) with an invalid/non-positive scale at %s — the tier '
            'is in the family but cannot enter the scale axis; the trend is descriptive'
            % ', '.join(map(str, _invalid_scale_points)))
    if _seed_dependence_blocks_claim:
        out['note'] = ((out.get('note') + '; ') if out.get('note') else '') + (
            'scale points share only a partial seed set (common across all: '
            '%s; complete paired seeds: %d) — the WLS fallback treats the '
            'paired draws as independent, so the trend is descriptive until '
            'the full seed set is repeated at every scale'
            % (out['seed_overlap'].get('common_seeds_all_points') or 'none',
               int(paired.get('n_complete_seeds') or 0)))
    if underpowered and _underpowered_points:
        _u = ('paired complete-seed n<%d' % int(min_seeds) if paired_primary
              else 'scale point(s) with n<%d: ' % int(min_seeds)
                   + ', '.join(map(str, _underpowered_points)))
        out['note'] = ((out.get('note') + '; ') if out.get('note') else '') + (
            'underpowered (%s) — the trend remains descriptive' % _u)
    return out
