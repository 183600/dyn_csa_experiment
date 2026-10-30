# ---- 确认性主对照的实际显著性：声明式 SESOI + TOST -------------------------
# 冻结清单回答「能否排除零」，但不回答「效应是否达到实际值得关心的最小
# 幅度」。后者需要预先声明的 SESOI（smallest effect size of interest）和
# 等价性/超越 margin 的检验。本块只做**描述性**读数：不改变冻结 p/Holm/
# confirmed，也不把事后选择的 margin 伪装成确认性判定。

CONFIRMATION_SESOI = {
    'version': 'confirmation-sesoi-1',
    'scale': 'log_ratio',
    'relative_pct': 1.0,
    # None -> 1 - alpha (95% CI at alpha=0.05).  The one-sided decision level
    # is alpha/2, so the CI and the two one-sided tests agree.
    'tost_conf': None,
    'family': 'primary_comparisons',
    'declared_in_frozen_manifest': False,
    'role': ('descriptive practical-significance readout for the primary '
             'comparisons; it never changes p_exact_signflip, p_holm, '
             'verified or confirmed'),
    'rationale': ('a 1% relative PPL difference is the smallest architecture-'
                  'level effect this study treats as practically relevant, '
                  'matching the pre-existing baseline/ablation margin; a '
                  'statistically detectable but smaller difference is not '
                  'reported as a practically meaningful gain'),
}


def _cp_number(value, name, *, positive=False, allow_none=False):
    if value is None and allow_none:
        return None
    if isinstance(value, (bool, np.bool_, str, bytes, bytearray)):
        raise ValueError(f'{name} must be a real number, got {value!r}')
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'{name} must be a real number, got {value!r}') from None
    if not math.isfinite(out):
        raise ValueError(f'{name} must be finite, got {value!r}')
    if positive and out <= 0.0:
        raise ValueError(f'{name} must be > 0, got {value!r}')
    return out


def _cp_normalise_one(raw, *, where, allow_comparisons=False):
    if not isinstance(raw, dict):
        raise ValueError(f'{where} must be an object, got {type(raw).__name__}')
    out = dict(raw)
    scale = out.get('scale', 'log_ratio')
    if not isinstance(scale, str) or scale not in ('log_ratio', 'raw_ppl'):
        raise ValueError(f'{where}.scale must be log_ratio or raw_ppl, got {scale!r}')
    units = [k for k in ('relative_pct', 'absolute_log_ratio', 'absolute_ppl')
             if out.get(k) is not None]
    if len(units) != 1:
        raise ValueError(
            f'{where} must declare exactly one of relative_pct / '
            f'absolute_log_ratio / absolute_ppl, got {sorted(units)}')
    unit = units[0]
    if scale == 'log_ratio' and unit == 'absolute_ppl':
        raise ValueError(f'{where}: absolute_ppl is a raw-PPL margin; use '
                         'absolute_log_ratio on the log_ratio scale')
    if scale == 'raw_ppl' and unit != 'absolute_ppl':
        raise ValueError(f'{where}: the raw_ppl scale needs absolute_ppl')
    val = _cp_number(out.get(unit), f'{where}.{unit}', positive=True)
    tost_conf = out.get('tost_conf')
    if tost_conf is not None:
        tost_conf = _cp_number(tost_conf, f'{where}.tost_conf')
        if not (0.0 < tost_conf < 1.0):
            raise ValueError(f'{where}.tost_conf must be in (0,1), got {tost_conf!r}')
    version = out.get('version', CONFIRMATION_SESOI['version'])
    if not isinstance(version, str) or not version.strip():
        raise ValueError(f'{where}.version must be a non-empty string')
    declared = out.get('declared_in_frozen_manifest')
    if declared is not None and not isinstance(declared, (bool, np.bool_)):
        raise ValueError(f'{where}.declared_in_frozen_manifest must be a boolean')
    comps = out.get('comparisons')
    if comps is not None and not allow_comparisons:
        raise ValueError(f'{where}.comparisons is only allowed at the top level')
    if comps is not None and not isinstance(comps, dict):
        raise ValueError(f'{where}.comparisons must be a mapping')
    return {
        'scale': scale,
        'unit': unit,
        'value': float(val),
        'relative_pct': (float(val) if unit == 'relative_pct' else None),
        'absolute_log_ratio': (float(val) if unit == 'absolute_log_ratio' else None),
        'absolute_ppl': (float(val) if unit == 'absolute_ppl' else None),
        'tost_conf': (None if tost_conf is None else float(tost_conf)),
        'version': version.strip(),
        'declared_in_frozen_manifest': (None if declared is None else bool(declared)),
        'comparisons': comps,
    }


def _cp_margin_from_unit(pol):
    if pol['scale'] == 'log_ratio':
        if pol['relative_pct'] is not None:
            if pol['relative_pct'] <= -100.0:
                raise ValueError('relative_pct must be > -100')
            return float(math.log1p(float(pol['relative_pct']) / 100.0))
        return float(pol['absolute_log_ratio'])
    return float(pol['absolute_ppl'])


def normalise_confirmation_sesoi(spec=None, *, where='confirmation SESOI'):
    """Validate one confirmation-level SESOI declaration.

    ``spec`` is a mapping with exactly one margin unit.  A bare number is
    accepted as ``relative_pct`` for convenience, but never as an absolute
    log/raw margin: the unit must be explicit for anything else.
    """
    if spec is None:
        spec = CONFIRMATION_SESOI
    elif isinstance(spec, (int, float, np.integer, np.floating)) and not isinstance(spec, (bool, np.bool_)):
        spec = {'relative_pct': spec}
    pol = _cp_normalise_one(spec, where=where, allow_comparisons=True)
    comps = {}
    for name, raw in (pol.get('comparisons') or {}).items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f'{where}.comparisons names must be non-empty strings, got {name!r}')
        if isinstance(raw, (int, float, np.integer, np.floating)) and not isinstance(raw, (bool, np.bool_)):
            # A numeric override uses the parent unit.
            if pol['scale'] == 'raw_ppl':
                raw = {'scale': 'raw_ppl', 'absolute_ppl': raw}
            elif pol['unit'] == 'absolute_log_ratio':
                raw = {'scale': 'log_ratio', 'absolute_log_ratio': raw}
            else:
                raw = {'scale': 'log_ratio', 'relative_pct': raw}
        elif isinstance(raw, dict):
            raw = dict(raw)
            raw.setdefault('scale', pol['scale'])
        else:
            raise ValueError(f'{where}.comparisons[{name!r}] must be a number or an object')
        one = _cp_normalise_one(raw, where=f'{where}.comparisons[{name!r}]')
        if one['scale'] != pol['scale']:
            raise ValueError(
                f'{where}.comparisons[{name!r}] changes scale from '
                f'{pol["scale"]!r} to {one["scale"]!r}; one family must share a scale')
        comps[name.strip()] = one
    pol['comparisons_normalised'] = comps
    pol['margin'] = _cp_margin_from_unit(pol)
    return pol


def _cp_declared_block(manifest):
    """Return a manifest's practical-significance block, top-level first.

    Both ``manifest['practical_significance']`` and
    ``manifest['statistics']['practical_significance']`` are accepted so a
    statistics-owned policy does not have to be duplicated at the top level.
    """
    m = manifest if isinstance(manifest, dict) else {}
    raw = m.get('practical_significance')
    if isinstance(raw, dict):
        return raw
    stat = m.get('statistics') if isinstance(m.get('statistics'), dict) else {}
    raw = stat.get('practical_significance')
    return raw if isinstance(raw, dict) else None


def confirmation_sesoi_for_manifest(manifest=None, *, spec=None, alpha=None):
    """Resolve the SESOI declaration for one manifest's primary family.

    A manifest-level ``practical_significance`` block is used when present;
    otherwise the library policy is used and explicitly marked as *not* part
    of the frozen manifest.  Named per-comparison overrides are validated
    against the declared primary names, so a typo cannot silently remove a
    margin from one comparison.
    """
    manifest = manifest if isinstance(manifest, dict) else {}
    source = 'library_policy'
    raw = spec
    if raw is None:
        raw = _cp_declared_block(manifest)
        if isinstance(raw, dict):
            source = 'manifest'
        else:
            raw = CONFIRMATION_SESOI
    pol = normalise_confirmation_sesoi(raw, where='practical_significance')
    comps = manifest.get('primary_comparisons')
    names = ([str(c.get('name')) for c in comps
              if isinstance(c, dict) and c.get('name')]
             if isinstance(comps, list) else [])
    if not names:
        # A plan preflight can be run before the comparison list exists.
        names = sorted(pol['comparisons_normalised'])
    if len(set(names)) != len(names):
        raise ValueError(
            'primary_comparisons contain duplicate names; a SESOI family '
            'cannot represent two indistinguishable comparison slots')
    unknown = sorted(set(pol['comparisons_normalised']) - set(names))
    if unknown:
        raise ValueError(
            'practical_significance names comparison(s) not in the declared '
            f'primary family: {unknown}; a typo would silently drop a margin')
    _stat = manifest.get('statistics') if isinstance(manifest.get('statistics'), dict) else {}
    a = _cp_number(alpha if alpha is not None else _stat.get('alpha', 0.05),
                   'alpha')
    if not (0.0 < a < 1.0):
        raise ValueError(f'alpha must be in (0,1), got {a!r}')
    conf = pol['tost_conf'] if pol['tost_conf'] is not None else (1.0 - a)
    if not (0.0 < conf < 1.0):
        raise ValueError(f'tost_conf must be in (0,1), got {conf!r}')
    margins = {}
    for name in names:
        one = pol['comparisons_normalised'].get(name)
        if one is None:
            margin = pol['margin']
            rel = pol['relative_pct']
            unit = pol['unit']
        else:
            margin = _cp_margin_from_unit(one)
            rel = one['relative_pct']
            unit = one['unit']
        margins[name] = {'margin': float(margin),
                         'relative_pct': (None if rel is None else float(rel)),
                         'unit': unit}
    return {
        'kind': 'confirmation_sesoi_policy',
        'version': pol['version'],
        'source': source,
        'declared_in_frozen_manifest': (source == 'manifest'
                                        and pol['declared_in_frozen_manifest'] is not False),
        'scale': pol['scale'],
        'tost_conf': float(conf),
        'alpha': float(a),
        'alpha_one_sided': float(a / 2.0),
        'default': {'margin': float(pol['margin']),
                    'relative_pct': pol['relative_pct'],
                    'unit': pol['unit']},
        'margins': margins,
        'n_declared': len(margins),
        'missing': sorted(set(names) - set(margins)),
        'family': str((CONFIRMATION_SESOI.get('family') if source == 'library_policy'
                       else (raw.get('family') if isinstance(raw, dict) else None))
                      or 'primary_comparisons'),
        'role': str((raw.get('role') if isinstance(raw, dict) else None)
                    or CONFIRMATION_SESOI['role']),
        'rationale': str((raw.get('rationale') if isinstance(raw, dict) else None)
                         or CONFIRMATION_SESOI['rationale']),
        'note': ('a library-policy margin is a descriptive sensitivity, not a '
                 'pre-registered decision rule: it cannot upgrade or downgrade '
                 '`confirmed`'),
    }


def confirmation_practical_significance(deltas, *, margin=None, alpha=0.05,
                                        tost_conf=None, expected_sign=None,
                                        seeds=None, ppl_a=None, ppl_b=None,
                                        policy_version=None,
                                        relative_pct=None, scale='log_ratio'):
    """Practical-significance readout for one paired primary comparison.

    ``deltas`` are paired per-seed values on ``scale``: log(PPL_a/PPL_b) for
    ``log_ratio`` (positive = arm a worse) or PPL_a-PPL_b for ``raw_ppl``.
    A positive ``margin`` is the SESOI in the same units.  The function
    reports a paired-t CI, TOST equivalence within +/-margin, and a one-sided
    "beyond the margin" p in the pre-registered direction.  The equivalence
    decision uses ``tost_conf`` (default ``1-alpha``), while the superiority
    claim keeps its own declared level ``alpha/2``; changing ``tost_conf``
    cannot move a beyond-margin verdict.

    It never selects the margin from the data and never changes the frozen
    p/Holm/verdict.  A missing margin is ``not_declared`` rather than a
    guessed threshold.
    """
    a = _cp_number(alpha, 'alpha')
    if not (0.0 < a < 1.0):
        raise ValueError(f'alpha must be in (0,1), got {a!r}')
    conf = (1.0 - a) if tost_conf is None else _cp_number(tost_conf, 'tost_conf')
    if not (0.0 < conf < 1.0):
        raise ValueError(f'tost_conf must be in (0,1), got {conf!r}')
    if not isinstance(scale, str) or scale not in ('log_ratio', 'raw_ppl'):
        # Silently falling back to log_ratio makes a raw-PPL margin mean
        # something else than the caller declared.  Fail closed instead.
        raise ValueError(
            f"scale must be 'log_ratio' or 'raw_ppl', got {scale!r}")
    # TOST decisions must use the level implied by the DECLARED confidence
    # level, not always alpha/2.  With the default tost_conf=1-alpha the two
    # are identical; with a stricter tost_conf=0.99 the old code used the
    # 95% decision level and could call a panel equivalent even though its
    # own 99% interval was wider than the margin.
    _alpha_one = (1.0 - conf) / 2.0
    # The beyond-margin superiority test is a separate decision with its own
    # declared level, alpha/2.  It must NOT inherit tost_conf: a looser
    # equivalence confidence would otherwise make a superiority claim at
    # 0.05 one-sided, and a stricter one (0.99) would suppress a claim that
    # clears the documented alpha/2 level.  Keep the two levels explicit.
    _alpha_superiority = a / 2.0
    sign = None
    if expected_sign is not None:
        if isinstance(expected_sign, str):
            s = expected_sign.strip().lower()
            if s in ('positive', '+', '+1', '1'):
                sign = 'positive'
            elif s in ('negative', '-', '-1'):
                sign = 'negative'
            else:
                raise ValueError(f'expected_sign must be positive/negative, got {expected_sign!r}')
        elif isinstance(expected_sign, (int, np.integer)) and not isinstance(expected_sign, (bool, np.bool_)):
            if int(expected_sign) == 1:
                sign = 'positive'
            elif int(expected_sign) == -1:
                sign = 'negative'
            else:
                raise ValueError(f'expected_sign must be +/-1, got {expected_sign!r}')
        else:
            raise ValueError(f'expected_sign must be positive/negative, got {expected_sign!r}')
    out = {
        'kind': 'confirmation_practical_significance',
        'version': (policy_version or CONFIRMATION_SESOI['version']),
        'role': ('descriptive practical-significance/equivalence readout; it '
                 'does not alter p_exact_signflip, p_holm, verified or confirmed'),
        'scale': scale,
        'unit': 'seed',
        'alpha': float(a),
        'alpha_one_sided': float(_alpha_one),
        'superiority_alpha_one_sided': float(_alpha_superiority),
        'tost_conf': float(conf),
        'expected_sign': sign,
        'n': 0,
        'n_dropped': 0,
        'seeds': [],
        'per_seed_log_ratio': [],
        'per_seed_ppl_a': [],
        'per_seed_ppl_b': [],
        'margin': (None if margin is None else _cp_number(margin, 'margin', positive=True)),
        'relative_pct': (None if relative_pct is None else _cp_number(relative_pct, 'relative_pct')),
        'mean': None,
        'relative_effect_pct': None,
        'geometric_ratio': None,
        'ci': None,
        'p_tost': None,
        'p_lo': None,
        'p_hi': None,
        'p_beyond': None,
        'equivalent': None,
        'beyond_sesoi': None,
        'status': 'not_estimable',
        'reason': None,
        'note': ('margin must be declared before looking at the result; a '
                 'post-hoc margin is descriptive, not confirmatory'),
    }
    if margin is None:
        out['status'] = 'not_declared'
        out['reason'] = 'no SESOI margin is declared for this comparison'
        return out
    if isinstance(deltas, (str, bytes, bytearray, dict)):
        raise ValueError('confirmation_practical_significance: deltas must be '
                         f'an iterable of numbers, not a {type(deltas).__name__}')
    try:
        raw = list(deltas if deltas is not None else [])
    except TypeError:
        raise ValueError('confirmation_practical_significance: deltas must be an iterable') from None
    seed_in = None
    if seeds is not None:
        if isinstance(seeds, (str, bytes, bytearray, dict)):
            raise ValueError('seeds must be an iterable of integers')
        seed_in = list(seeds)
        if len(seed_in) != len(raw):
            raise ValueError(f'seeds length {len(seed_in)} does not match deltas length {len(raw)}')
    d, used, kept = [], [], []
    for i, x in enumerate(raw):
        f = _finite_scalar(x)
        if f is None:
            continue
        d.append(float(f))
        kept.append(i)
        used.append(_as_seed_int(seed_in[i]) if seed_in is not None else i)
    if relative_pct is None and margin is not None and out['scale'] == 'log_ratio':
        _m = _cp_number(margin, 'margin', positive=True)
        try:
            relative_pct = float(math.expm1(_m) * 100.0)
        except OverflowError:
            relative_pct = None
    out['relative_pct'] = (None if relative_pct is None
                           else _cp_number(relative_pct, 'relative_pct'))
    out['n_dropped'] = len(raw) - len(d)
    out['n'] = len(d)
    out['seeds'] = [int(s) for s in used if s is not None]
    if seed_in is not None and any(s is None for s in used):
        raise ValueError('seeds must be integers')
    if len(set(out['seeds'])) != len(out['seeds']):
        raise ValueError('seeds contain duplicates')
    if len(d) < 2:
        out['status'] = 'not_estimable'
        out['reason'] = 'fewer than two finite paired seeds'
        out['per_seed_log_ratio'] = [float(x) for x in d]
        return out
    arr = np.asarray(d, dtype=float)
    # Store the analysis-scale vector so an audit can replay the test even
    # when the source PPL cells are absent.
    out['per_seed_log_ratio'] = [float(x) for x in d]
    if (ppl_a is None) != (ppl_b is None):
        raise ValueError('ppl_a and ppl_b must be provided together')
    if ppl_a is not None:
        if isinstance(ppl_a, (str, bytes, bytearray, dict)) or isinstance(
                ppl_b, (str, bytes, bytearray, dict)):
            raise ValueError('ppl_a/ppl_b must be iterables of PPL values')
        try:
            pa = list(ppl_a)
            pb = list(ppl_b)
        except TypeError:
            raise ValueError('ppl_a/ppl_b must be iterables') from None
        if len(pa) != len(raw) or len(pb) != len(raw):
            # The arrays are indexed by the SAME original positions as
            # ``deltas``; a shorter/longer provenance vector cannot be
            # aligned and must not be silently ignored.
            raise ValueError(
                'ppl_a/ppl_b length must match deltas '
                f'({len(pa)}/{len(pb)} vs {len(raw)})')
        # A PPL is a finite positive measurement.  Validating here keeps an
        # inf/NaN/string/None provenance cell from leaking into a strict-JSON
        # report (and from masking which input was malformed).
        out['per_seed_ppl_a'] = [
            _cp_number(pa[i], f'ppl_a[{i}]', positive=True) for i in kept]
        out['per_seed_ppl_b'] = [
            _cp_number(pb[i], f'ppl_b[{i}]', positive=True) for i in kept]
    _ms = _scaled_mean_std_if_extreme(arr, ddof=1)
    if _ms is not None:
        mean, sd = _ms
    else:
        mean = float(arr.mean())
        sd = float(arr.std(ddof=1))
    if not (math.isfinite(mean) and math.isfinite(sd)):
        out['status'] = 'not_estimable'
        out['reason'] = 'mean or SD is not representable'
        return out
    out['mean'] = float(mean)
    if out['scale'] == 'log_ratio':
        ratio = math.exp(mean) if -745.0 < mean < 709.0 else None
        out['geometric_ratio'] = (float(ratio) if ratio is not None and math.isfinite(ratio) else None)
        out['relative_effect_pct'] = (float(math.expm1(mean) * 100.0)
                                      if -745.0 < mean < 709.0 else None)
    se = (sd / math.sqrt(len(d))) if sd > 0.0 else 0.0
    df = len(d) - 1
    if se > 0.0:
        p_lo = float(_student_t_sf((mean + float(margin)) / se, float(df)))
        p_hi = float(_student_t_sf((float(margin) - mean) / se, float(df)))
        # Lower/upper one-sided tests against the margin, computed directly
        # (using 1-p would lose precision in the far tail).
        p_better = float(_student_t_sf(-(mean + float(margin)) / se, float(df)))
        p_worse = float(_student_t_sf((mean - float(margin)) / se, float(df)))
    else:
        # Degenerate constant panel: the t limit at the boundary is 0.5, not
        # 0.  ``_at(x)`` is the se -> 0 limit of P(T > x): 0 for x > 0, 1 for
        # x < 0, and 0.5 exactly on the boundary.
        def _at(x):
            # se -> 0 limit of P(T > x): 0 above the boundary, 1 below it.
            return 0.5 if x == 0.0 else (0.0 if x > 0.0 else 1.0)
        p_lo = _at(float(margin) + mean)
        p_hi = _at(float(margin) - mean)
        p_better = (1.0 - p_lo) if p_lo != 0.5 else 0.5
        p_worse = _at(mean - float(margin))
    p_tost = float(max(p_lo, p_hi))
    out.update({'p_lo': float(p_lo), 'p_hi': float(p_hi),
                'p_tost': p_tost, 'p_better': float(p_better),
                'p_worse': float(p_worse)})
    if mean > 0.0:
        point_direction = 'positive'
    elif mean < 0.0:
        point_direction = 'negative'
    else:
        point_direction = None
    out['point_direction'] = point_direction
    equivalent = bool(p_lo < _alpha_one and p_hi < _alpha_one)
    beyond_better = bool(p_better < _alpha_superiority and mean < 0.0)
    beyond_worse = bool(p_worse < _alpha_superiority and mean > 0.0)
    out['equivalent'] = equivalent
    out['beyond_sesoi_better'] = beyond_better
    out['beyond_sesoi_worse'] = beyond_worse
    if point_direction is None:
        # An exactly zero point effect is not evidence of practical
        # equivalence either: it is the degenerate no-effect case.
        status = 'zero_effect'
        out['equivalent'] = None
        out['p_tost'] = None
        if sign == 'negative':
            out['p_beyond'] = float(p_better)
            out['beyond_sesoi'] = False
        elif sign == 'positive':
            out['p_beyond'] = float(p_worse)
            out['beyond_sesoi'] = False
        else:
            out['p_beyond'] = None
            out['beyond_sesoi'] = None
    elif sign == 'negative':
        out['p_beyond'] = float(p_better)
        out['beyond_sesoi'] = beyond_better
        if beyond_better:
            status = 'beyond_sesoi'
        elif beyond_worse:
            status = 'direction_conflict'
        elif equivalent:
            status = 'equivalent_within_sesoi'
        else:
            status = 'indeterminate'
    elif sign == 'positive':
        out['p_beyond'] = float(p_worse)
        out['beyond_sesoi'] = beyond_worse
        if beyond_worse:
            status = 'beyond_sesoi'
        elif beyond_better:
            status = 'direction_conflict'
        elif equivalent:
            status = 'equivalent_within_sesoi'
        else:
            status = 'indeterminate'
    else:
        # No pre-registered direction: equivalence is still well defined; a
        # directional "beyond margin" claim is not.
        out['p_beyond'] = None
        out['beyond_sesoi'] = None
        if equivalent:
            status = 'equivalent_within_sesoi'
        else:
            status = 'indeterminate'
    # The paired-t interval is the same construction TOST uses; report it so
    # the reader can see the classification without re-deriving it.
    _tc = _t_critical(float(conf), float(df))
    if math.isfinite(_tc):
        lo, hi = mean - _tc * se, mean + _tc * se
        if math.isfinite(lo) and math.isfinite(hi):
            out['ci'] = [float(lo), float(hi)]
    out['status'] = status
    out['note'] = ('TOST/CI and the beyond-margin p use the declared margin '
                   '(manifest block when present, otherwise the library policy; '
                   'the latter is descriptive, not a pre-registered decision rule); '
                   'they are not added to the frozen Holm family and '
                   'never change `confirmed`.  `equivalent_within_sesoi` means '
                   'smaller than the practical margin, not exactly zero.')
    return out


def confirmation_sesoi_design_audit(manifest):
    """Design-time check for a future confirmation's practical-significance plan.

    The current frozen v3 manifest predates this block, so an absent
    declaration is a warning (``not_declared``), not a failure.  A manifest
    that explicitly sets ``statistics.require_practical_significance=true``
    must declare a valid margin for every primary comparison; otherwise the
    pre-test preflight fails instead of freezing an uninterpretable plan.
    """
    m = manifest if isinstance(manifest, dict) else {}
    stat = m.get('statistics') if isinstance(m.get('statistics'), dict) else {}
    required = stat.get('require_practical_significance') is True
    out = {'kind': 'confirmation_sesoi_design_audit', 'status': 'ok',
           'declared': False, 'required': bool(required),
           'policy': None, 'problems': [], 'warnings': [],
           'note': ('design-time only; the current v3 manifest has no SESOI '
                    'block, so the descriptive library policy is used at '
                    'analysis time')}
    if _cp_declared_block(m) is None:
        out['status'] = 'fail' if required else 'not_declared'
        msg = ('the manifest does not declare practical_significance; a '
               'post-hoc margin cannot be a pre-registered decision rule')
        (out['problems'] if required else out['warnings']).append(msg)
        return out
    try:
        pol = confirmation_sesoi_for_manifest(m)
    except Exception as e:
        out['status'] = 'fail'
        out['problems'].append(f'invalid practical_significance: {type(e).__name__}: {e}')
        return out
    out['declared'] = True
    out['policy'] = pol
    if pol.get('missing'):
        msg = (f'no SESOI margin for primary comparison(s) {pol["missing"]}; '
               'a partial declaration cannot be read as a family-wide plan')
        (out['problems'] if required else out['warnings']).append(msg)
    if not (isinstance(pol.get('tost_conf'), (int, float))
            and 0.0 < float(pol['tost_conf']) < 1.0):
        out['problems'].append('tost_conf is not a valid probability')
    if out['problems']:
        out['status'] = 'fail'
    elif pol.get('missing'):
        out['status'] = 'warnings'
    else:
        out['status'] = 'ok'
    return out


def confirmation_practical_family(comparisons, *, confirmed=(), policy=None,
                                  alpha=0.05, family_names=None):
    """Aggregate per-comparison practical-significance blocks into families.

    Two separate Holm families are reported: (a) the one-sided "effect exceeds
    the SESOI in the pre-registered direction" p, and (b) the TOST equivalence
    p.  They are never mixed with the frozen superiority family.
    """
    a = _cp_number(alpha, 'alpha')
    if not (0.0 < a < 1.0):
        raise ValueError(f'alpha must be in (0,1), got {a!r}')
    if not isinstance(comparisons, dict):
        raise ValueError('comparisons must be a mapping')
    names = list(family_names) if family_names is not None else list(comparisons)
    if isinstance(family_names, (str, bytes, bytearray)):
        raise ValueError('family_names must be an iterable of names, not a string')
    names = [str(n) for n in names]
    _declared = set(names)
    _unknown = sorted(set(comparisons) - _declared)
    if _unknown:
        raise ValueError(f'comparison(s) not in the declared family: {_unknown}')
    _omitted = sorted(_declared - set(comparisons))
    if _omitted:
        raise ValueError(
            f'comparison(s) omitted from the supplied rows: {_omitted}; a '
            'missing slot would shrink both Holm families')
    beyond_pairs, tost_pairs = [], []
    rows = {}
    for name in names:
        comp = comparisons.get(name)
        blk = (comp or {}).get('practical_significance') if isinstance(comp, dict) else None
        if not isinstance(blk, dict):
            blk = {'status': 'not_estimable', 'reason': 'no practical-significance block'}
        pb = blk.get('p_beyond')
        pt = blk.get('p_tost')
        beyond_pairs.append((name, pb if (isinstance(pb, (int, float)) and not isinstance(pb, (bool, np.bool_))
                                          and math.isfinite(float(pb))) else 1.0))
        tost_pairs.append((name, pt if (isinstance(pt, (int, float)) and not isinstance(pt, (bool, np.bool_))
                                        and math.isfinite(float(pt))) else 1.0))
        rows[name] = {
            'status': blk.get('status'),
            'reason': blk.get('reason'),
            'scale': blk.get('scale'),
            'margin': blk.get('margin'),
            'relative_pct': blk.get('relative_pct'),
            'n': blk.get('n'),
            'mean': blk.get('mean'),
            'relative_effect_pct': blk.get('relative_effect_pct'),
            'ci': blk.get('ci'),
            'p_tost': blk.get('p_tost'),
            'p_beyond': blk.get('p_beyond'),
            'equivalent': blk.get('equivalent'),
            'beyond_sesoi': blk.get('beyond_sesoi'),
            'expected_sign': blk.get('expected_sign'),
        }
    # The beyond-margin family uses the declared superiority level.  The TOST
    # family must use the level implied by the policy's declared confidence
    # (which may be stricter than 1-alpha); otherwise a 0.99-confidence
    # equivalence row could be called equivalent by a 0.95-level Holm test.
    _tost_raw = (policy or {}).get('tost_conf')
    _tost_conf = ((1.0 - a) if _tost_raw is None
                  else _cp_number(_tost_raw, 'policy.tost_conf'))
    if not (0.0 < _tost_conf < 1.0):
        raise ValueError(f'policy.tost_conf must be in (0,1), got {_tost_conf!r}')
    _tost_alpha_one = (1.0 - _tost_conf) / 2.0
    beyond_holm = holm_adjust(beyond_pairs, alpha=a / 2.0)
    tost_holm = holm_adjust(tost_pairs, alpha=_tost_alpha_one)
    for name in names:
        _b = beyond_holm.get(name) or {}
        _t = tost_holm.get(name) or {}
        _pbh = _b.get('p_holm')
        _pth = _t.get('p_holm')
        rows[name]['p_beyond_holm'] = (None if _pbh is None else float(_pbh))
        rows[name]['p_tost_holm'] = (None if _pth is None else float(_pth))
        rows[name]['practically_meaningful'] = bool(
            rows[name].get('beyond_sesoi') is True
            and _pbh is not None and float(_pbh) < a / 2.0)
        rows[name]['equivalent_holm'] = bool(
            rows[name].get('status') == 'equivalent_within_sesoi'
            and rows[name].get('equivalent') is True
            and _pth is not None and float(_pth) < _tost_alpha_one)
    _conf = {str(x) for x in (confirmed or [])}
    meaningful = sorted(n for n in names if rows[n].get('practically_meaningful'))
    _nondirectional = {n for n in names if rows[n].get('expected_sign') is None}
    equivalent = sorted(n for n in names if rows[n].get('equivalent_holm'))
    conflict = sorted(n for n in names if rows[n].get('status') == 'direction_conflict')
    not_declared = sorted(n for n in names if rows[n].get('status') == 'not_declared')
    not_estimable = sorted(n for n in names if rows[n].get('status') == 'not_estimable')
    indeterminate = sorted(n for n in names
                           if n not in meaningful and n not in equivalent
                           and n not in conflict and n not in not_declared
                           and n not in not_estimable)
    block = {
        'kind': 'confirmation_practical_significance_family',
        'version': (policy or {}).get('version', CONFIRMATION_SESOI['version']),
        'status': ('not_declared' if len(not_declared) == len(names)
                   else ('not_estimable' if len(not_estimable) == len(names)
                         else 'ok')),
        'alpha': float(a),
        'alpha_one_sided': float(a / 2.0),
        'tost_conf': float(_tost_conf),
        'tost_alpha_one_sided': float(_tost_alpha_one),
        'policy': policy,
        'family_names': list(names),
        'family_size': len(names),
        'comparisons': rows,
        'holm': {
            'beyond_sesoi': beyond_holm,
            'tost_equivalence': tost_holm,
            'alpha_one_sided': float(a / 2.0),
            'tost_alpha_one_sided': float(_tost_alpha_one),
            'note': ('two Holm families over the SAME declared primary set; '
                     'missing/unestimable slots enter at p=1 so the correction '
                     'family cannot shrink'),
        },
        'summary': {
            'n_comparisons': len(names),
            'n_declared_margin': sum(1 for n in names if rows[n].get('margin') is not None),
            'n_practically_meaningful': len(meaningful),
            'practically_meaningful': meaningful,
            'n_equivalent_within_sesoi': len(equivalent),
            'equivalent_within_sesoi': equivalent,
            'direction_conflict': conflict,
            'indeterminate': indeterminate,
            'not_declared': not_declared,
            'not_estimable': not_estimable,
            'frozen_confirmed': sorted(_conf),
            'frozen_confirmed_but_not_beyond_sesoi': sorted(
                (_conf - set(meaningful)) - _nondirectional),
            'frozen_confirmed_nondirectional': sorted(_conf & _nondirectional),
            'equivalent_but_not_frozen_confirmed': sorted(set(equivalent) - _conf),
        },
        'role': ('descriptive practical-significance readout; separate from '
                 'the frozen superiority family and never part of `confirmed`'),
        'note': ('a confirmed statistical difference below the SESOI cannot be '
                 'called practically meaningful; an equivalence result is '
                 'bounded by the declared margin, not proof of zero effect'),
    }
    return block
